from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
from dotenv import load_dotenv
from pydantic import ValidationError

from tastepack.config import TastepackConfig
from tastepack.logging import get_logger, redact_secrets
from tastepack.schema import TasteAnalysis


class GeminiAnalysisError(RuntimeError):
    """Raised when Gemini analysis cannot produce valid structured output."""


@dataclass
class GeminiRunTelemetry:
    """Provider diagnostics that are safe to persist in a tastepack."""

    operation_attempts: dict[str, int] = field(default_factory=dict)
    operation_durations_seconds: dict[str, float] = field(default_factory=dict)
    file_states: list[str] = field(default_factory=list)
    finish_reason: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)
    total_duration_seconds: float | None = None

    def record_attempt(self, operation_name: str) -> None:
        self.operation_attempts[operation_name] = self.operation_attempts.get(operation_name, 0) + 1

    def record_duration(self, operation_name: str, duration_seconds: float) -> None:
        self.operation_durations_seconds[operation_name] = (
            self.operation_durations_seconds.get(operation_name, 0.0) + duration_seconds
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "operation_attempts": self.operation_attempts,
            "operation_durations_seconds": self.operation_durations_seconds,
            "file_states": self.file_states,
            "finish_reason": self.finish_reason,
            "token_usage": self.token_usage,
            "total_duration_seconds": self.total_duration_seconds,
        }


logger = get_logger("gemini")


MOCK_ANALYSIS: dict[str, Any] = {
    "source_summary": "Mock narrated review of one dashboard design.",
    "transcript": "At twelve seconds I like the compact hierarchy and table density.",
    "assets": [
        {
            "id": "asset-1",
            "name": "Mock dashboard",
            "kind": "dashboard",
            "start_timestamp": "00:00:00",
            "end_timestamp": "00:00:20",
            "summary": "A dashboard with a metric strip and operational table.",
        }
    ],
    "preference_moments": [
        {
            "asset_id": "asset-1",
            "timestamp": "00:00:12.000",
            "sentiment": "positive",
            "preference": "Likes compact grouping where labels sit close to values.",
            "rationale": "The arrangement shortens the scan path across the dashboard.",
            "categories": ["layout", "information_hierarchy", "dashboard"],
            "confidence": 0.9,
        }
    ],
    "suggested_frames": [
        {
            "asset_id": "asset-1",
            "timestamp": "00:00:12.000",
            "reason": "Shows the preference moment with compact metric grouping.",
            "confidence": 0.9,
        }
    ],
    "visual_details": {
        "style": ["Restrained interface chrome with dense operational content"],
        "layout": ["Metric strip sits above the table and shares the same grid rhythm"],
        "information_hierarchy": ["Metric labels stay subordinate to numeric values"],
        "typography": ["Numbers are emphasized with stable tabular spacing"],
        "color": ["Neutral surfaces use color only for status accents"],
        "dashboard": ["Summary metrics remain close to the detailed operational table"],
        "presentation": [],
        "negative_preferences": ["Avoid decorative cards that reduce data density"],
    },
    "motion_details": {
        "animations": ["Fast hover state"],
        "interaction_details": ["Row hover clarifies the selected target"],
        "motion_preferences": ["Prefer functional feedback over decorative animation"],
    },
}


def parse_gemini_json(raw: str) -> TasteAnalysis:
    raw = _strip_json_fence(raw)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiAnalysisError("Gemini returned malformed JSON") from exc
    try:
        return TasteAnalysis.model_validate(payload)
    except ValidationError as exc:
        raise GeminiAnalysisError(f"Gemini JSON failed validation: {exc}") from exc


def _strip_json_fence(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def load_api_key(env_path: Path | None = None) -> str | None:
    if env_path:
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _error_code(exc: BaseException) -> int | None:
    for attr in ("code", "status_code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    return None


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError | ConnectionError | httpx.TransportError):
        return True
    code = _error_code(exc)
    if code == 429 and "quota" in str(exc).lower():
        return False
    return code in {408, 429} or (code is not None and code >= 500)


def _retry_after_seconds(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = headers.get("Retry-After")
    try:
        delay = float(value)
    except (TypeError, ValueError):
        return None
    return delay if delay >= 0 else None


def _is_ambiguous_transport_error(exc: BaseException) -> bool:
    return isinstance(
        exc,
        httpx.ReadError | httpx.ReadTimeout | httpx.WriteError | httpx.WriteTimeout,
    )


def call_with_retries(
    operation: Callable[[], Any],
    config: TastepackConfig,
    sleep: Callable[[float], None] = time.sleep,
    operation_name: str = "Gemini operation",
    retry_ambiguous_transport_errors: bool = True,
    on_attempt: Callable[[int], None] | None = None,
) -> Any:
    attempt = 1
    while True:
        if on_attempt:
            on_attempt(attempt)
        try:
            return operation()
        except Exception as exc:
            if (
                attempt >= config.gemini_max_retries
                or not _is_retryable_error(exc)
                or (
                    not retry_ambiguous_transport_errors
                    and _is_ambiguous_transport_error(exc)
                )
            ):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = config.gemini_retry_base_delay_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0, config.gemini_retry_jitter_seconds)
            logger.warning(
                "%s attempt %s failed with retryable error; retrying in %.1fs: %s",
                operation_name,
                attempt,
                delay,
                redact_secrets(exc),
            )
            sleep(delay)
            attempt += 1


def call_with_telemetry(
    operation: Callable[[], Any],
    config: TastepackConfig,
    operation_name: str,
    telemetry: GeminiRunTelemetry | None = None,
    sleep: Callable[[float], None] = time.sleep,
    retry_ambiguous_transport_errors: bool = True,
) -> Any:
    started_at = time.monotonic()
    try:
        return call_with_retries(
            operation,
            config,
            sleep=sleep,
            operation_name=operation_name,
            retry_ambiguous_transport_errors=retry_ambiguous_transport_errors,
            on_attempt=(
                lambda _attempt: telemetry.record_attempt(operation_name)
                if telemetry is not None
                else None
            ),
        )
    finally:
        if telemetry is not None:
            telemetry.record_duration(operation_name, time.monotonic() - started_at)


def _state_name(file_obj: Any) -> str:
    state = getattr(file_obj, "state", None)
    if hasattr(state, "name"):
        return str(state.name)
    return str(state or "")


def wait_for_file_active(
    files_client: Any,
    uploaded_file: Any,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    config: TastepackConfig | None = None,
    telemetry: GeminiRunTelemetry | None = None,
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    current_file = uploaded_file
    retry_config = config or TastepackConfig()
    while time.monotonic() < deadline:
        current_file = call_with_telemetry(
            lambda: files_client.get(
                name=uploaded_file.name,
                config=build_get_file_config(retry_config),
            ),
            retry_config,
            operation_name="file_status",
            telemetry=telemetry,
            sleep=sleep,
        )
        state = _state_name(current_file).upper()
        if telemetry is not None:
            telemetry.file_states.append(state)
        if state == "ACTIVE":
            return current_file
        if state in {"FAILED", "ERROR"}:
            raise GeminiAnalysisError(f"Gemini file processing failed with state {state}")
        sleep(poll_interval_seconds)
    raise GeminiAnalysisError("Timed out waiting for Gemini file processing")


def build_http_options(timeout_seconds: float) -> Any:
    from google.genai import types

    return types.HttpOptions(timeout=math.ceil(timeout_seconds * 1000))


def build_upload_config(config: TastepackConfig, display_name: str) -> Any:
    from google.genai import types

    return types.UploadFileConfig(
        http_options=build_http_options(config.gemini_upload_timeout_seconds),
        display_name=display_name,
    )


def build_get_file_config(config: TastepackConfig) -> Any:
    from google.genai import types

    return types.GetFileConfig(
        http_options=build_http_options(config.gemini_file_processing_timeout_seconds)
    )


def build_delete_file_config(config: TastepackConfig) -> Any:
    from google.genai import types

    return types.DeleteFileConfig(
        http_options=build_http_options(config.gemini_cleanup_timeout_seconds)
    )


def build_list_files_config(config: TastepackConfig) -> Any:
    from google.genai import types

    return types.ListFilesConfig(
        http_options=build_http_options(config.gemini_cleanup_timeout_seconds)
    )


def build_generation_config(config: TastepackConfig | None = None) -> Any:
    from google.genai import types

    config = config or TastepackConfig()
    return types.GenerateContentConfig(
        http_options=build_http_options(config.gemini_generation_timeout_seconds),
        response_mime_type="application/json",
        response_schema=TasteAnalysis,
    )


def _value_name(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "name"):
        return str(value.name)
    return str(value)


def _record_response_telemetry(response: Any, telemetry: GeminiRunTelemetry | None) -> None:
    if telemetry is None:
        return
    usage_metadata = getattr(response, "usage_metadata", None)
    for field_name in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
    ):
        value = getattr(usage_metadata, field_name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            telemetry.token_usage[field_name] = value
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        telemetry.finish_reason = _value_name(getattr(candidates[0], "finish_reason", None))


def cleanup_uploaded_files(
    files_client: Any,
    uploaded_file: Any | None,
    upload_display_name: str,
    config: TastepackConfig,
    telemetry: GeminiRunTelemetry | None = None,
) -> None:
    """Best-effort cleanup that never overrides the run's primary outcome."""
    file_names = {
        name
        for name in (getattr(uploaded_file, "name", None),)
        if isinstance(name, str) and name
    }
    try:
        remote_files = call_with_telemetry(
            lambda: files_client.list(config=build_list_files_config(config)),
            config,
            operation_name="cleanup_list",
            telemetry=telemetry,
        )
        for remote_file in remote_files:
            if getattr(remote_file, "display_name", None) == upload_display_name:
                remote_name = getattr(remote_file, "name", None)
                if isinstance(remote_name, str) and remote_name:
                    file_names.add(remote_name)
    except Exception as cleanup_exc:
        logger.warning(
            "Gemini Files cleanup reconciliation failed; a remote file may remain: %s",
            redact_secrets(cleanup_exc),
        )

    for file_name in file_names:
        try:
            call_with_telemetry(
                lambda file_name=file_name: files_client.delete(
                    name=file_name,
                    config=build_delete_file_config(config),
                ),
                config,
                operation_name="cleanup_delete",
                telemetry=telemetry,
            )
        except Exception as cleanup_exc:
            logger.warning(
                "Gemini Files cleanup failed; a remote file may remain: %s",
                redact_secrets(cleanup_exc),
            )


def analyze_video(
    video_path: Path,
    config: TastepackConfig,
    mock: bool = False,
    mock_payload_path: Path | None = None,
    telemetry: GeminiRunTelemetry | None = None,
) -> TasteAnalysis:
    started_at = time.monotonic()
    if mock:
        if mock_payload_path:
            analysis = parse_gemini_json(mock_payload_path.read_text(encoding="utf-8"))
        else:
            analysis = TasteAnalysis.model_validate(MOCK_ANALYSIS)
        if telemetry is not None:
            telemetry.finish_reason = "MOCK"
            telemetry.total_duration_seconds = time.monotonic() - started_at
        return analysis
    api_key = load_api_key()
    if not api_key:
        raise GeminiAnalysisError("GEMINI_API_KEY is required unless --mock-gemini is used")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise GeminiAnalysisError("google-genai is not installed") from exc

    client = genai.Client(api_key=api_key)
    uploaded = None
    upload_display_name = f"tastepack-{uuid4().hex}"
    try:
        uploaded = call_with_telemetry(
            lambda: client.files.upload(
                file=str(video_path),
                config=build_upload_config(config, upload_display_name),
            ),
            config,
            operation_name="upload",
            telemetry=telemetry,
            retry_ambiguous_transport_errors=False,
        )
        logger.debug("Uploaded video to Gemini Files API as %s", uploaded.name)
        active_file = wait_for_file_active(
            client.files,
            uploaded,
            timeout_seconds=config.gemini_file_processing_timeout_seconds,
            config=config,
            telemetry=telemetry,
        )
        logger.debug("Gemini file %s is ACTIVE", active_file.name)
        prompt = (
            "Analyze this narrated screen recording. Return strict JSON only matching the "
            "tastepack schema: source_summary, transcript, assets, preference_moments, "
            "suggested_frames, visual_details, motion_details. Avoid vague design language "
            "unless you explain the concrete visual properties."
        )
        response = call_with_telemetry(
            lambda: client.models.generate_content(
                model=config.gemini_model,
                contents=[active_file, prompt],
                config=build_generation_config(config),
            ),
            config,
            operation_name="generation",
            telemetry=telemetry,
            retry_ambiguous_transport_errors=False,
        )
        _record_response_telemetry(response, telemetry)
    except GeminiAnalysisError:
        raise
    except Exception as exc:
        raise GeminiAnalysisError(f"Gemini API request failed: {redact_secrets(exc)}") from exc
    finally:
        if config.cleanup_uploaded_files:
            cleanup_uploaded_files(
                client.files,
                uploaded,
                upload_display_name,
                config,
                telemetry,
            )
        try:
            client.close()
        except Exception as close_exc:
            logger.warning(
                "Could not close Gemini client cleanly: %s",
                redact_secrets(close_exc),
            )
        if telemetry is not None:
            telemetry.total_duration_seconds = time.monotonic() - started_at
    return parse_gemini_json(response.text or "")
