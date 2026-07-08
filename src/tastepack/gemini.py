from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

from tastepack.config import TastepackConfig
from tastepack.schema import TasteAnalysis


class GeminiAnalysisError(RuntimeError):
    """Raised when Gemini analysis cannot produce valid structured output."""


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
) -> Any:
    deadline = time.monotonic() + timeout_seconds
    current_file = uploaded_file
    while time.monotonic() < deadline:
        current_file = files_client.get(name=uploaded_file.name)
        state = _state_name(current_file).upper()
        if state == "ACTIVE":
            return current_file
        if state in {"FAILED", "ERROR"}:
            raise GeminiAnalysisError(f"Gemini file processing failed with state {state}")
        sleep(poll_interval_seconds)
    raise GeminiAnalysisError("Timed out waiting for Gemini file processing")


def build_generation_config() -> Any:
    from google.genai import types

    return types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=TasteAnalysis,
    )


def analyze_video(
    video_path: Path,
    config: TastepackConfig,
    mock: bool = False,
    mock_payload_path: Path | None = None,
) -> TasteAnalysis:
    if mock:
        if mock_payload_path:
            return parse_gemini_json(mock_payload_path.read_text(encoding="utf-8"))
        return TasteAnalysis.model_validate(MOCK_ANALYSIS)
    api_key = load_api_key()
    if not api_key:
        raise GeminiAnalysisError("GEMINI_API_KEY is required unless --mock-gemini is used")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise GeminiAnalysisError("google-genai is not installed") from exc

    client = genai.Client(api_key=api_key)
    try:
        uploaded = client.files.upload(file=str(video_path))
        active_file = wait_for_file_active(
            client.files,
            uploaded,
            timeout_seconds=config.request_timeout_seconds,
        )
        prompt = (
            "Analyze this narrated screen recording. Return strict JSON only matching the "
            "tastepack schema: source_summary, transcript, assets, preference_moments, "
            "suggested_frames, visual_details, motion_details. Avoid vague design language "
            "unless you explain the concrete visual properties."
        )
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=[active_file, prompt],
            config=build_generation_config(),
        )
    except GeminiAnalysisError:
        raise
    except Exception as exc:
        raise GeminiAnalysisError(f"Gemini API request failed: {exc}") from exc
    return parse_gemini_json(response.text or "")
