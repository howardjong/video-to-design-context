import json

import httpx
import pytest

from tastepack.config import TastepackConfig
from tastepack.gemini import (
    GeminiAnalysisError,
    GeminiRunTelemetry,
    analyze_video,
    build_generation_config,
    call_with_retries,
    load_api_key,
    parse_gemini_json,
    wait_for_file_active,
)
from tastepack.schema import TasteAnalysis


def valid_payload():
    return {
        "source_summary": "Narrated review of two dashboard examples.",
        "transcript": "At twelve seconds I like the dense table hierarchy.",
        "assets": [
            {
                "id": "asset-1",
                "name": "Metrics dashboard",
                "kind": "dashboard",
                "start_timestamp": "00:00:05",
                "end_timestamp": "00:00:25",
                "summary": "A dashboard with KPI cards and a dense table.",
            }
        ],
        "preference_moments": [
            {
                "asset_id": "asset-1",
                "timestamp": "00:00:12.500",
                "sentiment": "positive",
                "preference": "Likes compact hierarchy with labels close to values.",
                "rationale": "The proximity makes the scan path obvious.",
                "categories": ["layout", "information_hierarchy"],
                "confidence": 0.86,
            }
        ],
        "suggested_frames": [
            {
                "asset_id": "asset-1",
                "timestamp": "00:00:12.500",
                "reason": "Shows the KPI/table relationship.",
                "confidence": 0.91,
            }
        ],
        "visual_details": {
            "style": ["restrained chrome with strong data density"],
            "layout": ["KPI strip above table"],
            "information_hierarchy": ["labels are visually subordinate to values"],
            "typography": ["tabular numeric emphasis"],
            "color": ["neutral surface with status accents"],
            "dashboard": ["high signal metrics above operational table"],
            "presentation": [],
            "negative_preferences": ["Avoid vague hero cards in work tools"],
        },
        "motion_details": {
            "animations": ["short hover feedback"],
            "interaction_details": ["row hover clarifies target"],
            "motion_preferences": ["prefer fast functional feedback over decorative motion"],
        },
    }


def test_gemini_json_parser_accepts_fenced_json():
    payload = valid_payload()
    raw = "```json\n" + json.dumps(payload) + "\n```"

    analysis = parse_gemini_json(raw)

    assert isinstance(analysis, TasteAnalysis)
    assert analysis.assets[0].name == "Metrics dashboard"


def test_invalid_gemini_json_fails_gracefully():
    with pytest.raises(GeminiAnalysisError, match="malformed JSON"):
        parse_gemini_json("{not json")


def test_valid_json_with_invalid_schema_fails_gracefully():
    with pytest.raises(GeminiAnalysisError, match="failed validation"):
        parse_gemini_json(json.dumps({"transcript": "missing required fields"}))


def test_generation_config_requests_json_matching_schema():
    config = build_generation_config()

    assert config.response_mime_type == "application/json"
    assert config.response_schema is TasteAnalysis
    assert config.http_options.timeout == 300_000


def test_load_api_key_reads_dotenv_without_printing_secret(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text("GEMINI_API_KEY=secret-from-dotenv\n")

    key = load_api_key(env_path)

    captured = capsys.readouterr()
    assert key == "secret-from-dotenv"
    assert "secret-from-dotenv" not in captured.out
    assert "secret-from-dotenv" not in captured.err


class FakeFile:
    def __init__(self, name, state, display_name=None):
        self.name = name
        self.state = state
        self.display_name = display_name


class FakeState:
    def __init__(self, name):
        self.name = name


def test_wait_for_file_active_polls_until_active():
    seen_names = []
    files = iter(
        [
            FakeFile("files/abc", FakeState("PROCESSING")),
            FakeFile("files/abc", FakeState("ACTIVE")),
        ]
    )

    class FakeFiles:
        def get(self, name, config=None):
            seen_names.append(name)
            return next(files)

    active_file = wait_for_file_active(
        FakeFiles(),
        FakeFile("files/abc", FakeState("PROCESSING")),
        sleep=lambda _: None,
    )

    assert active_file.state.name == "ACTIVE"
    assert seen_names == ["files/abc", "files/abc"]


def test_wait_for_file_active_fails_cleanly_when_processing_fails():
    class FakeFiles:
        def get(self, name, config=None):
            return FakeFile(name, "FAILED")

    with pytest.raises(GeminiAnalysisError, match="failed"):
        wait_for_file_active(
            FakeFiles(),
            FakeFile("files/abc", "PROCESSING"),
            sleep=lambda _: None,
        )


def test_wait_for_file_active_retries_transient_poll_errors():
    responses = iter(
        [
            FakeApiError(503),
            FakeFile("files/abc", FakeState("ACTIVE")),
        ]
    )

    class FakeFiles:
        def get(self, name, config=None):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

    active_file = wait_for_file_active(
        FakeFiles(),
        FakeFile("files/abc", FakeState("PROCESSING")),
        sleep=lambda _: None,
    )

    assert active_file.state.name == "ACTIVE"


class FakeApiError(Exception):
    def __init__(self, code):
        super().__init__(f"api error {code}")
        self.code = code


def test_transient_gemini_errors_retry_then_succeed():
    attempts = []

    def operation():
        attempts.append("try")
        if len(attempts) < 3:
            raise FakeApiError(429)
        return "ok"

    result = call_with_retries(
        operation,
        TastepackConfig(gemini_max_retries=3),
        sleep=lambda _: None,
    )

    assert result == "ok"
    assert len(attempts) == 3


def test_httpx_transport_errors_retry_then_succeed():
    attempts = []

    def operation():
        attempts.append("try")
        if len(attempts) < 2:
            raise httpx.ReadTimeout("read timed out")
        return "ok"

    result = call_with_retries(
        operation,
        TastepackConfig(gemini_max_retries=2),
        sleep=lambda _: None,
    )

    assert result == "ok"
    assert len(attempts) == 2


def test_retry_logging_includes_attempt_and_error_without_secrets(caplog):
    attempts = []

    def operation():
        attempts.append("try")
        if len(attempts) < 2:
            raise FakeApiError(429)
        return "ok"

    result = call_with_retries(
        operation,
        TastepackConfig(gemini_max_retries=2),
        sleep=lambda _: None,
        operation_name="Gemini generate_content",
    )

    assert result == "ok"
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Gemini generate_content attempt 1 failed with retryable error" in messages
    assert "429" in messages
    assert "GEMINI_API_KEY" not in messages


def test_non_retryable_gemini_errors_fail_without_retry():
    attempts = []

    def operation():
        attempts.append("try")
        raise FakeApiError(401)

    with pytest.raises(FakeApiError):
        call_with_retries(operation, TastepackConfig(), sleep=lambda _: None)

    assert len(attempts) == 1


def test_hard_quota_errors_fail_without_retry():
    attempts = []

    def operation():
        attempts.append("try")
        error = FakeApiError(429)
        error.args = ("quota exhausted for this billing period",)
        raise error

    with pytest.raises(FakeApiError, match="quota exhausted"):
        call_with_retries(operation, TastepackConfig(), sleep=lambda _: None)

    assert len(attempts) == 1


def test_retry_after_header_overrides_local_backoff_delay():
    attempts = []
    delays = []

    def operation():
        attempts.append("try")
        if len(attempts) == 1:
            error = FakeApiError(429)
            error.response = type("Response", (), {"headers": {"Retry-After": "7"}})()
            raise error
        return "ok"

    result = call_with_retries(
        operation,
        TastepackConfig(gemini_max_retries=2),
        sleep=delays.append,
    )

    assert result == "ok"
    assert delays == [7.0]


def test_local_retry_backoff_includes_configured_jitter(monkeypatch):
    attempts = []
    delays = []
    monkeypatch.setattr(
        "tastepack.gemini.random",
        type("Random", (), {"uniform": staticmethod(lambda minimum, maximum: 0.25)})(),
        raising=False,
    )

    def operation():
        attempts.append("try")
        if len(attempts) == 1:
            raise FakeApiError(503)
        return "ok"

    result = call_with_retries(
        operation,
        TastepackConfig(
            gemini_max_retries=2,
            gemini_retry_base_delay_seconds=1.0,
            gemini_retry_jitter_seconds=0.5,
        ),
        sleep=delays.append,
    )

    assert result == "ok"
    assert delays == [1.25]


class FakeGeminiResponse:
    text = json.dumps(valid_payload())


class FakeGeminiResponseWithUsage:
    text = json.dumps(valid_payload())
    usage_metadata = type(
        "UsageMetadata",
        (),
        {
            "prompt_token_count": 11,
            "candidates_token_count": 22,
            "total_token_count": 33,
        },
    )()
    candidates = [type("Candidate", (), {"finish_reason": "STOP"})()]


class FakeGeminiFiles:
    def __init__(
        self,
        upload_error=None,
        delete_error=None,
        listed_files=None,
    ):
        self.upload_error = upload_error
        self.delete_error = delete_error
        self.listed_files = listed_files or []
        self.deleted_names = []
        self.upload_configs = []
        self.get_configs = []
        self.delete_configs = []
        self.list_configs = []

    def upload(self, file, config=None):
        self.upload_configs.append(config)
        if self.upload_error:
            raise self.upload_error
        return FakeFile("files/uploaded", FakeState("ACTIVE"), config.display_name)

    def get(self, name, config=None):
        self.get_configs.append(config)
        return FakeFile(name, FakeState("ACTIVE"))

    def delete(self, name, config=None):
        self.deleted_names.append(name)
        self.delete_configs.append(config)
        if self.delete_error:
            raise self.delete_error

    def list(self, config=None):
        self.list_configs.append(config)
        if callable(self.listed_files):
            return self.listed_files()
        return self.listed_files


class FakeGeminiModels:
    def __init__(self, error=None, response=None):
        self.error = error
        self.response = response or FakeGeminiResponse()
        self.generate_configs = []

    def generate_content(self, **kwargs):
        self.generate_configs.append(kwargs["config"])
        if self.error:
            raise self.error
        return self.response


class FakeGeminiClient:
    def __init__(
        self,
        generate_error=None,
        upload_error=None,
        delete_error=None,
        listed_files=None,
        close_error=None,
        response=None,
    ):
        self.files = FakeGeminiFiles(
            upload_error=upload_error,
            delete_error=delete_error,
            listed_files=listed_files,
        )
        self.models = FakeGeminiModels(generate_error, response=response)
        self.close_error = close_error
        self.close_calls = 0

    def close(self):
        self.close_calls += 1
        if self.close_error:
            raise self.close_error


def test_analyze_video_deletes_uploaded_file_after_success(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient()
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    analysis = analyze_video(tmp_path / "input.mp4", TastepackConfig())

    assert analysis.assets[0].name == "Metrics dashboard"
    assert fake_client.files.deleted_names == ["files/uploaded"]


def test_analyze_video_records_safe_telemetry(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient(response=FakeGeminiResponseWithUsage())
    telemetry = GeminiRunTelemetry()
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    analyze_video(tmp_path / "input.mp4", TastepackConfig(), telemetry=telemetry)

    assert telemetry.operation_attempts == {
        "upload": 1,
        "file_status": 1,
        "generation": 1,
        "cleanup_list": 1,
        "cleanup_delete": 1,
    }
    assert telemetry.file_states == ["ACTIVE"]
    assert telemetry.finish_reason == "STOP"
    assert telemetry.token_usage == {
        "prompt_token_count": 11,
        "candidates_token_count": 22,
        "total_token_count": 33,
    }
    assert telemetry.total_duration_seconds is not None
    assert all(duration >= 0 for duration in telemetry.operation_durations_seconds.values())


def test_analyze_video_applies_separate_operation_timeouts(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient()
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)
    config = TastepackConfig(
        gemini_upload_timeout_seconds=11,
        gemini_file_processing_timeout_seconds=12,
        gemini_generation_timeout_seconds=13,
        gemini_cleanup_timeout_seconds=14,
    )

    analyze_video(tmp_path / "input.mp4", config)

    assert fake_client.files.upload_configs[0].http_options.timeout == 11_000
    assert fake_client.files.get_configs[0].http_options.timeout == 12_000
    assert fake_client.models.generate_configs[0].http_options.timeout == 13_000
    assert fake_client.files.delete_configs[0].http_options.timeout == 14_000


def test_analyze_video_deletes_uploaded_file_after_generation_failure(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient(generate_error=FakeApiError(401))
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    with pytest.raises(GeminiAnalysisError, match="Gemini API request failed"):
        analyze_video(tmp_path / "input.mp4", TastepackConfig())

    assert fake_client.files.deleted_names == ["files/uploaded"]


def test_cleanup_failure_does_not_fail_successful_analysis_and_is_redacted(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("GEMINI_API_KEY", "actual-secret")
    fake_client = FakeGeminiClient(
        delete_error=RuntimeError("delete failed for actual-secret"),
    )
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    analysis = analyze_video(tmp_path / "input.mp4", TastepackConfig())

    assert analysis.assets[0].id == "asset-1"
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "Gemini Files cleanup failed" in messages
    assert "actual-secret" not in messages
    assert fake_client.close_calls == 1


def test_ambiguous_upload_timeout_does_not_retry_and_reconciles_run_scoped_orphan(
    tmp_path, monkeypatch
):
    fake_client = FakeGeminiClient(upload_error=httpx.ReadTimeout("upload timed out"))
    fake_client.files.listed_files = lambda: [
        FakeFile(
            "files/orphaned-upload",
            FakeState("ACTIVE"),
            fake_client.files.upload_configs[0].display_name,
        )
    ]
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    with pytest.raises(GeminiAnalysisError, match="Gemini API request failed"):
        analyze_video(
            tmp_path / "input.mp4",
            TastepackConfig(gemini_max_retries=3),
        )

    assert len(fake_client.files.upload_configs) == 1
    assert fake_client.files.upload_configs[0].display_name.startswith("tastepack-")
    assert fake_client.files.deleted_names == ["files/orphaned-upload"]
    assert fake_client.close_calls == 1
