import json

import pytest

from tastepack.config import TastepackConfig
from tastepack.gemini import (
    GeminiAnalysisError,
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
    def __init__(self, name, state):
        self.name = name
        self.state = state


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
        def get(self, name):
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
        def get(self, name):
            return FakeFile(name, "FAILED")

    with pytest.raises(GeminiAnalysisError, match="failed"):
        wait_for_file_active(
            FakeFiles(),
            FakeFile("files/abc", "PROCESSING"),
            sleep=lambda _: None,
        )


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


class FakeGeminiResponse:
    text = json.dumps(valid_payload())


class FakeGeminiFiles:
    def __init__(self, generate_error=None, delete_error=None):
        self.generate_error = generate_error
        self.delete_error = delete_error
        self.deleted_names = []

    def upload(self, file):
        return FakeFile("files/uploaded", FakeState("ACTIVE"))

    def get(self, name):
        return FakeFile(name, FakeState("ACTIVE"))

    def delete(self, name):
        self.deleted_names.append(name)
        if self.delete_error:
            raise self.delete_error


class FakeGeminiModels:
    def __init__(self, error=None):
        self.error = error

    def generate_content(self, **kwargs):
        if self.error:
            raise self.error
        return FakeGeminiResponse()


class FakeGeminiClient:
    def __init__(self, generate_error=None, delete_error=None):
        self.files = FakeGeminiFiles(delete_error=delete_error)
        self.models = FakeGeminiModels(generate_error)


def test_analyze_video_deletes_uploaded_file_after_success(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient()
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    analysis = analyze_video(tmp_path / "input.mp4", TastepackConfig())

    assert analysis.assets[0].name == "Metrics dashboard"
    assert fake_client.files.deleted_names == ["files/uploaded"]


def test_analyze_video_deletes_uploaded_file_after_generation_failure(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient(generate_error=FakeApiError(401))
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    with pytest.raises(GeminiAnalysisError, match="Gemini API request failed"):
        analyze_video(tmp_path / "input.mp4", TastepackConfig())

    assert fake_client.files.deleted_names == ["files/uploaded"]


def test_cleanup_failure_does_not_fail_successful_analysis(tmp_path, monkeypatch):
    fake_client = FakeGeminiClient(delete_error=RuntimeError("delete failed"))
    monkeypatch.setattr("tastepack.gemini.load_api_key", lambda: "fake-key")
    monkeypatch.setattr("google.genai.Client", lambda api_key: fake_client)

    with pytest.warns(RuntimeWarning, match="Could not delete uploaded Gemini file"):
        analysis = analyze_video(tmp_path / "input.mp4", TastepackConfig())

    assert analysis.assets[0].id == "asset-1"
