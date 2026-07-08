import json

import pytest

from tastepack.gemini import GeminiAnalysisError, load_api_key, parse_gemini_json
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
