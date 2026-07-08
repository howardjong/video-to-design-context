import pytest
from pydantic import ValidationError

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


def test_gemini_json_response_validates_against_schema():
    analysis = TasteAnalysis.model_validate(valid_payload())

    assert analysis.assets[0].start_seconds == 5
    assert analysis.preference_moments[0].timestamp_seconds == 12.5
    assert analysis.suggested_frames[0].confidence == 0.91


def test_missing_timestamps_fail_validation():
    payload = valid_payload()
    del payload["preference_moments"][0]["timestamp"]

    with pytest.raises(ValidationError):
        TasteAnalysis.model_validate(payload)


def test_preference_language_must_be_specific_not_vague():
    payload = valid_payload()
    payload["preference_moments"][0]["preference"] = "Clean design."
    payload["preference_moments"][0]["rationale"] = "Clean."

    with pytest.raises(ValidationError, match="specific design properties"):
        TasteAnalysis.model_validate(payload)
