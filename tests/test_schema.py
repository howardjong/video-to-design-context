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


def test_duplicate_asset_ids_fail_validation():
    payload = valid_payload()
    duplicate = payload["assets"][0].copy()
    duplicate["name"] = "Duplicate dashboard"
    payload["assets"].append(duplicate)

    with pytest.raises(ValidationError, match="Asset IDs must be unique"):
        TasteAnalysis.model_validate(payload)


def test_preference_outside_its_asset_range_fails_validation():
    payload = valid_payload()
    payload["preference_moments"][0]["timestamp"] = "00:00:30"

    with pytest.raises(ValidationError, match="Preference timestamp is outside asset range"):
        TasteAnalysis.model_validate(payload)


def test_suggested_frame_outside_its_asset_range_fails_validation():
    payload = valid_payload()
    payload["suggested_frames"][0]["timestamp"] = "00:00:30"

    with pytest.raises(ValidationError, match="Frame timestamp is outside asset range"):
        TasteAnalysis.model_validate(payload)


def test_zero_length_asset_range_fails_validation():
    payload = valid_payload()
    payload["assets"][0]["end_timestamp"] = "00:00:05"
    payload["preference_moments"][0]["timestamp"] = "00:00:05"
    payload["suggested_frames"][0]["timestamp"] = "00:00:05"

    with pytest.raises(ValidationError, match="Asset end timestamp must be after start timestamp"):
        TasteAnalysis.model_validate(payload)


def test_blank_asset_id_fails_validation():
    payload = valid_payload()
    payload["assets"][0]["id"] = "   "
    payload["preference_moments"][0]["asset_id"] = "   "
    payload["suggested_frames"][0]["asset_id"] = "   "

    with pytest.raises(ValidationError, match="Asset ID cannot be blank"):
        TasteAnalysis.model_validate(payload)


def test_assets_outside_the_probed_video_duration_fail_validation():
    payload = valid_payload()

    with pytest.raises(ValidationError, match="Asset range is outside video duration"):
        TasteAnalysis.model_validate(payload, context={"video_duration_seconds": 20.0})


def test_analysis_without_assets_fails_the_semantic_quality_gate():
    payload = valid_payload()
    payload["assets"] = []
    payload["preference_moments"] = []
    payload["suggested_frames"] = []

    with pytest.raises(ValidationError, match="Analysis must identify at least one asset"):
        TasteAnalysis.model_validate(payload)


def test_analysis_without_preference_moments_fails_the_semantic_quality_gate():
    payload = valid_payload()
    payload["preference_moments"] = []

    with pytest.raises(
        ValidationError,
        match="Analysis must identify at least one preference moment",
    ):
        TasteAnalysis.model_validate(payload)


def test_audio_analysis_requires_a_nonblank_transcript():
    payload = valid_payload()
    payload["transcript"] = "   "

    with pytest.raises(ValidationError, match="Analysis transcript is required for narrated video"):
        TasteAnalysis.model_validate(payload, context={"require_transcript": True})
