import json

from tastepack.artifacts import generate_artifacts
from tastepack.config import TastepackConfig
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


def test_markdown_artifacts_include_traceability_and_grouped_assets(tmp_path):
    analysis = TasteAnalysis.model_validate(valid_payload())
    frame_map = {12.5: "frames/asset-1_000012500.jpg"}

    generate_artifacts(
        output_dir=tmp_path,
        analysis=analysis,
        frame_map=frame_map,
        config=TastepackConfig(produce_pdf=False),
        source_video_name="input.mp4",
    )

    taste_packet = (tmp_path / "taste_packet.md").read_text()
    design_preferences = (tmp_path / "design_preferences.md").read_text()
    metadata = json.loads((tmp_path / "metadata.json").read_text())

    assert "Metrics dashboard" in taste_packet
    assert "00:00:12.500" in taste_packet
    assert "frames/asset-1_000012500.jpg" in taste_packet
    assert "Positive Preferences" in design_preferences
    assert "Negative Preferences" in design_preferences
    assert "Motion Preferences" in design_preferences
    assert "Typography Preferences" in design_preferences
    assert "Layout Preferences" in design_preferences
    assert "Dashboard Preferences" in design_preferences
    assert "Presentation Preferences" in design_preferences
    assert "Reusable Design Rules" in design_preferences
    assert metadata["source_video"] == "input.mp4"


def test_api_keys_are_never_written_to_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    analysis = TasteAnalysis.model_validate(valid_payload())

    generate_artifacts(
        output_dir=tmp_path,
        analysis=analysis,
        frame_map={},
        config=TastepackConfig(produce_pdf=False),
        source_video_name="input.mp4",
    )

    for artifact in tmp_path.glob("*"):
        if artifact.is_file():
            assert "super-secret-key" not in artifact.read_text(errors="ignore")
