from tastepack.config import TastepackConfig
from tastepack.schema import TasteAnalysis
from tastepack.segments import build_segment_plan, merge_segment_analyses


def make_analysis(asset_id, name, start, end, moment_time, frame_time, transcript):
    return TasteAnalysis.model_validate(
        {
            "source_summary": f"Review of {name}",
            "transcript": transcript,
            "assets": [
                {
                    "id": asset_id,
                    "name": name,
                    "kind": "website",
                    "start_timestamp": start,
                    "end_timestamp": end,
                    "summary": f"{name} example",
                }
            ],
            "preference_moments": [
                {
                    "asset_id": asset_id,
                    "timestamp": moment_time,
                    "sentiment": "positive",
                    "preference": "Likes the grouped navigation controls.",
                    "rationale": "The grouping reduces the visual scan path.",
                    "categories": ["layout"],
                    "confidence": 0.9,
                }
            ],
            "suggested_frames": [
                {
                    "asset_id": asset_id,
                    "timestamp": frame_time,
                    "reason": "Shows the navigation grouping.",
                    "confidence": 0.9,
                }
            ],
            "visual_details": {"layout": ["Grouped navigation controls"]},
            "motion_details": {"animations": ["Short hover feedback"]},
        }
    )


def test_segment_plan_uses_deterministic_overlapping_windows():
    plan = build_segment_plan(
        650,
        TastepackConfig(analysis_segment_seconds=300, analysis_segment_overlap_seconds=20),
    )

    assert [(segment.start_seconds, segment.end_seconds) for segment in plan] == [
        (0.0, 300.0),
        (280.0, 580.0),
        (560.0, 650.0),
    ]


def test_segment_merge_deduplicates_boundary_evidence_and_resolves_id_collisions():
    first = make_analysis("dashboard", "Metrics dashboard", 0, 300, 290, 290, "First segment.")
    boundary_duplicate = make_analysis(
        "dashboard",
        "Metrics dashboard",
        280,
        580,
        290,
        290,
        "Boundary segment.",
    )
    collision = make_analysis("dashboard", "Pricing page", 560, 650, 600, 600, "Final segment.")

    merged = merge_segment_analyses([first, boundary_duplicate, collision])

    assert [(asset.id, asset.start_seconds, asset.end_seconds) for asset in merged.assets] == [
        ("dashboard", 0.0, 580.0),
        ("dashboard-79f0fc86", 560.0, 650.0),
    ]
    assert len(merged.preference_moments) == 2
    assert len(merged.suggested_frames) == 2
    assert merged.transcript == "First segment.\n\nBoundary segment.\n\nFinal segment."
    assert merged.visual_details.layout == ["Grouped navigation controls"]
    assert merged.motion_details.animations == ["Short hover feedback"]
