from __future__ import annotations

import json
from pathlib import Path

from tastepack.config import TastepackConfig
from tastepack.pdf import generate_pdf_from_markdown
from tastepack.schema import AssetExample, PreferenceMoment, TasteAnalysis
from tastepack.timestamps import format_timestamp


def _bullet_list(items: list[str]) -> str:
    if not items:
        return "- None identified.\n"
    return "".join(f"- {item}\n" for item in items)


def _moments_for_asset(analysis: TasteAnalysis, asset: AssetExample) -> list[PreferenceMoment]:
    return [moment for moment in analysis.preference_moments if moment.asset_id == asset.id]


def _frame_for_timestamp(frame_map: dict[float, str], seconds: float) -> str | None:
    return frame_map.get(round(seconds, 3))


def build_taste_packet_markdown(
    analysis: TasteAnalysis,
    frame_map: dict[float, str],
    source_video_name: str,
) -> str:
    lines = [
        "# Taste Packet",
        "",
        "## Source Metadata",
        f"- Source video: `{source_video_name}`",
        f"- Assets identified: {len(analysis.assets)}",
        "",
        "## Source Summary",
        analysis.source_summary,
        "",
        "## Transcript",
        analysis.transcript,
        "",
        "## Assets and Preference Moments",
    ]
    for asset in analysis.assets:
        lines.extend(
            [
                "",
                f"### {asset.name}",
                f"- Asset ID: `{asset.id}`",
                f"- Kind: {asset.kind}",
                "- Range: "
                f"{format_timestamp(asset.start_seconds)} to "
                f"{format_timestamp(asset.end_seconds)}",
                f"- Summary: {asset.summary}",
                "",
                "Preference moments:",
            ]
        )
        for moment in _moments_for_asset(analysis, asset):
            frame_path = _frame_for_timestamp(frame_map, moment.timestamp_seconds)
            frame_note = f" Frame: `{frame_path}`." if frame_path else ""
            lines.append(
                "- "
                f"{format_timestamp(moment.timestamp_seconds)} "
                f"({moment.sentiment}, {moment.confidence:.2f}): "
                f"{moment.preference} Rationale: {moment.rationale}.{frame_note}"
            )
    return "\n".join(lines).strip() + "\n"


def build_design_preferences_markdown(analysis: TasteAnalysis) -> str:
    positives = [
        moment.preference
        for moment in analysis.preference_moments
        if moment.sentiment == "positive"
    ]
    negatives = analysis.visual_details.negative_preferences + [
        moment.preference
        for moment in analysis.preference_moments
        if moment.sentiment == "negative"
    ]
    reusable_rules = []
    reusable_rules.extend(analysis.visual_details.information_hierarchy)
    reusable_rules.extend(analysis.motion_details.interaction_details)
    reusable_rules.extend(analysis.motion_details.motion_preferences)

    return (
        "# Design Preferences\n\n"
        "## Positive Preferences\n"
        f"{_bullet_list(positives)}\n"
        "## Negative Preferences\n"
        f"{_bullet_list(negatives)}\n"
        "## Visual Style\n"
        f"{_bullet_list(analysis.visual_details.style)}\n"
        "## Layout Preferences\n"
        f"{_bullet_list(analysis.visual_details.layout)}\n"
        "## Information Hierarchy\n"
        f"{_bullet_list(analysis.visual_details.information_hierarchy)}\n"
        "## Typography Preferences\n"
        f"{_bullet_list(analysis.visual_details.typography)}\n"
        "## Color Preferences\n"
        f"{_bullet_list(analysis.visual_details.color)}\n"
        "## Motion Preferences\n"
        f"{_bullet_list(analysis.motion_details.motion_preferences)}\n"
        "## Interaction Details\n"
        f"{_bullet_list(analysis.motion_details.interaction_details)}\n"
        "## Dashboard Preferences\n"
        f"{_bullet_list(analysis.visual_details.dashboard)}\n"
        "## Presentation Preferences\n"
        f"{_bullet_list(analysis.visual_details.presentation)}\n"
        "## Reusable Design Rules\n"
        f"{_bullet_list(reusable_rules)}"
    )


def generate_artifacts(
    output_dir: Path,
    analysis: TasteAnalysis,
    frame_map: dict[float, str],
    config: TastepackConfig,
    source_video_name: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    taste_packet = build_taste_packet_markdown(analysis, frame_map, source_video_name)
    design_preferences = build_design_preferences_markdown(analysis)
    (output_dir / "taste_packet.md").write_text(taste_packet, encoding="utf-8")
    (output_dir / "design_preferences.md").write_text(design_preferences, encoding="utf-8")
    (output_dir / "transcript.md").write_text(
        f"# Transcript\n\n{analysis.transcript}\n",
        encoding="utf-8",
    )
    metadata = {
        "source_video": source_video_name,
        "gemini_model": config.gemini_model,
        "assets_count": len(analysis.assets),
        "preference_moments_count": len(analysis.preference_moments),
        "frames": frame_map,
        "config": config.model_dump(exclude={"request_timeout_seconds"}),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if config.produce_pdf:
        generate_pdf_from_markdown(taste_packet, output_dir / "taste_packet.pdf")
