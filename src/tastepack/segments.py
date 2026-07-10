from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from tastepack.config import TastepackConfig
from tastepack.schema import AssetExample, PreferenceMoment, SuggestedFrame, TasteAnalysis


@dataclass(frozen=True)
class VideoSegment:
    index: int
    start_seconds: float
    end_seconds: float


def build_segment_plan(
    duration_seconds: float | None, config: TastepackConfig
) -> list[VideoSegment]:
    if duration_seconds is None:
        return [VideoSegment(index=0, start_seconds=0.0, end_seconds=0.0)]
    if config.analysis_segment_overlap_seconds >= config.analysis_segment_seconds:
        raise ValueError(
            "analysis_segment_overlap_seconds must be smaller than analysis_segment_seconds"
        )
    if duration_seconds <= config.analysis_segment_seconds:
        return [VideoSegment(index=0, start_seconds=0.0, end_seconds=duration_seconds)]
    segments = []
    start_seconds = 0.0
    index = 0
    while start_seconds < duration_seconds:
        end_seconds = min(start_seconds + config.analysis_segment_seconds, duration_seconds)
        segments.append(
            VideoSegment(index=index, start_seconds=start_seconds, end_seconds=end_seconds)
        )
        if end_seconds == duration_seconds:
            break
        start_seconds = end_seconds - config.analysis_segment_overlap_seconds
        index += 1
    return segments


def _normal(value: str) -> str:
    return " ".join(value.lower().split())


def _overlaps(first: AssetExample, second: AssetExample) -> bool:
    return first.start_seconds <= second.end_seconds and second.start_seconds <= first.end_seconds


def _collision_id(asset: AssetExample) -> str:
    identity = "|".join(
        (
            asset.id,
            _normal(asset.name),
            _normal(asset.kind),
            f"{asset.start_seconds:.3f}",
            f"{asset.end_seconds:.3f}",
        )
    )
    return f"{asset.id}-{sha256(identity.encode()).hexdigest()[:8]}"


def _merge_assets(analyses: list[TasteAnalysis]) -> tuple[list[AssetExample], list[dict[str, str]]]:
    merged_assets: list[AssetExample] = []
    mappings: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for analysis in analyses:
        mapping: dict[str, str] = {}
        for asset in analysis.assets:
            matching_asset = next(
                (
                    current
                    for current in merged_assets
                    if current.kind == asset.kind
                    and _normal(current.name) == _normal(asset.name)
                    and _overlaps(current, asset)
                ),
                None,
            )
            if matching_asset is not None:
                matching_asset.start_seconds = min(
                    matching_asset.start_seconds, asset.start_seconds
                )
                matching_asset.end_seconds = max(matching_asset.end_seconds, asset.end_seconds)
                matching_asset.start_timestamp = matching_asset.start_seconds
                matching_asset.end_timestamp = matching_asset.end_seconds
                mapping[asset.id] = matching_asset.id
                continue
            final_id = asset.id
            if final_id in used_ids:
                final_id = _collision_id(asset)
            while final_id in used_ids:
                final_id = f"{final_id}-{len(used_ids)}"
            merged_assets.append(asset.model_copy(update={"id": final_id}))
            used_ids.add(final_id)
            mapping[asset.id] = final_id
        mappings.append(mapping)
    return merged_assets, mappings


def _unique_strings(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def merge_segment_analyses(analyses: list[TasteAnalysis]) -> TasteAnalysis:
    if not analyses:
        raise ValueError("At least one segment analysis is required")
    assets, mappings = _merge_assets(analyses)
    moments: list[PreferenceMoment] = []
    frames: dict[tuple[str, float], SuggestedFrame] = {}
    seen_moments: set[tuple[str, float, str, str, str]] = set()
    for analysis, mapping in zip(analyses, mappings, strict=True):
        for moment in analysis.preference_moments:
            remapped = moment.model_copy(update={"asset_id": mapping[moment.asset_id]})
            key = (
                remapped.asset_id,
                round(remapped.timestamp_seconds, 3),
                remapped.sentiment,
                _normal(remapped.preference),
                _normal(remapped.rationale),
            )
            if key not in seen_moments:
                seen_moments.add(key)
                moments.append(remapped)
        for frame in analysis.suggested_frames:
            remapped = frame.model_copy(update={"asset_id": mapping[frame.asset_id]})
            key = (remapped.asset_id, round(remapped.timestamp_seconds, 3))
            current = frames.get(key)
            if current is None or (remapped.confidence, remapped.reason) > (
                current.confidence,
                current.reason,
            ):
                frames[key] = remapped

    visual = analyses[0].visual_details.model_copy(
        update={
            field: _unique_strings(
                [item for analysis in analyses for item in getattr(analysis.visual_details, field)]
            )
            for field in type(analyses[0].visual_details).model_fields
        }
    )
    motion = analyses[0].motion_details.model_copy(
        update={
            field: _unique_strings(
                [item for analysis in analyses for item in getattr(analysis.motion_details, field)]
            )
            for field in type(analyses[0].motion_details).model_fields
        }
    )
    return TasteAnalysis.model_validate(
        {
            "source_summary": "\n\n".join(
                _unique_strings([analysis.source_summary for analysis in analyses])
            ),
            "transcript": "\n\n".join(
                _unique_strings([analysis.transcript for analysis in analyses])
            ),
            "assets": [asset.model_dump() for asset in assets],
            "preference_moments": [
                moment.model_dump()
                for moment in sorted(
                    moments, key=lambda item: (item.asset_id, item.timestamp_seconds)
                )
            ],
            "suggested_frames": [
                frame.model_dump()
                for frame in sorted(
                    frames.values(), key=lambda item: (item.asset_id, item.timestamp_seconds)
                )
            ],
            "visual_details": visual.model_dump(),
            "motion_details": motion.model_dump(),
        }
    )
