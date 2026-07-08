from __future__ import annotations

import subprocess
from pathlib import Path

from tastepack.config import TastepackConfig
from tastepack.schema import AssetExample, SuggestedFrame, TasteAnalysis


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract a frame."""


def select_frames(
    frames: list[SuggestedFrame],
    config: TastepackConfig,
    video_duration_seconds: float | None = None,
) -> list[SuggestedFrame]:
    by_key: dict[tuple[str, float], SuggestedFrame] = {}
    asset_counts: dict[str, int] = {}
    for frame in sorted(frames, key=lambda item: item.confidence, reverse=True):
        if frame.confidence < config.frame_confidence_threshold:
            continue
        seconds = frame.timestamp_seconds
        if video_duration_seconds is not None:
            seconds = min(seconds, video_duration_seconds)
        key = (frame.asset_id, round(seconds, 3))
        if key in by_key:
            continue
        if asset_counts.get(frame.asset_id, 0) >= config.max_frames_per_asset:
            continue
        frame.timestamp_seconds = seconds
        by_key[key] = frame
        asset_counts[frame.asset_id] = asset_counts.get(frame.asset_id, 0) + 1
        if len(by_key) >= config.max_total_frames:
            break
    return sorted(by_key.values(), key=lambda item: (item.asset_id, item.timestamp_seconds))


def select_frames_for_analysis(
    analysis: TasteAnalysis,
    config: TastepackConfig,
    video_duration_seconds: float | None = None,
) -> list[SuggestedFrame]:
    selected = select_frames(analysis.suggested_frames, config, video_duration_seconds)
    asset_ranges = {asset.id: asset for asset in analysis.assets}
    by_key: dict[tuple[str, float], SuggestedFrame] = {}
    for frame in selected:
        asset = asset_ranges.get(frame.asset_id)
        if asset:
            frame.timestamp_seconds = min(
                max(frame.timestamp_seconds, asset.start_seconds),
                asset.end_seconds,
            )
        key = (frame.asset_id, round(frame.timestamp_seconds, 3))
        current = by_key.get(key)
        if current is None or frame.confidence > current.confidence:
            by_key[key] = frame
    return sorted(by_key.values(), key=lambda item: (item.asset_id, item.timestamp_seconds))


def build_fallback_frames(
    assets: list[AssetExample],
    config: TastepackConfig,
) -> list[SuggestedFrame]:
    frames: list[SuggestedFrame] = []
    for asset in assets:
        seconds = asset.start_seconds
        asset_count = 0
        while seconds <= asset.end_seconds and asset_count < config.max_frames_per_asset:
            frame = SuggestedFrame(
                asset_id=asset.id,
                timestamp=seconds,
                reason="Fallback interval frame",
                confidence=config.frame_confidence_threshold,
            )
            frames.append(frame)
            if len(frames) >= config.max_total_frames:
                return frames
            seconds += config.fallback_interval_seconds
            asset_count += 1
    return frames


def frame_filename(frame: SuggestedFrame) -> str:
    millis = int(round(frame.timestamp_seconds * 1000))
    safe_asset_id = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in frame.asset_id
    )
    return f"{safe_asset_id}_{millis:09d}.jpg"


def build_ffmpeg_extract_command(
    video_path: Path,
    timestamp_seconds: float,
    output_path: Path,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-ss",
        f"{timestamp_seconds:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]


def extract_frames(
    video_path: Path,
    frames: list[SuggestedFrame],
    output_dir: Path,
    skip_ffmpeg: bool = False,
) -> dict[float, str]:
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_map: dict[float, str] = {}
    for frame in frames:
        relative_path = Path("frames") / frame_filename(frame)
        destination = output_dir / relative_path
        if skip_ffmpeg:
            destination.write_text(
                f"Mock frame for {frame.asset_id} at {frame.timestamp_seconds:.3f}\n",
                encoding="utf-8",
            )
        else:
            command = build_ffmpeg_extract_command(video_path, frame.timestamp_seconds, destination)
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise FrameExtractionError(completed.stderr.strip() or "ffmpeg extraction failed")
            if not destination.exists():
                raise FrameExtractionError(f"Extracted frame is missing: {destination}")
            if destination.stat().st_size == 0:
                raise FrameExtractionError(f"Extracted frame is empty: {destination}")
        frame_map[round(frame.timestamp_seconds, 3)] = str(relative_path)
    return frame_map
