from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from tastepack.config import TastepackConfig
from tastepack.logging import get_logger
from tastepack.schema import AssetExample, SuggestedFrame, TasteAnalysis
from tastepack.video import VideoValidationError, assert_source_unchanged


class FrameExtractionError(RuntimeError):
    """Raised when ffmpeg cannot extract a frame."""


logger = get_logger("frames")
END_OF_VIDEO_EPSILON_SECONDS = 0.1


@dataclass(frozen=True)
class ExtractedFrame:
    id: str
    asset_id: str
    timestamp_seconds: float
    relative_path: str
    reason: str
    confidence: float

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


def select_frames(
    frames: list[SuggestedFrame],
    config: TastepackConfig,
    video_duration_seconds: float | None = None,
) -> list[SuggestedFrame]:
    candidates_by_asset: dict[str, list[SuggestedFrame]] = {}
    seen_keys: set[tuple[str, float]] = set()
    for frame in sorted(frames, key=lambda item: item.confidence, reverse=True):
        if frame.confidence < config.frame_confidence_threshold:
            continue
        seconds = frame.timestamp_seconds
        if video_duration_seconds is not None:
            safe_end = max(0.0, video_duration_seconds - END_OF_VIDEO_EPSILON_SECONDS)
            seconds = min(seconds, safe_end)
        key = (frame.asset_id, round(seconds, 3))
        if key in seen_keys:
            continue
        candidates = candidates_by_asset.setdefault(frame.asset_id, [])
        if len(candidates) >= config.max_frames_per_asset:
            continue
        frame.timestamp_seconds = seconds
        candidates.append(frame)
        seen_keys.add(key)

    selected: list[SuggestedFrame] = []
    selected_keys: set[tuple[str, float]] = set()
    first_candidates = sorted(
        (candidates[0] for candidates in candidates_by_asset.values()),
        key=lambda item: item.confidence,
        reverse=True,
    )
    for frame in first_candidates:
        if len(selected) >= config.max_total_frames:
            break
        selected.append(frame)
        selected_keys.add((frame.asset_id, round(frame.timestamp_seconds, 3)))

    remaining = sorted(
        (frame for candidates in candidates_by_asset.values() for frame in candidates),
        key=lambda item: item.confidence,
        reverse=True,
    )
    for frame in remaining:
        if len(selected) >= config.max_total_frames:
            break
        key = (frame.asset_id, round(frame.timestamp_seconds, 3))
        if key not in selected_keys:
            selected.append(frame)
            selected_keys.add(key)
    return sorted(selected, key=lambda item: (item.asset_id, item.timestamp_seconds))


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
            max_seconds = asset.end_seconds
            if video_duration_seconds is not None:
                max_seconds = min(
                    max_seconds,
                    max(0.0, video_duration_seconds - END_OF_VIDEO_EPSILON_SECONDS),
                )
            if asset.start_seconds > max_seconds:
                continue
            frame.timestamp_seconds = min(
                max(frame.timestamp_seconds, asset.start_seconds),
                max_seconds,
            )
        key = (frame.asset_id, round(frame.timestamp_seconds, 3))
        current = by_key.get(key)
        if current is None or frame.confidence > current.confidence:
            by_key[key] = frame
    remaining_capacity = config.max_total_frames - len(by_key)
    if remaining_capacity > 0:
        fallback_frames = build_fallback_frames(
            analysis.assets,
            config,
            covered_asset_ids={frame.asset_id for frame in by_key.values()},
            max_total_frames=remaining_capacity,
            video_duration_seconds=video_duration_seconds,
        )
        for frame in fallback_frames:
            by_key.setdefault((frame.asset_id, round(frame.timestamp_seconds, 3)), frame)
    return sorted(by_key.values(), key=lambda item: (item.asset_id, item.timestamp_seconds))


def build_fallback_frames(
    assets: list[AssetExample],
    config: TastepackConfig,
    *,
    covered_asset_ids: set[str] | None = None,
    max_total_frames: int | None = None,
    video_duration_seconds: float | None = None,
) -> list[SuggestedFrame]:
    fallback_candidates: list[SuggestedFrame] = []
    covered_asset_ids = covered_asset_ids or set()
    for asset in assets:
        if asset.id in covered_asset_ids:
            continue
        end_seconds = asset.end_seconds
        if video_duration_seconds is not None:
            end_seconds = min(
                end_seconds,
                max(0.0, video_duration_seconds - END_OF_VIDEO_EPSILON_SECONDS),
            )
        if asset.start_seconds > end_seconds:
            continue
        seconds = asset.start_seconds
        asset_count = 0
        while seconds <= end_seconds and asset_count < config.max_frames_per_asset:
            frame = SuggestedFrame(
                asset_id=asset.id,
                timestamp=seconds,
                reason="Fallback interval frame",
                confidence=config.frame_confidence_threshold,
            )
            fallback_candidates.append(frame)
            seconds += config.fallback_interval_seconds
            asset_count += 1
    fallback_config = config.model_copy(
        update={"max_total_frames": max_total_frames or config.max_total_frames}
    )
    return select_frames(fallback_candidates, fallback_config, video_duration_seconds)


def build_coverage_frames(
    video_duration_seconds: float,
    config: TastepackConfig,
) -> list[SuggestedFrame]:
    """Return deterministic, model-independent frames for visual QA coverage."""
    safe_end = round(max(0.0, video_duration_seconds - END_OF_VIDEO_EPSILON_SECONDS), 3)
    timestamps: list[float] = [0.0]
    next_timestamp = config.qa_coverage_interval_seconds
    while next_timestamp < safe_end:
        timestamps.append(next_timestamp)
        next_timestamp += config.qa_coverage_interval_seconds
    if safe_end > 0 and safe_end not in timestamps:
        timestamps.append(safe_end)
    return [
        SuggestedFrame(
            asset_id="qa-coverage",
            timestamp=timestamp,
            reason="Independent QA coverage frame",
            confidence=1.0,
        )
        for timestamp in timestamps
    ]


def frame_filename(frame: SuggestedFrame) -> str:
    millis = int(round(frame.timestamp_seconds * 1000))
    safe_asset_id = "".join(
        char if char.isalnum() or char in "-_" else "-" for char in frame.asset_id
    )
    asset_digest = sha256(frame.asset_id.encode("utf-8")).hexdigest()[:10]
    return f"{safe_asset_id or 'asset'}-{asset_digest}_{millis:09d}.jpg"


def frame_id(frame: SuggestedFrame) -> str:
    payload = f"{frame.asset_id}\0{frame.timestamp_seconds:.3f}".encode()
    return f"frame-{sha256(payload).hexdigest()[:16]}"


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
        "-vf",
        "format=yuvj420p",
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output_path),
    ]


def validate_extracted_jpeg(destination: Path, relative_path: Path) -> None:
    try:
        with Image.open(destination) as image:
            if image.format != "JPEG":
                raise FrameExtractionError(f"Extracted frame is not a JPEG: {relative_path}")
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise FrameExtractionError(f"Extracted frame is an invalid JPEG: {relative_path}") from exc


def extract_frames(
    video_path: Path,
    frames: list[SuggestedFrame],
    output_dir: Path,
    skip_ffmpeg: bool = False,
    expected_source_metadata: dict[str, object] | None = None,
    ffmpeg_timeout_seconds: float = 30.0,
    relative_directory: Path = Path("frames"),
) -> list[ExtractedFrame]:
    if expected_source_metadata is not None:
        try:
            assert_source_unchanged(video_path, expected_source_metadata)
        except VideoValidationError as exc:
            raise FrameExtractionError(str(exc)) from exc
    frames_dir = output_dir / relative_directory
    frames_dir.mkdir(parents=True, exist_ok=True)
    extracted_frames: list[ExtractedFrame] = []
    for frame in frames:
        relative_path = relative_directory / frame_filename(frame)
        destination = output_dir / relative_path
        if skip_ffmpeg:
            destination.write_text(
                f"Mock frame for {frame.asset_id} at {frame.timestamp_seconds:.3f}\n",
                encoding="utf-8",
            )
        else:
            command = build_ffmpeg_extract_command(video_path, frame.timestamp_seconds, destination)
            logger.debug("Running ffmpeg frame extraction command: %s", command)
            try:
                completed = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=ffmpeg_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise FrameExtractionError(
                    "ffmpeg timed out while extracting frame "
                    f"at {frame.timestamp_seconds:.3f}s to {relative_path} "
                    f"after {ffmpeg_timeout_seconds:.1f}s"
                ) from exc
            if completed.returncode != 0:
                stderr = completed.stderr.strip() or "no stderr"
                raise FrameExtractionError(
                    "ffmpeg failed while extracting frame "
                    f"at {frame.timestamp_seconds:.3f}s to {relative_path}: {stderr}"
                )
            if not destination.exists():
                raise FrameExtractionError(
                    f"Extracted frame is missing after ffmpeg success at "
                    f"{frame.timestamp_seconds:.3f}s: {relative_path}"
                )
            if destination.stat().st_size == 0:
                raise FrameExtractionError(
                    f"Extracted frame is empty after ffmpeg success at "
                    f"{frame.timestamp_seconds:.3f}s: {relative_path}"
                )
            validate_extracted_jpeg(destination, relative_path)
        extracted_frames.append(
            ExtractedFrame(
                id=frame_id(frame),
                asset_id=frame.asset_id,
                timestamp_seconds=frame.timestamp_seconds,
                relative_path=str(relative_path),
                reason=frame.reason,
                confidence=frame.confidence,
            )
        )
    return extracted_frames
