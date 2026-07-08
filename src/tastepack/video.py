from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tastepack.config import TastepackConfig

SUPPORTED_EXTENSIONS = {".mp4", ".mov"}


class VideoValidationError(ValueError):
    """Raised when the input video or system dependencies are invalid."""


def validate_input_video(
    video_path: Path,
    require_tools: bool = True,
    config: TastepackConfig | None = None,
) -> dict[str, Any]:
    if not video_path.exists():
        raise VideoValidationError(f"Input video does not exist: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video extension: {video_path.suffix}. Expected .mp4 or .mov"
        )
    config = config or TastepackConfig()
    file_size = video_path.stat().st_size
    if file_size > config.max_file_size_bytes:
        raise VideoValidationError(
            f"Input video is larger than the Gemini Files API limit "
            f"({file_size} bytes > {config.max_file_size_bytes} bytes)"
        )
    if require_tools:
        missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
        if missing:
            raise VideoValidationError(f"Missing required system dependency: {', '.join(missing)}")
        metadata = probe_video_metadata(video_path)
        metadata["file_size_bytes"] = file_size
        _validate_metadata(metadata, config)
        return metadata
    return {
        "duration_seconds": None,
        "width": None,
        "height": None,
        "video_stream_count": None,
        "audio_stream_count": None,
        "video_codec": None,
        "audio_codec": None,
        "file_size_bytes": file_size,
    }


def _validate_metadata(metadata: dict[str, Any], config: TastepackConfig) -> None:
    duration = metadata.get("duration_seconds")
    if duration is None or duration <= 0:
        raise VideoValidationError("Input video has no positive duration")
    if duration > config.max_duration_seconds:
        raise VideoValidationError(
            f"Input video is longer than the configured limit "
            f"({duration:.3f}s > {config.max_duration_seconds:.3f}s)"
        )
    if not metadata.get("video_stream_count"):
        raise VideoValidationError("Input file has no video stream")
    if not metadata.get("audio_stream_count") and not config.allow_no_audio:
        raise VideoValidationError(
            "Input video has no audio stream. Use --allow-no-audio for visual-only runs."
        )


def probe_video_metadata(video_path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        (
            "format=duration:stream=index,codec_type,codec_name,width,height"
        ),
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffprobe failed"
        raise VideoValidationError(f"Could not inspect video with ffprobe: {message}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise VideoValidationError("Could not inspect video with ffprobe: invalid JSON") from exc

    streams = payload.get("streams") or []
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    duration_raw = (payload.get("format") or {}).get("duration")
    try:
        duration = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise VideoValidationError("Could not inspect video duration with ffprobe") from exc
    first_video = video_streams[0] if video_streams else {}
    first_audio = audio_streams[0] if audio_streams else {}
    return {
        "duration_seconds": duration,
        "width": first_video.get("width"),
        "height": first_video.get("height"),
        "video_stream_count": len(video_streams),
        "audio_stream_count": len(audio_streams),
        "video_codec": first_video.get("codec_name"),
        "audio_codec": first_audio.get("codec_name"),
        "file_size_bytes": video_path.stat().st_size,
    }


def probe_duration_seconds(video_path: Path) -> float | None:
    try:
        return probe_video_metadata(video_path)["duration_seconds"]
    except VideoValidationError:
        return None
