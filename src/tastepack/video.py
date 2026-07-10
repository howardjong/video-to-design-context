from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path
from typing import Any

from tastepack.config import TastepackConfig

SUPPORTED_EXTENSIONS = {".mp4", ".mov"}


class VideoValidationError(ValueError):
    """Raised when the input video or system dependencies are invalid."""


def source_fingerprint(video_path: Path) -> dict[str, int]:
    stat = video_path.stat()
    return {
        "file_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
    }


def source_sha256(video_path: Path) -> str:
    digest = sha256()
    with video_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_source_metadata(
    video_path: Path,
    expected_fingerprint: dict[str, int],
) -> dict[str, Any]:
    source_hash = source_sha256(video_path)
    if source_fingerprint(video_path) != expected_fingerprint:
        raise VideoValidationError("Input video changed during preflight")
    return {**expected_fingerprint, "source_sha256": source_hash}


def assert_source_unchanged(video_path: Path, expected_metadata: dict[str, Any]) -> None:
    expected_size = expected_metadata.get("file_size_bytes")
    expected_mtime = expected_metadata.get("source_mtime_ns")
    actual = source_fingerprint(video_path)
    if actual["file_size_bytes"] != expected_size or actual["source_mtime_ns"] != expected_mtime:
        raise VideoValidationError(
            "Input video changed after preflight "
            f"(size {expected_size} -> {actual['file_size_bytes']}, "
            f"mtime {expected_mtime} -> {actual['source_mtime_ns']})"
        )


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
    if video_path.is_symlink() or not video_path.is_file():
        raise VideoValidationError(f"Input video is not a regular file: {video_path}")
    fingerprint = source_fingerprint(video_path)
    file_size = fingerprint["file_size_bytes"]
    if file_size > config.max_file_size_bytes:
        raise VideoValidationError(
            f"Input video is larger than the Gemini Files API limit "
            f"({file_size} bytes > {config.max_file_size_bytes} bytes)"
        )
    if require_tools:
        missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
        if missing:
            raise VideoValidationError(f"Missing required system dependency: {', '.join(missing)}")
        metadata = probe_video_metadata(video_path, config.ffprobe_timeout_seconds)
        metadata.update(fingerprint)
        _validate_metadata(metadata, config)
        validate_media_decode(video_path, metadata, config)
        metadata.update(_stable_source_metadata(video_path, fingerprint))
        return metadata
    return {
        "duration_seconds": None,
        "width": None,
        "height": None,
        "video_stream_count": None,
        "audio_stream_count": None,
        "video_codec": None,
        "audio_codec": None,
        **_stable_source_metadata(video_path, fingerprint),
    }


def _validate_metadata(metadata: dict[str, Any], config: TastepackConfig) -> None:
    duration = metadata.get("duration_seconds")
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not math.isfinite(duration)
    ):
        raise VideoValidationError("Input video has an invalid duration")
    if duration <= 0:
        raise VideoValidationError("Input video has no positive duration")
    if duration > config.max_duration_seconds:
        raise VideoValidationError(
            f"Input video is longer than the configured limit "
            f"({duration:.3f}s > {config.max_duration_seconds:.3f}s)"
        )
    if not metadata.get("video_stream_count"):
        raise VideoValidationError("Input file has no video stream")
    width = metadata.get("width")
    height = metadata.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        raise VideoValidationError("Input video has invalid dimensions")
    if not isinstance(metadata.get("video_codec"), str) or not metadata["video_codec"].strip():
        raise VideoValidationError("Input video has no usable video codec")
    if not metadata.get("audio_stream_count") and not config.allow_no_audio:
        raise VideoValidationError(
            "Input video has no audio stream. Use --allow-no-audio for visual-only runs."
        )
    if metadata.get("audio_stream_count") and (
        not isinstance(metadata.get("audio_codec"), str) or not metadata["audio_codec"].strip()
    ):
        raise VideoValidationError("Input video has no usable audio codec")


def probe_video_metadata(video_path: Path, timeout_seconds: float = 30.0) -> dict[str, Any]:
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
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoValidationError(
            f"ffprobe timed out after {timeout_seconds:.1f}s while inspecting input video"
        ) from exc
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


def _run_ffmpeg_validation(
    command: list[str],
    timeout_seconds: float,
    label: str,
) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoValidationError(f"{label} timed out after {timeout_seconds:.1f}s") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "no ffmpeg diagnostics"
        raise VideoValidationError(f"{label} failed: {stderr}")
    return completed


def validate_media_decode(
    video_path: Path,
    metadata: dict[str, Any],
    config: TastepackConfig,
) -> None:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-xerror",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
    ]
    if metadata.get("audio_stream_count") and not config.allow_no_audio:
        command.extend(["-map", "0:a:0"])
    command.extend(["-f", "null", "-"])
    _run_ffmpeg_validation(
        command,
        config.ffmpeg_timeout_seconds,
        "Full media decode validation",
    )

    if metadata.get("audio_stream_count") and not config.allow_no_audio:
        loudness_command = [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-v",
            "info",
            "-i",
            str(video_path),
            "-map",
            "0:a:0",
            "-af",
            "volumedetect",
            "-f",
            "null",
            "-",
        ]
        completed = _run_ffmpeg_validation(
            loudness_command,
            config.ffmpeg_timeout_seconds,
            "Audio loudness validation",
        )
        match = re.search(r"mean_volume:\s*(-?(?:\d+(?:\.\d+)?|inf))\s*dB", completed.stderr)
        if not match:
            raise VideoValidationError("Could not determine audio loudness during preflight")
        volume_text = match.group(1)
        mean_volume = -math.inf if volume_text == "-inf" else float(volume_text)
        metadata["audio_mean_volume_db"] = mean_volume
        if mean_volume < config.min_audio_mean_volume_db:
            raise VideoValidationError(
                "Input audio is effectively silent "
                f"({mean_volume:.1f} dB < {config.min_audio_mean_volume_db:.1f} dB). "
                "Use --allow-no-audio only for intentionally visual-only runs."
            )
    else:
        metadata["audio_mean_volume_db"] = None


def mux_video_with_companion_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    config: TastepackConfig,
) -> None:
    """Create and fully validate the one MP4 sent to Gemini for a sidecar-audio bundle."""
    if audio_path.is_symlink() or not audio_path.is_file():
        raise VideoValidationError(f"Companion audio is not a regular file: {audio_path.name}")
    missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
    if missing:
        raise VideoValidationError(f"Missing required system dependency: {', '.join(missing)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _run_ffmpeg_validation(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-xerror",
                "-i",
                str(video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            config.ffmpeg_timeout_seconds,
            "Companion audio mux",
        )
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise VideoValidationError("Companion audio mux produced no analysis MP4")
        validate_input_video(output_path, config=config)
    except Exception:
        output_path.unlink(missing_ok=True)
        raise
