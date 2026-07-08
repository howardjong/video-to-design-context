from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

SUPPORTED_EXTENSIONS = {".mp4", ".mov"}


class VideoValidationError(ValueError):
    """Raised when the input video or system dependencies are invalid."""


def validate_input_video(video_path: Path, require_tools: bool = True) -> None:
    if not video_path.exists():
        raise VideoValidationError(f"Input video does not exist: {video_path}")
    if video_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported video extension: {video_path.suffix}. Expected .mp4 or .mov"
        )
    if require_tools:
        missing = [tool for tool in ("ffmpeg", "ffprobe") if shutil.which(tool) is None]
        if missing:
            raise VideoValidationError(f"Missing required system dependency: {', '.join(missing)}")


def probe_duration_seconds(video_path: Path) -> float | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None
