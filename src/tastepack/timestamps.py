from __future__ import annotations

from typing import Any


class TimestampError(ValueError):
    """Raised when a timestamp cannot be normalized."""


def normalize_timestamp(raw: Any) -> float:
    if isinstance(raw, bool):
        raise TimestampError("Boolean values are not valid timestamps")
    if isinstance(raw, int | float):
        seconds = float(raw)
        if seconds < 0:
            raise TimestampError("Timestamp cannot be negative")
        return seconds
    if not isinstance(raw, str):
        raise TimestampError(f"Unsupported timestamp value: {raw!r}")

    value = raw.strip()
    if not value:
        raise TimestampError("Timestamp cannot be blank")
    if value.startswith("-"):
        raise TimestampError("Timestamp cannot be negative")
    if ":" not in value:
        try:
            return normalize_timestamp(float(value))
        except ValueError as exc:
            raise TimestampError(f"Invalid timestamp: {raw}") from exc

    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise TimestampError(f"Invalid timestamp: {raw}")
    try:
        numeric = [float(part) for part in parts]
    except ValueError as exc:
        raise TimestampError(f"Invalid timestamp: {raw}") from exc

    if any(part < 0 for part in numeric):
        raise TimestampError("Timestamp cannot be negative")
    if len(numeric) == 2:
        minutes, seconds = numeric
        hours = 0.0
    else:
        hours, minutes, seconds = numeric
    if minutes >= 60 or seconds >= 60:
        raise TimestampError(f"Invalid timestamp: {raw}")
    return (hours * 3600) + (minutes * 60) + seconds


def format_timestamp(seconds: float) -> str:
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    whole_seconds = int(seconds)
    if milliseconds == 1000:
        whole_seconds += 1
        milliseconds = 0
    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
