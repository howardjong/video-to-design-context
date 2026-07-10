from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TastepackConfig(BaseModel):
    gemini_model: str = "gemini-3.5-flash"
    frame_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    max_frames_per_asset: int = Field(default=6, ge=1)
    max_total_frames: int = Field(default=24, ge=1)
    produce_pdf: bool = True
    fallback_interval_seconds: float = Field(default=2.0, gt=0)
    verbosity: Literal["quiet", "normal", "debug"] = "normal"
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    max_duration_seconds: float = Field(default=1800.0, gt=0)
    max_file_size_bytes: int = Field(default=2_147_483_648, gt=0)
    allow_no_audio: bool = False
    ffprobe_timeout_seconds: float = Field(default=30.0, gt=0)
    ffmpeg_timeout_seconds: float = Field(default=600.0, gt=0)
    frame_extraction_timeout_seconds: float = Field(default=30.0, gt=0)
    min_audio_mean_volume_db: float = Field(default=-60.0, le=0)
    gemini_max_retries: int = Field(default=3, ge=1)
    gemini_retry_base_delay_seconds: float = Field(default=1.0, gt=0)
    gemini_retry_jitter_seconds: float = Field(default=0.25, ge=0)
    gemini_upload_timeout_seconds: float = Field(default=600.0, gt=0)
    gemini_file_processing_timeout_seconds: float = Field(default=600.0, gt=0)
    gemini_generation_timeout_seconds: float = Field(default=300.0, gt=0)
    gemini_cleanup_timeout_seconds: float = Field(default=30.0, gt=0)
    cleanup_uploaded_files: bool = True

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_request_timeout(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if (
            "gemini_file_processing_timeout_seconds" not in values
            and "request_timeout_seconds" in values
        ):
            values = dict(values)
            values["gemini_file_processing_timeout_seconds"] = values["request_timeout_seconds"]
        return values

    @classmethod
    def from_sources(
        cls,
        config_path: Path | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> TastepackConfig:
        values: dict[str, Any] = {}
        if config_path:
            with config_path.open("r", encoding="utf-8") as handle:
                values.update(json.load(handle))
        if overrides:
            values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)
