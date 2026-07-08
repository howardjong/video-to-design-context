from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class TastepackConfig(BaseModel):
    gemini_model: str = "gemini-3.5-flash"
    frame_confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    max_frames_per_asset: int = Field(default=6, ge=1)
    max_total_frames: int = Field(default=24, ge=1)
    produce_pdf: bool = True
    fallback_interval_seconds: float = Field(default=2.0, gt=0)
    verbosity: Literal["quiet", "normal", "debug"] = "normal"
    request_timeout_seconds: float = Field(default=120.0, gt=0)

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
