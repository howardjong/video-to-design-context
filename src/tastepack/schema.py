from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from tastepack.timestamps import TimestampError, normalize_timestamp

VAGUE_TERMS = {"clean", "modern", "sleek", "nice", "beautiful", "polished"}


def _contains_only_vague_language(text: str) -> bool:
    tokens = {token.strip(" .,!?:;").lower() for token in text.split()}
    meaningful = {token for token in tokens if token}
    return bool(meaningful) and meaningful.issubset(VAGUE_TERMS)


class AssetExample(BaseModel):
    id: str
    name: str
    kind: str
    start_timestamp: str | int | float
    end_timestamp: str | int | float
    summary: str
    start_seconds: float = Field(default=0.0)
    end_seconds: float = Field(default=0.0)

    @model_validator(mode="after")
    def normalize_range(self) -> AssetExample:
        try:
            self.start_seconds = normalize_timestamp(self.start_timestamp)
            self.end_seconds = normalize_timestamp(self.end_timestamp)
        except TimestampError as exc:
            raise ValueError(str(exc)) from exc
        if self.end_seconds < self.start_seconds:
            raise ValueError("Asset end timestamp must be after start timestamp")
        return self


class PreferenceMoment(BaseModel):
    asset_id: str
    timestamp: str | int | float
    sentiment: Literal["positive", "negative", "mixed"]
    preference: str
    rationale: str
    categories: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    timestamp_seconds: float = Field(default=0.0)

    @field_validator("preference", "rationale")
    @classmethod
    def reject_generic_language(cls, value: str, info: ValidationInfo) -> str:
        if _contains_only_vague_language(value):
            raise ValueError(
                f"{info.field_name} must explain specific design properties, not vague labels"
            )
        return value

    @model_validator(mode="after")
    def normalize_time(self) -> PreferenceMoment:
        try:
            self.timestamp_seconds = normalize_timestamp(self.timestamp)
        except TimestampError as exc:
            raise ValueError(str(exc)) from exc
        return self


class SuggestedFrame(BaseModel):
    asset_id: str
    timestamp: str | int | float
    reason: str
    confidence: float = Field(ge=0, le=1)
    timestamp_seconds: float = Field(default=0.0)

    @model_validator(mode="after")
    def normalize_time(self) -> SuggestedFrame:
        try:
            self.timestamp_seconds = normalize_timestamp(self.timestamp)
        except TimestampError as exc:
            raise ValueError(str(exc)) from exc
        return self


class VisualDetails(BaseModel):
    style: list[str] = Field(default_factory=list)
    layout: list[str] = Field(default_factory=list)
    information_hierarchy: list[str] = Field(default_factory=list)
    typography: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)
    dashboard: list[str] = Field(default_factory=list)
    presentation: list[str] = Field(default_factory=list)
    negative_preferences: list[str] = Field(default_factory=list)


class MotionDetails(BaseModel):
    animations: list[str] = Field(default_factory=list)
    interaction_details: list[str] = Field(default_factory=list)
    motion_preferences: list[str] = Field(default_factory=list)


class TasteAnalysis(BaseModel):
    source_summary: str
    transcript: str
    assets: list[AssetExample]
    preference_moments: list[PreferenceMoment]
    suggested_frames: list[SuggestedFrame]
    visual_details: VisualDetails
    motion_details: MotionDetails

    @model_validator(mode="after")
    def ensure_asset_references_exist(self) -> TasteAnalysis:
        asset_ids = {asset.id for asset in self.assets}
        for moment in self.preference_moments:
            if moment.asset_id not in asset_ids:
                raise ValueError(f"Preference references unknown asset: {moment.asset_id}")
        for frame in self.suggested_frames:
            if frame.asset_id not in asset_ids:
                raise ValueError(f"Frame references unknown asset: {frame.asset_id}")
        return self
