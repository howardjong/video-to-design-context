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

    @field_validator("id")
    @classmethod
    def reject_blank_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Asset ID cannot be blank")
        return value

    @model_validator(mode="after")
    def normalize_range(self) -> AssetExample:
        try:
            self.start_seconds = normalize_timestamp(self.start_timestamp)
            self.end_seconds = normalize_timestamp(self.end_timestamp)
        except TimestampError as exc:
            raise ValueError(str(exc)) from exc
        if self.end_seconds <= self.start_seconds:
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
    def ensure_asset_references_exist(self, info: ValidationInfo) -> TasteAnalysis:
        if not self.assets:
            raise ValueError("Analysis must identify at least one asset")
        if not self.preference_moments:
            raise ValueError("Analysis must identify at least one preference moment")
        assets_by_id = {asset.id: asset for asset in self.assets}
        if len(assets_by_id) != len(self.assets):
            raise ValueError("Asset IDs must be unique")
        context = info.context or {}
        video_duration_seconds = context.get("video_duration_seconds")
        if context.get("require_transcript") and not self.transcript.strip():
            raise ValueError("Analysis transcript is required for narrated video")
        if video_duration_seconds is not None:
            for asset in self.assets:
                if asset.end_seconds > video_duration_seconds:
                    raise ValueError("Asset range is outside video duration")
        for moment in self.preference_moments:
            asset = assets_by_id.get(moment.asset_id)
            if asset is None:
                raise ValueError(f"Preference references unknown asset: {moment.asset_id}")
            if not asset.start_seconds <= moment.timestamp_seconds <= asset.end_seconds:
                raise ValueError("Preference timestamp is outside asset range")
        for frame in self.suggested_frames:
            asset = assets_by_id.get(frame.asset_id)
            if asset is None:
                raise ValueError(f"Frame references unknown asset: {frame.asset_id}")
            if not asset.start_seconds <= frame.timestamp_seconds <= asset.end_seconds:
                raise ValueError("Frame timestamp is outside asset range")
        return self
