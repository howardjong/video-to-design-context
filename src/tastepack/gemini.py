from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from tastepack.config import TastepackConfig
from tastepack.schema import TasteAnalysis


class GeminiAnalysisError(RuntimeError):
    """Raised when Gemini analysis cannot produce valid structured output."""


MOCK_ANALYSIS: dict[str, Any] = {
    "source_summary": "Mock narrated review of one dashboard design.",
    "transcript": "At twelve seconds I like the compact hierarchy and table density.",
    "assets": [
        {
            "id": "asset-1",
            "name": "Mock dashboard",
            "kind": "dashboard",
            "start_timestamp": "00:00:00",
            "end_timestamp": "00:00:20",
            "summary": "A dashboard with a metric strip and operational table.",
        }
    ],
    "preference_moments": [
        {
            "asset_id": "asset-1",
            "timestamp": "00:00:12.000",
            "sentiment": "positive",
            "preference": "Likes compact grouping where labels sit close to values.",
            "rationale": "The arrangement shortens the scan path across the dashboard.",
            "categories": ["layout", "information_hierarchy", "dashboard"],
            "confidence": 0.9,
        }
    ],
    "suggested_frames": [
        {
            "asset_id": "asset-1",
            "timestamp": "00:00:12.000",
            "reason": "Shows the preference moment with compact metric grouping.",
            "confidence": 0.9,
        }
    ],
    "visual_details": {
        "style": ["Restrained interface chrome with dense operational content"],
        "layout": ["Metric strip sits above the table and shares the same grid rhythm"],
        "information_hierarchy": ["Metric labels stay subordinate to numeric values"],
        "typography": ["Numbers are emphasized with stable tabular spacing"],
        "color": ["Neutral surfaces use color only for status accents"],
        "dashboard": ["Summary metrics remain close to the detailed operational table"],
        "presentation": [],
        "negative_preferences": ["Avoid decorative cards that reduce data density"],
    },
    "motion_details": {
        "animations": ["Fast hover state"],
        "interaction_details": ["Row hover clarifies the selected target"],
        "motion_preferences": ["Prefer functional feedback over decorative animation"],
    },
}


def parse_gemini_json(raw: str) -> TasteAnalysis:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GeminiAnalysisError("Gemini returned malformed JSON") from exc
    try:
        return TasteAnalysis.model_validate(payload)
    except ValidationError as exc:
        raise GeminiAnalysisError(f"Gemini JSON failed validation: {exc}") from exc


def analyze_video(
    video_path: Path,
    config: TastepackConfig,
    mock: bool = False,
    mock_payload_path: Path | None = None,
) -> TasteAnalysis:
    if mock:
        if mock_payload_path:
            return parse_gemini_json(mock_payload_path.read_text(encoding="utf-8"))
        return TasteAnalysis.model_validate(MOCK_ANALYSIS)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise GeminiAnalysisError("GEMINI_API_KEY is required unless --mock-gemini is used")
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - dependency is declared.
        raise GeminiAnalysisError("google-genai is not installed") from exc

    client = genai.Client(api_key=api_key)
    uploaded = client.files.upload(file=str(video_path))
    prompt = (
        "Analyze this narrated screen recording. Return strict JSON only matching the "
        "tastepack schema: source_summary, transcript, assets, preference_moments, "
        "suggested_frames, visual_details, motion_details. Avoid vague design language "
        "unless you explain the concrete visual properties."
    )
    response = client.models.generate_content(
        model=config.gemini_model,
        contents=[uploaded, prompt],
    )
    return parse_gemini_json(response.text or "")
