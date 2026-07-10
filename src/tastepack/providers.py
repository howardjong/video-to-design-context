from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tastepack.config import TastepackConfig
from tastepack.schema import TasteAnalysis


class VideoAnalysisProvider(Protocol):
    """Minimal boundary for a provider that natively understands video and audio."""

    def analyze(self, config: TastepackConfig) -> TasteAnalysis: ...


@dataclass(frozen=True)
class SynthesisEvaluationProtocol:
    primary_video_provider: str = "gemini-3.5-flash"
    candidate_synthesis_provider: str = "gpt-5.6"
    required_uat_metrics: tuple[str, ...] = (
        "asset and timestamp accuracy",
        "preference evidence fidelity",
        "frame-to-moment traceability",
        "cost and latency",
    )
    promotion_rule: str = "Adopt only after measured UAT improvement without weaker evidence."


FUTURE_SYNTHESIS_EVALUATION = SynthesisEvaluationProtocol()
