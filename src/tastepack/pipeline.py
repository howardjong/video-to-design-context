from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import uuid4

from tastepack.artifacts import generate_artifacts
from tastepack.config import TastepackConfig
from tastepack.frames import build_coverage_frames, extract_frames, select_frames_for_analysis
from tastepack.gemini import (
    GeminiRunTelemetry,
    analyze_video,
    build_gemini_provider_metadata,
)
from tastepack.logging import get_logger, redact_secrets
from tastepack.qa import (
    QualityAuditProvider,
    audit_staged_pack,
    validate_source_transcript,
)
from tastepack.schema import TasteAnalysis
from tastepack.video import validate_input_video

logger = get_logger("pipeline")


class FailureCategory(StrEnum):
    INPUT = "input"
    PROVIDER = "provider"
    SYSTEM = "system"


class PipelineFailure(RuntimeError):
    def __init__(
        self,
        step: str,
        reason: str,
        next_step: str,
        category: FailureCategory,
    ) -> None:
        self.step = step
        self.reason = reason
        self.next_step = next_step
        self.category = category
        super().__init__(self.format_message())

    def format_message(self) -> str:
        return f"Step: {self.step}\nWhy: {self.reason}\nNext: {self.next_step}"


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    video_metadata: dict[str, object]
    analysis: TasteAnalysis
    provider_metadata: dict[str, object]


@dataclass(frozen=True)
class PipelineDependencies:
    validate_input_video: Callable[..., dict[str, object]] = validate_input_video
    analyze_video: Callable[..., TasteAnalysis] = analyze_video
    select_frames_for_analysis: Callable[..., list[Any]] = select_frames_for_analysis
    extract_frames: Callable[..., list[Any]] = extract_frames
    generate_artifacts: Callable[..., None] = generate_artifacts
    promote_output: Callable[[Path, Path], None] | None = None


LifecycleCallback = Callable[[str, dict[str, Any]], None]


def run_step(
    step: str,
    next_step: str,
    callback: Callable[[], Any],
    category: FailureCategory,
) -> Any:
    logger.debug("Starting step: %s", step)
    try:
        result = callback()
    except PipelineFailure:
        raise
    except Exception as exc:
        safe_reason = redact_secrets(exc)
        logger.exception("Step failed: %s: %s", step, safe_reason)
        raise PipelineFailure(step, safe_reason, next_step, category) from exc
    logger.debug("Finished step: %s", step)
    return result


def validate_output_directory(out: Path) -> None:
    if out.exists() and not out.is_dir():
        raise ValueError(f"Output path exists and is not a directory: {out}")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        probe_dir = Path(tempfile.mkdtemp(prefix=f".{out.name}.preflight-", dir=out.parent))
        shutil.rmtree(probe_dir)
    except OSError as exc:
        raise ValueError(f"Output parent is not writable: {out.parent}: {exc}") from exc


def validate_analysis_for_video(
    analysis: TasteAnalysis,
    video_metadata: dict[str, object],
    config: TastepackConfig,
) -> TasteAnalysis:
    return TasteAnalysis.model_validate(
        analysis.model_dump(),
        context={
            "video_duration_seconds": video_metadata.get("duration_seconds"),
            "require_transcript": not config.allow_no_audio,
        },
    )


def promote_output(staging_dir: Path, out: Path) -> None:
    if not out.exists():
        staging_dir.replace(out)
        return
    if not out.is_dir():
        raise RuntimeError(f"Output path exists and is not a directory: {out}")
    backup_dir = out.with_name(f".{out.name}.backup-{uuid4().hex}")
    out.replace(backup_dir)
    try:
        staging_dir.replace(out)
    except Exception:
        backup_dir.replace(out)
        raise
    shutil.rmtree(backup_dir)


def run_processing_job(
    input_video: Path,
    out: Path,
    config: TastepackConfig,
    *,
    mock_gemini: bool = False,
    mock_payload: Path | None = None,
    skip_ffmpeg: bool = False,
    preflight_metadata: dict[str, object] | None = None,
    gemini_permit: Callable[[], Any] | None = None,
    retry_observer: Callable[[float, BaseException], None] | None = None,
    precomputed_analysis: TasteAnalysis | None = None,
    precomputed_provider_metadata: dict[str, object] | None = None,
    lifecycle_callback: LifecycleCallback | None = None,
    dependencies: PipelineDependencies | None = None,
    analysis_video: Path | None = None,
    source_transcript: Path | None = None,
    qa_provider: QualityAuditProvider | None = None,
    mock_qa: bool = False,
) -> PipelineResult:
    dependencies = dependencies or PipelineDependencies(promote_output=promote_output)
    promote = dependencies.promote_output or promote_output
    staging_dir: Path | None = None
    provider_video = analysis_video or input_video

    def emit_lifecycle(state: str, payload: dict[str, Any]) -> None:
        if lifecycle_callback is None:
            return
        try:
            lifecycle_callback(state, payload)
        except Exception as exc:
            raise PipelineFailure(
                "Job state persistence",
                redact_secrets(exc),
                "Restore the jobs directory and retry after confirming its permissions.",
                FailureCategory.SYSTEM,
            ) from exc

    try:
        run_step(
            "Output preflight",
            "Verify the output path is writable and is a directory.",
            lambda: validate_output_directory(out),
            FailureCategory.SYSTEM,
        )
        video_metadata = preflight_metadata or run_step(
            "Video preflight",
            "Verify the input path, format, ffmpeg/ffprobe installation, audio stream, "
            "duration, and size before retrying.",
            lambda: dependencies.validate_input_video(
                input_video,
                require_tools=not skip_ffmpeg,
                config=config,
            ),
            FailureCategory.INPUT,
        )
        duration = video_metadata.get("duration_seconds")
        if not isinstance(duration, int | float) or isinstance(duration, bool):
            duration = None
        if config.qa_enabled:
            source_transcript = run_step(
                "QA preflight",
                "Provide a non-empty timestamped Markdown transcript before retrying.",
                lambda: validate_source_transcript(source_transcript),
                FailureCategory.INPUT,
            )

        gemini_telemetry = GeminiRunTelemetry()
        if precomputed_analysis is None:

            def analyze_with_provider_permit() -> TasteAnalysis:
                context = gemini_permit() if gemini_permit is not None else nullcontext()
                with context:
                    emit_lifecycle("gemini_started", {})
                    return dependencies.analyze_video(
                        provider_video,
                        config,
                        mock=mock_gemini,
                        mock_payload_path=mock_payload,
                        telemetry=gemini_telemetry,
                        video_duration_seconds=duration,
                        retry_observer=retry_observer,
                    )

            analysis = run_step(
                "Gemini analysis",
                "Fix the Gemini response/schema issue or retry after resolving API availability.",
                analyze_with_provider_permit,
                FailureCategory.PROVIDER,
            )
        else:
            analysis = precomputed_analysis
        analysis = run_step(
            "Analysis validation",
            "Record a shorter, clearer video or correct the Gemini analysis response "
            "before retrying.",
            lambda: validate_analysis_for_video(analysis, video_metadata, config),
            FailureCategory.PROVIDER,
        )
        provider_metadata = precomputed_provider_metadata or build_gemini_provider_metadata(
            config,
            gemini_telemetry,
        )
        if precomputed_analysis is None:
            emit_lifecycle(
                "analysis_validated",
                {
                    "analysis": analysis.model_dump(mode="json"),
                    "provider_metadata": provider_metadata,
                },
            )
        selected_frames = run_step(
            "Frame selection",
            "Check Gemini suggested frames, confidence thresholds, and fallback interval.",
            lambda: dependencies.select_frames_for_analysis(
                analysis,
                config,
                video_duration_seconds=duration,
            ),
            FailureCategory.SYSTEM,
        )
        if not selected_frames:
            raise PipelineFailure(
                "Frame selection",
                "No frames could be selected or generated",
                "Lower frame confidence threshold or adjust fallback interval.",
                FailureCategory.SYSTEM,
            )

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=str(out.parent)))
        logger.debug("Created staging directory: %s", staging_dir)
        extracted_frames = run_step(
            "Frame extraction",
            "Check ffmpeg output and frame timestamps; rerun with --verbosity debug.",
            lambda: dependencies.extract_frames(
                input_video,
                selected_frames,
                staging_dir,
                skip_ffmpeg=skip_ffmpeg,
                expected_source_metadata=video_metadata,
                ffmpeg_timeout_seconds=config.frame_extraction_timeout_seconds,
            ),
            FailureCategory.SYSTEM,
        )
        coverage_frames = []
        if config.qa_enabled:
            coverage_duration = duration
            if coverage_duration is None:
                coverage_duration = max(
                    (asset.end_seconds for asset in analysis.assets),
                    default=0.0,
                )
            if coverage_duration <= 0:
                raise PipelineFailure(
                    "QA evidence extraction",
                    "No usable video duration is available for independent coverage frames",
                    "Fix video preflight metadata before retrying.",
                    FailureCategory.SYSTEM,
                )
            coverage_selection = build_coverage_frames(float(coverage_duration), config)
            coverage_frames = run_step(
                "QA evidence extraction",
                "Check ffmpeg output and the configured QA coverage interval.",
                lambda: dependencies.extract_frames(
                    input_video,
                    coverage_selection,
                    staging_dir,
                    skip_ffmpeg=skip_ffmpeg,
                    expected_source_metadata=video_metadata,
                    ffmpeg_timeout_seconds=config.frame_extraction_timeout_seconds,
                    relative_directory=Path("evidence") / "coverage_frames",
                ),
                FailureCategory.SYSTEM,
            )
        run_step(
            "Artifact generation",
            "Inspect output permissions and PDF generation settings; retry with --no-pdf.",
            lambda: dependencies.generate_artifacts(
                staging_dir,
                analysis,
                extracted_frames,
                config,
                input_video.name,
                source_video_metadata=video_metadata,
                provider_metadata=provider_metadata,
                source_transcript_path=source_transcript if config.qa_enabled else None,
                coverage_frames=coverage_frames,
            ),
            FailureCategory.SYSTEM,
        )
        if config.qa_enabled:
            run_step(
                "QA audit",
                "Inspect the QA provider response, citations, and source evidence before retrying.",
                lambda: audit_staged_pack(
                    staging_dir,
                    config,
                    provider=qa_provider,
                    mock=mock_qa,
                ),
                FailureCategory.PROVIDER,
            )
        run_step(
            "Output promotion",
            "Check that the output path is writable and is a directory.",
            lambda: promote(staging_dir, out),
            FailureCategory.SYSTEM,
        )
        emit_lifecycle("output_promoted", {"output_dir": str(out)})
        staging_dir = None
        return PipelineResult(
            output_dir=out,
            video_metadata=video_metadata,
            analysis=analysis,
            provider_metadata=provider_metadata,
        )
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
