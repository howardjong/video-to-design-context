from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

import typer

from tastepack.artifacts import generate_artifacts
from tastepack.config import TastepackConfig
from tastepack.frames import extract_frames, select_frames_for_analysis
from tastepack.gemini import GeminiAnalysisError, analyze_video
from tastepack.logging import configure_logging, get_logger, redact_secrets
from tastepack.schema import TasteAnalysis
from tastepack.video import VideoValidationError, probe_duration_seconds, validate_input_video

app = typer.Typer(help="Create Claude-ready design taste context packs from narrated videos.")
logger = get_logger("cli")


class PipelineStepError(RuntimeError):
    def __init__(self, step: str, reason: str, next_step: str) -> None:
        self.step = step
        self.reason = reason
        self.next_step = next_step
        super().__init__(self.format_message())

    def format_message(self) -> str:
        return f"Step: {self.step}\nWhy: {self.reason}\nNext: {self.next_step}"


def run_step(step: str, next_step: str, callback):
    logger.debug("Starting step: %s", step)
    try:
        result = callback()
    except Exception as exc:
        safe_reason = redact_secrets(exc)
        logger.exception("Step failed: %s: %s", step, safe_reason)
        raise PipelineStepError(step, safe_reason, next_step) from exc
    logger.debug("Finished step: %s", step)
    return result


def validate_output_paths(out: Path, log_file: Path | None) -> None:
    if log_file and log_file.resolve().is_relative_to(out.resolve()):
        raise ValueError(f"Log file cannot be inside the output directory: {log_file}")
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


@app.callback()
def main() -> None:
    """Create Claude-ready design taste context packs from narrated videos."""


@app.command()
def process(
    input_video: Path = typer.Argument(
        ...,
        exists=False,
        help="Local MP4 or MOV screen recording.",
    ),
    out: Path = typer.Option(..., "--out", help="Output directory for the Claude-ready pack."),
    config_file: Path | None = typer.Option(None, "--config", help="Optional JSON config file."),
    gemini_model: str | None = typer.Option(None, "--model", help="Gemini model name."),
    frame_confidence_threshold: float | None = typer.Option(
        None, "--frame-confidence-threshold", min=0, max=1
    ),
    max_frames_per_asset: int | None = typer.Option(None, "--max-frames-per-asset", min=1),
    max_total_frames: int | None = typer.Option(None, "--max-total-frames", min=1),
    produce_pdf: bool | None = typer.Option(
        None,
        "--pdf/--no-pdf",
        help="Generate taste_packet.pdf.",
    ),
    fallback_interval_seconds: float | None = typer.Option(None, "--fallback-interval", min=0.1),
    max_duration_seconds: float | None = typer.Option(None, "--max-duration-seconds", min=0.1),
    max_file_size_mb: float | None = typer.Option(None, "--max-file-size-mb", min=0.1),
    allow_no_audio: bool | None = typer.Option(
        None,
        "--allow-no-audio/--require-audio",
        help="Allow visual-only videos.",
    ),
    gemini_max_retries: int | None = typer.Option(None, "--gemini-max-retries", min=1),
    gemini_retry_base_delay_seconds: float | None = typer.Option(
        None,
        "--gemini-retry-base-delay",
        min=0.1,
    ),
    cleanup_uploaded_files: bool | None = typer.Option(
        None,
        "--cleanup-uploaded-files/--no-cleanup-uploaded-files",
        help="Delete Gemini Files API upload after processing.",
    ),
    verbosity: str | None = typer.Option(None, "--verbosity", help="quiet, normal, or debug."),
    log_file: Path | None = typer.Option(None, "--log-file", help="Write troubleshooting logs."),
    mock_gemini: bool = typer.Option(
        False,
        "--mock-gemini",
        help="Use offline mock Gemini output.",
    ),
    mock_payload: Path | None = typer.Option(
        None,
        "--mock-payload",
        help="Path to mock JSON payload.",
    ),
    skip_ffmpeg: bool = typer.Option(False, "--skip-ffmpeg", help="Write mock frame files."),
) -> None:
    overrides = {
        "gemini_model": gemini_model,
        "frame_confidence_threshold": frame_confidence_threshold,
        "max_frames_per_asset": max_frames_per_asset,
        "max_total_frames": max_total_frames,
        "produce_pdf": produce_pdf,
        "fallback_interval_seconds": fallback_interval_seconds,
        "max_duration_seconds": max_duration_seconds,
        "max_file_size_bytes": int(max_file_size_mb * 1024 * 1024)
        if max_file_size_mb is not None
        else None,
        "allow_no_audio": allow_no_audio,
        "gemini_max_retries": gemini_max_retries,
        "gemini_retry_base_delay_seconds": gemini_retry_base_delay_seconds,
        "cleanup_uploaded_files": cleanup_uploaded_files,
        "verbosity": verbosity,
    }
    staging_dir: Path | None = None
    try:
        configure_logging("normal")
        run_step(
            "Output preflight",
            "Choose a log file outside --out and verify the output path.",
            lambda: validate_output_paths(out, log_file),
        )
        configure_logging("normal", log_file)
        if skip_ffmpeg and not mock_gemini:
            raise PipelineStepError(
                "Configuration",
                "--skip-ffmpeg requires --mock-gemini",
                "Remove --skip-ffmpeg for a live run or add --mock-gemini for offline testing.",
            )
        if mock_payload and not mock_gemini:
            raise PipelineStepError(
                "Configuration",
                "--mock-payload requires --mock-gemini",
                "Add --mock-gemini to use a local analysis fixture.",
            )
        config = run_step(
            "Configuration",
            "Check the config file JSON syntax and CLI flag values.",
            lambda: TastepackConfig.from_sources(config_file, overrides),
        )
        configure_logging(config.verbosity, log_file)
        logger.debug("Loaded configuration: %s", config.model_dump())
        video_metadata = run_step(
            "Video preflight",
            "Verify the input path, format, ffmpeg/ffprobe installation, audio stream, "
            "duration, and size before retrying.",
            lambda: validate_input_video(
                input_video,
                require_tools=not skip_ffmpeg,
                config=config,
            ),
        )
        duration = run_step(
            "Video duration probe",
            "Re-export the video if ffprobe cannot read its duration.",
            lambda: None if skip_ffmpeg else probe_duration_seconds(input_video),
        )
        analysis = run_step(
            "Gemini analysis",
            "Fix the Gemini response/schema issue or retry after resolving API availability.",
            lambda: analyze_video(
                input_video,
                config,
                mock=mock_gemini,
                mock_payload_path=mock_payload,
            ),
        )
        analysis = run_step(
            "Analysis validation",
            "Record a shorter, clearer video or correct the Gemini analysis response "
            "before retrying.",
            lambda: validate_analysis_for_video(analysis, video_metadata, config),
        )
        selected_frames = run_step(
            "Frame selection",
            "Check Gemini suggested frames, confidence thresholds, and fallback interval.",
            lambda: select_frames_for_analysis(
                analysis,
                config,
                video_duration_seconds=duration,
            ),
        )
        if not selected_frames:
            raise PipelineStepError(
                "Frame selection",
                "No frames could be selected or generated",
                "Lower frame confidence threshold or adjust fallback interval.",
            )
        out.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=str(out.parent))
        )
        logger.debug("Created staging directory: %s", staging_dir)
        extracted_frames = run_step(
            "Frame extraction",
            "Check ffmpeg output and frame timestamps; rerun with --verbosity debug.",
            lambda: extract_frames(
                input_video,
                selected_frames,
                staging_dir,
                skip_ffmpeg=skip_ffmpeg,
            ),
        )
        run_step(
            "Artifact generation",
            "Inspect output permissions and PDF generation settings; retry with --no-pdf.",
            lambda: generate_artifacts(
                staging_dir,
                analysis,
                extracted_frames,
                config,
                input_video.name,
                source_video_metadata=video_metadata,
            ),
        )
        run_step(
            "Output promotion",
            "Check that the output path is writable and is a directory.",
            lambda: promote_output(staging_dir, out),
        )
        staging_dir = None
    except (VideoValidationError, GeminiAnalysisError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(redact_secrets(exc)) from exc
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    typer.echo(f"Wrote tastepack output to {out}")


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
