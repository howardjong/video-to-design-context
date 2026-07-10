from __future__ import annotations

import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import typer

from tastepack.artifacts import generate_artifacts
from tastepack.config import TastepackConfig
from tastepack.frames import extract_frames, select_frames_for_analysis
from tastepack.gemini import GeminiAnalysisError, analyze_video
from tastepack.inbox_queue import (
    IntakePaths,
    QueueLockedError,
)
from tastepack.inbox_queue import (
    process_inbox as run_inbox,
)
from tastepack.logging import configure_logging, get_logger, redact_secrets
from tastepack.pipeline import PipelineDependencies, PipelineFailure, run_processing_job
from tastepack.schema import TasteAnalysis
from tastepack.video import validate_input_video

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
        run_processing_job(
            input_video,
            out,
            config,
            mock_gemini=mock_gemini,
            mock_payload=mock_payload,
            skip_ffmpeg=skip_ffmpeg,
            dependencies=PipelineDependencies(
                validate_input_video=validate_input_video,
                analyze_video=analyze_video,
                select_frames_for_analysis=select_frames_for_analysis,
                extract_frames=extract_frames,
                generate_artifacts=generate_artifacts,
                promote_output=promote_output,
            ),
        )
    except (PipelineFailure, GeminiAnalysisError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(redact_secrets(exc)) from exc

    typer.echo(f"Wrote tastepack output to {out}")


@app.command("process-inbox")
def process_inbox_command(
    data_dir: Path = typer.Option(Path("tastepack-data"), "--data-dir", help="Inbox data root."),
    config_file: Path | None = typer.Option(None, "--config", help="Optional JSON config file."),
    gemini_model: str | None = typer.Option(None, "--model", help="Gemini model name."),
    frame_confidence_threshold: float | None = typer.Option(
        None,
        "--frame-confidence-threshold",
        min=0,
        max=1,
    ),
    max_frames_per_asset: int | None = typer.Option(None, "--max-frames-per-asset", min=1),
    max_total_frames: int | None = typer.Option(None, "--max-total-frames", min=1),
    produce_pdf: bool | None = typer.Option(None, "--pdf/--no-pdf"),
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
        help="Delete Gemini Files API uploads after each video.",
    ),
    stable_seconds: float = typer.Option(
        10.0,
        "--stable-seconds",
        min=0,
        help="Require unchanged file size and mtime for this duration before claiming.",
    ),
    max_jobs: int | None = typer.Option(None, "--max-jobs", min=1),
    force: bool = typer.Option(False, "--force", help="Reprocess matching completed output."),
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
    paths = IntakePaths.from_root(data_dir)
    paths.ensure()
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
    effective_log_file = log_file or paths.logs / (
        "inbox-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".log"
    )
    try:
        if effective_log_file.resolve().is_relative_to(paths.output.resolve()):
            raise PipelineStepError(
                "Configuration",
                f"Log file cannot be inside the output directory: {effective_log_file}",
                "Choose a log file under the logs directory or another path outside output.",
            )
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
        config = TastepackConfig.from_sources(config_file, overrides)
        configure_logging(config.verbosity, effective_log_file)
        summary = run_inbox(
            data_dir,
            config,
            stable_seconds=stable_seconds,
            max_jobs=max_jobs,
            force=force,
            mock_gemini=mock_gemini,
            mock_payload=mock_payload,
            skip_ffmpeg=skip_ffmpeg,
        )
    except (PipelineStepError, QueueLockedError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(redact_secrets(exc)) from exc

    for job in summary.jobs:
        failure = job.get("failure")
        if isinstance(failure, dict):
            typer.echo(
                "\n".join(
                    (
                        f"Job: {job['source_name']}",
                        f"Step: {failure['step']}",
                        f"Why: {failure['reason']}",
                        f"Next: {failure['next']}",
                    )
                ),
                err=True,
            )
    typer.echo(
        f"Inbox summary: {summary.completed} complete, {summary.skipped} skipped, "
        f"{summary.failed} failed. Log: {effective_log_file}"
    )
    if summary.halted:
        typer.echo(
            "Step: Inbox processing\n"
            f"Why: {summary.halt_reason}\n"
            "Next: Resolve the provider or local-system failure before processing more videos.",
            err=True,
        )
        raise typer.Exit(code=3)
    if summary.failed:
        raise typer.Exit(code=1)


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
