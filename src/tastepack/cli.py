from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import typer

from tastepack.artifacts import generate_artifacts
from tastepack.config import TastepackConfig
from tastepack.frames import build_fallback_frames, extract_frames, select_frames_for_analysis
from tastepack.gemini import GeminiAnalysisError, analyze_video
from tastepack.video import VideoValidationError, probe_duration_seconds, validate_input_video

app = typer.Typer(help="Create Claude-ready design taste context packs from narrated videos.")


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
    produce_pdf: bool = typer.Option(True, "--pdf/--no-pdf", help="Generate taste_packet.pdf."),
    fallback_interval_seconds: float | None = typer.Option(None, "--fallback-interval", min=0.1),
    max_duration_seconds: float | None = typer.Option(None, "--max-duration-seconds", min=0.1),
    max_file_size_mb: float | None = typer.Option(None, "--max-file-size-mb", min=0.1),
    allow_no_audio: bool = typer.Option(
        False,
        "--allow-no-audio",
        help="Allow visual-only videos.",
    ),
    gemini_max_retries: int | None = typer.Option(None, "--gemini-max-retries", min=1),
    gemini_retry_base_delay_seconds: float | None = typer.Option(
        None,
        "--gemini-retry-base-delay",
        min=0.1,
    ),
    cleanup_uploaded_files: bool = typer.Option(
        True,
        "--cleanup-uploaded-files/--no-cleanup-uploaded-files",
        help="Delete Gemini Files API upload after processing.",
    ),
    verbosity: str | None = typer.Option(None, "--verbosity", help="quiet, normal, or debug."),
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
        config = TastepackConfig.from_sources(config_file, overrides)
        video_metadata = validate_input_video(
            input_video,
            require_tools=not skip_ffmpeg,
            config=config,
        )
        duration = None if skip_ffmpeg else probe_duration_seconds(input_video)
        analysis = analyze_video(
            input_video,
            config,
            mock=mock_gemini,
            mock_payload_path=mock_payload,
        )
        selected_frames = select_frames_for_analysis(
            analysis,
            config,
            video_duration_seconds=duration,
        )
        if not selected_frames:
            selected_frames = build_fallback_frames(analysis.assets, config)
        if not selected_frames:
            raise RuntimeError("No frames could be selected or generated")
        out.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=str(out.parent))
        )
        frame_map = extract_frames(
            input_video,
            selected_frames,
            staging_dir,
            skip_ffmpeg=skip_ffmpeg,
        )
        generate_artifacts(
            staging_dir,
            analysis,
            frame_map,
            config,
            input_video.name,
            source_video_metadata=video_metadata,
        )
        promote_output(staging_dir, out)
        staging_dir = None
    except (VideoValidationError, GeminiAnalysisError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc
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
    for staged_path in staging_dir.rglob("*"):
        relative_path = staged_path.relative_to(staging_dir)
        destination = out / relative_path
        if staged_path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_path, destination)
