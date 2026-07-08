from __future__ import annotations

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
        "verbosity": verbosity,
    }
    try:
        config = TastepackConfig.from_sources(config_file, overrides)
        validate_input_video(input_video, require_tools=not skip_ffmpeg)
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
        frame_map = extract_frames(input_video, selected_frames, out, skip_ffmpeg=skip_ffmpeg)
        generate_artifacts(out, analysis, frame_map, config, input_video.name)
    except (VideoValidationError, GeminiAnalysisError, ValueError, RuntimeError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    typer.echo(f"Wrote tastepack output to {out}")
