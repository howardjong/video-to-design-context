from __future__ import annotations

from pathlib import Path

import pytest

from tastepack.config import TastepackConfig
from tastepack.gemini import MOCK_ANALYSIS
from tastepack.pipeline import (
    FailureCategory,
    PipelineDependencies,
    PipelineFailure,
    run_processing_job,
)
from tastepack.schema import TasteAnalysis


def test_pipeline_reports_missing_input_as_structured_input_failure(tmp_path: Path) -> None:
    with pytest.raises(PipelineFailure) as raised:
        run_processing_job(
            tmp_path / "missing.mp4",
            tmp_path / "pack",
            TastepackConfig(produce_pdf=False),
            mock_gemini=True,
            skip_ffmpeg=True,
        )

    error = raised.value
    assert error.category is FailureCategory.INPUT
    assert error.step == "Video preflight"
    assert "does not exist" in error.reason
    assert "Next:" in str(error)


def test_pipeline_returns_structured_result_without_importing_cli(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake-video")
    output_dir = tmp_path / "pack"

    analysis = TasteAnalysis.model_validate(MOCK_ANALYSIS)
    dependencies = PipelineDependencies(
        validate_input_video=lambda *_args, **_kwargs: {
            "duration_seconds": 20.0,
            "source_sha256": "abc",
        },
        analyze_video=lambda *_args, **_kwargs: analysis,
        select_frames_for_analysis=lambda *_args, **_kwargs: [object()],
        extract_frames=lambda *_args, **_kwargs: [],
        generate_artifacts=lambda *_args, **_kwargs: None,
    )

    result = run_processing_job(
        input_video,
        output_dir,
        TastepackConfig(produce_pdf=False),
        mock_gemini=True,
        skip_ffmpeg=True,
        dependencies=dependencies,
    )

    assert result.output_dir == output_dir
    assert result.video_metadata["source_sha256"] == "abc"
    assert result.analysis.model_dump() == analysis.model_dump()


def test_pipeline_emits_validated_analysis_before_artifact_work(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake-video")
    analysis = TasteAnalysis.model_validate(MOCK_ANALYSIS)
    events = []
    dependencies = PipelineDependencies(
        validate_input_video=lambda *_args, **_kwargs: {
            "duration_seconds": 20.0,
            "source_sha256": "abc",
        },
        analyze_video=lambda *_args, **_kwargs: analysis,
        select_frames_for_analysis=lambda *_args, **_kwargs: [object()],
        extract_frames=lambda *_args, **_kwargs: [],
        generate_artifacts=lambda *_args, **_kwargs: None,
    )

    run_processing_job(
        input_video,
        tmp_path / "pack",
        TastepackConfig(produce_pdf=False),
        mock_gemini=True,
        skip_ffmpeg=True,
        dependencies=dependencies,
        lifecycle_callback=lambda state, payload: events.append((state, payload)),
    )

    states = [state for state, _payload in events]
    assert states.index("analysis_validated") < states.index("output_promoted")
    snapshot = next(payload for state, payload in events if state == "analysis_validated")
    assert snapshot["analysis"] == analysis.model_dump(mode="json")
