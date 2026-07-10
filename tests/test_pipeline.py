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
from tastepack.qa import MockQualityAuditProvider, QualityAuditError
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


def test_pipeline_uses_private_analysis_video_but_extracts_source_frames(tmp_path: Path) -> None:
    input_video = tmp_path / "source.mp4"
    analysis_video = tmp_path / "analysis-input.mp4"
    input_video.write_bytes(b"source")
    analysis_video.write_bytes(b"muxed")
    analysis = TasteAnalysis.model_validate(MOCK_ANALYSIS)
    analyzed_paths: list[Path] = []
    extracted_paths: list[Path] = []
    dependencies = PipelineDependencies(
        validate_input_video=lambda *_args, **_kwargs: {
            "duration_seconds": 20.0,
            "source_sha256": "source-hash",
        },
        analyze_video=lambda path, *_args, **_kwargs: (
            analyzed_paths.append(path) or analysis
        ),
        select_frames_for_analysis=lambda *_args, **_kwargs: [object()],
        extract_frames=lambda path, *_args, **_kwargs: extracted_paths.append(path) or [],
        generate_artifacts=lambda *_args, **_kwargs: None,
    )

    run_processing_job(
        input_video,
        tmp_path / "pack",
        TastepackConfig(produce_pdf=False),
        mock_gemini=True,
        skip_ffmpeg=True,
        analysis_video=analysis_video,
        dependencies=dependencies,
    )

    assert analyzed_paths == [analysis_video]
    assert extracted_paths == [input_video]


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


def test_qa_preflight_requires_source_transcript_before_gemini(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake-video")
    analyzed = False

    def fail_if_analyzed(*_args, **_kwargs):
        nonlocal analyzed
        analyzed = True
        raise AssertionError("Gemini must not be called")

    dependencies = PipelineDependencies(
        validate_input_video=lambda *_args, **_kwargs: {
            "duration_seconds": 20.0,
            "source_sha256": "abc",
        },
        analyze_video=fail_if_analyzed,
    )

    with pytest.raises(PipelineFailure, match="source transcript") as raised:
        run_processing_job(
            input_video,
            tmp_path / "pack",
            TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
            mock_gemini=True,
            skip_ffmpeg=True,
            dependencies=dependencies,
        )

    assert raised.value.category is FailureCategory.INPUT
    assert analyzed is False


def test_qa_preflight_rejects_untimestamped_transcript_before_gemini(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake-video")
    source_transcript = tmp_path / "source-transcript.md"
    source_transcript.write_text("No timestamp here.\n", encoding="utf-8")
    analyzed = False

    def fail_if_analyzed(*_args, **_kwargs):
        nonlocal analyzed
        analyzed = True
        raise AssertionError("Gemini must not be called")

    dependencies = PipelineDependencies(
        validate_input_video=lambda *_args, **_kwargs: {
            "duration_seconds": 20.0,
            "source_sha256": "abc",
        },
        analyze_video=fail_if_analyzed,
    )

    with pytest.raises(PipelineFailure, match="timestamped"):
        run_processing_job(
            input_video,
            tmp_path / "pack",
            TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
            mock_gemini=True,
            skip_ffmpeg=True,
            source_transcript=source_transcript,
            dependencies=dependencies,
        )

    assert analyzed is False


def test_pipeline_creates_evidence_and_reviewed_pack_with_mocked_qa(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake-video")
    source_transcript = tmp_path / "source-transcript.md"
    source_transcript.write_text("[00:00.000] I like the large heading.\n", encoding="utf-8")

    result = run_processing_job(
        input_video,
        tmp_path / "pack",
        TastepackConfig(
            produce_pdf=False,
            qa_enabled=True,
            qa_model="test-claude",
            qa_coverage_interval_seconds=3,
        ),
        mock_gemini=True,
        skip_ffmpeg=True,
        source_transcript=source_transcript,
        qa_provider=MockQualityAuditProvider(),
    )

    assert (result.output_dir / "evidence" / "source_transcript.md").read_text() == (
        "[00:00.000] I like the large heading.\n"
    )
    assert list((result.output_dir / "evidence" / "coverage_frames").glob("*.jpg"))
    assert (result.output_dir / "qa" / "audit.json").is_file()
    assert (result.output_dir / "START_HERE.md").is_file()


def test_qa_provider_failure_never_replaces_existing_complete_pack(tmp_path: Path) -> None:
    input_video = tmp_path / "input.mp4"
    input_video.write_bytes(b"fake-video")
    source_transcript = tmp_path / "source-transcript.md"
    source_transcript.write_text("[00:00.000] I like the large heading.\n", encoding="utf-8")
    output_dir = tmp_path / "pack"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("existing complete pack", encoding="utf-8")

    class ProviderFailure(MockQualityAuditProvider):
        def inventory(self, request):
            raise QualityAuditError("QA provider unavailable")

    with pytest.raises(PipelineFailure, match="QA provider unavailable") as raised:
        run_processing_job(
            input_video,
            output_dir,
            TastepackConfig(produce_pdf=False, qa_enabled=True, qa_model="test-claude"),
            mock_gemini=True,
            skip_ffmpeg=True,
            source_transcript=source_transcript,
            qa_provider=ProviderFailure(),
        )

    assert raised.value.category is FailureCategory.PROVIDER
    assert (output_dir / "keep.txt").read_text() == "existing complete pack"
    assert not (output_dir / "qa").exists()
