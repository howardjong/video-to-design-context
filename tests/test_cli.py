import json

import pytest
from typer.testing import CliRunner

from tastepack.cli import app, promote_output
from tastepack.gemini import MOCK_ANALYSIS
from tastepack.schema import TasteAnalysis
from tastepack.video import source_fingerprint

runner = CliRunner()


def test_cli_rejects_missing_input_file(tmp_path):
    result = runner.invoke(app, ["process", str(tmp_path / "missing.mp4"), "--out", str(tmp_path)])

    assert result.exit_code != 0
    assert "Step: Video preflight" in result.output
    assert "Why: Input video does not exist" in result.output
    assert "Next:" in result.output
    assert "does not exist" in result.output


def test_cli_rejects_unsupported_file_extension(tmp_path):
    video = tmp_path / "input.txt"
    video.write_text("not a video")

    result = runner.invoke(app, ["process", str(video), "--out", str(tmp_path / "out")])

    assert result.exit_code != 0
    assert "Unsupported video extension" in result.output


def test_cli_accepts_video_and_output_directory_in_mock_mode(tmp_path):
    video = tmp_path / "input movie.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude pack"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "taste_packet.md").exists()
    assert (output_dir / "design_preferences.md").exists()
    assert (output_dir / "transcript.md").exists()
    assert json.loads((output_dir / "metadata.json").read_text())["source_video"] == video.name


def test_cli_processes_inbox_in_mock_mode(tmp_path):
    data_dir = tmp_path / "tastepack-data"
    inbox = data_dir / "inbox"
    inbox.mkdir(parents=True)
    (inbox / "input.mp4").write_bytes(b"fake video for mocked mode")

    result = runner.invoke(
        app,
        [
            "process-inbox",
            "--data-dir",
            str(data_dir),
            "--mock-gemini",
            "--skip-ffmpeg",
            "--no-pdf",
            "--stable-seconds",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(list((data_dir / "archive").rglob("input.mp4"))) == 1
    outputs = list((data_dir / "output").glob("*/metadata.json"))
    assert len(outputs) == 1
    assert json.loads(outputs[0].read_text())["queue"]["run_key"]


def test_skip_ffmpeg_requires_mock_gemini(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a decodable video")
    output_dir = tmp_path / "claude-pack"

    result = runner.invoke(
        app,
        ["process", str(video), "--out", str(output_dir), "--skip-ffmpeg"],
    )

    assert result.exit_code != 0
    assert "--skip-ffmpeg requires --mock-gemini" in result.output
    assert not output_dir.exists()


def test_mock_payload_requires_mock_gemini(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not a decodable video")
    payload = tmp_path / "analysis.json"
    payload.write_text("{}")

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(tmp_path / "claude-pack"),
            "--mock-payload",
            str(payload),
        ],
    )

    assert result.exit_code != 0
    assert "--mock-payload requires --mock-gemini" in result.output


def test_log_file_inside_output_directory_is_rejected_before_processing(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--skip-ffmpeg",
            "--no-pdf",
            "--log-file",
            str(output_dir / "tastepack.log"),
        ],
    )

    assert result.exit_code != 0
    assert "Log file cannot be inside the output directory" in result.output
    assert not output_dir.exists()


def test_non_directory_output_path_fails_before_gemini_analysis(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_path = tmp_path / "claude-pack"
    output_path.write_text("not a directory")
    analysis_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal analysis_called
        analysis_called = True
        raise AssertionError("Gemini analysis must not run")

    monkeypatch.setattr("tastepack.cli.analyze_video", fail_if_called)

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_path),
            "--mock-gemini",
            "--skip-ffmpeg",
            "--no-pdf",
        ],
    )

    assert result.exit_code != 0
    assert "Output path exists and is not a directory" in result.output
    assert analysis_called is False


def test_config_file_boolean_values_apply_without_cli_overrides(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    config_path = tmp_path / "tastepack.json"
    config_path.write_text(
        json.dumps(
            {
                "produce_pdf": False,
                "allow_no_audio": True,
                "cleanup_uploaded_files": False,
            }
        )
    )
    output_dir = tmp_path / "claude-pack"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--config",
            str(config_path),
            "--mock-gemini",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code == 0, result.output
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["config"]["produce_pdf"] is False
    assert metadata["config"]["allow_no_audio"] is True
    assert metadata["config"]["cleanup_uploaded_files"] is False
    assert not (output_dir / "taste_packet.pdf").exists()


def test_unusable_output_parent_fails_before_gemini_analysis(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    blocked_parent = tmp_path / "blocked-parent"
    blocked_parent.write_text("not a directory")
    analysis_called = False

    def fail_if_called(*args, **kwargs):
        nonlocal analysis_called
        analysis_called = True
        raise AssertionError("Gemini analysis must not run")

    monkeypatch.setattr("tastepack.cli.analyze_video", fail_if_called)

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(blocked_parent / "claude-pack"),
            "--mock-gemini",
            "--skip-ffmpeg",
            "--no-pdf",
        ],
    )

    assert result.exit_code != 0
    assert "Output preflight" in result.output
    assert analysis_called is False


def test_out_of_video_gemini_analysis_fails_before_frame_extraction(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    metadata = {
        "duration_seconds": 10.0,
        "width": 1280,
        "height": 720,
        "video_stream_count": 1,
        "audio_stream_count": 1,
        "video_codec": "h264",
        "audio_codec": "aac",
        "file_size_bytes": video.stat().st_size,
    }
    monkeypatch.setattr("tastepack.cli.validate_input_video", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(
        "tastepack.cli.analyze_video",
        lambda *args, **kwargs: TasteAnalysis.model_validate(MOCK_ANALYSIS),
    )

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(tmp_path / "claude-pack"),
            "--mock-gemini",
            "--no-pdf",
        ],
    )

    assert result.exit_code != 0
    assert "Step: Analysis validation" in result.output
    assert "Asset range is outside video duration" in result.output
    assert not (tmp_path / "claude-pack").exists()


def test_cli_reuses_the_first_video_preflight_instead_of_reprobing_duration(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    metadata = {
        "duration_seconds": 20.0,
        "width": 1280,
        "height": 720,
        "video_stream_count": 1,
        "audio_stream_count": 1,
        "video_codec": "h264",
        "audio_codec": "aac",
        **source_fingerprint(video),
    }
    monkeypatch.setattr("tastepack.cli.validate_input_video", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(
        "tastepack.cli.probe_duration_seconds",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not reprobe")),
        raising=False,
    )
    monkeypatch.setattr("tastepack.cli.extract_frames", lambda *args, **kwargs: [])

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(tmp_path / "claude-pack"),
            "--mock-gemini",
            "--no-pdf",
        ],
    )

    assert result.exit_code == 0, result.output


def test_source_replacement_after_analysis_preserves_existing_output(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"video-before-preflight")
    metadata = {
        "duration_seconds": 20.0,
        "width": 1280,
        "height": 720,
        "video_stream_count": 1,
        "audio_stream_count": 1,
        "video_codec": "h264",
        "audio_codec": "aac",
        **source_fingerprint(video),
    }
    output_dir = tmp_path / "claude-pack"
    output_dir.mkdir()
    previous_packet = output_dir / "taste_packet.md"
    previous_packet.write_text("previous complete pack")
    monkeypatch.setattr("tastepack.cli.validate_input_video", lambda *args, **kwargs: metadata)

    def replace_source_during_analysis(*args, **kwargs):
        video.write_bytes(b"video-replaced-after-gemini-analysis")
        return TasteAnalysis.model_validate(MOCK_ANALYSIS)

    monkeypatch.setattr("tastepack.cli.analyze_video", replace_source_during_analysis)

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--skip-ffmpeg",
            "--no-pdf",
        ],
    )

    assert result.exit_code != 0
    assert "Step: Frame extraction" in result.output
    assert "changed after preflight" in result.output
    assert previous_packet.read_text() == "previous complete pack"
    assert not list(tmp_path.glob(".claude-pack.tmp-*"))


def test_existing_output_directory_is_replaced_on_success(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old pack content")

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code == 0, result.output
    assert not (output_dir / "stale.txt").exists()


def test_promote_output_replaces_an_existing_pack_without_stale_files(tmp_path):
    output_dir = tmp_path / "claude-pack"
    output_dir.mkdir()
    (output_dir / "stale.pdf").write_text("old pdf")
    (output_dir / "frames").mkdir()
    (output_dir / "frames" / "stale.jpg").write_text("old frame")

    staging_dir = tmp_path / ".claude-pack.tmp-test"
    staging_dir.mkdir()
    (staging_dir / "taste_packet.md").write_text("new complete pack")
    (staging_dir / "frames").mkdir()
    (staging_dir / "frames" / "current.jpg").write_text("new frame")

    promote_output(staging_dir, output_dir)

    assert (output_dir / "taste_packet.md").read_text() == "new complete pack"
    assert (output_dir / "frames" / "current.jpg").read_text() == "new frame"
    assert not (output_dir / "stale.pdf").exists()
    assert not (output_dir / "frames" / "stale.jpg").exists()
    assert not staging_dir.exists()


def test_failed_promotion_restores_the_previous_pack(tmp_path, monkeypatch):
    output_dir = tmp_path / "claude-pack"
    output_dir.mkdir()
    previous_packet = output_dir / "taste_packet.md"
    previous_packet.write_text("previous complete pack")
    staging_dir = tmp_path / ".claude-pack.tmp-test"
    staging_dir.mkdir()
    (staging_dir / "taste_packet.md").write_text("replacement pack")

    original_replace = type(staging_dir).replace

    def fail_when_promoting(self, target):
        if self == staging_dir and target == output_dir:
            raise OSError("simulated promotion failure")
        return original_replace(self, target)

    monkeypatch.setattr(type(staging_dir), "replace", fail_when_promoting)

    with pytest.raises(OSError, match="simulated promotion failure"):
        promote_output(staging_dir, output_dir)

    assert previous_packet.read_text() == "previous complete pack"
    assert not list(tmp_path.glob(".claude-pack.backup-*"))


def test_failed_run_does_not_promote_partial_output(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"

    def fail_artifacts(*args, **kwargs):
        raise RuntimeError("artifact failure")

    monkeypatch.setattr("tastepack.cli.generate_artifacts", fail_artifacts)

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code != 0
    assert not output_dir.exists()
    assert not list(tmp_path.glob(".claude-pack.tmp-*"))


def test_gemini_failure_does_not_promote_partial_output(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("schema-invalid Gemini output")

    monkeypatch.setattr("tastepack.cli.analyze_video", fail_analysis)

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code != 0
    assert "Step: Gemini analysis" in result.output
    assert "Why: schema-invalid Gemini output" in result.output
    assert "Next:" in result.output
    assert "schema-invalid Gemini output" in result.output
    assert not output_dir.exists()


def test_verbose_cli_logs_major_processing_steps(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
            "--verbosity",
            "debug",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Video preflight" in result.output
    assert "Gemini analysis" in result.output
    assert "Frame extraction" in result.output
    assert "Output promotion" in result.output


def test_cli_writes_debug_log_file_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-key")
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"
    log_file = tmp_path / "tastepack.log"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
            "--verbosity",
            "debug",
            "--log-file",
            str(log_file),
        ],
    )

    assert result.exit_code == 0, result.output
    log_text = log_file.read_text()
    assert "Starting step: Video preflight" in log_text
    assert "Finished step: Output promotion" in log_text
    assert "super-secret-key" not in log_text


def test_invalid_config_file_failure_identifies_configuration_step(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video")
    config_path = tmp_path / "tastepack.json"
    config_path.write_text("{not json")

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(tmp_path / "out"),
            "--config",
            str(config_path),
            "--mock-gemini",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code != 0
    assert "Step: Configuration" in result.output
    assert "Why:" in result.output
    assert "Next:" in result.output


def test_failed_run_leaves_existing_output_untouched(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"
    output_dir.mkdir()
    existing = output_dir / "taste_packet.md"
    existing.write_text("previous complete pack")

    def fail_artifacts(*args, **kwargs):
        raise RuntimeError("artifact failure")

    monkeypatch.setattr("tastepack.cli.generate_artifacts", fail_artifacts)

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code != 0
    assert existing.read_text() == "previous complete pack"


def test_successful_run_promotes_complete_staged_output(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"

    result = runner.invoke(
        app,
        [
            "process",
            str(video),
            "--out",
            str(output_dir),
            "--mock-gemini",
            "--no-pdf",
            "--skip-ffmpeg",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "taste_packet.md").exists()
    assert not list(tmp_path.glob(".claude-pack.tmp-*"))
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["run_status"] == "complete"
    assert metadata["provider"]["name"] == "gemini"
    assert metadata["provider"]["model"] == "gemini-3.5-flash"
    assert metadata["provider"]["prompt_version"] == "tastepack-video-analysis-v1"
    assert metadata["provider"]["schema_version"] == "tastepack-analysis-v1"
    assert metadata["provider"]["sdk_version"]
    assert metadata["provider"]["telemetry"]["finish_reason"] == "MOCK"
