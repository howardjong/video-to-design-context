import json

from typer.testing import CliRunner

from tastepack.cli import app

runner = CliRunner()


def test_cli_rejects_missing_input_file(tmp_path):
    result = runner.invoke(app, ["process", str(tmp_path / "missing.mp4"), "--out", str(tmp_path)])

    assert result.exit_code != 0
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


def test_existing_output_directory_is_handled_safely(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video for mocked mode")
    output_dir = tmp_path / "claude-pack"
    output_dir.mkdir()
    (output_dir / "keep.txt").write_text("do not delete")

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
    assert (output_dir / "keep.txt").read_text() == "do not delete"


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
    assert "schema-invalid Gemini output" in result.output
    assert not output_dir.exists()


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
