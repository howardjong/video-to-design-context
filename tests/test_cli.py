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
