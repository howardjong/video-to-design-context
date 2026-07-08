import pytest

from tastepack.config import TastepackConfig
from tastepack.video import VideoValidationError, validate_input_video


def test_video_validation_rejects_missing_files(tmp_path):
    with pytest.raises(VideoValidationError, match="does not exist"):
        validate_input_video(tmp_path / "missing.mp4", require_tools=False)


def test_video_validation_rejects_unsupported_extensions(tmp_path):
    video = tmp_path / "input.avi"
    video.write_bytes(b"not real video")

    with pytest.raises(VideoValidationError, match="Unsupported"):
        validate_input_video(video, require_tools=False)


def test_video_validation_checks_ffmpeg_and_ffprobe_are_available(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"not real video")

    def missing_ffprobe(tool):
        return "/usr/bin/ffmpeg" if tool == "ffmpeg" else None

    monkeypatch.setattr("tastepack.video.shutil.which", missing_ffprobe)

    with pytest.raises(VideoValidationError, match="ffprobe"):
        validate_input_video(video, require_tools=True)


def test_corrupt_video_fails_preflight_before_upload(tmp_path, monkeypatch):
    video = tmp_path / "corrupt.mp4"
    video.write_bytes(b"not a real mp4")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")

    def fake_run(command, capture_output, text, check):
        class Completed:
            returncode = 1
            stdout = ""
            stderr = "moov atom not found"

        return Completed()

    monkeypatch.setattr("tastepack.video.subprocess.run", fake_run)

    with pytest.raises(VideoValidationError, match="Could not inspect video"):
        validate_input_video(video, config=TastepackConfig())


def test_file_over_gemini_limit_fails_preflight(tmp_path, monkeypatch):
    video = tmp_path / "huge.mp4"
    video.write_bytes(b"tiny")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.Path.stat",
        lambda self: type("Stat", (), {"st_size": 3})(),
    )

    with pytest.raises(VideoValidationError, match="larger than"):
        validate_input_video(video, config=TastepackConfig(max_file_size_bytes=2))


def test_too_long_video_fails_preflight(tmp_path, monkeypatch):
    video = tmp_path / "long.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path: {
            "duration_seconds": 1801.0,
            "width": 1280,
            "height": 720,
            "video_stream_count": 1,
            "audio_stream_count": 1,
            "video_codec": "h264",
            "audio_codec": "aac",
            "file_size_bytes": 4,
        },
    )

    with pytest.raises(VideoValidationError, match="longer than"):
        validate_input_video(video, config=TastepackConfig(max_duration_seconds=1800))


def test_missing_audio_fails_unless_visual_only_is_allowed(tmp_path, monkeypatch):
    video = tmp_path / "silent.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path: {
            "duration_seconds": 10.0,
            "width": 1280,
            "height": 720,
            "video_stream_count": 1,
            "audio_stream_count": 0,
            "video_codec": "h264",
            "audio_codec": None,
            "file_size_bytes": 4,
        },
    )

    with pytest.raises(VideoValidationError, match="audio stream"):
        validate_input_video(video, config=TastepackConfig())

    metadata = validate_input_video(video, config=TastepackConfig(allow_no_audio=True))
    assert metadata["audio_stream_count"] == 0


def test_valid_video_preflight_returns_metadata(tmp_path, monkeypatch):
    video = tmp_path / "valid.mov"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path: {
            "duration_seconds": 12.5,
            "width": 640,
            "height": 360,
            "video_stream_count": 1,
            "audio_stream_count": 1,
            "video_codec": "h264",
            "audio_codec": "aac",
            "file_size_bytes": 4,
        },
    )

    metadata = validate_input_video(video, config=TastepackConfig())

    assert metadata["duration_seconds"] == 12.5
    assert metadata["width"] == 640
