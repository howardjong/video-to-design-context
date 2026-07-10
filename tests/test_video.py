import json
import subprocess
from hashlib import sha256

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


def test_video_validation_rejects_non_regular_files(tmp_path):
    video_directory = tmp_path / "not-a-video.mp4"
    video_directory.mkdir()

    with pytest.raises(VideoValidationError, match="not a regular file"):
        validate_input_video(video_directory, require_tools=False)


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

    def fake_run(command, capture_output, text, check, timeout):
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
    with pytest.raises(VideoValidationError, match="larger than"):
        validate_input_video(video, config=TastepackConfig(max_file_size_bytes=2))


def test_too_long_video_fails_preflight(tmp_path, monkeypatch):
    video = tmp_path / "long.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path, timeout_seconds: {
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
        lambda path, timeout_seconds: {
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

    monkeypatch.setattr("tastepack.video.validate_media_decode", lambda *args: None)
    metadata = validate_input_video(video, config=TastepackConfig(allow_no_audio=True))
    assert metadata["audio_stream_count"] == 0


def test_valid_video_preflight_returns_metadata(tmp_path, monkeypatch):
    video = tmp_path / "valid.mov"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path, timeout_seconds: {
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

    monkeypatch.setattr("tastepack.video.validate_media_decode", lambda *args: None)
    metadata = validate_input_video(video, config=TastepackConfig())

    assert metadata["duration_seconds"] == 12.5
    assert metadata["width"] == 640


def test_preflight_uses_bounded_full_decode_and_records_source_fingerprint(tmp_path, monkeypatch):
    video = tmp_path / "valid.mp4"
    video.write_bytes(b"fake")
    calls = []
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")

    def fake_run(command, capture_output, text, check, timeout):
        calls.append(command)
        if command[0] == "ffprobe":
            assert timeout == 2
            return type(
                "Completed",
                (),
                {
                    "returncode": 0,
                    "stdout": json.dumps(
                        {
                            "format": {"duration": "12.5"},
                            "streams": [
                                {
                                    "codec_type": "video",
                                    "codec_name": "h264",
                                    "width": 640,
                                    "height": 360,
                                },
                                {"codec_type": "audio", "codec_name": "aac"},
                            ],
                        }
                    ),
                    "stderr": "",
                },
            )()
        assert timeout == 3
        stderr = (
            "[Parsed_volumedetect_0] mean_volume: -20.0 dB"
            if "volumedetect" in command
            else ""
        )
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": stderr})()

    monkeypatch.setattr("tastepack.video.subprocess.run", fake_run)

    metadata = validate_input_video(
        video,
        config=TastepackConfig(
            ffprobe_timeout_seconds=2,
            ffmpeg_timeout_seconds=3,
        ),
    )

    assert metadata["source_mtime_ns"] == video.stat().st_mtime_ns
    assert metadata["file_size_bytes"] == video.stat().st_size
    assert metadata["source_sha256"] == sha256(video.read_bytes()).hexdigest()
    ffmpeg_calls = [command for command in calls if command[0] == "ffmpeg"]
    assert any("0:v:0" in command and "0:a:0" in command for command in ffmpeg_calls)
    assert any("volumedetect" in command for command in ffmpeg_calls)


def test_ffprobe_timeout_is_reported_as_a_preflight_error(tmp_path, monkeypatch):
    video = tmp_path / "slow.mp4"
    video.write_bytes(b"fake")

    def timeout_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("ffprobe", 2)

    monkeypatch.setattr("tastepack.video.subprocess.run", timeout_run)

    with pytest.raises(VideoValidationError, match="ffprobe timed out after 2.0s"):
        from tastepack.video import probe_video_metadata

        probe_video_metadata(video, timeout_seconds=2)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            {
                "duration_seconds": float("nan"),
                "width": 1280,
                "height": 720,
                "video_stream_count": 1,
                "audio_stream_count": 1,
                "video_codec": "h264",
                "audio_codec": "aac",
            },
            "invalid duration",
        ),
        (
            {
                "duration_seconds": 10.0,
                "width": 0,
                "height": 720,
                "video_stream_count": 1,
                "audio_stream_count": 1,
                "video_codec": "h264",
                "audio_codec": "aac",
            },
            "invalid dimensions",
        ),
        (
            {
                "duration_seconds": 10.0,
                "width": 1280,
                "height": 720,
                "video_stream_count": 1,
                "audio_stream_count": 1,
                "video_codec": "",
                "audio_codec": "aac",
            },
            "usable video codec",
        ),
    ],
)
def test_preflight_rejects_invalid_video_metadata(tmp_path, monkeypatch, metadata, message):
    video = tmp_path / "invalid.mp4"
    video.write_bytes(b"fake")
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path, timeout_seconds: metadata,
    )

    with pytest.raises(VideoValidationError, match=message):
        validate_input_video(video, config=TastepackConfig())


def test_decode_failure_and_silent_audio_are_rejected_unless_visual_only(tmp_path, monkeypatch):
    video = tmp_path / "audio.mp4"
    video.write_bytes(b"fake")
    metadata = {
        "duration_seconds": 10.0,
        "width": 1280,
        "height": 720,
        "video_stream_count": 1,
        "audio_stream_count": 1,
        "video_codec": "h264",
        "audio_codec": "aac",
    }
    monkeypatch.setattr("tastepack.video.shutil.which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(
        "tastepack.video.probe_video_metadata",
        lambda path, timeout_seconds: metadata,
    )

    def decode_failure(command, capture_output, text, check, timeout):
        return type(
            "Completed",
            (),
            {"returncode": 1, "stdout": "", "stderr": "bad audio packet"},
        )()

    monkeypatch.setattr("tastepack.video.subprocess.run", decode_failure)
    with pytest.raises(
        VideoValidationError,
        match="Full media decode validation failed: bad audio packet",
    ):
        validate_input_video(video, config=TastepackConfig())

    def silent_audio(command, capture_output, text, check, timeout):
        stderr = "mean_volume: -91.0 dB" if "volumedetect" in command else ""
        return type("Completed", (), {"returncode": 0, "stdout": "", "stderr": stderr})()

    monkeypatch.setattr("tastepack.video.subprocess.run", silent_audio)
    with pytest.raises(VideoValidationError, match="effectively silent"):
        validate_input_video(video, config=TastepackConfig())

    visual_only_metadata = validate_input_video(
        video,
        config=TastepackConfig(allow_no_audio=True),
    )
    assert visual_only_metadata["audio_mean_volume_db"] is None
