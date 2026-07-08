import pytest

from tastepack.video import VideoValidationError, validate_input_video


def test_video_validation_rejects_missing_files(tmp_path):
    with pytest.raises(VideoValidationError, match="does not exist"):
        validate_input_video(tmp_path / "missing.mp4", require_tools=False)


def test_video_validation_rejects_unsupported_extensions(tmp_path):
    video = tmp_path / "input.avi"
    video.write_bytes(b"not real video")

    with pytest.raises(VideoValidationError, match="Unsupported"):
        validate_input_video(video, require_tools=False)
