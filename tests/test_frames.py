from pathlib import Path

from tastepack.config import TastepackConfig
from tastepack.frames import (
    build_ffmpeg_extract_command,
    select_frames,
)
from tastepack.schema import SuggestedFrame


def test_duplicate_frame_timestamps_are_deduplicated():
    frames = [
        SuggestedFrame(asset_id="a", timestamp="00:00:01", reason="first", confidence=0.9),
        SuggestedFrame(asset_id="a", timestamp="1", reason="duplicate", confidence=0.95),
        SuggestedFrame(asset_id="a", timestamp="00:00:02", reason="second", confidence=0.8),
    ]

    selected = select_frames(frames, TastepackConfig())

    assert [frame.timestamp_seconds for frame in selected] == [1.0, 2.0]
    assert selected[0].reason == "duplicate"


def test_low_confidence_frames_are_filtered_by_threshold():
    frames = [
        SuggestedFrame(asset_id="a", timestamp="1", reason="weak", confidence=0.2),
        SuggestedFrame(asset_id="a", timestamp="2", reason="strong", confidence=0.9),
    ]

    selected = select_frames(frames, TastepackConfig(frame_confidence_threshold=0.5))

    assert [frame.reason for frame in selected] == ["strong"]


def test_out_of_range_frame_timestamps_are_clamped_to_video_duration():
    frames = [SuggestedFrame(asset_id="a", timestamp="99", reason="late", confidence=0.9)]

    selected = select_frames(frames, TastepackConfig(), video_duration_seconds=10)

    assert selected[0].timestamp_seconds == 10


def test_ffmpeg_command_generation_handles_spaces_in_paths(tmp_path: Path):
    video = tmp_path / "input movie.mp4"
    output = tmp_path / "frame one.jpg"

    command = build_ffmpeg_extract_command(video, 12.5, output)

    assert command == [
        "ffmpeg",
        "-y",
        "-ss",
        "12.500",
        "-i",
        str(video),
        "-frames:v",
        "1",
        "-q:v",
        "2",
        str(output),
    ]
