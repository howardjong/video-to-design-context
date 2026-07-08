from pathlib import Path

import pytest

from tastepack.config import TastepackConfig
from tastepack.frames import (
    build_fallback_frames,
    build_ffmpeg_extract_command,
    extract_frames,
    select_frames,
    select_frames_for_analysis,
)
from tastepack.schema import SuggestedFrame, TasteAnalysis


def analysis_with_frames(frame_timestamps):
    return TasteAnalysis.model_validate(
        {
            "source_summary": "Review",
            "transcript": "Narration",
            "assets": [
                {
                    "id": "a",
                    "name": "Asset A",
                    "kind": "website",
                    "start_timestamp": "00:00:05",
                    "end_timestamp": "00:00:10",
                    "summary": "First asset",
                }
            ],
            "preference_moments": [
                {
                    "asset_id": "a",
                    "timestamp": "00:00:06",
                    "sentiment": "positive",
                    "preference": "Likes close alignment between headline and action.",
                    "rationale": "The relationship makes the primary action easy to scan.",
                    "categories": ["layout"],
                    "confidence": 0.8,
                }
            ],
            "suggested_frames": [
                {
                    "asset_id": "a",
                    "timestamp": timestamp,
                    "reason": "Suggested by model",
                    "confidence": 0.9,
                }
                for timestamp in frame_timestamps
            ],
            "visual_details": {
                "style": [],
                "layout": [],
                "information_hierarchy": [],
                "typography": [],
                "color": [],
                "dashboard": [],
                "presentation": [],
                "negative_preferences": [],
            },
            "motion_details": {
                "animations": [],
                "interaction_details": [],
                "motion_preferences": [],
            },
        }
    )


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


def test_frame_timestamps_are_clamped_to_asset_range():
    analysis = analysis_with_frames(["00:00:12"])

    selected = select_frames_for_analysis(analysis, TastepackConfig())

    assert selected[0].timestamp_seconds == 10


def test_frames_are_deduplicated_after_asset_range_clamping():
    analysis = analysis_with_frames(["00:00:11", "00:00:12"])

    selected = select_frames_for_analysis(analysis, TastepackConfig())

    assert [frame.timestamp_seconds for frame in selected] == [10]


def test_fallback_frames_are_created_when_gemini_suggests_none():
    analysis = analysis_with_frames([])

    fallback = build_fallback_frames(analysis.assets, TastepackConfig(fallback_interval_seconds=2))

    assert [frame.timestamp_seconds for frame in fallback] == [5.0, 7.0, 9.0]
    assert all(frame.reason == "Fallback interval frame" for frame in fallback)


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


def test_frame_extraction_writes_expected_files(tmp_path, monkeypatch):
    video = tmp_path / "input movie.mp4"
    video.write_bytes(b"fake")
    frame = SuggestedFrame(asset_id="asset 1", timestamp="00:00:03", reason="test", confidence=0.9)

    def fake_run(command, capture_output, text, check):
        Path(command[-1]).write_bytes(b"jpeg")

        class Completed:
            returncode = 0
            stderr = ""

        return Completed()

    monkeypatch.setattr("tastepack.frames.subprocess.run", fake_run)

    frame_map = extract_frames(video, [frame], tmp_path / "pack")

    frame_path = tmp_path / "pack" / frame_map[3.0]
    assert frame_path.exists()
    assert frame_path.read_bytes() == b"jpeg"


def test_frame_extraction_fails_when_ffmpeg_writes_no_file(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    frame = SuggestedFrame(asset_id="asset", timestamp="00:00:03", reason="test", confidence=0.9)

    def fake_run(command, capture_output, text, check):
        class Completed:
            returncode = 0
            stderr = ""

        return Completed()

    monkeypatch.setattr("tastepack.frames.subprocess.run", fake_run)

    with pytest.raises(Exception, match="missing"):
        extract_frames(video, [frame], tmp_path / "pack")


def test_frame_extraction_failure_includes_timestamp_output_and_stderr(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    frame = SuggestedFrame(asset_id="asset", timestamp="00:00:03", reason="test", confidence=0.9)

    def fake_run(command, capture_output, text, check):
        class Completed:
            returncode = 1
            stderr = "invalid seek"

        return Completed()

    monkeypatch.setattr("tastepack.frames.subprocess.run", fake_run)

    with pytest.raises(Exception) as exc_info:
        extract_frames(video, [frame], tmp_path / "pack")

    message = str(exc_info.value)
    assert "ffmpeg failed" in message
    assert "3.000" in message
    assert "frames/asset_000003000.jpg" in message
    assert "invalid seek" in message


def test_frame_extraction_fails_when_ffmpeg_writes_empty_file(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    frame = SuggestedFrame(asset_id="asset", timestamp="00:00:03", reason="test", confidence=0.9)

    def fake_run(command, capture_output, text, check):
        Path(command[-1]).write_bytes(b"")

        class Completed:
            returncode = 0
            stderr = ""

        return Completed()

    monkeypatch.setattr("tastepack.frames.subprocess.run", fake_run)

    with pytest.raises(Exception, match="empty"):
        extract_frames(video, [frame], tmp_path / "pack")
