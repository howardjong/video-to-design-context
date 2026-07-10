import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tastepack.config import TastepackConfig
from tastepack.frames import (
    build_fallback_frames,
    build_ffmpeg_extract_command,
    extract_frames,
    frame_filename,
    select_frames,
    select_frames_for_analysis,
)
from tastepack.schema import AssetExample, SuggestedFrame, TasteAnalysis


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


def test_frame_selection_covers_each_asset_before_selecting_extra_frames():
    frames = [
        SuggestedFrame(asset_id="a", timestamp="1", reason="a1", confidence=0.99),
        SuggestedFrame(asset_id="a", timestamp="2", reason="a2", confidence=0.98),
        SuggestedFrame(asset_id="b", timestamp="3", reason="b1", confidence=0.97),
        SuggestedFrame(asset_id="b", timestamp="4", reason="b2", confidence=0.96),
        SuggestedFrame(asset_id="c", timestamp="5", reason="c1", confidence=0.95),
        SuggestedFrame(asset_id="c", timestamp="6", reason="c2", confidence=0.94),
    ]

    selected = select_frames(
        frames,
        TastepackConfig(
            frame_confidence_threshold=0,
            max_frames_per_asset=2,
            max_total_frames=3,
        ),
    )

    assert {frame.asset_id for frame in selected} == {"a", "b", "c"}


def test_frame_selection_falls_back_for_an_uncovered_asset_when_capacity_remains():
    analysis = analysis_with_frames(["00:00:06"])
    analysis.assets.append(
        AssetExample(
            id="b",
            name="Asset B",
            kind="website",
            start_timestamp="00:00:12",
            end_timestamp="00:00:18",
            summary="Second asset",
        )
    )
    analysis.suggested_frames.append(
        SuggestedFrame(
            asset_id="b",
            timestamp="00:00:14",
            reason="Weak model suggestion",
            confidence=0.1,
        )
    )

    selected = select_frames_for_analysis(
        analysis,
        TastepackConfig(frame_confidence_threshold=0.5, max_total_frames=4),
    )

    assert {frame.asset_id for frame in selected} == {"a", "b"}
    assert any(
        frame.asset_id == "b" and frame.reason == "Fallback interval frame" for frame in selected
    )


def test_out_of_range_frame_timestamps_are_clamped_before_video_eof():
    frames = [SuggestedFrame(asset_id="a", timestamp="99", reason="late", confidence=0.9)]

    selected = select_frames(frames, TastepackConfig(), video_duration_seconds=10)

    assert selected[0].timestamp_seconds == pytest.approx(9.9)


def test_out_of_asset_range_frames_are_rejected_before_selection():
    with pytest.raises(Exception, match="Frame timestamp is outside asset range"):
        analysis_with_frames(["00:00:12"])


def test_frames_without_asset_video_overlap_are_dropped():
    analysis = TasteAnalysis.model_validate(
        {
            "source_summary": "Review",
            "transcript": "Narration",
            "assets": [
                {
                    "id": "late-asset",
                    "name": "Late asset",
                    "kind": "website",
                    "start_timestamp": "00:00:15",
                    "end_timestamp": "00:00:20",
                    "summary": "A late asset range.",
                }
            ],
            "preference_moments": [
                {
                    "asset_id": "late-asset",
                    "timestamp": "00:00:16",
                    "sentiment": "positive",
                    "preference": "Likes the tightly grouped navigation controls.",
                    "rationale": "The grouping shortens the visual scan path.",
                    "categories": ["layout"],
                    "confidence": 0.9,
                }
            ],
            "suggested_frames": [
                {
                    "asset_id": "late-asset",
                    "timestamp": "00:00:16",
                    "reason": "Shows the late range.",
                    "confidence": 0.9,
                }
            ],
            "visual_details": {},
            "motion_details": {},
        }
    )

    selected = select_frames_for_analysis(
        analysis,
        TastepackConfig(),
        video_duration_seconds=10,
    )

    assert selected == []


def test_frames_are_deduplicated_within_the_asset_range():
    analysis = analysis_with_frames(["00:00:09", "00:00:09"])

    selected = select_frames_for_analysis(analysis, TastepackConfig())

    assert [frame.timestamp_seconds for frame in selected] == [9]


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
        "-vf",
        "format=yuvj420p",
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
        Image.new("RGB", (1, 1), "white").save(Path(command[-1]), format="JPEG")

        class Completed:
            returncode = 0
            stderr = ""

        return Completed()

    monkeypatch.setattr("tastepack.frames.subprocess.run", fake_run)

    extracted = extract_frames(video, [frame], tmp_path / "pack")

    frame_path = tmp_path / "pack" / extracted[0].relative_path
    assert frame_path.exists()
    with Image.open(frame_path) as image:
        assert image.format == "JPEG"


def test_frame_extraction_preserves_frames_from_different_assets_at_same_timestamp(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    frames = [
        SuggestedFrame(asset_id="asset-a", timestamp="00:00:03", reason="first", confidence=0.9),
        SuggestedFrame(asset_id="asset-b", timestamp="00:00:03", reason="second", confidence=0.9),
    ]

    extracted = extract_frames(video, frames, tmp_path / "pack", skip_ffmpeg=True)

    assert len(extracted) == 2
    assert {frame.asset_id for frame in extracted} == {"asset-a", "asset-b"}
    assert len({frame.relative_path for frame in extracted}) == 2


def test_frame_filenames_do_not_collide_for_distinct_sanitized_asset_ids(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    frames = [
        SuggestedFrame(asset_id="a/b", timestamp="00:00:03", reason="first", confidence=0.9),
        SuggestedFrame(asset_id="a?b", timestamp="00:00:03", reason="second", confidence=0.9),
    ]

    extracted = extract_frames(video, frames, tmp_path / "pack", skip_ffmpeg=True)

    assert len({frame.relative_path for frame in extracted}) == 2


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
    assert f"frames/{frame_filename(frame)}" in message
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


def test_frame_extraction_fails_when_ffmpeg_writes_an_invalid_jpeg(tmp_path, monkeypatch):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake")
    frame = SuggestedFrame(asset_id="asset", timestamp="00:00:03", reason="test", confidence=0.9)

    def fake_run(command, capture_output, text, check):
        Path(command[-1]).write_bytes(b"not a jpeg")

        class Completed:
            returncode = 0
            stderr = ""

        return Completed()

    monkeypatch.setattr("tastepack.frames.subprocess.run", fake_run)

    with pytest.raises(Exception, match="invalid JPEG"):
        extract_frames(video, [frame], tmp_path / "pack")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_real_ffmpeg_extracts_a_frame_clamped_before_video_eof(tmp_path):
    video = tmp_path / "two-seconds.mp4"
    created = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=32x32:d=2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert created.returncode == 0, created.stderr
    frame = SuggestedFrame(asset_id="asset", timestamp="99", reason="late", confidence=0.9)

    selected = select_frames([frame], TastepackConfig(), video_duration_seconds=2.0)
    extracted = extract_frames(video, selected, tmp_path / "pack")

    assert selected[0].timestamp_seconds == pytest.approx(1.9)
    with Image.open(tmp_path / "pack" / extracted[0].relative_path) as image:
        assert image.format == "JPEG"
