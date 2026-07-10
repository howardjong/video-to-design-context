from __future__ import annotations

import json
from pathlib import Path

import pytest

from tastepack.config import TastepackConfig
from tastepack.inbox_queue import (
    IntakePaths,
    QueueLockedError,
    acquire_dispatcher_lock,
    process_inbox,
)
from tastepack.video import VideoValidationError


def write_complete_pack(output_dir: Path, source_hash: str) -> None:
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text(
        json.dumps({"run_status": "complete", "source_sha256": source_hash})
    )


def test_inbox_processes_multiple_jobs_and_archives_sources(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    (paths.inbox / "first.mp4").write_bytes(b"first")
    (paths.inbox / "second.mov").write_bytes(b"second")
    processed: list[str] = []

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": f"hash-{path.stem}", "duration_seconds": 5.0}

    def runner(path: Path, out: Path, _config: TastepackConfig, **_kwargs):
        processed.append(path.name)
        write_complete_pack(out, f"hash-{path.stem}")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 2
    assert summary.failed == 0
    assert processed == ["first.mp4", "second.mov"]
    assert sorted(path.name for path in paths.inbox.iterdir()) == []
    assert [item["status"] for item in summary.jobs] == ["complete", "complete"]
    assert all((paths.archive / item["archive_path"]).is_file() for item in summary.jobs)
    assert all(
        (paths.output / item["output_path"] / "metadata.json").is_file()
        for item in summary.jobs
    )


def test_input_failure_moves_only_that_source_to_failed_and_continues(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    (paths.inbox / "bad.mp4").write_bytes(b"bad")
    (paths.inbox / "good.mp4").write_bytes(b"good")
    processed: list[str] = []

    def preflight(path: Path, **_kwargs):
        if path.name == "bad.mp4":
            raise VideoValidationError("missing audible audio")
        return {"source_sha256": "good-hash", "duration_seconds": 5.0}

    def runner(path: Path, out: Path, _config: TastepackConfig, **_kwargs):
        processed.append(path.name)
        write_complete_pack(out, "good-hash")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 1
    assert summary.failed == 1
    assert processed == ["good.mp4"]
    failed_job = next(job for job in summary.jobs if job["status"] == "failed")
    assert failed_job["failure"]["category"] == "input"
    assert (paths.failed / failed_job["failed_path"]).is_file()


def test_duplicate_source_and_config_is_skipped_without_running_again(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    (paths.inbox / "original.mp4").write_bytes(b"same-video")
    calls: list[str] = []

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": "same-hash", "duration_seconds": 5.0}

    def runner(path: Path, out: Path, _config: TastepackConfig, **_kwargs):
        calls.append(path.name)
        write_complete_pack(out, "same-hash")

    config = TastepackConfig(produce_pdf=False)
    first = process_inbox(intake, config, stable_seconds=0, preflight=preflight, runner=runner)
    (paths.inbox / "copied.mp4").write_bytes(b"same-video")
    second = process_inbox(intake, config, stable_seconds=0, preflight=preflight, runner=runner)

    assert first.completed == 1
    assert second.skipped == 1
    assert calls == ["original.mp4"]
    assert second.jobs[0]["status"] == "skipped"


def test_dispatcher_lock_rejects_second_runner(tmp_path: Path) -> None:
    paths = IntakePaths.from_root(tmp_path / "tastepack-data")
    paths.ensure()

    with acquire_dispatcher_lock(paths):
        with pytest.raises(QueueLockedError):
            with acquire_dispatcher_lock(paths):
                pass
