from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tastepack.config import TastepackConfig
from tastepack.inbox_queue import (
    GeminiGate,
    IntakePaths,
    ProviderCircuitBreaker,
    QueueLockedError,
    acquire_dispatcher_lock,
    process_inbox,
    watch_inbox,
)
from tastepack.pipeline import FailureCategory, PipelineFailure
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


def test_provider_failure_stops_later_jobs_and_leaves_them_in_inbox(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (paths.inbox / name).write_bytes(name.encode())
    calls: list[str] = []

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": f"hash-{path.stem}", "duration_seconds": 5.0}

    def runner(path: Path, _out: Path, _config: TastepackConfig, **_kwargs):
        calls.append(path.name)
        raise PipelineFailure(
            "Gemini analysis",
            "schema-invalid Gemini output",
            "Fix the Gemini response before continuing.",
            FailureCategory.PROVIDER,
        )

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        workers=1,
        preflight=preflight,
        runner=runner,
    )

    assert summary.halted is True
    assert summary.failed == 1
    assert calls == ["a.mp4"]
    assert sorted(path.name for path in paths.inbox.iterdir()) == ["b.mp4", "c.mp4"]


def test_gemini_concurrency_limit_is_shared_across_parallel_workers(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    for name in ("first.mp4", "second.mp4"):
        (paths.inbox / name).write_bytes(name.encode())
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": f"hash-{path.stem}", "duration_seconds": 5.0}

    def runner(path: Path, out: Path, _config: TastepackConfig, **kwargs):
        nonlocal active, maximum_active
        with kwargs["gemini_permit"]():
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
        write_complete_pack(out, f"hash-{path.stem}")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        workers=2,
        gemini_concurrency=1,
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 2
    assert maximum_active == 1


def test_parallel_job_waiting_for_open_circuit_returns_to_inbox(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    for name in ("a.mp4", "b.mp4"):
        (paths.inbox / name).write_bytes(name.encode())
    first_job_started = threading.Event()

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": f"hash-{path.stem}", "duration_seconds": 5.0}

    def runner(path: Path, _out: Path, _config: TastepackConfig, **kwargs):
        if path.name == "a.mp4":
            with kwargs["gemini_permit"]():
                first_job_started.set()
                time.sleep(0.05)
                raise PipelineFailure(
                    "Gemini analysis",
                    "schema-invalid Gemini output",
                    "Fix Gemini before continuing.",
                    FailureCategory.PROVIDER,
                )
        assert first_job_started.wait(timeout=1)
        with kwargs["gemini_permit"]():
            raise AssertionError("The circuit breaker should stop this Gemini attempt")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        workers=2,
        gemini_concurrency=1,
        preflight=preflight,
        runner=runner,
    )

    assert summary.halted is True
    assert summary.failed == 1
    assert summary.deferred == 1
    assert (paths.inbox / "b.mp4").is_file()


def test_watch_mode_processes_stable_inbox_and_stops_cleanly(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    (paths.inbox / "watch.mp4").write_bytes(b"watch")
    stop_event = threading.Event()
    observed = []

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": "watch-hash", "duration_seconds": 5.0}

    def runner(_path: Path, out: Path, _config: TastepackConfig, **_kwargs):
        write_complete_pack(out, "watch-hash")

    summary = watch_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        poll_seconds=0,
        stop_event=stop_event,
        on_summary=lambda item: (observed.append(item), stop_event.set()),
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 1
    assert len(observed) == 1


def test_shared_gemini_gate_observes_rate_limit_cooldown() -> None:
    now = [100.0]
    delays = []

    class RateLimitedError(RuntimeError):
        status_code = 429

    def sleep(delay: float) -> None:
        delays.append(delay)
        now[0] += delay

    gate = GeminiGate(
        1,
        ProviderCircuitBreaker(),
        sleep=sleep,
        clock=lambda: now[0],
    )
    gate.observe_retry(3.0, RateLimitedError("retry later"))

    with gate.permit():
        pass

    assert delays == [3.0]
