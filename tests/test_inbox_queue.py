from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from tastepack.config import TastepackConfig
from tastepack.gemini import MOCK_ANALYSIS
from tastepack.inbox_queue import (
    GeminiGate,
    IntakePaths,
    ProviderCircuitBreaker,
    QueueLockedError,
    RetryAcknowledgementRequired,
    acquire_dispatcher_lock,
    config_fingerprint,
    process_inbox,
    queue_status,
    retry_failed,
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
    assert not any(paths.processing.iterdir())


def test_inbox_processes_asset_bundle_and_archives_all_companions(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    bundle = paths.inbox / "van-holtz-site"
    bundle.mkdir()
    (bundle / "walkthrough.mp4").write_bytes(b"video")
    (bundle / "narration.mp3").write_bytes(b"audio reference")
    (bundle / "transcript.md").write_text("# Timestamped transcript\n", encoding="utf-8")
    processed: list[Path] = []

    def preflight(path: Path, **_kwargs):
        return {"source_sha256": "bundle-hash", "duration_seconds": 5.0}

    def runner(path: Path, out: Path, _config: TastepackConfig, **kwargs):
        processed.append(path)
        kwargs["lifecycle_callback"](
            "analysis_validated",
            {
                "analysis": MOCK_ANALYSIS,
                "provider_metadata": {"name": "gemini", "model": "gemini-3.5-flash"},
            },
        )
        write_complete_pack(out, "bundle-hash")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 1
    assert processed[0].name == "walkthrough.mp4"
    job = summary.jobs[0]
    assert job["source_name"] == "van-holtz-site"
    assert job["input_kind"] == "asset_bundle"
    assert job["source_video_name"] == "walkthrough.mp4"
    assert job["source_video_relative_path"] == "walkthrough.mp4"
    assert job["companion_assets"] == ["narration.mp3", "transcript.md"]
    assert job["output_path"].startswith("van-holtz-site--")
    archived_bundle = paths.archive / job["archive_path"]
    assert archived_bundle.is_dir()
    assert (archived_bundle / "walkthrough.mp4").is_file()
    assert (archived_bundle / "narration.mp3").is_file()
    assert (archived_bundle / "transcript.md").is_file()
    assert not (archived_bundle / "analysis-snapshot.json").exists()


def test_silent_bundle_uses_companion_audio_in_one_private_analysis_mp4(
    tmp_path: Path, monkeypatch
) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    bundle = paths.inbox / "silent-video"
    bundle.mkdir()
    (bundle / "walkthrough.mp4").write_bytes(b"silent video")
    (bundle / "narration.mp3").write_bytes(b"audible narration")
    (bundle / "transcript.md").write_text("[00:00.000] Hello\n", encoding="utf-8")
    preflight_allow_no_audio: list[bool] = []
    received_analysis_videos: list[Path] = []

    def preflight(path: Path, *, config: TastepackConfig, **_kwargs):
        preflight_allow_no_audio.append(config.allow_no_audio)
        if not config.allow_no_audio:
            raise VideoValidationError("Input audio is effectively silent")
        return {"source_sha256": "video-hash", "duration_seconds": 5.0}

    def fake_mux(video: Path, audio: Path, output: Path, _config: TastepackConfig) -> None:
        assert video.name == "walkthrough.mp4"
        assert audio.name == "narration.mp3"
        output.write_bytes(b"single analysis mp4")

    monkeypatch.setattr("tastepack.inbox_queue.mux_video_with_companion_audio", fake_mux)

    def runner(path: Path, out: Path, _config: TastepackConfig, **kwargs):
        assert path.name == "walkthrough.mp4"
        received_analysis_videos.append(kwargs["analysis_video"])
        write_complete_pack(out, "video-hash")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 1
    assert preflight_allow_no_audio == [False, True]
    assert received_analysis_videos[0].name == "analysis-input.mp4"
    job = summary.jobs[0]
    assert job["analysis_input"] == {
        "kind": "muxed_mp4_with_companion_audio",
        "source_video_relative_path": "walkthrough.mp4",
        "companion_audio_relative_path": "narration.mp3",
        "companion_audio_sha256": (
            "a61edb7887d61ebc0bc75c7e3f9944268a35a02a506b18f254063447d3c9878c"
        ),
    }
    assert not list(paths.archive.rglob("analysis-input.mp4"))
    metadata = json.loads((paths.output / job["output_path"] / "metadata.json").read_text())
    assert metadata["queue"]["analysis_input"] == job["analysis_input"]


def test_invalid_asset_bundle_moves_entire_bundle_to_failed_and_continues(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    invalid_bundle = paths.inbox / "missing-video"
    invalid_bundle.mkdir()
    (invalid_bundle / "narration.mp3").write_bytes(b"audio reference")
    (invalid_bundle / "transcript.md").write_text("# Transcript\n", encoding="utf-8")
    (paths.inbox / "good.mp4").write_bytes(b"video")
    processed: list[str] = []

    def preflight(path: Path, **_kwargs):
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
    assert "exactly one .mp4 or .mov video" in failed_job["failure"]["reason"]
    failed_bundle = paths.failed / failed_job["failed_path"]
    assert (failed_bundle / "narration.mp3").is_file()
    assert (failed_bundle / "transcript.md").is_file()


def test_retry_failed_asset_bundle_restores_all_assets_to_inbox(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    bundle = paths.inbox / "retry-bundle"
    bundle.mkdir()
    (bundle / "only-audio.mp3").write_bytes(b"audio reference")
    (bundle / "transcript.md").write_text("# Transcript\n", encoding="utf-8")

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
    )

    failed_job = summary.jobs[0]
    retried = retry_failed(intake, failed_job["job_id"])

    assert retried["status"] == "retry_queued"
    restored_bundle = paths.inbox / "retry-bundle"
    assert (restored_bundle / "only-audio.mp3").is_file()
    assert (restored_bundle / "transcript.md").is_file()


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


def test_recovery_resumes_persisted_analysis_without_calling_gemini(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    config = TastepackConfig(produce_pdf=False)
    source_hash = "recovery-hash"
    fingerprint = config_fingerprint(config)
    run_key = __import__("hashlib").sha256(f"{source_hash}:{fingerprint}".encode()).hexdigest()
    job_id = "recovery-job"
    processing_dir = paths.processing / job_id
    processing_dir.mkdir()
    source = processing_dir / "recovered.mp4"
    source.write_bytes(b"recovery-source")
    snapshot = processing_dir / "analysis-snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "analysis": MOCK_ANALYSIS,
                "provider_metadata": {"name": "gemini", "model": config.gemini_model},
                "source_sha256": source_hash,
                "config_fingerprint": fingerprint,
                "run_key": run_key,
            }
        )
    )
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "analysis_validated",
        "attempt": 1,
        "source_name": source.name,
        "claimed_path": f"{job_id}/{source.name}",
        "analysis_snapshot_path": f"{job_id}/{snapshot.name}",
        "source_sha256": source_hash,
        "config_fingerprint": fingerprint,
        "run_key": run_key,
        "output_path": f"recovered--{run_key[:12]}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "history": [],
    }
    (paths.jobs / f"{job_id}.json").write_text(json.dumps(manifest))
    resumed = []

    def preflight(_path: Path, **_kwargs):
        return {"source_sha256": source_hash, "duration_seconds": 5.0}

    def runner(path: Path, out: Path, _config: TastepackConfig, **kwargs):
        resumed.append(path.name)
        assert kwargs["precomputed_analysis"] is not None
        write_complete_pack(out, source_hash)

    summary = process_inbox(
        intake,
        config,
        stable_seconds=0,
        preflight=preflight,
        runner=runner,
    )

    assert summary.completed == 1
    assert resumed == ["recovered.mp4"]
    assert not (paths.inbox / "recovered.mp4").exists()
    assert len(list(paths.archive.rglob("recovered.mp4"))) == 1


def test_provider_failure_requires_explicit_acknowledgement_before_retry(tmp_path: Path) -> None:
    intake = tmp_path / "tastepack-data"
    paths = IntakePaths.from_root(intake)
    paths.ensure()
    (paths.inbox / "retry.mp4").write_bytes(b"retry")

    def preflight(_path: Path, **_kwargs):
        return {"source_sha256": "retry-hash", "duration_seconds": 5.0}

    def runner(_path: Path, _out: Path, _config: TastepackConfig, **_kwargs):
        raise PipelineFailure(
            "Gemini analysis",
            "temporary provider failure",
            "Resolve Gemini before retrying.",
            FailureCategory.PROVIDER,
        )

    summary = process_inbox(
        intake,
        TastepackConfig(produce_pdf=False),
        stable_seconds=0,
        preflight=preflight,
        runner=runner,
    )
    job_id = summary.jobs[0]["job_id"]

    status = queue_status(intake)
    assert status.state_counts["failed"] == 1
    with pytest.raises(RetryAcknowledgementRequired):
        retry_failed(intake, job_id)

    retried = retry_failed(intake, job_id, acknowledge_provider_retry=True)

    assert retried["status"] == "retry_queued"
    assert (paths.inbox / "retry.mp4").is_file()
