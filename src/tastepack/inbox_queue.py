from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from tastepack.artifacts import validate_complete_metadata
from tastepack.config import TastepackConfig
from tastepack.gemini import GEMINI_PROMPT_VERSION, GEMINI_SCHEMA_VERSION
from tastepack.logging import get_logger, redact_secrets
from tastepack.pipeline import FailureCategory, PipelineFailure, run_processing_job
from tastepack.video import VideoValidationError, validate_input_video

logger = get_logger("inbox")
SUPPORTED_EXTENSIONS = {".mp4", ".mov"}
TERMINAL_STATES = {"complete", "deferred", "failed", "skipped", "recovery_required"}


class QueueLockedError(RuntimeError):
    pass


class ProviderCircuitOpen(RuntimeError):
    pass


class ProviderCircuitBreaker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def trip(self, reason: str) -> None:
        with self._lock:
            self._reason = self._reason or reason

    def require_closed(self) -> None:
        if reason := self.reason:
            raise ProviderCircuitOpen(f"Gemini circuit breaker is open: {reason}")


class GeminiGate:
    def __init__(
        self,
        concurrency: int,
        circuit_breaker: ProviderCircuitBreaker,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if concurrency < 1:
            raise ValueError("gemini_concurrency must be at least 1")
        self._semaphore = threading.BoundedSemaphore(concurrency)
        self._circuit_breaker = circuit_breaker
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._next_allowed_at = 0.0

    @contextmanager
    def permit(self):
        self._circuit_breaker.require_closed()
        with self._semaphore:
            while True:
                self._circuit_breaker.require_closed()
                with self._lock:
                    delay = self._next_allowed_at - self._clock()
                if delay <= 0:
                    break
                self._sleep(delay)
            try:
                yield
            except Exception as exc:
                self._circuit_breaker.trip(redact_secrets(exc))
                raise

    def observe_retry(self, delay_seconds: float, exc: BaseException) -> None:
        if _error_code(exc) != 429:
            return
        with self._lock:
            self._next_allowed_at = max(self._next_allowed_at, self._clock() + delay_seconds)


@dataclass(frozen=True)
class IntakePaths:
    root: Path

    @classmethod
    def from_root(cls, root: Path) -> IntakePaths:
        return cls(root=root)

    @property
    def inbox(self) -> Path:
        return self.root / "inbox"

    @property
    def processing(self) -> Path:
        return self.root / "processing"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def archive(self) -> Path:
        return self.root / "archive"

    @property
    def failed(self) -> Path:
        return self.root / "failed"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def lock_file(self) -> Path:
        return self.root / ".dispatcher.lock"

    def ensure(self) -> None:
        for directory in (
            self.inbox,
            self.processing,
            self.output,
            self.archive,
            self.failed,
            self.jobs,
            self.logs,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class QueueSummary:
    jobs: list[dict[str, Any]] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    deferred: int = 0
    halted: bool = False
    halt_reason: str | None = None


@contextmanager
def acquire_dispatcher_lock(paths: IntakePaths):
    paths.root.mkdir(parents=True, exist_ok=True)
    handle = paths.lock_file.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise QueueLockedError(
                f"Another tastepack inbox dispatcher is already running for {paths.root}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "started_at": _timestamp()}))
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def process_inbox(
    root: Path,
    config: TastepackConfig,
    *,
    stable_seconds: float = 10.0,
    max_jobs: int | None = None,
    workers: int = 1,
    gemini_concurrency: int = 1,
    force: bool = False,
    mock_gemini: bool = False,
    mock_payload: Path | None = None,
    skip_ffmpeg: bool = False,
    preflight: Callable[..., dict[str, object]] = validate_input_video,
    runner: Callable[..., Any] = run_processing_job,
    sleep: Callable[[float], None] = time.sleep,
) -> QueueSummary:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if gemini_concurrency < 1:
        raise ValueError("gemini_concurrency must be at least 1")
    paths = IntakePaths.from_root(root)
    paths.ensure()
    summary = QueueSummary()
    circuit_breaker = ProviderCircuitBreaker()
    gemini_gate = GeminiGate(gemini_concurrency, circuit_breaker, sleep=sleep)

    with acquire_dispatcher_lock(paths):
        _recover_promoted_jobs(paths, summary)
        if summary.halted:
            return summary
        sources = iter(discover_stable_inputs(paths, stable_seconds=stable_seconds, sleep=sleep))
        pending: dict[Future[dict[str, Any]], dict[str, Any]] = {}
        exhausted = False
        claimed_count = 0
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="tastepack-inbox",
        ) as executor:
            while pending or not exhausted:
                while (
                    not exhausted
                    and not circuit_breaker.reason
                    and len(pending) < workers
                    and (max_jobs is None or claimed_count < max_jobs)
                ):
                    try:
                        source = next(sources)
                    except StopIteration:
                        exhausted = True
                        break
                    manifest = claim_input(paths, source)
                    claimed_count += 1
                    future = executor.submit(
                        _execute_claimed_job,
                        paths,
                        manifest,
                        config,
                        force=force,
                        mock_gemini=mock_gemini,
                        mock_payload=mock_payload,
                        skip_ffmpeg=skip_ffmpeg,
                        preflight=preflight,
                        runner=runner,
                        circuit_breaker=circuit_breaker,
                        gemini_gate=gemini_gate,
                    )
                    pending[future] = manifest
                if not pending:
                    break
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    manifest = pending.pop(future)
                    try:
                        future.result()
                    except Exception as exc:  # Defensive boundary for worker bugs.
                        _record_failure(paths, manifest, FailureCategory.SYSTEM, exc)
                        circuit_breaker.trip(redact_secrets(exc))
                    _record_summary(summary, manifest)
                if circuit_breaker.reason:
                    summary.halted = True
                    summary.halt_reason = circuit_breaker.reason
    return summary


def watch_inbox(
    root: Path,
    config: TastepackConfig,
    *,
    poll_seconds: float = 2.0,
    stop_event: threading.Event | None = None,
    on_summary: Callable[[QueueSummary], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    **process_options: Any,
) -> QueueSummary:
    if poll_seconds < 0:
        raise ValueError("poll_seconds must be non-negative")
    stop_event = stop_event or threading.Event()
    summary = QueueSummary()
    while not stop_event.is_set():
        summary = process_inbox(root, config, sleep=sleep, **process_options)
        if on_summary is not None:
            on_summary(summary)
        if summary.halted or stop_event.is_set():
            return summary
        sleep(poll_seconds)
    return summary


def _execute_claimed_job(
    paths: IntakePaths,
    manifest: dict[str, Any],
    config: TastepackConfig,
    *,
    force: bool,
    mock_gemini: bool,
    mock_payload: Path | None,
    skip_ffmpeg: bool,
    preflight: Callable[..., dict[str, object]],
    runner: Callable[..., Any],
    circuit_breaker: ProviderCircuitBreaker,
    gemini_gate: GeminiGate,
) -> dict[str, Any]:
    try:
        _process_claimed_job(
            paths,
            manifest,
            config,
            force=force,
            mock_gemini=mock_gemini,
            mock_payload=mock_payload,
            skip_ffmpeg=skip_ffmpeg,
            preflight=preflight,
            runner=runner,
            gemini_permit=gemini_gate.permit,
            retry_observer=gemini_gate.observe_retry,
        )
    except PipelineFailure as exc:
        if _was_caused_by(exc, ProviderCircuitOpen):
            _defer_claimed_job(paths, manifest, redact_secrets(exc))
        else:
            _record_failure(paths, manifest, exc.category, exc)
            if exc.category is not FailureCategory.INPUT:
                circuit_breaker.trip(redact_secrets(exc))
    except ProviderCircuitOpen as exc:
        _defer_claimed_job(paths, manifest, redact_secrets(exc))
    except VideoValidationError as exc:
        _record_failure(paths, manifest, FailureCategory.INPUT, exc)
    except Exception as exc:
        _record_failure(paths, manifest, FailureCategory.SYSTEM, exc)
        circuit_breaker.trip(redact_secrets(exc))
    return manifest


def _record_summary(summary: QueueSummary, manifest: dict[str, Any]) -> None:
    summary.jobs.append(manifest)
    if manifest["status"] == "complete":
        summary.completed += 1
    elif manifest["status"] == "skipped":
        summary.skipped += 1
    elif manifest["status"] == "deferred":
        summary.deferred += 1
    else:
        summary.failed += 1


def discover_stable_inputs(
    paths: IntakePaths,
    *,
    stable_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> list[Path]:
    candidates: list[Path] = []
    for path in sorted(paths.inbox.iterdir(), key=lambda item: item.name.casefold()):
        if not _is_supported_input(path):
            continue
        first = path.stat()
        if stable_seconds:
            sleep(stable_seconds)
        try:
            second = path.stat()
        except FileNotFoundError:
            continue
        if first.st_size == second.st_size and first.st_mtime_ns == second.st_mtime_ns:
            candidates.append(path)
        else:
            logger.info("Leaving unstable inbox file for a later run: %s", path.name)
    return candidates


def claim_input(paths: IntakePaths, source: Path) -> dict[str, Any]:
    job_id = uuid4().hex
    processing_dir = paths.processing / job_id
    processing_dir.mkdir(parents=True)
    claimed_path = processing_dir / source.name
    source.replace(claimed_path)
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "status": "claimed",
        "attempt": 1,
        "source_name": source.name,
        "claimed_path": str(claimed_path.relative_to(paths.processing)),
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
        "history": [{"state": "claimed", "at": _timestamp()}],
    }
    _write_manifest(paths, manifest)
    return manifest


def config_fingerprint(config: TastepackConfig) -> str:
    values = config.model_dump()
    for key in (
        "verbosity",
        "request_timeout_seconds",
        "ffprobe_timeout_seconds",
        "ffmpeg_timeout_seconds",
        "frame_extraction_timeout_seconds",
        "gemini_max_retries",
        "gemini_retry_base_delay_seconds",
        "gemini_retry_jitter_seconds",
        "gemini_upload_timeout_seconds",
        "gemini_file_processing_timeout_seconds",
        "gemini_generation_timeout_seconds",
        "gemini_cleanup_timeout_seconds",
        "cleanup_uploaded_files",
    ):
        values.pop(key, None)
    values["prompt_version"] = GEMINI_PROMPT_VERSION
    values["schema_version"] = GEMINI_SCHEMA_VERSION
    return _hash_json(values)


def _process_claimed_job(
    paths: IntakePaths,
    manifest: dict[str, Any],
    config: TastepackConfig,
    *,
    force: bool,
    mock_gemini: bool,
    mock_payload: Path | None,
    skip_ffmpeg: bool,
    preflight: Callable[..., dict[str, object]],
    runner: Callable[..., Any],
    gemini_permit: Callable[[], Any],
    retry_observer: Callable[[float, BaseException], None],
) -> None:
    claimed_path = paths.processing / manifest["claimed_path"]
    video_metadata = preflight(claimed_path, require_tools=not skip_ffmpeg, config=config)
    source_hash = video_metadata.get("source_sha256")
    if not isinstance(source_hash, str) or not source_hash:
        raise PipelineFailure(
            "Video preflight",
            "Preflight did not return a source_sha256 fingerprint",
            "Fix the preflight implementation before retrying this job.",
            FailureCategory.SYSTEM,
        )
    fingerprint = config_fingerprint(config)
    run_key = hashlib.sha256(f"{source_hash}:{fingerprint}".encode()).hexdigest()
    output_name = f"{_safe_stem(manifest['source_name'])}--{run_key[:12]}"
    output_dir = paths.output / output_name
    _transition(
        paths,
        manifest,
        "preflight_passed",
        source_sha256=source_hash,
        config_fingerprint=fingerprint,
        run_key=run_key,
        output_path=output_name,
        video_metadata=video_metadata,
    )

    existing_output = _find_complete_run(paths, run_key, source_hash)
    if not force and existing_output is not None:
        manifest["output_path"] = existing_output
        archive_path = _archive_source(paths, manifest, source_hash)
        _transition(paths, manifest, "skipped", archive_path=archive_path)
        return

    _transition(paths, manifest, "running")
    runner(
        claimed_path,
        output_dir,
        config,
        mock_gemini=mock_gemini,
        mock_payload=mock_payload,
        skip_ffmpeg=skip_ffmpeg,
        preflight_metadata=video_metadata,
        gemini_permit=gemini_permit,
        retry_observer=retry_observer,
    )
    if not _is_complete_pack(output_dir, source_hash):
        raise PipelineFailure(
            "Output validation",
            "The processing runner returned without a validated complete output pack",
            "Inspect artifact generation and output promotion before processing more videos.",
            FailureCategory.SYSTEM,
        )
    _annotate_pack(output_dir, manifest)
    _transition(paths, manifest, "output_promoted")
    archive_path = _archive_source(paths, manifest, source_hash)
    _transition(paths, manifest, "complete", archive_path=archive_path)


def _recover_promoted_jobs(paths: IntakePaths, summary: QueueSummary) -> None:
    for manifest_path in sorted(paths.jobs.glob("*.json")):
        manifest = _read_manifest(manifest_path)
        if not manifest or manifest.get("status") in TERMINAL_STATES:
            continue
        output_name = manifest.get("output_path")
        source_hash = manifest.get("source_sha256")
        claimed_path = manifest.get("claimed_path")
        if (
            isinstance(output_name, str)
            and isinstance(source_hash, str)
            and isinstance(claimed_path, str)
            and _is_complete_pack(paths.output / output_name, source_hash)
            and (paths.processing / claimed_path).is_file()
        ):
            try:
                _annotate_pack(paths.output / output_name, manifest)
                _transition(paths, manifest, "output_promoted")
                archive_path = _archive_source(paths, manifest, source_hash)
                _transition(paths, manifest, "complete", archive_path=archive_path)
                summary.jobs.append(manifest)
                summary.completed += 1
            except Exception as exc:
                _record_failure(paths, manifest, FailureCategory.SYSTEM, exc)
                summary.jobs.append(manifest)
                summary.failed += 1
                summary.halted = True
                summary.halt_reason = redact_secrets(exc)
        else:
            _transition(
                paths,
                manifest,
                "recovery_required",
                failure={
                    "category": FailureCategory.SYSTEM.value,
                    "step": "Recovery",
                    "reason": "A previous job stopped before a verified complete output was found",
                    "next": (
                        "Use retry-failed after reviewing the job manifest; "
                        "Gemini billing may recur."
                    ),
                },
            )


def _record_failure(
    paths: IntakePaths,
    manifest: dict[str, Any],
    category: FailureCategory,
    exc: BaseException,
) -> None:
    if isinstance(exc, PipelineFailure):
        step = exc.step
        next_step = exc.next_step
        reason = exc.reason
    elif category is FailureCategory.INPUT:
        step = "Video preflight"
        next_step = "Correct the input video and retry the failed job."
        reason = redact_secrets(exc)
    else:
        step = "Inbox processing"
        next_step = "Resolve this system failure before processing more inbox videos."
        reason = redact_secrets(exc)
    claimed_path = paths.processing / manifest["claimed_path"]
    failed_path = _failed_destination(paths, manifest)
    if claimed_path.exists():
        failed_path.parent.mkdir(parents=True, exist_ok=True)
        claimed_path.replace(failed_path)
    _transition(
        paths,
        manifest,
        "failed",
        failed_path=str(failed_path.relative_to(paths.failed)),
        failure={
            "category": category.value,
            "step": step,
            "reason": reason,
            "next": next_step,
        },
    )


def _defer_claimed_job(paths: IntakePaths, manifest: dict[str, Any], reason: str) -> None:
    claimed_path = paths.processing / manifest["claimed_path"]
    destination = _unique_destination(paths.inbox / manifest["source_name"])
    if claimed_path.exists():
        claimed_path.replace(destination)
    _transition(
        paths,
        manifest,
        "deferred",
        deferred_path=destination.name,
        deferred_reason=reason,
    )


def _archive_source(paths: IntakePaths, manifest: dict[str, Any], source_hash: str) -> str:
    claimed_path = paths.processing / manifest["claimed_path"]
    destination = paths.archive / f"{_safe_stem(manifest['source_name'])}--{source_hash[:12]}"
    destination = _unique_destination(destination / manifest["source_name"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    claimed_path.replace(destination)
    return str(destination.relative_to(paths.archive))


def _failed_destination(paths: IntakePaths, manifest: dict[str, Any]) -> Path:
    return paths.failed / manifest["job_id"] / manifest["source_name"]


def _annotate_pack(output_dir: Path, manifest: dict[str, Any]) -> None:
    metadata_path = output_dir / "metadata.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["queue"] = {
        "job_id": manifest["job_id"],
        "run_key": manifest["run_key"],
        "config_fingerprint": manifest["config_fingerprint"],
    }
    _atomic_write_json(metadata_path, payload)
    validate_complete_metadata(output_dir)


def _is_complete_pack(output_dir: Path, source_hash: str) -> bool:
    try:
        validate_complete_metadata(output_dir)
        metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    return metadata.get("source_sha256") == source_hash


def _find_complete_run(paths: IntakePaths, run_key: str, source_hash: str) -> str | None:
    for manifest_path in paths.jobs.glob("*.json"):
        manifest = _read_manifest(manifest_path)
        if not manifest or manifest.get("run_key") != run_key:
            continue
        output_name = manifest.get("output_path")
        if isinstance(output_name, str) and _is_complete_pack(
            paths.output / output_name,
            source_hash,
        ):
            return output_name
    return None


def _transition(paths: IntakePaths, manifest: dict[str, Any], state: str, **updates: Any) -> None:
    manifest.update(updates)
    manifest["status"] = state
    manifest["updated_at"] = _timestamp()
    manifest.setdefault("history", []).append({"state": state, "at": manifest["updated_at"]})
    _write_manifest(paths, manifest)


def _write_manifest(paths: IntakePaths, manifest: dict[str, Any]) -> None:
    _atomic_write_json(paths.jobs / f"{manifest['job_id']}.json", manifest)


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.error("Ignoring corrupt inbox manifest: %s", path.name)
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _is_supported_input(path: Path) -> bool:
    return (
        path.is_file()
        and not path.is_symlink()
        and not path.name.startswith(".")
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _safe_stem(source_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", Path(source_name).stem).strip("-").lower()
    return value[:80] or "video"


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10_000):
        candidate = path.with_stem(f"{path.stem}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not choose a unique archive path for {path.name}")


def _hash_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _timestamp() -> str:
    return datetime.now(UTC).isoformat()


def _error_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int):
            return value
    return None


def _was_caused_by(exc: BaseException, error_type: type[BaseException]) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, error_type):
            return True
        current = current.__cause__
    return False
