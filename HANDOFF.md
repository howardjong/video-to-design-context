# Tastepack Handoff

**Last updated:** 2026-07-10  
**Current branch:** `main`  
**Current operational model:** `gemini-3.5-flash`

This is the persistent working record for agents and operators. Update it when a
meaningful implementation, verification, operational, or UAT milestone completes.
It is an operational history, not a place for API keys, tokens, video contents, or
raw provider responses.

## Current State

- The project is a local-first CLI that turns narrated UI, dashboard, website, app,
  or presentation videos into a complete design tastepack.
- The durable inbox queue is implemented and shipped. It has a dispatcher lock,
  atomic input claims and output promotion, per-job manifests/logs, bounded local
  concurrency, a shared Gemini gate/cooldown, recovery, and circuit breaking.
- `tastepack-data/inbox/van-holtz-site/` is ready for the next real UAT. It contains
  one MP4 plus MP3 and timestamped Markdown companion files. The latest non-destructive
  status check reported one pending input; it has not been uploaded to Gemini yet.
- The next intended action is a real inbox UAT, not more implementation work:

  ```zsh
  set -a
  source .env
  set +a
  uv run tastepack process-inbox \
    --data-dir ./tastepack-data \
    --workers 1 \
    --gemini-concurrency 1 \
    --verbosity debug
  ```

## Non-Negotiable Behavior

- No partial tastepack is promoted. Every artifact, extracted frame, optional PDF,
  metadata record, and output validation must succeed before promotion.
- Gemini analysis failures are provider failures: stop future claims, diagnose the
  cause, and do not proceed with later videos in that drain.
- Input-only failures, such as corrupt or no-audio video, fail that job before Gemini
  upload and let the queue continue with other inputs.
- Secrets must never appear in terminal output, logs, manifests, or generated packs.
  Logs use secret redaction and errors must explain `Step`, `Why`, and `Next`.
- Existing complete packs are never overwritten by a failed rerun. Exact duplicate
  video/config inputs are skipped only after the previous pack validates complete.

## Intake Contract

`inbox/` accepts either a direct `.mp4`/`.mov` or a one-folder asset bundle. A bundle
is a non-hidden folder directly inside `inbox/` and must contain exactly one `.mp4`
or `.mov`, including in nested folders. Example:

```text
tastepack-data/inbox/van-holtz-site/
  van-holtz-site.mp4
  van-holtz-site.mp3
  van-holtz-site-transcript.md
```

Only the selected MP4/MOV is preflighted and sent to Gemini. Companion MP3/Markdown
files stay local, are listed in the job manifest, and move atomically with the bundle
to `archive/`, `failed/`, back to `inbox/` when deferred, or during retry/recovery.
Output uses `<bundle-name>--<run-key>/` for bundles. Private recovery snapshots live
beside the claimed bundle and are never archived as source content.

## Architecture Map

- `src/tastepack/cli.py`: Typer commands: `process`, `process-inbox`, `queue-status`,
  and `retry-failed`.
- `src/tastepack/pipeline.py`: shared hard-fail pipeline used by single-video and
  inbox jobs; creates artifacts in staging and promotes only complete packs.
- `src/tastepack/video.py`: strict `ffprobe`/`ffmpeg` preflight and source hashing.
- `src/tastepack/gemini.py`: Files API upload, `ACTIVE` polling, structured JSON
  analysis, retries for transient failures, and remote-file cleanup.
- `src/tastepack/inbox_queue.py`: durable queue state, bundle handling, manifests,
  recovery, locking, shared Gemini concurrency/rate gate, and circuit breaker.
- `tastepack-data/`: runtime state. Inspect `jobs/<job-id>.json` and
  `logs/<job-id>.log` first when investigating a failure.

## Completed Milestones

| Milestone | Outcome |
| --- | --- |
| Gemini integration and file lifecycle | Live Gemini text/video smoke validated Files API upload, `ACTIVE` polling, structured analysis, and cleanup. |
| Strict media and output hardening | Corrupt, no-audio, invalid, overlong, oversized, and undecodable inputs are rejected before upload; failed artifacts remain unpromoted. |
| Operator diagnostics | Step/Why/Next errors, debug logs, provider telemetry, and secret redaction are in place. |
| Durable batch queue | Sequential/bounded concurrent inbox processing, duplicate avoidance, circuit breaking, crash recovery, watch mode, status, and explicit provider retry acknowledgement are implemented. |
| Asset-bundle intake | One MP4/MOV plus local MP3/Markdown companions can now be processed as an atomic folder. Tests cover completion, failure, retry, snapshot isolation, and the CLI. |

## Verification Record

- Latest full local verification after bundle intake: `uv run pytest` passed with
  **143 tests**; `uv run ruff check .` passed.
- Latest bundle status check: `uv run tastepack queue-status --data-dir ./tastepack-data`
  reported **one pending input** for the real `van-holtz-site` bundle.
- Earlier live checks established that `gemini-3.5-flash` works with the Files API,
  narrated video, PDF generation, archive completion, no-audio rejection, corrupt
  preflight rejection, duplicate skip behavior, and cleanup telemetry. Repeat a live
  smoke before changing the default model, provider SDK, provider workflow, or media
  toolchain.

## Recovery And Troubleshooting

1. Run `uv run tastepack queue-status --data-dir ./tastepack-data` before acting.
2. Read the corresponding `jobs/<job-id>.json` and `logs/<job-id>.log`; preserve the
   `Step`, `Why`, and `Next` details in any handoff or bug report.
3. For `failed` media, correct the input then run `retry-failed JOB_ID`.
4. For provider failures or `recovery_required`, inspect the manifest first, then use
   `retry-failed JOB_ID --acknowledge-provider-retry`. The acknowledgement exists
   because the interrupted Gemini request may already have been billable.
5. Do not manually promote output directories or edit manifests to bypass validation.
   Do not re-upload a video after an ambiguous Gemini interruption without the explicit
   acknowledgement.

## Update Protocol

At the end of an agent session or a meaningful checkpoint:

1. Update **Current State**, **Verification Record**, and **Completed Milestones** as
   needed. Add a dated entry below for material decisions, failures, or UAT results.
2. Record commands and outcomes, never secrets or raw credential-bearing logs.
3. Run the relevant tests and `uv run ruff check .` before committing code or docs.
4. Commit and push the handoff update with the implementation it describes.

## Handoff History

- **2026-07-10:** Added this persistent handoff record. Asset-bundle support was
  committed and pushed in `b22a7d2`; the real `van-holtz-site` MP4/MP3/Markdown
  bundle is queued for live UAT.
- **2026-07-08:** Shipped the hard-fail Gemini/video pipeline and durable inbox queue
  through checkpoint commits. Live Gemini verification and strict failure controls
  were completed before handing off real user-media UAT.
