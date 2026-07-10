# tastepack

`tastepack` is a local-first CLI that turns a narrated screen-recording of UI,
dashboard, website, app, or presentation design examples into a Claude-ready
design taste context pack.

It uses Gemini video understanding for structured analysis and `ffmpeg` for
frame extraction. Secrets stay in environment variables and are not written to
generated artifacts.

## Install

From the repo:

```bash
uv sync
```

Run the CLI through `uv`:

```bash
uv run tastepack process input.mp4 --out ./claude-pack
```

After installing the package into an environment, the command is:

```bash
tastepack process input.mp4 --out ./claude-pack
```

## System Dependencies

Install `ffmpeg`, which also provides `ffprobe`.

On macOS with Homebrew:

```bash
brew install ffmpeg
```

The CLI validates both tools before live frame extraction.

Live runs also use strict preflight checks before uploading anything to Gemini:
the input must be a regular MP4/MOV file with valid codecs and dimensions, pass
bounded `ffprobe` plus full video/audio `ffmpeg` decode checks, have a positive
duration, stay under the configured file-size and duration limits, and include
materially audible audio unless visual-only mode is explicitly enabled. The
preflight record includes a `source_sha256` fingerprint and is rechecked before
frame extraction.

## Gemini API Key

Copy `.env.example` to `.env` and fill in your key:

```bash
cp .env.example .env
```

```text
GEMINI_API_KEY=your-key-here
```

`GOOGLE_API_KEY` is also accepted. Do not commit `.env`; it is ignored by git.

## Usage

Process an MP4 or MOV:

```bash
uv run tastepack process input.mp4 --out ./claude-pack
```

Use a specific Gemini model:

```bash
uv run tastepack process input.mp4 --out ./claude-pack --model gemini-3.5-flash
```

Tune frame selection:

```bash
uv run tastepack process input.mp4 \
  --out ./claude-pack \
  --frame-confidence-threshold 0.7 \
  --max-frames-per-asset 4 \
  --max-total-frames 20
```

Skip PDF generation:

```bash
uv run tastepack process input.mp4 --out ./claude-pack --no-pdf
```

Adjust strict preflight and Gemini retry behavior:

```bash
uv run tastepack process input.mp4 \
  --out ./claude-pack \
  --max-duration-seconds 1800 \
  --max-file-size-mb 2048 \
  --gemini-max-retries 3
```

For intentionally visual-only recordings, opt in explicitly:

```bash
uv run tastepack process input.mp4 --out ./claude-pack --allow-no-audio
```

By default, uploaded Gemini Files API files are cleaned up after processing. To
leave uploaded files in Gemini's temporary file store, use:

```bash
uv run tastepack process input.mp4 --out ./claude-pack --no-cleanup-uploaded-files
```

For troubleshooting, write detailed step-by-step logs to a file:

```bash
uv run tastepack process input.mp4 \
  --out ./claude-pack \
  --verbosity debug \
  --log-file ./tastepack.log
```

Failures include the pipeline step, the reason, and a concrete next action. Logs
redact API key values before writing to the terminal or log file.

## Inbox Queue

For repeatable local processing, place videos in `tastepack-data/inbox/` and drain
the queue:

```bash
uv run tastepack process-inbox --data-dir ./tastepack-data
```

Each source is atomically claimed, preflighted, fingerprinted, and processed into
`output/<source-name>--<run-key>/`. Successful sources move to `archive/`; rejected
inputs move to `failed/<job-id>/`; manifests and per-job logs live in `jobs/` and
`logs/`. The source SHA-256 plus output-affecting configuration produces the run key,
so exact duplicate content is skipped only after the earlier complete pack validates.

Use bounded local workers while keeping Gemini conservative by default:

```bash
uv run tastepack process-inbox \
  --data-dir ./tastepack-data \
  --workers 2 \
  --gemini-concurrency 1
```

`--workers` limits claimed local jobs. `--gemini-concurrency` limits simultaneous
Gemini analyses across those workers. A shared `429` cooldown is honored by all
workers. A corrupt or unsupported source fails independently, but Gemini, schema,
frame, artifact, disk, permission, or promotion failures open the circuit breaker:
no new source is claimed and unclaimed inputs remain in `inbox/`.

Use `--watch --poll-seconds 2` to keep draining stable files after they have remained
unchanged for `--stable-seconds` (default `10`). Watch mode does not process hidden,
symlinked, unsupported, or actively changing files.

Inspect and retry work without editing manifests:

```bash
uv run tastepack queue-status --data-dir ./tastepack-data
uv run tastepack retry-failed JOB_ID --data-dir ./tastepack-data
uv run tastepack retry-failed JOB_ID --data-dir ./tastepack-data --acknowledge-provider-retry
```

Provider failures and interrupted Gemini calls require explicit acknowledgement before
retrying because the original request may have reached Gemini. A validated analysis
snapshot is persisted before local artifact work; restart resumes those local steps
without another Gemini call. A job interrupted before Gemini is safely requeued.

## Output

The output directory contains:

```text
claude-pack/
  analysis.json
  design_preferences.md
  taste_packet.md
  taste_packet.pdf
  transcript.md
  metadata.json
  frames/
    ...
```

`analysis.json` is the canonical validated structured analysis. `taste_packet.md`
is the main Claude upload context; it includes source metadata, asset/example
ranges, preference moments, confidence scores, categories, and asset-scoped
frame references. Valid extracted frames are embedded in both this Markdown and
the PDF when generated.

The transcript and on-screen text are marked as **untrusted source evidence**.
They are context to assess, never instructions to follow. `metadata.json` records
the source hash, model, prompt/schema/SDK versions, Gemini usage/timing telemetry,
and the final `run_status` only after every staged artifact validates.

## Long Videos and Provider Policy

For videos longer than `analysis_segment_seconds` (default `300`), tastepack uploads
the video once and analyzes overlapping time windows from that same Gemini Files API
object. Every segment must validate before their asset ranges, moments, and suggested
frames are deterministically merged; a failed segment fails the entire pack. Gemini
uses `high` media resolution by default for text-heavy UI recordings and caps each
segment response at `gemini_max_output_tokens` (default `8192`).

Before generation, metadata records the segment count and
`estimated_max_output_tokens` as the cost-planning upper bound: multiply that cap by
your current Gemini output-token rate to estimate the maximum generation charge.
Gemini 3.5 Flash remains the native video/audio analyzer. A future GPT-5.6 synthesis
pass is only an evaluation candidate and must demonstrate UAT gains in timestamp
accuracy, evidence fidelity, traceability, cost, and latency before adoption.

`design_preferences.md` distills reusable preferences across visual style,
layout, information hierarchy, typography, color, motion, interaction details,
dashboard-specific preferences, presentation-specific preferences, and negative
preferences.

The run is hard-fail by default. If Gemini analysis, frame extraction, Markdown
generation, metadata writing, or PDF generation fails, `tastepack` removes its
temporary staging directory and does not promote a partial pack. Existing output
directories are left untouched on failure.

## Mocked and Offline Mode

Use `--mock-gemini` for local tests or demos without calling Gemini:

```bash
uv run tastepack process input.mp4 --out ./claude-pack --mock-gemini --skip-ffmpeg --no-pdf
```

Use a fixture payload:

```bash
uv run tastepack process input.mp4 \
  --out ./claude-pack \
  --mock-gemini \
  --mock-payload ./analysis.json \
  --skip-ffmpeg
```

`--skip-ffmpeg` writes mock frame placeholders and avoids requiring a real video.

## Config File

Pass a JSON config file with `--config`:

```json
{
  "gemini_model": "gemini-3.5-flash",
  "frame_confidence_threshold": 0.65,
  "max_frames_per_asset": 6,
  "max_total_frames": 24,
  "frame_association_tolerance_seconds": 1.0,
  "produce_pdf": true,
  "fallback_interval_seconds": 2,
  "max_duration_seconds": 1800,
  "max_file_size_bytes": 2147483648,
  "allow_no_audio": false,
  "ffprobe_timeout_seconds": 30,
  "ffmpeg_timeout_seconds": 600,
  "frame_extraction_timeout_seconds": 30,
  "min_audio_mean_volume_db": -60,
  "analysis_segment_seconds": 300,
  "analysis_segment_overlap_seconds": 2,
  "gemini_media_resolution": "high",
  "gemini_max_output_tokens": 8192,
  "gemini_max_retries": 3,
  "gemini_retry_base_delay_seconds": 1.0,
  "gemini_retry_jitter_seconds": 0.25,
  "gemini_upload_timeout_seconds": 600,
  "gemini_file_processing_timeout_seconds": 600,
  "gemini_generation_timeout_seconds": 300,
  "gemini_cleanup_timeout_seconds": 30,
  "cleanup_uploaded_files": true,
  "verbosity": "normal"
}
```

CLI flags override config-file values.

## Development

Run tests:

```bash
uv run pytest
```

Run lint:

```bash
uv run ruff check .
```

## Troubleshooting

If the CLI says `ffmpeg` or `ffprobe` is missing, install `ffmpeg` and retry.

If a command fails, read the `Step`, `Why`, and `Next` fields in the error
message. Rerun with `--verbosity debug --log-file ./tastepack.log` when you need
the exact pipeline step history.

If preflight rejects the video as corrupt or undecodable, open it locally or
re-export it as a standard MP4/MOV before retrying.

If preflight rejects a video with no audio, re-record with narration or rerun
with `--allow-no-audio` when visual-only analysis is intentional.

If Gemini returns malformed JSON, rerun with a shorter video or use
`--mock-gemini --mock-payload` to validate a fixture locally.

If the API key is missing, confirm `.env` contains `GEMINI_API_KEY` or export it
in your shell.

If PDF generation fails because of local font or rendering issues, rerun with
`--no-pdf`; the Markdown artifacts are still generated.

For inbox jobs, inspect `tastepack-data/logs/<job-id>.log` alongside the batch log
reported by `process-inbox`. Both logs preserve step, reason, and next action while
redacting API keys and authorization values.
