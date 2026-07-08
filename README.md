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
the input must be decodable, contain a video stream, have a positive duration,
stay under the configured file-size and duration limits, and include an audio
stream unless visual-only mode is explicitly enabled.

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

## Output

The output directory contains:

```text
claude-pack/
  design_preferences.md
  taste_packet.md
  taste_packet.pdf
  transcript.md
  metadata.json
  frames/
    ...
```

`taste_packet.md` is the main Claude upload context. It includes source metadata,
transcript, assets/examples, preference moments, timestamps, confidence scores,
and frame references.

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
  "produce_pdf": true,
  "fallback_interval_seconds": 2,
  "max_duration_seconds": 1800,
  "max_file_size_bytes": 2147483648,
  "allow_no_audio": false,
  "gemini_max_retries": 3,
  "gemini_retry_base_delay_seconds": 1.0,
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
