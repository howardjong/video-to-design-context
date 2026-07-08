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

If Gemini returns malformed JSON, rerun with a shorter video or use
`--mock-gemini --mock-payload` to validate a fixture locally.

If the API key is missing, confirm `.env` contains `GEMINI_API_KEY` or export it
in your shell.

If PDF generation fails because of local font or rendering issues, rerun with
`--no-pdf`; the Markdown artifacts are still generated.
