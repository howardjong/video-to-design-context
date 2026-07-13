# Tastepack Project Context

Last updated: 2026-07-13

## Purpose

Tastepack turns a narrated UI, website, dashboard, app, or presentation recording
into an evidence-backed design-taste packet for another model or a human reviewer.
The runtime is local-first: source media and credentials remain local, while only
the selected analysis video is sent to Gemini.

## Current Operating Contract

- The production video model is `gemini-3.5-flash`.
- Inbox input is either one MP4/MOV or one folder containing exactly one MP4/MOV
  plus optional MP3 and timestamped Markdown companions.
- A silent video with exactly one MP3 companion is privately muxed into one MP4 for
  Gemini. The MP3 and transcript are never standalone Gemini inputs.
- The original video remains the source of frame extraction and provenance. Companion
  files are recorded in the manifest and move with the source bundle on archive,
  failure, and retry.
- A complete output is atomic. It includes the analysis, packet Markdown, design
  preferences, transcript, metadata, frames, PDF when enabled, and `taste_packet.zip`.
  Source media and logs never enter the delivery ZIP.

## Quality and Safety Gates

- Gemini output is untrusted evidence, not the final answer. Cross-object schema and
  timestamp validation must succeed before any pack can be promoted.
- When enabled, evidence-grounded QA uses OpenAI Responses with
  `gpt-5.6-terra`, `high` reasoning, and `enforce` mode by default. It creates an
  independent frame inventory, audits claims against frames and the timestamped
  transcript, and only promotes a cited reviewed delivery.
- Provider, semantic-validation, frame, artifact, and promotion failures hard-fail
  without promoting partial tastepacks. Provider retries require explicit operator
  acknowledgement.
- Secrets must not be emitted to output, manifests, or logs. Runtime data under
  `tastepack-data/` is ignored by Git.

## Provider Decision

ModelArk Seed 2.0 Pro remains experimental and is not a production replacement.
On a full recording with embedded audio, it returned visually plausible observations
but invented narration and attributed unsupported preferences to the speaker. Gemini
reproduced the narration closely and extracted substantially more relevant preference
moments, although its raw timestamps and visual claims still require QA. Do not switch
providers without a new evidence-backed evaluation.

## Working Guidance

1. Read `README.md` and `tastepack-data/README.md` before changing intake, output,
   or provider behavior.
2. Check `git status --short --branch` first. Preserve unrelated local changes and
   never touch incidental `.DS_Store` files.
3. For queue incidents, use `queue-status`, then inspect the job manifest and per-job
   log before considering a retry.
4. Never manually promote output or edit a manifest to bypass validation.
5. Keep durable public project behavior here; keep machine-local operational history
   in the ignored `HANDOFF.md`.

## Verification Baseline

The evidence-first implementation has a green local suite (`uv run pytest -q`) and
Ruff validation (`uv run ruff check src tests scripts`). Re-run both after code
changes; use a live provider run only when explicitly authorized.
