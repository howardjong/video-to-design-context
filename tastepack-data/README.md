# Tastepack Intake

- `inbox/`: videos waiting to be processed, or one-folder asset bundles. A bundle
  contains exactly one `.mp4` or `.mov` plus optional local companions such as `.mp3`
  audio and a timestamped `.md` transcript.
- `processing/`: atomically claimed videos or bundles and private restart state.
- `output/`: completed tastepacks, one directory per source video.
- `archive/`: source videos or complete bundles for successfully completed or duplicate jobs.
- `failed/`: inputs or complete bundles retained after a failed run for investigation or retry.
- `jobs/`: atomic job manifests with state, source fingerprint, output identity, and failure details.
- `logs/`: batch and per-job troubleshooting logs stored outside packs.

Runtime files in these directories are ignored by Git. The `.gitkeep` files preserve
the empty directory structure in fresh clones.

The queue sends only the selected video to Gemini. Companion audio and transcript
files stay local, are listed in the job manifest, and are preserved when the bundle is
archived, failed, deferred, or retried.
