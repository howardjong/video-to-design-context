# Tastepack Intake

- `inbox/`: videos waiting to be processed.
- `processing/`: atomically claimed videos and private restart state.
- `output/`: completed tastepacks, one directory per source video.
- `archive/`: source videos for successfully completed or duplicate jobs.
- `failed/`: inputs retained after a failed run for investigation or retry.
- `jobs/`: atomic job manifests with state, source fingerprint, output identity, and failure details.
- `logs/`: batch and per-job troubleshooting logs stored outside packs.

Runtime files in these directories are ignored by Git. The `.gitkeep` files preserve
the empty directory structure in fresh clones.
