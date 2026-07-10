# Tastepack Intake

- `inbox/`: videos waiting to be processed.
- `output/`: completed tastepacks, one directory per source video.
- `failed/`: inputs retained after a failed run for investigation or retry.
- `logs/`: debug logs stored outside their corresponding pack directory.

Runtime files in these directories are ignored by Git. The `.gitkeep` files preserve
the empty directory structure in fresh clones.
