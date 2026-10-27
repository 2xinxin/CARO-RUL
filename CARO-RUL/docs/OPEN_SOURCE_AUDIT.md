# Public source-and-processed-data packaging audit

The distributed repository contains the CARO-RUL model code, benchmark and reviewer-requested scripts, environment metadata and MIT software license, **plus** the seven processed SW1--SW7 observation-sequence CSVs and provenance manifests.

Scope of included data: 7 devices; 69,355 retained acquisition rows; 67,022 waveform-backed rows; 2,333 additional SW5 peak-only rows. Original raw oscilloscope archives, trained checkpoints and run-level predictions are **not** in this package.

The public data release is verified with `scripts/verify_public_dataset.py`, which checks per-file SHA-256 hashes, row counts, waveform counts, chronological event indexes, and RUL targets. Unit tests and an actual-data loading/model smoke test should be completed as part of the publication workflow.

The MIT software license does not implicitly license third-party baseline source or experimental data. Dataset reuse terms and any raw waveform access must be confirmed separately by the respective rights holders. Source archive filenames are included for provenance, but their descriptive words are not verified operating-condition values.
