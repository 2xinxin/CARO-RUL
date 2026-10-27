# Public processed-data release and scope

This repository **includes** the processed event-sequence CSVs for seven SiC-PCSS devices (`dataset/processed/event_sequences/SW1.csv` through `SW7.csv`), their record counts, publication/source ID mapping, integrity manifests, and provenance metadata. They contain 69,355 chronological **observed acquisition** records, including 67,022 with waveform-derived descriptors. SW5 has 2,333 additional peak-only records.

A retained acquisition is not proof of a unique, independently counted laser-triggered physical switching operation. No synchronized pulse counter spans all historical acquisition/export segments. The event-index RUL target is the number of **later retained records** until the experimental end-of-life record; it is not verified physical switching-pulse count or elapsed operational time.

## Not included

- Original oscilloscope waveform archives (`private_raw_sources/`): generally far larger than the feature-level CSVs; paths/hashes in `dataset/raw/source_archives.csv` are provenance metadata, not downloadable files.
- Saved trained model weights and per-fold/run prediction files.
- Complete historical experiment result directories and all figure/table regeneration inputs.

Thus the release permits independent inspection of the **input processed data** and re-running the provided code, but it is **not** by itself a complete numerical review archive. The main benchmark and additional analyses must be run to regenerate metrics, and some post-processing expects separately generated upstream run outputs.

## Data identity and integrity

See `dataset/id_mapping.csv` for the mapping from acquisition-era names to SW1--SW7. The original segment and source archive labels are retained for provenance; archive names do **not** supersede the instrument/bias/optical settings reported in the manuscript. All seven dataset CSVs have SHA-256 hashes in `dataset/records.csv` and `dataset/file_manifest.csv`. Run:

```bash
python scripts/verify_public_dataset.py
```

## Reuse terms

The source-code `LICENSE` covers the authors' code, not automatically the experimental dataset. The supplied dataset materials carried a historical statement that public data redistribution was not cleared, but the processed CSVs are now included in this user-prepared public-release package. Before publication, the dataset rights holders must confirm permission and decide on an explicit dataset reuse license. Until then, no broader data reuse rights are claimed here.
