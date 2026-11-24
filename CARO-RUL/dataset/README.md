# SiC-PCSS Cross-Device RUL Benchmark v1.0.0

This directory is the authoritative benchmark used by the CARO-RUL manuscript.
It contains seven complete run-to-breakdown SiC photoconductive semiconductor
switch records with continuous publication identifiers `SW1`--`SW7`.

The primary task is remaining useful life prediction. At event index (t) in a
record of length (N), the target is `rul_observed_events = N - t`. One row is
one acquired switching event, and one complete device record is the independent
experimental unit.

## Record renumbering

| Benchmark ID | Source record ID | Events | Waveforms | Segments |
|---|---|---:|---:|---:|
| SW1 | SW2 | 4,632 | 4,632 | 1 |
| SW2 | SW3 | 15,070 | 15,070 | 4 |
| SW3 | SW5 | 1,807 | 1,807 | 1 |
| SW4 | SW6 | 14,374 | 14,374 | 4 |
| SW5 | SW7 | 10,101 | 7,768 | 4 |
| SW6 | EXP-20231108 | 3,199 | 3,199 | 1 |
| SW7 | EXP-20241205 | 20,172 | 20,172 | 6 |

The old identifiers remain in `source_record_id`; acquisition block names remain
in `source_segment_id`. The publication identifiers therefore change without
destroying provenance. The discarded legacy SW1 and SW4 peak-only sequences are
not part of this benchmark.

## Layout

```text
dataset/
├── processed/event_sequences/SW1.csv ... SW7.csv
├── raw/source_archives.csv
├── splits/policy.json
├── audit/validation_report.json
├── records.csv
├── id_mapping.csv
├── file_manifest.csv
├── dataset_manifest.json
├── DATA_CARD.md
└── DATA_DICTIONARY.md
```

## Evaluation protocol

Use leakage-safe leave-one-device-out evaluation. Split complete device records
before constructing windows. Fit imputation, normalization, feature selection,
and early stopping only on the six source devices. Test at 20%, 30%, 40%, 50%,
60%, 70%, and 80% of the held-out record. Report device-macro metrics and every
held-out device result.

Rebuild with the configured environment:

```powershell
C:\Users\Administrator\anaconda3\envs\myenv\python.exe scripts\build_caro_dataset_benchmark.py
```

The raw ZIP archives are preserved under `private_raw_sources/` and are
identified by size and SHA-256 hash in `raw/source_archives.csv`. Public
redistribution has not yet been cleared.
