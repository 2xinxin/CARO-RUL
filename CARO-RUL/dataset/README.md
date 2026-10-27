# SiC-PCSS Cross-Device RUL Benchmark v1.0.0

This repository contains the **processed, feature-level** benchmark corresponding to the CARO-RUL manuscript, with seven physical device records (SW1--SW7). The original full oscilloscope waveforms are not included.

The canonical prediction target is `rul_observed_events = N - event_index`, using the one-based index for a record of length `N`. One row represents a **retained acquisition observation**, not independently verified switching-pulse count. A complete physical device lifetime is the independent evaluation unit.

## Benchmark provenance

| Public ID | Acquisition-era ID | Events | Waveform-backed | Segments |
|---|---|---:|---:|---:|
| SW1 | SW2 | 4,632 | 4,632 | 1 |
| SW2 | SW3 | 15,070 | 15,070 | 4 |
| SW3 | SW5 | 1,807 | 1,807 | 1 |
| SW4 | SW6 | 14,374 | 14,374 | 4 |
| SW5 | SW7 | 10,101 | 7,768 | 4 |
| SW6 | EXP-20231108 | 3,199 | 3,199 | 1 |
| SW7 | EXP-20241205 | 20,172 | 20,172 | 6 |
| **Total** | | **69,355** | **67,022** | **21** |

The source IDs and segment names are retained as **provenance identifiers**, not as operating-condition measurements. Original legacy SW1 and SW4 peak-only archives were not admitted to the seven-device benchmark. The public SW5 contains 2,333 observations without the full waveform features; those entries have peak voltage and explicit `waveform_available = false`.

## Files

```text
dataset/
  processed/event_sequences/SW1.csv ... SW7.csv
  records.csv
  id_mapping.csv
  dataset_manifest.json
  file_manifest.csv
  splits/policy.json
  audit/validation_report.json
  DATA_CARD.md
  DATA_DICTIONARY.md
  raw/source_archives.csv
```

The `raw/source_archives.csv` file lists historical archive *metadata*; those large raw ZIP archives themselves are **not included**. Full provenance may include archive filenames with embedded terms such as voltages or energies; these filename strings should not be treated as authoritative per-device electrical or optical settings.

## Verification and evaluation

From the repository root, run `python scripts/verify_public_dataset.py`. Evaluate using held-out **whole devices**, fitting preprocessing and model selection on source devices only. See `configs/b016_caro_unified_5seed.yaml` and `docs/REPRODUCIBILITY.md`. Retrospective percentage-of-life checkpoints are endpoint-dependent and should not be presented as deployable online timepoints.

## License

Dataset reuse terms are not yet specified in the supplied materials. The repository MIT software license does not independently grant a dataset license. Confirm the applicable permissions and add explicit dataset terms before public publication. See `docs/DATA_ACCESS.md`.
