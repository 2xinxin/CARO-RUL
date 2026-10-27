# Reviewer reproducibility requirement map

This map distinguishes **files already in the repository** from the separate run-level artifacts required to verify all published numerical results.

| Reviewer request | Included public artifact | Still missing from this ZIP |
|---|---|---|
| Device data and source mapping | Seven `dataset/processed/event_sequences/SW*.csv` files, `dataset/records.csv`, `dataset/id_mapping.csv`, SHA-256 manifests | Original full waveforms (not needed to rerun feature-level models) |
| Preprocessing and feature definitions | `src/pcss_rich_benchmark/data.py`, `scripts/revision/extract_waveform_features.py`, `dataset/DATA_DICTIONARY.md` | Raw waveform exports for independently repeating descriptor extraction |
| Device folds and schedules | `configs/*.yaml`, `dataset/split_policy.json` | None for fold specification |
| Model and baseline configurations | `src/pcss_single_model/`, `src/tor_rul/`, `configs/` | Historical trained model checkpoints |
| Conservative IORH controls | `scripts/revision/run_iorh_controls.py` and public processed CSVs | Historical run-level predictions / outputs (can be regenerated) |
| Nested source-device selection | `scripts/revision/run_nested_source_selection.py` | Historical per-fold selection outputs |
| Forecasting-baseline audit | `scripts/revision/run_baseline_audit.py`, `vendor/README.md` | Licensed external source snapshots and historical audit runs |
| Table, calibration and intervention scripts | `scripts/revision/analyze_revision_evidence.py`, `scripts/revision/build_revision_tables.py` | Upstream B019/B020 and other historical run outputs needed to regenerate all tables directly |

The public repository is a **code-and-processed-data release**, **not** a complete collection of reported trained checkpoints, validation traces and run-level predictions. Any review response or paper statement claiming the latter must refer to a separately attached verified archive, or be corrected to match this repository.
