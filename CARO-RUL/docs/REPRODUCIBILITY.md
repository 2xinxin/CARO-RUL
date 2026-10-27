# Reproducibility guide

## Released input dataset

Processed device CSVs, source-ID mapping, and SHA-256 hashes are included in `dataset/`. Run `python scripts/verify_public_dataset.py` before training. No additional dataset download is needed. Historical trained model checkpoints, prediction files and paper-result outputs are not included in this release; experiments must be run to regenerate them. The B019/B020 upstream runs needed by `build_revision_tables.py` require separate execution/artifacts.

## Core protocol

- Split complete physical devices before preprocessing or window construction.
- Fit imputation and standardization on source-device prefixes only.
- Use 128 causal training checkpoints and 32 validation checkpoints per source device.
- Use the fixed seeds `17, 29, 41, 53, 67`.
- Report both percentage-of-recorded-life and endpoint-independent fixed-event-count schedules.
- Average seeds within device before device-level intervals or paired tests.

The exact contract is encoded in `configs/b016_caro_unified_5seed.yaml`.

## Main benchmark

```bash
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/b016_caro_unified_5seed.yaml
```

Use `--model NAME` and `--target SWN` to run a resumable subset. Do not combine a subset run with published aggregate claims until every configured target and seed is complete.

## Mechanism ablations and classical estimators

```bash
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/b017_caro_mechanism_5seed.yaml
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/b018_classical_unified_5seed.yaml
```

## Reviewer-requested controls

```bash
python scripts/revision/run_iorh_controls.py
python scripts/revision/run_nested_source_selection.py
python scripts/revision/analyze_revision_evidence.py
python scripts/revision/build_revision_tables.py
```

These scripts write to `results/revision/` and `outputs/`. Analysis scripts require their upstream result files and fail rather than silently substituting manuscript values.

## Audited forecasting baselines

Follow `vendor/README.md`, then run:

```bash
python scripts/revision/run_baseline_audit.py
```

The audit is deliberately separate from the lightweight installation because it depends on pinned third-party source trees with their own licenses.

## Feature extraction

`scripts/revision/extract_waveform_features.py` documents the exact finite-pair filtering, stable time ordering, 10% baseline window, first threshold crossings, FWHM, and positive pulse-area conventions used for raw waveforms.
