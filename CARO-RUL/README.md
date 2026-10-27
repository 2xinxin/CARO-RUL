# CARO-RUL

Source code and **processed seven-device benchmark dataset** for *From Switching Waveforms to Remaining Life: Risk-Aware Cross-Device Prognostics of SiC-PCSSs*.

CARO-RUL is a compact predictor that combines causal degradation attention pooling (CDAP) and an interval-aware ordered risk head (IORH). The repo includes device-level data, model and baseline implementations, exact main benchmark configurations, preprocessing rules, and scripts for reviewer-requested analyses.

> **Publication scope.** The dataset contains 7 independent SiC-PCSS run-to-breakdown records, **69,355 retained acquisition events**, and **67,022 waveform-backed events**. Each retained row is an **observed acquisition event**, not an independently verified physical switching pulse; there is no synchronized pulse counter covering all files. The original full oscilloscope waveforms are **not** contained here. Trained checkpoints, per-run predictions, and the numerical result archives needed to recreate all published figure/table values are also **not** included in this ZIP. These result artifacts need to be added separately if the repository is intended as a complete manuscript reproduction archive.

## Repository layout

```text
configs/                          Main and ablation experiment configurations
dataset/                          Public processed data + manifests + provenance
  processed/event_sequences/      SW1.csv ... SW7.csv (14 waveform/peak descriptors)
  records.csv                     Event and waveform counts by device
  id_mapping.csv                  Acquisition-era to publication ID mapping
  file_manifest.csv               SHA-256 integrity list
  DATA_CARD.md                    Dataset provenance and limits
  DATA_DICTIONARY.md              Column definitions
docs/                             Data access and reproducibility notes
examples/                         Synthetic data for lightweight tests
scripts/revision/                 Reviewer-requested experiments and analyses
scripts/verify_public_dataset.py  Verify all 7 CSVs and the SHA-256 manifest
src/                              CARO-RUL model, baselines, evaluation tools
tests/                            Unit tests
vendor/                           Instructions for external audited baselines
```

## Installation

```bash
conda env create -f environment.yml
conda activate caro-rul
pip install -e .
```

Or in an existing Python 3.10+ environment:

```bash
pip install -e ".[dev]"
```

## Verify the public release

From the repository root:

```bash
python scripts/verify_repository.py
python scripts/verify_public_dataset.py
pytest -q
```

The data integrity tool verifies the seven CSV checksums, one-based event indexes, observed-event RUL targets, waveform counts, and manifest records. The source archives referenced in the provenance metadata are **not** distributed here.

Optional synthetic-data pipeline check:

```bash
python examples/generate_synthetic_dataset.py
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/smoke_test.yaml --model caro_rul --target SW1
```

The synthetic test is not a reproduction of the manuscript metrics.

## Main seven-device experiments

The real processed files are already at `dataset/processed/event_sequences/`. No separate installation of the dataset is needed.

```bash
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/b016_caro_unified_5seed.yaml
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/b017_caro_mechanism_5seed.yaml
python src/pcss_rich_benchmark/run_baselines.py \
  --config configs/b018_classical_unified_5seed.yaml
```

These main runs can be computationally substantial. The reviewer-requested conservative controls, nested source-device selection, and baseline audit have separate commands in [Reproducibility](docs/REPRODUCIBILITY.md). The scripts generating consolidated revision tables require experiment result artifacts that are not bundled with this public release.

## Interpreting the dataset

All samples from each physical device remain in one fold. Source-only imputation/normalization is fitted anew for each held-out target. SW5 includes 2,333 peak-only events: other waveform fields are missing rather than fabricated. The label `rul_observed_events` denotes subsequent **retained** observations until the terminal record and cannot establish the number of physical switching operations missed between acquisitions. Source archive names are file provenance only, not authoritative operating-voltage or laser-energy metadata.

For the data table, original record IDs, column details, and known limits, see [Dataset README](dataset/README.md), [Data card](dataset/DATA_CARD.md) and [Data access](docs/DATA_ACCESS.md).

## Licensing and citation

The original CARO-RUL source code is under the repository's **MIT License**. **No independent license for reuse of the experimental dataset has been selected in the supplied materials**; do not assume the software license automatically covers the experimental records. Dataset rightsholders should add suitable license terms before relying on unrestricted downstream reuse. Third-party baseline source repositories are not vendored and retain their own licenses.

Use `CITATION.cff` until a final bibliographic citation is available.
