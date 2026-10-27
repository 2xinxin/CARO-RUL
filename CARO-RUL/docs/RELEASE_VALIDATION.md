# Combined code-and-dataset release checks (2026-10-27)

The GitHub-ready archive was built by combining the CARO-RUL Clean code ZIP with the supplied `dataset.zip`. The original seven numeric event-sequence CSVs were copied without modifying their contents. Repository-level documentation was updated to describe the public processed-data scope.

## Checks performed

- ZIP extraction and directory path-safety checks: passed.
- Dataset 7 physical records: passed.
- Observed acquisition rows: **69,355**; waveform-backed rows: **67,022**; SW5 peak-only rows: **2,333**.
- Acquisition segments in the 7 records: **21**.
- `event_index` is one-based contiguous; `rul_observed_events` is descending to zero: passed for all records.
- Record count/ID/segment and benchmark-CSV SHA-256 consistency: passed.
- Dataset `file_manifest.csv` sha256 and file size: passed after updating dataset documentation.
- `python scripts/verify_repository.py`: passed.
- `pytest -q`: **2 passed**.
- Real-data CPU CARO-RUL sample train/evaluation with outer target SW1, small smoke configuration: **1 complete, 0 failed**. This small run is a software test, not the manuscript's trained model or nMAE result.

## Current release boundaries

- Input processed data and code are present, but raw oscilloscope waveform archives and full historical benchmark output directories, saved model weights, and prediction files are absent.
- The article's final aggregate results have **not** been independently regenerated from scratch as part of this packaging check.
- Source archive filenames may contain apparent voltage/energy figures that are not independently validated device operating conditions.
- Confirm approval to publish the dataset and attach an explicit data-reuse license before public release. The code MIT license does not by itself apply to the dataset.

To repeat integrity checks, execute `python scripts/verify_public_dataset.py`, `python scripts/verify_repository.py`, and `pytest -q` from the repository root.
