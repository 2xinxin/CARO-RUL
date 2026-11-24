# Local release-candidate checks

Executed in Python 3.13.5, PyTorch 2.10.0+cpu with CPU thread limits:

- `scripts/verify_package.py`: PASS — all seven processed records, 69,355 events, dataset hashes, 1,470 B016 metric rows and 35 saved CARO models load and execute. Raw identity checks require private originals.
- `scripts/build_final_paper_evidence.py`: PASS — 105 historical intervention outcomes and 78 regular-grid at-risk checkpoints; early/late/missed counts 6/1/28, 7/2/26, 17/2/16 for lower quantile, width margin, constant TCN.
- `revision_experiments/summarize_input_ablation.py`: PASS — 105 run JSON records, 210 metric rows and 1,365 predictions; macro nMAE matches manuscript (0.4239/0.6814, 0.3779/0.6713, 0.3446/0.5690).
- `scripts/analyze_b016_unified.py`, `scripts/analyze_b020_evidence.py`, `revision_experiments/analyze_revision_evidence.py`, `revision_experiments/build_revision_tables.py`, `scripts/plot_caro_results.py`: all exited successfully.

Only analysis and saved-model checks were rerun for this packaging step; full historical training sweeps were not repeated. The new CPU experiment was already completed before this step. This is a local release candidate, not a verified public GitHub release. Original B020 training and boundary sensitivity drivers are not available in the supplied archive.
