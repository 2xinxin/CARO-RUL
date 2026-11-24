# CARO-RUL: From Switching Waveforms to Remaining Life

Version 1.0.0 — repository release candidate for the IEEE Access paper.

**Publication gate:** the original `dataset/dataset_manifest.json` and `dataset/DATA_CARD.md` explicitly state that public redistribution is not yet cleared. This locally assembled package contains the processed benchmark and experimental artifacts at the authors' request, but does not grant redistribution rights. Obtain and document authorization for the processed data, predictions and model weights before pushing to a public repository. Software licensing must also be confirmed by the rights holders; third-party licenses remain unchanged. No raw oscilloscope waveform exports or private raw source ZIPs are included. `dataset/raw/source_archives.csv` contains provenance metadata only.

## Layout

- `dataset/`: processed SW1–SW7 event sequences (69,355 rows), metadata, feature dictionary, mapping, split policy and original checksums.
- `src/`, `configs/`, `scripts/`: model implementations, training and evaluation, original B016/B017/B018 configurations and analysis tools.
- `models/caro_rul_lodo/`: 35 historical trained CARO-RUL checkpoints.
- `results/PCSS_RICH_BENCHMARK/`: archived B016–B020 metrics, predictions where available, and manifests.
- `revision_experiments/`: conservative controls (A2-002), nested selection (A2-003), audited baselines (A2-004), diagnostics (A2-005), tables (A2-006), new CPU input ablation (A3-input-ablation), and licensed upstream snapshots.
- `paper_results/`: final paper's device-level event MAE, quantile diagnostics, all 78 regular at-risk checkpoints, and 105 historical intervention outcomes.
- `generated/`: created by analysis commands; preserved results are not changed unless noted.

## Setup

Run from the `CARO-RUL/` directory. For the original environment use `conda env create -f environment.yml` then `conda activate caro-rul`. The new CPU ablation was run using Python 3.13.5 and PyTorch 2.10.0+cpu; for that environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
```

On Windows, activate with `.venv\Scripts\activate` and use PowerShell syntax for environment variables. The external baseline audit uses the included official snapshots; see `revision_experiments/vendor/VERSIONS.md` and their original dependency instructions.

## 1. Verify archived evidence without retraining

```bash
python scripts/verify_package.py
python scripts/build_final_paper_evidence.py
python revision_experiments/summarize_input_ablation.py --output revision_experiments/results/A3-input-ablation
python scripts/analyze_b016_unified.py
python scripts/analyze_b020_evidence.py
python revision_experiments/analyze_revision_evidence.py
python revision_experiments/build_revision_tables.py
python scripts/plot_caro_results.py
```

Some original analysis commands regenerate derived outputs in the existing results directories; use a working copy when preservation of byte-identical archive files is required. The raw-provenance verification is intentionally incomplete without private originals. The archived metrics are used for the original paper's main comparison; the CPU rerun is a separate matched experiment.

## 2. Retrain principal baseline comparisons

Use a separate working copy. The runner resumes by default, so remove/move the matching archived output directory in that copy to force a new training run. For example, before B016 retraining, move `results/PCSS_RICH_BENCHMARK/B016_CARO_UNIFIED_5SEED` to a backup location outside the results tree. Then run:

```bash
export PYTHONPATH="$PWD/src"
python -m pcss_rich_benchmark.run_baselines --config configs/b016_caro_unified_5seed.yaml
python -m pcss_rich_benchmark.run_baselines --config configs/b017_caro_mechanism_5seed.yaml
python -m pcss_rich_benchmark.run_baselines --config configs/b018_classical_unified_5seed.yaml
```

Move the B017 and B018 result directories too before their respective fresh runs. The frozen configuration files retain historical settings. CUDA is used when available; CPU results are not guaranteed numerically identical. The original B019 merged comparison and B020 stress-test outputs are included for analysis; the original standalone B019 merge and B020 training drivers were not found in the supplied archive. B020 diagnostics can be regenerated from its stored predictions.

## 3. Rerun reviewer-requested experiments

```bash
python revision_experiments/run_iorh_controls.py --device auto
python revision_experiments/run_nested_source_selection.py --device auto
python revision_experiments/run_baseline_audit.py --device auto
python revision_experiments/analyze_revision_evidence.py
python revision_experiments/build_revision_tables.py
```

Use a working copy and preserve the A2-* result folders before rerunning. Some controls use the saved B016 CARO checkpoints; `python scripts/export_caro_models.py` retrains the 35 checkpoints if required. The audited forecasting baselines may need extra dependencies from the upstream source snapshots.

## 4. Reproduce matched input ablation (105 fits)

```bash
OUT=revision_experiments/results/A3-input-ablation-new
python revision_experiments/run_input_ablation.py --mode age_only age_missing full --target SW1 --seed 17 --device cpu --threads 2 --output "$OUT"
python revision_experiments/run_input_ablation.py --mode age_only age_missing full --device cpu --threads 1 --output "$OUT"
python revision_experiments/summarize_input_ablation.py --output "$OUT"
```

Expected archived macro nMAE (retrospective / fixed count): age only 0.4239 / 0.6814; age plus missingness 0.3779 / 0.6713; full 0.3446 / 0.5690. All 105 JSON records are provided, together with 210 fold-schedule metrics and 1,365 predictions. The main historical B016 reference is 0.3445 / 0.5682. Seeds 17, 29, 41, 53, 67; seven device-held-out folds; 128 acquisition window; 14,117 parameters for each input variant.

## Scope and limitations

RUL is measured in remaining recorded acquisitions, not physical switching cycles or elapsed time. The seven devices are the independent units; seed variation measures optimization variability. Original waveform extraction rules are included, but repeating waveform descriptor extraction or raw-to-processed identity checks requires the nonpublic raw exports. The historical gap-boundary sensitivity result is recorded in the manuscript; its original standalone driver was not found in the archive. Not every historical exploratory run is necessarily preserved. The repository supports rerunning the principal comparisons and direct checks from the included result files, not a claim of bit-for-bit reproduction of every historical experiment.
