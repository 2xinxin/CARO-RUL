from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results" / "revision"
TABLES = PROJECT / "outputs" / "tables"
FIGURES = PROJECT / "outputs" / "figures"
BENCHMARK_RESULTS = PROJECT / "results" / "PCSS_RICH_BENCHMARK"
B016 = BENCHMARK_RESULTS / "B016_CARO_UNIFIED_5SEED" / "fold_metrics.csv"
B017 = BENCHMARK_RESULTS / "B017_CARO_MECHANISM_5SEED" / "fold_metrics.csv"
B019 = BENCHMARK_RESULTS / "B019_UNIFIED_27_BASELINES" / "fold_metrics.csv"
B020 = BENCHMARK_RESULTS / "B020_CARO_EVIDENCE_EXTENSION" / "fold_metrics.csv"
A2_003 = RESULTS / "A2-003"
A2_004 = RESULTS / "A2-004"


LABELS = {
    "population_median": "Population median", "population_geomean": "Population geometric mean",
    "initial_state_ridge": "Initial-state ridge", "ridge": "Ridge", "svr": "SVR",
    "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
    "extra_trees": "Extra trees", "wiener": "Wiener process", "mlp": "MLP", "cnn": "CNN",
    "rnn": "RNN", "gru": "GRU", "lstm": "LSTM", "bilstm": "BiLSTM", "tcn": "TCN",
    "deep_coral_tcn": "Deep CORAL--TCN", "groupdro_tcn": "GroupDRO--TCN",
    "dlinear_faithful": "DLinear", "dlinear": "DLinear", "transformer": "Transformer",
    "sa_mcd": "SA-MCD", "sa_mcd_adapted": "Adapted SA-MCD",
    "informer_faithful": "Informer", "nbeats_faithful": "N-BEATS", "nbeats": "N-BEATS",
    "patchtst_faithful": "PatchTST", "patchtst": "PatchTST",
    "timesnet_faithful": "TimesNet", "pathformer_adapted": "Pathformer", "caro_rul": "CARO-RUL",
    "cdap_point": "TCN + CDAP", "tcn_iorh": "TCN + IORH",
    "caro_no_recency": "CARO w/o recency", "caro_no_reliability": "CARO w/o reliability",
    "caro_no_trend": "CARO w/o trend", "caro_ordered_only": "CARO ordered-only",
}


def device_bootstrap(frame: pd.DataFrame, column: str, draws: int = 20000, seed: int = 20261026):
    values = frame.groupby("target")[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = sampled.mean(axis=1)
    return values.mean(), np.quantile(means, 0.025), np.quantile(means, 0.975), values.max()


def replace_audited_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    audited = pd.read_csv(A2_004 / "fold_metrics.csv")
    rename = {"dlinear": "dlinear_faithful", "nbeats": "nbeats_faithful", "patchtst": "patchtst_faithful"}
    audited["model"] = audited.model.map(rename)
    audited = audited.rename(columns={"mae_events": "mae", "oer": "overestimation_rate"})
    frame = frame[~frame.model.isin(rename.values())].copy()
    common = [c for c in frame.columns if c in audited.columns]
    return pd.concat([frame, audited[common]], ignore_index=True, sort=False)


def main_baseline_table() -> pd.DataFrame:
    frame = replace_audited_baselines(pd.read_csv(B019))
    rows = []
    for model, group in frame[frame.evaluation.eq("fractional_life")].groupby("model"):
        nmae, lo, hi, worst = device_bootstrap(group, "nmae")
        rows.append({
            "model": model, "label": LABELS.get(model, model), "nmae": nmae,
            "nmae_ci_low": lo, "nmae_ci_high": hi, "worst_device_nmae": worst,
            "nrmse": group.groupby("target").nrmse.mean().mean(),
            "oer": group.groupby("target").overestimation_rate.mean().mean(),
            "r2": group.groupby("target").r2.mean().mean(),
            "parameters": group.parameters.dropna().astype(float).median() if "parameters" in group else np.nan,
        })
    out = pd.DataFrame(rows).sort_values("nmae")
    out.to_csv(TABLES / "revision_main_baseline_table.csv", index=False)

    fixed = []
    for model, group in frame[frame.evaluation.eq("fixed_event_count")].groupby("model"):
        nmae, lo, hi, worst = device_bootstrap(group, "nmae")
        fixed.append({"model": model, "label": LABELS.get(model, model), "nmae": nmae,
                      "nmae_ci_low": lo, "nmae_ci_high": hi, "worst_device_nmae": worst,
                      "nrmse": group.groupby("target").nrmse.mean().mean(),
                      "oer": group.groupby("target").overestimation_rate.mean().mean(),
                      "r2": group.groupby("target").r2.mean().mean()})
    pd.DataFrame(fixed).sort_values("nmae").to_csv(TABLES / "revision_fixed_count_table.csv", index=False)
    return frame


def ablation_tables():
    frame = pd.read_csv(B016)
    rows = []
    for evaluation in ["fractional_life", "fixed_event_count"]:
        for model in ["tcn", "cdap_point", "tcn_iorh", "caro_rul"]:
            group = frame[(frame.evaluation.eq(evaluation)) & (frame.model.eq(model))]
            nmae, lo, hi, worst = device_bootstrap(group, "nmae")
            rows.append({"evaluation": evaluation, "model": model, "label": LABELS[model],
                         "nmae": nmae, "nmae_ci_low": lo, "nmae_ci_high": hi,
                         "worst_device_nmae": worst, "nrmse": group.groupby("target").nrmse.mean().mean(),
                         "r2": group.groupby("target").r2.mean().mean()})
    pd.DataFrame(rows).to_csv(TABLES / "revision_ablation_confidence_intervals.csv", index=False)

    mech = pd.read_csv(B017)
    mech_rows = []
    for (evaluation, model), group in mech.groupby(["evaluation", "model"]):
        nmae, lo, hi, worst = device_bootstrap(group, "nmae")
        mech_rows.append({"evaluation": evaluation, "model": model, "label": LABELS[model],
                          "nmae": nmae, "nmae_ci_low": lo, "nmae_ci_high": hi,
                          "worst_device_nmae": worst})
    pd.DataFrame(mech_rows).to_csv(TABLES / "revision_mechanism_ablation.csv", index=False)


def severe_corruption_table():
    frame = pd.read_csv(B020)
    severe = frame[frame.condition.isin(["waveform_dropout_50", "recent_missing_16"])].copy()
    out = severe.groupby(["evaluation", "condition", "model"], as_index=False).agg(
        nmae=("nmae", "mean"), nrmse=("nrmse", "mean"), oer=("overestimation_rate", "mean"), r2=("r2", "mean")
    )
    out["label"] = out.model.map(LABELS)
    out.to_csv(TABLES / "revision_severe_corruption.csv", index=False)


def nested_table():
    prediction = pd.read_csv(A2_003 / "outer_predictions.csv")
    prediction["absolute_error"] = (prediction.predicted_rul - prediction.true_rul).abs()
    prediction["signed_error"] = prediction.predicted_rul - prediction.true_rul
    prediction["normalized_absolute_error"] = prediction.absolute_error / prediction.lifetime
    fold_rows = []
    for (evaluation, target, seed), group in prediction.groupby(["evaluation", "outer_target", "seed"]):
        fold_rows.append({"evaluation": evaluation, "target": target, "seed": seed,
                          "nmae": group.normalized_absolute_error.mean(),
                          "mae_events": group.absolute_error.mean(),
                          "oer": (group.signed_error > 0).mean()})
    outer = pd.DataFrame(fold_rows)
    rows = []
    for evaluation, group in outer.groupby("evaluation"):
        nmae, lo, hi, worst = device_bootstrap(group, "nmae")
        rows.append({"evaluation": evaluation, "nmae": nmae, "nmae_ci_low": lo, "nmae_ci_high": hi,
                     "worst_device_nmae": worst, "mae_events": group.groupby("target").mae_events.mean().mean(),
                     "oer": group.groupby("target").oer.mean().mean()})
    pd.DataFrame(rows).to_csv(TABLES / "revision_nested_source_selection.csv", index=False)
    selected = pd.read_csv(A2_003 / "nested_selection.csv")
    selected.groupby(["residual_scale", "risk_margin"], as_index=False).size().sort_values(
        "size", ascending=False
    ).to_csv(TABLES / "revision_nested_choice_frequency.csv", index=False)


def plot_validation_traces():
    selected = pd.read_csv(A2_004 / "learning_rate_selection.csv")
    records = []
    for row in selected.itertuples(index=False):
        lr_text = str(float(row.selected_learning_rate))
        path = A2_004 / "training_traces" / f"{row.model}_target-{row.target}_seed-{row.seed}_lr-{lr_text}.csv"
        if not path.exists():
            candidates = list((A2_004 / "training_traces").glob(f"{row.model}_target-{row.target}_seed-{row.seed}_lr-*.csv"))
            path = min(candidates, key=lambda p: abs(float(p.stem.split("lr-")[-1]) - float(row.selected_learning_rate)))
        trace = pd.read_csv(path)
        trace["model"] = row.model
        trace["target"] = row.target
        trace["seed"] = row.seed
        records.append(trace)
    data = pd.concat(records, ignore_index=True)
    data.to_csv(A2_004 / "selected_training_traces.csv", index=False)

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8.2, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.50, 2.65), layout="constrained")
    styles = {"dlinear": ("#4C78A8", "o"), "nbeats": ("#F58518", "s"), "patchtst": ("#0B7A75", "^")}
    for model, (color, marker) in styles.items():
        sub = data[data.model.eq(model)]
        epochs = np.arange(1, int(sub.epoch.max()) + 1)
        matrix = []
        for _, run in sub.groupby(["target", "seed"]):
            values = np.full(len(epochs), np.nan)
            e = run.epoch.to_numpy(int) - 1
            values[e] = run.validation_mae_log.to_numpy(float)
            matrix.append(values)
        matrix = np.asarray(matrix)
        median = np.nanmedian(matrix, axis=0)
        q25 = np.nanpercentile(matrix, 25, axis=0)
        q75 = np.nanpercentile(matrix, 75, axis=0)
        valid = np.sum(np.isfinite(matrix), axis=0) >= 5
        ax.plot(epochs[valid], median[valid], color=color, lw=1.5, marker=marker, markevery=10,
                ms=3, label=LABELS[model])
        ax.fill_between(epochs[valid], q25[valid], q75[valid], color=color, alpha=0.15, linewidth=0)
    ax.set(xlabel="Epoch", ylabel="Source-validation MAE (log-RUL)")
    ax.grid(color="#E4E7EA", lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(FIGURES / "revision_baseline_validation_traces.pdf")
    fig.savefig(FIGURES / "revision_baseline_validation_traces.png", dpi=400)
    plt.close(fig)


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = main_baseline_table()
    ablation_tables()
    severe_corruption_table()
    nested_table()
    plot_validation_traces()
    manifest = {
        "rows_after_baseline_replacement": len(frame),
        "models": sorted(frame.model.unique().tolist()),
        "devices": sorted(frame.target.unique().tolist()),
        "seeds": sorted(int(x) for x in frame.seed.unique()),
        "independent_unit": "physical device; confidence intervals resample seven device-level seed means",
    }
    (RESULTS / "A2-006").mkdir(parents=True, exist_ok=True)
    (RESULTS / "A2-006" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
