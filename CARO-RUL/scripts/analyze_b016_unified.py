from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B016_CARO_UNIFIED_5SEED"
EXPECTED_MODELS = {
    "mlp", "cnn", "rnn", "gru", "lstm", "bilstm", "tcn",
    "deep_coral_tcn", "groupdro_tcn", "dlinear_faithful", "transformer",
    "sa_mcd", "informer_faithful", "nbeats_faithful", "patchtst_faithful",
    "timesnet_faithful", "pathformer_adapted", "sa_mcd_adapted",
    "cdap_point", "tcn_iorh", "caro_rul",
}
EXPECTED_TARGETS = {f"SW{index}" for index in range(1, 8)}
EXPECTED_SEEDS = {17, 29, 41, 53, 67}


def holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy())
    ordered = p_values.to_numpy()[order]
    adjusted = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * ordered)
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def paired_bootstrap(values: np.ndarray, seed: int = 20260917) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def main() -> None:
    metrics = pd.read_csv(RUN / "fold_metrics.csv")
    ok = metrics.loc[metrics.status.eq("complete")].copy()
    expected_rows = len(EXPECTED_MODELS) * len(EXPECTED_TARGETS) * len(EXPECTED_SEEDS) * 2
    keys = ok[["evaluation", "target", "seed", "model"]]
    integrity = {
        "expected_metric_rows": expected_rows,
        "actual_metric_rows": int(len(ok)),
        "duplicate_metric_keys": int(keys.duplicated().sum()),
        "models_complete": set(ok.model) == EXPECTED_MODELS,
        "targets_complete": set(ok.target) == EXPECTED_TARGETS,
        "seeds_complete": set(ok.seed.astype(int)) == EXPECTED_SEEDS,
        "evaluations": sorted(ok.evaluation.unique().tolist()),
        "failed_rows": int(metrics.status.eq("failed").sum()),
    }
    integrity["pass"] = all(
        [
            integrity["actual_metric_rows"] == expected_rows,
            integrity["duplicate_metric_keys"] == 0,
            integrity["models_complete"],
            integrity["targets_complete"],
            integrity["seeds_complete"],
            integrity["failed_rows"] == 0,
        ]
    )
    (RUN / "integrity_report.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    if not integrity["pass"]:
        raise RuntimeError(f"B016 is incomplete: {integrity}")

    summary = (
        ok.groupby(["evaluation", "family", "model"], as_index=False)
        .agg(
            device_seed_n=("nmae", "size"),
            nmae_mean=("nmae", "mean"),
            nmae_sd=("nmae", "std"),
            nmae_worst=("nmae", "max"),
            nrmse_mean=("nrmse", "mean"),
            r2_mean=("r2", "mean"),
            nbias_mean=("nbias", "mean"),
            oer_mean=("overestimation_rate", "mean"),
            parameters=("parameters", "max"),
            latency_ms=("latency_ms", "mean"),
        )
        .sort_values(["evaluation", "nmae_mean"])
    )
    summary.to_csv(RUN / "unified_summary.csv", index=False)

    device = (
        ok.groupby(["evaluation", "model", "target"], as_index=False)
        .agg(nmae=("nmae", "mean"), nrmse=("nrmse", "mean"), r2=("r2", "mean"))
    )
    device.to_csv(RUN / "device_five_seed_summary.csv", index=False)

    rows: list[dict[str, object]] = []
    for evaluation, frame in device.groupby("evaluation"):
        proposed = frame.loc[frame.model.eq("caro_rul")].set_index("target")["nmae"]
        for model in sorted(set(frame.model) - {"caro_rul"}):
            comparator = frame.loc[frame.model.eq(model)].set_index("target")["nmae"]
            common = proposed.index.intersection(comparator.index)
            gain = comparator.loc[common].to_numpy() - proposed.loc[common].to_numpy()
            try:
                p_value = float(
                    wilcoxon(gain, alternative="greater", zero_method="wilcox").pvalue
                )
            except ValueError:
                p_value = 1.0
            lower, upper = paired_bootstrap(gain)
            rows.append(
                {
                    "evaluation": evaluation,
                    "comparator": model,
                    "mean_nmae_gain": float(gain.mean()),
                    "relative_gain": float(gain.mean() / comparator.loc[common].mean()),
                    "device_wins": int(np.sum(gain > 0)),
                    "devices": int(len(gain)),
                    "wilcoxon_p_one_sided": p_value,
                    "bootstrap_ci_low": lower,
                    "bootstrap_ci_high": upper,
                }
            )
    tests = pd.DataFrame(rows)
    tests["holm_p"] = tests.groupby("evaluation", group_keys=False)[
        "wilcoxon_p_one_sided"
    ].apply(holm_adjust)
    tests.to_csv(RUN / "paired_tests.csv", index=False)

    ablation = summary.loc[
        summary.model.isin(["tcn", "cdap_point", "tcn_iorh", "caro_rul"])
    ].copy()
    ablation.to_csv(RUN / "module_ablation.csv", index=False)

    report = {
        "integrity": integrity,
        "best_by_evaluation": {
            evaluation: group.iloc[0][
                ["model", "nmae_mean", "nrmse_mean", "r2_mean", "nmae_worst"]
            ].to_dict()
            for evaluation, group in summary.groupby("evaluation", sort=False)
        },
        "caro": summary.loc[summary.model.eq("caro_rul")].to_dict("records"),
    }
    (RUN / "analysis_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
