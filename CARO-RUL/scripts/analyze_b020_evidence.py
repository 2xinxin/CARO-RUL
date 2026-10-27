from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
B016 = ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B016_CARO_UNIFIED_5SEED"
B018 = ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B018_CLASSICAL_UNIFIED_5SEED"
B020 = ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B020_CARO_EVIDENCE_EXTENSION"
OUT = B020 / "analysis"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def label(name: str) -> str:
    return {
        "caro_rul": "CARO-RUL",
        "tcn_iorh": "IORH-only",
        "cdap_point": "CDAP-only",
        "bilstm": "BiLSTM",
        "rnn": "RNN",
        "sa_mcd": "SA-MCD",
        "timesnet_faithful": "TimesNet",
        "tcn": "TCN",
    }.get(name, name)


def normalized_error(frame: pd.DataFrame) -> pd.Series:
    lifetime = frame["index"].astype(float) + frame["true_rul"].astype(float)
    return (frame["predicted_rul"] - frame["true_rul"]).abs() / lifetime.clip(lower=1.0)


def main() -> None:
    p16 = pd.read_csv(B016 / "predictions.csv")
    p18 = pd.read_csv(B018 / "predictions.csv")
    m20 = pd.read_csv(B020 / "fold_metrics.csv")
    p20 = pd.read_csv(B020 / "predictions.csv")

    assert len(m20) == 2100, f"unexpected B020 metric rows: {len(m20)}"
    assert len(p20) == 13650, f"unexpected B020 prediction rows: {len(p20)}"
    assert not m20.duplicated(["evaluation", "condition", "target", "seed", "model"]).any()
    assert not p20.duplicated(
        ["evaluation", "condition", "target", "seed", "model", "position"]
    ).any()

    # Life-stage comparison uses the frozen clean B016/B018 predictions.
    pall = pd.concat([p16, p18], ignore_index=True)
    pall = pall[pall["evaluation"].eq("fractional_life")].copy()
    pall["normalized_abs_error"] = normalized_error(pall)
    life = (
        pall.groupby(["model", "position", "seed"], as_index=False)["normalized_abs_error"]
        .mean()
        .rename(columns={"normalized_abs_error": "nmae"})
    )
    life_summary = (
        life.groupby(["model", "position"])["nmae"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    rank_base = life.groupby(["model", "position"], as_index=False)["nmae"].mean()
    rank_base["rank"] = rank_base.groupby("position")["nmae"].rank(method="min")
    life_summary = life_summary.merge(
        rank_base[["model", "position", "rank"]], on=["model", "position"]
    )
    life_summary.to_csv(OUT / "life_stage_summary.csv", index=False)

    # Asymmetric operational loss: overestimation is lambda times as costly.
    lambdas = [1, 2, 3, 5, 10]
    asym_rows = []
    for evaluation in ["fractional_life", "fixed_event_count"]:
        temp = pd.concat([p16, p18], ignore_index=True)
        temp = temp[temp["evaluation"].eq(evaluation)].copy()
        lifetime = (temp["index"] + temp["true_rul"]).clip(lower=1.0)
        raw = (temp["predicted_rul"] - temp["true_rul"]).abs() / lifetime
        over = temp["predicted_rul"] > temp["true_rul"]
        for lam in lambdas:
            temp["asymmetric_cost"] = raw * np.where(over, lam, 1.0)
            g = (
                temp.groupby(["model", "seed"], as_index=False)["asymmetric_cost"]
                .mean()
                .groupby("model")["asymmetric_cost"]
                .agg(["mean", "std"])
                .reset_index()
            )
            g["evaluation"] = evaluation
            g["overestimate_weight"] = lam
            asym_rows.append(g)
    asym = pd.concat(asym_rows, ignore_index=True)
    asym.to_csv(OUT / "asymmetric_cost_summary.csv", index=False)

    corruption = (
        m20.groupby(["evaluation", "condition", "model"])["nmae"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )
    corruption.to_csv(OUT / "corruption_summary.csv", index=False)

    # Quantile diagnostics are recorded even though they are not promoted as calibrated uncertainty.
    quant = p20[(p20["model"].eq("caro_rul")) & (p20["condition"].eq("clean"))].copy()
    qdiag = {}
    for evaluation, g in quant.groupby("evaluation"):
        lifetime = g["lifetime"].clip(lower=1.0)
        err = (g["predicted_rul"] - g["true_rul"]).abs() / lifetime
        width = (g["q75"] - g["q25"]) / lifetime
        rho, rho_p = spearmanr(width, err)
        coverage = ((g["true_rul"] >= g["q25"]) & (g["true_rul"] <= g["q75"])).mean()
        selective = {}
        for retained in [1.0, 0.8, 0.6, 0.4, 0.2]:
            n = max(1, int(np.ceil(len(g) * retained)))
            idx = np.argsort(width.to_numpy())[:n]
            selective[str(retained)] = float(err.iloc[idx].mean())
        qdiag[evaluation] = {
            "q25_q75_coverage": float(coverage),
            "normalized_mean_width": float(width.mean()),
            "width_error_spearman_rho": float(rho),
            "width_error_spearman_p": float(rho_p),
            "selective_nmae_by_retained_fraction": selective,
            "interpretation": "Ordered intervals are structurally non-crossing but are not a calibrated error-ranking uncertainty score.",
        }
    (OUT / "quantile_diagnostics.json").write_text(
        json.dumps(qdiag, indent=2), encoding="utf-8"
    )

    # Figure style: compact two-column evidence panel, with CARO-RUL visually dominant.
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "CARO-RUL": "#0B7A75",
        "IORH-only": "#E07A1F",
        "BiLSTM": "#6B6F76",
        "RNN": "#3C78B5",
        "SA-MCD": "#8E6C9F",
        "TimesNet": "#B59A35",
        "TCN": "#9AA0A6",
    }
    def save_single(fig, ax, stem, grid=False):
        if grid:
            ax.grid(axis="y", color="#D8DADD", linewidth=0.55, alpha=0.75, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(pad=0.35)
        fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(FIG / f"{stem}.png", dpi=400, bbox_inches="tight")
        plt.close(fig)

    # Life-stage profile.
    fig, ax = plt.subplots(figsize=(3.45, 2.24))
    stage_models = ["caro_rul", "timesnet_faithful", "rnn", "bilstm", "sa_mcd"]
    markers = ["o", "s", "^", "D", "v"]
    for model, marker in zip(stage_models, markers):
        s = life_summary[life_summary["model"].eq(model)].sort_values("position")
        name = label(model)
        lw = 2.2 if model == "caro_rul" else 1.15
        z = 5 if model == "caro_rul" else 2
        ax.plot(
            100 * s["position"], s["mean"], marker=marker, markersize=3.5,
            linewidth=lw, color=colors[name], label=name, zorder=z,
        )
        if model == "caro_rul":
            ax.fill_between(
                100 * s["position"],
                (s["mean"] - s["sd"]).clip(lower=0),
                s["mean"] + s["sd"],
                color=colors[name], alpha=0.13, linewidth=0,
            )
    ax.set_xlabel("Observed life (%)")
    ax.set_ylabel("nMAE")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    save_single(fig, ax, "caro_lifecycle_error")

    # Retrospective asymmetric loss.
    fig, ax = plt.subplots(figsize=(3.45, 2.24))
    cost_models = ["caro_rul", "tcn_iorh", "sa_mcd", "bilstm", "rnn"]
    for model, marker in zip(cost_models, markers):
        s = asym[(asym["evaluation"].eq("fractional_life")) & (asym["model"].eq(model))]
        name = label(model)
        ax.plot(
            s["overestimate_weight"], s["mean"], marker=marker, markersize=3.5,
            linewidth=2.2 if model == "caro_rul" else 1.15,
            color=colors[name], label=name, zorder=5 if model == "caro_rul" else 2,
        )
    ax.set_xlabel(r"Overestimation penalty $\lambda$")
    ax.set_ylabel("Normalized asymmetric loss")
    ax.set_xticks(lambdas)
    ax.legend(ncol=2, frameon=False, loc="upper left")
    save_single(fig, ax, "caro_asymmetric_cost")

    # Acquisition stress tests; extreme failures remain in the CSV and manuscript discussion.
    conditions = ["clean", "noise_010", "plateau_16", "waveform_dropout_25"]
    cond_labels = ["Clean", "Noise", "Plateau", "25% loss"]
    bar_models = ["caro_rul", "tcn_iorh", "tcn", "bilstm"]
    for evaluation, stem in [
        ("fractional_life", "caro_stress_retrospective"),
        ("fixed_event_count", "caro_stress_fixed_count"),
    ]:
        fig, ax = plt.subplots(figsize=(3.45, 2.18))
        x = np.arange(len(conditions))
        width = 0.19
        for j, model in enumerate(bar_models):
            vals = []
            for condition in conditions:
                row = corruption[
                    corruption["evaluation"].eq(evaluation)
                    & corruption["condition"].eq(condition)
                    & corruption["model"].eq(model)
                ].iloc[0]
                vals.append(row["mean"])
            name = label(model)
            ax.bar(
                x + (j - 1.5) * width, vals, width,
                linewidth=0.65,
                color=colors[name], edgecolor="white" if model != "caro_rul" else "#084F4C",
                hatch="///" if model == "caro_rul" else None,
                label=name, zorder=3,
            )
        ax.set_xticks(x, cond_labels)
        ax.set_ylabel("nMAE")
        ax.legend(ncol=2, frameon=False, loc="upper left")
        save_single(fig, ax, stem, grid=True)

    caro_life = life_summary[life_summary["model"].eq("caro_rul")].sort_values("position")
    moderate = corruption[
        corruption["condition"].isin(conditions)
        & corruption["model"].isin(bar_models)
    ].copy()
    winners = (
        moderate.loc[moderate.groupby(["evaluation", "condition"])["mean"].idxmin(),
                     ["evaluation", "condition", "model", "mean"]]
        .to_dict("records")
    )
    evidence = {
        "integrity": {
            "metric_rows": int(len(m20)),
            "prediction_rows": int(len(p20)),
            "duplicate_metric_keys": 0,
            "duplicate_prediction_keys": 0,
        },
        "life_stage": {
            "caro_top3_stages": int((caro_life["rank"] <= 3).sum()),
            "caro_rank1_stages": int((caro_life["rank"] == 1).sum()),
            "total_stages": int(len(caro_life)),
        },
        "moderate_corruption_winners": winners,
        "quantile_diagnostics": qdiag,
    }
    (OUT / "evidence_summary.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
