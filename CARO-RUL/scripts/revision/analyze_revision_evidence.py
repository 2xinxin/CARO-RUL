from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "results" / "revision"
CONTROL = HERE / "A2-002"
OUT = HERE / "A2-005"
FIG = ROOT / "outputs" / "figures"
TABLE = ROOT / "outputs" / "tables"
B016 = ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B016_CARO_UNIFIED_5SEED"
B020 = ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B020_CARO_EVIDENCE_EXTENSION"


LABEL = {
    "tcn_point": "TCN point",
    "tcn_lower_q375": r"TCN $q_{0.375}$",
    "tcn_asym_smoothl1_x2": "TCN asymmetric",
    "tcn_source_constant": "TCN + constant",
    "iorh_median_q50": r"IORH $q_{0.50}$",
    "iorh_q375": r"IORH $q_{0.375}$",
    "iorh_fixed_width_010": "IORH width margin",
    "iorh_source_selected_width": "IORH selected margin",
    "tcn": "TCN",
    "cdap_point": "TCN + CDAP",
    "tcn_iorh": "TCN + IORH",
    "caro_rul": "CARO-RUL",
}


def bootstrap_devices(frame: pd.DataFrame, value: str, *, draws=20000, seed=20261026):
    device = frame.groupby("target", as_index=False)[value].mean()
    values = device[value].to_numpy(float)
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    sampled = values[index]
    means = sampled.mean(axis=1)
    worst = sampled.max(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "worst_estimate": float(values.max()),
        "worst_ci_low": float(np.quantile(worst, 0.025)),
        "worst_ci_high": float(np.quantile(worst, 0.975)),
        "independent_devices": int(len(values)),
    }


def quantile_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pd.read_csv(B020 / "predictions.csv")
    p = p[(p.model.eq("caro_rul")) & (p.condition.eq("clean"))].copy()
    rows = []
    risk_rows = []
    quantiles = [(0.25, "q25"), (0.375, "q375"), (0.50, "q50"), (0.75, "q75")]
    for (evaluation, target), g in p.groupby(["evaluation", "target"]):
        lifetime = g.lifetime.to_numpy(float)
        truth = g.true_rul.to_numpy(float)
        width = (g.q75 - g.q25).to_numpy(float) / lifetime
        central_coverage = float(np.mean((truth >= g.q25.to_numpy(float)) & (truth <= g.q75.to_numpy(float))))
        point_error = (g.predicted_rul - g.true_rul).to_numpy(float)
        abs_error = np.abs(point_error) / lifetime
        over = (point_error > 0).astype(int)
        rho, rho_p = spearmanr(width, abs_error)
        auc = roc_auc_score(over, width) if len(np.unique(over)) == 2 else np.nan
        ap = average_precision_score(over, width) if over.sum() else np.nan
        risk_rows.append({
            "evaluation": evaluation, "target": target,
            "normalized_width_mean": float(width.mean()),
            "width_error_spearman": float(rho), "width_error_p": float(rho_p),
            "width_overestimate_auroc": float(auc), "width_overestimate_auprc": float(ap),
            "overestimation_prevalence": float(over.mean()),
        })
        for tau, column in quantiles:
            q = g[column].to_numpy(float)
            residual = truth - q
            pinball = np.maximum(tau * residual, (tau - 1.0) * residual) / lifetime
            rows.append({
                "evaluation": evaluation, "target": target, "quantile": tau,
                "empirical_coverage": float(np.mean(truth <= q)),
                "calibration_error": float(np.mean(truth <= q) - tau),
                "normalized_pinball": float(pinball.mean()),
                "normalized_width_q25_q75": float(width.mean()),
                "central_q25_q75_coverage": central_coverage,
                "n_predictions_including_seeds": int(len(g)),
            })
    return pd.DataFrame(rows), pd.DataFrame(risk_rows)


def decision_analysis(predictions: pd.DataFrame, reserve=512) -> pd.DataFrame:
    p = predictions[predictions.evaluation.eq("online_256")].copy()
    rows = []
    for (target, seed, method), g in p.groupby(["target", "seed", "method"]):
        g = g.sort_values("index")
        hit = g[g.predicted_rul <= reserve]
        selected = hit.iloc[0] if len(hit) else g.iloc[-1]
        actual = float(selected.true_rul)
        rows.append({
            "target": target, "seed": seed, "method": method,
            "reserve_events": reserve, "intervention_index": int(selected["index"]),
            "actual_rul_at_intervention": actual,
            "premature_replacement_events": max(actual - reserve, 0.0),
            "late_intervention_events": max(reserve - actual, 0.0),
            "triggered_before_terminal_checkpoint": bool(len(hit)),
        })
    return pd.DataFrame(rows)


def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.2, "axes.labelsize": 8.5, "axes.titlesize": 9,
        "legend.fontsize": 7.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def plot_calibration(qdiag: pd.DataFrame):
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65), sharex=True, sharey=True, layout="constrained")
    for ax, evaluation, title in zip(
        axes, ["fractional_life", "fixed_event_count"],
        ["Retrospective checkpoints", "Fixed event-count checkpoints"],
    ):
        s = qdiag[qdiag.evaluation.eq(evaluation)]
        for _, d in s.groupby("target"):
            ax.plot(d["quantile"], d.empirical_coverage, color="#AEB6BF", alpha=0.55, lw=0.8)
        macro = s.groupby("quantile", as_index=False).empirical_coverage.mean()
        ax.plot(macro["quantile"], macro.empirical_coverage, "o-", color="#0B7A75", lw=2.0, ms=4, label="Device-macro")
        ax.plot([0.2, 0.8], [0.2, 0.8], "--", color="#4D4D4D", lw=1.0, label="Ideal")
        ax.set_title(title)
        ax.set_xlabel("Nominal quantile level")
        ax.grid(color="#E4E7EA", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Empirical coverage")
    axes[1].legend(frameon=False, loc="upper left")
    fig.savefig(FIG / "revision_quantile_calibration.pdf")
    fig.savefig(FIG / "revision_quantile_calibration.png", dpi=400)
    plt.close(fig)


def plot_control_comparison(metrics: pd.DataFrame):
    setup_style()
    order = [
        "tcn_point", "tcn_lower_q375", "tcn_asym_smoothl1_x2", "tcn_source_constant",
        "iorh_median_q50", "iorh_q375", "iorh_fixed_width_010",
    ]
    colors = ["#8C8C8C", "#4C78A8", "#F58518", "#B279A2", "#72B7B2", "#54A24B", "#0B7A75"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.15), sharey=True, layout="constrained")
    for ax, evaluation, title in zip(
        axes, ["fractional_life", "fixed_event_count"],
        ["Retrospective", "Fixed event count"],
    ):
        s = metrics[(metrics.evaluation.eq(evaluation)) & (metrics.method.isin(order))]
        device = s.groupby(["method", "target"], as_index=False).nmae.mean()
        means = device.groupby("method").nmae.mean().reindex(order)
        # Device bootstrap intervals for the device-macro mean.
        low, high = [], []
        for method in order:
            ci = bootstrap_devices(s[s.method.eq(method)], "nmae")
            low.append(ci["estimate"] - ci["ci_low"])
            high.append(ci["ci_high"] - ci["estimate"])
        y = np.arange(len(order))[::-1]
        ax.errorbar(means.values, y, xerr=[low, high], fmt="none", ecolor="#4A4A4A", elinewidth=0.9, capsize=2, zorder=2)
        ax.scatter(means.values, y, c=colors, s=36, edgecolors="white", linewidths=0.5, zorder=3)
        ax.set_yticks(y, [LABEL[m] for m in order])
        ax.set_title(title)
        ax.set_xlabel("Device-macro nMAE")
        ax.grid(axis="x", color="#E4E7EA", lw=0.5, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG / "revision_conservative_controls.pdf")
    fig.savefig(FIG / "revision_conservative_controls.png", dpi=400)
    plt.close(fig)


def plot_decisions(decision: pd.DataFrame):
    setup_style()
    methods = ["tcn_point", "tcn_lower_q375", "tcn_asym_smoothl1_x2", "tcn_source_constant", "iorh_q375", "iorh_fixed_width_010"]
    s = decision[decision.method.isin(methods)]
    d = s.groupby(["method", "target"], as_index=False)[["premature_replacement_events", "late_intervention_events"]].mean()
    agg = d.groupby("method")[["premature_replacement_events", "late_intervention_events"]].mean().reindex(methods)
    fig, ax = plt.subplots(figsize=(3.50, 3.05), layout="constrained")
    y = np.arange(len(methods))[::-1]
    ax.barh(y, agg.premature_replacement_events, color="#4C78A8", label="Premature", zorder=2)
    ax.barh(y, agg.late_intervention_events, left=agg.premature_replacement_events, color="#E45756", label="Late", zorder=2)
    ax.set_yticks(y, [LABEL[m] for m in methods])
    ax.set_xlabel("Mean deviation from 512-event reserve")
    ax.grid(axis="x", color="#E4E7EA", lw=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.savefig(FIG / "revision_intervention_outcomes.pdf")
    fig.savefig(FIG / "revision_intervention_outcomes.png", dpi=400)
    plt.close(fig)


def plot_trajectories(predictions: pd.DataFrame):
    setup_style()
    chosen = ["SW1", "SW3", "SW5", "SW7"]
    p = predictions[(predictions.evaluation.eq("online_256")) & predictions.target.isin(chosen)]
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))
    for ax, target in zip(axes.ravel(), chosen):
        s = p[p.target.eq(target)]
        truth = s.groupby("index", as_index=False).true_rul.first()
        ax.plot(truth["index"], truth.true_rul, color="#202124", lw=1.8, label="True RUL")
        for method, color, ls in [
            ("tcn_point", "#8C8C8C", "--"),
            ("iorh_q375", "#54A24B", "-."),
            ("iorh_fixed_width_010", "#0B7A75", "-"),
        ]:
            m = s[s.method.eq(method)].groupby("index").predicted_rul.agg(["mean", "std"]).reset_index()
            ax.plot(m["index"], m["mean"], color=color, ls=ls, lw=1.35, label=LABEL[method])
            if method == "iorh_fixed_width_010":
                ax.fill_between(m["index"], (m["mean"] - m["std"]).clip(lower=0), m["mean"] + m["std"], color=color, alpha=0.14, linewidth=0)
        ax.set_title(target)
        ax.set_xlabel("Recorded acquisition event")
        ax.set_ylabel("Remaining recorded events")
        ax.grid(color="#E7E9EC", lw=0.45)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Reserve a dedicated header band so the shared legend never collides with
    # the SW1/SW3 panel titles after the figure is scaled in the manuscript.
    fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.11, top=0.86, wspace=0.28, hspace=0.42)
    fig.savefig(FIG / "revision_rul_trajectories.pdf")
    fig.savefig(FIG / "revision_rul_trajectories.png", dpi=400)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLE.mkdir(parents=True, exist_ok=True)
    controls = pd.read_csv(CONTROL / "fold_metrics.csv")
    predictions = pd.read_csv(CONTROL / "predictions.csv")
    b016 = pd.read_csv(B016 / "fold_metrics.csv")

    qdiag, risk = quantile_diagnostics()
    qdiag.to_csv(OUT / "quantile_by_device.csv", index=False)
    risk.to_csv(OUT / "width_risk_by_device.csv", index=False)
    decision = decision_analysis(predictions)
    decision.to_csv(OUT / "intervention_outcomes.csv", index=False)

    ci_rows = []
    for evaluation in ["fractional_life", "fixed_event_count"]:
        for method in sorted(controls.method.unique()):
            frame = controls[(controls.evaluation.eq(evaluation)) & (controls.method.eq(method))]
            for metric in ["nmae", "nrmse", "mae_events", "signed_error_events", "r2"]:
                ci_rows.append({"source": "controls", "evaluation": evaluation, "method": method, "metric": metric, **bootstrap_devices(frame, metric)})
        for model in ["tcn", "cdap_point", "tcn_iorh", "caro_rul", "bilstm"]:
            frame = b016[(b016.evaluation.eq(evaluation)) & (b016.model.eq(model))]
            for metric in ["nmae", "nrmse", "mae", "nbias", "r2"]:
                ci_rows.append({"source": "B016", "evaluation": evaluation, "method": model, "metric": metric, **bootstrap_devices(frame, metric)})
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "device_bootstrap_confidence_intervals.csv", index=False)

    qmacro = qdiag.groupby(["evaluation", "quantile"], as_index=False).agg(
        empirical_coverage=("empirical_coverage", "mean"),
        calibration_error=("calibration_error", "mean"),
        normalized_pinball=("normalized_pinball", "mean"),
        normalized_width_q25_q75=("normalized_width_q25_q75", "mean"),
        central_q25_q75_coverage=("central_q25_q75_coverage", "mean"),
    )
    qmacro.to_csv(TABLE / "revision_quantile_diagnostics.csv", index=False)
    control_summary = controls[controls.evaluation.isin(["fractional_life", "fixed_event_count"])].groupby(
        ["evaluation", "method"], as_index=False
    ).agg(nmae=("nmae", "mean"), nrmse=("nrmse", "mean"), mae_events=("mae_events", "mean"), signed_error_events=("signed_error_events", "mean"), oer=("overestimation_rate", "mean"), r2=("r2", "mean"))
    control_summary.to_csv(TABLE / "revision_conservative_controls.csv", index=False)
    decision_summary = decision.groupby("method", as_index=False).agg(
        premature_events=("premature_replacement_events", "mean"),
        late_events=("late_intervention_events", "mean"),
        actual_rul_at_intervention=("actual_rul_at_intervention", "mean"),
    )
    decision_summary.to_csv(TABLE / "revision_intervention_summary.csv", index=False)

    plot_calibration(qdiag)
    plot_control_comparison(controls)
    plot_decisions(decision)
    plot_trajectories(predictions)

    summary = {
        "integrity": {
            "control_metric_rows": int(len(controls)),
            "control_prediction_rows": int(len(predictions)),
            "control_duplicate_keys": int(controls.duplicated(["evaluation", "target", "seed", "method"]).sum()),
            "devices": sorted(controls.target.unique().tolist()),
            "seeds": sorted(int(x) for x in controls.seed.unique()),
        },
        "quantile_macro": qmacro.to_dict("records"),
        "width_risk_macro": risk.groupby("evaluation").mean(numeric_only=True).reset_index().to_dict("records"),
        "control_summary": control_summary.to_dict("records"),
        "decision_summary": decision_summary.to_dict("records"),
        "ci_method": "20,000 nonparametric resamples of seven device-level seed means; seeds are not independent units",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["integrity"], indent=2))


if __name__ == "__main__":
    main()
