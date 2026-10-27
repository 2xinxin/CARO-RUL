from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

all_results = pd.read_csv(
    ROOT / "results/PCSS_RICH_BENCHMARK/B016_CARO_UNIFIED_5SEED/fold_metrics.csv"
)
fractional = all_results[all_results.evaluation.eq("fractional_life")]
fixed = all_results[all_results.evaluation.eq("fixed_event_count")]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.2,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.1,
    "xtick.labelsize": 7.3,
    "ytick.labelsize": 7.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {"CARO-RUL": "#0B7A75", "BiLSTM": "#E07A1F", "RNN": "#3C78B5"}


def finish(fig, ax, stem, legend=True):
    ax.grid(axis="y", color="#D8DADD", lw=0.55, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    if legend:
        ax.legend(frameon=False, ncol=3, loc="upper left", columnspacing=0.8, handlelength=1.4)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


# Device-level error with seed variability.
devices = [f"SW{index}" for index in range(1, 8)]
frames = {
    "CARO-RUL": fractional[fractional.model.eq("caro_rul")],
    "BiLSTM": fractional[fractional.model.eq("bilstm")],
    "RNN": fractional[fractional.model.eq("rnn")],
}
fig, ax = plt.subplots(figsize=(3.45, 2.25))
x = np.arange(len(devices))
width = 0.24
for offset, (name, frame) in zip((-1, 0, 1), frames.items()):
    stats = frame.groupby("target").nmae.agg(["mean", "std"]).reindex(devices)
    ax.bar(
        x + offset * width, stats["mean"], width,
        color=COLORS[name], label=name, yerr=stats["std"], capsize=1.7,
        linewidth=0, zorder=3,
    )
ax.set_xticks(x, devices)
ax.set_ylabel(r"nMAE (mean $\pm$ SD)")
ax.set_ylim(0, 2.8)
finish(fig, ax, "caro_device_error")


# Optimization-seed stability.
fig, ax = plt.subplots(figsize=(3.45, 2.18))
for name, frame in frames.items():
    seed_mean = frame.groupby("seed").nmae.mean().reindex([17, 29, 41, 53, 67])
    ax.plot(
        seed_mean.index.astype(str), seed_mean, marker="o", ms=3.7,
        lw=1.7 if name == "CARO-RUL" else 1.25,
        color=COLORS[name], label=name,
    )
ax.set_ylabel("Device-macro nMAE")
ax.set_xlabel("Random seed")
finish(fig, ax, "caro_seed_stability")


# Four-way matched module ablation.
rename = {"tcn": "TCN", "cdap_point": "+ CDAP", "tcn_iorh": "+ IORH", "caro_rul": "CARO-RUL"}
order = list(rename)
means = [fractional[fractional.model.eq(model)].nmae.mean() for model in order]
stds = [fractional[fractional.model.eq(model)].groupby("seed").nmae.mean().std() for model in order]
fig, ax = plt.subplots(figsize=(3.45, 2.18))
ax.bar(
    np.arange(4), means, yerr=stds, capsize=2.2,
    color=["#B7BCC2", "#76B7B2", "#E8A65C", "#0B7A75"],
    edgecolor=["white", "white", "white", "#084F4C"],
    hatch=[None, None, None, "///"], linewidth=0.7, zorder=3,
)
ax.set_xticks(np.arange(4), [rename[m] for m in order], rotation=13)
ax.set_ylabel("Device-macro nMAE")
ax.set_ylim(0, 1.02)
finish(fig, ax, "caro_module_ablation", legend=False)


# Endpoint-independent fixed-event-count sensitivity.
fixed_order = ["rnn", "timesnet_faithful", "caro_rul", "bilstm", "tcn"]
fixed_label = ["RNN", "TimesNet", "CARO-RUL", "BiLSTM", "TCN"]
fixed_mean = [fixed[fixed.model.eq(model)].nmae.mean() for model in fixed_order]
fixed_sd = [fixed[fixed.model.eq(model)].groupby("seed").nmae.mean().std() for model in fixed_order]
fig, ax = plt.subplots(figsize=(3.45, 2.22))
ax.bar(
    np.arange(len(fixed_order)), fixed_mean, yerr=fixed_sd, capsize=2.2,
    color=["#3C78B5", "#8E6C9F", "#0B7A75", "#E07A1F", "#9AA0A6"],
    edgecolor=["white", "white", "#084F4C", "white", "white"],
    hatch=[None, None, "///", None, None], linewidth=0.7, zorder=3,
)
ax.set_xticks(np.arange(len(fixed_order)), fixed_label, rotation=14)
ax.set_ylabel("Device-macro nMAE")
finish(fig, ax, "caro_fixed_count_sensitivity", legend=False)
