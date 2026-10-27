from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "results" / "revision" / "A2-003"
sys.path.insert(0, str(PROJECT / "src"))

from pcss_rich_benchmark.data import (  # noqa: E402
    build_samples,
    fit_transform,
    load_records,
    make_sample,
    transformed_record,
)
from pcss_single_model.model_v7 import (  # noqa: E402
    CDAPRUL,
    CausalDegradationAttentionPooling,
    SequenceTCN,
)
from pcss_single_model.model_v9 import IntervalAwareOrderedRiskHead  # noqa: E402
from tor_rul.dash_experiment import _hazard_loss  # noqa: E402
from tor_rul.rul_dataset import SequenceRULDataset  # noqa: E402


FEATURES = [
    "peak_voltage_v", "voltage_min_v", "voltage_peak_to_peak_v",
    "voltage_mean_v", "voltage_std_v", "voltage_rms_v",
    "baseline_mean_v", "baseline_std_v", "baseline_corrected_peak_v",
    "baseline_corrected_trough_v", "peak_time_s", "rise_time_10_90_s",
    "fwhm_s", "positive_pulse_area_v_s",
]
DEVICES = [f"SW{i}" for i in range(1, 8)]
SEEDS = [17, 29, 41, 53, 67]
RESIDUALS = [0.0, 0.05, 0.10, 0.20]
ALPHAS = [0.0, 0.05, 0.10, 0.20, 0.30]
MAX_RUL = 1_000_000.0
WINDOW = 128
HIDDEN = 24


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(residual: float):
    return CDAPRUL(
        SequenceTCN(43, HIDDEN),
        CausalDegradationAttentionPooling(HIDDEN, residual_scale=residual),
        IntervalAwareOrderedRiskHead(HIDDEN, risk_margin=0.0),
    )


def fit_model(train, validation, residual, seed, device):
    seed_all(seed)
    model = build_model(residual).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 100003)
    train_loader = DataLoader(SequenceRULDataset(train, "log_pulses"), batch_size=128, shuffle=True, generator=generator)
    val_loader = DataLoader(SequenceRULDataset(validation, "log_pulses"), batch_size=128, shuffle=False)
    best, state, stale, trace = float("inf"), None, 0, []
    for epoch in range(1, 21):
        model.train()
        total, count = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = _hazard_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss) * len(y)
            count += len(y)
        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)["prediction"]
                val_total += float(nn.functional.l1_loss(pred, y, reduction="sum"))
                val_count += len(y)
        score = val_total / max(val_count, 1)
        trace.append({"epoch": epoch, "train_objective": total / count, "validation_mae_log": score})
        if score < best - 1e-5:
            best = score
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 5:
                break
    if state is None:
        raise RuntimeError("no valid checkpoint")
    model.load_state_dict(state)
    return model, best, trace


def predict_quantiles(model, samples, device):
    loader = DataLoader(SequenceRULDataset(samples, "log_pulses"), batch_size=256, shuffle=False)
    parts = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            parts.append(model(x.to(device))["quantiles"].cpu().numpy())
    return np.vstack(parts)


def to_events(z):
    return np.expm1(np.clip(z, 0, 1) * np.log1p(MAX_RUL))


def eval_samples(record, transformed):
    frac = [make_sample(record, transformed, int(np.floor(f * (record.length - 1))), WINDOW) for f in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    fixed_idx = [i for i in [256, 512, 768, 1024, 1280, 1536] if WINDOW <= i < record.length - 1]
    fixed = [make_sample(record, transformed, i, WINDOW) for i in fixed_idx]
    return {"fractional_life": frac, "fixed_event_count": fixed}


def nmae(samples, prediction):
    truth = np.asarray([s.target_pulses for s in samples], float)
    life = np.asarray([s.lifetime for s in samples], float)
    return float(np.mean(np.abs(prediction - truth) / life))


def write_frame(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    outers = args.outer or DEVICES
    seeds = args.seed or SEEDS
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    OUT.mkdir(parents=True, exist_ok=True)
    traces = OUT / "training_traces"
    traces.mkdir(exist_ok=True)
    checkpoints = OUT / "selected_checkpoints"
    checkpoints.mkdir(exist_ok=True)
    inner_path = OUT / "inner_predictions.csv"
    outer_path = OUT / "outer_predictions.csv"
    selection_path = OUT / "nested_selection.csv"
    inner_rows = pd.read_csv(inner_path).to_dict("records") if inner_path.exists() else []
    outer_rows = pd.read_csv(outer_path).to_dict("records") if outer_path.exists() else []
    selection_rows = pd.read_csv(selection_path).to_dict("records") if selection_path.exists() else []
    completed_inner = {(r["outer_target"], r["inner_target"], int(r["seed"]), float(r["residual_scale"])) for r in inner_rows}
    completed_outer = {(r["outer_target"], int(r["seed"])) for r in outer_rows}
    records = load_records(CODE / "dataset", DEVICES)

    for outer in outers:
        outer_sources = [d for d in DEVICES if d != outer]
        for seed in seeds:
            for inner in outer_sources:
                inner_train_ids = [d for d in outer_sources if d != inner]
                for residual in RESIDUALS:
                    key = (outer, inner, seed, residual)
                    if key in completed_inner:
                        continue
                    source_records = [records[d] for d in inner_train_ids]
                    transform = fit_transform(source_records, FEATURES, 0.75)
                    transformed = {d: transformed_record(records[d], transform) for d in inner_train_ids + [inner]}
                    train = [s for d in inner_train_ids for s in build_samples(records[d], transformed[d], WINDOW, 0.05, 0.75, 128)]
                    validation = [s for d in inner_train_ids for s in build_samples(records[d], transformed[d], WINDOW, 0.76, 0.88, 32)]
                    model, best, trace = fit_model(train, validation, residual, seed, device)
                    pd.DataFrame(trace).to_csv(traces / f"inner_outer-{outer}_target-{inner}_seed-{seed}_r-{residual:.2f}.csv", index=False)
                    for evaluation, samples in eval_samples(records[inner], transformed[inner]).items():
                        q = predict_quantiles(model, samples, device)
                        for pos, sample, values in zip(range(len(samples)), samples, q):
                            inner_rows.append({
                                "outer_target": outer, "inner_target": inner, "seed": seed,
                                "residual_scale": residual, "evaluation": evaluation,
                                "position": pos, "index": sample.index, "true_rul": sample.target_pulses,
                                "lifetime": sample.lifetime, "q25_z": values[0], "q375_z": values[1],
                                "q50_z": values[2], "q75_z": values[3], "validation_mae_log": best,
                            })
                    completed_inner.add(key)
                    write_frame(inner_path, inner_rows)
                    print(json.dumps({"outer": outer, "inner": inner, "seed": seed, "residual": residual}), flush=True)

            # Jointly select residual bound and width margin by mean of the two inner-device schedules.
            frame = pd.DataFrame(inner_rows)
            frame = frame[(frame.outer_target.eq(outer)) & (frame.seed.eq(seed))]
            scores = []
            for residual in RESIDUALS:
                subset = frame[frame.residual_scale.eq(residual)].copy()
                for alpha in ALPHAS:
                    z = subset.q375_z - alpha * (subset.q75_z - subset.q25_z)
                    subset["prediction"] = to_events(z.to_numpy())
                    per = subset.groupby(["inner_target", "evaluation"], as_index=False).apply(
                        lambda g: pd.Series({"nmae": float(np.mean(np.abs(g.prediction - g.true_rul) / g.lifetime))}),
                        include_groups=False,
                    )
                    scores.append({
                        "outer_target": outer, "seed": seed, "residual_scale": residual,
                        "risk_margin": alpha,
                        "nested_fractional_nmae": float(per[per.evaluation.eq("fractional_life")].nmae.mean()),
                        "nested_fixed_nmae": float(per[per.evaluation.eq("fixed_event_count")].nmae.mean()),
                        "selection_objective": float(per.nmae.mean()),
                    })
            score_frame = pd.DataFrame(scores).sort_values(["selection_objective", "nested_fixed_nmae", "risk_margin", "residual_scale"])
            chosen = score_frame.iloc[0].to_dict()
            selection_rows = [r for r in selection_rows if not (r["outer_target"] == outer and int(r["seed"]) == seed)]
            selection_rows.append(chosen)
            write_frame(selection_path, selection_rows)
            score_frame.to_csv(OUT / f"nested_grid_outer-{outer}_seed-{seed}.csv", index=False)

            if (outer, seed) in completed_outer:
                continue
            source_records = [records[d] for d in outer_sources]
            transform = fit_transform(source_records, FEATURES, 0.75)
            transformed = {d: transformed_record(records[d], transform) for d in DEVICES}
            train = [s for d in outer_sources for s in build_samples(records[d], transformed[d], WINDOW, 0.05, 0.75, 128)]
            validation = [s for d in outer_sources for s in build_samples(records[d], transformed[d], WINDOW, 0.76, 0.88, 32)]
            model, best, trace = fit_model(train, validation, float(chosen["residual_scale"]), seed, device)
            torch.save({
                "outer_target": outer, "seed": seed, "residual_scale": float(chosen["residual_scale"]),
                "risk_margin": float(chosen["risk_margin"]), "state_dict": model.state_dict(),
                "validation_mae_log": best,
            }, checkpoints / f"nested_caro_target-{outer}_seed-{seed}.pt")
            pd.DataFrame(trace).to_csv(traces / f"outer_target-{outer}_seed-{seed}.csv", index=False)
            for evaluation, samples in eval_samples(records[outer], transformed[outer]).items():
                q = predict_quantiles(model, samples, device)
                z = q[:, 1] - float(chosen["risk_margin"]) * (q[:, 3] - q[:, 0])
                prediction = to_events(z)
                for pos, sample, pred in zip(range(len(samples)), samples, prediction):
                    outer_rows.append({
                        "outer_target": outer, "seed": seed, "evaluation": evaluation,
                        "position": pos, "index": sample.index, "true_rul": sample.target_pulses,
                        "lifetime": sample.lifetime, "predicted_rul": float(pred),
                        "residual_scale": float(chosen["residual_scale"]),
                        "risk_margin": float(chosen["risk_margin"]),
                    })
            completed_outer.add((outer, seed))
            write_frame(outer_path, outer_rows)
            print(json.dumps({"outer": outer, "seed": seed, "selected": chosen, "status": "complete"}), flush=True)

    outer_frame = pd.DataFrame(outer_rows)
    summary = outer_frame.groupby(["evaluation"], as_index=False).apply(
        lambda g: pd.Series({
            "nmae": float(np.mean(np.abs(g.predicted_rul - g.true_rul) / g.lifetime)),
            "mae_events": float(np.mean(np.abs(g.predicted_rul - g.true_rul))),
            "oer": float(np.mean(g.predicted_rul > g.true_rul)),
        }), include_groups=False,
    )
    summary.to_csv(OUT / "outer_summary.csv", index=False)
    manifest = {
        "run_id": "A2-003", "status": "complete", "device": str(device),
        "outer_targets": outers, "seeds": seeds, "residual_grid": RESIDUALS,
        "risk_margin_grid": ALPHAS,
        "selection": "For each outer target and seed, train on five devices and validate on the sixth in turn; minimize the equal-weight mean of retrospective and fixed-count inner-device nMAE; retrain on all six outer-source devices.",
        "fixed_development_choices_not_reselected": {"window": 128, "quantile_levels": [0.25, 0.375, 0.5, 0.75], "quantile_loss_weight": 0.35},
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
