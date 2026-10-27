from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
OUT_ROOT = PROJECT / "results" / "revision" / "A2-002"
sys.path.insert(0, str(PROJECT / "src"))

from pcss_rich_benchmark.data import (  # noqa: E402
    FoldTransform,
    build_samples,
    fit_transform,
    load_records,
    make_sample,
    transformed_record,
)
from pcss_single_model.model_v9 import build_model as build_caro  # noqa: E402
from tor_rul.neural_models import TCNRegressor  # noqa: E402
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
MAX_RUL = 1_000_000.0
WINDOW = 128
HIDDEN = 24


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalized_log_to_events(values: np.ndarray) -> np.ndarray:
    return np.expm1(np.clip(values, 0.0, 1.0) * np.log1p(MAX_RUL))


def metric_row(samples, predictions: np.ndarray) -> dict[str, float]:
    truth = np.asarray([s.target_pulses for s in samples], dtype=float)
    lifetime = np.asarray([s.lifetime for s in samples], dtype=float)
    prediction = np.maximum(np.asarray(predictions, dtype=float), 0.0)
    error = prediction - truth
    denom = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "nmae": float(np.mean(np.abs(error) / lifetime)),
        "nrmse": float(np.sqrt(np.mean((error / lifetime) ** 2))),
        "mae_events": float(np.mean(np.abs(error))),
        "rmse_events": float(np.sqrt(np.mean(error**2))),
        "signed_error_events": float(np.mean(error)),
        "nbias": float(np.mean(error / lifetime)),
        "overestimation_rate": float(np.mean(error > 0)),
        "r2": float(1.0 - np.sum(error**2) / denom) if denom > 0 else math.nan,
    }


class ScalarFit:
    def __init__(self, model: nn.Module, best_validation: float, epochs: int, trace):
        self.model = model
        self.best_validation = best_validation
        self.epochs = epochs
        self.trace = trace


def scalar_loss(prediction: torch.Tensor, target: torch.Tensor, kind: str) -> torch.Tensor:
    error = target - prediction
    if kind == "lower_quantile":
        tau = 0.375
        return torch.maximum(tau * error, (tau - 1.0) * error).mean()
    if kind == "asymmetric_scalar":
        base = F.smooth_l1_loss(prediction, target, reduction="none")
        # Overestimation is given twice the cost of underestimation.
        return (base * torch.where(prediction > target, 2.0, 1.0)).mean()
    if kind == "point":
        return F.smooth_l1_loss(prediction, target)
    raise KeyError(kind)


def fit_scalar(kind, train, validation, seed, device, *, epochs=20, patience=5) -> ScalarFit:
    seed_all(seed)
    model = TCNRegressor(train[0].sequence.shape[0], HIDDEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 100003)
    train_loader = DataLoader(
        SequenceRULDataset(train, "log_pulses"), batch_size=128, shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        SequenceRULDataset(validation, "log_pulses"), batch_size=128, shuffle=False
    )
    best = float("inf")
    best_state = None
    stale = 0
    trace = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_sum = 0.0
        train_n = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            prediction = model(x)["prediction"]
            loss = scalar_loss(prediction, y, kind)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_sum += float(loss) * len(y)
            train_n += len(y)
        model.eval()
        val_abs = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)["prediction"]
                val_abs += float(F.l1_loss(pred, y, reduction="sum"))
                val_n += len(y)
        score = val_abs / max(val_n, 1)
        trace.append({"epoch": epoch, "train_objective": train_sum / train_n, "validation_mae_log": score})
        if score < best - 1e-5:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {kind}")
    model.load_state_dict(best_state)
    return ScalarFit(model, best, len(trace), trace)


def predict_scalar(model, samples, device) -> np.ndarray:
    loader = DataLoader(SequenceRULDataset(samples, "log_pulses"), batch_size=256, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device))["prediction"].cpu().numpy())
    return np.concatenate(out)


def predict_caro(model, samples, device) -> dict[str, np.ndarray]:
    loader = DataLoader(SequenceRULDataset(samples, "log_pulses"), batch_size=256, shuffle=False)
    values = {"q25": [], "q375": [], "q50": [], "q75": []}
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            output = model(x.to(device))
            q = output["quantiles"].cpu().numpy()
            for j, key in enumerate(values):
                values[key].append(q[:, j])
    return {key: np.concatenate(parts) for key, parts in values.items()}


def checkpoint_transform(payload) -> FoldTransform:
    item = payload["transform"]
    return FoldTransform(
        feature_columns=list(payload["feature_columns"]),
        median=np.asarray(item["median"], dtype=float),
        mean=np.asarray(item["mean"], dtype=float),
        std=np.asarray(item["std"], dtype=float),
    )


def choose_constant_delta(point_z: np.ndarray, samples) -> tuple[float, pd.DataFrame]:
    truth = np.asarray([s.target_pulses for s in samples], dtype=float)
    lifetime = np.asarray([s.lifetime for s in samples], dtype=float)
    rows = []
    for delta in np.linspace(0.0, 0.15, 31):
        pred = normalized_log_to_events(point_z - delta)
        rows.append({
            "delta_log": float(delta),
            "source_validation_nmae": float(np.mean(np.abs(pred - truth) / lifetime)),
            "source_validation_oer": float(np.mean(pred > truth)),
        })
    frame = pd.DataFrame(rows)
    chosen = frame.sort_values(["source_validation_nmae", "source_validation_oer", "delta_log"]).iloc[0]
    return float(chosen.delta_log), frame


def choose_width_margin(q: dict[str, np.ndarray], samples) -> tuple[float, pd.DataFrame]:
    truth = np.asarray([s.target_pulses for s in samples], dtype=float)
    lifetime = np.asarray([s.lifetime for s in samples], dtype=float)
    rows = []
    for alpha in np.linspace(0.0, 0.30, 13):
        z = q["q375"] - alpha * (q["q75"] - q["q25"])
        pred = normalized_log_to_events(z)
        rows.append({
            "width_alpha": float(alpha),
            "source_validation_nmae": float(np.mean(np.abs(pred - truth) / lifetime)),
            "source_validation_oer": float(np.mean(pred > truth)),
        })
    frame = pd.DataFrame(rows)
    chosen = frame.sort_values(["source_validation_nmae", "source_validation_oer", "width_alpha"]).iloc[0]
    return float(chosen.width_alpha), frame


def build_evaluations(target_record, transformed):
    fractional = [
        make_sample(target_record, transformed, int(np.floor(f * (target_record.length - 1))), WINDOW)
        for f in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    ]
    fixed_idx = [i for i in [256, 512, 768, 1024, 1280, 1536] if WINDOW <= i < target_record.length - 1]
    fixed = [make_sample(target_record, transformed, i, WINDOW) for i in fixed_idx]
    online_idx = list(range(256, target_record.length - 1, 256))
    if online_idx[-1] != target_record.length - 2:
        online_idx.append(target_record.length - 2)
    online = [make_sample(target_record, transformed, i, WINDOW) for i in online_idx]
    return {"fractional_life": fractional, "fixed_event_count": fixed, "online_256": online}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    targets = args.target or DEVICES
    seeds = args.seed or SEEDS
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    trace_dir = OUT_ROOT / "training_traces"
    trace_dir.mkdir(exist_ok=True)
    models_dir = OUT_ROOT / "checkpoints"
    models_dir.mkdir(exist_ok=True)

    records = load_records(CODE / "dataset", DEVICES)
    prediction_path = OUT_ROOT / "predictions.csv"
    metric_path = OUT_ROOT / "fold_metrics.csv"
    pred_rows = pd.read_csv(prediction_path).to_dict("records") if prediction_path.exists() else []
    metric_rows = pd.read_csv(metric_path).to_dict("records") if metric_path.exists() else []
    completed = {
        (str(r["evaluation"]), str(r["target"]), int(r["seed"]), str(r["method"]))
        for r in metric_rows
    }
    selection_rows = []

    for target in targets:
        source = [records[d] for d in DEVICES if d != target]
        transform = fit_transform(source, FEATURES, 0.75)
        transformed = {d: transformed_record(r, transform) for d, r in records.items()}
        train_domains = [build_samples(r, transformed[r.record_id], WINDOW, 0.05, 0.75, 128) for r in source]
        val_domains = [build_samples(r, transformed[r.record_id], WINDOW, 0.76, 0.88, 32) for r in source]
        train = [s for group in train_domains for s in group]
        validation = [s for group in val_domains for s in group]
        evaluations = build_evaluations(records[target], transformed[target])

        for seed in seeds:
            print(json.dumps({"target": target, "seed": seed, "status": "start"}), flush=True)
            fits = {}
            for kind in ["point", "lower_quantile", "asymmetric_scalar"]:
                checkpoint = models_dir / f"{kind}_target-{target}_seed-{seed}.pt"
                trace_path = trace_dir / f"{kind}_target-{target}_seed-{seed}.csv"
                if checkpoint.exists() and trace_path.exists():
                    fit = fit_scalar(kind, train, validation, seed, device, epochs=1, patience=1)
                    payload = torch.load(checkpoint, map_location=device, weights_only=False)
                    fit.model.load_state_dict(payload["state_dict"])
                    fit.best_validation = float(payload["best_validation"])
                    fit.epochs = int(payload["epochs"])
                    fit.trace = pd.read_csv(trace_path).to_dict("records")
                else:
                    fit = fit_scalar(kind, train, validation, seed, device)
                    torch.save({
                        "method": kind, "target": target, "seed": seed,
                        "best_validation": fit.best_validation, "epochs": fit.epochs,
                        "state_dict": fit.model.state_dict(),
                    }, checkpoint)
                    pd.DataFrame(fit.trace).to_csv(trace_path, index=False)
                fits[kind] = fit

            point_val_z = predict_scalar(fits["point"].model, validation, device)
            delta, delta_grid = choose_constant_delta(point_val_z, validation)
            delta_grid.assign(target=target, seed=seed).to_csv(
                OUT_ROOT / f"constant_selection_target-{target}_seed-{seed}.csv", index=False
            )

            caro_path = CODE / "models" / "caro_rul_lodo" / f"caro_rul_target-{target}_seed-{seed}.pt"
            payload = torch.load(caro_path, map_location="cpu", weights_only=False)
            # The saved transform must match the independently rebuilt outer-fold transform.
            saved_transform = checkpoint_transform(payload)
            if not (np.allclose(saved_transform.mean, transform.mean) and np.allclose(saved_transform.std, transform.std)):
                raise RuntimeError(f"Transform mismatch for {target}, seed {seed}")
            caro = build_caro("caro_rul", payload["input_channels"], payload["hidden_size"], payload["hazard_bins"]).to(device)
            caro.load_state_dict(payload["state_dict"])
            q_val = predict_caro(caro, validation, device)
            alpha, alpha_grid = choose_width_margin(q_val, validation)
            alpha_grid.assign(target=target, seed=seed).to_csv(
                OUT_ROOT / f"width_selection_target-{target}_seed-{seed}.csv", index=False
            )
            selection_rows.append({"target": target, "seed": seed, "constant_delta_log": delta, "selected_width_alpha": alpha})

            for evaluation, samples in evaluations.items():
                positions = [s.index for s in samples]
                q = predict_caro(caro, samples, device)
                method_z = {
                    "tcn_point": predict_scalar(fits["point"].model, samples, device),
                    "tcn_lower_q375": predict_scalar(fits["lower_quantile"].model, samples, device),
                    "tcn_asym_smoothl1_x2": predict_scalar(fits["asymmetric_scalar"].model, samples, device),
                    "tcn_source_constant": predict_scalar(fits["point"].model, samples, device) - delta,
                    "iorh_median_q50": q["q50"],
                    "iorh_q375": q["q375"],
                    "iorh_fixed_width_010": q["q375"] - 0.10 * (q["q75"] - q["q25"]),
                    "iorh_source_selected_width": q["q375"] - alpha * (q["q75"] - q["q25"]),
                }
                for method, z in method_z.items():
                    key = (evaluation, target, seed, method)
                    if key in completed:
                        continue
                    pred = normalized_log_to_events(z)
                    score = metric_row(samples, pred)
                    metric_rows.append({
                        "evaluation": evaluation, "target": target, "seed": seed,
                        "method": method, **score, "selected_constant_delta_log": delta,
                        "selected_width_alpha": alpha,
                    })
                    for sample, position, prediction in zip(samples, positions, pred):
                        row = {
                            "evaluation": evaluation, "target": target, "seed": seed,
                            "method": method, "index": position,
                            "true_rul": sample.target_pulses, "lifetime": sample.lifetime,
                            "predicted_rul": float(prediction),
                        }
                        if method.startswith("iorh_"):
                            idx = positions.index(position)
                            row.update({k: float(normalized_log_to_events(np.asarray([v[idx]]))[0]) for k, v in q.items()})
                        pred_rows.append(row)
                    completed.add(key)
            pd.DataFrame(metric_rows).to_csv(metric_path, index=False)
            pd.DataFrame(pred_rows).to_csv(prediction_path, index=False)
            pd.DataFrame(selection_rows).to_csv(OUT_ROOT / "selection_summary.csv", index=False)
            print(json.dumps({"target": target, "seed": seed, "status": "complete", "delta": delta, "alpha": alpha}), flush=True)

    manifest = {
        "run_id": "A2-002",
        "status": "complete",
        "targets": targets,
        "seeds": seeds,
        "device": str(device),
        "methods": sorted(pd.DataFrame(metric_rows).method.unique().tolist()),
        "data_contract": "outer LODO before windowing; source-only preprocessing; source temporal validation for checkpoint and correction selection",
        "training_budget": {"epochs": 20, "patience": 5, "batch_size": 128, "learning_rate": 1e-3, "weight_decay": 1e-4},
        "constant_grid_log": [float(x) for x in np.linspace(0.0, 0.15, 31)],
        "width_alpha_grid": [float(x) for x in np.linspace(0.0, 0.30, 13)],
        "asymmetric_overestimation_weight": 2.0,
    }
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
