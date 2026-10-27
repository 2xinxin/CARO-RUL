from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pcss_rich_benchmark.data import (  # noqa: E402
    RichSample,
    build_samples,
    fit_transform,
    load_records,
    make_sample,
    transformed_record,
)
from tor_rul.dash_experiment import (  # noqa: E402
    fit_classical_log_rul,
    fit_neural,
    predict_classical_log_rul,
    predict_clock_prior,
    predict_direct_log_rul,
)
from tor_rul.neural_models import CausalConvBlock  # noqa: E402


MAX_RUL = 1_000_000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics(samples: list[RichSample], prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray([sample.target_pulses for sample in samples], dtype=float)
    lifetime = np.asarray([sample.lifetime for sample in samples], dtype=float)
    prediction = np.maximum(np.asarray(prediction, dtype=float), 0.0)
    error = prediction - truth
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    return {
        "nmae": float(np.mean(np.abs(error) / lifetime)),
        "nrmse": float(np.sqrt(np.mean((error / lifetime) ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "nbias": float(np.mean(error / lifetime)),
        "overestimation_rate": float(np.mean(error > 0)),
        "r2": float(1.0 - np.sum(error**2) / denominator) if denominator > 0 else float("nan"),
    }


class DomainTCN(nn.Module):
    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.input = nn.Conv1d(channels, hidden, 1)
        self.blocks = nn.Sequential(*(CausalConvBlock(hidden, d) for d in (1, 2, 4, 8)))
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.norm(self.blocks(self.input(x))[..., -1])
        return torch.sigmoid(self.head(hidden).squeeze(-1)), hidden


def coral_penalty(features: list[torch.Tensor]) -> torch.Tensor:
    penalty = features[0].new_zeros(())
    pairs = 0
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            left, right = features[i], features[j]
            left_centered = left - left.mean(0, keepdim=True)
            right_centered = right - right.mean(0, keepdim=True)
            left_cov = left_centered.T @ left_centered / max(left.shape[0] - 1, 1)
            right_cov = right_centered.T @ right_centered / max(right.shape[0] - 1, 1)
            penalty = penalty + (left.mean(0) - right.mean(0)).pow(2).mean()
            penalty = penalty + (left_cov - right_cov).pow(2).mean()
            pairs += 1
    return penalty / max(pairs, 1)


def fit_domain_model(kind, train_domains, validation_domains, seed, device, hidden, epochs, patience):
    seed_everything(seed)
    channels = train_domains[0][0].sequence.shape[0]
    model = DomainTCN(channels, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_tensors = [
        (
            torch.from_numpy(np.stack([sample.sequence for sample in domain])).to(device),
            torch.tensor(
                [np.log1p(sample.target_pulses) / np.log1p(MAX_RUL) for sample in domain],
                dtype=torch.float32,
                device=device,
            ),
        )
        for domain in train_domains
    ]
    validation_tensors = [
        (
            torch.from_numpy(np.stack([sample.sequence for sample in domain])).to(device),
            torch.tensor(
                [np.log1p(sample.target_pulses) / np.log1p(MAX_RUL) for sample in domain],
                dtype=torch.float32,
                device=device,
            ),
        )
        for domain in validation_domains
    ]
    weights = torch.ones(len(train_domains), device=device) / len(train_domains)
    generator = torch.Generator(device=device).manual_seed(seed)
    best, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for _ in range(6):
            losses, features = [], []
            for x_all, y_all in train_tensors:
                take = min(20, len(x_all))
                index = torch.randint(len(x_all), (take,), generator=generator, device=device)
                prediction, hidden_value = model(x_all[index])
                losses.append(F.smooth_l1_loss(prediction, y_all[index]))
                features.append(hidden_value)
            stacked = torch.stack(losses)
            if kind == "deep_coral_tcn":
                loss = stacked.mean() + 0.1 * coral_penalty(features)
            else:
                with torch.no_grad():
                    weights.mul_(torch.exp(0.05 * stacked.detach()))
                    weights.div_(weights.sum())
                loss = torch.sum(weights * stacked)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            score = float(np.mean([F.l1_loss(model(x)[0], y).item() for x, y in validation_tensors]))
        if score < best - 1e-6:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("domain model produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, best


def predict_domain(model, samples, device):
    x = torch.from_numpy(np.stack([sample.sequence for sample in samples])).to(device)
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        normalized, _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    latency = (time.perf_counter() - start) * 1000 / len(samples)
    return np.expm1(normalized.cpu().numpy() * np.log1p(MAX_RUL)), latency


def initial_state_lifetime(source_records, source_transformed, target_record, target_transformed):
    def vector(record, transformed):
        values, missing = transformed
        width = min(64, len(values))
        return np.concatenate([values[:width].mean(0), values[:width].std(0), missing[:width].mean(0)])
    x = np.vstack([vector(record, source_transformed[record.record_id]) for record in source_records])
    y = np.log1p([record.length - 1 for record in source_records])
    model = Ridge(alpha=10.0).fit(x, y)
    return float(np.expm1(model.predict(vector(target_record, target_transformed)[None])[0]))


def wiener_prediction(source_records, target_record, test_samples):
    terminal = np.median([
        record.frame.peak_voltage_v.iloc[-max(16, record.length // 50):].median()
        / max(abs(record.frame.peak_voltage_v.iloc[:64].median()), 1e-8)
        for record in source_records
    ])
    values = target_record.frame.peak_voltage_v.to_numpy(float)
    scale = max(abs(np.median(values[:64])), 1e-8)
    normalized = values / scale
    prediction = []
    start = time.perf_counter()
    for sample in test_samples:
        width = min(64, sample.index + 1)
        block = normalized[sample.index - width + 1 : sample.index + 1]
        slope = float(np.polyfit(np.arange(width), block, 1)[0])
        estimate = 0.0 if abs(slope) < 1e-8 else max(0.0, (terminal - block[-1]) / slope)
        prediction.append(estimate)
    return np.asarray(prediction), (time.perf_counter() - start) * 1000 / len(test_samples)


def aggregate(output: Path) -> None:
    metrics_path = output / "fold_metrics.csv"
    if not metrics_path.exists():
        return
    frame = pd.read_csv(metrics_path)
    ok = frame[frame.status == "complete"].copy()
    if ok.empty:
        return
    group_columns = ["family", "model"]
    if "evaluation" in ok.columns:
        group_columns.append("evaluation")
    summary = ok.groupby(group_columns, as_index=False).agg(
        completed_folds=("target", "nunique"),
        nmae_mean=("nmae", "mean"),
        nmae_worst=("nmae", "max"),
        nrmse_mean=("nrmse", "mean"),
        r2_mean=("r2", "mean"),
        nbias_mean=("nbias", "mean"),
        overestimation_rate_mean=("overestimation_rate", "mean"),
        parameters=("parameters", "max"),
        latency_ms_mean=("latency_ms", "mean"),
    ).sort_values(["completed_folds", "nmae_mean"], ascending=[False, True])
    summary.to_csv(output / "scoreboard.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pcss_rich_benchmark/b001_all_baselines.yaml")
    parser.add_argument("--model", action="append", help="Optional model subset; repeat flag.")
    parser.add_argument("--target", action="append", help="Optional target subset; repeat flag.")
    args = parser.parse_args()
    config_path = ROOT / args.config
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / cfg["runtime"]["output"]
    output.mkdir(parents=True, exist_ok=True)
    dataset_root = ROOT / cfg["dataset"]["root"]
    record_ids = list(cfg["dataset"]["records"])
    records = load_records(dataset_root, record_ids)
    protocol = cfg["protocol"]
    training = cfg["training"]
    model_groups = cfg["models"]
    group_order = (
        "classical", "standard_neural", "domain_generalization", "modern", "proposed"
    )
    all_models = sum((list(model_groups.get(key, [])) for key in group_order), [])
    requested_models = args.model or all_models
    requested_targets = args.target or record_ids
    family = {model: key for key, models in model_groups.items() for model in models}
    requested_device = cfg["runtime"].get("device", "auto")
    device = torch.device("cuda" if requested_device == "auto" and torch.cuda.is_available() else requested_device if requested_device != "auto" else "cpu")
    metrics_path = output / "fold_metrics.csv"
    predictions_path = output / "predictions.csv"
    error_path = output / "errors.jsonl"
    metric_rows = pd.read_csv(metrics_path).to_dict("records") if cfg["runtime"].get("resume") and metrics_path.exists() else []
    prediction_rows = pd.read_csv(predictions_path).to_dict("records") if cfg["runtime"].get("resume") and predictions_path.exists() else []
    completed = {(str(row["target"]), int(row["seed"]), str(row["model"])) for row in metric_rows if row.get("status") == "complete"}
    failed = {(str(row["target"]), int(row["seed"]), str(row["model"])) for row in metric_rows if row.get("status") == "failed"}

    for target_id in requested_targets:
        target_record = records[target_id]
        source_records = [records[key] for key in record_ids if key != target_id]
        transform = fit_transform(source_records, cfg["dataset"]["feature_columns"], protocol["preprocessing_fit_max_fraction"])
        transformed = {key: transformed_record(record, transform) for key, record in records.items()}
        train_domains = [
            build_samples(record, transformed[record.record_id], protocol["window"], *protocol["train_fraction"], protocol["train_samples_per_source"])
            for record in source_records
        ]
        validation_domains = [
            build_samples(record, transformed[record.record_id], protocol["window"], *protocol["validation_fraction"], protocol["validation_samples_per_source"])
            for record in source_records
        ]
        train_samples = [sample for domain in train_domains for sample in domain]
        validation_samples = [sample for domain in validation_domains for sample in domain]
        fractional_samples = [
            make_sample(
                target_record,
                transformed[target_id],
                int(np.floor(fraction * (target_record.length - 1))),
                protocol["window"],
            )
            for fraction in protocol["test_checkpoints"]
        ]
        evaluation_sets = {
            "fractional_life": (
                [str(value) for value in protocol["test_checkpoints"]], fractional_samples
            )
        }
        if protocol.get("fixed_event_indices"):
            fixed_indices = [
                int(index) for index in protocol["fixed_event_indices"]
                if protocol["window"] <= int(index) < target_record.length
            ]
            evaluation_sets["fixed_event_count"] = (
                [str(index) for index in fixed_indices],
                [
                    make_sample(
                        target_record, transformed[target_id], index, protocol["window"]
                    )
                    for index in fixed_indices
                ],
            )
        source_lifetimes = np.asarray([record.length - 1 for record in source_records], dtype=float)
        initial_lifetime = initial_state_lifetime(source_records, transformed, target_record, transformed[target_id])

        for seed in training["screening_seeds"]:
            for model_name in requested_models:
                key = (target_id, int(seed), model_name)
                if key in completed or key in failed:
                    continue
                start_time = time.time()
                try:
                    epochs, validation_score, parameters = 0, float("nan"), 0
                    if model_name == "population_median":
                        fit = None
                    elif model_name == "population_geomean":
                        fit = None
                    elif model_name == "initial_state_ridge":
                        fit = None
                    elif model_name == "wiener":
                        fit = None
                    elif model_name in model_groups["classical"]:
                        fit = fit_classical_log_rul(model_name, train_samples, int(seed))
                        parameters = fit.parameters
                    elif model_name in model_groups["domain_generalization"]:
                        fit, epochs, validation_score = fit_domain_model(
                            model_name, train_domains, validation_domains, int(seed), device,
                            training["hidden_size"], training["epochs"], training["patience"],
                        )
                        parameters = sum(parameter.numel() for parameter in fit.parameters())
                    else:
                        fit = fit_neural(
                            model_name,
                            train_samples,
                            validation_samples,
                            window=protocol["window"],
                            hidden=training["hidden_size"],
                            bins=training["hazard_bins"],
                            epochs=training["epochs"],
                            patience=training["patience"],
                            batch_size=training["batch_size"],
                            learning_rate=training["learning_rate"],
                            weight_decay=training["weight_decay"],
                            gradient_clip=training["gradient_clip"],
                            seed=int(seed),
                            device=device,
                            target_mode="log_pulses",
                        )
                        parameters, epochs, validation_score = fit.parameters, fit.epochs, fit.best_validation
                    for evaluation, (positions, test_samples) in evaluation_sets.items():
                        if model_name == "population_median":
                            prediction = predict_clock_prior(
                                test_samples, float(np.median(source_lifetimes))
                            )
                            latency = 0.0
                        elif model_name == "population_geomean":
                            prediction = predict_clock_prior(
                                test_samples,
                                float(np.exp(np.mean(np.log(source_lifetimes)))),
                            )
                            latency = 0.0
                        elif model_name == "initial_state_ridge":
                            prediction = predict_clock_prior(test_samples, initial_lifetime)
                            latency = 0.0
                        elif model_name == "wiener":
                            prediction, latency = wiener_prediction(
                                source_records, target_record, test_samples
                            )
                        elif model_name in model_groups["classical"]:
                            prediction, latency = predict_classical_log_rul(
                                fit.model, test_samples
                            )
                        elif model_name in model_groups["domain_generalization"]:
                            prediction, latency = predict_domain(fit, test_samples, device)
                        else:
                            prediction, latency = predict_direct_log_rul(
                                fit, test_samples, device
                            )
                        score = metrics(test_samples, prediction)
                        row = {
                            "status": "complete",
                            "protocol": "caro_unified_lodo_v1",
                            "evaluation": evaluation,
                            "target": target_id,
                            "sources": "+".join(record.record_id for record in source_records),
                            "seed": int(seed),
                            "family": family[model_name],
                            "model": model_name,
                            **score,
                            "parameters": parameters,
                            "epochs": epochs,
                            "validation_mae": validation_score,
                            "latency_ms": latency,
                            "elapsed_seconds": time.time() - start_time,
                        }
                        metric_rows.append(row)
                        for position, sample, value in zip(positions, test_samples, prediction):
                            prediction_rows.append(
                                {
                                    "evaluation": evaluation,
                                    "target": target_id,
                                    "seed": int(seed),
                                    "model": model_name,
                                    "position": position,
                                    "index": sample.index,
                                    "true_rul": sample.target_pulses,
                                    "predicted_rul": float(value),
                                }
                            )
                    completed.add(key)
                    print(json.dumps({"target": target_id, "seed": int(seed), "model": model_name, "nmae": score["nmae"], "seconds": row["elapsed_seconds"]}), flush=True)
                except Exception as error:
                    failure = {
                        "status": "failed",
                        "protocol": "rich_feature_lodo_v1",
                        "target": target_id,
                        "sources": "+".join(record.record_id for record in source_records),
                        "seed": int(seed),
                        "family": family.get(model_name, "unknown"),
                        "model": model_name,
                        "nmae": float("nan"), "nrmse": float("nan"), "mae": float("nan"),
                        "rmse": float("nan"), "nbias": float("nan"),
                        "overestimation_rate": float("nan"), "r2": float("nan"),
                        "parameters": 0, "epochs": 0, "validation_mae": float("nan"),
                        "latency_ms": float("nan"), "elapsed_seconds": time.time() - start_time,
                    }
                    metric_rows.append(failure)
                    with error_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"key": key, "error": repr(error), "traceback": traceback.format_exc()}) + "\n")
                    failed.add(key)
                    print(json.dumps({"target": target_id, "model": model_name, "failed": repr(error)}), flush=True)
                write_csv(metrics_path, metric_rows)
                write_csv(predictions_path, prediction_rows)
                aggregate(output)

    manifest = {
        "run_id": cfg["run_id"],
        "status": "complete" if len(completed | failed) >= len(requested_targets) * len(training["screening_seeds"]) * len(requested_models) else "partial",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "dataset_manifest_sha256": sha256(dataset_root / "file_manifest.csv"),
        "device": str(device),
        "records": record_ids,
        "requested_targets": requested_targets,
        "requested_models": requested_models,
        "completed": len(completed),
        "failed": len(failed),
        "input_channels": len(cfg["dataset"]["feature_columns"]) * 3 + 1,
        "leakage_guards": [
            "record split before windowing",
            "source-training-prefix-only imputation and scaling",
            "no target endpoint used for fitting or model selection",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    aggregate(output)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
