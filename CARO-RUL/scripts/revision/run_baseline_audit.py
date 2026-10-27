from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


PROJECT = Path(__file__).resolve().parents[2]
VENDOR = PROJECT / "vendor"
TSLIB = VENDOR / "Time-Series-Library"
NBEATS = VENDOR / "N-BEATS" / "models" / "nbeats.py"
OUT = PROJECT / "results" / "revision" / "A2-004"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(TSLIB))

from pcss_rich_benchmark.data import build_samples, fit_transform, load_records, make_sample, transformed_record  # noqa: E402
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
MODELS = ["dlinear", "patchtst", "nbeats"]
LRS = [1e-3, 3e-4]
MAX_RUL = 1_000_000.0
WINDOW, HIDDEN = 128, 24


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def config():
    return SimpleNamespace(
        task_name="classification", seq_len=WINDOW, label_len=0, pred_len=0,
        enc_in=43, dec_in=43, c_out=43, d_model=HIDDEN, n_heads=4,
        e_layers=2, d_layers=1, d_ff=HIDDEN * 2, moving_avg=25, factor=3,
        distil=True, dropout=0.1, embed="fixed", freq="h", activation="gelu",
        output_attention=False, num_class=1, top_k=3, num_kernels=6,
        individual=False,
    )


def load_nbeats():
    spec = importlib.util.spec_from_file_location("revision_nbeats", NBEATS)
    if spec is None or spec.loader is None: raise ImportError(NBEATS)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class OfficialScalarAdapter(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name
        if name in {"dlinear", "patchtst"}:
            from models import DLinear, PatchTST
            self.model = {"dlinear": DLinear.Model, "patchtst": PatchTST.Model}[name](config())
        elif name == "nbeats":
            reference = load_nbeats()
            self.pre = nn.Conv1d(43, 1, 1)
            blocks = []
            for _ in range(4):
                blocks.append(reference.NBeatsBlock(
                    input_size=WINDOW, theta_size=WINDOW + 1,
                    basis_function=reference.GenericBasis(WINDOW, 1),
                    layers=4, layer_size=max(64, HIDDEN * 2),
                ))
            self.model = reference.NBeats(nn.ModuleList(blocks))
        else:
            raise KeyError(name)

    def stabilize_head(self, target_center):
        if hasattr(self.model, "projection") and isinstance(self.model.projection, nn.Linear):
            nn.init.zeros_(self.model.projection.weight)
            bias = float(np.log(target_center / max(1.0 - target_center, 1e-6)))
            nn.init.constant_(self.model.projection.bias, bias)

    def forward(self, x):
        if self.name == "nbeats":
            raw = self.model(self.pre(x).squeeze(1), torch.ones_like(x[:, 0])).squeeze(-1)
        else:
            sequence = x.transpose(1, 2)
            valid = torch.ones(sequence.shape[:2], device=x.device, dtype=x.dtype)
            raw = self.model(sequence, valid, None, None).squeeze(-1)
        return torch.sigmoid(raw)


def fit(name, train, validation, seed, lr, device):
    seed_all(seed)
    model = OfficialScalarAdapter(name).to(device)
    target_center = np.median([np.log1p(s.target_pulses) / np.log1p(MAX_RUL) for s in train])
    model.stabilize_head(float(np.clip(target_center, 0.05, 0.95)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed + 100003)
    train_loader = DataLoader(SequenceRULDataset(train, "log_pulses"), batch_size=128, shuffle=True, generator=gen)
    val_loader = DataLoader(SequenceRULDataset(validation, "log_pulses"), batch_size=128, shuffle=False)
    best, state, stale, trace = float("inf"), None, 0, []
    for epoch in range(1, 101):
        model.train(); total = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x); loss = F.smooth_l1_loss(pred, y)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += float(loss) * len(y); count += len(y)
        model.eval(); val_total = val_n = 0; val_values = []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device); pred = model(x)
                val_total += float(F.l1_loss(pred, y, reduction="sum")); val_n += len(y)
                val_values.append(pred.cpu().numpy())
        score = val_total / val_n
        vals = np.concatenate(val_values)
        trace.append({
            "epoch": epoch, "train_smoothl1": total / count, "validation_mae_log": score,
            "validation_pred_mean": float(vals.mean()), "validation_pred_std": float(vals.std()),
            "validation_saturation_low": float(np.mean(vals < 1e-3)),
            "validation_saturation_high": float(np.mean(vals > 1 - 1e-3)),
        })
        if score < best - 1e-5:
            best = score; state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; stale = 0
        else:
            stale += 1
            if stale >= 15: break
    model.load_state_dict(state)
    return model, best, trace


def predict(model, samples, device):
    loader = DataLoader(SequenceRULDataset(samples, "log_pulses"), batch_size=256, shuffle=False)
    values = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader: values.append(model(x.to(device)).cpu().numpy())
    z = np.concatenate(values)
    return z, np.expm1(np.clip(z, 0, 1) * np.log1p(MAX_RUL))


def metrics(samples, pred):
    truth = np.asarray([s.target_pulses for s in samples]); life = np.asarray([s.lifetime for s in samples])
    error = pred - truth; denom = np.sum((truth - truth.mean()) ** 2)
    return {
        "nmae": float(np.mean(np.abs(error) / life)), "nrmse": float(np.sqrt(np.mean((error / life) ** 2))),
        "mae_events": float(np.mean(np.abs(error))), "nbias": float(np.mean(error / life)),
        "oer": float(np.mean(error > 0)), "r2": float(1 - np.sum(error**2) / denom) if denom else np.nan,
        "prediction_min_events": float(pred.min()), "prediction_max_events": float(pred.max()),
        "prediction_saturated_near_max": float(np.mean(pred > 0.99 * MAX_RUL)),
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--target", action="append"); parser.add_argument("--seed", action="append", type=int); parser.add_argument("--model", action="append"); parser.add_argument("--device", default="auto"); args = parser.parse_args()
    targets, seeds, names = args.target or DEVICES, args.seed or SEEDS, args.model or MODELS
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu"))
    OUT.mkdir(parents=True, exist_ok=True); trace_dir = OUT / "training_traces"; trace_dir.mkdir(exist_ok=True); ckpt_dir = OUT / "checkpoints"; ckpt_dir.mkdir(exist_ok=True)
    metrics_path, predictions_path, selection_path = OUT / "fold_metrics.csv", OUT / "predictions.csv", OUT / "learning_rate_selection.csv"
    metric_rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.exists() else []
    pred_rows = pd.read_csv(predictions_path).to_dict("records") if predictions_path.exists() else []
    selection_rows = pd.read_csv(selection_path).to_dict("records") if selection_path.exists() else []
    done = {(r["target"], int(r["seed"]), r["model"]) for r in selection_rows}
    records = load_records(CODE / "dataset", DEVICES)
    for target in targets:
        source_ids = [d for d in DEVICES if d != target]; source = [records[d] for d in source_ids]
        transform = fit_transform(source, FEATURES, 0.75); transformed = {d: transformed_record(records[d], transform) for d in DEVICES}
        train = [s for d in source_ids for s in build_samples(records[d], transformed[d], WINDOW, 0.05, 0.75, 128)]
        validation = [s for d in source_ids for s in build_samples(records[d], transformed[d], WINDOW, 0.76, 0.88, 32)]
        fractional = [make_sample(records[target], transformed[target], int(np.floor(f * (records[target].length - 1))), WINDOW) for f in [0.2,0.3,0.4,0.5,0.6,0.7,0.8]]
        fixed_idx = [i for i in [256,512,768,1024,1280,1536] if i < records[target].length - 1]
        fixed = [make_sample(records[target], transformed[target], i, WINDOW) for i in fixed_idx]
        for seed in seeds:
            for name in names:
                if (target, seed, name) in done: continue
                candidates = []
                for lr in LRS:
                    model, score, trace = fit(name, train, validation, seed, lr, device)
                    pd.DataFrame(trace).to_csv(trace_dir / f"{name}_target-{target}_seed-{seed}_lr-{lr:g}.csv", index=False)
                    candidates.append((score, lr, model, trace))
                score, lr, model, trace = min(candidates, key=lambda x: (x[0], x[1]))
                torch.save({"model": name, "target": target, "seed": seed, "learning_rate": lr, "validation_mae_log": score, "state_dict": model.state_dict()}, ckpt_dir / f"{name}_target-{target}_seed-{seed}.pt")
                selection_rows.append({"target": target, "seed": seed, "model": name, "selected_learning_rate": lr, "validation_mae_log": score, "epochs": len(trace), "parameters": sum(p.numel() for p in model.parameters())})
                for evaluation, samples in [("fractional_life", fractional), ("fixed_event_count", fixed)]:
                    z, pred = predict(model, samples, device); row = {"target": target, "seed": seed, "model": name, "evaluation": evaluation, **metrics(samples, pred), "selected_learning_rate": lr, "validation_mae_log": score, "epochs": len(trace), "parameters": sum(p.numel() for p in model.parameters())}; metric_rows.append(row)
                    for pos, sample, zi, pi in zip(range(len(samples)), samples, z, pred): pred_rows.append({"target": target, "seed": seed, "model": name, "evaluation": evaluation, "position": pos, "index": sample.index, "true_rul": sample.target_pulses, "lifetime": sample.lifetime, "predicted_z": zi, "predicted_rul": pi})
                done.add((target, seed, name)); pd.DataFrame(metric_rows).to_csv(metrics_path, index=False); pd.DataFrame(pred_rows).to_csv(predictions_path, index=False); pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
                print(json.dumps({"target": target, "seed": seed, "model": name, "lr": lr, "validation": score}), flush=True)
    old = pd.read_csv(CODE / "results" / "PCSS_RICH_BENCHMARK" / "B016_CARO_UNIFIED_5SEED" / "fold_metrics.csv")
    old = old[old.model.isin(["dlinear_faithful", "nbeats_faithful", "patchtst_faithful"])]
    new = pd.DataFrame(metric_rows)
    audit = pd.concat([
        old.groupby(["evaluation", "model"], as_index=False).agg(nmae=("nmae", "mean"), worst=("nmae", "max"), validation_mae_log=("validation_mae", "mean"), epochs=("epochs", "mean")).assign(run="submitted"),
        new.groupby(["evaluation", "model"], as_index=False).agg(nmae=("nmae", "mean"), worst=("nmae", "max"), validation_mae_log=("validation_mae_log", "mean"), epochs=("epochs", "mean")).assign(run="stabilized_100epoch"),
    ], ignore_index=True)
    audit.to_csv(OUT / "submitted_vs_stabilized.csv", index=False)
    manifest = {"run_id": "A2-004", "status": "complete", "official_sources": {"Time-Series-Library": "4e938a1767106324dd753b2a44832bf870a0252e", "N-BEATS": "c746a4f13ffc957487e0c3279b182c3030836053"}, "changes_from_submitted_adapter": ["final scalar projection initialized at the source target median to prevent initial sigmoid saturation", "maximum epochs increased from 20 to 100", "patience increased from 5 to 15", "learning rate selected from 1e-3 and 3e-4 using source-only temporal validation"], "same_data_contract": "outer LODO, source-only preprocessing, identical windows and test schedules"}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))


if __name__ == "__main__": main()
