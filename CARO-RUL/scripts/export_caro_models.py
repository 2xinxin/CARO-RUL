"""Retrain and save the 35 final CARO-RUL LODO fold/seed checkpoints."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcss_rich_benchmark.data import (  # noqa: E402
    build_samples,
    fit_transform,
    load_records,
    transformed_record,
)
from tor_rul.dash_experiment import fit_neural  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    config_path = ROOT / "configs" / "b016_caro_unified_5seed.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    records = load_records(ROOT / cfg["dataset"]["root"], cfg["dataset"]["records"])
    protocol = cfg["protocol"]
    training = cfg["training"]
    out = ROOT / "models" / "caro_rul_lodo"
    out.mkdir(parents=True, exist_ok=True)
    requested = training["screening_seeds"]
    device_name = cfg["runtime"].get("device", "auto")
    device = torch.device(
        "cuda" if device_name == "auto" and torch.cuda.is_available()
        else device_name if device_name != "auto" else "cpu"
    )
    exported = []
    for target_id in cfg["dataset"]["records"]:
        source_records = [record for key, record in records.items() if key != target_id]
        transform = fit_transform(
            source_records,
            cfg["dataset"]["feature_columns"],
            protocol["preprocessing_fit_max_fraction"],
        )
        transformed = {
            key: transformed_record(record, transform) for key, record in records.items()
        }
        train_samples = [
            sample
            for record in source_records
            for sample in build_samples(
                record,
                transformed[record.record_id],
                protocol["window"],
                *protocol["train_fraction"],
                protocol["train_samples_per_source"],
            )
        ]
        validation_samples = [
            sample
            for record in source_records
            for sample in build_samples(
                record,
                transformed[record.record_id],
                protocol["window"],
                *protocol["validation_fraction"],
                protocol["validation_samples_per_source"],
            )
        ]
        for seed in requested:
            fit = fit_neural(
                "caro_rul",
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
            payload = {
                "format_version": 1,
                "model_name": "caro_rul",
                "target_device": target_id,
                "source_devices": [record.record_id for record in source_records],
                "seed": int(seed),
                "input_channels": len(cfg["dataset"]["feature_columns"]) * 3 + 1,
                "window": int(protocol["window"]),
                "hidden_size": int(training["hidden_size"]),
                "hazard_bins": int(training["hazard_bins"]),
                "feature_columns": cfg["dataset"]["feature_columns"],
                "transform": {
                    "median": transform.median.tolist(),
                    "mean": transform.mean.tolist(),
                    "std": transform.std.tolist(),
                },
                "validation_mae_log_rul": float(fit.best_validation),
                "epochs_completed": int(fit.epochs),
                "state_dict": {key: value.detach().cpu() for key, value in fit.model.state_dict().items()},
                "config_sha256": sha256(config_path),
                "dataset_manifest_sha256": sha256(ROOT / "dataset" / "file_manifest.csv"),
            }
            path = out / f"caro_rul_target-{target_id}_seed-{seed}.pt"
            torch.save(payload, path)
            exported.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
            print(f"saved {path.name}", flush=True)
    (out / "manifest.json").write_text(
        json.dumps({"count": len(exported), "device": str(device), "files": exported}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
