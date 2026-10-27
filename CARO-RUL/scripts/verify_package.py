"""Validate dataset hashes, result completeness, and saved CARO checkpoints."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pcss_single_model.model_v9 import build_model  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dataset = json.loads((ROOT / "dataset" / "dataset_manifest.json").read_text(encoding="utf-8"))
    records = pd.read_csv(ROOT / "dataset" / dataset["records"])
    assert len(dataset["record_ids"]) == 7
    assert int(records["event_count"].sum()) == 69355
    private_provenance_complete = True
    for row in records.itertuples(index=False):
        source_path = ROOT / row.source_csv
        if source_path.exists():
            assert sha256(source_path) == row.source_sha256
        else:
            private_provenance_complete = False
        assert sha256(ROOT / row.benchmark_csv) == row.benchmark_sha256
    file_manifest_path = ROOT / "dataset" / "file_manifest.csv"
    for row in pd.read_csv(file_manifest_path).itertuples(index=False):
        path = ROOT / "dataset" / row.relative_path
        assert path.stat().st_size == int(row.bytes)
        assert sha256(path) == row.sha256
    for row in pd.read_csv(ROOT / "dataset" / "raw" / "source_archives.csv").itertuples(index=False):
        path = ROOT / row.repository_relative_path
        if path.exists():
            assert path.stat().st_size == int(row.bytes)
            assert sha256(path) == row.sha256
        else:
            private_provenance_complete = False
    b016 = pd.read_csv(ROOT / "results" / "PCSS_RICH_BENCHMARK" / "B016_CARO_UNIFIED_5SEED" / "fold_metrics.csv")
    assert len(b016) == 1470
    assert set(b016["status"]) == {"complete"}
    manifest = json.loads((ROOT / "models" / "caro_rul_lodo" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["count"] == 35
    for item in manifest["files"]:
        payload = torch.load(ROOT / item["path"], map_location="cpu", weights_only=False)
        assert payload["dataset_manifest_sha256"] == sha256(file_manifest_path)
        model = build_model(
            payload["model_name"], payload["input_channels"],
            payload["hidden_size"], payload["hazard_bins"]
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        with torch.no_grad():
            output = model(torch.zeros(2, payload["input_channels"], payload["window"]))
        assert tuple(output["prediction"].shape) == (2,)
    scope = "full private provenance" if private_provenance_complete else "compact release"
    print(f"PASS ({scope}): 7 records, 69,355 events, B016 results, and 35 CARO models")


if __name__ == "__main__":
    main()
