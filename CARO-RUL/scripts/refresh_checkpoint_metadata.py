"""Rebind checkpoint provenance after a metadata-only dataset manifest update."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "caro_rul_lodo"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dataset_hash = sha256(ROOT / "dataset" / "file_manifest.csv")
    files = []
    for path in sorted(MODEL_ROOT.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payload["dataset_manifest_sha256"] = dataset_hash
        torch.save(payload, path)
        files.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    (MODEL_ROOT / "manifest.json").write_text(
        json.dumps({"count": len(files), "device": "metadata-refresh", "files": files}, indent=2),
        encoding="utf-8",
    )
    print(f"updated {len(files)} checkpoints for dataset manifest {dataset_hash}")


if __name__ == "__main__":
    main()
