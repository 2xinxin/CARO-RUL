"""Fast structural checks for the public source release."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "pyproject.toml",
    "environment.yml",
    "src/pcss_rich_benchmark/run_baselines.py",
    "src/pcss_single_model/model_v9.py",
    "configs/b016_caro_unified_5seed.yaml",
    "dataset/DATA_DICTIONARY.md",
]


def main() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).is_file()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")

    local_path = re.compile(r"(?:[A-Za-z]:\\\\|/home/|/Users/)")
    for config_path in sorted((ROOT / "configs").glob("*.yaml")):
        text = config_path.read_text(encoding="utf-8")
        if local_path.search(text):
            raise SystemExit(f"Machine-local path in {config_path.relative_to(ROOT)}")
        config = yaml.safe_load(text)
        if not isinstance(config, dict) or "runtime" not in config:
            raise SystemExit(f"Invalid configuration: {config_path.relative_to(ROOT)}")

    # Ignore locally generated test caches and the declared Git-ignored output
    # directories. Public ZIP packaging separately excludes these paths.
    ignored_roots = {"results", "outputs", "models", ".venv", "venv", ".git"}
    present = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(part in ignored_roots or part == "__pycache__" for part in relative.parts):
            continue
        if "private_raw_sources" in relative.parts or (path.is_file() and path.suffix in {".pt", ".ckpt", ".pth"}):
            present.append(relative.as_posix())
    if present:
        raise SystemExit(f"Private/generated artifacts found: {present[:10]}")
    print("Repository structure and public-release boundaries verified.")


if __name__ == "__main__":
    main()
