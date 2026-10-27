"""Generate a tiny non-scientific dataset for installation and CI checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "examples" / "synthetic_dataset"
SEQUENCES = OUT / "processed" / "event_sequences"
FEATURES = [
    "peak_voltage_v",
    "voltage_min_v",
    "voltage_peak_to_peak_v",
    "voltage_mean_v",
    "voltage_std_v",
    "voltage_rms_v",
    "baseline_mean_v",
    "baseline_std_v",
    "baseline_corrected_peak_v",
    "baseline_corrected_trough_v",
    "peak_time_s",
    "rise_time_10_90_s",
    "fwhm_s",
    "positive_pulse_area_v_s",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    SEQUENCES.mkdir(parents=True, exist_ok=True)
    manifest = []
    for device, length in enumerate((160, 176, 192), start=1):
        rng = np.random.default_rng(100 + device)
        event = np.arange(1, length + 1)
        life = event / length
        peak = 2.2 - 0.7 * life + 0.04 * rng.normal(size=length) + 0.06 * device
        matrix = np.column_stack(
            [
                peak,
                -0.08 - 0.02 * life,
                peak + 0.08,
                0.20 * peak,
                0.05 + 0.01 * life,
                0.45 * peak,
                np.full(length, 0.02),
                np.full(length, 0.005),
                peak - 0.02,
                -0.10 - 0.01 * life,
                1.0e-7 + 1.0e-8 * life,
                2.0e-8 + 2.0e-9 * life,
                5.0e-8 + 5.0e-9 * life,
                1.0e-7 * peak,
            ]
        )
        frame = pd.DataFrame(matrix, columns=FEATURES)
        frame.insert(0, "event_index", event)
        if device == 3:
            frame.loc[::17, "fwhm_s"] = np.nan
            frame.loc[::23, "positive_pulse_area_v_s"] = np.nan
        path = SEQUENCES / f"SW{device}.csv"
        frame.to_csv(path, index=False)
        manifest.append({"file": path.relative_to(OUT).as_posix(), "rows": length, "sha256": sha256(path)})
    pd.DataFrame(manifest).to_csv(OUT / "file_manifest.csv", index=False)
    print(f"Synthetic smoke-test dataset written to {OUT}")


if __name__ == "__main__":
    main()
