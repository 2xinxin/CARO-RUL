"""Reference feature extraction for the SiC-PCSS review archive.

The function below documents the exact conventions used to convert one voltage
waveform into the 14 descriptors consumed by the RUL models.  It intentionally
does not infer a sampling interval: time-valued descriptors require the
acquisition time vector supplied with the waveform.
"""

from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "peak_voltage",
    "minimum_voltage",
    "peak_to_peak_voltage",
    "mean_voltage",
    "standard_deviation",
    "rms_voltage",
    "baseline_mean",
    "baseline_standard_deviation",
    "baseline_corrected_peak",
    "baseline_corrected_trough",
    "peak_time",
    "rise_time_10_90",
    "full_width_half_maximum",
    "positive_pulse_area",
)


def _first_crossing(values: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(values >= threshold)
    return int(indices[0]) if indices.size else None


def extract_waveform_features(time_s, voltage_v) -> dict[str, float]:
    """Extract one causal descriptor row from a voltage waveform.

    Rules
    -----
    * Non-finite time/voltage pairs are removed together; remaining samples are
      stably sorted by time.  Fewer than three finite samples returns NaNs.
    * Baseline statistics use the first ``max(1, ceil(0.10*n))`` samples.
    * Rise time uses the first 10% and 90% crossings of the baseline-corrected
      pulse before (and including) its maximum.
    * FWHM spans the first and last samples at or above half the corrected peak.
    * Pulse area integrates only the positive baseline-corrected component.
    """
    t = np.asarray(time_s, dtype=float).reshape(-1)
    v = np.asarray(voltage_v, dtype=float).reshape(-1)
    if t.size != v.size:
        raise ValueError("time_s and voltage_v must have equal length")

    finite = np.isfinite(t) & np.isfinite(v)
    t, v = t[finite], v[finite]
    if t.size < 3:
        return {name: float("nan") for name in FEATURE_NAMES}
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]

    n_baseline = max(1, int(np.ceil(0.10 * t.size)))
    baseline = v[:n_baseline]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline, ddof=0))
    corrected = v - baseline_mean

    peak_index = int(np.argmax(v))
    corrected_peak = float(corrected[peak_index])
    pre_peak = corrected[: peak_index + 1]
    i10 = _first_crossing(pre_peak, 0.10 * corrected_peak)
    i90 = _first_crossing(pre_peak, 0.90 * corrected_peak)
    rise_time = float(t[i90] - t[i10]) if i10 is not None and i90 is not None and i90 >= i10 else float("nan")

    above_half = np.flatnonzero(corrected >= 0.50 * corrected_peak)
    fwhm = float(t[above_half[-1]] - t[above_half[0]]) if above_half.size else float("nan")
    positive_area = float(np.trapz(np.maximum(corrected, 0.0), t))

    values = (
        float(np.max(v)),
        float(np.min(v)),
        float(np.ptp(v)),
        float(np.mean(v)),
        float(np.std(v, ddof=0)),
        float(np.sqrt(np.mean(v**2))),
        baseline_mean,
        baseline_std,
        corrected_peak,
        float(np.min(corrected)),
        float(t[peak_index]),
        rise_time,
        fwhm,
        positive_area,
    )
    return dict(zip(FEATURE_NAMES, values, strict=True))

