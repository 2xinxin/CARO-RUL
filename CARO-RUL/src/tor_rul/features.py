from __future__ import annotations

import numpy as np


FEATURE_NAMES = (
    "last_rel",
    "mean_rel",
    "std_rel",
    "min_rel",
    "max_rel",
    "global_slope_rel",
    "recent_slope_rel",
    "recent_change_rel",
)


def _slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x -= x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x, values - values.mean()) / denom) if denom else 0.0


def causal_prefix_features(values: np.ndarray, recent_window: int = 128) -> np.ndarray:
    """Scale-relative features computed solely from the supplied prefix."""
    prefix = np.asarray(values, dtype=np.float64)
    if prefix.ndim != 1 or prefix.size < 2:
        raise ValueError("a one-dimensional prefix with at least two values is required")
    init_len = min(128, prefix.size)
    scale = float(np.median(np.abs(prefix[:init_len])))
    scale = max(scale, np.finfo(np.float64).eps)
    relative = prefix / scale
    recent = relative[-min(recent_window, relative.size) :]
    split = max(1, recent.size // 2)
    recent_change = float(np.mean(recent[split:]) - np.mean(recent[:split]))
    return np.asarray(
        [
            relative[-1],
            relative.mean(),
            relative.std(ddof=0),
            relative.min(),
            relative.max(),
            _slope(relative),
            _slope(recent),
            recent_change,
        ],
        dtype=np.float64,
    )

