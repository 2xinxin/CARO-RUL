from __future__ import annotations

import numpy as np


def summarize_rul(y_true: np.ndarray, y_pred: np.ndarray, lifetime: float) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    error = pred - truth
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_mae": float(np.mean(np.abs(error)) / max(lifetime, 1.0)),
        "bias": float(np.mean(error)),
    }

