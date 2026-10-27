from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Trace
from .features import causal_prefix_features


def checkpoint_index(length: int, fraction: float) -> int:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    return min(length - 2, max(1, int(np.floor(fraction * (length - 1)))))


def age_mirror_prediction(observed_index: int) -> float:
    """Predict remaining observations equal to elapsed observations."""
    return float(observed_index + 1)


def fit_ridge_remaining_ratio(
    train_traces: list[Trace], checkpoints: list[float]
):
    """Fit an exploratory prefix-only model for remaining/elapsed ratio."""
    features: list[np.ndarray] = []
    targets: list[float] = []
    for trace in train_traces:
        for fraction in checkpoints:
            idx = checkpoint_index(trace.length, fraction)
            features.append(causal_prefix_features(trace.voltage[: idx + 1]))
            remaining = trace.length - 1 - idx
            targets.append(float(remaining / (idx + 1)))
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(np.vstack(features), np.asarray(targets))
    return model


def ridge_rul_prediction(model, trace: Trace, observed_index: int) -> float:
    features = causal_prefix_features(trace.voltage[: observed_index + 1])[None, :]
    ratio = max(0.0, float(model.predict(features)[0]))
    return ratio * float(observed_index + 1)

