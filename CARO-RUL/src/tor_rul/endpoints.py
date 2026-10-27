from __future__ import annotations

import numpy as np

from .data import Trace


def time_to_terminal(trace: Trace) -> np.ndarray:
    """Exploratory label: observations remaining to the final recorded row."""
    return (trace.length - 1 - np.arange(trace.length)).astype(np.float64)


def threshold_crossing(
    trace: Trace,
    ratio: float,
    reference_window: int = 1,
    persistence: int = 5,
) -> int | None:
    """Return the first persistent decline crossing of ratio * initial reference.

    This reconstructs a candidate endpoint for sensitivity analysis. It does not
    assert that the threshold is a physical failure event.
    """
    if not 0.0 < ratio < 1.0:
        raise ValueError("ratio must be between zero and one")
    if reference_window < 1 or reference_window > trace.length:
        raise ValueError("invalid reference_window")
    if persistence < 1:
        raise ValueError("persistence must be positive")
    reference = float(np.median(trace.voltage[:reference_window]))
    below = trace.voltage <= ratio * reference
    if persistence == 1:
        hits = np.flatnonzero(below)
    else:
        hits = np.flatnonzero(
            np.convolve(below.astype(np.int16), np.ones(persistence, dtype=np.int16), mode="valid")
            == persistence
        )
    return int(hits[0]) if hits.size else None

