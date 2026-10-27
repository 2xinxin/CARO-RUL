from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trace:
    trace_id: str
    count: np.ndarray
    voltage: np.ndarray
    source_path: Path

    @property
    def length(self) -> int:
        return int(self.voltage.size)


def load_trace(path: str | Path, trace_id: str, has_header: bool) -> Trace:
    """Load one two-column degradation trace without silently dropping row 1."""
    source = Path(path)
    frame = pd.read_csv(source, header=0 if has_header else None)
    if frame.shape[1] != 2:
        raise ValueError(f"{source} must have exactly two columns, got {frame.shape[1]}")
    frame.columns = ["count", "peak_voltage"]
    count = frame["count"].to_numpy(dtype=np.float64)
    voltage = frame["peak_voltage"].to_numpy(dtype=np.float64)
    if count.size < 3:
        raise ValueError(f"{source} is too short for RUL analysis")
    if not np.isfinite(count).all() or not np.isfinite(voltage).all():
        raise ValueError(f"{source} contains non-finite values")
    if not np.allclose(np.diff(count), 1.0):
        raise ValueError(f"{source} count column is not contiguous")
    return Trace(trace_id=trace_id, count=count, voltage=voltage, source_path=source)

