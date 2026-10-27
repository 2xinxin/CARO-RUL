from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Trace
from .features import causal_prefix_features


@dataclass(frozen=True)
class RULSample:
    trace_id: str
    index: int
    sequence: np.ndarray
    tabular: np.ndarray
    target_fraction: float
    target_pulses: float
    lifetime: int


def _initial_scale(trace: Trace) -> float:
    width = min(128, trace.length)
    value = float(np.median(np.abs(trace.voltage[:width])))
    return max(value, np.finfo(np.float64).eps)


def make_sample(trace: Trace, index: int, window: int) -> RULSample:
    if index < 1 or index >= trace.length - 1:
        raise ValueError("sample index must leave at least one future observation")
    scale = _initial_scale(trace)
    start = max(0, index - window + 1)
    prefix = trace.voltage[: index + 1] / scale
    raw = prefix[start : index + 1]
    running_max = np.maximum.accumulate(prefix)[start : index + 1]
    running_min = np.minimum.accumulate(prefix)[start : index + 1]
    drawdown = (running_max - raw) / np.maximum(np.abs(running_max), 1e-8)
    rebound = (raw - running_min) / np.maximum(np.abs(running_max - running_min), 1e-8)
    if raw.size < window:
        pad = window - raw.size
        raw = np.pad(raw, (pad, 0), mode="edge")
        drawdown = np.pad(drawdown, (pad, 0), mode="edge")
        rebound = np.pad(rebound, (pad, 0), mode="edge")
    delta = np.diff(raw, prepend=raw[0])
    short_mean = np.convolve(raw, np.ones(9) / 9.0, mode="same")
    local_residual = raw - short_mean
    age = np.full(window, np.log1p(index) / np.log1p(50000.0), dtype=np.float64)
    initial_scale = np.full(
        window, np.log1p(scale) / np.log1p(10000.0), dtype=np.float64
    )
    sequence = np.stack(
        [raw, delta, drawdown, rebound, local_residual, age, initial_scale], axis=0
    ).astype(np.float32)
    prefix_features = causal_prefix_features(trace.voltage[: index + 1])
    tabular = np.concatenate(
        [prefix_features, [np.log1p(index) / np.log1p(50000.0)]]
    ).astype(np.float32)
    remaining = float(trace.length - 1 - index)
    fraction = remaining / float(trace.length - 1)
    return RULSample(
        trace_id=trace.trace_id,
        index=index,
        sequence=sequence,
        tabular=tabular,
        target_fraction=float(fraction),
        target_pulses=remaining,
        lifetime=trace.length - 1,
    )


def sample_indices(length: int, low: float, high: float, count: int) -> np.ndarray:
    lo = max(1, int(np.floor(low * (length - 1))))
    hi = min(length - 2, int(np.floor(high * (length - 1))))
    if hi <= lo:
        raise ValueError("invalid sampling interval")
    return np.unique(np.linspace(lo, hi, num=min(count, hi - lo + 1), dtype=int))


def build_samples(
    trace: Trace, window: int, low: float, high: float, count: int
) -> list[RULSample]:
    return [make_sample(trace, int(i), window) for i in sample_indices(trace.length, low, high, count)]


class SequenceRULDataset(Dataset):
    def __init__(self, samples: list[RULSample], target_mode: str = "fraction"):
        if not samples:
            raise ValueError("dataset cannot be empty")
        self.samples = samples
        self.target_mode = target_mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int):
        sample = self.samples[item]
        if self.target_mode == "log_pulses":
            target = np.log1p(sample.target_pulses) / np.log1p(1_000_000.0)
        elif self.target_mode == "fraction":
            target = sample.target_fraction
        else:
            raise ValueError(f"unknown target mode: {self.target_mode}")
        return (
            torch.from_numpy(sample.sequence),
            torch.tensor(target, dtype=torch.float32),
        )


def stack_tabular(samples: list[RULSample]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.vstack([sample.tabular for sample in samples]),
        np.asarray([sample.target_fraction for sample in samples], dtype=np.float64),
    )


def build_time_warp_augmented_samples(
    trace: Trace,
    window: int,
    factors: tuple[float, ...],
    samples_per_factor: int,
    low: float = 0.05,
    high: float = 0.75,
) -> list[RULSample]:
    """Create source-only lifetime-scale tasks by causal trajectory resampling."""
    original_grid = np.linspace(0.0, 1.0, trace.length)
    samples: list[RULSample] = []
    for factor in factors:
        length = max(window + 3, int(round(trace.length * factor)))
        warped_grid = np.linspace(0.0, 1.0, length)
        voltage = np.interp(warped_grid, original_grid, trace.voltage)
        warped = Trace(
            trace_id=f"{trace.trace_id}_warp_{factor:g}",
            count=np.arange(length, dtype=np.float64),
            voltage=voltage,
            source_path=trace.source_path,
        )
        samples.extend(build_samples(warped, window, low, high, samples_per_factor))
    return samples
