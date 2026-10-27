from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RichSample:
    trace_id: str
    index: int
    sequence: np.ndarray
    tabular: np.ndarray
    target_fraction: float
    target_pulses: float
    lifetime: int


@dataclass
class Record:
    record_id: str
    frame: pd.DataFrame

    @property
    def length(self) -> int:
        return len(self.frame)


@dataclass
class FoldTransform:
    feature_columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_records(dataset_root: Path, record_ids: list[str]) -> dict[str, Record]:
    records = {}
    for record_id in record_ids:
        path = dataset_root / "processed" / "event_sequences" / f"{record_id}.csv"
        frame = pd.read_csv(path)
        if not np.array_equal(frame.event_index.to_numpy(), np.arange(1, len(frame) + 1)):
            raise ValueError(f"non-contiguous event index: {record_id}")
        records[record_id] = Record(record_id, frame)
    return records


def fit_transform(
    records: list[Record], feature_columns: list[str], max_fraction: float
) -> FoldTransform:
    pieces = []
    for record in records:
        end = max(2, int(np.floor(max_fraction * (record.length - 1))) + 1)
        pieces.append(record.frame.loc[: end - 1, feature_columns].to_numpy(float))
    matrix = np.vstack(pieces)
    median = np.nanmedian(matrix, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    filled = np.where(np.isfinite(matrix), matrix, median)
    mean = filled.mean(axis=0)
    std = filled.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return FoldTransform(feature_columns, median, mean, std)


def transformed_record(record: Record, transform: FoldTransform) -> tuple[np.ndarray, np.ndarray]:
    raw = record.frame[transform.feature_columns].to_numpy(float)
    missing = ~np.isfinite(raw)
    filled = np.where(missing, transform.median, raw)
    values = (filled - transform.mean) / transform.std
    return values.astype(np.float32), missing.astype(np.float32)


def _slope(block: np.ndarray) -> np.ndarray:
    if len(block) < 2:
        return np.zeros(block.shape[1], dtype=np.float32)
    x = np.linspace(-1.0, 1.0, len(block), dtype=np.float64)
    denom = float(np.sum(x * x))
    centered = block - block.mean(axis=0, keepdims=True)
    return ((x[:, None] * centered).sum(axis=0) / denom).astype(np.float32)


def make_sample(
    record: Record,
    transformed: tuple[np.ndarray, np.ndarray],
    index: int,
    window: int,
) -> RichSample:
    if index < 1 or index >= record.length - 1:
        raise ValueError("sample index must leave a future event")
    values, missing = transformed
    start = max(0, index - window + 1)
    value_block = values[start : index + 1]
    missing_block = missing[start : index + 1]
    if len(value_block) < window:
        pad = window - len(value_block)
        value_block = np.pad(value_block, ((pad, 0), (0, 0)), mode="edge")
        missing_block = np.pad(missing_block, ((pad, 0), (0, 0)), mode="edge")
    delta = np.diff(value_block, axis=0, prepend=value_block[:1])
    age_value = np.log1p(index) / np.log1p(100_000.0)
    age = np.full((window, 1), age_value, dtype=np.float32)
    sequence = np.concatenate([value_block, delta, missing_block, age], axis=1).T.astype(np.float32)
    tabular = np.concatenate(
        [
            value_block[-1],
            value_block.mean(axis=0),
            value_block.std(axis=0),
            _slope(value_block),
            missing_block.mean(axis=0),
            np.asarray([age_value], dtype=np.float32),
        ]
    ).astype(np.float32)
    lifetime = record.length - 1
    remaining = lifetime - index
    return RichSample(
        trace_id=record.record_id,
        index=index,
        sequence=sequence,
        tabular=tabular,
        target_fraction=float(remaining / lifetime),
        target_pulses=float(remaining),
        lifetime=lifetime,
    )


def sample_indices(length: int, low: float, high: float, count: int) -> np.ndarray:
    lo = max(1, int(np.floor(low * (length - 1))))
    hi = min(length - 2, int(np.floor(high * (length - 1))))
    return np.unique(np.linspace(lo, hi, min(count, hi - lo + 1), dtype=int))


def build_samples(
    record: Record,
    transformed: tuple[np.ndarray, np.ndarray],
    window: int,
    low: float,
    high: float,
    count: int,
) -> list[RichSample]:
    return [
        make_sample(record, transformed, int(index), window)
        for index in sample_indices(record.length, low, high, count)
    ]


def make_relative_sample(
    record: Record,
    transformed: tuple[np.ndarray, np.ndarray],
    index: int,
    window: int,
    reference_events: int = 64,
) -> RichSample:
    sample = make_sample(record, transformed, index, window)
    values, missing = transformed
    reference = values[: min(reference_events, index + 1)]
    center = np.median(reference, axis=0)
    q75, q25 = np.percentile(reference, [75, 25], axis=0)
    scale = np.maximum(q75 - q25, 0.10)
    relative = np.clip((values - center) / scale, -12.0, 12.0).astype(np.float32)
    start = max(0, index - window + 1)
    block = relative[start : index + 1]
    mask_block = missing[start : index + 1]
    if len(block) < window:
        pad = window - len(block)
        block = np.pad(block, ((pad, 0), (0, 0)), mode="edge")
        mask_block = np.pad(mask_block, ((pad, 0), (0, 0)), mode="edge")
    delta = np.diff(block, axis=0, prepend=block[:1])
    age = np.zeros((window, 1), dtype=np.float32)
    sequence = np.concatenate([block, delta, mask_block, age], axis=1).T.astype(np.float32)
    return RichSample(sample.trace_id, sample.index, sequence, sample.tabular, sample.target_fraction, sample.target_pulses, sample.lifetime)


def build_relative_samples(
    record: Record,
    transformed: tuple[np.ndarray, np.ndarray],
    window: int,
    low: float,
    high: float,
    count: int,
    reference_events: int = 64,
) -> list[RichSample]:
    return [make_relative_sample(record, transformed, int(index), window, reference_events) for index in sample_indices(record.length, low, high, count)]
