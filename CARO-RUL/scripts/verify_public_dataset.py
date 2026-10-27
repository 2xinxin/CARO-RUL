"""Verify the released processed SiC-PCSS CSVs, their label convention and hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "dataset"
FEATURES = [
    "peak_voltage_v", "voltage_min_v", "voltage_peak_to_peak_v", "voltage_mean_v",
    "voltage_std_v", "voltage_rms_v", "baseline_mean_v", "baseline_std_v",
    "baseline_corrected_peak_v", "baseline_corrected_trough_v", "peak_time_s",
    "rise_time_10_90_s", "fwhm_s", "positive_pulse_area_v_s",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as inp:
        for chunk in iter(lambda: inp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check() -> None:
    manifest = json.loads((D / "dataset_manifest.json").read_text(encoding="utf-8"))
    expected = [f"SW{i}" for i in range(1, 8)]
    assert manifest["record_ids"] == expected, "Unexpected record list"
    checksums = pd.read_csv(D / "file_manifest.csv")
    for entry in checksums.itertuples(index=False):
        rel = Path(entry.relative_path)
        assert not rel.is_absolute() and ".." not in rel.parts
        f = D / rel
        assert f.is_file(), f"Missing dataset file: {rel}"
        assert f.stat().st_size == int(entry.bytes), f"Size mismatch: {rel}"
        assert sha256(f) == entry.sha256, f"SHA-256 mismatch: {rel}"
    records = pd.read_csv(D / "records.csv")
    assert list(records.record_id) == expected
    events = waveforms = segments = 0
    for r in records.itertuples(index=False):
        f = D / "processed" / "event_sequences" / f"{r.record_id}.csv"
        frame = pd.read_csv(f, low_memory=False)
        length = len(frame)
        assert length == int(r.event_count), f"Event count: {r.record_id}"
        assert set(FEATURES) <= set(frame.columns), f"Missing feature: {r.record_id}"
        assert frame.record_id.eq(r.record_id).all()
        assert frame.source_record_id.eq(r.source_record_id).all()
        assert np.array_equal(frame.event_index.to_numpy(), np.arange(1, length + 1)), r.record_id
        assert np.array_equal(frame.rul_observed_events.to_numpy(), np.arange(length - 1, -1, -1)), r.record_id
        assert frame.peak_voltage_v.notna().all(), r.record_id
        assert np.allclose(frame.life_fraction.to_numpy(), np.linspace(0, 1, length)), r.record_id
        wave = int(frame.waveform_available.astype(str).str.lower().eq("true").sum())
        seg = frame.segment_id.nunique()
        assert wave == int(r.waveform_available_count), f"Waveform count: {r.record_id}"
        assert seg == int(r.segment_count), f"Segment count: {r.record_id}"
        assert sha256(f) == r.benchmark_sha256, f"Record SHA-256: {r.record_id}"
        events += length
        waveforms += wave
        segments += seg
        print(f"PASS {r.record_id}: {length:,} observed events; {wave:,} waveform-backed; {seg} segments")
    assert events == 69_355 and waveforms == 67_022 and segments == 21
    assert events-waveforms == 2_333
    print(f"PASS dataset: 7 physical devices, {events:,} events, {waveforms:,} waveform-backed, {segments} segments")
    print(f"PASS SHA-256: {len(checksums)} listed dataset files match")


if __name__ == "__main__":
    check()
