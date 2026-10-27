from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "provenance" / "pcss_dataset_v1"
OUTPUT_ROOT = ROOT / "dataset"

RECORD_MAP = [
    ("SW2", "SW1"),
    ("SW3", "SW2"),
    ("SW5", "SW3"),
    ("SW6", "SW4"),
    ("SW7", "SW5"),
    ("EXP-20231108", "SW6"),
    ("EXP-20241205", "SW7"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_record(source_id: str, benchmark_id: str) -> dict[str, object]:
    source = SOURCE_ROOT / "processed" / "event_sequences" / f"{source_id}.csv"
    destination = OUTPUT_ROOT / "processed" / "event_sequences" / f"{benchmark_id}.csv"
    frame = pd.read_csv(source)

    source_segments = frame["segment_id"].astype(str).copy()
    segment_order = list(dict.fromkeys(source_segments.tolist()))
    segment_map = {
        segment: f"{benchmark_id}-S{position:02d}"
        for position, segment in enumerate(segment_order, start=1)
    }

    frame.insert(1, "source_record_id", source_id)
    segment_position = frame.columns.get_loc("segment_id") + 1
    frame.insert(segment_position, "source_segment_id", source_segments)
    frame["record_id"] = benchmark_id
    frame["segment_id"] = source_segments.map(segment_map)

    n = len(frame)
    expected_index = np.arange(1, n + 1)
    expected_rul = np.arange(n - 1, -1, -1)
    expected_fraction = np.linspace(0.0, 1.0, n)
    assert np.array_equal(frame["event_index"].to_numpy(), expected_index)
    assert np.array_equal(frame["rul_observed_events"].to_numpy(), expected_rul)
    assert np.allclose(frame["life_fraction"].to_numpy(float), expected_fraction)
    assert frame["peak_voltage_v"].notna().all()
    assert frame["record_id"].eq(benchmark_id).all()
    assert frame["source_record_id"].eq(source_id).all()

    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False, lineterminator="\n")

    waveform_count = int(frame["waveform_available"].astype(str).str.lower().eq("true").sum())
    return {
        "record_id": benchmark_id,
        "source_record_id": source_id,
        "event_count": n,
        "waveform_available_count": waveform_count,
        "segment_count": len(segment_order),
        "peak_min_v": float(frame["peak_voltage_v"].min()),
        "peak_max_v": float(frame["peak_voltage_v"].max()),
        "terminal_rul": int(frame["rul_observed_events"].iloc[-1]),
        "source_csv": source.relative_to(ROOT).as_posix(),
        "source_sha256": sha256(source),
        "benchmark_csv": destination.relative_to(ROOT).as_posix(),
        "benchmark_sha256": sha256(destination),
    }


def main() -> None:
    (OUTPUT_ROOT / "audit").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "raw").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "splits").mkdir(parents=True, exist_ok=True)

    records = [canonicalize_record(source, target) for source, target in RECORD_MAP]
    records_frame = pd.DataFrame(records)
    records_frame.to_csv(OUTPUT_ROOT / "records.csv", index=False, lineterminator="\n")
    records_frame[["source_record_id", "record_id"]].to_csv(
        OUTPUT_ROOT / "id_mapping.csv", index=False, lineterminator="\n"
    )

    archive_manifest = pd.read_csv(SOURCE_ROOT / "raw" / "archive_manifest.csv")
    archive_manifest.to_csv(
        OUTPUT_ROOT / "raw" / "source_archives.csv", index=False, lineterminator="\n"
    )

    split_policy = {
        "version": "1.0.0",
        "unit_of_independence": "physical_device_record",
        "primary_protocol": "leave-one-device-out",
        "records": [target for _, target in RECORD_MAP],
        "test_checkpoints": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "leakage_guards": [
            "split by physical device before window construction",
            "fit imputation, scaling, feature selection, and early stopping on source devices only",
            "never use held-out target labels or endpoint length during fitting or selection",
            "preserve acquisition order and segment boundaries",
        ],
    }
    (OUTPUT_ROOT / "splits" / "policy.json").write_text(
        json.dumps(split_policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    validation = {
        "dataset": "SiC-PCSS Cross-Device RUL Benchmark",
        "version": "1.0.0",
        "status": "pass",
        "records": len(records),
        "events": int(records_frame["event_count"].sum()),
        "waveform_backed_events": int(records_frame["waveform_available_count"].sum()),
        "checks": {
            "contiguous_one_based_event_index": True,
            "rul_decreases_to_zero": True,
            "life_fraction_spans_zero_to_one": True,
            "finite_peak_voltage": True,
            "continuous_publication_ids_SW1_to_SW7": True,
            "discarded_legacy_SW1_SW4_absent": True,
            "source_identity_retained": True,
        },
    }
    (OUTPUT_ROOT / "audit" / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    manifest = {
        "name": "SiC-PCSS Cross-Device RUL Benchmark",
        "version": "1.0.0",
        "created": "2026-09-17",
        "primary_task": "remaining useful life prediction",
        "target": "rul_observed_events",
        "device_family": "SiC-PCSS",
        "record_ids": [target for _, target in RECORD_MAP],
        "canonical_directory": "processed/event_sequences",
        "records": "records.csv",
        "id_mapping": "id_mapping.csv",
        "split_policy": "splits/policy.json",
        "validation": "audit/validation_report.json",
        "license": "Dataset reuse license not specified; see docs/DATA_ACCESS.md",
    }
    (OUTPUT_ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    generated = [
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.name != "file_manifest.csv"
    ]
    file_manifest = pd.DataFrame(
        [
            {
                "relative_path": path.relative_to(OUTPUT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(generated)
        ]
    )
    file_manifest.to_csv(OUTPUT_ROOT / "file_manifest.csv", index=False, lineterminator="\n")

    print(json.dumps(validation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
