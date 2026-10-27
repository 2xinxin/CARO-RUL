# Data dictionary

Each CSV in `processed/event_sequences/` represents one physical device and is
ordered by `event_index`. All seven files share the same core descriptor columns; SW5 also includes a `sidecar_peak_delta_v` provenance column for acquisition-side peak matching.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `record_id` | string | -- | Continuous benchmark identifier, SW1--SW7. |
| `source_record_id` | string | -- | Acquisition-era identifier retained for provenance. |
| `event_index` | integer | events | One-based event order within the device lifetime. |
| `segment_id` | string | -- | Canonical segment identifier such as `SW4-S02`. |
| `source_segment_id` | string | -- | Original acquisition block or folder label. |
| `segment_event_index` | integer | events | Order recorded inside the acquisition segment. |
| `peak_voltage_v` | float | V | Waveform maximum or acquisition-side peak entry. |
| `rul_observed_events` | integer | events | Later observed events before terminal breakdown. |
| `life_fraction` | float | 0--1 | Descriptive normalized position in the complete record. |
| `peak_source` | category | -- | Source of the peak measurement. |
| `waveform_available` | boolean | -- | Whether the original waveform is recoverable. |
| `source_archive` | string | -- | Immutable top-level raw archive. |
| `source_member` | string | -- | Member path supporting the event. |
| `qc_status` | category | -- | Event-level provenance or availability status. |
| `is_segment_start` | boolean | -- | First event of an acquisition segment. |

Waveform-derived columns are blank when `waveform_available` is false. They
include waveform size and time bounds; minimum, maximum, peak-to-peak, mean,
standard deviation, and RMS voltage; baseline statistics; baseline-corrected
peak and trough; peak time; 10--90% rise time; FWHM; positive pulse area; and
source checksums. Repeated-peak fields describe quantization runs and must not be
interpreted as instructions to delete observations.
