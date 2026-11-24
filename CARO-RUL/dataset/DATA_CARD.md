# Dataset card

## Scope

The SiC-PCSS Cross-Device RUL Benchmark supports remaining useful life prediction
for laboratory-fabricated silicon-carbide photoconductive semiconductor switches.
It contains seven physical-device lifetime records and 69,355 observed switching
events. Every experiment ended in destructive breakdown. Voltage was measured
across the load resistor.

## Observations, targets, and independence

- Observation: one acquired switching event.
- Independent unit: one complete device lifetime record.
- Primary target: the number of later observed events before the terminal event.
- Canonical order: one-based `event_index`.
- Measurement unit: volts.

The target is expressed in observed acquisition events because no synchronized
hardware pulse counter is available. Segment folder names are provenance, not a
substitute for the observed event index.

## Provenance and admission

All seven benchmark records are raw-archive-backed. Six have complete waveform
coverage. SW5 has 10,101 ordered peak observations, of which 7,768 retain the
original waveform; the remaining 2,333 are recovered from acquisition-side peak
files and are explicitly marked. The devices share the laboratory wafer context,
but complete machine-readable operating metadata are unavailable for some dated
experiments.

The obsolete peak-only records formerly called SW1 and SW4 are excluded. They
lacked original waveform archives and are not silently relabeled as benchmark
devices. `id_mapping.csv` records the exact mapping from acquisition-era names to
publication identifiers.

## Recommended use

The primary protocol is leave-one-device-out evaluation over SW1--SW7. Windows
from a held-out device must remain target-only. Source-only preprocessing must be
refitted inside every fold. Random row-level train/test splits, target-dependent
normalization, and endpoint-informed model selection are prohibited.

## Limitations

- Seven devices remain a small independent sample.
- Complete operating-condition metadata are unavailable for some dated records.
- SW5 contains 2,333 events without recoverable full waveforms.
- Consecutive equal peaks are retained because they represent separate switching
  events under oscilloscope quantization; they are not assumed to be duplicates.
- The current license does not authorize public redistribution.
