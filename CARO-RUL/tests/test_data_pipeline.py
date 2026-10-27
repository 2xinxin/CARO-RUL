import numpy as np
import pandas as pd

from pcss_rich_benchmark.data import Record, fit_transform, make_sample, transformed_record


FEATURES = [f"f{i}" for i in range(14)]


def test_source_transform_and_sample_shapes():
    frame = pd.DataFrame(np.arange(1400, dtype=float).reshape(100, 14), columns=FEATURES)
    frame.insert(0, "event_index", np.arange(1, 101))
    frame.loc[3, "f2"] = np.nan
    record = Record("SWX", frame)
    transform = fit_transform([record], FEATURES, 0.75)
    values, missing = transformed_record(record, transform)
    sample = make_sample(record, (values, missing), index=50, window=32)
    assert sample.sequence.shape == (43, 32)
    assert sample.tabular.shape == (71,)
    assert missing[3, 2] == 1.0
    assert 0.0 < sample.target_fraction < 1.0
