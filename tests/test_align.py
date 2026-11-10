"""Specification for align_topics. Synthetic data only, so no logs needed."""

import numpy as np
import pandas as pd
import pytest

from aeropulse.align.timeline import align_topics

STEP_US = 20_000  # 50 Hz grid


def make_topic(timestamps, col):
    ts = np.asarray(timestamps, dtype="int64")
    return pd.DataFrame({"timestamp": ts, col: np.arange(len(ts), dtype="float64")})


@pytest.fixture
def fast_slow():
    fast = make_topic(range(0, 1_000_001, 4_000), "x")     # 250 Hz, like an IMU
    slow = make_topic(range(0, 1_000_001, 200_000), "y")   # 5 Hz, like GPS
    return {"a": fast, "b": slow}


def test_regular_grid(fast_slow):
    out = align_topics(fast_slow, rate_hz=50, max_age_s=1.0)
    assert out.index.name == "t_us"
    assert out.index[0] == 0
    assert out.index[-1] == 1_000_000
    assert (np.diff(out.index.to_numpy()) == STEP_US).all()


def test_grid_covers_only_common_span():
    a = make_topic(range(0, 1_000_001, 4_000), "x")
    b = make_topic(range(300_000, 800_001, 100_000), "y")
    out = align_topics({"a": a, "b": b}, rate_hz=50, max_age_s=1.0)
    assert out.index[0] == 300_000
    assert out.index[-1] <= 800_000


def test_column_names(fast_slow):
    out = align_topics(fast_slow, rate_hz=50, max_age_s=1.0)
    assert sorted(out.columns) == ["a.x", "b.y"]


def test_causal_never_uses_future_samples(fast_slow):
    out = align_topics(fast_slow, rate_hz=50, max_age_s=1.0)
    # b has samples at 0 (y=0) and 200_000 (y=1).
    assert out.loc[180_000, "b.y"] == 0   # the sample at 200_000 is in the future here
    assert out.loc[200_000, "b.y"] == 1   # an exact timestamp match counts as "past"


def test_stale_values_become_nan():
    a = make_topic(range(0, 1_000_001, 4_000), "x")
    b = make_topic([0, 100_000, 900_000, 1_000_000], "y")   # gap: dropout
    out = align_topics({"a": a, "b": b}, rate_hz=50, max_age_s=0.25)
    assert out.loc[340_000, "b.y"] == 1         # age 0.24 s, still valid
    assert np.isnan(out.loc[360_000, "b.y"])    # age 0.26 s, stale
    assert out.loc[900_000, "b.y"] == 2         # fresh again after the gap


def test_per_topic_max_age():
    a = make_topic(range(0, 1_000_001, 4_000), "x")
    b = make_topic([0, 100_000, 900_000, 1_000_000], "y")
    out = align_topics({"a": a, "b": b}, rate_hz=50, max_age_s={"a": 0.02, "b": 1.0})
    assert out.loc[500_000, "b.y"] == 1      # b allowed to be 0.4 s old
    assert not out["a.x"].isna().any()       # a is fresh within 20 ms everywhere


def test_span_topics_define_grid_and_others_may_be_nan():
    a = make_topic(range(0, 1_000_001, 4_000), "x")
    late = make_topic(range(500_000, 700_001, 100_000), "z")   # optional, starts late
    out = align_topics({"a": a, "late": late}, rate_hz=50, max_age_s=0.15,
                       span_topics=["a"])
    assert out.index[0] == 0 and out.index[-1] == 1_000_000
    assert np.isnan(out.loc[400_000, "late.z"])   # before it exists
    assert out.loc[520_000, "late.z"] == 0
    assert np.isnan(out.loc[900_000, "late.z"])   # stale after it stops


def test_inputs_not_modified(fast_slow):
    before = {k: v.copy() for k, v in fast_slow.items()}
    align_topics(fast_slow, rate_hz=50, max_age_s=1.0)
    for k in fast_slow:
        pd.testing.assert_frame_equal(fast_slow[k], before[k])
