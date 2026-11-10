"""Tests for evaluation logic: splits, persistence, events, false alarms."""

import numpy as np
import pandas as pd

from aeropulse.models.evaluate import (
    alarm_episodes,
    event_metrics,
    merge_ranges,
    persistence,
    session_folds,
)

HOP = 200_000


def windows(flights: dict[str, int]) -> pd.DataFrame:
    rows = [{"flight_id": f, "t_us": i * HOP} for f, n in flights.items() for i in range(n)]
    return pd.DataFrame(rows)


def test_session_folds_never_share_a_session():
    rng = np.random.default_rng(0)
    flights = pd.DataFrame({
        "flight_id": [f"f{i}" for i in range(60)],
        "session": [f"s{i // 3}" for i in range(60)],
        "flight_primary": rng.choice(["Normal", "Mechanical"], 60, p=[0.7, 0.3]),
    })
    df = flights.loc[flights.index.repeat(5)].reset_index(drop=True)
    seen_test = set()
    for tr, te in session_folds(df, n_splits=5):
        assert not set(df.loc[tr, "session"]) & set(df.loc[te, "session"])
        assert not set(df.loc[te, "flight_id"]) & seen_test
        seen_test |= set(df.loc[te, "flight_id"])
    assert seen_test == set(flights["flight_id"])   # every flight tested once


def test_persistence_needs_k_consecutive_in_same_flight():
    df = windows({"a": 6, "b": 3})
    raw = np.array([1, 1, 1, 0, 1, 1,   1, 1, 1], bool)
    out = persistence(df, raw, k=3)
    assert out.tolist() == [0, 0, 1, 0, 0, 0,   0, 0, 1]   # run in "a" does not carry into "b"


def test_persistence_breaks_on_time_gap():
    df = pd.DataFrame({"flight_id": ["a"] * 4, "t_us": [0, HOP, 10 * HOP, 11 * HOP]})
    assert persistence(df, np.ones(4, bool), k=3).tolist() == [False] * 4


def test_alarm_episodes_count_onsets():
    df = windows({"a": 8})
    pred = np.array([0, 1, 1, 0, 1, 0, 1, 1], bool)
    assert alarm_episodes(df, pred) == 3


def test_merge_ranges_and_event_detection_latency():
    ranges = pd.DataFrame({"flight_id": ["a", "a", "a"], "class": ["X", "Y", "X"],
                           "start_us": [1_000_000, 1_500_000, 5_000_000],
                           "end_us": [2_000_000, 2_500_000, 6_000_000]})
    events = merge_ranges(ranges)
    assert len(events) == 2 and events.loc[0, "event_class"] == "X + Y"
    df = windows({"a": 40})                         # 0 .. 7.8 s
    pred = np.zeros(40, bool)
    pred[10] = True                                 # alarm at 2.0 s
    ev = event_metrics(df, pred, events)
    assert ev["detected"].tolist() == [True, False]
    assert np.isclose(ev.loc[0, "latency_s"], 1.0)  # 2.0 s alarm - 1.0 s start
