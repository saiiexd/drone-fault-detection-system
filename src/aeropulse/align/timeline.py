"""Put topics logged at different rates onto one regular, causal timeline.

Why not just join on ``timestamp``: each topic is published on its own clock
tick, so timestamps from different topics almost never match exactly. An
exact join returns nearly nothing, and an outer join returns mostly NaNs.

Why causal (as-of backward) and not linear interpolation: interpolating at
time t uses the next sample after t, i.e. future data. A live detector cannot
see the future, so features built that way would make offline results look
better than they could ever be online. (The UAV-SEAD annotation tool uses
``np.interp``; that is fine for plotting, not for features.)

Why a staleness limit: forward-filling across a dropout would repeat an old
value as if it were current. Past ``max_age_s`` the value becomes NaN, so a
missing signal stays visible as missing.
"""

import numpy as np
import pandas as pd


def align_topics(
    topics: dict[str, pd.DataFrame],
    rate_hz: float,
    max_age_s: float | dict[str, float],
    span_topics: list[str] | None = None,
) -> pd.DataFrame:
    """Align several topics onto one regular time grid.

    Input: name -> DataFrame with an int64 ``timestamp`` column (microseconds,
    sorted ascending) and numeric value columns.

    Output contract (enforced by tests/test_align.py):
    - Index named ``t_us`` (int64), a regular grid with step round(1e6 / rate_hz).
    - The grid starts at the latest first-timestamp and ends at or before the
      earliest last-timestamp across the span topics (default: all topics),
      i.e. the span where all of them exist. Other topics are joined onto that
      grid and are NaN where they have no data, so an optional topic that
      starts late does not shorten the flight.
    - One column per input value column, named f"{topic}.{column}".
    - Causal: the value at grid time t comes from the most recent sample with
      timestamp <= t. Never from a later sample.
    - Stale values become NaN: if that most recent sample is older than the
      topic's max age (a single float for all topics, or a per-topic dict),
      the value is NaN.
    - Inputs must not be modified.
    """
    if not topics:
        raise ValueError("no topics to align")
    span = list(topics) if span_topics is None else span_topics
    missing = [s for s in span if s not in topics]
    if missing:
        raise ValueError(f"span topics not provided: {missing}")
    step_us = round(1e6 / rate_hz)

    start = max(int(topics[s]["timestamp"].iloc[0]) for s in span)
    end = min(int(topics[s]["timestamp"].iloc[-1]) for s in span)
    if end < start:
        raise ValueError("topics do not overlap in time")
    grid = pd.DataFrame({"t_us": np.arange(start, end + 1, step_us, dtype=np.int64)})

    for name, df in topics.items():
        age_s = max_age_s[name] if isinstance(max_age_s, dict) else max_age_s
        values = df.rename(
            columns={c: f"{name}.{c}" for c in df.columns if c != "timestamp"}
        )
        grid = pd.merge_asof(
            grid,
            values,
            left_on="t_us",
            right_on="timestamp",
            direction="backward",           # only samples at or before t: causal
            tolerance=round(age_s * 1e6),   # older than this -> NaN: staleness
        ).drop(columns="timestamp")

    return grid.set_index("t_us")
