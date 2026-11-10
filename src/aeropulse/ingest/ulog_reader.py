"""Read selected uORB topics from a ULog file into pandas DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog


def instance_name(topic: str, multi_id: int) -> str:
    """Key for one topic instance, e.g. ``actuator_outputs_1``.

    The ``{topic}_{multi_id}`` form matches pyulog's own ulog2csv naming. It is
    always applied, even to single-instance topics, so a key means the same
    thing in every log, whether or not a second instance happens to exist.
    """
    return f"{topic}_{multi_id}"


def load_topics(path: Path, wanted: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load the wanted topics (all topics if None) from one .ulg file.

    Returns one DataFrame per (topic, multi_id) instance, keyed by
    ``instance_name``. Each has an int64 ``timestamp`` column in microseconds
    since boot, sorted ascending, plus the topic's fields as logged.

    Decisions:
    - Missing topics are skipped, not raised. Topic sets differ across PX4
      versions and vehicle setups (e.g. GPS topics exist in 4 of the 10 sample
      logs), so the caller decides whether a missing topic is fatal.
    - ``timestamp`` is the time key. ``timestamp_sample`` (when the sensor
      measured, not when the message was published) would be more accurate,
      but only 1 of 38 topics in the sample logs (PX4 v1.8) has it. It is
      kept as an ordinary column where present.
    - Rows are stably sorted by timestamp in case the log is out of order.
      Values are not otherwise modified; cleaning happens in a later layer.
    """
    ulog = ULog(str(path), message_name_filter_list=wanted)
    out: dict[str, pd.DataFrame] = {}
    for d in ulog.data_list:
        df = pd.DataFrame(d.data)
        df["timestamp"] = df["timestamp"].astype(np.int64)
        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values("timestamp", kind="stable", ignore_index=True)
        out[instance_name(d.name, d.multi_id)] = df
    return out
