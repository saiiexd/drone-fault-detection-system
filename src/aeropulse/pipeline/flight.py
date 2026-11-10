"""Per-flight processing: ULog -> raw, cleaned and labelled layers + metadata.

Layers (all Parquet, hive-partitioned so DuckDB can prune by flight/topic):
- raw:     lake/raw/topic=<instance>/flight_id=<id>/part-0.parquet  (optional,
           --with-raw) every topic as logged, original field names.
- cleaned: lake/cleaned/flight_id=<id>/part-0.parquet
           canonical signals (aeropulse.signals) on a causal 50 Hz grid,
           derived quantities, and per-row labels.
- features: lake/features/flight_id=<id>/part-0.parquet
           2 s trailing-window features every 0.2 s (aeropulse.features).
"""

import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog

from aeropulse.align.timeline import align_topics
from aeropulse.features import batch_features
from aeropulse.ingest.ulog_reader import instance_name
from aeropulse.signals import MAX_MOTORS, SIGNALS

RATE_HZ = 50
VIB_WINDOW = "100ms"          # IMU vibration window, computed at native rate
AGE_PERIODS = 5               # a value is stale after 5 missed samples ...
AGE_MIN_S, AGE_MAX_S = 0.05, 5.0   # ... clamped to this range
PWM_FALLBACK = (1000.0, 2000.0)


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Atomic write: a crash never leaves a half-written file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)


# Motor count comes from the airframe type, NOT from actuator_outputs.noutputs:
# the quad airframe in UAV-SEAD (SYS_AUTOSTART 4014) logs noutputs=8, but only
# outputs 0-3 drive motors (4-5 are auxiliary, 6-7 constant).
MOTORS_BY_MAV_TYPE = {2: 4, 13: 6, 14: 8, 15: 3}   # quad, hexa, octo, tri
RANGE_TOL_US = 2 * 10**6
MAX_LOG_US = 24 * 3600 * 10**6   # timestamps beyond start + 24 h are corrupt
LEAD_GAP_WINDOW_US = 10 * 10**6  # gaps in the first 10 s are logger start-up
LEAD_GAP_MIN_US = 10**6


def ulog_metadata(ulog: ULog) -> dict:
    p = ulog.initial_parameters
    n_outputs = None
    try:
        nout = ulog.get_dataset("actuator_outputs", 0).data["noutputs"]
        n_outputs = int(Counter(nout.tolist()).most_common(1)[0][0])
    except (KeyError, IndexError, ValueError):
        pass
    mav_type = p.get("MAV_TYPE")
    return {
        "px4_version": ulog.get_version_info_str(),
        "ver_hw": ulog.msg_info_dict.get("ver_hw"),
        "mav_type": mav_type,
        "sys_autostart": p.get("SYS_AUTOSTART"),
        # PX4 v1.12 renamed these to PWM_MAIN_MIN/MAX (RflyMAD real flights).
        "pwm_min": p.get("PWM_MIN", p.get("PWM_MAIN_MIN")),
        "pwm_max": p.get("PWM_MAX", p.get("PWM_MAIN_MAX")),
        "n_outputs": n_outputs,
        "n_motors": MOTORS_BY_MAV_TYPE.get(int(mav_type)) if mav_type is not None else None,
        "start_us": int(ulog.start_timestamp),
        "header_end_us": int(ulog.last_timestamp),
        "n_dropouts": len(ulog.dropouts),
        "dropout_ms": float(sum(d.duration for d in ulog.dropouts)),
        "n_topics": len(ulog.data_list),
    }


def ulog_frames(ulog: ULog) -> tuple[dict[str, pd.DataFrame], int]:
    """All topics as DataFrames. Drops samples with corrupt timestamps.

    Some logs contain timestamps near 2**64 (corrupt). Cast to int64 they
    would wrap to negative numbers and sort to the front, so they are removed
    while still unsigned. Returns (frames, number of samples dropped).
    """
    start = ulog.start_timestamp
    out, dropped = {}, 0
    for d in ulog.data_list:
        ts = d.data["timestamp"]
        ok = (ts >= start) & (ts <= start + MAX_LOG_US)
        dropped += int((~ok).sum())
        df = pd.DataFrame({k: v[ok] for k, v in d.data.items()})
        df["timestamp"] = df["timestamp"].astype(np.int64)
        if not df["timestamp"].is_monotonic_increasing:
            df = df.sort_values("timestamp", kind="stable", ignore_index=True)
        if len(df):
            out[instance_name(d.name, d.multi_id)] = df
    return out, dropped


def leading_gap_end(frames: dict[str, pd.DataFrame], topics: list[str]) -> int | None:
    """End of the last >1 s gap within the first 10 s of the span topics.

    In ~15% of UAV-SEAD logs all topics pause together right after logging
    starts (logger start-up). Data before such a gap is trimmed rather than
    rejecting the whole flight; gaps later in the flight still count.
    """
    first = max(int(frames[t]["timestamp"].iloc[0]) for t in topics)
    cut = None
    for t in topics:
        ts = frames[t]["timestamp"].to_numpy()
        gaps = np.diff(ts)
        idx = np.flatnonzero((gaps > LEAD_GAP_MIN_US) & (ts[:-1] < first + LEAD_GAP_WINDOW_US))
        if len(idx):
            cut = max(cut or 0, int(ts[idx[-1] + 1]))
    return cut


def resolve_signals(frames: dict[str, pd.DataFrame]) -> dict[str, tuple[str, str, float]]:
    """Canonical name -> (topic instance, field, scale) actually used for this log."""
    chosen = {}
    for sig in SIGNALS:
        for cand in sig.candidates:
            topic, field, scale = (*cand, 1.0)[:3]
            if topic in frames and field in frames[topic].columns:
                chosen[sig.name] = (topic, field, float(scale))
                break
    return chosen


def add_imu_vibration(df: pd.DataFrame) -> pd.DataFrame:
    """Causal rolling std of |acc| and |gyro| over VIB_WINDOW at native rate.

    Taking one IMU sample per 20 ms grid step would throw away the
    high-frequency content that reveals propeller/motor vibration, so it is
    summarised here, before downsampling. Time-based rolling windows in pandas
    are right-closed: only past samples are used.
    """
    t = pd.to_timedelta(df["timestamp"], unit="us")
    acc = np.sqrt((df[["acc_x", "acc_y", "acc_z"]] ** 2).sum(axis=1))
    gyro = np.sqrt((df[["gyro_x", "gyro_y", "gyro_z"]] ** 2).sum(axis=1))
    df = df.copy()
    df["acc_vib"] = acc.set_axis(t).rolling(VIB_WINDOW).std().to_numpy()
    df["gyro_vib"] = gyro.set_axis(t).rolling(VIB_WINDOW).std().to_numpy()
    return df


# State topics may be published only when the state changes (e.g. landed /
# armed in PX4 v1.17: once at takeoff, once at landing). Their last value stays
# true until the next message, so they never go stale.
STATE_TOPICS = ("vehicle_land_detected_", "vehicle_status_")
STATE_MAX_AGE_S = 24 * 3600.0


def max_age(df: pd.DataFrame, topic: str = "") -> float:
    if topic.startswith(STATE_TOPICS):
        return STATE_MAX_AGE_S
    gaps = np.diff(df["timestamp"].to_numpy())
    period = float(np.median(gaps)) / 1e6 if len(gaps) else AGE_MAX_S
    return float(np.clip(AGE_PERIODS * period, AGE_MIN_S, AGE_MAX_S))


def quat_to_euler(w, x, y, z):
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    tilt = np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))   # body z vs vertical
    return roll, pitch, yaw, tilt


def label_rows(t_us: np.ndarray, ranges: list[tuple[str, str, int, int]]):
    """Per-row anomaly label and class from (class, signal, start, end) ranges."""
    classes = [set() for _ in range(len(t_us))]
    for c, _sig, a, b in ranges:
        for i in np.flatnonzero((t_us >= a) & (t_us <= b)):
            classes[i].add(c)
    anomaly = np.array([bool(s) for s in classes], dtype=bool)
    cls = np.array([" + ".join(sorted(s)) if s else "Normal" for s in classes], dtype=object)
    return anomaly, cls


def build_cleaned(frames, chosen, meta, ranges) -> pd.DataFrame:
    by_topic: dict[str, dict[str, str]] = {}
    for name, (topic, field, _scale) in chosen.items():
        by_topic.setdefault(topic, {})[field] = name
    topics = {}
    n_motors = min(meta["n_motors"] or meta["n_outputs"] or 4, MAX_MOTORS)
    for topic, fields in by_topic.items():
        df = frames[topic][["timestamp", *fields]].rename(columns=fields)
        for name in fields.values():
            if chosen[name][2] != 1.0:
                df[name] = df[name] * chosen[name][2]
        if topic.startswith("actuator_motors"):
            # PX4 marks a stopped/failed motor as NaN; physically it gives 0 thrust.
            active = [f"motor_{i}" for i in range(1, n_motors + 1) if f"motor_{i}" in df]
            df[active] = df[active].fillna(0.0)
        topics[topic] = df
    imu_topic = chosen["gyro_x"][0]
    topics[imu_topic] = add_imu_vibration(topics[imu_topic])

    required = {s.name for s in SIGNALS if s.required}
    span = sorted({chosen[n][0] for n in required})
    ages = {topic: max_age(df, topic) for topic, df in topics.items()}
    aligned = align_topics(topics, RATE_HZ, ages, span_topics=span)
    aligned.columns = [c.split(".", 1)[1] for c in aligned.columns]

    out = pd.DataFrame(index=aligned.index)
    for sig in SIGNALS:
        out[sig.name] = aligned[sig.name] if sig.name in aligned else np.nan
    out["acc_vib"], out["gyro_vib"] = aligned["acc_vib"], aligned["gyro_vib"]

    out["roll"], out["pitch"], out["yaw"], out["tilt"] = quat_to_euler(
        out["q_w"], out["q_x"], out["q_y"], out["q_z"])

    # Motor commands normalised to 0..1 with the vehicle's own PWM range, so
    # vehicles with different ESC calibrations are comparable.
    if chosen["motor_1"][0].startswith("actuator_motors"):
        lo, hi = 0.0, 1.0          # already normalised by PX4
    else:
        lo, hi = meta["pwm_min"] or PWM_FALLBACK[0], meta["pwm_max"] or PWM_FALLBACK[1]
    n = min(meta["n_motors"] or meta["n_outputs"] or 4, MAX_MOTORS)
    for i in range(1, MAX_MOTORS + 1):
        out[f"motor_{i}_n"] = (out[f"motor_{i}"] - lo) / (hi - lo) if i <= n else np.nan
        if i > n:
            out[f"motor_{i}"] = np.nan

    out["label_anomaly"], out["label_class"] = label_rows(out.index.to_numpy(), ranges)
    out = out.reset_index()
    out.insert(1, "t_s", (out["t_us"] - out["t_us"].iloc[0]) / 1e6)
    return out


def process_flight(entry: dict, raw_root: Path, lake: Path, write_raw: bool = False) -> dict:
    """Build all per-flight layers. Returns one catalog row (never raises)."""
    flight_id = entry["sha256"][:16]
    row = {"flight_id": flight_id, "key": entry["key"], "session": entry["key"].split("/")[0],
           "flight_class": entry["class"], "error": None}
    try:
        ulog = ULog(str(raw_root / entry["path"]))
        row |= ulog_metadata(ulog)
        frames, row["corrupt_ts_samples"] = ulog_frames(ulog)
        del ulog
        # True end of data; the header's last timestamp can be corrupt.
        row["end_us"] = max(int(df["timestamp"].iloc[-1]) for df in frames.values())
        row["duration_s"] = (row["end_us"] - row["start_us"]) / 1e6

        # Optional: the .ulg files are the raw source of truth and every build
        # re-parses them, so a Parquet copy of all topics (~half the log size)
        # only helps ad-hoc SQL on topics outside the canonical signal set.
        for topic, df in (frames.items() if write_raw else ()):
            write_parquet(df, lake / "raw" / f"topic={topic}" / f"flight_id={flight_id}"
                          / "part-0.parquet")

        chosen = resolve_signals(frames)
        missing_req = [s.name for s in SIGNALS if s.required and s.name not in chosen]
        row["missing_required"] = ",".join(missing_req) or None
        row["n_signals_found"] = len(chosen)
        row["signal_sources"] = ";".join(f"{k}={t}.{f}" + (f"*{s:g}" if s != 1 else "")
                                         for k, (t, f, s) in chosen.items())

        ranges = [(a["class"], sig, int(s), int(e))
                  for a in entry["annotations"] for sig, rr in a["ranges"] for s, e in rr]
        row["n_ranges"] = len(ranges)
        # Annotators often dragged a range slightly past the end of the log
        # (up to ~1 s). Allow RANGE_TOL_US either side; a range far outside the
        # log would indicate a clock mismatch and fails the check.
        row["ranges_in_span"] = all(
            row["start_us"] - RANGE_TOL_US <= s <= e <= row["end_us"] + RANGE_TOL_US
            for _, _, s, e in ranges)

        if missing_req:
            return row

        span_topics = sorted({chosen[s.name][0] for s in SIGNALS if s.required})
        cut = leading_gap_end(frames, span_topics)
        row["trimmed_start_s"] = 0.0
        if cut is not None:
            first = max(int(frames[t]["timestamp"].iloc[0]) for t in span_topics)
            row["trimmed_start_s"] = (cut - first) / 1e6
            frames = {k: df[df["timestamp"] >= cut].reset_index(drop=True)
                      for k, df in frames.items()}
            frames = {k: df for k, df in frames.items() if len(df)}
            chosen = {k: v for k, v in chosen.items() if v[0] in frames}
        gaps = [np.diff(frames[t]["timestamp"].to_numpy()).max(initial=0) for t in span_topics]
        row["max_gap_s"] = float(max(gaps, default=0)) / 1e6

        cleaned = build_cleaned(frames, chosen, row, ranges)
        req = [s.name for s in SIGNALS if s.required]
        row["n_rows"] = len(cleaned)
        row["nan_frac_required"] = float(cleaned[req].isna().any(axis=1).mean())
        row["anomaly_frac"] = float(cleaned["label_anomaly"].mean())
        row["airborne_frac"] = float((cleaned["landed"] == 0).mean())
        write_parquet(cleaned, lake / "cleaned" / f"flight_id={flight_id}" / "part-0.parquet")

        feats = batch_features(cleaned)
        row["n_windows"] = len(feats)
        write_parquet(feats, lake / "features" / f"flight_id={flight_id}" / "part-0.parquet")
    except Exception as e:  # noqa: BLE001 - one bad log must not stop the batch
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def remove_flight_outputs(lake: Path, flight_id: str) -> None:
    for layer in ("cleaned", "features"):
        shutil.rmtree(lake / layer / f"flight_id={flight_id}", ignore_errors=True)
    for d in (lake / "raw").glob(f"topic=*/flight_id={flight_id}"):
        shutil.rmtree(d, ignore_errors=True)
