"""RflyMAD flights (real and HIL) as phase 6 frame tables.

Each flight's ULog is turned into the 50 Hz MAVLink-equivalent stream
(stream.ulog_adapter) and then into frames and features with the unchanged
phase 6 code, so every model in phase 7 sees the same representation.

Labels come from RflyMAD's own fault-injection topic `rfly_ctrl_lxl`
(checked on real flights, 2026-11-09):
    id          1500 (real) or 0 (HIL) = no fault; 123450 = motor fault active
    mode        1 = step (the multiplier applies at once), 2 = ramp (it
                declines linearly, e.g. 1.0 -> 0.9 over ~60-100 s)
    controls[0..3]  per-motor thrust multiplier while the fault is active
                (e.g. 0.85 = the motor gives 85% of its commanded thrust)
The fault starts when `id` becomes 123450 and ends when it changes again.
Both times are on the autopilot clock, like every other topic, so labels
are exact. Severity = 1 - the lowest multiplier reached (0.15 for 85%).
Each frame also gets `loss`, the thrust loss in effect at that moment: at
the start of a ramp it is ~0, so for ramps the useful question is how much
loss had built up when the alarm fired, not latency from the start.

Motor PWM range comes from each log's PWM_MAIN_MIN/MAX parameters.

Usage:
    uv run python -m aeropulse.sim2real.rflymad --raw data/raw/rflymad --set real \
        --out data/stream/rflymad_real_frames.parquet
"""

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from pyulog import ULog

from aeropulse.stream.dataset import AIRBORNE_STATES
from aeropulse.stream.features import (
    FEATURES,
    IDX,
    RATE_HZ,
    STALE_US,
    WINDOW,
    MotorScale,
    all_window_features,
    frames_from_messages,
)
from aeropulse.stream.mavlink_io import read_tlog
from aeropulse.stream.ulog_adapter import ulog_messages

MOTOR_FAULT_ID = 123450
MODES = ("hover", "waypoint", "velocity", "circling", "acce", "dece")
NO_FAULT_IDS = {0, 1500}    # real flights use 1500, HIL flights 0


def recording_day(ulog_path: Path) -> str:
    """Date from RflyMAD's log name, e.g. log_28_2023-6-2-10-16-50.ulg -> 2023-6-2."""
    m = re.search(r"log_\d+_(\d{4}-\d{1,2}-\d{1,2})-", ulog_path.name)
    return m.group(1) if m else "unknown"


def fault_info(ulog_path: Path) -> dict:
    u = ULog(str(ulog_path), message_name_filter_list=["rfly_ctrl_lxl"])
    p = u.initial_parameters
    info = {"pwm_min": float(p.get("PWM_MAIN_MIN", 1000)), "pwm_max": float(p.get("PWM_MAIN_MAX", 2000)),
            "airframe": int(p.get("SYS_AUTOSTART", 0)), "t_on": None, "t_off": None,
            "severity": 0.0, "n_failed": 0, "fault_ids": [], "fault_mode": None,
            "loss_t": np.empty(0, np.int64), "loss": np.empty(0)}
    try:
        d = u.get_dataset("rfly_ctrl_lxl").data
    except (KeyError, IndexError, ValueError):
        return info
    ids = d["id"].astype(np.int64)
    info["fault_ids"] = sorted(set(ids.tolist()) - NO_FAULT_IDS)
    on = np.flatnonzero(ids == MOTOR_FAULT_ID)
    if len(on):
        i0 = on[0]
        off = np.flatnonzero((ids != MOTOR_FAULT_ID) & (np.arange(len(ids)) > i0))
        info["t_on"] = int(d["timestamp"][i0])
        info["t_off"] = int(d["timestamp"][off[0]]) if len(off) else None
        # Rows whose multipliers are all 0 are a reset artifact (seen at the end
        # of HIL logs with `id` still 123450), not a 100% thrust loss.
        all_mult = np.stack([d[f"controls[{k}]"] for k in range(4)], axis=1)
        active = (ids == MOTOR_FAULT_ID) & (all_mult > 0).any(axis=1)
        mult = all_mult[active]
        info["loss_t"] = d["timestamp"][active].astype(np.int64)
        info["loss"] = 1 - mult.min(axis=1)
        info["severity"] = float(info["loss"].max())
        info["n_failed"] = int((mult.min(axis=0) < 0.999).sum())
        info["fault_mode"] = {1: "step", 2: "ramp"}.get(int(d["mode"][i0]), str(int(d["mode"][i0])))
    return info


def flight_table(ulog_path: Path, case: str, source: str, tlog_path: Path | None = None,
                 window: int = WINDOW, rate_hz: float = RATE_HZ, stale_us: int = STALE_US) -> pd.DataFrame:
    """Frames and labels for one flight. By default the stream is rebuilt at
    50 Hz from the ULog; with `tlog_path` the recorded telemetry-radio stream
    is used instead (labels still come from the ULog: same autopilot clock,
    checked at lag 0 on real flights)."""
    info = fault_info(ulog_path)
    ms = MotorScale(info["pwm_min"], info["pwm_max"], 4)
    messages = read_tlog(tlog_path) if tlog_path else ulog_messages(ulog_path, n_motors=4)
    t_us, frames, _ = frames_from_messages(messages, ms, stale_us)
    df = pd.DataFrame(all_window_features(frames, window, rate_hz), columns=FEATURES)
    df.insert(0, "key", case)
    df.insert(1, "t_us", t_us)
    df["landed_state"] = frames[:, IDX["landed_state"]]
    df["airborne"] = df["landed_state"].isin(AIRBORNE_STATES).to_numpy()
    t_on, t_off = info["t_on"], info["t_off"] or (int(t_us[-1]) + 1 if len(t_us) else None)
    y = (t_us >= t_on) & (t_us < t_off) if t_on is not None else np.zeros(len(df), bool)
    df["y"] = y
    loss = np.zeros(len(df))
    if len(info["loss_t"]):
        j = np.searchsorted(info["loss_t"], t_us, side="right") - 1
        loss = np.where((j >= 0) & y, info["loss"][np.clip(j, 0, None)], 0.0)
    df["loss"] = loss
    df["fault_mode"] = info["fault_mode"]
    df["use"] = df["airborne"].to_numpy() & np.isfinite(df[FEATURES].to_numpy()).any(axis=1)
    df["t_inject_us"] = t_on if t_on is not None else pd.NA
    df["t_fault_end_us"] = t_off if t_on is not None else pd.NA
    df["t_impact_us"] = pd.NA            # partial faults: the vehicle keeps flying
    df["severity"] = info["severity"]
    df["n_failed"] = info["n_failed"]
    df["source"] = source
    df["flight_mode"] = next((m for m in MODES if f"/{m}" in f"/{case}"), "other")
    # Grouping for splits. Real: the recording day, as for UAV-SEAD (phase 4):
    # flights of one day share vehicle, battery, weather and site, and 31 of
    # the 51 no-fault flights come from two days with almost no fault flights,
    # so any finer split could learn the day instead of the fault. HIL: every
    # "TestCase_<n>_<id>" is an independent simulated run.
    df["group"] = recording_day(ulog_path) if source == "real" else case.split("/")[-1]
    df["flight_class"] = "Mechanical" if t_on is not None else "Normal"
    df["other_fault"] = bool(set(info["fault_ids"]) - {MOTOR_FAULT_ID})
    return df


def _one(args):
    path, case, source = args
    try:
        return flight_table(path, case, source), None
    except Exception as e:  # noqa: BLE001 - report and continue
        return None, f"{case}: {type(e).__name__}: {e}"


def build(raw: Path, set_name: str, out: Path, workers: int = 4) -> pd.DataFrame:
    cases = json.loads((raw / f"cases_{set_name}.json").read_text())
    jobs = [(raw / c["ulg"], case, set_name) for case, c in sorted(cases.items()) if "ulg" in c]
    parts, errors = [], []
    with ProcessPoolExecutor(workers) as pool:
        for fut in as_completed([pool.submit(_one, j) for j in jobs]):
            df, err = fut.result()
            if err:
                errors.append(err)
            elif df is not None and len(df):
                parts.append(df)
    # Workers finish in random order; sort so the table (and every model fit
    # on it, e.g. early stopping's validation split) is reproducible.
    table = pd.concat(parts, ignore_index=True).sort_values(["key", "t_us"], kind="stable")
    table = table[~table["other_fault"]].reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    for e in errors:
        print("  error", e)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/rflymad"))
    ap.add_argument("--set", choices=["real", "hil"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    t = build(args.raw, args.set, args.out)
    fl = t.groupby("key").agg(cls=("flight_class", "first"), sev=("severity", "first"),
                              mode=("flight_mode", "first"), pos=("y", "sum"), used=("use", "sum"))
    print(fl.groupby(["cls", "mode"]).size().to_string())
    print(fl[fl["cls"] == "Mechanical"]["sev"].round(2).value_counts().sort_index().to_string())
    print(f"{t['key'].nunique()} flights, {len(t)} frames, {int(t['y'].sum())} positive")


if __name__ == "__main__":
    main()
