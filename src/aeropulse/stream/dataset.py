"""Frame-level training data from recorded MAVLink streams (.tlog).

For each flight in a scenario manifest (phase 5/6 SITL format):
- frames and features come from the tlog, through the same FrameBuilder
  and window_features the live service uses;
- the fault start comes from the ULog (`vehicle_command`, exact). The tlog
  and the ULog share the autopilot boot clock (checked: roll angles agree at
  zero lag), so the label needs no clock conversion;
- impact is the first frame after injection at less than IMPACT_Z_M above
  the takeoff point. Frames after impact are kept for display but excluded
  from training and evaluation: after impact the question "did a motor fail?"
  is moot, and PX4's land detector often still reports "in air" (phase 5 §4).

Labels per frame:
    y        1 between injection and impact, else 0
    use      airborne (landed_state 2/3/4) and not after impact
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeropulse.stream.features import (
    FEATURES,
    IDX,
    MotorScale,
    all_window_features,
    frames_from_messages,
)
from aeropulse.stream.mavlink_io import decimate, read_tlog

IMPACT_Z_M = 0.3
RADIO_HZ = 15                # frame rate on a telemetry radio (ATTITUDE)
RADIO_WINDOW = 8             # 0.5 s at 15 Hz
RADIO_STALE_US = 1_200_000   # hold 1-2 Hz messages until the next one
AIRBORNE_STATES = (2, 3, 4)   # MAV_LANDED_STATE: in air, takeoff, landing


def impact_time_us(t_us: np.ndarray, z: np.ndarray, t_inj: int) -> int | None:
    after = (t_us >= t_inj) & (z > -IMPACT_Z_M)
    return int(t_us[np.argmax(after)]) if after.any() else None


def flight_frame_table(tlog: Path, key: str, t_inj: int | None,
                       motor_scale: MotorScale | None = None, radio: bool = False) -> pd.DataFrame:
    """radio=True thins the recording to a real telemetry radio's rates
    (mavlink_io.RADIO_RATES_HZ) and uses the radio window/staleness settings."""
    msgs = read_tlog(tlog)
    if radio:
        msgs = decimate(msgs)
        t_us, frames, rx_ns = frames_from_messages(msgs, motor_scale, RADIO_STALE_US)
        feats = all_window_features(frames, RADIO_WINDOW, RADIO_HZ)
    else:
        t_us, frames, rx_ns = frames_from_messages(msgs, motor_scale)
        feats = all_window_features(frames)
    df = pd.DataFrame(feats, columns=FEATURES)
    df.insert(0, "key", key)
    df.insert(1, "t_us", t_us)
    df.insert(2, "rx_ns", rx_ns)
    z = frames[:, IDX["z"]]
    df["z"] = z
    df["landed_state"] = frames[:, IDX["landed_state"]]
    if radio and t_inj is not None:   # impact from the full-rate recording, not 15 Hz samples
        t_full, f_full, _ = frames_from_messages(read_tlog(tlog), motor_scale)
        t_imp = impact_time_us(t_full, f_full[:, IDX["z"]], t_inj)
    else:
        t_imp = impact_time_us(t_us, z, t_inj) if t_inj is not None else None
    df["t_inject_us"] = t_inj if t_inj is not None else pd.NA
    df["t_impact_us"] = t_imp if t_imp is not None else pd.NA
    after_imp = t_us >= t_imp if t_imp is not None else np.zeros(len(df), bool)
    df["y"] = ((t_us >= t_inj) & ~after_imp) if t_inj is not None else False
    df["airborne"] = df["landed_state"].isin(AIRBORNE_STATES).to_numpy()
    df["use"] = df["airborne"].to_numpy() & ~after_imp & np.isfinite(feats).any(axis=1)
    return df


def build(raw: Path, out: Path, radio: bool = False) -> pd.DataFrame:
    manifest = json.loads((raw / "manifest.json").read_text())
    parts = []
    for f in manifest["files"]:
        if not f.get("tlog"):
            continue
        ranges = [r for a in f["annotations"] for _, rr in a["ranges"] for r in rr]
        t_inj = int(ranges[0][0]) if ranges else None
        df = flight_frame_table(raw / f["tlog"], f["key"], t_inj, radio=radio)
        df["flight_class"] = f["class"]
        parts.append(df)
    table = pd.concat(parts, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    return table


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--radio", action="store_true",
                    help="thin the recordings to telemetry-radio rates (15/2/1 Hz)")
    args = ap.parse_args()
    t = build(args.raw, args.out, args.radio)
    per = t.groupby("key").agg(cls=("flight_class", "first"), frames=("t_us", "size"),
                               used=("use", "sum"), pos=("y", "sum"))
    print(per.to_string())
    print(f"{len(t)} frames, {int(t['use'].sum())} used, {int(t['y'].sum())} positive")


if __name__ == "__main__":
    main()
