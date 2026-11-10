"""Phase 7 E: does motor-fault detection survive a real telemetry radio link?

Every phase 7 result so far used the 50 Hz MAVLink stream rebuilt from the
onboard ULog, i.e. what a companion computer wired to the autopilot would
see. A ground station on a telemetry radio sees much less. RflyMAD recorded
that radio stream (QGroundControl .tlog) on every real flight:

    ATTITUDE, LOCAL_POSITION_NED   15 Hz
    ATTITUDE_TARGET                 2 Hz   (thrust demand, attitude setpoint)
    SERVO_OUTPUT_RAW                1 Hz   (motor outputs)
    ATTITUDE_QUATERNION, HIGHRES_IMU  not sent

Same 111 real flights, same labels (ULog fault topic; tlog and ULog share the
autopilot clock, checked at lag 0), same model recipe and day-grouped CV with
nested thresholds as experiment B's real->real ceiling. Only the input
stream changes:

    R50      50 Hz rebuilt from the ULog, 0.5 s window   (= B ceiling)
    R15_0.5  recorded radio stream, 0.5 s window (8 frames)
    R15_2    recorded radio stream, 2 s window (30 frames)

On the radio stream a held value stays valid for 1.2 s (stale_us), so the
1 Hz motor outputs are not NaN between messages; it also means they can be
up to a second old. Persistence stays 3 frames (0.2 s at 15 Hz vs 0.06 s at 50 Hz).

Second part (--part sitl): the catastrophic case. The phase 6 SITL recordings
(50 Hz) are thinned to the same radio rates (mavlink_io.decimate) and the
phase 6 protocol is re-run (flight-grouped CV, alarm before impact). Impact
times come from the full-rate stream, so thinning cannot move them.

Usage:
    uv run python -m aeropulse.sim2real.link_rate --out reports/phase7 --part real
    uv run python -m aeropulse.sim2real.link_rate --out reports/phase7 --part sitl
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from aeropulse.sim2real.rflymad import flight_table
from aeropulse.sim2real.run import ap, ap_boot, cv_with_inner_threshold, events, frame_s, prepare

RADIO_HZ = 15
RADIO_STALE_US = 1_200_000
VARIANTS = {"R15_0.5": 8, "R15_2": 30}     # window in frames at 15 Hz


def _one(args):
    raw, case, c, window = args
    return flight_table(raw / c["ulg"], case, "real", tlog_path=raw / c["tlog"],
                        window=window, rate_hz=RADIO_HZ, stale_us=RADIO_STALE_US)


def build(raw: Path, window: int, workers: int = 4) -> pd.DataFrame:
    cases = json.loads((raw / "cases_real.json").read_text())
    jobs = [(raw, case, c, window) for case, c in sorted(cases.items()) if "ulg" in c and "tlog" in c]
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_one, jobs))
    t = pd.concat(parts, ignore_index=True).sort_values(["key", "t_us"], kind="stable")
    return t[~t["other_fault"]].reset_index(drop=True)


def availability(df: pd.DataFrame) -> dict:
    u = df["use"]
    return {f"has_{c.split('__')[0]}": float(df.loc[u, c].notna().mean())
            for c in ["tilt_err__last", "thrust_sp__last", "m_max__last", "vz__last", "acc_z__mean"]}


def evaluate(name: str, df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    df = prepare(df)
    s, a = cv_with_inner_threshold(df)
    ev = events(df, a).assign(experiment=name)
    f = ev[ev["fault"]]
    lo, hi = ap_boot(df, s)
    hours = ev["neg_s"].sum() / 3600
    by = f.groupby("fault_mode").agg(recall=("detected", "mean"), lat=("latency_s", "median"),
                                     loss=("loss_at_alarm", "median"))
    row = {"variant": name, "flights": int(df["key"].nunique()),
           "frame_hz": round(1 / frame_s(df), 1),
           "ap": ap(df, s), "ap_lo": lo, "ap_hi": hi, "chance_ap": float(df.loc[df["use"], "y"].mean()),
           "event_recall": float(f["detected"].mean()),
           "false_alarms_per_h": float(ev["false_alarms"].sum() / hours), "neg_hours": float(hours),
           "step_recall": float(by.loc["step", "recall"]), "step_latency_s": float(by.loc["step", "lat"]),
           "ramp_recall": float(by.loc["ramp", "recall"]), "ramp_loss_at_alarm": float(by.loc["ramp", "loss"]),
           **availability(df)}
    return row, ev


def sitl_radio(raw: Path, frames50: Path, out: Path) -> pd.DataFrame:
    from aeropulse.stream.dataset import flight_frame_table
    from aeropulse.stream.evaluate import flight_results, summary
    from aeropulse.stream.train import cross_validate

    full = pd.read_parquet(frames50, columns=["key", "flight_class", "t_inject_us", "t_impact_us"])
    meta = full.groupby("key").first()
    parts = []
    for key, m in meta.iterrows():
        t_inj = None if pd.isna(m["t_inject_us"]) else int(m["t_inject_us"])
        d = flight_frame_table(raw / key / "telemetry.tlog", key, t_inj, radio=True)
        if t_inj is not None:   # exact impact from the 50 Hz stream
            t_imp = int(m["t_impact_us"])
            after = d["t_us"].to_numpy() >= t_imp
            d["t_impact_us"] = t_imp
            d["y"] = (d["t_us"].to_numpy() >= t_inj) & ~after
            d["use"] = d["use"].to_numpy() & ~after
        d["flight_class"] = m["flight_class"]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    scores, alarm, _ = cross_validate(df, log=lambda *_: None)
    rows = [{"stream": "radio 15/2/1 Hz", "detector": k, **summary(df, scores[k], alarm[k])} for k in scores]
    ref = pd.read_csv(out.parent / "phase6" / "cv_detectors.csv").assign(stream="50 Hz (phase 6)")
    table = pd.concat([ref, pd.DataFrame(rows)], ignore_index=True)
    table.to_csv(out / "link_rate_sitl.csv", index=False)
    fr = flight_results(df, alarm["gbt_fast"])
    fr[fr["fault"]].to_csv(out / "link_rate_sitl_flights.csv", index=False)
    return table


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--raw", type=Path, default=Path("data/raw/rflymad"))
    ap_.add_argument("--real50", type=Path, default=Path("data/stream/rflymad_real_frames.parquet"))
    ap_.add_argument("--sitl-raw", type=Path, default=Path("data/raw/sitl6"))
    ap_.add_argument("--sitl-frames", type=Path, default=Path("data/stream/sitl6_frames.parquet"))
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--part", choices=["real", "sitl"], default="real")
    args = ap_.parse_args()
    if args.part == "sitl":
        t = sitl_radio(args.sitl_raw, args.sitl_frames, args.out)
        pd.set_option("display.width", 250)
        cols = ["stream", "detector", "before_impact", "fault_flights", "latency_median_s",
                "latency_max_s", "margin_median_s", "false_alarms", "airborne_neg_h", "frame_ap"]
        print(t[cols].round(3).to_string(index=False))
        return
    rows, evs = [], []
    r, ev = evaluate("R50", pd.read_parquet(args.real50))
    rows.append(r)
    evs.append(ev)
    for name, window in VARIANTS.items():
        df = build(args.raw, window)
        print(f"{name}: {df['key'].nunique()} flights, {len(df)} frames", flush=True)
        r, ev = evaluate(name, df)
        rows.append(r)
        evs.append(ev)
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "link_rate.csv", index=False)
    pd.concat(evs).to_csv(args.out / "link_rate_events.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(3).T.to_string())


if __name__ == "__main__":
    main()
