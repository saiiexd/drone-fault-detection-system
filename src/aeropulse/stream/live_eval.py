"""End-to-end live test: the real service attached to SITL, on held-out flights.

1. Start the detection service (its own process, bound to UDP 14550).
2. Fly the `phase6_live` scenarios (new seeds; includes CA_FAILURE_MODE=0
   failures the model never trained on). The scenario runner does not record
   telemetry itself here; the service is the ground station.
3. Stop the service and score its decision log against the ULogs.

Latency is split by clock, so no two clocks are ever compared:
- detection latency (autopilot boot clock): first alarm frame time minus the
  injection time, which PX4 logged in the ULog on the same clock;
- compute latency (host monotonic clock): message received -> decision made
  (Decision.proc_us), and per-frame inter-arrival on the host.
SITL runs on the same host over loopback UDP, so transport delay is not
measured here; on hardware it would be measured with MAVLink TIMESYNC.

Replay mode (--from-tlog): feed a recorded session (the service's own
session.tlog) through a model offline, e.g. the radio-rate model on the
held-out live flights, and score it the same way.

Usage:
    uv run python -m aeropulse.stream.live_eval --out reports/phase6 --data data/raw/sitl6_live
    uv run python -m aeropulse.stream.live_eval --out reports/phase6/radio --analyse-only \
        --from-tlog data/stream/live_eval/session.tlog --model data/models/stream_gbt_radio.joblib \
        --session-dir data/stream/live_eval_radio
    uv run python -m aeropulse.stream.live_eval --out reports/phase6 --data data/raw/sitl6_live --analyse-only
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aeropulse.sim.scenarios import injection_time_us
from aeropulse.stream.dataset import IMPACT_Z_M


def run_flights(data: Path, session_dir: Path, model: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    svc = subprocess.Popen([sys.executable, "-m", "aeropulse.stream.service", "--model", str(model),
                            "--session-dir", str(session_dir), "--port", "8000"],
                           stdout=open(session_dir / "service.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(5)
    try:
        subprocess.run([sys.executable, "-m", "aeropulse.sim.scenarios", "--out", str(data),
                        "--set", "phase6_live", "--no-tlog", "--skip-done"], check=True)
    finally:
        svc.terminate()
        svc.wait(timeout=20)


def replay_to_decisions(tlog: Path, model: Path, session_dir: Path) -> None:
    """Run a recorded session through a model and write decisions.jsonl as
    the service would (input thinned to radio rates for a radio model)."""
    from dataclasses import asdict

    from aeropulse.stream.detector import LiveDetector
    from aeropulse.stream.mavlink_io import decimate, read_tlog
    det = LiveDetector(model)
    src = read_tlog(tlog)
    if det.radio:
        src = decimate(src)
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / "decisions.jsonl", "w") as f:
        for msg, rx in src:
            d = det.update(msg, rx)
            if d is not None:
                f.write(json.dumps(asdict(d)) + "\n")


def load_decisions(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    disp = pd.json_normalize(df.pop("display"))
    return pd.concat([df, disp], axis=1)


def analyse(data: Path, session_dir: Path) -> tuple[pd.DataFrame, dict]:
    dec = load_decisions(session_dir / "decisions.jsonl")
    runs = json.loads((data / "runs.json").read_text())
    # A fault counts only if the log shows it happened (phase 5 fault_effect):
    # with CA_FAILURE_MODE=0, PX4 v1.17 + Gazebo accepts "motor off" but the
    # motor keeps running, because the stop is done by PX4's own allocator.
    effect = {f["key"]: f.get("fault_effect") or {}
              for f in json.loads((data / "manifest.json").read_text())["files"]}
    sessions = [g for _, g in dec.groupby("session", sort=True) if g["airborne"].any()]
    ok_runs = [r for r in runs if not r.get("error")]
    if len(sessions) != len(ok_runs):
        print(f"warning: {len(sessions)} airborne sessions vs {len(ok_runs)} successful runs")
    rows = []
    for run, s in zip(ok_runs, sessions, strict=False):   # mismatch is warned above
        ulg = sorted((data / run["scenario"] / "log").rglob("*.ulg"))[-1]
        t = s["t_us"].to_numpy()
        alarm = s["alarm"].to_numpy()
        onset = s["onset"].to_numpy()
        air = s["airborne"].to_numpy()
        fe = effect.get(run["scenario"], {})
        happened = bool(run["unit"]) and fe.get("motor_cmd_after", 1.0) < 0.1
        row = {"key": run["scenario"], "fault": happened, "injected": bool(run["unit"]),
               "motor_cmd_after": fe.get("motor_cmd_after"),
               "ca_failure_mode": run.get("ca_failure_mode", 1),
               "takeoff_alt_m": run.get("takeoff_alt_m"), "frames": len(s)}
        t_inj = injection_time_us(ulg) if happened else None
        if t_inj is not None:
            assert t[0] <= t_inj <= t[-1], f"{run['scenario']}: injection outside session"
            after = (t >= t_inj) & (s["alt"].to_numpy() < IMPACT_Z_M)
            t_imp = int(t[np.argmax(after)]) if after.any() else None
            pre = air & (t < t_inj)
            hit = np.flatnonzero(alarm & (t >= t_inj))
            row["time_to_impact_s"] = (t_imp - t_inj) / 1e6 if t_imp else np.nan
            row["latency_s"] = (t[hit[0]] - t_inj) / 1e6 if len(hit) else np.nan
            row["margin_s"] = (t_imp - t[hit[0]]) / 1e6 if len(hit) and t_imp else np.nan
            row["before_impact"] = bool(len(hit) and t_imp and t[hit[0]] < t_imp)
        else:
            pre = air
        row["false_alarms"] = int((onset & pre).sum())
        dt = np.diff(t)
        frame_s = float(dt[dt < 1_000_000].mean()) / 1e6     # 0.02 at 50 Hz, ~0.067 on a radio
        row["airborne_neg_s"] = pre.sum() * frame_s
        rows.append(row)
    table = pd.DataFrame(rows)
    proc = dec["proc_us"].to_numpy()
    rx = dec["rx_ns"].to_numpy()
    gaps = np.diff(rx)[np.diff(dec["session"].to_numpy()) == 0] / 1e6
    faults = table[table["fault"]]
    summ = {
        "flights": int(len(table)), "fault_flights": int(len(faults)),
        "before_impact": int(faults["before_impact"].sum()),
        "detected": int(faults["latency_s"].notna().sum()),
        "latency_median_s": float(faults["latency_s"].median()),
        "latency_max_s": float(faults["latency_s"].max()),
        "margin_median_s": float(faults["margin_s"].median()),
        "false_alarms": int(table["false_alarms"].sum()),
        "airborne_neg_h": float(table["airborne_neg_s"].sum() / 3600),
        "proc_us_p50": float(np.percentile(proc, 50)), "proc_us_p99": float(np.percentile(proc, 99)),
        "proc_us_max": float(proc.max()),
        "frame_gap_ms_p50": float(np.percentile(gaps, 50)),
        "frame_gap_ms_p99": float(np.percentile(gaps, 99)),
        "frames": int(len(dec)),
    }
    inert = table[table["injected"] & ~table["fault"]]
    summ["injected_without_effect"] = int(len(inert))
    summ["alarms_on_injected_without_effect"] = int(inert["false_alarms"].sum())
    return table, summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/sitl6_live"))
    ap.add_argument("--session-dir", type=Path, default=Path("data/stream/live_eval"))
    ap.add_argument("--model", type=Path, default=Path("data/models/stream_gbt.joblib"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--from-tlog", type=Path, default=None)
    args = ap.parse_args()
    if args.from_tlog:
        replay_to_decisions(args.from_tlog, args.model, args.session_dir)
    elif not args.analyse_only:
        run_flights(args.data, args.session_dir, args.model)
    table, summ = analyse(args.data, args.session_dir)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "live_per_flight.csv", index=False)
    (args.out / "live_summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 200)
    print(table.round(3).to_string(index=False))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
