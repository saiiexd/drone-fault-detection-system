"""Evaluation of the live (frame-level) motor-failure detector.

The question for a quadcopter is not "is this window anomalous?" but
"does the alarm come before the vehicle hits the ground?". So the main
metrics are per flight:

- detected before impact: an alarm between injection and impact;
- latency: first alarm minus injection, on the autopilot clock (exact
  injection time from the ULog; the alarm time is the frame's boot time);
- margin: impact minus first alarm (how much warning was left);
- false alarms: alarm onsets on airborne frames without a fault (nominal
  flights and the pre-fault part of fault flights), per airborne hour.

Frame-level average precision (AP) is reported as a threshold-free summary.

Thresholds are chosen on out-of-fold scores of the *training* flights only
(inner cross-validation), never on the test flights. Rule: among thresholds
with zero false alarms on validation, keep those that detect the most fault
flights before impact, and take the middle of that range (geometric mean of
its ends, since scores are probabilities spanning decades). Taking the lowest
(as phase 4 effectively did) gives the hair-trigger 0.001 threshold seen in
phase 4; the middle keeps a margin on both sides.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PERSIST = 3       # consecutive alarming frames (60 ms at 50 Hz)


class Persistence:
    """Online k-of-k filter: alarm once k consecutive frames exceed the threshold."""

    def __init__(self, k: int = PERSIST):
        self.k, self.run = k, 0

    def step(self, over: bool) -> bool:
        self.run = self.run + 1 if over else 0
        return self.run >= self.k


def persist(over: np.ndarray, k: int = PERSIST) -> np.ndarray:
    """Vectorised Persistence over one flight; identical output to stepping."""
    # run length at i = i - (index of the last False at or before i)
    i = np.arange(len(over))
    last_false = np.maximum.accumulate(np.where(~np.asarray(over, bool), i, -1))
    return (i - last_false) >= k


def alarms(df: pd.DataFrame, score: np.ndarray, thr: float, k: int = PERSIST) -> np.ndarray:
    """Alarm per frame; only airborne frames can alarm (the live gate)."""
    over = (score >= thr) & df["airborne"].to_numpy()
    out = np.zeros(len(df), bool)
    for idx in df.groupby("key", sort=False).indices.values():
        out[idx] = persist(over[idx], k)
    return out


def flight_results(df: pd.DataFrame, alarm: np.ndarray) -> pd.DataFrame:
    t_all, k_all = df["t_us"].to_numpy(), df["key"].to_numpy()
    dt = np.diff(t_all)[k_all[1:] == k_all[:-1]]
    frame_s = float(dt[dt < 1_000_000].mean()) / 1e6   # 0.02 at 50 Hz; mean, as thinned streams alternate 60/80 ms
    rows = []
    for key, idx in df.groupby("key", sort=False).indices.items():
        d, a = df.iloc[idx], alarm[idx]
        use, y, t = d["use"].to_numpy(), d["y"].to_numpy(), d["t_us"].to_numpy()
        neg = use & ~y
        onset = a & ~np.r_[False, a[:-1]]
        row = {"key": key, "fault": bool(y.any()),
               "false_alarms": int((onset & neg).sum()),
               "airborne_neg_s": neg.sum() * frame_s}
        if y.any():
            t_inj, t_imp = int(d["t_inject_us"].iloc[0]), d["t_impact_us"].iloc[0]
            hit = np.flatnonzero(a & (t >= t_inj))
            row["time_to_impact_s"] = (int(t_imp) - t_inj) / 1e6 if pd.notna(t_imp) else np.nan
            if len(hit):
                t_al = int(t[hit[0]])
                row["latency_s"] = (t_al - t_inj) / 1e6
                row["margin_s"] = ((int(t_imp) - t_al) / 1e6) if pd.notna(t_imp) else np.nan
            else:
                row["latency_s"] = row["margin_s"] = np.nan
            row["before_impact"] = bool(row["margin_s"] > 0) if np.isfinite(row["margin_s"]) else False
        rows.append(row)
    return pd.DataFrame(rows)


def summary(df: pd.DataFrame, score: np.ndarray, alarm: np.ndarray) -> dict:
    fr = flight_results(df, alarm)
    faults = fr[fr["fault"]]
    use = df["use"].to_numpy()
    y = df["y"].to_numpy()[use]
    hours = fr["airborne_neg_s"].sum() / 3600
    lat = faults["latency_s"].dropna()
    return {
        "fault_flights": int(len(faults)),
        "detected": int(faults["latency_s"].notna().sum()),
        "before_impact": int(faults["before_impact"].sum()),
        "latency_median_s": float(lat.median()) if len(lat) else np.nan,
        "latency_max_s": float(lat.max()) if len(lat) else np.nan,
        "margin_median_s": float(faults["margin_s"].median()) if len(faults) else np.nan,
        "false_alarms": int(fr["false_alarms"].sum()),
        "false_alarms_per_h": float(fr["false_alarms"].sum() / hours) if hours else np.nan,
        "airborne_neg_h": float(hours),
        "frame_ap": float(average_precision_score(y, np.nan_to_num(score[use], nan=-1e9)))
        if y.any() else np.nan,
    }


def choose_threshold(df: pd.DataFrame, score: np.ndarray, n_grid: int = 200) -> float:
    """Middle of the zero-false-alarm range that detects most faults before impact."""
    use = df["use"].to_numpy()
    s = score[use & np.isfinite(score)]
    if len(s) == 0:          # detector's input absent from this stream (e.g. no IMU on a radio)
        return float("inf")
    grid = np.unique(np.quantile(s, np.linspace(0.5, 1.0, n_grid)))
    rows = []
    for thr in grid:
        fr = flight_results(df, alarms(df, score, thr))
        rows.append((thr, int(fr["false_alarms"].sum()),
                     int(fr.loc[fr["fault"], "before_impact"].sum())))
    res = pd.DataFrame(rows, columns=["thr", "fa", "hits"])
    ok = res[res["fa"] == 0]
    if ok.empty:                       # no zero-FA threshold: take the fewest FAs
        ok = res[res["fa"] == res["fa"].min()]
    best = ok[ok["hits"] == ok["hits"].max()]["thr"].to_numpy()
    return float(np.sqrt(best.min() * best.max())) if best.min() > 0 else float(np.median(best))
