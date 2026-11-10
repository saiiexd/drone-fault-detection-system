"""Phase 7: the sim-to-real experiment, with the gaps separated.

Phase 5 found that "a model trained on X detects faults in Y" mixes three
gaps: simulation vs reality, octorotor vs quadcopter, and catastrophic vs
gradual faults. Each experiment below holds all but one fixed where the data
allows it. All use the phase 6 representation (50 Hz MAVLink-equivalent
frames, 0.5 s window features) and the same model class (stream.train).

A. naive transfer: the phase 6 SITL model (x500, motor OFF, PX4 v1.17 sim)
   applied unchanged to RflyMAD real flights (X200/X450/X680 quads, partial
   thrust loss, PX4 v1.12). Every gap at once.
B. sim-to-real only: train on RflyMAD HIL (the real flight controller and
   firmware with simulated physics and the same fault model) and test on
   RflyMAD real flights, against real->real cross-validation as the ceiling.
   Fault model, firmware, airframe family and label source are shared, so the
   difference is the physics (and sensors). Reported as AP with a bootstrap
   95% interval over flight groups, and the transfer ratio
   AP(HIL->real) / AP(real->real).
B2. recalibrated: B's HIL model, but its threshold is set on real *no-fault*
   flights only (lowest threshold within the false-alarm budget on the
   training days' no-fault flights, day-grouped folds). Realistic: normal real
   flights are cheap to collect, real fault labels are not.
C. fault severity: B's results split by thrust-loss severity, the
   catastrophic-vs-gradual axis inside one dataset.

Metrics: frame AP on airborne frames (threshold-free, the main number);
event recall (alarm during the fault window) and latency at a threshold set
on the *training* domain for <= 2 false alarms per airborne hour; false
alarms per hour on the test domain. Partial faults don't crash the vehicle,
so "before impact" (phase 6) does not apply here.

Usage:
    uv run python -m aeropulse.sim2real.run --real data/stream/rflymad_real_frames.parquet \
        --hil data/stream/rflymad_hil_frames.parquet --sitl-model data/models/stream_gbt.joblib \
        --out reports/phase7
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from aeropulse.stream.evaluate import alarms
from aeropulse.stream.features import FEATURES
from aeropulse.stream.train import NEG_STRIDE, make_model

SEED = 0
FA_BUDGET_PER_H = 2.0
POST_FAULT_GRACE_S = 2.0
N_BOOT = 300


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Frames right after a fault clears are excluded from false-alarm counts
    and training (the vehicle is still recovering), as SITL excluded post-impact."""
    df = df.copy()
    end = pd.to_numeric(df["t_fault_end_us"], errors="coerce").to_numpy()
    grace = np.isfinite(end) & (df["t_us"].to_numpy() >= end) & \
        (df["t_us"].to_numpy() < end + POST_FAULT_GRACE_S * 1e6)
    df.loc[grace, "use"] = False
    return df.reset_index(drop=True)


def fit(df: pd.DataFrame):
    use, y = df["use"].to_numpy(), df["y"].to_numpy()
    stride = (df.groupby("key").cumcount().to_numpy() % NEG_STRIDE) == 0
    m = use & (y | stride)
    X = df.loc[m, FEATURES].to_numpy(copy=True)
    # A signal the stream never carries (e.g. acceleration on a telemetry
    # radio) gives an all-NaN column, which HistGradientBoosting cannot bin.
    # A constant is never split on, so predictions ignore that feature.
    X[:, np.isnan(X).all(axis=0)] = 0.0
    return make_model().fit(X, y[m])


def score(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(df[FEATURES].to_numpy())[:, 1]


def frame_s(df: pd.DataFrame) -> float:
    """Frame period in seconds (0.02 for the 50 Hz stream, ~0.067 for the
    15 Hz radio link): mean spacing of frames within flights, ignoring gaps
    over 1 s. The mean, not the median: thinned streams alternate 60/80 ms."""
    t, k = df["t_us"].to_numpy(), df["key"].to_numpy()
    dt = np.diff(t)[k[1:] == k[:-1]]
    return float(dt[dt < 1_000_000].mean()) / 1e6


def events(df: pd.DataFrame, alarm: np.ndarray) -> pd.DataFrame:
    dt = frame_s(df)
    rows = []
    for key, idx in df.groupby("key", sort=False).indices.items():
        d, a = df.iloc[idx], alarm[idx]
        t, y, use = d["t_us"].to_numpy(), d["y"].to_numpy(), d["use"].to_numpy()
        onset = a & ~np.r_[False, a[:-1]]
        neg = use & ~y
        row = {"key": key, "fault": bool(y.any()), "severity": float(d["severity"].iloc[0]),
               "fault_mode": d["fault_mode"].iloc[0],
               "flight_mode": d["flight_mode"].iloc[0], "group": d["group"].iloc[0],
               "false_alarms": int((onset & neg).sum()), "neg_s": neg.sum() * dt}
        if y.any():
            hit = np.flatnonzero(a & y)
            row["detected"] = bool(len(hit))
            row["latency_s"] = (t[hit[0]] - int(d["t_inject_us"].iloc[0])) / 1e6 if len(hit) else np.nan
            row["loss_at_alarm"] = float(d["loss"].to_numpy()[hit[0]]) if len(hit) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fa_threshold(df: pd.DataFrame, s: np.ndarray, budget: float = FA_BUDGET_PER_H) -> float:
    """Most event recall within the false-alarm budget; middle of the best range."""
    use = df["use"].to_numpy()
    grid = np.unique(np.quantile(s[use], np.linspace(0.5, 0.9999, 150)))
    res = []
    for thr in grid:
        ev = events(df, alarms(df, s, thr))
        h = ev["neg_s"].sum() / 3600
        res.append((thr, ev["false_alarms"].sum() / h if h else 0.0,
                    ev.loc[ev["fault"], "detected"].mean()))
    r = pd.DataFrame(res, columns=["thr", "fa_h", "recall"])
    ok = r[r["fa_h"] <= budget]
    if ok.empty:
        return float(r["thr"].max())
    best = ok[ok["recall"] >= ok["recall"].max() - 1e-9]["thr"].to_numpy()
    return float(np.sqrt(best.min() * best.max())) if best.min() > 0 else float(np.median(best))


def normal_only_threshold(df: pd.DataFrame, s: np.ndarray, budget: float = FA_BUDGET_PER_H) -> float:
    """Lowest threshold with <= budget false alarms per airborne hour on the
    no-fault flights of df. Needs no fault labels."""
    nf = df["flight_class"].to_numpy() == "Normal"
    d, sc = df[nf].reset_index(drop=True), s[nf]
    use = d["use"].to_numpy()
    hours = use.sum() * frame_s(d) / 3600
    for thr in np.unique(np.quantile(sc[use], np.linspace(0.9, 1.0, 400))):
        a = alarms(d, sc, thr)
        onsets = (a & ~np.r_[False, a[:-1]] & use).sum()
        if onsets / hours <= budget:
            return float(thr)
    return float(sc.max() + 1e-9)


def day_folds_alarms(df: pd.DataFrame, s: np.ndarray, n: int = 5) -> np.ndarray:
    """Alarms with a per-fold threshold from the other days' no-fault flights."""
    a = np.zeros(len(df), bool)
    for tr, te in folds(df, n, SEED):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        thr = normal_only_threshold(dtr, s[tr])
        a[te] = alarms(dte, s[te], thr)
    return a


def ap(df: pd.DataFrame, s: np.ndarray) -> float:
    u = df["use"].to_numpy()
    return float(average_precision_score(df["y"].to_numpy()[u], s[u]))


def ap_boot(df: pd.DataFrame, s: np.ndarray, n: int = N_BOOT) -> tuple[float, float]:
    """95% interval of AP, resampling flight groups (not frames: frames of
    one flight are strongly correlated)."""
    rng = np.random.default_rng(SEED)
    u = df["use"].to_numpy()
    g = df["group"].astype(str).to_numpy()[u]
    y, sc = df["y"].to_numpy()[u], s[u]
    groups = np.unique(g)
    idx = {k: np.flatnonzero(g == k) for k in groups}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(groups, len(groups))])
        if y[pick].any() and (~y[pick]).any():
            vals.append(average_precision_score(y[pick], sc[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarise(name: str, df: pd.DataFrame, s: np.ndarray, thr: float) -> tuple[dict, pd.DataFrame]:
    ev = events(df, alarms(df, s, thr)).assign(experiment=name)
    f = ev[ev["fault"]]
    h = ev["neg_s"].sum() / 3600
    lo, hi = ap_boot(df, s)
    return {"experiment": name, "ap": ap(df, s), "ap_lo": lo, "ap_hi": hi,
            "chance_ap": float(df.loc[df["use"], "y"].mean()), "threshold": thr,
            "event_recall": float(f["detected"].mean()), "fault_flights": int(len(f)),
            "latency_median_s": float(f["latency_s"].median()),
            "false_alarms_per_h": float(ev["false_alarms"].sum() / h) if h else np.nan,
            "neg_hours": float(h)}, ev


def folds(df: pd.DataFrame, n: int, seed: int):
    fl = df.groupby("key").agg(cls=("flight_class", "first"), group=("group", "first")).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(fl, fl["cls"], fl["group"].astype(str)):
        yield (np.flatnonzero(df["key"].isin(set(fl.loc[tr, "key"])).to_numpy()),
               np.flatnonzero(df["key"].isin(set(fl.loc[te, "key"])).to_numpy()))


def oof(df: pd.DataFrame, n: int = 5, seed: int = SEED) -> np.ndarray:
    s = np.full(len(df), np.nan)
    for tr, te in folds(df, n, seed):
        s[te] = score(fit(df.iloc[tr]), df.iloc[te])
    return s


def cv_with_inner_threshold(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Real->real: test-fold scores and alarms, threshold from inner OOF."""
    s = np.full(len(df), np.nan)
    a = np.zeros(len(df), bool)
    for k, (tr, te) in enumerate(folds(df, 5, SEED)):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        thr = fa_threshold(dtr, oof(dtr, 4, SEED + 1 + k))
        s[te] = score(fit(dtr), dte)
        a[te] = alarms(dte, s[te], thr)
    return s, a


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--real", type=Path, required=True)
    ap_.add_argument("--hil", type=Path, required=True)
    ap_.add_argument("--sitl-model", type=Path, required=True)
    ap_.add_argument("--out", type=Path, required=True)
    ap_.add_argument("--only", choices=["all", "A"], default="all",
                     help="A: re-run only experiment A and merge into existing results")
    args = ap_.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    real, hil = prepare(pd.read_parquet(args.real)), prepare(pd.read_parquet(args.hil))
    for name, d in (("real", real), ("hil", hil)):
        fl = d.groupby("key")["flight_class"].first().value_counts().to_dict()
        print(f"{name}: {d['key'].nunique()} flights {fl}, {int(d['use'].sum())} airborne frames, "
              f"{int(d['y'].sum())} fault frames")

    rows, evs = [], []

    # A. SITL model, unchanged (its own threshold), on real and HIL
    b = joblib.load(args.sitl_model)
    for name, d in (("A_sitl_to_real", real), ("A_sitl_to_hil", hil)):
        r, ev = summarise(name, d, score(b["model"], d), b["threshold"])
        rows.append(r)
        evs.append(ev)

    if args.only == "A":
        old = pd.read_csv(args.out / "sim2real_summary.csv")
        old_ev = pd.read_csv(args.out / "sim2real_events.csv")
        rows = rows + old[~old["experiment"].str.startswith("A_")].to_dict("records")
        evs = evs + [old_ev[~old_ev["experiment"].str.startswith("A_")]]
        finish(args, rows, evs)
        return

    # B. HIL -> real (threshold from HIL out-of-fold scores), HIL->HIL CV, real->real CV
    hil_oof = oof(hil)
    thr_hil = fa_threshold(hil, hil_oof)
    r, ev = summarise("B_hil_cv", hil, hil_oof, thr_hil)
    rows.append(r)
    evs.append(ev)
    m_hil = fit(hil)
    r, ev = summarise("B_hil_to_real", real, score(m_hil, real), thr_hil)
    rows.append(r)
    evs.append(ev)
    s_hr = score(m_hil, real)
    ev = events(real, day_folds_alarms(real, s_hr)).assign(experiment="B2_hil_to_real_recalibrated")
    f = ev[ev["fault"]]
    rows.append({"experiment": "B2_hil_to_real_recalibrated", "ap": rows[-1]["ap"],
                 "ap_lo": rows[-1]["ap_lo"], "ap_hi": rows[-1]["ap_hi"],
                 "chance_ap": rows[-1]["chance_ap"], "threshold": np.nan,
                 "event_recall": float(f["detected"].mean()), "fault_flights": int(len(f)),
                 "latency_median_s": float(f["latency_s"].median()),
                 "false_alarms_per_h": float(ev["false_alarms"].sum() / (ev["neg_s"].sum() / 3600)),
                 "neg_hours": float(ev["neg_s"].sum() / 3600)})
    evs.append(ev)

    s_rr, a_rr = cv_with_inner_threshold(real)
    ev = events(real, a_rr).assign(experiment="B_real_cv")
    f = ev[ev["fault"]]
    lo, hi = ap_boot(real, s_rr)
    rows.append({"experiment": "B_real_cv", "ap": ap(real, s_rr), "ap_lo": lo, "ap_hi": hi,
                 "chance_ap": float(real.loc[real["use"], "y"].mean()), "threshold": np.nan,
                 "event_recall": float(f["detected"].mean()), "fault_flights": int(len(f)),
                 "latency_median_s": float(f["latency_s"].median()),
                 "false_alarms_per_h": float(ev["false_alarms"].sum() / (ev["neg_s"].sum() / 3600)),
                 "neg_hours": float(ev["neg_s"].sum() / 3600)})
    evs.append(ev)
    finish(args, rows, evs)


def finish(args, rows: list[dict], evs: list[pd.DataFrame]) -> None:
    table = pd.DataFrame(rows)
    ev_all = pd.concat(evs, ignore_index=True)
    ratio = table.set_index("experiment").loc["B_hil_to_real", "ap"] / \
        table.set_index("experiment").loc["B_real_cv", "ap"]

    # C. severity breakdown of event recall
    sev = ev_all[ev_all["fault"]].assign(
        sev_bin=lambda e: pd.cut(e["severity"], [0, 0.1, 0.2, 0.4, 0.7, 1.0],
                                 labels=["<=10%", "10-20%", "20-40%", "40-70%", ">70%"]))
    by_sev = sev.pivot_table(index="sev_bin", columns="experiment", values="detected",
                             aggfunc="mean", observed=False)
    n_sev = sev[sev["experiment"] == "B_real_cv"].groupby("sev_bin", observed=False).size()
    by_sev.insert(0, "n_real_flights", n_sev)
    # step vs ramp: event recall, latency (step) and loss at the alarm (ramp)
    by_mode = sev.groupby(["experiment", "fault_mode"]).agg(
        n=("key", "size"), recall=("detected", "mean"), latency_median_s=("latency_s", "median"),
        loss_at_alarm_median=("loss_at_alarm", "median")).reset_index()
    by_mode.to_csv(args.out / "sim2real_by_fault_mode.csv", index=False)

    table.to_csv(args.out / "sim2real_summary.csv", index=False)
    ev_all.to_csv(args.out / "sim2real_events.csv", index=False)
    by_sev.to_csv(args.out / "sim2real_by_severity.csv")
    (args.out / "sim2real_headline.json").write_text(json.dumps(
        {"transfer_ratio_ap_hil_to_real_over_real_cv": float(ratio)}, indent=2))
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print(f"\ntransfer ratio AP(HIL->real)/AP(real->real) = {ratio:.2f}")
    print(by_sev.round(2).to_string())
    print(by_mode.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
