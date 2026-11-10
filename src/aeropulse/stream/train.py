"""Phase 6: train and evaluate the live motor-failure detector on SITL streams.

Data: frame tables from stream.dataset (MAVLink tlogs of phase 6 SITL quad
flights). Each flight is its own group: flights are independent simulator
runs (fresh container, own seed), so grouping by flight is the honest split
here (UAV-SEAD needed grouping by recording day instead).

Detectors, simplest first:
- rule_tilt_err:  attitude-tracking error (deg) in the latest frame. "The
                  vehicle is not where its controller wants it."
- rule_lift:      lift deficit, mean acc_z + 9.81 over 0.5 s (m/s²): specific
                  force along the body axis collapses towards 0 in free fall.
- gbt_fast:       HistGradientBoosting on the 0.5 s window features.

Protocol: 5-fold StratifiedGroupKFold over flights. Inside each training
part, a 4-fold inner CV gives out-of-fold scores on which the threshold is
chosen (evaluate.choose_threshold). The test fold is scored once.

Usage:
    uv run python -m aeropulse.stream.train --frames data/stream/sitl6_frames.parquet \
        --out reports/phase6 --models data/models
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold

from aeropulse.stream.dataset import RADIO_HZ, RADIO_STALE_US, RADIO_WINDOW
from aeropulse.stream.evaluate import PERSIST, alarms, choose_threshold, flight_results, summary
from aeropulse.stream.features import FEATURES, RATE_HZ, STALE_US, WINDOW

SEED = 0
N_FOLDS = 5
NEG_STRIDE = 5       # keep every 5th negative frame for fitting (0.1 s apart)

RULES = {
    "rule_tilt_err": lambda X: X["tilt_err__last"].to_numpy(),
    "rule_lift": lambda X: (X["acc_z__mean"] + 9.81).to_numpy(),
}


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=1.0, class_weight="balanced", random_state=SEED)


def fit_mask(df: pd.DataFrame) -> np.ndarray:
    use, y = df["use"].to_numpy(), df["y"].to_numpy()
    stride = (df.groupby("key").cumcount().to_numpy() % NEG_STRIDE) == 0
    return use & (y | stride)


def fit(df: pd.DataFrame) -> HistGradientBoostingClassifier:
    m = fit_mask(df)
    X = df.loc[m, FEATURES].to_numpy(copy=True)
    X[:, np.isnan(X).all(axis=0)] = 0.0   # signal absent from the stream: constant, never split
    return make_model().fit(X, df.loc[m, "y"].to_numpy())


def flight_folds(df: pd.DataFrame, n: int, seed: int):
    fl = df.groupby("key").agg(cls=("flight_class", "first")).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(fl, fl["cls"], fl["key"]):
        yield (np.flatnonzero(df["key"].isin(set(fl.loc[tr, "key"])).to_numpy()),
               np.flatnonzero(df["key"].isin(set(fl.loc[te, "key"])).to_numpy()))


def oof_scores(df: pd.DataFrame, n: int, seed: int) -> np.ndarray:
    s = np.full(len(df), np.nan)
    for tr, te in flight_folds(df, n, seed):
        s[te] = fit(df.iloc[tr]).predict_proba(df.iloc[te][FEATURES].to_numpy())[:, 1]
    return s


def cross_validate(df: pd.DataFrame, log=print):
    scores = {k: np.full(len(df), np.nan) for k in ["gbt_fast", *RULES]}
    alarm = {k: np.zeros(len(df), bool) for k in scores}
    thr_log = []
    for fold, (tr, te) in enumerate(flight_folds(df, N_FOLDS, SEED)):
        dtr, dte = df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
        inner = oof_scores(dtr, 4, SEED + 1 + fold)
        thr = {"gbt_fast": choose_threshold(dtr, inner)}
        model = fit(dtr)
        s_te = {"gbt_fast": model.predict_proba(dte[FEATURES].to_numpy())[:, 1]}
        for name, rule in RULES.items():
            thr[name] = choose_threshold(dtr, rule(dtr))
            s_te[name] = rule(dte)
        for k in scores:
            scores[k][te] = s_te[k]
            alarm[k][te] = alarms(dte, s_te[k], thr[k])
        thr_log.append({"fold": fold, **thr})
        log(f"  fold {fold}: test flights {dte['key'].nunique()}, thresholds "
            + ", ".join(f"{k}={v:.3g}" for k, v in thr.items()))
    return scores, alarm, pd.DataFrame(thr_log)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--radio", action="store_true",
                    help="frames were built with dataset --radio: store radio window/rate in the bundle")
    ap.add_argument("--name", default="stream_gbt", help="model file name (without .joblib)")
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_parquet(args.frames)
    n_f = df.groupby("key")["flight_class"].first().value_counts().to_dict()
    print(f"{len(df)} frames, {int(df['use'].sum())} airborne pre-impact, "
          f"{int(df['y'].sum())} positive; flights {n_f}")

    scores, alarm, thr = cross_validate(df)
    args.out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{"detector": k, **summary(df, scores[k], alarm[k])} for k in scores])
    per_flight = pd.concat([flight_results(df, alarm[k]).assign(detector=k) for k in scores])
    table.to_csv(args.out / "cv_detectors.csv", index=False)
    per_flight.to_csv(args.out / "cv_per_flight.csv", index=False)
    thr.to_csv(args.out / "cv_thresholds.csv", index=False)
    pd.DataFrame({"key": df["key"], "t_us": df["t_us"], "y": df["y"], "use": df["use"],
                  **{f"score_{k}": v for k, v in scores.items()},
                  **{f"alarm_{k}": v for k, v in alarm.items()}}).to_parquet(
        args.out / "cv_scores.parquet", index=False)

    # Final model on all flights; threshold from out-of-fold scores of all flights.
    oof = oof_scores(df, N_FOLDS, SEED + 100)
    final_thr = choose_threshold(df, oof)
    model = fit(df)
    imp = permutation_like_importance(model, df)
    bundle = {"model": model, "features": FEATURES, "threshold": final_thr,
              "persist": PERSIST,
              # The live detector rebuilds frames and windows from these, so a
              # model can only be served with the stream settings it was trained on.
              "window": RADIO_WINDOW if args.radio else WINDOW,
              "rate_hz": RADIO_HZ if args.radio else RATE_HZ,
              "stale_us": RADIO_STALE_US if args.radio else STALE_US,
              "radio": bool(args.radio),
              "rule_thresholds": {k: choose_threshold(df, r(df)) for k, r in RULES.items()},
              "trained_on": f"PX4 v1.17 SITL x500 MAVLink streams, {df['key'].nunique()} flights",
              "trained_at": time.strftime("%Y-%m-%d")}
    args.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.models / f"{args.name}.joblib")
    (args.out / "final_model.json").write_text(json.dumps(
        {"threshold": final_thr, "rule_thresholds": bundle["rule_thresholds"],
         "top_features": imp.head(12).round(4).to_dict()}, indent=2))

    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))
    print("\nfault flights, gbt_fast:")
    pf = per_flight[(per_flight["detector"] == "gbt_fast") & per_flight["fault"]]
    print(pf[["key", "time_to_impact_s", "latency_s", "margin_s", "before_impact"]]
          .round(3).to_string(index=False))
    print(f"\nfinal threshold {final_thr:.4g}; top features:\n{imp.head(8).round(4).to_string()}")
    print(f"{time.time() - t0:.0f} s")


def permutation_like_importance(model, df: pd.DataFrame, n: int = 20000) -> pd.Series:
    """Drop in AP when one feature is shuffled (on a sample of used frames)."""
    from sklearn.inspection import permutation_importance
    d = df[df["use"]]
    d = pd.concat([d[d["y"]], d[~d["y"]].sample(min(n, (~d["y"]).sum()), random_state=SEED)])
    r = permutation_importance(model, d[FEATURES].to_numpy(), d["y"].to_numpy(), scoring="average_precision",
                               n_repeats=3, random_state=SEED, n_jobs=-1)
    return pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False)


if __name__ == "__main__":
    main()
