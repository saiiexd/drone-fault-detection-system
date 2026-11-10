"""Phase 4b: dedicated motor-fault (Mechanical) detector.

The general detectors are trained on "any anomaly", where motor faults are
only ~9% of positives, so they mostly learn estimator anomalies. Here the task
is Mechanical vs Normal: windows of the other fault classes are removed from
both training and evaluation. Same session-grouped cross-validation, same two
operating points, same metrics as run.py.

Usage:
    uv run python -m aeropulse.models.mechanical --lake data/lake \
        --out reports/phase4 --models data/models
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from aeropulse.models.dataset import MODEL_FEATURES, load_ranges, load_windows
from aeropulse.models.detectors import EkfTestRatio, GbtBinary, PhysicsRule
from aeropulse.models.evaluate import (
    PERSIST,
    TARGET_FA_PER_H,
    best_f1_threshold,
    cross_validate,
    fa_budget_threshold,
    merge_ranges,
    session_folds,
    training_mask,
)
from aeropulse.models.run import detector_table


class GbtMechanical(GbtBinary):
    name = "gbt_mechanical"


def make_mech_detectors():
    return [EkfTestRatio(), PhysicsRule(), GbtMechanical()]


def mechanical_task(df: pd.DataFrame) -> pd.DataFrame:
    keep = df["y_class"].isin(["Normal", "Mechanical"])
    out = df[keep].copy()
    out["label_anomaly"] = (out["y_class"] == "Mechanical").to_numpy()
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    args = ap.parse_args()
    t0 = time.time()

    df = mechanical_task(load_windows(args.lake))
    events = merge_ranges(load_ranges(args.lake))
    events = events[events["event_class"].str.contains("Mechanical")]
    print(f"{len(df)} windows, positives {int(df['label_anomaly'].sum())}, "
          f"{len(events)} Mechanical events")

    scores, alarms, thresholds, _ = cross_validate(df, make_mech_detectors)
    table = pd.concat(
        [detector_table(df, scores, alarms[op], thresholds[op], events, sub)
         .assign(operating_point=op, airframe=af)
         for op in ("f1", "fa")
         for af, sub in (("all", None), ("octo", (df["airframe"] == "octo").to_numpy()))])
    table.to_csv(args.out / "mechanical_detectors.csv", index=False)
    pd.DataFrame({"flight_id": df["flight_id"], "t_us": df["t_us"],
                  "label": df["label_anomaly"], "airframe": df["airframe"],
                  **{f"score_{k}": v for k, v in scores.items()}}).to_parquet(
        args.out / "mechanical_oof.parquet", index=False)

    # Final model: fit on 3/4 of sessions, thresholds from the held-out 1/4.
    X, y = df[MODEL_FEATURES], df["label_anomaly"].to_numpy()
    ok = training_mask(df)
    fit_idx, val_idx = next(session_folds(df, n_splits=4, seed=123))
    det = GbtMechanical().fit(X.iloc[fit_idx[ok[fit_idx]]], y[fit_idx[ok[fit_idx]]])
    s_val = det.score(X.iloc[val_idx])
    rule_val = PhysicsRule().score(X.iloc[val_idx])
    bundle = {
        "detector": det.model, "features": MODEL_FEATURES,
        "threshold_f1": best_f1_threshold(y[val_idx], s_val),
        "threshold_fa": fa_budget_threshold(df.iloc[val_idx], s_val),
        "physics_rule_threshold_fa": fa_budget_threshold(df.iloc[val_idx], rule_val),
        "persist_windows": PERSIST, "target_fa_per_h": TARGET_FA_PER_H,
        "trained_on": "UAV-SEAD Mechanical vs Normal, in-air windows (octorotor + quad)",
    }
    args.models.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.models / "mechanical_gbt.joblib")
    (args.out / "mechanical_final_thresholds.json").write_text(json.dumps(
        {k: float(v) for k, v in bundle.items() if "threshold" in k}, indent=2))

    pd.set_option("display.width", 220)
    cols = ["detector", "airframe", "operating_point", "precision", "recall", "f1",
            "avg_precision", "event_recall", "n_events", "latency_median_s", "false_alarms_per_h"]
    print(table[cols].round(3).to_string(index=False))
    print(f"\nfinal -> {args.models / 'mechanical_gbt.joblib'}; {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
