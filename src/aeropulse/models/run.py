"""Phase 4: train and evaluate baseline detectors on UAV-SEAD.

Usage:
    uv run python -m aeropulse.models.run --lake data/lake --out reports/phase4 \
        --models data/models
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from aeropulse.models.dataset import CLASSES, MODEL_FEATURES, load_ranges, load_windows
from aeropulse.models.detectors import GbtBinary, make_classifier, make_detectors
from aeropulse.models.evaluate import (
    PERSIST,
    TARGET_FA_PER_H,
    best_f1_threshold,
    cross_validate,
    event_metrics,
    fa_budget_threshold,
    false_alarms_per_hour,
    merge_ranges,
    session_folds,
    training_mask,
    window_metrics,
)


def detector_table(df, scores, alarms, thresholds, events, subset=None) -> pd.DataFrame:
    rows = []
    mask = np.ones(len(df), bool) if subset is None else subset
    sub = df[mask]
    y = sub["label_anomaly"].to_numpy()
    for name in scores:
        s, a = scores[name][mask], alarms[name][mask]
        thr = float(np.median(thresholds[name]))  # per-fold thresholds differ
        wm = window_metrics(y, s, -np.inf)          # placeholder for AP and counts
        tp, fp, fn = (a & y).sum(), (a & ~y).sum(), (~a & y).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        ev = event_metrics(sub, a, events)
        fa, hours = false_alarms_per_hour(sub, a)
        rows.append({
            "detector": name,
            "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "avg_precision": wm["avg_precision"],
            "event_recall": ev["detected"].mean() if len(ev) else np.nan,
            "n_events": len(ev),
            "latency_median_s": ev["latency_s"].median() if len(ev) else np.nan,
            "false_alarms_per_h": fa, "normal_hours": hours,
            "threshold_median": thr,
        })
    return pd.DataFrame(rows)


def per_class_event_recall(df, alarms, events) -> pd.DataFrame:
    out = {}
    for name, a in alarms.items():
        ev = event_metrics(df, a, events)
        ev["cls"] = ev["event_class"].str.split(" \\+ ").str[0]
        out[name] = ev.groupby("event_class")["detected"].mean()
    counts = event_metrics(df, next(iter(alarms.values())), events).groupby("event_class").size()
    return pd.DataFrame(out).assign(n_events=counts)


def classifier_cv(df) -> tuple[np.ndarray, np.ndarray]:
    X, yc = df[MODEL_FEATURES], df["y_class"].to_numpy()
    train_ok = training_mask(df)
    pred = np.empty(len(df), dtype=object)
    proba = np.zeros(len(df))
    for tr, te in session_folds(df):
        tr = tr[train_ok[tr]]
        clf = make_classifier().fit(X.iloc[tr], yc[tr])
        p = clf.predict_proba(X.iloc[te])
        pred[te] = clf.classes_[p.argmax(1)]
        proba[te] = p.max(1)
    return pred, proba


def train_final(df, models_dir: Path) -> dict:
    """Final detector + classifier on all usable flights.

    The alarm threshold still comes from held-out sessions: fit on 3/4 of
    the sessions, choose the threshold on the rest, then keep that threshold.
    """
    X, y, yc = df[MODEL_FEATURES], df["label_anomaly"].to_numpy(), df["y_class"].to_numpy()
    ok = training_mask(df)
    fit_idx, val_idx = next(session_folds(df, n_splits=4, seed=123))
    det = GbtBinary().fit(X.iloc[fit_idx[ok[fit_idx]]], y[fit_idx[ok[fit_idx]]])
    s_val = det.score(X.iloc[val_idx])
    thr = best_f1_threshold(y[val_idx], s_val)
    thr_fa = fa_budget_threshold(df.iloc[val_idx], s_val)
    clf = make_classifier().fit(X[ok], yc[ok])
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle = {"detector": det.model, "threshold": thr, "threshold_fa": thr_fa,
              "persist_windows": PERSIST, "target_fa_per_h": TARGET_FA_PER_H,
              "classifier": clf,
              "features": MODEL_FEATURES, "trained_on": "UAV-SEAD usable in-air windows",
              "n_windows": int(ok.sum())}
    path = models_dir / "uavsead_gbt.joblib"
    joblib.dump(bundle, path)
    return {"path": str(path), "threshold_f1": thr, "threshold_fa": thr_fa}


def fmt(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f") if hasattr(df, "to_markdown") else df.to_string()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = load_windows(args.lake)
    events = merge_ranges(load_ranges(args.lake))
    print(f"{len(df)} in-air windows, {df['flight_id'].nunique()} flights, "
          f"{df['session'].nunique()} sessions, positives {int(df['label_anomaly'].sum())} "
          f"({time.time() - t0:.0f} s)")

    scores, alarms, thresholds, fold = cross_validate(df, make_detectors)
    overall = pd.concat([detector_table(df, scores, alarms[op], thresholds[op], events)
                         .assign(operating_point=op) for op in ("f1", "fa")])
    by_airframe = pd.concat(
        [detector_table(df, scores, alarms[op], thresholds[op], events,
                        (df["airframe"] == af).to_numpy()).assign(airframe=af, operating_point=op)
         for op in ("f1", "fa") for af in ("octo", "quad")])
    per_class = pd.concat({op: per_class_event_recall(df, alarms[op], events)
                           for op in ("f1", "fa")}, axis=1)

    pred, proba = classifier_cv(df)
    yc = df["y_class"].to_numpy()
    report = classification_report(yc, pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = pd.DataFrame(confusion_matrix(yc, pred, labels=CLASSES), index=CLASSES, columns=CLASSES)
    anomalous = df["label_anomaly"].to_numpy()
    type_acc_given_anomaly = float((pred[anomalous] == yc[anomalous]).mean())

    final = train_final(df, args.models)

    pd.DataFrame({"flight_id": df["flight_id"], "t_us": df["t_us"], "fold": fold,
                  "label_anomaly": df["label_anomaly"], "y_class": yc,
                  "airframe": df["airframe"], "flight_class": df["flight_class"],
                  **{f"score_{k}": v for k, v in scores.items()},
                  "type_pred": pred, "type_proba": proba}).to_parquet(
        args.out / "oof_predictions.parquet", index=False)
    overall.to_csv(args.out / "detectors_overall.csv", index=False)
    by_airframe.to_csv(args.out / "detectors_by_airframe.csv", index=False)
    per_class.to_csv(args.out / "event_recall_by_class.csv")
    cm.to_csv(args.out / "type_confusion.csv")
    summary = {
        "windows": len(df), "flights": int(df["flight_id"].nunique()),
        "sessions": int(df["session"].nunique()),
        "positives": int(df["label_anomaly"].sum()),
        "type_macro_f1": report["macro avg"]["f1-score"],
        "type_accuracy_given_anomaly": type_acc_given_anomaly,
        "type_per_class": {c: {k: report[c][k] for k in ("precision", "recall", "f1-score", "support")}
                           for c in CLASSES},
        "final_model": final, "runtime_s": round(time.time() - t0, 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    pd.set_option("display.width", 200)
    print("\n== detectors, all test folds\n" + overall.round(3).to_string(index=False))
    print("\n== by airframe\n" + by_airframe.round(3).to_string(index=False))
    print("\n== event recall by class\n" + per_class.round(3).to_string())
    print("\n== fault type (window level)\n" + cm.to_string())
    print(f"type macro F1 {summary['type_macro_f1']:.3f}, "
          f"type accuracy on anomalous windows {type_acc_given_anomaly:.3f}")
    print(f"\nfinal model -> {final['path']} (thresholds f1 {final['threshold_f1']:.3f}, "
          f"fa {final['threshold_fa']:.3f}); "
          f"{summary['runtime_s']} s")


if __name__ == "__main__":
    main()
