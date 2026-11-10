"""Phase 7 D: the airframe gap with real data on both sides.

The phase 4 motor-fault model (gbt_mechanical) was trained on real UAV-SEAD
flights, 63 of whose 64 motor-fault events are on an octorotor. Here it is
applied to RflyMAD's real quadcopter flights, through the unchanged phase 3
pipeline (2 s windows from the ULog). The ceiling is the same phase 4 recipe
cross-validated on RflyMAD itself (grouped by test case).

Both sides are real flights, so the gap measured is airframe/vehicle/fault
model (octorotor motor problems vs quad partial thrust loss), not sim vs
real. It is not a pure airframe effect: vehicles, sites and the way faults
arise also differ, and that is stated wherever the number is used.

Usage:
    uv run python -m aeropulse.sim2real.octo_to_quad --raw data/raw/rflymad \
        --lake data/lake_rflymad --models data/models --out reports/phase7
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from aeropulse.models.dataset import MODEL_FEATURES, load_windows
from aeropulse.models.detectors import PhysicsRule
from aeropulse.models.evaluate import false_alarms_per_hour, persistence, training_mask
from aeropulse.models.mechanical import GbtMechanical
from aeropulse.pipeline.run import build as build_lake
from aeropulse.sim2real.rflymad import fault_info, recording_day


def write_manifest(raw: Path) -> dict:
    """Phase 3 manifest for RflyMAD real flights. The key starts with the
    recording day, which the pipeline uses as the session (grouping) id."""
    cases = json.loads((raw / "cases_real.json").read_text())
    files = []
    for case, c in sorted(cases.items()):
        if "ulg" not in c:
            continue
        path = raw / c["ulg"]
        info = fault_info(path)
        if set(info["fault_ids"]) - {123450}:
            continue
        data = path.read_bytes()
        ann = [{"class": "Normal", "note": "", "ranges": []}]
        cls = "Normal"
        if info["t_on"] is not None:
            cls = "Mechanical"
            end = info["t_off"] or 2**62
            ann = [{"class": cls, "note": f"severity {info['severity']:.2f}",
                    "ranges": [["rfly_ctrl_lxl motor fault", [[info["t_on"], end]]]]}]
        files.append({"key": f"{recording_day(path)}/{case}", "path": c["ulg"], "class": cls,
                      "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "annotations": ann, "severity": info["severity"]})
    manifest = {"source": "rflymad_real", "revision": "kaggle xianglile/rflymad v10", "files": files}
    (raw / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def event_recall(df: pd.DataFrame, alarm: np.ndarray) -> tuple[float, int]:
    hits, n = 0, 0
    for _, idx in df.groupby("flight_id").indices.items():
        y = df["label_anomaly"].to_numpy()[idx]
        if y.any():
            n += 1
            hits += bool((alarm[idx] & y).any())
    return (hits / n if n else np.nan), n


def evaluate(df: pd.DataFrame, s: np.ndarray, thr: float) -> dict:
    y = df["label_anomaly"].to_numpy()
    a = persistence(df, s >= thr)
    rec, n = event_recall(df, a)
    fa, hours = false_alarms_per_hour(df, a)
    return {"ap": float(average_precision_score(y, s)), "chance_ap": float(y.mean()),
            "event_recall": rec, "fault_flights": n, "false_alarms_per_h": fa,
            "normal_hours": hours, "threshold": thr}


def _cv_scores(df: pd.DataFrame, cols: list[str], n: int, seed: int) -> np.ndarray:
    fl = df.groupby("flight_id").agg(s=("session", "first"), c=("flight_primary", "first")).reset_index()
    out = np.full(len(df), np.nan)
    ok = training_mask(df)
    for tr, te in StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(fl, fl["c"], fl["s"]):
        trm = df["flight_id"].isin(set(fl.loc[tr, "flight_id"])).to_numpy()
        tem = df["flight_id"].isin(set(fl.loc[te, "flight_id"])).to_numpy()
        det = GbtMechanical().fit(df.loc[trm & ok, cols], df["label_anomaly"].to_numpy()[trm & ok])
        out[tem] = det.score(df.loc[tem, cols])
    return out


def quad_cv(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Day-grouped 5-fold CV of the phase 4 recipe on RflyMAD. The alarm
    threshold of each fold is the 99.9th percentile of no-fault scores from an
    inner 3-fold CV on that fold's training days only (no test data)."""
    # Some UAV-SEAD features never exist on RflyMAD (e.g. external-vision
    # residuals: no motion capture), and an all-NaN column breaks binning.
    cols = [c for c in MODEL_FEATURES if df[c].notna().any()]
    fl = df.groupby("flight_id").agg(s=("session", "first"), c=("flight_primary", "first")).reset_index()
    scores, alarm = np.full(len(df), np.nan), np.zeros(len(df), bool)
    ok = training_mask(df)
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                                 .split(fl, fl["c"], fl["s"])):
        trm = df["flight_id"].isin(set(fl.loc[tr, "flight_id"])).to_numpy()
        tem = df["flight_id"].isin(set(fl.loc[te, "flight_id"])).to_numpy()
        dtr = df[trm].reset_index(drop=True)
        inner = _cv_scores(dtr, cols, 3, 1 + k)
        normal = (dtr["flight_primary"] == "Normal").to_numpy()
        thr = float(np.quantile(inner[normal], 0.999))
        det = GbtMechanical().fit(df.loc[trm & ok, cols], df["label_anomaly"].to_numpy()[trm & ok])
        scores[tem] = det.score(df.loc[tem, cols])
        alarm[tem] = persistence(df[tem], scores[tem] >= thr)
    return scores, alarm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/rflymad"))
    ap.add_argument("--lake", type=Path, default=Path("data/lake_rflymad"))
    ap.add_argument("--models", type=Path, default=Path("data/models"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    if not args.skip_build:
        m = write_manifest(args.raw)
        print(f"manifest: {len(m['files'])} flights")
        cat = build_lake(args.raw, args.lake, workers=6)
        print(f"lake: {int(cat['usable'].sum())}/{len(cat)} usable")
    df = load_windows(args.lake)
    df = df[df["y_class"].isin(["Normal", "Mechanical"])].reset_index(drop=True)
    print(f"{df['flight_id'].nunique()} flights, {len(df)} windows, "
          f"{int(df['label_anomaly'].sum())} fault windows; airframes {df['airframe'].value_counts().to_dict()}")

    mech = joblib.load(args.models / "mechanical_gbt.joblib")
    X = df[MODEL_FEATURES]
    rows = []
    s_octo = mech["detector"].predict_proba(X)[:, 1]
    for op in ("threshold_f1", "threshold_fa"):
        rows.append({"experiment": f"D_uavsead_to_rflymad_real ({op})", **evaluate(df, s_octo, mech[op])})
    s_rule = PhysicsRule().score(X)
    rows.append({"experiment": "D_physics_rule (uavsead threshold)",
                 **evaluate(df, s_rule, mech["physics_rule_threshold_fa"])})
    s_cv, a_cv = quad_cv(df)
    y = df["label_anomaly"].to_numpy()
    rec, n = event_recall(df, a_cv)
    fa, hours = false_alarms_per_hour(df, a_cv)
    rows.append({"experiment": "D_rflymad_real_cv (same recipe, day-grouped, nested threshold)",
                 "ap": float(average_precision_score(y, s_cv)), "chance_ap": float(y.mean()),
                 "event_recall": rec, "fault_flights": n, "false_alarms_per_h": fa,
                 "normal_hours": hours, "threshold": np.nan})
    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "octo_to_quad.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
