"""Apply saved phase 4 models to another lake (e.g. PX4 SITL flights).

Per flight: when the fault was injected (exact, from the log), the first alarm
after it (latency), alarm onsets before it (false alarms), and the predicted
fault type at the first alarm.

Usage:
    uv run python -m aeropulse.models.apply --lake data/lake_sitl \
        --models data/models --out reports/phase5
"""

import argparse
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from aeropulse.db import DB_NAME
from aeropulse.models.dataset import MODEL_FEATURES, load_windows
from aeropulse.models.detectors import PhysicsRule
from aeropulse.models.evaluate import alarm_episodes, persistence


def score_windows(df: pd.DataFrame, models: Path) -> pd.DataFrame:
    mech = joblib.load(models / "mechanical_gbt.joblib")
    gen = joblib.load(models / "uavsead_gbt.joblib")
    X = df[MODEL_FEATURES]
    out = df[["flight_id", "t_us"]].copy()
    out["p_mech"] = mech["detector"].predict_proba(X)[:, 1]
    out["rule"] = PhysicsRule().score(X)
    out["alarm_gbt"] = persistence(df, out["p_mech"].to_numpy() >= mech["threshold_fa"])
    out["alarm_gbt_f1"] = out["p_mech"].to_numpy() >= mech["threshold_f1"]
    out["alarm_rule"] = persistence(df, out["rule"].to_numpy() >= mech["physics_rule_threshold_fa"])
    proba = gen["classifier"].predict_proba(X)
    out["type_pred"] = gen["classifier"].classes_[proba.argmax(1)]
    out["type_proba"] = proba.max(1)
    return out


def per_flight(df: pd.DataFrame, scored: pd.DataFrame, lake: Path) -> pd.DataFrame:
    con = duckdb.connect(str(lake / DB_NAME), read_only=True)
    flights = con.execute("select flight_id, key, flight_class from flights").fetchdf()
    ranges = con.execute("select flight_id, min(start_us) inject_us from anomaly_ranges "
                         "group by 1").fetchdf()
    con.close()
    info = flights.merge(ranges, on="flight_id", how="left").set_index("flight_id")
    rows = []
    for fid, idx in scored.groupby("flight_id").indices.items():
        s, w = scored.iloc[idx], df.iloc[idx]
        inj = info.loc[fid, "inject_us"]
        row = {"key": info.loc[fid, "key"], "class": info.loc[fid, "flight_class"],
               "in_air_s": len(s) * 0.2}
        for det in ("alarm_gbt", "alarm_gbt_f1", "alarm_rule"):
            a = s[det].to_numpy()
            pre = s["t_us"].to_numpy() < inj if pd.notna(inj) else np.ones(len(s), bool)
            row[f"{det}_false"] = alarm_episodes(w[pre], a[pre])
            if pd.notna(inj):
                hits = s[(s["t_us"] >= inj) & s[det]]
                row[f"{det}_latency_s"] = (hits["t_us"].iloc[0] - inj) / 1e6 if len(hits) else np.nan
                if det == "alarm_gbt" and len(hits):
                    row["type_at_alarm"] = hits["type_pred"].iloc[0]
                    row["type_proba"] = round(float(hits["type_proba"].iloc[0]), 2)
        if pd.notna(inj):
            post = s[s["t_us"] >= inj]
            row["in_air_after_inject_s"] = len(post) * 0.2
            row["max_p_mech_after"] = round(float(post["p_mech"].max()), 3) if len(post) else np.nan
        row["max_p_mech_before"] = round(float(s.loc[s["t_us"] < (inj if pd.notna(inj) else np.inf),
                                                     "p_mech"].max()), 3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("key")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = load_windows(args.lake)
    scored = score_windows(df, args.models)
    table = per_flight(df, scored, args.lake)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "uavsead_models_on_sitl.csv", index=False)
    scored.to_parquet(args.out / "sitl_scores.parquet", index=False)
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
