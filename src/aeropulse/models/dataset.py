"""Load the modelling dataset (feature windows) from the lake.

Scope: usable flights, windows entirely in the air (airborne_frac == 1).
Ground segments are excluded because "on the ground" is trivially different
from flight and would inflate every metric.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from aeropulse.db import DB_NAME
from aeropulse.features import FEATURE_COLUMNS

# Inputs to models. airborne_frac is constant (1) after the in-air filter.
MODEL_FEATURES = [c for c in FEATURE_COLUMNS if c != "airborne_frac"]

# A window or flight with several classes is assigned the most specific one:
# a motor problem explains estimator anomalies more often than the reverse.
CLASS_PRIORITY = ["Mechanical", "Global Position", "Altitude", "External Position"]
CLASSES = ["Normal", *CLASS_PRIORITY]


def primary_class(label: str) -> str:
    if label == "Normal":
        return label
    parts = label.split(" + ")
    return min(parts, key=lambda p: CLASS_PRIORITY.index(p) if p in CLASS_PRIORITY else 99)


def load_windows(lake: Path) -> pd.DataFrame:
    con = duckdb.connect(str(lake / DB_NAME), read_only=True)
    cols = ", ".join(f'w."{c}"' for c in MODEL_FEATURES)
    df = con.execute(f"""
        select w.flight_id, w.t_us, w.label_anomaly, w.label_class, w.label_frac,
               f.session, f.flight_class, f.n_motors, {cols}
        from features w join flights f using (flight_id)
        where f.usable and w.airborne_frac = 1
        order by w.flight_id, w.t_us
    """).fetchdf()
    con.close()
    df[MODEL_FEATURES] = df[MODEL_FEATURES].astype(np.float32)
    df = df.copy()   # defragment before adding columns
    df["label_anomaly"] = df["label_anomaly"].astype(bool)
    df["y_class"] = df["label_class"].map(primary_class)
    df["flight_primary"] = df["flight_class"].map(primary_class)
    df["airframe"] = df["n_motors"].map({4: "quad", 8: "octo"}).fillna("other")
    return df


def load_ranges(lake: Path) -> pd.DataFrame:
    con = duckdb.connect(str(lake / DB_NAME), read_only=True)
    r = con.execute("select * from anomaly_ranges").fetchdf()
    con.close()
    return r
