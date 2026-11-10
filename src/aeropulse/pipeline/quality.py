"""Flight-level data-quality rules, applied to the catalog.

Each rule is a boolean column ``q_<name>``; ``usable`` is their AND. Flights
that fail are kept in the catalog (with the reason visible) but are excluded
from modelling by filtering on ``usable``. Thresholds are deliberately simple
and listed here so they are easy to audit and change.
"""

import pandas as pd

MIN_DURATION_S = 10.0     # shorter logs are mostly aborted takeoffs
MAX_GAP_S = 1.0           # a gap in a required topic longer than this
MAX_NAN_FRAC = 0.05       # share of rows with any required signal missing
MIN_AIRBORNE_FRAC = 0.2   # a "flight" that never leaves the ground


def apply_rules(cat: pd.DataFrame) -> pd.DataFrame:
    cat = cat.copy()
    cat["q_parsed"] = cat["error"].isna()
    cat["q_required_signals"] = cat["missing_required"].isna() & cat["q_parsed"]
    cat["q_duration"] = cat["duration_s"].fillna(0) >= MIN_DURATION_S
    cat["q_no_long_gap"] = cat["max_gap_s"].fillna(float("inf")) <= MAX_GAP_S
    cat["q_nan_frac"] = cat["nan_frac_required"].fillna(1.0) <= MAX_NAN_FRAC
    cat["q_airborne"] = cat["airborne_frac"].fillna(0) >= MIN_AIRBORNE_FRAC
    cat["q_labels_in_span"] = cat["ranges_in_span"].fillna(False).astype(bool)
    cat["q_motor_count_known"] = cat["n_motors"].notna()
    cat["q_anomaly_has_ranges"] = (cat["flight_class"] == "Normal") | (cat["n_ranges"].fillna(0) > 0)
    rules = [c for c in cat.columns if c.startswith("q_")]
    cat["usable"] = cat[rules].all(axis=1)
    cat["failed_rules"] = cat[rules].apply(
        lambda r: ",".join(c[2:] for c in rules if not r[c]) or None, axis=1)
    return cat
