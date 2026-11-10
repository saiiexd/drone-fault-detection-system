"""Window features from the cleaned 50 Hz timeline.

Batch (phase 3/4) and streaming (phase 6) both call ``window_features`` on
the same cleaned columns. In streaming, it runs on a ring buffer of the last
WINDOW_ROWS rows and takes the last row. One implementation means no
training-serving skew: the live model sees exactly the features it was
trained on.

Design choices:
- Causal trailing windows: a feature at time t uses rows in (t - 2 s, t].
- Motor features are aggregated across however many motors the vehicle has
  (max, min, spread, saturation fraction), so models do not depend on motor
  count or numbering (UAV-SEAD is an octorotor; RflyMAD and SITL are quads).
- Absolute position and heading are excluded: they describe where and in
  which direction a flight happened, which identifies the session rather than
  the fault and would leak across flights from the same day.
"""

import numpy as np
import pandas as pd

from aeropulse.signals import MAX_MOTORS

RATE_HZ = 50
WINDOW_ROWS = 100          # 2.0 s
MIN_ROWS = 90              # tolerate up to 10% NaN rows inside a window
HOP_ROWS = 10              # 0.2 s between batch windows
SAT_HI = 0.95              # normalised motor command treated as saturated

MOTOR_N = [f"motor_{i}_n" for i in range(1, MAX_MOTORS + 1)]

STAT_COLS = [
    "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z", "acc_vib", "gyro_vib",
    "ctrl_roll", "ctrl_pitch", "ctrl_yaw", "ctrl_thrust",
    "rate_roll", "rate_pitch", "rate_yaw", "roll", "pitch", "tilt",
    "vel_x", "vel_y", "vel_z",
    "tr_vel", "tr_pos", "tr_hgt", "tr_mag",
    "m_max", "m_min", "m_mean", "m_spread", "m_sat_frac",
    "ext_res_x", "ext_res_y", "ext_res_z", "batt_v",
]
STATS = ["mean", "std", "min", "max"]


def row_derived(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Per-row quantities computed from one row only (no time context)."""
    d = pd.DataFrame(index=cleaned.index)
    motors = cleaned[MOTOR_N]
    d["m_max"] = motors.max(axis=1)
    d["m_min"] = motors.min(axis=1)
    d["m_mean"] = motors.mean(axis=1)
    d["m_spread"] = d["m_max"] - d["m_min"]
    n_active = motors.notna().sum(axis=1)
    d["m_sat_frac"] = (motors >= SAT_HI).sum(axis=1) / n_active.where(n_active > 0)
    for a in "xyz":   # estimate minus external measurement
        d[f"ext_res_{a}"] = (cleaned[f"pos_{a}"] - cleaned[f"ext_{a}"]).abs()
    d["alt"] = -cleaned["pos_z"]          # NED z is down
    d["airborne"] = (cleaned["landed"] == 0).astype(float).where(cleaned["landed"].notna())
    return d


def window_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Trailing-window features for every row (NaN until a window is full)."""
    base = pd.concat([cleaned, row_derived(cleaned)], axis=1)
    roll = base[STAT_COLS].rolling(WINDOW_ROWS, min_periods=MIN_ROWS)
    parts = [roll.agg(s).add_suffix(f"__{s}") for s in STATS]
    extra = pd.DataFrame({
        "alt_delta": base["alt"] - base["alt"].shift(WINDOW_ROWS - 1),
        "airborne_frac": base["airborne"].rolling(WINDOW_ROWS, min_periods=MIN_ROWS).mean(),
    }, index=base.index)
    return pd.concat([*parts, extra], axis=1)


def batch_features(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Windows every HOP_ROWS rows, labelled at the window end (decision time)."""
    feats = window_features(cleaned)
    idx = np.arange(WINDOW_ROWS - 1, len(cleaned), HOP_ROWS)
    out = feats.iloc[idx].copy()
    out.insert(0, "t_us", cleaned["t_us"].iloc[idx].to_numpy())
    out.insert(1, "t_s", cleaned["t_s"].iloc[idx].to_numpy())
    out["label_anomaly"] = cleaned["label_anomaly"].iloc[idx].to_numpy()
    out["label_class"] = cleaned["label_class"].iloc[idx].to_numpy()
    out["label_frac"] = (cleaned["label_anomaly"].astype(float)
                         .rolling(WINDOW_ROWS, min_periods=1).mean().iloc[idx].to_numpy())
    feature_cols = [c for c in out.columns if "__" in c] + ["alt_delta"]
    return out[out[feature_cols].notna().any(axis=1)].reset_index(drop=True)


FEATURE_COLUMNS = [f"{c}__{s}" for s in STATS for c in STAT_COLS] + ["alt_delta", "airborne_frac"]
