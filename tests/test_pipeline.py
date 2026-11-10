"""Unit tests for pipeline logic (synthetic data) + one end-to-end flight."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeropulse.features import WINDOW_ROWS, window_features
from aeropulse.pipeline.flight import label_rows, process_flight, quat_to_euler, resolve_signals
from aeropulse.pipeline.quality import apply_rules
from aeropulse.signals import MAX_MOTORS, SIGNAL_NAMES

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "uav_sead"


def test_label_rows_overlapping_ranges():
    t = np.arange(0, 10)
    anomaly, cls = label_rows(t, [("A", "s", 2, 4), ("B", "s", 4, 6)])
    assert anomaly.tolist() == [False, False, True, True, True, True, True, False, False, False]
    assert cls[3] == "A" and cls[4] == "A + B" and cls[6] == "B" and cls[0] == "Normal"


def test_quat_identity_and_roll():
    r, p, y, tilt = quat_to_euler(1.0, 0.0, 0.0, 0.0)
    assert (r, p, y, tilt) == (0, 0, 0, 0)
    a = np.pi / 6   # 30 deg roll
    r, p, y, tilt = quat_to_euler(np.cos(a / 2), np.sin(a / 2), 0.0, 0.0)
    assert np.isclose(r, a) and np.isclose(tilt, a) and np.isclose(p, 0)


def test_resolve_signals_falls_back_to_newer_px4_topic():
    frames = {"vehicle_angular_velocity_0": pd.DataFrame({"timestamp": [0], "xyz[0]": [1.0]})}
    assert resolve_signals(frames)["rate_roll"] == ("vehicle_angular_velocity_0", "xyz[0]", 1.0)
    frames["vehicle_attitude_0"] = pd.DataFrame({"timestamp": [0], "rollspeed": [1.0]})
    assert resolve_signals(frames)["rate_roll"] == ("vehicle_attitude_0", "rollspeed", 1.0)


def test_resolve_signals_applies_thrust_sign_for_new_px4():
    frames = {"vehicle_thrust_setpoint_0": pd.DataFrame({"timestamp": [0], "xyz[2]": [-0.6]})}
    assert resolve_signals(frames)["ctrl_thrust"] == ("vehicle_thrust_setpoint_0", "xyz[2]", -1.0)


def test_leading_gap_is_found_only_near_start():
    from aeropulse.pipeline.flight import leading_gap_end
    early = pd.DataFrame({"timestamp": np.r_[0:1_000_000:4_000, 4_000_000:60_000_000:4_000]})
    late = pd.DataFrame({"timestamp": np.r_[0:30_000_000:4_000, 33_000_000:60_000_000:4_000]})
    assert leading_gap_end({"a": early}, ["a"]) == 4_000_000
    assert leading_gap_end({"a": late}, ["a"]) is None   # mid-flight gap: not trimmed


def synthetic_cleaned(n=300, n_motors=4):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({c: rng.normal(size=n) for c in SIGNAL_NAMES})
    for c in ["acc_vib", "gyro_vib", "roll", "pitch", "yaw", "tilt"]:
        df[c] = rng.normal(size=n)
    for i in range(1, MAX_MOTORS + 1):
        df[f"motor_{i}_n"] = rng.uniform(size=n) if i <= n_motors else np.nan
    df["landed"] = 0.0
    return df


def test_window_features_are_causal():
    df = synthetic_cleaned()
    full = window_features(df)
    cut = 200
    changed = df.copy()
    changed.iloc[cut + 1:, :] = 999.0   # change only the future
    partial = window_features(changed)
    pd.testing.assert_frame_equal(full.iloc[: cut + 1], partial.iloc[: cut + 1])


def test_window_features_need_full_window():
    feats = window_features(synthetic_cleaned())
    assert feats["gyro_x__mean"].iloc[:89].isna().all()
    assert feats["gyro_x__mean"].iloc[WINDOW_ROWS - 1:].notna().all()


def test_motor_features_independent_of_motor_count():
    quad, octo = synthetic_cleaned(n_motors=4), synthetic_cleaned(n_motors=8)
    for df in (quad, octo):
        for i in range(1, MAX_MOTORS + 1):
            col = f"motor_{i}_n"
            df[col] = df[col].where(df[col].isna(), 1.0)   # all active motors saturated
    for df in (quad, octo):
        assert (window_features(df)["m_sat_frac__mean"].dropna() == 1.0).all()


def test_quality_rules_flag_reasons():
    cat = pd.DataFrame([
        {"error": None, "missing_required": None, "duration_s": 30, "max_gap_s": 0.1,
         "nan_frac_required": 0.0, "airborne_frac": 0.9, "ranges_in_span": True,
         "flight_class": "Normal", "n_ranges": 0, "n_motors": 8},
        {"error": None, "missing_required": None, "duration_s": 30, "max_gap_s": 5.0,
         "nan_frac_required": 0.2, "airborne_frac": 0.9, "ranges_in_span": True,
         "flight_class": "Mechanical", "n_ranges": 0, "n_motors": 8},
    ])
    out = apply_rules(cat)
    assert out["usable"].tolist() == [True, False]
    assert set(out.loc[1, "failed_rules"].split(",")) == {"no_long_gap", "nan_frac",
                                                          "anomaly_has_ranges"}


@pytest.mark.skipif(not (RAW / "manifest.json").exists(), reason="no data")
def test_process_flight_end_to_end(tmp_path):
    entries = json.loads((RAW / "manifest.json").read_text())["files"]
    entry = next((e for e in entries if e["key"] == "2019-01-15/07_28_00"), None)
    if entry is None:
        pytest.skip("reference flight not in manifest")
    if not (RAW / entry["path"]).exists():
        pytest.skip("reference flight .ulg not on disk")
    row = process_flight(entry, RAW, tmp_path)
    assert row["error"] is None and row["missing_required"] is None
    assert row["n_motors"] == 8
    fid = row["flight_id"]
    cleaned = pd.read_parquet(tmp_path / "cleaned" / f"flight_id={fid}" / "part-0.parquet")
    assert (np.diff(cleaned["t_us"]) == 20_000).all()
    labelled_s = cleaned["label_anomaly"].sum() / 50
    range_s = sum(e - s for a in entry["annotations"] for _, rr in a["ranges"] for s, e in rr) / 1e6
    assert abs(labelled_s - range_s) < 0.1
    feats = pd.read_parquet(tmp_path / "features" / f"flight_id={fid}" / "part-0.parquet")
    assert len(feats) == row["n_windows"] > 0


def test_failed_motor_nan_counts_as_zero_thrust():
    from aeropulse.pipeline.flight import build_cleaned, resolve_signals
    t = np.arange(0, 5_000_000, 4_000, dtype=np.int64)
    n = len(t)

    def topic(**cols):
        return pd.DataFrame({"timestamp": t, **{k: np.full(n, v, float) for k, v in cols.items()}})

    motors = topic(**{f"control[{i}]": 0.7 for i in range(4)})
    motors.loc[n // 2:, "control[0]"] = np.nan          # motor 1 stops half-way
    frames = {
        "sensor_combined_0": topic(**{f"gyro_rad[{i}]": 0.0 for i in range(3)},
                                   **{f"accelerometer_m_s2[{i}]": 0.0 for i in range(3)}),
        "vehicle_torque_setpoint_0": topic(**{f"xyz[{i}]": 0.0 for i in range(3)}),
        "vehicle_thrust_setpoint_0": topic(**{"xyz[2]": -0.7}),
        "actuator_motors_0": motors,
        "vehicle_attitude_0": topic(**{"q[0]": 1.0, "q[1]": 0.0, "q[2]": 0.0, "q[3]": 0.0}),
        "vehicle_local_position_0": topic(x=0.0, y=0.0, z=-10.0, vx=0.0, vy=0.0, vz=0.0),
    }
    meta = {"n_motors": 4, "n_outputs": 16, "pwm_min": None, "pwm_max": None}
    out = build_cleaned(frames, resolve_signals(frames), meta, [])
    assert np.isclose(out["ctrl_thrust"].iloc[10], 0.7)          # sign flipped
    assert np.isclose(out["motor_1_n"].iloc[10], 0.7)            # already normalised
    assert out["motor_1_n"].iloc[-1] == 0.0                      # stopped -> 0, not NaN
    assert out["motor_5_n"].isna().all()                         # quad: no 5th motor


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/raw/sitl/manifest.json").exists(),
                    reason="no SITL data")
def test_sitl_label_starts_at_logged_injection_command():
    from aeropulse.sim.scenarios import injection_time_us
    sitl = Path(__file__).resolve().parents[1] / "data" / "raw" / "sitl"
    entries = json.loads((sitl / "manifest.json").read_text())["files"]
    faulty = [e for e in entries if e["class"] != "Normal"]
    if not faulty:
        pytest.skip("no fault runs yet")
    e = faulty[0]
    start = e["annotations"][0]["ranges"][0][1][0][0]
    assert start == injection_time_us(sitl / e["path"])


def test_state_topics_never_go_stale():
    from aeropulse.pipeline.flight import max_age
    once = pd.DataFrame({"timestamp": np.array([0, 50_000_000], dtype=np.int64)})
    assert max_age(once, "vehicle_land_detected_0") >= 3600
    fast = pd.DataFrame({"timestamp": np.arange(0, 1_000_000, 4_000, dtype=np.int64)})
    assert max_age(fast, "sensor_combined_0") == pytest.approx(0.05)   # clamped minimum
