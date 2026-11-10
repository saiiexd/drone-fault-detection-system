"""Phase 6 streaming: frames, features, persistence, live/batch equivalence.

Synthetic tlogs are written with pymavlink's encoder, so these tests need no
downloaded or simulated data (they run in CI). The adapter check at the end
uses a real SITL flight and is skipped when the data is absent.
"""

from pathlib import Path

import joblib
import numpy as np
import pytest
from pymavlink.dialects.v20 import common as mavlink
from sklearn.ensemble import HistGradientBoostingClassifier

from aeropulse.stream.dataset import flight_frame_table
from aeropulse.stream.detector import LiveDetector
from aeropulse.stream.evaluate import Persistence, persist
from aeropulse.stream.features import (
    FEATURES,
    IDX,
    STALE_US,
    WINDOW,
    FrameBuilder,
    all_window_features,
    frames_from_messages,
)
from aeropulse.stream.mavlink_io import read_tlog

ROOT = Path(__file__).resolve().parents[1]


def write_tlog(path: Path, n_frames: int = 300, fail_at: int | None = None,
               t0_ms: int = 10_000, drop_lpos_after: int | None = None) -> None:
    """A hovering quad at 50 Hz; optionally a 'motor failure' (tilt, sink, thrust 1)."""
    mav = mavlink.MAVLink(None, srcSystem=1, srcComponent=1)
    rng = np.random.default_rng(0)
    with open(path, "wb") as f:
        def put(msg, t_ms):
            f.write(int(1_700_000_000e6 + t_ms * 1e3).to_bytes(8, "big") + msg.pack(mav))
        for i in range(n_frames):
            t = t0_ms + 20 * i
            failed = fail_at is not None and i >= fail_at
            k = (i - fail_at) if failed else 0
            roll = 0.02 * rng.standard_normal() + (0.05 * k if failed else 0.0)
            put(mavlink.MAVLink_extended_sys_state_message(0, 2), t)
            put(mavlink.MAVLink_attitude_target_message(
                t, 0, [1.0, 0.0, 0.0, 0.0], 0, 0, 0, 1.0 if failed else 0.7), t)
            if drop_lpos_after is None or i < drop_lpos_after:
                put(mavlink.MAVLink_local_position_ned_message(
                    t, 0, 0, -10 + (0.02 * k * k if failed else 0), 0, 0, 0.5 * k if failed else 0.0), t)
            put(mavlink.MAVLink_highres_imu_message(
                t * 1000, 0, 0, -3.0 if failed else -9.8, *([0] * 10), 0, 0), t)
            put(mavlink.MAVLink_servo_output_raw_message(
                t * 1000, 0, *([1000 if failed else 700] * 3 + [0 if failed else 700]),
                *([0] * 12)), t)
            q = [np.cos(roll / 2), np.sin(roll / 2), 0.0, 0.0]
            put(mavlink.MAVLink_attitude_quaternion_message(t, *q, 0, 0, 0, [0.0] * 4), t)
            put(mavlink.MAVLink_attitude_message(t, roll, 0.0, 0.0, 0.1 * k if failed else 0.0, 0, 0), t)


def test_persist_matches_online_filter():
    over = np.random.default_rng(1).random(2000) > 0.4
    p = Persistence(3)
    assert np.array_equal(persist(over, 3), np.array([p.step(bool(o)) for o in over]))


def test_tlog_roundtrip_and_frames(tmp_path):
    path = tmp_path / "f.tlog"
    write_tlog(path, n_frames=50)
    t_us, frames, _ = frames_from_messages(read_tlog(path))
    assert len(t_us) == 50
    assert np.all(np.diff(t_us) == 20_000)
    assert frames[-1, IDX["thrust_sp"]] == pytest.approx(0.7)
    assert frames[-1, IDX["m_max"]] == pytest.approx(0.7)       # 700 / 1000
    assert frames[-1, IDX["landed_state"]] == 2


def test_frames_are_causal_and_go_stale(tmp_path):
    path = tmp_path / "f.tlog"
    write_tlog(path, n_frames=100, drop_lpos_after=50)
    t_us, frames, _ = frames_from_messages(read_tlog(path))
    vz = frames[:, IDX["vz"]]
    stale_from = 50 + STALE_US // 20_000 + 1
    assert np.isfinite(vz[:50]).all()
    assert np.isnan(vz[stale_from:]).all()   # held value expires instead of lingering


def test_features_only_use_the_past(tmp_path):
    path = tmp_path / "f.tlog"
    write_tlog(path, n_frames=200, fail_at=150)
    _, frames, _ = frames_from_messages(read_tlog(path))
    full = all_window_features(frames)
    prefix = all_window_features(frames[:150])
    assert np.allclose(full[:150], prefix, equal_nan=True)


def test_live_replay_equals_batch(tmp_path):
    """The live detector must give the training pipeline's scores, frame by frame."""
    path = tmp_path / "f.tlog"
    write_tlog(path, n_frames=400, fail_at=300)
    df = flight_frame_table(path, "synthetic", t_inj=(10_000 + 20 * 300) * 1000)
    X, y = df[FEATURES].to_numpy(), df["y"].to_numpy()
    ok = np.isfinite(X).any(axis=1)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X[ok], y[ok])
    bundle = tmp_path / "m.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "persist": 3, "window": WINDOW}, bundle)

    batch = model.predict_proba(X)[:, 1]
    det = LiveDetector(bundle)
    live = [d.score for d in (det.update(m, rx) for m, rx in read_tlog(path)) if d is not None]
    live = np.array(live)
    assert len(live) == len(batch)
    assert np.isnan(live[:WINDOW - 1]).all()
    assert np.allclose(live[WINDOW - 1:], batch[WINDOW - 1:], atol=1e-6)


def test_session_reset_on_reboot(tmp_path):
    a, b = tmp_path / "a.tlog", tmp_path / "b.tlog"
    write_tlog(a, n_frames=100, t0_ms=50_000)
    write_tlog(b, n_frames=100, t0_ms=5_000)      # boot clock restarted
    joined = tmp_path / "ab.tlog"
    joined.write_bytes(a.read_bytes() + b.read_bytes())
    df = flight_frame_table(a, "a", None)
    X = df[FEATURES].to_numpy()
    ok = np.isfinite(X).any(axis=1)
    y = np.r_[np.zeros(ok.sum() // 2), np.ones(ok.sum() - ok.sum() // 2)].astype(bool)
    model = HistGradientBoostingClassifier(max_iter=5).fit(X[ok], y)
    bundle = tmp_path / "m.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "persist": 3, "window": WINDOW}, bundle)
    det = LiveDetector(bundle)
    out = [d for d in (det.update(m, rx) for m, rx in read_tlog(joined)) if d is not None]
    sessions = np.array([d.session for d in out])
    assert sessions[0] == 0 and sessions[-1] == 1
    first_b = int(np.argmax(sessions == 1))
    # the new session starts with an empty window again
    assert all(np.isnan(d.score) for d in out[first_b:first_b + WINDOW - 1])


def test_frame_builder_motor_scale():
    from aeropulse.stream.features import MotorScale

    class M:
        port = 0
        time_usec = 1

        def get_type(self):
            return "SERVO_OUTPUT_RAW"
    m = M()
    for i, v in enumerate([1500, 2000, 1000, 1250], 1):
        setattr(m, f"servo{i}_raw", v)
    fb = FrameBuilder(MotorScale(1000, 2000, 4))
    fb.update(m)
    assert fb.val[IDX["m_max"]] == pytest.approx(1.0)
    assert fb.val[IDX["m_sat_frac"]] == pytest.approx(0.25)


SITL_FLIGHT = ROOT / "data/raw/sitl6/p6_motor_off_5"


@pytest.mark.skipif(not (SITL_FLIGHT / "telemetry.tlog").exists(), reason="SITL data not present")
def test_ulog_adapter_matches_recorded_stream():
    from aeropulse.stream.ulog_adapter import ulog_messages
    tA, fA, _ = frames_from_messages(read_tlog(SITL_FLIGHT / "telemetry.tlog"))
    tB, fB, _ = frames_from_messages(ulog_messages(next((SITL_FLIGHT / "log").rglob("*.ulg"))))
    FA, FB = all_window_features(fA), all_window_features(fB)
    j = np.clip(np.searchsorted(tB, tA), 0, len(tB) - 1)
    ok = np.abs(tB[j] - tA) <= 10_000
    for name in ["tilt_err__mean", "thrust_sp__mean", "vz__mean", "rate_rp__max", "acc_z__mean"]:
        i = FEATURES.index(name)
        a, b = FA[ok, i], FB[j[ok], i]
        m = np.isfinite(a) & np.isfinite(b)
        assert np.corrcoef(a[m], b[m])[0, 1] > 0.95, name


def test_euler_fallback_matches_quaternion():
    """Without ATTITUDE_QUATERNION (telemetry radio), the attitude is rebuilt
    from ATTITUDE's Euler angles; it must equal the true quaternion."""
    from aeropulse.stream.features import euler_to_quat
    rng = np.random.default_rng(3)
    for roll, pitch, yaw in rng.uniform([-1, -1, -3], [1, 1, 3], size=(20, 3)):
        w, x, y, z = euler_to_quat(roll, pitch, yaw)
        assert np.isclose(w * w + x * x + y * y + z * z, 1.0)
        # back to Euler with the same formulas PX4/MAVLink use
        assert np.isclose(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), roll)
        assert np.isclose(np.arcsin(2 * (w * y - z * x)), pitch)
        assert np.isclose(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)), yaw)


def test_radio_live_replay_equals_batch(tmp_path):
    """Radio mode: settings stored in the bundle; live scores on the thinned
    stream equal the batch (training) scores."""
    from aeropulse.stream.dataset import RADIO_HZ, RADIO_STALE_US, RADIO_WINDOW
    from aeropulse.stream.mavlink_io import decimate
    path = tmp_path / "f.tlog"
    write_tlog(path, n_frames=1500, fail_at=1200)
    df = flight_frame_table(path, "synthetic", t_inj=(10_000 + 20 * 1200) * 1000, radio=True)
    assert 1e6 / np.mean(np.diff(df["t_us"])) == pytest.approx(15, rel=0.02)   # 15 Hz on average
    X, y = df[FEATURES].to_numpy(copy=True), df["y"].to_numpy()
    ok = np.isfinite(X).any(axis=1)
    X[:, np.isnan(X).all(axis=0)] = 0.0
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(X[ok], y[ok])
    bundle = tmp_path / "m.joblib"
    joblib.dump({"model": model, "threshold": 0.5, "persist": 3, "window": RADIO_WINDOW,
                 "rate_hz": RADIO_HZ, "stale_us": RADIO_STALE_US, "radio": True}, bundle)
    batch = model.predict_proba(df[FEATURES].to_numpy())[:, 1]
    det = LiveDetector(bundle)
    live = np.array([d.score for d in (det.update(m, rx) for m, rx in decimate(read_tlog(path)))
                     if d is not None])
    assert len(live) == len(batch)
    assert np.allclose(live[RADIO_WINDOW - 1:], batch[RADIO_WINDOW - 1:], atol=1e-6)


def test_decimate_hits_radio_rates(tmp_path):
    from aeropulse.stream.mavlink_io import decimate
    path = tmp_path / "f.tlog"
    write_tlog(path, n_frames=1500)                       # 30 s at 50 Hz
    counts = {}
    for m, _ in decimate(read_tlog(path)):
        counts[m.get_type()] = counts.get(m.get_type(), 0) + 1
    assert counts["ATTITUDE"] == pytest.approx(15 * 30, abs=2)
    assert counts["ATTITUDE_TARGET"] == pytest.approx(2 * 30, abs=1)
    assert counts["SERVO_OUTPUT_RAW"] == pytest.approx(30, abs=1)
    assert "ATTITUDE_QUATERNION" not in counts and "HIGHRES_IMU" not in counts


def test_browser_payloads_are_strict_json():
    import json

    from aeropulse.stream.service import browser_safe
    rec = {"score": float("nan"), "display": {"acc_z": float("nan"), "alt": 3.0}, "x": [float("inf"), 1.0]}
    out = json.dumps(browser_safe(rec), allow_nan=False)        # raises if any NaN/inf survives
    assert json.loads(out) == {"score": None, "display": {"acc_z": None, "alt": 3.0}, "x": [None, 1.0]}
