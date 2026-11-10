"""Phase 7: RflyMAD label parsing and the event metric for non-crashing faults."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeropulse.sim2real.rflymad import fault_info, flight_table, recording_day
from aeropulse.sim2real.run import events, prepare

ROOT = Path(__file__).resolve().parents[1]
REAL_407 = ROOT / "data/raw/rflymad/Real-Motor/acce/407_1/log_28_2023-6-2-10-16-50/log_28_2023-6-2-10-16-50.ulg"


def test_recording_day_from_log_name():
    assert recording_day(Path("log_28_2023-6-2-10-16-50.ulg")) == "2023-6-2"
    assert recording_day(Path("log_105_2023-12-31-04-55-26.ulg")) == "2023-12-31"
    assert recording_day(Path("weird.ulg")) == "unknown"


@pytest.mark.skipif(not REAL_407.exists(), reason="RflyMAD real flight not downloaded")
def test_fault_label_from_rfly_topic():
    # Values read from the flight's rfly_ctrl_lxl topic and its TestInfo sheet
    # (fault parameter [1, 0.85, 1, 1, 1], case 407).
    info = fault_info(REAL_407)
    assert info["t_on"] == 134_923_291
    assert info["t_off"] == 149_218_257
    assert info["severity"] == pytest.approx(0.15, abs=1e-4)
    assert info["n_failed"] == 1
    assert info["fault_mode"] == "step"
    assert (info["pwm_min"], info["pwm_max"]) == (1000.0, 1950.0)


def _flight(key, fault, n=1000, t_on=400, t_off=700):
    t = np.arange(n) * 20_000
    y = (np.arange(n) >= t_on) & (np.arange(n) < t_off) if fault else np.zeros(n, bool)
    return pd.DataFrame({
        "key": key, "t_us": t, "y": y, "use": True, "airborne": True,
        "t_inject_us": t_on * 20_000 if fault else pd.NA,
        "t_fault_end_us": t_off * 20_000 if fault else pd.NA,
        "severity": 0.2 if fault else 0.0, "fault_mode": "step" if fault else None,
        "flight_mode": "hover", "group": "d1",
        "loss": np.where(y, 0.2, 0.0)})


def test_events_detection_latency_and_false_alarms():
    df = prepare(pd.concat([_flight("f", True), _flight("n", False)], ignore_index=True))
    alarm = np.zeros(len(df), bool)
    alarm[450:460] = True             # in the fault window: 50 frames = 1.0 s after onset
    alarm[1000 + 100:1000 + 110] = True   # one episode on the no-fault flight
    alarm[720:725] = True             # right after the fault cleared: inside the grace period
    ev = events(df, alarm).set_index("key")
    assert ev.loc["f", "detected"]
    assert ev.loc["f", "latency_s"] == pytest.approx(1.0)
    assert ev.loc["f", "loss_at_alarm"] == pytest.approx(0.2)
    assert ev.loc["f", "false_alarms"] == 0    # post-fault grace is not counted
    assert ev.loc["n", "false_alarms"] == 1


@pytest.mark.skipif(not REAL_407.exists(), reason="RflyMAD real flight not downloaded")
def test_radio_stream_table_uses_ulog_labels():
    from aeropulse.sim2real.run import frame_s
    tlog = next(REAL_407.parents[1].glob("*.tlog"))
    df = flight_table(REAL_407, "Real-Motor/acce/407_1", "real", tlog_path=tlog,
                      window=8, rate_hz=15, stale_us=1_200_000)
    assert frame_s(df) == pytest.approx(1 / 15, rel=0.05)        # the radio's ATTITUDE rate
    pos = df.loc[df["y"], "t_us"]
    assert pos.min() >= 134_923_291 and pos.max() < 149_218_257  # inside the ULog fault window
    assert df.loc[df["use"], "thrust_sp__last"].notna().all()     # 2 Hz setpoint held, not NaN
