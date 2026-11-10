"""Rebuild the 50 Hz MAVLink stream a flight *would* have produced, from its ULog.

Why: a real vehicle's telemetry radio is slow. RflyMAD's real tlogs carry
ATTITUDE at 15 Hz, ATTITUDE_TARGET at 2 Hz and SERVO_OUTPUT_RAW at 1 Hz, with
no quaternion or IMU messages (measured 2026-11-09). A detector that needs
50 Hz must run where the data is fast: on a companion computer wired to the
autopilot, or on a fast link. The ULog holds the onboard topics those
messages are made from, so this adapter emits the same messages, with the
same field meanings, on a 50 Hz grid, and feeds them to the unchanged
FrameBuilder. tests/test_stream.py and stream.validate_adapter check it
against a recorded tlog of the same SITL flight.

Mapping (PX4 mavlink module, v1.12-v1.17):
    ATTITUDE              vehicle_attitude.q -> roll, pitch; vehicle_angular_velocity.xyz
                          (v1.12 also has it; older logs: vehicle_attitude.*speed)
    ATTITUDE_QUATERNION   vehicle_attitude.q
    ATTITUDE_TARGET       vehicle_attitude_setpoint.q_d, |thrust_body|
    LOCAL_POSITION_NED    vehicle_local_position x, y, z, vx, vy, vz
    HIGHRES_IMU           sensor_combined.accelerometer_m_s2
    SERVO_OUTPUT_RAW      actuator_outputs.output[0..n]
    EXTENDED_SYS_STATE    vehicle_land_detected.landed -> 1 on ground, 2 in air

At each 20 ms tick every message carries the latest sample at or before the
tick (causal), and ATTITUDE goes last, so the frame sees fresh values.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyulog import ULog

TOPICS = ["vehicle_attitude", "vehicle_angular_velocity", "vehicle_attitude_setpoint",
          "vehicle_local_position", "sensor_combined", "actuator_outputs",
          "vehicle_land_detected"]
TICK_US = 20_000


class Msg(SimpleNamespace):
    def get_type(self) -> str:
        return self._type


def _get(ulog: ULog, name: str):
    try:
        return ulog.get_dataset(name, 0).data
    except (KeyError, IndexError, ValueError):
        return None


def _hold(t_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Index of the latest source sample at or before each grid time (-1 = none)."""
    return np.searchsorted(t_src, t_grid, side="right") - 1


def ulog_messages(path: Path, n_motors: int = 4):
    """Yield (msg, rx_ns) like read_tlog; rx_ns is the boot time in ns."""
    ulog = ULog(str(path), message_name_filter_list=TOPICS)
    att = _get(ulog, "vehicle_attitude")
    if att is None:
        return
    rates = _get(ulog, "vehicle_angular_velocity")
    sp = _get(ulog, "vehicle_attitude_setpoint")
    lp = _get(ulog, "vehicle_local_position")
    imu = _get(ulog, "sensor_combined")
    act = _get(ulog, "actuator_outputs")
    land = _get(ulog, "vehicle_land_detected")

    t0, t1 = int(att["timestamp"][0]), int(att["timestamp"][-1])
    grid = np.arange(t0 - t0 % TICK_US + TICK_US, t1, TICK_US)
    q = np.c_[att["q[0]"], att["q[1]"], att["q[2]"], att["q[3]"]]
    roll = np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2))
    pitch = np.arcsin(np.clip(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1))
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))

    def idx(d):
        return _hold(d["timestamp"], grid) if d is not None else np.full(len(grid), -1)

    i_att, i_rate, i_sp, i_lp, i_imu, i_act, i_land = map(idx, (att, rates, sp, lp, imu, act, land))
    if rates is not None:
        rate_xyz = np.c_[rates["xyz[0]"], rates["xyz[1]"], rates["xyz[2]"]]
    else:
        rate_xyz = np.c_[att["rollspeed"], att["pitchspeed"], att["yawspeed"]]
        i_rate = i_att
    if sp is not None:
        thrust = np.sqrt(sum(sp[f"thrust_body[{k}]"] ** 2 for k in range(3))) \
            if "thrust_body[0]" in sp else np.abs(sp.get("thrust", np.zeros(len(sp["timestamp"]))))

    for g, ia, ir, isp, ilp, ii, iact, il in zip(grid, i_att, i_rate, i_sp, i_lp, i_imu, i_act, i_land,
                                                 strict=True):
        t_ms = int(g // 1000)
        rx = int(g) * 1000
        if il >= 0:
            yield Msg(_type="EXTENDED_SYS_STATE", landed_state=1 if land["landed"][il] else 2), rx
        if ia < 0 or ir < 0:
            continue
        yield Msg(_type="ATTITUDE_QUATERNION", time_boot_ms=int(att["timestamp"][ia] // 1000),
                  q1=q[ia, 0], q2=q[ia, 1], q3=q[ia, 2], q4=q[ia, 3]), rx
        if isp >= 0:
            yield Msg(_type="ATTITUDE_TARGET", time_boot_ms=int(sp["timestamp"][isp] // 1000),
                      q=[sp[f"q_d[{k}]"][isp] for k in range(4)], thrust=float(thrust[isp])), rx
        if ilp >= 0:
            yield Msg(_type="LOCAL_POSITION_NED", time_boot_ms=int(lp["timestamp"][ilp] // 1000),
                      x=lp["x"][ilp], y=lp["y"][ilp], z=lp["z"][ilp],
                      vx=lp["vx"][ilp], vy=lp["vy"][ilp], vz=lp["vz"][ilp]), rx
        if ii >= 0:
            yield Msg(_type="HIGHRES_IMU", time_usec=int(imu["timestamp"][ii]),
                      xacc=imu["accelerometer_m_s2[0]"][ii], yacc=imu["accelerometer_m_s2[1]"][ii],
                      zacc=imu["accelerometer_m_s2[2]"][ii]), rx
        if iact >= 0:
            yield Msg(_type="SERVO_OUTPUT_RAW", time_usec=int(act["timestamp"][iact]), port=0,
                      **{f"servo{k + 1}_raw": float(act[f"output[{k}]"][iact]) for k in range(n_motors)}), rx
        yield Msg(_type="ATTITUDE", time_boot_ms=t_ms, roll=roll[ia], pitch=pitch[ia], yaw=yaw[ia],
                  rollspeed=rate_xyz[ir, 0], pitchspeed=rate_xyz[ir, 1],
                  yawspeed=rate_xyz[ir, 2]), rx
