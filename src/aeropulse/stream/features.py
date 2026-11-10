"""Fast features for live motor-failure detection from the MAVLink stream.

Phase 5 showed that a quadcopter that loses a motor hits the ground about
1.4 s later, while the phase 4 features used 2 s windows. So this module
uses 0.5 s windows on 50 Hz frames built from MAVLink only.

Two steps, each with one implementation shared by training and live use:

1. ``FrameBuilder`` turns MAVLink messages into one frame (a fixed vector of
   signals) per ATTITUDE message, the 50 Hz clock. Every other signal is the
   latest value received so far (sample-and-hold), so a frame at time t only
   uses data from <= t (causal). Values older than STALE_US become NaN.
2. ``window_features`` maps a (n, WINDOW, n_signals) array of frame windows
   to (n, n_features). Training calls it on all windows of a flight at once;
   the live detector calls it on the single latest window. Same code, so the
   live model sees exactly the features it was trained on.

Signals deliberately NOT used as features:
- The minimum motor output, motor spread and mean. In SITL the injected
  motor's output is forced to 0 by the failure injector itself, so these
  would encode the label rather than the physics. A real broken propeller
  or ESC does not zero the commanded output. Only the maximum output and
  the saturated fraction are kept: healthy motors saturating while they
  compensate is a physical response.
- Absolute position, altitude and heading: they identify where a flight
  happened, not whether a motor failed.
"""

from dataclasses import dataclass

import numpy as np

RATE_HZ = 50
WINDOW = 25            # 0.5 s of frames
STALE_US = 250_000     # a 50 Hz signal unseen for 0.25 s is treated as missing
SAT_HI = 0.95

SIGNALS = [
    "rate_roll", "rate_pitch", "rate_yaw", "roll", "pitch",      # ATTITUDE
    "qw", "qx", "qy", "qz",                                       # ATTITUDE_QUATERNION
    "sp_qw", "sp_qx", "sp_qy", "sp_qz", "thrust_sp",               # ATTITUDE_TARGET
    "vx", "vy", "vz", "z",                                        # LOCAL_POSITION_NED
    "acc_x", "acc_y", "acc_z",                                    # HIGHRES_IMU
    "m_max", "m_sat_frac",                                        # SERVO_OUTPUT_RAW
    "landed_state",                                               # EXTENDED_SYS_STATE
]
IDX = {s: i for i, s in enumerate(SIGNALS)}

# message type -> (signals it sets, time field)
_SOURCES = {
    "ATTITUDE_QUATERNION": ("qw", "qx", "qy", "qz"),
    "ATTITUDE_TARGET": ("sp_qw", "sp_qx", "sp_qy", "sp_qz", "thrust_sp"),
    "LOCAL_POSITION_NED": ("vx", "vy", "vz", "z"),
    "HIGHRES_IMU": ("acc_x", "acc_y", "acc_z"),
    "SERVO_OUTPUT_RAW": ("m_max", "m_sat_frac"),
}


@dataclass
class MotorScale:
    """Maps raw SERVO_OUTPUT_RAW values to 0..1. Gazebo SITL: 0..1000; real
    PX4 hardware: PWM microseconds, typically 1000..2000."""
    lo: float = 0.0
    hi: float = 1000.0
    n_motors: int = 4


def euler_to_quat(roll, pitch, yaw):
    """ZYX Euler angles (MAVLink ATTITUDE) -> quaternion (w, x, y, z)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)


class FrameBuilder:
    """stale_us: how long a held value stays valid. 0.25 s suits the 50 Hz
    SITL/companion link; a telemetry radio sending some messages at 1-2 Hz
    needs more (phase 7 uses 1.2 s), or those signals would be NaN most of the
    time."""

    def __init__(self, motor_scale: MotorScale | None = None, stale_us: int = STALE_US):
        self.ms = motor_scale or MotorScale()
        self.stale_us = stale_us
        self.val = np.full(len(SIGNALS), np.nan)
        self.t_set = np.full(len(SIGNALS), -np.inf)   # boot-clock µs of each value
        self.state_sticky = {IDX["landed_state"]}

    def _set(self, names, values, t_us: float) -> None:
        for n, v in zip(names, values, strict=True):
            self.val[IDX[n]] = v
            self.t_set[IDX[n]] = t_us

    def update(self, msg) -> tuple[int, np.ndarray] | None:
        """Feed one message. Returns (t_us, frame) on each ATTITUDE message."""
        t = msg.get_type()
        if t == "ATTITUDE":
            t_us = int(msg.time_boot_ms) * 1000
            self._set(("rate_roll", "rate_pitch", "rate_yaw", "roll", "pitch"),
                      (msg.rollspeed, msg.pitchspeed, msg.yawspeed, msg.roll, msg.pitch), t_us)
            # A telemetry radio may send no ATTITUDE_QUATERNION: then the
            # attitude quaternion is rebuilt from this message's Euler angles.
            # Streams that carry the quaternion are unaffected.
            q_age = t_us - self.t_set[IDX["qw"]]
            if (q_age > self.stale_us or q_age < -self.stale_us) and hasattr(msg, "yaw"):
                self._set(("qw", "qx", "qy", "qz"), euler_to_quat(msg.roll, msg.pitch, msg.yaw), t_us)
            frame = self.val.copy()
            age = t_us - self.t_set
            stale = (age > self.stale_us) | (age < -self.stale_us)   # negative: clock restarted
            for i in self.state_sticky:   # state messages are slow (1 Hz) and never go stale
                stale[i] = False
            frame[stale] = np.nan
            return t_us, frame
        if t == "ATTITUDE_QUATERNION":
            self._set(_SOURCES[t], (msg.q1, msg.q2, msg.q3, msg.q4), msg.time_boot_ms * 1000)
        elif t == "ATTITUDE_TARGET":
            q = msg.q
            self._set(_SOURCES[t], (q[0], q[1], q[2], q[3], msg.thrust), msg.time_boot_ms * 1000)
        elif t == "LOCAL_POSITION_NED":
            self._set(_SOURCES[t], (msg.vx, msg.vy, msg.vz, msg.z), msg.time_boot_ms * 1000)
        elif t == "HIGHRES_IMU":
            self._set(_SOURCES[t], (msg.xacc, msg.yacc, msg.zacc), msg.time_usec)
        elif t == "SERVO_OUTPUT_RAW" and msg.port == 0:
            raw = np.array([getattr(msg, f"servo{i}_raw") for i in range(1, self.ms.n_motors + 1)],
                           dtype=float)
            m = np.clip((raw - self.ms.lo) / (self.ms.hi - self.ms.lo), 0.0, 1.0)
            self._set(_SOURCES[t], (m.max(), (m >= SAT_HI).mean()), msg.time_usec)
        elif t == "EXTENDED_SYS_STATE":
            self.val[IDX["landed_state"]] = msg.landed_state
        return None


def body_z(qw, qx, qy, qz):
    """Body z axis (thrust direction, reversed) expressed in the NED world frame."""
    return np.stack([2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx),
                     1 - 2 * (qx ** 2 + qy ** 2)], axis=-1)


def derived(frames: np.ndarray) -> dict[str, np.ndarray]:
    """Per-frame physical quantities; frames has shape (..., n_signals)."""
    g = lambda s: frames[..., IDX[s]]  # noqa: E731
    z_act = body_z(g("qw"), g("qx"), g("qy"), g("qz"))
    z_sp = body_z(g("sp_qw"), g("sp_qx"), g("sp_qy"), g("sp_qz"))
    cos_err = np.clip((z_act * z_sp).sum(-1), -1.0, 1.0)
    return {
        "tilt": np.degrees(np.arccos(np.clip(np.cos(g("roll")) * np.cos(g("pitch")), -1, 1))),
        # Angle between where the controller wants the thrust axis and where
        # it is: large when the vehicle cannot follow its attitude setpoint.
        "tilt_err": np.degrees(np.arccos(cos_err)),
        "rate_rp": np.hypot(g("rate_roll"), g("rate_pitch")),
        "yaw_rate": np.abs(g("rate_yaw")),
        "vz": g("vz"),                         # NED: positive = sinking
        "acc_z": g("acc_z"),                   # specific force, about -9.8 in hover
        "thrust_sp": g("thrust_sp"),
        "m_max": g("m_max"),
        "m_sat_frac": g("m_sat_frac"),
    }


STATS = ("last", "mean", "min", "max")
# Acceleration keeps only its window mean. HIGHRES_IMU carries raw single
# accelerometer samples, while the ULog (used to rebuild streams for real
# flights, see ulog_adapter) holds averaged ones: on the same SITL flight their
# instantaneous values correlate poorly (acc_xy last r = 0.18), the 0.5 s mean
# of acc_z agrees (r > 0.97). Vibration-sensitive extremes would also differ
# between vehicles. Lateral acceleration is dropped for the same reason.
STAT_SPEC = {s: STATS for s in ["tilt", "tilt_err", "rate_rp", "yaw_rate", "vz",
                                "thrust_sp", "m_max", "m_sat_frac"]}
STAT_SPEC["acc_z"] = ("mean",)
FEATURES = ([f"{s}__{st}" for s, sts in STAT_SPEC.items() for st in sts]
            + ["tilt__delta", "vz__delta", "rate_rp__dmax", "yaw_rate__dmax"])
_REDUCE = {"last": lambda x: x[:, -1], "mean": lambda x: np.nanmean(x, 1),
           "min": lambda x: np.nanmin(x, 1), "max": lambda x: np.nanmax(x, 1)}


def window_features(win: np.ndarray, rate_hz: float = RATE_HZ) -> np.ndarray:
    """(n, WINDOW, n_signals) frame windows -> (n, len(FEATURES)) features."""
    d = derived(win)
    cols = []
    with np.errstate(all="ignore"), _nanwarn_off():
        for s, sts in STAT_SPEC.items():
            cols += [_REDUCE[st](d[s]) for st in sts]
        cols.append(d["tilt"][:, -1] - d["tilt"][:, 0])
        cols.append(d["vz"][:, -1] - d["vz"][:, 0])
        # largest frame-to-frame change, per second: angular acceleration
        cols.append(np.nanmax(np.abs(np.diff(d["rate_rp"], axis=1)), 1) * rate_hz)
        cols.append(np.nanmax(np.abs(np.diff(d["yaw_rate"], axis=1)), 1) * rate_hz)
    return np.stack(cols, axis=1).astype(np.float32)


class _nanwarn_off:
    """Silence 'All-NaN slice' warnings: an all-NaN window gives NaN, on purpose."""

    def __enter__(self):
        import warnings
        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("ignore", RuntimeWarning)

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def frames_from_messages(messages, motor_scale: MotorScale | None = None, stale_us: int = STALE_US):
    """All frames of a message stream: (t_us array, frames array, rx_ns array)."""
    fb = FrameBuilder(motor_scale, stale_us)
    ts, fs, rx = [], [], []
    for msg, rx_ns in messages:
        out = fb.update(msg)
        if out is not None:
            ts.append(out[0])
            fs.append(out[1])
            rx.append(rx_ns)
    if not ts:
        return np.empty(0, np.int64), np.empty((0, len(SIGNALS))), np.empty(0, np.int64)
    return np.array(ts, np.int64), np.vstack(fs), np.array(rx, np.int64)


def all_window_features(frames: np.ndarray, window: int = WINDOW, rate_hz: float = RATE_HZ) -> np.ndarray:
    """Features at every frame (NaN rows until the first window is full)."""
    n = len(frames)
    out = np.full((n, len(FEATURES)), np.nan, np.float32)
    if n >= window:
        win = np.lib.stride_tricks.sliding_window_view(frames, window, axis=0)  # (n-W+1, C, W)
        out[window - 1:] = window_features(np.moveaxis(win, -1, 1), rate_hz)
    return out
