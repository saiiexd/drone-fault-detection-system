"""Online motor-failure detector: one MAVLink message in, maybe one decision out.

Wraps the same FrameBuilder, window_features and persistence filter used in
training (stream.features, stream.evaluate), with a ring buffer holding the
last WINDOW frames. tests/test_stream.py checks that replaying a tlog
through this class gives the same scores as the offline batch path.

A PX4 reboot (e.g. a new SITL container) restarts the boot clock. When the
frame time jumps backwards, the detector starts a new session and clears its
state, so windows never mix two flights.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from aeropulse.stream.dataset import AIRBORNE_STATES
from aeropulse.stream.evaluate import Persistence
from aeropulse.stream.features import (
    IDX,
    RATE_HZ,
    SIGNALS,
    STALE_US,
    WINDOW,
    FrameBuilder,
    MotorScale,
    derived,
    window_features,
)


@dataclass
class Decision:
    session: int
    t_us: int              # autopilot boot clock of the frame (event time)
    rx_ns: int             # host receive time of the triggering message
    proc_us: float         # host time spent in update() for this frame
    score: float
    alarm: bool
    onset: bool            # first alarming frame of an episode
    airborne: bool
    display: dict = field(default_factory=dict)


class LiveDetector:
    def __init__(self, bundle_path: Path, motor_scale: MotorScale | None = None):
        # One row per call: starting OpenMP threads costs more than the work.
        # Measured on one SITL model: 5.8 ms -> 1.3 ms median, 12.5 -> 1.7 ms p99.
        # (Process-wide setting; the service process only runs this detector.)
        threadpool_limits(1)
        b = joblib.load(bundle_path)
        self.model, self.threshold, self.k = b["model"], b["threshold"], b["persist"]
        # Stream settings travel with the model (older bundles: 50 Hz defaults).
        self.window = b.get("window", WINDOW)
        self.rate_hz = b.get("rate_hz", RATE_HZ)
        self.stale_us = b.get("stale_us", STALE_US)
        self.radio = b.get("radio", False)
        self.motor_scale = motor_scale
        self.session = 0
        self._reset()

    def _reset(self) -> None:
        self.fb = FrameBuilder(self.motor_scale, self.stale_us)
        self.buf = np.full((self.window, len(SIGNALS)), np.nan)
        self.n = 0
        self.persist = Persistence(self.k)
        self.prev_alarm = False
        self.last_t = None

    def update(self, msg, rx_ns: int) -> Decision | None:
        t0 = time.perf_counter_ns()
        out = self.fb.update(msg)
        if out is None:
            return None
        t_us, frame = out
        if self.last_t is not None and t_us < self.last_t - 1_000_000:
            self.session += 1          # boot clock restarted: new flight
            self._reset()
            t_us, frame = self.fb.update(msg)   # rebuild the frame with no old values
        self.last_t = t_us
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = frame
        self.n += 1
        if self.n >= self.window:
            x = window_features(self.buf[None], self.rate_hz)
            score = float(self.model.predict_proba(x)[0, 1])
        else:
            score = float("nan")
        ls = frame[IDX["landed_state"]]
        airborne = bool(np.isfinite(ls) and int(ls) in AIRBORNE_STATES)
        alarm = self.persist.step(bool(score >= self.threshold) and airborne)
        onset = alarm and not self.prev_alarm
        self.prev_alarm = alarm
        d = {k: float(v) for k, v in derived(frame).items()
             if k in ("tilt", "tilt_err", "thrust_sp", "vz", "acc_z", "m_max")}
        d["alt"] = float(-frame[IDX["z"]])
        proc_us = (time.perf_counter_ns() - t0) / 1e3
        return Decision(self.session, t_us, rx_ns, proc_us, score, alarm, onset, airborne, d)
