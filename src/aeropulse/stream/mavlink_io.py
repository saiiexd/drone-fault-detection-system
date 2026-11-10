"""MAVLink telemetry in: a live UDP link or a recorded .tlog file.

Both sources yield the same thing, the decoded message plus the host time at
which it was received. The detector, the training-set builder and the tests
therefore consume identical input. A model trained on recorded streams sees
exactly what it will see live (no training-serving skew).

Why MAVLink and not the ULog: the ULog is written on the vehicle and is only
available after landing. A ground station sees only the MAVLink stream, which
has fewer signals and lower rates (50 Hz here, against 250 Hz IMU in the log).

PX4 SITL's GCS link (UDP 14550) streams these at 50 Hz by default (measured
2026-11-09): ATTITUDE, ATTITUDE_QUATERNION, ATTITUDE_TARGET, LOCAL_POSITION_NED,
SERVO_OUTPUT_RAW. HIGHRES_IMU is not streamed by default, so it is requested
with MAV_CMD_SET_MESSAGE_INTERVAL.

A .tlog is the standard ground-station recording format (QGroundControl and
MAVProxy write it too): each packet prefixed by an 8-byte receive time in µs.
"""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

from pymavlink import mavutil

GCS_URL = "udpin:0.0.0.0:14550"
MAV_CMD_SET_MESSAGE_INTERVAL = 511
REQUEST_HZ = {"HIGHRES_IMU": 50.0}


def connect(url: str = GCS_URL):
    # source_system 255 is the conventional ground-station system id
    return mavutil.mavlink_connection(url, source_system=255, source_component=190)


def request_streams(conn) -> None:
    """Ask the autopilot for the streams that are not on by default."""
    for name, hz in REQUEST_HZ.items():
        msg_id = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}")
        conn.mav.command_long_send(conn.target_system or 1, conn.target_component or 1,
                                   MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                   msg_id, 1e6 / hz, 0, 0, 0, 0, 0)


class LiveLink:
    """Receive from PX4 over UDP, keep the link alive, optionally record a tlog.

    PX4 only keeps sending to a GCS it hears from, so a heartbeat goes out
    once per second. Streams are (re)requested every 5 s because a request
    sent before the autopilot is ready is silently dropped.
    """

    def __init__(self, url: str = GCS_URL, tlog: Path | None = None):
        self.conn = connect(url)
        self.tlog = None
        if tlog:
            tlog.parent.mkdir(parents=True, exist_ok=True)
            self.tlog = open(tlog, "wb")
        self._last_hb = 0.0
        self._last_req = 0.0

    def _keepalive(self) -> None:
        now = time.monotonic()
        if now - self._last_hb >= 1.0:
            self.conn.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                         mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
            self._last_hb = now
        if self.conn.target_system and now - self._last_req >= 5.0:
            request_streams(self.conn)
            self._last_req = now

    def messages(self, stop: threading.Event | None = None) -> Iterator[tuple[object, int]]:
        """Yield (message, host receive time in ns, monotonic clock)."""
        while stop is None or not stop.is_set():
            self._keepalive()
            msg = self.conn.recv_match(blocking=True, timeout=0.2)
            if msg is None:
                continue
            rx_ns = time.monotonic_ns()
            if msg.get_type() == "BAD_DATA":
                continue
            if self.tlog:
                us = int(time.time() * 1e6)
                self.tlog.write(us.to_bytes(8, "big") + msg.get_msgbuf())
            yield msg, rx_ns

    def close(self) -> None:
        if self.tlog:
            self.tlog.close()
        self.conn.close()


# What a real telemetry radio delivered on RflyMAD's real flights (QGroundControl
# tlogs, measured 2026-11-09). Messages not listed were not sent at all.
RADIO_RATES_HZ = {"ATTITUDE": 15.0, "LOCAL_POSITION_NED": 15.0, "ATTITUDE_TARGET": 2.0,
                  "SERVO_OUTPUT_RAW": 1.0, "EXTENDED_SYS_STATE": 1.0, "HEARTBEAT": 5.0}


def _msg_time_us(msg) -> int | None:
    if hasattr(msg, "time_boot_ms"):
        return int(msg.time_boot_ms) * 1000
    if hasattr(msg, "time_usec"):
        return int(msg.time_usec)
    return None


def decimate(messages, rates_hz: dict[str, float] = RADIO_RATES_HZ):
    """Thin a stream to per-message rates, dropping unlisted messages. Used to
    turn a 50 Hz SITL recording into what a telemetry radio would carry.
    Messages without a time field are thinned on the receive clock."""
    due: dict[str, float] = {}
    for msg, rx_ns in messages:
        name = msg.get_type()
        hz = rates_hz.get(name)
        if hz is None:
            continue
        t = _msg_time_us(msg)
        t = t if t is not None else rx_ns // 1000
        period = 1e6 / hz
        d = due.get(name)
        if d is None or t < d - 2 * period or t > d + 2 * period:   # start, reboot or gap
            d = t
        if t >= d:
            due[name] = d + period     # running schedule: averages exactly `hz`
            yield msg, rx_ns
        else:
            due[name] = d


def read_tlog(path: Path) -> Iterator[tuple[object, int]]:
    """Replay a tlog. The receive time is the recorded wall-clock time in ns."""
    conn = mavutil.mavlink_connection(str(path), robust_parsing=True)
    while True:
        msg = conn.recv_match()
        if msg is None:
            break
        if msg.get_type() == "BAD_DATA":
            continue
        yield msg, int(msg._timestamp * 1e9)
    conn.close()


class Recorder(threading.Thread):
    """Background tlog recorder used while scripted SITL flights run."""

    def __init__(self, tlog: Path, url: str = GCS_URL):
        super().__init__(daemon=True)
        self.link = LiveLink(url, tlog)
        self.stop_event = threading.Event()
        self.n = 0

    def run(self) -> None:
        for _ in self.link.messages(self.stop_event):
            self.n += 1
        self.link.close()

    def stop(self) -> int:
        self.stop_event.set()
        self.join(timeout=5)
        return self.n
