"""Canonical signal set: one stable column name per physical quantity.

PX4 renames and moves fields between versions (e.g. body rates lived in
``vehicle_attitude`` in v1.8 and moved to ``vehicle_angular_velocity`` later;
external vision moved from ``vehicle_vision_position`` to
``vehicle_visual_odometry``). Each canonical signal therefore lists candidate
(topic instance, field) sources in priority order; the first one present in a
log is used. Downstream code only ever sees canonical names, so a new PX4
version means adding a candidate here, not changing features or models.
"""

from dataclasses import dataclass

MAX_MOTORS = 8


@dataclass(frozen=True)
class Signal:
    name: str
    # (topic instance, field) or (topic instance, field, scale); scale fixes
    # sign/unit conventions that changed between PX4 versions.
    candidates: tuple[tuple, ...]
    required: bool = False


def _vec(prefix: str, topic: str, field: str, axes=("x", "y", "z"), required=False):
    return [Signal(f"{prefix}_{a}", ((topic, f"{field}[{i}]"),), required)
            for i, a in enumerate(axes)]


SIGNALS: list[Signal] = [
    # IMU (body frame), native 250 Hz in v1.8 logs
    *_vec("gyro", "sensor_combined_0", "gyro_rad", required=True),
    *_vec("acc", "sensor_combined_0", "accelerometer_m_s2", required=True),
    # Controller output: normalised torque (-1..1) and thrust (0..1) demands
    # v1.14+ replaced actuator_controls with torque/thrust setpoints; thrust is
    # along body z (NED, pointing down), so upward thrust is negative there.
    *[Signal(f"ctrl_{a}",
             (("actuator_controls_0_0", f"control[{i}]"),
              ("vehicle_torque_setpoint_0", f"xyz[{i}]") if i < 3
              else ("vehicle_thrust_setpoint_0", "xyz[2]", -1.0)),
             required=True)
      for i, a in enumerate(("roll", "pitch", "yaw", "thrust"))],
    # Per-motor commands. v1.13+: actuator_motors, normalised 0..1 by PX4 (NaN
    # for a stopped motor; filled with 0 in the pipeline). Older logs: PWM in
    # actuator_outputs, normalised with PWM_MIN/PWM_MAX. actuator_outputs is
    # not used when actuator_motors exists: its scale depends on the output
    # driver (e.g. 0..1000 in Gazebo SITL). Only the first `n_motors` are real.
    *[Signal(f"motor_{i + 1}",
             (("actuator_motors_0", f"control[{i}]"), ("actuator_outputs_0", f"output[{i}]")),
             required=(i == 0))
      for i in range(MAX_MOTORS)],
    # Attitude estimate (quaternion w, x, y, z) and body rates
    *[Signal(f"q_{c}", (("vehicle_attitude_0", f"q[{i}]"),), required=True)
      for i, c in enumerate("wxyz")],
    *[Signal(f"rate_{a}",
             (("vehicle_attitude_0", f"{a}speed"),
              ("vehicle_angular_velocity_0", f"xyz[{i}]")))
      for i, a in enumerate(("roll", "pitch", "yaw"))],
    # Local position/velocity estimate (NED: z is down)
    *[Signal(f"pos_{a}", (("vehicle_local_position_0", a),), required=True) for a in "xyz"],
    *[Signal(f"vel_{a}", (("vehicle_local_position_0", f"v{a}"),), required=True) for a in "xyz"],
    # Estimator consistency: innovation test ratios (>1 means measurement rejected)
    Signal("tr_vel", (("estimator_status_0", "vel_test_ratio"),)),
    Signal("tr_pos", (("estimator_status_0", "pos_test_ratio"),)),
    Signal("tr_hgt", (("estimator_status_0", "hgt_test_ratio"),)),
    Signal("tr_mag", (("estimator_status_0", "mag_test_ratio"),)),
    # External (indoor) position measurement
    *[Signal(f"ext_{a}",
             (("vehicle_vision_position_0", a), ("vehicle_visual_odometry_0", a)))
      for a in "xyz"],
    # Height above ground from a rangefinder
    Signal("dist_bottom", (("distance_sensor_0", "current_distance"),)),
    # GPS quality
    Signal("gps_eph", (("vehicle_gps_position_0", "eph"), ("sensor_gps_0", "eph"))),
    Signal("gps_sats", (("vehicle_gps_position_0", "satellites_used"),
                        ("sensor_gps_0", "satellites_used"))),
    # Power
    Signal("batt_v", (("battery_status_0", "voltage_v"),)),
    # Flight state: on ground vs in air, armed (arming_state 2 = ARMED)
    Signal("landed", (("vehicle_land_detected_0", "landed"),)),
    Signal("arming_state", (("vehicle_status_0", "arming_state"),)),
]

SIGNAL_NAMES = [s.name for s in SIGNALS]
