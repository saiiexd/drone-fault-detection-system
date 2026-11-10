# Phase 2 findings: UAV-SEAD log exploration and timeline alignment

Sample: 10 annotated UAV-SEAD flights (4 Normal, 2 Mechanical, 2 External Position, 1 Altitude, 1 Global Position), chosen as the smallest files per class. They are short (24-89 s) and almost all PX4 v1.8.x, so the findings below may not hold for the full dataset. Re-check them in phase 3.

Reproduce:
```sh
uv run python scripts/explore_topics.py --raw data/raw/uav_sead --out reports/phase2
uv run python scripts/plot_flight.py --raw data/raw/uav_sead --key 2019-01-15/07_28_00 --out reports/phase2
uv run pytest
```

## 1. Topics and logging rates

38 distinct topics (318 topic instances in total). 26 topics appear in all 10 logs. Full table: `reports/phase2/topic_summary.csv`; per flight: `reports/phase2/topic_inventory.csv`.

Signals relevant to fault detection (median logging rate across flights):

| Instance | Rate | What it is |
|---|---|---|
| `sensor_combined_0` | 250 Hz | IMU: gyro (rad/s) and accelerometer (m/s²) |
| `actuator_controls_0_0` | 250 Hz | Controller output: roll/pitch/yaw torque and thrust commands, normalised |
| `vehicle_attitude_0` | 31 Hz | Estimated attitude (quaternion) and body rates |
| `vehicle_vision_position_0` | 19 Hz | External (indoor) position measurements |
| `vehicle_local_position_0` | 10 Hz | EKF position/velocity estimate |
| `actuator_outputs_0`, `_1` | 10 Hz | Per-motor PWM commands (`_0` = the 4 motors) |
| `ekf2_innovations_0`, `estimator_status_0` | 5 Hz | Estimator residuals and health flags |
| `vehicle_gps_position_0` | 12.5 Hz | GPS, present in only 4/10 logs |
| `vehicle_air_data_0` (barometer) | 88 Hz | Present in only 3/10 logs |

Takeaways:
- **These are logging rates, not sensor rates.** The PX4 logger subsamples topics, which is why motor outputs are only 10 Hz here although the motors are updated much faster. A live MAVLink stream will have different rates again (phase 5/6).
- **Motor outputs at 10 Hz bound detection latency.** A detector that relies on per-motor PWM cannot react faster than about 100 ms from that signal alone. `actuator_controls_0` (250 Hz) is the faster proxy.
- **Topic availability varies:** GPS 4/10, barometer 3/10, magnetometer 3/10, optical flow 6/10. Features must tolerate missing topics, or be limited to the 26 always-present ones.
- Only `actuator_controls_0` has `timestamp_sample`; everything is aligned on `timestamp`.

## 2. Data-quality observations

- **Timing is regular.** For the continuously streamed always-present topics, the median 95th-percentile/median gap ratio is ≤ 1.1. Exceptions:
  - vision topics at 1.8, which reflects the external tracking system;
  - event-like topics published irregularly (`sensor_preflight`, `vehicle_command`).
- **One log has a 4.96 s gap in every topic** (`2018-07-__/sess015/log002`, about 1 s after start). The ULog file records **no dropout** for it. After alignment, 14.1% of that flight's rows are NaN, as the staleness rule intends. The batch pipeline needs a quality check for gaps like this.
- One log (`2018-09-03/2016-02-11/19_35_56`) has 2 recorded dropouts, both short (sensor_combined max gap 8 ms).
- Two logs have no PX4 version string. Two logs are inside an obviously mis-dated folder (`2018-09-03/2016-02-11`), so folder dates are not reliable metadata.

## 3. Labels: usable

- **Clock:** label ranges are in **microseconds on the log's own clock** (time since boot). All 8 ranges in the 6 anomalous sample flights fall inside their log's `[start_timestamp, last_timestamp]`.
- **Meaning:** a label's signal name (e.g. `Position.X`, `Actuator Outputs`) is the **plot panel** the annotator marked in the annotation tool, not a single field. The mapping is defined in `server/plotting.py` of [ulog_annotation_tool](https://github.com/aykutkabaoglu/ulog_annotation_tool) (commit `3e1492f`). For example:
  - `Actuator Outputs` → `actuator_outputs.output[0..3]`
  - `Position.X` → local position x, vision position x, GPS/global position (x/y swapped in that file)
  - `Position.Z` → local z, distance sensor, vision z, GPS z
- **Frequency** across all 1,396 annotations: Position.X 168, Position.Y 90, Position.Z 84, Actuator Outputs 31, Actuator Controls 11, others ≤ 4.
- **Caveat:** the paper says ranges mark regions of interest, not exact onset or offset. Latency measured against these labels is approximate.

## 4. Alignment design (`src/aeropulse/align/timeline.py`)

- **Why not join on timestamp:** each topic has its own clock ticks, so exact matches across topics almost never happen.
- **Causal as-of join:** each grid point takes the most recent sample at or before it (`pd.merge_asof`, `direction="backward"`). Linear interpolation would use the next, future sample. A live detector can't do that, so offline results would be inflated. The annotation tool itself uses `np.interp`, which is acceptable for plots but not for features.
- **Staleness limit:** samples older than `max_age_s` give NaN instead of being repeated across a gap.
- **Grid:** 50 Hz on the common span of all selected topics. 50 Hz is 5× the slowest key signal (motor outputs, 10 Hz) and keeps the 250 Hz IMU at a manageable size. **Open:** high-frequency IMU content (vibration) is lost by taking one sample every 20 ms. If vibration matters, compute IMU window features (e.g. RMS over 20 ms) at 250 Hz first, then align them. Decide in phase 3/4.
- **Open:** one `max_age_s` for all topics is a simplification. A per-topic value (e.g. 3-5× each topic's period) is more correct.

Measured: 36/36 tests pass. All 10 flights align. Nine have 0% NaN rows; the gap flight has 14.1%.

## 5. What the faults look like (2 Mechanical vs 1 Normal flight)

> **Correction (phase 3).** All 10 sample flights are from an **octorotor coaxial** airframe (`MAV_TYPE=14`, `SYS_AUTOSTART=12001`, 8 active motor outputs), not a quadcopter.
> - The original phase 2 plots showed only motors 1-4. Those plots were removed.
> - Replacement plots use the max, mean and min across all 8 motors. They are in `reports/phase3/`, made with `scripts/plot_flight.py`.
> - The annotation tool's "Actuator Outputs" panel also shows only `output[0..3]`, so annotators saw half of the motors.
> - The observations below still hold for the 8-motor view.

- **In both Mechanical flights, during the labelled range:**
  - The thrust command climbs to its 1.0 limit, and at least one motor sits at the 2000 µs PWM maximum (saturation).
  - Altitude still drops: 1.9 m → 0.1 m in one flight, 4.4 m → 0.1 m in the other.
  - That is the physical signature of lost lift: the controller asks for full power and the drone still sinks.
- **In the Normal flight:** thrust stays around 0.82 with no saturation. The altitude drops only at commanded landing, with thrust decreasing.
- **Caution:** in `2019-01-15/07_28_00` one motor is near or at saturation for most of the flight, before the label starts. Saturation alone is therefore not a fault signal. The combination of saturation, maximum thrust and altitude loss looks discriminative, but this is based on 2 flights and is a **hypothesis for phase 4, not a result**.
- **Yaw rate barely reacts** in these flights (one 1.6 rad/s spike). Yaw rate is a weak signal for this fault type here.

## Next (phase 3)

- Choose the signal set: the always-present topics in §1. Decide how to handle optional ones (GPS, barometer).
- Write raw, cleaned and feature Parquet layers, partitioned by flight.
- Add quality checks: gaps over threshold, missing version, missing required topics.
- Download a larger, less biased sample, not just the smallest files.
