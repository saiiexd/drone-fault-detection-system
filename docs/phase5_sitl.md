# Phase 5: PX4 SITL in Docker with fault injection

Measured on 2026-11-08 (CachyOS host, Docker 29.9.3, no GPU used).

Reproduce:
```sh
docker build -t aeropulse/px4-sitl:v1.17.0 docker/sitl                       # 945 s, 14.8 GB on disk
uv run python -m aeropulse.sim.scenarios --out data/raw/sitl                 # 16 flights, ~22 min
uv run python -m aeropulse.pipeline.run --raw data/raw/sitl --lake data/lake_sitl
uv run python -m aeropulse.models.apply --lake data/lake_sitl --models data/models --out reports/phase5
uv run python scripts/plot_flight.py --lake data/lake_sitl --key motor_off_5 --out reports/phase5 \
    --scores reports/phase5/sitl_scores.parquet --xlim 25 45
```

## 1. What was built

- **Docker image `docker/sitl/Dockerfile`:**
  - Based on the official `px4io/px4-dev:v1.17.0` build image. Gazebo Harmonic is added with PX4's own `Tools/setup/ubuntu.sh`, because the upstream simulation image doesn't exist yet.
  - PX4 **v1.17.0** (latest stable) is compiled for SITL at build time.
  - Runs headless with the x500 quadcopter. PX4 runs as a daemon, so `docker exec … px4-<command>` works.
- **Scenario runner `src/aeropulse/sim/scenarios.py`** (MAVSDK 4.1), for each flight:
  1. Starts a fresh container, so every flight starts from the same simulator state.
  2. Sets the parameters (`SYS_FAILURE_EN=1`, `CA_FAILURE_MODE=1`, no RC).
  3. Arms, takes off to 10 m, and flies random waypoints (seeded, so runs are reproducible).
  4. Injects the fault over MAVLink (`MAV_CMD_INJECT_FAILURE`), lands or kills the vehicle, and collects the ULog.
- **Exact labels:** the injected command is logged by PX4 in `vehicle_command` on the same clock as every sensor, so the label start is read from the log, exact to the microsecond. A test checks this.
- **Effect check (`fault_effect`):** the log is checked to confirm the fault really happened (e.g. the failed motor's command after injection). An accepted command alone doesn't count.
- **Same pipeline:** SITL logs go through the same phase 3 pipeline into `data/lake_sitl` (source `px4_sitl`). All 16 flights pass the quality rules.

## 2. Problems found and fixed

| Problem | Evidence | Fix |
|---|---|---|
| PX4 timed out waiting for Gazebo under `--network host` | `Timed out waiting for Gazebo world`, while `gz sim` alone worked | Gazebo's transport uses multicast discovery; pinned it to loopback with `GZ_IP=127.0.0.1` |
| Arming denied right after connecting | `Arming denied: Resolve system health failures first` | Retry arming until PX4's preflight checks accept it (took 2 s) |
| Wrong motor source for v1.17 logs | `actuator_outputs` is 0-1000 in Gazebo SITL, and there are no `PWM_MIN`/`PWM_MAX` parameters | Prefer `actuator_motors` (0-1). Its NaN for a stopped motor is filled with 0, so a dead motor doesn't vanish from the max/min features |
| Thrust sign changed in v1.14+ | `vehicle_thrust_setpoint.xyz[2]` is negative for upward thrust (NED) | Per-candidate scale factor (−1) in `signals.py` |
| "landed"/"armed" turned NaN after takeoff | Published only on change in v1.17, so the staleness rule expired them | State topics never go stale. This also fixed 28 UAV-SEAD flights (phase 3 §8), and the models were retrained |
| **Sensor faults (GPS off, barometer stuck) had no effect** | No ACK (MAVSDK `TIMEOUT`); GPS still at 1.0 Hz and the barometer still varying after injection | See §3 |

## 3. What PX4 v1.17 + Gazebo can actually inject

Confirmed by reading the v1.17.0 source in the image and by the logs:
- **Sensor failures** (`gps`, `baro`, …) are handled only by `simulator_mavlink`, the Gazebo *Classic* interface. The new-Gazebo `gz_bridge` ignores them.
- **Motor `stuck` and `wrong`** (`FailureInjector.cpp`) only falsify the **ESC telemetry** (zeroing it, or scaling voltage/current/RPM). The motor keeps producing thrust.
- **Motor `off`** really stops the motor, *but only with `CA_FAILURE_MODE=1`* (set by the scenario runner). Correction found in phase 6 (2026-11-09): the injector only flags the motor as failed, and PX4's own control allocator then stops it and re-allocates. With `CA_FAILURE_MODE=0` the motor keeps running (command ~0.74 after injection). So every SITL motor failure here is one that PX4 knows about and handles.

The current PX4 docs (`docs.px4.io/main`) describe newer behaviour from the development branch, where Gazebo supports sensor faults and `wrong` stops the motor. **Docs and release disagree**, and this was found only because the effect was checked in the logs instead of trusting the command.

So the SITL dataset is **8 nominal + 8 motor-off flights** (each of the 4 motors twice, injected 13-28 s after reaching altitude). In all 8 fault runs the failed motor's command goes from 0.72-0.75 to 0.0.

## 4. What a quadcopter motor failure looks like

`reports/phase5/flight_motor_off_5.png`: within about 0.2 s of injection, thrust demand jumps to 1.0, the remaining motors saturate, tilt oscillates up to 170°, and the vehicle falls. Unlike the octorotor in UAV-SEAD, **a quadcopter can't survive losing a motor**: altitude reaches the ground **1.24-1.63 s** after injection (from 7.3-11.2 m).

PX4's land detector mostly keeps reporting "not landed" while the vehicle lies upside down (tilt 180°). So windows after a crash count as "in air". That doesn't affect the latency numbers below, which use the first alarm after injection, but it would matter for any in-air time statistic.

## 5. Models trained on real UAV-SEAD flights, applied to SITL

The motor-fault model (`gbt_mechanical`) and the physics rule were trained only on real octorotor/quad UAV-SEAD flights, with no SITL data. Per-flight results are in `reports/phase5/uavsead_models_on_sitl.csv`; latency against impact is in `latency_vs_impact.csv`.

| Threshold on `gbt_mechanical` (+ 0.6 s persistence) | Detected | Median latency | Worst | Alarm before impact | False alarms in 8 nominal flights |
|---|---|---|---|---|---|
| **0.001** (selected by the phase 4 budget) | 8/8 | 0.73 s | 0.94 s | **8/8** | 1 |
| 0.05 | 8/8 | 1.73 s | 4.74 s | 3/8 | 0 |
| 0.2 | 6/8 | 2.89 s | 4.94 s | 0/8 | 0 |
| 0.5 | 6/8 | 3.65 s | 6.14 s | 0/8 | 0 |
| 0.9 | 5/8 | 6.14 s | 10.84 s | 0/8 | 0 |

The **physics rule** at its deployable threshold detected 8/8 in 1.92-2.34 s: always **after** impact.

What this shows:
- **The real-to-sim signal is there.** A model trained on real octorotor faults ranks the simulated quad motor failures as anomalous (max probability after injection 0.38-1.00, against ≤ 0.003 in nominal flights).
- **But warning before impact needs a hair-trigger threshold.** Only the degenerate 0.001 threshold beats impact in all 8 runs, and on real test days that same threshold costs 5.5 false alarms per hour (phase 4 §4). At any moderate threshold, the detector reports the crash, not the failure.
- **The reasons are structural.** The features use 2 s windows, and the training faults (octorotor) degrade over seconds, while a quad motor loss is catastrophic in about 1.4 s.
- **Fault type doesn't transfer.** At the first alarm, the UAV-SEAD type classifier said "Normal" or "External Position" in all 8 runs, never "Mechanical".
- **Fragility.** Before the state-topic fix and retraining, the same 8 flights gave a median latency of about 1.9 s. With 8 fault flights and 0.12 in-air hours of nominal flight, these numbers have wide uncertainty. They are evidence, not a benchmark.

## 6. Consequences for phases 6 and 7

- **Phase 6 (live demo on the quad).**
  - Use faster features: a 0.5 s window, and per-row signals such as motor-command collapse, thrust saturation and tilt rate. These are computable from MAVLink.
  - Train or validate on SITL quad flights. Otherwise the live demo will mostly flag the crash.
  - The detection latency target must be compared with time-to-impact (about 1.4 s from 10 m).
- **Phase 7 (sim-to-real).** The gap measured here mixes three effects: sim vs real, **quad vs octorotor**, and catastrophic vs gradual fault dynamics. The phase 7 design must separate them, e.g. by using RflyMAD's real quad flights as the real test set.
- **More fault types in SITL** need either Gazebo Classic (`simulator_mavlink`), PX4 SIH, or a PX4 version newer than v1.17. All three are untested here.

## 7. Not done (and why)

- **SITL in GitHub Actions.** The image is 14.8 GB, larger than a standard runner's free disk (~14 GB). CI runs the unit tests only (`.github/workflows/tests.yml`; 23 passed, 5 data-dependent skipped in a clean checkout). SITL in CI would need a slimmer image pushed to a registry.

## 8. Watching a flight: 3D view and demo world

The simulator runs headless, because recording flights needs no graphics. To watch one, `scenarios.py --gui` starts a second container from the same image that runs only the Gazebo user interface (`gz sim -g`) and joins the simulation over gz-transport on loopback.

| Decision | Why |
|---|---|
| Viewer in its own container, same image | Gazebo is already in the image; nothing to install on the host. The window is drawn through the host's X server (`/tmp/.X11-unix`). |
| Both containers run as the host user | gz-transport only connects processes in the same partition, which defaults to `hostname:username`. A root simulator and a user viewer never see each other. |
| Minimal window layout (`docker/sitl/viewer.gui.config`) | Only the 3D view: Gazebo's editing panels took half the window. |
| Camera follows the drone with gain 0.3 (`/gui/track`) | Gazebo's default follow gain (0.01) loses a falling drone. |

`--world town` loads `docker/sitl/worlds/town.sdf`: PX4's default world (same physics, gravity, magnetic field and location) with a main street, shops, trees, vehicles and street furniture. All scenery is at least 45 m from the takeoff point, beyond the scripted flight area (waypoints within 25 m on each axis). The 20 models are Gazebo Fuel models (Open Robotics, CC0), downloaded into the image at build time so the world loads offline.

Two model problems were fixed while building it:
- Some models reference their textures by bare file name while the files sit in `materials/textures/`, so they rendered black. The Dockerfile copies the textures next to the meshes.
- Three house models and the gas station use only OGRE 1 material scripts, which Gazebo Harmonic's OGRE 2 renderer ignores; they rendered black and were replaced with other buildings.

Detection is unaffected by the scenery. On the same flight (`live_motor_off_1`), the default world gave the first alarm 0.10 s after the motor was cut, with impact at 1.96 s (`reports/phase6/live_per_flight.csv`); two runs in the town world gave 0.10 s and 0.09 s, with impact at 1.95 s and 1.94 s. Only startup takes longer, by about 30 s, to load the models. All recorded flights use the default world.
