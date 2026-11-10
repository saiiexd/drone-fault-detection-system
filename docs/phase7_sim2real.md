# Phase 7: the sim-to-real experiment, with the gaps separated

Measured on 2026-11-09.

Reproduce (data: ~5.3 GB):
```sh
uv run python scripts/download_rflymad.py --out data/raw/rflymad --set real --per-group 12
uv run python scripts/download_rflymad.py --out data/raw/rflymad --set hil  --per-group 12
uv run python -m aeropulse.sim2real.rflymad --set real --out data/stream/rflymad_real_frames.parquet
uv run python -m aeropulse.sim2real.rflymad --set hil  --out data/stream/rflymad_hil_frames.parquet
uv run python -m aeropulse.sim2real.run --real data/stream/rflymad_real_frames.parquet \
    --hil data/stream/rflymad_hil_frames.parquet --sitl-model data/models/stream_gbt.joblib --out reports/phase7   # ~10 min
uv run python -m aeropulse.sim2real.octo_to_quad --out reports/phase7                                          # ~5 min
uv run python -m aeropulse.sim2real.link_rate --out reports/phase7 --part real                                 # ~15 min
uv run python -m aeropulse.sim2real.link_rate --out reports/phase7 --part sitl                                 # ~4 min
```

## 1. Question and metric

Phase 5 found that "a model trained on X detects faults in Y" mixes three gaps:
- simulation vs reality;
- octorotor vs quadcopter;
- catastrophic vs gradual faults.

Each experiment below holds the others fixed as far as the data allows.

**Main metric: frame-level average precision (AP)** on airborne frames, with a 95% bootstrap interval that resamples *flight groups* (recording days for real flights), because frames within a flight are strongly correlated. AP is threshold-free, so it measures whether the model *ranks* faulty moments above healthy ones, separately from whether its alarm threshold still works.

**Transfer ratio** = AP(sim → real) / AP(real → real). The model is said to transfer if the ratio is close to 1 *and* a threshold chosen without real fault labels keeps false alarms within budget (≤ 2 per airborne hour) on real flights.

## 2. Data: RflyMAD

RflyMAD (Le, Quan et al., arXiv:2311.11340, IJRR 2024; Beihang University). Non-commercial research use. Downloaded per file through Kaggle's public API; there is no Hugging Face mirror.

| Set | What it is | Flights used | Fault flights | Faults |
|---|---|---|---|---|
| **Real** | Real quadcopters (airframe 4011, PWM 1000-1950), PX4 v1.12, 13 recording days | 111 (52 no-fault) | 59 | One to four motors at 70-90% thrust (10-30% loss); 33 *step*, 26 *ramp* (declining over 60-100 s) |
| **HIL** | The real flight controller and firmware, simulated physics | 120 (72 no-fault) | 48 | Partial loss on one to four motors, step and ramp |
| SITL (phase 6) | Gazebo x500, PX4 v1.17 | 52 (20 nominal) | 32 | One motor fully off |

The selection keeps 12 cases per flight mode (hover, waypoint, velocity, circling, accelerate), evenly spaced. That was needed because bandwidth here was ~1.5 MB/s.

**Labels** come from RflyMAD's own fault-injection topic `rfly_ctrl_lxl`, logged on the autopilot clock:
- `id` = 123450 while the motor fault is active;
- `mode` 1 = step, 2 = ramp;
- `controls[0..3]` = per-motor thrust multipliers.

So labels are exact. A test checks one flight against its test-info sheet (fault parameter [1, 0.85, 1, 1, 1]).

Three label problems were found and fixed:
- HIL marks "no fault" as `id` 0, not 1500.
- Ramp faults start at a multiplier of ~1.0. Each frame therefore carries the thrust loss in effect *at that moment*, so for ramps the question becomes "how much loss had built up at the alarm".
- In some HIL logs the last rows have all multipliers at 0 while `id` still says "fault". That's a reset artifact, not a 100% loss.

**Representation.** The real vehicles' telemetry radio is slow (`ATTITUDE` 15 Hz, `ATTITUDE_TARGET` 2 Hz, `SERVO_OUTPUT_RAW` 1 Hz). So every flight is rebuilt as the 50 Hz MAVLink stream PX4 would produce, from the ULog's onboard topics (`stream/ulog_adapter.py`), and passed through the unchanged phase 6 frame and feature code. On a SITL flight that has both a recorded tlog and a ULog, the rebuilt features match the recorded ones (r > 0.95 for attitude, rates, velocity, thrust and motors; tested).

**Grouping.** Real flights are grouped by recording day, as UAV-SEAD was in phase 4. 31 of the 52 no-fault flights come from two days with almost no fault flights (25 May, 15 June), so a finer split could learn the day. All airframe parameters are identical across flights, so vehicle type is not a confound.

## 3. Results

Chance AP is the fraction of airborne frames inside a fault: 0.169 for real flights, 0.341 for HIL.

| Experiment | Train → test | AP [95% CI] | Event recall | False alarms / h | Median latency |
|---|---|---|---|---|---|
| A | SITL (phase 6 model) → real | 0.197 [0.129-0.284] | 2/59 | 1.2 | 1.10 s |
| A | SITL → HIL | 0.322 [0.255-0.398] (chance 0.341) | 0/48 | 29.2 | n/a |
| B | HIL → HIL (5-fold CV) | 0.996 [0.990-0.999] | 48/48 | 1.8 | 0.17 s |
| **B** | **HIL → real** (threshold from HIL) | **0.704 [0.519-0.829]** | 56/59 | **768** | 1.49 s |
| B2 | HIL → real (threshold from real no-fault flights) | 0.704 | 36/59 | 4.3 | 8.66 s |
| **B ceiling** | **real → real** (day-grouped CV, nested threshold) | **0.886 [0.790-0.939]** | 46/59 | 0.0 | 6.19 s |

Transfer ratio AP(HIL → real) / AP(real → real) = **0.80**. Real no-fault airborne time: 2.5 h. Files: `reports/phase7/sim2real_*.csv`.

### A. Naive transfer: the phase 6 SITL model on RflyMAD

**It does not transfer at all.**
- On real flights its AP is 0.197 [0.129-0.284], against 0.169 for chance.
- It alarms on 2 of 59 real fault flights.
- It never alarms on a HIL fault flight.

The reason is visible in the model itself. Phase 6's detector relies almost entirely on *peak angular acceleration* (phase 6 §4), the signature of a motor switching fully off with PX4 re-allocating around it. A 10-30% loss produces no such spike:
- After a real step fault, the 0.5 s mean tilt-tracking error goes from 1.1° to 5.7° (median over flights, first 2 s).
- After a SITL motor-off, the window maximum reaches 166° within 1 s.

The SITL model also never fired on HIL, where faults scale the motor signal without PX4 knowing (phase 6 §5 showed that SITL in this setup can only produce failures PX4 knows about).

### B. Sim-to-real only: HIL → real

HIL shares the firmware, fault model, label source and vehicle family with the real flights. So the gap here is physics and sensors.

- **The ranking transfers fairly well.** AP(HIL → real) = 0.70 [0.52-0.83] against AP(real → real) = 0.89 [0.79-0.94]: **transfer ratio 0.80**.
- **The operating point does not transfer.** A threshold chosen on HIL (≤ 2 false alarms/h there) gives **768 false alarms per hour** on real flights.
  - Cause: HIL is nearly noise-free. Before a fault, the median tilt-tracking error is 0.04° in HIL against 1.0° on the real vehicles, about 25× quieter, and HIL hovers at a lower motor command (0.22 vs 0.39). A HIL-calibrated threshold sits inside the real vehicles' normal noise.
  - The same effect makes HIL cross-validation look perfect (AP 0.996): in a noise-free world, a 0.4% thrust loss is visible.
- **B2, recalibrating on real no-fault flights only.** Keep the HIL model but set its threshold on real no-fault flights from other days (no real fault labels needed). False alarms fall to **4.3 per hour** at 61% event recall. That's still 2× the budget: thresholds set on some days don't fully hold on others, the same day-to-day drift phase 4 found on UAV-SEAD.
- **B2 is fragile.** Two earlier runs differed only in the row order of the training table, which changes HistGradientBoosting's early-stopping split. AP(HIL → real) stayed within 0.70-0.72, but B2 gave 37-61% recall at 4.3-11.8 false alarms per hour. The table is now sorted, so results are reproducible (checked: identical rebuild). The spread is a property of the method: a threshold set on noisy normal data from a few days is unstable.

### C. Fault severity and dynamics

Real flights only, by the final thrust loss. Event recall:

| Final thrust loss | Real flights | real → real | HIL → real (recalibrated, B2) | SITL → real (A) |
|---|---|---|---|---|
| ≤ 10% | 9 | 0.89 | 0.44 | 0.00 |
| 10-20% | 38 | 0.76 | 0.63 | 0.00 |
| 20-40% | 12 | 0.75 | 0.67 | 0.17 |

By fault dynamics (real flights):

| | Step (33) | Ramp (26) |
|---|---|---|
| real → real: recall | 0.64 | 0.96 |
| real → real: median latency / loss at alarm | 1.84 s after onset | 13.3 s after onset, at **8.9%** thrust loss |
| HIL → real (B2): recall | 0.52 | 0.73 (at 9.7% loss) |

- **Within the real data, severity barely matters across 10-30%.** Recall is about 0.75-0.89 in every bin. These faults are all mild, and the controller compensates for each of them.
- **Ramps are caught more often than steps, but late.** A ramp keeps getting worse until it crosses the detector's sensitivity, about 9% loss. A small step stays small, and a third of real step faults are never flagged.
- **The catastrophic/gradual axis can only be read across datasets.** SITL motor-off (100% loss, crash in ~1.4 s) is caught in 0.1 s by a model that sees nothing in real 10-30% faults. Within one dataset there is no catastrophic real fault to compare against.

### D. Airframe gap (real data on both sides): UAV-SEAD octorotor model → RflyMAD real quads

The phase 4 motor-fault model was trained on real UAV-SEAD flights (63 of its 64 motor-fault events on an octorotor). Here it is applied through the unchanged phase 3 pipeline (2 s windows; 108 of 111 flights usable).

| Model → real RflyMAD quads | AP (chance 0.157) | Event recall (56 fault flights) | False alarms / h (1.36 h) |
|---|---|---|---|
| UAV-SEAD `gbt_mechanical`, F1 threshold (0.964) | 0.204 | 0.07 | 0.7 |
| UAV-SEAD `gbt_mechanical`, deployable threshold (0.001) | 0.204 | 0.91 | **363** |
| UAV-SEAD physics rule | 0.170 | 0.05 | 0.0 |
| Same recipe trained on RflyMAD (day-grouped CV, nested threshold) | **0.903** | 0.73 | 0.7 |

- The octorotor model is **near chance on quads** (AP 0.20 vs 0.16).
- At its phase 4 deployable threshold (the degenerate 0.001 flagged in phase 4), it does catch 91% of faults, but at **363 false alarms per hour**. That is the hair-trigger problem predicted in phase 4, now measured on new data.
- The *features and recipe* are not the problem. The same recipe trained on RflyMAD reaches AP 0.90 with 73% event recall at 0.74 false alarms per hour (day-grouped folds; thresholds from an inner CV on training days only).
- This is not a pure airframe effect. Vehicles, sites and fault mechanisms all differ between the datasets. What it shows is that a motor-fault model does not carry over between vehicle types, even when both are real.

### E. The telemetry radio link

Everything above used the 50 Hz stream rebuilt from the onboard log, which is what a companion computer wired to the autopilot sees. A ground station on a telemetry radio sees much less. RflyMAD recorded that radio stream on every real flight:

| Message | Radio rate |
|---|---|
| `ATTITUDE`, `LOCAL_POSITION_NED` | 15 Hz |
| `ATTITUDE_TARGET` | 2 Hz |
| `SERVO_OUTPUT_RAW` | 1 Hz |
| quaternion, IMU | not sent |

The radio tlogs share the autopilot clock with the ULogs (roll angles agree at lag 0, RMS 0.0002 rad), so the same labels apply.

Changes for radio input, with every 50 Hz default unchanged:
- frame rate, window and staleness became parameters;
- slow messages are held for up to 1.2 s;
- without a quaternion message, the attitude is rebuilt from `ATTITUDE`'s Euler angles (tested).

**Real partial faults:** same 111 flights, recipe and day-grouped nested CV as the B ceiling. Only the input changes.

| Input | AP [95% CI] | Event recall | False alarms / h | Step: recall, latency | Ramp: recall, loss at alarm |
|---|---|---|---|---|---|
| 50 Hz rebuilt from ULog | 0.886 [0.79-0.94] | 0.78 | 0.0 | 0.64, 1.84 s | 0.96, 8.9% |
| **Real radio**, 0.5 s window | **0.874** [0.77-0.93] | 0.73 | 1.2 | 0.61, 1.93 s | 0.88, 9.8% |
| Real radio, 2 s window | 0.879 [0.78-0.93] | 0.78 | 1.2 | 0.67, 2.79 s | 0.92, 9.2% |

**Catastrophic faults:** the 52 phase 6 SITL flights, thinned to the radio's rates (`mavlink_io.decimate`, checked to give 15.0 / 2.0 / 1.0 Hz). Phase 6 protocol; impact times from the full-rate stream.

| Input | Alarm before impact | Median / worst latency | Median margin | False alarms |
|---|---|---|---|---|
| 50 Hz (phase 6) | 32/32 | 0.10 / 0.11 s | 1.74 s | 0 in 0.73 h |
| **Radio rates**, model | **32/32** | **0.25 / 0.66 s** | 1.51 s | 0 in 0.73 h |
| Radio rates, tracking-error rule | 25/32 | 1.06 / 2.86 s | 0.66 s | 0 |
| Radio rates, lift rule | unavailable (no IMU on the radio) | | | |

The radio-rate model was also checked on the 16 **held-out** live flights: their recorded 50 Hz session, thinned and replayed through the live detector. Result: 8/8 before impact, median 0.32 s (max 0.60 s), 0 false alarms in 0.20 h. The impact time there comes from 15 Hz frames, so margins may be up to 67 ms optimistic.

What this shows:
- **The radio link costs little for partial faults.** AP drops by 0.01, inside the interval. Their signature, the controller pushing the weak motor harder and demanding more thrust, develops over seconds, and 15 Hz rates plus 1-2 Hz setpoints and motor outputs are enough to see it.
- **For catastrophic faults the model still beats impact, but with 2.5× the latency.** The worst case, 0.66 s, uses half of the shortest time to impact (1.28 s). The 15 Hz angular rates carry the detection.
- **The physics rule does not survive the radio.** Attitude-tracking error needs the setpoint, which arrives only twice a second, so the rule misses impact in 7 of 32 flights.
- Persistence stays 3 frames, which is 0.2 s at 15 Hz instead of 0.06 s.

A deployable radio model is saved as `data/models/stream_gbt_radio.joblib`. It stores its window, rate and staleness in the bundle, and the service thins a 50 Hz input to radio rates when given that model (`--model data/models/stream_gbt_radio.joblib`).

## 4. Answer to the phase 7 question

**The models do not transfer in the sense defined in §1.**

| Gap | Experiment | Ranking (AP) | Operating point |
|---|---|---|---|
| Sim (Gazebo, motor-off) → real (partial) | A | at chance | n/a |
| Sim (HIL, same fault model) → real | B, B2 | 80% of the real ceiling | fails: 768/h; 4.3/h (2× budget, unstable) after recalibration on real normal flights |
| Octorotor → quad (both real) | D | 22% of the real ceiling | fails: 363/h or 7% recall |

What would make transfer work, in order of likely value:
1. **A small amount of real fault data.** Real → real is the only setting that meets the false-alarm budget.
2. **Noise realism in simulation.** The main HIL → real difference is noise level (25×), and it can be measured from real no-fault flights.
3. **Per-vehicle normalisation**, e.g. features relative to each flight's own hover baseline, since motor command levels differ by vehicle.

## 5. Limitations

- 111 real and 120 HIL flights, a subset of RflyMAD selected for bandwidth (12 cases per flight mode). Intervals are wide: AP(HIL → real) [0.52-0.83].
- Real flights cover only 10-30% thrust loss. The catastrophic end of the severity axis exists only in simulation (SITL motor-off), so "catastrophic vs gradual" can't be separated from "sim vs real" for severe faults.
- Real no-fault airborne time is 2.5 h, so false-alarm rates near zero are imprecise.
- The radio experiment (E) uses one radio configuration, the one RflyMAD recorded. Other telemetry setups stream other rates, and radio packet loss was only present as recorded, not varied.
- The catastrophic radio result is simulated (SITL thinned to radio rates), not a real crash.
