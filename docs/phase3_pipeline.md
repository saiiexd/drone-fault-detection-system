# Phase 3: batch pipeline (ULog → Parquet lake → DuckDB), orchestrated by Dagster

All numbers below were measured on 2026-11-08 on the full annotated UAV-SEAD set, excluding "Uncategorized". Hardware: i5-12450HX, 16 GB RAM, 6 worker processes.

Reproduce:
```sh
uv run python scripts/download_uavsead.py --out data/raw/uav_sead --all
DAGSTER_HOME=$PWD/data/dagster_home uv run dagster asset materialize \
    -m aeropulse.orchestration.definitions --select '*'
uv run pytest
```

## 1. Result

| | Value |
|---|---|
| Logs downloaded and verified (SHA-256) | 1,254 (11.42 GB) |
| Build time, all layers | 104 s (126 s including Dagster start-up and checks) |
| Usable flights | **1,148** (44.97 h) |
| Cleaned rows (50 Hz) | 8,797,722 |
| Raw IMU samples | 42,408,719 |
| Lake size | raw 5.9 GB, cleaned 1.3 GB, features 643 MB |
| Tests | 47 passed |

Usable flights per class:

| Class | Flights | Usable | Usable hours |
|---|---|---|---|
| Normal | 899 | 829 | 35.47 |
| External Position | 193 | 169 | 3.95 |
| Altitude | 73 | 68 | 2.11 |
| Global Position | 40 | 38 | 1.58 |
| Mechanical | 41 | 36 | 1.59 |
| Multi-class (4 kinds) | 8 | 8 | 0.27 |

In-air feature windows, by label at the window end:

| Label | Windows |
|---|---|
| Normal | 730,687 |
| External Position | 17,741 |
| Global Position | 10,000 |
| Altitude | 7,433 |
| **Mechanical** | **3,376 (0.46% of Normal)** |

**This class imbalance is the main constraint for phase 4.**

## 2. Architecture

```
manifest.json + .ulg ──► process_flight (one process per log, 6 in parallel)
                            ├─ raw/topic=<instance>/flight_id=<id>/      every topic as logged
                            ├─ cleaned/flight_id=<id>/                   canonical signals, causal 50 Hz, labels
                            └─ features/flight_id=<id>/                  2 s windows every 0.2 s
                         ──► catalog/flights.parquet (metadata + quality flags)
                             catalog/anomaly_ranges.parquet
                         ──► aeropulse.duckdb (views only, no data copies)
```

| Module | Role |
|---|---|
| `src/aeropulse/signals.py` | Canonical signal names with candidate sources per PX4 version |
| `src/aeropulse/pipeline/flight.py` | Per-flight processing |
| `src/aeropulse/features.py` | Window features, shared with the future streaming detector |
| `src/aeropulse/pipeline/quality.py` | Quality rules |
| `src/aeropulse/pipeline/run.py` | Command-line build |
| `src/aeropulse/db.py` | DuckDB views |
| `src/aeropulse/orchestration/definitions.py` | Dagster asset and 5 asset checks |

## 3. Design decisions

- **Hive-partitioned Parquet** (`flight_id=…`). Directory names act as columns, and DuckDB skips files that a filter excludes. Measured: a one-flight query reads **1 of 1,251 files** (63 ms), and a full scan of 8.8 M cleaned rows with a join and group-by takes 280 ms.
- **Raw layer keeps every topic as logged.** Later layers can be rebuilt without re-parsing the ULog files, which is the slowest step. It costs 5.9 GB.
- **Different PX4 versions.** The logs cover PX4 v1.7.3 to v1.13.0, and 386 logs have no version string. `signals.py` maps each canonical signal to candidate sources in priority order (e.g. body rates from `vehicle_attitude` in older versions, `vehicle_angular_velocity` in newer ones). DuckDB views use `union_by_name` so schemas that differ by version coexist. Result: only 1 of 1,252 parsed logs lacks a required signal.
- **Staleness limit per topic:** 5 sample periods, clamped to 0.05-5 s. This replaces the single phase 2 limit.
- **IMU vibration** (rolling std of |acc| and |gyro| over 100 ms) is computed at the native 250 Hz before downsampling. This answers the phase 2 open question about losing high-frequency content.
- **Motor features don't depend on motor count** (max, min, mean, spread, saturation fraction across active motors). Motor commands are normalised with each vehicle's own `PWM_MIN`/`PWM_MAX`.
- **Absolute position and heading are excluded from features.** They identify the flight's place and day, not the fault.
- **Idempotent builds.** Each run deletes and rebuilds each flight's outputs, and files are written atomically (temporary file, then rename). A run never mixes outputs from different code versions, and a crash never leaves half-written files.
- **Dagster** wraps the same `build()` function. It adds lineage, run history with metadata, and asset checks. The pipeline still runs without it: `python -m aeropulse.pipeline.run`.

## 4. Data problems found at full scale, and fixes

| Problem | Evidence | Fix |
|---|---|---|
| **Two airframes, not one.** 1,075 logs are an octorotor coaxial (`MAV_TYPE` 14, 8 motors); 177 are a quadcopter (`MAV_TYPE` 2, airframe 4014). | Quad logs report `noutputs=8`, but only outputs 0-3 vary like motors | Motor count comes from `MAV_TYPE`, not `noutputs` |
| Corrupt timestamps near 2^64 | 10,152 samples; the header end time gave a "10-billion-hour" duration | Samples beyond start + 24 h dropped before the int64 cast; duration taken from the data |
| Logger start-up gap: all topics pause in the first seconds | 186 of 197 long gaps were in the first 10% of a log | Data before a gap of more than 1 s in the first 10 s is trimmed (79 flights, median 3.5 s trimmed) |
| Label ranges end slightly after the log ends | 11 flights, overrun of 0.03-1.1 s | ±2 s tolerance; labels apply only to rows that exist |
| Two files fail to parse | `ULogException: Header too short` (SHA-256 matches the source, so the files themselves are broken) | Recorded in the catalog as `parsed=False` |
| `ProcessPoolExecutor(max_tasks_per_child=25)` hung | Stopped at exactly 150 = 6 × 25 flights, no worker processes left, no OOM in the kernel log | Option removed |

Quality rule failures (a flight can fail several):

| Rule | Flights failing |
|---|---|
| Long gap in mid-flight | 65 |
| Less than 20% airborne | 46 |
| More than 5% missing required signals | 20 |
| Missing required signals | 3 |
| Parse error | 2 |
| Motor count unknown | 2 |
| Duration under 10 s | 2 |
| Anomaly class without ranges | 2 |

## 5. Asset checks (last Dagster run)

| Check | Result |
|---|---|
| `labels_match_ranges`: every cleaned row's label recomputed in SQL from the ranges | **pass**, 0 of 8,797,722 rows differ |
| `labels_inside_logs` | pass, 0 flights |
| `required_signals_coverage` | pass, 0.08% missing |
| `usable_flights_per_class` | pass |
| `no_parse_errors` (warning) | fails: the 2 corrupt files above (expected) |

## 6. What this means for later phases

- **Phase 4.**
  - Mechanical faults are 0.46% of in-air windows, so accuracy is meaningless. Report precision, recall and F1, false alarms per flight-hour, and detection latency.
  - Splits must group by **session (day)**, not only by flight: flights on the same day share vehicle, place and conditions.
  - The octorotor and the quadcopter must be reported separately.
- **Phase 7 (sim-to-real).** Real Mechanical labels exist almost only for the octorotor: 36 usable flights, against 1 quad flight with Altitude + Mechanical. RflyMAD and PX4 SITL are quadcopters. A sim-trained quad model therefore cannot be tested on real quad motor faults from UAV-SEAD alone; the airframe difference is part of the gap. Phase 7 needs either RflyMAD's real-flight subset as the real test set, or an explicit cross-airframe experiment. This must be decided before phase 7, not after.

## 7. Plots

`reports/phase3/flight_*.png`, made with `scripts/plot_flight.py`:
- two octorotor Mechanical flights (thrust at its limit, a motor saturated, altitude loss);
- one Normal flight;
- one quad flight labelled Altitude + Mechanical: thrust and motors collapse, tilt reaches 30°, and after the crash the altitude estimate drifts to -20 m.

## 8. Update after phase 5: state topics must not go stale

PX4 v1.17 SITL logs showed that `vehicle_land_detected` (landed) and `vehicle_status` (arming state) can be published **only when they change**, e.g. once at takeoff. The per-topic staleness limit (5 sample periods, at most 5 s) turned "not landed" into NaN a few seconds after takeoff, and those windows dropped out of the in-air filter.

**Fix:** state topics never go stale (`STATE_TOPICS` in `pipeline/flight.py`, with a test). The UAV-SEAD lake was rebuilt:

| | Before | After |
|---|---|---|
| Usable flights | 1,148 | **1,176** (46.1 h) |
| In-air windows | 769,266 | **783,610** |
| Flights failing the "airborne" rule | 46 | 15 |

The affected flights were mostly the newer PX4 v1.10 logs. The phase 4 numbers are from the rebuilt lake.

## 9. Update: raw Parquet layer made optional

The raw layer (every topic as logged) took 5.9 GB, half the size of the ULog files it copies. Every build re-parses the ULogs anyway, so it saved no build time. It's now off by default: use `--with-raw` on the CLI, or `write_raw: true` in the Dagster config. The existing copies were deleted, and the DuckDB views were rebuilt without the `raw_*` views. Lake size is now 1.9 GB (cleaned 1.3 GB + features 0.6 GB).
