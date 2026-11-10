# Phase 4: baseline fault detection on UAV-SEAD, with leakage-safe evaluation

All numbers below are out-of-fold results measured on 2026-11-08, after the state-topic fix described in phase 5. Every flight was tested exactly once, by a model that never saw any flight from the same recording day.

Reproduce:
```sh
uv run python -m aeropulse.models.run        --lake data/lake --out reports/phase4 --models data/models   # ~10 min
uv run python -m aeropulse.models.mechanical --lake data/lake --out reports/phase4 --models data/models   # ~3 min
uv run python scripts/plot_phase4.py --pred reports/phase4/oof_predictions.parquet --out reports/phase4/pr_curves.png
```

## 1. Setup

- **Data:**
  - 783,610 in-air windows (2 s, one every 0.2 s) from 1,176 usable flights recorded on 139 days.
  - 39,629 windows (5.1%) fall inside a labelled anomaly.
  - Ground segments are excluded, because "on the ground" is trivially different from flight.
- **Splits:** 5-fold `StratifiedGroupKFold`.
  - **Grouped by session (recording day):** flights from one day share vehicle, place, weather and often the same fault, so a per-flight or random split would let a model recognise the day.
  - **Stratified by flight class**, so every fold has every class.
- **Thresholds:** chosen on a validation split cut from the *training* days (another grouped split), never on the test fold.
- **Training data:**
  - Normal windows are thinned to one per second, because consecutive windows overlap by 90%.
  - Windows that overlap the edge of a label range are left out of training (still evaluated), because their label is ambiguous.
- **Two operating points:**
  - **f1:** the threshold that maximises window F1 on validation, applied to raw windows.
  - **fa (deployable):** the lowest threshold with ≤ 2 false alarms per in-air hour on validation, plus a **persistence filter** (3 consecutive alarming windows, i.e. 0.6 s).
- **Metrics:**
  - Window precision, recall and F1, and average precision (AP, the area under the precision-recall curve; it doesn't depend on a threshold).
  - **Event recall:** a merged label range counts as detected if an alarm falls in [start − 1 s, end].
  - **Latency:** first alarm minus range start. It is approximate because the ranges are approximate.
  - **False alarms per in-air hour** on Normal flights, counted as alarm onsets.
- Tests for splits, persistence, events and false alarms are in `tests/test_evaluate.py`.

## 2. Detectors (simplest first)

| Detector | Training | Idea |
|---|---|---|
| `ekf_test_ratio` | none | PX4's own estimator consistency check (max of velocity, position, height and magnetometer test ratios). Asks whether the autopilot already flags these anomalies. |
| `physics_rule` | none | Share of saturated motors × metres of altitude lost in the window. "Full power but still sinking." |
| `isolation_forest` | normal windows only | Unsupervised outlier score |
| `gbt_binary` | all labels | `HistGradientBoostingClassifier`, any anomaly vs normal, class-balanced |
| `gbt_mechanical` | Mechanical vs Normal | The same model, trained only on motor faults |
| type classifier | all labels | The same model, 5 classes (Normal + 4 fault types) |

## 3. Results: any anomaly vs normal (492 events, 34.4 h of normal flight)

| Detector | Operating point | Precision | Recall | F1 | AP | Event recall | Median latency | False alarms/h |
|---|---|---|---|---|---|---|---|---|
| ekf_test_ratio | f1 | 0.142 | 0.329 | 0.199 | 0.143 | 0.547 | −0.44 s | 51.0 |
| physics_rule | f1 | 0.117 | 0.154 | 0.133 | 0.092 | 0.465 | 0.63 s | 89.8 |
| isolation_forest | f1 | 0.417 | 0.226 | 0.293 | 0.228 | 0.313 | 1.81 s | 27.5 |
| **gbt_binary** | f1 | **0.518** | **0.336** | **0.408** | **0.393** | 0.459 | 0.99 s | 26.1 |
| ekf_test_ratio | fa | 0.325 | 0.137 | 0.193 | | 0.248 | 1.06 s | 4.9 |
| isolation_forest | fa | 0.607 | 0.035 | 0.065 | | 0.108 | 6.50 s | 2.1 |
| **gbt_binary** | fa | **0.700** | 0.244 | **0.362** | | **0.329** | 2.72 s | 4.3 |

Event recall per class for `gbt_binary` at the deployable point (number of events in brackets):

| Class | Event recall |
|---|---|
| External Position (209) | 0.53 |
| Mechanical (62) | 0.58 |
| Global Position (86) | 0.13 |
| Altitude (132) | 0.02 |

What this shows:
- **Generic anomaly detection on UAV-SEAD is hard.** The best AP is 0.39, against 0.05 for chance. The labels are *state-estimation* anomalies judged by experts against what really happened. Altitude and global-position errors mostly don't show up in the vehicle's internal signals, which is what makes them estimator errors in the first place.
- **PX4's own EKF test ratios are a weak detector** (AP 0.14). The autopilot doesn't already know about most of these anomalies, which is the dataset's premise.
- **Negative latency** for the EKF test ratio means its alarms fire just before the annotated start. Annotators marked when an anomaly became *visible* in the plots, which can lag the estimator's own signal.

## 4. Results: motor faults (Mechanical vs Normal)

64 Mechanical events (63 octorotor, 1 quadcopter); 3,445 positive windows (0.46%).

| Detector | Operating point | Precision | Recall | F1 | AP | Event recall | Median latency | False alarms/h |
|---|---|---|---|---|---|---|---|---|
| ekf_test_ratio | f1 | 0.012 | 0.498 | 0.024 | 0.011 | 0.828 | −0.29 s | 88.8 |
| physics_rule | f1 | 0.383 | 0.375 | 0.379 | 0.262 | 0.812 | 1.94 s | 4.0 |
| **gbt_mechanical** | f1 | **0.546** | **0.696** | **0.612** | **0.552** | 0.828 | **1.29 s** | **3.4** |
| physics_rule | fa | 0.380 | 0.316 | 0.345 | | 0.750 | 2.36 s | 3.5 |
| gbt_mechanical | fa | 0.396 | 0.747 | 0.518 | | **0.859** | 1.18 s | 5.5 |

What this shows:
- **A dedicated model matters.** The general `gbt_binary` reaches only AP 0.12 on motor faults, because they are 9% of its positives (see `reports/phase4/pr_curves.png`). Trained only on motor faults, the same model reaches **0.55**.
- **The two-feature physics rule is a strong, interpretable baseline:** AP 0.26 and event recall 81%, with no training.
- **The EKF test ratio is useless for motor faults** (AP 0.011, chance is 0.005).
- **The false-alarm budget did not transfer.** Thresholds chosen for ≤ 2 false alarms per hour on validation days gave 3.5-5.5 per hour on test days. Score distributions shift between recording days.
- **The selected `gbt_mechanical` deployable threshold is degenerate: 0.001.** The model scores normal windows at almost exactly 0, so the false-alarm budget isn't binding, and the search returns the lowest candidate it tries. That makes the operating point a hair-trigger. The 5.5 false alarms per hour above is its measured cost. Phase 5 shows the consequence.

## 5. Fault-type classification (5 classes, window level)

| True \ Predicted | Normal | Mechanical | Global Pos. | Altitude | External Pos. |
|---|---|---|---|---|---|
| Normal | 724,724 | 1,269 | 3,869 | 5,283 | 8,836 |
| Mechanical | 1,397 | **2,044** | 0 | 0 | 4 |
| Global Position | 8,201 | 32 | **1,088** | 2 | 740 |
| Altitude | 6,853 | 0 | 6 | **568** | 35 |
| External Position | 7,602 | 359 | 150 | 34 | **10,514** |

Per-class F1:

| Class | F1 |
|---|---|
| Normal | 0.97 |
| Mechanical | 0.57 |
| External Position | 0.54 |
| Global Position | 0.14 |
| Altitude | 0.09 |

Macro F1 is 0.46. Motor faults are rarely confused with estimator faults (4 of 3,445). Most errors are confusion with Normal.

## 6. Saved models (`data/models/`, gitignored)

| File | Content |
|---|---|
| `uavsead_gbt.joblib` | `gbt_binary` + type classifier, feature list, thresholds (f1 0.682, fa 0.837), persistence setting |
| `mechanical_gbt.joblib` | `gbt_mechanical`, thresholds (f1 0.964, fa 0.001), physics-rule threshold for the false-alarm budget (0.045) |

Thresholds come from held-out days (fit on 3/4 of days, threshold on the remaining 1/4).

## 7. Limitations and next steps

- **Labels are approximate ranges** drawn by annotators, so latency numbers are approximate. SITL in phase 5 gives exact injection times.
- **Motor faults are almost all octorotor** (63 of 64 events).
- **The threshold search needs a lower bound and a calibration step** (e.g. pick the threshold from the false-alarm *and* precision constraints, or calibrate the probabilities) to avoid degenerate thresholds like 0.001.
- **Deep models (1D-CNN, autoencoder)** were not trained. They are worth trying only against these numbers (AP 0.55 for motor faults, 0.39 for any anomaly), with the same folds.
