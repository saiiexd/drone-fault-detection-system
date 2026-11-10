"""Evaluation: grouped cross-validation and detection metrics.

Leakage rules:
- Folds are grouped by session (recording day). Flights from one day share
  vehicle, place, weather and often the same fault, so a random or per-flight
  split would let the model recognise the day instead of the fault.
- The alarm threshold is chosen on a validation split carved from the
  training sessions, never on the test fold.

Metrics (test folds only, in-air windows of usable flights):
- window: precision, recall, F1 at the chosen threshold; average precision
  (area under the precision-recall curve, threshold-free).
- events: each merged label range is one event. Detected if any alarm falls
  in [start - EVENT_EARLY_S, end]. Latency = first alarm - range start.
  Ranges are approximate (paper: "regions of interest"), so latency is too.
- false alarms per in-air hour on Normal flights: alarm episodes (runs of
  consecutive alarming windows) divided by in-air time.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold

from aeropulse.features import HOP_ROWS, RATE_HZ
from aeropulse.models.dataset import MODEL_FEATURES

N_FOLDS = 5
SEED = 0
HOP_S = HOP_ROWS / RATE_HZ
EVENT_EARLY_S = 1.0
NORMAL_TRAIN_STRIDE = 5    # keep every 5th normal window (1 s apart) for training


def session_folds(df: pd.DataFrame, n_splits: int = N_FOLDS, seed: int = SEED):
    """Yield (train_idx, test_idx), grouped by session, stratified by flight class."""
    flights = df.groupby("flight_id").agg(session=("session", "first"),
                                          cls=("flight_primary", "first")).reset_index()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in sgkf.split(flights, flights["cls"], flights["session"]):
        tr_f, te_f = set(flights.loc[tr, "flight_id"]), set(flights.loc[te, "flight_id"])
        yield (np.flatnonzero(df["flight_id"].isin(tr_f).to_numpy()),
               np.flatnonzero(df["flight_id"].isin(te_f).to_numpy()))


def training_mask(df: pd.DataFrame) -> np.ndarray:
    """Windows used for fitting: drop ambiguous ones and thin out normal ones.

    A window labelled normal that partly overlaps a labelled range
    (0 < label_frac) is ambiguous, so it is excluded from training (it stays in
    evaluation). Consecutive windows overlap by 90%, so normal windows are
    subsampled; anomalous ones are all kept.
    """
    ambiguous = ~df["label_anomaly"].to_numpy() & (df["label_frac"].to_numpy() > 0)
    keep_normal = (df.groupby("flight_id").cumcount().to_numpy() % NORMAL_TRAIN_STRIDE) == 0
    return ~ambiguous & (df["label_anomaly"].to_numpy() | keep_normal)


def best_f1_threshold(y: np.ndarray, score: np.ndarray) -> float:
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f1[:-1])]) if len(t) else float("inf")


def window_metrics(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = score >= thr
    tp, fp, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "avg_precision": float(average_precision_score(y, score)) if y.any() else np.nan,
            "n_windows": int(len(y)), "n_positive": int(y.sum())}


def alarm_episodes(df: pd.DataFrame, pred: np.ndarray) -> int:
    """Number of alarm onsets: an alarm window whose predecessor (same flight,
    one hop earlier) was not alarming."""
    t = df["t_us"].to_numpy()
    f = df["flight_id"].to_numpy()
    prev_same = np.r_[False, (f[1:] == f[:-1]) & (np.diff(t) <= HOP_S * 1e6 * 1.5)]
    prev_pred = np.r_[False, pred[:-1]]
    return int((pred & ~(prev_same & prev_pred)).sum())


def merge_ranges(ranges: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping label ranges per flight into events."""
    out = []
    for fid, g in ranges.sort_values("start_us").groupby("flight_id"):
        cur = None
        for s, e, c in g[["start_us", "end_us", "class"]].itertuples(index=False):
            if cur and s <= cur[1]:
                cur[1] = max(cur[1], e)
                cur[2].add(c)
            else:
                if cur:
                    out.append((fid, *cur[:2], " + ".join(sorted(cur[2]))))
                cur = [s, e, {c}]
        out.append((fid, *cur[:2], " + ".join(sorted(cur[2]))))
    return pd.DataFrame(out, columns=["flight_id", "start_us", "end_us", "event_class"])


def event_metrics(df: pd.DataFrame, pred: np.ndarray, events: pd.DataFrame) -> pd.DataFrame:
    """One row per event that has in-air windows in the evaluated set."""
    t = df["t_us"].to_numpy()
    rows = []
    by_flight = {fid: idx for fid, idx in df.groupby("flight_id").indices.items()}
    for ev in events.itertuples(index=False):
        idx = by_flight.get(ev.flight_id)
        if idx is None:
            continue
        inside = idx[(t[idx] >= ev.start_us) & (t[idx] <= ev.end_us)]
        if not len(inside):
            continue   # event happened on the ground or in an excluded segment
        win = idx[(t[idx] >= ev.start_us - EVENT_EARLY_S * 1e6) & (t[idx] <= ev.end_us)]
        hits = win[pred[win]]
        rows.append({"flight_id": ev.flight_id, "event_class": ev.event_class,
                     "duration_s": (ev.end_us - ev.start_us) / 1e6,
                     "detected": bool(len(hits)),
                     "latency_s": (t[hits[0]] - ev.start_us) / 1e6 if len(hits) else np.nan})
    return pd.DataFrame(rows)


def false_alarms_per_hour(df: pd.DataFrame, pred: np.ndarray) -> tuple[float, float]:
    normal = (df["flight_class"] == "Normal").to_numpy()
    sub = df[normal]
    hours = len(sub) * HOP_S / 3600
    return alarm_episodes(sub, pred[normal]) / hours if hours else np.nan, hours


PERSIST = 3            # alarm only after 3 consecutive alarming windows (0.6 s)
TARGET_FA_PER_H = 2.0  # false-alarm budget for the deployable operating point


def persistence(df: pd.DataFrame, raw: np.ndarray, k: int = PERSIST) -> np.ndarray:
    """True where the last k windows of the same flight (contiguous) all alarm.

    Vectorised run-length count: a run restarts after a non-alarm window or
    at a break (new flight or a time gap).
    """
    t, f = df["t_us"].to_numpy(), df["flight_id"].to_numpy()
    brk = np.r_[True, (f[1:] != f[:-1]) | (np.diff(t) > HOP_S * 1e6 * 1.5)]
    idx = np.arange(len(raw))
    start = np.maximum.accumulate(np.where(~raw, idx + 1, np.where(brk, idx, 0)))
    run = np.where(raw, idx - start + 1, 0)
    return run >= k


def fa_budget_threshold(df: pd.DataFrame, score: np.ndarray,
                        target: float = TARGET_FA_PER_H) -> float:
    """Lowest threshold whose persistence-filtered alarms stay within the
    false-alarm budget on the Normal flights of this (validation) set."""
    normal = (df["flight_class"] == "Normal").to_numpy()
    sub, s = df[normal], score[normal]
    hours = len(sub) * HOP_S / 3600
    for thr in np.unique(np.quantile(s, np.linspace(0.90, 1.0, 201))):
        if alarm_episodes(sub, persistence(sub, s >= thr)) / hours <= target:
            return float(thr)
    return float(np.max(s) + 1e-9)


def cross_validate(df: pd.DataFrame, make_detectors, log=print):
    """Out-of-fold scores and alarms at two operating points per detector.

    - "f1": threshold maximising window F1 on validation, no persistence.
    - "fa": threshold meeting TARGET_FA_PER_H on validation, with persistence.
    """
    X = df[MODEL_FEATURES]
    y = df["label_anomaly"].to_numpy()
    train_ok = training_mask(df)
    names = [d.name for d in make_detectors()]
    scores = {n: np.full(len(df), np.nan) for n in names}
    alarms = {op: {n: np.zeros(len(df), dtype=bool) for n in names} for op in ("f1", "fa")}
    thresholds = {op: {n: [] for n in names} for op in ("f1", "fa")}
    fold = np.full(len(df), -1)

    for k, (tr, te) in enumerate(session_folds(df)):
        fold[te] = k
        # inner split of the training sessions: fit vs threshold validation
        inner_tr, inner_val = next(session_folds(df.iloc[tr], n_splits=4, seed=SEED + k))
        fit_idx, val_idx = tr[inner_tr], tr[inner_val]
        fit_idx = fit_idx[train_ok[fit_idx]]
        val_df, test_df = df.iloc[val_idx], df.iloc[te]
        for det in make_detectors():
            det.fit(X.iloc[fit_idx], y[fit_idx])
            s_val, s = det.score(X.iloc[val_idx]), det.score(X.iloc[te])
            thr_f1 = best_f1_threshold(y[val_idx], s_val)
            thr_fa = fa_budget_threshold(val_df, s_val)
            scores[det.name][te] = s
            alarms["f1"][det.name][te] = s >= thr_f1
            alarms["fa"][det.name][te] = persistence(test_df, s >= thr_fa)
            thresholds["f1"][det.name].append(thr_f1)
            thresholds["fa"][det.name].append(thr_fa)
        log(f"fold {k}: train {len(fit_idx)} / val {len(val_idx)} / test {len(te)} windows, "
            f"test positives {int(y[te].sum())}")
    return scores, alarms, thresholds, fold
