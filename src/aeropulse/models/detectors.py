"""Anomaly detectors, simplest first. Each returns a score: higher = more anomalous.

1. ekf_test_ratio  - PX4's own estimator consistency check (no training).
                     Answers: does the autopilot already flag these anomalies?
2. physics_rule    - motors saturated while altitude falls (no training).
3. isolation_forest- unsupervised, fitted on normal windows only.
4. gbt_binary      - supervised gradient-boosted trees, anomaly vs normal.

All share fit(X, y) / score(X); rules ignore fit. Thresholds are chosen
outside the detector, on a validation split (see evaluate.py).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from aeropulse.models.dataset import MODEL_FEATURES

SEED = 0


class EkfTestRatio:
    name = "ekf_test_ratio"
    cols = ["tr_vel__max", "tr_pos__max", "tr_hgt__max", "tr_mag__max"]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.cols].max(axis=1).fillna(0).to_numpy()


class PhysicsRule:
    """Share of saturated motors x metres of altitude lost in the window.

    Saturation alone is common in healthy flight (phase 2 finding); it only
    indicates lost lift when the vehicle sinks at the same time.
    """
    name = "physics_rule"

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        sat = X["m_sat_frac__mean"].fillna(0)
        sink = (-X["alt_delta"]).clip(lower=0).fillna(0)
        return (sat * sink).to_numpy()


class IsoForest:
    name = "isolation_forest"

    def __init__(self, max_train: int = 200_000):
        self.max_train = max_train

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        normal = X[~y]
        if len(normal) > self.max_train:
            normal = normal.sample(self.max_train, random_state=SEED)
        self.medians = normal.median()
        self.model = IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1)
        self.model.fit(normal.fillna(self.medians))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return -self.model.score_samples(X.fillna(self.medians))


class GbtBinary:
    name = "gbt_binary"

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            class_weight="balanced", random_state=SEED)
        self.model.fit(X, y)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


def make_detectors():
    return [EkfTestRatio(), PhysicsRule(), IsoForest(), GbtBinary()]


def make_classifier() -> HistGradientBoostingClassifier:
    """Fault-type classifier (Normal + 4 fault classes)."""
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
        class_weight="balanced", random_state=SEED)


__all__ = ["MODEL_FEATURES", "make_detectors", "make_classifier"]
