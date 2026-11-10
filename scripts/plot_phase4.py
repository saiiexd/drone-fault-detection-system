"""Precision-recall curves of the phase 4 detectors from out-of-fold scores.

Usage:
    uv run python scripts/plot_phase4.py --pred reports/phase4/oof_predictions.parquet \
        --out reports/phase4/pr_curves.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

DETECTORS = [("ekf_test_ratio", "EKF test ratio"), ("physics_rule", "Physics rule"),
             ("isolation_forest", "Isolation Forest"), ("gbt_binary", "Gradient-boosted trees")]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.pred)

    any_y = df["label_anomaly"].to_numpy()
    # Mechanical vs normal: drop windows of the other fault classes.
    mech = df[(df["y_class"] == "Mechanical") | ~df["label_anomaly"]]
    tasks = [("Any labelled anomaly vs normal", df, any_y),
             ("Mechanical (motor) fault vs normal", mech, (mech["y_class"] == "Mechanical").to_numpy())]

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE,
                             constrained_layout=True, sharey=True)
    for ax, (title, d, y) in zip(axes, tasks, strict=True):
        ax.set_facecolor(SURFACE)
        for (col, label), color in zip(DETECTORS, SERIES, strict=True):
            s = d[f"score_{col}"].to_numpy()
            p, r, _ = precision_recall_curve(y, s)
            ap_ = average_precision_score(y, s)
            ax.plot(r, p, color=color, lw=2, label=f"{label} (AP {ap_:.2f})")
        base = y.mean()
        ax.axhline(base, color=MUTED, lw=1, ls="--")
        ax.text(1.0, base, f"chance {base:.3f}", ha="right", va="bottom", fontsize=8, color=MUTED)
        ax.set_title(f"{title}  (n={len(y):,}, positives {int(y.sum()):,})",
                     loc="left", fontsize=9, color=TEXT)
        ax.set_xlabel("Recall")
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    axes[0].set_ylabel("Precision")
    fig.suptitle("Phase 4 detectors, out-of-fold (5-fold, grouped by session), in-air windows",
                 x=0.01, ha="left", fontsize=10, color=TEXT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(args.out)


if __name__ == "__main__":
    assert np is not None
    main()
