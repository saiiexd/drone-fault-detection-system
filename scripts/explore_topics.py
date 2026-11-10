"""Build a per-log, per-topic inventory of logging rates and gaps.

Usage:
    uv run python scripts/explore_topics.py --raw data/raw/uav_sead --out reports/phase2
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aeropulse.ingest.ulog_reader import load_topics


def topic_stats(df: pd.DataFrame) -> dict:
    t = df["timestamp"].to_numpy()
    gaps = np.diff(t)
    row = {"n_msgs": len(t), "n_fields": df.shape[1] - 1}
    if len(gaps) >= 2:
        med = float(np.median(gaps))
        row |= {
            "rate_hz": 1e6 / med,
            "jitter_p95_over_median": float(np.percentile(gaps, 95)) / med,
            "max_gap_ms": float(gaps.max()) / 1e3,
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.raw / "manifest.json").read_text())
    rows = []
    for entry in manifest["files"]:
        for name, df in load_topics(args.raw / entry["path"]).items():
            rows.append({"flight": entry["key"], "class": entry["class"],
                         "instance": name, **topic_stats(df)})
    inv = pd.DataFrame(rows)
    inv.to_csv(args.out / "topic_inventory.csv", index=False, float_format="%.3f")

    n_flights = inv["flight"].nunique()
    summary = (inv.groupby("instance")
                  .agg(flights=("flight", "nunique"),
                       rate_hz=("rate_hz", "median"),
                       jitter=("jitter_p95_over_median", "median"),
                       worst_gap_ms=("max_gap_ms", "max"))
                  .sort_values(["flights", "rate_hz"], ascending=False))
    summary.to_csv(args.out / "topic_summary.csv", float_format="%.2f")
    print(f"{len(inv)} topic instances across {n_flights} flights")
    print(summary.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
