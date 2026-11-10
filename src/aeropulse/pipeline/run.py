"""Build the lake from downloaded UAV-SEAD logs.

Usage:
    uv run python -m aeropulse.pipeline.run --raw data/raw/uav_sead --lake data/lake
    uv run python -m aeropulse.pipeline.run ... --limit 20      # quick trial
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from aeropulse.db import build_database
from aeropulse.pipeline.flight import process_flight, remove_flight_outputs, write_parquet
from aeropulse.pipeline.quality import apply_rules


def anomaly_ranges(entries: list[dict]) -> pd.DataFrame:
    rows = [{"flight_id": e["sha256"][:16], "class": a["class"], "signal": sig,
             "start_us": int(s), "end_us": int(t)}
            for e in entries for a in e["annotations"] for sig, rr in a["ranges"]
            for s, t in rr]
    return pd.DataFrame(rows, columns=["flight_id", "class", "signal", "start_us", "end_us"])


def build(raw: Path, lake: Path, workers: int, limit: int | None = None,
          write_raw: bool = False) -> pd.DataFrame:
    manifest = json.loads((raw / "manifest.json").read_text())
    entries = manifest["files"][:limit] if limit else manifest["files"]

    # Idempotent: every run rebuilds each flight from scratch, so outputs never
    # mix results from different code versions.
    for e in entries:
        remove_flight_outputs(lake, e["sha256"][:16])

    rows, t0 = [], time.time()
    # No max_tasks_per_child: with it, the pool hung once every worker had hit
    # its task limit (6 workers x 25 tasks = 150 flights, then no progress).
    with ProcessPoolExecutor(workers) as pool:
        futures = [pool.submit(process_flight, e, raw, lake, write_raw) for e in entries]
        for i, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(entries):
                print(f"{i}/{len(entries)} flights, {time.time() - t0:.0f} s", flush=True)

    catalog = apply_rules(pd.DataFrame(rows).sort_values("key", ignore_index=True))
    catalog["source"] = manifest.get("source", "uav_sead")
    catalog["dataset_revision"] = manifest["revision"]
    write_parquet(catalog, lake / "catalog" / "flights.parquet")
    write_parquet(anomaly_ranges(entries), lake / "catalog" / "anomaly_ranges.parquet")
    build_database(lake)
    return catalog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--lake", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--with-raw", action="store_true",
                    help="also write the raw Parquet layer (every topic, ~0.5x log size)")
    args = ap.parse_args()
    cat = build(args.raw, args.lake, args.workers, args.limit, args.with_raw)
    print(f"\n{len(cat)} flights, {cat['usable'].sum()} usable, "
          f"{cat['error'].notna().sum()} errors")
    print(cat.groupby("flight_class")["usable"].agg(["size", "sum"]).to_string())
    fails = cat["failed_rules"].dropna().str.split(",").explode().value_counts()
    if len(fails):
        print("\nfailed rules:\n" + fails.to_string())


if __name__ == "__main__":
    main()
