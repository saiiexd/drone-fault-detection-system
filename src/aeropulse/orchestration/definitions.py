"""Dagster definitions: the batch pipeline as assets with data-quality checks.

Dagster adds a lineage graph, a run history with metadata, and checks that
are evaluated after every build. The work itself is the same
``aeropulse.pipeline.run.build`` function used by the CLI.

    uv run dagster dev                                   # web UI on localhost:3000
    uv run dagster asset materialize -m aeropulse.orchestration.definitions --select '*'
"""

import os
from pathlib import Path

import duckdb
from dagster import (
    AssetCheckResult,
    AssetCheckSeverity,
    AssetExecutionContext,
    AssetSpec,
    Config,
    Definitions,
    MaterializeResult,
    MetadataValue,
    asset,
    asset_check,
)

from aeropulse.db import DB_NAME
from aeropulse.pipeline.run import build

ROOT = Path(__file__).resolve().parents[3]
RAW = Path(os.environ.get("AEROPULSE_RAW", ROOT / "data" / "raw" / "uav_sead"))
LAKE = Path(os.environ.get("AEROPULSE_LAKE", ROOT / "data" / "lake"))

uav_sead_raw = AssetSpec(
    "uav_sead_raw",
    description="Downloaded UAV-SEAD ULog files + manifest (scripts/download_uavsead.py).",
)


class LakeConfig(Config):
    workers: int = 6
    limit: int | None = None
    write_raw: bool = False


@asset(deps=[uav_sead_raw], description="Raw, cleaned and feature Parquet layers, "
       "flight catalog with quality flags, and DuckDB views.")
def uav_sead_lake(context: AssetExecutionContext, config: LakeConfig) -> MaterializeResult:
    cat = build(RAW, LAKE, config.workers, config.limit, config.write_raw)
    by_class = cat.groupby("flight_class")["usable"].agg(["size", "sum"])
    context.log.info("\n" + by_class.to_string())
    return MaterializeResult(metadata={
        "flights": len(cat),
        "usable": int(cat["usable"].sum()),
        "errors": int(cat["error"].notna().sum()),
        "per_class": MetadataValue.md(f"```\n{by_class.to_string()}\n```"),
    })


def _query(sql: str):
    con = duckdb.connect(str(LAKE / DB_NAME), read_only=True)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


@asset_check(asset=uav_sead_lake, description="No log fails to parse.")
def no_parse_errors() -> AssetCheckResult:
    (n,), = _query("select count(*) from flights where error is not null")
    return AssetCheckResult(passed=n == 0, severity=AssetCheckSeverity.WARN,
                            metadata={"errors": n})


@asset_check(asset=uav_sead_lake, description="At most 5% of flights lack required signals.")
def required_signals_coverage() -> AssetCheckResult:
    (frac,), = _query("select avg((missing_required is not null)::int) from flights")
    return AssetCheckResult(passed=frac <= 0.05, metadata={"missing_frac": float(frac)})


@asset_check(asset=uav_sead_lake, description="Every label range lies inside its log.")
def labels_inside_logs() -> AssetCheckResult:
    (n,), = _query("select count(*) from flights where not ranges_in_span")
    return AssetCheckResult(passed=n == 0, metadata={"flights_with_bad_ranges": n})


@asset_check(asset=uav_sead_lake, description="Every cleaned row's label agrees with the "
             "label ranges, recomputed independently in SQL.")
def labels_match_ranges() -> AssetCheckResult:
    (wrong, total), = _query("""
        select count(*) filter (where c.label_anomaly != exists (
                   select 1 from anomaly_ranges r
                   where r.flight_id = c.flight_id and c.t_us between r.start_us and r.end_us)),
               count(*)
        from cleaned c
    """)
    return AssetCheckResult(passed=wrong == 0, metadata={"mismatched_rows": wrong,
                                                         "rows": total})


@asset_check(asset=uav_sead_lake, description="Every fault class keeps usable flights.")
def usable_flights_per_class() -> AssetCheckResult:
    rows = _query("select flight_class, sum(usable::int) from flights group by 1")
    empty = [c for c, n in rows if n == 0]
    return AssetCheckResult(passed=not empty, metadata={"classes_without_usable": str(empty)})


defs = Definitions(
    assets=[uav_sead_raw, uav_sead_lake],
    asset_checks=[no_parse_errors, required_signals_coverage, labels_inside_logs,
                  labels_match_ranges, usable_flights_per_class],
)
