"""DuckDB database of views over the Parquet lake.

The .duckdb file stores only view definitions, no data copies: queries read
the Parquet files directly and DuckDB prunes partitions (flight_id, topic)
from the hive-style directory names.

    import duckdb
    con = duckdb.connect("data/lake/aeropulse.duckdb", read_only=True)
    con.sql("select flight_class, count(*) from flights where usable group by 1")
"""

from pathlib import Path

import duckdb

DB_NAME = "aeropulse.duckdb"


def build_database(lake: Path) -> Path:
    lake = lake.resolve()
    path = lake / DB_NAME
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    con.execute(f"create view flights as select * from read_parquet('{lake}/catalog/flights.parquet')")
    con.execute(f"create view anomaly_ranges as select * from "
                f"read_parquet('{lake}/catalog/anomaly_ranges.parquet')")
    for layer in ("cleaned", "features"):
        if any((lake / layer).glob("flight_id=*/*.parquet")):
            con.execute(f"create view {layer} as select * from read_parquet("
                        f"'{lake}/{layer}/flight_id=*/*.parquet', hive_partitioning = true, "
                        f"union_by_name = true)")
    # One view per raw topic instance, e.g. raw_sensor_combined_0. union_by_name
    # lets logs from different PX4 versions (different field sets) coexist.
    for topic_dir in sorted((lake / "raw").glob("topic=*")):
        topic = topic_dir.name.split("=", 1)[1]
        con.execute(f'create view "raw_{topic}" as select * from read_parquet('
                    f"'{topic_dir}/flight_id=*/*.parquet', hive_partitioning = true, "
                    f"union_by_name = true)")
    con.close()
    return path
