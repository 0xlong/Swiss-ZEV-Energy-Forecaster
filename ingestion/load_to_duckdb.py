"""
load_to_duckdb.py
-----------------
Bulk-load the three validated raw CSVs into a single DuckDB warehouse file
under a `raw` schema. Downstream dbt staging models read from `raw.*`.

ELT pattern:
    APIs ─► ingestion/*.py ─► data/raw/*.csv ─► load_to_duckdb.py ─► raw.*
                                                  (this file)

All paths and metadata come from `config` so dbt's profiles.yml and any
ML / dashboard scripts share the same warehouse path.

Usage:
    python ingestion/load_to_duckdb.py
"""

import logging
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# Source CSVs and target raw tables. The CSVs are expected to exist already
# (produced by pvgis_client.py, bwed_load_profile.py, entsoe_client.py,
# openmeteo_client.py).
TABLES = [
    ("raw.pvgis_hourly",           config.PVGIS_CSV,                ["pv_power_kw"]),
    ("raw.load_hourly",            config.LOAD_CSV,                 ["load_kw"]),
    ("raw.entsoe_prices",          config.ENTSOE_CSV,               ["price_eur_mwh"]),
    ("raw.openmeteo_hourly",       config.OPENMETEO_CSV,            ["ghi_w_m2", "temp_c"]),
    ("raw.entsoe_load_forecast",   config.ENTSOE_LOAD_FORECAST_CSV, ["load_forecast_mw"]),
    ("raw.entsoe_genmix",          config.ENTSOE_GENMIX_CSV,        ["generation_mw"]),
]

# load_hourly has 5 known DST duplicate ts_utc rows (one per spring-forward
# transition 2015–2019) caused by nonexistent='shift_forward' in
# bwed_load_profile.py mapping 02:00 local → 03:00+02:00 UTC, which collides
# with the real 03:00 row.  These are deduped (averaged) in stg_load_hourly.
# entsoe_genmix is long-format (one row per gen_type per timestamp), so
# (asset_id, ts_utc) naturally has multiple rows — not a true duplicate.
KNOWN_DUPLICATE_TABLES = {"raw.load_hourly", "raw.entsoe_genmix"}


def load(con: duckdb.DuckDBPyConnection, table: str, csv_path: Path) -> int:
    """Replace `table` with the contents of `csv_path`. Returns row count."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected {csv_path} — run the ingestion script first.")
    con.execute(
        f"CREATE OR REPLACE TABLE {table} AS "
        f"SELECT * FROM read_csv_auto(?, header=True)",
        [str(csv_path)],
    )
    (rows,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
    log.info("  %s ← %s  (%d rows)", table, csv_path.name, rows)
    return rows


def validate(con: duckdb.DuckDBPyConnection, table: str, non_null_cols: list[str]) -> None:
    """Assert: zero nulls in key value columns; warn (not fail) on known DST dupes."""
    for col in non_null_cols:
        (nulls,) = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} IS NULL").fetchone()
        assert nulls == 0, f"{table}.{col} has {nulls} nulls"

    # All three tables carry asset_id; pvgis carries ts_local-as-UTC, the other
    # two carry ts_utc. Use whichever timestamp column exists.
    cols = {r[0] for r in con.execute(f"DESCRIBE {table}").fetchall()}
    ts_col = "ts_utc" if "ts_utc" in cols else "ts_local"
    (dupes,) = con.execute(
        f"SELECT COUNT(*) FROM ("
        f"  SELECT asset_id, {ts_col}, COUNT(*) c FROM {table} "
        f"  GROUP BY asset_id, {ts_col} HAVING c > 1)"
    ).fetchone()
    if dupes > 0 and table in KNOWN_DUPLICATE_TABLES:
        log.warning("  %s: %d known DST duplicate rows — will be averaged in stg_load_hourly", table, dupes)
    else:
        assert dupes == 0, f"{table}: {dupes} unexpected duplicate (asset_id, {ts_col}) rows"


def main() -> None:
    log.info("=== Loading raw CSVs into %s ===", config.DUCKDB_PATH)
    config.DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(config.DUCKDB_PATH))
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")

    total = 0
    for table, csv_path, non_null_cols in TABLES:
        rows = load(con, table, csv_path)
        validate(con, table, non_null_cols)
        total += rows

    log.info("Validation passed — %d rows across %d tables", total, len(TABLES))
    log.info("=== Done. Run `cd dbt_project && dbt build` next ===")
    con.close()


if __name__ == "__main__":
    main()
