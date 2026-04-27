"""
features.py
-----------
Shared utilities for loading features from DuckDB and chronological splitting.

Functions
---------
load_price_features()
    Read fct_hourly_market from DuckDB, drop lag-null rows.

load_pv_features()
    Read fct_pv_features from DuckDB, drop lag-null rows.

chronological_split(df, train_months=48, val_months=6, test_months=6)
    Split df into (train, val, test) strictly by calendar month.
    Never shuffles — preserves temporal order.
    Default boundaries for 2015-2019:
        train  2015-01 → 2018-12  (48 months)
        val    2019-01 → 2019-06  (6 months)
        test   2019-07 → 2019-12  (6 months)
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# dbt-duckdb schema: target.schema='main' + custom '+schema: marts' = 'main_marts'
_MARTS_SCHEMA = "main_marts"

# Lag columns that must be non-null for a row to be usable (first 168 h are NaN)
_PRICE_LAG_COLS = ["price_lag_1h", "price_lag_24h", "price_lag_168h"]
_PV_LAG_COLS    = ["pv_lag_1h", "pv_lag_24h", "pv_lag_168h"]


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(config.DUCKDB_PATH), read_only=True)


# ---------------------------------------------------------------------------
# Feature loaders
# ---------------------------------------------------------------------------

def load_price_features() -> pd.DataFrame:
    """Read fct_hourly_market from DuckDB.

    Returns
    -------
    pd.DataFrame
        All columns from fct_hourly_market, sorted by ts_utc ascending.
        Rows where any price-lag column is null are dropped (first 168 h).
        ts_utc is UTC-aware datetime; ts_local is tz-naive.
    """
    with _connect() as con:
        df = con.execute(
            f"SELECT * FROM {_MARTS_SCHEMA}.fct_hourly_market ORDER BY ts_utc"
        ).df()

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.dropna(subset=_PRICE_LAG_COLS).reset_index(drop=True)
    return df


def load_pv_features() -> pd.DataFrame:
    """Read fct_pv_features from DuckDB.

    Returns
    -------
    pd.DataFrame
        All columns from fct_pv_features, sorted by ts_utc ascending.
        Rows where any PV-lag column is null are dropped (first 168 h).
        ts_utc is UTC-aware datetime; ts_local is tz-naive.
    """
    with _connect() as con:
        df = con.execute(
            f"SELECT * FROM {_MARTS_SCHEMA}.fct_pv_features ORDER BY ts_utc"
        ).df()

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df = df.dropna(subset=_PV_LAG_COLS).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Chronological splitter
# ---------------------------------------------------------------------------

def chronological_split(
    df: pd.DataFrame,
    train_months: int = 48,
    val_months: int = 6,
    test_months: int = 6,
    ts_col: str = "ts_utc",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split *df* into (train, val, test) by calendar month. Never shuffles.

    Uses period arithmetic so boundaries are always at clean month edges
    regardless of DST gaps or missing hours in the source data.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *ts_col* as a UTC-aware datetime column, sorted ascending.
    train_months, val_months, test_months : int
        Number of calendar months in each partition.
    ts_col : str
        Name of the UTC timestamp column.

    Returns
    -------
    (train, val, test) : tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Three non-overlapping DataFrames with reset indices.
    """
    df = df.sort_values(ts_col).reset_index(drop=True)

    # tz-aware → period conversion drops tz info intentionally; suppress the pandas warning
    month_periods = df[ts_col].dt.tz_localize(None).dt.to_period("M")
    start_period  = month_periods.iloc[0]

    train_end = start_period + train_months - 1          # inclusive last month of train
    val_end   = train_end   + val_months                 # inclusive last month of val

    test_end   = val_end + test_months                   # inclusive last month of test

    train_mask = month_periods <= train_end
    val_mask   = (month_periods > train_end) & (month_periods <= val_end)
    test_mask  = (month_periods > val_end)   & (month_periods <= test_end)

    return (
        df[train_mask].reset_index(drop=True),
        df[val_mask].reset_index(drop=True),
        df[test_mask].reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== features.py - sanity check ===")

    print("Loading price features from DuckDB...")
    price_df = load_price_features()
    print(f"  rows: {len(price_df)}")
    print(f"  cols: {list(price_df.columns)}")
    print(f"  ts_utc range: {price_df['ts_utc'].min()} to {price_df['ts_utc'].max()}")
    assert price_df[_PRICE_LAG_COLS].isna().sum().sum() == 0, "Null price lags found!"
    print("  OK: no null lags")

    print("Loading PV features from DuckDB...")
    pv_df = load_pv_features()
    print(f"  rows: {len(pv_df)}")
    print(f"  cols: {list(pv_df.columns)}")
    print(f"  ts_utc range: {pv_df['ts_utc'].min()} to {pv_df['ts_utc'].max()}")
    assert pv_df[_PV_LAG_COLS].isna().sum().sum() == 0, "Null PV lags found!"
    print("  OK: no null lags")

    print("Chronological split on price features (48/6/6 months)...")
    train, val, test = chronological_split(price_df)
    print(f"  train: {len(train):>6} rows  "
          f"{train['ts_utc'].min().date()} to {train['ts_utc'].max().date()}")
    print(f"  val:   {len(val):>6} rows  "
          f"{val['ts_utc'].min().date()} to {val['ts_utc'].max().date()}")
    print(f"  test:  {len(test):>6} rows  "
          f"{test['ts_utc'].min().date()} to {test['ts_utc'].max().date()}")

    # Boundary assertions
    assert str(train["ts_utc"].max())[:7] == "2018-12", \
        f"Train should end 2018-12, got {train['ts_utc'].max()}"
    assert str(val["ts_utc"].min())[:7] == "2019-01", \
        f"Val should start 2019-01, got {val['ts_utc'].min()}"
    assert str(val["ts_utc"].max())[:7] == "2019-06", \
        f"Val should end 2019-06, got {val['ts_utc'].max()}"
    assert str(test["ts_utc"].min())[:7] == "2019-07", \
        f"Test should start 2019-07, got {test['ts_utc'].min()}"
    assert str(test["ts_utc"].max())[:7] == "2019-12", \
        f"Test should end 2019-12, got {test['ts_utc'].max()}"

    # No overlap (indices were reset, so compare ts_utc sets)
    train_ts = set(train["ts_utc"])
    val_ts   = set(val["ts_utc"])
    test_ts  = set(test["ts_utc"])
    assert len(train_ts & val_ts) == 0,  "Overlap between train and val!"
    assert len(val_ts   & test_ts) == 0, "Overlap between val and test!"
    assert len(train_ts & test_ts) == 0, "Overlap between train and test!"
    print("  OK: no overlap between splits")

    total    = len(train) + len(val) + len(test)
    overflow = len(price_df) - total
    assert overflow >= 0, f"Split produced more rows ({total}) than source ({len(price_df)})!"
    if overflow:
        print(f"  {overflow} row(s) beyond test_end excluded (boundary date outside all splits - expected)")
    print(f"  OK: {total} rows across splits")

    print("=== PASSED ===")
