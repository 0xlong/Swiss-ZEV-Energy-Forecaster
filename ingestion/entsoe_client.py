"""
entsoe_client.py
----------------
Fetches hourly day-ahead electricity prices for Switzerland from the
ENTSO-E Transparency Platform (bidding zone 10YCH-SWISSGRIDZ).

Prices are published in EUR/MWh. A CHF conversion column is NOT added
here; that conversion belongs in the dbt intermediate layer where
fx-rate data is available.

Requires: entsoe-py, python-dotenv
API key : set ENTSOE_API_KEY or securityToken_entso in .env (either name works)
Period  : config.START_UTC (2015-01-01) → config.END_UTC (2021-01-01 exclusive).

Usage:
    python ingestion/entsoe_client.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

# Make project root importable so `import config` works regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_prices(api_key: str, retries: int = 3) -> pd.Series:
    """
    Query ENTSO-E day-ahead prices for Switzerland (2018-2020).

    Returns a pd.Series with tz-aware UTC DatetimeIndex and float values (EUR/MWh).
    entsoe-py handles XML parsing and pagination internally.
    """
    client = EntsoePandasClient(api_key=api_key)

    for attempt in range(1, retries + 1):
        log.info("Querying ENTSO-E day-ahead prices (attempt %d/%d) …", attempt, retries)
        try:
            series = client.query_day_ahead_prices(
                config.ENTSOE_BIDDING_ZONE,
                start=config.START_UTC,
                end=config.END_UTC,
            )
            log.info(
                "Received %d price points — %s to %s",
                len(series),
                series.index.min(),
                series.index.max(),
            )
            return series
        except NoMatchingDataError:
            log.error("No data returned for %s %s–%s",
                      config.ENTSOE_BIDDING_ZONE, config.START_UTC, config.END_UTC)
            raise
        except Exception as exc:
            log.warning("Request failed: %s", exc)
            if attempt < retries:
                wait = 10 * attempt
                log.info("Retrying in %d s …", wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
def parse_prices(series: pd.Series) -> pd.DataFrame:
    """
    Normalise the raw price Series into a clean DataFrame.

    Columns:
        asset_id        – propagated identifier (links to PV/load data)
        ts_utc          – tz-aware UTC timestamp (canonical for joins)
        ts_local        – naive Europe/Zurich wall time (for display/dbt)
        price_eur_mwh   – day-ahead spot price [EUR/MWh]
        price_eur_kwh   – same value divided by 1000 (convenient for CHF savings calc)
    """
    df = series.rename("price_eur_mwh").to_frame()
    df.index.name = "ts_utc"
    df = df.reset_index()

    # Ensure the UTC column is tz-aware
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    # Derive naive local time
    df["ts_local"] = df["ts_utc"].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)

    # Per-kWh column — useful when computing CHF savings against retail tariff
    df["price_eur_kwh"] = df["price_eur_mwh"] / 1000.0

    df.insert(0, "asset_id", config.ASSET_ID)
    df = df[["asset_id", "ts_utc", "ts_local", "price_eur_mwh", "price_eur_kwh"]]

    log.info(
        "Parsed %d hourly rows — %s to %s",
        len(df),
        df["ts_local"].min(),
        df["ts_local"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Fail-fast sanity checks. A corrupt price series must not silently flow into dbt."""
    # Compute expected hourly rows over [START_UTC, END_UTC).
    expected_hours = int((config.END_UTC - config.START_UTC).total_seconds() // 3600)
    # Allow ±2 % for rare missing ENTSO-E data (publication gaps, DST transitions)
    assert len(df) >= expected_hours * 0.98, (
        f"Expected ~{expected_hours} rows ({config.START_UTC.date()}–"
        f"{(config.END_UTC - pd.Timedelta('1h')).date()}), got {len(df)}. "
        "Check API response or date window."
    )

    # Prices can be negative (surplus periods) but extreme values signal a unit error
    assert df["price_eur_mwh"].min() >= -500, "Price below -500 EUR/MWh — check units."
    assert df["price_eur_mwh"].max() <= 3_000, "Price above 3 000 EUR/MWh — check units."

    assert df["ts_utc"].is_monotonic_increasing, "ts_utc is not sorted."

    null_counts = df[["ts_utc", "ts_local", "price_eur_mwh"]].isnull().sum()
    assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts}"

    log.info("Validation passed ✓")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame) -> None:
    """Persist clean CSV for DuckDB ingestion."""
    df.to_csv(config.ENTSOE_CSV, index=False, float_format="%.4f")
    log.info("Saved → %s  (%d rows × %d cols)", config.ENTSOE_CSV, len(df), len(df.columns))

    log.info("--- ENTSO-E price summary ---")
    log.info("  Period (UTC)  : %s to %s", df["ts_utc"].min(), df["ts_utc"].max())
    log.info("  Min price     : %.2f EUR/MWh", df["price_eur_mwh"].min())
    log.info("  Max price     : %.2f EUR/MWh", df["price_eur_mwh"].max())
    log.info("  Mean price    : %.2f EUR/MWh", df["price_eur_mwh"].mean())
    log.info("  Median price  : %.2f EUR/MWh", df["price_eur_mwh"].median())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=== ENTSO-E day-ahead price ingestion starting ===")
    log.info("Zone  : %s (Switzerland)", config.ENTSOE_BIDDING_ZONE)
    log.info("Period: %s – %s (UTC)",
             config.START_UTC.date(), (config.END_UTC - pd.Timedelta("1h")).date())

    api_key = config.get_entsoe_api_key()
    series  = fetch_prices(api_key)
    df      = parse_prices(series)
    validate(df)
    save(df)

    log.info("=== Done. Load data into DuckDB with load_to_duckdb.py ===")


if __name__ == "__main__":
    main()
