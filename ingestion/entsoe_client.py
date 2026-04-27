"""
entsoe_client.py
----------------
Fetches hourly data from the ENTSO-E Transparency Platform for Switzerland
(bidding zone 10YCH-SWISSGRIDZ):

1. Day-ahead electricity prices (EUR/MWh)
2. Day-ahead total load forecast (MW)
3. Actual generation per type (MW) — stored in long format

Requires: entsoe-py, python-dotenv
API key : set ENTSOE_API_KEY or securityToken_entso in .env (either name works)
Period  : config.START_UTC (2015-01-01) → config.END_UTC (2020-01-01 exclusive).

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
# Generic retry helper
# ---------------------------------------------------------------------------
def _retry(fn, description: str, retries: int = 3):
    """Call *fn* with retries and exponential back-off."""
    for attempt in range(1, retries + 1):
        log.info("Querying ENTSO-E %s (attempt %d/%d) …", description, attempt, retries)
        try:
            result = fn()
            return result
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


# ═══════════════════════════════════════════════════════════════════════════
# 1. Day-ahead prices (existing functionality, refactored)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_prices(client: EntsoePandasClient, retries: int = 3) -> pd.Series:
    """
    Query ENTSO-E day-ahead prices for Switzerland.
    Returns a pd.Series with tz-aware UTC DatetimeIndex and float values (EUR/MWh).
    """
    def _call():
        series = client.query_day_ahead_prices(
            config.ENTSOE_BIDDING_ZONE,
            start=config.START_UTC,
            end=config.END_UTC,
        )
        log.info("Received %d price points — %s to %s",
                 len(series), series.index.min(), series.index.max())
        return series

    return _retry(_call, "day-ahead prices", retries)


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

    log.info("Parsed %d hourly price rows — %s to %s",
             len(df), df["ts_local"].min(), df["ts_local"].max())
    return df


def validate_prices(df: pd.DataFrame) -> None:
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

    log.info("Price validation passed ✓  (%d rows)", len(df))


def save_prices(df: pd.DataFrame) -> None:
    """Persist clean price CSV for DuckDB ingestion."""
    df.to_csv(config.ENTSOE_CSV, index=False, float_format="%.4f")
    log.info("Saved → %s  (%d rows × %d cols)", config.ENTSOE_CSV, len(df), len(df.columns))

    log.info("--- ENTSO-E price summary ---")
    log.info("  Period (UTC)  : %s to %s", df["ts_utc"].min(), df["ts_utc"].max())
    log.info("  Min price     : %.2f EUR/MWh", df["price_eur_mwh"].min())
    log.info("  Max price     : %.2f EUR/MWh", df["price_eur_mwh"].max())
    log.info("  Mean price    : %.2f EUR/MWh", df["price_eur_mwh"].mean())
    log.info("  Median price  : %.2f EUR/MWh", df["price_eur_mwh"].median())


# ═══════════════════════════════════════════════════════════════════════════
# 2. Day-ahead load forecast (new)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_load_forecast(client: EntsoePandasClient, retries: int = 3) -> pd.Series:
    """
    Query ENTSO-E day-ahead total load forecast for Switzerland.
    Returns hourly MW values published by Swissgrid (typically at 18:00 CET
    for the next day).
    """
    def _call():
        series = client.query_load_forecast(
            config.ENTSOE_BIDDING_ZONE,
            start=config.START_UTC,
            end=config.END_UTC,
        )
        log.info("Received %d load-forecast points — %s to %s",
                 len(series), series.index.min(), series.index.max())
        return series

    return _retry(_call, "load forecast", retries)


def parse_load_forecast(data) -> pd.DataFrame:
    """
    Normalise the raw load-forecast data into a clean DataFrame.

    entsoe-py may return a Series or DataFrame depending on the response.
    We extract the first (or only) column as the forecast value.

    Columns:
        asset_id, ts_utc, ts_local, load_forecast_mw
    """
    # Handle DataFrame (multi-column) or Series
    if isinstance(data, pd.DataFrame):
        # Take the first numeric column as load forecast
        series = data.iloc[:, 0]
    else:
        series = data

    df = series.rename("load_forecast_mw").to_frame()
    df.index.name = "ts_utc"
    df = df.reset_index()

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["ts_local"] = df["ts_utc"].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)
    df.insert(0, "asset_id", config.ASSET_ID)
    df = df[["asset_id", "ts_utc", "ts_local", "load_forecast_mw"]]

    log.info("Parsed %d hourly load-forecast rows — %s to %s",
             len(df), df["ts_local"].min(), df["ts_local"].max())
    return df


def validate_load_forecast(df: pd.DataFrame) -> None:
    """Sanity checks for the load forecast series."""
    expected_hours = int((config.END_UTC - config.START_UTC).total_seconds() // 3600)
    assert len(df) >= expected_hours * 0.98, (
        f"Expected ~{expected_hours} load-forecast rows, got {len(df)}."
    )

    # Swiss demand typically 4–12 GW; extreme values signal unit errors
    assert df["load_forecast_mw"].min() >= 0, "Negative load forecast — check units."
    assert df["load_forecast_mw"].max() <= 20_000, "Load > 20 GW — check units."

    null_counts = df[["ts_utc", "load_forecast_mw"]].isnull().sum()
    assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts}"

    log.info("Load-forecast validation passed ✓  (%d rows)", len(df))


def save_load_forecast(df: pd.DataFrame) -> None:
    """Persist clean load-forecast CSV."""
    df.to_csv(config.ENTSOE_LOAD_FORECAST_CSV, index=False, float_format="%.1f")
    log.info("Saved → %s  (%d rows × %d cols)",
             config.ENTSOE_LOAD_FORECAST_CSV, len(df), len(df.columns))

    log.info("--- ENTSO-E load forecast summary ---")
    log.info("  Period (UTC)  : %s to %s", df["ts_utc"].min(), df["ts_utc"].max())
    log.info("  Min load      : %.0f MW", df["load_forecast_mw"].min())
    log.info("  Max load      : %.0f MW", df["load_forecast_mw"].max())
    log.info("  Mean load     : %.0f MW", df["load_forecast_mw"].mean())


# ═══════════════════════════════════════════════════════════════════════════
# 3. Actual generation per type (new)
# ═══════════════════════════════════════════════════════════════════════════

def fetch_genmix(client: EntsoePandasClient, retries: int = 3) -> pd.DataFrame:
    """
    Query ENTSO-E actual generation per production type for Switzerland.
    Returns a wide DataFrame — one column per generation type (MW).

    psr_type=None fetches all production types in a single call.
    """
    def _call():
        df = client.query_generation(
            config.ENTSOE_BIDDING_ZONE,
            start=config.START_UTC,
            end=config.END_UTC,
            psr_type=None,
        )
        log.info("Received generation mix — %d rows × %d types, %s to %s",
                 len(df), len(df.columns), df.index.min(), df.index.max())
        return df

    return _retry(_call, "generation mix", retries)


def parse_genmix(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the wide generation DataFrame to long format.

    Output columns:
        asset_id, ts_utc, ts_local, gen_type, generation_mw

    Column names from entsoe-py may be tuples (type, consumption/generation)
    or plain strings depending on the version — handle both.
    """
    # entsoe-py sometimes returns MultiIndex columns: (type, 'Actual Aggregated')
    # Flatten to just the generation type name.
    if isinstance(raw_df.columns, pd.MultiIndex):
        # Keep only 'Actual Aggregated' columns (drop consumption if present)
        gen_cols = [col for col in raw_df.columns
                    if "consumption" not in str(col[1]).lower()]
        raw_df = raw_df[gen_cols]
        raw_df.columns = [col[0] for col in raw_df.columns]

    # Melt wide → long
    raw_df.index.name = "ts_utc"
    raw_df = raw_df.reset_index()
    raw_df["ts_utc"] = pd.to_datetime(raw_df["ts_utc"], utc=True)

    long = raw_df.melt(
        id_vars=["ts_utc"],
        var_name="gen_type",
        value_name="generation_mw",
    )

    # Drop rows where generation is NaN (type not reporting in that interval)
    long = long.dropna(subset=["generation_mw"])

    # Clean up type names: "Fossil Gas" → "fossil_gas", "Solar" → "solar", etc.
    long["gen_type"] = (
        long["gen_type"]
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    long["ts_local"] = long["ts_utc"].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)
    long.insert(0, "asset_id", config.ASSET_ID)
    long = long[["asset_id", "ts_utc", "ts_local", "gen_type", "generation_mw"]]
    long = long.sort_values(["ts_utc", "gen_type"]).reset_index(drop=True)

    unique_types = sorted(long["gen_type"].unique())
    log.info("Parsed %d gen-mix rows across %d types: %s",
             len(long), len(unique_types), ", ".join(unique_types))
    return long


def validate_genmix(df: pd.DataFrame) -> None:
    """Sanity checks for the generation-mix long table."""
    expected_hours = int((config.END_UTC - config.START_UTC).total_seconds() // 3600)
    unique_timestamps = df["ts_utc"].nunique()
    # Should have at least 98% of expected hours represented
    assert unique_timestamps >= expected_hours * 0.98, (
        f"Expected ~{expected_hours} unique timestamps, got {unique_timestamps}."
    )

    assert df["generation_mw"].min() >= -500, "Generation below -500 MW — check units."
    assert df["generation_mw"].max() <= 20_000, "Generation > 20 GW — check units."

    null_counts = df[["ts_utc", "gen_type", "generation_mw"]].isnull().sum()
    assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts}"

    log.info("Gen-mix validation passed ✓  (%d rows, %d timestamps)",
             len(df), unique_timestamps)


def save_genmix(df: pd.DataFrame) -> None:
    """Persist clean gen-mix CSV in long format."""
    df.to_csv(config.ENTSOE_GENMIX_CSV, index=False, float_format="%.1f")
    log.info("Saved → %s  (%d rows × %d cols)",
             config.ENTSOE_GENMIX_CSV, len(df), len(df.columns))

    # Summary by type
    log.info("--- ENTSO-E generation mix summary ---")
    log.info("  Period (UTC)  : %s to %s", df["ts_utc"].min(), df["ts_utc"].max())
    summary = df.groupby("gen_type")["generation_mw"].agg(["mean", "max", "count"])
    for gen_type, row in summary.iterrows():
        log.info("  %-25s  mean=%7.0f MW  max=%7.0f MW  (%d rows)",
                 gen_type, row["mean"], row["max"], row["count"])


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("=== ENTSO-E ingestion starting ===")
    log.info("Zone  : %s (Switzerland)", config.ENTSOE_BIDDING_ZONE)
    log.info("Period: %s – %s (UTC)",
             config.START_UTC.date(), (config.END_UTC - pd.Timedelta("1h")).date())

    api_key = config.get_entsoe_api_key()
    client = EntsoePandasClient(api_key=api_key)

    # --- 1. Day-ahead prices (existing) ---
    log.info("")
    log.info("── Day-ahead prices ──")
    series = fetch_prices(client)
    df_prices = parse_prices(series)
    validate_prices(df_prices)
    save_prices(df_prices)

    # --- 2. Load forecast (new) ---
    log.info("")
    log.info("── Day-ahead load forecast ──")
    series_load = fetch_load_forecast(client)
    df_load = parse_load_forecast(series_load)
    validate_load_forecast(df_load)
    save_load_forecast(df_load)

    # --- 3. Generation mix (new) ---
    log.info("")
    log.info("── Actual generation per type ──")
    raw_gen = fetch_genmix(client)
    df_gen = parse_genmix(raw_gen)
    validate_genmix(df_gen)
    save_genmix(df_gen)

    log.info("")
    log.info("=== Done. Load data into DuckDB with load_to_duckdb.py ===")


if __name__ == "__main__":
    main()
