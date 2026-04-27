"""
openmeteo_client.py
-------------------
Fetches hourly historical weather data from the Open-Meteo Archive API
for the Baar warehouse location (lat 47.195, lon 8.527).

Variables fetched:
    - shortwave_radiation  (GHI, W/m²)  — primary PV driver
    - temperature_2m       (°C)         — panel efficiency correction
    - cloud_cover          (%)          — cloud-vs-clear sky signal
    - wind_speed_10m       (m/s)        — panel cooling

No API key required. Rate limit: ~10 000 requests/day for non-commercial use.
Window: config.START_UTC → config.END_UTC (2015-01-01 to 2020-01-01 exclusive).

Follows the same extract → parse → validate → save pattern as pvgis_client.py.

Usage:
    python ingestion/openmeteo_client.py
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

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
def fetch_openmeteo(retries: int = 3) -> dict:
    """
    Call the Open-Meteo Historical Weather API and return parsed JSON.

    The archive endpoint accepts start_date/end_date as YYYY-MM-DD strings
    and returns hourly arrays for the requested variables.
    """
    # Open-Meteo uses inclusive date range — end_date is the last day included.
    # config.END_UTC is exclusive (2020-01-01), so the last day we want is 2019-12-31.
    end_date = (config.END_UTC - pd.Timedelta("1D")).strftime("%Y-%m-%d")
    start_date = config.START_UTC.strftime("%Y-%m-%d")

    params = {
        "latitude": config.LAT,
        "longitude": config.LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation,temperature_2m,cloud_cover,wind_speed_10m",
        "timezone": "UTC",
    }

    for attempt in range(1, retries + 1):
        log.info("Calling Open-Meteo Archive API (attempt %d/%d) …", attempt, retries)
        log.info("  Window: %s to %s", start_date, end_date)
        try:
            resp = requests.get(config.OPENMETEO_URL, params=params, timeout=120)
            resp.raise_for_status()
            data = resp.json()

            # Open-Meteo returns {"error": true, "reason": "..."} on bad params
            if data.get("error"):
                raise ValueError(f"Open-Meteo error: {data.get('reason', 'unknown')}")

            log.info(
                "HTTP %s — response received (%.1f KB)",
                resp.status_code,
                len(resp.content) / 1024,
            )
            return data
        except requests.exceptions.HTTPError as exc:
            log.error("HTTP error: %s", exc)
            raise
        except requests.exceptions.RequestException as exc:
            log.warning("Request failed: %s", exc)
            if attempt < retries:
                wait = 5 * attempt
                log.info("Retrying in %d s …", wait)
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------
def parse_hourly(raw: dict) -> pd.DataFrame:
    """
    Extract hourly arrays from the Open-Meteo JSON response.

    The response contains:
        raw["hourly"]["time"]                — ISO timestamps (UTC)
        raw["hourly"]["shortwave_radiation"] — GHI W/m²
        raw["hourly"]["temperature_2m"]      — °C
        raw["hourly"]["cloud_cover"]         — %
        raw["hourly"]["wind_speed_10m"]      — m/s

    Returns a DataFrame with the canonical project schema.
    """
    hourly = raw["hourly"]

    df = pd.DataFrame(
        {
            "ts_utc": pd.to_datetime(hourly["time"], utc=True),
            "ghi_w_m2": hourly["shortwave_radiation"],
            "temp_c": hourly["temperature_2m"],
            "cloud_pct": hourly["cloud_cover"],
            "wind_ms": hourly["wind_speed_10m"],
        }
    )

    # Derive naive local time (Europe/Zurich) for display / dbt
    df["ts_local"] = df["ts_utc"].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)

    # Add asset metadata
    df.insert(0, "asset_id", config.ASSET_ID)

    # Reorder columns to match project convention
    df = df[["asset_id", "ts_utc", "ts_local", "ghi_w_m2", "temp_c", "cloud_pct", "wind_ms"]]

    log.info(
        "Parsed %d hourly rows — %s to %s",
        len(df),
        df["ts_utc"].min(),
        df["ts_utc"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """
    Fail-fast sanity checks on the weather DataFrame.

    Expected hourly count is derived from config.START_UTC → config.END_UTC.
    For 2015-01-01 to 2020-01-01 (exclusive) that is 43,824 hours.
    """
    # Compute expected hourly rows dynamically from the config window
    expected_hours = int((config.END_UTC - config.START_UTC).total_seconds() // 3600)
    actual = len(df)
    assert actual >= expected_hours * 0.98, (
        f"Expected ~{expected_hours} rows ({config.START_UTC.date()}–"
        f"{(config.END_UTC - pd.Timedelta('1h')).date()}), got {actual}. "
        "Check date window or API response."
    )
    # Warn if exact match fails but still within tolerance
    if actual != expected_hours:
        log.warning(
            "Row count %d differs from exact expectation %d (delta=%+d). "
            "This may be due to DST transitions or API quirks.",
            actual, expected_hours, actual - expected_hours,
        )

    # Value range checks
    assert df["ghi_w_m2"].min() >= 0, "Negative GHI — check units."
    assert df["ghi_w_m2"].max() <= 1500, "GHI above 1500 W/m² — check units."
    assert df["cloud_pct"].min() >= 0, "Negative cloud cover."
    assert df["cloud_pct"].max() <= 100, "Cloud cover above 100%."
    assert df["temp_c"].min() >= -50, "Temperature below -50 °C — suspicious."
    assert df["temp_c"].max() <= 60, "Temperature above 60 °C — suspicious."

    # Monotonic timestamps
    assert df["ts_utc"].is_monotonic_increasing, "ts_utc is not sorted."

    # No nulls in key columns
    null_counts = df[["ts_utc", "ts_local", "ghi_w_m2", "temp_c", "cloud_pct", "wind_ms"]].isnull().sum()
    assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts}"

    log.info("Validation passed ✓  (%d rows)", len(df))


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame) -> None:
    """Persist clean CSV for DuckDB ingestion."""
    df.to_csv(config.OPENMETEO_CSV, index=False)
    log.info(
        "Saved → %s  (%d rows × %d cols)",
        config.OPENMETEO_CSV, len(df), len(df.columns),
    )

    log.info("--- Open-Meteo weather summary ---")
    log.info("  Period (UTC)  : %s to %s", df["ts_utc"].min(), df["ts_utc"].max())
    log.info("  GHI range     : %.1f – %.1f W/m²", df["ghi_w_m2"].min(), df["ghi_w_m2"].max())
    log.info("  Temp range    : %.1f – %.1f °C", df["temp_c"].min(), df["temp_c"].max())
    log.info("  Cloud range   : %.0f – %.0f %%", df["cloud_pct"].min(), df["cloud_pct"].max())
    log.info("  Wind range    : %.1f – %.1f m/s", df["wind_ms"].min(), df["wind_ms"].max())
    log.info("  Mean GHI      : %.1f W/m²", df["ghi_w_m2"].mean())
    log.info("  Mean temp     : %.1f °C", df["temp_c"].mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=== Open-Meteo hourly weather ingestion starting ===")
    log.info("Asset : %s @ lat=%.3f lon=%.3f", config.ASSET_ID, config.LAT, config.LON)
    log.info("Period: %s – %s (UTC)", config.START_UTC.date(), (config.END_UTC - pd.Timedelta("1h")).date())

    raw = fetch_openmeteo()
    df  = parse_hourly(raw)
    validate(df)
    save(df)

    log.info("=== Done. Load data into DuckDB with load_to_duckdb.py ===")


if __name__ == "__main__":
    main()
