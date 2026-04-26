"""
pvgis_client.py
---------------
Fetches hourly simulated PV power output from the PVGIS API (EU JRC).
No API key required. Saves raw JSON and a clean CSV/Parquet into data/.

Asset: 100 kWp rooftop installation on the Baar logistics warehouse.
       Lat/Lon and time range come from `config` (single source of truth).

Note:
    PVGIS SARAH2 radiation database only covers 2005–2020.
    API calls have a rate limit of 30 calls/second per IP address.

Usage:
    python ingestion/pvgis_client.py
"""

import json
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
# PVGIS request params (asset-shaped values come from config)
# ---------------------------------------------------------------------------
PARAMS = {
    # --- Location ---
    "lat": config.LAT,
    "lon": config.LON,

    # --- Time range ---
    "startyear": config.START_YEAR,
    "endyear": config.END_YEAR,

    # --- PV system specification ---
    "pvcalculation": 1,     # 1 = return PV power output (not raw radiation only)
    "peakpower": config.PV_PEAK_KWP,
    "loss": config.PV_SYSTEM_LOSS_PCT,
    "mountingplace": "building",  # rooftop-mounted (affects free-air cooling assumption)

    # --- Panel orientation ---
    # optimalangles=1 lets PVGIS pick the best tilt+azimuth automatically.
    # Set to 0 and provide angle/aspect manually if you know the roof geometry.
    "optimalangles": 1,

    # --- Output ---
    "outputformat": "json",
    "browser": 0,           # 0 = raw JSON (no HTML wrapper)
}


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def fetch_pvgis(params: dict, retries: int = 3) -> dict:
    """Call the PVGIS seriescalc API and return the parsed JSON response."""
    for attempt in range(1, retries + 1):
        log.info("Calling PVGIS API (attempt %d/%d) …", attempt, retries)
        try:
            resp = requests.get(config.PVGIS_URL, params=params, timeout=120)
            resp.raise_for_status()
            log.info("HTTP %s — response received (%.1f KB)", resp.status_code, len(resp.content) / 1024)
            return resp.json()
        except requests.exceptions.HTTPError as exc:
            log.error("HTTP error: %s", exc)
            # PVGIS returns 400 with a helpful message — surface it
            try:
                log.error("PVGIS says: %s", resp.json().get("message", resp.text[:300]))
            except Exception:
                pass
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
    Extract the hourly series from the PVGIS JSON response.

    PVGIS returns a list of dicts under raw["outputs"]["hourly"].
    Each dict has:
        time    – string "YYYYDDHHMM" (local time, no timezone offset)
        P       – PV power output [W]  ← the main value we need
        G(i)    – In-plane irradiance [W/m²]
        H_sun   – Sun height [°]
        T2m     – 2-m air temperature [°C]
        WS10m   – Wind speed at 10 m [m/s]
        Int     – Solar radiation reconstructed flag
    """
    hourly_records = raw["outputs"]["hourly"]
    df = pd.DataFrame(hourly_records)

    # Parse timestamp — format is "YYYYDDHHMM" where DD is day-of-year... 
    # actually PVGIS uses "YYYYMMDDHHM" — let's confirm and parse robustly.
    # The actual format is "202201010050" → YYYYMMDDHHMM
    df["ts_local"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M")

    # Rename columns to snake_case for DuckDB / dbt compatibility
    df = df.rename(columns={
        "P":     "pv_power_w",        # Watts — divide by 1000 for kW
        "G(i)":  "irradiance_w_m2",   # In-plane irradiance
        "H_sun": "sun_height_deg",
        "T2m":   "temp_2m_c",
        "WS10m": "wind_speed_m_s",
        "Int":   "reconstruction_flag",
    })

    # Convert power to kW (more natural unit for energy analytics)
    df["pv_power_kw"] = df["pv_power_w"] / 1000.0

    # Drop the raw time string (we have ts_local now)
    df = df.drop(columns=["time"], errors="ignore")

    # Reorder columns
    col_order = [
        "ts_local", "pv_power_kw", "pv_power_w",
        "irradiance_w_m2", "sun_height_deg",
        "temp_2m_c", "wind_speed_m_s", "reconstruction_flag",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    # Add metadata columns so DuckDB / dbt knows which asset this belongs to
    df.insert(0, "asset_id", config.ASSET_ID)
    df.insert(1, "lat", config.LAT)
    df.insert(2, "lon", config.LON)
    df.insert(3, "peak_power_kwp", config.PV_PEAK_KWP)

    log.info("Parsed %d hourly rows — %s to %s",
             len(df), df["ts_local"].min(), df["ts_local"].max())
    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Basic sanity checks — fail loudly rather than silently corrupt data."""
    num_years = len(config.YEARS)
    expected_hours = num_years * 365 * 24  # approximate (ignoring leap years)
    assert len(df) >= expected_hours * 0.98, (
        f"Expected ~{expected_hours} rows ({num_years} yr), got {len(df)}. Check startyear/endyear."
    )
    assert df["pv_power_kw"].min() >= 0, "Negative PV power — something is wrong."
    assert df["pv_power_kw"].max() <= config.PV_PEAK_KWP * 1.2, (
        "PV power exceeds 120 % of nameplate — check units."
    )
    assert df["ts_local"].is_monotonic_increasing, "Timestamps are not sorted."
    null_counts = df[["ts_local", "pv_power_kw", "irradiance_w_m2"]].isnull().sum()
    assert null_counts.sum() == 0, f"Unexpected nulls:\n{null_counts}"
    log.info("Validation passed ✓")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save(raw: dict, df: pd.DataFrame) -> None:
    """Persist raw JSON (for audit) and clean CSV (for DuckDB ingestion)."""
    # Raw JSON — full API response including metadata and system info
    # with open(config.PVGIS_RAW_JSON, "w", encoding="utf-8") as f:
    #     json.dump(raw, f, indent=2)
    # log.info("Raw JSON saved → %s", config.PVGIS_RAW_JSON)

    # Clean CSV
    df.to_csv(config.PVGIS_CSV, index=False)
    log.info("Clean CSV saved → %s  (%d rows × %d cols)", config.PVGIS_CSV, len(df), len(df.columns))

    # Print a quick summary of what the API returned
    meta_inputs  = raw.get("inputs", {})
    meta_outputs = raw.get("outputs", {}).get("totals", {}).get("fixed", {})
    log.info("--- PVGIS system summary ---")
    log.info("  Optimal slope angle : %.1f °", meta_inputs.get("mounting_system", {}).get("fixed", {}).get("slope", {}).get("value", float("nan")))
    log.info("  Optimal azimuth     : %.1f °", meta_inputs.get("mounting_system", {}).get("fixed", {}).get("azimuth", {}).get("value", float("nan")))
    log.info("  Annual PV production: %.0f kWh/yr", meta_outputs.get("E_y", float("nan")))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=== PVGIS hourly ingestion starting ===")
    log.info("Asset : %s kWp @ lat=%.3f lon=%.3f", PARAMS["peakpower"], PARAMS["lat"], PARAMS["lon"])
    log.info("Period: %s – %s", PARAMS["startyear"], PARAMS["endyear"])

    raw = fetch_pvgis(PARAMS)
    df  = parse_hourly(raw)
    validate(df)
    save(raw, df)

    log.info("=== Done. Load data into DuckDB with load_to_duckdb.py ===")


if __name__ == "__main__":
    main()
