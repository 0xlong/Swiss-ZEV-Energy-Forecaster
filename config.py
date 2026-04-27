"""
config.py
---------
Single source of truth for ingestion / dbt / forecasting parameters.

All ingestion scripts import from this module instead of defining their own
asset metadata, time windows, output paths, or API endpoints. Secrets stay
in `.env` and are pulled via `os.getenv` (load_dotenv runs at import).

Asset: 100 kWp rooftop PV on the Baar logistics warehouse, Zug, Switzerland.
"""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Populate env vars from .env so any importer (ingestion, notebooks, dbt
# python models) gets the ENTSO-E key without repeating load_dotenv calls.
load_dotenv(ROOT / ".env")

# ---------------------------------------------------------------------------
# Asset metadata
# ---------------------------------------------------------------------------
ASSET_ID = "baar_warehouse_100kwp"
LAT, LON = 47.195, 8.527
TIMEZONE = "Europe/Zurich"

PV_PEAK_KWP = 100
PV_SYSTEM_LOSS_PCT = 14
ANNUAL_DEMAND_KWH = 500_000
BDEW_PROFILE_TYPE = "g1"   # commercial weekdays 8–18h

# ---------------------------------------------------------------------------
# Time range (single source for the analysis window)
# ---------------------------------------------------------------------------
START_YEAR, END_YEAR = 2015, 2019   # it gives data from start of 2015 till end of 2019
YEARS = list(range(START_YEAR, END_YEAR + 1))
START_UTC = pd.Timestamp(f"{START_YEAR}-01-01", tz="UTC")
END_UTC = pd.Timestamp(f"{END_YEAR + 1}-01-01", tz="UTC")  # exclusive

# ---------------------------------------------------------------------------
# API config
# ---------------------------------------------------------------------------
PVGIS_URL = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
ENTSOE_BIDDING_ZONE = "10YCH-SWISSGRIDZ"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
LOAD_CSV = DATA_DIR / "load_hourly.csv"
PVGIS_CSV = DATA_DIR / "pvgis_hourly.csv"
#PVGIS_RAW_JSON = DATA_DIR / "pvgis_hourly_raw.json"
ENTSOE_CSV = DATA_DIR / "entsoe_day_ahead_prices.csv"
ENTSOE_LOAD_FORECAST_CSV = DATA_DIR / "entsoe_load_forecast.csv"
ENTSOE_GENMIX_CSV = DATA_DIR / "entsoe_genmix.csv"
OPENMETEO_CSV = DATA_DIR / "openmeteo_hourly.csv"

# DuckDB warehouse — single-file database, shared by load_to_duckdb.py,
# dbt's profiles.yml, and downstream ML / dashboard scripts.
DUCKDB_PATH = ROOT / "data" / "swiss_zev.duckdb"


# ---------------------------------------------------------------------------
# Secret accessors
# ---------------------------------------------------------------------------
def get_entsoe_api_key() -> str:
    """Return the ENTSO-E API key. Accepts the legacy `securityToken_entso` name."""
    key = os.getenv("ENTSOE_API_KEY") or os.getenv("securityToken_entso")
    if not key:
        raise EnvironmentError(
            "ENTSO-E API key not found. Set ENTSOE_API_KEY or securityToken_entso in .env"
        )
    return key
