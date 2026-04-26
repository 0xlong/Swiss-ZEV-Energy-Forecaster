"""
load_profile.py
---------------
Generates a synthetic building load profile for the Baar logistics warehouse
using BDEW standard load profiles (G1 — business weekdays 8–18h).

The BDEW (Bundesverband der Energie- und Wasserwirtschaft) profiles are
industry-standard consumption curves used across the German/Swiss energy
sector. We scale them to ~500 MWh/yr to match a medium-sized commercial
warehouse.

No external API call needed — the profile is computed locally from
coefficients bundled inside the oemof-demand (demandlib) library.

Asset: Baar logistics warehouse (~500 MWh/yr consumption)
Period: config.START_YEAR → config.END_YEAR (single source of truth)
Profile type: G1 (Gewerbe werktags 8–18 Uhr — commercial weekdays 8am–6pm)

Output columns: asset_id, ts_utc (canonical, tz-aware UTC), ts_local
(naive Europe/Zurich wall time), load_kw.

Usage:
    python ingestion/load_profile.py
"""

import datetime
import logging
import sys
from pathlib import Path

import pandas as pd

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
# Swiss public holidays (Zug canton)
# ---------------------------------------------------------------------------
def _easter_based_holidays(year: int) -> dict:
    """Good Friday, Easter Monday, Ascension, Whit Monday — all move with Easter."""
    try:
        from dateutil.easter import easter
    except ImportError:
        return {}
    e = easter(year)
    return {
        e - datetime.timedelta(days=2):  "Good Friday",
        e + datetime.timedelta(days=1):  "Easter Monday",
        e + datetime.timedelta(days=39): "Ascension Day",
        e + datetime.timedelta(days=50): "Whit Monday",
    }


def get_swiss_holidays(year: int) -> dict:
    """
    Return a dict of {datetime.date: name} for Swiss public holidays.

    Uses workalendar if available (canton-specific). Falls back to a
    hardcoded list of the most important national holidays plus the
    Easter-based movable feasts computed from dateutil.easter — this
    avoids overstating commercial load on ~4 weekdays per year.
    """
    try:
        from workalendar.europe import Switzerland
        cal = Switzerland()
        holidays = {d: name for d, name in cal.holidays(year)}
        log.info("  Loaded %d Swiss holidays for %d via workalendar", len(holidays), year)
        return holidays
    except ImportError:
        log.info("  workalendar not installed — using hardcoded Swiss holidays for %d", year)
        holidays = {
            datetime.date(year, 1, 1):  "New Year's Day",
            datetime.date(year, 1, 2):  "Berchtoldstag",
            datetime.date(year, 8, 1):  "Swiss National Day",
            datetime.date(year, 12, 25): "Christmas Day",
            datetime.date(year, 12, 26): "St. Stephen's Day",
        }
        holidays.update(_easter_based_holidays(year))
        return holidays


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
def generate_load_profile() -> pd.DataFrame:
    """
    Generate an hourly load profile for each year using BDEW G1 coefficients.

    Returns a DataFrame with columns:
        asset_id, ts_utc, ts_local, load_kw

    The profile is generated at 15-minute resolution by demandlib, then
    resampled to hourly means for consistency with PVGIS / ENTSO-E / Open-Meteo.

    Timestamps: demandlib produces naive wall-clock datetimes. We interpret
    them as Europe/Zurich wall time, DST-localize, then convert to UTC so
    ts_utc is canonical for joins. ts_local is kept as naive local time for
    display.
    """
    from demandlib import bdew

    all_years = []

    for year in config.YEARS:
        log.info("Generating BDEW %s profile for %d …", config.BDEW_PROFILE_TYPE.upper(), year)

        holidays = get_swiss_holidays(year)

        # Initialize the BDEW Standard Load Profile generator
        e_slp = bdew.ElecSlp(year=year, holidays=holidays)

        # get_scaled_power_profiles returns values in kW (power)
        # conversion_factor=4 converts from kWh (energy per 15 min) to kW (power)
        power_profile = e_slp.get_scaled_power_profiles(
            {config.BDEW_PROFILE_TYPE: config.ANNUAL_DEMAND_KWH},
            conversion_factor=4,
        )

        df_year = power_profile[[config.BDEW_PROFILE_TYPE]].copy()
        df_year.columns = ["load_kw"]

        # Resample 15-min → hourly (mean power over the hour) while still naive.
        df_year = df_year.resample("1h").mean()

        # DST-aware uplift: naive wall-clock → Europe/Zurich → UTC.
        # ambiguous=False: synthetic BDEW data has exactly 24 h on the October
        # fallback day (no repeated 2 AM), so "infer" fails. False treats the
        # ambiguous hour as standard time (CET), the conventional choice.
        # nonexistent="shift_forward": spring-forward skips one hour; demandlib
        # still emits a value for 2 AM so we shift it to 03:00.
        df_year.index = df_year.index.tz_localize(
            config.TIMEZONE,
            ambiguous=False,
            nonexistent="shift_forward",
        ).tz_convert("UTC")

        annual_kwh = df_year["load_kw"].sum()
        log.info("  %d annual energy: %.0f kWh (target: %d kWh, diff: %.2f%%)",
                 year, annual_kwh, config.ANNUAL_DEMAND_KWH,
                 100 * (annual_kwh - config.ANNUAL_DEMAND_KWH) / config.ANNUAL_DEMAND_KWH)

        all_years.append(df_year)

    df = pd.concat(all_years)
    df = df.reset_index().rename(columns={"index": "ts_utc"})

    # Derive naive local time from UTC — always consistent with ts_utc.
    df["ts_local"] = df["ts_utc"].dt.tz_convert(config.TIMEZONE).dt.tz_localize(None)

    df.insert(0, "asset_id", config.ASSET_ID)
    df = df[["asset_id", "ts_utc", "ts_local", "load_kw"]]

    return df


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> None:
    """Basic sanity checks on the generated load profile."""
    # Exact row count: account for leap years. UTC has no DST gaps.
    expected_hours = sum(
        366 if pd.Timestamp(f"{y}-01-01").is_leap_year else 365 for y in config.YEARS
    ) * 24
    assert len(df) == expected_hours, (
        f"Expected {expected_hours} rows ({len(config.YEARS)} yr), got {len(df)}."
    )
    assert df["load_kw"].min() >= 0, "Negative load — something is wrong."

    # Per-year 5% tolerance — a single bad year must not hide behind good ones.
    years_by_ts = df["ts_local"].dt.year
    for year in config.YEARS:
        yearly_kwh = df.loc[years_by_ts == year, "load_kw"].sum()
        pct_diff = abs(yearly_kwh - config.ANNUAL_DEMAND_KWH) / config.ANNUAL_DEMAND_KWH * 100
        assert pct_diff < 5, (
            f"{year}: {yearly_kwh:.0f} kWh is {pct_diff:.1f}% off "
            f"target {config.ANNUAL_DEMAND_KWH} kWh — check scaling."
        )

    assert df["ts_utc"].is_monotonic_increasing, "ts_utc is not sorted."
    assert df[["ts_utc", "ts_local", "load_kw"]].isnull().sum().sum() == 0, "Unexpected nulls."

    log.info("Validation passed — %d rows, %d-year total %.0f kWh (target %d kWh/yr)",
             len(df), len(config.YEARS), df["load_kw"].sum(), config.ANNUAL_DEMAND_KWH)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save(df: pd.DataFrame) -> None:
    """Persist the generated load profile as CSV."""
    # float_format persists rounded kW values to disk (3 dp is plenty at this scale).
    df.to_csv(config.LOAD_CSV, index=False, float_format="%.3f")
    log.info("Saved → %s  (%d rows × %d cols)", config.LOAD_CSV, len(df), len(df.columns))

    log.info("--- Load profile summary ---")
    log.info("  Period (UTC): %s to %s", df["ts_utc"].min(), df["ts_utc"].max())
    log.info("  Peak load   : %.1f kW", df["load_kw"].max())
    log.info("  Min load    : %.1f kW", df["load_kw"].min())
    log.info("  Mean load   : %.1f kW", df["load_kw"].mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    log.info("=== BDEW load profile generation starting ===")
    log.info("Asset  : %s", config.ASSET_ID)
    log.info("Profile: BDEW %s (commercial weekdays 8–18h)", config.BDEW_PROFILE_TYPE.upper())
    log.info("Target : %d kWh/yr (%d MWh/yr)", config.ANNUAL_DEMAND_KWH, config.ANNUAL_DEMAND_KWH // 1000)
    log.info("Years  : %s", config.YEARS)

    df = generate_load_profile()
    validate(df)
    save(df)

    log.info("=== Done. Load data into DuckDB with load_to_duckdb.py ===")


if __name__ == "__main__":
    main()
