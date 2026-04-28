-- Staging model for the battery dispatch LP output.
--
-- WHAT THIS MODEL DOES
-- --------------------
-- Picks up the raw table written by optimization/battery_dispatch.py and:
--   1. Casts every column to the correct SQL type.
--   2. Derives ts_local (Europe/Zurich wall time) from ts_utc — same pattern
--      used by every other staging model in this project.
--   3. Clips LP outputs to 0 with GREATEST(..., 0.0).
--
-- WHY THE GREATEST(..., 0.0) GUARDS?
-- -----------------------------------
-- The CBC solver can return values like -0.0000000001 for variables that are
-- mathematically zero.  This is normal floating-point noise, but it would
-- make downstream energy balance checks fail (e.g. "grid_export_kw < 0 is
-- physically impossible").  Clamping here in the staging layer means the
-- marts layer never sees physically impossible values — ELT principle: raw
-- stays faithful, staging fixes known artefacts.
--
-- WHY DERIVE ts_local HERE, NOT IN THE RAW TABLE?
-- ------------------------------------------------
-- The Python script writes ts_utc as a naive timestamp (no timezone info).
-- The AT TIME ZONE 'UTC' cast labels it as UTC, and the second AT TIME ZONE
-- 'Europe/Zurich' converts it to Swiss wall time.  This is the same two-step
-- pattern used in stg_pvgis_hourly.sql and stg_load_hourly.sql — one
-- consistent approach across all staging models in the project.
SELECT
    asset_id,
    CAST(battery_capacity_kwh AS DOUBLE)                                        AS battery_capacity_kwh,
    CAST(ts_utc AS TIMESTAMP) AT TIME ZONE 'UTC'                                AS ts_utc,
    (CAST(ts_utc AS TIMESTAMP) AT TIME ZONE 'UTC') AT TIME ZONE 'Europe/Zurich' AS ts_local,
    GREATEST(CAST(charge_kw      AS DOUBLE), 0.0)                               AS charge_kw,
    GREATEST(CAST(discharge_kw   AS DOUBLE), 0.0)                               AS discharge_kw,
    GREATEST(CAST(soc_kwh        AS DOUBLE), 0.0)                               AS soc_kwh,
    GREATEST(CAST(grid_import_kw AS DOUBLE), 0.0)                               AS grid_import_kw,
    GREATEST(CAST(grid_export_kw AS DOUBLE), 0.0)                               AS grid_export_kw
FROM {{ source('raw', 'battery_simulations') }}
