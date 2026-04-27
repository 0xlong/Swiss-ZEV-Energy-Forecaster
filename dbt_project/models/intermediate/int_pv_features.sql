-- PV production feature table for ML (Phase D).
-- Target: pv_power_kw from PVGIS hourly simulation.
-- Features: weather (GHI, temp, cloud, wind), calendar (hour, day of year,
-- day of week), and PV production lags (1h, 24h, 168h).
WITH pv_base AS (
    SELECT
        asset_id,
        ts_utc,
        ts_local,
        pv_power_kw,
        LAG(pv_power_kw, 1)   OVER (PARTITION BY asset_id ORDER BY ts_utc) AS pv_lag_1h,
        LAG(pv_power_kw, 24)  OVER (PARTITION BY asset_id ORDER BY ts_utc) AS pv_lag_24h,
        LAG(pv_power_kw, 168) OVER (PARTITION BY asset_id ORDER BY ts_utc) AS pv_lag_168h,
        EXTRACT(hour FROM ts_local)::INTEGER      AS hour_of_day,
        EXTRACT(doy  FROM ts_local)::INTEGER      AS day_of_year,
        EXTRACT(dow  FROM ts_local)::INTEGER      AS day_of_week
    FROM {{ ref('stg_pvgis_hourly') }}
)

SELECT
    p.asset_id,
    p.ts_utc,
    p.ts_local,
    -- Target
    p.pv_power_kw,
    -- PV lags
    p.pv_lag_1h,
    p.pv_lag_24h,
    p.pv_lag_168h,
    -- Calendar
    p.hour_of_day,
    p.day_of_year,
    p.day_of_week,
    -- Weather features
    w.ghi_w_m2,
    w.temp_c,
    w.cloud_pct,
    w.wind_ms
FROM pv_base p
LEFT JOIN {{ ref('stg_openmeteo_hourly') }} w
    ON p.asset_id = w.asset_id AND p.ts_utc = w.ts_utc
