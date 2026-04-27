-- Open-Meteo archive hourly weather data for the asset location.
-- Columns: GHI (shortwave radiation), temperature, cloud cover, wind speed.
SELECT
    asset_id,
    CAST(ts_utc AS TIMESTAMPTZ)     AS ts_utc,
    CAST(ts_local AS TIMESTAMP)     AS ts_local,
    CAST(ghi_w_m2 AS DOUBLE)       AS ghi_w_m2,
    CAST(temp_c AS DOUBLE)         AS temp_c,
    CAST(cloud_pct AS INTEGER)     AS cloud_pct,
    CAST(wind_ms AS DOUBLE)        AS wind_ms
FROM {{ source('raw', 'openmeteo_hourly') }}
