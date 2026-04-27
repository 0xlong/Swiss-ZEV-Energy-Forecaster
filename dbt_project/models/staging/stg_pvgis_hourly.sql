-- PVGIS labels its timestamps as 'ts_local' but they are UTC by the API spec.
-- The API returns representative timestamps at HH:10 (10 min into each hour);
-- we floor to HH:00 so this model aligns with stg_load_hourly and
-- stg_entsoe_prices which are both on the hour.
SELECT
    asset_id,
    DATE_TRUNC('hour', CAST(ts_local AS TIMESTAMP)) AT TIME ZONE 'UTC'                                 AS ts_utc,
    (DATE_TRUNC('hour', CAST(ts_local AS TIMESTAMP)) AT TIME ZONE 'UTC') AT TIME ZONE 'Europe/Zurich'  AS ts_local,
    CAST(pv_power_kw AS DOUBLE)                                                                        AS pv_power_kw,
    CAST(irradiance_w_m2 AS DOUBLE)                                                                    AS irradiance_w_m2,
    CAST(sun_height_deg AS DOUBLE)                                                                     AS sun_height_deg,
    CAST(temp_2m_c AS DOUBLE)                                                                          AS temp_2m_c,
    CAST(wind_speed_m_s AS DOUBLE)                                                                     AS wind_speed_m_s
FROM {{ source('raw', 'pvgis_hourly') }}