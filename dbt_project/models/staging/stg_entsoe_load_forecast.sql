-- ENTSO-E day-ahead total load forecast for Switzerland (Swissgrid).
SELECT
    asset_id,
    CAST(ts_utc AS TIMESTAMPTZ)         AS ts_utc,
    CAST(ts_local AS TIMESTAMP)         AS ts_local,
    CAST(load_forecast_mw AS DOUBLE)   AS load_forecast_mw
FROM {{ source('raw', 'entsoe_load_forecast') }}
