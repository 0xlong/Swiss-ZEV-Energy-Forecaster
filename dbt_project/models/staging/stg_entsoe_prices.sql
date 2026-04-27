SELECT
    asset_id,
    CAST(ts_utc AS TIMESTAMPTZ)     AS ts_utc,
    CAST(ts_local AS TIMESTAMP)     AS ts_local,
    CAST(price_eur_mwh AS DOUBLE)   AS price_eur_mwh,
    CAST(price_eur_kwh AS DOUBLE)   AS price_eur_kwh
FROM {{ source('raw', 'entsoe_prices') }}
