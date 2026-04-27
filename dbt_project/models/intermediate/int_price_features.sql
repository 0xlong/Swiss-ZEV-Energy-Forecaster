-- ML feature table: price lags (1h, 24h, 168h), hour-of-day, day-of-week,
-- is_weekend, and CH public holiday flag. Weather and generation-mix columns
-- are deferred to Phase D when Open-Meteo and ENTSO-E gen-mix are ingested.
WITH base AS (
    SELECT
        asset_id,
        ts_utc,
        ts_local,
        price_eur_mwh,
        price_eur_kwh,
        LAG(price_eur_mwh, 1)   OVER (PARTITION BY asset_id ORDER BY ts_utc) AS price_lag_1h,
        LAG(price_eur_mwh, 24)  OVER (PARTITION BY asset_id ORDER BY ts_utc) AS price_lag_24h,
        LAG(price_eur_mwh, 168) OVER (PARTITION BY asset_id ORDER BY ts_utc) AS price_lag_168h,
        EXTRACT(hour FROM ts_local)::INTEGER AS hour_of_day,
        EXTRACT(dow  FROM ts_local)::INTEGER AS day_of_week
    FROM {{ ref('stg_entsoe_prices') }}
)
SELECT
    b.*,
    (b.day_of_week IN (0, 6))      AS is_weekend,
    h.holiday_date IS NOT NULL     AS is_holiday
FROM base b
LEFT JOIN {{ ref('dim_ch_holidays') }} h
    ON CAST(b.ts_local AS DATE) = h.holiday_date
