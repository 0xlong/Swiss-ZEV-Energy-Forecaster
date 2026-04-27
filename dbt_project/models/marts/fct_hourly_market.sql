-- ML feature table for the Phase D price and PV forecasting models.
-- Phase D writes predictions back to this schema via fct_forecasts.
SELECT
    asset_id,
    ts_utc,
    ts_local,
    price_eur_mwh,
    price_eur_kwh,
    price_lag_1h,
    price_lag_24h,
    price_lag_168h,
    hour_of_day,
    day_of_week,
    is_weekend,
    is_holiday
FROM {{ ref('int_price_features') }}
