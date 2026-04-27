-- ML feature table for the Phase D price forecasting model.
-- Exposes all price features: raw price, lags, calendar, weather, load
-- forecast, and lagged gen-mix shares.
SELECT
    asset_id,
    ts_utc,
    ts_local,
    -- Target
    price_eur_mwh,
    price_eur_kwh,
    -- Price lags
    price_lag_1h,
    price_lag_24h,
    price_lag_168h,
    -- Calendar
    hour_of_day,
    day_of_week,
    is_weekend,
    is_holiday,
    -- Weather
    ghi_w_m2,
    temp_c,
    cloud_pct,
    wind_ms,
    -- Load forecast
    load_forecast_mw,
    -- Gen-mix (lagged 24h — no look-ahead)
    solar_share_lag24h,
    hydro_share_lag24h,
    nuclear_share_lag24h,
    total_gen_mw_lag24h
FROM {{ ref('int_price_features') }}
