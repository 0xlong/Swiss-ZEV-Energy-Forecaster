-- Mart wrapper for the PV forecasting model.
-- Exposes PV target, weather features, calendar, and PV lags.
SELECT
    asset_id,
    ts_utc,
    ts_local,
    -- Target
    pv_power_kw,
    -- PV lags
    pv_lag_1h,
    pv_lag_24h,
    pv_lag_168h,
    -- Calendar
    hour_of_day,
    day_of_year,
    day_of_week,
    -- Weather
    ghi_w_m2,
    temp_c,
    cloud_pct,
    wind_ms
FROM {{ ref('int_pv_features') }}
