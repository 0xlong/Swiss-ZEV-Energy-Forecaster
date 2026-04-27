-- ML feature table: price lags (1h, 24h, 168h), hour-of-day, day-of-week,
-- is_weekend, CH public holiday flag, weather features (GHI, temp, cloud),
-- load forecast, and lagged generation-mix shares (24h lag to avoid
-- day-ahead look-ahead leakage).
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
),

holidays AS (
    SELECT
        b.*,
        (b.day_of_week IN (0, 6))      AS is_weekend,
        h.holiday_date IS NOT NULL     AS is_holiday
    FROM base b
    LEFT JOIN {{ ref('dim_ch_holidays') }} h
        ON CAST(b.ts_local AS DATE) = h.holiday_date
),

-- Join weather, load forecast, and lagged gen-mix
enriched AS (
    SELECT
        p.*,
        -- Weather features
        w.ghi_w_m2,
        w.temp_c,
        w.cloud_pct,
        w.wind_ms,
        -- Load forecast
        lf.load_forecast_mw,
        -- Gen-mix shares lagged 24h to avoid look-ahead leakage:
        -- when forecasting tomorrow's price, we only know yesterday's actual mix
        LAG(g.solar_share, 24)   OVER (PARTITION BY p.asset_id ORDER BY p.ts_utc) AS solar_share_lag24h,
        LAG(g.hydro_share, 24)   OVER (PARTITION BY p.asset_id ORDER BY p.ts_utc) AS hydro_share_lag24h,
        LAG(g.nuclear_share, 24) OVER (PARTITION BY p.asset_id ORDER BY p.ts_utc) AS nuclear_share_lag24h,
        LAG(g.total_gen_mw, 24)  OVER (PARTITION BY p.asset_id ORDER BY p.ts_utc) AS total_gen_mw_lag24h
    FROM holidays p
    LEFT JOIN {{ ref('stg_openmeteo_hourly') }} w
        ON p.asset_id = w.asset_id AND p.ts_utc = w.ts_utc
    LEFT JOIN {{ ref('stg_entsoe_load_forecast') }} lf
        ON p.asset_id = lf.asset_id AND p.ts_utc = lf.ts_utc
    LEFT JOIN {{ ref('stg_entsoe_genmix') }} g
        ON p.asset_id = g.asset_id AND p.ts_utc = g.ts_utc
)

SELECT * FROM enriched
