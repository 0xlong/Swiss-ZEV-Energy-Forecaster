-- Monthly rollup of battery-aware ZEV KPIs.
-- Aggregates mrt_zev_kpis_battery_daily → one row per (asset_id, month, capacity).
--
-- This is the table the Streamlit Page 3 slider queries:
--   SELECT * FROM main_marts.mrt_zev_kpis_battery_monthly
--   WHERE battery_capacity_kwh = <slider_value>
--   ORDER BY month
--
-- Monthly granularity is fine for the payback and KPI trend charts; the daily
-- table is available if the dashboard ever needs finer resolution.
--
-- Ratios are re-derived from the monthly sums (not averaged from daily ratios)
-- to ensure they are volume-weighted.  Averaging daily ratios would give equal
-- weight to a zero-PV winter day and a peak summer day — incorrect.
SELECT
    asset_id,
    battery_capacity_kwh,
    DATE_TRUNC('month', date)                                                       AS month,
    SUM(total_pv_kwh)                                                               AS total_pv_kwh,
    SUM(total_load_kwh)                                                             AS total_load_kwh,
    SUM(self_consumption_kwh)                                                       AS self_consumption_kwh,
    SUM(own_covered_load_kwh)                                                       AS own_covered_load_kwh,
    SUM(grid_import_kwh)                                                            AS grid_import_kwh,
    SUM(grid_export_kwh)                                                            AS grid_export_kwh,
    ROUND(
        SUM(self_consumption_kwh) / NULLIF(SUM(total_pv_kwh), 0),
        4
    )                                                                               AS self_consumption_ratio,
    ROUND(
        SUM(own_covered_load_kwh) / NULLIF(SUM(total_load_kwh), 0),
        4
    )                                                                               AS autarky_ratio,
    ROUND(SUM(co2_avoided_kg), 2)                                                   AS co2_avoided_kg,
    ROUND(SUM(net_chf_cost), 2)                                                     AS net_chf_cost
FROM {{ ref('mrt_zev_kpis_battery_daily') }}
GROUP BY asset_id, battery_capacity_kwh, DATE_TRUNC('month', date)
