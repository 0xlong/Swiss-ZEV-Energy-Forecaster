-- Daily ZEV KPIs with battery, grouped by battery_capacity_kwh.
-- Mirrors mrt_zev_kpis_daily but reads from fct_asset_performance_battery.
--
-- WHY battery_capacity_kwh IS A GROUP-BY KEY
-- -------------------------------------------
-- The dashboard's capacity slider works by filtering this table:
--   WHERE battery_capacity_kwh = 100
-- Including it in GROUP BY means each slider position gets its own set of
-- daily rows without any additional joins at query time — fast and simple.
--
-- KPI DEFINITIONS WITH BATTERY
-- -----------------------------
-- self_consumption_ratio = on_site_pv / total_pv
--   "What fraction of the solar generation stayed on-site?"
--   "On-site" means used directly or stored in the battery; the battery's
--   round-trip losses are counted as consumed PV (because that PV never
--   reached the grid — it was committed to the battery).
--
-- autarky_ratio = own_covered_load / total_load
--   "What fraction of building demand was met without buying from the grid?"
--   Battery discharge contributes here: stored PV pushed out at night
--   reduces grid import and raises autarky.
--
-- net_chf_cost = SUM(grid_import × retail − grid_export × feed_in)
--   Net electricity bill for the day.  Positive = we owe the grid money.
--   Compare battery=0 vs battery=X to get the daily monetary saving.
--   (The dashboard computes savings by subtracting from the 0-kWh baseline.)
--
-- co2_avoided_kg uses on_site_pv (not own_covered_load) because CO₂ is
-- avoided by *not importing* grid electricity, and the PV generation that
-- stayed on-site (directly used + battery charge) represents that avoided
-- import.  The carbon intensity (~30 g/kWh) is from dim_asset.
SELECT
    asset_id,
    battery_capacity_kwh,
    CAST(ts_local AS DATE)                                                                  AS date,
    SUM(pv_power_kw)                                                                        AS total_pv_kwh,
    SUM(load_kw)                                                                            AS total_load_kwh,
    SUM(on_site_pv_kw)                                                                      AS self_consumption_kwh,
    SUM(own_covered_load_kw)                                                                AS own_covered_load_kwh,
    SUM(grid_import_kw)                                                                     AS grid_import_kwh,
    SUM(grid_export_kw)                                                                     AS grid_export_kwh,
    ROUND(
        COALESCE(SUM(on_site_pv_kw) / NULLIF(SUM(pv_power_kw), 0), 0),
        4
    )                                                                                       AS self_consumption_ratio,
    ROUND(
        COALESCE(SUM(own_covered_load_kw) / NULLIF(SUM(load_kw), 0), 0),
        4
    )                                                                                       AS autarky_ratio,
    ROUND(SUM(on_site_pv_kw) * grid_carbon_g_per_kwh / 1000.0, 3)                         AS co2_avoided_kg,
    ROUND(SUM(net_chf_cost), 2)                                                             AS net_chf_cost
FROM {{ ref('fct_asset_performance_battery') }}
GROUP BY
    asset_id,
    battery_capacity_kwh,
    CAST(ts_local AS DATE),
    grid_carbon_g_per_kwh
