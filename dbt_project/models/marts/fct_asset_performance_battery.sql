-- Per-asset per-hour fact table extended with battery dispatch columns.
-- Grain: (asset_id, ts_utc, battery_capacity_kwh)
--
-- WHY A PARALLEL TABLE INSTEAD OF MODIFYING fct_asset_performance?
-- -----------------------------------------------------------------
-- fct_asset_performance has grain (asset_id, ts_utc) with a UNIQUE constraint
-- on ts_utc.  Adding battery_capacity_kwh would multiply every row by 5 (one
-- per pre-computed capacity), breaking the unique constraint and making every
-- Page 2 dashboard query return 5× as many rows — breaking the charts.
--
-- A parallel table keeps both grains intact:
--   fct_asset_performance        → grain (asset_id, ts_utc)         → Page 2
--   fct_asset_performance_battery → grain (asset_id, ts_utc, capacity) → Page 3
--
-- KEY DERIVED COLUMNS
-- -------------------
-- on_site_pv_kw: PV that stays on-site = pv_power_kw - grid_export_kw.
--   Includes PV used directly AND PV stored in the battery.
--   Even though some stored energy is lost to round-trip efficiency, the
--   PV generation itself was "used on-site" — it never went to the grid.
--
-- own_covered_load_kw: load covered by own resources = load_kw - grid_import_kw.
--   This is the load fraction the building met without buying from the grid.
--
-- net_chf_cost: hourly electricity bill = imports × retail − exports × feed_in.
--   Summing this over a year and comparing across battery capacities gives the
--   annual savings from storage — the core financial input for payback analysis.
--
-- GREATEST(..., 0.0) guards against LP floating-point noise that could produce
-- tiny negative values (e.g. on_site_pv when grid_export slightly exceeds pv).
SELECT
    s.asset_id,
    s.ts_utc,
    s.ts_local,
    s.battery_capacity_kwh,
    b.pv_power_kw,
    b.load_kw,
    GREATEST(b.pv_power_kw - s.grid_export_kw, 0.0)    AS on_site_pv_kw,
    GREATEST(b.load_kw     - s.grid_import_kw, 0.0)    AS own_covered_load_kw,
    s.charge_kw,
    s.discharge_kw,
    s.soc_kwh,
    s.grid_import_kw,
    s.grid_export_kw,
    a.retail_chf_per_kwh,
    a.feed_in_chf_per_kwh,
    a.grid_carbon_g_per_kwh,
    s.grid_import_kw * a.retail_chf_per_kwh
        - s.grid_export_kw * a.feed_in_chf_per_kwh     AS net_chf_cost
FROM {{ ref('stg_battery_simulations') }} s
INNER JOIN {{ ref('int_hourly_energy_balance') }} b
    ON s.asset_id = b.asset_id AND s.ts_utc = b.ts_utc
INNER JOIN {{ ref('dim_asset') }} a
    ON s.asset_id = a.asset_id
