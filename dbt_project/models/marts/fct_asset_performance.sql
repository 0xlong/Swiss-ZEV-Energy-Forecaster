-- Per-asset per-hour operational facts, enriched with tariff and carbon
-- constants from dim_asset. Battery columns (soc_kwh, charge_kw, discharge_kw)
-- are reserved for Phase F (PuLP battery optimizer).
SELECT
    b.asset_id,
    b.ts_utc,
    b.ts_local,
    b.pv_power_kw,
    b.load_kw,
    b.direct_self_consumption_kw,
    b.surplus_kw,
    b.deficit_kw,
    a.retail_chf_per_kwh,
    a.feed_in_chf_per_kwh,
    a.grid_carbon_g_per_kwh
FROM {{ ref('int_hourly_energy_balance') }} b
INNER JOIN {{ ref('dim_asset') }} a
    USING (asset_id)
