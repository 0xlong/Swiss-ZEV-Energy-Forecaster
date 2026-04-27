-- Daily ZEV KPIs for the Baar warehouse.
-- kW × 1h = kWh because the source grain is hourly.
-- Carbon intensity: ~30 gCO2/kWh for CH grid (mostly hydro + nuclear).
-- Source: IEA "Electricity 2024" and Swiss Federal Office of Energy (SFOE)
-- grid emission factor. Will be replaced by per-hour ENTSO-E gen-mix
-- calculation in Phase D.
-- co2_avoided_kg: divide grams by 1000 for a more readable column.
SELECT
    asset_id,
    CAST(ts_local AS DATE)                                                              AS date,
    SUM(pv_power_kw)                                                                    AS total_pv_kwh,
    SUM(load_kw)                                                                        AS total_load_kwh,
    SUM(direct_self_consumption_kw)                                                     AS self_consumption_kwh,
    SUM(surplus_kw)                                                                     AS surplus_kwh,
    SUM(deficit_kw)                                                                     AS deficit_kwh,
    ROUND(
        COALESCE(SUM(direct_self_consumption_kw) / NULLIF(SUM(pv_power_kw), 0), 0),
        4
    )                                                                                   AS self_consumption_ratio,
    ROUND(
        COALESCE(SUM(direct_self_consumption_kw) / NULLIF(SUM(load_kw), 0), 0),
        4
    )                                                                                   AS autarky_ratio,
    ROUND(
        SUM(direct_self_consumption_kw) * grid_carbon_g_per_kwh / 1000.0,
        3
    )                                                                                   AS co2_avoided_kg,
    ROUND(
        SUM(direct_self_consumption_kw) * (retail_chf_per_kwh - feed_in_chf_per_kwh),
        2
    )                                                                                   AS chf_saved
FROM {{ ref('fct_asset_performance') }}
GROUP BY
    asset_id,
    CAST(ts_local AS DATE),
    grid_carbon_g_per_kwh,
    retail_chf_per_kwh,
    feed_in_chf_per_kwh
