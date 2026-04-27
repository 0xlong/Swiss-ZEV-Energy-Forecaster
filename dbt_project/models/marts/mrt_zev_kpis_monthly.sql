SELECT
    asset_id,
    DATE_TRUNC('month', date)                                                       AS month,
    SUM(total_pv_kwh)                                                               AS total_pv_kwh,
    SUM(total_load_kwh)                                                             AS total_load_kwh,
    SUM(self_consumption_kwh)                                                       AS self_consumption_kwh,
    SUM(surplus_kwh)                                                                AS surplus_kwh,
    SUM(deficit_kwh)                                                                AS deficit_kwh,
    ROUND(
        SUM(self_consumption_kwh) / NULLIF(SUM(total_pv_kwh), 0),
        4
    )                                                                               AS self_consumption_ratio,
    ROUND(
        SUM(self_consumption_kwh) / NULLIF(SUM(total_load_kwh), 0),
        4
    )                                                                               AS autarky_ratio,
    ROUND(SUM(co2_avoided_kg), 2)                                                   AS co2_avoided_kg,
    ROUND(SUM(chf_saved), 2)                                                        AS chf_saved
FROM {{ ref('mrt_zev_kpis_daily') }}
GROUP BY asset_id, DATE_TRUNC('month', date)
