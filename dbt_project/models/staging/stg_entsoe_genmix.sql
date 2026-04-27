-- ENTSO-E actual generation per production type for Switzerland.
-- Raw data is in long format (one row per gen_type per timestamp).
-- This model pivots to wide: one row per (asset_id, ts_utc) with columns
-- for each major generation type plus shares and total.
WITH raw_long AS (
    SELECT
        asset_id,
        CAST(ts_utc AS TIMESTAMPTZ)      AS ts_utc,
        CAST(ts_local AS TIMESTAMP)      AS ts_local,
        gen_type,
        CAST(generation_mw AS DOUBLE)   AS generation_mw
    FROM {{ source('raw', 'entsoe_genmix') }}
),

pivoted AS (
    SELECT
        asset_id,
        ts_utc,
        ts_local,
        SUM(CASE WHEN gen_type = 'solar' THEN generation_mw ELSE 0 END)                             AS solar_mw,
        SUM(CASE WHEN gen_type LIKE 'hydro%' THEN generation_mw ELSE 0 END)                         AS hydro_mw,
        SUM(CASE WHEN gen_type = 'nuclear' THEN generation_mw ELSE 0 END)                           AS nuclear_mw,
        SUM(CASE WHEN gen_type = 'fossil_gas' THEN generation_mw ELSE 0 END)                        AS gas_mw,
        SUM(CASE WHEN gen_type = 'wind_onshore' THEN generation_mw ELSE 0 END)                      AS wind_mw,
        SUM(generation_mw)                                                                           AS total_gen_mw
    FROM raw_long
    GROUP BY asset_id, ts_utc, ts_local
)

SELECT
    asset_id,
    ts_utc,
    ts_local,
    solar_mw,
    hydro_mw,
    nuclear_mw,
    gas_mw,
    wind_mw,
    total_gen_mw,
    CASE WHEN total_gen_mw > 0 THEN solar_mw   / total_gen_mw ELSE 0 END AS solar_share,
    CASE WHEN total_gen_mw > 0 THEN hydro_mw   / total_gen_mw ELSE 0 END AS hydro_share,
    CASE WHEN total_gen_mw > 0 THEN nuclear_mw / total_gen_mw ELSE 0 END AS nuclear_share
FROM pivoted
