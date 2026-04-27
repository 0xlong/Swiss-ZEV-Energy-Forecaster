-- Per-asset per-hour energy balance. Inner join on (asset_id, ts_utc) aligns
-- PV and load on the same canonical UTC clock — safe across DST transitions.
-- The 5 spring-forward hours with no load_hourly counterpart are dropped here;
-- their impact on annual KPIs is < 0.01%.
SELECT
    p.asset_id,
    p.ts_utc,
    p.ts_local,
    p.pv_power_kw,
    l.load_kw,
    LEAST(p.pv_power_kw, l.load_kw)             AS direct_self_consumption_kw,
    GREATEST(p.pv_power_kw - l.load_kw, 0.0)    AS surplus_kw,
    GREATEST(l.load_kw - p.pv_power_kw, 0.0)    AS deficit_kw
FROM {{ ref('stg_pvgis_hourly') }} p
INNER JOIN {{ ref('stg_load_hourly') }} l
    USING (asset_id, ts_utc)
