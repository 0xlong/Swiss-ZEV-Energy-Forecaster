-- 5 duplicate (asset_id, ts_utc) rows exist in raw (one per spring-forward
-- DST transition 2015–2019). Average load_kw for the duplicated hour so the
-- mart KPIs account for that hour rather than silently dropping it.
SELECT
    asset_id,
    CAST(ts_utc AS TIMESTAMPTZ)  AS ts_utc,
    CAST(ts_local AS TIMESTAMP)  AS ts_local,
    AVG(CAST(load_kw AS DOUBLE)) AS load_kw
FROM {{ source('raw', 'load_hourly') }}
GROUP BY asset_id, ts_utc, ts_local
