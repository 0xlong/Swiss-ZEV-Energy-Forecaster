-- ML predictions table. Written by ml/predict.py via
-- CREATE OR REPLACE TABLE main.fct_forecasts.
-- This dbt model reads the table back so predictions appear in the dbt
-- lineage graph and get tested like any other mart.
SELECT
    asset_id,
    ts_utc,
    ts_local,
    target,
    y_pred,
    y_true,
    model_version
FROM {{ source('main', 'fct_forecasts') }}
