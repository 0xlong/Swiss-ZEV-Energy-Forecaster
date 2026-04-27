"""
predict.py
----------
Load both trained XGBoost models, run inference over the test window
(Jul–Dec 2019), and write predictions to main.fct_forecasts in DuckDB.

Run:
    python ml/predict.py

The dbt model fct_forecasts.sql reads this table back so predictions appear
in the dbt lineage graph and get tested like any other mart.

Schema written to main.fct_forecasts:
    asset_id      TEXT
    ts_utc        TIMESTAMPTZ
    ts_local      TIMESTAMP
    target        TEXT    — 'price_eur_mwh' or 'pv_power_kw'
    y_pred        DOUBLE
    y_true        DOUBLE
    model_version TEXT
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
import duckdb  # noqa: E402
from ml.features import (  # noqa: E402
    load_price_features,
    load_pv_features,
    chronological_split,
)

print("=== Step 9 — Generate predictions & write to DuckDB ===")

# ---------------------------------------------------------------------------
# Chunk 1 — Load both trained model files from disk
# ---------------------------------------------------------------------------
# joblib.load() deserialises the dict we saved in Steps 7 and 8:
#   "model"        — the fitted XGBRegressor with all its trees
#   "feature_cols" — the exact column list used at training time
#   "model_version"— date-stamped string (e.g. "xgb_v1_2026-04-27")
#
# We load feature_cols FROM the pkl rather than re-deriving it here.
# Why?  If someone adds or renames a column in dbt and forgets to retrain,
# using the pkl's column list means predict.py will raise a KeyError
# immediately — a loud, obvious failure — instead of silently feeding the
# wrong features to the model and producing garbage predictions.
# ---------------------------------------------------------------------------

print("\nChunk 1: loading trained models from disk …")

MODEL_DIR = Path(__file__).resolve().parent / "models"

price_bundle = joblib.load(MODEL_DIR / "price_xgb.pkl")
price_model       = price_bundle["model"]
price_feature_cols = price_bundle["feature_cols"]
price_version     = price_bundle["model_version"]

pv_bundle = joblib.load(MODEL_DIR / "pv_xgb.pkl")
pv_model       = pv_bundle["model"]
pv_feature_cols = pv_bundle["feature_cols"]
pv_version     = pv_bundle["model_version"]

print(f"  price model  : {price_version}  ({len(price_feature_cols)} features)")
print(f"  PV model     : {pv_version}  ({len(pv_feature_cols)} features)")

# ---------------------------------------------------------------------------
# Chunk 2 — Reload feature data and isolate the test window
# ---------------------------------------------------------------------------
# We call the same loaders used during training so the data is identical.
# chronological_split() returns (train, val, test) — we keep only test_df.
#
# Why re-load instead of caching the split from training?
#   predict.py is designed to run independently of the training scripts.
#   It must be able to regenerate predictions at any time without relying
#   on in-memory state from a previous training run.
#
# The test window is Jul–Dec 2019: the 6 months the models never saw.
# We restrict inference to this window so we never present in-sample
# (training-period) predictions as if they were genuine forecasts.
# Showing train-period predictions would be dishonest in a dashboard —
# the model already "memorized" those rows to some degree.
# ---------------------------------------------------------------------------

print("\nChunk 2: reloading feature data and isolating test window …")

price_df = load_price_features()
_, _, price_test = chronological_split(price_df)

pv_df = load_pv_features()
_, _, pv_test = chronological_split(pv_df)

print(f"  price test rows : {len(price_test):,}  "
      f"{price_test['ts_utc'].min().date()} → {price_test['ts_utc'].max().date()}")
print(f"  PV test rows    : {len(pv_test):,}  "
      f"{pv_test['ts_utc'].min().date()} → {pv_test['ts_utc'].max().date()}")

# ---------------------------------------------------------------------------
# Chunk 3 — Run inference with each model
# ---------------------------------------------------------------------------
# model.predict(X) runs every tree in the ensemble and sums their outputs
# to produce one prediction per row.  This is fast — no training happens.
#
# Price predictions: no clipping.  Day-ahead electricity prices can be
# negative (oversupply + must-run generation), so we leave the model free
# to predict below 0.  XGBoost rarely does this; if it does, it's valid.
#
# PV predictions: clip to 0.  Physics forbids negative solar production.
# XGBoost doesn't know that — it might predict -0.3 kW at 05:59 local
# as an artefact of interpolating between night and dawn.  Clipping to 0
# is a one-line physics constraint that costs nothing.
# ---------------------------------------------------------------------------

print("\nChunk 3: running inference …")

price_pred = price_model.predict(price_test[price_feature_cols])
pv_pred    = np.clip(pv_model.predict(pv_test[pv_feature_cols]), 0, None)

print(f"  price predictions : {len(price_pred):,} rows  "
      f"range [{price_pred.min():.2f}, {price_pred.max():.2f}] EUR/MWh")
print(f"  PV predictions    : {len(pv_pred):,} rows  "
      f"range [{pv_pred.min():.2f}, {pv_pred.max():.2f}] kW")

# ---------------------------------------------------------------------------
# Chunk 4 — Build the long-format prediction table
# ---------------------------------------------------------------------------
# We stack price and PV rows into one DataFrame using a "target" column
# to distinguish them.  This is the "long" (unpivoted) format.
#
# Why long format instead of two separate tables or wide format?
#   - One table → one dbt model, one test block, one dashboard query.
#   - Adding a third forecast target in the future (e.g. load) is just
#     appending more rows, not adding a column or a new table.
#   - The dashboard can filter WHERE target = 'price_eur_mwh' trivially.
#
# Columns:
#   asset_id      — propagated from the feature table (identifies the site)
#   ts_utc        — UTC timestamp (canonical join key for the dashboard)
#   ts_local      — Europe/Zurich local time (human-readable display)
#   target        — which quantity was predicted ('price_eur_mwh' / 'pv_power_kw')
#   y_pred        — model prediction
#   y_true        — actual value (from the mart; used for evaluation charts)
#   model_version — which pkl was used (date-stamped; critical for re-trains)
# ---------------------------------------------------------------------------

print("\nChunk 4: building long-format predictions DataFrame …")

def _build_rows(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    target_col: str,
    version: str,
) -> pd.DataFrame:
    """Return a long-format slice for one model."""
    return pd.DataFrame({
        "asset_id"     : test_df["asset_id"].values,
        "ts_utc"       : test_df["ts_utc"].values,
        "ts_local"     : test_df["ts_local"].values,
        "target"       : target_col,
        "y_pred"       : y_pred.astype(float),
        "y_true"       : test_df[target_col].values.astype(float),
        "model_version": version,
    })

price_rows = _build_rows(price_test, price_pred, "price_eur_mwh", price_version)
pv_rows    = _build_rows(pv_test,    pv_pred,    "pv_power_kw",   pv_version)

fct_df = pd.concat([price_rows, pv_rows], ignore_index=True)

print(f"  total rows : {len(fct_df):,}  "
      f"({len(price_rows):,} price + {len(pv_rows):,} PV)")
print(f"  columns    : {list(fct_df.columns)}")
print(f"  targets    : {sorted(fct_df['target'].unique())}")

# ---------------------------------------------------------------------------
# Chunk 5 — Write to DuckDB and verify
# ---------------------------------------------------------------------------
# We open DuckDB in read-write mode (no read_only=True).
# CREATE OR REPLACE TABLE overwrites any previous run of predict.py cleanly —
# no manual DROP TABLE needed, no risk of stale rows from a previous version.
#
# We register the DataFrame as a view first so DuckDB can SELECT from it
# directly without a separate INSERT step.
#
# Why write to main.fct_forecasts (not main_marts.fct_forecasts)?
#   The dbt mart fct_forecasts.sql uses {{ source('main', 'fct_forecasts') }}.
#   dbt sources point at the raw / pre-dbt layer; dbt models are the
#   transformed output.  Writing to main.fct_forecasts means the table is
#   a raw input that dbt reads and surfaces in its lineage graph — exactly
#   the same pattern as the other raw.* tables loaded by load_to_duckdb.py.
#   predict.py IS the "ingestion" step for ML predictions.
#
# Verification: we read the row count back immediately after writing so we
# know the commit succeeded and didn't silently write 0 rows.
# ---------------------------------------------------------------------------

print("\nChunk 5: writing to DuckDB …")

with duckdb.connect(str(config.DUCKDB_PATH)) as con:
    con.register("predictions_view", fct_df)
    con.execute(
        "CREATE OR REPLACE TABLE main.fct_forecasts AS SELECT * FROM predictions_view"
    )
    written = con.execute("SELECT COUNT(*) FROM main.fct_forecasts").fetchone()[0]
    by_target = con.execute(
        "SELECT target, COUNT(*) AS n FROM main.fct_forecasts GROUP BY target ORDER BY target"
    ).fetchall()

print(f"  Written {written:,} rows to main.fct_forecasts")
for target, n in by_target:
    print(f"    {target:<20s} {n:>6,} rows")

print(f"\n=== Step 9 complete — fct_forecasts ready in DuckDB ===")
print("  Next: cd dbt_project && dbt build  (picks up fct_forecasts source)")
print("  Then: python ml/evaluate.py")
