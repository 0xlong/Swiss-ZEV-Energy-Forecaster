"""
train_pv_forecast.py
--------------------
Train an XGBoost model on fct_pv_features to predict solar PV production.

Run:
    python ml/train_pv_forecast.py

Exit codes:
    0 — model trained, acceptance gate passed, .pkl saved
    1 — acceptance gate failed (model no better than persistence baseline)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Chunk 1 — Load data & split into train / val / test
# ---------------------------------------------------------------------------
# Same structure as the price model:
# We load fct_pv_features (one row per UTC hour, 2015-2019).
# The data includes PV production, weather (GHI, temp, cloud, wind),
# calendar signals, and 3 production lags — all pre-computed by dbt.
#
# Split follows the same boundaries:
#   train  Jan 2015 – Dec 2018  (48 months, ~35 k rows)
#   val    Jan 2019 – Jun 2019  (6 months — early-stopping signal)
#   test   Jul 2019 – Dec 2019  (6 months — final exam, never seen)
#
# Why keep the same split? PV data covers the same 2015-2019 window
# so there's no reason to change the boundaries.
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import load_pv_features, chronological_split  # noqa: E402

print("=== Step 8 — Train PV production forecast model ===")
print("Chunk 1: loading PV features from DuckDB …")

df = load_pv_features()
print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
print(f"  Date range: {df['ts_utc'].min().date()} → {df['ts_utc'].max().date()}")

train_df, val_df, test_df = chronological_split(df)
print(
    f"  train {len(train_df):>6,} rows  "
    f"{train_df['ts_utc'].min().date()} → {train_df['ts_utc'].max().date()}"
)
print(
    f"  val   {len(val_df):>6,} rows  "
    f"{val_df['ts_utc'].min().date()} → {val_df['ts_utc'].max().date()}"
)
print(
    f"  test  {len(test_df):>6,} rows  "
    f"{test_df['ts_utc'].min().date()} → {test_df['ts_utc'].max().date()}"
)

# ---------------------------------------------------------------------------
# Chunk 2 — Separate input features (X) from the target (y)
# ---------------------------------------------------------------------------
# Target: pv_power_kw — the actual solar output in kilowatts each hour.
# Drop identifiers (ts_utc, ts_local, asset_id) — they are row labels, not
# signals.  No leakage concern here: there is no "pv_power_kw / 1000" column.
#
# The 10 features the model gets to use:
#   lags     — pv_lag_1h, pv_lag_24h, pv_lag_168h
#              (yesterday same hour and last-week same hour are strong signals
#               for whether the sun was shining at this time of day recently)
#   calendar — hour_of_day, day_of_year, day_of_week
#              (captures the solar arc across the sky throughout the year;
#               day_of_year encodes seasonality — summer days are longer)
#   weather  — ghi_w_m2, temp_c, cloud_pct, wind_ms
#              (GHI = global horizontal irradiance — this is the dominant
#               physical driver; cloud_pct is the random "sunny vs cloudy"
#               component that makes any forecast hard)
# ---------------------------------------------------------------------------

print("\nChunk 2: separating features (X) from target (y) …")

TARGET    = "pv_power_kw"
DROP_COLS = ["ts_utc", "ts_local", "asset_id", TARGET]

feature_cols = [c for c in df.columns if c not in DROP_COLS]
print(f"  Target:   {TARGET}")
print(f"  Features: {feature_cols}")

X_train, y_train = train_df[feature_cols], train_df[TARGET]
X_val,   y_val   = val_df[feature_cols],   val_df[TARGET]
X_test,  y_test  = test_df[feature_cols],  test_df[TARGET]

print(f"  X_train shape: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")

# ---------------------------------------------------------------------------
# Chunk 3 — Train XGBoost with early stopping on the validation set
# ---------------------------------------------------------------------------
# Identical hyperparameters to the price model — this is intentional:
# for a portfolio project, consistent settings across models makes it easy
# to explain that no per-model hyperparameter hunting was done (which would
# risk overfitting to the test set).
#
# The model will naturally figure out that GHI dominates:
# GHI (sunlight) → panel output is near-linear physics, so a tree that
# splits on "ghi > 400 W/m²" can immediately separate day/night and
# sunny/cloudy hours.  The lags and cloud_pct fill in the residual noise.
#
# early_stopping_rounds=50: if val MAE doesn't improve for 50 consecutive
# trees, stop.  For a physics-driven target like PV, convergence is often
# faster than for market prices — expect fewer than 500 trees used.
# ---------------------------------------------------------------------------

from xgboost import XGBRegressor  # noqa: E402

print("\nChunk 3: training XGBRegressor with early stopping …")

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,        # each tree sees 80 % of training rows
    colsample_bytree=0.8, # each tree sees 80 % of features
    random_state=42,
    n_jobs=-1,
    eval_metric="mae",
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

best_round = model.best_iteration + 1
print(f"  Stopped at round {best_round} / 500  (early stopping after 50 non-improving rounds)")
print(f"  Best val MAE: {model.best_score:.3f} kW")

# ---------------------------------------------------------------------------
# Chunk 4 — Evaluate on the test set: overall + daytime vs nighttime
# ---------------------------------------------------------------------------
# For PV we split on productive hours rather than peak/off-peak:
#
#   Nighttime (hour_of_day < 6 or > 20): production is always ~0 kW.
#   XGBoost will learn this trivially — MAE near 0, so overall MAE gets
#   dragged down artificially if we don't separate them.
#
#   Daytime (6 ≤ hour_of_day ≤ 20): this is where the model must actually
#   work — predicting how much the panels produce on a sunny vs cloudy day.
#   This is the number that matters for the ZEV self-consumption calculation.
#
# We clip predictions to 0 before scoring: physics forbids negative
# production, but XGBoost doesn't know that.  A tiny negative prediction
# at dawn/dusk is a rounding artefact, not a modelling failure.
#
# MAPE note: when y_true = 0 (nighttime), MAPE is undefined (div-by-zero).
# We substitute 1 in the denominator — same guard used in the price model —
# but MAPE is only meaningful in the daytime split.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

print("\nChunk 4: evaluating on the 6-month test window …")

# Clip negatives: XGBoost can predict slightly below 0 near dawn/dusk
y_pred = np.clip(model.predict(X_test), 0, None)

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}

overall = _metrics(y_test.values, y_pred)

# Daytime = hours where PV can actually produce (6:00–20:59 local)
daytime_mask  = test_df["hour_of_day"].between(6, 20)
nighttime_mask = ~daytime_mask

daytime_m  = _metrics(y_test[daytime_mask].values,  y_pred[daytime_mask])
nighttime_m = _metrics(y_test[nighttime_mask].values, y_pred[nighttime_mask])

print(f"  {'':12s}  {'MAE':>8}  {'MAPE':>8}  {'RMSE':>8}")
print(f"  {'Overall':12s}  {overall['MAE']:>8.3f}  {overall['MAPE']:>7.2f}%  {overall['RMSE']:>8.3f}")
print(f"  {'Daytime 6-20h':13s} {daytime_m['MAE']:>8.3f}  {daytime_m['MAPE']:>7.2f}%  {daytime_m['RMSE']:>8.3f}")
print(f"  {'Nighttime':12s}  {nighttime_m['MAE']:>8.3f}  {nighttime_m['MAPE']:>7.2f}%  {nighttime_m['RMSE']:>8.3f}")
print("  (MAPE is meaningful for daytime only — nighttime y_true ≈ 0)")

# ---------------------------------------------------------------------------
# Chunk 5 — Baseline comparison & acceptance gate
# ---------------------------------------------------------------------------
# The baseline for PV is persistence (lag-24h): "predict the same production
# as yesterday at this exact hour."  This is weaker than seasonal-naive
# (same hour last week) because a single cloudy day breaks it completely.
#
# Why persistence and not seasonal-naive?
#   The build plan (§13) only specifies a hard MAE gate for the price model.
#   For PV, the bar is deliberately softer: if XGBoost with GHI as a feature
#   can't beat a model that just copies yesterday, something is fundamentally
#   broken.  Seasonal-naive would be a tougher bar and also harder to beat
#   (same weekday last week correlates with seasonal sun angle better than
#   same hour yesterday during cloudy spells).
#
# Gate: model MAE < persistence MAE  (any improvement counts, no % threshold)
# We use pv_lag_24h from test_df directly — dbt already computed it.
# ---------------------------------------------------------------------------

print("\nChunk 5: comparing against persistence baseline …")

persist_pred = test_df["pv_lag_24h"].values
persist_mae  = np.mean(np.abs(y_test.values - persist_pred))
model_mae    = overall["MAE"]

print(f"  Persistence (lag-24h) MAE : {persist_mae:.3f} kW")
print(f"  XGBoost MAE               : {model_mae:.3f} kW")
print(f"  Acceptance gate           : model MAE < persistence MAE ({persist_mae:.3f})")

if model_mae >= persist_mae:
    print(f"\n  FAIL — model MAE {model_mae:.3f} ≥ persistence {persist_mae:.3f}")
    print("  Not saving model. Check weather feature join and retrain.")
    sys.exit(1)

improvement_pct = (1 - model_mae / persist_mae) * 100
print(f"  PASS — {improvement_pct:.1f} % better than persistence ✓")

# ---------------------------------------------------------------------------
# Chunk 6 — Persist the trained model to disk
# ---------------------------------------------------------------------------
# Same structure as the price model pkl:
#   "model"        — fitted XGBRegressor (all trees)
#   "feature_cols" — list of column names used during training
#   "model_version"— date-stamped string so re-trains are distinguishable
#
# predict.py loads both pkl files and runs inference over the test window,
# writing results to main.fct_forecasts in DuckDB.
# feature_cols is bundled in the pkl so predict.py never has to know which
# columns were used — it just reads the list from the file.
# ---------------------------------------------------------------------------

import joblib    # noqa: E402
import datetime  # noqa: E402

print("\nChunk 6: saving model to disk …")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

model_version = f"xgb_v1_{datetime.date.today()}"
model_path    = MODEL_DIR / "pv_xgb.pkl"

joblib.dump(
    {"model": model, "feature_cols": feature_cols, "model_version": model_version},
    model_path,
)

print(f"  Saved → {model_path}")
print(f"  model_version: {model_version}")
print(f"  features bundled: {len(feature_cols)}")
print(f"\n=== Step 8 complete — PV model ready ===")
