"""
train_price_forecast.py
-----------------------
Train an XGBoost model on fct_hourly_market to predict day-ahead electricity prices.

Run:
    python ml/train_price_forecast.py

Exit codes:
    0 — model trained, acceptance gate passed, .pkl saved
    1 — acceptance gate failed (model not 15 % better than seasonal-naive)
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Chunk 1 — Load data & split into train / val / test
# ---------------------------------------------------------------------------
# We pull every row from fct_hourly_market (one per UTC hour, 2015-2019) and
# slice it into three non-overlapping windows in strict time order.
# Train  Jan 2015 – Dec 2018  (48 months, ~35 k rows the model learns from)
# Val    Jan 2019 – Jun 2019  (6 months — used only to stop training early)
# Test   Jul 2019 – Dec 2019  (6 months — the "final exam" the model never saw)
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.features import load_price_features, chronological_split  # noqa: E402

print("=== Step 7 — Train price forecast model ===")
print("Chunk 1: loading features from DuckDB …")

df = load_price_features()
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
# The target is the column we want to predict: price_eur_mwh.
# The model must NOT see ts_utc / ts_local / asset_id during training — they
# are identifiers, not signals (and ts_utc would let the model cheat by
# memorising the training period's prices by timestamp).
# Every remaining column becomes an input feature X.
# ---------------------------------------------------------------------------

print("\nChunk 2: separating features (X) from target (y) …")

TARGET    = "price_eur_mwh"
# price_eur_kwh = price_eur_mwh / 1000 — same value, different unit → leakage
DROP_COLS = ["ts_utc", "ts_local", "asset_id", TARGET, "price_eur_kwh"]

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
# XGBoost builds trees one at a time (up to n_estimators=500).
# After each tree it scores itself on the val set (eval_set).
# If the val MAE hasn't improved in 50 consecutive rounds it stops early —
# this prevents the model from memorising the training data (overfitting).
# learning_rate=0.05 makes each tree contribute only a small step so the
# model converges carefully rather than jumping around.
# ---------------------------------------------------------------------------

from xgboost import XGBRegressor  # noqa: E402

print("\nChunk 3: training XGBRegressor with early stopping …")

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,           # each tree sees 80 % of training rows (reduces variance)
    colsample_bytree=0.8,    # each tree sees 80 % of features (reduces variance)
    random_state=42,
    n_jobs=-1,               # use all CPU cores
    eval_metric="mae",
    early_stopping_rounds=50,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,           # suppress per-round output; we print our own summary
)

best_round = model.best_iteration + 1
print(f"  Stopped at round {best_round} / 500  (early stopping after 50 non-improving rounds)")
print(f"  Best val MAE: {model.best_score:.3f} EUR/MWh")

# ---------------------------------------------------------------------------
# Chunk 4 — Evaluate on the test set: overall + peak vs off-peak
# ---------------------------------------------------------------------------
# model.predict(X_test) runs all 481 trees on the 6-month test window and
# returns one predicted price per hour.  We then score three ways:
#   MAE   — average absolute miss in EUR/MWh (most interpretable)
#   MAPE  — miss as % of actual price (good for stakeholder comms)
#   RMSE  — penalises large spike misses more than MAE does
#
# Peak split (8–20 h local): daytime hours have the highest prices and drive
# the bulk of ZEV CHF savings, so a separate score there matters.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

print("\nChunk 4: evaluating on the 6-month test window …")

y_pred = model.predict(X_test)

def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae  = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return {"MAE": mae, "MAPE": mape, "RMSE": rmse}

overall = _metrics(y_test.values, y_pred)

peak_mask    = test_df["hour_of_day"].between(8, 19)   # 08:00–19:59 local
offpeak_mask = ~peak_mask

peak_m    = _metrics(y_test[peak_mask].values,    y_pred[peak_mask])
offpeak_m = _metrics(y_test[offpeak_mask].values, y_pred[offpeak_mask])

print(f"  {'':12s}  {'MAE':>8}  {'MAPE':>8}  {'RMSE':>8}")
print(f"  {'Overall':12s}  {overall['MAE']:>8.3f}  {overall['MAPE']:>7.2f}%  {overall['RMSE']:>8.3f}")
print(f"  {'Peak 8-20h':12s}  {peak_m['MAE']:>8.3f}  {peak_m['MAPE']:>7.2f}%  {peak_m['RMSE']:>8.3f}")
print(f"  {'Off-peak':12s}  {offpeak_m['MAE']:>8.3f}  {offpeak_m['MAPE']:>7.2f}%  {offpeak_m['RMSE']:>8.3f}")

# ---------------------------------------------------------------------------
# Chunk 5 — Baseline comparison & acceptance gate
# ---------------------------------------------------------------------------
# price_lag_168h is already in our test data — it's the actual price from
# exactly 1 week ago at that same hour (computed by dbt, no extra work).
# That IS the seasonal-naive prediction.
#
# Gate: model MAE must be ≤ 85 % of seasonal-naive MAE.
# If the model can't beat "just copy last week" by 15 %, something is wrong
# with the features — we don't save the model and exit with code 1.
# ---------------------------------------------------------------------------

print("\nChunk 5: comparing against seasonal-naive baseline …")

sn_pred = test_df["price_lag_168h"].values
sn_mae  = np.mean(np.abs(y_test.values - sn_pred))
model_mae = overall["MAE"]
threshold = 0.85 * sn_mae

print(f"  Seasonal-naive MAE : {sn_mae:.3f} EUR/MWh")
print(f"  XGBoost MAE        : {model_mae:.3f} EUR/MWh")
print(f"  Acceptance gate    : model MAE ≤ {threshold:.3f}  (85 % × {sn_mae:.3f})")

if model_mae > threshold:
    print(f"\n  FAIL — model MAE {model_mae:.3f} > gate {threshold:.3f}")
    print("  Not saving model. Diagnose feature quality and retrain.")
    sys.exit(1)

improvement_pct = (1 - model_mae / sn_mae) * 100
print(f"  PASS — {improvement_pct:.1f} % better than seasonal-naive ✓")

# ---------------------------------------------------------------------------
# Chunk 6 — Persist the trained model to disk
# ---------------------------------------------------------------------------
# joblib serialises the fitted XGBRegressor (all 481 trees + metadata) to a
# .pkl file.  predict.py loads it with joblib.load() — no retraining needed.
# model_version embeds today's date so we can tell apart future re-trains.
# ---------------------------------------------------------------------------

import joblib  # noqa: E402
import datetime  # noqa: E402

print("\nChunk 6: saving model to disk …")

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

model_version = f"xgb_v1_{datetime.date.today()}"
model_path    = MODEL_DIR / "price_xgb.pkl"

joblib.dump({"model": model, "feature_cols": feature_cols, "model_version": model_version}, model_path)

print(f"  Saved → {model_path}")
print(f"  model_version: {model_version}")
print(f"  features bundled: {len(feature_cols)}")
print(f"\n=== Step 7 complete — price model ready ===")
