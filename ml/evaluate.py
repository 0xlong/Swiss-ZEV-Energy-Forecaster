"""
evaluate.py
-----------
Evaluate both ML forecasts against naive baselines on the held-out test window
(Jul–Dec 2019) and persist results to disk.

Run:
    python ml/evaluate.py

Requires predict.py to have been run first so main.fct_forecasts exists in DuckDB.

Outputs
-------
docs/metrics.json               — MAE / MAPE / RMSE for XGBoost + 2 baselines,
                                  both targets, overall + time-of-day split.
docs/forecast_vs_baseline.png   — 2-panel chart: price (6-month view) +
                                  PV production (2-week zoom).
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from ml.features import (  # noqa: E402
    load_price_features,
    load_pv_features,
    chronological_split,
)

print("=== Step 10 — Evaluate forecasts vs baselines ===")

# ---------------------------------------------------------------------------
# Chunk 1 — Load ML predictions from DuckDB
# ---------------------------------------------------------------------------
# predict.py wrote price and PV predictions to main.fct_forecasts.
# We read it here and immediately split on the `target` column so the rest
# of the script works with two clean DataFrames (one per forecast target).
#
# Why read the stored table instead of re-running predict.py logic?
#   evaluate.py is an audit step, not an inference step.  Reading the
#   written table means we evaluate exactly what was saved to DuckDB —
#   including any bugs that may have crept in during the write.  If we
#   recomputed predictions here, we would silently hide those bugs.
# ---------------------------------------------------------------------------

print("\nChunk 1: loading ML predictions from main.fct_forecasts …")

with duckdb.connect(str(config.DUCKDB_PATH), read_only=True) as con:
    fc_df = con.execute(
        "SELECT * FROM main.fct_forecasts ORDER BY target, ts_utc"
    ).df()

fc_df["ts_utc"]   = pd.to_datetime(fc_df["ts_utc"], utc=True)
fc_df["ts_local"] = pd.to_datetime(fc_df["ts_local"])

price_fc = fc_df[fc_df["target"] == "price_eur_mwh"].reset_index(drop=True)
pv_fc    = fc_df[fc_df["target"] == "pv_power_kw"].reset_index(drop=True)

print(f"  price rows : {len(price_fc):,}  "
      f"{price_fc['ts_utc'].min().date()} → {price_fc['ts_utc'].max().date()}")
print(f"  PV rows    : {len(pv_fc):,}  "
      f"{pv_fc['ts_utc'].min().date()} → {pv_fc['ts_utc'].max().date()}")
print(f"  price model : {price_fc['model_version'].iloc[0]}")
print(f"  PV model    : {pv_fc['model_version'].iloc[0]}")

# ---------------------------------------------------------------------------
# Chunk 2 — Attach baseline lag columns and time-of-day labels
# ---------------------------------------------------------------------------
# The naive baselines (persistence / seasonal-naive) are not stored in
# fct_forecasts — they only appeared inside the training scripts as
# acceptance-gate comparisons.  Here we re-derive them from the feature
# tables so the evaluation script is self-contained.
#
# We use the *pre-computed* lag columns that dbt already wrote into
# fct_hourly_market and fct_pv_features (window functions LAG(col, 24)
# and LAG(col, 168)).  No new computation is needed — just a left-join
# on ts_utc to pull those columns into our prediction DataFrames.
#
# We also pull `hour_of_day` from the feature tables so we can later split
# the metrics into peak/off-peak (price) and daytime/nighttime (PV) slices.
#
# Why not store baselines in fct_forecasts during predict.py?
#   predict.py's job is ML inference.  Mixing baseline columns there would
#   couple the inference step to evaluation logic, making both harder to
#   test independently.  Baselines are cheap to recompute and belong here.
# ---------------------------------------------------------------------------

print("\nChunk 2: attaching baseline lag columns from feature tables …")

price_df = load_price_features()
_, _, price_test_feats = chronological_split(price_df)

pv_df = load_pv_features()
_, _, pv_test_feats = chronological_split(pv_df)

# Single merge per target: bring in both lag columns + hour_of_day at once
price_fc = price_fc.merge(
    price_test_feats[["ts_utc", "price_lag_24h", "price_lag_168h", "hour_of_day"]],
    on="ts_utc",
    how="left",
)
pv_fc = pv_fc.merge(
    pv_test_feats[["ts_utc", "pv_lag_24h", "pv_lag_168h", "hour_of_day"]],
    on="ts_utc",
    how="left",
)

null_price = price_fc[["price_lag_24h", "price_lag_168h"]].isna().sum().sum()
null_pv    = pv_fc[["pv_lag_24h", "pv_lag_168h"]].isna().sum().sum()
print(f"  null price lags after merge : {null_price}")
print(f"  null PV lags after merge    : {null_pv}")

if null_price > 0 or null_pv > 0:
    print("  WARNING: null lags found — timestamps in fct_forecasts do not "
          "align with the feature tables.  Check predict.py and dbt build.")

# ---------------------------------------------------------------------------
# Chunk 3 — Compute MAE / MAPE / RMSE for every (model, split) combination
# ---------------------------------------------------------------------------
# We measure three metrics for each combination:
#
#   MAE   — mean absolute error.  Same unit as the target (EUR/MWh or kW).
#            The most interpretable metric for non-technical stakeholders:
#            "on average, our price forecast is off by X EUR/MWh".
#
#   MAPE  — mean absolute percentage error.  Normalised so 4 % and 8 % are
#            directly comparable across different price regimes.
#            MAPE is UNDEFINED when y_true = 0 (div-by-zero).  We guard with
#            np.where(y_true == 0, 1, y_true) — capping the contribution of
#            a zero-actual row at 100 %.  For PV nighttime this makes MAPE
#            meaningless, which we flag explicitly in the output.
#
#   RMSE  — root mean squared error.  Penalises large spikes more than MAE
#            because errors are squared before averaging.  A much higher RMSE
#            than MAE signals that the model occasionally produces big misses.
#
# Time-of-day splits:
#   Price → overall | peak 08:00–19:59 | off-peak  (matches train_price_forecast)
#   PV    → overall | daytime 06:00–20:59 | nighttime  (matches train_pv_forecast)
#
# Models evaluated per split:
#   xgboost          — the ML model's y_pred from fct_forecasts
#   persistence_24h  — predict same value as 24 h ago (lag-24)
#   seasonal_naive   — predict same value as 168 h ago (lag-168, same weekday)
# ---------------------------------------------------------------------------

print("\nChunk 3: computing metrics for all models and splits …")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, MAPE, RMSE as a dict with 4-decimal precision."""
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    )
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": round(mae, 4), "mape": round(mape, 4), "rmse": round(rmse, 4)}


def _slice(df: pd.DataFrame, mask: pd.Series, col: str) -> dict[str, float]:
    """Evaluate one prediction column on a boolean-masked slice of df."""
    sub = df[mask]
    return _metrics(sub["y_true"].values, sub[col].values)


# ── Price ──────────────────────────────────────────────────────────────────
peak_mask    = price_fc["hour_of_day"].between(8, 19)
offpeak_mask = ~peak_mask

price_metrics: dict = {
    "model_version": price_fc["model_version"].iloc[0],
    "overall": {
        "xgboost":             _metrics(price_fc["y_true"].values, price_fc["y_pred"].values),
        "persistence_24h":     _metrics(price_fc["y_true"].values, price_fc["price_lag_24h"].values),
        "seasonal_naive_168h": _metrics(price_fc["y_true"].values, price_fc["price_lag_168h"].values),
    },
    "peak_8_19h": {
        "xgboost":             _slice(price_fc, peak_mask,    "y_pred"),
        "persistence_24h":     _slice(price_fc, peak_mask,    "price_lag_24h"),
        "seasonal_naive_168h": _slice(price_fc, peak_mask,    "price_lag_168h"),
    },
    "offpeak": {
        "xgboost":             _slice(price_fc, offpeak_mask, "y_pred"),
        "persistence_24h":     _slice(price_fc, offpeak_mask, "price_lag_24h"),
        "seasonal_naive_168h": _slice(price_fc, offpeak_mask, "price_lag_168h"),
    },
}

# ── PV ─────────────────────────────────────────────────────────────────────
daytime_mask  = pv_fc["hour_of_day"].between(6, 20)
night_mask    = ~daytime_mask

pv_metrics: dict = {
    "model_version": pv_fc["model_version"].iloc[0],
    "overall": {
        "xgboost":             _metrics(pv_fc["y_true"].values, pv_fc["y_pred"].values),
        "persistence_24h":     _metrics(pv_fc["y_true"].values, pv_fc["pv_lag_24h"].values),
        "seasonal_naive_168h": _metrics(pv_fc["y_true"].values, pv_fc["pv_lag_168h"].values),
    },
    "daytime_6_20h": {
        "xgboost":             _slice(pv_fc, daytime_mask, "y_pred"),
        "persistence_24h":     _slice(pv_fc, daytime_mask, "pv_lag_24h"),
        "seasonal_naive_168h": _slice(pv_fc, daytime_mask, "pv_lag_168h"),
    },
    "nighttime": {
        "xgboost":             _slice(pv_fc, night_mask, "y_pred"),
        "persistence_24h":     _slice(pv_fc, night_mask, "pv_lag_24h"),
        "seasonal_naive_168h": _slice(pv_fc, night_mask, "pv_lag_168h"),
    },
}

print("  done")

# ---------------------------------------------------------------------------
# Chunk 4 — Print formatted comparison table
# ---------------------------------------------------------------------------
# A human-readable table so you can sanity-check the numbers immediately
# without opening metrics.json.  Rows group by target + split; columns are
# the three models side by side.
#
# We also print one-liner improvement summaries vs the toughest baseline:
#   Price → seasonal-naive (168h) is harder to beat than persistence (24h)
#           because last week's prices correlate with this week's better than
#           yesterday's prices do over a 6-month window.
#   PV    → persistence (24h) is the gate used in training; we report vs that.
# ---------------------------------------------------------------------------

print("\nChunk 4: metrics comparison table\n")


def _print_block(title: str, splits: dict[str, dict[str, dict]]) -> None:
    """Print a formatted 3-column metrics table for one target."""
    w_split, w_model = 18, 22
    header = (
        f"  {'Split':<{w_split}}  {'Model':<{w_model}}"
        f"  {'MAE':>8}  {'MAPE':>8}  {'RMSE':>8}"
    )
    print(f"  ── {title} ──")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for split_name, models in splits.items():
        for model_name, m in models.items():
            print(
                f"  {split_name:<{w_split}}  {model_name:<{w_model}}"
                f"  {m['mae']:>8.3f}  {m['mape']:>7.2f}%  {m['rmse']:>8.3f}"
            )
        print()


_print_block(
    "PRICE (EUR/MWh)",
    {
        "overall":    price_metrics["overall"],
        "peak 8-19h": price_metrics["peak_8_19h"],
        "off-peak":   price_metrics["offpeak"],
    },
)

_print_block(
    "PV PRODUCTION (kW)  ← MAPE is meaningless at nighttime",
    {
        "overall":       pv_metrics["overall"],
        "daytime 6-20h": pv_metrics["daytime_6_20h"],
        "nighttime":     pv_metrics["nighttime"],
    },
)

# One-liner improvement summaries
price_sn_mae  = price_metrics["overall"]["seasonal_naive_168h"]["mae"]
price_xgb_mae = price_metrics["overall"]["xgboost"]["mae"]
pv_pers_mae   = pv_metrics["overall"]["persistence_24h"]["mae"]
pv_xgb_mae    = pv_metrics["overall"]["xgboost"]["mae"]

price_lift = (1 - price_xgb_mae / price_sn_mae) * 100
pv_lift    = (1 - pv_xgb_mae   / pv_pers_mae)  * 100

print(f"  Price XGBoost: {price_lift:.1f}% better than seasonal-naive (overall MAE)")
print(f"  PV XGBoost   : {pv_lift:.1f}% better than persistence     (overall MAE)")

# ---------------------------------------------------------------------------
# Chunk 5 — Plot actual vs forecast → docs/forecast_vs_baseline.png
# ---------------------------------------------------------------------------
# Two stacked subplots in one figure:
#
#   Top (price, full 6-month window):
#     Actual | XGBoost | Seasonal-naive baseline
#     The full Jul–Dec 2019 window (~4,416 points) is still legible for
#     prices because the signal spans a wide EUR/MWh range and the model
#     clearly tracks the actual better than the baseline.
#     Seasonal-naive is the toughest price baseline, so showing it makes
#     the model's superiority visible even at small scale.
#
#   Bottom (PV, 2-week zoom — 1–14 Jul 2019):
#     Actual | XGBoost | Persistence baseline
#     Zooming to 2 weeks is essential for PV: the intraday solar arc
#     (rise → peak → set) is only visible at day-level granularity.
#     Plotting 6 months would compress every day's curve into a solid bar.
#     July is chosen because it's high-irradiance and shows the model's
#     ability to track variability between sunny and cloudy days.
#
#   Alpha layering (actual=0.9, model=0.75, baseline=0.5) keeps all three
#   curves visible when they overlap without completely obscuring each other.
# ---------------------------------------------------------------------------

print("\nChunk 5: generating forecast_vs_baseline.png …")

import matplotlib       # noqa: E402
matplotlib.use("Agg")  # non-interactive backend — write to file, no display needed
import matplotlib.pyplot as plt        # noqa: E402
import matplotlib.dates as mdates      # noqa: E402

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DOCS_DIR.mkdir(exist_ok=True)

fig, (ax_price, ax_pv) = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
fig.suptitle(
    "XGBoost Forecasts vs Actuals & Baselines — Test Window Jul–Dec 2019",
    fontsize=12,
)

# ── Price (full 6-month) ───────────────────────────────────────────────────
ax_price.plot(
    price_fc["ts_utc"], price_fc["y_true"],
    color="#1f77b4", lw=0.6, alpha=0.9, label="Actual",
)
ax_price.plot(
    price_fc["ts_utc"], price_fc["y_pred"],
    color="#ff7f0e", lw=0.6, alpha=0.75, label="XGBoost",
)
ax_price.plot(
    price_fc["ts_utc"], price_fc["price_lag_168h"],
    color="#aec7e8", lw=0.5, alpha=0.5, label="Seasonal-naive (lag-168h)",
)
ax_price.set_ylabel("EUR/MWh", fontsize=10)
ax_price.set_title("CH Day-ahead Electricity Price (Jul–Dec 2019)", fontsize=10)
ax_price.legend(fontsize=8, loc="upper left")
ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax_price.xaxis.set_major_locator(mdates.MonthLocator())
ax_price.grid(True, alpha=0.25)

# ── PV (2-week zoom) ───────────────────────────────────────────────────────
zoom_start = pd.Timestamp("2019-07-01", tz="UTC")
zoom_end   = pd.Timestamp("2019-07-14 23:00:00", tz="UTC")
pv_zoom    = pv_fc[pv_fc["ts_utc"].between(zoom_start, zoom_end)]

ax_pv.plot(
    pv_zoom["ts_utc"], pv_zoom["y_true"],
    color="#2ca02c", lw=0.9, alpha=0.9, label="Actual",
)
ax_pv.plot(
    pv_zoom["ts_utc"], pv_zoom["y_pred"],
    color="#d62728", lw=0.9, alpha=0.75, label="XGBoost",
)
ax_pv.plot(
    pv_zoom["ts_utc"], pv_zoom["pv_lag_24h"],
    color="#98df8a", lw=0.8, alpha=0.55, label="Persistence (lag-24h)",
)
ax_pv.set_ylabel("kW", fontsize=10)
ax_pv.set_title("Solar PV Production — 2-week zoom (1–14 Jul 2019)", fontsize=10)
ax_pv.legend(fontsize=8, loc="upper left")
ax_pv.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax_pv.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax_pv.grid(True, alpha=0.25)

out_png = DOCS_DIR / "forecast_vs_baseline.png"
fig.savefig(out_png, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved → {out_png}")

# ---------------------------------------------------------------------------
# Chunk 6 — Write docs/metrics.json
# ---------------------------------------------------------------------------
# All computed metrics are serialised to a plain JSON file so the Streamlit
# dashboard (Phase E) and any future CI quality gate can read them without
# needing to connect to DuckDB or re-run the ML pipeline.
#
# Structure mirrors the internal dicts: one section per target, each with
# model_version, overall metrics, and time-of-day split metrics, all nested
# under a "models" key for extensibility.
#
# Why JSON rather than a DuckDB table?
#   The dashboard needs these numbers at startup — before DuckDB is open.
#   A plain JSON file is zero-dependency, commitable to git, and readable
#   by any CI badge script without the full ML stack installed.
# ---------------------------------------------------------------------------

print("\nChunk 6: writing docs/metrics.json …")

metrics_payload = {
    "generated_at": str(datetime.date.today()),
    "test_window": {
        "start": str(price_fc["ts_utc"].min().date()),
        "end":   str(price_fc["ts_utc"].max().date()),
    },
    "price_eur_mwh": price_metrics,
    "pv_power_kw":   pv_metrics,
}

out_json = DOCS_DIR / "metrics.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(metrics_payload, f, indent=2)

print(f"  Saved → {out_json}")

print("\n=== Step 10 complete ===")
print("  docs/forecast_vs_baseline.png — 2-panel chart (price 6-month + PV 2-week)")
print("  docs/metrics.json             — MAE/MAPE/RMSE for all models & splits")
