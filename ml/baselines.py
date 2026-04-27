"""
baselines.py
------------
Naive baseline models (persistence and seasonal naive) to benchmark ML models.

Baselines are defined *before* any ML training so they set the bar:
if XGBoost can't beat them by ≥15 %, the diagnosis is feature quality,
not model choice.

Functions
---------
persistence(y, lag=24)
    Predict the value `lag` hours ago (default: yesterday same hour).

seasonal_naive(y, lag=168)
    Predict the value `lag` hours ago (default: same hour last week).

compute_baselines(df, target_col, ts_col="ts_utc")
    Convenience wrapper: takes a feature DataFrame, computes both baselines,
    and returns a tidy DataFrame ready for evaluation.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Core baseline functions
# ---------------------------------------------------------------------------
def persistence(y: pd.Series, lag: int = 24) -> pd.Series:
    """Persistence (lag-*n*) forecast: predict `y(t) = y(t - lag)`.

    For hourly data the default ``lag=24`` predicts "same hour yesterday".
    The first ``lag`` values will be NaN (no history available).
    """
    return y.shift(lag)


def seasonal_naive(y: pd.Series, lag: int = 168) -> pd.Series:
    """Seasonal-naive forecast: predict `y(t) = y(t - lag)`.

    For hourly data the default ``lag=168`` (= 7 × 24) predicts
    "same hour, same weekday, last week".
    The first ``lag`` values will be NaN.
    """
    return y.shift(lag)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def compute_baselines(
    df: pd.DataFrame,
    target_col: str,
    ts_col: str = "ts_utc",
) -> pd.DataFrame:
    """Build a DataFrame with actual values and both baseline predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least ``ts_col`` and ``target_col``.
        Assumed to be sorted chronologically (ascending ``ts_col``).
    target_col : str
        Name of the target column (e.g. ``"price_eur_mwh"`` or
        ``"pv_power_kw"``).
    ts_col : str, default ``"ts_utc"``
        Name of the timestamp column.

    Returns
    -------
    pd.DataFrame
        Columns: ``ts_utc``, ``y_true``, ``y_pred_persistence``,
        ``y_pred_seasonal_naive``.
        Rows where any baseline is NaN (first 168 hours) are **dropped**
        so the returned frame is ready for metric computation.
    """
    # Ensure chronological order
    df = df.sort_values(ts_col).reset_index(drop=True)

    result = pd.DataFrame(
        {
            "ts_utc": df[ts_col],
            "y_true": df[target_col],
            "y_pred_persistence": persistence(df[target_col], lag=24),
            "y_pred_seasonal_naive": seasonal_naive(df[target_col], lag=168),
        }
    )

    # Drop the initial rows where lags haven't "warmed up" yet
    result = result.dropna(subset=["y_pred_persistence", "y_pred_seasonal_naive"])
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick sanity check (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np

    print("─── baselines.py — sanity check ───")

    # Synthetic hourly data: 2 weeks = 336 hours
    n_hours = 336
    rng = np.random.default_rng(42)
    ts = pd.date_range("2019-01-01", periods=n_hours, freq="h", tz="UTC")
    # Sine wave with daily period + noise → mimics a price signal
    y = 50.0 + 20.0 * np.sin(2 * np.pi * np.arange(n_hours) / 24) + rng.normal(0, 3, n_hours)

    df = pd.DataFrame({"ts_utc": ts, "price_eur_mwh": y})
    baselines = compute_baselines(df, target_col="price_eur_mwh")

    print(f"Input rows:    {len(df)}")
    print(f"Baseline rows: {len(baselines)}  (first 168 h dropped)")
    print(f"Columns:       {list(baselines.columns)}")

    # Quick MAE
    mae_pers = (baselines["y_true"] - baselines["y_pred_persistence"]).abs().mean()
    mae_sn   = (baselines["y_true"] - baselines["y_pred_seasonal_naive"]).abs().mean()
    print(f"Persistence  MAE: {mae_pers:.2f}")
    print(f"Seasonal-naive MAE: {mae_sn:.2f}")
    print("─── OK ───")
