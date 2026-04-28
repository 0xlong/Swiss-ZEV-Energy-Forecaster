import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent       # pages/
_dashboard = _here.parent                      # dashboard/
_root = _dashboard.parent                      # project root
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_dashboard))

import pandas as pd                            # noqa: E402
import plotly.graph_objects as go             # noqa: E402
import streamlit as st                         # noqa: E402
from db import get_conn                        # noqa: E402

st.set_page_config(page_title="Market Forecast", layout="wide")

# ---------------------------------------------------------------------------
# Metrics from pre-computed JSON
# ---------------------------------------------------------------------------
with open(_root / "docs" / "metrics.json") as f:
    metrics = json.load(f)

price_m = metrics["price_eur_mwh"]
pv_m = metrics["pv_power_kw"]

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_price_forecast() -> pd.DataFrame:
    df = get_conn().execute("""
        SELECT
            f.ts_local,
            f.ts_utc,
            f.y_pred                        AS xgboost,
            f.y_true                        AS actual,
            m.price_lag_168h                AS seasonal_naive
        FROM main_marts.fct_forecasts f
        LEFT JOIN main_marts.fct_hourly_market m USING (ts_utc)
        WHERE f.target = 'price_eur_mwh'
        ORDER BY f.ts_utc
    """).df()
    df["ts_local"] = pd.to_datetime(df["ts_local"])
    return df


@st.cache_data(ttl=3600)
def load_pv_forecast() -> pd.DataFrame:
    df = get_conn().execute("""
        SELECT
            f.ts_local,
            f.ts_utc,
            f.y_pred                        AS xgboost,
            f.y_true                        AS actual,
            p.pv_lag_24h                    AS persistence
        FROM main_marts.fct_forecasts f
        LEFT JOIN main_marts.fct_pv_features p USING (ts_utc)
        WHERE f.target = 'pv_power_kw'
        ORDER BY f.ts_utc
    """).df()
    df["ts_local"] = pd.to_datetime(df["ts_local"])
    return df


@st.cache_data(ttl=3600)
def load_mae_by_hour(target: str) -> pd.DataFrame:
    return get_conn().execute(f"""
        SELECT
            EXTRACT(hour FROM ts_local)::INTEGER    AS hour_of_day,
            AVG(ABS(y_pred - y_true))               AS mae
        FROM main_marts.fct_forecasts
        WHERE target = '{target}'
        GROUP BY 1
        ORDER BY 1
    """).df()


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("Market Forecast")
st.caption("Test window: H2 2019 (Jul–Dec) — genuine out-of-sample predictions")

tab_price, tab_pv = st.tabs(["CH Day-Ahead Price", "PV Production"])

# ===========================================================================
# Tab 1 — Price
# ===========================================================================
with tab_price:
    p = price_m
    overall = p["overall"]["xgboost"]
    peak = p["peak_8_19h"]["xgboost"]
    baseline_overall = p["overall"]["seasonal_naive_168h"]
    improvement = (1 - overall["mae"] / baseline_overall["mae"]) * 100

    st.markdown(
        f"**XGBoost beats seasonal-naive by {improvement:.1f}% MAE on the H2 2019 test window.**"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall MAE", f"{overall['mae']:.3f} EUR/MWh")
    c2.metric("Peak MAE (08–19h)", f"{peak['mae']:.3f} EUR/MWh")
    c3.metric("Overall MAPE", f"{overall['mape']:.2f}%")
    c4.metric("vs Seasonal-naive", f"{improvement:.1f}% better")

    st.divider()

    df_price = load_price_forecast()
    min_d, max_d = df_price["ts_local"].dt.date.min(), df_price["ts_local"].dt.date.max()

    col_s, col_e = st.columns(2)
    start = col_s.date_input("From", value=min_d, min_value=min_d, max_value=max_d, key="p_start")
    end = col_e.date_input("To", value=max_d, min_value=min_d, max_value=max_d, key="p_end")

    mask = (df_price["ts_local"].dt.date >= start) & (df_price["ts_local"].dt.date <= end)
    dfp = df_price[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dfp["ts_local"], y=dfp["actual"],
        name="Actual", line=dict(color="#1f77b4", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=dfp["ts_local"], y=dfp["xgboost"],
        name="XGBoost", line=dict(color="#ff7f0e", width=1.2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=dfp["ts_local"], y=dfp["seasonal_naive"],
        name="Seasonal-naive (168h)", line=dict(color="#d62728", width=1, dash="dash"),
    ))
    fig.update_layout(
        title="CH Day-Ahead Price: Actual vs Forecast vs Baseline",
        xaxis_title="Date",
        yaxis_title="EUR/MWh",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    # MAE by hour
    mae_h = load_mae_by_hour("price_eur_mwh")
    colors = ["#ff7f0e" if 8 <= h <= 19 else "#aec7e8" for h in mae_h["hour_of_day"]]
    fig2 = go.Figure(go.Bar(x=mae_h["hour_of_day"], y=mae_h["mae"], marker_color=colors))
    fig2.update_layout(
        title="XGBoost MAE by Hour of Day  (orange = peak 08–19h)",
        xaxis=dict(title="Hour of day", dtick=1),
        yaxis_title="MAE (EUR/MWh)",
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

# ===========================================================================
# Tab 2 — PV
# ===========================================================================
with tab_pv:
    pv = pv_m
    overall_pv = pv["overall"]["xgboost"]
    day_pv = pv["daytime_6_20h"]["xgboost"]
    baseline_pv = pv["overall"]["persistence_24h"]
    improvement_pv = (1 - overall_pv["mae"] / baseline_pv["mae"]) * 100

    st.markdown(
        f"**XGBoost beats persistence (lag-24h) by {improvement_pv:.1f}% MAE on the H2 2019 test window.**"
    )
    st.caption("MAPE is not reported for PV: near-zero actuals at night make the ratio meaningless.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall MAE", f"{overall_pv['mae']:.2f} kW")
    c2.metric("Daytime MAE (06–20h)", f"{day_pv['mae']:.2f} kW")
    c3.metric("Overall RMSE", f"{overall_pv['rmse']:.2f} kW")
    c4.metric("vs Persistence", f"{improvement_pv:.1f}% better")

    st.divider()

    df_pv = load_pv_forecast()

    # Default to a 2-week zoom (first two weeks of July 2019) for legibility
    default_start = df_pv["ts_local"].dt.date.min()
    default_end = default_start + pd.Timedelta(days=13)
    min_d_pv, max_d_pv = df_pv["ts_local"].dt.date.min(), df_pv["ts_local"].dt.date.max()

    col_s2, col_e2 = st.columns(2)
    start2 = col_s2.date_input("From", value=default_start, min_value=min_d_pv, max_value=max_d_pv, key="pv_start")
    end2 = col_e2.date_input("To", value=default_end, min_value=min_d_pv, max_value=max_d_pv, key="pv_end")

    mask2 = (df_pv["ts_local"].dt.date >= start2) & (df_pv["ts_local"].dt.date <= end2)
    dfv = df_pv[mask2]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=dfv["ts_local"], y=dfv["actual"],
        name="Actual", line=dict(color="#1f77b4", width=1.5),
    ))
    fig3.add_trace(go.Scatter(
        x=dfv["ts_local"], y=dfv["xgboost"],
        name="XGBoost", line=dict(color="#ff7f0e", width=1.2, dash="dot"),
    ))
    fig3.add_trace(go.Scatter(
        x=dfv["ts_local"], y=dfv["persistence"],
        name="Persistence (lag-24h)", line=dict(color="#d62728", width=1, dash="dash"),
    ))
    fig3.update_layout(
        title="PV Production: Actual vs Forecast vs Baseline",
        xaxis_title="Date",
        yaxis_title="kW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        height=420,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # MAE by hour (PV)
    mae_h_pv = load_mae_by_hour("pv_power_kw")
    colors_pv = ["#ff7f0e" if 6 <= h <= 20 else "#aec7e8" for h in mae_h_pv["hour_of_day"]]
    fig4 = go.Figure(go.Bar(x=mae_h_pv["hour_of_day"], y=mae_h_pv["mae"], marker_color=colors_pv))
    fig4.update_layout(
        title="XGBoost MAE by Hour of Day  (orange = daytime 06–20h)",
        xaxis=dict(title="Hour of day", dtick=1),
        yaxis_title="MAE (kW)",
        height=320,
        margin=dict(l=40, r=20, t=60, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig4, use_container_width=True)
