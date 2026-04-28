import json
import sys
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from db import get_conn  # noqa: E402  (db.py is in the same directory)

st.set_page_config(
    page_title="Swiss ZEV Energy Forecaster",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Pre-computed ML metrics (written by evaluate.py, no DB needed)
# ---------------------------------------------------------------------------
with open(_ROOT / "docs" / "metrics.json") as f:
    metrics = json.load(f)

price_m = metrics["price_eur_mwh"]
pv_m = metrics["pv_power_kw"]

price_mae = price_m["overall"]["xgboost"]["mae"]
price_baseline_mae = price_m["overall"]["seasonal_naive_168h"]["mae"]
price_improvement = (1 - price_mae / price_baseline_mae) * 100

pv_mae = pv_m["overall"]["xgboost"]["mae"]
pv_baseline_mae = pv_m["overall"]["persistence_24h"]["mae"]
pv_improvement = (1 - pv_mae / pv_baseline_mae) * 100


# ---------------------------------------------------------------------------
# Summary KPIs from DB
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_summary():
    conn = get_conn()
    return conn.execute("""
        SELECT
            AVG(self_consumption_ratio) AS avg_sc_ratio,
            AVG(autarky_ratio)          AS avg_autarky,
            SUM(co2_avoided_kg) / 1000  AS total_co2_t,
            SUM(chf_saved) / 5          AS avg_chf_yr
        FROM main_marts.mrt_zev_kpis_monthly
    """).fetchone()


summary = load_summary()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.title("Swiss ZEV Energy Forecaster")
st.caption(
    "100 kWp rooftop PV — Baar logistics warehouse, Zug, Switzerland "
    "(lat 47.195, lon 8.527) · 2015–2019 analysis window"
)

st.markdown("""
This dashboard models a **ZEV** (Zusammenschluss zum Eigenverbrauch) installation:
a 100 kWp rooftop PV system on a 500 MWh/yr warehouse in Baar. It quantifies
self-consumption, autarky ratio, CO₂ savings, and annual CHF savings under Swiss
ZEV regulation, and provides day-ahead forecasts for electricity prices and PV output.
""")

st.divider()

st.subheader("Key results")

col1, col2, col3, col4 = st.columns(4)
with col1.container(border=True):
    st.metric(
        "Avg self-consumption",
        f"{summary[0]:.1%}",
        help="The percentage of our own generated solar power that we use directly in the building, rather than sending it to the grid.",
    )
with col2.container(border=True):
    st.metric(
        "Avg autarky ratio",
        f"{summary[1]:.1%}",
        help="The percentage of the building's total energy needs that is covered by our own solar power.",
    )
with col3.container(border=True):
    st.metric(
        "CO₂ avoided (total)",
        f"{summary[2]:.1f} t",
        help="The total amount of carbon dioxide emissions prevented by using our solar power instead of drawing from the grid.",
    )
with col4.container(border=True):
    st.metric(
        "CHF saved (avg/yr)",
        f"CHF {summary[3]:,.0f}",
        help="The average money saved per year by using our own solar power instead of buying it from the grid.",
    )


col_price, col_pv = st.columns(2)

with col_price.container(border=True):
    st.subheader("Price forecast")
    st.metric(
        "XGBoost MAE", 
        f"{price_mae:.3f} EUR/MWh",
        help="Mean Absolute Error: The average difference between our AI forecasted electricity price and the actual price."
    )
    st.metric(
        "vs Seasonal-naive (168h)",
        f"{price_improvement:.1f}% better",
        help=f"How much better our AI model is compared to simply guessing that next week's price will be the same as this week's price. (Baseline MAE: {price_baseline_mae:.3f} EUR/MWh)",
    )
    st.caption(f"Test window: H2 2019 · {price_m['model_version']}")

with col_pv.container(border=True):
    st.subheader("PV forecast")
    st.metric(
        "XGBoost MAE", 
        f"{pv_mae:.2f} kW",
        help="Mean Absolute Error: The average difference between our AI forecasted solar production and the actual production."
    )
    st.metric(
        "vs Persistence (lag-24h)",
        f"{pv_improvement:.1f}% better",
        help=f"How much better our AI model is compared to simply guessing that tomorrow's solar production will be the same as today's. (Baseline MAE: {pv_baseline_mae:.2f} kW)",
    )
    st.caption(f"Test window: H2 2019 · {pv_m['model_version']}")