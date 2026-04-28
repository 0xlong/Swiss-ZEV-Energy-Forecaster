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
col1.metric(
    "Avg self-consumption",
    f"{summary[0]:.1%}",
    help="Fraction of PV generation consumed on-site (unbatteried)",
)
col2.metric(
    "Avg autarky ratio",
    f"{summary[1]:.1%}",
    help="Fraction of building load covered by own PV generation",
)
col3.metric(
    "CO₂ avoided (total)",
    f"{summary[2]:.1f} t",
    help="5-year total · 30 gCO₂/kWh Swiss grid intensity",
)
col4.metric(
    "CHF saved (avg/yr)",
    f"CHF {summary[3]:,.0f}",
    help="Retail 0.28 − feed-in 0.08 CHF/kWh spread applied to self-consumed kWh",
)

st.divider()

col_price, col_pv = st.columns(2)

col_price.subheader("Price forecast")
col_price.metric("XGBoost MAE", f"{price_mae:.3f} EUR/MWh")
col_price.metric(
    "vs Seasonal-naive (168h)",
    f"{price_improvement:.1f}% better",
    help=f"Baseline MAE: {price_baseline_mae:.3f} EUR/MWh",
)
col_price.caption(f"Test window: H2 2019 · {price_m['model_version']}")

col_pv.subheader("PV forecast")
col_pv.metric("XGBoost MAE", f"{pv_mae:.2f} kW")
col_pv.metric(
    "vs Persistence (lag-24h)",
    f"{pv_improvement:.1f}% better",
    help=f"Baseline MAE: {pv_baseline_mae:.2f} kW",
)
col_pv.caption(f"Test window: H2 2019 · {pv_m['model_version']}")

st.divider()

st.markdown("""
**Pages (sidebar):**
- **Market Forecast** — CH day-ahead price and PV output: XGBoost vs naive baselines, MAE by hour of day
- **Asset Performance** — monthly energy balance, self-consumption and autarky trends, CHF/CO₂ KPIs
- **ZEV Economics** — battery dispatch optimizer (Phase F, coming soon)
""")
