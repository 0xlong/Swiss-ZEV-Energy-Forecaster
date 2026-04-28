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

st.set_page_config(page_title="Asset Performance", layout="wide")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_monthly() -> pd.DataFrame:
    df = get_conn().execute("""
        SELECT
            month,
            total_pv_kwh,
            total_load_kwh,
            self_consumption_kwh,
            surplus_kwh,
            deficit_kwh,
            self_consumption_ratio,
            autarky_ratio,
            co2_avoided_kg,
            chf_saved
        FROM main_marts.mrt_zev_kpis_monthly
        ORDER BY month
    """).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.title("Asset Performance")
st.caption("Baar warehouse · 100 kWp rooftop PV · 500 MWh/yr load · unbatteried · 2015–2019")

df = load_monthly()

# ---------------------------------------------------------------------------
# Summary KPI tiles (full 5-year aggregate)
# ---------------------------------------------------------------------------
avg_sc = df["self_consumption_ratio"].mean()
avg_autarky = df["autarky_ratio"].mean()
total_co2_t = df["co2_avoided_kg"].sum() / 1000
avg_chf_yr = df["chf_saved"].sum() / 5

c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg self-consumption", f"{avg_sc:.1%}",
          help="Fraction of PV generation consumed on-site")
c2.metric("Avg autarky ratio", f"{avg_autarky:.1%}",
          help="Fraction of building load covered by own PV")
c3.metric("CO₂ avoided (5-yr total)", f"{total_co2_t:.1f} t",
          help="30 gCO₂/kWh Swiss grid intensity · IEA Electricity 2024 / SFOE")
c4.metric("CHF saved (avg/yr)", f"CHF {avg_chf_yr:,.0f}",
          help="(Retail 0.28 − feed-in 0.08) CHF/kWh × self-consumed kWh")

st.divider()

# ---------------------------------------------------------------------------
# Year filter
# ---------------------------------------------------------------------------
years = sorted(df["month"].dt.year.unique().tolist())
selected = st.multiselect("Filter by year", options=years, default=years)
if not selected:
    selected = years
dff = df[df["month"].dt.year.isin(selected)]

# ---------------------------------------------------------------------------
# Stacked bar — monthly energy flows
# ---------------------------------------------------------------------------
fig_flows = go.Figure()
fig_flows.add_trace(go.Bar(
    x=dff["month"], y=dff["self_consumption_kwh"] / 1000,
    name="PV self-consumed", marker_color="#f9a825",
))
fig_flows.add_trace(go.Bar(
    x=dff["month"], y=dff["deficit_kwh"] / 1000,
    name="Grid import", marker_color="#1565c0",
))
fig_flows.add_trace(go.Bar(
    x=dff["month"], y=dff["surplus_kwh"] / 1000,
    name="PV exported to grid", marker_color="#81c784",
))
fig_flows.update_layout(
    barmode="stack",
    title="Monthly Energy Flows (MWh)",
    xaxis_title="Month",
    yaxis_title="MWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=400,
    margin=dict(l=40, r=20, t=60, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_flows, use_container_width=True)

# ---------------------------------------------------------------------------
# KPI ratio lines
# ---------------------------------------------------------------------------
fig_ratios = go.Figure()
fig_ratios.add_trace(go.Scatter(
    x=dff["month"], y=dff["self_consumption_ratio"] * 100,
    name="Self-consumption ratio", line=dict(color="#f9a825", width=2),
))
fig_ratios.add_trace(go.Scatter(
    x=dff["month"], y=dff["autarky_ratio"] * 100,
    name="Autarky ratio", line=dict(color="#1565c0", width=2),
))
fig_ratios.update_layout(
    title="Monthly KPI Ratios",
    xaxis_title="Month",
    yaxis=dict(title="%", ticksuffix="%", range=[0, 105]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=340,
    margin=dict(l=40, r=20, t=60, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_ratios, use_container_width=True)

# ---------------------------------------------------------------------------
# CHF saved + CO₂ avoided side by side
# ---------------------------------------------------------------------------
col_chf, col_co2 = st.columns(2)

fig_chf = go.Figure(go.Bar(
    x=dff["month"], y=dff["chf_saved"],
    marker_color="#2e7d32", name="CHF saved",
))
fig_chf.update_layout(
    title="CHF Saved per Month",
    xaxis_title="Month",
    yaxis_title="CHF",
    height=300,
    margin=dict(l=40, r=20, t=50, b=40),
)
col_chf.plotly_chart(fig_chf, use_container_width=True)

fig_co2 = go.Figure(go.Bar(
    x=dff["month"], y=dff["co2_avoided_kg"],
    marker_color="#00796b", name="CO₂ avoided (kg)",
))
fig_co2.update_layout(
    title="CO₂ Avoided per Month",
    xaxis_title="Month",
    yaxis_title="kg CO₂",
    height=300,
    margin=dict(l=40, r=20, t=50, b=40),
)
col_co2.plotly_chart(fig_co2, use_container_width=True)

st.caption(
    "Self-consumption is high (>80%) because the 100 kWp system (avg ~12 kW output) "
    "is much smaller than the warehouse load (avg ~57 kW, peak ~239 kW). "
    "Almost all PV is absorbed on-site; only summer weekend peaks create grid surplus. "
    "A battery would shift that surplus and raise the autarky ratio."
)
