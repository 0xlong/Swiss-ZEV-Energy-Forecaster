import sys
from pathlib import Path

_here      = Path(__file__).resolve().parent   # pages/
_dashboard = _here.parent                       # dashboard/
_root      = _dashboard.parent                  # project root
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_dashboard))

import pandas as pd                             # noqa: E402
import plotly.graph_objects as go              # noqa: E402
import streamlit as st                          # noqa: E402
from db import get_conn                         # noqa: E402

from config import (                            # noqa: E402
    BATTERY_CAPACITIES_KWH,
    BATTERY_COST_CHF_PER_KWH,
    FEED_IN_CHF_PER_KWH,
    RETAIL_CHF_PER_KWH,
)

st.set_page_config(page_title="ZEV Economics", layout="wide")

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_battery_monthly(capacity_kwh: int) -> pd.DataFrame:
    df = get_conn().execute("""
        SELECT month, total_pv_kwh, total_load_kwh,
               self_consumption_kwh, own_covered_load_kwh,
               grid_import_kwh, grid_export_kwh,
               self_consumption_ratio, autarky_ratio,
               co2_avoided_kg, net_chf_cost
        FROM main_marts.mrt_zev_kpis_battery_monthly
        WHERE battery_capacity_kwh = ?
        ORDER BY month
    """, [float(capacity_kwh)]).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


@st.cache_data(ttl=3600)
def load_all_capacities_annual() -> pd.DataFrame:
    df = get_conn().execute("""
        SELECT
            battery_capacity_kwh,
            SUM(total_pv_kwh)            AS total_pv_kwh,
            SUM(total_load_kwh)          AS total_load_kwh,
            SUM(self_consumption_kwh)    AS self_consumption_kwh,
            SUM(own_covered_load_kwh)    AS own_covered_load_kwh,
            SUM(net_chf_cost)            AS total_net_chf_cost,
            SUM(co2_avoided_kg)          AS total_co2_kg,
            SUM(self_consumption_kwh) / NULLIF(SUM(total_pv_kwh), 0)   AS self_consumption_ratio,
            SUM(own_covered_load_kwh) / NULLIF(SUM(total_load_kwh), 0) AS autarky_ratio
        FROM main_marts.mrt_zev_kpis_battery_monthly
        GROUP BY battery_capacity_kwh
        ORDER BY battery_capacity_kwh
    """).df()
    return df


# ---------------------------------------------------------------------------
# Page header + formula reference
# ---------------------------------------------------------------------------
st.title("ZEV Economics")
st.caption(
    "Battery dispatch optimizer · PuLP rolling 24-hour LP · perfect-foresight simulation · "
    f"retail {RETAIL_CHF_PER_KWH:.2f} CHF/kWh · feed-in {FEED_IN_CHF_PER_KWH:.2f} CHF/kWh · "
    "efficiency 0.95 one-way (≈ 0.90 round-trip)"
)

with st.expander("How every metric on this page is calculated (First Principles)"):
    st.markdown(f"""
**Energy flows (The "Border" Approach)**
*Instead of tracking complex internal battery flows, we just look at what crosses the building's border (the electricity meter).*

| Metric | The Simple Math | Plain English |
|---|---|---|
| **On-site PV** | `Total PV − Exported PV` | How much solar did we keep? *(If we made 100kW and exported 10kW, we kept 90kW. Whether it powered a lightbulb or charged the battery doesn't matter).* |
| **Own-covered Load** | `Total Load − Imported Load` | How self-sufficient were we? *(If we needed 100kW and bought 30kW, we provided 70kW ourselves via solar/battery).* |
| **Hourly Bill** | `(Import × {RETAIL_CHF_PER_KWH} CHF) − (Export × {FEED_IN_CHF_PER_KWH} CHF)` | Money leaving our pocket minus money earned from the grid. |

**KPI ratios (from monthly rollup)**

| Metric | The Simple Math | Interpretation |
|---|---|---|
| **Self-consumption** | `Total On-site PV ÷ Total PV` | Of all the solar we made, what percentage did we actually use or store? |
| **Autarky** | `Total Own-covered Load ÷ Total Load` | Of all the energy we needed, what percentage did we provide ourselves? |

**Financial metrics (on this page)**

| Metric | The Simple Math | Interpretation |
|---|---|---|
| **Annual Savings** | `(Bill without battery − Bill with battery) ÷ 5` | How much money the battery saves us per year on average. |
| **Battery Investment**| `Capacity × {BATTERY_COST_CHF_PER_KWH:.0f} CHF/kWh` | Upfront cost of buying the battery. |
| **Simple Payback** | `Investment ÷ Annual Savings` | How many years it takes for the battery to pay for itself. |

*Note: Ratios are volume-weighted (Total kWh ÷ Total kWh), so a sunny July day naturally impacts the yearly score more than a dark January day.*
""")

st.divider()

# ---------------------------------------------------------------------------
# Capacity slider
# ---------------------------------------------------------------------------
st.markdown(
    "**Select a battery size** to see how it changes the economics. "
    "Five sizes were pre-computed offline (the LP takes ~10 s per size) "
    "so the slider responds instantly — it just filters a database table."
)
selected_kwh = st.select_slider(
    "Battery capacity",
    options=BATTERY_CAPACITIES_KWH,
    value=100,
    format_func=lambda x: f"{x} kWh",
)

df_sel  = load_battery_monthly(selected_kwh)
df_base = load_battery_monthly(0)
df_all  = load_all_capacities_annual()

sel_row  = df_all[df_all["battery_capacity_kwh"] == float(selected_kwh)].iloc[0]
base_row = df_all[df_all["battery_capacity_kwh"] == 0.0].iloc[0]

N_YEARS          = 5
annual_cost_base = base_row["total_net_chf_cost"] / N_YEARS
annual_cost_sel  = sel_row["total_net_chf_cost"]  / N_YEARS
annual_savings   = annual_cost_base - annual_cost_sel

# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
st.subheader(f"{selected_kwh} kWh battery — 5-year summary vs. no battery")
st.caption(
    "All four tiles compare the selected battery size against the 0 kWh baseline "
    "(same 5-year dataset, same hours). Delta arrows show the battery's contribution."
)

c1, c2, c3, c4 = st.columns(4)

sc_delta  = sel_row["self_consumption_ratio"] - base_row["self_consumption_ratio"]
aut_delta = sel_row["autarky_ratio"]          - base_row["autarky_ratio"]

c1.metric(
    "Self-consumption ratio",
    f"{sel_row['self_consumption_ratio']:.1%}",
    delta=f"{sc_delta:+.1%}",
    help=(
        "Of all the solar we generated, what % did we keep? "
        "(Calculation: Total PV minus what went down the drain to the grid). "
        "Includes both direct use and battery charging."
    ),
)
c2.metric(
    "Autarky ratio",
    f"{sel_row['autarky_ratio']:.1%}",
    delta=f"{aut_delta:+.1%}",
    help=(
        "Of all the energy we needed, what % did we provide ourselves? "
        "(Calculation: Total Load minus what we bought from the grid). "
        "Rises when the battery covers evening demand."
    ),
)
c3.metric(
    "Annual CHF saved vs. no battery",
    f"CHF {annual_savings:,.0f}",
    help=(
        f"How much less we pay the power company each year thanks to the battery. "
        f"Baseline annual cost: CHF {annual_cost_base:,.0f}. "
        f"Cost with battery: CHF {annual_cost_sel:,.0f}."
    ),
)

battery_investment = selected_kwh * BATTERY_COST_CHF_PER_KWH
if annual_savings > 0:
    payback_years = battery_investment / annual_savings
    c4.metric(
        "Simple payback period",
        f"{payback_years:.1f} yr",
        help=(
            f"How many years until the battery pays for itself. "
            f"(Investment: CHF {battery_investment:,.0f} ÷ Savings: CHF {annual_savings:,.0f}/yr). "
            "Does not account for battery degradation, O&M, or electricity price inflation."
        ),
    )
else:
    c4.metric("Simple payback period", "N/A",
              help="No measurable savings at this capacity.")

st.divider()

# ---------------------------------------------------------------------------
# Monthly energy flow chart
# ---------------------------------------------------------------------------
st.subheader("Monthly energy flows")
st.caption(
    f"Each month: amber = PV self-consumed on-site (direct use + battery), "
    "blue = grid import, green = PV surplus exported. "
    "Amber + blue ≈ total building load (~500 MWh/yr). "
    "A larger battery shifts green (exports) into amber (self-consumed), reducing blue (imports)."
)

fig_flows = go.Figure()
fig_flows.add_trace(go.Bar(
    x=df_sel["month"],
    y=df_sel["self_consumption_kwh"] / 1000,
    name="PV self-consumed (on-site)",
    marker_color="#f9a825",
))
fig_flows.add_trace(go.Bar(
    x=df_sel["month"],
    y=df_sel["grid_import_kwh"] / 1000,
    name="Grid import",
    marker_color="#1565c0",
))
fig_flows.add_trace(go.Bar(
    x=df_sel["month"],
    y=df_sel["grid_export_kwh"] / 1000,
    name="PV exported to grid",
    marker_color="#81c784",
))
fig_flows.update_layout(
    barmode="stack",
    xaxis_title="Month",
    yaxis_title="MWh",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=380,
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_flows, use_container_width=True)

# ---------------------------------------------------------------------------
# KPI ratio comparison lines
# ---------------------------------------------------------------------------
st.subheader("Self-consumption and autarky — with vs. without battery")
st.caption(
    "Grey dashed lines = 0 kWh baseline (no battery). "
    "Coloured solid lines = selected battery. "
    "The vertical gap between dashed and solid is the battery's contribution. "
    "Summer dips in self-consumption = weekends/holidays when PV exceeds idle building load."
)

fig_ratios = go.Figure()
fig_ratios.add_trace(go.Scatter(
    x=df_base["month"], y=df_base["self_consumption_ratio"] * 100,
    name="Self-consumption (no battery)",
    line=dict(color="#bdbdbd", width=1, dash="dash"),
))
fig_ratios.add_trace(go.Scatter(
    x=df_base["month"], y=df_base["autarky_ratio"] * 100,
    name="Autarky (no battery)",
    line=dict(color="#bdbdbd", width=1, dash="dot"),
))
fig_ratios.add_trace(go.Scatter(
    x=df_sel["month"], y=df_sel["self_consumption_ratio"] * 100,
    name=f"Self-consumption ({selected_kwh} kWh battery)",
    line=dict(color="#f9a825", width=2),
))
fig_ratios.add_trace(go.Scatter(
    x=df_sel["month"], y=df_sel["autarky_ratio"] * 100,
    name=f"Autarky ({selected_kwh} kWh battery)",
    line=dict(color="#1565c0", width=2),
))
fig_ratios.update_layout(
    xaxis_title="Month",
    yaxis=dict(title="%", ticksuffix="%", range=[0, 105]),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=340,
    margin=dict(l=40, r=20, t=40, b=40),
    hovermode="x unified",
)
st.plotly_chart(fig_ratios, use_container_width=True)

# ---------------------------------------------------------------------------
# Payback chart
# ---------------------------------------------------------------------------
st.subheader("Payback period and annual savings — all battery sizes")
st.caption(
    f"Investment = capacity × {BATTERY_COST_CHF_PER_KWH:.0f} CHF/kWh installed · "
    "savings = reduction in annual net electricity cost vs 0 kWh baseline · "
    "payback = investment ÷ savings · "
    "highlighted bar = currently selected size · "
    "diminishing returns after 50 kWh: the available daily surplus (~200 kWh) "
    "is mostly captured by the first 50 kWh of storage."
)

payback_data = []
for _, row in df_all.iterrows():
    cap = int(row["battery_capacity_kwh"])
    if cap == 0:
        continue
    ann_sav = (base_row["total_net_chf_cost"] - row["total_net_chf_cost"]) / N_YEARS
    inv     = cap * BATTERY_COST_CHF_PER_KWH
    pb      = inv / ann_sav if ann_sav > 0 else None
    payback_data.append({
        "capacity_kwh":   cap,
        "investment_chf": inv,
        "annual_savings": ann_sav,
        "payback_years":  pb,
    })

df_pb     = pd.DataFrame(payback_data)
x_labels  = [f"{r['capacity_kwh']} kWh" for _, r in df_pb.iterrows()]
is_sel    = [r["capacity_kwh"] == selected_kwh for _, r in df_pb.iterrows()]

col_pb, col_sav = st.columns(2)

fig_pb = go.Figure(go.Bar(
    x=x_labels,
    y=df_pb["payback_years"],
    marker_color=["#f9a825" if s else "#b0bec5" for s in is_sel],
    text=[f"{v:.1f} yr" if v else "N/A" for v in df_pb["payback_years"]],
    textposition="outside",
))
fig_pb.update_layout(
    title="Simple payback period (years)",
    xaxis_title="Battery capacity",
    yaxis_title="Years",
    height=320,
    margin=dict(l=40, r=20, t=50, b=40),
)
col_pb.plotly_chart(fig_pb, use_container_width=True)

fig_sav = go.Figure(go.Bar(
    x=x_labels,
    y=df_pb["annual_savings"],
    marker_color=["#2e7d32" if s else "#a5d6a7" for s in is_sel],
    text=[f"CHF {v:,.0f}" for v in df_pb["annual_savings"]],
    textposition="outside",
))
fig_sav.update_layout(
    title="Annual savings vs. no battery (CHF / yr)",
    xaxis_title="Battery capacity",
    yaxis_title="CHF / year",
    height=320,
    margin=dict(l=40, r=20, t=50, b=40),
)
col_sav.plotly_chart(fig_sav, use_container_width=True)

# ---------------------------------------------------------------------------
# Background + methodology
# ---------------------------------------------------------------------------
st.divider()

col_bg, col_method = st.columns([3, 2])

with col_bg:
    st.markdown("""
### Background

**ZEV (Zusammenschluss zum Eigenverbrauch)**

Swiss federal regulation (Art. 17 EnG): multiple consumers behind one grid connection
point can pool PV generation. Each kWh consumed internally saves ~20 Rp/kWh
(retail 28 Rp − feed-in 8 Rp). That spread is the entire economic case for rooftop solar.

**LEG (Lokale Elektrizitätsgemeinschaft)**

Revised Energy Act (effective 2025): extends ZEV economics across the low-voltage grid —
multiple buildings in a neighbourhood share one PV asset. Raises achievable
self-consumption ratios and is the regulatory basis for Enshift's ENSOL PLUS product.

**Why the battery improvement is modest here**

The 100 kWp system is small relative to the 500 MWh/yr warehouse load.
On weekdays the building absorbs almost all PV output directly (avg 57 kW load vs ~12 kW avg PV).
Surplus only occurs on summer weekends and holidays when the building is idle —
roughly 73,000 kWh/yr, or ~200 kWh/day on peak surplus days.
A 50 kWh battery captures most of that; larger sizes show diminishing returns.
""")

with col_method:
    st.markdown("""
### Methodology

| Parameter | Value |
|---|---|
| Optimizer | Rolling 24-h LP (PuLP / CBC) |
| Foresight | Perfect (actual historical data) |
| Round-trip efficiency | 0.95 × 0.95 ≈ 90% |
| Max charge/discharge rate | 50 kW |
| Installed cost assumption | 500 CHF/kWh |
| Dataset | 2015–2019, 43,674 h/capacity |

**Perfect foresight** means the optimizer is given the actual historical PV
and load — as if it knew the future exactly. This produces the *theoretical
upper bound* on battery value, which is the standard input for payback
analysis. A real-time controller with imperfect forecasts would achieve
roughly 70–80% of these savings.

**Simple payback** does not account for battery degradation (~2%/yr capacity
loss), O&M costs, or electricity price inflation.
""")
