import streamlit as st

st.set_page_config(page_title="ZEV Economics", layout="wide")

st.title("ZEV Economics")
st.caption("Battery dispatch optimizer — Phase F")

st.info(
    "This page will expose the battery optimizer results once Phase F (PuLP linear program) "
    "is complete. It will include a battery-capacity slider (0–200 kWh), payback-period chart, "
    "and a self-consumption vs autarky comparison with and without storage.",
    icon=None,
)

st.divider()

st.markdown("""
### Background

**ZEV (Zusammenschluss zum Eigenverbrauch)**

Swiss federal regulation (Art. 17 EnG) that allows multiple consumers behind a single
grid connection point to pool their PV generation. Each kWh consumed internally avoids
paying the full retail tariff (~28 Rp/kWh); only the surplus is exported at the feed-in
rate (~8 Rp/kWh). The ~20 Rp/kWh spread is the core economic driver for rooftop solar
in Switzerland.

**LEG (Lokale Elektrizitätsgemeinschaft)**

Introduced in the revised Energy Act (effective 2025), LEG extends ZEV economics across
the low-voltage grid — multiple buildings in a neighbourhood can share a single PV asset.
This materially raises achievable self-consumption ratios and is the regulatory basis for
Enshift's ENSOL PLUS community energy contracting product.

**Why a battery changes the numbers**

The Baar warehouse load is dominated by daytime working hours (BDEW G1 profile). PV surplus
occurs mainly on summer weekends and public holidays, when the building is largely idle.
A battery shifts that midday surplus into evening hours, pushing more kWh through the
on-site bucket and reducing grid import. The Phase F optimizer will quantify exactly how
many kWh — and CHF — a given battery size recovers.
""")
