"""Compare BDEW G1 (our generated load) with EMPA Swiss commercial archetypes.
Both are normalized to peak=1, so only shape is compared."""

import matplotlib.pyplot as plt
import pandas as pd

OUT = "data/raw/bdew_vs_empa.png"

# --- BDEW G1 (our generated warehouse load) ---
bdew = pd.read_csv("data/raw/load_hourly.csv", parse_dates=["ts_local"])
bdew["hour"] = bdew["ts_local"].dt.hour
bdew["month"] = bdew["ts_local"].dt.month
bdew["dow"] = bdew["ts_local"].dt.dayofweek
bdew["is_weekend"] = bdew["dow"] >= 5

# --- EMPA commercial ---
EMPA = "data/commercial_hour_profile_EMPA.csv"
empa = pd.read_csv(EMPA)
empa.columns = [c.strip() for c in empa.columns]
ELEC = "Electricity demand [J](Hourly)"
empa[ELEC] = empa[ELEC] / 3.6e6  # J/h -> kW

idx = pd.date_range("2019-01-01 00:00", periods=8760, freq="h")
empa = empa.sort_values(["Building_type", "Construction_period", "retrofit_scenario"]).reset_index(drop=True)
combos = empa.groupby(["Building_type", "Construction_period", "retrofit_scenario"]).ngroups
empa["ts"] = list(idx) * combos
empa["hour"] = empa["ts"].dt.hour
empa["month"] = empa["ts"].dt.month
empa["dow"] = empa["ts"].dt.dayofweek
empa["is_weekend"] = empa["dow"] >= 5

# Use middle-era + no retrofit for broad comparability (exists for Offices, Shops, Schools)
empa_slice = empa[(empa["Construction_period"] == "1981-1990") & (empa["retrofit_scenario"] == "No retrofit")]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("BDEW G1 vs EMPA Swiss Commercial Archetypes — shape comparison (peak = 1)",
             fontsize=13, fontweight="bold")

def norm(s):
    return s / s.max()

# --- Panel 1: Weekday median hourly shape ---
ax = axes[0]
b_wk = bdew[~bdew["is_weekend"]].groupby("hour")["load_kw"].median()
ax.plot(b_wk.index, norm(b_wk), label="BDEW G1 (warehouse)", lw=2.5, color="black", zorder=5)
colors = {"Offices": "#2980b9", "Shops": "#8e44ad", "Schools": "#27ae60"}
for btype in ["Offices", "Shops", "Schools"]:
    sub = empa_slice[empa_slice["Building_type"] == btype]
    s = sub[~sub["is_weekend"]].groupby("hour")[ELEC].median()
    ax.plot(s.index, norm(s), label=f"EMPA {btype}", color=colors[btype], lw=1.8, alpha=0.9)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Normalized load (peak = 1)")
ax.set_title("Weekday median shape")
ax.set_xticks(range(0, 24, 3))
ax.legend()
ax.grid(alpha=0.3)

# --- Panel 2: Weekend median hourly shape ---
ax = axes[1]
b_we = bdew[bdew["is_weekend"]].groupby("hour")["load_kw"].median()
ax.plot(b_we.index, norm(b_we), label="BDEW G1 (warehouse)", lw=2.5, color="black", zorder=5)
for btype in ["Offices", "Shops", "Schools"]:
    sub = empa_slice[empa_slice["Building_type"] == btype]
    s = sub[sub["is_weekend"]].groupby("hour")[ELEC].median()
    ax.plot(s.index, norm(s), label=f"EMPA {btype}", color=colors[btype], lw=1.8, alpha=0.9)
ax.set_xlabel("Hour of day")
ax.set_title("Weekend median shape")
ax.set_xticks(range(0, 24, 3))
ax.legend()
ax.grid(alpha=0.3)

# --- Panel 3: Seasonality — monthly energy share (% of annual) ---
ax = axes[2]
b_m = bdew.groupby("month")["load_kw"].sum()
ax.plot(b_m.index, 100 * b_m / b_m.sum(), label="BDEW G1 (warehouse)",
        lw=2.5, color="black", marker="o", zorder=5)
for btype in ["Offices", "Shops", "Schools"]:
    sub = empa_slice[empa_slice["Building_type"] == btype]
    m = sub.groupby("month")[ELEC].sum()
    ax.plot(m.index, 100 * m / m.sum(), label=f"EMPA {btype}",
            color=colors[btype], lw=1.8, marker="o", alpha=0.9)
ax.axhline(100 / 12, color="grey", ls="--", alpha=0.5, label="Flat (1/12)")
ax.set_xlabel("Month")
ax.set_ylabel("% of annual energy")
ax.set_title("Seasonality — monthly share of annual load")
ax.set_xticks(range(1, 13))
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved -> {OUT}")

# --- Numeric similarity metrics ---
print("\nShape correlation (weekday hourly, peak-normalized) vs BDEW G1:")
b_norm = norm(b_wk)
for btype in ["Offices", "Shops", "Schools"]:
    sub = empa_slice[empa_slice["Building_type"] == btype]
    e = norm(sub[~sub["is_weekend"]].groupby("hour")[ELEC].median())
    print(f"  {btype:<10} Pearson r = {b_norm.corr(e):.3f}")

print("\nSeasonality correlation (monthly share) vs BDEW G1:")
b_month = (b_m / b_m.sum())
for btype in ["Offices", "Shops", "Schools"]:
    sub = empa_slice[empa_slice["Building_type"] == btype]
    m = sub.groupby("month")[ELEC].sum()
    m_share = m / m.sum()
    print(f"  {btype:<10} Pearson r = {b_month.corr(m_share):.3f}")
