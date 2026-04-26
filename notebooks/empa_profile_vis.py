"""Quick visual — EMPA Swiss commercial load profiles."""

import matplotlib.pyplot as plt
import pandas as pd

CSV = "data/commercial_hour_profile_EMPA.csv"
OUT = "data/raw/empa_profile_vis.png"

ELEC_COL = "Electricity demand [J](Hourly) "
HEAT_COL = "Hot water + Space Heating demand [J](Hourly)"
COOL_COL = "Space cooling demand [J](Hourly)"

# J/hour -> kW (average power over the hour)
J_PER_HOUR_TO_KW = 1 / 3.6e6

df = pd.read_csv(CSV)
df.columns = [c.strip() for c in df.columns]
ELEC_COL, HEAT_COL, COOL_COL = ELEC_COL.strip(), HEAT_COL.strip(), COOL_COL.strip()

# Parse date: "MM/DD HH:MM:SS" — no year, attach 2020 (leap-safe? 8760 rows => non-leap, use 2019)
# EnergyPlus uses "24:00:00" (next-day midnight), which %H:%M:%S rejects.
# Each (type, period, retrofit) block has exactly 8760 sequential hours,
# so just attach a synthetic 2019 hourly index to each block.
idx = pd.date_range("2019-01-01 00:00", periods=8760, freq="h")
df = df.sort_values(["Building_type", "Construction_period", "retrofit_scenario"]).reset_index(drop=True)
combos = df.groupby(["Building_type", "Construction_period", "retrofit_scenario"]).ngroups
df["ts"] = list(idx) * combos
df["hour"] = df["ts"].dt.hour
df["month"] = df["ts"].dt.month
df["dow"] = df["ts"].dt.dayofweek

for c in [ELEC_COL, HEAT_COL, COOL_COL]:
    df[c] = df[c] * J_PER_HOUR_TO_KW  # now kW

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("EMPA Swiss Commercial Load Profiles — per m² normalized views",
             fontsize=13, fontweight="bold")

# Take the newest+no-retrofit slice as "typical modern stock"
newest = "2010-2015"
base = df[(df["Construction_period"] == newest) & (df["retrofit_scenario"] == "No retrofit")]

# --- 1. Median hourly electricity shape, all 5 building types ---
ax = axes[0, 0]
colors = {"Hospitals": "#c0392b", "Offices": "#2980b9",
          "Restaurants": "#e67e22", "Schools": "#27ae60", "Shops": "#8e44ad"}
for btype, sub in base.groupby("Building_type"):
    shape = sub.groupby("hour")[ELEC_COL].median()
    # normalize to relative shape (kW per unit) - peak = 1
    shape_norm = shape / shape.max()
    ax.plot(shape_norm.index, shape_norm.values, label=btype,
            color=colors[btype], lw=2)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Relative electricity demand (peak = 1)")
ax.set_title("Daily electricity shape by building type\n(2010–2015, no retrofit, normalized)")
ax.set_xticks(range(0, 24, 3))
ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

# --- 2. Shops: winter vs summer electricity (warehouse analogue) ---
ax = axes[0, 1]
shops = base[base["Building_type"] == "Shops"].copy()
winter = shops[shops["month"].isin([12, 1, 2])]
summer = shops[shops["month"].isin([6, 7, 8])]
w = winter.groupby("hour")[ELEC_COL].median()
s = summer.groupby("hour")[ELEC_COL].median()
ax.plot(w.index, w.values, label="Winter (Dec–Feb)", color="#1f6faf", lw=2)
ax.plot(s.index, s.values, label="Summer (Jun–Aug)", color="#e87722", lw=2)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Electricity demand (kW)")
ax.set_title("Shops — winter vs summer electricity shape\n(warehouse analogue, EMPA kW per archetype unit)")
ax.set_xticks(range(0, 24, 3))
ax.legend()
ax.grid(alpha=0.3)

# --- 3. Annual energy by building type & end-use (stacked bars) ---
ax = axes[1, 0]
agg = base.groupby("Building_type")[[ELEC_COL, HEAT_COL, COOL_COL]].sum() / 1000  # kW*h -> MWh
agg = agg.sort_values(ELEC_COL, ascending=False)
bars_btm = [0] * len(agg)
labels_map = {ELEC_COL: "Electricity", HEAT_COL: "Heat + DHW", COOL_COL: "Cooling"}
colors_e = {ELEC_COL: "#2980b9", HEAT_COL: "#c0392b", COOL_COL: "#16a085"}
for col in [ELEC_COL, HEAT_COL, COOL_COL]:
    ax.bar(agg.index, agg[col], bottom=bars_btm,
           label=labels_map[col], color=colors_e[col], alpha=0.85)
    bars_btm = [b + v for b, v in zip(bars_btm, agg[col])]
ax.set_ylabel("Annual energy (MWh per archetype unit)")
ax.set_title("Annual end-use breakdown — 2010–2015 stock, no retrofit")
ax.legend()
ax.grid(axis="y", alpha=0.3)

# --- 4. Retrofit impact on heating demand (Shops, all construction periods) ---
ax = axes[1, 1]
shop_all = df[df["Building_type"] == "Shops"]
pivot = shop_all.groupby(["Construction_period", "retrofit_scenario"])[HEAT_COL].sum() / 1000
pivot = pivot.unstack("retrofit_scenario")
periods = sorted(pivot.index.tolist())
pivot = pivot.loc[periods]
x = range(len(periods))
width = 0.38
if "No retrofit" in pivot.columns:
    ax.bar([i - width / 2 for i in x], pivot["No retrofit"].values, width,
           label="No retrofit", color="#c0392b", alpha=0.85)
if "Full retrofit" in pivot.columns:
    ax.bar([i + width / 2 for i in x], pivot["Full retrofit"].values, width,
           label="Full retrofit", color="#27ae60", alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(periods, rotation=35, ha="right")
ax.set_ylabel("Annual heat+DHW (MWh)")
ax.set_title("Shops — retrofit impact on heating, by construction era")
ax.legend()
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved -> {OUT}")

# Print summary table
print("\nAnnual electricity demand (MWh per archetype unit, 2010-2015 stock, no retrofit):")
for bt, val in base.groupby("Building_type")[ELEC_COL].sum().sort_values(ascending=False).items():
    print(f"  {bt:<14}{val / 1000:>8.1f}")
