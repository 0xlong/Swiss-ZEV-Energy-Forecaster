"""Quick visual — load profile diagnostics (run from project root)."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

df = pd.read_csv("data/raw/load_hourly.csv", parse_dates=["ts_local"])
df = df.sort_values("ts_local")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle("BDEW G1 Load Profile — Baar Warehouse (500 MWh/yr)", fontsize=13, fontweight="bold")

# --- 1. Full 3-year time series (daily mean to reduce clutter) ---
ax = axes[0]
daily = df.set_index("ts_local")["load_kw"].resample("D").mean()
ax.plot(daily.index, daily.values, lw=0.8, color="#1f6faf")
ax.set_ylabel("Load (kW, daily mean)")
ax.set_title("3-year daily-mean load")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="y", alpha=0.3)

# --- 2. Median day-of-week profile (winter vs summer) ---
ax = axes[1]
df["month"] = df["ts_local"].dt.month
df["hour"] = df["ts_local"].dt.hour
df["dow"] = df["ts_local"].dt.dayofweek  # 0=Mon

winter = df[df["month"].isin([12, 1, 2])]
summer = df[df["month"].isin([6, 7, 8])]

winter_hourly = winter.groupby("hour")["load_kw"].median()
summer_hourly = summer.groupby("hour")["load_kw"].median()

ax.plot(winter_hourly.index, winter_hourly.values, label="Winter (Dec–Feb)", color="#1f6faf", lw=2)
ax.plot(summer_hourly.index, summer_hourly.values, label="Summer (Jun–Aug)", color="#e87722", lw=2)
ax.set_xlabel("Hour of day (local)")
ax.set_ylabel("Load (kW, median)")
ax.set_title("Median hourly shape — winter vs summer")
ax.set_xticks(range(0, 24, 2))
ax.legend()
ax.grid(alpha=0.3)

# --- 3. Monthly total energy (GWh) ---
ax = axes[2]
monthly = df.set_index("ts_local")["load_kw"].resample("ME").sum() / 1000  # kWh → MWh
bars = ax.bar(monthly.index, monthly.values, width=20, color="#1f6faf", alpha=0.8)
ax.set_ylabel("Energy (MWh)")
ax.set_title("Monthly energy consumption")
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="y", alpha=0.3)

# Annotate annual totals
for year in [2018, 2019, 2020]:
    mask = monthly.index.year == year
    total = monthly[mask].sum()
    mid_month = pd.Timestamp(f"{year}-07-01")
    ax.text(mid_month, monthly.max() * 0.92, f"{year}: {total:.0f} MWh",
            ha="center", fontsize=9, color="#333")

plt.tight_layout()
out = "data/raw/load_profile_vis.png"
plt.savefig(out, dpi=150)
print(f"Saved -> {out}")
plt.show()
