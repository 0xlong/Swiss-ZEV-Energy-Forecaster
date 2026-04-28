"""
optimization/battery_dispatch.py
---------------------------------
Phase F — Rolling 24-hour Linear Program battery dispatcher.

WHAT THIS SCRIPT DOES
---------------------
For each battery size in BATTERY_CAPACITIES_KWH it loops through every
calendar day in the dataset (2015-2019), solves a small 24-variable LP, and
records the optimal charge/discharge schedule for that day.

All results land in raw.battery_simulations in DuckDB.  The dbt pipeline then
picks them up, joins them with the existing energy balance, and recomputes the
ZEV KPIs for each battery size — which the Streamlit dashboard exposes via a
capacity slider.

HOW TO RUN (from the project root, with the venv active)
---------------------------------------------------------
    python optimization/battery_dispatch.py

Typical runtime: ~20-60 seconds total (4 non-trivial LP sizes × ~1,825 days).
"""

import sys
from pathlib import Path

import duckdb
import pandas as pd
import pulp

# ---------------------------------------------------------------------------
# Ensure the project root is importable regardless of CWD.
# This lets `python optimization/battery_dispatch.py` work from any directory.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import (
    BATTERY_CAPACITIES_KWH,
    BATTERY_EFFICIENCY,
    BATTERY_POWER_CAP_KW,
    DUCKDB_PATH,
    FEED_IN_CHF_PER_KWH,
    RETAIL_CHF_PER_KWH,
)


# ---------------------------------------------------------------------------
# WHY LINEAR PROGRAMMING?
# ---------------------------------------------------------------------------
# A naive approach would be a rule-based heuristic: "charge when PV > load,
# discharge at night."  That works, but it's not *optimal* — it might charge
# too much in the morning and have no room for a sunny afternoon.
#
# A Linear Program (LP) guarantees the globally optimal answer by checking
# every legal combination of charge/discharge values simultaneously.  It can
# do this efficiently because every relationship in our problem is linear:
#   - doubling the charge rate doubles the energy stored
#   - doubling grid import doubles the electricity bill
# If any relationship were non-linear (e.g. battery efficiency degrades as
# SoC approaches 100%) we'd need a harder solver.  Ours are all linear.
#
# WHY ROLLING 24-HOUR WINDOW?
# ---------------------------
# We could solve one giant LP over all 5 years at once, but:
#   1. It would have 5×365×24×5 = 219,000 variables — slow and memory-hungry.
#   2. A real battery controller only has a 24-hour day-ahead forecast anyway.
# A rolling 24-hour window mimics the real-world day-ahead controller and
# produces a result that is achievable in practice (not just theoretically).
#
# WHY "PERFECT FORESIGHT"?
# ------------------------
# Instead of feeding our XGBoost forecasts into the optimizer, we feed the
# *actual* historical data — as if the controller knew the future exactly.
# This gives the theoretical maximum value of a battery, which is the standard
# input for a financial payback analysis. The README documents this honestly.
# ---------------------------------------------------------------------------


def _solve_day(
    pv: list,
    load: list,
    capacity_kwh: float,
    power_cap_kw: float,
    efficiency: float,
    prev_soc: float,
) -> dict:
    """
    Solve the 24-hour battery dispatch LP for one calendar day.

    INTUITION
    ---------
    Think of PuLP as a very fast accountant.  You hand it 24 hourly PV and
    load values, the physical limits of the battery, and say: "find the
    charge/discharge schedule with the lowest possible electricity bill."
    It considers every legal combination and returns the best one in
    milliseconds.

    DECISION VARIABLES  (what the solver is allowed to change)
    ------------------
    For each hour t in [0..23] we declare five non-negative variables:

      charge[t]      kW flowing INTO  the battery this hour
      discharge[t]   kW flowing OUT   of the battery this hour
      grid_import[t] kW bought  FROM the grid this hour  (≥ 0 always)
      grid_export[t] kW sold    TO   the grid this hour  (≥ 0 always)
      soc[t]         kWh stored IN   the battery at the END of this hour

    CONSTRAINTS  (physical rules the solver must obey)
    -----------
    1. Energy balance — supply must equal demand every single hour:
           pv[t] + grid_import[t] + discharge[t]
               == load[t] + grid_export[t] + charge[t]
       Think of it as Kirchhoff's current law for energy flows.

    2. SoC evolution — the battery level after hour t equals the level
       before, plus energy put in (scaled down by efficiency), minus energy
       taken out (scaled up because we must put in more than we get out):
           soc[t] = soc[t-1] + charge[t]*η - discharge[t]/η
       η < 1 (e.g. 0.95) means both charging and discharging lose a
       fraction to heat, so the ROUND-TRIP loss is (1 - η²) ≈ 10%.

    3. SoC bounds — battery cannot be over-full or below empty:
           0 ≤ soc[t] ≤ capacity_kwh

    4. Power limits — charge/discharge rate cannot exceed the inverter:
           0 ≤ charge[t]    ≤ power_cap_kw
           0 ≤ discharge[t] ≤ power_cap_kw

    OBJECTIVE  (what the solver minimises)
    ---------
    Total net electricity cost over 24 hours:
        Σ grid_import[t] × retail_price  −  Σ grid_export[t] × feed_in_price

    Because retail (28 Rp) >> feed_in (8 Rp), the solver is strongly
    incentivised to store surplus PV rather than export it.

    WHY NO BINARY FLAG AGAINST SIMULTANEOUS CHARGE+DISCHARGE?
    ----------------------------------------------------------
    Charging costs efficiency×x energy, discharging recovers x/efficiency.
    Doing both at the same time wastes (1 - η²) of every kWh cycled — the
    solver sees this cost and never charges+discharges simultaneously.
    Adding an explicit binary "anti-simultaneous" flag would require integer
    variables, turning the LP into a MILP (Mixed Integer LP), which is
    10-100× slower.  We don't need it.

    Parameters
    ----------
    pv, load    : lists of 24 floats, kW for each hour of the day.
    capacity_kwh: maximum state of charge (kWh).
    power_cap_kw: maximum charge or discharge rate (kW).
    efficiency  : one-way efficiency η (e.g. 0.95 → round-trip ≈ 0.9025).
    prev_soc    : SoC (kWh) carried over from the last hour of the previous day.

    Returns
    -------
    dict with keys: status, charge, discharge, soc, grid_import, grid_export
    (each of the last five is a list of 24 floats).
    """
    n = 24

    # LpProblem("name", LpMinimize) creates a blank minimisation problem.
    prob = pulp.LpProblem("battery_dispatch", pulp.LpMinimize)

    # LpVariable(name, lowBound, upBound) declares a decision variable.
    # lowBound=0 enforces non-negativity (no negative energy flows).
    # upBound enforces physical limits (inverter rating, battery size).
    charge      = [pulp.LpVariable(f"charge_{t}",    lowBound=0, upBound=power_cap_kw)  for t in range(n)]
    discharge   = [pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=power_cap_kw)  for t in range(n)]
    grid_import = [pulp.LpVariable(f"import_{t}",    lowBound=0)                         for t in range(n)]
    grid_export = [pulp.LpVariable(f"export_{t}",    lowBound=0)                         for t in range(n)]
    soc         = [pulp.LpVariable(f"soc_{t}",       lowBound=0, upBound=capacity_kwh)  for t in range(n)]

    # --- Objective function ---
    # prob += expression  adds the objective.
    # pulp.lpSum is PuLP's Σ (equivalent to Python's sum() but works on LP variables).
    prob += pulp.lpSum(
        grid_import[t] * RETAIL_CHF_PER_KWH - grid_export[t] * FEED_IN_CHF_PER_KWH
        for t in range(n)
    )

    # --- Constraints ---
    for t in range(n):

        # Constraint 1 — energy balance: every kW generated must go somewhere.
        # pv available + bought from grid + taken from battery
        #   == building consumes + sold to grid + put into battery
        prob += (
            pv[t] + grid_import[t] + discharge[t]
            == load[t] + grid_export[t] + charge[t]
        )

        # Constraint 2 — SoC evolution: battery level this hour = last hour ± flows.
        # The efficiency term is asymmetric by design:
        #   charging:    we put in charge[t] kWh, but only efficiency×charge[t] is stored
        #   discharging: we get discharge[t] kWh out, but the battery loses discharge[t]/η
        # This asymmetry correctly models the round-trip heat loss.
        if t == 0:
            # Hour 0: "previous hour" is the last hour of yesterday (prev_soc).
            prob += soc[t] == prev_soc + charge[t] * efficiency - discharge[t] / efficiency
        else:
            prob += soc[t] == soc[t - 1] + charge[t] * efficiency - discharge[t] / efficiency

    # Solve silently — msg=0 suppresses the CBC solver's console output.
    # CBC (Coin-or Branch and Cut) is the open-source solver bundled with PuLP;
    # no licence required and fast enough for 24-variable LPs.
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # pulp.value() extracts the optimal numeric value from each LP variable object.
    # The "or 0.0" guard converts None (infeasible problem) to zero so the caller
    # can always work with numbers.
    return {
        "status":      pulp.LpStatus[prob.status],
        "charge":      [pulp.value(charge[t])      or 0.0 for t in range(n)],
        "discharge":   [pulp.value(discharge[t])   or 0.0 for t in range(n)],
        "soc":         [pulp.value(soc[t])          or 0.0 for t in range(n)],
        "grid_import": [pulp.value(grid_import[t]) or 0.0 for t in range(n)],
        "grid_export": [pulp.value(grid_export[t]) or 0.0 for t in range(n)],
    }


def run_optimizer() -> None:
    """
    Main loop: iterate over every (battery_size × calendar_day) combination,
    collect LP results, and write them to raw.battery_simulations in DuckDB.

    WHY WE READ FROM main_intermediate.int_hourly_energy_balance
    ------------------------------------------------------------
    This is the dbt table that has already joined PV generation and building
    load on a shared UTC timestamp.  It is the single canonical grain for
    energy balance in this project — if we read from the raw CSVs directly,
    we'd re-introduce the DST duplicate and HH:10 timestamp problems that
    the staging models already fixed.

    WHY WE WRITE TO raw.battery_simulations (not straight to a mart)
    ----------------------------------------------------------------
    The project follows an ELT pattern: external data lands in the raw schema,
    dbt handles all transformations.  Treating the optimizer output exactly
    like any other data source (CSV, API, ML predictions) keeps the lineage
    graph complete and makes the results testable via dbt tests.

    WHY SoC CARRIES OVER DAY-TO-DAY
    --------------------------------
    A real battery doesn't reset to empty at midnight.  If the evening charge
    from today's surplus PV sits unused overnight, it is available tomorrow
    morning — reducing the first few hours of grid import.  Carrying prev_soc
    forward correctly models this continuity.  It resets to 0.0 only at the
    start of each year (when we start a new BATTERY_CAPACITIES_KWH iteration).
    """
    con = duckdb.connect(str(DUCKDB_PATH))

    # Read the full energy balance — only the 4 columns the optimizer needs.
    df = con.execute("""
        SELECT ts_utc, asset_id, pv_power_kw, load_kw
        FROM main_intermediate.int_hourly_energy_balance
        ORDER BY ts_utc
    """).df()

    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    df["date"]   = df["ts_utc"].dt.date

    n_days   = df["date"].nunique()
    n_hours  = len(df)
    print(f"  Dataset: {n_hours:,} hours across {n_days} days")

    all_rows = []

    for capacity_kwh in BATTERY_CAPACITIES_KWH:
        print(f"  [{capacity_kwh:>3} kWh] running LP over {n_days} days ...", end="", flush=True)

        # Battery starts the multi-year run fully empty.
        # SoC is carried forward day-to-day within the loop.
        prev_soc    = 0.0
        non_optimal = 0

        for date, day_df in df.groupby("date", sort=True):
            day_df = day_df.sort_values("ts_utc").reset_index(drop=True)

            # Safety check — our validated dataset should always have 24 hours/day.
            # DST spring-forward creates a 23-hour day in wall-clock time, but we
            # work in UTC so every day is exactly 24 hours.
            if len(day_df) != 24:
                print(f"\n  WARNING: {date} has {len(day_df)} hours — skipping")
                continue

            pv_hrs   = day_df["pv_power_kw"].tolist()
            load_hrs = day_df["load_kw"].tolist()

            if capacity_kwh == 0:
                # Zero-capacity shortcut: no charge/discharge is physically possible
                # when the battery holds 0 kWh.  We skip the LP and compute the
                # trivial result (deficit = grid_import, surplus = grid_export).
                #
                # WHY INCLUDE 0 kWh AT ALL?
                # The 0-capacity run is a free regression test: the KPIs it produces
                # must exactly match the existing fct_asset_performance KPIs from Phase C.
                # Any divergence reveals a bug in the new dbt models.
                result = {
                    "status":      "Optimal",
                    "charge":      [0.0] * 24,
                    "discharge":   [0.0] * 24,
                    "soc":         [0.0] * 24,
                    "grid_import": [max(load_hrs[t] - pv_hrs[t], 0.0) for t in range(24)],
                    "grid_export": [max(pv_hrs[t] - load_hrs[t], 0.0) for t in range(24)],
                }
            else:
                result = _solve_day(
                    pv_hrs, load_hrs,
                    float(capacity_kwh), BATTERY_POWER_CAP_KW,
                    BATTERY_EFFICIENCY, prev_soc,
                )
                if result["status"] != "Optimal":
                    non_optimal += 1

            # Carry today's final SoC into tomorrow's hour-0 constraint.
            # The "or 0.0" converts None (infeasible) to a safe default.
            prev_soc = result["soc"][-1] or 0.0

            for t in range(24):
                all_rows.append({
                    "ts_utc":               day_df["ts_utc"].iloc[t],
                    "asset_id":             day_df["asset_id"].iloc[t],
                    "battery_capacity_kwh": float(capacity_kwh),
                    "charge_kw":            result["charge"][t],
                    "discharge_kw":         result["discharge"][t],
                    "soc_kwh":              result["soc"][t],
                    "grid_import_kw":       result["grid_import"][t],
                    "grid_export_kw":       result["grid_export"][t],
                })

        suffix = f" ({non_optimal} non-optimal days)" if non_optimal else ""
        print(f" done{suffix}")

    results_df = pd.DataFrame(all_rows)

    # Write to DuckDB.
    # con.register() makes the DataFrame visible to DuckDB SQL under a stable
    # name, which is cleaner than relying on DuckDB scanning local variables.
    # CREATE OR REPLACE means re-running this script is always safe — no manual
    # cleanup needed between runs (e.g. after a config change or re-optimise).
    con.execute("CREATE SCHEMA IF NOT EXISTS raw")
    con.execute("DROP TABLE IF EXISTS raw.battery_simulations")
    con.register("_battery_results", results_df)
    con.execute("CREATE TABLE raw.battery_simulations AS SELECT * FROM _battery_results")

    n_written   = con.execute("SELECT COUNT(*) FROM raw.battery_simulations").fetchone()[0]
    n_expected  = len(BATTERY_CAPACITIES_KWH) * n_hours
    print(f"\n  Wrote {n_written:,} rows (expected {n_expected:,}) -> raw.battery_simulations")
    if n_written != n_expected:
        print("  WARNING: row count mismatch — check for incomplete days")

    con.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Phase F — Battery Dispatch Optimizer")
    print(f"  Capacities : {BATTERY_CAPACITIES_KWH} kWh")
    print(f"  Efficiency : {BATTERY_EFFICIENCY} one-way  (round-trip ~{BATTERY_EFFICIENCY**2:.4f})")
    print(f"  Power cap  : {BATTERY_POWER_CAP_KW} kW charge/discharge")
    print(f"  Tariffs    : retail {RETAIL_CHF_PER_KWH} CHF/kWh · feed-in {FEED_IN_CHF_PER_KWH} CHF/kWh")
    print("=" * 60)
    run_optimizer()
    print("\nDone.  Next step: dbt build --select stg_battery_simulations+")
