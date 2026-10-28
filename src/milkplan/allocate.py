"""Allocate next month's forecast milk to dairies with a linear program, once per forecast scenario.

The four dairies are SYNTHETIC examples sized to Danish national volumes. They are not Arla sites.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass(frozen=True)
class Site:
    name: str
    min_kt: float      # contracted minimum intake per month
    max_kt: float      # processing capacity per month
    value: float       # value per tonne of milk processed here (relative units)


SITES = (
    Site("Fresh milk dairy", min_kt=100, max_kt=130, value=4.0),
    Site("Cheese dairy", min_kt=150, max_kt=180, value=3.0),
    Site("Powder plant", min_kt=60, max_kt=130, value=2.0),
    Site("Butter plant", min_kt=0, max_kt=40, value=1.5),
)
SPOT_VALUE = 0.5          # surplus sold as raw milk
SHORTFALL_PENALTY = 10.0  # cost per tonne a site falls below its minimum


def allocate(supply_kt: float, sites=SITES) -> pd.DataFrame:
    """Maximise value of processed milk. Variables per site: allocated x_i and shortfall s_i; plus spot sale.
    Constraints: sum(x) + spot = supply; x_i + s_i >= min_i; 0 <= x_i <= max_i; s_i, spot >= 0."""
    n = len(sites)
    # variable order: x_0..x_{n-1}, s_0..s_{n-1}, spot
    c = np.concatenate([[-s.value for s in sites], [SHORTFALL_PENALTY] * n, [-SPOT_VALUE]])
    A_eq = np.concatenate([np.ones(n), np.zeros(n), [1.0]])[None, :]
    b_eq = [supply_kt]
    A_ub = np.hstack([-np.eye(n), -np.eye(n), np.zeros((n, 1))])
    b_ub = [-s.min_kt for s in sites]
    bounds = [(0, s.max_kt) for s in sites] + [(0, s.min_kt) for s in sites] + [(0, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x, s, spot = res.x[:n], res.x[n:2 * n], res.x[-1]
    rows = [{"site": site.name, "allocated_kt": x[i], "min_kt": site.min_kt, "max_kt": site.max_kt,
             "utilisation_pct": 100 * x[i] / site.max_kt, "shortfall_kt": s[i]} for i, site in enumerate(sites)]
    rows.append({"site": "Spot sale (surplus)", "allocated_kt": spot, "min_kt": 0, "max_kt": np.nan,
                 "utilisation_pct": np.nan, "shortfall_kt": 0.0})
    return pd.DataFrame(rows).round(2)


def allocate_scenarios(forecast: pd.DataFrame, fat_pct: pd.Series | None = None) -> pd.DataFrame:
    """forecast has month, p10, p50, p90. Returns the plan per month and scenario; fat in tonnes when
    a fat % per month is given (indexed by month)."""
    out = []
    for _, row in forecast.iterrows():
        for scen in ("p10", "p50", "p90"):
            plan = allocate(float(row[scen]))
            plan.insert(0, "month", row["month"])
            plan.insert(1, "scenario", scen)
            plan.insert(2, "supply_kt", round(float(row[scen]), 2))
            if fat_pct is not None:
                plan["fat_t"] = (plan.allocated_kt * 1000 * float(fat_pct[row["month"]]) / 100).round(0)
            out.append(plan)
    return pd.concat(out, ignore_index=True)


def summarise(plan: pd.DataFrame) -> pd.DataFrame:
    """One row per month and scenario: supply, milk to spot sale, total shortfall and value."""
    value = {s.name: s.value for s in SITES} | {"Spot sale (surplus)": SPOT_VALUE}
    p = plan.assign(value=plan.site.map(value) * plan.allocated_kt - SHORTFALL_PENALTY * plan.shortfall_kt)
    g = p.groupby(["month", "scenario"])
    return pd.DataFrame({
        "supply_kt": g.supply_kt.first(),
        "spot_kt": g.apply(lambda d: d.loc[d.site == "Spot sale (surplus)", "allocated_kt"].sum()),
        "shortfall_kt": g.shortfall_kt.sum(),
        "value": g.value.sum(),
    }).round(2).reset_index()
