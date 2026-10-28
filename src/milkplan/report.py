"""Run the whole chain and write Power BI-ready CSVs (tidy, one fact per row) and PNG charts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from milkplan import allocate, data, forecast

OUT = Path(__file__).resolve().parents[2] / "output"
START = "2005-01-01"


def run(geo: str = "DK", offline: bool = True, out: Path = OUT, origins: int = 36) -> dict:
    if not offline:
        data.fetch(geo)
    out.mkdir(parents=True, exist_ok=True)
    df = data.load(geo)
    y = df.milk_kt.dropna()
    y = y[y.index >= START]
    checks = data.validate(df.milk_kt, START)
    if checks["missing_months"] or checks["negative_values"]:
        raise ValueError(f"data checks failed: {checks}")

    bt = forecast.backtest(y, origins=origins)
    scores = forecast.score(bt)
    best = scores.index[0]
    fc = forecast.forecast_with_intervals(y, best, bt)
    coverage = {m: round(forecast.interval_coverage(bt, m), 2) for m in forecast.MODELS}

    fat = df.fat_pct.dropna()
    fat_fc = forecast.seasonal_naive(fat, len(fc))
    fc["fat_pct"] = fat_fc.values
    plan = allocate.allocate_scenarios(fc, fat_fc)
    summary = allocate.summarise(plan)

    # Power BI-ready tables
    df.reset_index(names="month").to_csv(out / "history.csv", index=False)
    bt.to_csv(out / "backtest.csv", index=False)
    scores.reset_index().melt(id_vars="model", var_name="horizon", value_name="mape_pct").to_csv(out / "scores.csv", index=False)
    fc.assign(model=best).to_csv(out / "forecast.csv", index=False)
    plan.to_csv(out / "allocation.csv", index=False)
    summary.to_csv(out / "allocation_summary.csv", index=False)

    _charts(y, fc, scores, summary, best, out)
    result = {
        "geo": geo, "data_checks": checks, "best_model": best,
        "mean_mape_best": float(scores.loc[best, "mean_mape"]),
        "mean_mape_seasonal_naive": float(scores.loc["seasonal_naive", "mean_mape"]),
        "mape_by_horizon": scores.drop(columns="mean_mape").to_dict(orient="index"),
        "interval_coverage_80": coverage,
        "forecast": fc.assign(month=fc.month.dt.strftime("%Y-%m")).round(1).to_dict(orient="records"),
        "spot_kt_by_scenario": summary.groupby("scenario").spot_kt.sum().round(1).to_dict(),
        "shortfall_kt_by_scenario": summary.groupby("scenario").shortfall_kt.sum().round(1).to_dict(),
    }
    (out / "run_summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _charts(y, fc, scores, summary, best, out):
    fig, ax = plt.subplots(figsize=(10, 4))
    hist = y[y.index >= "2021-01-01"]
    ax.plot(hist.index, hist.values, color="#1f4e79", label="Delivered to dairies (Eurostat)")
    ax.plot(fc.month, fc.p50, color="#c0504d", label=f"Forecast ({best}, P50)")
    ax.fill_between(fc.month, fc.p10, fc.p90, color="#c0504d", alpha=0.2, label="P10 to P90")
    ax.set_ylabel("Thousand tonnes per month")
    ax.set_title("Danish raw milk delivered to dairies: history and 6-month forecast")
    ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out / "forecast.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for model, row in scores.drop(columns="mean_mape").iterrows():
        ax.plot(row.index.astype(int), row.values, marker="o", label=model, linewidth=2.5 if model == best else 1.2)
    ax.set_xlabel("Months ahead"); ax.set_ylabel("MAPE (%)")
    ax.set_title("Rolling-origin backtest, last 36 origins")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "backtest_mape.png", dpi=120); plt.close(fig)

    piv = summary.pivot(index="month", columns="scenario", values="spot_kt")
    fig, ax = plt.subplots(figsize=(8, 4))
    piv.index = piv.index.strftime("%Y-%m")
    piv.plot.bar(ax=ax, color=["#9bbb59", "#4f81bd", "#c0504d"])
    ax.set_ylabel("Thousand tonnes to spot sale")
    ax.set_title("Milk the example dairies cannot take, by forecast scenario")
    fig.tight_layout(); fig.savefig(out / "allocation_spot.png", dpi=120); plt.close(fig)
