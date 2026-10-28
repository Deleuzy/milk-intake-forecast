"""Monthly milk-intake forecasting: four models, a rolling-origin backtest, and intervals from backtest errors."""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
HORIZON = 6


def _future_index(y: pd.Series, h: int) -> pd.DatetimeIndex:
    return pd.date_range(y.index[-1] + pd.offsets.MonthBegin(1), periods=h, freq="MS")


def seasonal_naive(y: pd.Series, h: int) -> pd.Series:
    """Same month last year. The baseline every other model has to beat."""
    idx = _future_index(y, h)
    return pd.Series([y[d - pd.DateOffset(years=1)] for d in idx], index=idx)


def seasonal_naive_trend(y: pd.Series, h: int) -> pd.Series:
    """Same month last year, scaled by the growth of the last 12 months over the 12 before."""
    growth = y.iloc[-12:].sum() / y.iloc[-24:-12].sum()
    return seasonal_naive(y, h) * growth


def ets(y: pd.Series, h: int) -> pd.Series:
    """Holt-Winters with damped additive trend and multiplicative seasonality."""
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="mul", seasonal_periods=12).fit()
    return fit.forecast(h)


def sarima(y: pd.Series, h: int) -> pd.Series:
    """SARIMA(1,1,1)(0,1,1)12 on log volume."""
    fit = SARIMAX(np.log(y), order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=False)
    return np.exp(fit.forecast(h))


def gbm(y: pd.Series, h: int) -> pd.Series:
    """Gradient boosting on year-over-year growth. Features are only what is known at the forecast origin:
    the latest yoy growth, the 3-month average yoy growth, the target month and the horizon."""
    g = (y / y.shift(12)).dropna()
    rows, targets = [], []
    for i in range(3, len(g)):
        origin = g.index[i]
        for k in range(1, h + 1):
            tgt = origin + pd.DateOffset(months=k)
            if tgt not in g.index:
                break
            rows.append([g.iloc[i], g.iloc[i - 2:i + 1].mean(), tgt.month, k])
            targets.append(g[tgt])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0)
    model.fit(np.array(rows), np.array(targets))
    idx = _future_index(y, h)
    X = [[g.iloc[-1], g.iloc[-3:].mean(), d.month, k] for k, d in enumerate(idx, 1)]
    growth = model.predict(np.array(X))
    return pd.Series([y[d - pd.DateOffset(years=1)] * gr for d, gr in zip(idx, growth)], index=idx)


MODELS = {
    "seasonal_naive": seasonal_naive,
    "seasonal_naive_trend": seasonal_naive_trend,
    "ets": ets,
    "sarima": sarima,
    "gbm_yoy": gbm,
}


def backtest(y: pd.Series, origins: int = 36, h: int = HORIZON, models=None) -> pd.DataFrame:
    """Rolling origin: for each of the last `origins` months, train only on data up to that month and
    forecast the next h months. Returns one row per (model, origin, horizon) with the percentage error."""
    models = models or MODELS
    last_origin = len(y) - h
    rows = []
    for o in range(last_origin - origins, last_origin):
        train = y.iloc[: o + 1]
        actual = y.iloc[o + 1: o + 1 + h]
        assert train.index.max() < actual.index.min()  # no leakage
        for name, fn in models.items():
            pred = fn(train, h)
            for k, (d, a) in enumerate(actual.items(), 1):
                rows.append({"model": name, "origin": train.index[-1], "target": d, "horizon": k,
                             "actual": a, "forecast": float(pred[d]), "pct_error": (a - float(pred[d])) / a})
    return pd.DataFrame(rows)


def score(bt: pd.DataFrame) -> pd.DataFrame:
    s = bt.assign(ape=bt.pct_error.abs() * 100).groupby(["model", "horizon"]).ape.mean().unstack()
    s["mean_mape"] = s.mean(axis=1)
    return s.sort_values("mean_mape").round(2)


def forecast_with_intervals(y: pd.Series, model: str, bt: pd.DataFrame, h: int = HORIZON) -> pd.DataFrame:
    """Point forecast from the chosen model; P10/P50/P90 from that model's own backtest errors per horizon.
    P50 is the point corrected by the median error, so a model that is biased in backtest is corrected too."""
    point = MODELS[model](y, h)
    errs = bt[bt.model == model]
    out = []
    for k, (d, p) in enumerate(point.items(), 1):
        e = errs[errs.horizon == k].pct_error
        # actual = forecast / (1 - e)  =>  quantiles of e give the band
        lo, mid, hi = np.quantile(e, [0.10, 0.50, 0.90])
        out.append({"month": d, "horizon": k, "point": float(p),
                    "p10": p / (1 - lo), "p50": p / (1 - mid), "p90": p / (1 - hi)})
    return pd.DataFrame(out)


def interval_coverage(bt: pd.DataFrame, model: str, split: float = 0.5) -> float:
    """Honest check of the band: quantiles from the first half of the backtest origins,
    coverage measured on the second half."""
    b = bt[bt.model == model].sort_values("origin")
    origins = b.origin.unique()
    cut = origins[int(len(origins) * split)]
    first, second = b[b.origin < cut], b[b.origin >= cut]
    hits = []
    for k, grp in second.groupby("horizon"):
        e = first[first.horizon == k].pct_error
        lo, hi = np.quantile(e, 0.10), np.quantile(e, 0.90)
        hits += list((grp.pct_error >= lo) & (grp.pct_error <= hi))
    return float(np.mean(hits))
