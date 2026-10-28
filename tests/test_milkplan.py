import json

import numpy as np
import pandas as pd
import pytest

from milkplan import allocate, data, forecast


@pytest.fixture(scope="module")
def df():
    return data.load("DK")


@pytest.fixture(scope="module")
def y(df):
    s = df.milk_kt.dropna()
    return s[s.index >= "2005-01-01"]


def test_parse_reads_known_values(df):
    assert df.loc["2026-08-01", "milk_kt"] == pytest.approx(513.5)
    assert df.loc["2026-08-01", "fat_pct"] == pytest.approx(4.38)
    assert df.index.freqstr == "MS"


def test_parse_rejects_unknown_product():
    doc = json.loads(data.raw_path("DK").read_text())
    with pytest.raises(KeyError):
        data.parse(doc, "NOT_A_PRODUCT", "PRD", "THS_T")


def test_validation_reports_gaps_and_jumps_instead_of_filling():
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    s = pd.Series(100.0, index=idx)
    s.iloc[5] = np.nan
    s.iloc[20] = 200.0
    checks = data.validate(s, "2020-01-01")
    assert checks["missing_months"] == ["2020-06-01"]
    assert "2021-09-01" in checks["yoy_jumps_over_25pct"]


def test_real_data_passes_checks(df):
    checks = data.validate(df.milk_kt, "2005-01-01")
    assert checks["missing_months"] == [] and checks["negative_values"] == 0


@pytest.mark.parametrize("name", list(forecast.MODELS))
def test_models_return_h_future_months(y, name):
    out = forecast.MODELS[name](y, 6)
    assert len(out) == 6 and out.index[0] == y.index[-1] + pd.offsets.MonthBegin(1)
    assert (out > 0).all()


def test_backtest_never_trains_on_the_future(y):
    small = {"seasonal_naive": forecast.seasonal_naive, "seasonal_naive_trend": forecast.seasonal_naive_trend}
    bt = forecast.backtest(y, origins=6, h=3, models=small)
    assert (bt.origin < bt.target).all()
    assert len(bt) == 6 * 3 * 2


def test_seasonal_naive_is_last_year(y):
    out = forecast.seasonal_naive(y, 2)
    assert out.iloc[0] == y[out.index[0] - pd.DateOffset(years=1)]


def test_intervals_are_ordered(y):
    small = {"seasonal_naive": forecast.seasonal_naive}
    bt = forecast.backtest(y, origins=12, h=6, models=small)
    fc = forecast.forecast_with_intervals(y, "seasonal_naive", bt)
    assert ((fc.p10 <= fc.p50) & (fc.p50 <= fc.p90)).all()


def test_allocation_respects_supply_capacity_and_minimums():
    for supply in (300, 380, 480, 560):
        plan = allocate.allocate(supply)
        assert plan.allocated_kt.sum() == pytest.approx(supply, abs=1e-6)
        sites = plan[plan.site != "Spot sale (surplus)"]
        assert (sites.allocated_kt <= sites.max_kt + 1e-6).all()
        assert ((sites.allocated_kt + sites.shortfall_kt) >= sites.min_kt - 1e-6).all()


def test_allocation_fills_highest_value_site_first():
    plan = allocate.allocate(400).set_index("site")
    assert plan.loc["Fresh milk dairy", "allocated_kt"] == pytest.approx(130)
    assert plan.loc["Spot sale (surplus)", "allocated_kt"] == pytest.approx(0)


def test_shortfall_only_when_supply_below_total_minimum():
    total_min = sum(s.min_kt for s in allocate.SITES)
    assert allocate.allocate(total_min).shortfall_kt.sum() == pytest.approx(0)
    assert allocate.allocate(total_min - 50).shortfall_kt.sum() == pytest.approx(50)


def test_surplus_goes_to_spot_sale():
    cap = sum(s.max_kt for s in allocate.SITES)
    plan = allocate.allocate(cap + 25).set_index("site")
    assert plan.loc["Spot sale (surplus)", "allocated_kt"] == pytest.approx(25)
