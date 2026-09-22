import numpy as np
import pandas as pd
import pytest

from data_universe.sources.base import BAR_COLUMNS, filter_regular_hours
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


def test_daily_bars_has_expected_columns_dtype_and_index():
    market = FakeMarket()
    bars = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    assert list(bars.columns) == BAR_COLUMNS
    assert all(bars[c].dtype == "float64" for c in BAR_COLUMNS)
    assert bars.index.name == "timestamp"
    assert str(bars.index.tz) == "America/New_York"
    assert len(bars) > 0


def test_daily_bars_deterministic_across_calls():
    market = FakeMarket()
    a = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    b = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    pd.testing.assert_frame_equal(a, b)


def test_daily_bars_differ_by_ticker():
    market = FakeMarket()
    aapl = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    msft = market.daily_bars("MSFT", "2023-01-01", "2023-03-01")
    assert not aapl["close"].equals(msft["close"])


def test_planted_beta_is_recoverable_by_ols():
    market = FakeMarket(beta=1.5, alpha=0.0, seed=42)
    mkt_bars = market.daily_bars("SPY", "2010-01-01", "2023-01-01")
    stock_bars = market.daily_bars("AAPL", "2010-01-01", "2023-01-01")
    mkt_ret = np.log(mkt_bars["close"]).diff().dropna()
    stock_ret = np.log(stock_bars["close"]).diff().dropna()
    beta, alpha = np.polyfit(mkt_ret.to_numpy(), stock_ret.to_numpy(), 1)
    assert beta == pytest.approx(1.5, abs=0.05)
    assert alpha == pytest.approx(0.0, abs=0.0005)


def test_proxy_ticker_bars_are_deterministic_too():
    # SPY against itself must have beta 1, not the planted stock beta -- checked
    # indirectly here via determinism; the beta-vs-proxy relationship itself is
    # exercised by test_planted_beta_is_recoverable_by_ols.
    market = FakeMarket(beta=1.5, seed=7)
    spy_a = market.daily_bars("SPY", "2015-01-01", "2020-01-01")
    spy_b = market.daily_bars("SPY", "2015-01-01", "2020-01-01")
    pd.testing.assert_frame_equal(spy_a, spy_b)


def test_vix_closes_are_constant_at_configured_level():
    market = FakeMarket(vix_level=16.0)
    closes = market.vix_closes("2023-01-01", "2023-01-10")
    assert (closes == 16.0).all()
    assert len(closes) > 0


def test_hourly_bars_filtering_removes_pre_open_bar():
    market = FakeMarket()
    bars = market.hourly_bars("AAPL", "2023-01-03", "2023-01-03")
    filtered = filter_regular_hours(bars)
    assert len(filtered) < len(bars)
    assert len(filtered) > 0


def test_fake_polygon_source_daily_bars_delegates_to_market():
    market = FakeMarket(seed=99)
    source = FakePolygonSource(market)
    direct = market.daily_bars("AAPL", "2023-01-01", "2023-02-01")
    via_source = source.daily_bars("AAPL", "2023-01-01", "2023-02-01")
    pd.testing.assert_frame_equal(direct, via_source)


def test_fake_polygon_source_index_daily_bars_wraps_vix():
    source = FakePolygonSource(FakeMarket(vix_level=20.0))
    bars = source.index_daily_bars("I:VIX", "2023-01-01", "2023-01-10")
    assert (bars["close"] == 20.0).all()
    assert list(bars.columns) == BAR_COLUMNS


def test_fake_polygon_source_rejects_unknown_index_ticker():
    source = FakePolygonSource()
    with pytest.raises(ValueError):
        source.index_daily_bars("I:SPX", "2023-01-01", "2023-01-10")


def test_fake_polygon_source_default_market_is_a_fake_market():
    source = FakePolygonSource()
    assert isinstance(source.market, FakeMarket)
