import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_universe.sources import adjust
from data_universe.sources.base import BAR_COLUMNS
from data_universe.sources.fakes import FakeMarket, FakePolygonSource

_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_hourly_bars_mimic_raw_vendor_stamps_including_pre_and_post():
    market = FakeMarket()
    bars = market.hourly_bars("AAPL", "2023-01-03", "2023-01-03")
    # Nine start-stamped bars per session, like raw vendor aggregates:
    # a pre-open 08:00 stamp and an after-hours 16:00 stamp.
    assert len(bars) == 9
    assert [t.hour for t in bars.index] == list(range(8, 17))


def test_fake_polygon_source_hourly_bars_filters_to_regular_hours():
    source = FakePolygonSource()
    bars = source.hourly_bars("AAPL", "2023-01-03", "2023-01-03")
    assert len(bars) == 7
    assert [t.hour for t in bars.index] == list(range(9, 16))


def test_same_date_has_same_price_across_request_windows():
    market = FakeMarket()
    narrow = market.daily_bars("AAPL", "2023-02-01", "2023-02-28")
    wide = market.daily_bars("AAPL", "2023-01-01", "2023-03-31")
    pd.testing.assert_frame_equal(narrow, wide.loc[narrow.index])
    narrow_h = market.hourly_bars("AAPL", "2023-02-01", "2023-02-10")
    wide_h = market.hourly_bars("AAPL", "2023-01-01", "2023-03-01")
    pd.testing.assert_frame_equal(narrow_h, wide_h.loc[narrow_h.index])


def test_bars_are_identical_across_processes_with_different_hash_seeds():
    # PYTHONHASHSEED randomizes hash() per process; the fake must not depend on
    # it, or the same code would emit different bars on different machines/runs.
    code = (
        "from data_universe.sources.fakes import FakeMarket\n"
        "m = FakeMarket()\n"
        "print(m.daily_bars('SPY', '2024-01-01', '2024-02-01')['close'].iloc[-1])\n"
        "print(m.hourly_bars('AAPL', '2024-01-02', '2024-01-02')['close'].tolist())"
    )
    outputs = []
    for hashseed in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=_REPO_ROOT,
            env={**os.environ, "PYTHONHASHSEED": hashseed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout.strip())
    assert outputs[0] == outputs[1]


def test_start_before_epoch_raises():
    market = FakeMarket()
    with pytest.raises(ValueError, match="epoch"):
        market.daily_bars("AAPL", "1989-12-01", "2023-01-01")
    with pytest.raises(ValueError, match="epoch"):
        market.hourly_bars("AAPL", "1989-12-01", "2023-01-01")


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


# --- v0.6.0: dividends, splits, raw and split-adjusted series ---------------------


def test_real_adjustment_reproduces_fake_total_return_series_up_to_scale():
    market = FakeMarket(splits={"AAPL": [("2024-03-01", 4.0)]})
    args = ("AAPL", "2023-06-01", "2024-06-01")
    total_return = market.daily_bars(*args)
    split_adjusted = market.split_adjusted_bars(*args)
    raw = market.raw_bars(*args)
    dividends = market.dividends(*args)
    assert len(dividends) >= 3
    rebuilt = adjust.apply_total_return(
        split_adjusted, adjust.dividend_factors(raw["close"], dividends)
    )
    ratio = rebuilt["close"] / total_return["close"]
    assert ratio.max() / ratio.min() - 1.0 < 1e-12


def test_raw_bars_undo_splits_and_volume_moves_inversely():
    market = FakeMarket(splits={"AAPL": [("2024-03-01", 4.0)]})
    split_adjusted = market.split_adjusted_bars("AAPL", "2024-02-26", "2024-03-05")
    raw = market.raw_bars("AAPL", "2024-02-26", "2024-03-05")
    before = raw.index < pd.Timestamp("2024-03-01", tz="America/New_York")
    assert (raw["close"] / split_adjusted["close"])[before].tolist() == pytest.approx([4.0] * 4)
    assert (raw["close"] / split_adjusted["close"])[~before].tolist() == pytest.approx([1.0] * 3)
    assert (raw["volume"] * 4.0)[before].tolist() == pytest.approx(
        split_adjusted["volume"][before].tolist()
    )


def test_dividend_series_are_window_independent():
    market = FakeMarket()
    narrow = market.raw_bars("AAPL", "2023-02-01", "2023-02-28")
    wide = market.raw_bars("AAPL", "2023-01-01", "2023-03-31")
    pd.testing.assert_frame_equal(narrow, wide.loc[narrow.index])
    all_divs = market.dividends("AAPL", "2022-01-01", "2023-12-31")
    some = market.dividends("AAPL", "2023-01-01", "2023-06-30")
    assert set(some["ex_dividend_date"]) <= set(all_divs["ex_dividend_date"])


def test_dividend_factor_is_exactly_one_minus_yield():
    market = FakeMarket(dividend_yield=0.01)
    raw = market.raw_bars("MSFT", "2023-01-01", "2023-12-31")
    factors = adjust.dividend_factors(
        raw["close"], market.dividends("MSFT", "2023-01-01", "2023-12-31")
    )
    assert factors.tolist() == pytest.approx([0.99] * len(factors))


def test_no_dividends_for_index_tickers_or_zero_yield():
    assert FakeMarket().dividends("I:VIX", "2020-01-01", "2023-01-01").empty
    zero = FakeMarket(dividend_yield=0.0)
    assert zero.dividends("AAPL", "2020-01-01", "2023-01-01").empty
    pd.testing.assert_frame_equal(
        zero.raw_bars("AAPL", "2023-01-01", "2023-02-01"),
        zero.daily_bars("AAPL", "2023-01-01", "2023-02-01"),
    )


def test_fake_source_exposes_raw_split_adjusted_and_dividends():
    market = FakeMarket()
    source = FakePolygonSource(market)
    args = ("AAPL", "2023-01-01", "2023-06-01")
    pd.testing.assert_frame_equal(source.daily_bars_raw(*args), market.raw_bars(*args))
    pd.testing.assert_frame_equal(
        source.daily_bars_split_adjusted(*args), market.split_adjusted_bars(*args)
    )
    pd.testing.assert_frame_equal(source.dividends(*args), market.dividends(*args))
