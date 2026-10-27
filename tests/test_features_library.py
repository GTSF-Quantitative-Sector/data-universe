import numpy as np
import pandas as pd
import pytest

from data_universe.features import core, registry
from data_universe.features.library import FEATURE_NAMES
from data_universe.sources.fakes import FakeMarket, FakePolygonSource, FakeRateSource


@pytest.fixture
def source():
    return FakePolygonSource(FakeMarket())


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_every_registered_feature_computes_without_error(name, source):
    if name in ("iv_index", "vrp_signal_rv21"):
        ticker = "SPY"
    else:
        ticker = "AAPL"
    result = registry.compute(
        registry.FEATURE_REGISTRY, name, ticker, "2024-03-01", "2024-04-01", source,
        rate_source=FakeRateSource(),
    )
    assert isinstance(result, pd.Series)
    assert not result.empty


def test_ret_1d_matches_manual_pct_change(source):
    from data_universe.features.library import ret_1d

    result = ret_1d("AAPL", "2024-01-01", "2024-04-01", source)
    bars = source.daily_bars("AAPL", "2024-01-01", "2024-04-01")
    closes = pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())
    expected = closes.pct_change()
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_dollar_volume_1d_equals_close_times_volume(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "dollar_volume_1d", "AAPL", "2024-03-01", "2024-03-10", source
    )
    # Dollar volume is a price level, so it uses raw bars -- not total-return ones.
    bars = source.daily_bars_raw("AAPL", "2024-03-01", "2024-03-10")
    expected = (bars["close"] * bars["volume"]).values
    assert result.values == pytest.approx(expected)


def test_beta_ols_60d_recovers_planted_beta(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "beta_ols_60d", "AAPL", "2024-06-03", "2024-08-01", source
    )
    from data_universe.sources.fakes import PLANTED_BETA
    assert result.dropna().iloc[-1] == pytest.approx(PLANTED_BETA, abs=0.3)


def test_iv_index_for_spy_uses_vix(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "iv_index", "SPY", "2024-03-01", "2024-03-10", source
    )
    from data_universe.sources.fakes import PLANTED_VIX
    assert result.values == pytest.approx(PLANTED_VIX / 100.0)


def test_iv_index_for_non_proxy_ticker_raises_not_implemented(source):
    with pytest.raises(NotImplementedError):
        registry.compute(
            registry.FEATURE_REGISTRY, "iv_index", "AAPL", "2024-03-01", "2024-03-10", source
        )


def test_ret_excess_1d_requires_rate_source(source):
    with pytest.raises(ValueError, match="rate_source"):
        registry.compute(
            registry.FEATURE_REGISTRY, "ret_excess_1d", "AAPL",
            "2024-03-01", "2024-04-01", source,
        )


def test_ret_excess_1d_subtracts_daily_rate(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "ret_excess_1d", "AAPL",
        "2024-03-01", "2024-04-01", source, rate_source=FakeRateSource(annual_rate=0.05),
    )
    raw = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-03-01", "2024-04-01", source
    )
    expected = raw - 0.05 / 252.0
    pd.testing.assert_series_equal(result, expected.rename("ret_excess_1d"))


class _GapRateSource:
    """FakeRateSource missing one session -- exercises the forward-fill so a
    rate-less session carries the previous day's rate rather than 0."""

    def __init__(self, missing: str, annual_rate: float = 0.05) -> None:
        self._missing = pd.Timestamp(missing)
        self._inner = FakeRateSource(annual_rate=annual_rate)

    def daily_rate(self, series_id="DTB3", start=None, end=None):
        rate = self._inner.daily_rate(series_id=series_id, start=start, end=end)
        return rate.drop(index=self._missing, errors="ignore")


def test_ret_excess_1d_carries_previous_rate_over_a_missing_session(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "ret_excess_1d", "AAPL",
        "2024-03-01", "2024-04-01", source,
        rate_source=_GapRateSource("2024-03-05"),
    )
    raw = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-03-01", "2024-04-01", source
    )
    day = pd.Timestamp("2024-03-05")
    assert result.loc[day] == pytest.approx(raw.loc[day] - 0.05 / 252.0)


def _manual_regression(source, rate_source=None, window=60):
    """Independent rolling OLS on (excess) log returns over a wide window."""
    asset = source.daily_bars("AAPL", "2024-01-01", "2024-08-01")["close"]
    market = source.daily_bars("SPY", "2024-01-01", "2024-08-01")["close"]
    asset.index = asset.index.tz_localize(None).normalize()
    market.index = market.index.tz_localize(None).normalize()
    ar = np.log(asset / asset.shift(1))
    mr = np.log(market / market.shift(1)).reindex(ar.index)
    if rate_source is not None:
        rf = rate_source.daily_rate(start="2023-12-01", end="2024-08-01")
        rf = rf.reindex(rf.index.union(ar.index)).ffill().reindex(ar.index)
        ar = ar - np.log1p(rf)
        mr = mr - np.log1p(rf)
    return core.rolling_ols_beta_alpha(ar, mr, window)


def test_alpha_ols_60d_with_rate_source_is_excess_return_regression(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "alpha_ols_60d", "AAPL",
        "2024-06-03", "2024-08-01", source, rate_source=FakeRateSource(0.04),
    )
    _, expected_alpha = _manual_regression(source, rate_source=FakeRateSource(0.04))
    pd.testing.assert_series_equal(
        result, expected_alpha.loc[result.index], check_names=False
    )


def test_beta_alpha_without_rate_source_is_raw_return_regression(source):
    beta = registry.compute(
        registry.FEATURE_REGISTRY, "beta_ols_60d", "AAPL",
        "2024-06-03", "2024-08-01", source,
    )
    alpha = registry.compute(
        registry.FEATURE_REGISTRY, "alpha_ols_90d", "AAPL",
        "2024-06-03", "2024-08-01", source,
    )
    expected_beta, expected_alpha = _manual_regression(source, window=60)
    pd.testing.assert_series_equal(beta, expected_beta.loc[beta.index], check_names=False)
    _, expected_alpha_90 = _manual_regression(source, window=90)
    pd.testing.assert_series_equal(
        alpha, expected_alpha_90.loc[alpha.index], check_names=False
    )
