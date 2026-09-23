import pandas as pd
import pytest

from data_universe.features import registry
from data_universe.features.library import FEATURE_NAMES
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


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
        registry.FEATURE_REGISTRY, name, ticker, "2024-03-01", "2024-04-01", source
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
    bars = source.daily_bars("AAPL", "2024-03-01", "2024-03-10")
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
