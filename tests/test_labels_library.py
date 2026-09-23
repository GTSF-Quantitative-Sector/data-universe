import pandas as pd
import pytest

from data_universe.labels import registry
from data_universe.labels.library import LABEL_NAMES
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


@pytest.fixture
def source():
    return FakePolygonSource(FakeMarket())


@pytest.mark.parametrize("name", LABEL_NAMES)
def test_every_registered_label_computes_without_error(name, source):
    ticker = "SPY" if name == "fwd_vrp_expost" else "AAPL"
    result = registry.compute(
        registry.LABEL_REGISTRY, name, ticker, "2024-03-01", "2024-04-01", source
    )
    assert isinstance(result, pd.Series)
    assert not result.empty


def test_fwd_ret_15d_shifts_returns_backward(source):
    from data_universe.labels.library import fwd_ret_15d

    result = fwd_ret_15d("AAPL", "2024-01-01", "2024-05-01", source)
    bars = source.daily_bars("AAPL", "2024-01-01", "2024-05-01")
    closes = pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())
    expected = closes.shift(-15) / closes - 1.0
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_fwd_ret_15d_is_nan_near_the_end_of_the_requested_range(source):
    result = registry.compute(
        registry.LABEL_REGISTRY, "fwd_ret_15d", "AAPL", "2024-03-01", "2024-03-20", source
    )
    assert result.iloc[-1:].isna().all()  # last row has no future data beyond `end`
