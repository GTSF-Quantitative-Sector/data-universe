"""Generic no-lookahead property, run over every registered feature AND
label: extending the requested `end` date further into the future must
never change a value already computed for an earlier date. It may only
fill in previously-NaN values (e.g. a forward label's last few rows, which
were NaN purely because no future data existed within the smaller window).
"""
import pandas as pd
import pytest

import data_universe.features.library  # noqa: F401  (populates FEATURE_REGISTRY)
import data_universe.labels.library  # noqa: F401  (populates LABEL_REGISTRY)
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.features.registry import compute as compute_feature
from data_universe.labels.registry import LABEL_REGISTRY
from data_universe.labels.registry import compute as compute_label
from data_universe.sources.fakes import FakeMarket, FakePolygonSource

_SHORT_END = "2024-04-01"
_LONG_END = "2024-06-01"
_START = "2024-03-01"


def _ticker_for(name: str) -> str:
    return "SPY" if name in ("iv_index", "vrp_signal_rv21", "fwd_vrp_expost") else "AAPL"


@pytest.mark.parametrize("name", sorted(FEATURE_REGISTRY))
def test_feature_values_are_unchanged_by_extending_end_date(name):
    source = FakePolygonSource(FakeMarket())
    ticker = _ticker_for(name)
    short = compute_feature(FEATURE_REGISTRY, name, ticker, _START, _SHORT_END, source)
    long = compute_feature(FEATURE_REGISTRY, name, ticker, _START, _LONG_END, source)
    common_index = short.index
    long_over_common = long.loc[common_index]
    pd.testing.assert_series_equal(short, long_over_common, check_names=False)


@pytest.mark.parametrize("name", sorted(LABEL_REGISTRY))
def test_label_values_are_unchanged_by_extending_end_date(name):
    source = FakePolygonSource(FakeMarket())
    ticker = _ticker_for(name)
    short = compute_label(LABEL_REGISTRY, name, ticker, _START, _SHORT_END, source)
    long = compute_label(LABEL_REGISTRY, name, ticker, _START, _LONG_END, source)
    common_index = short.index
    long_over_common = long.loc[common_index]
    # Extending `end` may only turn previously-NaN rows into real values
    # (more future data became available) -- it may never change a value
    # that was already non-NaN.
    already_known = short.dropna().index
    pd.testing.assert_series_equal(
        short.loc[already_known], long_over_common.loc[already_known], check_names=False
    )
