import pandas as pd
import pytest

import data_universe.features.library  # noqa: F401  (populates FEATURE_REGISTRY)
from data_universe.features import registry
from data_universe.sources.fakes import FakePolygonSource


def test_register_rejects_fwd_prefixed_names():
    local_registry = {}
    with pytest.raises(ValueError):
        @registry.register("fwd_something", registry=local_registry)
        def _fn(ticker, start, end, source):
            return pd.Series(dtype="float64")


def test_register_rejects_duplicate_names():
    local_registry = {}

    @registry.register("dup", registry=local_registry)
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    with pytest.raises(ValueError):
        @registry.register("dup", registry=local_registry)
        def _fn2(ticker, start, end, source):
            return pd.Series(dtype="float64")


def test_register_stores_metadata():
    local_registry = {}

    @registry.register(
        "with_meta",
        warmup_trading_days=5,
        project="beta",
        description="desc",
        registry=local_registry,
    )
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    spec = local_registry["with_meta"]
    assert spec.warmup_trading_days == 5
    assert spec.project == "beta"
    assert spec.description == "desc"
    assert spec.fn is _fn


def test_compute_pads_start_backward_by_warmup_and_slices_result():
    local_registry = {}
    seen_ranges = []

    @registry.register("recorder", warmup_trading_days=3, registry=local_registry)
    def _fn(ticker, start, end, source):
        seen_ranges.append((start, end))
        bars = source.daily_bars(ticker, start, end)
        return pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())

    source = FakePolygonSource()
    result = registry.compute(
        local_registry, "recorder", "AAPL", "2024-02-01", "2024-02-10", source
    )

    called_start, called_end = seen_ranges[0]
    assert pd.Timestamp(called_start) < pd.Timestamp("2024-02-01")
    assert called_end == "2024-02-10"
    assert result.index.min() >= pd.Timestamp("2024-02-01")
    assert result.index.max() <= pd.Timestamp("2024-02-10")


def test_compute_raises_for_unknown_name():
    with pytest.raises(KeyError):
        registry.compute({}, "nope", "AAPL", "2024-02-01", "2024-02-10", FakePolygonSource())


def test_compute_with_weekend_start_matches_first_session_after():
    source = FakePolygonSource()
    weekend = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-06-01", "2024-06-14", source
    )
    monday = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-06-03", "2024-06-14", source
    )
    pd.testing.assert_series_equal(weekend, monday)
    assert pd.notna(weekend.iloc[0])


class _HolidaySource(FakePolygonSource):
    """FakePolygonSource that drops 2024-01-01 (a Monday) from every daily_bars
    result, simulating a market holiday the fake's business-day calendar lacks."""

    def daily_bars(self, ticker, start, end):
        bars = super().daily_bars(ticker, start, end)
        dates = bars.index.tz_localize(None).normalize()
        return bars.loc[dates != pd.Timestamp("2024-01-01")]


def test_compute_with_holiday_start_matches_first_session_after():
    source = _HolidaySource()
    holiday = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-01-01", "2024-01-12", source
    )
    next_day = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-01-02", "2024-01-12", source
    )
    pd.testing.assert_series_equal(holiday, next_day)
    assert not holiday.isna().iloc[0]
