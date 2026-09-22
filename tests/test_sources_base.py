import pandas as pd
import pytest

from data_universe.sources.base import (
    BAR_COLUMNS,
    DataSource,
    DataSourceError,
    RateLimitedError,
    empty_bars,
    filter_regular_hours,
)


def test_empty_bars_has_expected_columns_and_index_name():
    bars = empty_bars()
    assert list(bars.columns) == BAR_COLUMNS
    assert bars.empty
    assert bars.index.name == "timestamp"


def test_empty_bars_columns_are_float64():
    bars = empty_bars()
    assert all(bars[c].dtype == "float64" for c in BAR_COLUMNS)


def test_filter_regular_hours_on_empty_frame_returns_empty():
    filtered = filter_regular_hours(empty_bars())
    assert filtered.empty


def test_filter_regular_hours_drops_bars_outside_930_to_1600():
    index = pd.DatetimeIndex(
        [
            "2023-01-03 09:00:00",
            "2023-01-03 09:30:00",
            "2023-01-03 12:00:00",
            "2023-01-03 16:00:00",
            "2023-01-03 16:30:00",
        ],
        tz="America/New_York",
        name="timestamp",
    )
    bars = pd.DataFrame(
        {c: [1.0, 2.0, 3.0, 4.0, 5.0] for c in BAR_COLUMNS},
        index=index,
    )
    filtered = filter_regular_hours(bars)
    assert len(filtered) == 3
    assert filtered.index.tolist() == index[1:4].tolist()


def test_filter_regular_hours_returns_a_copy():
    index = pd.DatetimeIndex(["2023-01-03 10:00:00"], tz="America/New_York", name="timestamp")
    bars = pd.DataFrame({c: [1.0] for c in BAR_COLUMNS}, index=index)
    filtered = filter_regular_hours(bars)
    filtered.iloc[0, 0] = 999.0
    assert bars.iloc[0, 0] == 1.0


def test_data_source_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DataSource()


def test_data_source_subclass_must_implement_both_abstract_methods():
    class Incomplete(DataSource):
        def daily_bars(self, ticker, start, end):
            return empty_bars()

    with pytest.raises(TypeError):
        Incomplete()


def test_data_source_subclass_with_both_methods_instantiates():
    class Complete(DataSource):
        def daily_bars(self, ticker, start, end):
            return empty_bars()

        def hourly_bars(self, ticker, start, end):
            return empty_bars()

    Complete()  # should not raise -- DataSource imposes no constructor


def test_rate_limited_error_is_a_data_source_error():
    assert issubclass(RateLimitedError, DataSourceError)
