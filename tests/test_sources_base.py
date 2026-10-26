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


def test_filter_regular_hours_keeps_hourly_bars_overlapping_930_to_1600():
    index = pd.DatetimeIndex(
        [
            "2023-01-03 08:00:00",
            "2023-01-03 09:00:00",
            "2023-01-03 10:00:00",
            "2023-01-03 15:00:00",
            "2023-01-03 16:00:00",
            "2023-01-03 17:00:00",
        ],
        tz="America/New_York",
        name="timestamp",
    )
    bars = pd.DataFrame(
        {c: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] for c in BAR_COLUMNS},
        index=index,
    )
    filtered = filter_regular_hours(bars, bar_minutes=60)
    # Start-stamped: the 09:00 bar covers 9:30-10:00, the 15:00 bar covers
    # 15:00-16:00; 08:00 is fully pre-open and 16:00 fully after-hours.
    assert filtered.index.tolist() == index[1:4].tolist()


def test_filter_regular_hours_with_one_minute_bars():
    index = pd.DatetimeIndex(
        [
            "2023-01-03 09:29:00",
            "2023-01-03 09:30:00",
            "2023-01-03 15:59:00",
            "2023-01-03 16:00:00",
        ],
        tz="America/New_York",
        name="timestamp",
    )
    bars = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in BAR_COLUMNS}, index=index)
    filtered = filter_regular_hours(bars, bar_minutes=1)
    # Half-open window overlap: [09:29,09:30) misses, [09:30,09:31) and
    # [15:59,16:00) intersect [09:30,16:00), [16:00,16:01) doesn't.
    assert filtered.index.tolist() == index[1:3].tolist()


def test_filter_regular_hours_converts_other_timezones_to_new_york():
    # Summer: 13:00Z = 09:00 EDT (kept hourly bar), 20:00Z = 16:00 EDT (dropped).
    index = pd.DatetimeIndex(
        ["2023-07-05 13:00:00", "2023-07-05 20:00:00"], tz="UTC", name="timestamp"
    )
    bars = pd.DataFrame({c: [1.0, 2.0] for c in BAR_COLUMNS}, index=index)
    filtered = filter_regular_hours(bars, bar_minutes=60)
    assert filtered.index.tolist() == index[:1].tolist()


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
