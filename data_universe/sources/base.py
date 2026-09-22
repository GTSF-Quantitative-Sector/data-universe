"""Shared frame conventions and the `DataSource` interface.

Bar frames: tz-aware `DatetimeIndex` named "timestamp" in America/New_York,
columns open/high/low/close/volume/vwap as float64. Daily bars are stamped
at midnight of the session date; intraday bars are filtered to regular
trading hours (9:30-16:00 ET) by default. Prices are split- and
dividend-adjusted unless a function name says otherwise.
"""
from __future__ import annotations

import abc

import pandas as pd

NY_TZ = "America/New_York"
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]
REGULAR_OPEN = "09:30"
REGULAR_CLOSE = "16:00"


class DataSourceError(Exception):
    """Base class for all data_universe source errors."""


class RateLimitedError(DataSourceError):
    """Raised when an upstream API returns 429 after exhausting retries."""


def empty_bars() -> pd.DataFrame:
    """Return an empty, correctly-typed bars frame (see module docstring for the schema)."""
    index = pd.DatetimeIndex([], name="timestamp", tz=NY_TZ)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=index)


def filter_regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter an intraday bars frame down to regular trading hours (9:30-16:00 ET, inclusive).

    Args:
        bars: A bars frame with a tz-aware America/New_York DatetimeIndex.

    Returns:
        A copy containing only rows within regular trading hours.
    """
    if bars.empty:
        return bars.copy()
    localized = bars.index.tz_convert(NY_TZ)
    times = localized.time
    start = pd.Timestamp(REGULAR_OPEN).time()
    end = pd.Timestamp(REGULAR_CLOSE).time()
    mask = (times >= start) & (times <= end)
    return bars.loc[mask].copy()


class DataSource(abc.ABC):
    """Common interface implemented by every data source in this package."""

    @abc.abstractmethod
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return daily adjusted OHLCV bars for `ticker` between `start` and `end` (inclusive).

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.
        """

    @abc.abstractmethod
    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return hourly adjusted OHLCV bars for `ticker` between `start` and `end` (inclusive),
        filtered to regular trading hours by default.

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.
        """
