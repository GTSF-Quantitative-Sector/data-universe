"""Shared frame conventions and the `DataSource` interface.

Bar frames: tz-aware `DatetimeIndex` named "timestamp" in America/New_York,
columns open/high/low/close/volume/vwap as float64. Daily bars are stamped
at midnight of the session date. Intraday bars are stamped at the START of
their window (the vendor convention), and are filtered to regular trading
hours by window overlap -- a bar survives iff any part of its
`[stamp, stamp + bar_minutes)` window intersects 09:30-16:00 ET. For hourly
bars that keeps the 09:00-15:00 stamps; the 09:00 bar includes 9:00-9:30
pre-market trades, which is accepted (see `filter_regular_hours`).

Adjustment: `daily_bars`/`hourly_bars` are total-return adjusted -- splits plus
cash dividends (`sources/adjust.py`). Adjusted levels are anchored at the
window's last bar, so only ratios (returns, vol, beta) are comparable across
calls; `daily_bars_raw` gives real, unadjusted price levels (needed for option
strikes, price filters and dollar volume). Volume is split-adjusted share
counts in adjusted frames and raw share counts in raw frames.
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


class EntitlementError(DataSourceError):
    """Raised on a vendor's HTTP 401/403: the API key is missing or invalid, or the plan
    doesn't include the requested dataset. Distinct from an empty result."""


def empty_bars() -> pd.DataFrame:
    """Return an empty, correctly-typed bars frame (see module docstring for the schema)."""
    index = pd.DatetimeIndex([], name="timestamp", tz=NY_TZ)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=index)


def filter_regular_hours(bars: pd.DataFrame, bar_minutes: int = 60) -> pd.DataFrame:
    """Filter a start-stamped intraday bars frame down to bars whose window
    overlaps regular trading hours (09:30-16:00 ET).

    A bar is kept iff its `[stamp, stamp + bar_minutes)` window intersects
    `[09:30, 16:00)` -- vendors like Polygon/Massive stamp each aggregate at
    the start of its window, so for hourly bars (`bar_minutes=60`) the 09:00
    stamp survives (it holds the 9:30-10:00 regular-hour trades) and the
    16:00 stamp is dropped (it is entirely after-hours). The trade-off, a
    deliberate choice: the kept 09:00-stamped bar also contains 9:00-9:30
    pre-market trades.

    Args:
        bars: A bars frame with a tz-aware America/New_York DatetimeIndex.
        bar_minutes: Length of each bar's window in minutes.

    Returns:
        A copy containing only rows overlapping regular trading hours.
    """
    if bars.empty:
        return bars.copy()
    localized = bars.index.tz_convert(NY_TZ)
    stamp_minutes = localized.hour * 60 + localized.minute + localized.second / 60
    open_minutes = int(REGULAR_OPEN[:2]) * 60 + int(REGULAR_OPEN[3:])
    close_minutes = int(REGULAR_CLOSE[:2]) * 60 + int(REGULAR_CLOSE[3:])
    mask = (stamp_minutes < close_minutes) & (stamp_minutes + bar_minutes > open_minutes)
    return bars.loc[mask].copy()


class DataSource(abc.ABC):
    """Common interface implemented by every data source in this package."""

    @abc.abstractmethod
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return daily total-return (split + cash dividend) adjusted OHLCV bars for
        `ticker` between `start` and `end` (inclusive), anchored at the last bar.

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.
        """

    @abc.abstractmethod
    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return hourly total-return adjusted OHLCV bars for `ticker` between `start` and
        `end` (inclusive), filtered to regular trading hours by `filter_regular_hours`
        (start-stamped window-overlap convention -- see that function's docstring).

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.
        """

    def daily_bars_raw(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return daily **unadjusted** OHLCV bars -- the prices and share volume actually
        traded on each date. Needed wherever a real price level matters: option strike
        selection, price filters, dollar volume.

        Not abstract, so existing custom sources keep working; a source that doesn't
        override it raises `NotImplementedError` when a raw-price feature asks for it.

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.

        Raises:
            NotImplementedError: If the source doesn't provide raw bars.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not provide raw (unadjusted) daily bars; "
            "dollar-volume features and the PEAD event table need daily_bars_raw()"
        )
