"""Trading session calendar, derived from a market proxy's own daily bars --
no separate holiday-calendar dependency. The proxy's bars only exist for
days it actually traded, so its dates *are* the trading calendar (subject to
whatever the underlying `DataSource` actually returns; `FakePolygonSource`
in tests only knows about weekends, not holidays, which is an accepted
simplification for fakes).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from data_universe.sources.base import DataSource


def sessions(
    source: DataSource, start: str, end: str, proxy_ticker: str = "SPY"
) -> pd.DatetimeIndex:
    """Trading session dates between `start` and `end` (inclusive), derived
    from `proxy_ticker`'s daily bars.

    Args:
        source: Any `DataSource` (real or fake) to pull the proxy's daily bars from.
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        proxy_ticker: The market-proxy ticker whose trading days define the
            calendar (default "SPY").

    Returns:
        A tz-naive `DatetimeIndex` named "date", sorted ascending.
    """
    bars = source.daily_bars(proxy_ticker, start, end)
    dates = bars.index.tz_localize(None).normalize()
    return pd.DatetimeIndex(dates, name="date")


def next_session(session_index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the first session strictly after `date`, or None if out of range."""
    position = session_index.searchsorted(date, side="right")
    if position >= len(session_index):
        return None
    return session_index[position]


def previous_session(
    session_index: pd.DatetimeIndex, date: pd.Timestamp
) -> Optional[pd.Timestamp]:
    """Return the last session strictly before `date`, or None if out of range."""
    position = session_index.searchsorted(date, side="left") - 1
    if position < 0:
        return None
    return session_index[position]


def shift_sessions(
    session_index: pd.DatetimeIndex, date: pd.Timestamp, n: int
) -> Optional[pd.Timestamp]:
    """Return the session `n` trading days from `date` (n may be negative to
    go backward). `date` must itself be a session in `session_index`.

    Args:
        session_index: A calendar as returned by `sessions()`.
        date: A session present in `session_index`.
        n: Number of trading days to shift.

    Returns:
        The shifted session date, or None if the shift goes out of range.
    """
    position = session_index.get_loc(date) + n
    if position < 0 or position >= len(session_index):
        return None
    return session_index[position]
