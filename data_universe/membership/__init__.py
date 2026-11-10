"""Point-in-time index membership. Currently S&P 500 only -- every table
carries an `index_id` column so the Dow and Nasdaq-100 can follow later as
data, not new code. See `data_universe/membership/tables/sp500/README.md`
for provenance of the committed data.
"""
from __future__ import annotations

import pandas as pd

from data_universe.membership import grid as _grid
from data_universe.membership import history as _history
from data_universe.membership import intervals as _intervals
from data_universe.membership import store as _store


def sp500_intervals() -> pd.DataFrame:
    """`index_id, security_id, start_date, end_date_excl` for every security
    that was ever an S&P 500 member. `end_date_excl` is NaN for an open
    (current) membership. Security_ids look like `AAPL-19960102`, named
    after each security's first membership interval -- not tickers: renames
    and rejoins share one id, reused tickers get different ones."""
    return _intervals.membership_intervals("sp500")


def membership_grid(
    index_id: str, sessions, strict: bool = False, by: str = "security_id"
) -> pd.DataFrame:
    """Boolean membership grid: one row per session in `sessions`, one column
    per member key, True where `start_date <= t < end_date_excl` (membership
    at the close of that session).

    Args:
        index_id: e.g. "sp500".
        sessions: Session dates to build rows for.
        strict: Raise `MembershipStaleError` instead of warning when the
            committed data is past `membership_max_staleness_days`.
        by: "security_id" (default; `<ticker>-<YYYYMMDD>` of each security's
            first membership interval -- resolved, not one-per-interval) or
            "ticker" (upstream's ticker symbol for each membership interval;
            renames are NOT linked).
    """
    return _grid.membership_grid(index_id, sessions, strict=strict, by=by)


def mask_like(features: pd.DataFrame, index_id: str, by: str = "security_id") -> pd.DataFrame:
    """Boolean mask with the same index and columns as `features`: True where
    the column's key was a member of `index_id` at the close of that row's
    date; columns `index_id` never held are all False.

    Args:
        features: The frame whose index/columns the mask mirrors.
        index_id: e.g. "sp500".
        by: How to interpret `features`' columns -- "security_id" (default;
            `<ticker>-<YYYYMMDD>` of each security's first interval -- see
            `membership.identity`) or "ticker"
            (upstream's ticker symbol for each membership interval; renames
            are NOT linked).
    """
    return _grid.mask_like(features, index_id, by=by)


def members_on(index_id: str, date: str, by: str = "security_id") -> list:
    """Sorted member keys of `index_id` at the close of `date`.

    Args:
        index_id: e.g. "sp500".
        date: Session date, "YYYY-MM-DD".
        by: "security_id" (default; `<ticker>-<YYYYMMDD>` of each security's
            first membership interval -- resolved, not one-per-interval) or
            "ticker" (upstream's ticker symbol for each membership interval;
            renames are NOT linked).
    """
    return _grid.members_on(index_id, date, by=by)


def status(index_id: str) -> dict:
    """`{verified_through, age_days, n_securities, warnings}` for `index_id`,
    where `warnings` is non-empty once the committed data is older than
    `membership_max_staleness_days` (default 45 days)."""
    return _grid.status(index_id)


def security_master(index_id: str = "sp500") -> pd.DataFrame:
    """The security_master table: one row per resolved security_id with
    name, latest_ticker, cik, FIGIs and notes."""
    return _store.load_security_master(index_id)


def ticker_history(index_id: str = "sp500") -> pd.DataFrame:
    """[security_id, ticker, valid_from, valid_to_excl, source]: which ticker
    symbol each security traded under over time."""
    return _store.load_ticker_history(index_id)


def cik_history(index_id: str = "sp500") -> pd.DataFrame:
    """[security_id, cik, valid_from, valid_to_excl, source]: which SEC CIK
    each security filed under over time."""
    return _store.load_cik_history(index_id)


def symbol_segments(index_id: str, security_id: str, start, end) -> pd.DataFrame:
    """[ticker, valid_from, valid_to_excl, source] covering [start, end]
    inclusive for one security (see `membership/history.py`)."""
    return _history.symbol_segments(index_id, security_id, start, end)


def cik_segments(index_id: str, security_id: str, start, end) -> pd.DataFrame:
    """[cik, valid_from, valid_to_excl, source] covering [start, end]
    inclusive for one security (see `membership/history.py`)."""
    return _history.cik_segments(index_id, security_id, start, end)
