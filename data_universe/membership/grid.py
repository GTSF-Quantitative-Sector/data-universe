"""Boolean membership grid and the small query functions built on it.
Never stored -- rebuilt from `intervals.membership_intervals()` (keyed by
security_id) or `store.load_ticker_intervals()` (keyed by ticker) on every
call.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data_universe import config
from data_universe.membership import intervals as intervals_module
from data_universe.membership import store
from data_universe.membership.errors import (
    MembershipError,
    MembershipStaleError,
    MembershipStaleWarning,
)

_MEMBER_KEYS = ("security_id", "ticker")


def _key_intervals(index_id: str, by: str) -> pd.DataFrame:
    """`index_id, key, start_date, end_date_excl` membership intervals where
    `key` is a security_id (`by="security_id"`, via `intervals.membership_intervals`)
    or upstream's ticker symbol for each membership interval (`by="ticker"`,
    straight from `store.load_ticker_intervals` -- see
    `data_universe/membership/tables/sp500/README.md`, "Provisional identity").

    Raises:
        ValueError: If `by` is not "security_id" or "ticker".
    """
    if by == "security_id":
        frame = intervals_module.membership_intervals(index_id)
        return frame.rename(columns={"security_id": "key"})
    if by == "ticker":
        return store.load_ticker_intervals(index_id).rename(columns={"ticker": "key"})
    raise ValueError(f"by must be one of {_MEMBER_KEYS}, got {by!r}")


def membership_grid(
    index_id: str, sessions, strict: bool = False, by: str = "security_id"
) -> pd.DataFrame:
    """Boolean membership grid: one row per session, one column per member key.

    `grid.loc[t, k]` is True if `start_date <= t < end_date_excl` for `k`'s
    membership interval -- i.e. membership at the close of session `t`.

    Args:
        index_id: e.g. "sp500".
        sessions: Trading session dates to build rows for (e.g. from
            `data_universe.calendar.sessions()`).
        strict: If True, raise `MembershipStaleError` when the data is older
            than `config.get_membership_max_staleness_days()`. If False
            (default), emit `MembershipStaleWarning` and proceed anyway.
        by: Column key -- "security_id" (default; provisional
            `<ticker>-<YYYYMMDD>` ids that WILL change when identity
            resolution lands, so don't persist them) or "ticker" (upstream's
            ticker symbol for each membership interval; renames like
            FB -> META are NOT linked, each interval keys on its own symbol).

    Returns:
        A bool DataFrame, columns=sorted member keys. The row index is
        `sessions` itself (normalized, name "date") -- same order, including
        duplicates, as what the caller passed in. `sessions` need not be
        sorted; it is sorted and deduped internally for the interval lookup
        and the result is scattered back into the caller's requested order.

    Raises:
        ValueError: If `by` is not "security_id" or "ticker".
    """
    _check_staleness(index_id, strict)
    member_intervals = _key_intervals(index_id, by)

    requested_index = pd.DatetimeIndex(sessions).normalize()
    unique_sorted = requested_index.unique().sort_values()
    lookup_index = (
        unique_sorted.tz_localize(None) if unique_sorted.tz is not None else unique_sorted
    )

    columns = sorted(member_intervals["key"].unique())
    column_of = {key: position for position, key in enumerate(columns)}
    values = np.zeros((len(unique_sorted), len(columns)), dtype=bool)

    for _, row in member_intervals.iterrows():
        start = pd.Timestamp(row["start_date"])
        end = row["end_date_excl"]
        first = lookup_index.searchsorted(start, side="left")
        last = (
            len(lookup_index)
            if pd.isna(end)
            else lookup_index.searchsorted(pd.Timestamp(end), side="left")
        )
        values[first:last, column_of[row["key"]]] = True

    sorted_grid = pd.DataFrame(values, index=unique_sorted, columns=columns)
    requested_lookup = (
        requested_index.tz_localize(None) if requested_index.tz is not None else requested_index
    )
    positions = lookup_index.searchsorted(requested_lookup, side="left")
    result = sorted_grid.iloc[positions]
    result.index = pd.DatetimeIndex(requested_index, name="date")
    return result


def mask_like(features: pd.DataFrame, index_id: str, by: str = "security_id") -> pd.DataFrame:
    """Boolean mask with the same index and columns as `features`. A column
    `features` has that `index_id` never held as a member is all False.

    `membership_grid` is looked up with `features.index` normalized to
    midnight for the membership comparison, but the mask handed back is
    reindexed onto `features.index` itself (same values, order, name, and
    tz) so it lines up row-for-row with `features` even when `features` is
    sampled intraday (e.g. hourly bars).

    Args:
        features: The frame whose index/columns the mask mirrors. Columns
            are interpreted per `by`.
        index_id: e.g. "sp500".
        by: Column key, as in `membership_grid`. With `by="ticker"` zero
            matched columns is legitimate (non-members are simply all
            False); with `by="security_id"` it almost always means the
            columns were tickers, so that case raises `MembershipError`
            instead of silently returning an all-False mask.

    Raises:
        MembershipError: `by="security_id"` and NONE of `features`' columns
            is a known security_id for `index_id`.
        ValueError: If `by` is not "security_id" or "ticker".
    """
    grid = membership_grid(index_id, features.index, by=by)
    grid.index = features.index
    if (
        by == "security_id"
        and len(features.columns) > 0
        and not set(features.columns) & set(grid.columns)
    ):
        raise MembershipError(
            f"None of {index_id!r}'s security_ids matches any column of `features` "
            "-- security_ids are provisional '<ticker>-<YYYYMMDD>' ids like "
            "'AAPL-19960102', not tickers. Pass by=\"ticker\" for a ticker-keyed panel."
        )
    return grid.reindex(columns=features.columns, fill_value=False)


def members_on(index_id: str, date: str, by: str = "security_id") -> list:
    """Members of `index_id` at the close of `date`: security_ids by
    default, upstream ticker symbols with `by="ticker"`."""
    grid = membership_grid(index_id, pd.DatetimeIndex([pd.Timestamp(date)]), by=by)
    return sorted(grid.columns[grid.iloc[0]])


def status(index_id: str) -> dict:
    """`{verified_through, age_days, n_securities, warnings}` for `index_id`."""
    manifest = store.load_manifest(index_id)
    verified_through = manifest.get("verified_through")
    age_days = None
    warnings_list = []
    if verified_through:
        age_days = (
            pd.Timestamp.now(tz="UTC").normalize() - pd.Timestamp(verified_through, tz="UTC")
        ).days
        if age_days > config.get_membership_max_staleness_days():
            warnings_list.append(
                f"verified_through ({verified_through}) is {age_days} days old, past the "
                f"{config.get_membership_max_staleness_days()}-day staleness threshold"
            )
    security_map = store.load_security_map(index_id)
    n_securities = security_map["security_id"].nunique() if len(security_map) else 0
    return {
        "verified_through": verified_through,
        "age_days": age_days,
        "n_securities": n_securities,
        "warnings": warnings_list,
    }


def _check_staleness(index_id: str, strict: bool) -> None:
    result = status(index_id)
    if not result["warnings"]:
        return
    message = result["warnings"][0]
    if strict:
        raise MembershipStaleError(message)
    warnings.warn(message, MembershipStaleWarning)
