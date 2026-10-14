"""Boolean membership grid and the small query functions built on it.
Never stored -- rebuilt from `intervals.membership_intervals()` on every call.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from data_universe import config
from data_universe.membership import intervals as intervals_module
from data_universe.membership import store
from data_universe.membership.errors import MembershipStaleError, MembershipStaleWarning


def membership_grid(index_id: str, sessions, strict: bool = False) -> pd.DataFrame:
    """Boolean membership grid: one row per session, one column per security_id.

    `grid.loc[t, s]` is True if `start_date <= t < end_date_excl` for security
    `s`'s membership interval -- i.e. membership at the close of session `t`.

    Args:
        index_id: e.g. "sp500".
        sessions: Trading session dates to build rows for (e.g. from
            `data_universe.calendar.sessions()`).
        strict: If True, raise `MembershipStaleError` when the data is older
            than `config.get_membership_max_staleness_days()`. If False
            (default), emit `MembershipStaleWarning` and proceed anyway.

    Returns:
        A bool DataFrame, columns=sorted security_id. The row index is
        `sessions` itself (normalized, name "date") -- same order, including
        duplicates, as what the caller passed in. `sessions` need not be
        sorted; it is sorted and deduped internally for the interval lookup
        and the result is scattered back into the caller's requested order.
    """
    _check_staleness(index_id, strict)
    member_intervals = intervals_module.membership_intervals(index_id)

    requested_index = pd.DatetimeIndex(sessions).normalize()
    unique_sorted = requested_index.unique().sort_values()
    lookup_index = (
        unique_sorted.tz_localize(None) if unique_sorted.tz is not None else unique_sorted
    )

    columns = sorted(member_intervals["security_id"].unique())
    column_of = {security_id: position for position, security_id in enumerate(columns)}
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
        values[first:last, column_of[row["security_id"]]] = True

    sorted_grid = pd.DataFrame(values, index=unique_sorted, columns=columns)
    requested_lookup = (
        requested_index.tz_localize(None) if requested_index.tz is not None else requested_index
    )
    positions = lookup_index.searchsorted(requested_lookup, side="left")
    result = sorted_grid.iloc[positions]
    result.index = pd.DatetimeIndex(requested_index, name="date")
    return result


def mask_like(features: pd.DataFrame, index_id: str) -> pd.DataFrame:
    """Boolean mask with the same index and columns as `features`. A column
    `features` has that `index_id` never held as a member is all False.

    `membership_grid` is looked up with `features.index` normalized to
    midnight for the membership comparison, but the mask handed back is
    reindexed onto `features.index` itself (same values, order, name, and
    tz) so it lines up row-for-row with `features` even when `features` is
    sampled intraday (e.g. hourly bars)."""
    grid = membership_grid(index_id, features.index)
    grid.index = features.index
    return grid.reindex(columns=features.columns, fill_value=False)


def members_on(index_id: str, date: str) -> list:
    """Securities that were members of `index_id` at the close of `date`."""
    grid = membership_grid(index_id, pd.DatetimeIndex([pd.Timestamp(date)]))
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
