"""Join ticker_intervals + security_map into per-security membership
intervals. Always derived at call time from the committed tables, never
stored -- see spec D6.

A `split` override (see `membership/identity.py`) produces several
security_map rows for one ticker interval: the rows whose start_date lies
inside the interval partition it, each piece resolving to its own security.
"""
from __future__ import annotations

import pandas as pd

from data_universe.membership import store
from data_universe.membership.errors import MembershipValidationError
from data_universe.membership.schema import MEMBERSHIP_INTERVALS_COLUMNS


def membership_intervals(index_id: str) -> pd.DataFrame:
    """`index_id, security_id, start_date, end_date_excl` for every security
    that was ever a member of `index_id`.

    Touching intervals resolved to the same `security_id` (e.g. FB -> META)
    are coalesced into one row; a real gap for the same `security_id`
    (e.g. AMD leaving and rejoining) stays as separate rows. `split`
    overrides cut a ticker interval into per-security pieces (e.g. CB's
    2016-01-19 change of company).

    Raises:
        MembershipValidationError: a ticker_intervals row has no matching
            security_map row, or two intervals for one security_id overlap.
    """
    ticker_intervals = store.load_ticker_intervals(index_id)
    security_map = store.load_security_map(index_id)

    # Partition each ticker interval by the security_map rows inside it;
    # group the map once (ticker -> sorted [(start, sid)]) instead of
    # re-filtering per interval.
    by_ticker = {}
    for _, row in security_map.iterrows():
        by_ticker.setdefault(row["ticker"], []).append(
            (row["start_date"], row["security_id"]))
    for rows_for_ticker in by_ticker.values():
        rows_for_ticker.sort()

    pieces = []
    unresolved = []
    for _, iv in ticker_intervals.iterrows():
        ticker, start, end = iv["ticker"], iv["start_date"], iv["end_date_excl"]
        in_interval = [
            r for r in by_ticker.get(ticker, [])
            if r[0] >= start and (pd.isna(end) or r[0] < end)
        ]
        if start not in {r[0] for r in in_interval}:
            unresolved.append((ticker, start))
            continue
        bounds = [r[0] for r in in_interval] + (
            [end] if pd.notna(end) else [None]
        )
        for i, sub_start in enumerate(bounds[:-1]):
            sub_end = bounds[i + 1]
            sid = next(r[1] for r in in_interval if r[0] == sub_start)
            pieces.append((index_id, sid, sub_start, sub_end))
    if unresolved:
        raise MembershipValidationError(
            f"{len(unresolved)} ticker_intervals row(s) have no matching security_map "
            f"row: {unresolved[:5]}"
        )

    # coalesce touching pieces of the same security
    frame = pd.DataFrame(pieces, columns=MEMBERSHIP_INTERVALS_COLUMNS)
    rows = []
    for security_id, group in frame.sort_values("start_date").groupby("security_id"):
        current_start, current_end = None, None
        for _, row in group.iterrows():
            start, end = row["start_date"], row["end_date_excl"]
            if current_start is None:
                current_start, current_end = start, end
            elif pd.notna(current_end) and start == current_end:
                current_end = end  # touching: extend the open/closed interval
            else:
                _append_interval(rows, index_id, security_id, current_start, current_end, start)
                current_start, current_end = start, end
        rows.append((index_id, security_id, current_start, current_end))

    return pd.DataFrame(rows, columns=MEMBERSHIP_INTERVALS_COLUMNS)


def _append_interval(rows, index_id, security_id, start, end, next_start):
    if pd.isna(end):
        raise MembershipValidationError(
            f"security_id {security_id} has an interval starting at {next_start} "
            f"that overlaps an already-open interval starting at {start}"
        )
    if next_start < end:
        raise MembershipValidationError(
            f"security_id {security_id} has overlapping intervals around {start}..{end} "
            f"and {next_start}"
        )
    rows.append((index_id, security_id, start, end))
