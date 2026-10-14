"""Join ticker_intervals + security_map into per-security membership
intervals. Always derived at call time from the committed tables, never
stored -- see spec D6.
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
    (e.g. AMD leaving and rejoining) stays as separate rows.

    Raises:
        MembershipValidationError: a ticker_intervals row has no matching
            security_map row, or two intervals for one security_id overlap.
    """
    ticker_intervals = store.load_ticker_intervals(index_id)
    security_map = store.load_security_map(index_id)

    merged = ticker_intervals.merge(
        security_map[["index_id", "ticker", "start_date", "security_id"]],
        on=["index_id", "ticker", "start_date"],
        how="left",
    )
    unresolved = merged[merged["security_id"].isna()]
    if len(unresolved):
        raise MembershipValidationError(
            f"{len(unresolved)} ticker_intervals row(s) have no matching security_map "
            f"row: {unresolved[['ticker', 'start_date']].to_dict('records')}"
        )

    rows = []
    for security_id, group in merged.sort_values("start_date").groupby("security_id"):
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
