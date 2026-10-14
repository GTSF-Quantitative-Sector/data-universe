"""Turn a sequence of raw upstream membership snapshots into per-ticker
intervals. Pure and mechanical -- it does not know what a security is,
only what a ticker's presence/absence looks like over time. Identity
resolution (deciding which intervals belong to the same company) is
`identity.py`'s job, added in PR2.
"""
from __future__ import annotations

from typing import Optional


def derive_ticker_intervals(
    snapshots: list[tuple[str, set]]
) -> list[tuple[str, str, Optional[str]]]:
    """Derive `(ticker, start_date, end_date_excl)` tuples from upstream snapshots.

    Args:
        snapshots: `(date, set_of_tickers)` pairs, sorted ascending by date.
            Each date is a file row; the ticker set is that row's full
            membership list.

    Returns:
        One tuple per run of consecutive snapshots in which a ticker was
        present. `end_date_excl` is None for a ticker still present in the
        last snapshot (an open interval).
    """
    open_start: dict = {}
    intervals: list = []
    previous: set = set()
    for date, members in snapshots:
        for ticker in sorted(members - previous):
            open_start[ticker] = date
        for ticker in sorted(previous - members):
            intervals.append((ticker, open_start.pop(ticker), date))
        previous = members
    for ticker in sorted(open_start):
        intervals.append((ticker, open_start[ticker], None))
    return intervals
