"""Column name/order constants for every membership table. Every reader and
writer in this package imports these rather than spelling out column lists,
so a column rename is a one-line change.
"""
from __future__ import annotations

TICKER_INTERVALS_COLUMNS = ["index_id", "ticker", "start_date", "end_date_excl"]

SECURITY_MAP_COLUMNS = [
    "index_id",
    "ticker",
    "start_date",
    "security_id",
    "evidence",
    "reviewed",
]

SECURITY_MASTER_COLUMNS = [
    "security_id",
    "name",
    "latest_ticker",
    "cik",
    "composite_figi",
    "share_class_figi",
    "notes",
]

TICKER_HISTORY_COLUMNS = ["security_id", "ticker", "valid_from", "valid_to_excl", "source"]

TAIL_CHANGES_COLUMNS = ["effective_date", "action", "ticker", "source", "note"]

MEMBERSHIP_INTERVALS_COLUMNS = ["index_id", "security_id", "start_date", "end_date_excl"]
