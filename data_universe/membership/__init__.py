"""Point-in-time index membership. Currently S&P 500 only -- every table
carries an `index_id` column so the Dow and Nasdaq-100 can follow later as
data, not new code. See `data_universe/membership/tables/sp500/README.md`
for provenance of the committed data.
"""
from __future__ import annotations

import pandas as pd

from data_universe.membership import grid as _grid
from data_universe.membership import intervals as _intervals


def sp500_intervals() -> pd.DataFrame:
    """`index_id, security_id, start_date, end_date_excl` for every security
    that was ever an S&P 500 member. `end_date_excl` is NaN for an open
    (current) membership."""
    return _intervals.membership_intervals("sp500")


def membership_grid(index_id: str, sessions, strict: bool = False) -> pd.DataFrame:
    return _grid.membership_grid(index_id, sessions, strict=strict)


def mask_like(features: pd.DataFrame, index_id: str) -> pd.DataFrame:
    return _grid.mask_like(features, index_id)


def members_on(index_id: str, date: str) -> list:
    return _grid.members_on(index_id, date)


def status(index_id: str) -> dict:
    return _grid.status(index_id)
