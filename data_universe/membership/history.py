"""Per-date history views over the committed identity tables:
`symbol_segments`/`cik_segments` turn `ticker_history`/`cik_history` rows
into a complete [start, end] coverage for one security, filling uncovered
stretches with source "extended".
"""
from __future__ import annotations

import pandas as pd

from data_universe.membership import store
from data_universe.membership.errors import MembershipError


def _check_security(index_id: str, security_id: str) -> None:
    master = store.load_security_master(index_id)
    if security_id not in set(master["security_id"]):
        raise MembershipError(
            f"unknown security_id {security_id!r} for index {index_id!r}"
        )


def _segments(frame: pd.DataFrame, security_id: str, start, end,
              value_col: str) -> pd.DataFrame:
    columns = [value_col, "valid_from", "valid_to_excl", "source"]
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"start {start!r} is after end {end!r}")
    start_ts = pd.Timestamp(start)
    close = pd.Timestamp(end) + pd.Timedelta(days=1)

    rows = frame[frame["security_id"] == security_id]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    clipped = []
    for _, row in rows.iterrows():
        vf = pd.Timestamp(row["valid_from"])
        raw_vt = row["valid_to_excl"]
        vt = None if (raw_vt == "" or pd.isna(raw_vt)) else pd.Timestamp(raw_vt)
        lo = max(vf, start_ts)
        hi = min(vt, close) if vt is not None else close
        if lo < hi:
            clipped.append([row[value_col], lo, hi, row["source"], vf])
    clipped.sort(key=lambda r: r[1])
    if not clipped:
        # The window sits entirely outside the history: extend from the
        # nearest row (first row's value when before it, last's when after).
        ordered = rows.assign(_vf=pd.to_datetime(rows["valid_from"])).sort_values("_vf")
        if start_ts < ordered["_vf"].iloc[0]:
            value = ordered[value_col].iloc[0]
        else:
            value = ordered[value_col].iloc[-1]
        return pd.DataFrame(
            [[value, start_ts, close, "extended"]], columns=columns
        )

    out = []
    expected = start_ts
    for value, lo, hi, src, _vf in clipped:
        if lo > expected:
            fill_value = clipped[0][0] if not out else out[-1][0]
            out.append([fill_value, expected, lo, "extended"])
        out.append([value, lo, hi, src])
        expected = hi
    if expected < close:
        fill_value = clipped[-1][0] if clipped else ""
        if clipped:
            out.append([fill_value, expected, close, "extended"])
    return pd.DataFrame(
        {
            value_col: [r[0] for r in out],
            "valid_from": [r[1] for r in out],
            "valid_to_excl": [r[2] for r in out],
            "source": [r[3] for r in out],
        },
        columns=columns,
    )


def symbol_segments(index_id: str, security_id: str, start, end) -> pd.DataFrame:
    """[ticker, valid_from, valid_to_excl, source] covering [start, end]
    inclusive for one security (last row's valid_to_excl = end + 1 day)."""
    _check_security(index_id, security_id)
    return _segments(
        store.load_ticker_history(index_id), security_id, start, end, "ticker"
    )


def cik_segments(index_id: str, security_id: str, start, end) -> pd.DataFrame:
    """[cik, valid_from, valid_to_excl, source] covering [start, end]
    inclusive for one security (last row's valid_to_excl = end + 1 day)."""
    _check_security(index_id, security_id)
    return _segments(
        store.load_cik_history(index_id), security_id, start, end, "cik"
    )
