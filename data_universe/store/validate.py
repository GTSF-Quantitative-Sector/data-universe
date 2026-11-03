"""Validation flags for daily bar data -- flags, never fixes.

`validate_bars` compares one ticker's raw and split-adjusted daily frames and
reports the session dates worth a second look: impossible OHLC rows,
duplicated stamps, calendar gaps vs a session index, >40% close jumps the
vendor's own split factor doesn't explain (a possible unrecorded split --
genuine crashes and large special dividends flag too, which is by design;
`OPEN_QUESTIONS.md` item 33), and runs of repeated closes. Flags are
on-demand: nothing is written back to the store and nothing alters the data.
`validate_tickers` runs a list of tickers with per-ticker failure isolation,
fetching the calendar's sessions once per call.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from data_universe.store.raw import _row_dates

FLAG_COLUMNS = [
    "ohlc_bad", "duplicate_stamp", "calendar_gap", "jump_no_split", "stale_run",
]
JUMP_LOG_THRESHOLD = math.log(1.4)
STALE_MIN_RUN = 5
_FACTOR_TOL = 1e-6


def ohlc_flags(bars: pd.DataFrame) -> pd.Series:
    """Per row: non-finite or non-positive OHLC, high < max(open, close),
    low > min(open, close), low > high, or non-finite/negative volume."""
    opens = bars["open"].to_numpy(dtype="float64")
    highs = bars["high"].to_numpy(dtype="float64")
    lows = bars["low"].to_numpy(dtype="float64")
    closes = bars["close"].to_numpy(dtype="float64")
    volume = bars["volume"].to_numpy(dtype="float64")
    bad = (
        ~np.isfinite(opens) | ~np.isfinite(highs) | ~np.isfinite(lows)
        | ~np.isfinite(closes) | ~np.isfinite(volume)
        | (opens <= 0) | (highs <= 0) | (lows <= 0) | (closes <= 0)
        | (highs < np.maximum(opens, closes))
        | (lows > np.minimum(opens, closes))
        | (lows > highs)
        | (volume < 0)
    )
    return pd.Series(bad, index=bars.index)


def duplicate_flags(bars: pd.DataFrame) -> pd.Series:
    """Rows whose session date occurs more than once."""
    dates = _row_dates(bars, None)
    return pd.Series(dates.duplicated(keep=False), index=bars.index)


def stale_run_flags(bars: pd.DataFrame, min_run: int = STALE_MIN_RUN) -> pd.Series:
    """Every row of a maximal run of >= `min_run` consecutive identical closes."""
    closes = bars["close"].to_numpy(dtype="float64")
    flags = np.zeros(len(closes), dtype=bool)
    start = 0
    for i in range(1, len(closes) + 1):
        if i == len(closes) or closes[i] != closes[start]:
            if i - start >= min_run:
                flags[start:i] = True
            start = i
    return pd.Series(flags, index=bars.index)


def jump_no_split_flags(
    raw: pd.DataFrame, split_adjusted: pd.DataFrame,
    threshold: float = JUMP_LOG_THRESHOLD,
) -> pd.Series:
    """Indexed by session date: |log(close_t / close_{t-1})| > `threshold`
    while the split factor raw/split_adjusted is unchanged (|f_t/f_{t-1} - 1|
    <= 1e-6) -- a jump the vendor's own split adjustment doesn't explain.
    Duplicate dates in raw are dropped keep='last'; dates missing from either
    frame are skipped."""
    raw_dates = _row_dates(raw, None)
    raw = raw.iloc[~raw_dates.duplicated(keep="last")]
    raw_close = pd.Series(
        raw["close"].to_numpy(dtype="float64"), index=_row_dates(raw, None)
    )
    adj_dates = _row_dates(split_adjusted, None)
    adj = split_adjusted.iloc[~adj_dates.duplicated(keep="last")]
    adj_close = pd.Series(
        adj["close"].to_numpy(dtype="float64"), index=_row_dates(adj, None)
    )
    common = raw_close.index.intersection(adj_close.index).sort_values()
    raw_close, adj_close = raw_close.loc[common], adj_close.loc[common]
    with np.errstate(invalid="ignore", divide="ignore"):
        log_move = np.log(raw_close / raw_close.shift(1)).abs()
    factor = raw_close / adj_close
    factor_change = (factor / factor.shift(1) - 1.0).abs()
    return (log_move > threshold) & (factor_change <= _FACTOR_TOL)


def calendar_gaps(bars: pd.DataFrame, sessions) -> pd.DatetimeIndex:
    """Sessions inside [first bar date, last bar date] with no bar."""
    dates = _row_dates(bars, None)
    sessions = pd.DatetimeIndex(sessions)
    in_range = sessions[(sessions >= dates.min()) & (sessions <= dates.max())]
    return in_range.difference(dates)


def validate_bars(
    raw: pd.DataFrame,
    split_adjusted: pd.DataFrame,
    sessions=None,
    *,
    jump_threshold: float = JUMP_LOG_THRESHOLD,
    min_run: int = STALE_MIN_RUN,
) -> pd.DataFrame:
    """One row per flagged session date: a bool `FLAG_COLUMNS` frame indexed
    by tz-naive `date`, sorted. ohlc_bad/stale_run/jump_no_split come from
    `raw`; duplicate_stamp when the date is duplicated in either frame;
    calendar_gap needs `sessions` (all False when None)."""
    raw_dates = _row_dates(raw, None)
    adj_dates = _row_dates(split_adjusted, None)

    def flagged_dates(values, dates):
        return set(pd.DatetimeIndex(dates)[np.asarray(values, dtype=bool)])

    flag_sets = {
        "ohlc_bad": flagged_dates(ohlc_flags(raw).to_numpy(), raw_dates),
        "duplicate_stamp": (
            flagged_dates(raw_dates.duplicated(keep=False), raw_dates)
            | flagged_dates(adj_dates.duplicated(keep=False), adj_dates)
        ),
        "calendar_gap": (
            set(calendar_gaps(raw, sessions)) if sessions is not None else set()
        ),
        "jump_no_split": set(
            jump_no_split_flags(raw, split_adjusted, jump_threshold)
            .loc[lambda s: s].index
        ),
        "stale_run": flagged_dates(stale_run_flags(raw, min_run).to_numpy(), raw_dates),
    }
    all_dates = sorted(set().union(*flag_sets.values()))
    if not all_dates:
        return pd.DataFrame(
            {column: pd.Series(dtype=bool) for column in FLAG_COLUMNS},
            index=pd.DatetimeIndex([], name="date"),
        )
    out = pd.DataFrame(
        False,
        index=pd.DatetimeIndex(all_dates, name="date"),
        columns=FLAG_COLUMNS,
    )
    for column, dates in flag_sets.items():
        if dates:
            out.loc[out.index.isin(dates), column] = True
    return out


def validate_ticker(
    source, ticker: str, start: str, end: str,
    calendar_ticker: str = "SPY", sessions=None, **kwargs,
) -> pd.DataFrame:
    """`validate_bars` for one ticker's raw and split-adjusted daily frames
    over [start, end]. `sessions` defaults to the row dates of
    `source.daily_bars_raw(calendar_ticker, start, end)`."""
    raw = source.daily_bars_raw(ticker, start, end)
    split_adjusted = source.daily_bars_split_adjusted(ticker, start, end)
    if sessions is None:
        sessions = _row_dates(
            source.daily_bars_raw(calendar_ticker, start, end), None
        )
    return validate_bars(raw, split_adjusted, sessions, **kwargs)


@dataclass
class ValidationReport:
    flags: pd.DataFrame
    failed: Dict[str, str] = field(default_factory=dict)


def validate_tickers(
    source, tickers, start: str, end: str,
    calendar_ticker: str = "SPY", **kwargs,
) -> ValidationReport:
    """`validate_ticker` over a list of tickers. The calendar's sessions are
    fetched once per call; one ticker's failure is recorded in
    `report.failed` and never stops the rest. `flags` has columns
    `ticker, date, *FLAG_COLUMNS`, sorted by (ticker, date)."""
    sessions = _row_dates(
        source.daily_bars_raw(calendar_ticker, start, end), None
    )
    frames = []
    failed: Dict[str, str] = {}
    for ticker in tickers:
        try:
            frame = validate_ticker(
                source, ticker, start, end, calendar_ticker, sessions, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - per-ticker failure isolation
            failed[ticker] = f"{type(exc).__name__}: {exc}"
            continue
        if len(frame):
            part = frame.reset_index()
            part.insert(0, "ticker", ticker)
            frames.append(part)
    if frames:
        flags = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["ticker", "date"])
            .reset_index(drop=True)
        )
    else:
        flags = pd.DataFrame(columns=["ticker", "date"] + FLAG_COLUMNS)
    return ValidationReport(flags=flags, failed=failed)
