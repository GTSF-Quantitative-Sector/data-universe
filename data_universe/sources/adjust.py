"""Total-return (split + cash-dividend) adjustment of bar frames. Pure -- no I/O.

Convention (backward, multiplicative -- the "adjusted close" convention): for each cash
ex-dividend date `e` inside a frame, the factor is

    f_e = 1 - D_e / C(prev)

where `D_e` is the summed cash amount of every dividend record with that ex-date (regular,
special and capital-gain distributions alike) and `C(prev)` is the close of the last
session strictly before `e`. Every bar whose New York session date is strictly before `e`
has its open/high/low/close/vwap multiplied by `f_e`; volume is never touched. The
return across `e` is then `C(e) / (C(prev) - D_e)`.

**Anchoring:** only ex-dates after the frame's first bar and on/before its last bar are
used, so the frame's last bar equals the split-adjusted price and absolute adjusted levels
depend on the requested window. Returns, volatilities and every other ratio are identical
for any window -- compare levels only within one frame, and use raw (unadjusted) bars for
real price levels. See `OPEN_QUESTIONS.md` item 21.

**Cash basis:** `C(prev)` must be on the same basis as `D_e`. `DIVIDEND_CASH_BASIS` records
which close series the vendor's `cash_amount` matches: `"raw"` (as declared, divide by raw
closes) or `"split_adjusted"` (divide by split-adjusted closes). The factor is identical
either way once the basis is right. See `OPEN_QUESTIONS.md` item 20 -- `python -m
data_universe.probe` checks it against a known AAPL dividend.

Not adjusted (known limitation): spin-offs, stock dividends, rights issues.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_universe.sources.base import NY_TZ, DataSourceError

DIVIDEND_CASH_BASIS = "raw"
DIVIDEND_COLUMNS = ["ex_dividend_date", "cash_amount", "currency", "dividend_type"]
PRICE_COLUMNS = ["open", "high", "low", "close", "vwap"]


def empty_dividends() -> pd.DataFrame:
    """An empty, correctly-typed dividends frame (columns `DIVIDEND_COLUMNS`)."""
    return pd.DataFrame(
        {
            "ex_dividend_date": pd.Series(dtype="datetime64[ns]"),
            "cash_amount": pd.Series(dtype="float64"),
            "currency": pd.Series(dtype="object"),
            "dividend_type": pd.Series(dtype="object"),
        }
    )


def _session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Tz-naive New York session date of each stamp in `index`."""
    if index.tz is not None:
        index = index.tz_convert(NY_TZ).tz_localize(None)
    return index.normalize()


def dividend_factors(closes: pd.Series, dividends: pd.DataFrame) -> pd.Series:
    """Per-ex-date backward adjustment factors for one frame.

    Args:
        closes: Daily closes (tz-aware New York or tz-naive index), on the same basis as
            `dividends["cash_amount"]` -- see `DIVIDEND_CASH_BASIS`.
        dividends: A frame with `DIVIDEND_COLUMNS`.

    Returns:
        A float Series indexed by tz-naive ex-date, one factor `f_e` per ex-date strictly
        after the first close's date and on/before the last's, sorted ascending. Empty if
        none qualify.

    Raises:
        DataSourceError: A record has a non-USD currency, or a factor falls outside (0, 1)
            (a dividend at least as large as the prior close, or a non-positive amount).
    """
    empty = pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="ex_dividend_date"))
    if closes.empty or dividends.empty:
        return empty
    dates = _session_dates(pd.DatetimeIndex(closes.index))
    values = closes.to_numpy(dtype="float64")
    order = np.argsort(dates.values, kind="stable")
    dates, values = dates[order], values[order]

    frame = dividends.copy()
    frame["ex_dividend_date"] = pd.to_datetime(frame["ex_dividend_date"]).dt.normalize()
    ex_dates = frame["ex_dividend_date"]
    frame = frame[(ex_dates > dates[0]) & (ex_dates <= dates[-1])]
    if frame.empty:
        return empty
    currencies = frame["currency"].fillna("").astype(str).str.upper()
    bad = frame[(currencies != "") & (currencies != "USD")]
    if len(bad):
        raise DataSourceError(
            f"Non-USD dividend record(s) can't be adjusted against USD prices: "
            f"{bad[['ex_dividend_date', 'currency']].to_dict('records')}"
        )
    amounts = frame.groupby("ex_dividend_date")["cash_amount"].sum().sort_index()

    factors = []
    for ex_date, amount in amounts.items():
        position = dates.searchsorted(ex_date, side="left") - 1
        prior_close = values[position]
        factor = 1.0 - float(amount) / prior_close
        if not 0.0 < factor < 1.0:
            raise DataSourceError(
                f"Dividend of {amount} on {ex_date.date()} against prior close "
                f"{prior_close} gives an adjustment factor {factor} outside (0, 1)"
            )
        factors.append(factor)
    return pd.Series(factors, index=pd.DatetimeIndex(amounts.index, name="ex_dividend_date"))


def apply_total_return(bars: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """Apply `dividend_factors` output to a daily or intraday bars frame.

    Args:
        bars: A bars frame (see `sources/base.py`).
        factors: As returned by `dividend_factors`.

    Returns:
        A copy of `bars` with each price column multiplied by the product of every factor
        whose ex-date is strictly after the bar's New York session date. Volume unchanged.
    """
    result = bars.copy()
    if bars.empty or factors.empty:
        return result
    factors = factors.sort_index()
    ex_dates = pd.DatetimeIndex(factors.index).values
    # suffix[i] = product of factors[i:], suffix[n] = 1
    suffix = np.append(np.cumprod(factors.to_numpy()[::-1])[::-1], 1.0)
    positions = np.searchsorted(ex_dates, _session_dates(bars.index).values, side="right")
    multiplier = suffix[positions]
    for column in PRICE_COLUMNS:
        if column in result.columns:
            result[column] = result[column].to_numpy(dtype="float64") * multiplier
    return result
