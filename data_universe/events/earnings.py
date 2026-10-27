"""PEAD event table: one row per qualifying 8-K Item 2.02 filing, combining
`EdgarSource` (announcement dates), a price `DataSource` (liquidity/vol
inputs), and the options toolkit (the ATM straddle chosen at day0).

Every "spec issue to flag" from `PLAN.md` §8 gets its own named column here,
never silently resolved:
  1. sigma_pre_daily      -- daily (not annualized) stdev of the 30 log
                             returns (31 closes) ending at pre_session.
  2. adv_usd_pre20 / adv_usd_day0_day1 -- only these two liquidity windows;
                             no post-entry-window average is computed.
  4. days_to_expiry        -- nearest expiry strictly after day0; a bias
                             the project team may want to adjust, not fixed
                             here.
  5. next_earnings_session -- the REALIZED next filing's event date, not a
                             forecast; NaT for the most recent filing.
  6. call_volume/put_volume/is_stale -- is_stale is true whenever either
                             leg of the chosen straddle traded zero volume
                             on day0 (a stale daily close).
(Issue 3, weekly-ranking lookahead, is out of scope -- no ranking column
exists in this table at all.)

Price bases (v0.6.0): `sigma_pre_daily` uses total-return closes
(`price_source.daily_bars`); everything that is a price *level* uses raw bars
(`price_source.daily_bars_raw`) -- the ATM `spot` (historical option strikes
are raw, so a split-adjusted spot picks the wrong strike for any event before
a later split) and both `adv_usd_*` columns (real dollars traded). The option
chain is fetched filtered to expiries in (day0, day0 + `_EXPIRY_SEARCH_DAYS`]
and strikes within `_STRIKE_BAND` of spot, instead of the whole chain.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_universe import calendar
from data_universe.features import core
from data_universe.options import chain
from data_universe.sources.base import NY_TZ

_EVENT_COLUMNS = [
    "ticker", "event_date", "pre_session", "day0", "day1", "sigma_pre_daily",
    "adv_usd_pre20", "adv_usd_day0_day1", "days_to_expiry", "next_earnings_session",
    "call_volume", "put_volume", "is_stale",
]

# Monthly expiries are at most 35 days apart, so this window always contains the
# nearest listed expiry after day0.
_EXPIRY_SEARCH_DAYS = 40
# ATM strike search band, as a fraction of spot.
_STRIKE_BAND = 0.20


def build_pead_events(
    ticker: str,
    edgar_source,
    price_source,
    options_source,
    cik: Optional[str] = None,
    calendar_start: str = "2015-01-01",
    calendar_end: Optional[str] = None,
) -> pd.DataFrame:
    """Build the PEAD event table for one ticker.

    Args:
        ticker: Underlying ticker symbol.
        edgar_source: Object with `.get_cik(ticker)` and
            `.earnings_8k_filings(cik)` (see `sources/edgar.py`).
        price_source: A `DataSource`-shaped object providing `.daily_bars` and
            `.daily_bars_raw`.
        options_source: Object with `.contracts_as_of`/`.daily_close`
            (see `sources/polygon_options.py`; used via `options/chain.py`).
        cik: SEC CIK, or None to look it up via `edgar_source.get_cik`.
        calendar_start: Start of the trading calendar used to resolve
            sessions around each filing.
        calendar_end: End of the trading calendar. Defaults to today.

    Returns:
        A DataFrame with one row per qualifying filing (see module
        docstring for the column list), sorted ascending by `event_date`.
        A filing too close to either edge of the available calendar/price
        history to compute is skipped, not raised on.
    """
    cik = cik or edgar_source.get_cik(ticker)
    filings = edgar_source.earnings_8k_filings(cik)
    if filings.empty:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    end = calendar_end or pd.Timestamp.today().strftime("%Y-%m-%d")
    session_index = calendar.sessions(price_source, calendar_start, end)

    rows = []
    for _, filing in filings.iterrows():
        row = _build_one_event(ticker, filing, session_index, price_source, options_source)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows, columns=_EVENT_COLUMNS)
    table = table.sort_values("event_date").reset_index(drop=True)
    # next_earnings_session: the following row's event_date, known only in hindsight.
    table["next_earnings_session"] = table["event_date"].shift(-1)
    return table


def _build_one_event(ticker, filing, session_index, price_source, options_source):
    # EDGAR acceptanceDateTime is ISO UTC -- convert to New York before
    # normalizing so an evening filing on date D lands on D, not D+1.
    accepted = pd.Timestamp(filing["acceptance_datetime"])
    if accepted.tz is not None:
        accepted = accepted.tz_convert(NY_TZ)
    event_date = accepted.tz_localize(None).normalize()

    day0 = (
        event_date
        if event_date in session_index
        else calendar.next_session(session_index, event_date)
    )
    if day0 is None:
        return None
    pre_session = calendar.previous_session(session_index, day0)
    day1 = calendar.next_session(session_index, day0)
    if pre_session is None or day1 is None:
        return None

    pre_window_start = calendar.shift_sessions(session_index, pre_session, -30)
    if pre_window_start is None:
        return None
    window = (pre_window_start.strftime("%Y-%m-%d"), day1.strftime("%Y-%m-%d"))
    bars = price_source.daily_bars(ticker, *window)
    raw_bars = price_source.daily_bars_raw(ticker, *window)
    closes = bars["close"].copy()
    closes.index = bars.index.tz_localize(None).normalize()
    raw_closes = raw_bars["close"].copy()
    raw_closes.index = raw_bars.index.tz_localize(None).normalize()

    log_returns = core.log_returns(closes)
    sigma_pre_daily = log_returns.loc[log_returns.index <= pre_session].tail(30).std(ddof=1)

    dollar_volume = raw_bars["close"] * raw_bars["volume"]
    dollar_volume.index = raw_closes.index
    adv_usd_pre20 = dollar_volume.loc[dollar_volume.index <= pre_session].tail(20).mean()
    if any(d not in index for d in (day0, day1) for index in (closes.index, raw_closes.index)):
        return None  # halt/no bar on day0 or day1 -- skip just this event
    adv_usd_day0_day1 = dollar_volume.loc[[day0, day1]].mean()

    day0_str = day0.strftime("%Y-%m-%d")
    spot = float(raw_closes.loc[day0])
    try:
        option_chain = chain.as_of(
            options_source,
            ticker,
            day0_str,
            expiration_gte=(day0 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            expiration_lte=(day0 + pd.Timedelta(days=_EXPIRY_SEARCH_DAYS)).strftime("%Y-%m-%d"),
            strike_gte=spot * (1.0 - _STRIKE_BAND),
            strike_lte=spot * (1.0 + _STRIKE_BAND),
        )
        expiry = chain.nearest_expiry_after(option_chain, day0_str)
        strike = chain.atm_strike(option_chain, spot)
        straddle = chain.straddle(options_source, option_chain, strike, expiry, day0_str)
    except (ValueError, KeyError):
        return None

    days_to_expiry = (expiry - day0).days
    call_volume = straddle["call_volume"]
    put_volume = straddle["put_volume"]
    is_stale = bool(call_volume == 0 or put_volume == 0)

    return {
        "ticker": ticker,
        "event_date": event_date,
        "pre_session": pre_session,
        "day0": day0,
        "day1": day1,
        "sigma_pre_daily": sigma_pre_daily,
        "adv_usd_pre20": adv_usd_pre20,
        "adv_usd_day0_day1": adv_usd_day0_day1,
        "days_to_expiry": days_to_expiry,
        "next_earnings_session": np.nan,  # filled in by build_pead_events after sorting
        "call_volume": call_volume,
        "put_volume": put_volume,
        "is_stale": is_stale,
    }
