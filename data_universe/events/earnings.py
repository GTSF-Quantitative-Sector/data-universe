"""PEAD event table: one row per qualifying 8-K Item 2.02 filing, combining
`EdgarSource` (announcement dates), a price `DataSource` (liquidity/vol
inputs), and the options toolkit (the ATM straddle chosen at day0).

Every "spec issue to flag" from `PLAN.md` §8 gets its own named column here,
never silently resolved:
  1. sigma_pre_daily      -- daily (not annualized) stdev of log returns,
                             30 trading days ending at pre_session.
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
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_universe import calendar
from data_universe.features import core
from data_universe.options import chain

_EVENT_COLUMNS = [
    "ticker", "event_date", "pre_session", "day0", "day1", "sigma_pre_daily",
    "adv_usd_pre20", "adv_usd_day0_day1", "days_to_expiry", "next_earnings_session",
    "call_volume", "put_volume", "is_stale",
]


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
        price_source: A `DataSource`-shaped object providing `.daily_bars`.
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
    event_date = pd.Timestamp(filing["acceptance_datetime"]).tz_localize(None).normalize()

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

    pre_window_start = calendar.shift_sessions(session_index, pre_session, -29)
    if pre_window_start is None:
        return None
    bars = price_source.daily_bars(
        ticker, pre_window_start.strftime("%Y-%m-%d"), day1.strftime("%Y-%m-%d")
    )
    closes = bars["close"].copy()
    closes.index = bars.index.tz_localize(None).normalize()

    log_returns = core.log_returns(closes)
    sigma_pre_daily = log_returns.loc[log_returns.index <= pre_session].tail(30).std(ddof=1)

    dollar_volume = bars["close"] * bars["volume"]
    dollar_volume.index = closes.index
    adv_usd_pre20 = dollar_volume.loc[dollar_volume.index <= pre_session].tail(20).mean()
    adv_usd_day0_day1 = dollar_volume.loc[[day0, day1]].mean()

    try:
        option_chain = chain.as_of(options_source, ticker, day0.strftime("%Y-%m-%d"))
        expiry = chain.nearest_expiry_after(option_chain, day0.strftime("%Y-%m-%d"))
        spot = closes.loc[day0]
        strike = chain.atm_strike(option_chain, spot)
        straddle = chain.straddle(
            options_source, option_chain, strike, expiry, day0.strftime("%Y-%m-%d")
        )
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
