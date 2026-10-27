"""PEAD event table: one row per qualifying 8-K Item 2.02 filing, combining
`EdgarSource` (announcement dates), a price `DataSource` (liquidity/vol
inputs), and the options toolkit (the ATM straddles chosen at day0 and at the
pre-announcement session).

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

v0.7.0 additions (appended after the v1 columns, which are unchanged in
name, order, values and skip rules): anything a new column can't compute is
missing -- NaN / NaT / None / `<NA>` -- and NEVER causes the row to be
skipped.

Timing: `acceptance_time_et` is the EDGAR acceptanceDateTime expressed in
America/New_York (tz-aware); `announce_timing` buckets it by the regular
session clock -- "pre_open" before 09:30, "market_hours" 09:30-16:00,
"post_close" at 16:00 or later (`REGULAR_OPEN`/`REGULAR_CLOSE` from
`sources/base.py`). The bucket is purely clock-based, so weekend and holiday
filings classify the same way.

Sessions: a "post_close" filing on a session day is announced after that
day's close, so `pre_announcement_session` (the last session whose close
precedes the news) is `day0` itself and `reaction_session` (the first close
that reflects it) is `day1`. Every other filing -- pre_open, market_hours,
or any filing on a weekend/holiday -- uses `pre_session` and `day0`, which
maps off-session filings onto the surrounding sessions.

Straddles: the day-0 straddle v1 already computed (then dropped) is now
exported (`strike`, `expiry`, leg tickers/prices, `straddle_price`,
`spot_raw`). A second "pre" straddle (`pre_*` columns) is priced at the
close of `pre_announcement_session` against the chain as of that day, with
the same expiry rule as day0 -- the nearest expiry strictly after `day0` --
and the strike band around `pre_spot_raw`. When `pre_announcement_session`
IS `day0` (a post-close filing on a session day) the pre straddle copies the
day0 values and makes no extra requests. `implied_move =
pre_straddle_price / pre_spot_raw`; `ret_reaction` is the total-return move
from `pre_announcement_session`'s close to `reaction_session`'s;
`surprise_ratio = |ret_reaction| / implied_move` only when `implied_move`
is positive. `cik`/`cik_source` record where the filing CIK came from
("explicit", "sec_current", or a `cik_resolver` label).

`build_pead_panel` builds the same table over a whole index: an event is
kept iff `start <= event_date <= end` AND the ticker was a member at the
close of its `pre_announcement_session` under one of the intervals in its
resolved-CIK group (half-open `[start_date, end_date_excl)`; a NaN end means
the interval is open).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_universe import calendar
from data_universe.features import core
from data_universe.membership import grid as membership_grid
from data_universe.membership import store as membership_store
from data_universe.options import chain
from data_universe.sources.base import NY_TZ, REGULAR_CLOSE, REGULAR_OPEN
from data_universe.sources.edgar import EdgarSource
from data_universe.sources.massive import MassiveOptionsSource

V1_EVENT_COLUMNS = [
    "ticker", "event_date", "pre_session", "day0", "day1", "sigma_pre_daily",
    "adv_usd_pre20", "adv_usd_day0_day1", "days_to_expiry", "next_earnings_session",
    "call_volume", "put_volume", "is_stale",
]

V2_EVENT_COLUMNS = [
    "acceptance_time_et", "announce_timing", "pre_announcement_session", "reaction_session",
    "strike", "expiry", "call_ticker", "put_ticker", "call_price", "put_price",
    "straddle_price", "spot_raw",
    "pre_strike", "pre_expiry", "pre_call_ticker", "pre_put_ticker", "pre_call_price",
    "pre_put_price", "pre_straddle_price", "pre_spot_raw", "pre_days_to_expiry",
    "pre_call_volume", "pre_put_volume", "pre_is_stale",
    "implied_move", "ret_reaction", "surprise_ratio",
    "cik", "cik_source",
]

EVENT_COLUMNS = V1_EVENT_COLUMNS + V2_EVENT_COLUMNS

# Monthly expiries are at most 35 days apart, so this window always contains the
# nearest listed expiry after day0.
_EXPIRY_SEARCH_DAYS = 40
# ATM strike search band, as a fraction of spot.
_STRIKE_BAND = 0.20
# Calendar days of lookback a panel run pads before `start`, so events near the
# range edge still get their 31-close sigma window and session resolution.
_PANEL_LOOKBACK_DAYS = 90

# (ticker, as_of "YYYY-MM-DD") -> (cik, cik_source label), or None to fall
# through to `edgar_source.get_cik(ticker)`.
CikResolver = Callable[[str, str], Optional[Tuple[str, str]]]


@dataclass
class PeadPanelReport:
    events: pd.DataFrame
    succeeded: Dict[str, int] = field(default_factory=dict)  # ticker -> events kept
    failed: Dict[str, str] = field(default_factory=dict)     # ticker -> "ExcType: message"


def build_pead_events(
    ticker: str,
    edgar_source,
    price_source,
    options_source,
    cik: Optional[str] = None,
    calendar_start: str = "2015-01-01",
    calendar_end: Optional[str] = None,
    cik_source: Optional[str] = None,
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
        cik_source: Label for where `cik` came from, recorded in the
            `cik_source` column. Defaults to "explicit" when `cik` is given;
            passing it without `cik` raises ValueError.

    Returns:
        A DataFrame with one row per qualifying filing (see module
        docstring for the column list), sorted ascending by `event_date`.
        A filing too close to either edge of the available calendar/price
        history to compute is skipped, not raised on.
    """
    if cik is None:
        if cik_source is not None:
            raise ValueError("cik_source requires an explicit cik")
        cik = edgar_source.get_cik(ticker)
        cik_source = "sec_current"
    else:
        cik_source = cik_source or "explicit"
    cik = str(int(cik))
    filings = edgar_source.earnings_8k_filings(cik)
    if filings.empty:
        return _empty_events_frame()

    end = calendar_end or pd.Timestamp.today().strftime("%Y-%m-%d")
    session_index = calendar.sessions(price_source, calendar_start, end)

    rows = []
    for _, filing in filings.iterrows():
        row = _build_one_event(
            ticker, filing, session_index, price_source, options_source, cik, cik_source
        )
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    table = table.sort_values("event_date").reset_index(drop=True)
    # next_earnings_session: the following row's event_date, known only in hindsight.
    table["next_earnings_session"] = table["event_date"].shift(-1)
    table["pre_is_stale"] = table["pre_is_stale"].astype("boolean")
    return table


def build_pead_panel(
    start: str,
    end: Optional[str] = None,
    index_id: str = "sp500",
    *,
    edgar_source=None,
    price_source=None,
    options_source=None,
    tickers: Optional[Sequence[str]] = None,
    cik_overrides: Optional[Dict[str, str]] = None,
    cik_resolver: Optional[CikResolver] = None,
    strict: bool = False,
) -> PeadPanelReport:
    """Build the PEAD event table for every member of an index, point-in-time.

    Membership rule: an event is kept iff `start <= event_date <= end` AND the
    ticker was an index member at the close of its `pre_announcement_session`
    under one of the membership intervals in its resolved-CIK group --
    `start_date <= pre_announcement_session < end_date_excl`, where a NaN end
    means the interval is open. Intervals are keyed by ticker
    (`store.load_ticker_intervals`) until Phase 4's identity resolution.

    Per candidate interval the CIK is resolved, in order: `cik_overrides`
    (label "explicit"), then `cik_resolver(ticker, as_of)` where `as_of` is
    `max(interval start, last session before `start`)` -- a resolver returning
    None falls through -- then `edgar_source.get_cik(ticker)` (label
    "sec_current"). Each ticker's intervals are grouped by (cik, label); one
    `build_pead_events` call per group, with that group's events kept only
    inside that group's own intervals.

    Args:
        start: First event date, "YYYY-MM-DD" (inclusive).
        end: Last event date, "YYYY-MM-DD" (inclusive). Defaults to today.
        index_id: Membership index, e.g. "sp500".
        edgar_source: Object with `.get_cik`/`.earnings_8k_filings`.
            Defaults to `EdgarSource()`.
        price_source: Price bars source. Defaults to one shared
            `MassiveOptionsSource()` (same default for `options_source`).
        options_source: Options chain/close source. See `price_source`.
        tickers: Optional subset restricting the candidates. A requested
            ticker with no candidate interval lands in `failed` (non-fatal).
        cik_overrides: `{ticker: cik}` used before any resolver/EDGAR lookup.
        cik_resolver: Optional `(ticker, as_of) -> (cik, label)` hook for
            as-of-date CIK lookup (e.g. for renamed tickers).
        strict: Passed to the membership staleness check -- True raises
            `MembershipStaleError` instead of warning on stale tables.

    Returns:
        A `PeadPanelReport`: `events` sorted by (event_date, ticker) with
        exactly `EVENT_COLUMNS`; `succeeded` mapping ticker -> kept-event
        count (0 when nothing qualified); `failed` mapping ticker ->
        "ExcType: message" for one ticker's isolated failure, which never
        stops the rest of the panel.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError(f"start {start!r} is after end {end!r}")
    edgar_source = edgar_source if edgar_source is not None else EdgarSource()
    if price_source is None or options_source is None:
        massive = MassiveOptionsSource()
        if price_source is None:
            price_source = massive
        if options_source is None:
            options_source = massive

    membership_grid._check_staleness(index_id, strict)
    ticker_intervals = membership_store.load_ticker_intervals(index_id)

    calendar_start = (
        pd.Timestamp(start) - pd.Timedelta(days=_PANEL_LOOKBACK_DAYS)
    ).strftime("%Y-%m-%d")
    # Same arguments build_pead_events is called with below, so the proxy's
    # bars request is shared through the source's cache.
    session_index = calendar.sessions(price_source, calendar_start, end)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # `pre_announcement_session` is never earlier than the last session before
    # `start`, so an interval already ended by then can never contain one.
    lower = calendar.previous_session(session_index, start_ts)
    if lower is None:
        lower = start_ts

    interval_starts = pd.to_datetime(ticker_intervals["start_date"])
    interval_ends = pd.to_datetime(ticker_intervals["end_date_excl"])
    candidates = ticker_intervals.loc[
        (interval_starts <= end_ts) & (interval_ends.isna() | (interval_ends > lower))
    ]

    failed: Dict[str, str] = {}
    if tickers is not None:
        requested = set(tickers)
        for missing in sorted(requested - set(candidates["ticker"])):
            failed[missing] = f"not a member of {index_id} between {start} and {end}"
        candidates = candidates.loc[candidates["ticker"].isin(requested)]

    events_frames = []
    succeeded: Dict[str, int] = {}
    for ticker, group in candidates.groupby("ticker"):
        try:
            cik_groups: Dict[Tuple[str, str], list] = {}
            for _, row in group.iterrows():
                as_of = max(pd.Timestamp(row["start_date"]), lower).strftime("%Y-%m-%d")
                key = _resolve_cik(
                    edgar_source, ticker, as_of, cik_overrides, cik_resolver
                )
                cik_groups.setdefault(key, []).append(row)
            ticker_frames = []
            for (cik, label), interval_rows in cik_groups.items():
                events = build_pead_events(
                    ticker,
                    edgar_source,
                    price_source,
                    options_source,
                    cik=cik,
                    cik_source=label,
                    calendar_start=calendar_start,
                    calendar_end=end,
                )
                kept = _events_in_intervals(events, interval_rows, start_ts, end_ts)
                if len(kept):
                    ticker_frames.append(kept)
            if ticker_frames:
                combined = pd.concat(ticker_frames, ignore_index=True)
                events_frames.append(combined)
                succeeded[ticker] = len(combined)
            else:
                succeeded[ticker] = 0
        except Exception as exc:  # noqa: BLE001 - isolate any failure, report it
            failed[ticker] = f"{type(exc).__name__}: {exc}"

    if events_frames:
        events = pd.concat(events_frames, ignore_index=True)
        events = events.sort_values(["event_date", "ticker"]).reset_index(drop=True)
    else:
        events = _empty_events_frame()
    events = events[EVENT_COLUMNS]
    events["pre_is_stale"] = events["pre_is_stale"].astype("boolean")
    return PeadPanelReport(events=events, succeeded=succeeded, failed=failed)


def _empty_events_frame() -> pd.DataFrame:
    """An empty frame with exactly `EVENT_COLUMNS` and the right dtypes."""
    frame = pd.DataFrame(columns=EVENT_COLUMNS)
    frame["pre_is_stale"] = frame["pre_is_stale"].astype("boolean")
    return frame


def _announce_timing(acceptance_time_et: pd.Timestamp) -> str:
    """Bucket an ET acceptance time by the regular-session clock: before 09:30
    is "pre_open", 09:30 up to (not including) 16:00 is "market_hours", 16:00
    or later is "post_close". Purely clock-based -- weekend and holiday dates
    classify the same way."""
    minutes = (
        acceptance_time_et.hour * 60
        + acceptance_time_et.minute
        + acceptance_time_et.second / 60.0
    )
    open_minutes = int(REGULAR_OPEN[:2]) * 60 + int(REGULAR_OPEN[3:])
    close_minutes = int(REGULAR_CLOSE[:2]) * 60 + int(REGULAR_CLOSE[3:])
    if minutes < open_minutes:
        return "pre_open"
    if minutes < close_minutes:
        return "market_hours"
    return "post_close"


def _atm_straddle(options_source, ticker, quote_session, day0, spot) -> dict:
    """The ATM straddle at `strike`/`expiry` priced on `quote_session`: the
    chain as of `quote_session` filtered to expiries in
    (day0, day0 + `_EXPIRY_SEARCH_DAYS`] and strikes within `_STRIKE_BAND` of
    `spot`, then the nearest expiry strictly after `day0` and the strike
    closest to `spot`. Returns the `chain.straddle` dict plus `strike` and
    `expiry`. Raises the same ValueError/KeyError the callers catch."""
    day0_str = day0.strftime("%Y-%m-%d")
    quote_str = quote_session.strftime("%Y-%m-%d")
    option_chain = chain.as_of(
        options_source,
        ticker,
        quote_str,
        expiration_gte=(day0 + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        expiration_lte=(day0 + pd.Timedelta(days=_EXPIRY_SEARCH_DAYS)).strftime("%Y-%m-%d"),
        strike_gte=spot * (1.0 - _STRIKE_BAND),
        strike_lte=spot * (1.0 + _STRIKE_BAND),
    )
    expiry = chain.nearest_expiry_after(option_chain, day0_str)
    strike = chain.atm_strike(option_chain, spot)
    straddle = chain.straddle(options_source, option_chain, strike, expiry, quote_str)
    straddle["strike"] = strike
    straddle["expiry"] = expiry
    return straddle


def _missing_pre_straddle() -> dict:
    """Every `pre_*` column's missing value -- used when there is no raw bar
    at `pre_announcement_session` or the pre straddle can't be priced. Never
    skips the row."""
    return {
        "pre_strike": np.nan,
        "pre_expiry": pd.NaT,
        "pre_call_ticker": None,
        "pre_put_ticker": None,
        "pre_call_price": np.nan,
        "pre_put_price": np.nan,
        "pre_straddle_price": np.nan,
        "pre_spot_raw": np.nan,
        "pre_days_to_expiry": np.nan,
        "pre_call_volume": np.nan,
        "pre_put_volume": np.nan,
        "pre_is_stale": pd.NA,
    }


def _resolve_cik(edgar_source, ticker, as_of, cik_overrides, cik_resolver):
    """`(normalized_cik, label)` for one membership interval, trying
    `cik_overrides` ("explicit"), then `cik_resolver(ticker, as_of)` (a None
    return falls through), then `edgar_source.get_cik` ("sec_current")."""
    if cik_overrides and ticker in cik_overrides:
        cik, label = cik_overrides[ticker], "explicit"
    elif cik_resolver is not None:
        resolved = cik_resolver(ticker, as_of)
        if resolved is None:
            cik, label = edgar_source.get_cik(ticker), "sec_current"
        else:
            cik, label = resolved
    else:
        cik, label = edgar_source.get_cik(ticker), "sec_current"
    return str(int(cik)), label


def _events_in_intervals(events, interval_rows, start_ts, end_ts) -> pd.DataFrame:
    """Events with `start_ts <= event_date <= end_ts` whose
    `pre_announcement_session` falls inside one of `interval_rows`
    (`start_date <= session < end_date_excl`; a NaN end is an open interval)."""
    if events.empty:
        return events
    keep = (events["event_date"] >= start_ts) & (events["event_date"] <= end_ts)
    member = pd.Series(False, index=events.index)
    for row in interval_rows:
        in_interval = events["pre_announcement_session"] >= pd.Timestamp(row["start_date"])
        interval_end = row["end_date_excl"]
        if pd.notna(interval_end):
            in_interval = in_interval & (
                events["pre_announcement_session"] < pd.Timestamp(interval_end)
            )
        member = member | in_interval
    return events.loc[keep & member]


def _build_one_event(
    ticker, filing, session_index, price_source, options_source, cik, cik_source
):
    # EDGAR acceptanceDateTime is ISO UTC -- convert to New York before
    # normalizing so an evening filing on date D lands on D, not D+1.
    accepted = pd.Timestamp(filing["acceptance_datetime"])
    if accepted.tz is not None:
        accepted = accepted.tz_convert(NY_TZ)
    acceptance_time_et = (
        accepted if accepted.tz is not None else accepted.tz_localize(NY_TZ)
    )
    event_date = accepted.tz_localize(None).normalize()
    announce_timing = _announce_timing(acceptance_time_et)

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

    # A post_close filing ON a session day is announced after that day's
    # close, so day0's close precedes the news and day1's reflects it. Every
    # other filing -- including weekend/holiday ones -- is bracketed by
    # pre_session and day0.
    post_close_on_session = event_date == day0 and announce_timing == "post_close"
    pre_announcement_session = day0 if post_close_on_session else pre_session
    reaction_session = day1 if post_close_on_session else day0

    spot = float(raw_closes.loc[day0])
    try:
        straddle = _atm_straddle(options_source, ticker, day0, day0, spot)
    except (ValueError, KeyError):
        return None

    days_to_expiry = (straddle["expiry"] - day0).days
    call_volume = straddle["call_volume"]
    put_volume = straddle["put_volume"]
    is_stale = bool(call_volume == 0 or put_volume == 0)
    straddle_price = straddle["call_price"] + straddle["put_price"]

    if post_close_on_session:
        # The pre straddle priced at day0 IS the day0 straddle -- copy the
        # values instead of making a second chain/price request.
        pre = {
            "pre_strike": straddle["strike"],
            "pre_expiry": straddle["expiry"],
            "pre_call_ticker": straddle["call_ticker"],
            "pre_put_ticker": straddle["put_ticker"],
            "pre_call_price": straddle["call_price"],
            "pre_put_price": straddle["put_price"],
            "pre_straddle_price": straddle_price,
            "pre_spot_raw": spot,
            "pre_days_to_expiry": float(days_to_expiry),
            "pre_call_volume": call_volume,
            "pre_put_volume": put_volume,
            "pre_is_stale": is_stale,
        }
    else:
        pre_spot_raw = (
            float(raw_closes.loc[pre_announcement_session])
            if pre_announcement_session in raw_closes.index
            else np.nan
        )
        pre = None
        if np.isfinite(pre_spot_raw) and pre_spot_raw > 0:
            try:
                pre_straddle = _atm_straddle(
                    options_source, ticker, pre_announcement_session, day0, pre_spot_raw
                )
            except (ValueError, KeyError):
                pass  # no pre straddle -- the pre_* columns stay missing
            else:
                pre = {
                    "pre_strike": pre_straddle["strike"],
                    "pre_expiry": pre_straddle["expiry"],
                    "pre_call_ticker": pre_straddle["call_ticker"],
                    "pre_put_ticker": pre_straddle["put_ticker"],
                    "pre_call_price": pre_straddle["call_price"],
                    "pre_put_price": pre_straddle["put_price"],
                    "pre_straddle_price": (
                        pre_straddle["call_price"] + pre_straddle["put_price"]
                    ),
                    "pre_spot_raw": pre_spot_raw,
                    "pre_days_to_expiry": float(
                        (pre_straddle["expiry"] - pre_announcement_session).days
                    ),
                    "pre_call_volume": pre_straddle["call_volume"],
                    "pre_put_volume": pre_straddle["put_volume"],
                    "pre_is_stale": bool(
                        pre_straddle["call_volume"] == 0 or pre_straddle["put_volume"] == 0
                    ),
                }
        if pre is None:
            pre = _missing_pre_straddle()

    implied_move = (
        pre["pre_straddle_price"] / pre["pre_spot_raw"]
        if np.isfinite(pre["pre_straddle_price"]) and np.isfinite(pre["pre_spot_raw"])
        else np.nan
    )
    if pre_announcement_session in closes.index and reaction_session in closes.index:
        ret_reaction = float(
            closes.loc[reaction_session] / closes.loc[pre_announcement_session] - 1.0
        )
    else:
        ret_reaction = np.nan
    if np.isfinite(ret_reaction) and np.isfinite(implied_move) and implied_move > 0:
        surprise_ratio = abs(ret_reaction) / implied_move
    else:
        surprise_ratio = np.nan

    row = {
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
        "acceptance_time_et": acceptance_time_et,
        "announce_timing": announce_timing,
        "pre_announcement_session": pre_announcement_session,
        "reaction_session": reaction_session,
        "strike": straddle["strike"],
        "expiry": straddle["expiry"],
        "call_ticker": straddle["call_ticker"],
        "put_ticker": straddle["put_ticker"],
        "call_price": straddle["call_price"],
        "put_price": straddle["put_price"],
        "straddle_price": straddle_price,
        "spot_raw": spot,
        "implied_move": implied_move,
        "ret_reaction": ret_reaction,
        "surprise_ratio": surprise_ratio,
        "cik": cik,
        "cik_source": cik_source,
    }
    row.update(pre)
    return row
