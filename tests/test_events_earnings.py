import numpy as np
import pandas as pd
import pytest

from data_universe.events.earnings import EVENT_COLUMNS, V1_EVENT_COLUMNS, build_pead_events
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


class FakeEdgarSource:
    def __init__(self, cik="320193", filing_dates=None):
        self._cik = cik
        self._filing_dates = filing_dates or [
            "2024-02-01T20:15:00.000Z",
            "2024-05-02T20:15:00.000Z",
        ]

    def get_cik(self, ticker):
        return self._cik

    def earnings_8k_filings(self, cik):
        rows = [
            {
                "filing_date": pd.Timestamp(d[:10]),
                "acceptance_datetime": pd.Timestamp(d),
                "form": "8-K",
                "items": "2.02,9.01",
            }
            for d in self._filing_dates
        ]
        return pd.DataFrame(rows, columns=["filing_date", "acceptance_datetime", "form", "items"])


class FakeOptionsSource(FakePolygonSource):
    def contracts_as_of(self, underlying, as_of_date, **filters):  # filters ignored
        expiries = ["2024-02-16", "2024-03-15", "2024-05-17", "2024-06-21"]
        rows = []
        for expiry in expiries:
            for strike in (95.0, 100.0, 105.0):
                expiry_code = expiry.replace("-", "")[2:]
                strike_code = f"{int(strike*1000):08d}"
                rows.append(
                    {
                        "ticker": f"O:{underlying}{expiry_code}C{strike_code}",
                        "expiration_date": expiry,
                        "strike_price": strike,
                        "contract_type": "call",
                    }
                )
                rows.append(
                    {
                        "ticker": f"O:{underlying}{expiry_code}P{strike_code}",
                        "expiration_date": expiry,
                        "strike_price": strike,
                        "contract_type": "put",
                    }
                )
        return pd.DataFrame(rows)

    def daily_close(self, option_ticker, date):
        return {"close": 2.5, "volume": 10.0}


def test_build_pead_events_returns_one_row_per_filing_with_named_columns():
    price_source = FakePolygonSource()
    edgar_source = FakeEdgarSource()
    options_source = FakeOptionsSource()

    events = build_pead_events(
        "AAPL",
        edgar_source,
        price_source,
        options_source,
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )

    expected_columns = {
        "ticker", "event_date", "pre_session", "day0", "day1", "sigma_pre_daily",
        "adv_usd_pre20", "adv_usd_day0_day1", "days_to_expiry", "next_earnings_session",
        "call_volume", "put_volume", "is_stale",
    }
    assert expected_columns.issubset(set(events.columns))
    assert len(events) == 2
    assert (events["ticker"] == "AAPL").all()
    assert events["sigma_pre_daily"].notna().all()
    assert events["days_to_expiry"].gt(0).all()
    # next_earnings_session is the *following* filing's event date -- known only in hindsight
    assert events.iloc[0]["next_earnings_session"] == events.iloc[1]["event_date"]
    assert pd.isna(events.iloc[1]["next_earnings_session"])
    assert not events["is_stale"].iloc[0]  # fake options source always returns volume=10


def test_sigma_pre_daily_is_std_of_30_log_returns_ending_at_pre_session():
    price_source = FakePolygonSource()
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-01T20:15:00.000Z"]),
        price_source,
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 1
    row = events.iloc[0]
    bars = price_source.daily_bars("AAPL", "2023-10-01", "2024-02-05")
    closes = pd.Series(
        bars["close"].values, index=bars.index.tz_localize(None).normalize()
    )
    # 31 closes ending at pre_session -> exactly 30 log returns.
    pre_closes = closes.loc[closes.index <= row["pre_session"]].tail(31)
    expected = np.log(pre_closes / pre_closes.shift(1)).dropna().std(ddof=1)
    assert len(pre_closes) == 31
    assert row["sigma_pre_daily"] == pytest.approx(expected)


class _DayDroppingSource(FakePolygonSource):
    """Drops given dates from one ticker's daily_bars (a halt), while leaving
    the SPY calendar -- and therefore the session index -- intact."""

    def __init__(self, drop_ticker, drop_dates):
        super().__init__()
        self._drop_ticker = drop_ticker
        self._drop_dates = {pd.Timestamp(d) for d in drop_dates}

    def daily_bars(self, ticker, start, end):
        bars = super().daily_bars(ticker, start, end)
        if ticker != self._drop_ticker:
            return bars
        dates = bars.index.tz_localize(None).normalize()
        return bars.loc[~dates.isin(self._drop_dates)]


def test_event_with_missing_day1_bar_is_skipped_not_fatal():
    # 2024-02-02 is day1 for the 2024-02-01 filing; the second filing's bars
    # are intact, so only the first event drops out.
    price_source = _DayDroppingSource("AAPL", {"2024-02-02"})
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(),
        price_source,
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 1
    assert events.iloc[0]["event_date"] == pd.Timestamp("2024-05-02")


def test_evening_utc_acceptance_maps_to_prior_new_york_date():
    # 2024-02-02T01:30Z is 2024-02-01 20:30 in New York -- the event date
    # must be the New York calendar date, not the UTC one.
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-02T01:30:00.000Z"]),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 1
    assert events.iloc[0]["event_date"] == pd.Timestamp("2024-02-01")


class FakeOptionsSourceStalePuts(FakeOptionsSource):
    """The put leg never trades on day0 (untraded contract: NaN close, zero
    volume) -- the event must survive with is_stale=True rather than die."""

    def daily_close(self, option_ticker, date):
        if option_ticker[-9] == "P":  # call/put flag precedes the 8-digit strike
            return {"close": float("nan"), "volume": 0.0}
        return super().daily_close(option_ticker, date)


def test_untraded_option_contract_marks_event_stale_not_dropped():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(),
        FakePolygonSource(),
        FakeOptionsSourceStalePuts(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 2
    assert events["is_stale"].all()


# --- v0.6.0: raw spot / raw dollars / filtered chain ------------------------------


class RecordingOptionsSource(FakePolygonSource):
    """Builds a chain with one strike near the raw spot and one near the split-adjusted
    spot on the expiry after day0, honors no filters, and records every call."""

    def __init__(self, market, strikes):
        super().__init__(market)
        self._strikes = strikes
        self.chain_calls = []
        self.priced = []

    def contracts_as_of(self, underlying, as_of_date, **filters):
        self.chain_calls.append((as_of_date, filters))
        rows = []
        for strike in self._strikes:
            for kind, flag in (("call", "C"), ("put", "P")):
                rows.append(
                    {
                        "ticker": f"O:{underlying}240216{flag}{int(strike * 1000):08d}",
                        "expiration_date": "2024-02-16",
                        "strike_price": strike,
                        "contract_type": kind,
                    }
                )
        return pd.DataFrame(rows)

    def daily_close(self, option_ticker, date):
        self.priced.append(option_ticker)
        return {"close": 2.5, "volume": 10.0}


def _split_market():
    # AAPL splits 4-for-1 AFTER the 2024-02-01 event: the raw spot that day is 4x the
    # split-adjusted one, and historical option strikes are on the raw basis.
    return FakeMarket(splits={"AAPL": [("2024-03-01", 4.0)]})


def test_atm_strike_uses_raw_spot_when_a_split_happens_later():
    market = _split_market()
    source = FakePolygonSource(market)
    raw_spot = source.daily_bars_raw("AAPL", "2024-02-01", "2024-02-01")["close"].iloc[0]
    split_spot = source.daily_bars_split_adjusted("AAPL", "2024-02-01", "2024-02-01")[
        "close"
    ].iloc[0]
    assert raw_spot == pytest.approx(4.0 * split_spot)
    options = RecordingOptionsSource(market, strikes=[round(raw_spot), round(split_spot)])
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-01T20:15:00.000Z"]),
        source,
        options,
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 1
    chosen_strikes = {int(t[-8:]) / 1000 for t in options.priced}
    assert chosen_strikes == {float(round(raw_spot))}


def test_chain_is_requested_with_expiry_and_strike_filters():
    market = _split_market()
    source = FakePolygonSource(market)
    raw_spot = source.daily_bars_raw("AAPL", "2024-02-01", "2024-02-01")["close"].iloc[0]
    options = RecordingOptionsSource(market, strikes=[round(raw_spot)])
    build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-01T20:15:00.000Z"]),
        source,
        options,
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    # chain_calls[0] is day0's request; v2's pre-announcement chain call (as of
    # pre_session) follows it.
    as_of_date, filters = options.chain_calls[0]
    assert as_of_date == "2024-02-01"
    assert filters["expiration_gte"] == "2024-02-02"
    assert filters["expiration_lte"] == "2024-03-12"
    assert filters["strike_gte"] == pytest.approx(0.8 * raw_spot)
    assert filters["strike_lte"] == pytest.approx(1.2 * raw_spot)


def test_adv_usd_pre20_is_raw_dollar_volume():
    market = _split_market()
    source = FakePolygonSource(market)
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-01T20:15:00.000Z"]),
        source,
        FakeOptionsSource(market),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    raw = source.daily_bars_raw("AAPL", "2023-10-01", "2024-02-05")
    dollars = pd.Series(
        (raw["close"] * raw["volume"]).values, index=raw.index.tz_localize(None).normalize()
    )
    expected = dollars.loc[dollars.index <= row["pre_session"]].tail(20).mean()
    assert row["adv_usd_pre20"] == pytest.approx(expected)


# --- v0.7.0: event table v2 columns (appended, never row-skipping) ----------


def test_v2_columns_are_appended_after_the_v1_ones():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert list(events.columns) == EVENT_COLUMNS
    assert list(events.columns[:13]) == V1_EVENT_COLUMNS


class _EmptyEdgar(FakeEdgarSource):
    def earnings_8k_filings(self, cik):
        return pd.DataFrame(columns=["filing_date", "acceptance_datetime", "form", "items"])


def test_empty_filings_give_an_empty_frame_with_event_columns():
    events = build_pead_events(
        "AAPL",
        _EmptyEdgar(),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert events.empty
    assert list(events.columns) == EVENT_COLUMNS


def test_announce_timing_is_clock_based_on_the_et_acceptance_time():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(
            filing_dates=[
                "2024-02-01T12:00:00.000Z",  # 07:00 EST -> pre_open
                "2024-02-05T14:30:00.000Z",  # 09:30:00 EST -> market_hours
                "2024-02-06T21:00:00.000Z",  # 16:00:00 EST -> post_close
            ]
        ),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert list(events["announce_timing"]) == ["pre_open", "market_hours", "post_close"]


def test_announce_timing_uses_the_et_wall_clock_across_dst():
    # The same UTC clock time lands on different ET wall-clock times across DST:
    # 20:15Z is 15:15 EST in February (market_hours) but 16:15 EDT in May
    # (post_close).
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(
            filing_dates=["2024-02-01T20:15:00.000Z", "2024-05-02T20:15:00.000Z"]
        ),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert list(events["announce_timing"]) == ["market_hours", "post_close"]
    assert "America/New_York" in str(events["acceptance_time_et"].dtype)
    assert events["acceptance_time_et"].iloc[0] == pd.Timestamp(
        "2024-02-01T15:15:00", tz="America/New_York"
    )


def test_post_close_on_a_session_day_maps_to_day0_and_day1():
    # 2024-05-02T20:15Z is 16:15 EDT -- announced after that session's close.
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T20:15:00.000Z"]),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    assert row["event_date"] == row["day0"] == pd.Timestamp("2024-05-02")
    assert row["pre_announcement_session"] == row["day0"]
    assert row["reaction_session"] == row["day1"] == pd.Timestamp("2024-05-03")


def test_pre_open_and_market_hours_map_to_pre_session_and_day0():
    for filing_dates in (["2024-05-02T11:00:00.000Z"], ["2024-05-02T19:15:00.000Z"]):
        events = build_pead_events(
            "AAPL",
            FakeEdgarSource(filing_dates=filing_dates),
            FakePolygonSource(),
            FakeOptionsSource(),
            calendar_start="2023-10-01",
            calendar_end="2024-06-01",
        )
        row = events.iloc[0]
        assert row["pre_announcement_session"] == row["pre_session"] == pd.Timestamp(
            "2024-05-01"
        )
        assert row["reaction_session"] == row["day0"] == pd.Timestamp("2024-05-02")


def test_saturday_filing_maps_to_the_surrounding_sessions():
    # 2024-02-03 is a Saturday (the fake calendar knows weekends, not holidays).
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-03T18:00:00.000Z"]),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    assert row["event_date"] == pd.Timestamp("2024-02-03")
    assert row["day0"] == pd.Timestamp("2024-02-05")  # Monday
    assert row["pre_announcement_session"] == row["pre_session"] == pd.Timestamp(
        "2024-02-02"  # Friday
    )
    assert row["reaction_session"] == row["day0"]


class FakeOptionsSourceDistinctLegPrices(FakeOptionsSource):
    """Calls close at 3.0 (volume 9), puts at 1.5 (volume 7), so the two legs
    are distinguishable in the exported columns."""

    def daily_close(self, option_ticker, date):
        if option_ticker[-9] == "P":  # call/put flag precedes the 8-digit strike
            return {"close": 1.5, "volume": 7.0}
        return {"close": 3.0, "volume": 9.0}


def test_day0_straddle_columns_export_strike_expiry_legs_and_prices():
    price_source = FakePolygonSource()
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-02-01T20:15:00.000Z"]),
        price_source,
        FakeOptionsSourceDistinctLegPrices(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    spot = float(
        price_source.daily_bars_raw("AAPL", "2024-02-01", "2024-02-01")["close"].iloc[0]
    )
    expected_strike = min((95.0, 100.0, 105.0), key=lambda k: abs(k - spot))
    strike_code = f"{int(expected_strike * 1000):08d}"
    assert row["strike"] == expected_strike
    assert row["expiry"] == pd.Timestamp("2024-02-16")
    assert row["call_ticker"] == f"O:AAPL240216C{strike_code}"
    assert row["put_ticker"] == f"O:AAPL240216P{strike_code}"
    assert row["call_price"] == 3.0
    assert row["put_price"] == 1.5
    assert row["straddle_price"] == pytest.approx(4.5)
    assert row["spot_raw"] == pytest.approx(spot)
    assert row["call_volume"] == 9.0
    assert row["put_volume"] == 7.0


class DateRecordingOptionsSource(FakeOptionsSource):
    """Records the as_of date + filters of every chain request and the date of
    every contract close."""

    def __init__(self, market=None):
        super().__init__(market)
        self.chain_calls = []
        self.close_calls = []

    def contracts_as_of(self, underlying, as_of_date, **filters):
        self.chain_calls.append((as_of_date, filters))
        return super().contracts_as_of(underlying, as_of_date)

    def daily_close(self, option_ticker, date):
        self.close_calls.append((option_ticker, date))
        return super().daily_close(option_ticker, date)


def test_pre_straddle_is_priced_at_pre_announcement_session_for_a_pre_open_filing():
    price_source = FakePolygonSource()
    options = DateRecordingOptionsSource()
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T11:00:00.000Z"]),  # 07:00 EDT
        price_source,
        options,
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    assert row["pre_announcement_session"] == pd.Timestamp("2024-05-01")
    pre_spot = float(
        price_source.daily_bars_raw("AAPL", "2024-05-01", "2024-05-01")["close"].iloc[0]
    )

    # The day0 chain request comes first, then the pre chain as of pre_session.
    assert [call[0] for call in options.chain_calls] == ["2024-05-02", "2024-05-01"]
    _, filters = options.chain_calls[1]
    assert filters["expiration_gte"] == "2024-05-03"  # day0 + 1
    assert filters["expiration_lte"] == "2024-06-11"  # day0 + _EXPIRY_SEARCH_DAYS
    assert filters["strike_gte"] == pytest.approx(0.8 * pre_spot)
    assert filters["strike_lte"] == pytest.approx(1.2 * pre_spot)

    # Two legs priced on day0, then two legs priced on pre_session.
    assert [d for _, d in options.close_calls] == ["2024-05-02"] * 2 + ["2024-05-01"] * 2

    assert row["pre_spot_raw"] == pytest.approx(pre_spot)
    assert row["pre_expiry"] == pd.Timestamp("2024-05-17")
    assert row["pre_days_to_expiry"] == pytest.approx(16.0)
    assert row["pre_straddle_price"] == pytest.approx(5.0)
    assert row["implied_move"] == pytest.approx(5.0 / pre_spot)


def test_post_close_filing_copies_the_day0_straddle_without_extra_requests():
    options = DateRecordingOptionsSource()
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T20:15:00.000Z"]),  # 16:15 EDT
        FakePolygonSource(),
        options,
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(options.chain_calls) == 1
    row = events.iloc[0]
    for pre_col, day0_col in [
        ("pre_strike", "strike"),
        ("pre_expiry", "expiry"),
        ("pre_call_ticker", "call_ticker"),
        ("pre_put_ticker", "put_ticker"),
        ("pre_call_price", "call_price"),
        ("pre_put_price", "put_price"),
        ("pre_straddle_price", "straddle_price"),
        ("pre_spot_raw", "spot_raw"),
        ("pre_days_to_expiry", "days_to_expiry"),
        ("pre_call_volume", "call_volume"),
        ("pre_put_volume", "put_volume"),
        ("pre_is_stale", "is_stale"),
    ]:
        assert row[pre_col] == row[day0_col], pre_col


class EmptyChainOnDate(FakeOptionsSource):
    """The listed chain comes back empty on one as_of date (nothing priced
    there) -- the pre straddle must go missing, never the row."""

    def __init__(self, empty_date, market=None):
        super().__init__(market)
        self._empty_date = empty_date

    def contracts_as_of(self, underlying, as_of_date, **filters):
        if as_of_date == self._empty_date:
            return pd.DataFrame(
                columns=["ticker", "expiration_date", "strike_price", "contract_type"]
            )
        return super().contracts_as_of(underlying, as_of_date)


def test_missing_pre_straddle_keeps_the_row_with_pre_columns_missing():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T11:00:00.000Z"]),  # pre_session 05-01
        FakePolygonSource(),
        EmptyChainOnDate("2024-05-01"),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 1
    row = events.iloc[0]
    assert row["days_to_expiry"] > 0  # v1 columns intact
    assert row["spot_raw"] > 0
    assert not row["is_stale"]
    for column in [
        "pre_strike", "pre_expiry", "pre_call_ticker", "pre_put_ticker",
        "pre_call_price", "pre_put_price", "pre_straddle_price", "pre_spot_raw",
        "pre_days_to_expiry", "pre_call_volume", "pre_put_volume", "pre_is_stale",
    ]:
        assert pd.isna(row[column]), column
    assert events["pre_is_stale"].dtype == pd.BooleanDtype()
    assert np.isnan(row["implied_move"])
    assert np.isnan(row["surprise_ratio"])


class StalePutOnDate(FakeOptionsSource):
    """The put leg trades zero volume (stale close) only on one date."""

    def __init__(self, stale_date, market=None):
        super().__init__(market)
        self._stale_date = stale_date

    def daily_close(self, option_ticker, date):
        if date == self._stale_date and option_ticker[-9] == "P":
            return {"close": float("nan"), "volume": 0.0}
        return super().daily_close(option_ticker, date)


def test_stale_pre_leg_marks_pre_is_stale_nullable_boolean():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T11:00:00.000Z"]),
        FakePolygonSource(),
        StalePutOnDate("2024-05-01"),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    assert row["pre_is_stale"]  # the pre put leg traded zero volume
    assert events["pre_is_stale"].dtype == pd.BooleanDtype()
    assert not row["is_stale"]  # the day0 legs traded fine
    assert np.isnan(row["pre_straddle_price"])  # the stale leg has no close
    assert np.isnan(row["implied_move"])


def test_ret_reaction_is_the_total_return_move_between_the_two_sessions():
    price_source = FakePolygonSource()
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T11:00:00.000Z"]),
        price_source,
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    row = events.iloc[0]
    bars = price_source.daily_bars("AAPL", "2024-04-01", "2024-05-10")
    closes = pd.Series(
        bars["close"].values, index=bars.index.tz_localize(None).normalize()
    )
    expected = (
        closes.loc[pd.Timestamp("2024-05-02")] / closes.loc[pd.Timestamp("2024-05-01")]
        - 1.0
    )
    assert row["ret_reaction"] == pytest.approx(expected)
    assert row["surprise_ratio"] == pytest.approx(abs(expected) / row["implied_move"])


def test_ret_reaction_is_nan_when_the_pre_announcement_adjusted_bar_is_missing():
    # The event is kept -- a missing pre_announcement_session bar only takes
    # out ret_reaction/surprise_ratio (the pre straddle still prices on raw bars).
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(filing_dates=["2024-05-02T11:00:00.000Z"]),
        _DayDroppingSource("AAPL", {"2024-05-01"}),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert len(events) == 1
    row = events.iloc[0]
    assert np.isnan(row["ret_reaction"])
    assert np.isnan(row["surprise_ratio"])
    assert np.isfinite(row["implied_move"])


def test_cik_defaults_to_the_sec_current_lookup():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(cik="0000320193"),
        FakePolygonSource(),
        FakeOptionsSource(),
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert (events["cik"] == "320193").all()  # normalized via str(int(cik))
    assert (events["cik_source"] == "sec_current").all()


def test_explicit_cik_is_normalized_and_labeled_explicit():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(),
        FakePolygonSource(),
        FakeOptionsSource(),
        cik="0000320193",
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert (events["cik"] == "320193").all()
    assert (events["cik_source"] == "explicit").all()


def test_explicit_cik_with_a_custom_label_records_that_label():
    events = build_pead_events(
        "AAPL",
        FakeEdgarSource(),
        FakePolygonSource(),
        FakeOptionsSource(),
        cik="320193",
        cik_source="asof_lookup",
        calendar_start="2023-10-01",
        calendar_end="2024-06-01",
    )
    assert (events["cik"] == "320193").all()
    assert (events["cik_source"] == "asof_lookup").all()


def test_cik_source_without_cik_raises_value_error():
    with pytest.raises(ValueError):
        build_pead_events(
            "AAPL",
            FakeEdgarSource(),
            FakePolygonSource(),
            FakeOptionsSource(),
            cik_source="asof_lookup",
            calendar_start="2023-10-01",
            calendar_end="2024-06-01",
        )
