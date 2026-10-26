import numpy as np
import pandas as pd
import pytest

from data_universe.events.earnings import build_pead_events
from data_universe.sources.fakes import FakePolygonSource


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
    def contracts_as_of(self, underlying, as_of_date):
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
