import pandas as pd

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
