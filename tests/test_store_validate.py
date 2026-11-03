"""`store.validate` flags: OHLC sanity, duplicates, calendar gaps,
unrecorded-split jumps, stale runs, per-ticker isolation."""
import numpy as np
import pandas as pd
import pytest

from data_universe.sources.fakes import FakeMarket, FakePolygonSource
from data_universe.store import (
    FLAG_COLUMNS,
    StoreBackedSource,
    validate_bars,
    validate_tickers,
)
from data_universe.store.raw import _row_dates
from data_universe.store.validate import calendar_gaps
from tests.test_store_source import CountingFake

NY = "America/New_York"


@pytest.fixture(autouse=True)
def _last(monkeypatch):
    monkeypatch.setattr(
        "data_universe.store.source._last_complete_date",
        lambda: pd.Timestamp("2024-12-31"),
    )


def _bars(dates, closes, **cols):
    index = pd.DatetimeIndex(pd.to_datetime(list(dates)), tz=NY, name="timestamp")
    n = len(index)
    data = {
        "open": cols.get("open", closes), "high": cols.get("high", closes),
        "low": cols.get("low", closes), "close": closes,
        "volume": cols.get("volume", [1000.0] * n),
        "vwap": cols.get("vwap", closes),
    }
    return pd.DataFrame(data, index=index)


def test_clean_fake_year_has_no_flags():
    source = FakePolygonSource()
    raw = source.daily_bars_raw("AAA", "2024-01-01", "2024-12-31")
    adj = source.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-12-31")
    sessions = _row_dates(source.daily_bars_raw("SPY", "2024-01-01", "2024-12-31"), None)
    out = validate_bars(raw, adj, sessions)
    assert out.empty
    assert list(out.columns) == FLAG_COLUMNS


def test_ohlc_flags_each_bad_row_only():
    dates = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
    bars = _bars(
        dates, [10.0, 10.0, -5.0, 10.0],
        high=[11.0, 9.0, 11.0, 11.0],   # day 2: high < close
        volume=[1000.0, 1000.0, 1000.0, np.nan],
    )
    out = validate_bars(bars, bars.copy())
    assert out["ohlc_bad"].sum() == 3
    flagged = set(out.index[out["ohlc_bad"]])
    assert flagged == {
        pd.Timestamp("2024-01-03"),  # high < close
        pd.Timestamp("2024-01-04"),  # non-positive close
        pd.Timestamp("2024-01-05"),  # NaN volume
    }


def test_duplicate_stamp_flags_only_that_date():
    dates = ["2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"]
    bars = _bars(dates, [10.0, 10.1, 10.2, 10.3])
    out = validate_bars(bars, bars.copy())
    assert out.loc[pd.Timestamp("2024-01-03"), "duplicate_stamp"]
    assert not out["ohlc_bad"].any()


def test_calendar_gap_flags_dropped_session_only():
    source = FakePolygonSource()
    raw = source.daily_bars_raw("AAA", "2024-01-01", "2024-01-31")
    sessions = _row_dates(source.daily_bars_raw("SPY", "2024-01-01", "2024-01-31"), None)
    dropped = _row_dates(raw, None)[5]
    out = validate_bars(raw.iloc[[i for i in range(len(raw)) if i != 5]],
                        raw.copy(), sessions)
    assert dropped in set(out.index[out["calendar_gap"]])
    # sessions outside [first, last] bar dates are never flagged
    gaps = calendar_gaps(raw, pd.DatetimeIndex(["2023-01-01", "2030-01-01"]))
    assert gaps.empty


def test_recorded_split_is_not_a_jump():
    market = FakeMarket(splits={"AAA": [("2024-03-01", 4.0)]})
    source = FakePolygonSource(market)
    raw = source.daily_bars_raw("AAA", "2024-01-01", "2024-06-30")
    adj = source.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-06-30")
    out = validate_bars(raw, adj)
    assert not out["jump_no_split"].any()


def test_unrecorded_split_flags_at_the_step_date():
    source = FakePolygonSource()
    raw = source.daily_bars_raw("AAA", "2024-01-01", "2024-06-30")
    adj = source.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-06-30")
    x = _row_dates(raw, None)[60]
    mask = _row_dates(raw, None) < x
    for frame in (raw, adj):
        for column in ("open", "high", "low", "close", "vwap"):
            frame.loc[mask, column] = frame.loc[mask, column] * 3.0
    out = validate_bars(raw, adj)
    assert list(out.index[out["jump_no_split"]]) == [x]


def test_stale_run_flags_full_runs_only():
    dates = pd.bdate_range("2024-01-01", periods=10)
    closes = [10.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 11.0, 11.0, 12.0]
    out = validate_bars(_bars(dates, closes), _bars(dates, closes))
    flagged = set(out.index[out["stale_run"]])
    assert flagged == set(pd.DatetimeIndex(dates[:5]))  # run of 5, not 4


def test_validate_tickers_isolates_failures_and_fetches_calendar_once(tmp_path):
    class Failing(CountingFake):
        def daily_bars_raw(self, ticker, start, end):
            # raise before super() so CountingFake counts it exactly once
            if ticker == "BAD":
                raise ValueError("boom")
            return super().daily_bars_raw(ticker, start, end)

    upstream = Failing()
    report = validate_tickers(upstream, ["AAA", "BAD", "BBB"], "2024-01-01", "2024-03-31")
    assert report.failed == {"BAD": "ValueError: boom"}
    assert upstream.calls[("daily_bars_raw", "SPY")] == 1  # calendar fetched once
    assert set(report.flags.columns) == {"ticker", "date", *FLAG_COLUMNS}
    assert (report.flags["ticker"] != "BAD").all()


def test_validate_through_store_backed_source_reuses_store(tmp_path):
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    validate_tickers(sbs, ["AAA"], "2024-01-01", "2024-03-31")
    second = CountingFake()
    sbs2 = StoreBackedSource(second, root=tmp_path)
    validate_tickers(sbs2, ["AAA"], "2024-01-01", "2024-03-31")
    assert second.calls == {}
