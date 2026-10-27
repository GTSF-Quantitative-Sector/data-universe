from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.base import RateLimitedError
from data_universe.sources.polygon import PolygonSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_polygon_key("test-key")
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 429:
            raise RuntimeError(f"HTTP {self.status_code}")


def _day_agg(day_ms, o, h, lo, c, v, vw):
    return {"t": day_ms, "o": o, "h": h, "l": lo, "c": c, "v": v, "vw": vw}


def test_daily_bars_sends_bearer_header_not_url_param():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    source = PolygonSource(session=session)
    source.daily_bars_split_adjusted("AAPL", "2024-01-01", "2024-01-05")
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "apiKey" not in (kwargs.get("params") or {})
    assert "apiKey" not in session.get.call_args[0][0]


def test_daily_bars_raises_without_a_key():
    config.set_polygon_key(None)
    source = PolygonSource(session=Mock())
    with pytest.raises(ValueError):
        source.daily_bars_split_adjusted("AAPL", "2024-01-01", "2024-01-05")


def test_daily_bars_parses_results_into_frame_normalized_to_midnight_ny():
    ts_ms = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(
        200, {"results": [_day_agg(ts_ms, 100.0, 101.0, 99.0, 100.5, 1000.0, 100.2)]}
    )
    source = PolygonSource(session=session)
    bars = source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert bars.index[0] == pd.Timestamp("2024-01-02", tz="America/New_York")
    assert bars.iloc[0]["close"] == 100.5


def test_daily_bars_follows_next_url_pagination():
    ts1 = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    ts2 = int(pd.Timestamp("2024-01-03", tz="America/New_York").timestamp() * 1000)
    page1 = FakeResponse(
        200,
        {
            "results": [_day_agg(ts1, 1, 1, 1, 1, 1, 1)],
            "next_url": (
                "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/"
                "2024-01-03/2024-01-05?cursor=abc"
            ),
        },
    )
    page2 = FakeResponse(200, {"results": [_day_agg(ts2, 2, 2, 2, 2, 2, 2)]})
    session = Mock()
    session.get.side_effect = [page1, page2]
    source = PolygonSource(session=session)
    bars = source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-05")
    assert len(bars) == 2
    assert session.get.call_count == 2


def test_daily_bars_is_a_single_range_request_not_one_per_day():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]})
    source = PolygonSource(session=session)
    source.daily_bars_split_adjusted("AAPL", "2024-01-01", "2024-01-31")
    assert session.get.call_count == 1


def test_hourly_bars_filters_to_regular_hours():
    def _stamp(hhmm):
        return int(pd.Timestamp(f"2024-01-02 {hhmm}", tz="America/New_York").timestamp() * 1000)

    session = Mock()
    session.get.return_value = FakeResponse(
        200,
        {
            "results": [
                _day_agg(_stamp("08:00:00"), 1, 1, 1, 1, 1, 1),
                _day_agg(_stamp("09:00:00"), 2, 2, 2, 2, 2, 2),
                _day_agg(_stamp("15:00:00"), 3, 3, 3, 3, 3, 3),
                _day_agg(_stamp("16:00:00"), 4, 4, 4, 4, 4, 4),
            ]
        },
    )
    source = PolygonSource(session=session)
    bars = source.hourly_bars("AAPL", "2024-01-02", "2024-01-02")
    # Start-stamped overlap: 08:00 fully pre-open and 16:00 fully after-hours
    # are dropped; 09:00 (holds 9:30-10:00) and 15:00 (holds 15:00-16:00) kept.
    assert bars["close"].tolist() == [2.0, 3.0]


def test_index_daily_bars_for_vix():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 16, 16, 16, 16, 0, 16)]})
    source = PolygonSource(session=session)
    bars = source.index_daily_bars("I:VIX", "2024-01-02", "2024-01-02")
    assert bars.iloc[0]["close"] == 16.0
    assert "I:VIX" in session.get.call_args[0][0]


def test_empty_results_returns_empty_bars_frame():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    source = PolygonSource(session=session)
    bars = source.daily_bars_split_adjusted("AAPL", "2024-01-01", "2024-01-05")
    assert bars.empty


def test_429_retries_with_backoff_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("data_universe.sources.http.time.sleep", lambda s: sleeps.append(s))
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.side_effect = [
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]}),
    ]
    source = PolygonSource(session=session, backoff_seconds=0.01)
    bars = source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert len(sleeps) == 2
    assert sleeps == [0.01, 0.02]  # doubles each retry


def test_429_exhausting_retries_raises_rate_limited_error(monkeypatch):
    monkeypatch.setattr("data_universe.sources.http.time.sleep", lambda s: None)
    session = Mock()
    session.get.return_value = FakeResponse(429)
    source = PolygonSource(session=session, max_retries=2, backoff_seconds=0.01)
    with pytest.raises(RateLimitedError):
        source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")


def test_repeat_call_is_served_from_cache():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]})
    source = PolygonSource(session=session)
    source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    assert session.get.call_count == 1


# --- v0.6.0: total-return composition, retries, entitlement ------------------------


def _ms(day, hhmm="00:00"):
    return int(pd.Timestamp(f"{day} {hhmm}", tz="America/New_York").timestamp() * 1000)


def _routing_session(raw_closes, split_closes=None, dividends=(), hourly=()):
    """A Mock session answering aggs (by the `adjusted` param) and dividends requests."""
    split_closes = split_closes or raw_closes

    def get(url, params=None, headers=None, timeout=None):
        if "/v3/reference/dividends" in url:
            return FakeResponse(200, {"results": list(dividends)})
        if "/range/1/hour/" in url:
            return FakeResponse(200, {"results": list(hourly)})
        closes = raw_closes if params["adjusted"] == "false" else split_closes
        return FakeResponse(
            200, {"results": [_day_agg(_ms(d), c, c, c, c, 1000.0, c) for d, c in closes]}
        )

    session = Mock()
    session.get.side_effect = get
    return session


def test_daily_bars_is_total_return_adjusted_across_an_ex_dividend_date():
    raw = [("2024-01-02", 100.0), ("2024-01-03", 102.0), ("2024-01-04", 101.0)]
    dividend = {"ex_dividend_date": "2024-01-04", "cash_amount": 1.0, "currency": "USD"}
    source = PolygonSource(session=_routing_session(raw, dividends=[dividend]))
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-04")
    factor = 1.0 - 1.0 / 102.0
    assert bars["close"].tolist() == pytest.approx([100.0 * factor, 102.0 * factor, 101.0])
    # Return across the ex-date is close_e / (close_{e-1} - D): here 101 / 101 = 0.
    assert bars["close"].iloc[2] / bars["close"].iloc[1] == pytest.approx(101.0 / 101.0)
    assert bars["volume"].tolist() == [1000.0] * 3  # volume never dividend-adjusted


def test_daily_bars_issues_split_adjusted_raw_and_dividends_requests_once():
    raw = [("2024-01-02", 100.0), ("2024-01-03", 102.0)]
    session = _routing_session(raw)
    source = PolygonSource(session=session)
    source.daily_bars("AAPL", "2024-01-02", "2024-01-03")
    calls = session.get.call_args_list
    assert len(calls) == 3
    adjusted_flags = sorted(
        c[1]["params"]["adjusted"] for c in calls if "/v2/aggs/" in c[0][0]
    )
    assert adjusted_flags == ["false", "true"]
    dividend_call = next(c for c in calls if "/v3/reference/dividends" in c[0][0])
    assert dividend_call[1]["params"]["ticker"] == "AAPL"
    source.daily_bars("AAPL", "2024-01-02", "2024-01-03")
    assert session.get.call_count == 3  # every primitive served from cache


def test_daily_bars_raw_sends_adjusted_false():
    session = _routing_session([("2024-01-02", 400.0)], split_closes=[("2024-01-02", 100.0)])
    source = PolygonSource(session=session)
    assert source.daily_bars_raw("AAPL", "2024-01-02", "2024-01-02")["close"].iloc[0] == 400.0
    assert session.get.call_args[1]["params"]["adjusted"] == "false"


def test_dividends_are_filtered_to_the_window_locally():
    dividends = [
        {"ex_dividend_date": "2023-12-01", "cash_amount": 0.5, "currency": "USD"},
        {"ex_dividend_date": "2024-01-03", "cash_amount": 0.25, "currency": "USD"},
    ]
    source = PolygonSource(session=_routing_session([], dividends=dividends))
    frame = source.dividends("AAPL", "2024-01-01", "2024-01-31")
    assert frame["ex_dividend_date"].tolist() == [pd.Timestamp("2024-01-03")]
    assert frame["cash_amount"].tolist() == [0.25]


def test_hourly_bars_use_the_same_daily_dividend_factors():
    raw = [("2024-01-02", 100.0), ("2024-01-03", 102.0), ("2024-01-04", 101.0)]
    dividend = {"ex_dividend_date": "2024-01-04", "cash_amount": 1.0, "currency": "USD"}
    hourly = [
        _day_agg(_ms("2024-01-03", "10:00"), 50, 50, 50, 50, 1, 50),
        _day_agg(_ms("2024-01-04", "10:00"), 60, 60, 60, 60, 1, 60),
    ]
    source = PolygonSource(session=_routing_session(raw, dividends=[dividend], hourly=hourly))
    bars = source.hourly_bars("AAPL", "2024-01-02", "2024-01-04")
    assert bars["close"].tolist() == pytest.approx([50.0 * (1.0 - 1.0 / 102.0), 60.0])


def test_5xx_and_connection_errors_are_retried(monkeypatch):
    import requests

    sleeps = []
    monkeypatch.setattr("data_universe.sources.http.time.sleep", lambda s: sleeps.append(s))
    session = Mock()
    session.get.side_effect = [
        FakeResponse(503),
        requests.ConnectionError("boom"),
        FakeResponse(200, {"results": [_day_agg(_ms("2024-01-02"), 1, 1, 1, 1, 1, 1)]}),
    ]
    source = PolygonSource(session=session, backoff_seconds=0.01)
    bars = source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert sleeps == [0.01, 0.02]


def test_persistent_5xx_raises_data_source_error_without_query_string(monkeypatch):
    from data_universe.sources.base import DataSourceError

    monkeypatch.setattr("data_universe.sources.http.time.sleep", lambda s: None)
    session = Mock()
    session.get.return_value = FakeResponse(502)
    source = PolygonSource(session=session, max_retries=2)
    with pytest.raises(DataSourceError) as info:
        source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    assert "HTTP 502" in str(info.value)
    assert "?" not in str(info.value) and "adjusted" not in str(info.value)


@pytest.mark.parametrize("status", [401, 403])
def test_401_and_403_raise_entitlement_error_without_retrying(status):
    from data_universe.sources.base import EntitlementError

    session = Mock()
    session.get.return_value = FakeResponse(status)
    source = PolygonSource(session=session)
    with pytest.raises(EntitlementError):
        source.daily_bars_split_adjusted("AAPL", "2024-01-02", "2024-01-02")
    assert session.get.call_count == 1
