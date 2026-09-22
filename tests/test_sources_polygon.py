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
    source.daily_bars("AAPL", "2024-01-01", "2024-01-05")
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "apiKey" not in (kwargs.get("params") or {})
    assert "apiKey" not in session.get.call_args[0][0]


def test_daily_bars_raises_without_a_key():
    config.set_polygon_key(None)
    source = PolygonSource(session=Mock())
    with pytest.raises(ValueError):
        source.daily_bars("AAPL", "2024-01-01", "2024-01-05")


def test_daily_bars_parses_results_into_frame_normalized_to_midnight_ny():
    ts_ms = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(
        200, {"results": [_day_agg(ts_ms, 100.0, 101.0, 99.0, 100.5, 1000.0, 100.2)]}
    )
    source = PolygonSource(session=session)
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
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
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-05")
    assert len(bars) == 2
    assert session.get.call_count == 2


def test_daily_bars_is_a_single_range_request_not_one_per_day():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]})
    source = PolygonSource(session=session)
    source.daily_bars("AAPL", "2024-01-01", "2024-01-31")
    assert session.get.call_count == 1


def test_hourly_bars_filters_to_regular_hours():
    pre_open = int(pd.Timestamp("2024-01-02 09:00:00", tz="America/New_York").timestamp() * 1000)
    in_hours = int(pd.Timestamp("2024-01-02 10:00:00", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(
        200,
        {
            "results": [
                _day_agg(pre_open, 1, 1, 1, 1, 1, 1),
                _day_agg(in_hours, 2, 2, 2, 2, 2, 2),
            ]
        },
    )
    source = PolygonSource(session=session)
    bars = source.hourly_bars("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert bars.iloc[0]["close"] == 2.0


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
    bars = source.daily_bars("AAPL", "2024-01-01", "2024-01-05")
    assert bars.empty


def test_429_retries_with_backoff_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("data_universe.sources.polygon.time.sleep", lambda s: sleeps.append(s))
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.side_effect = [
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]}),
    ]
    source = PolygonSource(session=session, backoff_seconds=0.01)
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert len(sleeps) == 2
    assert sleeps == [0.01, 0.02]  # doubles each retry


def test_429_exhausting_retries_raises_rate_limited_error(monkeypatch):
    monkeypatch.setattr("data_universe.sources.polygon.time.sleep", lambda s: None)
    session = Mock()
    session.get.return_value = FakeResponse(429)
    source = PolygonSource(session=session, max_retries=2, backoff_seconds=0.01)
    with pytest.raises(RateLimitedError):
        source.daily_bars("AAPL", "2024-01-02", "2024-01-02")


def test_repeat_call_is_served_from_cache():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]})
    source = PolygonSource(session=session)
    source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    assert session.get.call_count == 1
