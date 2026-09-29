from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.base import RateLimitedError
from data_universe.sources.massive import TICKER_COLUMNS, MassiveSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_massive_key("massive-key")
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


def _ticker(symbol, exchange="XNAS", ticker_type="CS", name=None):
    return {
        "ticker": symbol,
        "name": name or f"{symbol} Inc.",
        "market": "stocks",
        "type": ticker_type,
        "primary_exchange": exchange,
        "active": True,
        "cik": "0000000001",
        "composite_figi": f"BBG{symbol}",
        "share_class_figi": f"BBGS{symbol}",
        "currency_name": "usd",
        "last_updated_utc": "2026-09-01T00:00:00Z",
        "locale": "us",  # an extra field the frame should ignore
    }


def test_defaults_to_the_massive_host():
    assert config.get_massive_base_url() == "https://api.massive.com"
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    MassiveSource(session=session).list_tickers()
    assert session.get.call_args[0][0] == "https://api.massive.com/v3/reference/tickers"


def test_sends_bearer_header_not_url_param():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    MassiveSource(session=session).list_tickers()
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer massive-key"
    assert "apiKey" not in (kwargs.get("params") or {})
    assert "apiKey" not in session.get.call_args[0][0]


def test_falls_back_to_the_polygon_key():
    config.set_massive_key(None)
    config.set_polygon_key("polygon-key")
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    MassiveSource(session=session).list_tickers()
    assert session.get.call_args[1]["headers"]["Authorization"] == "Bearer polygon-key"


def test_massive_key_wins_over_the_polygon_key():
    config.set_polygon_key("polygon-key")
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    MassiveSource(session=session).list_tickers()
    assert session.get.call_args[1]["headers"]["Authorization"] == "Bearer massive-key"


def test_raises_without_any_key():
    config.set_massive_key(None)
    with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
        MassiveSource(session=Mock()).list_tickers()


def test_env_var_seeds_the_key(monkeypatch):
    monkeypatch.setenv("MASSIVE_API_KEY", "from-env")
    monkeypatch.setenv("MASSIVE_BASE_URL", "https://example.test")
    config.reset()
    assert config.get_massive_key() == "from-env"
    assert config.get_massive_base_url() == "https://example.test"


def test_list_tickers_sends_filters_and_parses_columns():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_ticker("AAPL")]})
    frame = MassiveSource(session=session).list_tickers(
        market="stocks", ticker_type="CS", exchange="XNAS", active=True
    )
    params = session.get.call_args[1]["params"]
    assert params["market"] == "stocks"
    assert params["type"] == "CS"
    assert params["exchange"] == "XNAS"
    assert params["active"] == "true"
    assert params["limit"] == 1000
    assert list(frame.columns) == TICKER_COLUMNS
    assert frame.iloc[0]["ticker"] == "AAPL"
    assert frame.iloc[0]["primary_exchange"] == "XNAS"


def test_list_tickers_omits_unset_filters():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    MassiveSource(session=session).list_tickers()
    params = session.get.call_args[1]["params"]
    assert "type" not in params
    assert "exchange" not in params


def test_list_tickers_follows_next_url_and_drops_params_on_later_pages():
    next_url = "https://api.massive.com/v3/reference/tickers?cursor=abc"
    session = Mock()
    session.get.side_effect = [
        FakeResponse(200, {"results": [_ticker("AAA")], "next_url": next_url}),
        FakeResponse(200, {"results": [_ticker("BBB")]}),
    ]
    frame = MassiveSource(session=session).list_tickers()
    assert frame["ticker"].tolist() == ["AAA", "BBB"]
    assert session.get.call_count == 2
    assert session.get.call_args_list[1][0][0] == next_url
    assert session.get.call_args_list[1][1]["params"] is None


def test_list_tickers_empty_returns_typed_empty_frame():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    frame = MassiveSource(session=session).list_tickers()
    assert frame.empty
    assert list(frame.columns) == TICKER_COLUMNS


def test_list_tickers_missing_field_is_missing_not_an_error():
    thin = {"ticker": "XYZ", "name": "XYZ Corp"}
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [thin]})
    frame = MassiveSource(session=session).list_tickers()
    assert pd.isna(frame.iloc[0]["cik"])


def test_list_tickers_is_cached():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_ticker("AAPL")]})
    source = MassiveSource(session=session)
    source.list_tickers(market="stocks")
    source.list_tickers(market="stocks")
    assert session.get.call_count == 1


def test_ticker_details_returns_results_object():
    session = Mock()
    session.get.return_value = FakeResponse(
        200, {"results": {"ticker": "AAPL", "primary_exchange": "XNAS"}, "status": "OK"}
    )
    details = MassiveSource(session=session).ticker_details("AAPL")
    assert details["primary_exchange"] == "XNAS"
    assert session.get.call_args[0][0] == "https://api.massive.com/v3/reference/tickers/AAPL"


def test_inherited_daily_bars_use_the_massive_host():
    ts_ms = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(
        200, {"results": [{"t": ts_ms, "o": 1, "h": 2, "l": 1, "c": 2, "v": 10, "vw": 1.5}]}
    )
    bars = MassiveSource(session=session).daily_bars("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    url = session.get.call_args[0][0]
    assert url.startswith("https://api.massive.com/v2/aggs/ticker/AAPL/range/1/day/")


def test_inherited_429_retry_then_rate_limited_error(monkeypatch):
    monkeypatch.setattr("data_universe.sources.polygon.time.sleep", lambda s: None)
    session = Mock()
    session.get.return_value = FakeResponse(429)
    source = MassiveSource(session=session, max_retries=2, backoff_seconds=0.01)
    with pytest.raises(RateLimitedError):
        source.list_tickers()
