from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.polygon_options import PolygonOptionsSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_polygon_key("test-key")
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def _contract(ticker, expiration_date, strike_price, contract_type):
    return {
        "ticker": ticker,
        "expiration_date": expiration_date,
        "strike_price": strike_price,
        "contract_type": contract_type,
    }


def test_contracts_as_of_parses_results():
    session = Mock()
    session.get.return_value = FakeResponse(
        {
            "results": [
                _contract("O:AAPL240621C00190000", "2024-06-21", 190.0, "call"),
                _contract("O:AAPL240621P00190000", "2024-06-21", 190.0, "put"),
            ]
        }
    )
    source = PolygonOptionsSource(session=session)
    chain = source.contracts_as_of("AAPL", "2024-01-02")
    assert list(chain.columns) == ["ticker", "expiration_date", "strike_price", "contract_type"]
    assert len(chain) == 2


def test_contracts_as_of_sends_bearer_header():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    source.contracts_as_of("AAPL", "2024-01-02")
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_contracts_as_of_follows_next_url_pagination():
    page1 = FakeResponse(
        {
            "results": [_contract("O:AAPL240621C00190000", "2024-06-21", 190.0, "call")],
            "next_url": "https://api.polygon.io/v3/reference/options/contracts?cursor=abc",
        }
    )
    page2 = FakeResponse(
        {"results": [_contract("O:AAPL240621P00190000", "2024-06-21", 190.0, "put")]}
    )
    session = Mock()
    session.get.side_effect = [page1, page2]
    source = PolygonOptionsSource(session=session)
    # A past expiry window -> one (expired=true) query, so exactly the two pages.
    chain = source.contracts_as_of("AAPL", "2024-01-02", expiration_lte="2024-07-01")
    assert len(chain) == 2
    assert session.get.call_count == 2


def test_contracts_as_of_empty_results_has_expected_columns():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    chain = source.contracts_as_of("AAPL", "2024-01-02")
    assert chain.empty
    assert list(chain.columns) == ["ticker", "expiration_date", "strike_price", "contract_type"]


def test_daily_close_parses_close_and_volume():
    session = Mock()
    session.get.return_value = FakeResponse({"close": 5.5, "volume": 120})
    source = PolygonOptionsSource(session=session)
    result = source.daily_close("O:AAPL240621C00190000", "2024-06-20")
    assert result == {"close": 5.5, "volume": 120.0}


def test_daily_close_missing_close_is_nan_missing_volume_is_zero():
    session = Mock()
    session.get.return_value = FakeResponse({})
    source = PolygonOptionsSource(session=session)
    result = source.daily_close("O:AAPL240621C00190000", "2024-06-20")
    assert pd.isna(result["close"])
    assert result["volume"] == 0.0


def test_daily_close_404_not_found_is_nan_close_zero_volume():
    # Untraded contract/date: the endpoint answers HTTP 404 NOT_FOUND, which
    # must surface as a stale close, not an exception that kills the caller.
    session = Mock()
    session.get.return_value = FakeResponse(
        {"status": "NOT_FOUND", "message": "Data not found."}, status_code=404
    )
    source = PolygonOptionsSource(session=session)
    result = source.daily_close("O:AAPL240621C00190000", "2024-06-20")
    assert pd.isna(result["close"])
    assert result["volume"] == 0.0


def test_daily_close_reraises_non_404_http_errors():
    # 400 is not retryable (5xx now is -- see sources/http.py), so it surfaces at once.
    session = Mock()
    session.get.return_value = FakeResponse({}, status_code=400)
    source = PolygonOptionsSource(session=session)
    with pytest.raises(requests.HTTPError):
        source.daily_close("O:AAPL240621C00190000", "2024-06-20")


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    source.contracts_as_of("AAPL", "2024-01-02", expiration_lte="2024-07-01")
    source.contracts_as_of("AAPL", "2024-01-02", expiration_lte="2024-07-01")
    assert session.get.call_count == 1


# --- v0.6.0: filters, expired/active union, host -----------------------------------


def test_contracts_as_of_sends_only_the_given_filters(monkeypatch):
    monkeypatch.setattr("data_universe.sources.polygon_options._today", lambda: "2026-10-26")
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    PolygonOptionsSource(session=session).contracts_as_of(
        "AAPL", "2019-07-01", expiration_gte="2019-07-02", expiration_lte="2019-08-10",
        strike_gte=160.0, strike_lte=240.0,
    )
    assert session.get.call_count == 1
    params = session.get.call_args[1]["params"]
    assert params["expiration_date.gte"] == "2019-07-02"
    assert params["expiration_date.lte"] == "2019-08-10"
    assert params["strike_price.gte"] == 160.0
    assert params["strike_price.lte"] == 240.0
    assert params["expired"] == "true"
    assert "contract_type" not in params


def test_window_reaching_today_also_queries_unexpired_and_dedupes(monkeypatch):
    monkeypatch.setattr("data_universe.sources.polygon_options._today", lambda: "2026-10-26")
    expired = _contract("O:AAPL261023C00190000", "2026-10-23", 190.0, "call")
    listed = _contract("O:AAPL261120C00190000", "2026-11-20", 190.0, "call")

    def get(url, params=None, headers=None, timeout=None):
        if params["expired"] == "true":
            return FakeResponse({"results": [expired]})
        return FakeResponse({"results": [listed, expired]})

    session = Mock()
    session.get.side_effect = get
    chain = PolygonOptionsSource(session=session).contracts_as_of(
        "AAPL", "2026-10-20", expiration_lte="2026-11-30"
    )
    assert sorted(c[1]["params"]["expired"] for c in session.get.call_args_list) == [
        "false", "true",
    ]
    assert chain["ticker"].tolist() == [expired["ticker"], listed["ticker"]]


def test_massive_options_source_uses_the_massive_host():
    from data_universe.sources.massive import MassiveOptionsSource

    config.set_massive_key("massive-key")
    session = Mock()
    session.get.return_value = FakeResponse({"close": 1.0, "volume": 3.0})
    MassiveOptionsSource(session=session).daily_close("O:AAPL240621C00190000", "2024-06-20")
    url = session.get.call_args[0][0]
    assert url.startswith("https://api.massive.com/v1/open-close/")
    assert session.get.call_args[1]["headers"]["Authorization"] == "Bearer massive-key"


def test_polygon_options_source_honors_a_base_url_override():
    config.set_polygon_base_url("https://example.test")
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    PolygonOptionsSource(session=session).contracts_as_of(
        "AAPL", "2024-01-02", expiration_lte="2024-02-01"
    )
    assert session.get.call_args[0][0] == "https://example.test/v3/reference/options/contracts"
