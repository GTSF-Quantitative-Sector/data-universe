from unittest.mock import Mock

import pytest

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
        pass


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
    chain = source.contracts_as_of("AAPL", "2024-01-02")
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


def test_daily_close_defaults_missing_fields_to_zero():
    session = Mock()
    session.get.return_value = FakeResponse({})
    source = PolygonOptionsSource(session=session)
    result = source.daily_close("O:AAPL240621C00190000", "2024-06-20")
    assert result == {"close": 0.0, "volume": 0.0}


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    source.contracts_as_of("AAPL", "2024-01-02")
    source.contracts_as_of("AAPL", "2024-01-02")
    assert session.get.call_count == 1
