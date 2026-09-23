import pandas as pd
import pytest

from data_universe.options.chain import as_of, atm_strike, nearest_expiry_after, straddle


def _chain():
    return pd.DataFrame(
        [
            {
                "ticker": "O:AAPL240621C00190000",
                "expiration_date": "2024-06-21",
                "strike_price": 190.0,
                "contract_type": "call",
            },
            {
                "ticker": "O:AAPL240621P00190000",
                "expiration_date": "2024-06-21",
                "strike_price": 190.0,
                "contract_type": "put",
            },
            {
                "ticker": "O:AAPL240621C00195000",
                "expiration_date": "2024-06-21",
                "strike_price": 195.0,
                "contract_type": "call",
            },
            {
                "ticker": "O:AAPL240719C00190000",
                "expiration_date": "2024-07-19",
                "strike_price": 190.0,
                "contract_type": "call",
            },
        ]
    )


class StubSource:
    def __init__(self, chain=None, closes=None):
        self._chain = chain
        self._closes = closes or {}

    def contracts_as_of(self, underlying, as_of_date):
        return self._chain

    def daily_close(self, option_ticker, date):
        return self._closes[(option_ticker, date)]


def test_as_of_delegates_to_source_contracts_as_of():
    stub = StubSource(chain=_chain())
    result = as_of(stub, "AAPL", "2024-01-02")
    pd.testing.assert_frame_equal(result, _chain())


def test_atm_strike_picks_closest_strike():
    assert atm_strike(_chain(), spot=191.0) == 190.0
    assert atm_strike(_chain(), spot=194.0) == 195.0


def test_atm_strike_raises_on_empty_chain():
    empty = _chain().iloc[0:0]
    with pytest.raises(ValueError):
        atm_strike(empty, spot=100.0)


def test_nearest_expiry_after_picks_first_future_expiry():
    result = nearest_expiry_after(_chain(), "2024-06-01")
    assert result == pd.Timestamp("2024-06-21")


def test_nearest_expiry_after_skips_expiries_on_or_before_date():
    result = nearest_expiry_after(_chain(), "2024-06-21")
    assert result == pd.Timestamp("2024-07-19")


def test_nearest_expiry_after_raises_when_no_future_expiry():
    with pytest.raises(ValueError):
        nearest_expiry_after(_chain(), "2024-12-31")


def test_straddle_returns_prices_and_volumes():
    stub = StubSource(
        chain=_chain(),
        closes={
            ("O:AAPL240621C00190000", "2024-06-20"): {"close": 6.0, "volume": 100.0},
            ("O:AAPL240621P00190000", "2024-06-20"): {"close": 5.5, "volume": 80.0},
        },
    )
    result = straddle(
        stub,
        _chain(),
        strike=190.0,
        expiry=pd.Timestamp("2024-06-21"),
        quote_date="2024-06-20",
    )
    assert result == {
        "call_ticker": "O:AAPL240621C00190000",
        "put_ticker": "O:AAPL240621P00190000",
        "call_price": 6.0,
        "put_price": 5.5,
        "call_volume": 100.0,
        "put_volume": 80.0,
    }


def test_straddle_raises_when_missing_a_leg():
    stub = StubSource(chain=_chain())
    with pytest.raises(ValueError):
        straddle(
            stub,
            _chain(),
            strike=999.0,
            expiry=pd.Timestamp("2024-06-21"),
            quote_date="2024-06-20",
        )
