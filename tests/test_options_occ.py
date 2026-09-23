from datetime import date

import pytest

from data_universe.options.occ import OccContract, build, parse


def test_parse_known_ticker():
    contract = parse("O:AAPL240621C00190000")
    assert contract == OccContract(
        underlying="AAPL", expiry=date(2024, 6, 21), is_call=True, strike=190.0
    )


def test_parse_put():
    contract = parse("O:AAPL240621P00190000")
    assert contract.is_call is False


def test_parse_decimal_strike():
    contract = parse("O:AAPL240621C00190500")
    assert contract.strike == pytest.approx(190.5)


def test_parse_rejects_malformed_ticker():
    with pytest.raises(ValueError):
        parse("AAPL240621C00190000")  # missing "O:" prefix


def test_build_known_contract():
    ticker = build("AAPL", date(2024, 6, 21), is_call=True, strike=190.0)
    assert ticker == "O:AAPL240621C00190000"


def test_build_decimal_strike():
    ticker = build("AAPL", date(2024, 6, 21), is_call=False, strike=7.25)
    assert ticker == "O:AAPL240621P00007250"


@pytest.mark.parametrize(
    "underlying,expiry,is_call,strike",
    [
        ("AAPL", date(2024, 6, 21), True, 190.0),
        ("SPY", date(2025, 1, 17), False, 450.5),
        ("MSFT", date(2023, 12, 15), True, 7.25),
        ("TSLA", date(2026, 3, 20), False, 1000.125),
    ],
)
def test_round_trip(underlying, expiry, is_call, strike):
    ticker = build(underlying, expiry, is_call, strike)
    contract = parse(ticker)
    assert contract.underlying == underlying
    assert contract.expiry == expiry
    assert contract.is_call == is_call
    assert contract.strike == pytest.approx(strike, abs=0.001)
