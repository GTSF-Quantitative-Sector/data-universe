"""Option chain helpers: listed contracts as of a date, ATM strike selection,
nearest-expiry-after selection, and straddle construction.

Every function here takes a chain `DataFrame` (columns `ticker`,
`expiration_date`, `strike_price`, `contract_type`) or a `source` object
exposing `.contracts_as_of`/`.daily_close` -- never a concrete class -- so
these helpers work with `PolygonOptionsSource` or any test double with the
same shape.
"""
from __future__ import annotations

import pandas as pd


def as_of(source, underlying: str, as_of_date: str) -> pd.DataFrame:
    """The listed option chain for `underlying` as of `as_of_date`.

    Args:
        source: An object with `.contracts_as_of(underlying, as_of_date)`.
        underlying: Underlying ticker symbol.
        as_of_date: Date "YYYY-MM-DD".

    Returns:
        A chain DataFrame (see module docstring for the schema).
    """
    return source.contracts_as_of(underlying, as_of_date)


def atm_strike(chain: pd.DataFrame, spot: float) -> float:
    """The strike closest to `spot` present in `chain`.

    Args:
        chain: A chain DataFrame.
        spot: The underlying's spot price.

    Returns:
        The closest available strike price.

    Raises:
        ValueError: If `chain` has no strikes.
    """
    strikes = chain["strike_price"].unique()
    if len(strikes) == 0:
        raise ValueError("Chain has no strikes")
    return float(min(strikes, key=lambda k: abs(k - spot)))


def nearest_expiry_after(chain: pd.DataFrame, date: str) -> pd.Timestamp:
    """The nearest expiration strictly after `date`.

    Args:
        chain: A chain DataFrame.
        date: Reference date, "YYYY-MM-DD".

    Returns:
        The nearest expiration date strictly after `date`.

    Raises:
        ValueError: If no expiration in `chain` is after `date`.
    """
    reference = pd.Timestamp(date)
    expiries = pd.to_datetime(chain["expiration_date"]).unique()
    future = sorted(e for e in expiries if pd.Timestamp(e) > reference)
    if not future:
        raise ValueError(f"No expiration in chain after {date}")
    return pd.Timestamp(future[0])


def straddle(
    source, chain: pd.DataFrame, strike: float, expiry: pd.Timestamp, quote_date: str
) -> dict:
    """The ATM straddle (one call + one put at `strike`/`expiry`) priced on `quote_date`.

    Args:
        source: An object with `.daily_close(option_ticker, date)`.
        chain: A chain DataFrame, used to find the call/put OCC tickers for
            `strike`/`expiry`.
        strike: The straddle's strike.
        expiry: The straddle's expiration date.
        quote_date: Date "YYYY-MM-DD" to price the straddle on.

    Returns:
        A dict with `call_ticker`, `put_ticker`, `call_price`, `put_price`,
        `call_volume`, `put_volume`.

    Raises:
        ValueError: If `chain` doesn't have both a call and a put at
            `strike`/`expiry`.
    """
    expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d")
    matches = chain[
        (chain["strike_price"] == strike)
        & (pd.to_datetime(chain["expiration_date"]).dt.strftime("%Y-%m-%d") == expiry_str)
    ]
    calls = matches[matches["contract_type"] == "call"]
    puts = matches[matches["contract_type"] == "put"]
    if calls.empty or puts.empty:
        raise ValueError(f"Chain missing a call/put at strike={strike}, expiry={expiry_str}")
    call_ticker = calls.iloc[0]["ticker"]
    put_ticker = puts.iloc[0]["ticker"]
    call_data = source.daily_close(call_ticker, quote_date)
    put_data = source.daily_close(put_ticker, quote_date)
    return {
        "call_ticker": call_ticker,
        "put_ticker": put_ticker,
        "call_price": call_data["close"],
        "put_price": put_data["close"],
        "call_volume": call_data["volume"],
        "put_volume": put_data["volume"],
    }
