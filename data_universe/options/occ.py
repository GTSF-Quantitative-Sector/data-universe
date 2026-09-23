"""OCC-style option ticker parsing and building (Polygon's convention, e.g.
"O:AAPL240621C00190000").

Format: `O:{underlying}{YYMMDD}{C|P}{strike * 1000, zero-padded to 8 digits}`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_PATTERN = re.compile(
    r"^O:(?P<underlying>[A-Z]+)(?P<expiry>\d{6})(?P<type>[CP])(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class OccContract:
    """A parsed OCC option contract ticker.

    Attributes:
        underlying: Underlying ticker symbol.
        expiry: Expiration date.
        is_call: True for a call, False for a put.
        strike: Strike price in dollars (may have fractional cents, e.g. 190.5).
    """

    underlying: str
    expiry: date
    is_call: bool
    strike: float


def parse(ticker: str) -> OccContract:
    """Parse an OCC-style option ticker, e.g. "O:AAPL240621C00190000".

    Args:
        ticker: OCC ticker, prefixed with "O:" (Polygon's convention).

    Returns:
        The parsed `OccContract`.

    Raises:
        ValueError: If `ticker` doesn't match the expected format.
    """
    match = _PATTERN.match(ticker)
    if not match:
        raise ValueError(f"Not a valid OCC option ticker: {ticker!r}")
    expiry = date(
        2000 + int(match["expiry"][0:2]),
        int(match["expiry"][2:4]),
        int(match["expiry"][4:6]),
    )
    strike = int(match["strike"]) / 1000.0
    return OccContract(
        underlying=match["underlying"],
        expiry=expiry,
        is_call=match["type"] == "C",
        strike=strike,
    )


def build(underlying: str, expiry: date, is_call: bool, strike: float) -> str:
    """Build an OCC-style option ticker from its parts.

    Args:
        underlying: Underlying ticker symbol.
        expiry: Expiration date.
        is_call: True for a call, False for a put.
        strike: Strike price in dollars (supports fractional cents, rounded
            to the nearest tenth of a cent -- OCC's own granularity).

    Returns:
        The OCC ticker string, e.g. "O:AAPL240621C00190000".
    """
    expiry_str = expiry.strftime("%y%m%d")
    type_char = "C" if is_call else "P"
    strike_int = round(strike * 1000)
    strike_str = f"{strike_int:08d}"
    return f"O:{underlying}{expiry_str}{type_char}{strike_str}"
