"""30-day constant-maturity ATM implied volatility (IV30).

STUB -- owner: vrp-research data track (Lionel). Intended method, once
IV30 is needed for real: take the two listed expiries bracketing 30
calendar days from `query_date` (via `data_universe.options.chain`),
compute each one's ATM implied vol (via `chain.atm_strike` and
`data_universe.options.black_scholes.implied_vol`), convert each to total
variance (`sigma**2 * T`), linearly interpolate total variance at
`T = 30/365`, then convert back to a volatility
(`sqrt(interpolated_total_variance / T)`). This module intentionally does
not implement that yet.
"""
from __future__ import annotations


def iv30(underlying: str, query_date: str) -> float:
    """Not implemented. See module docstring for the intended method.

    Args:
        underlying: Underlying ticker symbol.
        query_date: Date "YYYY-MM-DD" to compute IV30 as of.

    Raises:
        NotImplementedError: Always. Owner: vrp-research data track (Lionel).
    """
    raise NotImplementedError(
        "IV30 is not implemented yet. Owner: vrp-research data track (Lionel). "
        "Intended method: interpolate ATM IV across the two expiries "
        "bracketing 30 days, in total variance -- see module docstring."
    )
