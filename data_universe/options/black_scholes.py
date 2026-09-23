"""Black-Scholes pricing, vega, and implied volatility for European options.

Uses a continuous dividend yield `q`. `implied_vol` uses a bracketed solver
(Brent's method) rather than Newton alone, and returns NaN instead of
raising when the target price sits outside the model's no-arbitrage bounds
or a volatility root can't be bracketed in `[1e-6, 5.0]` (0.0001% to 500%
annualized vol -- wide enough for any real quote).
"""
from __future__ import annotations

import math

from scipy import optimize, stats

_MIN_SIGMA = 1e-6
_MAX_SIGMA = 5.0


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes price of a European option.

    Args:
        S: Spot price of the underlying, in dollars.
        K: Strike price, in dollars.
        T: Time to expiry, in years.
        r: Continuously-compounded risk-free rate (decimal, annualized).
        q: Continuously-compounded dividend yield (decimal, annualized).
        sigma: Annualized volatility (decimal).
        is_call: True for a call, False for a put.

    Returns:
        The option's theoretical price, in dollars.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if is_call:
        return S * math.exp(-q * T) * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(
            d2
        )
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * math.exp(-q * T) * stats.norm.cdf(-d1)


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega: sensitivity of price to a 1.0 (100 percentage
    point) change in `sigma`. Identical formula for calls and puts.

    Args: see `price` (vega doesn't depend on `is_call`).

    Returns:
        Vega, in dollars per 1.0 change in `sigma`.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * stats.norm.pdf(d1) * math.sqrt(T)


def _no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float, is_call: bool):
    discounted_s = S * math.exp(-q * T)
    discounted_k = K * math.exp(-r * T)
    if is_call:
        return max(discounted_s - discounted_k, 0.0), discounted_s
    return max(discounted_k - discounted_s, 0.0), discounted_k


def implied_vol(
    target_price: float, S: float, K: float, T: float, r: float, q: float, is_call: bool
) -> float:
    """Solve for the annualized volatility that reprices `target_price`,
    using a bracketed root finder (Brent's method).

    Args:
        target_price: Observed option price, in dollars.
        S: Spot price of the underlying, in dollars.
        K: Strike price, in dollars.
        T: Time to expiry, in years.
        r: Continuously-compounded risk-free rate (decimal, annualized).
        q: Continuously-compounded dividend yield (decimal, annualized).
        is_call: True for a call, False for a put.

    Returns:
        The implied volatility (decimal, annualized), or `float("nan")` if
        `target_price` is outside the model's no-arbitrage bounds, `T <= 0`,
        `target_price` isn't finite, or no root can be bracketed.
    """
    if T <= 0 or not math.isfinite(target_price):
        return float("nan")
    lower, upper = _no_arbitrage_bounds(S, K, T, r, q, is_call)
    if target_price < lower or target_price > upper:
        return float("nan")

    def objective(sigma: float) -> float:
        return price(S, K, T, r, q, sigma, is_call) - target_price

    f_lo = objective(_MIN_SIGMA)
    f_hi = objective(_MAX_SIGMA)
    if f_lo * f_hi > 0:
        return float("nan")
    try:
        return optimize.brentq(objective, _MIN_SIGMA, _MAX_SIGMA)
    except ValueError:
        return float("nan")
