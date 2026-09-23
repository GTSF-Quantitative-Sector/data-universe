"""Pure math used by registered features and labels: log returns, three
realized-volatility estimators, and rolling OLS beta/alpha. No I/O, no
cache, no config -- every function here takes and returns plain
`pd.Series`/`pd.DataFrame` so it can be unit tested without any data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def log_returns(closes: pd.Series) -> pd.Series:
    """Day-over-day log returns. First value is always NaN (no prior close)."""
    return np.log(closes / closes.shift(1))


def realized_vol_cc(closes: pd.Series, window: int = 21) -> pd.Series:
    """Annualized close-to-close realized volatility: rolling stdev of log
    returns over `window` trading days, scaled by sqrt(252).
    """
    returns = log_returns(closes)
    return returns.rolling(window).std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR)


def realized_vol_parkinson(highs: pd.Series, lows: pd.Series, window: int = 21) -> pd.Series:
    """Annualized Parkinson realized volatility from the high/low range,
    over a rolling `window` of trading days.
    """
    log_range_sq = np.log(highs / lows) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * log_range_sq.rolling(window).mean()
    return np.sqrt(variance * _TRADING_DAYS_PER_YEAR)


def realized_vol_garman_klass(
    opens: pd.Series, highs: pd.Series, lows: pd.Series, closes: pd.Series, window: int = 21
) -> pd.Series:
    """Annualized Garman-Klass realized volatility, over a rolling `window`
    of trading days.
    """
    log_hl = np.log(highs / lows) ** 2
    log_co = np.log(closes / opens) ** 2
    daily_variance = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    variance = daily_variance.rolling(window).mean()
    return np.sqrt(variance.clip(lower=0.0) * _TRADING_DAYS_PER_YEAR)


def rolling_ols_beta_alpha(
    asset_returns: pd.Series, market_returns: pd.Series, window: int
) -> tuple:
    """Rolling univariate OLS of `asset_returns` on `market_returns`:
    `asset = alpha + beta * market`.

    Args:
        asset_returns: Per-period returns of the asset.
        market_returns: Per-period returns of the market proxy, same index.
        window: Rolling window length in periods.

    Returns:
        `(beta, alpha)`, both `pd.Series` aligned to `asset_returns.index`.
    """
    covariance = asset_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var(ddof=1)
    beta = covariance / market_variance
    alpha = asset_returns.rolling(window).mean() - beta * market_returns.rolling(window).mean()
    return beta, alpha
