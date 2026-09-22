"""Deterministic fake data sources for tests and for users without an API key.

`FakeMarket` generates a small, reproducible synthetic market: a market proxy
(default "SPY") daily/hourly random walk, and every other ticker's return is a
fixed linear function of the proxy's return -- `stock_return = alpha + beta *
proxy_return + noise` -- so tests can check statistics (e.g. rolling OLS beta)
against a known planted truth instead of against the code itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_universe.sources.base import BAR_COLUMNS, NY_TZ

DEFAULT_SEED = 20240101
PLANTED_BETA = 1.5
PLANTED_ALPHA = 0.0
PLANTED_VIX = 16.0


def _session_index(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    """Build a tz-aware America/New_York DatetimeIndex of business-day sessions.

    Args:
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        freq: "1d" for one stamp per session (midnight), "1h" for seven
            hourly stamps (09:00-15:00) per session.
    """
    if freq == "1d":
        return pd.bdate_range(start, end, tz=NY_TZ, name="timestamp")
    if freq == "1h":
        sessions = pd.bdate_range(start, end)
        hours = pd.timedelta_range("09:00:00", "15:00:00", freq="1h")
        stamps = [pd.Timestamp(day.date()) + h for day in sessions for h in hours]
        return pd.DatetimeIndex(stamps, name="timestamp").tz_localize(NY_TZ)
    raise ValueError(f"Unsupported freq: {freq!r}")


def _bars_from_log_returns(
    index: pd.DatetimeIndex, log_returns: np.ndarray, start_price: float
) -> pd.DataFrame:
    """Turn a series of log returns into a synthetic OHLCV bars frame."""
    closes = start_price * np.exp(np.cumsum(log_returns))
    opens = np.roll(closes, 1)
    opens[0] = start_price
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    volume = np.full(len(index), 1_000_000.0)
    vwap = (opens + closes) / 2.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "vwap": vwap,
        },
        index=index,
    )[BAR_COLUMNS]


class FakeMarket:
    """A reproducible synthetic market shared by tests across the whole package.

    Args:
        seed: RNG seed. The same seed always produces the same bars.
        beta: Planted beta of every non-proxy ticker against `proxy_ticker`.
        alpha: Planted daily log-return alpha of every non-proxy ticker.
        proxy_ticker: The market proxy ticker (default "SPY").
        vix_level: Constant VIX close used for every session.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        beta: float = PLANTED_BETA,
        alpha: float = PLANTED_ALPHA,
        proxy_ticker: str = "SPY",
        vix_level: float = PLANTED_VIX,
    ) -> None:
        self.seed = seed
        self.beta = beta
        self.alpha = alpha
        self.proxy_ticker = proxy_ticker
        self.vix_level = vix_level

    def _rng(self, ticker: str, freq: str) -> np.random.Generator:
        """Deterministic per-(seed, ticker, freq) generator, so repeated calls agree."""
        key = abs(hash((self.seed, ticker, freq))) % (2**32)
        return np.random.default_rng(key)

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Deterministic daily bars. `ticker == proxy_ticker` is the market itself;
        any other ticker is `alpha + beta * proxy_return + noise`.
        """
        index = _session_index(start, end, "1d")
        n = len(index)
        market_returns = self._rng(self.proxy_ticker, "1d").normal(0.0003, 0.01, size=n)
        if ticker == self.proxy_ticker:
            returns = market_returns
        else:
            noise = self._rng(ticker, "1d").normal(0.0, 0.005, size=n)
            returns = self.alpha + self.beta * market_returns + noise
        return _bars_from_log_returns(index, returns, start_price=100.0)

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Deterministic hourly bars, same planted alpha/beta relationship as `daily_bars`.
        Includes one pre-open (09:00) stamp per session so `filter_regular_hours` has
        something real to filter in tests.
        """
        index = _session_index(start, end, "1h")
        n = len(index)
        market_returns = self._rng(self.proxy_ticker, "1h").normal(0.00005, 0.003, size=n)
        if ticker == self.proxy_ticker:
            returns = market_returns
        else:
            noise = self._rng(ticker, "1h").normal(0.0, 0.0015, size=n)
            returns = self.alpha + self.beta * market_returns + noise
        return _bars_from_log_returns(index, returns, start_price=100.0)

    def vix_closes(self, start: str, end: str) -> pd.Series:
        """Constant VIX close series (`vix_level`) over the given session range."""
        index = _session_index(start, end, "1d")
        return pd.Series(self.vix_level, index=index, name="close")


class FakePolygonSource:
    """A `DataSource`-shaped wrapper around `FakeMarket`, for tests that want an
    object with source-style methods instead of calling the generator directly.

    Args:
        market: The `FakeMarket` to delegate to. Defaults to `FakeMarket()`.
    """

    def __init__(self, market: Optional[FakeMarket] = None) -> None:
        self.market = market or FakeMarket()

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.daily_bars`."""
        return self.market.daily_bars(ticker, start, end)

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.hourly_bars`."""
        return self.market.hourly_bars(ticker, start, end)

    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fake index aggregates. Only "I:VIX" is supported.

        Args:
            ticker: Must be "I:VIX".
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Raises:
            ValueError: If `ticker` is not "I:VIX".
        """
        if ticker != "I:VIX":
            raise ValueError(f"FakePolygonSource only fakes I:VIX, got {ticker!r}")
        closes = self.market.vix_closes(start, end)
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": pd.Series(0.0, index=closes.index),
                "vwap": closes,
            }
        )[BAR_COLUMNS]
