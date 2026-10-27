"""Deterministic fake data sources for tests and for users without an API key.

`FakeMarket` generates a small, reproducible synthetic market: a market proxy
(default "SPY") daily/hourly random walk, and every other ticker's return is a
fixed linear function of the proxy's return -- `stock_return = alpha + beta *
proxy_return + noise` -- so tests can check statistics (e.g. rolling OLS beta)
against a known planted truth instead of against the code itself. The same
`seed` produces the same bars in any process (the RNG is keyed by SHA-256, not
`hash()`) and for any request window (each series is generated from `_EPOCH`
and sliced, so a wider window's bars are a strict extension of a narrower
one's -- the same date always has the same price).

Corporate actions: the generated series is the **total-return** series
(`daily_bars`). Each non-index ticker also pays a deterministic quarterly cash
dividend (`dividend_yield` of the prior close, every 63rd session with a
per-ticker phase) and may carry configured `splits`; `split_adjusted_bars`
and `raw_bars` are derived from the total-return series by taking those
actions back out, and `dividends` reports the cash amounts on the basis
`adjust.DIVIDEND_CASH_BASIS` names -- so `adjust.apply_total_return(split_adjusted,
adjust.dividend_factors(raw_close, dividends))` reproduces `daily_bars` up to a
constant scale, exactly like the real source.
"""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pandas as pd

from data_universe.sources import adjust
from data_universe.sources.base import BAR_COLUMNS, NY_TZ, filter_regular_hours

DEFAULT_SEED = 20240101
PLANTED_BETA = 1.5
PLANTED_ALPHA = 0.0
PLANTED_VIX = 16.0
DEFAULT_DIVIDEND_YIELD = 0.004  # per quarter, as a fraction of the prior close
_DIVIDEND_EVERY = 63  # sessions between ex-dates

_EPOCH = "1990-01-01"
_STAMPS_PER_SESSION = {"1d": 1, "1h": 9}
# Per-freq (loc, scale) for the market's log returns, and the noise scale for
# non-proxy tickers.
_DAILY_PARAMS = (0.0003, 0.01, 0.005)
_HOURLY_PARAMS = (0.00005, 0.003, 0.0015)


def _session_index(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    """Build a tz-aware America/New_York DatetimeIndex of business-day sessions.

    Args:
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        freq: "1d" for one stamp per session (midnight), "1h" for nine
            hourly stamps (08:00-16:00) per session, mimicking raw vendor
            aggregates (pre-open and after-hours stamps included).
    """
    if freq == "1d":
        return pd.bdate_range(start, end, tz=NY_TZ, name="timestamp")
    if freq == "1h":
        sessions = pd.bdate_range(start, end)
        hours = np.arange(8, 17)
        stamps = sessions.repeat(len(hours)) + pd.to_timedelta(
            np.tile(hours, len(sessions)), unit="h"
        )
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
        seed: RNG seed. The same seed produces the same bars in any process
            and for any request window.
        beta: Planted beta of every non-proxy ticker against `proxy_ticker`.
        alpha: Planted daily log-return alpha of every non-proxy ticker.
        proxy_ticker: The market proxy ticker (default "SPY").
        vix_level: Constant VIX close used for every session.
        dividend_yield: Quarterly cash dividend as a fraction of the prior close,
            paid by every ticker not starting with "I:". 0 disables dividends.
        splits: Optional `{ticker: [("YYYY-MM-DD", ratio), ...]}` -- e.g. `4.0` for a
            4-for-1 split effective (first split-adjusted session) on that date.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        beta: float = PLANTED_BETA,
        alpha: float = PLANTED_ALPHA,
        proxy_ticker: str = "SPY",
        vix_level: float = PLANTED_VIX,
        dividend_yield: float = DEFAULT_DIVIDEND_YIELD,
        splits: Optional[dict] = None,
    ) -> None:
        self.seed = seed
        self.beta = beta
        self.alpha = alpha
        self.proxy_ticker = proxy_ticker
        self.vix_level = vix_level
        self.dividend_yield = dividend_yield
        self.splits = splits or {}

    def _rng(self, ticker: str, freq: str) -> np.random.Generator:
        """Deterministic per-(seed, ticker, freq) generator, stable across
        processes (`hash()` would be salted differently by each process's
        PYTHONHASHSEED).
        """
        key = int.from_bytes(
            hashlib.sha256(f"{self.seed}|{ticker}|{freq}".encode()).digest()[:8], "little"
        )
        return np.random.default_rng(key)

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Deterministic daily bars. `ticker == proxy_ticker` is the market itself;
        any other ticker is `alpha + beta * proxy_return + noise`.

        Raises:
            ValueError: If `start` is before `_EPOCH`.
        """
        return self._bars(ticker, start, end, "1d")

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Deterministic hourly bars, same planted alpha/beta relationship as
        `daily_bars`. Nine start-stamped bars per session (08:00-16:00),
        mimicking raw vendor aggregates -- including a pre-open (08:00) and an
        after-hours (16:00) stamp, so `filter_regular_hours`/`hourly_bars`
        callers filter them like the real sources do.

        Raises:
            ValueError: If `start` is before `_EPOCH`.
        """
        return self._bars(ticker, start, end, "1h")

    def _bars(self, ticker: str, start: str, end: str, freq: str) -> pd.DataFrame:
        """Generate the per-(ticker, freq) sequence from `_EPOCH` through `end`
        and slice it down to `[start, end]`, so every window of the same series
        sees identical prices for a given date.
        """
        return self._slice(self._full_bars(ticker, end, freq), start, freq)

    @staticmethod
    def _slice(bars: pd.DataFrame, start: str, freq: str) -> pd.DataFrame:
        if pd.Timestamp(start) < pd.Timestamp(_EPOCH):
            raise ValueError(f"start {start!r} precedes FakeMarket's epoch {_EPOCH!r}")
        offset = np.busday_count(_EPOCH, start) * _STAMPS_PER_SESSION[freq]
        return bars.iloc[offset:]

    def _full_bars(self, ticker: str, end: str, freq: str) -> pd.DataFrame:
        """The unsliced total-return series from `_EPOCH` through `end`."""
        index = _session_index(_EPOCH, end, freq)
        loc, scale, noise_scale = _DAILY_PARAMS if freq == "1d" else _HOURLY_PARAMS
        market_returns = self._rng(self.proxy_ticker, freq).normal(loc, scale, size=len(index))
        if ticker == self.proxy_ticker:
            returns = market_returns
        else:
            noise = self._rng(ticker, freq).normal(0.0, noise_scale, size=len(index))
            returns = self.alpha + self.beta * market_returns + noise
        return _bars_from_log_returns(index, returns, start_price=100.0)

    def _ex_positions(self, ticker: str, n: int) -> np.ndarray:
        """Positions (in the epoch-based daily index) of `ticker`'s ex-dividend dates."""
        if ticker.startswith("I:") or self.dividend_yield == 0:
            return np.array([], dtype=int)
        digest = hashlib.sha256(f"{self.seed}|{ticker}|div".encode()).digest()
        phase = int.from_bytes(digest[:8], "little") % _DIVIDEND_EVERY
        positions = np.arange(phase, n, _DIVIDEND_EVERY)
        return positions[positions > 0]

    def _corporate_action_series(self, ticker: str, end: str):
        """`(split_adjusted, raw, ex_positions)` daily frames from `_EPOCH` through `end`."""
        total_return = self._full_bars(ticker, end, "1d")
        n = len(total_return)
        ex_positions = self._ex_positions(ticker, n)
        is_ex = np.zeros(n, dtype=bool)
        is_ex[ex_positions] = True
        # K(t): product of (1 - y) over ex-dates on/before t takes the dividends back out.
        dividend_factor = (1.0 - self.dividend_yield) ** np.cumsum(is_ex)
        dates = total_return.index.tz_localize(None).normalize()
        # M(t): product of split ratios effective strictly after t (present-anchored).
        split_multiplier = np.ones(n)
        for date, ratio in self.splits.get(ticker, []):
            split_multiplier[dates < pd.Timestamp(date)] *= float(ratio)

        price_columns = adjust.PRICE_COLUMNS
        split_adjusted = total_return.copy()
        split_adjusted[price_columns] = total_return[price_columns].mul(dividend_factor, axis=0)
        raw = split_adjusted.copy()
        raw[price_columns] = split_adjusted[price_columns].mul(split_multiplier, axis=0)
        raw["volume"] = split_adjusted["volume"] / split_multiplier
        return split_adjusted, raw, ex_positions

    def split_adjusted_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily bars with dividends taken out but splits still adjusted -- what the
        vendor's `adjusted=true` returns."""
        split_adjusted, _, _ = self._corporate_action_series(ticker, end)
        return self._slice(split_adjusted, start, "1d")

    def raw_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily bars as actually traded: no split or dividend adjustment; volume in
        raw shares."""
        _, raw, _ = self._corporate_action_series(ticker, end)
        return self._slice(raw, start, "1d")

    def dividends(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Cash dividends with an ex-date in `[start, end]`, `adjust.DIVIDEND_COLUMNS`.

        Each amount is `dividend_yield` times the prior session's close on the
        `adjust.DIVIDEND_CASH_BASIS` basis, so its adjustment factor is exactly
        `1 - dividend_yield`.
        """
        split_adjusted, raw, ex_positions = self._corporate_action_series(ticker, end)
        basis = raw if adjust.DIVIDEND_CASH_BASIS == "raw" else split_adjusted
        dates = split_adjusted.index.tz_localize(None).normalize()
        ex_dates = dates[ex_positions]
        amounts = self.dividend_yield * basis["close"].to_numpy()[ex_positions - 1]
        keep = (ex_dates >= pd.Timestamp(start)) & (ex_dates <= pd.Timestamp(end))
        if not keep.any():
            return adjust.empty_dividends()
        return pd.DataFrame(
            {
                "ex_dividend_date": ex_dates[keep],
                "cash_amount": amounts[keep],
                "currency": "USD",
                "dividend_type": "CD",
            }
        )[adjust.DIVIDEND_COLUMNS].reset_index(drop=True)

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
        """See `FakeMarket.daily_bars` (the total-return series)."""
        return self.market.daily_bars(ticker, start, end)

    def daily_bars_split_adjusted(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.split_adjusted_bars`."""
        return self.market.split_adjusted_bars(ticker, start, end)

    def daily_bars_raw(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.raw_bars`."""
        return self.market.raw_bars(ticker, start, end)

    def dividends(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.dividends`."""
        return self.market.dividends(ticker, start, end)

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """`FakeMarket.hourly_bars` filtered to regular hours, honoring the
        `DataSource.hourly_bars` contract like `PolygonSource` does: 7 bars
        per session, stamped 09:00-15:00 ET."""
        return filter_regular_hours(
            self.market.hourly_bars(ticker, start, end), bar_minutes=60
        )

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


class FakeRateSource:
    """A constant risk-free rate source mirroring `FredSource.daily_rate`'s
    units (decimal per-trading-day rate), for tests and users without a FRED
    key.

    Args:
        annual_rate: Decimal annualized rate (0.05 = 5%).
    """

    def __init__(self, annual_rate: float = 0.05) -> None:
        self.annual_rate = annual_rate

    def daily_rate(
        self,
        series_id: str = "DTB3",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """`annual_rate / 252` on every business day in `[start, end]`.

        Args:
            series_id: Series name to give the returned Series.
            start: Start date "YYYY-MM-DD"; None means the fake epoch.
            end: End date "YYYY-MM-DD"; None means today.

        Returns:
            A decimal daily-rate `pd.Series` indexed by tz-naive business-day
            date named "date", named `series_id` -- same units as
            `FredSource.daily_rate`.
        """
        index = pd.bdate_range(
            start or _EPOCH, end or pd.Timestamp.today().strftime("%Y-%m-%d"), name="date"
        )
        return pd.Series(self.annual_rate / 252.0, index=index, name=series_id)
