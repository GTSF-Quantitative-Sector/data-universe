"""Polygon.io stock and index aggregates, and stock dividends.

Synchronous, using a pooled `requests.Session` (unlike sec_parser's `async`
aiohttp calls). TLS verification stays on -- never pass `verify=False`. The
API key is sent as `Authorization: Bearer <key>`, never as a URL/query
parameter, so it never shows up in logs or tracebacks. One range request is
issued per (ticker, window) call, following `next_url` pagination rather
than paging by day.

Transient failures (HTTP 429/5xx, connection errors, timeouts) are retried with
doubling backoff by `sources.http.get_with_retries`; HTTP 401/403 raise
`EntitlementError` (bad key or a dataset the plan doesn't include).

Adjustment: the vendor's `adjusted=true` is **split-only**. `daily_bars` and
`hourly_bars` are total-return (split + cash dividend) adjusted here, from three
cached primitives -- `daily_bars_split_adjusted` (`adjusted=true`),
`daily_bars_raw` (`adjusted=false`) and `dividends` -- via `sources/adjust.py`
(window-end anchored; see that module). The composites themselves are not cached.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from data_universe import config
from data_universe.cache import cached
from data_universe.sources import adjust
from data_universe.sources.base import (
    BAR_COLUMNS,
    NY_TZ,
    DataSource,
    EntitlementError,
    empty_bars,
    filter_regular_hours,
)
from data_universe.sources.http import get_with_retries, url_path


class PolygonSource(DataSource):
    """Stock and index aggregates and stock dividends from Polygon.io.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a
            fresh pooled session.
        max_retries: Max retries of a transient failure (429/5xx/connection
            error/timeout) before raising.
        backoff_seconds: Base backoff in seconds, doubled on each retry.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._session = session or requests.Session()
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def _base_url(self) -> str:
        """REST base URL for this source. Subclasses override this to target another host."""
        return config.get_polygon_base_url()

    def _headers(self) -> dict:
        key = config.get_polygon_key()
        if key is None:
            raise ValueError("Polygon API key not set. Call config.set_polygon_key() first.")
        return {"Authorization": f"Bearer {key}"}

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """GET a URL with retries; raise `EntitlementError` on 401/403 and
        `requests.HTTPError` (via `raise_for_status`) on any other error status."""
        response = get_with_retries(
            self._session,
            url,
            params=params,
            headers=self._headers(),
            timeout=30,
            max_retries=self._max_retries,
            backoff_seconds=self._backoff_seconds,
        )
        if response.status_code in (401, 403):
            raise EntitlementError(
                f"{type(self).__name__}: HTTP {response.status_code} for {url_path(url)} -- "
                "API key missing/invalid, or the plan doesn't include this dataset"
            )
        response.raise_for_status()
        return response.json()

    def _paginate(self, url: str, params: Optional[dict]) -> list:
        """Every `results` row across `next_url` pages."""
        rows: list = []
        while url:
            data = self._get(url, params=params)
            rows.extend(data.get("results") or [])
            url = data.get("next_url")
            params = None  # next_url already carries its own query string
        return rows

    def _aggs(
        self,
        ticker: str,
        start: str,
        end: str,
        timespan: str,
        normalize: bool,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        """Fetch one (ticker, window) range request's worth of aggregates,
        following `next_url` pagination. `adjusted=True` is the vendor's
        split-only adjustment; `adjusted=False` is raw.
        """
        base = self._base_url()
        url = f"{base}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}"
        params = {
            "adjusted": "true" if adjusted else "false",
            "sort": "asc",
            "limit": 50000,
        }
        rows = self._paginate(url, params)
        if not rows:
            return empty_bars()
        index = pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True).tz_convert(NY_TZ)
        if normalize:
            index = index.normalize()
        index = index.set_names("timestamp")
        frame = pd.DataFrame(
            {
                "open": [float(r["o"]) for r in rows],
                "high": [float(r["h"]) for r in rows],
                "low": [float(r["l"]) for r in rows],
                "close": [float(r["c"]) for r in rows],
                "volume": [float(r["v"]) for r in rows],
                "vwap": [float(r.get("vw", r["c"])) for r in rows],
            },
            index=index,
        )
        return frame[BAR_COLUMNS]

    @cached("daily_bars")
    def daily_bars_split_adjusted(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily bars adjusted for splits only (the vendor's `adjusted=true`). One
        range request per (ticker, window). Shares the v0.5.0 `daily_bars` cache
        namespace, whose entries hold exactly this series."""
        return self._aggs(ticker, start, end, "day", normalize=True, adjusted=True)

    @cached("daily_bars_raw")
    def daily_bars_raw(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `DataSource.daily_bars_raw`: unadjusted (`adjusted=false`) daily bars."""
        return self._aggs(ticker, start, end, "day", normalize=True, adjusted=False)

    @cached("dividends")
    def dividends(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Cash dividends with an ex-dividend date in `[start, end]`.

        Uses `/v3/reference/dividends`, following `next_url` pagination. The
        `ex_dividend_date.gte/.lte` request filters only narrow the download -- the
        returned frame is always filtered to `[start, end]` locally.

        Args:
            ticker: Stock ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A frame with `adjust.DIVIDEND_COLUMNS` (`ex_dividend_date` tz-naive), sorted by
            ex-date.
        """
        url = f"{self._base_url()}/v3/reference/dividends"
        params = {
            "ticker": ticker,
            "ex_dividend_date.gte": start,
            "ex_dividend_date.lte": end,
            "limit": 1000,
        }
        rows = self._paginate(url, params)
        rows = [r for r in rows if r.get("ex_dividend_date") and r.get("cash_amount") is not None]
        if not rows:
            return adjust.empty_dividends()
        frame = pd.DataFrame(
            {
                "ex_dividend_date": pd.to_datetime([r["ex_dividend_date"] for r in rows]),
                "cash_amount": [float(r["cash_amount"]) for r in rows],
                "currency": [r.get("currency") for r in rows],
                "dividend_type": [r.get("dividend_type") for r in rows],
            }
        )
        in_window = (frame["ex_dividend_date"] >= pd.Timestamp(start)) & (
            frame["ex_dividend_date"] <= pd.Timestamp(end)
        )
        frame = frame.loc[in_window, adjust.DIVIDEND_COLUMNS]
        return frame.sort_values("ex_dividend_date", kind="stable").reset_index(drop=True)

    def _dividend_factors(self, ticker: str, start: str, end: str) -> pd.Series:
        """`adjust.dividend_factors` for one window, on the configured cash basis."""
        if adjust.DIVIDEND_CASH_BASIS == "raw":
            closes = self.daily_bars_raw(ticker, start, end)["close"]
        else:
            closes = self.daily_bars_split_adjusted(ticker, start, end)["close"]
        return adjust.dividend_factors(closes, self.dividends(ticker, start, end))

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `DataSource.daily_bars`: total-return (split + cash dividend)
        adjusted, anchored at the window's last bar. Three cached requests per
        (ticker, window): split-adjusted bars, raw bars, dividends."""
        bars = self.daily_bars_split_adjusted(ticker, start, end)
        return adjust.apply_total_return(bars, self._dividend_factors(ticker, start, end))

    @cached("hourly_bars")
    def hourly_bars_split_adjusted(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Hourly bars adjusted for splits only, filtered to regular trading hours."""
        bars = self._aggs(ticker, start, end, "hour", normalize=False, adjusted=True)
        return filter_regular_hours(bars, bar_minutes=60)

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `DataSource.hourly_bars`: regular-hours hourly bars, total-return
        adjusted with the same per-session dividend factors as `daily_bars`."""
        bars = self.hourly_bars_split_adjusted(ticker, start, end)
        return adjust.apply_total_return(bars, self._dividend_factors(ticker, start, end))

    @cached("index_daily_bars")
    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily aggregates for an index ticker, e.g. "I:VIX". Indices pay no
        dividends, so these are not total-return adjusted.

        Requires indices access on the Polygon plan -- see `OPEN_QUESTIONS.md`.

        Args:
            ticker: Index ticker, e.g. "I:VIX".
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".
        """
        return self._aggs(ticker, start, end, "day", normalize=True)
