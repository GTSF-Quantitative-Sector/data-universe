"""Polygon.io stock and index aggregates.

Synchronous, using a pooled `requests.Session` (unlike sec_parser's `async`
aiohttp calls). TLS verification stays on -- never pass `verify=False`. The
API key is sent as `Authorization: Bearer <key>`, never as a URL/query
parameter, so it never shows up in logs or tracebacks. One range request is
issued per (ticker, window) call, following `next_url` pagination rather
than paging by day.
"""
from __future__ import annotations

import time
from typing import Optional

import pandas as pd
import requests

from data_universe import config
from data_universe.cache import cached
from data_universe.sources.base import (
    BAR_COLUMNS,
    NY_TZ,
    DataSource,
    RateLimitedError,
    empty_bars,
    filter_regular_hours,
)


class PolygonSource(DataSource):
    """Stock and index aggregates from Polygon.io.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a
            fresh pooled session.
        max_retries: Max number of 429 retries before raising `RateLimitedError`.
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

    def _headers(self) -> dict:
        key = config.get_polygon_key()
        if key is None:
            raise ValueError("Polygon API key not set. Call config.set_polygon_key() first.")
        return {"Authorization": f"Bearer {key}"}

    def _get(self, url: str, params: Optional[dict] = None) -> dict:
        """GET a Polygon URL, retrying on 429 with doubling backoff up to `max_retries`."""
        attempt = 0
        while True:
            response = self._session.get(url, params=params, headers=self._headers(), timeout=30)
            if response.status_code == 429:
                attempt += 1
                if attempt > self._max_retries:
                    raise RateLimitedError(f"Polygon rate limit exceeded for {url}")
                time.sleep(self._backoff_seconds * (2 ** (attempt - 1)))
                continue
            response.raise_for_status()
            return response.json()

    def _aggs(
        self, ticker: str, start: str, end: str, timespan: str, normalize: bool
    ) -> pd.DataFrame:
        """Fetch one (ticker, window) range request's worth of aggregates,
        following `next_url` pagination.
        """
        base = config.get_polygon_base_url()
        url = f"{base}/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}
        rows: list = []
        while url:
            data = self._get(url, params=params)
            rows.extend(data.get("results") or [])
            url = data.get("next_url")
            params = None  # next_url already carries its own query string
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
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `DataSource.daily_bars`. One range request per (ticker, window)."""
        return self._aggs(ticker, start, end, "day", normalize=True)

    @cached("hourly_bars")
    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `DataSource.hourly_bars`. Filtered to regular trading hours by default."""
        bars = self._aggs(ticker, start, end, "hour", normalize=False)
        return filter_regular_hours(bars)

    @cached("index_daily_bars")
    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Daily aggregates for an index ticker, e.g. "I:VIX".

        Requires indices access on the Polygon plan -- see `OPEN_QUESTIONS.md`.

        Args:
            ticker: Index ticker, e.g. "I:VIX".
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".
        """
        return self._aggs(ticker, start, end, "day", normalize=True)
