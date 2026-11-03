"""Massive (formerly Polygon.io) market data.

Polygon.io rebranded as Massive on 2025-10-30. The REST API is the same -- same paths,
same `Authorization: Bearer <key>` auth -- but the host is now `api.massive.com`
(`api.polygon.io` is still served for an extended period) and existing Polygon keys keep
working. `MassiveSource` is therefore `PolygonSource` pointed at the Massive host, so it
inherits `daily_bars`, `hourly_bars`, `index_daily_bars`, the 429 retry/backoff and the
cache, and adds the two reference-data calls the universe collector needs:
`list_tickers` and `ticker_details`. `MassiveOptionsSource` adds the options methods
of `PolygonOptionsSource` on the same host.

Key lookup: `config.get_massive_key()` first (env `MASSIVE_API_KEY`), then the Polygon key
(env `POLYGON_API_KEY`). The key is only ever sent as a Bearer header, never in a URL.

The endpoint paths and response fields below follow Massive's REST docs
(`/v3/reference/tickers`, `/v3/reference/tickers/{ticker}`) but are mocked in every test,
never called live from CI -- see `OPEN_QUESTIONS.md` item 15.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from data_universe import config
from data_universe.cache import cached
from data_universe.sources.polygon import PolygonSource
from data_universe.sources.polygon_options import PolygonOptionsSource

TICKER_COLUMNS = [
    "ticker",
    "name",
    "market",
    "type",
    "primary_exchange",
    "active",
    "cik",
    "composite_figi",
    "share_class_figi",
    "currency_name",
    "last_updated_utc",
]


class MassiveSource(PolygonSource):
    """Stock/index aggregates (inherited) and ticker reference data from Massive.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a fresh
            pooled session.
        max_retries: Max number of 429 retries before raising `RateLimitedError`.
        backoff_seconds: Base backoff in seconds, doubled on each retry.
    """

    def _base_url(self) -> str:
        return config.get_massive_base_url()

    def _headers(self) -> dict:
        key = config.get_massive_key() or config.get_polygon_key()
        if key is None:
            raise ValueError(
                "Massive API key not set. Call config.set_massive_key() "
                "or set the MASSIVE_API_KEY environment variable."
            )
        return {"Authorization": f"Bearer {key}"}

    @cached("list_tickers")
    def list_tickers(
        self,
        market: str = "stocks",
        ticker_type: str = None,
        exchange: str = None,
        active: bool = True,
    ) -> pd.DataFrame:
        """List reference tickers, following `next_url` pagination (1000 per page).

        With no filters beyond `market`, this returns every active ticker in the market
        (stocks, ETFs, ADRs, warrants, ...), which takes on the order of ten to fifteen
        requests. On a rate-limited plan those requests are retried on 429 with backoff.

        Args:
            market: Massive market, e.g. "stocks", "indices".
            ticker_type: Optional Massive ticker type code, e.g. "CS" (common stock),
                "ADRC", "ETF". None means all types.
            exchange: Optional primary-exchange MIC (ISO 10383), e.g. "XNAS" or "XNYS".
                None means all exchanges.
            active: True for currently active tickers, False for delisted ones.

        Returns:
            A frame with one row per ticker and the columns in `TICKER_COLUMNS`
            (a field Massive omits for a ticker is missing).
        """
        url = f"{self._base_url()}/v3/reference/tickers"
        params = {
            "market": market,
            "active": "true" if active else "false",
            "sort": "ticker",
            "order": "asc",
            "limit": 1000,
        }
        if ticker_type:
            params["type"] = ticker_type
        if exchange:
            params["exchange"] = exchange

        results: list = []
        while url:
            data = self._get(url, params=params)
            results.extend(data.get("results") or [])
            url = data.get("next_url")
            params = None  # next_url already carries its own cursor/query string

        records = []
        for result in results:
            record = {}
            for column in TICKER_COLUMNS:
                record[column] = result.get(column)
            records.append(record)
        return pd.DataFrame(records, columns=TICKER_COLUMNS)

    @cached("ticker_details")
    def ticker_details(self, ticker: str, date: Optional[str] = None) -> dict:
        """Reference details for one ticker (name, primary exchange, CIK, FIGIs, SIC, ...).

        With `date` the record is as of that date ("YYYY-MM-DD") -- confirmed
        2026-11-03: FB on 2020-06-01 resolves to Meta's CIK (the SEC's current
        ticker map has no "FB" at all), and DOW resolves to Dow Chemical
        (CIK 29915) before 2019 but Dow Inc. (CIK 1751788) after the
        re-registration. Without `date` it's today's record, as before. Also a
        cheap way to check that the key and base URL work before a big pull.

        Args:
            ticker: Ticker symbol, e.g. "AAPL".
            date: Optional "YYYY-MM-DD" as-of date for the record.

        Returns:
            The `results` object from the API, as a dict (empty if absent).
        """
        params = {"date": date} if date is not None else None
        data = self._get(f"{self._base_url()}/v3/reference/tickers/{ticker}", params=params)
        return data.get("results") or {}

    def as_of_cik(self, ticker: str, date: str) -> Optional[str]:
        """The CIK Massive records for `ticker` as of `date` ("YYYY-MM-DD"), or None.

        A 404 (no record for the symbol on that date, e.g. GOOGL on 2015-01-02
        and 2016-01-04)
        or a record without a `cik` both return None -- "unknown here" -- so
        callers can fall back to another source instead of guessing. Any other
        error (rate limit, 5xx, entitlement) propagates.

        Args:
            ticker: Ticker symbol, e.g. "FB".
            date: As-of date, "YYYY-MM-DD".

        Returns:
            The CIK as an unpadded digit string (e.g. "1326801"), or None.
        """
        try:
            details = self.ticker_details(ticker, date=date)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            raise
        cik = details.get("cik")
        return str(int(cik)) if cik else None


class MassiveOptionsSource(MassiveSource, PolygonOptionsSource):
    """`PolygonOptionsSource`'s contract-reference and daily-close methods against the
    Massive host and key (`MassiveSource._base_url`/`_headers`), plus everything
    `MassiveSource` has. Method resolution order: MassiveOptionsSource, MassiveSource,
    PolygonOptionsSource, PolygonSource.
    """
