"""Massive (formerly Polygon.io) market data.

Polygon.io rebranded as Massive on 2025-10-30. The REST API is the same -- same paths,
same `Authorization: Bearer <key>` auth -- but the host is now `api.massive.com`
(`api.polygon.io` is still served for an extended period) and existing Polygon keys keep
working. `MassiveSource` is therefore `PolygonSource` pointed at the Massive host, so it
inherits `daily_bars`, `hourly_bars`, `index_daily_bars`, the 429 retry/backoff and the
cache, and adds the two reference-data calls the universe collector needs:
`list_tickers` and `ticker_details`.

Key lookup: `config.get_massive_key()` first (env `MASSIVE_API_KEY`), then the Polygon key
(env `POLYGON_API_KEY`). The key is only ever sent as a Bearer header, never in a URL.

The endpoint paths and response fields below follow Massive's REST docs
(`/v3/reference/tickers`, `/v3/reference/tickers/{ticker}`) but are mocked in every test,
never called live from CI -- see `OPEN_QUESTIONS.md` item 15.
"""
from __future__ import annotations

import pandas as pd

from data_universe import config
from data_universe.cache import cached
from data_universe.sources.polygon import PolygonSource

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
    def ticker_details(self, ticker: str) -> dict:
        """Reference details for one ticker (name, primary exchange, CIK, FIGIs, SIC, ...).

        Also a cheap way to check that the key and base URL work before a big pull.

        Args:
            ticker: Ticker symbol, e.g. "AAPL".

        Returns:
            The `results` object from the API, as a dict (empty if absent).
        """
        data = self._get(f"{self._base_url()}/v3/reference/tickers/{ticker}")
        return data.get("results") or {}
