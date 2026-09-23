"""Polygon.io options contract reference and per-contract daily aggregates.

Subclasses `PolygonSource` to reuse its Bearer-header auth, 429 backoff, and
`next_url` pagination rather than re-implementing them. Endpoint paths
follow Polygon's documented options reference/aggregates conventions but
were not re-verified against live docs in this session -- see the existing
flag in `OPEN_QUESTIONS.md` on options endpoints.
"""
from __future__ import annotations

import pandas as pd

from data_universe import config
from data_universe.cache import cached
from data_universe.sources.polygon import PolygonSource

_CONTRACT_COLUMNS = ["ticker", "expiration_date", "strike_price", "contract_type"]


class PolygonOptionsSource(PolygonSource):
    """Options contract reference and per-contract daily aggregates from Polygon.io.

    Constructor and auth/backoff/pagination are inherited from `PolygonSource`.
    """

    @cached("contracts_as_of")
    def contracts_as_of(self, underlying: str, as_of_date: str) -> pd.DataFrame:
        """The listed option chain for `underlying` as of `as_of_date`,
        including contracts that have since expired.

        Args:
            underlying: Underlying ticker symbol.
            as_of_date: Date "YYYY-MM-DD".

        Returns:
            A DataFrame with columns `ticker` (OCC), `expiration_date`,
            `strike_price`, `contract_type` ("call"/"put"), one row per contract.
        """
        base = config.get_polygon_base_url()
        url = f"{base}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying,
            "as_of": as_of_date,
            "expired": "true",
            "limit": 1000,
        }
        rows: list = []
        while url:
            data = self._get(url, params=params)
            rows.extend(data.get("results") or [])
            url = data.get("next_url")
            params = None  # next_url already carries its own query string
        if not rows:
            return pd.DataFrame(columns=_CONTRACT_COLUMNS)
        return pd.DataFrame(
            {
                "ticker": [r["ticker"] for r in rows],
                "expiration_date": [r["expiration_date"] for r in rows],
                "strike_price": [float(r["strike_price"]) for r in rows],
                "contract_type": [r["contract_type"] for r in rows],
            }
        )

    @cached("daily_close")
    def daily_close(self, option_ticker: str, date: str) -> dict:
        """A single option contract's daily close and volume.

        Args:
            option_ticker: OCC option ticker, e.g. "O:AAPL240621C00190000".
            date: Date "YYYY-MM-DD".

        Returns:
            A dict with `close` (dollars) and `volume` (contracts traded).
            `volume == 0` means a stale close (no trades that day) -- see
            `OPEN_QUESTIONS.md` and the PEAD event table's `is_stale` column
            (Phase 6), which is computed from this, not here.
        """
        base = config.get_polygon_base_url()
        url = f"{base}/v1/open-close/{option_ticker}/{date}"
        data = self._get(url, params={"adjusted": "true"})
        return {"close": float(data.get("close", 0.0)), "volume": float(data.get("volume", 0.0))}
