"""Polygon.io options contract reference and per-contract daily aggregates.

Subclasses `PolygonSource` to reuse its Bearer-header auth, 429 backoff, and
`next_url` pagination rather than re-implementing them. Endpoint paths
follow Polygon's documented options reference/aggregates conventions but
were not re-verified against live docs in this session -- see the existing
flag in `OPEN_QUESTIONS.md` on options endpoints.
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd
import requests

from data_universe.cache import cached
from data_universe.sources.polygon import PolygonSource

_CONTRACT_COLUMNS = ["ticker", "expiration_date", "strike_price", "contract_type"]


def _today() -> str:
    """Today's UTC date, "YYYY-MM-DD" (a function so tests can pin it)."""
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


class PolygonOptionsSource(PolygonSource):
    """Options contract reference and per-contract daily aggregates from Polygon.io.

    Constructor and auth/backoff/pagination are inherited from `PolygonSource`,
    including `_base_url()` -- see `massive.MassiveOptionsSource` for the Massive host.
    """

    @cached("contracts_as_of")
    def contracts_as_of(
        self,
        underlying: str,
        as_of_date: str,
        expiration_gte: Optional[str] = None,
        expiration_lte: Optional[str] = None,
        strike_gte: Optional[float] = None,
        strike_lte: Optional[float] = None,
        contract_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """The listed option chain for `underlying` as of `as_of_date`, optionally
        narrowed server-side by expiry/strike/type (a full chain for a liquid name is
        many pages; a filtered one is usually one).

        Contracts that have since expired need `expired=true`, which excludes
        contracts still listed today; so when the expiry window can reach today
        (`expiration_lte` is None or >= today), a second `expired=false` query is
        made and the two are unioned. See `OPEN_QUESTIONS.md` item 22.

        Args:
            underlying: Underlying ticker symbol.
            as_of_date: Date "YYYY-MM-DD".
            expiration_gte: Earliest expiration, "YYYY-MM-DD", inclusive.
            expiration_lte: Latest expiration, "YYYY-MM-DD", inclusive.
            strike_gte: Lowest strike, inclusive.
            strike_lte: Highest strike, inclusive.
            contract_type: "call" or "put".

        Returns:
            A DataFrame with columns `ticker` (OCC), `expiration_date`,
            `strike_price`, `contract_type` ("call"/"put"), one row per contract,
            sorted by expiration, strike, type and ticker.
        """
        url = f"{self._base_url()}/v3/reference/options/contracts"
        params = {"underlying_ticker": underlying, "as_of": as_of_date, "limit": 1000}
        filters = {
            "expiration_date.gte": expiration_gte,
            "expiration_date.lte": expiration_lte,
            "strike_price.gte": strike_gte,
            "strike_price.lte": strike_lte,
            "contract_type": contract_type,
        }
        params.update({k: v for k, v in filters.items() if v is not None})
        rows = self._paginate(url, {**params, "expired": "true"})
        if expiration_lte is None or expiration_lte >= _today():
            rows += self._paginate(url, {**params, "expired": "false"})
        if not rows:
            return pd.DataFrame(columns=_CONTRACT_COLUMNS)
        frame = pd.DataFrame(
            {
                "ticker": [r["ticker"] for r in rows],
                "expiration_date": [r["expiration_date"] for r in rows],
                "strike_price": [float(r["strike_price"]) for r in rows],
                "contract_type": [r["contract_type"] for r in rows],
            }
        )
        frame = frame.drop_duplicates("ticker")
        frame = frame.sort_values(["expiration_date", "strike_price", "contract_type", "ticker"])
        return frame.reset_index(drop=True)

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

            The endpoint answers HTTP 404 `{"status": "NOT_FOUND"}` when it
            has no data for the contract/date (e.g. an untraded contract);
            that case returns `{"close": NaN, "volume": 0.0}` so callers see
            a stale close rather than an exception. A 200 response missing
            `close` likewise yields NaN; a missing `volume` yields 0.0.

        Raises:
            requests.HTTPError: For non-404 HTTP errors.
        """
        url = f"{self._base_url()}/v1/open-close/{option_ticker}/{date}"
        try:
            data = self._get(url, params={"adjusted": "true"})
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return {"close": float("nan"), "volume": 0.0}
            raise
        close = data.get("close")
        return {
            "close": float(close) if close is not None else float("nan"),
            "volume": float(data.get("volume") or 0.0),
        }
