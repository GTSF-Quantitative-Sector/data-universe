"""FRED risk-free rate series (default `DTB3`, the 3-month Treasury bill
secondary-market discount rate).

Synchronous, using a pooled `requests.Session`. FRED's public API has no
Authorization-header option -- its documented auth mechanism is an
`api_key` query parameter, which is what this module uses; this is FRED's
own requirement, not an exception to the "never put a key in a URL" rule
that applies to Polygon specifically.

Unit conversion: FRED reports `DTB3` as an annualized percent (e.g. `5.25`
means 5.25%). This converts to a decimal *daily* rate via
`(value / 100) / 252`. Whether 252 (trading days) or a calendar-day count is
the right divisor, and whether a bank-discount-rate series like `DTB3`
should be converted to a bond-equivalent yield first, are both flagged in
`OPEN_QUESTIONS.md` -- this is a stated assumption, not a verified one.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from data_universe import config
from data_universe.cache import cached


class FredSource:
    """Risk-free rate series from the FRED API.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a
            fresh pooled session.
        base_url: FRED API base URL.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        base_url: str = "https://api.stlouisfed.org/fred",
    ) -> None:
        self._session = session or requests.Session()
        self._base_url = base_url

    @cached("daily_rate")
    def daily_rate(
        self,
        series_id: str = "DTB3",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.Series:
        """Daily risk-free rate, as a decimal (not percent) per-trading-day rate.

        Args:
            series_id: FRED series id. Default "DTB3".
            start: Start date "YYYY-MM-DD", or None for the earliest available.
            end: End date "YYYY-MM-DD", or None for the latest available.

        Returns:
            A decimal daily-rate `pd.Series` indexed by tz-naive session date,
            named `series_id`. FRED's own "." missing-observation marker is
            dropped, not forward-filled -- callers apply their own gap policy.
        """
        key = config.get_fred_key()
        if key is None:
            raise ValueError("FRED API key not set. Call config.set_fred_key() first.")
        params = {"series_id": series_id, "api_key": key, "file_type": "json"}
        if start:
            params["observation_start"] = start
        if end:
            params["observation_end"] = end
        url = f"{self._base_url}/series/observations"
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        dates = []
        values = []
        for obs in data.get("observations", []):
            if obs["value"] == ".":
                continue  # FRED's missing-observation marker
            dates.append(pd.Timestamp(obs["date"]))
            values.append(float(obs["value"]) / 100.0 / 252.0)
        index = pd.DatetimeIndex(dates, name="date")
        return pd.Series(values, index=index, name=series_id)
