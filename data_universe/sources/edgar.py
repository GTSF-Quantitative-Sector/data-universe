"""EDGAR 8-K Item 2.02 (Results of Operations) earnings-announcement dates,
and ticker-to-CIK lookup.

Uses SEC's per-company submissions JSON
(`https://data.sec.gov/submissions/CIK##########.json`), whose
`filings.recent` arrays include each filing's form type, filing date,
acceptance timestamp, and (for 8-Ks) a comma-separated `items` field listing
disclosed Item numbers -- e.g. "2.02,9.01". The exact submissions JSON shape
is flagged as unverified-against-live-docs in `OPEN_QUESTIONS.md`; this
module is written against that documented shape and mocked in every test.

CIK lookup uses SEC's public ticker mapping
(`https://www.sec.gov/files/company_tickers.json`), the same file
`sec_parser` bundles a snapshot of.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import requests

from data_universe.cache import cached

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EARNINGS_COLUMNS = ["filing_date", "acceptance_datetime", "form", "items"]


class EdgarSource:
    """8-K Item 2.02 earnings announcement dates and CIK lookup from SEC EDGAR.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a
            fresh pooled session.
        user_agent: SEC requires a descriptive User-Agent identifying the
            requester (e.g. "org-name contact@example.org"); requests without
            one may be rejected.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        user_agent: str = "data_universe (contact: gtsf-quant@example.org)",
    ) -> None:
        self._session = session or requests.Session()
        self._user_agent = user_agent

    def _get(self, url: str) -> dict:
        response = self._session.get(url, headers={"User-Agent": self._user_agent}, timeout=30)
        response.raise_for_status()
        return response.json()

    @cached("get_cik")
    def get_cik(self, ticker: str) -> str:
        """Look up a ticker's SEC CIK via the public ticker-to-CIK mapping.

        Args:
            ticker: Stock ticker symbol, case-insensitive.

        Returns:
            The CIK as a string of digits (e.g. "320193").

        Raises:
            ValueError: If `ticker` isn't found in the mapping.
        """
        data = self._get(_TICKERS_URL)
        ticker_upper = ticker.upper()
        for entry in data.values():
            if str(entry.get("ticker", "")).upper() == ticker_upper:
                return str(entry["cik_str"])
        raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers.json")

    @cached("earnings_8k_filings")
    def earnings_8k_filings(self, cik: str) -> pd.DataFrame:
        """Every 8-K (not 8-K/A) filing disclosing Item 2.02, for a given CIK.

        Args:
            cik: SEC Central Index Key, digits only (e.g. "320193" for Apple).

        Returns:
            A DataFrame with columns `filing_date`, `acceptance_datetime`
            (as reported by EDGAR), `form`, `items`; one row per qualifying
            filing, sorted ascending by `acceptance_datetime`.
        """
        url = _SUBMISSIONS_URL.format(cik=int(cik))
        data = self._get(url)
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        acceptance = recent.get("acceptanceDateTime", [])
        items = recent.get("items", [])
        rows = []
        for form, filing_date, accepted_at, item_str in zip(forms, filing_dates, acceptance, items):
            if form != "8-K":
                continue  # excludes "8-K/A" and every other form type
            item_codes = [c.strip() for c in (item_str or "").split(",")]
            if "2.02" not in item_codes:
                continue
            rows.append(
                {
                    "filing_date": pd.Timestamp(filing_date),
                    "acceptance_datetime": pd.Timestamp(accepted_at),
                    "form": form,
                    "items": item_str,
                }
            )
        frame = pd.DataFrame(rows, columns=_EARNINGS_COLUMNS)
        return frame.sort_values("acceptance_datetime").reset_index(drop=True)
