"""EDGAR 8-K Item 2.02 (Results of Operations) earnings-announcement dates,
and ticker-to-CIK lookup.

Uses SEC's per-company submissions JSON
(`https://data.sec.gov/submissions/CIK##########.json`). Its
`filings.recent` arrays cover only roughly the latest year / ~1,000 filings;
older filings live in the extra pages listed under `filings.files` as
`{"name": "CIK##########-submissions-NNN.json", ...}`, each fetched from
`https://data.sec.gov/submissions/<name>` and carrying the same column
arrays at the top level. Shape verified against live data.sec.gov on
2026-10-26: `items` is a comma-separated string of Item numbers (e.g.
"2.02,9.01"), `acceptanceDateTime` is ISO UTC.

CIK lookup uses SEC's public ticker mapping
(`https://www.sec.gov/files/company_tickers.json`), the same file
`sec_parser` bundles a snapshot of. It is downloaded once per process
(`ticker_map`) and lists *current* tickers only -- a renamed or delisted
ticker (e.g. FB) is not in it; pass `cik=` explicitly for those.

Fair access: SEC allows at most 10 requests/second per client and requires a
descriptive User-Agent with a contact. Every request from every `EdgarSource`
in the process goes through one throttle spacing requests at least
`_SEC_MIN_INTERVAL_SECONDS` apart (<= 9/s). Transient failures (429/5xx,
connection errors, timeouts) are retried with backoff; a 403 -- SEC's answer
to a placeholder User-Agent or a rate-limit violation -- is raised at once.
The User-Agent is the constructor argument, else `config.get_user_agent()`
(env `DATA_UNIVERSE_USER_AGENT`), else a placeholder that triggers a one-time
warning.
"""
from __future__ import annotations

import threading
import time
import warnings
from typing import Optional

import pandas as pd
import requests

from data_universe import config
from data_universe.cache import cached
from data_universe.sources.base import DataSourceError
from data_universe.sources.http import get_with_retries, url_path

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
_SUBMISSIONS_FILES_URL = "https://data.sec.gov/submissions/{name}"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_EARNINGS_COLUMNS = ["filing_date", "acceptance_datetime", "form", "items"]
PLACEHOLDER_USER_AGENT = "data_universe (contact: gtsf-quant@example.org)"

_SEC_MIN_INTERVAL_SECONDS = 0.11
_throttle_lock = threading.Lock()
_last_request_at: Optional[float] = None
_warned_placeholder = False


def _throttle() -> None:
    """Block until at least `_SEC_MIN_INTERVAL_SECONDS` have passed since the previous
    EDGAR request made anywhere in this process."""
    global _last_request_at
    with _throttle_lock:
        now = time.monotonic()
        if _last_request_at is not None:
            wait = _last_request_at + _SEC_MIN_INTERVAL_SECONDS - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
        _last_request_at = now


def _item_202_rows(page: dict) -> list:
    """Rows for every 8-K (not 8-K/A) disclosing Item 2.02 in one
    submissions page -- a dict with parallel `form`/`filingDate`/
    `acceptanceDateTime`/`items` arrays, as found at `filings.recent` or at
    the top level of each `filings.files` page.
    """
    rows = []
    for form, filing_date, accepted_at, item_str in zip(
        page.get("form", []),
        page.get("filingDate", []),
        page.get("acceptanceDateTime", []),
        page.get("items", []),
    ):
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
    return rows


class EdgarSource:
    """8-K Item 2.02 earnings announcement dates and CIK lookup from SEC EDGAR.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a
            fresh pooled session.
        user_agent: SEC requires a descriptive User-Agent identifying the
            requester (e.g. "org-name contact@example.org"); requests without
            one may be rejected. None uses `config.get_user_agent()`, then a
            placeholder (with a one-time warning).
        max_retries: Max retries of a transient failure before raising.
        backoff_seconds: Base backoff in seconds, doubled on each retry.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        user_agent: Optional[str] = None,
        max_retries: int = 5,
        backoff_seconds: float = 1.0,
    ) -> None:
        self._session = session or requests.Session()
        self._user_agent = user_agent
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def _resolved_user_agent(self) -> str:
        global _warned_placeholder
        user_agent = self._user_agent or config.get_user_agent()
        if user_agent:
            return user_agent
        if not _warned_placeholder:
            _warned_placeholder = True
            warnings.warn(
                "EdgarSource is using a placeholder User-Agent; SEC may block it. Set "
                "DATA_UNIVERSE_USER_AGENT (or config.set_user_agent) to "
                "'<org> <contact email>'.",
                UserWarning,
                stacklevel=3,
            )
        return PLACEHOLDER_USER_AGENT

    def _get(self, url: str) -> dict:
        response = get_with_retries(
            self._session,
            url,
            headers={"User-Agent": self._resolved_user_agent()},
            timeout=30,
            max_retries=self._max_retries,
            backoff_seconds=self._backoff_seconds,
            throttle=_throttle,
        )
        if response.status_code == 403:
            raise DataSourceError(
                f"SEC EDGAR returned 403 for {url_path(url)} -- usually a missing or "
                "placeholder User-Agent, or more than 10 requests/second. Set "
                "DATA_UNIVERSE_USER_AGENT to '<org> <contact email>'."
            )
        response.raise_for_status()
        return response.json()

    @cached("company_tickers")
    def ticker_map(self) -> pd.Series:
        """SEC's current ticker -> CIK mapping, downloaded once per process.

        Returns:
            A Series indexed by upper-case ticker, values CIK digit strings, named
            "cik". On a duplicate ticker the first entry wins.
        """
        data = self._get(_TICKERS_URL)
        tickers, ciks = [], []
        for entry in data.values():
            ticker = str(entry.get("ticker", "")).upper()
            if ticker:
                tickers.append(ticker)
                ciks.append(str(entry["cik_str"]))
        mapping = pd.Series(ciks, index=pd.Index(tickers, name="ticker"), name="cik", dtype=object)
        return mapping[~mapping.index.duplicated(keep="first")]

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
        mapping = self.ticker_map()
        ticker_upper = ticker.upper()
        if ticker_upper in mapping.index:
            return str(mapping.loc[ticker_upper])
        raise ValueError(
            f"Ticker {ticker!r} not found in SEC company_tickers.json (current tickers "
            "only -- pass cik= explicitly for a renamed or delisted ticker)"
        )

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
        filings = data.get("filings", {})
        rows = _item_202_rows(filings.get("recent", {}))
        for entry in filings.get("files", []):
            older = self._get(_SUBMISSIONS_FILES_URL.format(name=entry["name"]))
            rows.extend(_item_202_rows(older))
        frame = pd.DataFrame(rows, columns=_EARNINGS_COLUMNS).drop_duplicates()
        return frame.sort_values("acceptance_datetime").reset_index(drop=True)
