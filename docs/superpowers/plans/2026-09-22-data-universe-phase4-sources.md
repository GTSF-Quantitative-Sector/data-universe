# Data Universe Phase 4 (Sources) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the real data sources — Polygon stock/index aggregates, FRED risk-free rate, EDGAR 8-K Item 2.02 earnings dates + CIK lookup, an optional sec_parser adapter for point-in-time S&P 500 membership, and a trading-session calendar derived from a market proxy's bars. Every source is tested against a mocked `requests.Session`; nothing touches the network.

**Architecture:** Each source is a small class taking an injectable `requests.Session` (for tests) and reading credentials from `data_universe.config` at call time (never at import time, so a test can set a key just before the call it's testing). `PolygonSource` implements the `DataSource` interface from Phase 2 and reuses `filter_regular_hours`/`empty_bars`/`BAR_COLUMNS`. Every network-calling method is decorated with `@cached(...)` from Phase 3, so a repeat call in the same process is served from cache and logged. `sources/sec_adapter.py` is the one module allowed to fail at *call* time (not import time) with a clear `ImportError` pointing at `pip install ".[sec]"`, since `sec_parser` is an optional dependency. `calendar.py` has no source of its own — it derives trading sessions from whatever `DataSource` you hand it (a fake in tests, `PolygonSource` for real use), because the market proxy's own bars only exist for days it actually traded.

**Tech Stack:** Python ≥3.9, requests (HTTP), pandas (frames), pytest + `unittest.mock` (mocked sessions, no new test dependency).

**Spec:** `PLAN.md` (repo root) §3 (`sources/polygon.py`, `sources/fred.py`, `sources/edgar.py`, `sources/sec_adapter.py`, `calendar.py`), and the "Polygon" vendor-background section (Bearer header, `POLYGON_BASE_URL`, no invented endpoints/fields). This plan implements Phase 4 of `PLAN.md` §10 only: "Sources: Polygon stocks and indices, FRED, EDGAR, sec_parser adapter, calendar — all tested with mocks." The options toolkit, flat-files bulk S3 download, features/labels, the PEAD event table, and `precompute.py` are later phases and are **not** built here.

## Global Constraints

- **No test may touch the network.** Every source test injects a mocked `requests.Session` (a `unittest.mock.Mock` whose `.get` returns a canned fake response) — never a real HTTP call, never `requests_mock`/`responses` as a new dependency.
- The Polygon API key is sent as `Authorization: Bearer <key>` — never as a URL/query parameter, never logged. (FRED's public API has no Bearer-header option — its `api_key` query parameter is FRED's own documented auth mechanism, not a violation of this rule; SEC EDGAR needs no key at all, only a descriptive `User-Agent`.)
- Do **not** invent Polygon endpoint paths or response fields "confidently" — every field access in `polygon.py` is exactly what the aggregates endpoint's response shape needs (`t, o, h, l, c, v, vw`, plus `next_url` for pagination), which is the one Polygon endpoint shape stated directly in `PLAN.md`'s own examples. Anything less certain (index `I:VIX` entitlement, exact options endpoints) is Phase 5's problem or already flagged in the repo's `OPEN_QUESTIONS.md`, not re-litigated here.
- One range request per `(ticker, window)` call — never one request per day.
- TLS verification stays on (`requests.Session` defaults to `verify=True`; never pass `verify=False`).
- Synchronous only — no `async def` anywhere in this phase.
- Every `PolygonSource`/`FredSource`/`EdgarSource` network-calling method is decorated with `@cached(...)` from `data_universe.cache` (Phase 3).
- Bars returned by `PolygonSource` follow the Phase 2 frame convention exactly: daily bars stamped at midnight `America/New_York`; hourly bars filtered to regular hours (9:30–16:00 ET) via `filter_regular_hours`.
- Docstrings on every public function/class/method, stating units (decimal vs. percent, daily vs. annualized) where relevant.

---

### Task 1: `sources/polygon.py` — stock/index aggregates, pagination, 429 backoff

**Files:**
- Create: `data_universe/sources/polygon.py`
- Test: `tests/test_sources_polygon.py`

**Interfaces:**
- Consumes: `data_universe.config.get_polygon_key()`/`get_polygon_base_url()` (Phase 2), `data_universe.cache.cached` (Phase 3), `data_universe.sources.base.{DataSource, BAR_COLUMNS, NY_TZ, RateLimitedError, empty_bars, filter_regular_hours}` (Phase 2).
- Produces: `class PolygonSource(session=None, max_retries=5, backoff_seconds=1.0)` implementing `DataSource`, with `.daily_bars(ticker, start, end) -> pd.DataFrame`, `.hourly_bars(ticker, start, end) -> pd.DataFrame`, `.index_daily_bars(ticker, start, end) -> pd.DataFrame`. Task 5's `calendar.py` consumes `.daily_bars` through the `DataSource` interface (it doesn't import `PolygonSource` directly, so this doesn't create a hard dependency).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_polygon.py`:

```python
from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.base import RateLimitedError
from data_universe.sources.polygon import PolygonSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_polygon_key("test-key")
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400 and self.status_code != 429:
            raise RuntimeError(f"HTTP {self.status_code}")


def _day_agg(day_ms, o, h, l, c, v, vw):
    return {"t": day_ms, "o": o, "h": h, "l": l, "c": c, "v": v, "vw": vw}


def test_daily_bars_sends_bearer_header_not_url_param():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    source = PolygonSource(session=session)
    source.daily_bars("AAPL", "2024-01-01", "2024-01-05")
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "apiKey" not in (kwargs.get("params") or {})
    assert "apiKey" not in session.get.call_args[0][0]


def test_daily_bars_raises_without_a_key():
    config.set_polygon_key(None)
    source = PolygonSource(session=Mock())
    with pytest.raises(ValueError):
        source.daily_bars("AAPL", "2024-01-01", "2024-01-05")


def test_daily_bars_parses_results_into_frame_normalized_to_midnight_ny():
    ts_ms = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(
        200, {"results": [_day_agg(ts_ms, 100.0, 101.0, 99.0, 100.5, 1000.0, 100.2)]}
    )
    source = PolygonSource(session=session)
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert bars.index[0] == pd.Timestamp("2024-01-02", tz="America/New_York")
    assert bars.iloc[0]["close"] == 100.5


def test_daily_bars_follows_next_url_pagination():
    ts1 = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    ts2 = int(pd.Timestamp("2024-01-03", tz="America/New_York").timestamp() * 1000)
    page1 = FakeResponse(
        200,
        {
            "results": [_day_agg(ts1, 1, 1, 1, 1, 1, 1)],
            "next_url": "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2024-01-03/2024-01-05?cursor=abc",
        },
    )
    page2 = FakeResponse(200, {"results": [_day_agg(ts2, 2, 2, 2, 2, 2, 2)]})
    session = Mock()
    session.get.side_effect = [page1, page2]
    source = PolygonSource(session=session)
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-05")
    assert len(bars) == 2
    assert session.get.call_count == 2


def test_daily_bars_is_a_single_range_request_not_one_per_day():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]})
    source = PolygonSource(session=session)
    source.daily_bars("AAPL", "2024-01-01", "2024-01-31")
    assert session.get.call_count == 1


def test_hourly_bars_filters_to_regular_hours():
    pre_open = int(pd.Timestamp("2024-01-02 09:00:00", tz="America/New_York").timestamp() * 1000)
    in_hours = int(pd.Timestamp("2024-01-02 10:00:00", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(
        200,
        {
            "results": [
                _day_agg(pre_open, 1, 1, 1, 1, 1, 1),
                _day_agg(in_hours, 2, 2, 2, 2, 2, 2),
            ]
        },
    )
    source = PolygonSource(session=session)
    bars = source.hourly_bars("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert bars.iloc[0]["close"] == 2.0


def test_index_daily_bars_for_vix():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 16, 16, 16, 16, 0, 16)]})
    source = PolygonSource(session=session)
    bars = source.index_daily_bars("I:VIX", "2024-01-02", "2024-01-02")
    assert bars.iloc[0]["close"] == 16.0
    assert "I:VIX" in session.get.call_args[0][0]


def test_empty_results_returns_empty_bars_frame():
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": []})
    source = PolygonSource(session=session)
    bars = source.daily_bars("AAPL", "2024-01-01", "2024-01-05")
    assert bars.empty


def test_429_retries_with_backoff_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr("data_universe.sources.polygon.time.sleep", lambda s: sleeps.append(s))
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.side_effect = [
        FakeResponse(429),
        FakeResponse(429),
        FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]}),
    ]
    source = PolygonSource(session=session, backoff_seconds=0.01)
    bars = source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    assert len(bars) == 1
    assert len(sleeps) == 2
    assert sleeps == [0.01, 0.02]  # doubles each retry


def test_429_exhausting_retries_raises_rate_limited_error(monkeypatch):
    monkeypatch.setattr("data_universe.sources.polygon.time.sleep", lambda s: None)
    session = Mock()
    session.get.return_value = FakeResponse(429)
    source = PolygonSource(session=session, max_retries=2, backoff_seconds=0.01)
    with pytest.raises(RateLimitedError):
        source.daily_bars("AAPL", "2024-01-02", "2024-01-02")


def test_repeat_call_is_served_from_cache():
    ts = int(pd.Timestamp("2024-01-02", tz="America/New_York").timestamp() * 1000)
    session = Mock()
    session.get.return_value = FakeResponse(200, {"results": [_day_agg(ts, 1, 1, 1, 1, 1, 1)]})
    source = PolygonSource(session=session)
    source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    source.daily_bars("AAPL", "2024-01-02", "2024-01-02")
    assert session.get.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_polygon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.polygon'`

- [ ] **Step 3: Write `data_universe/sources/polygon.py`**

```python
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

    def _aggs(self, ticker: str, start: str, end: str, timespan: str, normalize: bool) -> pd.DataFrame:
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_polygon.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/sources/polygon.py tests/test_sources_polygon.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/sources/polygon.py tests/test_sources_polygon.py
git commit -m "feat: add PolygonSource with pagination, 429 backoff, and caching"
```

---

### Task 2: `sources/fred.py` — risk-free rate series

**Files:**
- Create: `data_universe/sources/fred.py`
- Test: `tests/test_sources_fred.py`

**Interfaces:**
- Consumes: `data_universe.config.get_fred_key()` (Phase 2), `data_universe.cache.cached` (Phase 3).
- Produces: `class FredSource(session=None, base_url="https://api.stlouisfed.org/fred")` with `.daily_rate(series_id="DTB3", start=None, end=None) -> pd.Series` (decimal daily rate, indexed by tz-naive session date). The beta project's excess-return feature (a later phase) calls this directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_fred.py`:

```python
from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.fred import FredSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_fred_key("test-fred-key")
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def test_daily_rate_raises_without_a_key():
    config.set_fred_key(None)
    source = FredSource(session=Mock())
    with pytest.raises(ValueError):
        source.daily_rate()


def test_daily_rate_converts_percent_to_decimal_daily_rate():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"observations": [{"date": "2024-01-02", "value": "5.25"}]}
    )
    source = FredSource(session=session)
    rate = source.daily_rate()
    assert rate.index[0] == pd.Timestamp("2024-01-02")
    assert rate.iloc[0] == pytest.approx(5.25 / 100 / 252)


def test_daily_rate_skips_missing_observations():
    session = Mock()
    session.get.return_value = FakeResponse(
        {
            "observations": [
                {"date": "2024-01-01", "value": "."},
                {"date": "2024-01-02", "value": "5.25"},
            ]
        }
    )
    source = FredSource(session=session)
    rate = source.daily_rate()
    assert len(rate) == 1
    assert rate.index[0] == pd.Timestamp("2024-01-02")


def test_daily_rate_sends_api_key_and_series_id_as_params():
    session = Mock()
    session.get.return_value = FakeResponse({"observations": []})
    source = FredSource(session=session)
    source.daily_rate(series_id="DGS3MO", start="2024-01-01", end="2024-02-01")
    _, kwargs = session.get.call_args
    params = kwargs["params"]
    assert params["api_key"] == "test-fred-key"
    assert params["series_id"] == "DGS3MO"
    assert params["observation_start"] == "2024-01-01"
    assert params["observation_end"] == "2024-02-01"


def test_daily_rate_defaults_to_dtb3():
    session = Mock()
    session.get.return_value = FakeResponse({"observations": []})
    source = FredSource(session=session)
    source.daily_rate()
    _, kwargs = session.get.call_args
    assert kwargs["params"]["series_id"] == "DTB3"


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse({"observations": [{"date": "2024-01-02", "value": "5.0"}]})
    source = FredSource(session=session)
    source.daily_rate()
    source.daily_rate()
    assert session.get.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_fred.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.fred'`

- [ ] **Step 3: Write `data_universe/sources/fred.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_fred.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/sources/fred.py tests/test_sources_fred.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/sources/fred.py tests/test_sources_fred.py
git commit -m "feat: add FredSource for the daily risk-free rate series"
```

---

### Task 3: `sources/edgar.py` — 8-K Item 2.02 earnings dates and CIK lookup

**Files:**
- Create: `data_universe/sources/edgar.py`
- Test: `tests/test_sources_edgar.py`

**Interfaces:**
- Consumes: `data_universe.cache.cached` (Phase 3).
- Produces: `class EdgarSource(session=None, user_agent=...)` with `.get_cik(ticker) -> str` and `.earnings_8k_filings(cik) -> pd.DataFrame` (columns: `filing_date, acceptance_datetime, form, items`, one row per 8-K filing disclosing Item 2.02, sorted by `acceptance_datetime`, `8-K/A` excluded). Phase 6's PEAD event table builder (`events/earnings.py`) calls `get_cik` then `earnings_8k_filings` to build each ticker's announcement timeline.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_edgar.py`:

```python
from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe.cache import reset_cache
from data_universe.sources.edgar import EdgarSource


@pytest.fixture(autouse=True)
def _reset_state():
    reset_cache()
    yield
    reset_cache()


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def test_get_cik_sends_descriptive_user_agent():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    )
    source = EdgarSource(session=session, user_agent="my-org contact@example.org")
    cik = source.get_cik("AAPL")
    assert cik == "320193"
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["User-Agent"] == "my-org contact@example.org"


def test_get_cik_is_case_insensitive():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    )
    source = EdgarSource(session=session)
    assert source.get_cik("aapl") == "320193"


def test_get_cik_raises_for_unknown_ticker():
    session = Mock()
    session.get.return_value = FakeResponse({"0": {"cik_str": 1, "ticker": "AAPL", "title": "x"}})
    source = EdgarSource(session=session)
    with pytest.raises(ValueError):
        source.get_cik("NOPE")


def _submissions(rows):
    """rows: list of (form, filing_date, acceptance_datetime, items)."""
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "acceptanceDateTime": [r[2] for r in rows],
                "items": [r[3] for r in rows],
            }
        }
    }


def test_earnings_8k_filings_keeps_only_item_202():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions(
            [
                ("8-K", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02,9.01"),
                ("8-K", "2024-02-01", "2024-02-01T16:05:00.000Z", "5.02"),
            ]
        )
    )
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert len(filings) == 1
    assert filings.iloc[0]["filing_date"] == pd.Timestamp("2024-01-02")


def test_earnings_8k_filings_excludes_8ka():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions(
            [
                ("8-K/A", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02"),
                ("8-K", "2024-04-01", "2024-04-01T16:05:00.000Z", "2.02"),
            ]
        )
    )
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert len(filings) == 1
    assert filings.iloc[0]["filing_date"] == pd.Timestamp("2024-04-01")


def test_earnings_8k_filings_sorted_by_acceptance_datetime():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions(
            [
                ("8-K", "2024-04-01", "2024-04-01T16:05:00.000Z", "2.02"),
                ("8-K", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02"),
            ]
        )
    )
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert list(filings["filing_date"]) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-04-01")]


def test_earnings_8k_filings_empty_when_no_filings():
    session = Mock()
    session.get.return_value = FakeResponse(_submissions([]))
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert filings.empty
    assert list(filings.columns) == ["filing_date", "acceptance_datetime", "form", "items"]


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions([("8-K", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02")])
    )
    source = EdgarSource(session=session)
    source.earnings_8k_filings("320193")
    source.earnings_8k_filings("320193")
    assert session.get.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_edgar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.edgar'`

- [ ] **Step 3: Write `data_universe/sources/edgar.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_edgar.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/sources/edgar.py tests/test_sources_edgar.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/sources/edgar.py tests/test_sources_edgar.py
git commit -m "feat: add EdgarSource for 8-K Item 2.02 dates and CIK lookup"
```

---

### Task 4: `sources/sec_adapter.py` — optional sec_parser wrapper

**Files:**
- Create: `data_universe/sources/sec_adapter.py`
- Test: `tests/test_sources_sec_adapter.py`
- Modify: `data_universe/__init__.py`

**Interfaces:**
- Consumes: `sec.lookups`/`sec.stock` from the optional `sec_parser` dependency (installed via `pip install ".[sec]"`), imported lazily inside each function — never at module import time, so `import data_universe.sources.sec_adapter` always succeeds even without `sec_parser` installed.
- Produces: `sp500(query_date: Optional[str] = None) -> list` and `fundamentals(ticker: str)`, both raising `ImportError` with a `pip install ".[sec]"` hint if `sec_parser` isn't installed. `data_universe/__init__.py` re-exports this module as `universe` per `PLAN.md` §3 (`q.universe.sp500(query_date=...)`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_sec_adapter.py`:

```python
import pytest

from data_universe.sources import sec_adapter


def test_sp500_raises_helpful_import_error_without_sec_parser():
    # sec_parser is not installed in this environment (it's an optional
    # extra), so this exercises the real failure path, not a mock.
    with pytest.raises(ImportError) as exc_info:
        sec_adapter.sp500(query_date="2019-01-01")
    assert "pip install" in str(exc_info.value)
    assert ".[sec]" in str(exc_info.value)


def test_fundamentals_raises_helpful_import_error_without_sec_parser():
    with pytest.raises(ImportError) as exc_info:
        sec_adapter.fundamentals("AAPL")
    assert "pip install" in str(exc_info.value)


def test_sp500_delegates_to_sec_parser_lookups(monkeypatch):
    calls = []

    class FakeLookups:
        @staticmethod
        def get_sp500_tickers(query_date=None):
            calls.append(query_date)
            return ["AAPL", "MSFT"]

    monkeypatch.setattr(sec_adapter, "_import_sec_parser", lambda: FakeLookups)
    result = sec_adapter.sp500(query_date="2019-01-01")
    assert result == ["AAPL", "MSFT"]
    assert calls == ["2019-01-01"]


def test_sp500_defaults_query_date_to_none():
    calls = []

    class FakeLookups:
        @staticmethod
        def get_sp500_tickers(query_date=None):
            calls.append(query_date)
            return []

    import data_universe.sources.sec_adapter as mod

    original = mod._import_sec_parser
    mod._import_sec_parser = lambda: FakeLookups
    try:
        mod.sp500()
    finally:
        mod._import_sec_parser = original
    assert calls == [None]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_sec_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.sec_adapter'`

- [ ] **Step 3: Write `data_universe/sources/sec_adapter.py`**

```python
"""Optional wrapper around sec_parser
(https://github.com/GTSF-Quantitative-Sector/sec_parser) for fundamentals
and point-in-time S&P 500 membership. Install with `pip install ".[sec]"`
to enable this module -- `sec_parser` is imported lazily inside each
function, never at module import time, so importing this module never fails
just because the optional dependency isn't installed.
"""
from __future__ import annotations

from typing import Optional

_INSTALL_HINT = (
    "sec_parser is not installed. Install it with `pip install '.[sec]'` "
    "to use data_universe.sources.sec_adapter."
)


def _import_sec_parser():
    """Import and return sec_parser's `lookups` module, or raise a helpful ImportError."""
    try:
        from sec import lookups
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return lookups


def sp500(query_date: Optional[str] = None) -> list:
    """Point-in-time S&P 500 constituent tickers, delegating to sec_parser.

    Args:
        query_date: Date "YYYY-MM-DD" to look up membership as of, or None
            for the current membership. **Never pass None for historical
            research** -- it returns today's membership, which introduces
            survivorship bias into any backtest over past dates.

    Returns:
        List of ticker symbols.

    Raises:
        ImportError: If `sec_parser` isn't installed.
    """
    lookups = _import_sec_parser()
    return lookups.get_sp500_tickers(query_date=query_date)


def fundamentals(ticker: str):
    """Return a sec_parser `stock.Stock` for fundamentals access
    (`get_ebit`, `get_revenue`, `get_wacc`, etc. -- see the sec_parser README
    for its full API).

    Args:
        ticker: Stock ticker symbol.

    Raises:
        ImportError: If `sec_parser` isn't installed.
    """
    try:
        from sec import stock
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return stock.Stock(ticker)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_sec_adapter.py -v`
Expected: all tests PASS

- [ ] **Step 5: Update `data_universe/__init__.py` to re-export `sec_adapter` as `universe`**

```python
"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (Ticker,
precompute, feature/label registries). This phase adds `universe` (the
optional sec_parser adapter) on top of Phase 3's `get_cache`/`simulate`.
"""
from data_universe import config
from data_universe.cache import get_cache, simulate
from data_universe.sources import sec_adapter as universe

__version__ = "0.1.0"

__all__ = ["config", "get_cache", "simulate", "universe", "__version__"]
```

- [ ] **Step 6: Run ruff**

Run: `ruff check data_universe/sources/sec_adapter.py data_universe/__init__.py tests/test_sources_sec_adapter.py`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add data_universe/sources/sec_adapter.py data_universe/__init__.py tests/test_sources_sec_adapter.py
git commit -m "feat: add optional sec_parser adapter, exported as data_universe.universe"
```

---

### Task 5: `calendar.py` — trading sessions derived from a market proxy

**Files:**
- Create: `data_universe/calendar.py`
- Test: `tests/test_calendar.py`

**Interfaces:**
- Consumes: any `data_universe.sources.base.DataSource` (Phase 2) via its `.daily_bars(ticker, start, end)` method — in tests, `data_universe.sources.fakes.FakePolygonSource` (Phase 2).
- Produces: `sessions(source, start, end, proxy_ticker="SPY") -> pd.DatetimeIndex`, `next_session(session_index, date) -> Optional[pd.Timestamp]`, `previous_session(session_index, date) -> Optional[pd.Timestamp]`, `shift_sessions(session_index, date, n) -> Optional[pd.Timestamp]`. Phase 6's PEAD event table (`day0_session`, `pre_session`) and Phase 6's feature `compute()` (fetching `warmup` trading days of history before `start`) both build on these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calendar.py`:

```python
import pandas as pd
import pytest

from data_universe.calendar import next_session, previous_session, sessions, shift_sessions
from data_universe.sources.fakes import FakePolygonSource


@pytest.fixture
def session_index():
    source = FakePolygonSource()
    return sessions(source, "2024-01-01", "2024-01-31")  # January 2024: Mon Jan 1 is a holiday but FakeMarket doesn't know that -- bdate_range only skips weekends


def test_sessions_is_tz_naive_and_named_date():
    source = FakePolygonSource()
    idx = sessions(source, "2024-01-01", "2024-01-10")
    assert idx.tz is None
    assert idx.name == "date"


def test_sessions_excludes_weekends():
    source = FakePolygonSource()
    idx = sessions(source, "2024-01-01", "2024-01-10")  # 2024-01-06/07 is a Sat/Sun
    assert pd.Timestamp("2024-01-06") not in idx
    assert pd.Timestamp("2024-01-07") not in idx
    assert pd.Timestamp("2024-01-08") in idx


def test_sessions_uses_configured_proxy_ticker():
    source = FakePolygonSource()
    default_proxy = sessions(source, "2024-01-01", "2024-01-10")
    other_proxy = sessions(source, "2024-01-01", "2024-01-10", proxy_ticker="QQQ")
    # Same fake calendar (bdate_range) regardless of ticker -- this just proves
    # the call succeeds with a non-default proxy and returns the same session set.
    assert list(default_proxy) == list(other_proxy)


def test_next_session_returns_next_business_day(session_index):
    friday = pd.Timestamp("2024-01-05")
    result = next_session(session_index, friday)
    assert result == pd.Timestamp("2024-01-08")


def test_next_session_returns_none_past_end_of_calendar(session_index):
    result = next_session(session_index, pd.Timestamp("2024-01-31"))
    assert result is None


def test_previous_session_returns_prior_business_day(session_index):
    monday = pd.Timestamp("2024-01-08")
    result = previous_session(session_index, monday)
    assert result == pd.Timestamp("2024-01-05")


def test_previous_session_returns_none_before_start_of_calendar(session_index):
    result = previous_session(session_index, pd.Timestamp("2024-01-01"))
    assert result is None


def test_shift_sessions_forward(session_index):
    start = pd.Timestamp("2024-01-02")
    result = shift_sessions(session_index, start, 2)
    assert result == pd.Timestamp("2024-01-04")


def test_shift_sessions_backward(session_index):
    start = pd.Timestamp("2024-01-10")
    result = shift_sessions(session_index, start, -2)
    assert result == pd.Timestamp("2024-01-08")


def test_shift_sessions_out_of_range_returns_none(session_index):
    result = shift_sessions(session_index, pd.Timestamp("2024-01-30"), 10)
    assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calendar.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.calendar'`

- [ ] **Step 3: Write `data_universe/calendar.py`**

```python
"""Trading session calendar, derived from a market proxy's own daily bars --
no separate holiday-calendar dependency. The proxy's bars only exist for
days it actually traded, so its dates *are* the trading calendar (subject to
whatever the underlying `DataSource` actually returns; `FakePolygonSource`
in tests only knows about weekends, not holidays, which is an accepted
simplification for fakes).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from data_universe.sources.base import DataSource


def sessions(
    source: DataSource, start: str, end: str, proxy_ticker: str = "SPY"
) -> pd.DatetimeIndex:
    """Trading session dates between `start` and `end` (inclusive), derived
    from `proxy_ticker`'s daily bars.

    Args:
        source: Any `DataSource` (real or fake) to pull the proxy's daily bars from.
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        proxy_ticker: The market-proxy ticker whose trading days define the
            calendar (default "SPY").

    Returns:
        A tz-naive `DatetimeIndex` named "date", sorted ascending.
    """
    bars = source.daily_bars(proxy_ticker, start, end)
    dates = bars.index.tz_localize(None).normalize()
    return pd.DatetimeIndex(dates, name="date")


def next_session(session_index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the first session strictly after `date`, or None if out of range."""
    position = session_index.searchsorted(date, side="right")
    if position >= len(session_index):
        return None
    return session_index[position]


def previous_session(session_index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]:
    """Return the last session strictly before `date`, or None if out of range."""
    position = session_index.searchsorted(date, side="left") - 1
    if position < 0:
        return None
    return session_index[position]


def shift_sessions(
    session_index: pd.DatetimeIndex, date: pd.Timestamp, n: int
) -> Optional[pd.Timestamp]:
    """Return the session `n` trading days from `date` (n may be negative to
    go backward). `date` must itself be a session in `session_index`.

    Args:
        session_index: A calendar as returned by `sessions()`.
        date: A session present in `session_index`.
        n: Number of trading days to shift.

    Returns:
        The shifted session date, or None if the shift goes out of range.
    """
    position = session_index.get_loc(date) + n
    if position < 0 or position >= len(session_index):
        return None
    return session_index[position]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calendar.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/calendar.py tests/test_calendar.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/calendar.py tests/test_calendar.py
git commit -m "feat: add trading-session calendar derived from a market proxy"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only).

**Interfaces:** none — this task only runs the suite built by Tasks 1-5 plus Phases 2-3's existing tests.

- [ ] **Step 1: Run the full local suite exactly as CI will**

Run: `ruff check .`
Expected: `All checks passed!`

Run: `pytest -v`
Expected: every test from Phases 2-3 plus this phase's five new test files PASSES, 0 failures, 0 errors.

- [ ] **Step 2: Confirm no network access is required**

Run: `python -c "import socket; socket.socket = None; import data_universe; from data_universe.sources import polygon, fred, edgar, sec_adapter; from data_universe import calendar; print('ok - no import-time network use')"`
Expected: prints `ok - no import-time network use`

- [ ] **Step 3: Nothing to commit** — this task only verifies Tasks 1-5's commits.

---

## Self-Review Notes (for the executor)

- `EdgarSource`'s submissions-JSON field names (`filings.recent.form/filingDate/acceptanceDateTime/items`) and the ticker-mapping file's `cik_str`/`ticker` fields are written from documented SEC EDGAR conventions but were not re-verified against live docs in this session — this matches `OPEN_QUESTIONS.md`'s existing flag on EDGAR endpoints; do not "fix" the mocked field names to something else without checking real EDGAR responses first.
- `FredSource`'s `api_key`-in-URL is an intentional, documented exception to the Polygon-specific "never in a URL" rule — do not "fix" it to a Bearer header; FRED's API doesn't support one.
- The options toolkit (`options/`), flat-files bulk S3 download (`sources/flatfiles.py`), features/labels, the PEAD event table, and `precompute.py` remain out of scope for this plan — `PolygonSource`, `FredSource`, `EdgarSource`, `sec_adapter`, and `calendar` are now ready for those later phases to build on, but do not start building them here.
