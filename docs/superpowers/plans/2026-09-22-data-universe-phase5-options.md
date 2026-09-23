# Data Universe Phase 5 (Options Toolkit) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the options toolkit shared by the VRP and PEAD projects: OCC ticker parsing/building, Black-Scholes pricing/vega/implied-vol, a Polygon options contract-reference + per-contract-aggregates source, chain helpers (ATM strike, nearest-expiry-after, straddle), and the documented `iv30` stub.

**Architecture:** `options/occ.py` and `options/black_scholes.py` are pure math, no I/O. `sources/polygon_options.py` adds a thin `PolygonOptionsSource` on top of Phase 4's `PolygonSource` (same auth/backoff/pagination machinery, reused by inheritance rather than duplicated) for the two options-specific endpoints: contract reference and per-contract daily aggregates. `options/chain.py` sits above that: `as_of()` calls a source's `contracts_as_of`, while `atm_strike`/`nearest_expiry_after`/`straddle` are pure functions over the chain `DataFrame` (plus, for `straddle`, a source to fetch two contract closes from) — decoupling chain logic from any specific source implementation. `options/iv30.py` stays a documented stub; nothing here builds it.

**Tech Stack:** Python ≥3.9, scipy (`optimize.brentq`, `stats.norm`) for Black-Scholes/IV, pandas for chain frames, requests for the options source, pytest + `unittest.mock` for source tests.

**Spec:** `PLAN.md` (repo root) §"Options toolkit" and the `options/` and `sources/polygon_options.py` entries in §2's file layout. This plan implements Phase 5 of `PLAN.md` §10 only: "Options toolkit: OCC, Black-Scholes and IV, chain helpers, IV30 stub." Features/labels, the PEAD event table, and `precompute.py` are later phases and are **not** built here — `chain.straddle`'s output (`call_price`, `put_price`, `call_volume`, `put_volume`) is deliberately raw; computing `is_stale` from it is Phase 6's `events/earnings.py` job, not this one's.

## Global Constraints

- `options/occ.py` and `options/black_scholes.py` contain **no I/O** — no `requests`, no `data_universe.config`, no `data_universe.cache`. Every function takes plain numbers/strings and returns plain numbers/dataclasses.
- `implied_vol` uses a **bracketed** solver (`scipy.optimize.brentq`), not Newton alone, and returns `float("nan")` — never raises — when the target price is outside the model's no-arbitrage bounds or no root can be bracketed.
- `PolygonOptionsSource` reuses `PolygonSource`'s auth/backoff/pagination (`_get`, `_headers`) rather than re-implementing them — subclass it, don't copy-paste Phase 4's code.
- No test may touch the network — every `PolygonOptionsSource` test injects a mocked `requests.Session`, same pattern as Phase 4.
- No `async def` anywhere in this phase.
- Do not invent Polygon options endpoint fields with false confidence — `contracts_as_of`/`daily_close`'s field names (`ticker, expiration_date, strike_price, contract_type` / `close, volume`) are written against Polygon's documented reference/aggregates conventions but are **not** re-verified live in this session; this matches the existing flag in `OPEN_QUESTIONS.md` on options endpoints. Do not "fix" the mocked field names without checking real Polygon responses first.
- `iv30.py` stays a stub that raises `NotImplementedError` naming its owner (vrp-research data track, Lionel) — do not implement the real interpolation logic.
- Docstrings on every public function/class/method, stating units (decimal vs. percent, annualized vs. per-period, dollars) where relevant.

---

### Task 1: `options/occ.py` — OCC ticker parsing and building

**Files:**
- Create: `data_universe/options/occ.py`
- Test: `tests/test_options_occ.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `@dataclass(frozen=True) class OccContract(underlying: str, expiry: date, is_call: bool, strike: float)`, `parse(ticker: str) -> OccContract`, `build(underlying: str, expiry: date, is_call: bool, strike: float) -> str`. Task 3's `PolygonOptionsSource.contracts_as_of` and Task 4's `chain.straddle` both work with OCC ticker strings that round-trip through these two functions.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_options_occ.py`:

```python
from datetime import date

import pytest

from data_universe.options.occ import OccContract, build, parse


def test_parse_known_ticker():
    contract = parse("O:AAPL240621C00190000")
    assert contract == OccContract(
        underlying="AAPL", expiry=date(2024, 6, 21), is_call=True, strike=190.0
    )


def test_parse_put():
    contract = parse("O:AAPL240621P00190000")
    assert contract.is_call is False


def test_parse_decimal_strike():
    contract = parse("O:AAPL240621C00190500")
    assert contract.strike == pytest.approx(190.5)


def test_parse_rejects_malformed_ticker():
    with pytest.raises(ValueError):
        parse("AAPL240621C00190000")  # missing "O:" prefix


def test_build_known_contract():
    ticker = build("AAPL", date(2024, 6, 21), is_call=True, strike=190.0)
    assert ticker == "O:AAPL240621C00190000"


def test_build_decimal_strike():
    ticker = build("AAPL", date(2024, 6, 21), is_call=False, strike=7.25)
    assert ticker == "O:AAPL240621P00007250"


@pytest.mark.parametrize(
    "underlying,expiry,is_call,strike",
    [
        ("AAPL", date(2024, 6, 21), True, 190.0),
        ("SPY", date(2025, 1, 17), False, 450.5),
        ("MSFT", date(2023, 12, 15), True, 7.25),
        ("TSLA", date(2026, 3, 20), False, 1000.125),
    ],
)
def test_round_trip(underlying, expiry, is_call, strike):
    ticker = build(underlying, expiry, is_call, strike)
    contract = parse(ticker)
    assert contract.underlying == underlying
    assert contract.expiry == expiry
    assert contract.is_call == is_call
    assert contract.strike == pytest.approx(strike, abs=0.001)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_options_occ.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.options.occ'`

- [ ] **Step 3: Write `data_universe/options/occ.py`**

```python
"""OCC-style option ticker parsing and building (Polygon's convention, e.g.
"O:AAPL240621C00190000").

Format: `O:{underlying}{YYMMDD}{C|P}{strike * 1000, zero-padded to 8 digits}`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

_PATTERN = re.compile(
    r"^O:(?P<underlying>[A-Z]+)(?P<expiry>\d{6})(?P<type>[CP])(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class OccContract:
    """A parsed OCC option contract ticker.

    Attributes:
        underlying: Underlying ticker symbol.
        expiry: Expiration date.
        is_call: True for a call, False for a put.
        strike: Strike price in dollars (may have fractional cents, e.g. 190.5).
    """

    underlying: str
    expiry: date
    is_call: bool
    strike: float


def parse(ticker: str) -> OccContract:
    """Parse an OCC-style option ticker, e.g. "O:AAPL240621C00190000".

    Args:
        ticker: OCC ticker, prefixed with "O:" (Polygon's convention).

    Returns:
        The parsed `OccContract`.

    Raises:
        ValueError: If `ticker` doesn't match the expected format.
    """
    match = _PATTERN.match(ticker)
    if not match:
        raise ValueError(f"Not a valid OCC option ticker: {ticker!r}")
    expiry = date(
        2000 + int(match["expiry"][0:2]),
        int(match["expiry"][2:4]),
        int(match["expiry"][4:6]),
    )
    strike = int(match["strike"]) / 1000.0
    return OccContract(
        underlying=match["underlying"],
        expiry=expiry,
        is_call=match["type"] == "C",
        strike=strike,
    )


def build(underlying: str, expiry: date, is_call: bool, strike: float) -> str:
    """Build an OCC-style option ticker from its parts.

    Args:
        underlying: Underlying ticker symbol.
        expiry: Expiration date.
        is_call: True for a call, False for a put.
        strike: Strike price in dollars (supports fractional cents, rounded
            to the nearest tenth of a cent -- OCC's own granularity).

    Returns:
        The OCC ticker string, e.g. "O:AAPL240621C00190000".
    """
    expiry_str = expiry.strftime("%y%m%d")
    type_char = "C" if is_call else "P"
    strike_int = round(strike * 1000)
    strike_str = f"{strike_int:08d}"
    return f"O:{underlying}{expiry_str}{type_char}{strike_str}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_options_occ.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/options/occ.py tests/test_options_occ.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/options/occ.py tests/test_options_occ.py
git commit -m "feat: add OCC option ticker parsing and building"
```

---

### Task 2: `options/black_scholes.py` — price, vega, implied vol

**Files:**
- Create: `data_universe/options/black_scholes.py`
- Test: `tests/test_options_black_scholes.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `price(S, K, T, r, q, sigma, is_call) -> float`, `vega(S, K, T, r, q, sigma) -> float`, `implied_vol(target_price, S, K, T, r, q, is_call) -> float`. Phase 6's VRP feature track (`iv_index`, `vrp_signal_rv21`) and Phase 6's PEAD event table (`implied_move` via `chain.straddle`) both build on `implied_vol`; `iv30.py` (Task 5) references this module's approach in its docstring without importing it (it stays a stub).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_options_black_scholes.py`:

```python
import math

import pytest

from data_universe.options.black_scholes import implied_vol, price, vega


def test_price_at_expiry_equals_intrinsic_value_call():
    assert price(S=110, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=True) == 10
    assert price(S=90, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=True) == 0


def test_price_at_expiry_equals_intrinsic_value_put():
    assert price(S=90, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=False) == 10
    assert price(S=110, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=False) == 0


def test_call_price_increases_with_volatility():
    low = price(S=100, K=100, T=1, r=0.02, q=0.0, sigma=0.1, is_call=True)
    high = price(S=100, K=100, T=1, r=0.02, q=0.0, sigma=0.4, is_call=True)
    assert high > low


def test_vega_is_positive_for_a_live_option():
    v = vega(S=100, K=100, T=1, r=0.02, q=0.0, sigma=0.2)
    assert v > 0


def test_vega_is_zero_at_expiry():
    assert vega(S=100, K=100, T=0, r=0.02, q=0.0, sigma=0.2) == 0


@pytest.mark.parametrize("moneyness", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("T", [0.05, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_recovers_planted_sigma(moneyness, T, is_call):
    S = 100.0
    K = S / moneyness
    r, q, sigma = 0.03, 0.01, 0.25
    target_price = price(S, K, T, r, q, sigma, is_call)
    recovered = implied_vol(target_price, S, K, T, r, q, is_call)
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_returns_nan_when_price_below_no_arbitrage_bound():
    result = implied_vol(target_price=-5.0, S=100, K=100, T=1, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)


def test_implied_vol_returns_nan_when_price_above_no_arbitrage_bound():
    # A call can never be worth more than the (discounted) spot price.
    result = implied_vol(target_price=1000.0, S=100, K=100, T=1, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)


def test_implied_vol_returns_nan_at_zero_time_to_expiry():
    result = implied_vol(target_price=10.0, S=110, K=100, T=0, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)


def test_implied_vol_never_raises_on_bad_inputs():
    # Should return nan, not raise, even for a degenerate/nonsensical price.
    result = implied_vol(target_price=float("inf"), S=100, K=100, T=1, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_options_black_scholes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.options.black_scholes'`

- [ ] **Step 3: Write `data_universe/options/black_scholes.py`**

```python
"""Black-Scholes pricing, vega, and implied volatility for European options.

Uses a continuous dividend yield `q`. `implied_vol` uses a bracketed solver
(Brent's method) rather than Newton alone, and returns NaN instead of
raising when the target price sits outside the model's no-arbitrage bounds
or a volatility root can't be bracketed in `[1e-6, 5.0]` (0.0001% to 500%
annualized vol -- wide enough for any real quote).
"""
from __future__ import annotations

import math

from scipy import optimize, stats

_MIN_SIGMA = 1e-6
_MAX_SIGMA = 5.0


def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price(S: float, K: float, T: float, r: float, q: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes price of a European option.

    Args:
        S: Spot price of the underlying, in dollars.
        K: Strike price, in dollars.
        T: Time to expiry, in years.
        r: Continuously-compounded risk-free rate (decimal, annualized).
        q: Continuously-compounded dividend yield (decimal, annualized).
        sigma: Annualized volatility (decimal).
        is_call: True for a call, False for a put.

    Returns:
        The option's theoretical price, in dollars.
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if is_call:
        return S * math.exp(-q * T) * stats.norm.cdf(d1) - K * math.exp(-r * T) * stats.norm.cdf(
            d2
        )
    return K * math.exp(-r * T) * stats.norm.cdf(-d2) - S * math.exp(-q * T) * stats.norm.cdf(-d1)


def vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega: sensitivity of price to a 1.0 (100 percentage
    point) change in `sigma`. Identical formula for calls and puts.

    Args: see `price` (vega doesn't depend on `is_call`).

    Returns:
        Vega, in dollars per 1.0 change in `sigma`.
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return S * math.exp(-q * T) * stats.norm.pdf(d1) * math.sqrt(T)


def _no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float, is_call: bool):
    discounted_s = S * math.exp(-q * T)
    discounted_k = K * math.exp(-r * T)
    if is_call:
        return max(discounted_s - discounted_k, 0.0), discounted_s
    return max(discounted_k - discounted_s, 0.0), discounted_k


def implied_vol(
    target_price: float, S: float, K: float, T: float, r: float, q: float, is_call: bool
) -> float:
    """Solve for the annualized volatility that reprices `target_price`,
    using a bracketed root finder (Brent's method).

    Args:
        target_price: Observed option price, in dollars.
        S: Spot price of the underlying, in dollars.
        K: Strike price, in dollars.
        T: Time to expiry, in years.
        r: Continuously-compounded risk-free rate (decimal, annualized).
        q: Continuously-compounded dividend yield (decimal, annualized).
        is_call: True for a call, False for a put.

    Returns:
        The implied volatility (decimal, annualized), or `float("nan")` if
        `target_price` is outside the model's no-arbitrage bounds, `T <= 0`,
        `target_price` isn't finite, or no root can be bracketed.
    """
    if T <= 0 or not math.isfinite(target_price):
        return float("nan")
    lower, upper = _no_arbitrage_bounds(S, K, T, r, q, is_call)
    if target_price < lower or target_price > upper:
        return float("nan")

    def objective(sigma: float) -> float:
        return price(S, K, T, r, q, sigma, is_call) - target_price

    f_lo = objective(_MIN_SIGMA)
    f_hi = objective(_MAX_SIGMA)
    if f_lo * f_hi > 0:
        return float("nan")
    try:
        return optimize.brentq(objective, _MIN_SIGMA, _MAX_SIGMA)
    except ValueError:
        return float("nan")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_options_black_scholes.py -v`
Expected: all tests PASS (72 parametrized cases from the moneyness/T/is_call grid plus the rest)

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/options/black_scholes.py tests/test_options_black_scholes.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/options/black_scholes.py tests/test_options_black_scholes.py
git commit -m "feat: add Black-Scholes price, vega, and bracketed implied-vol solver"
```

---

### Task 3: `sources/polygon_options.py` — contract reference and per-contract daily aggregates

**Files:**
- Create: `data_universe/sources/polygon_options.py`
- Test: `tests/test_sources_polygon_options.py`

**Interfaces:**
- Consumes: `data_universe.sources.polygon.PolygonSource` (Phase 4, for `_get`/`_headers`/backoff/pagination), `data_universe.cache.cached` (Phase 3).
- Produces: `class PolygonOptionsSource(session=None, max_retries=5, backoff_seconds=1.0)` (same constructor as `PolygonSource`, inherited) with `.contracts_as_of(underlying, as_of_date) -> pd.DataFrame` (columns `ticker, expiration_date, strike_price, contract_type`) and `.daily_close(option_ticker, date) -> dict` (keys `close, volume`). Task 4's `chain.as_of`/`chain.straddle` consume both methods through this exact interface.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_polygon_options.py`:

```python
from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.polygon_options import PolygonOptionsSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_polygon_key("test-key")
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, json_data, status_code=200):
        self.status_code = status_code
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def _contract(ticker, expiration_date, strike_price, contract_type):
    return {
        "ticker": ticker,
        "expiration_date": expiration_date,
        "strike_price": strike_price,
        "contract_type": contract_type,
    }


def test_contracts_as_of_parses_results():
    session = Mock()
    session.get.return_value = FakeResponse(
        {
            "results": [
                _contract("O:AAPL240621C00190000", "2024-06-21", 190.0, "call"),
                _contract("O:AAPL240621P00190000", "2024-06-21", 190.0, "put"),
            ]
        }
    )
    source = PolygonOptionsSource(session=session)
    chain = source.contracts_as_of("AAPL", "2024-01-02")
    assert list(chain.columns) == ["ticker", "expiration_date", "strike_price", "contract_type"]
    assert len(chain) == 2


def test_contracts_as_of_sends_bearer_header():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    source.contracts_as_of("AAPL", "2024-01-02")
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"


def test_contracts_as_of_follows_next_url_pagination():
    page1 = FakeResponse(
        {
            "results": [_contract("O:AAPL240621C00190000", "2024-06-21", 190.0, "call")],
            "next_url": "https://api.polygon.io/v3/reference/options/contracts?cursor=abc",
        }
    )
    page2 = FakeResponse(
        {"results": [_contract("O:AAPL240621P00190000", "2024-06-21", 190.0, "put")]}
    )
    session = Mock()
    session.get.side_effect = [page1, page2]
    source = PolygonOptionsSource(session=session)
    chain = source.contracts_as_of("AAPL", "2024-01-02")
    assert len(chain) == 2
    assert session.get.call_count == 2


def test_contracts_as_of_empty_results_has_expected_columns():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    chain = source.contracts_as_of("AAPL", "2024-01-02")
    assert chain.empty
    assert list(chain.columns) == ["ticker", "expiration_date", "strike_price", "contract_type"]


def test_daily_close_parses_close_and_volume():
    session = Mock()
    session.get.return_value = FakeResponse({"close": 5.5, "volume": 120})
    source = PolygonOptionsSource(session=session)
    result = source.daily_close("O:AAPL240621C00190000", "2024-06-20")
    assert result == {"close": 5.5, "volume": 120.0}


def test_daily_close_defaults_missing_fields_to_zero():
    session = Mock()
    session.get.return_value = FakeResponse({})
    source = PolygonOptionsSource(session=session)
    result = source.daily_close("O:AAPL240621C00190000", "2024-06-20")
    assert result == {"close": 0.0, "volume": 0.0}


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse({"results": []})
    source = PolygonOptionsSource(session=session)
    source.contracts_as_of("AAPL", "2024-01-02")
    source.contracts_as_of("AAPL", "2024-01-02")
    assert session.get.call_count == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_polygon_options.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.polygon_options'`

- [ ] **Step 3: Write `data_universe/sources/polygon_options.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_polygon_options.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/sources/polygon_options.py tests/test_sources_polygon_options.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/sources/polygon_options.py tests/test_sources_polygon_options.py
git commit -m "feat: add PolygonOptionsSource for contract reference and daily closes"
```

---

### Task 4: `options/chain.py` — ATM strike, nearest-expiry-after, straddle

**Files:**
- Create: `data_universe/options/chain.py`
- Test: `tests/test_options_chain.py`

**Interfaces:**
- Consumes: any object exposing `.contracts_as_of(underlying, as_of_date) -> pd.DataFrame` (for `as_of`) or `.daily_close(option_ticker, date) -> dict` (for `straddle`) — `PolygonOptionsSource` (Task 3) satisfies both, and tests use a lightweight stub so this task's tests don't depend on Task 3's mocked-HTTP machinery.
- Produces: `as_of(source, underlying, as_of_date) -> pd.DataFrame`, `atm_strike(chain, spot) -> float`, `nearest_expiry_after(chain, date) -> pd.Timestamp`, `straddle(source, chain, strike, expiry, quote_date) -> dict` (keys `call_ticker, put_ticker, call_price, put_price, call_volume, put_volume`). Phase 6's PEAD event table builder is the primary consumer of `atm_strike`, `nearest_expiry_after`, and `straddle`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_options_chain.py`:

```python
import pandas as pd
import pytest

from data_universe.options.chain import as_of, atm_strike, nearest_expiry_after, straddle


def _chain():
    return pd.DataFrame(
        [
            {"ticker": "O:AAPL240621C00190000", "expiration_date": "2024-06-21", "strike_price": 190.0, "contract_type": "call"},
            {"ticker": "O:AAPL240621P00190000", "expiration_date": "2024-06-21", "strike_price": 190.0, "contract_type": "put"},
            {"ticker": "O:AAPL240621C00195000", "expiration_date": "2024-06-21", "strike_price": 195.0, "contract_type": "call"},
            {"ticker": "O:AAPL240719C00190000", "expiration_date": "2024-07-19", "strike_price": 190.0, "contract_type": "call"},
        ]
    )


class StubSource:
    def __init__(self, chain=None, closes=None):
        self._chain = chain
        self._closes = closes or {}

    def contracts_as_of(self, underlying, as_of_date):
        return self._chain

    def daily_close(self, option_ticker, date):
        return self._closes[(option_ticker, date)]


def test_as_of_delegates_to_source_contracts_as_of():
    stub = StubSource(chain=_chain())
    result = as_of(stub, "AAPL", "2024-01-02")
    pd.testing.assert_frame_equal(result, _chain())


def test_atm_strike_picks_closest_strike():
    assert atm_strike(_chain(), spot=191.0) == 190.0
    assert atm_strike(_chain(), spot=194.0) == 195.0


def test_atm_strike_raises_on_empty_chain():
    empty = _chain().iloc[0:0]
    with pytest.raises(ValueError):
        atm_strike(empty, spot=100.0)


def test_nearest_expiry_after_picks_first_future_expiry():
    result = nearest_expiry_after(_chain(), "2024-06-01")
    assert result == pd.Timestamp("2024-06-21")


def test_nearest_expiry_after_skips_expiries_on_or_before_date():
    result = nearest_expiry_after(_chain(), "2024-06-21")
    assert result == pd.Timestamp("2024-07-19")


def test_nearest_expiry_after_raises_when_no_future_expiry():
    with pytest.raises(ValueError):
        nearest_expiry_after(_chain(), "2024-12-31")


def test_straddle_returns_prices_and_volumes():
    stub = StubSource(
        chain=_chain(),
        closes={
            ("O:AAPL240621C00190000", "2024-06-20"): {"close": 6.0, "volume": 100.0},
            ("O:AAPL240621P00190000", "2024-06-20"): {"close": 5.5, "volume": 80.0},
        },
    )
    result = straddle(
        stub, _chain(), strike=190.0, expiry=pd.Timestamp("2024-06-21"), quote_date="2024-06-20"
    )
    assert result == {
        "call_ticker": "O:AAPL240621C00190000",
        "put_ticker": "O:AAPL240621P00190000",
        "call_price": 6.0,
        "put_price": 5.5,
        "call_volume": 100.0,
        "put_volume": 80.0,
    }


def test_straddle_raises_when_missing_a_leg():
    stub = StubSource(chain=_chain())
    with pytest.raises(ValueError):
        straddle(
            stub, _chain(), strike=999.0, expiry=pd.Timestamp("2024-06-21"), quote_date="2024-06-20"
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_options_chain.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.options.chain'`

- [ ] **Step 3: Write `data_universe/options/chain.py`**

```python
"""Option chain helpers: listed contracts as of a date, ATM strike selection,
nearest-expiry-after selection, and straddle construction.

Every function here takes a chain `DataFrame` (columns `ticker`,
`expiration_date`, `strike_price`, `contract_type`) or a `source` object
exposing `.contracts_as_of`/`.daily_close` -- never a concrete class -- so
these helpers work with `PolygonOptionsSource` or any test double with the
same shape.
"""
from __future__ import annotations

import pandas as pd


def as_of(source, underlying: str, as_of_date: str) -> pd.DataFrame:
    """The listed option chain for `underlying` as of `as_of_date`.

    Args:
        source: An object with `.contracts_as_of(underlying, as_of_date)`.
        underlying: Underlying ticker symbol.
        as_of_date: Date "YYYY-MM-DD".

    Returns:
        A chain DataFrame (see module docstring for the schema).
    """
    return source.contracts_as_of(underlying, as_of_date)


def atm_strike(chain: pd.DataFrame, spot: float) -> float:
    """The strike closest to `spot` present in `chain`.

    Args:
        chain: A chain DataFrame.
        spot: The underlying's spot price.

    Returns:
        The closest available strike price.

    Raises:
        ValueError: If `chain` has no strikes.
    """
    strikes = chain["strike_price"].unique()
    if len(strikes) == 0:
        raise ValueError("Chain has no strikes")
    return float(min(strikes, key=lambda k: abs(k - spot)))


def nearest_expiry_after(chain: pd.DataFrame, date: str) -> pd.Timestamp:
    """The nearest expiration strictly after `date`.

    Args:
        chain: A chain DataFrame.
        date: Reference date, "YYYY-MM-DD".

    Returns:
        The nearest expiration date strictly after `date`.

    Raises:
        ValueError: If no expiration in `chain` is after `date`.
    """
    reference = pd.Timestamp(date)
    expiries = pd.to_datetime(chain["expiration_date"]).unique()
    future = sorted(e for e in expiries if pd.Timestamp(e) > reference)
    if not future:
        raise ValueError(f"No expiration in chain after {date}")
    return pd.Timestamp(future[0])


def straddle(
    source, chain: pd.DataFrame, strike: float, expiry: pd.Timestamp, quote_date: str
) -> dict:
    """The ATM straddle (one call + one put at `strike`/`expiry`) priced on `quote_date`.

    Args:
        source: An object with `.daily_close(option_ticker, date)`.
        chain: A chain DataFrame, used to find the call/put OCC tickers for
            `strike`/`expiry`.
        strike: The straddle's strike.
        expiry: The straddle's expiration date.
        quote_date: Date "YYYY-MM-DD" to price the straddle on.

    Returns:
        A dict with `call_ticker`, `put_ticker`, `call_price`, `put_price`,
        `call_volume`, `put_volume`.

    Raises:
        ValueError: If `chain` doesn't have both a call and a put at
            `strike`/`expiry`.
    """
    expiry_str = pd.Timestamp(expiry).strftime("%Y-%m-%d")
    matches = chain[
        (chain["strike_price"] == strike)
        & (pd.to_datetime(chain["expiration_date"]).dt.strftime("%Y-%m-%d") == expiry_str)
    ]
    calls = matches[matches["contract_type"] == "call"]
    puts = matches[matches["contract_type"] == "put"]
    if calls.empty or puts.empty:
        raise ValueError(f"Chain missing a call/put at strike={strike}, expiry={expiry_str}")
    call_ticker = calls.iloc[0]["ticker"]
    put_ticker = puts.iloc[0]["ticker"]
    call_data = source.daily_close(call_ticker, quote_date)
    put_data = source.daily_close(put_ticker, quote_date)
    return {
        "call_ticker": call_ticker,
        "put_ticker": put_ticker,
        "call_price": call_data["close"],
        "put_price": put_data["close"],
        "call_volume": call_data["volume"],
        "put_volume": put_data["volume"],
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_options_chain.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/options/chain.py tests/test_options_chain.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/options/chain.py tests/test_options_chain.py
git commit -m "feat: add chain helpers (as_of, atm_strike, nearest_expiry_after, straddle)"
```

---

### Task 5: `options/iv30.py` — documented stub

**Files:**
- Create: `data_universe/options/iv30.py`
- Test: `tests/test_options_iv30.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `iv30(underlying: str, query_date: str) -> float`, always raising `NotImplementedError`. Phase 6's `features/library.py` (`iv_index` for single names, `vrp_signal_rv21`, and the label `vrp_expost`) calls this and lets the `NotImplementedError` propagate, per `PLAN.md`'s "stub until IV30 exists."

- [ ] **Step 1: Write the failing test**

Create `tests/test_options_iv30.py`:

```python
import pytest

from data_universe.options.iv30 import iv30


def test_iv30_raises_not_implemented_naming_its_owner():
    with pytest.raises(NotImplementedError) as exc_info:
        iv30("AAPL", "2024-01-02")
    message = str(exc_info.value)
    assert "vrp-research" in message
    assert "Lionel" in message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_options_iv30.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.options.iv30'`

- [ ] **Step 3: Write `data_universe/options/iv30.py`**

```python
"""30-day constant-maturity ATM implied volatility (IV30).

STUB -- owner: vrp-research data track (Lionel). Intended method, once
IV30 is needed for real: take the two listed expiries bracketing 30
calendar days from `query_date` (via `data_universe.options.chain`),
compute each one's ATM implied vol (via `chain.atm_strike` and
`data_universe.options.black_scholes.implied_vol`), convert each to total
variance (`sigma**2 * T`), linearly interpolate total variance at
`T = 30/365`, then convert back to a volatility
(`sqrt(interpolated_total_variance / T)`). This module intentionally does
not implement that yet.
"""
from __future__ import annotations


def iv30(underlying: str, query_date: str) -> float:
    """Not implemented. See module docstring for the intended method.

    Args:
        underlying: Underlying ticker symbol.
        query_date: Date "YYYY-MM-DD" to compute IV30 as of.

    Raises:
        NotImplementedError: Always. Owner: vrp-research data track (Lionel).
    """
    raise NotImplementedError(
        "IV30 is not implemented yet. Owner: vrp-research data track (Lionel). "
        "Intended method: interpolate ATM IV across the two expiries "
        "bracketing 30 days, in total variance -- see module docstring."
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_options_iv30.py -v`
Expected: PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/options/iv30.py tests/test_options_iv30.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/options/iv30.py tests/test_options_iv30.py
git commit -m "feat: add documented iv30 stub (owner: vrp-research data track)"
```

---

### Task 6: Full-suite verification

**Files:** none (verification only).

**Interfaces:** none — this task only runs the suite built by Tasks 1-5 plus Phases 2-4's existing tests.

- [ ] **Step 1: Run the full local suite exactly as CI will**

Run: `ruff check .`
Expected: `All checks passed!`

Run: `pytest -v`
Expected: every test from Phases 2-4 plus this phase's six new test files PASSES, 0 failures, 0 errors.

- [ ] **Step 2: Confirm no network access is required**

```
python -c "
import socket

def _blocked(*args, **kwargs):
    raise AssertionError('network access attempted')

socket.socket.connect = _blocked

import data_universe
from data_universe.options import occ, black_scholes, chain, iv30
from data_universe.sources import polygon_options
print('ok - no import-time network use')
"
```

Expected: prints `ok - no import-time network use`

- [ ] **Step 3: Nothing to commit** — this task only verifies Tasks 1-5's commits.

---

## Self-Review Notes (for the executor)

- `chain.py` deliberately takes duck-typed `source` objects, not `PolygonOptionsSource` by name, so Task 4 can be implemented and tested before or independently of Task 3 if the executor reorders them — but this plan orders Task 3 before Task 4 anyway since `PolygonOptionsSource` is the real object `chain.as_of` will be handed outside of tests.
- `chain.straddle`'s returned dict has no `is_stale` key — do not add one here. Computing `is_stale` from `call_volume`/`put_volume` is explicitly Phase 6's `events/earnings.py` job (spec issue #6 in `PLAN.md`), so that logic lives with the PEAD event table, not in the generic chain helper.
- Features/labels, the PEAD event table, `precompute.py`, and the real `iv30` implementation remain out of scope — `occ`, `black_scholes`, `PolygonOptionsSource`, and `chain` are now ready for Phase 6 to build on, but do not start building Phase 6 here.
