# Phase 6: Features, Labels, and the PEAD Event Table

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the feature/label registries (strictly separated, `fwd_` prefix enforced only on
labels), the first concrete feature/label sets for the beta, VRP, and PEAD projects, the PEAD
event table builder, and a generic no-lookahead test that runs over every registered
feature/label.

**Architecture:** `features/core.py` holds pure, I/O-free math (log returns, three realized-vol
estimators, rolling OLS beta/alpha). `features/registry.py` defines `FEATURE_REGISTRY`, the
`@register` decorator, and a shared `compute()` that pads the requested date range backward by
each feature's declared warmup (via `calendar.py`), calls the registered function, and slices the
result back to `[start, end]`. `labels/registry.py` reuses that same `compute()` but its
`@register` enforces the opposite naming rule (`fwd_` required, not forbidden). `features/
library.py` / `labels/library.py` register the concrete functions against that shared interface.
`events/earnings.py` is a separate, higher-level builder that composes `EdgarSource`,
`PolygonSource`, and `options/chain.py` into one PEAD event row per filing.

**Tech Stack:** pandas, numpy (already in requirements.txt — no new dependencies this phase).

**Spec:** `/Users/vaibhav.wudaru/data-universe/PLAN.md` — see §3 (`data_universe/features`,
`labels`, `events` in the file layout), §6 ("Features vs. labels — enforcement"), §7
("Project-specific pieces"), §8 ("Spec issues — flagged with named columns"), §10 phase 6.

## Global Constraints

- Package name is `data_universe` (not `data-universe`); import as `import data_universe as q`.
- No test may touch the network. Every source-like collaborator in tests is a fake or a
  `unittest.mock.Mock`/hand-written stub — never `requests` against a real host.
- Feature names must **not** start with `fwd_`; label names **must** start with `fwd_`.
  Violating either raises at `@register(...)` decoration time (import time), not later.
- Every function operating on price data assumes split/dividend-adjusted prices unless its name
  says "raw"/"unadjusted" (`PLAN.md` §4) — none of this phase's functions need raw prices, so none
  are named that way.
- Daily feature/label frames are tz-naive, indexed by session date (`PLAN.md` §4). Single-ticker
  results are a `pd.Series` indexed by date, named after the feature/label.
- Every "spec issue to flag" in `PLAN.md` §8 gets its own **named column** in the PEAD event
  table — never silently resolved into one blended number.
- `next_earnings_session` is a realized (lookback) value, not a forecast — document this
  explicitly in its docstring, matching `PLAN.md` §8 issue 5.
- Bump `__version__` in `data_universe/__init__.py` is tracked as a known gap (not yet honored in
  any commit); this phase does not need to fix that on its own, but do not make it worse — no
  action required here beyond what prior phases already did (i.e., none).
- `ruff check .` and `pytest -q` must both be green after every task's commit.

## Existing interfaces this plan builds on (read-only reference, not to be modified)

```python
# data_universe/sources/base.py
NY_TZ = "America/New_York"
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]
class DataSource(abc.ABC):
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...

# data_universe/sources/fakes.py
class FakePolygonSource:
    def __init__(self, market: Optional[FakeMarket] = None) -> None: ...
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
        # only ticker == "I:VIX" is supported; anything else raises ValueError
PLANTED_BETA = 1.5
PLANTED_ALPHA = 0.0
# FakeMarket: stock_return = alpha + beta * proxy_return + noise, proxy_ticker default "SPY"

# data_universe/calendar.py
def sessions(source: DataSource, start: str, end: str, proxy_ticker: str = "SPY") -> pd.DatetimeIndex: ...
def next_session(session_index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]: ...
def previous_session(session_index: pd.DatetimeIndex, date: pd.Timestamp) -> Optional[pd.Timestamp]: ...
def shift_sessions(session_index: pd.DatetimeIndex, date: pd.Timestamp, n: int) -> Optional[pd.Timestamp]: ...

# data_universe/options/chain.py
def as_of(source, underlying: str, as_of_date: str) -> pd.DataFrame: ...
def atm_strike(chain: pd.DataFrame, spot: float) -> float: ...
def nearest_expiry_after(chain: pd.DataFrame, date: str) -> pd.Timestamp: ...
def straddle(source, chain, strike: float, expiry: pd.Timestamp, quote_date: str) -> dict:
    # -> {"call_ticker", "put_ticker", "call_price", "put_price", "call_volume", "put_volume"}

# data_universe/options/iv30.py
def iv30(underlying: str, query_date: str) -> float:
    # always raises NotImplementedError("...Owner: vrp-research data track (Lionel)...")

# data_universe/sources/edgar.py
class EdgarSource:
    def get_cik(self, ticker: str) -> str: ...
    def earnings_8k_filings(self, cik: str) -> pd.DataFrame:
        # columns: filing_date, acceptance_datetime, form, items; sorted ascending by acceptance_datetime
```

---

### Task 1: `features/core.py` — pure math

**Files:**
- Create: `data_universe/features/core.py`
- Create: `data_universe/features/__init__.py` (empty)
- Test: `tests/test_features_core.py`

**Interfaces:**
- Consumes: nothing (pure functions over `pd.Series`/`pd.DataFrame` — no I/O, no cache, no config).
- Produces (used by Task 4/5):
  - `log_returns(closes: pd.Series) -> pd.Series`
  - `realized_vol_cc(closes: pd.Series, window: int = 21) -> pd.Series`
  - `realized_vol_parkinson(highs: pd.Series, lows: pd.Series, window: int = 21) -> pd.Series`
  - `realized_vol_garman_klass(opens: pd.Series, highs: pd.Series, lows: pd.Series, closes: pd.Series, window: int = 21) -> pd.Series`
  - `rolling_ols_beta_alpha(asset_returns: pd.Series, market_returns: pd.Series, window: int) -> tuple[pd.Series, pd.Series]` (returns `(beta, alpha)`, both indexed like the inputs)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_core.py
import numpy as np
import pandas as pd
import pytest

from data_universe.features import core


def _closes(values):
    index = pd.bdate_range("2024-01-02", periods=len(values), name="date")
    return pd.Series(values, index=index, dtype="float64")


def test_log_returns_basic():
    closes = _closes([100.0, 105.0, 100.0])
    result = core.log_returns(closes)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.log(105.0 / 100.0))
    assert result.iloc[2] == pytest.approx(np.log(100.0 / 105.0))


def test_realized_vol_cc_annualized_matches_manual_calc():
    rng = np.random.default_rng(0)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=40)))
    closes = _closes(values)
    result = core.realized_vol_cc(closes, window=21)
    returns = core.log_returns(closes)
    expected_last = returns.iloc[-21:].std(ddof=1) * np.sqrt(252)
    assert result.iloc[-1] == pytest.approx(expected_last)
    assert result.iloc[:20].isna().all()  # not enough history for the first 20 rows


def test_realized_vol_parkinson_positive_and_uses_high_low_range():
    index = pd.bdate_range("2024-01-02", periods=25, name="date")
    highs = pd.Series(101.0, index=index)
    lows = pd.Series(99.0, index=index)
    result = core.realized_vol_parkinson(highs, lows, window=21)
    assert (result.dropna() > 0).all()
    flat_highs = pd.Series(100.0, index=index)
    flat_lows = pd.Series(100.0, index=index)
    zero_range = core.realized_vol_parkinson(flat_highs, flat_lows, window=21)
    assert zero_range.dropna().eq(0.0).all()


def test_realized_vol_garman_klass_zero_when_flat():
    index = pd.bdate_range("2024-01-02", periods=25, name="date")
    flat = pd.Series(100.0, index=index)
    result = core.realized_vol_garman_klass(flat, flat, flat, flat, window=21)
    assert result.dropna().eq(0.0).all()


def test_rolling_ols_beta_alpha_recovers_planted_relationship():
    rng = np.random.default_rng(1)
    n = 120
    index = pd.bdate_range("2024-01-02", periods=n, name="date")
    market_returns = pd.Series(rng.normal(0.0003, 0.01, size=n), index=index)
    planted_beta, planted_alpha = 1.5, 0.0002
    noise = pd.Series(rng.normal(0.0, 0.0005, size=n), index=index)
    asset_returns = planted_alpha + planted_beta * market_returns + noise

    beta, alpha = core.rolling_ols_beta_alpha(asset_returns, market_returns, window=60)

    assert beta.iloc[-1] == pytest.approx(planted_beta, abs=0.15)
    assert alpha.iloc[-1] == pytest.approx(planted_alpha, abs=0.001)
    assert beta.iloc[:59].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_core.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_universe.features'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/features/__init__.py
```

```python
# data_universe/features/core.py
"""Pure math used by registered features and labels: log returns, three
realized-volatility estimators, and rolling OLS beta/alpha. No I/O, no
cache, no config -- every function here takes and returns plain
`pd.Series`/`pd.DataFrame` so it can be unit tested without any data source.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252


def log_returns(closes: pd.Series) -> pd.Series:
    """Day-over-day log returns. First value is always NaN (no prior close)."""
    return np.log(closes / closes.shift(1))


def realized_vol_cc(closes: pd.Series, window: int = 21) -> pd.Series:
    """Annualized close-to-close realized volatility: rolling stdev of log
    returns over `window` trading days, scaled by sqrt(252).
    """
    returns = log_returns(closes)
    return returns.rolling(window).std(ddof=1) * np.sqrt(_TRADING_DAYS_PER_YEAR)


def realized_vol_parkinson(highs: pd.Series, lows: pd.Series, window: int = 21) -> pd.Series:
    """Annualized Parkinson realized volatility from the high/low range,
    over a rolling `window` of trading days.
    """
    log_range_sq = np.log(highs / lows) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    variance = factor * log_range_sq.rolling(window).mean()
    return np.sqrt(variance * _TRADING_DAYS_PER_YEAR)


def realized_vol_garman_klass(
    opens: pd.Series, highs: pd.Series, lows: pd.Series, closes: pd.Series, window: int = 21
) -> pd.Series:
    """Annualized Garman-Klass realized volatility, over a rolling `window`
    of trading days.
    """
    log_hl = np.log(highs / lows) ** 2
    log_co = np.log(closes / opens) ** 2
    daily_variance = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    variance = daily_variance.rolling(window).mean()
    return np.sqrt(variance.clip(lower=0.0) * _TRADING_DAYS_PER_YEAR)


def rolling_ols_beta_alpha(
    asset_returns: pd.Series, market_returns: pd.Series, window: int
) -> tuple:
    """Rolling univariate OLS of `asset_returns` on `market_returns`:
    `asset = alpha + beta * market`.

    Args:
        asset_returns: Per-period returns of the asset.
        market_returns: Per-period returns of the market proxy, same index.
        window: Rolling window length in periods.

    Returns:
        `(beta, alpha)`, both `pd.Series` aligned to `asset_returns.index`.
    """
    covariance = asset_returns.rolling(window).cov(market_returns)
    market_variance = market_returns.rolling(window).var(ddof=1)
    beta = covariance / market_variance
    alpha = asset_returns.rolling(window).mean() - beta * market_returns.rolling(window).mean()
    return beta, alpha
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_core.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/features/core.py data_universe/features/__init__.py tests/test_features_core.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/features/__init__.py data_universe/features/core.py tests/test_features_core.py
git commit -m "feat: add pure feature math (returns, realized vol estimators, rolling beta/alpha)"
```

---

### Task 2: `features/registry.py` — registry, `@register`, `compute()`

**Files:**
- Create: `data_universe/features/registry.py`
- Test: `tests/test_features_registry.py`

**Interfaces:**
- Consumes: `data_universe.sources.base.DataSource`, `data_universe.calendar.sessions` /
  `shift_sessions` (Existing interfaces section above), `data_universe.sources.fakes.
  FakePolygonSource` (for tests).
- Produces (used by Task 3, 4, 6, 7):
  - `FEATURE_REGISTRY: dict[str, FeatureSpec]`
  - `class FeatureSpec: fn: Callable, warmup_trading_days: int, project: str, description: str`
  - `register(name: str, warmup_trading_days: int = 0, project: str = "", description: str = "", registry: dict | None = None, forbidden_prefix: str = "fwd_") -> Callable` — decorator
    factory. Raises `ValueError` at decoration time if `name.startswith(forbidden_prefix)`.
    Registers into `registry` (defaults to `FEATURE_REGISTRY`) under `name`; raises `ValueError`
    on a duplicate name.
  - `compute(registry: dict, name: str, ticker: str, start: str, end: str, source: DataSource, calendar_proxy: str = "SPY", **kwargs) -> pd.Series` — looks up `name` in `registry`, pads
    `start` backward by `warmup_trading_days` trading sessions (via `calendar.sessions`/
    `shift_sessions`), calls `fn(ticker, padded_start, end, source, **kwargs)`, slices the
    resulting `pd.Series` back to `[start, end]` inclusive, and returns it.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_registry.py
import pandas as pd
import pytest

from data_universe.features import registry
from data_universe.sources.fakes import FakePolygonSource


def test_register_rejects_fwd_prefixed_names():
    local_registry = {}
    with pytest.raises(ValueError):
        @registry.register("fwd_something", registry=local_registry)
        def _fn(ticker, start, end, source):
            return pd.Series(dtype="float64")


def test_register_rejects_duplicate_names():
    local_registry = {}

    @registry.register("dup", registry=local_registry)
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    with pytest.raises(ValueError):
        @registry.register("dup", registry=local_registry)
        def _fn2(ticker, start, end, source):
            return pd.Series(dtype="float64")


def test_register_stores_metadata():
    local_registry = {}

    @registry.register("with_meta", warmup_trading_days=5, project="beta", description="desc", registry=local_registry)
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    spec = local_registry["with_meta"]
    assert spec.warmup_trading_days == 5
    assert spec.project == "beta"
    assert spec.description == "desc"
    assert spec.fn is _fn


def test_compute_pads_start_backward_by_warmup_and_slices_result():
    local_registry = {}
    seen_ranges = []

    @registry.register("recorder", warmup_trading_days=3, registry=local_registry)
    def _fn(ticker, start, end, source):
        seen_ranges.append((start, end))
        bars = source.daily_bars(ticker, start, end)
        return pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())

    source = FakePolygonSource()
    result = registry.compute(
        local_registry, "recorder", "AAPL", "2024-02-01", "2024-02-10", source
    )

    called_start, called_end = seen_ranges[0]
    assert pd.Timestamp(called_start) < pd.Timestamp("2024-02-01")
    assert called_end == "2024-02-10"
    assert result.index.min() >= pd.Timestamp("2024-02-01")
    assert result.index.max() <= pd.Timestamp("2024-02-10")


def test_compute_raises_for_unknown_name():
    with pytest.raises(KeyError):
        registry.compute({}, "nope", "AAPL", "2024-02-01", "2024-02-10", FakePolygonSource())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_universe.features.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/features/registry.py
"""The feature registry: `@register` (rejects `fwd_`-prefixed names -- those
belong in `labels/registry.py`) and `compute()`, the shared lookup-pad-call-
slice logic reused by both `features/library.py` and (via import)
`labels/registry.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from data_universe import calendar
from data_universe.sources.base import DataSource

FORBIDDEN_PREFIX = "fwd_"


@dataclass
class FeatureSpec:
    fn: Callable
    warmup_trading_days: int
    project: str
    description: str


FEATURE_REGISTRY: dict = {}


def register(
    name: str,
    warmup_trading_days: int = 0,
    project: str = "",
    description: str = "",
    registry: Optional[dict] = None,
    forbidden_prefix: str = FORBIDDEN_PREFIX,
) -> Callable:
    """Decorator factory registering a feature function under `name`.

    Args:
        name: Feature name. Must NOT start with `forbidden_prefix` ("fwd_").
        warmup_trading_days: Trading days of history `compute()` fetches
            before the requested `start`, so the first returned value isn't
            NaN purely for lack of history.
        project: Free-text owning project tag (e.g. "beta", "vrp", "pead").
        description: Human-readable description.
        registry: The dict to register into. Defaults to `FEATURE_REGISTRY`.
        forbidden_prefix: Prefix that is not allowed for a feature name.

    Raises:
        ValueError: If `name` starts with `forbidden_prefix`, or if `name`
            is already registered.
    """
    target = FEATURE_REGISTRY if registry is None else registry

    def decorator(fn: Callable) -> Callable:
        if name.startswith(forbidden_prefix):
            raise ValueError(
                f"Feature name {name!r} must not start with {forbidden_prefix!r} "
                "-- that prefix is reserved for labels/registry.py."
            )
        if name in target:
            raise ValueError(f"{name!r} is already registered")
        target[name] = FeatureSpec(
            fn=fn, warmup_trading_days=warmup_trading_days, project=project, description=description
        )
        return fn

    return decorator


def compute(
    registry: dict,
    name: str,
    ticker: str,
    start: str,
    end: str,
    source: DataSource,
    calendar_proxy: str = "SPY",
    **kwargs,
) -> pd.Series:
    """Look up `name` in `registry`, pad `start` backward by its declared
    warmup, call its function, and slice the result back to `[start, end]`.

    Args:
        registry: A registry dict as populated by `register()`.
        name: Registered feature/label name.
        ticker: Underlying ticker symbol.
        start: Requested start date, "YYYY-MM-DD".
        end: Requested end date, "YYYY-MM-DD".
        source: A `DataSource` passed through to the registered function.
        calendar_proxy: Market-proxy ticker used to derive the trading
            calendar for warmup padding (default "SPY").
        **kwargs: Passed through to the registered function.

    Returns:
        A `pd.Series` indexed by date, sliced to `[start, end]` inclusive.

    Raises:
        KeyError: If `name` is not registered.
    """
    if name not in registry:
        raise KeyError(f"{name!r} is not registered")
    spec = registry[name]
    padded_start = start
    if spec.warmup_trading_days > 0:
        calendar_start = (pd.Timestamp(start) - pd.Timedelta(days=spec.warmup_trading_days * 3 + 20))
        session_index = calendar.sessions(
            source, calendar_start.strftime("%Y-%m-%d"), end, proxy_ticker=calendar_proxy
        )
        shifted = calendar.shift_sessions(
            session_index, pd.Timestamp(start), -spec.warmup_trading_days
        )
        if shifted is not None:
            padded_start = shifted.strftime("%Y-%m-%d")
    result = spec.fn(ticker, padded_start, end, source, **kwargs)
    return result.loc[(result.index >= pd.Timestamp(start)) & (result.index <= pd.Timestamp(end))]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/features/registry.py tests/test_features_registry.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/features/registry.py tests/test_features_registry.py
git commit -m "feat: add feature registry with fwd_-name rejection and padded compute()"
```

---

### Task 3: `labels/registry.py` — registry sharing `compute()`, `fwd_` enforced

**Files:**
- Create: `data_universe/labels/__init__.py` (empty)
- Create: `data_universe/labels/registry.py`
- Test: `tests/test_labels_registry.py`

**Interfaces:**
- Consumes: `data_universe.features.registry.compute` (reused as-is), `data_universe.features.
  registry.FeatureSpec` (reused as the metadata container — labels use the same shape).
- Produces (used by Task 5, 6, 7):
  - `LABEL_REGISTRY: dict[str, FeatureSpec]`
  - `register(name, warmup_trading_days=0, project="", description="", registry=None) -> Callable`
    — same shape as `features.registry.register`, but requires `name.startswith("fwd_")`
    (opposite check) and defaults `registry` to `LABEL_REGISTRY`.
  - `compute(name, ticker, start, end, source, **kwargs) -> pd.Series` — thin wrapper calling
    `features.registry.compute(LABEL_REGISTRY, name, ...)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_registry.py
import pandas as pd
import pytest

from data_universe.labels import registry
from data_universe.sources.fakes import FakePolygonSource


def test_register_requires_fwd_prefix():
    local_registry = {}
    with pytest.raises(ValueError):
        @registry.register("not_forward", registry=local_registry)
        def _fn(ticker, start, end, source):
            return pd.Series(dtype="float64")


def test_register_accepts_fwd_prefixed_name():
    local_registry = {}

    @registry.register("fwd_ok", registry=local_registry)
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    assert "fwd_ok" in local_registry


def test_compute_delegates_to_shared_compute():
    local_registry = {}

    @registry.register("fwd_recorder", registry=local_registry)
    def _fn(ticker, start, end, source):
        bars = source.daily_bars(ticker, start, end)
        return pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())

    source = FakePolygonSource()
    result = registry.compute(
        local_registry, "fwd_recorder", "AAPL", "2024-02-01", "2024-02-10", source
    )
    assert result.index.min() >= pd.Timestamp("2024-02-01")
    assert result.index.max() <= pd.Timestamp("2024-02-10")


def test_label_registry_module_default_populated_by_decorator():
    @registry.register("fwd_default_target")
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    assert "fwd_default_target" in registry.LABEL_REGISTRY
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_labels_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_universe.labels'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/labels/__init__.py
```

```python
# data_universe/labels/registry.py
"""The label registry. Reuses `features.registry.compute()` verbatim (same
pad-call-slice logic) but enforces the opposite naming rule: every label
name MUST start with `fwd_`, signaling at a glance (and at import time) that
it looks forward and must never be used as a model input feature.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from data_universe.features.registry import FeatureSpec
from data_universe.features.registry import compute as _shared_compute
from data_universe.sources.base import DataSource

REQUIRED_PREFIX = "fwd_"

LABEL_REGISTRY: dict = {}


def register(
    name: str,
    warmup_trading_days: int = 0,
    project: str = "",
    description: str = "",
    registry: Optional[dict] = None,
    required_prefix: str = REQUIRED_PREFIX,
) -> Callable:
    """Decorator factory registering a label function under `name`.

    Args:
        name: Label name. Must start with `required_prefix` ("fwd_").
        warmup_trading_days: See `features.registry.register`.
        project: Free-text owning project tag.
        description: Human-readable description.
        registry: The dict to register into. Defaults to `LABEL_REGISTRY`.
        required_prefix: Prefix every label name must start with.

    Raises:
        ValueError: If `name` doesn't start with `required_prefix`, or if
            `name` is already registered.
    """
    target = LABEL_REGISTRY if registry is None else registry

    def decorator(fn: Callable) -> Callable:
        if not name.startswith(required_prefix):
            raise ValueError(
                f"Label name {name!r} must start with {required_prefix!r} "
                "-- non-fwd_ names belong in features/registry.py."
            )
        if name in target:
            raise ValueError(f"{name!r} is already registered")
        target[name] = FeatureSpec(
            fn=fn, warmup_trading_days=warmup_trading_days, project=project, description=description
        )
        return fn

    return decorator


def compute(
    registry: dict,
    name: str,
    ticker: str,
    start: str,
    end: str,
    source: DataSource,
    calendar_proxy: str = "SPY",
    **kwargs,
) -> pd.Series:
    """Thin wrapper around `features.registry.compute` -- identical
    pad-call-slice behavior, just documented under the label vocabulary.
    """
    return _shared_compute(registry, name, ticker, start, end, source, calendar_proxy, **kwargs)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_labels_registry.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/labels/__init__.py data_universe/labels/registry.py tests/test_labels_registry.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/labels/__init__.py data_universe/labels/registry.py tests/test_labels_registry.py
git commit -m "feat: add label registry enforcing fwd_ prefix, sharing compute() with features"
```

---

### Task 4: `features/library.py` — concrete features

**Files:**
- Create: `data_universe/features/library.py`
- Test: `tests/test_features_library.py`

**Interfaces:**
- Consumes: `data_universe.features.core` (Task 1), `data_universe.features.registry.register`
  (Task 2), `data_universe.options.iv30.iv30` (existing stub), `data_universe.sources.fakes.
  FakePolygonSource`/`FakeMarket` (existing, for tests).
- Produces (used by Task 5, 7): registers into `FEATURE_REGISTRY`: `ret_1d`, `ret_excess_1d`,
  `ret_21d`, `rv_cc_21d`, `rv_parkinson_21d`, `rv_gk_21d`, `dollar_volume_1d`, `adv_usd_20d`,
  `beta_ols_60d`, `beta_ols_90d`, `alpha_ols_60d`, `alpha_ols_90d`, `iv_index`,
  `vrp_signal_rv21`. Every function has signature
  `(ticker: str, start: str, end: str, source: DataSource, **kwargs) -> pd.Series`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_features_library.py
import pandas as pd
import pytest

from data_universe.features import registry
from data_universe.features.library import FEATURE_NAMES
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


@pytest.fixture
def source():
    return FakePolygonSource(FakeMarket())


@pytest.mark.parametrize("name", FEATURE_NAMES)
def test_every_registered_feature_computes_without_error(name, source):
    if name == "iv_index":
        ticker = "SPY"
    else:
        ticker = "AAPL"
    result = registry.compute(
        registry.FEATURE_REGISTRY, name, ticker, "2024-03-01", "2024-04-01", source
    )
    assert isinstance(result, pd.Series)
    assert not result.empty


def test_ret_1d_matches_manual_pct_change(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-03-01", "2024-04-01", source
    )
    bars = source.daily_bars("AAPL", "2024-01-01", "2024-04-01")
    closes = pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())
    expected = closes.pct_change().loc[result.index]
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_dollar_volume_1d_equals_close_times_volume(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "dollar_volume_1d", "AAPL", "2024-03-01", "2024-03-10", source
    )
    bars = source.daily_bars("AAPL", "2024-03-01", "2024-03-10")
    expected = (bars["close"] * bars["volume"]).values
    assert (result.values == pytest.approx(expected)).all()


def test_beta_ols_60d_recovers_planted_beta(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "beta_ols_60d", "AAPL", "2024-06-01", "2024-08-01", source
    )
    from data_universe.sources.fakes import PLANTED_BETA
    assert result.dropna().iloc[-1] == pytest.approx(PLANTED_BETA, abs=0.3)


def test_iv_index_for_spy_uses_vix(source):
    result = registry.compute(
        registry.FEATURE_REGISTRY, "iv_index", "SPY", "2024-03-01", "2024-03-10", source
    )
    from data_universe.sources.fakes import PLANTED_VIX
    assert (result == pytest.approx(PLANTED_VIX / 100.0)).all()


def test_iv_index_for_non_proxy_ticker_raises_not_implemented(source):
    with pytest.raises(NotImplementedError):
        registry.compute(
            registry.FEATURE_REGISTRY, "iv_index", "AAPL", "2024-03-01", "2024-03-10", source
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_features_library.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_universe.features.library'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/features/library.py
"""Concrete registered features for the beta, VRP, and PEAD projects. See
`PLAN.md` §7 for the project-level rationale behind each one.
"""
from __future__ import annotations

import pandas as pd

from data_universe.features import core
from data_universe.features.registry import register
from data_universe.options.iv30 import iv30

_PROXY_TICKER = "SPY"


def _daily_frame(ticker: str, start: str, end: str, source) -> pd.DataFrame:
    """Daily bars for `ticker`, re-indexed to a tz-naive date index."""
    bars = source.daily_bars(ticker, start, end)
    frame = bars.copy()
    frame.index = bars.index.tz_localize(None).normalize()
    frame.index.name = "date"
    return frame


@register("ret_1d", warmup_trading_days=1, project="beta", description="1-day simple return")
def ret_1d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_frame(ticker, start, end, source)["close"]
    return closes.pct_change().rename("ret_1d")


@register(
    "ret_excess_1d",
    warmup_trading_days=1,
    project="beta",
    description="1-day return minus the daily risk-free rate (0 if no rate_source given)",
)
def ret_excess_1d(ticker: str, start: str, end: str, source, rate_source=None, **kwargs) -> pd.Series:
    raw = ret_1d(ticker, start, end, source)
    if rate_source is None:
        return raw.rename("ret_excess_1d")
    rate = rate_source.daily_rate(start=start, end=end).reindex(raw.index).fillna(0.0)
    return (raw - rate).rename("ret_excess_1d")


@register("ret_21d", warmup_trading_days=21, project="beta", description="21-trading-day simple return")
def ret_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_frame(ticker, start, end, source)["close"]
    return closes.pct_change(21).rename("ret_21d")


@register(
    "rv_cc_21d",
    warmup_trading_days=21,
    project="vrp",
    description="Annualized 21d close-to-close realized volatility",
)
def rv_cc_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_frame(ticker, start, end, source)["close"]
    return core.realized_vol_cc(closes, window=21).rename("rv_cc_21d")


@register(
    "rv_parkinson_21d",
    warmup_trading_days=21,
    project="vrp",
    description="Annualized 21d Parkinson realized volatility",
)
def rv_parkinson_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    frame = _daily_frame(ticker, start, end, source)
    return core.realized_vol_parkinson(frame["high"], frame["low"], window=21).rename("rv_parkinson_21d")


@register(
    "rv_gk_21d",
    warmup_trading_days=21,
    project="vrp",
    description="Annualized 21d Garman-Klass realized volatility",
)
def rv_gk_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    frame = _daily_frame(ticker, start, end, source)
    return core.realized_vol_garman_klass(
        frame["open"], frame["high"], frame["low"], frame["close"], window=21
    ).rename("rv_gk_21d")


@register("dollar_volume_1d", warmup_trading_days=0, project="pead", description="close * volume")
def dollar_volume_1d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    frame = _daily_frame(ticker, start, end, source)
    return (frame["close"] * frame["volume"]).rename("dollar_volume_1d")


@register(
    "adv_usd_20d",
    warmup_trading_days=20,
    project="pead",
    description="20-trading-day rolling average dollar volume",
)
def adv_usd_20d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    dollar_volume = dollar_volume_1d(ticker, start, end, source)
    return dollar_volume.rolling(20).mean().rename("adv_usd_20d")


def _beta_alpha(ticker: str, start: str, end: str, source, window: int):
    asset_closes = _daily_frame(ticker, start, end, source)["close"]
    market_closes = _daily_frame(_PROXY_TICKER, start, end, source)["close"]
    asset_returns = core.log_returns(asset_closes)
    market_returns = core.log_returns(market_closes).reindex(asset_returns.index)
    return core.rolling_ols_beta_alpha(asset_returns, market_returns, window)


@register("beta_ols_60d", warmup_trading_days=61, project="beta", description="60d rolling OLS beta vs. SPY")
def beta_ols_60d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    beta, _ = _beta_alpha(ticker, start, end, source, window=60)
    return beta.rename("beta_ols_60d")


@register("beta_ols_90d", warmup_trading_days=91, project="beta", description="90d rolling OLS beta vs. SPY")
def beta_ols_90d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    beta, _ = _beta_alpha(ticker, start, end, source, window=90)
    return beta.rename("beta_ols_90d")


@register("alpha_ols_60d", warmup_trading_days=61, project="beta", description="60d rolling OLS alpha vs. SPY")
def alpha_ols_60d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    _, alpha = _beta_alpha(ticker, start, end, source, window=60)
    return alpha.rename("alpha_ols_60d")


@register("alpha_ols_90d", warmup_trading_days=91, project="beta", description="90d rolling OLS alpha vs. SPY")
def alpha_ols_90d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    _, alpha = _beta_alpha(ticker, start, end, source, window=90)
    return alpha.rename("alpha_ols_90d")


@register(
    "iv_index",
    warmup_trading_days=0,
    project="vrp",
    description="Implied-vol index level: VIX/100 for SPY, else options.iv30 (NotImplementedError today)",
)
def iv_index(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    if ticker != _PROXY_TICKER:
        # Single-name IV30 is not implemented yet -- see options/iv30.py.
        iv30(ticker, end)
    frame = source.index_daily_bars("I:VIX", start, end)
    closes = frame["close"] / 100.0
    closes.index = closes.index.tz_localize(None).normalize()
    return closes.rename("iv_index")


@register(
    "vrp_signal_rv21",
    warmup_trading_days=21,
    project="vrp",
    description="iv_index minus trailing rv_cc_21d (ex-ante VRP signal)",
)
def vrp_signal_rv21(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    iv = iv_index(ticker, start, end, source)
    rv = rv_cc_21d(ticker, start, end, source)
    return (iv - rv).rename("vrp_signal_rv21")


FEATURE_NAMES = [
    "ret_1d",
    "ret_excess_1d",
    "ret_21d",
    "rv_cc_21d",
    "rv_parkinson_21d",
    "rv_gk_21d",
    "dollar_volume_1d",
    "adv_usd_20d",
    "beta_ols_60d",
    "beta_ols_90d",
    "alpha_ols_60d",
    "alpha_ols_90d",
    "iv_index",
    "vrp_signal_rv21",
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_features_library.py -v`
Expected: PASS (all cases, including the `FEATURE_NAMES` parametrization)

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/features/library.py tests/test_features_library.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/features/library.py tests/test_features_library.py
git commit -m "feat: register beta/VRP/PEAD concrete features"
```

---

### Task 5: `labels/library.py` — concrete labels

**Files:**
- Create: `data_universe/labels/library.py`
- Test: `tests/test_labels_library.py`

**Interfaces:**
- Consumes: `data_universe.features.core` (Task 1), `data_universe.features.library.iv_index`
  / `rv_cc_21d` (Task 4, called directly as plain functions, not through the registry, to avoid
  re-deriving the same math), `data_universe.labels.registry.register` (Task 3).
- Produces (used by Task 7): registers into `LABEL_REGISTRY`: `fwd_rv_cc_21d`, `fwd_vrp_expost`,
  `fwd_ret_15d`. Same function signature as features:
  `(ticker: str, start: str, end: str, source: DataSource, **kwargs) -> pd.Series`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_labels_library.py
import pandas as pd
import pytest

from data_universe.labels import registry
from data_universe.labels.library import LABEL_NAMES
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


@pytest.fixture
def source():
    return FakePolygonSource(FakeMarket())


@pytest.mark.parametrize("name", LABEL_NAMES)
def test_every_registered_label_computes_without_error(name, source):
    ticker = "SPY" if name == "fwd_vrp_expost" else "AAPL"
    result = registry.compute(
        registry.LABEL_REGISTRY, name, ticker, "2024-03-01", "2024-04-01", source
    )
    assert isinstance(result, pd.Series)
    assert not result.empty


def test_fwd_ret_15d_shifts_returns_backward(source):
    result = registry.compute(
        registry.LABEL_REGISTRY, "fwd_ret_15d", "AAPL", "2024-03-01", "2024-03-20", source
    )
    bars = source.daily_bars("AAPL", "2024-01-01", "2024-05-01")
    closes = pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())
    forward = (closes.shift(-15) / closes - 1.0)
    aligned = forward.loc[result.dropna().index]
    pd.testing.assert_series_equal(result.dropna(), aligned, check_names=False)


def test_fwd_ret_15d_is_nan_near_the_end_of_the_requested_range(source):
    result = registry.compute(
        registry.LABEL_REGISTRY, "fwd_ret_15d", "AAPL", "2024-03-01", "2024-03-20", source
    )
    assert result.iloc[-1:].isna().all()  # last row has no future data beyond `end`
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_labels_library.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_universe.labels.library'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/labels/library.py
"""Concrete registered labels. Every function name here starts with `fwd_`
(enforced by `labels.registry.register`) as a loud reminder that these look
forward and must never be fed into a model as an input feature.

Important limitation shared by every label here: `compute()` only fetches
data through the requested `end` (never beyond it -- see `PLAN.md` §6), so
rows near the end of any requested range are genuinely NaN for lack of
future data, not a bug.
"""
from __future__ import annotations

import pandas as pd

from data_universe.features.library import iv_index, rv_cc_21d
from data_universe.labels.registry import register


def _daily_closes(ticker: str, start: str, end: str, source) -> pd.Series:
    bars = source.daily_bars(ticker, start, end)
    closes = bars["close"].copy()
    closes.index = bars.index.tz_localize(None).normalize()
    return closes


@register(
    "fwd_rv_cc_21d",
    warmup_trading_days=0,
    project="vrp",
    description="Realized close-to-close vol over the 21 trading days AFTER each date",
)
def fwd_rv_cc_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    trailing = rv_cc_21d(ticker, start, end, source)
    return trailing.shift(-21).rename("fwd_rv_cc_21d")


@register(
    "fwd_vrp_expost",
    warmup_trading_days=0,
    project="vrp",
    description="iv_index minus fwd_rv_cc_21d (realized, ex-post VRP)",
)
def fwd_vrp_expost(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    iv = iv_index(ticker, start, end, source)
    fwd_rv = fwd_rv_cc_21d(ticker, start, end, source)
    return (iv - fwd_rv).rename("fwd_vrp_expost")


@register(
    "fwd_ret_15d",
    warmup_trading_days=0,
    project="pead",
    description="Simple return from each date to 15 trading days later",
)
def fwd_ret_15d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_closes(ticker, start, end, source)
    return (closes.shift(-15) / closes - 1.0).rename("fwd_ret_15d")


LABEL_NAMES = ["fwd_rv_cc_21d", "fwd_vrp_expost", "fwd_ret_15d"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_labels_library.py -v`
Expected: PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/labels/library.py tests/test_labels_library.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/labels/library.py tests/test_labels_library.py
git commit -m "feat: register fwd_rv_cc_21d, fwd_vrp_expost, fwd_ret_15d labels"
```

---

### Task 6: `events/earnings.py` — PEAD event table

**Files:**
- Create: `data_universe/events/__init__.py` (empty)
- Create: `data_universe/events/earnings.py`
- Test: `tests/test_events_earnings.py`

**Interfaces:**
- Consumes: `data_universe.sources.edgar.EdgarSource` (shape: `.get_cik(ticker)`,
  `.earnings_8k_filings(cik)`), `data_universe.sources.base.DataSource` (shape:
  `.daily_bars(ticker, start, end)`), `data_universe.options.chain` (`as_of`, `atm_strike`,
  `nearest_expiry_after`, `straddle`), `data_universe.calendar` (`sessions`, `next_session`,
  `previous_session`).
- Produces (used by Phase 7's `precompute.py`, not built this phase):
  `build_pead_events(ticker: str, edgar_source, price_source: DataSource, options_source, cik: str | None = None, calendar_start: str = "2015-01-01", calendar_end: str | None = None) -> pd.DataFrame`
  — columns: `ticker`, `event_date`, `pre_session`, `day0`, `day1`, `sigma_pre_daily`,
  `adv_usd_pre20`, `adv_usd_day0_day1`, `days_to_expiry`, `next_earnings_session`,
  `call_volume`, `put_volume`, `is_stale`. One row per qualifying 8-K filing that has enough
  surrounding calendar/chain data to be priced (a filing near the very edge of the available
  calendar is skipped, not crashed on).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_events_earnings.py
import pandas as pd

from data_universe.events.earnings import build_pead_events
from data_universe.sources.fakes import FakePolygonSource


class FakeEdgarSource:
    def __init__(self, cik="320193", filing_dates=None):
        self._cik = cik
        self._filing_dates = filing_dates or ["2024-02-01T20:15:00.000Z", "2024-05-02T20:15:00.000Z"]

    def get_cik(self, ticker):
        return self._cik

    def earnings_8k_filings(self, cik):
        rows = [
            {
                "filing_date": pd.Timestamp(d[:10]),
                "acceptance_datetime": pd.Timestamp(d),
                "form": "8-K",
                "items": "2.02,9.01",
            }
            for d in self._filing_dates
        ]
        return pd.DataFrame(rows, columns=["filing_date", "acceptance_datetime", "form", "items"])


class FakeOptionsSource(FakePolygonSource):
    def contracts_as_of(self, underlying, as_of_date):
        expiries = ["2024-02-16", "2024-03-15", "2024-05-17", "2024-06-21"]
        rows = []
        for expiry in expiries:
            for strike in (95.0, 100.0, 105.0):
                rows.append(
                    {"ticker": f"O:{underlying}{expiry.replace('-', '')[2:]}C{int(strike*1000):08d}",
                     "expiration_date": expiry, "strike_price": strike, "contract_type": "call"}
                )
                rows.append(
                    {"ticker": f"O:{underlying}{expiry.replace('-', '')[2:]}P{int(strike*1000):08d}",
                     "expiration_date": expiry, "strike_price": strike, "contract_type": "put"}
                )
        return pd.DataFrame(rows)

    def daily_close(self, option_ticker, date):
        return {"close": 2.5, "volume": 10.0}


def test_build_pead_events_returns_one_row_per_filing_with_named_columns():
    price_source = FakePolygonSource()
    edgar_source = FakeEdgarSource()
    options_source = FakeOptionsSource()

    events = build_pead_events(
        "AAPL", edgar_source, price_source, options_source,
        calendar_start="2024-01-01", calendar_end="2024-06-01",
    )

    expected_columns = {
        "ticker", "event_date", "pre_session", "day0", "day1", "sigma_pre_daily",
        "adv_usd_pre20", "adv_usd_day0_day1", "days_to_expiry", "next_earnings_session",
        "call_volume", "put_volume", "is_stale",
    }
    assert expected_columns.issubset(set(events.columns))
    assert len(events) == 2
    assert (events["ticker"] == "AAPL").all()
    assert events["sigma_pre_daily"].notna().all()
    assert events["days_to_expiry"].gt(0).all()
    # next_earnings_session is the *following* filing's event date -- known only in hindsight
    assert events.iloc[0]["next_earnings_session"] == events.iloc[1]["event_date"]
    assert pd.isna(events.iloc[1]["next_earnings_session"])
    assert not events["is_stale"].iloc[0]  # fake options source always returns volume=10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_events_earnings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_universe.events'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/events/__init__.py
```

```python
# data_universe/events/earnings.py
"""PEAD event table: one row per qualifying 8-K Item 2.02 filing, combining
`EdgarSource` (announcement dates), a price `DataSource` (liquidity/vol
inputs), and the options toolkit (the ATM straddle chosen at day0).

Every "spec issue to flag" from `PLAN.md` §8 gets its own named column here,
never silently resolved:
  1. sigma_pre_daily      -- daily (not annualized) stdev of log returns,
                             30 trading days ending at pre_session.
  2. adv_usd_pre20 / adv_usd_day0_day1 -- only these two liquidity windows;
                             no post-entry-window average is computed.
  4. days_to_expiry        -- nearest expiry strictly after day0; a bias
                             the project team may want to adjust, not fixed
                             here.
  5. next_earnings_session -- the REALIZED next filing's event date, not a
                             forecast; NaT for the most recent filing.
  6. call_volume/put_volume/is_stale -- is_stale is true whenever either
                             leg of the chosen straddle traded zero volume
                             on day0 (a stale daily close).
(Issue 3, weekly-ranking lookahead, is out of scope -- no ranking column
exists in this table at all.)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_universe import calendar
from data_universe.features import core
from data_universe.options import chain

_EVENT_COLUMNS = [
    "ticker", "event_date", "pre_session", "day0", "day1", "sigma_pre_daily",
    "adv_usd_pre20", "adv_usd_day0_day1", "days_to_expiry", "next_earnings_session",
    "call_volume", "put_volume", "is_stale",
]


def build_pead_events(
    ticker: str,
    edgar_source,
    price_source,
    options_source,
    cik: Optional[str] = None,
    calendar_start: str = "2015-01-01",
    calendar_end: Optional[str] = None,
) -> pd.DataFrame:
    """Build the PEAD event table for one ticker.

    Args:
        ticker: Underlying ticker symbol.
        edgar_source: Object with `.get_cik(ticker)` and
            `.earnings_8k_filings(cik)` (see `sources/edgar.py`).
        price_source: A `DataSource`-shaped object providing `.daily_bars`.
        options_source: Object with `.contracts_as_of`/`.daily_close`
            (see `sources/polygon_options.py`; used via `options/chain.py`).
        cik: SEC CIK, or None to look it up via `edgar_source.get_cik`.
        calendar_start: Start of the trading calendar used to resolve
            sessions around each filing.
        calendar_end: End of the trading calendar. Defaults to today.

    Returns:
        A DataFrame with one row per qualifying filing (see module
        docstring for the column list), sorted ascending by `event_date`.
        A filing too close to either edge of the available calendar/price
        history to compute is skipped, not raised on.
    """
    cik = cik or edgar_source.get_cik(ticker)
    filings = edgar_source.earnings_8k_filings(cik)
    if filings.empty:
        return pd.DataFrame(columns=_EVENT_COLUMNS)

    end = calendar_end or pd.Timestamp.today().strftime("%Y-%m-%d")
    session_index = calendar.sessions(price_source, calendar_start, end)

    rows = []
    for position, filing in filings.iterrows():
        row = _build_one_event(ticker, filing, session_index, price_source, options_source)
        if row is not None:
            rows.append(row)

    table = pd.DataFrame(rows, columns=_EVENT_COLUMNS)
    table = table.sort_values("event_date").reset_index(drop=True)
    # next_earnings_session: the following row's event_date, known only in hindsight.
    table["next_earnings_session"] = table["event_date"].shift(-1)
    return table


def _build_one_event(ticker, filing, session_index, price_source, options_source):
    event_date = pd.Timestamp(filing["acceptance_datetime"]).tz_localize(None).normalize()

    day0 = event_date if event_date in session_index else calendar.next_session(session_index, event_date)
    if day0 is None:
        return None
    pre_session = calendar.previous_session(session_index, day0)
    day1 = calendar.next_session(session_index, day0)
    if pre_session is None or day1 is None:
        return None

    pre_window_start = calendar.shift_sessions(session_index, pre_session, -29)
    if pre_window_start is None:
        return None
    bars = price_source.daily_bars(
        ticker, pre_window_start.strftime("%Y-%m-%d"), day1.strftime("%Y-%m-%d")
    )
    closes = bars["close"].copy()
    closes.index = bars.index.tz_localize(None).normalize()

    log_returns = core.log_returns(closes)
    sigma_pre_daily = log_returns.loc[log_returns.index <= pre_session].tail(30).std(ddof=1)

    dollar_volume = (bars["close"] * bars["volume"])
    dollar_volume.index = closes.index
    adv_usd_pre20 = dollar_volume.loc[dollar_volume.index <= pre_session].tail(20).mean()
    adv_usd_day0_day1 = dollar_volume.loc[[day0, day1]].mean()

    try:
        option_chain = chain.as_of(options_source, ticker, day0.strftime("%Y-%m-%d"))
        expiry = chain.nearest_expiry_after(option_chain, day0.strftime("%Y-%m-%d"))
        spot = closes.loc[day0]
        strike = chain.atm_strike(option_chain, spot)
        straddle = chain.straddle(options_source, option_chain, strike, expiry, day0.strftime("%Y-%m-%d"))
    except (ValueError, KeyError):
        return None

    days_to_expiry = (expiry - day0).days
    call_volume = straddle["call_volume"]
    put_volume = straddle["put_volume"]
    is_stale = bool(call_volume == 0 or put_volume == 0)

    return {
        "ticker": ticker,
        "event_date": event_date,
        "pre_session": pre_session,
        "day0": day0,
        "day1": day1,
        "sigma_pre_daily": sigma_pre_daily,
        "adv_usd_pre20": adv_usd_pre20,
        "adv_usd_day0_day1": adv_usd_day0_day1,
        "days_to_expiry": days_to_expiry,
        "next_earnings_session": np.nan,  # filled in by build_pead_events after sorting
        "call_volume": call_volume,
        "put_volume": put_volume,
        "is_stale": is_stale,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_events_earnings.py -v`
Expected: PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/events/__init__.py data_universe/events/earnings.py tests/test_events_earnings.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/events/__init__.py data_universe/events/earnings.py tests/test_events_earnings.py
git commit -m "feat: add PEAD event table builder with named spec-issue columns"
```

---

### Task 7: Generic no-lookahead test + public API wiring

**Files:**
- Create: `tests/test_features_no_lookahead.py`
- Modify: `data_universe/__init__.py`
- Modify: `tests/test_public_api.py` (extend, don't replace)

**Interfaces:**
- Consumes: `data_universe.features.registry.FEATURE_REGISTRY`, `data_universe.labels.registry.
  LABEL_REGISTRY`, both `compute()` functions, `data_universe.features.library`,
  `data_universe.labels.library` (imported so registration side effects run).
- Produces: nothing new for later phases; this closes out Phase 6's registries by proving the
  no-lookahead property generically, and extends `data_universe/__init__.py`'s public API.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features_no_lookahead.py
"""Generic no-lookahead property, run over every registered feature AND
label: extending the requested `end` date further into the future must
never change a value already computed for an earlier date. It may only
fill in previously-NaN values (e.g. a forward label's last few rows, which
were NaN purely because no future data existed within the smaller window).
"""
import pandas as pd
import pytest

import data_universe.features.library  # noqa: F401  (populates FEATURE_REGISTRY)
import data_universe.labels.library  # noqa: F401  (populates LABEL_REGISTRY)
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.features.registry import compute as compute_feature
from data_universe.labels.registry import LABEL_REGISTRY
from data_universe.labels.registry import compute as compute_label
from data_universe.sources.fakes import FakeMarket, FakePolygonSource

_SHORT_END = "2024-04-01"
_LONG_END = "2024-06-01"
_START = "2024-03-01"


def _ticker_for(name: str) -> str:
    return "SPY" if name in ("iv_index", "fwd_vrp_expost") else "AAPL"


@pytest.mark.parametrize("name", sorted(FEATURE_REGISTRY))
def test_feature_values_are_unchanged_by_extending_end_date(name):
    source = FakePolygonSource(FakeMarket())
    ticker = _ticker_for(name)
    short = compute_feature(FEATURE_REGISTRY, name, ticker, _START, _SHORT_END, source)
    long = compute_feature(FEATURE_REGISTRY, name, ticker, _START, _LONG_END, source)
    common_index = short.index
    long_over_common = long.loc[common_index]
    pd.testing.assert_series_equal(short, long_over_common, check_names=False)


@pytest.mark.parametrize("name", sorted(LABEL_REGISTRY))
def test_label_values_are_unchanged_by_extending_end_date(name):
    source = FakePolygonSource(FakeMarket())
    ticker = _ticker_for(name)
    short = compute_label(LABEL_REGISTRY, name, ticker, _START, _SHORT_END, source)
    long = compute_label(LABEL_REGISTRY, name, ticker, _START, _LONG_END, source)
    common_index = short.index
    long_over_common = long.loc[common_index]
    # Extending `end` may only turn previously-NaN rows into real values
    # (more future data became available) -- it may never change a value
    # that was already non-NaN.
    already_known = short.dropna().index
    pd.testing.assert_series_equal(
        short.loc[already_known], long_over_common.loc[already_known], check_names=False
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_features_no_lookahead.py -v`
Expected: FAIL — either `ModuleNotFoundError` if any Task 4/5 module is missing, or an assertion
failure if the pre-existing implementation has an actual lookahead bug. If it's an assertion
failure, fix the offending feature/label function in `features/library.py` or `labels/library.py`
before proceeding (do not weaken this test to make it pass).

- [ ] **Step 3: Wire up the public API**

```python
# data_universe/__init__.py
"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (Ticker,
precompute) on top of Phase 6's feature/label registries.
"""
from data_universe import config
from data_universe.cache import get_cache, simulate
from data_universe.sources import sec_adapter as universe
import data_universe.features.library  # noqa: F401  (populates FEATURE_REGISTRY)
import data_universe.labels.library  # noqa: F401  (populates LABEL_REGISTRY)
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.labels.registry import LABEL_REGISTRY

__version__ = "0.1.0"

__all__ = [
    "config",
    "get_cache",
    "simulate",
    "universe",
    "FEATURE_REGISTRY",
    "LABEL_REGISTRY",
    "__version__",
]
```

- [ ] **Step 4: Extend the public API test**

Read `tests/test_public_api.py` first to match its existing style, then add (don't replace):

```python
def test_feature_and_label_registries_are_exported():
    import data_universe as q

    assert "ret_1d" in q.FEATURE_REGISTRY
    assert "fwd_ret_15d" in q.LABEL_REGISTRY
```

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all tests pass (prior phases' ~158 plus this phase's new tests)

- [ ] **Step 6: Run ruff on the whole tree**

Run: `ruff check .`
Expected: `All checks passed!`

- [ ] **Step 7: Commit**

```bash
git add tests/test_features_no_lookahead.py data_universe/__init__.py tests/test_public_api.py
git commit -m "test: add generic no-lookahead property over the feature/label registries"
```
