# Data Universe Phase 2 (Skeleton) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `data_universe` package skeleton — directory layout, packaging, config module, the `DataSource` interface and frame conventions, and a deterministic fake data source — so `pip install -e .` works, `pytest` runs green, and `ruff check .` is clean.

**Architecture:** Pure-Python package under `data_universe/`, installed with `setup.py` (explicit package list, no `find_packages()`). Configuration is a module-level mutable dict in `data_universe/config.py`, mirroring `sec_parser`'s `constants.py` global-setter pattern but with env-var-overrides-YAML precedence added. `data_universe/sources/base.py` defines the bar-frame conventions and an abstract `DataSource` interface; `data_universe/sources/fakes.py` implements a deterministic synthetic market (`FakeMarket`) with a planted beta against a market proxy, used by this phase's own tests and by every later phase's tests.

**Tech Stack:** Python ≥3.9, pandas, numpy, requests, pyarrow, scipy, pyyaml (core); pytest, ruff (dev).

**Spec:** `PLAN.md` (repo root) — sections 1-3, 4 (frame conventions), and 5 (cache design notes, not built this phase). This plan implements Phase 2 of `PLAN.md` §10 only: "Skeleton: layout, config.py, setup.py, CI, .gitignore, sources/fakes.py. pytest runs (even if nearly empty) and ruff passes." Cache (`cache.py`), the real Polygon/FRED/EDGAR sources, options toolkit, features/labels, and precompute are later phases and are **not** built here — do not add them.

## Global Constraints

- Package import name is `data_universe` (repo name `data-universe` is not a valid Python identifier).
- Never hardcode an API key or put one in a URL — this phase adds no real network calls, but `config.py` must be shaped so no later phase is tempted to.
- No `async def` anywhere (sec_parser uses `async`; the whole point of this library is to be synchronous).
- `setup.py` has a bare `VERSION` string constant; bump it (and `data_universe.__version__`) on every change that ships in a later phase — not required to change within this single phase's tasks, but the constant must exist now.
- Env vars override `config/config.yaml`, never the other way around.
- `.gitignore` must cover `data/`, `config/config.yaml`, `.venv/`, caches, and build artifacts — never commit keys or data.
- Docstrings on every public function/class/method, stating types/units where relevant (this phase has no annualized/percent quantities yet, but state format for dates etc.).
- Bars frame convention (from `PLAN.md` §4): tz-aware `DatetimeIndex` named `timestamp` in `America/New_York`; columns `open, high, low, close, volume, vwap` as `float64`. Intraday bars filtered to regular hours (9:30–16:00 ET) by default.

---

### Task 1: Repo scaffolding, packaging, and tooling config

**Files:**
- Create: `.gitignore`
- Create: `setup.py`
- Create: `requirements.txt`
- Create: `ruff.toml`
- Create: `data_universe/__init__.py`
- Create: `data_universe/sources/__init__.py`
- Create: `data_universe/options/__init__.py`
- Create: `data_universe/features/__init__.py`
- Create: `data_universe/labels/__init__.py`
- Create: `data_universe/events/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: importable package `data_universe` with `data_universe.__version__ == "0.1.0"`; installed in editable mode so later tasks' tests can `import data_universe...` without path hacks.

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# data & secrets
data/
config/config.yaml

# python
__pycache__/
*.pyc
.venv/
*.egg-info/
build/
dist/
.pytest_cache/
.ruff_cache/

# claude code worktrees
.claude/
```

- [ ] **Step 2: Create `data_universe/__init__.py`**

```python
"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (cache, Ticker,
precompute, feature/label registries). This phase only exposes `config` and
`__version__`.
"""
from data_universe import config

__version__ = "0.1.0"

__all__ = ["config", "__version__"]
```

- [ ] **Step 3: Create empty sub-package `__init__.py` files**

`data_universe/sources/__init__.py`:
```python
"""Data sources: Polygon, FRED, EDGAR, sec_parser adapter, and deterministic fakes."""
```

`data_universe/options/__init__.py`:
```python
"""Options toolkit: OCC ticker parsing, Black-Scholes/implied vol, chain helpers, IV30."""
```

`data_universe/features/__init__.py`:
```python
"""Feature registry and library. Features may only use information available by the
close of their own date -- see `data_universe.features.registry`."""
```

`data_universe/labels/__init__.py`:
```python
"""Label registry and library. Labels look forward in time and exist only for
evaluation/research -- see `data_universe.labels.registry`."""
```

`data_universe/events/__init__.py`:
```python
"""Event tables (e.g. the PEAD earnings event table) -- one row per ticker per event,
not a daily feature."""
```

`tests/__init__.py`:
```python
```
(empty file — makes `tests` a package so `pytest` can be run as `pytest tests/`)

- [ ] **Step 4: Create `setup.py`**

```python
from setuptools import setup

VERSION = "0.1.0"
DESCRIPTION = "GTSF Quant Sector shared data and feature library."

setup(
    name="data_universe",
    version=VERSION,
    description=DESCRIPTION,
    packages=[
        "data_universe",
        "data_universe.sources",
        "data_universe.options",
        "data_universe.features",
        "data_universe.labels",
        "data_universe.events",
    ],
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "pyarrow",
        "scipy",
        "pyyaml",
    ],
    extras_require={
        "sec": [
            "sec_parser @ git+https://github.com/GTSF-Quantitative-Sector/sec_parser.git",
        ],
        "s3": ["boto3"],
        "dev": ["pytest", "ruff"],
    },
    python_requires=">=3.9",
)
```

- [ ] **Step 5: Create `requirements.txt`**

```
pandas
numpy
requests
pyarrow
scipy
pyyaml
```

- [ ] **Step 6: Create `ruff.toml`**

```toml
line-length = 100
target-version = "py39"

[lint]
select = ["E", "F", "I"]
```

- [ ] **Step 7: Install the package in editable mode with dev extras**

Run: `pip install -e ".[dev]"`
Expected: installs cleanly, no errors. (`sec_parser` from the `sec` extra is NOT installed here — it's optional and unrelated to this phase.)

- [ ] **Step 8: Verify the package imports and pytest/ruff run clean on an empty suite**

Run: `python -c "import data_universe; print(data_universe.__version__)"`
Expected: prints `0.1.0`

Run: `pytest --collect-only`
Expected: `no tests ran` or `0 tests collected` (no errors)

Run: `ruff check .`
Expected: `All checks passed!`

- [ ] **Step 9: Commit**

```bash
git add .gitignore setup.py requirements.txt ruff.toml data_universe tests
git commit -m "chore: scaffold data_universe package layout and tooling"
```

---

### Task 2: `config.py` — global settings with env-var-over-YAML precedence

**Files:**
- Create: `data_universe/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (leaf module).
- Produces: `set_polygon_key`, `get_polygon_key`, `set_polygon_base_url`, `get_polygon_base_url`, `set_polygon_flatfiles_endpoint`, `get_polygon_flatfiles_endpoint`, `set_polygon_s3_access_key`, `get_polygon_s3_access_key`, `set_polygon_s3_secret_key`, `get_polygon_s3_secret_key`, `set_fred_key`, `get_fred_key`, `set_project`, `get_project`, `set_data_dir`, `get_data_dir`, `set_cache_size`, `get_cache_size`, `load_yaml(path: str = "config/config.yaml") -> None`, `reset() -> None`. Every later phase's `@cached` decorator reads `get_project()`; every source reads `get_polygon_key()` / `get_polygon_base_url()` / etc.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import pytest

from data_universe import config


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def test_polygon_key_defaults_to_none_and_is_settable():
    assert config.get_polygon_key() is None
    config.set_polygon_key("secret")
    assert config.get_polygon_key() == "secret"


def test_polygon_base_url_default():
    assert config.get_polygon_base_url() == "https://api.polygon.io"


def test_polygon_base_url_settable():
    config.set_polygon_base_url("https://example.test")
    assert config.get_polygon_base_url() == "https://example.test"


def test_project_defaults_to_none_and_is_settable():
    assert config.get_project() is None
    config.set_project("vrp-research")
    assert config.get_project() == "vrp-research"


def test_cache_size_default_is_256():
    assert config.get_cache_size() == 256


def test_cache_size_settable():
    config.set_cache_size(64)
    assert config.get_cache_size() == 64


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("POLYGON_BASE_URL", "https://from-env.test")
    config.reset()
    assert config.get_polygon_base_url() == "https://from-env.test"


def test_load_yaml_sets_values_not_set_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    config.reset()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("fred_api_key: from-yaml\n")
    config.load_yaml(str(yaml_path))
    assert config.get_fred_key() == "from-yaml"


def test_load_yaml_does_not_override_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "from-env")
    config.reset()
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("fred_api_key: from-yaml\n")
    config.load_yaml(str(yaml_path))
    assert config.get_fred_key() == "from-env"


def test_load_yaml_missing_file_is_a_noop(tmp_path):
    config.reset()
    config.load_yaml(str(tmp_path / "does-not-exist.yaml"))
    assert config.get_fred_key() is None


def test_data_dir_default_is_under_package_parent():
    assert config.get_data_dir().endswith("data")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError` or `AttributeError: module 'data_universe.config' has no attribute ...` (module doesn't exist yet).

- [ ] **Step 3: Write `data_universe/config.py`**

```python
"""Global configuration for the data_universe package.

Precedence: environment variable > `config/config.yaml` > hardcoded default.
The Polygon API key is only ever held in memory and sent as an
`Authorization: Bearer` header by source modules -- never in a URL, never
logged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
_DEFAULT_CACHE_SIZE = 256
_DEFAULT_POLYGON_BASE_URL = "https://api.polygon.io"

# Maps internal state keys to the environment variable that seeds them.
_ENV_VARS = {
    "polygon_key": "POLYGON_API_KEY",
    "polygon_base_url": "POLYGON_BASE_URL",
    "polygon_flatfiles_endpoint": "POLYGON_FLATFILES_ENDPOINT",
    "polygon_s3_access_key": "POLYGON_S3_ACCESS_KEY",
    "polygon_s3_secret_key": "POLYGON_S3_SECRET_KEY",
    "fred_key": "FRED_API_KEY",
    "data_dir": "DATA_UNIVERSE_DATA_DIR",
    "cache_size": "DATA_UNIVERSE_CACHE_SIZE",
}

# Maps a config/config.yaml top-level key to the internal state key it feeds.
_YAML_KEYS = {
    "polygon_api_key": "polygon_key",
    "polygon_base_url": "polygon_base_url",
    "polygon_flatfiles_endpoint": "polygon_flatfiles_endpoint",
    "polygon_s3_access_key": "polygon_s3_access_key",
    "polygon_s3_secret_key": "polygon_s3_secret_key",
    "fred_api_key": "fred_key",
    "data_dir": "data_dir",
    "cache_size": "cache_size",
}

_state: dict = {}


def _default(state_key: str, default):
    """Read a state key's initial value from its environment variable, else `default`."""
    env_var = _ENV_VARS.get(state_key)
    if env_var is None:
        return default
    value = os.environ.get(env_var)
    if value is None:
        return default
    if state_key == "cache_size":
        return int(value)
    return value


def reset() -> None:
    """Reset all configuration to defaults (re-reading environment variables).

    Intended for tests; not part of normal usage.
    """
    _state.clear()
    _state["polygon_key"] = _default("polygon_key", None)
    _state["polygon_base_url"] = _default("polygon_base_url", _DEFAULT_POLYGON_BASE_URL)
    _state["polygon_flatfiles_endpoint"] = _default("polygon_flatfiles_endpoint", None)
    _state["polygon_s3_access_key"] = _default("polygon_s3_access_key", None)
    _state["polygon_s3_secret_key"] = _default("polygon_s3_secret_key", None)
    _state["fred_key"] = _default("fred_key", None)
    _state["data_dir"] = _default("data_dir", _DEFAULT_DATA_DIR)
    _state["cache_size"] = _default("cache_size", _DEFAULT_CACHE_SIZE)
    _state["project"] = None


def set_polygon_key(key: str) -> None:
    """Set the Polygon.io API key. Sent as an `Authorization: Bearer` header, never a URL param.

    Args:
        key: Polygon.io API key.
    """
    _state["polygon_key"] = key


def get_polygon_key() -> Optional[str]:
    """Return the configured Polygon.io API key, or None if unset."""
    return _state["polygon_key"]


def set_polygon_base_url(url: str) -> None:
    """Set the Polygon REST base URL (default `https://api.polygon.io`).

    Args:
        url: Base URL, no trailing slash.
    """
    _state["polygon_base_url"] = url


def get_polygon_base_url() -> str:
    """Return the configured Polygon REST base URL."""
    return _state["polygon_base_url"]


def set_polygon_flatfiles_endpoint(url: str) -> None:
    """Set the S3-compatible flat-files endpoint for bulk options history.

    Args:
        url: S3-compatible endpoint URL.
    """
    _state["polygon_flatfiles_endpoint"] = url


def get_polygon_flatfiles_endpoint() -> Optional[str]:
    """Return the configured flat-files endpoint, or None if unset."""
    return _state["polygon_flatfiles_endpoint"]


def set_polygon_s3_access_key(key: str) -> None:
    """Set the S3 access key used for Polygon flat-files downloads."""
    _state["polygon_s3_access_key"] = key


def get_polygon_s3_access_key() -> Optional[str]:
    """Return the configured S3 access key, or None if unset."""
    return _state["polygon_s3_access_key"]


def set_polygon_s3_secret_key(key: str) -> None:
    """Set the S3 secret key used for Polygon flat-files downloads."""
    _state["polygon_s3_secret_key"] = key


def get_polygon_s3_secret_key() -> Optional[str]:
    """Return the configured S3 secret key, or None if unset."""
    return _state["polygon_s3_secret_key"]


def set_fred_key(key: str) -> None:
    """Set the FRED API key used to fetch the risk-free rate series."""
    _state["fred_key"] = key


def get_fred_key() -> Optional[str]:
    """Return the configured FRED API key, or None if unset."""
    return _state["fred_key"]


def set_project(name: str) -> None:
    """Tag subsequent cache accesses with a project name, e.g. "vrp-research", "pead", "beta".

    Args:
        name: Project identifier.
    """
    _state["project"] = name


def get_project() -> Optional[str]:
    """Return the current project tag, or None if unset."""
    return _state["project"]


def set_data_dir(path: str) -> None:
    """Set the root directory for raw/interim/processed data and cache logs."""
    _state["data_dir"] = path


def get_data_dir() -> str:
    """Return the configured data directory."""
    return _state["data_dir"]


def set_cache_size(n: int) -> None:
    """Set the shared LRU cache capacity (number of entries) used by `data_universe.cache`."""
    _state["cache_size"] = n


def get_cache_size() -> int:
    """Return the configured cache capacity."""
    return _state["cache_size"]


def load_yaml(path: str = "config/config.yaml") -> None:
    """Load settings from a YAML file. Environment variables still take precedence:
    a key present in both the YAML file and the environment keeps its environment value.

    Args:
        path: Path to a YAML file with a flat map of setting names (see
            `config/config.example.yaml`) to values. A missing file is a no-op.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    for yaml_key, state_key in _YAML_KEYS.items():
        if yaml_key not in data:
            continue
        env_var = _ENV_VARS.get(state_key)
        if env_var and os.environ.get(env_var) is not None:
            continue  # environment variable wins
        value = data[yaml_key]
        _state[state_key] = int(value) if state_key == "cache_size" else value


reset()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/config.py tests/test_config.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/config.py tests/test_config.py
git commit -m "feat: add config module with env-over-yaml precedence"
```

---

### Task 3: `sources/base.py` — frame conventions and the `DataSource` interface

**Files:**
- Create: `data_universe/sources/base.py`
- Test: `tests/test_sources_base.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `NY_TZ: str`, `BAR_COLUMNS: list[str]`, `REGULAR_OPEN: str`, `REGULAR_CLOSE: str`, `DataSourceError(Exception)`, `RateLimitedError(DataSourceError)`, `empty_bars() -> pd.DataFrame`, `filter_regular_hours(bars: pd.DataFrame) -> pd.DataFrame`, `class DataSource(abc.ABC)` with abstract methods `daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame` and `hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame`. Task 4's `FakePolygonSource` and every later phase's real sources implement this interface.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_base.py`:

```python
import pandas as pd
import pytest

from data_universe.sources.base import (
    BAR_COLUMNS,
    DataSource,
    DataSourceError,
    RateLimitedError,
    empty_bars,
    filter_regular_hours,
)


def test_empty_bars_has_expected_columns_and_index_name():
    bars = empty_bars()
    assert list(bars.columns) == BAR_COLUMNS
    assert bars.empty
    assert bars.index.name == "timestamp"


def test_empty_bars_columns_are_float64():
    bars = empty_bars()
    assert all(bars[c].dtype == "float64" for c in BAR_COLUMNS)


def test_filter_regular_hours_on_empty_frame_returns_empty():
    filtered = filter_regular_hours(empty_bars())
    assert filtered.empty


def test_filter_regular_hours_drops_bars_outside_930_to_1600():
    index = pd.DatetimeIndex(
        [
            "2023-01-03 09:00:00",
            "2023-01-03 09:30:00",
            "2023-01-03 12:00:00",
            "2023-01-03 16:00:00",
            "2023-01-03 16:30:00",
        ],
        tz="America/New_York",
        name="timestamp",
    )
    bars = pd.DataFrame(
        {c: [1.0, 2.0, 3.0, 4.0, 5.0] for c in BAR_COLUMNS},
        index=index,
    )
    filtered = filter_regular_hours(bars)
    assert len(filtered) == 3
    assert filtered.index.tolist() == index[1:4].tolist()


def test_filter_regular_hours_returns_a_copy():
    index = pd.DatetimeIndex(["2023-01-03 10:00:00"], tz="America/New_York", name="timestamp")
    bars = pd.DataFrame({c: [1.0] for c in BAR_COLUMNS}, index=index)
    filtered = filter_regular_hours(bars)
    filtered.iloc[0, 0] = 999.0
    assert bars.iloc[0, 0] == 1.0


def test_data_source_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        DataSource()


def test_data_source_subclass_must_implement_both_abstract_methods():
    class Incomplete(DataSource):
        def daily_bars(self, ticker, start, end):
            return empty_bars()

    with pytest.raises(TypeError):
        Incomplete()


def test_data_source_subclass_with_both_methods_instantiates():
    class Complete(DataSource):
        def daily_bars(self, ticker, start, end):
            return empty_bars()

        def hourly_bars(self, ticker, start, end):
            return empty_bars()

    Complete()  # should not raise -- DataSource takes no required __init__ args


def test_rate_limited_error_is_a_data_source_error():
    assert issubclass(RateLimitedError, DataSourceError)
```

Note: `test_data_source_subclass_with_both_methods_instantiates` calls `Complete()` to confirm `DataSource` doesn't force a particular constructor signature — `DataSource` must not define `__init__`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_base.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.base'`

- [ ] **Step 3: Write `data_universe/sources/base.py`**

```python
"""Shared frame conventions and the `DataSource` interface.

Bar frames: tz-aware `DatetimeIndex` named "timestamp" in America/New_York,
columns open/high/low/close/volume/vwap as float64. Daily bars are stamped
at midnight of the session date; intraday bars are filtered to regular
trading hours (9:30-16:00 ET) by default. Prices are split- and
dividend-adjusted unless a function name says otherwise.
"""
from __future__ import annotations

import abc

import pandas as pd

NY_TZ = "America/New_York"
BAR_COLUMNS = ["open", "high", "low", "close", "volume", "vwap"]
REGULAR_OPEN = "09:30"
REGULAR_CLOSE = "16:00"


class DataSourceError(Exception):
    """Base class for all data_universe source errors."""


class RateLimitedError(DataSourceError):
    """Raised when an upstream API returns 429 after exhausting retries."""


def empty_bars() -> pd.DataFrame:
    """Return an empty, correctly-typed bars frame (see module docstring for the schema)."""
    index = pd.DatetimeIndex([], name="timestamp", tz=NY_TZ)
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in BAR_COLUMNS}, index=index)


def filter_regular_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Filter an intraday bars frame down to regular trading hours (9:30-16:00 ET, inclusive).

    Args:
        bars: A bars frame with a tz-aware America/New_York DatetimeIndex.

    Returns:
        A copy containing only rows within regular trading hours.
    """
    if bars.empty:
        return bars.copy()
    localized = bars.index.tz_convert(NY_TZ)
    times = localized.time
    start = pd.Timestamp(REGULAR_OPEN).time()
    end = pd.Timestamp(REGULAR_CLOSE).time()
    mask = (times >= start) & (times <= end)
    return bars.loc[mask].copy()


class DataSource(abc.ABC):
    """Common interface implemented by every data source in this package."""

    @abc.abstractmethod
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return daily adjusted OHLCV bars for `ticker` between `start` and `end` (inclusive).

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.
        """

    @abc.abstractmethod
    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Return hourly adjusted OHLCV bars for `ticker` between `start` and `end` (inclusive),
        filtered to regular trading hours by default.

        Args:
            ticker: Underlying ticker symbol.
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Returns:
            A bars frame following the module-level frame convention.
        """
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_base.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/sources/base.py tests/test_sources_base.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/sources/base.py tests/test_sources_base.py
git commit -m "feat: add DataSource interface and bar frame conventions"
```

---

### Task 4: `sources/fakes.py` — deterministic fake market with a planted beta

**Files:**
- Create: `data_universe/sources/fakes.py`
- Test: `tests/test_sources_fakes.py`

**Interfaces:**
- Consumes: `data_universe.sources.base.BAR_COLUMNS`, `NY_TZ`, `filter_regular_hours` (Task 3).
- Produces: `DEFAULT_SEED: int`, `PLANTED_BETA: float`, `PLANTED_ALPHA: float`, `PLANTED_VIX: float`, `class FakeMarket(seed=DEFAULT_SEED, beta=PLANTED_BETA, alpha=PLANTED_ALPHA, proxy_ticker="SPY", vix_level=PLANTED_VIX)` with `.daily_bars(ticker, start, end) -> pd.DataFrame`, `.hourly_bars(ticker, start, end) -> pd.DataFrame`, `.vix_closes(start, end) -> pd.Series`; `class FakePolygonSource(market: FakeMarket | None = None)` implementing `daily_bars`/`hourly_bars`/`index_daily_bars(ticker, start, end)`. Later phases' beta-project and VRP-project feature tests (Phase 6) will import `FakeMarket`/`FakePolygonSource` directly — do not rename these.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sources_fakes.py`:

```python
import numpy as np
import pandas as pd
import pytest

from data_universe.sources.base import BAR_COLUMNS, filter_regular_hours
from data_universe.sources.fakes import FakeMarket, FakePolygonSource


def test_daily_bars_has_expected_columns_dtype_and_index():
    market = FakeMarket()
    bars = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    assert list(bars.columns) == BAR_COLUMNS
    assert all(bars[c].dtype == "float64" for c in BAR_COLUMNS)
    assert bars.index.name == "timestamp"
    assert str(bars.index.tz) == "America/New_York"
    assert len(bars) > 0


def test_daily_bars_deterministic_across_calls():
    market = FakeMarket()
    a = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    b = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    pd.testing.assert_frame_equal(a, b)


def test_daily_bars_differ_by_ticker():
    market = FakeMarket()
    aapl = market.daily_bars("AAPL", "2023-01-01", "2023-03-01")
    msft = market.daily_bars("MSFT", "2023-01-01", "2023-03-01")
    assert not aapl["close"].equals(msft["close"])


def test_planted_beta_is_recoverable_by_ols():
    market = FakeMarket(beta=1.5, alpha=0.0, seed=42)
    mkt_bars = market.daily_bars("SPY", "2010-01-01", "2023-01-01")
    stock_bars = market.daily_bars("AAPL", "2010-01-01", "2023-01-01")
    mkt_ret = np.log(mkt_bars["close"]).diff().dropna()
    stock_ret = np.log(stock_bars["close"]).diff().dropna()
    beta, alpha = np.polyfit(mkt_ret.to_numpy(), stock_ret.to_numpy(), 1)
    assert beta == pytest.approx(1.5, abs=0.05)
    assert alpha == pytest.approx(0.0, abs=0.0005)


def test_proxy_ticker_bars_are_not_beta_scaled():
    # SPY against itself must have beta 1, not the planted stock beta.
    market = FakeMarket(beta=1.5, seed=7)
    spy_a = market.daily_bars("SPY", "2015-01-01", "2020-01-01")
    spy_b = market.daily_bars("SPY", "2015-01-01", "2020-01-01")
    pd.testing.assert_frame_equal(spy_a, spy_b)


def test_vix_closes_are_constant_at_configured_level():
    market = FakeMarket(vix_level=16.0)
    closes = market.vix_closes("2023-01-01", "2023-01-10")
    assert (closes == 16.0).all()
    assert len(closes) > 0


def test_hourly_bars_filtering_removes_pre_open_bar():
    market = FakeMarket()
    bars = market.hourly_bars("AAPL", "2023-01-03", "2023-01-03")
    filtered = filter_regular_hours(bars)
    assert len(filtered) < len(bars)
    assert len(filtered) > 0


def test_fake_polygon_source_daily_bars_delegates_to_market():
    market = FakeMarket(seed=99)
    source = FakePolygonSource(market)
    direct = market.daily_bars("AAPL", "2023-01-01", "2023-02-01")
    via_source = source.daily_bars("AAPL", "2023-01-01", "2023-02-01")
    pd.testing.assert_frame_equal(direct, via_source)


def test_fake_polygon_source_index_daily_bars_wraps_vix():
    source = FakePolygonSource(FakeMarket(vix_level=20.0))
    bars = source.index_daily_bars("I:VIX", "2023-01-01", "2023-01-10")
    assert (bars["close"] == 20.0).all()
    assert list(bars.columns) == BAR_COLUMNS


def test_fake_polygon_source_rejects_unknown_index_ticker():
    source = FakePolygonSource()
    with pytest.raises(ValueError):
        source.index_daily_bars("I:SPX", "2023-01-01", "2023-01-10")


def test_fake_polygon_source_default_market_is_a_fake_market():
    source = FakePolygonSource()
    assert isinstance(source.market, FakeMarket)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sources_fakes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.sources.fakes'`

- [ ] **Step 3: Write `data_universe/sources/fakes.py`**

```python
"""Deterministic fake data sources for tests and for users without an API key.

`FakeMarket` generates a small, reproducible synthetic market: a market proxy
(default "SPY") daily/hourly random walk, and every other ticker's return is a
fixed linear function of the proxy's return -- `stock_return = alpha + beta *
proxy_return + noise` -- so tests can check statistics (e.g. rolling OLS beta)
against a known planted truth instead of against the code itself.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from data_universe.sources.base import BAR_COLUMNS, NY_TZ

DEFAULT_SEED = 20240101
PLANTED_BETA = 1.5
PLANTED_ALPHA = 0.0
PLANTED_VIX = 16.0


def _session_index(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    """Build a tz-aware America/New_York DatetimeIndex of business-day sessions.

    Args:
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        freq: "1d" for one stamp per session (midnight), "1h" for seven
            hourly stamps (09:00-15:00) per session.
    """
    if freq == "1d":
        return pd.bdate_range(start, end, tz=NY_TZ, name="timestamp")
    if freq == "1h":
        sessions = pd.bdate_range(start, end)
        hours = pd.timedelta_range("09:00:00", "15:00:00", freq="1h")
        stamps = [pd.Timestamp(day.date()) + h for day in sessions for h in hours]
        return pd.DatetimeIndex(stamps, name="timestamp").tz_localize(NY_TZ)
    raise ValueError(f"Unsupported freq: {freq!r}")


def _bars_from_log_returns(
    index: pd.DatetimeIndex, log_returns: np.ndarray, start_price: float
) -> pd.DataFrame:
    """Turn a series of log returns into a synthetic OHLCV bars frame."""
    closes = start_price * np.exp(np.cumsum(log_returns))
    opens = np.roll(closes, 1)
    opens[0] = start_price
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    volume = np.full(len(index), 1_000_000.0)
    vwap = (opens + closes) / 2.0
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "vwap": vwap,
        },
        index=index,
    )[BAR_COLUMNS]


class FakeMarket:
    """A reproducible synthetic market shared by tests across the whole package.

    Args:
        seed: RNG seed. The same seed always produces the same bars.
        beta: Planted beta of every non-proxy ticker against `proxy_ticker`.
        alpha: Planted daily log-return alpha of every non-proxy ticker.
        proxy_ticker: The market proxy ticker (default "SPY").
        vix_level: Constant VIX close used for every session.
    """

    def __init__(
        self,
        seed: int = DEFAULT_SEED,
        beta: float = PLANTED_BETA,
        alpha: float = PLANTED_ALPHA,
        proxy_ticker: str = "SPY",
        vix_level: float = PLANTED_VIX,
    ) -> None:
        self.seed = seed
        self.beta = beta
        self.alpha = alpha
        self.proxy_ticker = proxy_ticker
        self.vix_level = vix_level

    def _rng(self, ticker: str, freq: str) -> np.random.Generator:
        """Deterministic per-(seed, ticker, freq) generator, so repeated calls agree."""
        key = abs(hash((self.seed, ticker, freq))) % (2**32)
        return np.random.default_rng(key)

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Deterministic daily bars. `ticker == proxy_ticker` is the market itself;
        any other ticker is `alpha + beta * proxy_return + noise`.
        """
        index = _session_index(start, end, "1d")
        n = len(index)
        market_returns = self._rng(self.proxy_ticker, "1d").normal(0.0003, 0.01, size=n)
        if ticker == self.proxy_ticker:
            returns = market_returns
        else:
            noise = self._rng(ticker, "1d").normal(0.0, 0.005, size=n)
            returns = self.alpha + self.beta * market_returns + noise
        return _bars_from_log_returns(index, returns, start_price=100.0)

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Deterministic hourly bars, same planted alpha/beta relationship as `daily_bars`.
        Includes one pre-open (09:00) stamp per session so `filter_regular_hours` has
        something real to filter in tests.
        """
        index = _session_index(start, end, "1h")
        n = len(index)
        market_returns = self._rng(self.proxy_ticker, "1h").normal(0.00005, 0.003, size=n)
        if ticker == self.proxy_ticker:
            returns = market_returns
        else:
            noise = self._rng(ticker, "1h").normal(0.0, 0.0015, size=n)
            returns = self.alpha + self.beta * market_returns + noise
        return _bars_from_log_returns(index, returns, start_price=100.0)

    def vix_closes(self, start: str, end: str) -> pd.Series:
        """Constant VIX close series (`vix_level`) over the given session range."""
        index = _session_index(start, end, "1d")
        return pd.Series(self.vix_level, index=index, name="close")


class FakePolygonSource:
    """A `DataSource`-shaped wrapper around `FakeMarket`, for tests that want an
    object with source-style methods instead of calling the generator directly.

    Args:
        market: The `FakeMarket` to delegate to. Defaults to `FakeMarket()`.
    """

    def __init__(self, market: Optional[FakeMarket] = None) -> None:
        self.market = market or FakeMarket()

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.daily_bars`."""
        return self.market.daily_bars(ticker, start, end)

    def hourly_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """See `FakeMarket.hourly_bars`."""
        return self.market.hourly_bars(ticker, start, end)

    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fake index aggregates. Only "I:VIX" is supported.

        Args:
            ticker: Must be "I:VIX".
            start: Start date, "YYYY-MM-DD".
            end: End date, "YYYY-MM-DD".

        Raises:
            ValueError: If `ticker` is not "I:VIX".
        """
        if ticker != "I:VIX":
            raise ValueError(f"FakePolygonSource only fakes I:VIX, got {ticker!r}")
        closes = self.market.vix_closes(start, end)
        return pd.DataFrame(
            {
                "open": closes,
                "high": closes,
                "low": closes,
                "close": closes,
                "volume": pd.Series(0.0, index=closes.index),
                "vwap": closes,
            }
        )[BAR_COLUMNS]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sources_fakes.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/sources/fakes.py tests/test_sources_fakes.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/sources/fakes.py tests/test_sources_fakes.py
git commit -m "feat: add deterministic FakeMarket/FakePolygonSource with planted beta"
```

---

### Task 5: CI workflow and full-suite verification

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `setup.py`'s `dev` extra (Task 1), the full `tests/` suite (Tasks 2-4).
- Produces: a GitHub Actions job that runs `ruff check .` and `pytest` on push and pull request, on a machine with no API keys.

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install package
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"
      - name: Ruff
        run: ruff check .
      - name: Pytest
        run: pytest
```

- [ ] **Step 2: Run the full local suite exactly as CI will**

Run: `ruff check .`
Expected: `All checks passed!`

Run: `pytest -v`
Expected: every test from Tasks 2-4 PASSES, 0 failures, 0 errors.

- [ ] **Step 3: Confirm no network access is required**

Run: `python -c "import socket; socket.socket = None; import data_universe, data_universe.sources.fakes; print('ok - no import-time network use')"`
Expected: prints `ok - no import-time network use` (import succeeds even with `socket.socket` disabled, proving nothing in this phase touches the network at import time).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "chore: add CI workflow running ruff and pytest"
```

---

## Self-Review Notes (for the executor)

- Task 1 has no tests of its own (pure scaffolding) — its verification is the import + empty-collection + ruff checks in Step 8, which is why it's a step, not a skipped TDD cycle.
- `DataSource` must not define `__init__`, or `test_data_source_subclass_with_both_methods_instantiates`'s `Complete()` call will fail for the wrong reason (a signature mismatch) instead of proving the interface itself imposes no constructor.
- Cache (`data_universe/cache.py`), real `PolygonSource`/`FredSource`/`EdgarSource`/`sec_adapter`, `calendar.py`, `ticker.py`, options toolkit, features/labels, `precompute.py`, and the docs (README/CONTRACTS.md/MAINTENANCE.md/OPEN_QUESTIONS.md) are Phases 3-8 of `PLAN.md` and are explicitly **not** part of this plan — do not add them here even if it seems convenient.
