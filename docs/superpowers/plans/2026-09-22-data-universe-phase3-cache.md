# Data Universe Phase 3 (Cache) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the shared in-memory LRU cache — the main EOW deliverable per `PLAN.md` — with per-access logging, a `@cached` decorator for source methods, and a `simulate()` replay tool, all fully tested.

**Architecture:** A single `data_universe/cache.py` module. `Cache` wraps an `OrderedDict` + `threading.Lock`; the lock only ever guards dict mutation, never the actual `loader()` call, so one slow load can't block lookups for other keys. A module-level singleton (`get_cache()`/`reset_cache()`) is what the `@cached("namespace")` decorator writes through; the decorator derives the full cache namespace as `f"{type(self).__name__}.{namespace}"` from the bound method's `self`, so two source classes decorated with the same string never collide. `simulate()` replays a logged sequence of `(namespace, key)` lookups against candidate cache sizes without re-issuing any real loads.

**Tech Stack:** Python ≥3.9, pandas (DataFrame/Series copy + parquet I/O), threading (stdlib), pytest.

**Spec:** `PLAN.md` (repo root) §5 ("The cache — design specifics") and §3's `cache.py` API sketch. This plan implements Phase 3 of `PLAN.md` §10 only: "Cache: complete with all tests." Real sources (Polygon/FRED/EDGAR), `calendar.py`, `ticker.py`, the options toolkit, features/labels, and `precompute.py` are later phases and are **not** built here.

## Global Constraints

- Do **not** use `functools.lru_cache` — it can't share one capacity across call sites, can't report stats, and can't log accesses. Use `OrderedDict` + `threading.Lock` as specified.
- One shared capacity across all data calls, default 256 entries, configurable via `data_universe.config.get_cache_size()`/`set_cache_size()` (already implemented in Phase 2).
- Cache namespace **must** include the calling class's name (e.g. `"PolygonSource.daily_bars"`), so a fake source and a real source never share entries even under the same decorator string.
- On a cache **hit**, return a **copy** of any `pd.DataFrame`/`pd.Series` value (`.copy()`), so a caller mutating its result can't corrupt the cache. Non-DataFrame/Series values are returned as-is except `dict`/`list`, which get `copy.deepcopy`.
- The actual `loader()` call happens **outside** the lock. A double in-flight load for the same key under concurrent misses is a known, accepted limitation — do not add extra synchronization to prevent it.
- Access log has exactly these columns, in this order: `timestamp, namespace, key_repr, hit, bytes, load_seconds, project`. `project` comes from `data_universe.config.get_project()` at the moment of the access, not at cache-construction time.
- `cache.flush_log(path=None)` with no `path` writes to `<config.get_data_dir()>/cache_logs/access_log_<timestamp>.parquet`, creating the directory if needed.
- Out of scope for this phase (do not build): TTLs, disk caching of raw API responses, byte limits, chunking of overlapping date ranges into sub-ranges.
- Docstrings on every public function/class/method, stating units (seconds, bytes) where relevant.

---

### Task 1: Core `Cache` class — get_or_load, eviction, stats, access log, flush_log

**Files:**
- Create: `data_universe/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `data_universe.config.get_project()`, `get_data_dir()`, `get_cache_size()` (all Phase 2).
- Produces: `make_key(args: tuple, kwargs: dict) -> tuple` (key normalization); `class Cache(capacity: Optional[int] = None)` with `.get_or_load(namespace: str, key: Hashable, loader: Callable[[], Any]) -> Any`, `.clear() -> None`, `.stats() -> dict` (keys: `size, hits, misses, evictions, hit_rate, bytes_held`), `.access_log() -> pd.DataFrame` (columns: `timestamp, namespace, key_repr, hit, bytes, load_seconds, project`), `.flush_log(path: Optional[str] = None) -> str`. Task 2's `cached` decorator and `get_cache()`/`reset_cache()` singleton wrap this class without modifying it. Task 3's `simulate()` consumes the `pd.DataFrame` shape produced by `.access_log()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cache.py`:

```python
import datetime as dt

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import Cache, make_key


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def test_get_or_load_calls_loader_only_once_on_repeat_hit():
    cache = Cache(capacity=10)
    calls = []

    def loader():
        calls.append(1)
        return "value"

    key = make_key((), {})
    first = cache.get_or_load("ns", key, loader)
    second = cache.get_or_load("ns", key, loader)
    assert first == "value"
    assert second == "value"
    assert len(calls) == 1


def test_stats_hits_and_misses():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("a",), lambda: 1)
    cache.get_or_load("ns", ("a",), lambda: 1)
    cache.get_or_load("ns", ("b",), lambda: 2)
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 2
    assert stats["hit_rate"] == pytest.approx(1 / 3)


def test_lru_eviction_order():
    cache = Cache(capacity=2)
    cache.get_or_load("ns", ("a",), lambda: "A")
    cache.get_or_load("ns", ("b",), lambda: "B")
    cache.get_or_load("ns", ("c",), lambda: "C")  # evicts "a" (least recently used)
    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["evictions"] == 1
    calls = []
    cache.get_or_load("ns", ("a",), lambda: calls.append(1) or "A-reloaded")
    assert calls == [1]  # "a" was evicted, had to reload


def test_lru_recently_used_key_is_not_evicted():
    cache = Cache(capacity=2)
    cache.get_or_load("ns", ("a",), lambda: "A")
    cache.get_or_load("ns", ("b",), lambda: "B")
    cache.get_or_load("ns", ("a",), lambda: "A")  # touch "a", making "b" the LRU
    cache.get_or_load("ns", ("c",), lambda: "C")  # should evict "b", not "a"
    calls = []
    cache.get_or_load("ns", ("a",), lambda: calls.append(1) or "A-reloaded")
    assert calls == []  # "a" still cached, no reload


def test_dataframe_hits_return_independent_copies():
    cache = Cache(capacity=10)
    df = pd.DataFrame({"x": [1, 2, 3]})
    cache.get_or_load("ns", ("k",), lambda: df)
    result = cache.get_or_load("ns", ("k",), lambda: df)
    result.iloc[0, 0] = 999
    fresh = cache.get_or_load("ns", ("k",), lambda: df)
    assert fresh.iloc[0, 0] != 999


def test_series_hits_return_independent_copies():
    cache = Cache(capacity=10)
    series = pd.Series([1, 2, 3])
    cache.get_or_load("ns", ("k",), lambda: series)
    result = cache.get_or_load("ns", ("k",), lambda: series)
    result.iloc[0] = 999
    fresh = cache.get_or_load("ns", ("k",), lambda: series)
    assert fresh.iloc[0] != 999


def test_access_log_records_project_tag():
    config.set_project("vrp-research")
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    log = cache.access_log()
    assert (log["project"] == "vrp-research").all()


def test_access_log_has_expected_columns_and_order():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    log = cache.access_log()
    assert list(log.columns) == [
        "timestamp",
        "namespace",
        "key_repr",
        "hit",
        "bytes",
        "load_seconds",
        "project",
    ]


def test_access_log_marks_hit_and_miss_correctly():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    cache.get_or_load("ns", ("k",), lambda: 1)
    log = cache.access_log()
    assert list(log["hit"]) == [False, True]


def test_flush_log_writes_parquet(tmp_path):
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    out_path = tmp_path / "log.parquet"
    written = cache.flush_log(str(out_path))
    assert written == str(out_path)
    assert out_path.exists()
    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == 1


def test_flush_log_default_path_under_data_dir_cache_logs(tmp_path):
    config.set_data_dir(str(tmp_path))
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    written = cache.flush_log()
    assert (tmp_path / "cache_logs").exists()
    assert written.startswith(str(tmp_path / "cache_logs"))


def test_clear_resets_stats_and_store():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    cache.clear()
    stats = cache.stats()
    assert stats == {
        "size": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "hit_rate": 0.0,
        "bytes_held": 0,
    }


def test_cache_defaults_capacity_from_config():
    config.set_cache_size(3)
    cache = Cache()
    for k in ("a", "b", "c", "d"):
        cache.get_or_load("ns", (k,), lambda k=k: k)
    assert cache.stats()["size"] == 3


def test_make_key_normalizes_dates_and_list_order():
    key1 = make_key((), {"start": dt.date(2024, 1, 1), "tickers": ["AAPL", "MSFT"]})
    key2 = make_key((), {"tickers": ["MSFT", "AAPL"], "start": "2024-01-01"})
    assert key1 == key2


def test_make_key_normalizes_dicts():
    key1 = make_key((), {"opts": {"a": 1, "b": 2}})
    key2 = make_key((), {"opts": {"b": 2, "a": 1}})
    assert key1 == key2


def test_make_key_distinguishes_different_args():
    assert make_key(("AAPL",), {}) != make_key(("MSFT",), {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.cache'`

- [ ] **Step 3: Write `data_universe/cache.py` (Cache class and key normalization only — no decorator/singleton/simulate yet)**

```python
"""A simple, shared in-memory LRU cache with per-access logging.

This is deliberately simple: one `OrderedDict` + one `threading.Lock` per
`Cache` instance, one shared capacity across every data call that uses it.
The goal of this version is to work correctly and measure real traffic --
later caching rules (chunking overlapping date ranges, byte limits, TTLs,
disk backing) come from replaying the access log this module writes, not
from guessing now. See `data_universe.config` for the default capacity,
project tag, and data directory this module reads.

Known limitations (by design, not bugs):
- Overlapping date-range requests are separate cache keys; nothing here
  chunks or merges them.
- Two concurrent misses on the same key can both call `loader()` -- the
  lock only ever guards `OrderedDict` mutation, never the load itself, so
  one slow call can't block lookups for other keys.
"""
from __future__ import annotations

import copy
import datetime as dt
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Hashable, Optional

import pandas as pd

from data_universe import config

_ACCESS_LOG_COLUMNS = [
    "timestamp",
    "namespace",
    "key_repr",
    "hit",
    "bytes",
    "load_seconds",
    "project",
]


def _normalize(value: Any) -> Hashable:
    """Normalize one argument value so equivalent calls produce equal cache keys.

    Dates/datetimes become ISO date strings, dicts become sorted tuples of
    (key, normalized value) pairs, and lists/tuples/sets become sorted tuples
    of normalized elements (falling back to insertion order if elements
    aren't mutually orderable).
    """
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return tuple(sorted((k, _normalize(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        normalized = [_normalize(v) for v in value]
        try:
            return tuple(sorted(normalized))
        except TypeError:
            return tuple(normalized)
    return value


def make_key(args: tuple, kwargs: dict) -> tuple:
    """Build a normalized, hashable cache key from a call's positional args and kwargs.

    Args:
        args: Positional arguments of the call (excluding `self`).
        kwargs: Keyword arguments of the call.

    Returns:
        A hashable tuple that compares equal for calls a human would consider
        equivalent (same dates in different representations, same list
        regardless of order, same dict regardless of key order).
    """
    norm_args = tuple(_normalize(a) for a in args)
    norm_kwargs = tuple(sorted((k, _normalize(v)) for k, v in kwargs.items()))
    return (norm_args, norm_kwargs)


def _sizeof(value: Any) -> int:
    """Best-effort byte size of a cached value, used for `stats()` and the access log."""
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(deep=True))
    try:
        return sys.getsizeof(value)
    except TypeError:
        return 0


def _copy_value(value: Any) -> Any:
    """Return a defensive copy of a cached value so a caller mutating it can't
    corrupt the cache. DataFrames/Series get `.copy()`; dicts/lists get
    `copy.deepcopy`; everything else (assumed immutable) is returned as-is.
    """
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.copy()
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    return value


class Cache:
    """A shared, thread-safe, in-memory LRU cache with an access log.

    Args:
        capacity: Maximum number of entries. Defaults to
            `data_universe.config.get_cache_size()` when None.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        self._capacity = capacity if capacity is not None else config.get_cache_size()
        self._lock = threading.Lock()
        self._store: "OrderedDict[tuple, Any]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._log: list = []

    def get_or_load(self, namespace: str, key: Hashable, loader: Callable[[], Any]) -> Any:
        """Return the cached value for `(namespace, key)`, loading it via `loader()`
        on a miss. The actual load happens outside the lock.

        Args:
            namespace: Cache namespace (see `cached()` for how source methods
                build this to include the calling class's name).
            key: A hashable key, typically built with `make_key()`.
            loader: A zero-argument callable that produces the value on a miss.

        Returns:
            A defensive copy of the cached (or freshly loaded) value.
        """
        full_key = (namespace, key)
        with self._lock:
            if full_key in self._store:
                self._store.move_to_end(full_key)
                value = self._store[full_key]
                hit = True
            else:
                value = None
                hit = False

        if hit:
            result = value
            load_seconds = 0.0
        else:
            start = time.perf_counter()
            result = loader()
            load_seconds = time.perf_counter() - start

        with self._lock:
            if not hit:
                self._store[full_key] = result
                self._store.move_to_end(full_key)
                while len(self._store) > self._capacity:
                    self._store.popitem(last=False)
                    self._evictions += 1
                self._misses += 1
            else:
                self._hits += 1
            self._log.append(
                {
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "namespace": namespace,
                    "key_repr": repr(key),
                    "hit": hit,
                    "bytes": _sizeof(result),
                    "load_seconds": load_seconds,
                    "project": config.get_project(),
                }
            )
        return _copy_value(result)

    def clear(self) -> None:
        """Drop every entry and reset all counters and the access log."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._log.clear()

    def stats(self) -> dict:
        """Return `{size, hits, misses, evictions, hit_rate, bytes_held}`.

        `hit_rate` is `hits / (hits + misses)`, 0.0 if there have been no accesses.
        `bytes_held` is the summed best-effort byte size of every currently cached value.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0
            bytes_held = sum(_sizeof(v) for v in self._store.values())
            return {
                "size": len(self._store),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "bytes_held": bytes_held,
            }

    def access_log(self) -> pd.DataFrame:
        """Return the full access log as a DataFrame, one row per lookup, in
        chronological order. See module docstring for the known limitations.
        """
        with self._lock:
            rows = list(self._log)
        return pd.DataFrame(rows, columns=_ACCESS_LOG_COLUMNS)

    def flush_log(self, path: Optional[str] = None) -> str:
        """Write the access log to parquet and return the path written.

        Args:
            path: Destination path. Defaults to
                `<config.get_data_dir()>/cache_logs/access_log_<timestamp>.parquet`,
                creating the `cache_logs` directory if needed.
        """
        log = self.access_log()
        if path is None:
            cache_log_dir = Path(config.get_data_dir()) / "cache_logs"
            cache_log_dir.mkdir(parents=True, exist_ok=True)
            ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S%f")
            path = str(cache_log_dir / f"access_log_{ts}.parquet")
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        log.to_parquet(path, index=False)
        return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: all tests PASS

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/cache.py tests/test_cache.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/cache.py tests/test_cache.py
git commit -m "feat: add Cache class with LRU eviction, stats, and access logging"
```

---

### Task 2: `@cached` decorator and the process-wide singleton

**Files:**
- Modify: `data_universe/cache.py`
- Modify: `tests/test_cache.py`

**Interfaces:**
- Consumes: `Cache`, `make_key` (Task 1).
- Produces: `get_cache() -> Cache` (singleton, created on first use), `reset_cache() -> None` (test-only, replaces the singleton), `cached(namespace: str) -> Callable` decorator for instance methods. Every real source module in a later phase decorates its data-fetching methods with `@cached("<method_name>")`; the decorator alone is responsible for building the full namespace (`f"{type(self).__name__}.{namespace}"`) and the key (`make_key(args, kwargs)`), and for routing through `get_cache()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache.py`:

```python
from data_universe.cache import cached, get_cache, reset_cache


@pytest.fixture(autouse=True)
def _reset_cache_singleton():
    reset_cache()
    yield
    reset_cache()


def test_get_cache_returns_singleton():
    a = get_cache()
    b = get_cache()
    assert a is b


def test_reset_cache_creates_a_new_instance():
    a = get_cache()
    reset_cache()
    b = get_cache()
    assert a is not b


def test_cached_decorator_hits_on_repeat_call_with_same_args():
    calls = []

    class Source:
        @cached("thing")
        def thing(self, x, y=1):
            calls.append((x, y))
            return x + y

    s = Source()
    assert s.thing(1, y=2) == 3
    assert s.thing(1, y=2) == 3
    assert len(calls) == 1


def test_cached_decorator_misses_on_different_args():
    calls = []

    class Source:
        @cached("thing")
        def thing(self, x):
            calls.append(x)
            return x * 2

    s = Source()
    assert s.thing(1) == 2
    assert s.thing(2) == 4
    assert len(calls) == 2


def test_cached_decorator_namespace_includes_class_name():
    class SourceA:
        @cached("daily_bars")
        def daily_bars(self, ticker):
            return f"A:{ticker}"

    class SourceB:
        @cached("daily_bars")
        def daily_bars(self, ticker):
            return f"B:{ticker}"

    a_result = SourceA().daily_bars("AAPL")
    b_result = SourceB().daily_bars("AAPL")
    assert a_result == "A:AAPL"
    assert b_result == "B:AAPL"
    namespaces = set(get_cache().access_log()["namespace"])
    assert "SourceA.daily_bars" in namespaces
    assert "SourceB.daily_bars" in namespaces


def test_cached_decorator_two_instances_of_same_class_share_cache():
    calls = []

    class Source:
        @cached("thing")
        def thing(self, x):
            calls.append(x)
            return x

    Source().thing(1)
    Source().thing(1)  # different instance, same class and args -> should hit
    assert len(calls) == 1


def test_load_happens_outside_lock_allows_concurrent_different_keys():
    import threading
    import time as time_module

    cache = get_cache()

    def slow_loader(tag):
        time_module.sleep(0.2)
        return tag

    results = {}

    def worker(key):
        results[key] = cache.get_or_load("ns", (key,), lambda: slow_loader(key))

    start = time_module.perf_counter()
    threads = [threading.Thread(target=worker, args=(k,)) for k in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time_module.perf_counter() - start
    assert results == {"a": "a", "b": "b"}
    assert elapsed < 0.35  # would be >= 0.4s if the lock serialized the two loads
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py -v -k "cached or singleton or lock"`
Expected: FAIL — `ImportError: cannot import name 'cached' from 'data_universe.cache'`

- [ ] **Step 3: Append to `data_universe/cache.py`**

```python
import functools

_default_cache: Optional[Cache] = None
_default_cache_lock = threading.Lock()


def get_cache() -> Cache:
    """Return the process-wide default `Cache`, creating it on first use."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = Cache()
        return _default_cache


def reset_cache() -> None:
    """Replace the process-wide default `Cache` with a fresh one. Test-only."""
    global _default_cache
    with _default_cache_lock:
        _default_cache = None


def cached(namespace: str) -> Callable:
    """Decorator for `DataSource` instance methods: caches on `(ClassName.namespace, args)`.

    Args:
        namespace: A short name for the wrapped method, e.g. "daily_bars". The
            actual cache namespace also includes the bound instance's class
            name (`f"{type(self).__name__}.{namespace}"`), so two different
            source classes decorated with the same `namespace` string never
            share entries.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            full_namespace = f"{type(self).__name__}.{namespace}"
            key = make_key(args, kwargs)
            return get_cache().get_or_load(full_namespace, key, lambda: func(self, *args, **kwargs))

        return wrapper

    return decorator
```

Add the `functools` import to the top-level import block (next to `copy`, `datetime`, etc.) rather than inline, and move `Optional[Cache]`'s use above the class definitions is not required -- `_default_cache: Optional[Cache] = None` can go directly below the `Cache` class since `Optional`/`Callable` are already imported at the top of the file from Task 1.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: all tests PASS (Task 1's tests plus Task 2's)

- [ ] **Step 5: Run ruff**

Run: `ruff check data_universe/cache.py tests/test_cache.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add data_universe/cache.py tests/test_cache.py
git commit -m "feat: add cached() decorator and process-wide cache singleton"
```

---

### Task 3: `simulate()`, public API export, and full-suite verification

**Files:**
- Modify: `data_universe/cache.py`
- Modify: `tests/test_cache.py`
- Modify: `data_universe/__init__.py`

**Interfaces:**
- Consumes: the `pd.DataFrame` shape produced by `Cache.access_log()` (Task 1): columns `namespace, key_repr, hit, bytes, load_seconds, project` (order doesn't matter for `simulate`, only presence of `namespace`, `key_repr`, `load_seconds`).
- Produces: `simulate(log: pd.DataFrame, sizes: list[int]) -> pd.DataFrame` with columns `size, hit_rate, load_seconds_saved`, one row per requested size, in the order given. `data_universe/__init__.py` re-exports `get_cache` and `simulate` per `PLAN.md` §3's public API sketch (`q.get_cache()`, `q.simulate(...)`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cache.py`:

```python
from data_universe.cache import simulate


def test_simulate_returns_expected_columns_and_size_order():
    cache = Cache(capacity=2)
    for k in ("a", "b", "c", "a", "b", "c"):
        cache.get_or_load("ns", (k,), lambda k=k: k)
    log = cache.access_log()
    result = simulate(log, [1, 2, 4])
    assert list(result.columns) == ["size", "hit_rate", "load_seconds_saved"]
    assert list(result["size"]) == [1, 2, 4]


def test_simulate_hit_rate_increases_with_size():
    # A tiny real cache (capacity=1) means most accesses in the log are real
    # misses; a bigger simulated cache should recover more hits from the same
    # repeating sequence of keys.
    cache = Cache(capacity=1)
    keys = ["a", "b", "c", "d"] * 3
    for k in keys:
        cache.get_or_load("ns", (k,), lambda k=k: k)
    log = cache.access_log()
    result = simulate(log, [1, 2, 4]).set_index("size")
    assert result.loc[1, "hit_rate"] <= result.loc[2, "hit_rate"]
    assert result.loc[2, "hit_rate"] <= result.loc[4, "hit_rate"]
    assert result.loc[4, "hit_rate"] > result.loc[1, "hit_rate"]


def test_simulate_load_seconds_saved_is_nonnegative_and_grows_with_size():
    cache = Cache(capacity=1)
    keys = ["a", "b", "a", "b", "a", "b"]
    for k in keys:
        cache.get_or_load("ns", (k,), lambda k=k: k)
    log = cache.access_log()
    result = simulate(log, [1, 2]).set_index("size")
    assert result.loc[1, "load_seconds_saved"] >= 0.0
    assert result.loc[2, "load_seconds_saved"] >= result.loc[1, "load_seconds_saved"]


def test_simulate_on_empty_log_returns_zero_rates():
    empty_log = get_cache().access_log()  # freshly reset, no accesses yet
    result = simulate(empty_log, [1, 10]).set_index("size")
    assert (result["hit_rate"] == 0.0).all()
    assert (result["load_seconds_saved"] == 0.0).all()
```

Also add, in a new file `tests/test_public_api.py`:

```python
import data_universe as q


def test_get_cache_is_exported():
    assert q.get_cache is not None
    assert q.get_cache() is q.get_cache()


def test_simulate_is_exported():
    assert q.simulate is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py tests/test_public_api.py -v -k "simulate or public_api"`
Expected: FAIL — `ImportError: cannot import name 'simulate' from 'data_universe.cache'` and/or `AttributeError: module 'data_universe' has no attribute 'get_cache'`

- [ ] **Step 3: Append `simulate()` to `data_universe/cache.py`**

```python
def simulate(log: pd.DataFrame, sizes: list) -> pd.DataFrame:
    """Replay a logged sequence of `(namespace, key)` lookups against candidate
    cache sizes, without re-issuing any real loads, and report the hit rate
    and load-seconds saved at each size.

    For each row, in the log's original chronological order: if the
    `(namespace, key_repr)` pair is already in the simulated cache of the
    given size, it's a simulated hit -- if that row's real `load_seconds` was
    greater than zero (i.e. it was a real miss when the log was recorded),
    that load is counted as saved. Otherwise it's a simulated miss, inserted
    into the simulated cache, evicting the least-recently-used entry if the
    size is exceeded.

    Args:
        log: An access log as returned by `Cache.access_log()`. Must have
            `namespace`, `key_repr`, and `load_seconds` columns.
        sizes: Candidate cache capacities to simulate, in the order to report them.

    Returns:
        A DataFrame with one row per size: `size`, `hit_rate`, `load_seconds_saved`.
    """
    rows = []
    for size in sizes:
        store: "OrderedDict[tuple, bool]" = OrderedDict()
        hits = 0
        misses = 0
        saved = 0.0
        for _, row in log.iterrows():
            sim_key = (row["namespace"], row["key_repr"])
            if sim_key in store:
                store.move_to_end(sim_key)
                hits += 1
                saved += float(row["load_seconds"])
            else:
                misses += 1
                store[sim_key] = True
                if len(store) > size:
                    store.popitem(last=False)
        total = hits + misses
        hit_rate = hits / total if total else 0.0
        rows.append({"size": size, "hit_rate": hit_rate, "load_seconds_saved": saved})
    return pd.DataFrame(rows, columns=["size", "hit_rate", "load_seconds_saved"])
```

- [ ] **Step 4: Update `data_universe/__init__.py` to re-export the cache API**

```python
"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (Ticker,
precompute, feature/label registries). This phase adds `get_cache` and
`simulate` on top of Phase 2's `config`.
"""
from data_universe import config
from data_universe.cache import get_cache, simulate

__version__ = "0.1.0"

__all__ = ["config", "get_cache", "simulate", "__version__"]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_cache.py tests/test_public_api.py -v`
Expected: all tests PASS

- [ ] **Step 6: Run the full local suite exactly as CI will**

Run: `ruff check .`
Expected: `All checks passed!`

Run: `pytest -v`
Expected: every test from Phase 2 plus this phase's `test_cache.py`/`test_public_api.py` PASSES, 0 failures, 0 errors.

- [ ] **Step 7: Confirm no network access is required**

Run: `python -c "import socket; socket.socket = None; import data_universe; data_universe.get_cache().stats(); print('ok - no import-time or stats-time network use')"`
Expected: prints `ok - no import-time or stats-time network use`

- [ ] **Step 8: Commit**

```bash
git add data_universe/cache.py data_universe/__init__.py tests/test_cache.py tests/test_public_api.py
git commit -m "feat: add cache.simulate() and export get_cache/simulate from the public API"
```

---

## Self-Review Notes (for the executor)

- Task 2's decorator test `test_cached_decorator_two_instances_of_same_class_share_cache` is intentional: the namespace is per-class, not per-instance, matching `PLAN.md` §5 ("namespace = f'{type(self).__name__}.{decorator_namespace}'" — no instance identity in the key). Do not add instance-scoping; a later phase's `Ticker("AAPL")` and a second `Ticker("AAPL")` instance are expected to share cache entries.
- `test_load_happens_outside_lock_allows_concurrent_different_keys` is timing-based and could be flaky on a very slow CI runner; if it flakes, widen the threshold (e.g. `0.35` → `0.5`) rather than removing the concurrency assertion — it's the only test proving the "load outside the lock" requirement.
- `calendar.py`, `ticker.py`, real sources, options toolkit, features/labels, and `precompute.py` remain out of scope — do not add them even though `Cache`/`cached` are now ready for them to use.
