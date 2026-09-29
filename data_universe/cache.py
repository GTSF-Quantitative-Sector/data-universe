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
import functools
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


def _sizeof(value: Any, _seen: Optional[set] = None) -> int:
    """Best-effort byte size of a cached value, used for `stats()` and the access log.

    `sys.getsizeof()` alone only counts a container's own overhead, not what it
    references, so a dict or list of strings (e.g. `MassiveSource.ticker_details`'s
    return value) would otherwise size as a few hundred bytes no matter how much
    text it holds. Recurses into dict/list/tuple values to add those in; `_seen`
    guards against double-counting shared references or looping on a cycle.
    """
    if isinstance(value, pd.DataFrame):
        return int(value.memory_usage(deep=True).sum())
    if isinstance(value, pd.Series):
        return int(value.memory_usage(deep=True))

    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return 0
    _seen.add(value_id)

    try:
        size = sys.getsizeof(value)
    except TypeError:
        size = 0

    if isinstance(value, dict):
        for k, v in value.items():
            size += _sizeof(k, _seen) + _sizeof(v, _seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            size += _sizeof(item, _seen)

    return size


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
            return get_cache().get_or_load(
                full_namespace, key, lambda: func(self, *args, **kwargs)
            )

        return wrapper

    return decorator


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
