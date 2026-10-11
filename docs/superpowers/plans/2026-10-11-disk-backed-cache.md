# Disk-Backed Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional disk-backed second tier to `data_universe.cache.Cache`, so cached
values survive process restarts and can be shared between multiple colocated processes
reading from the same disk, without changing default behavior for existing callers/tests.

**Architecture:** `Cache` gains an optional `disk_dir` constructor argument. When set, a
`diskcache.Cache(disk_dir, size_limit=...)` instance backs the existing in-memory
`OrderedDict` LRU as a second tier: a miss in memory checks disk before calling the
caller's `loader()`; a value found on disk is promoted into the in-memory tier (and still
subject to its eviction) rather than hitting the network again. A value loaded fresh is
written to both tiers. `disk_dir=None` (the default) disables the disk tier entirely --
every existing `Cache(capacity=N)` call site and all 285 existing tests keep behaving
exactly as they do today. The process-wide singleton (`get_cache()`) turns the disk tier on
automatically when `config.get_cache_disk_dir()` is set (new config key, env var
`DATA_UNIVERSE_CACHE_DISK_DIR` / yaml key `cache_disk_dir`), so multiple applications
pointed at the same directory share the same persisted entries across restarts.

**Tech Stack:** Python, `diskcache` (new dependency; SQLite-backed, thread- and
process-safe out of the box -- avoids hand-rolling file locking for concurrent readers).

**Spec:** This plan is the spec; it was agreed in conversation (no separate design doc).
Context: `data_universe/cache.py`'s own module docstring currently lists "disk backing" as
a deferred decision ("come from replaying the access log ... not from guessing now"). The
project owner explicitly asked for it now, for the stated reason that multiple
applications running from one disk (colocated) should get hot local data instead of
re-pulling from Polygon/Massive/FRED/EDGAR every time, and that it must survive a process
restart (a pure in-memory cache cannot).

## Global Constraints

- Default behavior (`disk_dir=None`) must be byte-for-byte identical to the current
  `Cache` -- every one of the current 285 tests must keep passing unmodified except where
  this plan explicitly says to update a test (new stats/log columns).
- `diskcache` is a new `install_requires` entry in `setup.py` and a new line in
  `requirements.txt`, matching the existing style (no version pin, consistent with every
  other dependency in those files).
- Config precedence stays **environment variable > `config/config.yaml` > hardcoded
  default**, matching every other `config.py` setting.
- `ruff check .` must pass with zero errors before each commit (matches every prior phase).
- Keep the two-tier design transparent to callers: `@cached(...)`-decorated methods and
  `get_or_load()`'s signature do not change. Only `Cache.__init__`, `Cache.stats()`, and
  the access log gain new, additive fields.

---

### Task 1: Add `diskcache` dependency

**Files:**
- Modify: `setup.py`
- Modify: `requirements.txt`

**Interfaces:**
- Produces: `diskcache` importable in the environment for every later task.

- [ ] **Step 1: Add the dependency**

In `setup.py`, add `"diskcache"` to `install_requires` (alphabetical-ish, matching the
existing list's style -- after `"lxml"`):

```python
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "pyarrow",
        "scipy",
        "pyyaml",
        "lxml",
        "diskcache",
    ],
```

In `requirements.txt`, add a line `diskcache` after `lxml`.

- [ ] **Step 2: Install and verify import**

Run: `pip install -e . && python -c "import diskcache; print(diskcache.__version__)"`
Expected: prints a version string, no ImportError.

- [ ] **Step 3: Commit**

```bash
git add setup.py requirements.txt
git commit -m "chore: add diskcache dependency for the disk-backed cache tier"
```

---

### Task 2: Add disk-cache config keys

**Files:**
- Modify: `data_universe/config.py`
- Modify: `config/config.example.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing new -- follows the exact existing pattern of `cache_size` /
  `DATA_UNIVERSE_CACHE_SIZE` / `get_cache_size()` already in this file.
- Produces: `config.set_cache_disk_dir(path)`, `config.get_cache_disk_dir() -> Optional[str]`,
  `config.set_cache_disk_size_limit(n)`, `config.get_cache_disk_size_limit() -> Optional[int]`.
  Both default to `None` (disk tier off) unless set via env var or yaml.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py` (follow the existing fixture pattern in that file -- it
already resets `config` between tests):

```python
def test_cache_disk_dir_defaults_to_none():
    assert config.get_cache_disk_dir() is None


def test_cache_disk_dir_set_and_get():
    config.set_cache_disk_dir("/tmp/some-cache-dir")
    assert config.get_cache_disk_dir() == "/tmp/some-cache-dir"


def test_cache_disk_dir_env_var_wins_over_set(monkeypatch):
    monkeypatch.setenv("DATA_UNIVERSE_CACHE_DISK_DIR", "/env/dir")
    config.reset()
    assert config.get_cache_disk_dir() == "/env/dir"


def test_cache_disk_size_limit_defaults_to_none():
    assert config.get_cache_disk_size_limit() is None


def test_cache_disk_size_limit_set_and_get():
    config.set_cache_disk_size_limit(1_000_000)
    assert config.get_cache_disk_size_limit() == 1_000_000


def test_load_yaml_reads_cache_disk_dir(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("cache_disk_dir: /yaml/dir\ncache_disk_size_limit_bytes: 500\n")
    config.load_yaml(str(p))
    assert config.get_cache_disk_dir() == "/yaml/dir"
    assert config.get_cache_disk_size_limit() == 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -k "disk" -v`
Expected: FAIL -- `AttributeError: module 'data_universe.config' has no attribute
'get_cache_disk_dir'` (and similar for the others).

- [ ] **Step 3: Implement**

In `data_universe/config.py`, add to `_ENV_VARS`:

```python
    "cache_disk_dir": "DATA_UNIVERSE_CACHE_DISK_DIR",
    "cache_disk_size_limit": "DATA_UNIVERSE_CACHE_DISK_SIZE_LIMIT",
```

Add to `_YAML_KEYS`:

```python
    "cache_disk_dir": "cache_disk_dir",
    "cache_disk_size_limit_bytes": "cache_disk_size_limit",
```

In `_default()`, the `cache_size` branch does `int(value)` for one specific state key.
Extend that same branch to also cover `cache_disk_size_limit`:

```python
    if state_key in ("cache_size", "cache_disk_size_limit"):
        return int(value)
```

In `reset()`, add:

```python
    _state["cache_disk_dir"] = _default("cache_disk_dir", None)
    _state["cache_disk_size_limit"] = _default("cache_disk_size_limit", None)
```

Add the four new functions (place them right after `get_cache_size()`):

```python
def set_cache_disk_dir(path: Optional[str]) -> None:
    """Set the directory for the disk-backed cache tier. None disables it (default)."""
    _state["cache_disk_dir"] = path


def get_cache_disk_dir() -> Optional[str]:
    """Return the configured disk-cache directory, or None if the disk tier is off."""
    return _state["cache_disk_dir"]


def set_cache_disk_size_limit(n: Optional[int]) -> None:
    """Set the disk-cache byte size limit. None uses diskcache's own default (1GB)."""
    _state["cache_disk_size_limit"] = n


def get_cache_disk_size_limit() -> Optional[int]:
    """Return the configured disk-cache byte size limit, or None for diskcache's default."""
    return _state["cache_disk_size_limit"]
```

In `load_yaml()`'s loop, the existing `int(value) if state_key == "cache_size" else value`
line needs the same two-key treatment:

```python
        value = data[yaml_key]
        if state_key in ("cache_size", "cache_disk_size_limit"):
            value = int(value)
        _state[state_key] = value
```

In `config/config.example.yaml`, add after `cache_size: 256`:

```yaml
# Disk-backed cache tier: set cache_disk_dir to persist cached values across process
# restarts and share them between colocated processes reading the same disk. Leave empty
# to keep the cache in-memory only (default).
cache_disk_dir: ""
cache_disk_size_limit_bytes: ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: PASS, all tests including the 6 new ones and every pre-existing one.

- [ ] **Step 5: Commit**

```bash
git add data_universe/config.py config/config.example.yaml tests/test_config.py
git commit -m "feat: add cache_disk_dir / cache_disk_size_limit config keys"
```

---

### Task 3: Add the disk tier to `Cache`

**Files:**
- Modify: `data_universe/cache.py`
- Test: `tests/test_cache.py`

**Interfaces:**
- Consumes: `config.get_cache_disk_dir()`, `config.get_cache_disk_size_limit()` from Task 2.
- Produces: `Cache(capacity=None, disk_dir=None, disk_size_limit=None)`. `stats()` gains
  `disk_hits` and `disk_bytes_held` keys (0 when the disk tier is off). The access log
  gains a `source` column (`"memory"`, `"disk"`, or `"load"`) in place of the plain `hit`
  boolean's binary information -- `hit` stays as the existing boolean (`True` for both
  `"memory"` and `"disk"`) so nothing reading `hit` today breaks; `source` is additive.
  `clear()` now also clears the disk tier when one is configured.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_cache.py`:

```python
import diskcache


def test_disk_dir_none_behaves_exactly_as_before(tmp_path):
    # No disk tier configured -- nothing written to disk at all.
    cache = Cache(capacity=10, disk_dir=None)
    cache.get_or_load("ns", ("k",), lambda: "value")
    assert list(tmp_path.iterdir()) == []


def test_disk_hit_avoids_calling_loader_again(tmp_path):
    calls = []
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: calls.append(1) or "value")
    assert len(calls) == 1

    # Fresh Cache, same disk_dir, simulating a process restart.
    cache2 = Cache(capacity=10, disk_dir=str(tmp_path))
    result = cache2.get_or_load("ns", ("k",), lambda: calls.append(1) or "value")
    assert result == "value"
    assert len(calls) == 1  # loader not called again -- served from disk


def test_disk_hit_is_promoted_into_memory_tier(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: "value")

    cache2 = Cache(capacity=10, disk_dir=str(tmp_path))
    cache2.get_or_load("ns", ("k",), lambda: "value")
    assert cache2.stats()["size"] == 1  # now also in cache2's in-memory tier


def test_disk_hit_counts_separately_from_memory_hit(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: "value")  # miss, written to disk + memory

    cache2 = Cache(capacity=10, disk_dir=str(tmp_path))
    cache2.get_or_load("ns", ("k",), lambda: "value")  # disk hit
    stats = cache2.stats()
    assert stats["disk_hits"] == 1
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_access_log_source_column_distinguishes_memory_disk_and_load(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: "value")  # load
    cache.get_or_load("ns", ("k",), lambda: "value")  # memory hit
    log = cache.access_log()
    assert list(log["source"]) == ["load", "memory"]


def test_disk_hit_access_log_source_is_disk(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: "value")

    cache2 = Cache(capacity=10, disk_dir=str(tmp_path))
    cache2.get_or_load("ns", ("k",), lambda: "value")
    log = cache2.access_log()
    assert list(log["source"]) == ["disk"]


def test_disk_tier_survives_separate_cache_instances_sharing_a_directory(tmp_path):
    # Two "processes" (two Cache instances) pointed at the same directory.
    writer = Cache(capacity=10, disk_dir=str(tmp_path))
    writer.get_or_load("ns", ("shared-key",), lambda: "written-by-writer")

    reader = Cache(capacity=10, disk_dir=str(tmp_path))
    calls = []
    result = reader.get_or_load(
        "ns", ("shared-key",), lambda: calls.append(1) or "should-not-load"
    )
    assert result == "written-by-writer"
    assert calls == []


def test_dataframe_disk_round_trip_preserves_values(tmp_path):
    df = pd.DataFrame({"x": [1, 2, 3]})
    writer = Cache(capacity=10, disk_dir=str(tmp_path))
    writer.get_or_load("ns", ("k",), lambda: df)

    reader = Cache(capacity=10, disk_dir=str(tmp_path))
    result = reader.get_or_load("ns", ("k",), lambda: pd.DataFrame())
    pd.testing.assert_frame_equal(result, df)


def test_disk_hit_result_is_an_independent_copy(tmp_path):
    writer = Cache(capacity=10, disk_dir=str(tmp_path))
    df = pd.DataFrame({"x": [1, 2, 3]})
    writer.get_or_load("ns", ("k",), lambda: df)

    reader = Cache(capacity=10, disk_dir=str(tmp_path))
    result = reader.get_or_load("ns", ("k",), lambda: df)
    result.iloc[0, 0] = 999
    fresh = reader.get_or_load("ns", ("k",), lambda: df)
    assert fresh.iloc[0, 0] != 999


def test_clear_also_clears_disk_tier(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: "value")
    cache.clear()

    reader = Cache(capacity=10, disk_dir=str(tmp_path))
    calls = []
    reader.get_or_load("ns", ("k",), lambda: calls.append(1) or "reloaded")
    assert calls == [1]  # disk entry gone too, had to reload


def test_stats_disk_bytes_held_reflects_disk_tier_when_configured(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path))
    cache.get_or_load("ns", ("k",), lambda: "x" * 10_000)
    assert cache.stats()["disk_bytes_held"] > 9_000


def test_stats_disk_bytes_held_is_zero_when_disk_tier_off():
    cache = Cache(capacity=10, disk_dir=None)
    cache.get_or_load("ns", ("k",), lambda: "x" * 10_000)
    assert cache.stats()["disk_bytes_held"] == 0


def test_disk_size_limit_is_passed_through_to_diskcache(tmp_path):
    cache = Cache(capacity=10, disk_dir=str(tmp_path), disk_size_limit=123_456)
    assert cache._disk.size_limit == 123_456


def test_get_cache_singleton_enables_disk_tier_from_config(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_UNIVERSE_CACHE_DISK_DIR", str(tmp_path))
    config.reset()
    reset_cache()
    cache = get_cache()
    assert cache._disk is not None
    config.reset()
    reset_cache()


def test_get_cache_singleton_has_no_disk_tier_by_default():
    cache = get_cache()
    assert cache._disk is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cache.py -k "disk" -v`
Expected: FAIL -- `TypeError: Cache.__init__() got an unexpected keyword argument 'disk_dir'`.

- [ ] **Step 3: Implement**

In `data_universe/cache.py`:

Add the import at the top:

```python
import diskcache
```

Update the module docstring's "Known limitations" section to remove disk backing from the
deferred list and describe the new tier (replace the whole docstring):

```python
"""A shared, two-tier LRU cache (in-memory, optionally disk-backed) with per-access logging.

The in-memory tier is one `OrderedDict` + one `threading.Lock` per `Cache` instance, one
shared capacity across every data call that uses it -- deliberately simple, and still the
only tier by default (`disk_dir=None`). Passing `disk_dir` adds a `diskcache`-backed second
tier: a miss in memory checks disk before calling `loader()`, so values survive a process
restart and can be shared between multiple processes reading from the same directory
(e.g. several applications colocated on one disk, all hitting the same Polygon/Massive/FRED
data). A value found on disk is promoted into the in-memory tier; a value loaded fresh is
written to both. See `data_universe.config` for the default capacity, project tag, data
directory, and disk-cache directory/size-limit this module reads.

Known limitations (by design, not bugs):
- Overlapping date-range requests are separate cache keys; nothing here
  chunks or merges them.
- Two concurrent misses on the same key can both call `loader()` -- the
  lock only ever guards `OrderedDict` mutation, never the load itself, so
  one slow call can't block lookups for other keys. The disk tier does not
  change this: two processes racing on the same miss can both load and both
  write to disk, with the second write winning.
"""
```

Replace the `Cache` class's `__init__`, `get_or_load`, `clear`, and `stats` with:

```python
class Cache:
    """A shared, thread-safe, two-tier (memory + optional disk) LRU cache with an access log.

    Args:
        capacity: Maximum number of in-memory entries. Defaults to
            `data_universe.config.get_cache_size()` when None.
        disk_dir: Directory for the disk-backed second tier. None (default) keeps the
            cache purely in-memory, matching this class's original behavior exactly.
        disk_size_limit: Byte size limit for the disk tier. None uses diskcache's own
            default. Ignored when `disk_dir` is None.
    """

    def __init__(
        self,
        capacity: Optional[int] = None,
        disk_dir: Optional[str] = None,
        disk_size_limit: Optional[int] = None,
    ) -> None:
        self._capacity = capacity if capacity is not None else config.get_cache_size()
        self._lock = threading.Lock()
        self._store: "OrderedDict[tuple, Any]" = OrderedDict()
        self._hits = 0
        self._disk_hits = 0
        self._misses = 0
        self._evictions = 0
        self._log: list = []
        self._disk: Optional[diskcache.Cache] = None
        if disk_dir is not None:
            kwargs = {}
            if disk_size_limit is not None:
                kwargs["size_limit"] = disk_size_limit
            self._disk = diskcache.Cache(disk_dir, **kwargs)

    def get_or_load(self, namespace: str, key: Hashable, loader: Callable[[], Any]) -> Any:
        """Return the cached value for `(namespace, key)`, loading it via `loader()`
        on a miss in both tiers. The actual load happens outside the lock.

        Checks the in-memory tier first, then the disk tier (if configured) before
        calling `loader()`. A disk hit is promoted into the in-memory tier. A fresh load
        is written to both tiers.

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
                source = "memory"
            else:
                value = None
                source = None

        if source == "memory":
            result = value
            load_seconds = 0.0
        else:
            disk_result = self._disk_get(full_key)
            if disk_result is not None:
                result = disk_result[0]
                load_seconds = 0.0
                source = "disk"
            else:
                start = time.perf_counter()
                result = loader()
                load_seconds = time.perf_counter() - start
                source = "load"

        with self._lock:
            if source == "memory":
                self._hits += 1
            else:
                self._store[full_key] = result
                self._store.move_to_end(full_key)
                while len(self._store) > self._capacity:
                    self._store.popitem(last=False)
                    self._evictions += 1
                if source == "disk":
                    self._disk_hits += 1
                else:
                    self._misses += 1
                    if self._disk is not None:
                        self._disk.set(full_key, result)
            self._log.append(
                {
                    "timestamp": pd.Timestamp.now(tz="UTC"),
                    "namespace": namespace,
                    "key_repr": repr(key),
                    "hit": source in ("memory", "disk"),
                    "source": source,
                    "bytes": _sizeof(result),
                    "load_seconds": load_seconds,
                    "project": config.get_project(),
                }
            )
        return _copy_value(result)

    def _disk_get(self, full_key: tuple) -> Optional[tuple]:
        """Return `(value,)` from the disk tier, or None on a miss or no disk tier."""
        if self._disk is None:
            return None
        sentinel = object()
        value = self._disk.get(full_key, default=sentinel)
        if value is sentinel:
            return None
        return (value,)

    def clear(self) -> None:
        """Drop every entry (memory and disk) and reset all counters and the access log."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._disk_hits = 0
            self._misses = 0
            self._evictions = 0
            self._log.clear()
            if self._disk is not None:
                self._disk.clear()

    def stats(self) -> dict:
        """Return `{size, hits, disk_hits, misses, evictions, hit_rate, bytes_held,
        disk_bytes_held}`.

        `hit_rate` is `(hits + disk_hits) / (hits + disk_hits + misses)`, 0.0 if there
        have been no accesses. `bytes_held` is the summed best-effort byte size of every
        currently in-memory value; `disk_bytes_held` is the disk tier's reported size (0
        when no disk tier is configured).
        """
        with self._lock:
            total = self._hits + self._disk_hits + self._misses
            hit_rate = (self._hits + self._disk_hits) / total if total else 0.0
            bytes_held = sum(_sizeof(v) for v in self._store.values())
            disk_bytes_held = int(self._disk.volume()) if self._disk is not None else 0
            return {
                "size": len(self._store),
                "hits": self._hits,
                "disk_hits": self._disk_hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": hit_rate,
                "bytes_held": bytes_held,
                "disk_bytes_held": disk_bytes_held,
            }
```

Update `_ACCESS_LOG_COLUMNS` to add `"source"` right after `"hit"`:

```python
_ACCESS_LOG_COLUMNS = [
    "timestamp",
    "namespace",
    "key_repr",
    "hit",
    "source",
    "bytes",
    "load_seconds",
    "project",
]
```

Update `get_cache()` to wire the config-driven singleton:

```python
def get_cache() -> Cache:
    """Return the process-wide default `Cache`, creating it on first use.

    The disk tier turns on automatically when `config.get_cache_disk_dir()` is set
    (env var `DATA_UNIVERSE_CACHE_DISK_DIR` or `config/config.yaml`'s `cache_disk_dir`),
    so every process in the same deployment that reads that config shares persisted
    entries from the same directory.
    """
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = Cache(
                disk_dir=config.get_cache_disk_dir(),
                disk_size_limit=config.get_cache_disk_size_limit(),
            )
        return _default_cache
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cache.py -v`
Expected: PASS -- every new test and every pre-existing test (update any pre-existing
test that asserted the old `_ACCESS_LOG_COLUMNS` list or the old `stats()` dict shape;
see Step 5).

- [ ] **Step 5: Fix the two pre-existing tests this change touches**

`test_access_log_has_expected_columns_and_order` currently asserts the column list
without `"source"` -- update it to match the new list above.

`test_clear_resets_stats_and_store` currently asserts the exact `stats()` dict without
`disk_hits`/`disk_bytes_held` -- update it to:

```python
def test_clear_resets_stats_and_store():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    cache.clear()
    stats = cache.stats()
    assert stats == {
        "size": 0,
        "hits": 0,
        "disk_hits": 0,
        "misses": 0,
        "evictions": 0,
        "hit_rate": 0.0,
        "bytes_held": 0,
        "disk_bytes_held": 0,
    }
```

Run: `pytest tests/test_cache.py -v`
Expected: PASS, all tests (30 pre-existing + ~16 new).

- [ ] **Step 6: Run the full suite and ruff**

Run: `pytest -q && ruff check .`
Expected: all tests pass; ruff reports no errors.

- [ ] **Step 7: Commit**

```bash
git add data_universe/cache.py tests/test_cache.py
git commit -m "feat: add optional disk-backed second tier to Cache"
```

---

### Task 4: Docs

**Files:**
- Modify: `README.md`
- Modify: `CONTRACTS.md`
- Modify: `MAINTENANCE.md`

**Interfaces:**
- Consumes: the finished Task 2/3 behavior -- grep every quoted name/column list against
  the actual code before writing prose (same discipline as the Phase 8 docs work).

- [ ] **Step 1: README.md**

In the "Configuration" table (after the `DATA_UNIVERSE_CACHE_SIZE` row), add:

```markdown
| `DATA_UNIVERSE_CACHE_DISK_DIR` | Directory for the disk-backed cache tier. Unset keeps the cache in-memory only (default) |
| `DATA_UNIVERSE_CACHE_DISK_SIZE_LIMIT` | Byte size limit for the disk tier. Unset uses `diskcache`'s own default |
```

In the `cache.py` module-tour bullet, replace:

```markdown
- **`cache.py`** — a shared in-memory LRU `Cache` with per-access logging. `get_cache()`
  returns the process-wide singleton; the `@cached("namespace")` decorator wraps `DataSource`
  methods; `simulate(log, sizes)` replays a logged access sequence against candidate cache
```

with:

```markdown
- **`cache.py`** — a shared, two-tier (in-memory, optionally disk-backed) LRU `Cache` with
  per-access logging. `get_cache()` returns the process-wide singleton, whose disk tier
  turns on when `cache_disk_dir` is configured, so colocated processes share persisted
  entries across restarts; the `@cached("namespace")` decorator wraps `DataSource`
  methods; `simulate(log, sizes)` replays a logged access sequence against candidate cache
```

- [ ] **Step 2: CONTRACTS.md**

In the config precedence table (the one with `DATA_UNIVERSE_CACHE_SIZE` / `cache_size`),
add the two new rows following that table's existing two-column format -- read the
surrounding rows first to match the exact column headers before adding:

```markdown
| `DATA_UNIVERSE_CACHE_DISK_DIR` | `cache_disk_dir` |
| `DATA_UNIVERSE_CACHE_DISK_SIZE_LIMIT` | `cache_disk_size_limit` |
```

- [ ] **Step 3: MAINTENANCE.md**

Add a new numbered item (after the existing 5, matching that file's one-paragraph-each
style):

```markdown
## 6. Disk-backed cache directory

When `cache_disk_dir` points at a shared disk (multiple colocated processes/applications
reading the same path), that directory grows unbounded up to `cache_disk_size_limit`
(default: `diskcache`'s own 1GB limit, which evicts least-recently-used entries once hit).
There is no automatic cleanup beyond that limit and no process separating one project's
entries from another's within the same directory -- give each project its own
`cache_disk_dir` if that isolation matters.
```

- [ ] **Step 4: Verify no drift**

Re-read README.md's Configuration table, CONTRACTS.md's config table, and
`data_universe/config.py` side by side -- every env var in one must appear in the other
two with the matching key name. Fix any mismatch found.

- [ ] **Step 5: Commit**

```bash
git add README.md CONTRACTS.md MAINTENANCE.md
git commit -m "docs: document the disk-backed cache tier"
```

---

## Self-Review Notes (for the plan author, already applied above)

- Verified `diskcache.Cache.get(key, default=...)` and `.set(key, value)` accept
  arbitrary pickleable keys (the `full_key` tuple here is `(str, (tuple, tuple))` of
  plain strings/numbers -- always pickleable) -- no custom key-hashing needed.
- Verified `diskcache.Cache` is itself safe for multiple separate `diskcache.Cache(...)`
  instances (even across processes) pointed at the same directory -- it's SQLite-backed
  with its own internal locking, which is exactly why Task 3's tests construct two
  separate `Cache` instances against the same `tmp_path` to simulate two processes rather
  than needing real subprocesses.
- Checked `diskcache.Cache.volume()` exists and returns the on-disk size in bytes, used
  for `disk_bytes_held`.
- Confirmed no existing test constructs `Cache()` expecting disk writes as a side effect,
  so making `disk_dir` default to `None` cannot silently change any existing test's
  on-disk footprint.
