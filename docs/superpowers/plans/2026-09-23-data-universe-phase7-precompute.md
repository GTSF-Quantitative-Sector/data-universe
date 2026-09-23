# Phase 7 — Precompute to Parquet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `data_universe/precompute.py` — batch-compute registered features/labels for a
list of tickers over a date range, write them to parquet in long format, upsert on rerun
instead of duplicating rows, and isolate one ticker/feature's failure from the rest of the
batch — plus `load_feature()` to read the result back.

**Architecture:** `precompute()` loops over `(feature_name, ticker)` pairs, looking each name
up in `FEATURE_REGISTRY` first and `LABEL_REGISTRY` second, calling the shared
`features.registry.compute()` machinery already built in Phase 6. Each pair's failure is
caught individually and recorded in a `PrecomputeReport`, never raised out of the batch.
Successful, non-NaN rows for a given feature name are gathered across all its tickers into
one long-format `(date, ticker, value)` DataFrame and upserted into
`<out_dir>/<feature_name>.parquet` — read-merge-dedupe-on-`(date,ticker)`-write. `load_feature()`
reads that same file back, optionally filtered to a ticker subset.

**Tech Stack:** pandas, pyarrow (parquet), the existing `features.registry`/`labels.registry`
`compute()` and `FEATURE_REGISTRY`/`LABEL_REGISTRY`, `data_universe.config` for the default
output directory, `data_universe.sources.polygon.PolygonSource` as the default real source,
`data_universe.sources.fakes.FakePolygonSource` for tests.

**Spec:** `PLAN.md` §3 (`precompute.py` section) and §10 item 7.

## Global Constraints

- Output layout: `data/processed/<feature_name>.parquet`, long format `date, ticker, value`
  (PLAN.md §3).
- Rerunning `precompute` for a ticker/date range already present upserts on `(date, ticker)`
  rather than duplicating rows (PLAN.md §3).
- One ticker's failure (e.g. a missing filing, a 404 from Polygon) is caught, recorded in
  `PrecomputeReport.failed`, and does not stop the batch (PLAN.md §3).
- `PrecomputeReport` is a `@dataclass` with `succeeded: dict[tuple[str, str], int]` (keyed
  `(ticker, feature)` → rows written) and `failed: dict[tuple[str, str], str]` (keyed
  `(ticker, feature)` → error message) (PLAN.md §3).
- Public signatures per PLAN.md §3:
  `precompute(tickers: list[str], feature_names: list[str], start: str, end: str, out_dir: str | None = None) -> PrecomputeReport`
  and `load_feature(name: str, tickers: list[str] | None = None) -> pd.DataFrame`. This plan
  adds one extra trailing keyword-only parameter to each — `source` on `precompute` (defaults
  to a real `PolygonSource()`, overridable for tests) and `out_dir` on `load_feature` (mirrors
  `precompute`'s override, defaults from `config.get_data_dir()`) — neither changes the
  documented positional call shape from `PLAN.md`'s README example.
- Ruff: `line-length = 100`, `select = ["E", "F", "I"]` (see `ruff.toml`). Run
  `.venv/bin/ruff check .` after every task.
- Every commit ends with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

### Existing interfaces this plan consumes (do not redefine)

```python
# data_universe/features/registry.py
FEATURE_REGISTRY: dict  # name -> FeatureSpec
def compute(registry, name, ticker, start, end, source, calendar_proxy="SPY", **kwargs) -> pd.Series: ...
# Raises KeyError if name not in registry.

# data_universe/labels/registry.py
LABEL_REGISTRY: dict  # name -> FeatureSpec (same shape)
def compute(registry, name, ticker, start, end, source, calendar_proxy="SPY", **kwargs) -> pd.Series: ...

# data_universe/config.py
def get_data_dir() -> str: ...

# data_universe/sources/polygon.py
class PolygonSource:
    def __init__(self, session=None, max_retries=5, backoff_seconds=1.0) -> None: ...

# data_universe/sources/fakes.py
class FakePolygonSource:
    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
```

A `compute()` result `pd.Series` is indexed by tz-naive `date`; its `.name` is the feature name
(not relied upon here — this plan renames the column explicitly when building long frames).

---

### Task 1: `PrecomputeReport` + single-feature, single-ticker happy path

**Files:**
- Create: `data_universe/precompute.py`
- Test: `tests/test_precompute.py`

**Interfaces:**
- Produces: `PrecomputeReport` dataclass (`succeeded: dict[tuple[str,str], int]`,
  `failed: dict[tuple[str,str], str]`); `_resolve_registry(name) -> tuple[dict, str]` returning
  `(FEATURE_REGISTRY, "feature")` or `(LABEL_REGISTRY, "label")`, raising `KeyError` if `name`
  is in neither; `_output_path(name, out_dir) -> Path`; `precompute(...)` (partial — single
  pair working, no upsert logic yet, that's Task 2).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precompute.py
import pandas as pd
import pytest

from data_universe import precompute as pc
from data_universe.sources.fakes import FakePolygonSource


def test_precompute_writes_one_feature_one_ticker_to_parquet(tmp_path):
    source = FakePolygonSource()
    report = pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert report.failed == {}
    assert report.succeeded[("AAPL", "ret_1d")] > 0

    out_file = tmp_path / "ret_1d.parquet"
    assert out_file.exists()
    frame = pd.read_parquet(out_file)
    assert list(frame.columns) == ["date", "ticker", "value"]
    assert (frame["ticker"] == "AAPL").all()
    assert frame["value"].notna().all()
    assert len(frame) == report.succeeded[("AAPL", "ret_1d")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_universe.precompute'`

- [ ] **Step 3: Write minimal implementation**

```python
# data_universe/precompute.py
"""Batch-compute registered features/labels to parquet.

Long format `date, ticker, value`, one file per feature/label name under
`<out_dir>/<name>.parquet` (default `<config.get_data_dir()>/processed`).
Reruns upsert on `(date, ticker)` rather than duplicating rows (see
`_upsert_parquet`), and one ticker/feature pair's failure is caught and
recorded in `PrecomputeReport.failed` rather than stopping the batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_universe import config
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.features.registry import compute as _compute
from data_universe.labels.registry import LABEL_REGISTRY
from data_universe.sources.base import DataSource
from data_universe.sources.polygon import PolygonSource


@dataclass
class PrecomputeReport:
    succeeded: Dict[Tuple[str, str], int] = field(default_factory=dict)
    failed: Dict[Tuple[str, str], str] = field(default_factory=dict)


def _resolve_registry(name: str) -> dict:
    """Return the registry dict (`FEATURE_REGISTRY` or `LABEL_REGISTRY`) containing `name`.

    Raises:
        KeyError: If `name` is registered in neither.
    """
    if name in FEATURE_REGISTRY:
        return FEATURE_REGISTRY
    if name in LABEL_REGISTRY:
        return LABEL_REGISTRY
    raise KeyError(f"{name!r} is not registered in FEATURE_REGISTRY or LABEL_REGISTRY")


def _output_dir(out_dir: Optional[str]) -> Path:
    return Path(out_dir) if out_dir is not None else Path(config.get_data_dir()) / "processed"


def _output_path(name: str, out_dir: Optional[str]) -> Path:
    return _output_dir(out_dir) / f"{name}.parquet"


def precompute(
    tickers: List[str],
    feature_names: List[str],
    start: str,
    end: str,
    out_dir: Optional[str] = None,
    source: Optional[DataSource] = None,
) -> PrecomputeReport:
    """Compute every (feature_name, ticker) pair over `[start, end]` and write results
    to `<out_dir>/<feature_name>.parquet` in long format (`date, ticker, value`).

    Args:
        tickers: Ticker symbols to compute for.
        feature_names: Names registered in `FEATURE_REGISTRY` or `LABEL_REGISTRY`.
        start: Start date, "YYYY-MM-DD".
        end: End date, "YYYY-MM-DD".
        out_dir: Output directory. Defaults to `<config.get_data_dir()>/processed`.
        source: Data source passed to `compute()`. Defaults to a real `PolygonSource()`.

    Returns:
        A `PrecomputeReport` of rows written and errors, keyed by `(ticker, feature_name)`.
        A failure for one pair never stops the rest of the batch.
    """
    active_source = source if source is not None else PolygonSource()
    report = PrecomputeReport()

    for name in feature_names:
        try:
            registry = _resolve_registry(name)
        except KeyError as exc:
            for ticker in tickers:
                report.failed[(ticker, name)] = str(exc)
            continue

        for ticker in tickers:
            try:
                series = _compute(registry, name, ticker, start, end, active_source)
            except Exception as exc:  # noqa: BLE001 - isolate any failure, report it
                report.failed[(ticker, name)] = str(exc)
                continue
            clean = series.dropna()
            frame = pd.DataFrame(
                {"date": clean.index, "ticker": ticker, "value": clean.to_numpy()}
            )
            path = _output_path(name, out_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            report.succeeded[(ticker, name)] = len(frame)

    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_universe/precompute.py tests/test_precompute.py
git commit -m "feat: add precompute() single-pair happy path with PrecomputeReport"
```

---

### Task 2: Upsert-on-rerun (merge on `(date, ticker)`, no duplication)

**Files:**
- Modify: `data_universe/precompute.py`
- Test: `tests/test_precompute.py`

**Interfaces:**
- Consumes: `_output_path` from Task 1.
- Produces: `_upsert_parquet(path: Path, new_rows: pd.DataFrame) -> None`, used by `precompute`
  in place of the unconditional `frame.to_parquet(path, ...)` write from Task 1. Also changes
  `precompute` to accumulate all tickers' rows for a given `feature_name` into one frame before
  writing once per name (so a rerun that only touches some tickers still merges correctly
  against rows already on disk for other tickers).

- [ ] **Step 1: Write the failing test**

```python
def test_precompute_rerun_upserts_instead_of_duplicating(tmp_path):
    source = FakePolygonSource()
    pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    first_len = len(pd.read_parquet(tmp_path / "ret_1d.parquet"))

    # Rerun the exact same window -- should not duplicate rows.
    report = pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert len(frame) == first_len
    assert report.succeeded[("AAPL", "ret_1d")] == first_len

    # Extend the window -- new dates get merged in, old ones untouched, no dupes.
    pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-02-15",
        out_dir=str(tmp_path), source=source,
    )
    frame2 = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert len(frame2) > first_len
    assert frame2["date"].is_unique or not frame2.duplicated(subset=["date", "ticker"]).any()


def test_precompute_merges_across_tickers_for_same_feature(tmp_path):
    source = FakePolygonSource()
    pc.precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source)
    pc.precompute(["MSFT"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source)

    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}
    assert not frame.duplicated(subset=["date", "ticker"]).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: FAIL on `test_precompute_merges_across_tickers_for_same_feature` — the second
`precompute` call overwrites `ret_1d.parquet` entirely, losing AAPL's rows (Task 1's
unconditional `to_parquet` per ticker/name pair, last write wins).

- [ ] **Step 3: Write minimal implementation**

Replace the inner loop body of `precompute` (from Task 1) with row-accumulation per
`feature_name`, then a single upsert-write per name:

```python
def _upsert_parquet(path: Path, new_rows: pd.DataFrame) -> None:
    """Merge `new_rows` into the parquet file at `path` on `(date, ticker)`, new rows
    winning on a key collision. Creates the file (and its parent directory) if absent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined.drop_duplicates(subset=["date", "ticker"], keep="last")
    combined = combined.sort_values(["date", "ticker"]).reset_index(drop=True)
    combined.to_parquet(path, index=False)


def precompute(
    tickers: List[str],
    feature_names: List[str],
    start: str,
    end: str,
    out_dir: Optional[str] = None,
    source: Optional[DataSource] = None,
) -> PrecomputeReport:
    """See module docstring. (Args/Returns unchanged from Task 1.)"""
    active_source = source if source is not None else PolygonSource()
    report = PrecomputeReport()

    for name in feature_names:
        try:
            registry = _resolve_registry(name)
        except KeyError as exc:
            for ticker in tickers:
                report.failed[(ticker, name)] = str(exc)
            continue

        rows_by_ticker: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                series = _compute(registry, name, ticker, start, end, active_source)
            except Exception as exc:  # noqa: BLE001 - isolate any failure, report it
                report.failed[(ticker, name)] = str(exc)
                continue
            clean = series.dropna()
            frame = pd.DataFrame(
                {"date": clean.index, "ticker": ticker, "value": clean.to_numpy()}
            )
            rows_by_ticker[ticker] = frame
            report.succeeded[(ticker, name)] = len(frame)

        if rows_by_ticker:
            new_rows = pd.concat(rows_by_ticker.values(), ignore_index=True)
            _upsert_parquet(_output_path(name, out_dir), new_rows)

    return report
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: PASS (all 4 tests so far)

- [ ] **Step 5: Commit**

```bash
git add data_universe/precompute.py tests/test_precompute.py
git commit -m "feat: upsert precompute output on (date, ticker) instead of overwriting"
```

---

### Task 3: Per-pair failure isolation (unregistered name + a raising source)

**Files:**
- Modify: `tests/test_precompute.py` (new tests only — no further production code expected;
  if a test fails, fix `precompute` in `data_universe/precompute.py` to match)

**Interfaces:**
- Consumes: `precompute`, `PrecomputeReport` as they stand after Task 2.

- [ ] **Step 1: Write the failing test**

```python
class _RaisingSource:
    """Wraps a real source but raises for one specific ticker, to test isolation."""

    def __init__(self, inner, bad_ticker: str) -> None:
        self._inner = inner
        self._bad_ticker = bad_ticker

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker == self._bad_ticker:
            raise ValueError(f"simulated failure for {ticker}")
        return self._inner.daily_bars(ticker, start, end)

    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self._inner.index_daily_bars(ticker, start, end)


def test_precompute_isolates_unregistered_feature_name(tmp_path):
    source = FakePolygonSource()
    report = pc.precompute(
        ["AAPL"], ["not_a_real_feature"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert report.succeeded == {}
    assert "not_a_real_feature" in report.failed[("AAPL", "not_a_real_feature")]
    assert not (tmp_path / "not_a_real_feature.parquet").exists()


def test_precompute_isolates_one_ticker_failure_from_the_rest(tmp_path):
    source = _RaisingSource(FakePolygonSource(), bad_ticker="BADCO")
    report = pc.precompute(
        ["AAPL", "BADCO"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert ("AAPL", "ret_1d") in report.succeeded
    assert "simulated failure for BADCO" in report.failed[("BADCO", "ret_1d")]

    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert set(frame["ticker"]) == {"AAPL"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: These should already PASS against the Task 2 implementation (failure isolation and
unregistered-name handling were built in Tasks 1–2). This step confirms that — no
implementation change anticipated. If either fails, treat it as a real bug in
`data_universe/precompute.py` and fix it before continuing (most likely spot: the
`_resolve_registry` `except KeyError` branch, or the per-ticker `try/except` in the inner loop).

- [ ] **Step 3: (only if Step 2 failed) Fix the implementation, then re-run to confirm PASS.**

- [ ] **Step 4: Run full test file to verify all pass**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_precompute.py
git commit -m "test: cover per-pair failure isolation in precompute()"
```

---

### Task 4: `load_feature()`

**Files:**
- Modify: `data_universe/precompute.py`
- Test: `tests/test_precompute.py`

**Interfaces:**
- Produces: `load_feature(name: str, tickers: Optional[List[str]] = None, out_dir: Optional[str] = None) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

```python
def test_load_feature_reads_back_precomputed_values(tmp_path):
    source = FakePolygonSource()
    pc.precompute(["AAPL", "MSFT"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source)

    frame = pc.load_feature("ret_1d", out_dir=str(tmp_path))
    assert list(frame.columns) == ["date", "ticker", "value"]
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}

    filtered = pc.load_feature("ret_1d", tickers=["AAPL"], out_dir=str(tmp_path))
    assert set(filtered["ticker"]) == {"AAPL"}


def test_load_feature_raises_a_clear_error_when_nothing_precomputed(tmp_path):
    with pytest.raises(FileNotFoundError, match="ret_1d"):
        pc.load_feature("ret_1d", out_dir=str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: FAIL — `AttributeError: module 'data_universe.precompute' has no attribute 'load_feature'`

- [ ] **Step 3: Write minimal implementation**

```python
def load_feature(
    name: str,
    tickers: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Read back a precomputed feature/label written by `precompute()`.

    Args:
        name: Feature/label name (must have been precomputed at least once).
        tickers: Optional subset of tickers to filter to. Defaults to all tickers present.
        out_dir: Output directory `precompute()` wrote to. Defaults to
            `<config.get_data_dir()>/processed`, matching `precompute()`'s default.

    Returns:
        A long-format `date, ticker, value` DataFrame.

    Raises:
        FileNotFoundError: If `name` has never been precomputed to `out_dir`.
    """
    path = _output_path(name, out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No precomputed data for {name!r} at {path} -- run precompute() first."
        )
    frame = pd.read_parquet(path)
    if tickers is not None:
        frame = frame[frame["ticker"].isin(tickers)].reset_index(drop=True)
    return frame
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_precompute.py -v`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Commit**

```bash
git add data_universe/precompute.py tests/test_precompute.py
git commit -m "feat: add load_feature() to read back precomputed parquet output"
```

---

### Task 5: Label support + wire into the public API

**Files:**
- Modify: `data_universe/precompute.py` (no change expected — `_resolve_registry` already
  checks `LABEL_REGISTRY`; this task's test proves it), `data_universe/__init__.py`,
  `tests/test_precompute.py`, `tests/test_public_api.py`

**Interfaces:**
- Consumes: `LABEL_REGISTRY` (already imported in `precompute.py` since Task 1).
- Produces: `data_universe.precompute` and `data_universe.load_feature` exported from
  `data_universe/__init__.py`'s public API (per PLAN.md §3's `q.precompute(...)` /
  `q.load_feature(...)` usage).

- [ ] **Step 1: Write the failing test**

```python
def test_precompute_accepts_a_label_name(tmp_path):
    source = FakePolygonSource()
    report = pc.precompute(
        ["AAPL"], ["fwd_ret_15d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert report.failed == {}
    assert report.succeeded[("AAPL", "fwd_ret_15d")] > 0
    frame = pd.read_parquet(tmp_path / "fwd_ret_15d.parquet")
    assert (frame["ticker"] == "AAPL").all()
```

Add to `tests/test_public_api.py`:

```python
def test_precompute_and_load_feature_are_exported():
    assert callable(q.precompute)
    assert callable(q.load_feature)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_precompute.py tests/test_public_api.py -v`
Expected: `test_precompute_accepts_a_label_name` PASSES already (no code change needed --
confirms existing `_resolve_registry` behavior). `test_precompute_and_load_feature_are_exported`
FAILS with `AttributeError: module 'data_universe' has no attribute 'precompute'`.

- [ ] **Step 3: Write minimal implementation**

In `data_universe/__init__.py`, add the import and `__all__` entries (keep existing lines,
insert alphabetically among the `from data_universe...` imports per the `ruff` `I001`
import-sort rule):

```python
from data_universe.precompute import load_feature, precompute
```

and extend `__all__`:

```python
__all__ = [
    "config",
    "get_cache",
    "simulate",
    "universe",
    "FEATURE_REGISTRY",
    "LABEL_REGISTRY",
    "precompute",
    "load_feature",
    "__version__",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_precompute.py tests/test_public_api.py -v`
Expected: PASS (all tests)

- [ ] **Step 5: Run ruff and the full suite, then commit**

```bash
.venv/bin/ruff check .
.venv/bin/pytest -q
git add data_universe/__init__.py data_universe/precompute.py tests/test_precompute.py tests/test_public_api.py
git commit -m "feat: support label names in precompute() and export precompute/load_feature"
```

---

## Self-Review Notes (already folded into the tasks above)

- **Spec coverage:** `PrecomputeReport` (Task 1), upsert-on-rerun (Task 2), per-pair failure
  isolation (Task 1 built it, Task 3 proves it), `load_feature` (Task 4), public API wiring
  (Task 5). `PLAN.md` §3's long-format `date, ticker, value` output is asserted directly in
  every parquet-reading test.
- **No placeholders:** every step has runnable code, not descriptions.
- **Type/name consistency:** `PrecomputeReport.succeeded`/`failed` keyed `(ticker, name)`
  throughout every task and test; `_output_path`/`_output_dir`/`_upsert_parquet`/
  `_resolve_registry` names are introduced once (Task 1 or 2) and reused verbatim afterward.
