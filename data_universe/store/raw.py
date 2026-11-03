"""Durable per-(dataset, key) raw store: one parquet file plus a JSON
manifest per dataset/key pair, atomic writes, fcntl locking.

Layout under `root` (default `<config.get_data_dir()>/raw`):

    <root>/<dataset>/<quote(key, safe="")>.parquet       -- the raw rows
    <root>/<dataset>/<quoted>.manifest.json             -- fetch records
    <root>/<dataset>/.<quoted>.lock                     -- flock target

The manifest records every upstream fetch as
`{"start", "end", "status" ("ok"|"empty"|"not_entitled"), "rows",
"fetched_at", "source", "family", "tool_version", "rebased"}` so `coverage`
(the union of ok/empty ranges) is the durable statement of what the file
holds -- read coverage is never inferred from the rows themselves. A
manifest whose `schema_version` isn't `STORE_SCHEMA_VERSION` is treated as
absent everywhere: `read`/`coverage` return nothing and the next write
discards the old rows and records.

Writes are atomic (temp file in the same directory + `os.replace`) so a
reader never sees a half-written file. `lock(dataset, key)` takes an
exclusive `fcntl.flock` on the lock file (a no-op where fcntl is missing);
callers take it around read-modify-write -- the write methods don't lock
themselves.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

import pandas as pd

from data_universe import config
from data_universe.sources.base import NY_TZ

try:
    import fcntl
except ImportError:  # Windows has no fcntl -- writes stay atomic, unlocked
    fcntl = None

STORE_SCHEMA_VERSION = 1


def _row_dates(frame: pd.DataFrame, date_column) -> pd.DatetimeIndex:
    """Tz-naive normalized row dates: the `date_column` if given, else the
    (possibly tz-aware) index converted to America/New_York."""
    if date_column is not None:
        return pd.DatetimeIndex(pd.to_datetime(frame[date_column])).normalize()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(NY_TZ).tz_localize(None)
    return index.normalize()


def _fetch_record(start, end, status, rows, source, family, rebased) -> dict:
    import data_universe

    def _stamp(value):
        return pd.Timestamp(value).strftime("%Y-%m-%d")

    return {
        "start": _stamp(start),
        "end": _stamp(end),
        "status": status,
        "rows": int(rows),
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source": source,
        "family": family,
        "tool_version": data_universe.__version__,
        "rebased": bool(rebased),
    }


class RawStore:
    """Read/write access to the on-disk raw store. `read` memoizes loaded
    frames per (dataset, key) keyed on the data file's (st_ino, st_mtime_ns,
    st_size) -- `os.replace` gives a new inode per write -- and returns
    copies; the memo is dropped on own writes."""

    def __init__(self, root=None) -> None:
        self.root = (
            Path(root) if root is not None else Path(config.get_data_dir()) / "raw"
        )
        self._memo = {}
        self._memo_lock = threading.Lock()

    def _paths(self, dataset: str, key: str):
        quoted = quote(str(key), safe="")
        directory = self.root / dataset
        return (
            directory / f"{quoted}.parquet",
            directory / f"{quoted}.manifest.json",
            directory / f".{quoted}.lock",
        )

    @contextlib.contextmanager
    def lock(self, dataset: str, key: str):
        """Exclusive `fcntl.flock` on `<root>/<dataset>/.<quoted>.lock`
        (no-op where fcntl is unavailable -- the atomic writes still apply)."""
        _, _, lock_path = self._paths(dataset, key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "a") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield

    def _skeleton(self, dataset: str, key: str) -> dict:
        return {
            "dataset": dataset,
            "key": str(key),
            "schema_version": STORE_SCHEMA_VERSION,
            "fetches": [],
        }

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    def manifest(self, dataset: str, key: str) -> dict:
        """The parsed manifest, or an empty skeleton when absent/unreadable."""
        _, manifest_path, _ = self._paths(dataset, key)
        found = self._read_json(manifest_path)
        return found if found is not None else self._skeleton(dataset, key)

    def _drop_memo(self, dataset: str, key: str) -> None:
        with self._memo_lock:
            self._memo.pop((dataset, key), None)

    def read(self, dataset: str, key: str) -> Optional[pd.DataFrame]:
        """The stored frame, or None when absent or the manifest's
        schema_version isn't current."""
        data_path, manifest_path, _ = self._paths(dataset, key)
        if not data_path.exists():
            return None
        try:
            stat = data_path.stat()
        except OSError:
            return None
        stat_key = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        with self._memo_lock:
            entry = self._memo.get((dataset, key))
            if entry is not None and entry[0] == stat_key:
                return entry[1].copy()
        manifest = self._read_json(manifest_path)
        if manifest is None or manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            return None
        try:
            frame = pd.read_parquet(data_path)
        except Exception:  # noqa: BLE001 - unreadable data reads as absent
            return None
        with self._memo_lock:
            self._memo[(dataset, key)] = (stat_key, frame)
        return frame.copy()

    def coverage(self, dataset: str, key: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Inclusive [start, end] ranges the store provably holds: the merged
        union of ok/empty fetch records (adjacent within one day merges).
        [] on schema mismatch."""
        manifest = self.manifest(dataset, key)
        if manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            return []
        intervals = sorted(
            (pd.Timestamp(record["start"]), pd.Timestamp(record["end"]))
            for record in manifest.get("fetches", [])
            if record.get("status") in ("ok", "empty")
        )
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + pd.Timedelta(days=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def _write_atomic(self, path: Path, writer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp"
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            writer(tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def upsert_range(
        self, dataset: str, key: str, start, end, fresh: pd.DataFrame,
        date_column, record: dict,
    ) -> None:
        """Drop existing rows dated in [start, end], add `fresh` rows dated in
        [start, end], sort by row date, write the parquet, then append
        `record` to the manifest. A schema-mismatched or unreadable existing
        file/manifest is treated as absent."""
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        data_path, manifest_path, _ = self._paths(dataset, key)
        manifest = self._read_json(manifest_path)
        if manifest is None or manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            manifest = self._skeleton(dataset, key)
            existing = None
        elif data_path.exists():
            try:
                existing = pd.read_parquet(data_path)
            except Exception:  # noqa: BLE001 - unreadable data treated as absent
                existing = None
                manifest["fetches"] = []
        else:
            existing = None

        parts = []
        if existing is not None and len(existing):
            old_dates = _row_dates(existing, date_column)
            parts.append(existing.iloc[((old_dates < start) | (old_dates > end))])
        if len(fresh):
            fresh_dates = _row_dates(fresh, date_column)
            parts.append(fresh.iloc[(fresh_dates >= start) & (fresh_dates <= end)])
        if parts:
            frame = pd.concat(parts)
            order = _row_dates(frame, date_column).argsort(kind="stable")
            frame = frame.iloc[order]
            if date_column is not None:
                frame = frame.reset_index(drop=True)
        else:
            frame = fresh.iloc[0:0]

        self._write_atomic(data_path, lambda tmp: frame.to_parquet(tmp))
        manifest["fetches"].append(record)
        self._write_atomic(
            manifest_path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2))
        )
        self._drop_memo(dataset, key)

    def replace_all(self, dataset: str, key: str, fresh: pd.DataFrame,
                    record: dict) -> None:
        """Overwrite data with `fresh` and the manifest's fetches with [record]."""
        data_path, manifest_path, _ = self._paths(dataset, key)
        manifest = self._skeleton(dataset, key)
        manifest["fetches"] = [record]
        self._write_atomic(data_path, lambda tmp: fresh.to_parquet(tmp))
        self._write_atomic(
            manifest_path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2))
        )
        self._drop_memo(dataset, key)

    def record(self, dataset: str, key: str, record: dict) -> None:
        """Append a fetch record without touching the data file. A missing,
        mismatched or unreadable manifest starts a fresh skeleton -- the stale
        data file goes with it so old-schema rows can't read as current."""
        data_path, manifest_path, _ = self._paths(dataset, key)
        manifest = self._read_json(manifest_path)
        if manifest is None or manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            manifest = self._skeleton(dataset, key)
            if data_path.exists():
                data_path.unlink()
            self._drop_memo(dataset, key)
        manifest["fetches"].append(record)
        self._write_atomic(
            manifest_path, lambda tmp: tmp.write_text(json.dumps(manifest, indent=2))
        )

    def invalidate(self, dataset: str, key: str) -> None:
        """Remove the data file and manifest (under the lock)."""
        data_path, manifest_path, _ = self._paths(dataset, key)
        with self.lock(dataset, key):
            for path in (data_path, manifest_path):
                if path.exists():
                    path.unlink()
        self._drop_memo(dataset, key)

    def status(self) -> pd.DataFrame:
        """One row per manifest file: dataset, key, family, coverage_start,
        coverage_end, n_intervals, n_fetches, last_fetched_at, last_status,
        schema_version."""
        rows = []
        if self.root.exists():
            for manifest_path in sorted(self.root.glob("*/*.manifest.json")):
                manifest = self._read_json(manifest_path)
                if manifest is None:
                    continue
                records = manifest.get("fetches", [])
                coverage = self.coverage(manifest["dataset"], manifest["key"])
                families = {r.get("family") for r in records if r.get("family")}
                last = records[-1] if records else {}
                rows.append(
                    {
                        "dataset": manifest["dataset"],
                        "key": manifest["key"],
                        "family": sorted(families)[0] if families else None,
                        "coverage_start": coverage[0][0] if coverage else pd.NaT,
                        "coverage_end": coverage[-1][1] if coverage else pd.NaT,
                        "n_intervals": len(coverage),
                        "n_fetches": len(records),
                        "last_fetched_at": last.get("fetched_at"),
                        "last_status": last.get("status"),
                        "schema_version": manifest.get("schema_version"),
                    }
                )
        return pd.DataFrame(
            rows,
            columns=[
                "dataset", "key", "family", "coverage_start", "coverage_end",
                "n_intervals", "n_fetches", "last_fetched_at", "last_status",
                "schema_version",
            ],
        )
