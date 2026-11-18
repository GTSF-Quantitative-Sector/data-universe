"""Versioned, per-key partitioned derived datasets (precompute v2).

Layout under `root` (default `<config.get_data_dir()>/processed`):

    <root>/<name>/<fingerprint>/dataset.json              -- version manifest
    <root>/<name>/<fingerprint>/<quoted key>.parquet      -- the rows
    <root>/<name>/<fingerprint>/<quoted key>.coverage.json -- compute records
    <root>/<name>/<fingerprint>/.<quoted key>.lock        -- flock target
    <root>/<name>/<fingerprint>/.dataset.lock             -- manifest flock

Each `<name>` may hold several `<fingerprint>` versions side by side; the
fingerprint is a hash of the spec (code version, params, source family,
conventions) so a formula or input change lands in a new directory instead
of silently mixing with old rows. `dataset.json` records the spec that was
hashed plus the stored column list (null until the first non-empty write);
`coverage.json` records every computed range as
`{"start", "end", "rows", "computed_at", "tool_version", "source"}` with
inclusive "YYYY-MM-DD" dates -- coverage is the durable statement of what
a key file provably holds, never inferred from the rows.

Writes are atomic (temp file in the same directory + `os.replace`) and
`lock(name, fingerprint, key)` takes an exclusive `fcntl.flock` (a no-op
where fcntl is missing); callers hold it around read-modify-write --
`replace_range` assumes the caller holds the key lock, while `invalidate`
takes the key lock itself.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote, unquote

import pandas as pd
import pyarrow.parquet

from data_universe import config
from data_universe.sources.base import NY_TZ

try:
    import fcntl
except ImportError:  # Windows has no fcntl -- writes stay atomic, unlocked
    fcntl = None

DERIVED_SCHEMA_VERSION = 1

_STATUS_COLUMNS = [
    "name", "kind", "fingerprint", "schema_version", "key", "coverage_start",
    "coverage_end", "n_intervals", "rows", "last_computed_at", "tool_version",
]


def _row_dates(frame: pd.DataFrame, date_column) -> pd.DatetimeIndex:
    """Tz-naive normalized row dates from `date_column` (or the index)."""
    if date_column is not None:
        return pd.DatetimeIndex(pd.to_datetime(frame[date_column])).normalize()
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:
        index = index.tz_convert(NY_TZ).tz_localize(None)
    return index.normalize()


class DerivedStore:
    """Read/write access to the on-disk derived store."""

    def __init__(self, root=None) -> None:
        self.root = (
            Path(root) if root is not None else Path(config.get_data_dir()) / "processed"
        )

    def version_dir(self, name: str, fingerprint: str) -> Path:
        return self.root / name / fingerprint

    def _paths(self, name: str, fingerprint: str, key: str):
        quoted = quote(str(key), safe="")
        directory = self.version_dir(name, fingerprint)
        return (
            directory / f"{quoted}.parquet",
            directory / f"{quoted}.coverage.json",
            directory / f".{quoted}.lock",
        )

    @contextlib.contextmanager
    def _file_lock(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
            yield

    @contextlib.contextmanager
    def lock(self, name: str, fingerprint: str, key: str):
        """Exclusive `fcntl.flock` on the key's lock file (no-op where fcntl
        is unavailable -- the atomic writes still apply)."""
        _, _, lock_path = self._paths(name, fingerprint, key)
        with self._file_lock(lock_path):
            yield

    @contextlib.contextmanager
    def _dataset_lock(self, name: str, fingerprint: str):
        with self._file_lock(self.version_dir(name, fingerprint) / ".dataset.lock"):
            yield

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_atomic(path: Path, writer) -> None:
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

    def _write_json(self, path: Path, payload: dict) -> None:
        self._write_atomic(path, lambda tmp: tmp.write_text(json.dumps(payload, indent=2)))

    # -- dataset manifests -------------------------------------------------

    def dataset_manifest(self, name: str, fingerprint: str) -> Optional[dict]:
        """The parsed `dataset.json`, or None when absent/unreadable."""
        return self._read_json(self.version_dir(name, fingerprint) / "dataset.json")

    def versions(self, name: str) -> List[dict]:
        """Every parseable `dataset.json` under `<root>/<name>/*/` (any
        schema_version), sorted by `created_at`."""
        directory = self.root / name
        found = []
        if directory.is_dir():
            for manifest_path in directory.glob("*/dataset.json"):
                manifest = self._read_json(manifest_path)
                if manifest is not None:
                    found.append(manifest)
        return sorted(found, key=lambda m: str(m.get("created_at", "")))

    def names(self) -> List[str]:
        """Dataset names with at least one versioned directory."""
        if not self.root.is_dir():
            return []
        return sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir() and any(path.glob("*/dataset.json"))
        )

    def ensure_dataset(self, name: str, fingerprint: str, manifest: dict) -> dict:
        """Write `dataset.json` if absent (atomic, under `.dataset.lock`) and
        return the stored manifest -- the first writer's spec wins."""
        path = self.version_dir(name, fingerprint) / "dataset.json"
        with self._dataset_lock(name, fingerprint):
            existing = self._read_json(path)
            if existing is None:
                self._write_json(path, manifest)
                existing = manifest
        return existing

    def set_columns(self, name: str, fingerprint: str, columns: List[str]) -> None:
        """Record the stored column list on first non-empty write; a later
        write proposing different columns is a version bump, not an update."""
        with self._dataset_lock(name, fingerprint):
            manifest = self.dataset_manifest(name, fingerprint)
            if manifest is None:
                return
            stored = manifest.get("columns")
            if stored is None:
                manifest["columns"] = list(columns)
                self._write_json(
                    self.version_dir(name, fingerprint) / "dataset.json", manifest
                )
            elif list(stored) != list(columns):
                raise ValueError(
                    f"output columns {list(columns)} differ from this version's "
                    f"{list(stored)} -- bump the version"
                )

    # -- per-key data --------------------------------------------------------

    def keys(self, name: str, fingerprint: str) -> List[str]:
        """Unquoted key stems of the version's `*.parquet`/`*.coverage.json`."""
        directory = self.version_dir(name, fingerprint)
        found = set()
        if directory.is_dir():
            for path in directory.glob("*.parquet"):
                found.add(unquote(path.stem))
            for path in directory.glob("*.coverage.json"):
                found.add(unquote(path.name[: -len(".coverage.json")]))
        return sorted(found)

    def read(self, name: str, fingerprint: str, key: str) -> Optional[pd.DataFrame]:
        """The stored frame, or None when absent/unreadable."""
        data_path, _, _ = self._paths(name, fingerprint, key)
        if not data_path.exists():
            return None
        try:
            return pd.read_parquet(data_path)
        except Exception:  # noqa: BLE001 - unreadable data reads as absent
            return None

    def _coverage_skeleton(self, name: str, fingerprint: str, key: str) -> dict:
        return {
            "schema_version": DERIVED_SCHEMA_VERSION,
            "name": name,
            "fingerprint": fingerprint,
            "key": str(key),
            "records": [],
        }

    def records(self, name: str, fingerprint: str, key: str) -> List[dict]:
        """The raw coverage records, or [] when absent/wrong schema."""
        _, coverage_path, _ = self._paths(name, fingerprint, key)
        payload = self._read_json(coverage_path)
        if payload is None or payload.get("schema_version") != DERIVED_SCHEMA_VERSION:
            return []
        return list(payload.get("records", []))

    def coverage(
        self, name: str, fingerprint: str, key: str
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Merged [start, end] coverage (adjacent within one day merges).
        [] on schema mismatch or unreadable file."""
        intervals = sorted(
            (pd.Timestamp(record["start"]), pd.Timestamp(record["end"]))
            for record in self.records(name, fingerprint, key)
        )
        merged = []
        for start, end in intervals:
            if merged and start <= merged[-1][1] + pd.Timedelta(days=1):
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        return merged

    def replace_range(
        self,
        name: str,
        fingerprint: str,
        key: str,
        start,
        end,
        fresh: pd.DataFrame,
        record: Optional[dict] = None,
        date_column: str = "date",
    ) -> None:
        """Drop rows dated in [start, end], add `fresh` rows dated in
        [start, end], stable-sort by date, write atomically (an empty result
        unlinks the data file), then append `record` to coverage.json.
        CALLER holds the key lock. Unreadable existing data/coverage is
        treated as absent."""
        start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        data_path, coverage_path, _ = self._paths(name, fingerprint, key)

        existing = None
        if data_path.exists():
            try:
                existing = pd.read_parquet(data_path)
            except Exception:  # noqa: BLE001 - unreadable data treated as absent
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
            frame = frame.iloc[order].reset_index(drop=True)
        else:
            frame = fresh.iloc[0:0]

        if len(frame):
            self._write_atomic(data_path, lambda tmp: frame.to_parquet(tmp, index=False))
        elif data_path.exists():
            data_path.unlink()

        if record is not None:
            payload = self._read_json(coverage_path)
            if payload is None or payload.get("schema_version") != DERIVED_SCHEMA_VERSION:
                payload = self._coverage_skeleton(name, fingerprint, key)
            payload["records"].append(record)
            self._write_json(coverage_path, payload)

    def invalidate(
        self,
        name: str,
        fingerprint: str,
        key: str,
        start=None,
        end=None,
    ) -> bool:
        """Remove the key's data+coverage (no dates) or delete rows dated in
        [start, end] and subtract the range from every coverage record
        (either bound may be None = open-ended). Returns whether anything
        changed. Takes the key lock itself."""
        data_path, coverage_path, _ = self._paths(name, fingerprint, key)
        with self.lock(name, fingerprint, key):
            if start is None and end is None:
                changed = False
                for path in (data_path, coverage_path):
                    if path.exists():
                        path.unlink()
                        changed = True
                return changed

            s = pd.Timestamp(start).normalize() if start is not None else None
            e = pd.Timestamp(end).normalize() if end is not None else None
            changed = False

            manifest = self.dataset_manifest(name, fingerprint)
            date_column = (manifest or {}).get("date_column") or "date"
            if data_path.exists():
                try:
                    existing = pd.read_parquet(data_path)
                except Exception:  # noqa: BLE001 - unreadable data treated as absent
                    existing = None
                if existing is not None and len(existing):
                    dates = _row_dates(existing, date_column)
                    in_range = pd.Series(True, index=dates)
                    if s is not None:
                        in_range &= dates >= s
                    if e is not None:
                        in_range &= dates <= e
                    if bool(in_range.any()):
                        kept = existing.loc[~in_range.to_numpy()]
                        if len(kept):
                            self._write_atomic(
                                data_path, lambda tmp: kept.to_parquet(tmp, index=False)
                            )
                        else:
                            data_path.unlink()
                        changed = True

            payload = self._read_json(coverage_path)
            if payload is not None and payload.get("schema_version") == DERIVED_SCHEMA_VERSION:
                records = payload.get("records", [])
                day = pd.Timedelta(days=1)
                new_records = []
                for record in records:
                    r_start = pd.Timestamp(record["start"])
                    r_end = pd.Timestamp(record["end"])
                    # subtract [s, e]: keep [r_start, s-1d] and [e+1d, r_end]
                    if s is not None:
                        left_end = min(s - day, r_end)
                        if r_start <= left_end:
                            new_records.append(
                                {**record, "start": str(r_start.date()),
                                 "end": str(left_end.date())}
                            )
                    if e is not None:
                        right_start = max(e + day, r_start)
                        if right_start <= r_end:
                            new_records.append(
                                {**record, "start": str(right_start.date()),
                                 "end": str(r_end.date())}
                            )
                if new_records != records:
                    payload["records"] = new_records
                    self._write_json(coverage_path, payload)
                    changed = True
            return changed

    def remove_version(self, name: str, fingerprint: str) -> None:
        shutil.rmtree(self.version_dir(name, fingerprint), ignore_errors=True)

    def status(self) -> pd.DataFrame:
        """One row per (name, fingerprint, key): kind, schema_version,
        coverage span, interval count, row count, last computed_at,
        tool_version."""
        rows = []
        if self.root.is_dir():
            for manifest_path in sorted(self.root.glob("*/*/dataset.json")):
                manifest = self._read_json(manifest_path)
                if manifest is None:
                    continue
                name = manifest.get("name") or manifest_path.parent.parent.name
                fingerprint = manifest.get("fingerprint") or manifest_path.parent.name
                for key in self.keys(name, fingerprint):
                    coverage = self.coverage(name, fingerprint, key)
                    records = self.records(name, fingerprint, key)
                    data_path = (
                        self.version_dir(name, fingerprint)
                        / f"{quote(key, safe='')}.parquet"
                    )
                    n_rows = 0
                    if data_path.exists():
                        try:
                            n_rows = int(
                                pyarrow.parquet.ParquetFile(data_path).metadata.num_rows
                            )
                        except Exception:  # noqa: BLE001 - unreadable reads as 0
                            n_rows = 0
                    computed = [r.get("computed_at") for r in records if r.get("computed_at")]
                    rows.append(
                        {
                            "name": name,
                            "kind": manifest.get("kind"),
                            "fingerprint": fingerprint,
                            "schema_version": manifest.get("schema_version"),
                            "key": key,
                            "coverage_start": coverage[0][0] if coverage else pd.NaT,
                            "coverage_end": coverage[-1][1] if coverage else pd.NaT,
                            "n_intervals": len(coverage),
                            "rows": n_rows,
                            "last_computed_at": max(computed) if computed else None,
                            "tool_version": manifest.get("tool_version"),
                        }
                    )
        return pd.DataFrame(rows, columns=_STATUS_COLUMNS)
