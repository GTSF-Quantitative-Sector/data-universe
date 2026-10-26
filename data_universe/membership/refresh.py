"""Rebuild the committed point-in-time S&P 500 tables from upstream.

Upstream is fja05680/sp500's "S&P 500 Historical Components & Changes (Updated).csv": one
row per membership-change date, `date,tickers`, where `tickers` is that day's full
comma-separated member list. `derive.derive_ticker_intervals` turns those snapshots into
per-ticker intervals; this module adds the download, validation and table writing around it.

    python -m data_universe.membership.refresh --dry-run   # report changes, write nothing
    python -m data_universe.membership.refresh             # upstream's latest commit
    python -m data_universe.membership.refresh --ref SHA   # a specific upstream commit

Tables are written to `store.tables_dir("sp500")`: the packaged
`data_universe/membership/tables/sp500` directory in a repo checkout, or
`DATA_UNIVERSE_MEMBERSHIP_TABLES_DIR` when that is set. Everything is written to a staging
directory and validated (`intervals.membership_intervals`) before any committed file is
replaced.

Identity stays provisional (see `tables/sp500/README.md`): an interval with no existing
`security_map` row gets the default `<ticker>-<YYYYMMDD>` security_id (evidence
`new_default`, not reviewed). Existing `security_map`/`security_master` rows are kept for
every interval that still exists, so a refresh never discards review work; rows whose
interval no longer exists upstream are dropped and reported. `ticker_history.csv` and
`tail_changes.csv` are left untouched.

`verified_through` in the manifest is the UTC date of the refresh run: the day the tables
were last confirmed against upstream's latest commit, which is what the staleness check in
`grid.status()` measures.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests

from data_universe import config
from data_universe.membership import intervals, store
from data_universe.membership.derive import derive_ticker_intervals
from data_universe.membership.errors import MembershipValidationError
from data_universe.membership.schema import (
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TICKER_INTERVALS_COLUMNS,
)

INDEX_ID = "sp500"
UPSTREAM_REPO = "fja05680/sp500"
UPSTREAM_FILENAME = "S&P 500 Historical Components & Changes (Updated).csv"
TOOL_VERSION = "membership-refresh-1"
EXPECTED_MEMBERS = (480, 520)
WRITTEN_FILES = ("ticker_intervals.csv", "security_map.csv", "security_master.csv", "manifest.json")

_COMMIT_URL = "https://api.github.com/repos/{repo}/commits/{ref}"
_RAW_URL = "https://raw.githubusercontent.com/{repo}/{sha}/{path}"


@dataclass
class RefreshResult:
    """What a refresh found upstream and how it changed the tables.

    `added`/`removed` are `(ticker, start_date)` interval keys; `end_changed` is
    `(ticker, start_date, old_end_date_excl, new_end_date_excl)`, with None for an open
    interval; `dropped_reviewed` lists reviewed security_map rows whose interval no longer
    exists upstream.
    """

    commit_sha: str
    commit_date: str
    upstream_sha256: str
    n_snapshot_rows: int
    last_snapshot_date: str
    n_latest_members: int
    n_intervals_before: int
    n_intervals_after: int
    tables_dir: str
    written: bool
    added: List[Tuple[str, str]] = field(default_factory=list)
    removed: List[Tuple[str, str]] = field(default_factory=list)
    end_changed: List[Tuple[str, str, Optional[str], Optional[str]]] = field(default_factory=list)
    n_reviewed_kept: int = 0
    dropped_reviewed: List[Tuple[str, str, str]] = field(default_factory=list)


def resolve_commit(ref: str, session: requests.Session) -> Tuple[str, str]:
    """Resolve an upstream branch, tag, "HEAD" (default branch) or SHA via the GitHub API.

    Returns:
        `(full_commit_sha, committer_date)`, the date as GitHub reports it (ISO UTC).
    """
    response = session.get(
        _COMMIT_URL.format(repo=UPSTREAM_REPO, ref=ref),
        headers={"Accept": "application/vnd.github+json"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["sha"], data["commit"]["committer"]["date"]


def download_snapshots(commit_sha: str, session: requests.Session) -> bytes:
    """Raw bytes of the upstream snapshot CSV at `commit_sha`."""
    url = _RAW_URL.format(repo=UPSTREAM_REPO, sha=commit_sha, path=quote(UPSTREAM_FILENAME))
    response = session.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def parse_snapshots(raw: bytes) -> List[Tuple[str, set]]:
    """Parse upstream CSV bytes into `(date, set_of_tickers)` pairs, sorted by date.

    Raises:
        MembershipValidationError: the header isn't exactly `date,tickers`, a row doesn't
            have two fields, a date isn't YYYY-MM-DD or isn't strictly after the previous
            row's, a member list contains an empty ticker, or there are no rows.
    """
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    header = next(reader, None)
    if header != ["date", "tickers"]:
        raise MembershipValidationError(
            f"Unexpected upstream header {header!r}, expected ['date', 'tickers']"
        )
    snapshots: List[Tuple[str, set]] = []
    for line_number, row in enumerate(reader, start=2):
        if not row:
            continue
        if len(row) != 2:
            raise MembershipValidationError(f"Line {line_number}: expected 2 fields, got {row!r}")
        date, tickers = row
        try:
            canonical = dt.date.fromisoformat(date).isoformat() == date
        except ValueError:
            canonical = False
        if not canonical:
            raise MembershipValidationError(f"Line {line_number}: invalid date {date!r}")
        if snapshots and date <= snapshots[-1][0]:
            raise MembershipValidationError(
                f"Line {line_number}: date {date} is not after the previous row's "
                f"{snapshots[-1][0]}"
            )
        members = {ticker.strip() for ticker in tickers.split(",")}
        if "" in members:
            raise MembershipValidationError(
                f"Line {line_number}: empty ticker in the {date} member list"
            )
        snapshots.append((date, members))
    if not snapshots:
        raise MembershipValidationError("Upstream file has no snapshot rows")
    return snapshots


def build_tables(
    snapshots: List[Tuple[str, set]],
    security_map: pd.DataFrame,
    security_master: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive ticker_intervals/security_map/security_master from `snapshots`.

    Args:
        snapshots: As returned by `parse_snapshots`.
        security_map: The current security_map (kept for intervals that still exist).
        security_master: The current security_master (kept for ids still referenced).

    Returns:
        `(ticker_intervals, security_map, security_master, dropped_map_rows)`.
    """
    derived = derive_ticker_intervals(snapshots)
    ticker_intervals = pd.DataFrame(
        [(INDEX_ID, ticker, start, end) for ticker, start, end in derived],
        columns=TICKER_INTERVALS_COLUMNS,
    )

    existing = security_map[security_map["index_id"] == INDEX_ID]
    existing_by_key = {(row["ticker"], row["start_date"]): row for _, row in existing.iterrows()}
    map_rows = []
    for ticker, start, _ in derived:
        row = existing_by_key.get((ticker, start))
        if row is None:
            identity = (f"{ticker}-{start.replace('-', '')}", "new_default", False)
        else:
            identity = (row["security_id"], row["evidence"], bool(row["reviewed"]))
        map_rows.append([INDEX_ID, ticker, start, *identity])
    new_map = pd.DataFrame(map_rows, columns=SECURITY_MAP_COLUMNS)
    derived_keys = {(ticker, start) for ticker, start, _ in derived}
    is_dropped = [
        key not in derived_keys for key in zip(existing["ticker"], existing["start_date"])
    ]
    dropped = existing.loc[np.array(is_dropped, dtype=bool)]

    latest_ticker = {}
    for start, ticker, security_id in sorted(
        zip(new_map["start_date"], new_map["ticker"], new_map["security_id"])
    ):
        latest_ticker[security_id] = ticker
    existing_master = {row["security_id"]: row for _, row in security_master.iterrows()}
    master_rows = []
    for security_id in sorted(latest_ticker):
        row = existing_master.get(security_id)
        if row is None:
            master_rows.append(
                {"security_id": security_id, "latest_ticker": latest_ticker[security_id]}
            )
        else:
            master_rows.append(row[SECURITY_MASTER_COLUMNS].to_dict())
    new_master = pd.DataFrame(master_rows, columns=SECURITY_MASTER_COLUMNS)
    return ticker_intervals, new_map, new_master, dropped


def _load_existing() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not store.tables_dir(INDEX_ID).is_dir():
        return (
            pd.DataFrame(columns=TICKER_INTERVALS_COLUMNS),
            pd.DataFrame(columns=SECURITY_MAP_COLUMNS),
            pd.DataFrame(columns=SECURITY_MASTER_COLUMNS),
        )
    return (
        store.load_ticker_intervals(INDEX_ID),
        store.load_security_map(INDEX_ID),
        store.load_security_master(INDEX_ID),
    )


def _interval_ends(frame: pd.DataFrame) -> dict:
    return {
        (ticker, start): None if pd.isna(end) else end
        for ticker, start, end in zip(frame["ticker"], frame["start_date"], frame["end_date_excl"])
    }


def _stage_and_install(
    ticker_intervals: pd.DataFrame,
    security_map: pd.DataFrame,
    security_master: pd.DataFrame,
    manifest: dict,
    target: Path,
) -> None:
    """Write the tables to a staging directory, validate them there, then copy them over
    `target` -- so a table that fails validation never replaces a committed file."""
    previous = config.get_membership_tables_dir()
    with tempfile.TemporaryDirectory() as staging_root:
        config.set_membership_tables_dir(staging_root)
        try:
            store.save_ticker_intervals(INDEX_ID, ticker_intervals)
            store.save_security_map(INDEX_ID, security_map)
            store.save_security_master(INDEX_ID, security_master)
            store.save_manifest(INDEX_ID, manifest)
            intervals.membership_intervals(INDEX_ID)
        finally:
            config.set_membership_tables_dir(previous)
        target.mkdir(parents=True, exist_ok=True)
        for name in WRITTEN_FILES:
            shutil.copyfile(Path(staging_root) / INDEX_ID / name, target / name)


def refresh(
    ref: str = "HEAD",
    dry_run: bool = False,
    session: Optional[requests.Session] = None,
    now: Optional[dt.datetime] = None,
) -> RefreshResult:
    """Download upstream at `ref`, rebuild the sp500 tables, and write them unless `dry_run`.

    Args:
        ref: Upstream branch, tag, full commit SHA, or "HEAD" (upstream's default branch).
        dry_run: Compute and report the changes without writing anything.
        session: Injectable `requests.Session` (for tests). Defaults to a fresh one.
        now: Injectable timezone-aware "now" (for tests); stamps `retrieved_at` and
            `verified_through`. Defaults to the current UTC time.

    Returns:
        A `RefreshResult` describing upstream and the table changes.

    Raises:
        MembershipValidationError: the upstream file fails `parse_snapshots`' checks, the
            latest snapshot's member count is outside `EXPECTED_MEMBERS`, or the rebuilt
            tables have an unmapped or overlapping interval.
        requests.HTTPError: GitHub returned an error status.
    """
    session = session or requests.Session()
    now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    commit_sha, commit_date = resolve_commit(ref, session)
    raw = download_snapshots(commit_sha, session)
    snapshots = parse_snapshots(raw)
    last_date, latest_members = snapshots[-1]
    low, high = EXPECTED_MEMBERS
    if not low <= len(latest_members) <= high:
        raise MembershipValidationError(
            f"The latest upstream snapshot ({last_date}) has {len(latest_members)} members, "
            f"outside the expected {low}-{high}"
        )

    old_intervals, old_map, old_master = _load_existing()
    ticker_intervals, security_map, security_master, dropped = build_tables(
        snapshots, old_map, old_master
    )
    old_ends, new_ends = _interval_ends(old_intervals), _interval_ends(ticker_intervals)
    target = store.tables_dir(INDEX_ID)
    upstream_sha256 = hashlib.sha256(raw).hexdigest()
    result = RefreshResult(
        commit_sha=commit_sha,
        commit_date=commit_date,
        upstream_sha256=upstream_sha256,
        n_snapshot_rows=len(snapshots),
        last_snapshot_date=last_date,
        n_latest_members=len(latest_members),
        n_intervals_before=len(old_intervals),
        n_intervals_after=len(ticker_intervals),
        tables_dir=str(target),
        written=not dry_run,
        added=sorted(set(new_ends) - set(old_ends)),
        removed=sorted(set(old_ends) - set(new_ends)),
        end_changed=sorted(
            (ticker, start, old_ends[(ticker, start)], new_ends[(ticker, start)])
            for ticker, start in set(old_ends) & set(new_ends)
            if old_ends[(ticker, start)] != new_ends[(ticker, start)]
        ),
        n_reviewed_kept=int(security_map["reviewed"].sum()),
        dropped_reviewed=[
            (row["ticker"], row["start_date"], row["security_id"])
            for _, row in dropped.iterrows()
            if bool(row["reviewed"])
        ],
    )
    if dry_run:
        return result

    manifest = {
        "n_derived_intervals": len(ticker_intervals),
        "n_snapshot_rows": len(snapshots),
        "retrieved_at": now.isoformat(),
        "tool_version": TOOL_VERSION,
        "upstream_commit_date": commit_date,
        "upstream_commit_sha": commit_sha,
        "upstream_filename": UPSTREAM_FILENAME,
        "upstream_last_snapshot_date": last_date,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_sha256": upstream_sha256,
        "verified_through": now.date().isoformat(),
    }
    _stage_and_install(ticker_intervals, security_map, security_master, manifest, target)
    return result


def format_summary(result: RefreshResult, max_listed: int = 25) -> str:
    """Human-readable report of a `RefreshResult`, as printed by the CLI. Lists at most
    `max_listed` interval changes; dropped reviewed mappings are always listed in full."""
    lines = [
        f"upstream: {UPSTREAM_REPO}@{result.commit_sha} (committed {result.commit_date})",
        f"file sha256: {result.upstream_sha256}",
        f"snapshots: {result.n_snapshot_rows} rows, last change {result.last_snapshot_date} "
        f"({result.n_latest_members} members)",
        f"intervals: {result.n_intervals_before} -> {result.n_intervals_after} "
        f"({len(result.added)} new, {len(result.end_changed)} with a changed end date, "
        f"{len(result.removed)} removed)",
    ]
    changes = (
        [f"new: {ticker} from {start}" for ticker, start in result.added]
        + [
            f"end changed: {ticker} from {start}: {old or 'open'} -> {new or 'open'}"
            for ticker, start, old, new in result.end_changed
        ]
        + [f"removed: {ticker} from {start}" for ticker, start in result.removed]
    )
    lines += [f"  {change}" for change in changes[:max_listed]]
    if len(changes) > max_listed:
        lines.append(f"  ... and {len(changes) - max_listed} more")
    lines.append(f"reviewed security_map rows kept: {result.n_reviewed_kept}")
    lines += [
        f"  WARNING: dropped reviewed mapping {ticker} from {start} -> {security_id}"
        for ticker, start, security_id in result.dropped_reviewed
    ]
    lines.append(
        f"wrote {', '.join(WRITTEN_FILES)} to {result.tables_dir}"
        if result.written
        else f"dry run: nothing written (target {result.tables_dir})"
    )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_universe.membership.refresh",
        description=f"Rebuild the sp500 membership tables from {UPSTREAM_REPO}.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="upstream branch, tag or commit SHA (default: HEAD, upstream's default branch)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report what would change without writing"
    )
    args = parser.parse_args(argv)
    print(format_summary(refresh(ref=args.ref, dry_run=args.dry_run)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
