"""File IO for data_universe.membership's committed reference tables.

This is the only module in the package that opens a file. `intervals.py`
and `grid.py` call into this module and never touch a path themselves (as
will `identity.py`, once identity resolution lands) -- that split keeps
every file-format decision (CSV dtypes, the manifest's JSON shape, where
`tables_dir()` resolves to) in one place.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from data_universe import config
from data_universe.membership.errors import MembershipError
from data_universe.membership.schema import (
    CIK_EVIDENCE_COLUMNS,
    CIK_HISTORY_COLUMNS,
    CIK_TICKER_EVIDENCE_COLUMNS,
    IDENTITY_EVIDENCE_COLUMNS,
    IDENTITY_OVERRIDES_COLUMNS,
    IDENTITY_REVIEW_COLUMNS,
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TAIL_CHANGES_COLUMNS,
    TICKER_HISTORY_COLUMNS,
    TICKER_INTERVALS_COLUMNS,
)

# Explicit sort keys: every saved table gets a deterministic row order so an
# --offline rebuild reproduces the committed bytes exactly. cik_evidence is
# sorted numerically by CIK.
_SORT_KEYS = {
    "ticker_intervals.csv": ["index_id", "ticker", "start_date"],
    "security_map.csv": ["index_id", "ticker", "start_date"],
    "security_master.csv": ["security_id"],
    "ticker_history.csv": ["security_id", "valid_from"],
    "cik_history.csv": ["security_id", "valid_from"],
    "tail_changes.csv": ["effective_date", "ticker"],
    "identity_evidence.csv": ["ticker", "date"],
    "identity_review.csv": ["kind", "ticker", "start_date", "other_ticker", "other_start_date"],
    "cik_evidence.csv": None,         # numeric by cik
    "cik_ticker_evidence.csv": None,  # numeric by cik, then date
}

_PACKAGE_TABLES_DIR = Path(__file__).resolve().parent / "tables"


def tables_dir(index_id: str) -> Path:
    """Directory holding `index_id`'s committed tables.

    Resolves `config.get_membership_tables_dir()` if set, else the
    packaged `data_universe/membership/tables/<index_id>` directory.
    """
    override = config.get_membership_tables_dir()
    root = Path(override) if override else _PACKAGE_TABLES_DIR
    return root / index_id


def _require_tables_dir(index_id: str) -> Path:
    """`tables_dir(index_id)`, but raise `MembershipError` when the whole
    directory is missing -- i.e. `index_id` is unknown (a typo or a not-yet-
    built index), as opposed to a known index missing one specific file."""
    directory = tables_dir(index_id)
    if not directory.is_dir():
        raise MembershipError(
            f"Unknown index_id {index_id!r}: no tables directory found at {directory}"
        )
    return directory


def _load_csv(index_id: str, filename: str, columns: list) -> pd.DataFrame:
    path = _require_tables_dir(index_id) / filename
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    # reindex tolerates an older-schema file missing a column (e.g. an
    # evidence file predating a new field) -- missing cells read as NaN
    return frame.reindex(columns=columns)


def _save_csv(index_id: str, filename: str, frame: pd.DataFrame, columns: list,
              sort_by: list = None) -> None:
    """Write `frame` as `filename` with explicit sort keys and an atomic
    temp-file + os.replace write -- deterministic bytes and no partial files."""
    directory = tables_dir(index_id)
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame[columns]
    keys = sort_by if sort_by is not None else _SORT_KEYS.get(filename)
    if keys is None:  # cik-keyed evidence: numeric cik, then date
        by = ["_cik_num"] + (["date"] if "date" in frame.columns else [])
        frame = frame.assign(
            _cik_num=pd.to_numeric(frame["cik"], errors="coerce")
        ).sort_values(by, kind="mergesort").drop(columns=["_cik_num"])
    else:
        frame = frame.sort_values(keys, kind="mergesort")
    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f".{filename}.", suffix=".tmp"
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, directory / filename)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_ticker_intervals(index_id: str) -> pd.DataFrame:
    """`index_id, ticker, start_date, end_date_excl`.

    `end_date_excl` is NaN for an open interval.
    """
    return _load_csv(index_id, "ticker_intervals.csv", TICKER_INTERVALS_COLUMNS)


def save_ticker_intervals(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "ticker_intervals.csv", frame, TICKER_INTERVALS_COLUMNS)


def load_security_map(index_id: str) -> pd.DataFrame:
    """`index_id, ticker, start_date, security_id, evidence, reviewed`. `reviewed` is bool."""
    frame = _load_csv(index_id, "security_map.csv", SECURITY_MAP_COLUMNS)
    if len(frame):
        frame["reviewed"] = frame["reviewed"].astype(str).str.lower() == "true"
    return frame


def save_security_map(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "security_map.csv", frame, SECURITY_MAP_COLUMNS)


def load_security_master(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "security_master.csv", SECURITY_MASTER_COLUMNS)


def save_security_master(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "security_master.csv", frame, SECURITY_MASTER_COLUMNS)




def load_tail_changes(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "tail_changes.csv", TAIL_CHANGES_COLUMNS)


def save_tail_changes(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "tail_changes.csv", frame, TAIL_CHANGES_COLUMNS)


def load_manifest(index_id: str) -> dict:
    path = _require_tables_dir(index_id) / "manifest.json"
    if not path.exists():
        return {}
    with open(path) as handle:
        return json.load(handle)


def save_manifest(index_id: str, manifest: dict) -> None:
    directory = tables_dir(index_id)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_ticker_history(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "ticker_history.csv", TICKER_HISTORY_COLUMNS)


def save_ticker_history(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "ticker_history.csv", frame, TICKER_HISTORY_COLUMNS)


def load_cik_history(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "cik_history.csv", CIK_HISTORY_COLUMNS)


def save_cik_history(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "cik_history.csv", frame, CIK_HISTORY_COLUMNS)


def load_identity_evidence(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "identity_evidence.csv", IDENTITY_EVIDENCE_COLUMNS)


def save_identity_evidence(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "identity_evidence.csv", frame, IDENTITY_EVIDENCE_COLUMNS)


def load_cik_evidence(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "cik_evidence.csv", CIK_EVIDENCE_COLUMNS)


def save_cik_evidence(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "cik_evidence.csv", frame, CIK_EVIDENCE_COLUMNS)


def load_identity_review(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "identity_review.csv", IDENTITY_REVIEW_COLUMNS)


def save_identity_review(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "identity_review.csv", frame, IDENTITY_REVIEW_COLUMNS)


def load_identity_overrides(index_id: str) -> pd.DataFrame:
    """Human-authored overrides -- tools only ever READ this file."""
    return _load_csv(index_id, "identity_overrides.csv", IDENTITY_OVERRIDES_COLUMNS)


def load_cik_ticker_evidence(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "cik_ticker_evidence.csv", CIK_TICKER_EVIDENCE_COLUMNS)


def save_cik_ticker_evidence(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "cik_ticker_evidence.csv", frame, CIK_TICKER_EVIDENCE_COLUMNS)
