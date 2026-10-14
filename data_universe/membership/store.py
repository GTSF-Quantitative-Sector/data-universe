"""File IO for data_universe.membership's committed reference tables.

This is the only module in the package that opens a file. `intervals.py`,
`grid.py` and `identity.py` call into this module and never touch a path
themselves -- that split keeps every file-format decision (CSV dtypes, the
manifest's JSON shape, where `tables_dir()` resolves to) in one place.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_universe import config
from data_universe.membership.errors import MembershipError
from data_universe.membership.schema import (
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TAIL_CHANGES_COLUMNS,
    TICKER_HISTORY_COLUMNS,
    TICKER_INTERVALS_COLUMNS,
)

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
    return frame[columns]


def _save_csv(index_id: str, filename: str, frame: pd.DataFrame, columns: list) -> None:
    directory = tables_dir(index_id)
    directory.mkdir(parents=True, exist_ok=True)
    frame[columns].sort_values(columns[:2]).to_csv(directory / filename, index=False)


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


def load_ticker_history(index_id: str) -> pd.DataFrame:
    return _load_csv(index_id, "ticker_history.csv", TICKER_HISTORY_COLUMNS)


def save_ticker_history(index_id: str, frame: pd.DataFrame) -> None:
    _save_csv(index_id, "ticker_history.csv", frame, TICKER_HISTORY_COLUMNS)


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
