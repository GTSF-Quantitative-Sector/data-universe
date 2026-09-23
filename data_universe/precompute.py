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
        A failure for one pair never stops the rest of the batch. Rows for existing
        `(date, ticker)` keys already on disk are upserted, never duplicated.
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
