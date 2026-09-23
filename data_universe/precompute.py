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
