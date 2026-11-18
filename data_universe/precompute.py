"""Batch-compute registered features/labels and generic datasets into
versioned, per-key derived parquet stores.

Layout (see `store.derived.DerivedStore`): `<out_dir>/<name>/<fingerprint>/`
holds one `dataset.json` manifest plus one `<key>.parquet` /
`<key>.coverage.json` pair per key. The fingerprint is a hash of the spec --
code version, explicit parameters, source family, bar conventions -- so a
formula or input change lands in a new version directory instead of mixing
with old rows.

Run semantics per (name, key): the final cap C is the last date whose full
horizon is provably complete (`_last_complete_date()` for features,
`horizon_trading_days`/`final_lag_sessions` sessions earlier for
labels/datasets). Only gaps inside [start, C] not already in coverage are
computed (or all of it with `force=True`); rows past M = min(end, L) are a
provisional tail -- written but never covered, so a later run recomputes
exactly them. `replace_range` is the tombstone mechanism: recomputing a
range drops every stored row in it first, so values that turn NaN disappear.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data_universe import calendar, config
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.features.registry import compute as _compute
from data_universe.labels.registry import LABEL_REGISTRY
from data_universe.sources import adjust
from data_universe.sources.base import NY_TZ, DataSource
from data_universe.sources.polygon import PolygonSource
from data_universe.store import source as _store_source
from data_universe.store.derived import DERIVED_SCHEMA_VERSION, DerivedStore


@dataclass
class PrecomputeReport:
    """Per-(key, name) outcomes of a `precompute`/`precompute_dataset` run.

    `succeeded` counts rows written this run; `failed` maps to the error;
    `skipped` means nothing was computed (already covered / no final dates);
    `versions` records the fingerprint each name resolved to.
    """

    succeeded: Dict[Tuple[str, str], int] = field(default_factory=dict)
    failed: Dict[Tuple[str, str], str] = field(default_factory=dict)
    skipped: Dict[Tuple[str, str], str] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)


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


def _tool_version() -> str:
    import data_universe

    return data_universe.__version__


def _output_dir(out_dir: Optional[str]) -> Path:
    return Path(out_dir) if out_dir is not None else Path(config.get_data_dir()) / "processed"


# -- fingerprints -----------------------------------------------------------


def canonical(value):
    """JSON-safe canonical form of a parameter value for fingerprinting.

    Scalars pass through (non-finite floats raise); dates become ISO
    strings; mappings/lists/sets recurse; functions and classes become
    ``{"callable": ...}``; objects exposing ``precompute_fingerprint()``
    become ``{"type": ..., "fingerprint": ...}``; anything else becomes
    ``{"type": ...}`` -- other attributes are NEVER read (sources hold keys).
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite float {value!r} cannot be fingerprinted")
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, (set, frozenset)):
        items = [canonical(v) for v in value]
        items.sort(
            key=lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        return items
    if inspect.isroutine(value) or inspect.isclass(value):
        return {"callable": f"{value.__module__}.{value.__qualname__}"}
    fingerprint_fn = getattr(value, "precompute_fingerprint", None)
    if callable(fingerprint_fn):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fingerprint": canonical(fingerprint_fn()),
        }
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _fingerprint(spec: dict) -> str:
    blob = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _source_family(source) -> Optional[str]:
    """Vendor family tag for fingerprinting: `StoreBackedSource` reports its
    upstream's family; other sources use `store_family` or their class name."""
    if source is None:
        return None
    if isinstance(source, _store_source.StoreBackedSource):
        return source.family
    return getattr(source, "store_family", None) or type(source).__name__


def _explicit_params(fn: Callable, merged: dict) -> dict:
    """The function's explicit keyword parameters (everything after the
    first four positional `ticker, start, end, source`, minus *args/**kwargs)
    with effective values: the passed value when present in `merged`, else
    the default; parameters without a default that weren't passed are omitted."""
    params = {}
    for pname, param in list(inspect.signature(fn).parameters.items())[4:]:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        if pname in merged:
            params[pname] = canonical(merged[pname])
        elif param.default is not inspect.Parameter.empty:
            params[pname] = canonical(param.default)
    return params


def _registry_spec(name: str, registry: dict, kind: str, source, merged: dict) -> dict:
    spec_obj = registry[name]
    return {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "name": name,
        "kind": kind,
        "code_version": spec_obj.version,
        "horizon_trading_days": (
            0 if kind == "feature" else spec_obj.horizon_trading_days
        ),
        "source_family": _source_family(source),
        "conventions": {
            "bars": adjust.ADJUSTMENT_CONVENTION,
            "dividend_cash_basis": adjust.DIVIDEND_CASH_BASIS,
        },
        "params": _explicit_params(spec_obj.fn, merged),
    }


def _dataset_spec(
    name: str,
    version: int,
    params: Optional[dict],
    source,
    final_lag_sessions: int,
    key_column: str,
    date_column: str,
) -> dict:
    return {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "name": name,
        "kind": "dataset",
        "code_version": version,
        "params": canonical(params or {}),
        "source_family": _source_family(source),
        "final_lag_sessions": final_lag_sessions,
        "key_column": key_column,
        "date_column": date_column,
    }


def _manifest(name: str, kind: str, fingerprint: str, spec: dict,
              key_column: str, date_column: str) -> dict:
    return {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "name": name,
        "kind": kind,
        "fingerprint": fingerprint,
        "spec": spec,
        "key_column": key_column,
        "date_column": date_column,
        "columns": None,
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "tool_version": _tool_version(),
    }


# -- result normalization ----------------------------------------------------


def _norm_index_dates(index) -> pd.DatetimeIndex:
    dates = pd.DatetimeIndex(index)
    if dates.tz is not None:
        dates = dates.tz_convert(NY_TZ).tz_localize(None)
    return dates.normalize()


def _norm_column_dates(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values)
    if getattr(dates.dt, "tz", None) is not None:
        dates = dates.dt.tz_convert(NY_TZ).dt.tz_localize(None)
    return dates.dt.normalize()


def _normalize_result(
    result,
    *,
    is_registry: bool,
    key: str,
    key_column: str,
    date_column: str,
    a: pd.Timestamp,
    b: pd.Timestamp,
) -> pd.DataFrame:
    """Coerce a feature/label/builder result into the stored column order
    `[date_column, key_column, *value columns]`, dates normalized, rows
    outside [a, b] dropped. Registry results drop all-NaN rows; generic
    results keep NaNs and must carry `date_column`."""
    if isinstance(result, pd.Series):
        frame = result.rename("value").to_frame()
        value_cols = ["value"]
        dates_from_index = True
    elif isinstance(result, pd.DataFrame):
        frame = result
        if is_registry:
            clash = [c for c in frame.columns if c in (date_column, key_column)]
            if clash:
                raise ValueError(
                    f"feature returned columns {clash} clashing with the stored "
                    f"{date_column!r}/{key_column!r} columns"
                )
            value_cols = list(frame.columns)
            dates_from_index = True
        else:
            if len(frame) and date_column not in frame.columns:
                raise ValueError(
                    f"dataset builder must return a {date_column!r} column "
                    "when the result is non-empty"
                )
            value_cols = [
                c for c in frame.columns if c not in (date_column, key_column)
            ]
            dates_from_index = date_column not in frame.columns
    else:
        raise TypeError(
            f"expected a pandas Series/DataFrame, got {type(result).__name__}"
        )

    n = len(frame)
    out = pd.DataFrame(index=range(n))
    if dates_from_index:
        out[date_column] = _norm_index_dates(frame.index)
    else:
        # positional assignment: the builder's frame may carry any index
        out[date_column] = _norm_column_dates(frame[date_column]).to_numpy()
        if key_column in frame.columns:
            supplied = frame[key_column]
            if len(supplied) and not bool((supplied == key).all()):
                raise ValueError(
                    f"builder returned key_column {key_column!r} values "
                    f"{sorted(set(supplied))} but was called for key {key!r}"
                )
    out[key_column] = key
    for col in value_cols:
        out[col] = frame[col].to_numpy() if n else np.nan
    out = out[[date_column, key_column, *value_cols]]

    if is_registry and value_cols:
        out = out.dropna(subset=value_cols, how="all")
    if len(out):
        out[date_column] = pd.to_datetime(out[date_column])
        mask = (out[date_column] >= a) & (out[date_column] <= b)
        out = out.loc[mask].reset_index(drop=True)
    else:
        out[date_column] = pd.to_datetime(out[date_column])
    return out


# -- shared run loop -----------------------------------------------------------


def _execute(
    plans: List[dict],
    source,
    start,
    end,
    force: bool,
    out_dir: Optional[str],
    report: PrecomputeReport,
) -> PrecomputeReport:
    """Run every (name, key) pair in `plans` against `store`, isolating
    failures per pair. `start`/`end` are already-validated Timestamps."""
    store = DerivedStore(out_dir)
    last_complete = _store_source._last_complete_date()
    m = min(end, last_complete)
    sessions = None
    sessions_loaded = False

    def _calendar() -> pd.DatetimeIndex:
        nonlocal sessions, sessions_loaded
        if not sessions_loaded:
            sessions_loaded = True
            if m >= start:
                sessions = calendar.sessions(
                    source, start.strftime("%Y-%m-%d"), m.strftime("%Y-%m-%d")
                )
            else:
                sessions = pd.DatetimeIndex([])
        return sessions

    ensured = set()
    for plan in plans:
        name = plan["name"]
        fingerprint = plan["fingerprint"]
        horizon = plan["horizon"]
        report.versions[name] = fingerprint

        if horizon == 0:
            cap = m
        else:
            sess = _calendar()
            cap = sess[-1 - horizon] if len(sess) > horizon else None
        final_nonempty = cap is not None and cap >= start

        for key in plan["keys"]:
            try:
                coverage = store.coverage(name, fingerprint, key)
                if force or not final_nonempty:
                    gaps = [(start, cap)] if final_nonempty else []
                else:
                    gaps = _store_source._missing_intervals(coverage, start, cap)
                if gaps:
                    a = min(g[0] for g in gaps)
                    b = max(g[1] for g in gaps)
                else:
                    a = b = None

                provisional = horizon == 0 and end > m
                if provisional:
                    if a is None:
                        a = max(start, m + pd.Timedelta(days=1))
                    b = end

                if a is None:
                    report.skipped[(key, name)] = (
                        "up to date" if final_nonempty else "no final dates in window"
                    )
                    continue

                if horizon == 0:
                    compute_end = b
                else:
                    sess = _calendar()
                    pos = sess.searchsorted(b, side="right")
                    if pos == 0:
                        compute_end = b
                    else:
                        idx = pos - 1 + horizon
                        compute_end = sess[idx] if idx < len(sess) else b

                a_str = a.strftime("%Y-%m-%d")
                d_str = compute_end.strftime("%Y-%m-%d")
                result = plan["compute"](key, a_str, d_str)
                rows = _normalize_result(
                    result,
                    is_registry=plan["is_registry"],
                    key=key,
                    key_column=plan["key_column"],
                    date_column=plan["date_column"],
                    a=a,
                    b=b,
                )

                record = None
                if cap is not None:
                    record_end = min(b, cap)
                    if a <= record_end:
                        covered = rows[
                            (rows[plan["date_column"]] >= a)
                            & (rows[plan["date_column"]] <= record_end)
                        ]
                        record = {
                            "start": a.strftime("%Y-%m-%d"),
                            "end": record_end.strftime("%Y-%m-%d"),
                            "rows": int(len(covered)),
                            "computed_at": pd.Timestamp.now(tz="UTC").isoformat(),
                            "tool_version": _tool_version(),
                            "source": _source_family(source),
                        }

                with store.lock(name, fingerprint, key):
                    if (name, fingerprint) not in ensured:
                        store.ensure_dataset(name, fingerprint, plan["manifest"])
                        ensured.add((name, fingerprint))
                    if len(rows):
                        store.set_columns(
                            name, fingerprint, list(rows.columns)
                        )
                    store.replace_range(
                        name,
                        fingerprint,
                        key,
                        a,
                        b,
                        rows,
                        record,
                        date_column=plan["date_column"],
                    )
                report.succeeded[(key, name)] = len(rows)
            except Exception as exc:  # noqa: BLE001 - isolate any failure, report it
                report.failed[(key, name)] = str(exc)
    return report


# -- public API ---------------------------------------------------------------


def precompute(
    tickers: List[str],
    feature_names: List[str],
    start: str,
    end: str,
    out_dir: Optional[str] = None,
    source: Optional[DataSource] = None,
    *,
    feature_kwargs: Optional[Dict[str, dict]] = None,
    force: bool = False,
    **kwargs,
) -> PrecomputeReport:
    """Compute every (feature_name, ticker) pair into the versioned derived
    store under `out_dir` (default `<config.get_data_dir()>/processed`).

    Only uncovered gaps inside the final window are computed; covered pairs
    land in `report.skipped`. `force=True` recomputes the whole final window
    (the tombstone mechanism for corrected inputs). `feature_kwargs` maps a
    feature name to extra kwargs for that feature only; `**kwargs` reach
    every feature as before (e.g. `rate_source=FredSource()`).
    """
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if s > e:
        raise ValueError(f"start {start!r} is after end {end!r}")
    tickers = [tickers] if isinstance(tickers, str) else list(tickers)
    names = [feature_names] if isinstance(feature_names, str) else list(feature_names)
    tickers = list(dict.fromkeys(tickers))
    names = list(dict.fromkeys(names))
    feature_kwargs = feature_kwargs or {}
    unknown = set(feature_kwargs) - set(names)
    if unknown:
        raise ValueError(
            f"feature_kwargs keys {sorted(unknown)} are not in feature_names"
        )
    active_source = source if source is not None else PolygonSource()
    report = PrecomputeReport()
    plans = []

    for name in names:
        try:
            registry = _resolve_registry(name)
        except KeyError as exc:
            for ticker in tickers:
                report.failed[(ticker, name)] = str(exc)
            continue
        kind = "feature" if registry is FEATURE_REGISTRY else "label"
        spec_obj = registry[name]
        if kind == "label" and spec_obj.horizon_trading_days is None:
            for ticker in tickers:
                report.failed[(ticker, name)] = (
                    f"{name!r} is a label without horizon_trading_days -- "
                    "register it with horizon_trading_days=N to precompute it"
                )
            continue
        merged = {**kwargs, **feature_kwargs.get(name, {})}
        spec_dict = _registry_spec(name, registry, kind, active_source, merged)
        fingerprint = _fingerprint(spec_dict)
        key_column, date_column = "ticker", "date"
        plans.append(
            {
                "name": name,
                "kind": kind,
                "fingerprint": fingerprint,
                "horizon": 0 if kind == "feature" else spec_obj.horizon_trading_days,
                "keys": tickers,
                "key_column": key_column,
                "date_column": date_column,
                "is_registry": True,
                "manifest": _manifest(name, kind, fingerprint, spec_dict,
                                      key_column, date_column),
                "compute": lambda key, a, d, _n=name, _r=registry, _m=merged: _compute(
                    _r, _n, key, a, d, active_source, **_m
                ),
            }
        )

    return _execute(plans, active_source, s, e, force, out_dir, report)


def precompute_dataset(
    name: str,
    builder: Callable,
    keys: List[str],
    start: str,
    end: str,
    *,
    version: int = 1,
    params: Optional[dict] = None,
    final_lag_sessions: int = 0,
    source=None,
    key_column: str = "key",
    date_column: str = "date",
    out_dir: Optional[str] = None,
    force: bool = False,
) -> PrecomputeReport:
    """Compute a generic (non-registry) dataset: `builder(key, start, end)`
    returns a frame carrying `date_column` (and optionally `key_column`).

    `final_lag_sessions` is the maturity lag used for the final cap (0 means
    every row up to the last complete date is final). `version` and `params`
    feed the fingerprint; `source` only contributes its family tag (and the
    session calendar when `final_lag_sessions > 0`, which requires one).
    """
    s, e = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    if s > e:
        raise ValueError(f"start {start!r} is after end {end!r}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(name)):
        raise ValueError(
            f"dataset name {name!r} must match [A-Za-z0-9][A-Za-z0-9_.-]* "
            "-- it becomes a directory name"
        )
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"version must be an int >= 1, got {version!r}")
    if name in FEATURE_REGISTRY or name in LABEL_REGISTRY:
        raise ValueError(
            f"{name!r} is a registered feature/label -- use precompute()"
        )
    if final_lag_sessions < 0:
        raise ValueError("final_lag_sessions must be >= 0")
    if final_lag_sessions > 0 and source is None:
        raise ValueError(
            "final_lag_sessions > 0 needs a source for the session calendar"
        )
    keys = [keys] if isinstance(keys, str) else list(keys)
    spec_dict = _dataset_spec(
        name, version, params, source, final_lag_sessions, key_column, date_column
    )
    fingerprint = _fingerprint(spec_dict)
    plan = {
        "name": name,
        "kind": "dataset",
        "fingerprint": fingerprint,
        "horizon": final_lag_sessions,
        "keys": list(dict.fromkeys(keys)),
        "key_column": key_column,
        "date_column": date_column,
        "is_registry": False,
        "manifest": _manifest(name, "dataset", fingerprint, spec_dict,
                              key_column, date_column),
        "compute": lambda key, a, d: builder(key, a, d),
    }
    report = PrecomputeReport()
    return _execute([plan], source, s, e, force, out_dir, report)


# -- loading ------------------------------------------------------------------


def _current_registry_spec(name: str) -> Optional[dict]:
    """The spec fields a stored version must match to count as 'current'
    for a registered name: code_version, horizon, conventions."""
    if name in FEATURE_REGISTRY:
        registry, kind = FEATURE_REGISTRY, "feature"
    elif name in LABEL_REGISTRY:
        registry, kind = LABEL_REGISTRY, "label"
    else:
        return None
    spec_obj = registry[name]
    return {
        "code_version": spec_obj.version,
        "horizon_trading_days": (
            0 if kind == "feature" else spec_obj.horizon_trading_days
        ),
        "conventions": {
            "bars": adjust.ADJUSTMENT_CONVENTION,
            "dividend_cash_basis": adjust.DIVIDEND_CASH_BASIS,
        },
    }


def _spec_matches(
    stored_spec: dict,
    *,
    current: Optional[dict] = None,
    source=None,
    params: Optional[dict] = None,
    code_version=None,
) -> bool:
    if current is not None:
        for field_name, expected in current.items():
            if stored_spec.get(field_name) != expected:
                return False
    if source is not None and stored_spec.get("source_family") != _source_family(source):
        return False
    if params:
        stored_params = stored_spec.get("params") or {}
        for pname, pvalue in params.items():
            if pname not in stored_params or stored_params[pname] != canonical(pvalue):
                return False
    if code_version is not None and stored_spec.get("code_version") != code_version:
        return False
    return True


def _version_listing(versions: List[dict]) -> str:
    if not versions:
        return ""
    lines = ["Available versions:"]
    for manifest in versions:
        spec = manifest.get("spec") or {}
        lines.append(
            f"  fingerprint={manifest.get('fingerprint')} "
            f"code_version={spec.get('code_version')} "
            f"source_family={spec.get('source_family')} "
            f"params={json.dumps(spec.get('params'), sort_keys=True)}"
        )
    return "\n".join(lines) + "\n"


def _select_version(
    store: DerivedStore,
    name: str,
    *,
    version: Optional[str] = None,
    source=None,
    params: Optional[dict] = None,
    current: Optional[dict] = None,
    code_version=None,
) -> dict:
    """Pick the single dataset.json manifest matching the requested filters."""
    all_versions = store.versions(name)
    candidates = [
        v for v in all_versions if v.get("schema_version") == DERIVED_SCHEMA_VERSION
    ]
    if version is not None:
        match = [v for v in candidates if v.get("fingerprint") == version]
        if not match:
            raise FileNotFoundError(
                f"No version {version!r} of {name!r}.\n"
                + _version_listing(all_versions)
                + "Pass version=<fingerprint> to select one explicitly."
            )
        return match[0]
    filtered = [
        v
        for v in candidates
        if _spec_matches(
            v.get("spec") or {},
            current=current,
            source=source,
            params=params,
            code_version=code_version,
        )
    ]
    if not filtered:
        note = (
            "Only stale or incompatible versions exist -- rerun precompute() "
            "to write a current one, or pass version=<fingerprint>."
            if candidates
            else "Run precompute()/precompute_dataset() first."
        )
        raise FileNotFoundError(
            f"No compatible precomputed data for {name!r}.\n"
            + _version_listing(all_versions)
            + note
        )
    if len(filtered) > 1:
        fingerprints = [v.get("fingerprint") for v in filtered]
        raise ValueError(
            f"Multiple precomputed versions of {name!r} match "
            f"({fingerprints}) -- pass version=, source= or params to "
            "disambiguate."
        )
    return filtered[0]


def _load_selected(
    store: DerivedStore,
    manifest: dict,
    *,
    keys_filter: Optional[List[str]],
    start=None,
    end=None,
) -> pd.DataFrame:
    name = manifest["name"]
    fingerprint = manifest["fingerprint"]
    date_column = manifest.get("date_column") or "date"
    key_column = manifest.get("key_column") or "ticker"
    wanted = set(keys_filter) if keys_filter is not None else None
    frames = []
    for key in store.keys(name, fingerprint):
        if wanted is not None and key not in wanted:
            continue
        frame = store.read(name, fingerprint, key)
        if frame is not None and len(frame):
            frames.append(frame)
    stored_columns = manifest.get("columns") or [date_column, key_column]
    if frames:
        out = pd.concat(frames, ignore_index=True)
        if len(out):
            dates = pd.to_datetime(out[date_column])
            if start is not None:
                out = out[dates >= pd.Timestamp(start)]
                dates = dates.loc[out.index]
            if end is not None:
                out = out[dates <= pd.Timestamp(end)]
            out = out.sort_values([date_column, key_column]).reset_index(drop=True)
            rest = [c for c in out.columns if c not in (date_column, key_column)]
            out = out[[date_column, key_column, *rest]]
        return out
    return pd.DataFrame(columns=stored_columns)


def load_feature(
    name: str,
    tickers: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    *,
    start=None,
    end=None,
    version: Optional[str] = None,
    source=None,
    **params,
) -> pd.DataFrame:
    """Read back precomputed rows for a registered feature/label.

    Selects the version matching the current code version, horizon and bar
    conventions (plus `source=`/`version=`/`**params` filters when given).
    Falls back to a legacy unversioned `<name>.parquet` (pre-v0.13 layout)
    with a warning when no versioned data exists and no filters were given.
    """
    store = DerivedStore(out_dir)
    registered = name in FEATURE_REGISTRY or name in LABEL_REGISTRY
    filters_given = version is not None or source is not None or bool(params)
    if isinstance(tickers, str):
        tickers = [tickers]
    try:
        manifest = _select_version(
            store,
            name,
            version=version,
            source=source,
            params=params or None,
            current=_current_registry_spec(name) if registered else None,
        )
    except FileNotFoundError:
        if not filters_given and not store.versions(name):
            legacy = store.root / f"{name}.parquet"
            if legacy.exists():
                warnings.warn(
                    f"{name!r} is unversioned pre-v0.13 precompute output -- "
                    "recompute with precompute() to get a versioned dataset.",
                    UserWarning,
                    stacklevel=2,
                )
                frame = pd.read_parquet(legacy)
                if tickers is not None:
                    frame = frame[frame["ticker"].isin(tickers)]
                dates = pd.to_datetime(frame["date"])
                if start is not None:
                    frame = frame[dates >= pd.Timestamp(start)]
                    dates = dates.loc[frame.index]
                if end is not None:
                    frame = frame[dates <= pd.Timestamp(end)]
                return frame.sort_values(["date", "ticker"]).reset_index(drop=True)
        raise
    return _load_selected(
        store, manifest, keys_filter=tickers, start=start, end=end
    )


def load_dataset(
    name: str,
    keys: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    *,
    start=None,
    end=None,
    version: Optional[str] = None,
    code_version=None,
    source=None,
    **params,
) -> pd.DataFrame:
    """Read back a generic dataset written by `precompute_dataset`.

    Selects the version matching `version=` (fingerprint), `code_version=`,
    `source=` (source family) and `**params`; raises
    FileNotFoundError/ValueError when zero or several versions match.
    """
    store = DerivedStore(out_dir)
    if isinstance(keys, str):
        keys = [keys]
    manifest = _select_version(
        store,
        name,
        version=version,
        source=source,
        params=params or None,
        code_version=code_version,
    )
    return _load_selected(store, manifest, keys_filter=keys, start=start, end=end)


def invalidate(
    name: str,
    keys: Optional[List[str]] = None,
    start=None,
    end=None,
    version: Optional[str] = None,
    out_dir: Optional[str] = None,
) -> int:
    """Drop derived data for `name`. With no keys/dates/version, removes every
    version dir plus the legacy `<name>.parquet`; with only `version`, that
    version dir; otherwise invalidates the (version, key) pairs in range.
    Returns the count of (version, key) pairs changed/removed."""
    store = DerivedStore(out_dir)
    if isinstance(keys, str):
        keys = [keys]
    manifests = store.versions(name)
    if version is not None:
        manifests = [m for m in manifests if m.get("fingerprint") == version]

    if keys is None and start is None and end is None:
        removed = 0
        for manifest in manifests:
            fingerprint = manifest["fingerprint"]
            removed += len(store.keys(name, fingerprint))
            store.remove_version(name, fingerprint)
        if version is None:
            legacy = store.root / f"{name}.parquet"
            legacy_lock = store.root / f".{name}.parquet.lock"
            if legacy.exists():
                legacy.unlink()
                removed += 1
            if legacy_lock.exists():
                legacy_lock.unlink()
        return removed

    changed = 0
    wanted = set(keys) if keys is not None else None
    for manifest in manifests:
        fingerprint = manifest["fingerprint"]
        key_list = sorted(wanted) if wanted is not None else store.keys(name, fingerprint)
        for key in key_list:
            if store.invalidate(name, fingerprint, key, start, end):
                changed += 1
    return changed


def precompute_status(out_dir: Optional[str] = None) -> pd.DataFrame:
    """`DerivedStore.status()` plus `code_version`, `source_family`, `params`
    (JSON) and `current` -- True/False for registered names (spec matches the
    live registry), None for generic datasets or unregistered names."""
    store = DerivedStore(out_dir)
    status = store.status()
    for extra in ("code_version", "source_family", "params", "current"):
        status[extra] = None
    for idx, row in status.iterrows():
        manifest = store.dataset_manifest(row["name"], row["fingerprint"])
        spec = (manifest or {}).get("spec") or {}
        status.at[idx, "code_version"] = spec.get("code_version")
        status.at[idx, "source_family"] = spec.get("source_family")
        status.at[idx, "params"] = json.dumps(spec.get("params"), sort_keys=True)
        current_spec = _current_registry_spec(row["name"])
        if current_spec is not None:
            status.at[idx, "current"] = all(
                spec.get(field_name) == expected
                for field_name, expected in current_spec.items()
            )
    return status
