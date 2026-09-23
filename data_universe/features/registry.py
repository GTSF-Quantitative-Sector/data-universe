"""The feature registry: `@register` (rejects `fwd_`-prefixed names -- those
belong in `labels/registry.py`) and `compute()`, the shared lookup-pad-call-
slice logic reused by both `features/library.py` and (via import)
`labels/registry.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from data_universe import calendar
from data_universe.sources.base import DataSource

FORBIDDEN_PREFIX = "fwd_"


@dataclass
class FeatureSpec:
    fn: Callable
    warmup_trading_days: int
    project: str
    description: str


FEATURE_REGISTRY: dict = {}


def register(
    name: str,
    warmup_trading_days: int = 0,
    project: str = "",
    description: str = "",
    registry: Optional[dict] = None,
    forbidden_prefix: str = FORBIDDEN_PREFIX,
) -> Callable:
    """Decorator factory registering a feature function under `name`.

    Args:
        name: Feature name. Must NOT start with `forbidden_prefix` ("fwd_").
        warmup_trading_days: Trading days of history `compute()` fetches
            before the requested `start`, so the first returned value isn't
            NaN purely for lack of history.
        project: Free-text owning project tag (e.g. "beta", "vrp", "pead").
        description: Human-readable description.
        registry: The dict to register into. Defaults to `FEATURE_REGISTRY`.
        forbidden_prefix: Prefix that is not allowed for a feature name.

    Raises:
        ValueError: If `name` starts with `forbidden_prefix`, or if `name`
            is already registered.
    """
    target = FEATURE_REGISTRY if registry is None else registry

    def decorator(fn: Callable) -> Callable:
        if name.startswith(forbidden_prefix):
            raise ValueError(
                f"Feature name {name!r} must not start with {forbidden_prefix!r} "
                "-- that prefix is reserved for labels/registry.py."
            )
        if name in target:
            raise ValueError(f"{name!r} is already registered")
        target[name] = FeatureSpec(
            fn=fn,
            warmup_trading_days=warmup_trading_days,
            project=project,
            description=description,
        )
        return fn

    return decorator


def compute(
    registry: dict,
    name: str,
    ticker: str,
    start: str,
    end: str,
    source: DataSource,
    calendar_proxy: str = "SPY",
    **kwargs,
) -> pd.Series:
    """Look up `name` in `registry`, pad `start` backward by its declared
    warmup, call its function, and slice the result back to `[start, end]`.

    Args:
        registry: A registry dict as populated by `register()`.
        name: Registered feature/label name.
        ticker: Underlying ticker symbol.
        start: Requested start date, "YYYY-MM-DD".
        end: Requested end date, "YYYY-MM-DD".
        source: A `DataSource` passed through to the registered function.
        calendar_proxy: Market-proxy ticker used to derive the trading
            calendar for warmup padding (default "SPY").
        **kwargs: Passed through to the registered function.

    Returns:
        A `pd.Series` indexed by date, sliced to `[start, end]` inclusive.

    Raises:
        KeyError: If `name` is not registered.
    """
    if name not in registry:
        raise KeyError(f"{name!r} is not registered")
    spec = registry[name]
    padded_start = start
    if spec.warmup_trading_days > 0:
        calendar_start = pd.Timestamp(start) - pd.Timedelta(days=spec.warmup_trading_days * 3 + 20)
        session_index = calendar.sessions(
            source, calendar_start.strftime("%Y-%m-%d"), end, proxy_ticker=calendar_proxy
        )
        shifted = calendar.shift_sessions(
            session_index, pd.Timestamp(start), -spec.warmup_trading_days
        )
        if shifted is not None:
            padded_start = shifted.strftime("%Y-%m-%d")
    result = spec.fn(ticker, padded_start, end, source, **kwargs)
    return result.loc[(result.index >= pd.Timestamp(start)) & (result.index <= pd.Timestamp(end))]
