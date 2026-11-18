"""The label registry. Reuses `features.registry.compute()` verbatim (same
pad-call-slice logic) but enforces the opposite naming rule: every label
name MUST start with `fwd_`, signaling at a glance (and at import time) that
it looks forward and must never be used as a model input feature.
"""
from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from data_universe.features.registry import FeatureSpec
from data_universe.features.registry import compute as _shared_compute
from data_universe.sources.base import DataSource

REQUIRED_PREFIX = "fwd_"

LABEL_REGISTRY: dict = {}


def register(
    name: str,
    warmup_trading_days: int = 0,
    project: str = "",
    description: str = "",
    registry: Optional[dict] = None,
    required_prefix: str = REQUIRED_PREFIX,
    version: int = 1,
    horizon_trading_days: Optional[int] = None,
) -> Callable:
    """Decorator factory registering a label function under `name`.

    Args:
        name: Label name. Must start with `required_prefix` ("fwd_").
        warmup_trading_days: See `features.registry.register`.
        project: Free-text owning project tag.
        description: Human-readable description.
        registry: The dict to register into. Defaults to `LABEL_REGISTRY`.
        required_prefix: Prefix every label name must start with.
        version: Code version stamped into precompute fingerprints -- bump
            whenever the formula (or anything it calls) changes.
        horizon_trading_days: Trading days the label looks forward; precompute
            only writes rows whose full horizon is final. None means the
            label is not precompute-safe.

    Raises:
        ValueError: If `name` doesn't start with `required_prefix`, if
            `name` is already registered, or if `version`/`horizon_trading_days`
            fail validation.
    """
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"version must be an int >= 1, got {version!r}")
    if horizon_trading_days is not None and (
        not isinstance(horizon_trading_days, int)
        or isinstance(horizon_trading_days, bool)
        or horizon_trading_days < 0
    ):
        raise ValueError(
            "horizon_trading_days must be None or an int >= 0, "
            f"got {horizon_trading_days!r}"
        )
    target = LABEL_REGISTRY if registry is None else registry

    def decorator(fn: Callable) -> Callable:
        if not name.startswith(required_prefix):
            raise ValueError(
                f"Label name {name!r} must start with {required_prefix!r} "
                "-- non-fwd_ names belong in features/registry.py."
            )
        if name in target:
            raise ValueError(f"{name!r} is already registered")
        target[name] = FeatureSpec(
            fn=fn,
            warmup_trading_days=warmup_trading_days,
            project=project,
            description=description,
            version=version,
            horizon_trading_days=horizon_trading_days,
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
    """Thin wrapper around `features.registry.compute` -- identical
    pad-call-slice behavior, just documented under the label vocabulary.
    """
    return _shared_compute(registry, name, ticker, start, end, source, calendar_proxy, **kwargs)
