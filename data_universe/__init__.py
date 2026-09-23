"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (Ticker,
precompute) on top of Phase 6's feature/label registries.
"""
import data_universe.features.library  # noqa: F401  (populates FEATURE_REGISTRY)
import data_universe.labels.library  # noqa: F401  (populates LABEL_REGISTRY)
from data_universe import config
from data_universe.cache import get_cache, simulate
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.labels.registry import LABEL_REGISTRY
from data_universe.precompute import load_feature, precompute
from data_universe.sources import sec_adapter as universe

__version__ = "0.1.0"

__all__ = [
    "config",
    "get_cache",
    "simulate",
    "universe",
    "FEATURE_REGISTRY",
    "LABEL_REGISTRY",
    "precompute",
    "load_feature",
    "__version__",
]
