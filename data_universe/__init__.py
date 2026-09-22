"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (Ticker,
precompute, feature/label registries). This phase adds `get_cache` and
`simulate` on top of Phase 2's `config`.
"""
from data_universe import config
from data_universe.cache import get_cache, simulate

__version__ = "0.1.0"

__all__ = ["config", "get_cache", "simulate", "__version__"]
