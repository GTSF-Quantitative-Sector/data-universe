"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (Ticker,
precompute, feature/label registries). This phase adds `universe` (the
optional sec_parser adapter) on top of Phase 3's `get_cache`/`simulate`.
"""
from data_universe import config
from data_universe.cache import get_cache, simulate
from data_universe.sources import sec_adapter as universe

__version__ = "0.1.0"

__all__ = ["config", "get_cache", "simulate", "universe", "__version__"]
