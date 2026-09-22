"""GTSF Quant Sector shared data and feature library.

Public API is re-exported here as later phases add modules (cache, Ticker,
precompute, feature/label registries). This phase only exposes `config` and
`__version__`.
"""
from data_universe import config

__version__ = "0.1.0"

__all__ = ["config", "__version__"]
