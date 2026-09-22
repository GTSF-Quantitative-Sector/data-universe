"""Global configuration for the data_universe package.

Precedence: environment variable > `config/config.yaml` > hardcoded default.
The Polygon API key is only ever held in memory and sent as an
`Authorization: Bearer` header by source modules -- never in a URL, never
logged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DEFAULT_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
_DEFAULT_CACHE_SIZE = 256
_DEFAULT_POLYGON_BASE_URL = "https://api.polygon.io"

# Maps internal state keys to the environment variable that seeds them.
_ENV_VARS = {
    "polygon_key": "POLYGON_API_KEY",
    "polygon_base_url": "POLYGON_BASE_URL",
    "polygon_flatfiles_endpoint": "POLYGON_FLATFILES_ENDPOINT",
    "polygon_s3_access_key": "POLYGON_S3_ACCESS_KEY",
    "polygon_s3_secret_key": "POLYGON_S3_SECRET_KEY",
    "fred_key": "FRED_API_KEY",
    "data_dir": "DATA_UNIVERSE_DATA_DIR",
    "cache_size": "DATA_UNIVERSE_CACHE_SIZE",
}

# Maps a config/config.yaml top-level key to the internal state key it feeds.
_YAML_KEYS = {
    "polygon_api_key": "polygon_key",
    "polygon_base_url": "polygon_base_url",
    "polygon_flatfiles_endpoint": "polygon_flatfiles_endpoint",
    "polygon_s3_access_key": "polygon_s3_access_key",
    "polygon_s3_secret_key": "polygon_s3_secret_key",
    "fred_api_key": "fred_key",
    "data_dir": "data_dir",
    "cache_size": "cache_size",
}

_state: dict = {}


def _default(state_key: str, default):
    """Read a state key's initial value from its environment variable, else `default`."""
    env_var = _ENV_VARS.get(state_key)
    if env_var is None:
        return default
    value = os.environ.get(env_var)
    if value is None:
        return default
    if state_key == "cache_size":
        return int(value)
    return value


def reset() -> None:
    """Reset all configuration to defaults (re-reading environment variables).

    Intended for tests; not part of normal usage.
    """
    _state.clear()
    _state["polygon_key"] = _default("polygon_key", None)
    _state["polygon_base_url"] = _default("polygon_base_url", _DEFAULT_POLYGON_BASE_URL)
    _state["polygon_flatfiles_endpoint"] = _default("polygon_flatfiles_endpoint", None)
    _state["polygon_s3_access_key"] = _default("polygon_s3_access_key", None)
    _state["polygon_s3_secret_key"] = _default("polygon_s3_secret_key", None)
    _state["fred_key"] = _default("fred_key", None)
    _state["data_dir"] = _default("data_dir", _DEFAULT_DATA_DIR)
    _state["cache_size"] = _default("cache_size", _DEFAULT_CACHE_SIZE)
    _state["project"] = None


def set_polygon_key(key: str) -> None:
    """Set the Polygon.io API key. Sent as an `Authorization: Bearer` header, never a URL param.

    Args:
        key: Polygon.io API key.
    """
    _state["polygon_key"] = key


def get_polygon_key() -> Optional[str]:
    """Return the configured Polygon.io API key, or None if unset."""
    return _state["polygon_key"]


def set_polygon_base_url(url: str) -> None:
    """Set the Polygon REST base URL (default `https://api.polygon.io`).

    Args:
        url: Base URL, no trailing slash.
    """
    _state["polygon_base_url"] = url


def get_polygon_base_url() -> str:
    """Return the configured Polygon REST base URL."""
    return _state["polygon_base_url"]


def set_polygon_flatfiles_endpoint(url: str) -> None:
    """Set the S3-compatible flat-files endpoint for bulk options history.

    Args:
        url: S3-compatible endpoint URL.
    """
    _state["polygon_flatfiles_endpoint"] = url


def get_polygon_flatfiles_endpoint() -> Optional[str]:
    """Return the configured flat-files endpoint, or None if unset."""
    return _state["polygon_flatfiles_endpoint"]


def set_polygon_s3_access_key(key: str) -> None:
    """Set the S3 access key used for Polygon flat-files downloads."""
    _state["polygon_s3_access_key"] = key


def get_polygon_s3_access_key() -> Optional[str]:
    """Return the configured S3 access key, or None if unset."""
    return _state["polygon_s3_access_key"]


def set_polygon_s3_secret_key(key: str) -> None:
    """Set the S3 secret key used for Polygon flat-files downloads."""
    _state["polygon_s3_secret_key"] = key


def get_polygon_s3_secret_key() -> Optional[str]:
    """Return the configured S3 secret key, or None if unset."""
    return _state["polygon_s3_secret_key"]


def set_fred_key(key: str) -> None:
    """Set the FRED API key used to fetch the risk-free rate series."""
    _state["fred_key"] = key


def get_fred_key() -> Optional[str]:
    """Return the configured FRED API key, or None if unset."""
    return _state["fred_key"]


def set_project(name: str) -> None:
    """Tag subsequent cache accesses with a project name, e.g. "vrp-research", "pead", "beta".

    Args:
        name: Project identifier.
    """
    _state["project"] = name


def get_project() -> Optional[str]:
    """Return the current project tag, or None if unset."""
    return _state["project"]


def set_data_dir(path: str) -> None:
    """Set the root directory for raw/interim/processed data and cache logs."""
    _state["data_dir"] = path


def get_data_dir() -> str:
    """Return the configured data directory."""
    return _state["data_dir"]


def set_cache_size(n: int) -> None:
    """Set the shared LRU cache capacity (number of entries) used by `data_universe.cache`."""
    _state["cache_size"] = n


def get_cache_size() -> int:
    """Return the configured cache capacity."""
    return _state["cache_size"]


def load_yaml(path: str = "config/config.yaml") -> None:
    """Load settings from a YAML file. Environment variables still take precedence:
    a key present in both the YAML file and the environment keeps its environment value.

    Args:
        path: Path to a YAML file with a flat map of setting names (see
            `config/config.example.yaml`) to values. A missing file is a no-op.
    """
    import yaml

    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        data = yaml.safe_load(f) or {}
    for yaml_key, state_key in _YAML_KEYS.items():
        if yaml_key not in data:
            continue
        env_var = _ENV_VARS.get(state_key)
        if env_var and os.environ.get(env_var) is not None:
            continue  # environment variable wins
        value = data[yaml_key]
        _state[state_key] = int(value) if state_key == "cache_size" else value


reset()
