import pytest


@pytest.fixture(autouse=True)
def _strip_disk_cache_env_vars(monkeypatch):
    """Strip the disk-cache env vars for every test.

    `config.reset()` re-reads the environment, and `get_cache()` then builds a
    disk-backed `Cache` in whatever directory `DATA_UNIVERSE_CACHE_DISK_DIR` points at.
    If that env var happens to be set on the machine running the tests (e.g. a real
    deployment's shell), test runs would otherwise read/write the real shared disk
    cache with mock data instead of staying purely in-memory.
    """
    monkeypatch.delenv("DATA_UNIVERSE_CACHE_DISK_DIR", raising=False)
    monkeypatch.delenv("DATA_UNIVERSE_CACHE_DISK_SIZE_LIMIT", raising=False)
    yield
