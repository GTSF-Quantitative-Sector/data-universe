import datetime as dt

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import Cache, make_key


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def test_get_or_load_calls_loader_only_once_on_repeat_hit():
    cache = Cache(capacity=10)
    calls = []

    def loader():
        calls.append(1)
        return "value"

    key = make_key((), {})
    first = cache.get_or_load("ns", key, loader)
    second = cache.get_or_load("ns", key, loader)
    assert first == "value"
    assert second == "value"
    assert len(calls) == 1


def test_stats_hits_and_misses():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("a",), lambda: 1)
    cache.get_or_load("ns", ("a",), lambda: 1)
    cache.get_or_load("ns", ("b",), lambda: 2)
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 2
    assert stats["size"] == 2
    assert stats["hit_rate"] == pytest.approx(1 / 3)


def test_lru_eviction_order():
    cache = Cache(capacity=2)
    cache.get_or_load("ns", ("a",), lambda: "A")
    cache.get_or_load("ns", ("b",), lambda: "B")
    cache.get_or_load("ns", ("c",), lambda: "C")  # evicts "a" (least recently used)
    stats = cache.stats()
    assert stats["size"] == 2
    assert stats["evictions"] == 1
    calls = []
    cache.get_or_load("ns", ("a",), lambda: calls.append(1) or "A-reloaded")
    assert calls == [1]  # "a" was evicted, had to reload


def test_lru_recently_used_key_is_not_evicted():
    cache = Cache(capacity=2)
    cache.get_or_load("ns", ("a",), lambda: "A")
    cache.get_or_load("ns", ("b",), lambda: "B")
    cache.get_or_load("ns", ("a",), lambda: "A")  # touch "a", making "b" the LRU
    cache.get_or_load("ns", ("c",), lambda: "C")  # should evict "b", not "a"
    calls = []
    cache.get_or_load("ns", ("a",), lambda: calls.append(1) or "A-reloaded")
    assert calls == []  # "a" still cached, no reload


def test_dataframe_hits_return_independent_copies():
    cache = Cache(capacity=10)
    df = pd.DataFrame({"x": [1, 2, 3]})
    cache.get_or_load("ns", ("k",), lambda: df)
    result = cache.get_or_load("ns", ("k",), lambda: df)
    result.iloc[0, 0] = 999
    fresh = cache.get_or_load("ns", ("k",), lambda: df)
    assert fresh.iloc[0, 0] != 999


def test_series_hits_return_independent_copies():
    cache = Cache(capacity=10)
    series = pd.Series([1, 2, 3])
    cache.get_or_load("ns", ("k",), lambda: series)
    result = cache.get_or_load("ns", ("k",), lambda: series)
    result.iloc[0] = 999
    fresh = cache.get_or_load("ns", ("k",), lambda: series)
    assert fresh.iloc[0] != 999


def test_access_log_records_project_tag():
    config.set_project("vrp-research")
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    log = cache.access_log()
    assert (log["project"] == "vrp-research").all()


def test_access_log_has_expected_columns_and_order():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    log = cache.access_log()
    assert list(log.columns) == [
        "timestamp",
        "namespace",
        "key_repr",
        "hit",
        "bytes",
        "load_seconds",
        "project",
    ]


def test_access_log_marks_hit_and_miss_correctly():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    cache.get_or_load("ns", ("k",), lambda: 1)
    log = cache.access_log()
    assert list(log["hit"]) == [False, True]


def test_flush_log_writes_parquet(tmp_path):
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    out_path = tmp_path / "log.parquet"
    written = cache.flush_log(str(out_path))
    assert written == str(out_path)
    assert out_path.exists()
    reloaded = pd.read_parquet(out_path)
    assert len(reloaded) == 1


def test_flush_log_default_path_under_data_dir_cache_logs(tmp_path):
    config.set_data_dir(str(tmp_path))
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    written = cache.flush_log()
    assert (tmp_path / "cache_logs").exists()
    assert written.startswith(str(tmp_path / "cache_logs"))


def test_clear_resets_stats_and_store():
    cache = Cache(capacity=10)
    cache.get_or_load("ns", ("k",), lambda: 1)
    cache.clear()
    stats = cache.stats()
    assert stats == {
        "size": 0,
        "hits": 0,
        "misses": 0,
        "evictions": 0,
        "hit_rate": 0.0,
        "bytes_held": 0,
    }


def test_cache_defaults_capacity_from_config():
    config.set_cache_size(3)
    cache = Cache()
    for k in ("a", "b", "c", "d"):
        cache.get_or_load("ns", (k,), lambda k=k: k)
    assert cache.stats()["size"] == 3


def test_make_key_normalizes_dates_and_list_order():
    key1 = make_key((), {"start": dt.date(2024, 1, 1), "tickers": ["AAPL", "MSFT"]})
    key2 = make_key((), {"tickers": ["MSFT", "AAPL"], "start": "2024-01-01"})
    assert key1 == key2


def test_make_key_normalizes_dicts():
    key1 = make_key((), {"opts": {"a": 1, "b": 2}})
    key2 = make_key((), {"opts": {"b": 2, "a": 1}})
    assert key1 == key2


def test_make_key_distinguishes_different_args():
    assert make_key(("AAPL",), {}) != make_key(("MSFT",), {})
