import data_universe as q


def test_get_cache_is_exported():
    assert q.get_cache is not None
    assert q.get_cache() is q.get_cache()


def test_simulate_is_exported():
    assert q.simulate is not None
