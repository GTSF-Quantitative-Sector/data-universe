import data_universe as q


def test_get_cache_is_exported():
    assert q.get_cache is not None
    assert q.get_cache() is q.get_cache()


def test_simulate_is_exported():
    assert q.simulate is not None


def test_feature_and_label_registries_are_exported():
    assert "ret_1d" in q.FEATURE_REGISTRY
    assert "fwd_ret_15d" in q.LABEL_REGISTRY


def test_precompute_and_load_feature_are_exported():
    assert callable(q.precompute)
    assert callable(q.load_feature)
