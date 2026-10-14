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


def test_membership_is_exported():
    import data_universe as q

    assert hasattr(q, "membership")
    assert "membership" in q.__all__


def test_membership_facade_exposes_expected_functions():
    from data_universe import membership

    for name in ("sp500_intervals", "membership_grid", "mask_like", "members_on", "status"):
        assert hasattr(membership, name), f"membership.{name} missing"
