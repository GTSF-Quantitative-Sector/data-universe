import pandas as pd
import pytest

from data_universe.labels import registry
from data_universe.sources.fakes import FakePolygonSource


def test_register_requires_fwd_prefix():
    local_registry = {}
    with pytest.raises(ValueError):
        @registry.register("not_forward", registry=local_registry)
        def _fn(ticker, start, end, source):
            return pd.Series(dtype="float64")


def test_register_accepts_fwd_prefixed_name():
    local_registry = {}

    @registry.register("fwd_ok", registry=local_registry)
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    assert "fwd_ok" in local_registry


def test_compute_delegates_to_shared_compute():
    local_registry = {}

    @registry.register("fwd_recorder", registry=local_registry)
    def _fn(ticker, start, end, source):
        bars = source.daily_bars(ticker, start, end)
        return pd.Series(bars["close"].values, index=bars.index.tz_localize(None).normalize())

    source = FakePolygonSource()
    result = registry.compute(
        local_registry, "fwd_recorder", "AAPL", "2024-02-01", "2024-02-10", source
    )
    assert result.index.min() >= pd.Timestamp("2024-02-01")
    assert result.index.max() <= pd.Timestamp("2024-02-10")


def test_label_registry_module_default_populated_by_decorator():
    @registry.register("fwd_default_target")
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    assert "fwd_default_target" in registry.LABEL_REGISTRY


def test_register_version_and_horizon_defaults_and_validation():
    local_registry = {}

    @registry.register("fwd_meta_default", registry=local_registry)
    def _fn(ticker, start, end, source):
        return pd.Series(dtype="float64")

    spec = local_registry["fwd_meta_default"]
    assert spec.version == 1
    assert spec.horizon_trading_days is None

    @registry.register("fwd_meta_h", horizon_trading_days=21, version=2,
                       registry=local_registry)
    def _fn2(ticker, start, end, source):
        return pd.Series(dtype="float64")

    spec2 = local_registry["fwd_meta_h"]
    assert spec2.horizon_trading_days == 21
    assert spec2.version == 2

    for bad in (0, -1, 1.5, "2", True):
        with pytest.raises(ValueError):
            registry.register(f"fwd_badver_{bad}", version=bad, registry=local_registry)
    for bad in (-1, 1.5, "x", True):
        with pytest.raises(ValueError):
            registry.register(f"fwd_badh_{bad}", horizon_trading_days=bad,
                              registry=local_registry)
