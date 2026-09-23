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
