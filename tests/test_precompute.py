"""Tests for precompute v2: versioned DerivedStore layout, coverage-gap
incremental runs, label maturity caps, provisional tails, fingerprints,
loading/version selection, invalidate/status, generic datasets."""
from __future__ import annotations

import importlib
import json
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

import data_universe.calendar as cal
from data_universe.features import registry
from data_universe.features.registry import FEATURE_REGISTRY, FeatureSpec
from data_universe.labels.registry import LABEL_REGISTRY
from data_universe.precompute import (
    invalidate,
    load_dataset,
    load_feature,
    precompute,
    precompute_dataset,
    precompute_status,
)
from data_universe.sources.fakes import FakePolygonSource, FakeRateSource
from data_universe.store import source as store_source
from data_universe.store.derived import DerivedStore
from data_universe.store.source import StoreBackedSource

pm = importlib.import_module("data_universe.precompute")


class _CountingSource:
    """Wraps FakePolygonSource counting calls per (method, ticker)."""

    def __init__(self, inner=None):
        self._inner = inner or FakePolygonSource()
        self.calls = {}

    def _count(self, method, ticker):
        self.calls[(method, ticker)] = self.calls.get((method, ticker), 0) + 1

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr):
            return attr

        def _wrapped(*args, **kwargs):
            ticker = args[0] if args else kwargs.get("ticker")
            self._count(name, ticker)
            return attr(*args, **kwargs)

        return _wrapped


def _spy_compute(monkeypatch):
    """Spy on the module's `_compute`; returns the list of call tuples."""
    calls = []
    original = pm._compute

    def _spy(registry_, name, ticker, start, end, source, **kwargs):
        calls.append((name, ticker, start, end))
        return original(registry_, name, ticker, start, end, source, **kwargs)

    monkeypatch.setattr(pm, "_compute", _spy)
    return calls


def _version_dirs(out_dir, name):
    directory = out_dir / name
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


# -- 1. layout + load ---------------------------------------------------------


def test_layout_and_load_feature_matches_direct_compute(tmp_path):
    source = FakePolygonSource()
    report = precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert report.failed == {}
    fingerprint = report.versions["ret_1d"]
    version_dir = tmp_path / "ret_1d" / fingerprint
    assert (version_dir / "dataset.json").exists()
    assert (version_dir / "AAPL.parquet").exists()
    assert (version_dir / "AAPL.coverage.json").exists()

    manifest = json.loads((version_dir / "dataset.json").read_text())
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "feature"
    assert manifest["spec"]["name"] == "ret_1d"
    assert manifest["columns"] == ["date", "ticker", "value"]

    frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert list(frame.columns) == ["date", "ticker", "value"]
    expected = (
        registry.compute(
            registry.FEATURE_REGISTRY, "ret_1d", "AAPL",
            "2024-01-02", "2024-01-31", source,
        )
        .dropna()
        .rename("value")
        .reset_index()
        .rename(columns={"index": "date"})
    )
    expected["ticker"] = "AAPL"
    expected = expected[["date", "ticker", "value"]]
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True), expected.reset_index(drop=True)
    )


# -- 2. skip on rerun ----------------------------------------------------------


def test_rerun_same_window_skips_with_zero_source_calls(tmp_path):
    source = _CountingSource()
    kwargs = dict(out_dir=str(tmp_path), source=source)
    first = precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", **kwargs)
    assert first.succeeded[("AAPL", "ret_1d")] > 0
    calls_after_first = dict(source.calls)
    assert calls_after_first

    before = load_feature("ret_1d", out_dir=str(tmp_path))
    second = precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", **kwargs)
    assert second.succeeded == {}
    assert second.skipped[("AAPL", "ret_1d")] == "up to date"
    assert source.calls == calls_after_first  # zero new upstream calls
    pd.testing.assert_frame_equal(
        before, load_feature("ret_1d", out_dir=str(tmp_path))
    )


# -- 3. incremental gaps --------------------------------------------------------


def test_later_end_computes_only_the_extension(tmp_path, monkeypatch):
    source = FakePolygonSource()
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    calls = _spy_compute(monkeypatch)
    report = precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-02-15",
                        out_dir=str(tmp_path), source=source)
    assert calls == [("ret_1d", "AAPL", "2024-02-01", "2024-02-15")]
    assert report.succeeded[("AAPL", "ret_1d")] > 0

    fresh_dir = tmp_path / "fresh"
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-02-15",
               out_dir=str(fresh_dir), source=source)
    pd.testing.assert_frame_equal(
        load_feature("ret_1d", out_dir=str(tmp_path)),
        load_feature("ret_1d", out_dir=str(fresh_dir)),
    )


def test_earlier_start_computes_only_the_prefix(tmp_path, monkeypatch):
    source = FakePolygonSource()
    precompute(["AAPL"], ["ret_1d"], "2024-01-10", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    calls = _spy_compute(monkeypatch)
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    assert calls == [("ret_1d", "AAPL", "2024-01-02", "2024-01-09")]


def test_hole_in_the_middle_computes_only_the_hole(tmp_path, monkeypatch):
    source = FakePolygonSource()
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-10",
               out_dir=str(tmp_path), source=source)
    precompute(["AAPL"], ["ret_1d"], "2024-01-20", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    calls = _spy_compute(monkeypatch)
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    assert calls == [("ret_1d", "AAPL", "2024-01-11", "2024-01-19")]


# -- 4./5. tombstone + force ----------------------------------------------------


def _tomb_spec(state):
    def fn(ticker, start, end, source, **kwargs):
        index = pd.bdate_range(start, end)
        values = pd.Series(1.0, index=index)
        cutoff = state.get("nan_from")
        if cutoff is not None:
            values[values.index >= pd.Timestamp(cutoff)] = np.nan
        return values

    return FeatureSpec(fn=fn, warmup_trading_days=0, project="t", description="")


def test_forced_rerun_drops_values_that_became_nan(tmp_path, monkeypatch):
    state = {}
    monkeypatch.setitem(FEATURE_REGISTRY, "tomb_feat", _tomb_spec(state))
    source = FakePolygonSource()
    precompute(["AAPL"], ["tomb_feat"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    assert len(load_feature("tomb_feat", out_dir=str(tmp_path))) > 20

    state["nan_from"] = "2024-01-20"
    precompute(["AAPL"], ["tomb_feat"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source, force=True)
    frame = load_feature("tomb_feat", out_dir=str(tmp_path))
    assert frame["date"].max() < pd.Timestamp("2024-01-20")
    assert (frame["value"] == 1.0).all()


def test_force_recomputes_a_covered_range(tmp_path, monkeypatch):
    source = FakePolygonSource()
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    calls = _spy_compute(monkeypatch)
    report = precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
                        out_dir=str(tmp_path), source=source, force=True)
    assert calls == [("ret_1d", "AAPL", "2024-01-02", "2024-01-31")]
    assert report.succeeded[("AAPL", "ret_1d")] > 0


# -- 6. versioning / fingerprints ------------------------------------------------


def test_explicit_params_change_the_fingerprint(tmp_path):
    source = FakePolygonSource()
    window = ("2024-03-01", "2024-06-28")
    precompute(["AAPL"], ["beta_ols_60d"], *window, out_dir=str(tmp_path),
               source=source, rate_source=FakeRateSource(0.05))
    precompute(["AAPL"], ["beta_ols_60d"], *window, out_dir=str(tmp_path),
               source=source, rate_source=FakeRateSource(0.02))
    assert len(_version_dirs(tmp_path, "beta_ols_60d")) == 2

    # ret_1d has no explicit params: rate_source rides **kwargs -> one fingerprint.
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source,
               rate_source=FakeRateSource(0.05))
    report = precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
                        out_dir=str(tmp_path), source=source)
    assert report.skipped[("AAPL", "ret_1d")] == "up to date"
    assert len(_version_dirs(tmp_path, "ret_1d")) == 1


def test_store_backed_and_plain_fake_share_a_fingerprint(tmp_path):
    window = ("2024-01-02", "2024-01-31")
    precompute(["AAPL"], ["ret_1d"], *window, out_dir=str(tmp_path),
               source=FakePolygonSource())
    report = precompute(
        ["AAPL"], ["ret_1d"], *window, out_dir=str(tmp_path),
        source=StoreBackedSource(FakePolygonSource(), root=tmp_path / "raw"),
    )
    assert report.skipped[("AAPL", "ret_1d")] == "up to date"
    assert len(_version_dirs(tmp_path, "ret_1d")) == 1


def test_code_version_bump_creates_a_new_version(tmp_path, monkeypatch):
    source = FakePolygonSource()
    window = ("2024-01-02", "2024-01-31")
    old_spec = FEATURE_REGISTRY["ret_1d"]
    precompute(["AAPL"], ["ret_1d"], *window, out_dir=str(tmp_path), source=source)
    old_fp = _version_dirs(tmp_path, "ret_1d")[0]

    bumped = FeatureSpec(
        fn=old_spec.fn, warmup_trading_days=old_spec.warmup_trading_days,
        project=old_spec.project, description=old_spec.description, version=2,
    )
    monkeypatch.setitem(FEATURE_REGISTRY, "ret_1d", bumped)
    precompute(["AAPL"], ["ret_1d"], *window, out_dir=str(tmp_path), source=source)
    dirs = _version_dirs(tmp_path, "ret_1d")
    assert len(dirs) == 2 and old_fp in dirs

    # load_feature defaults to the current (v2) version.
    frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert len(frame) > 0

    # A store holding only the stale v1 -> FileNotFoundError mentioning version=.
    stale_dir = tmp_path / "stale"
    monkeypatch.setitem(FEATURE_REGISTRY, "ret_1d", old_spec)
    precompute(["AAPL"], ["ret_1d"], *window, out_dir=str(stale_dir), source=source)
    monkeypatch.setitem(FEATURE_REGISTRY, "ret_1d", bumped)
    with pytest.raises(FileNotFoundError, match="version="):
        load_feature("ret_1d", out_dir=str(stale_dir))
    # Explicit version= still reads the stale one.
    old_frame = load_feature("ret_1d", out_dir=str(stale_dir), version=old_fp)
    assert len(old_frame) > 0


def test_ambiguous_versions_require_a_filter(tmp_path):
    source = FakePolygonSource()
    window = ("2024-03-01", "2024-06-28")
    precompute(["AAPL"], ["beta_ols_60d"], *window, out_dir=str(tmp_path),
               source=source, rate_source=FakeRateSource(0.05))
    precompute(["AAPL"], ["beta_ols_60d"], *window, out_dir=str(tmp_path),
               source=source, rate_source=FakeRateSource(0.02))
    with pytest.raises(ValueError, match="version="):
        load_feature("beta_ols_60d", out_dir=str(tmp_path))
    # params filter picks exactly one.
    frame = load_feature(
        "beta_ols_60d", out_dir=str(tmp_path),
        rate_source=FakeRateSource(0.05),
    )
    expected = (
        registry.compute(
            registry.FEATURE_REGISTRY, "beta_ols_60d", "AAPL",
            *window, source, rate_source=FakeRateSource(0.05),
        )
        .dropna().rename("value").reset_index().rename(columns={"index": "date"})
    )
    expected["ticker"] = "AAPL"
    pd.testing.assert_frame_equal(
        frame.reset_index(drop=True),
        expected[["date", "ticker", "value"]].reset_index(drop=True),
        check_dtype=False,
    )


# -- 7. feature_kwargs ------------------------------------------------------------


def test_feature_kwargs_reach_only_the_named_feature(tmp_path):
    source = FakePolygonSource()
    window = ("2024-03-01", "2024-06-28")
    rate = FakeRateSource(0.05)
    report = precompute(
        ["AAPL"], ["beta_ols_60d", "beta_ols_90d"], *window,
        out_dir=str(tmp_path), source=source,
        feature_kwargs={"beta_ols_60d": {"rate_source": rate}},
    )
    assert report.failed == {}
    f60 = load_feature("beta_ols_60d", out_dir=str(tmp_path))
    f90 = load_feature("beta_ols_90d", out_dir=str(tmp_path))
    e60 = registry.compute(
        registry.FEATURE_REGISTRY, "beta_ols_60d", "AAPL", *window,
        source, rate_source=rate,
    ).dropna()
    e90 = registry.compute(
        registry.FEATURE_REGISTRY, "beta_ols_90d", "AAPL", *window, source,
    ).dropna()
    pd.testing.assert_series_equal(
        f60.set_index("date")["value"].sort_index(), e60,
        check_names=False, check_freq=False,
    )
    pd.testing.assert_series_equal(
        f90.set_index("date")["value"].sort_index(), e90,
        check_names=False, check_freq=False,
    )


def test_unknown_feature_kwargs_key_raises(tmp_path):
    with pytest.raises(ValueError, match="feature_kwargs"):
        precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
                   out_dir=str(tmp_path), source=FakePolygonSource(),
                   feature_kwargs={"nope": {}})


# -- 8. labels ---------------------------------------------------------------------


def _label_coverage_end(out_dir, name):
    store = DerivedStore(out_dir)
    manifest = store.versions(name)[0]
    return store.coverage(name, manifest["fingerprint"], "AAPL")[-1][1]


def test_label_writes_only_final_rows_and_extends(tmp_path):
    source = FakePolygonSource()
    window = ("2024-01-02", "2024-03-29")
    report = precompute(["AAPL"], ["fwd_ret_15d"], *window,
                        out_dir=str(tmp_path), source=source)
    assert report.failed == {}
    # Expected cap: 15 sessions before the last session <= end.
    sess = cal.sessions(source, "2024-01-02", "2024-03-29")
    expected_cap = sess[-1 - 15]
    assert _label_coverage_end(tmp_path, "fwd_ret_15d") == expected_cap
    frame = load_feature("fwd_ret_15d", out_dir=str(tmp_path))
    assert frame["date"].max() <= expected_cap

    # Extending E later fills the newly-mature rows, equal to a fresh run.
    precompute(["AAPL"], ["fwd_ret_15d"], "2024-01-02", "2024-04-30",
               out_dir=str(tmp_path), source=source)
    fresh = tmp_path / "fresh"
    precompute(["AAPL"], ["fwd_ret_15d"], "2024-01-02", "2024-04-30",
               out_dir=str(fresh), source=source)
    pd.testing.assert_frame_equal(
        load_feature("fwd_ret_15d", out_dir=str(tmp_path)),
        load_feature("fwd_ret_15d", out_dir=str(fresh)),
    )
    assert _label_coverage_end(tmp_path, "fwd_ret_15d") > expected_cap


def test_forced_label_rerun_with_shorter_end_keeps_later_rows(tmp_path):
    source = FakePolygonSource()
    precompute(["AAPL"], ["fwd_ret_15d"], "2024-01-02", "2024-04-30",
               out_dir=str(tmp_path), source=source)
    before = load_feature("fwd_ret_15d", out_dir=str(tmp_path))
    precompute(["AAPL"], ["fwd_ret_15d"], "2024-01-02", "2024-03-29",
               out_dir=str(tmp_path), source=source, force=True)
    after = load_feature("fwd_ret_15d", out_dir=str(tmp_path))
    # Rows beyond the new (shorter) cap survive the forced rewrite.
    assert len(after) == len(before)
    pd.testing.assert_frame_equal(after, before)


def test_label_without_horizon_fails_with_guidance(tmp_path, monkeypatch):
    def fn(ticker, start, end, source, **kwargs):
        return pd.Series(1.0, index=pd.bdate_range(start, end))

    monkeypatch.setitem(
        LABEL_REGISTRY, "fwd_nohorizon",
        FeatureSpec(fn=fn, warmup_trading_days=0, project="t", description="",
                    horizon_trading_days=None),
    )
    report = precompute(["AAPL"], ["fwd_nohorizon"], "2024-01-02", "2024-01-31",
                        out_dir=str(tmp_path), source=FakePolygonSource())
    assert "horizon_trading_days" in report.failed[("AAPL", "fwd_nohorizon")]


# -- 9. provisional tail -------------------------------------------------------------


def test_provisional_tail_writes_but_does_not_cover(tmp_path, monkeypatch):
    state = {"v": 1.0}

    def fn(ticker, start, end, source, **kwargs):
        return pd.Series(state["v"], index=pd.bdate_range(start, end))

    monkeypatch.setitem(
        FEATURE_REGISTRY, "prov_feat",
        FeatureSpec(fn=fn, warmup_trading_days=0, project="t", description=""),
    )
    monkeypatch.setattr(
        store_source, "_last_complete_date", lambda: pd.Timestamp("2024-01-15")
    )
    source = FakePolygonSource()
    report = precompute(["AAPL"], ["prov_feat"], "2024-01-02", "2024-01-31",
                        out_dir=str(tmp_path), source=source)
    assert report.failed == {}
    frame = load_feature("prov_feat", out_dir=str(tmp_path))
    assert frame["date"].max() == pd.Timestamp("2024-01-31")  # provisional rows written
    assert _label_coverage_end(tmp_path, "prov_feat") == pd.Timestamp("2024-01-15")

    # Rerun: only (L, E] is recomputed and its rows replaced.
    state["v"] = 2.0
    calls = _spy_compute(monkeypatch)
    precompute(["AAPL"], ["prov_feat"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    assert calls == [("prov_feat", "AAPL", "2024-01-16", "2024-01-31")]
    frame = load_feature("prov_feat", out_dir=str(tmp_path))
    final = frame[frame["date"] <= pd.Timestamp("2024-01-15")]
    prov = frame[frame["date"] > pd.Timestamp("2024-01-15")]
    assert (final["value"] == 1.0).all()
    assert (prov["value"] == 2.0).all()


def test_provisional_tail_never_starts_before_requested_start(tmp_path, monkeypatch):
    # Whole window after L: the provisional start clamps to S, not M + 1 day.
    state = {"v": 1.0}

    def fn(ticker, start, end, source, **kwargs):
        return pd.Series(state["v"], index=pd.bdate_range(start, end))

    monkeypatch.setitem(
        FEATURE_REGISTRY, "prov_feat2",
        FeatureSpec(fn=fn, warmup_trading_days=0, project="t", description=""),
    )
    monkeypatch.setattr(
        store_source, "_last_complete_date", lambda: pd.Timestamp("2024-01-10")
    )
    calls = _spy_compute(monkeypatch)
    report = precompute(["AAPL"], ["prov_feat2"], "2024-01-20", "2024-01-31",
                        out_dir=str(tmp_path), source=FakePolygonSource())
    assert report.failed == {}
    assert calls == [("prov_feat2", "AAPL", "2024-01-20", "2024-01-31")]
    frame = load_feature("prov_feat2", out_dir=str(tmp_path))
    assert frame["date"].min() >= pd.Timestamp("2024-01-20")


def test_dataset_builder_filtered_index_keeps_row_pairing(tmp_path):
    # A gapped (filtered) index must not misalign dates and values.
    def builder(key, a, d):
        dates = pd.bdate_range(a, d)
        frame = pd.DataFrame(
            {
                "date": dates,
                "x": np.arange(len(dates)),
                "value": np.arange(len(dates), dtype=float),
            }
        )
        return frame[frame["x"] % 2 == 1]

    report = precompute_dataset("ev", builder, ["K"], "2024-01-02", "2024-01-31",
                                out_dir=str(tmp_path))
    assert report.failed == {}
    frame = load_dataset("ev", out_dir=str(tmp_path))
    expected = builder("K", "2024-01-02", "2024-01-31").reset_index(drop=True)
    expected.insert(1, "key", "K")
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)


def test_dataset_builder_string_index_keeps_row_pairing(tmp_path):
    # A non-RangeIndex (string labels) must not misalign dates and values.
    def builder(key, a, d):
        dates = pd.bdate_range(a, d)
        return pd.DataFrame(
            {
                "kind": [f"k{i}" for i in range(len(dates))],
                "date": dates,
                "value": np.arange(len(dates), dtype=float),
            }
        ).set_index("kind")

    report = precompute_dataset("ev2", builder, ["K"], "2024-01-02", "2024-01-31",
                                out_dir=str(tmp_path))
    assert report.failed == {}
    frame = load_dataset("ev2", out_dir=str(tmp_path))
    expected = builder("K", "2024-01-02", "2024-01-31").reset_index(drop=True)
    expected.insert(1, "key", "K")
    pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected)


def test_bare_string_ticker_is_treated_as_one_key(tmp_path):
    source = FakePolygonSource()
    report = precompute("AAPL", ["ret_1d"], "2024-01-02", "2024-01-31",
                        out_dir=str(tmp_path), source=source)
    assert ("AAPL", "ret_1d") in report.succeeded
    frame = load_feature("ret_1d", tickers="AAPL", out_dir=str(tmp_path))
    assert set(frame["ticker"]) == {"AAPL"}


# -- 10. multi-column registry feature -------------------------------------------------


def _multi_col_spec(columns):
    def fn(ticker, start, end, source, **kwargs):
        index = pd.bdate_range(start, end)
        frame = pd.DataFrame(index=index)
        for i, col in enumerate(columns):
            frame[col] = float(i + 1)
        # one all-NaN row and one partial row
        if len(frame) > 3:
            frame.iloc[1] = np.nan
            frame.iloc[2, 0] = np.nan
        return frame

    return FeatureSpec(fn=fn, warmup_trading_days=0, project="t", description="")


def test_multi_column_feature_stores_all_columns(tmp_path, monkeypatch):
    monkeypatch.setitem(
        FEATURE_REGISTRY, "mc_feat", _multi_col_spec(["a", "b"])
    )
    precompute(["AAPL"], ["mc_feat"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=FakePolygonSource())
    frame = load_feature("mc_feat", out_dir=str(tmp_path))
    assert list(frame.columns) == ["date", "ticker", "a", "b"]
    # The all-NaN row was dropped; the partial row kept.
    bdates = pd.bdate_range("2024-01-02", "2024-01-31")
    assert len(frame) == len(bdates) - 1
    partial = frame[frame["a"].isna()]
    assert len(partial) == 1 and partial["b"].notna().all()


def test_column_change_in_same_version_fails_with_bump_message(tmp_path, monkeypatch):
    state = {"cols": ["a", "b"]}

    def fn(ticker, start, end, source, **kwargs):
        return pd.DataFrame(
            {c: 1.0 for c in state["cols"]}, index=pd.bdate_range(start, end)
        )

    monkeypatch.setitem(
        FEATURE_REGISTRY, "mc_feat2",
        FeatureSpec(fn=fn, warmup_trading_days=0, project="t", description=""),
    )
    source = FakePolygonSource()
    precompute(["AAPL"], ["mc_feat2"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    state["cols"] = ["a", "c"]
    report = precompute(["AAPL"], ["mc_feat2"], "2024-01-02", "2024-01-31",
                        out_dir=str(tmp_path), source=source, force=True)
    assert "bump the version" in report.failed[("AAPL", "mc_feat2")]


# -- 11. invalidate + status ------------------------------------------------------------


def test_invalidate_key_range_version_and_name(tmp_path, monkeypatch):
    source = FakePolygonSource()
    window = ("2024-01-02", "2024-01-31")
    report = precompute(["AAPL", "MSFT"], ["ret_1d"], *window,
                        out_dir=str(tmp_path), source=source)
    fp = report.versions["ret_1d"]

    # Whole key.
    assert invalidate("ret_1d", keys=["AAPL"], out_dir=str(tmp_path)) == 1
    frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert set(frame["ticker"]) == {"MSFT"}

    # Date range, then a rerun recomputes only the hole.
    assert invalidate(
        "ret_1d", start="2024-01-10", end="2024-01-20", out_dir=str(tmp_path)
    ) == 1
    calls = _spy_compute(monkeypatch)
    precompute(["MSFT"], ["ret_1d"], *window,
               out_dir=str(tmp_path), source=source)
    assert calls == [("ret_1d", "MSFT", "2024-01-10", "2024-01-20")]

    # Whole version.
    assert invalidate("ret_1d", version=fp, out_dir=str(tmp_path)) == 1
    with pytest.raises(FileNotFoundError):
        load_feature("ret_1d", out_dir=str(tmp_path))


def test_invalidate_whole_name_removes_legacy_file_too(tmp_path):
    source = FakePolygonSource()
    precompute(["AAPL", "MSFT"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    legacy = tmp_path / "ret_1d.parquet"
    legacy.write_bytes(b"legacy")
    removed = invalidate("ret_1d", out_dir=str(tmp_path))
    assert removed == 3  # two keys + the legacy file
    assert not legacy.exists()
    assert _version_dirs(tmp_path, "ret_1d") == []


def test_precompute_status_fields(tmp_path):
    source = FakePolygonSource()
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    precompute_dataset(
        "events", lambda k, a, d: pd.DataFrame({"date": pd.bdate_range(a, d)}),
        ["K1"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path),
    )
    status = precompute_status(out_dir=str(tmp_path))
    assert {"name", "kind", "fingerprint", "schema_version", "key",
            "coverage_start", "coverage_end", "n_intervals", "rows",
            "last_computed_at", "tool_version", "code_version",
            "source_family", "params", "current"} <= set(status.columns)
    ret_row = status[status["name"] == "ret_1d"].iloc[0]
    assert ret_row["kind"] == "feature"
    assert ret_row["current"]
    assert ret_row["source_family"] == "FakePolygonSource"
    assert ret_row["code_version"] == 1
    ev_row = status[status["name"] == "events"].iloc[0]
    assert ev_row["kind"] == "dataset"
    assert ev_row["current"] is None or pd.isna(ev_row["current"])


# -- 12. legacy fallback ---------------------------------------------------------------


def test_legacy_parquet_fallback_warns_then_versions_win(tmp_path):
    legacy = tmp_path / "ret_1d.parquet"
    pd.DataFrame(
        {"date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
         "ticker": ["AAPL", "AAPL"], "value": [0.1, 0.2]}
    ).to_parquet(legacy, index=False)
    with pytest.warns(UserWarning, match="pre-v0.13"):
        frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert frame["value"].tolist() == [0.1, 0.2]

    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=FakePolygonSource())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert len(frame) > 2  # versioned data, not the 2-row legacy file


# -- 13. resume after partial failure -----------------------------------------------------


class _TickerRaisingSource:
    def __init__(self, inner, bad):
        self._inner = inner
        self._bad = bad

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def daily_bars_split_adjusted(self, ticker, start, end):
        if ticker == self._bad:
            raise ValueError(f"simulated failure for {ticker}")
        return self._inner.daily_bars_split_adjusted(ticker, start, end)

    def daily_bars(self, ticker, start, end):
        if ticker == self._bad:
            raise ValueError(f"simulated failure for {ticker}")
        return self._inner.daily_bars(ticker, start, end)


def test_resume_after_partial_failure_computes_only_the_rest(tmp_path, monkeypatch):
    window = ("2024-01-02", "2024-01-31")
    bad = _TickerRaisingSource(FakePolygonSource(), "MSFT")
    first = precompute(["AAPL", "MSFT"], ["ret_1d"], *window,
                       out_dir=str(tmp_path), source=bad)
    assert ("AAPL", "ret_1d") in first.succeeded
    assert "simulated failure" in first.failed[("MSFT", "ret_1d")]

    calls = _spy_compute(monkeypatch)
    # Same source family -> same fingerprint; MSFT no longer raises.
    healthy = _TickerRaisingSource(FakePolygonSource(), None)
    second = precompute(["AAPL", "MSFT"], ["ret_1d"], *window,
                        out_dir=str(tmp_path), source=healthy)
    assert second.skipped[("AAPL", "ret_1d")] == "up to date"
    assert ("MSFT", "ret_1d") in second.succeeded
    assert calls == [("ret_1d", "MSFT", "2024-01-02", "2024-01-31")]


# -- 14. kept behaviours -----------------------------------------------------------------


def test_unregistered_name_fails_for_every_ticker(tmp_path):
    report = precompute(["AAPL", "MSFT"], ["not_a_feature"], "2024-01-02",
                        "2024-01-31", out_dir=str(tmp_path),
                        source=FakePolygonSource())
    assert report.succeeded == {}
    assert set(report.failed) == {("AAPL", "not_a_feature"), ("MSFT", "not_a_feature")}
    assert DerivedStore(tmp_path).names() == []


def test_rate_source_forwarding_and_missing_rate_source_failure(tmp_path):
    source = FakePolygonSource()
    window = ("2024-03-01", "2024-03-15")
    report = precompute(["AAPL"], ["ret_excess_1d"], *window,
                        out_dir=str(tmp_path), source=source,
                        rate_source=FakeRateSource(0.05))
    assert report.failed == {}
    frame = load_feature("ret_excess_1d", out_dir=str(tmp_path))
    raw = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", *window, source
    )
    expected = (raw - 0.05 / 252.0).dropna().rename("value")
    written = frame.set_index("date")["value"].sort_index()
    pd.testing.assert_series_equal(written, expected, check_freq=False)

    bad = tmp_path / "bad"
    report = precompute(["AAPL"], ["ret_excess_1d"], *window,
                        out_dir=str(bad), source=source)
    assert "rate_source" in report.failed[("AAPL", "ret_excess_1d")]
    assert DerivedStore(bad).names() == []


def test_concurrent_writers_all_land(tmp_path):
    source = FakePolygonSource()
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]

    def _run(ticker):
        return precompute([ticker], ["ret_1d"], "2024-01-02", "2024-01-31",
                           out_dir=str(tmp_path), source=source)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_run, tickers))
    frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert set(frame["ticker"]) == set(tickers)
    assert not frame.duplicated(subset=["date", "ticker"]).any()


def test_failed_write_leaves_existing_data_and_no_temp_files(tmp_path, monkeypatch):
    source = FakePolygonSource()
    precompute(["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
               out_dir=str(tmp_path), source=source)
    data_path = tmp_path / "ret_1d"
    parquet = next(data_path.glob("*/*.parquet"))
    before = parquet.read_bytes()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    report = precompute(["MSFT"], ["ret_1d"], "2024-01-02", "2024-02-15",
                        out_dir=str(tmp_path), source=source)
    assert "disk full" in report.failed[("MSFT", "ret_1d")]
    assert parquet.read_bytes() == before
    assert list(data_path.glob("*/*.tmp")) == []


def test_start_after_end_raises(tmp_path):
    with pytest.raises(ValueError):
        precompute(["AAPL"], ["ret_1d"], "2024-02-01", "2024-01-01",
                   out_dir=str(tmp_path), source=FakePolygonSource())
    with pytest.raises(ValueError):
        precompute_dataset("x", lambda k, a, d: pd.DataFrame(), ["K"],
                           "2024-02-01", "2024-01-01", out_dir=str(tmp_path))


# -- 15. precompute_dataset -----------------------------------------------------------------


def _events_builder(key, a, d):
    dates = pd.bdate_range(a, d)
    out = {"date": [], "kind": [], "amount": []}
    for dt in dates:
        out["date"] += [dt, dt]
        out["kind"] += ["x", "y"]
        out["amount"] += [1.5, np.nan]
    return pd.DataFrame(out)


def test_precompute_dataset_roundtrip_skip_and_extend(tmp_path):
    report = precompute_dataset(
        "events", _events_builder, ["K1"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path),
    )
    assert report.failed == {}
    frame = load_dataset("events", out_dir=str(tmp_path))
    assert list(frame.columns) == ["date", "key", "kind", "amount"]
    assert (frame["key"] == "K1").all()
    assert frame["amount"].isna().any()  # generic datasets keep NaNs

    again = precompute_dataset(
        "events", _events_builder, ["K1"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path),
    )
    assert again.skipped[("K1", "events")] == "up to date"

    extended = precompute_dataset(
        "events", _events_builder, ["K1"], "2024-01-02", "2024-02-15",
        out_dir=str(tmp_path),
    )
    assert extended.succeeded[("K1", "events")] > 0
    frame = load_dataset("events", keys=["K1"], out_dir=str(tmp_path),
                         start="2024-02-01", end="2024-02-15")
    assert frame["date"].min() >= pd.Timestamp("2024-02-01")


def test_precompute_dataset_key_column_mismatch_fails(tmp_path):
    def bad_builder(key, a, d):
        dates = pd.bdate_range(a, d)
        return pd.DataFrame({"date": dates, "key": ["OTHER"] * len(dates)})

    report = precompute_dataset("events", bad_builder, ["K1"],
                                "2024-01-02", "2024-01-31", out_dir=str(tmp_path))
    assert "K1" in report.failed[("K1", "events")]


def test_precompute_dataset_final_lag_uses_session_calendar(tmp_path):
    source = FakePolygonSource()
    lag = 2
    report = precompute_dataset(
        "lagged", _events_builder, ["K1"], "2024-01-02", "2024-03-29",
        final_lag_sessions=lag, source=source, out_dir=str(tmp_path),
    )
    assert report.failed == {}
    sess = cal.sessions(source, "2024-01-02", "2024-03-29")
    expected_cap = sess[-1 - lag]
    store = DerivedStore(tmp_path)
    manifest = store.versions("lagged")[0]
    coverage = store.coverage("lagged", manifest["fingerprint"], "K1")
    assert coverage[-1][1] == expected_cap


def test_precompute_dataset_rejects_registered_names_and_bad_lag(tmp_path):
    with pytest.raises(ValueError, match="registered"):
        precompute_dataset("ret_1d", _events_builder, ["K"], "2024-01-02",
                           "2024-01-31", out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="final_lag_sessions"):
        precompute_dataset("x", _events_builder, ["K"], "2024-01-02",
                           "2024-01-31", final_lag_sessions=-1,
                           out_dir=str(tmp_path))
    with pytest.raises(ValueError, match="source"):
        precompute_dataset("x", _events_builder, ["K"], "2024-01-02",
                           "2024-01-31", final_lag_sessions=2,
                           out_dir=str(tmp_path))


def test_precompute_dataset_validates_name_and_version(tmp_path):
    for bad_name in ("../evil", "a/b", ".hidden", ""):
        with pytest.raises(ValueError, match="directory"):
            precompute_dataset(bad_name, _events_builder, ["K"], "2024-01-02",
                               "2024-01-31", out_dir=str(tmp_path))
    for bad_version in (0, -1, 1.5, "1", True):
        with pytest.raises(ValueError, match="version must be an int"):
            precompute_dataset("ok", _events_builder, ["K"], "2024-01-02",
                               "2024-01-31", version=bad_version,
                               out_dir=str(tmp_path))


def test_dataset_params_land_in_dataset_json_without_secrets(tmp_path):
    class _Credentials:
        def __init__(self):
            self.api_key = "sk-live-SECRET-123"

    precompute_dataset(
        "events", _events_builder, ["K1"], "2024-01-02", "2024-01-31",
        params={"creds": _Credentials(), "note": "hello"},
        out_dir=str(tmp_path),
    )
    store = DerivedStore(tmp_path)
    manifest = store.dataset_manifest("events", store.versions("events")[0]["fingerprint"])
    text = json.dumps(manifest["spec"])
    assert "sk-live-SECRET-123" not in text
    creds = manifest["spec"]["params"]["creds"]
    assert set(creds) == {"type"} and creds["type"].endswith("_Credentials")
    assert manifest["spec"]["params"]["note"] == "hello"


# -- registry / source fingerprint methods -----------------------------------------------


def test_canonical_handles_common_types():
    assert pm.canonical(None) is None
    assert pm.canonical(True) is True
    assert pm.canonical(3) == 3
    assert pm.canonical(1.5) == 1.5
    assert pm.canonical(np.float64(2.5)) == 2.5
    assert pm.canonical(pd.Timestamp("2024-01-02")) == "2024-01-02T00:00:00"
    assert pm.canonical({"b": 1, "a": [True, None]}) == {"b": 1, "a": [True, None]}
    assert pm.canonical({2, 1}) == [1, 2]
    with pytest.raises(ValueError):
        pm.canonical(float("nan"))
