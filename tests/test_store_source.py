"""`StoreBackedSource`: parity with the upstream across windows, one-fetch
durable reuse, completed-session splitting, split rebases (stored and live
parts), entitlement/empty/family guards, cache bypass, and the precompute
acceptance case."""
from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import get_cache, reset_cache
from data_universe.precompute import precompute
from data_universe.sources import adjust
from data_universe.sources.base import (
    DataSourceError,
    EntitlementError,
    empty_bars,
)
from data_universe.sources.fakes import FakePolygonSource
from data_universe.sources.polygon import PolygonSource
from data_universe.store import StoreBackedSource
from data_universe.store.raw import _row_dates
from tests.test_sources_polygon import _routing_session


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    config.reset()
    reset_cache()
    monkeypatch.setattr(
        "data_universe.store.source._last_complete_date",
        lambda: pd.Timestamp("2024-12-31"),
    )
    yield
    config.reset()
    reset_cache()


class CountingFake(FakePolygonSource):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = {}

    def _count(self, method, ticker):
        self.calls[(method, ticker)] = self.calls.get((method, ticker), 0) + 1

    def daily_bars_split_adjusted(self, ticker, start, end):
        self._count("daily_bars_split_adjusted", ticker)
        return super().daily_bars_split_adjusted(ticker, start, end)

    def daily_bars_raw(self, ticker, start, end):
        self._count("daily_bars_raw", ticker)
        return super().daily_bars_raw(ticker, start, end)

    def index_daily_bars(self, ticker, start, end):
        self._count("index_daily_bars", ticker)
        return super().index_daily_bars(ticker, start, end)

    def dividends(self, ticker, start, end):
        self._count("dividends", ticker)
        return super().dividends(ticker, start, end)


class SplitLater(CountingFake):
    """`daily_bars_split_adjusted` becomes /4 (volume x4) before `split_date`
    once `split_applied` -- mimicking a vendor rebasing history after a split."""

    def __init__(self, *args, split_date="2020-09-01", **kwargs):
        super().__init__(*args, **kwargs)
        self.split_applied = False
        self.split_date = pd.Timestamp(split_date)

    def daily_bars_split_adjusted(self, ticker, start, end):
        bars = super().daily_bars_split_adjusted(ticker, start, end)
        if not self.split_applied:
            return bars
        dates = bars.index.tz_localize(None).normalize()
        mask = dates < self.split_date
        out = bars.copy()
        for column in ("open", "high", "low", "close", "vwap"):
            out.loc[mask, column] = out.loc[mask, column] / 4.0
        out.loc[mask, "volume"] = out.loc[mask, "volume"] * 4.0
        return out


def _assert_equal(actual, expected):
    pd.testing.assert_frame_equal(actual, expected, check_freq=False)


def test_parity_across_a_sequence_of_windows(tmp_path):
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    windows = [
        ("2024-01-01", "2024-03-31"),   # cold
        ("2024-02-01", "2024-02-15"),   # sub-window, no fetch
        ("2024-04-01", "2024-06-30"),   # extension after a gap
        ("2024-01-15", "2024-06-15"),   # overlap both sides
        ("2023-11-01", "2024-01-31"),   # backfill before
        ("2024-01-01", "2024-06-30"),   # spans both earlier gaps -> hull
    ]
    for s, e in windows:
        for method, key in (
            ("daily_bars_split_adjusted", "AAA"),
            ("daily_bars_raw", "AAA"),
            ("dividends", "AAA"),
            ("index_daily_bars", "I:VIX"),
        ):
            _assert_equal(getattr(sbs, method)(key, s, e),
                          getattr(upstream, method)(key, s, e))
        _assert_equal(
            sbs.daily_bars("AAA", s, e),
            adjust.apply_total_return(
                upstream.daily_bars_split_adjusted("AAA", s, e),
                adjust.dividend_factors(
                    upstream.daily_bars_raw("AAA", s, e)["close"],
                    upstream.dividends("AAA", s, e),
                ),
            ),
        )


def test_each_dataset_is_fetched_once_and_reused(tmp_path):
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    sbs.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-03-31")
    assert upstream.calls[("daily_bars_split_adjusted", "AAA")] == 1
    sbs.daily_bars_split_adjusted("AAA", "2024-02-01", "2024-02-15")
    assert upstream.calls[("daily_bars_split_adjusted", "AAA")] == 1
    sbs.daily_bars_split_adjusted("AAA", "2024-04-01", "2024-04-30")
    assert upstream.calls[("daily_bars_split_adjusted", "AAA")] == 2
    # Two disjoint gaps in one window -> one hull fetch.
    upstream.calls.clear()
    sbs.daily_bars_raw("AAA", "2024-01-01", "2024-01-31")
    sbs.daily_bars_raw("AAA", "2024-04-01", "2024-04-30")
    upstream.calls.clear()
    sbs.daily_bars_raw("AAA", "2024-01-01", "2024-04-30")
    assert upstream.calls[("daily_bars_raw", "AAA")] == 1
    # A brand-new source on the same root needs nothing.
    fresh = StoreBackedSource(CountingFake(), root=tmp_path)
    fresh.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-04-30")
    assert fresh.upstream.calls == {}


def test_completed_sessions_split_stored_and_live_parts(tmp_path):
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    s, e = "2024-10-01", "2025-01-05"  # L is pinned at 2024-12-31
    first = sbs.daily_bars_split_adjusted("AAA", s, e)
    coverage = sbs.store.coverage("daily_bars_split_adjusted", "AAA")
    assert coverage[-1][1] == pd.Timestamp("2024-12-31")
    _assert_equal(first, upstream.daily_bars_split_adjusted("AAA", s, e))
    before = dict(upstream.calls)
    sbs.daily_bars_split_adjusted("AAA", s, e)
    # One live call (the [L+1, e] part is passed through each time), no
    # store fetch -- coverage was already complete.
    assert upstream.calls[("daily_bars_split_adjusted", "AAA")] == \
        before[("daily_bars_split_adjusted", "AAA")] + 1


def test_stored_part_rebases_when_a_split_changes_the_overlap(tmp_path):
    upstream = SplitLater(split_date="2020-09-01")
    sbs = StoreBackedSource(upstream, root=tmp_path)
    sbs.daily_bars_split_adjusted("AAA", "2020-01-01", "2020-06-30")
    upstream.split_applied = True
    out = sbs.daily_bars_split_adjusted("AAA", "2020-01-01", "2020-12-31")
    _assert_equal(out, upstream.daily_bars_split_adjusted("AAA", "2020-01-01", "2020-12-31"))
    records = sbs.store.manifest("daily_bars_split_adjusted", "AAA")["fetches"]
    assert len(records) == 1
    assert records[0]["rebased"] is True
    assert records[0]["start"] == "2020-01-01" and records[0]["end"] == "2020-12-31"
    assert sbs.store.manifest("daily_bars_raw", "AAA")["fetches"] == []


def test_live_part_overlap_rebases_on_a_split_inside_the_live_window(tmp_path):
    # L pinned at 2024-12-31, the split lands 2025-01-01 -- inside the live part.
    upstream = SplitLater(split_date="2025-01-01")
    sbs = StoreBackedSource(upstream, root=tmp_path)
    sbs.daily_bars_split_adjusted("AAA", "2024-10-01", "2024-12-31")
    upstream.split_applied = True
    out = sbs.daily_bars_split_adjusted("AAA", "2024-10-01", "2025-01-03")
    _assert_equal(out, upstream.daily_bars_split_adjusted("AAA", "2024-10-01", "2025-01-03"))
    records = sbs.store.manifest("daily_bars_split_adjusted", "AAA")["fetches"]
    assert any(r["rebased"] for r in records)
    assert not out.index.duplicated().any()


def test_entitlement_error_is_recorded_not_covered_and_raised(tmp_path):
    class Failing(CountingFake):
        def daily_bars_split_adjusted(self, ticker, start, end):
            self._count("daily_bars_split_adjusted", ticker)
            raise EntitlementError("403")

    upstream = Failing()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    with pytest.raises(EntitlementError):
        sbs.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-01-31")
    records = sbs.store.manifest("daily_bars_split_adjusted", "AAA")["fetches"]
    assert [r["status"] for r in records] == ["not_entitled"]
    assert sbs.store.coverage("daily_bars_split_adjusted", "AAA") == []
    with pytest.raises(EntitlementError):
        sbs.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-01-31")
    assert upstream.calls[("daily_bars_split_adjusted", "AAA")] == 2


def test_empty_upstream_marks_coverage_and_is_not_refetched(tmp_path):
    class Empty(CountingFake):
        def daily_bars_split_adjusted(self, ticker, start, end):
            self._count("daily_bars_split_adjusted", ticker)
            return empty_bars()

    upstream = Empty()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    out = sbs.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-01-31")
    pd.testing.assert_frame_equal(out, empty_bars())
    records = sbs.store.manifest("daily_bars_split_adjusted", "AAA")["fetches"]
    assert records[0]["status"] == "empty"
    sbs.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-01-31")
    assert upstream.calls[("daily_bars_split_adjusted", "AAA")] == 1


def test_family_guard_rejects_a_foreign_store(tmp_path):
    sbs = StoreBackedSource(CountingFake(), root=tmp_path)
    sbs.daily_bars_raw("AAA", "2024-01-01", "2024-01-31")
    foreign = StoreBackedSource(PolygonSource(session=Mock()), root=tmp_path)
    with pytest.raises(DataSourceError, match="family"):
        foreign.daily_bars_raw("AAA", "2024-01-01", "2024-01-31")


def test_store_fetches_bypass_the_http_cache(tmp_path):
    config.set_polygon_key("test-key")
    raw = [("2024-01-02", 100.0), ("2024-01-03", 101.0)]
    session = _routing_session(raw)
    sbs = StoreBackedSource(PolygonSource(session=session), root=tmp_path)
    out = sbs.daily_bars_split_adjusted("AAA", "2024-01-02", "2024-01-03")
    assert len(out) == 2 and session.get.call_count >= 1
    log = get_cache().access_log()
    assert not log["namespace"].astype(str).str.startswith("PolygonSource.").any()
    assert get_cache().stats()["misses"] == 0


def test_dividends_pass_through_on_a_non_raw_cash_basis(tmp_path, monkeypatch):
    monkeypatch.setattr(adjust, "DIVIDEND_CASH_BASIS", "split_adjusted")
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    sbs.dividends("AAA", "2024-01-01", "2024-06-30")
    sbs.dividends("AAA", "2024-01-01", "2024-06-30")
    assert upstream.calls[("dividends", "AAA")] == 2
    assert not (tmp_path / "dividends").exists()


def test_unknown_attributes_and_hourly_bars_delegate(tmp_path):
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    assert sbs.market is upstream.market
    pd.testing.assert_frame_equal(
        sbs.hourly_bars("AAA", "2024-01-02", "2024-01-05"),
        upstream.hourly_bars("AAA", "2024-01-02", "2024-01-05"),
        check_freq=False,
    )


def test_precompute_rerun_makes_zero_upstream_calls(tmp_path):
    # SPY included because iv_index only exists for the SPY proxy; the
    # window ends before the pinned L, so the second run is fully stored.
    root = tmp_path / "raw"
    out_dir = tmp_path / "processed"
    upstream = CountingFake()
    precompute(
        ["AAA", "BBB", "SPY"],
        ["ret_1d", "rv_cc_21d", "beta_ols_60d", "dollar_volume_1d", "iv_index"],
        "2024-01-01", "2024-06-01",
        out_dir=str(out_dir),
        source=StoreBackedSource(upstream, root=root),
    )
    second = CountingFake()
    precompute(
        ["AAA", "BBB", "SPY"],
        ["ret_1d", "rv_cc_21d", "beta_ols_60d", "dollar_volume_1d", "iv_index"],
        "2024-01-01", "2024-06-01",
        out_dir=str(tmp_path / "processed2"),
        source=StoreBackedSource(second, root=root),
    )
    for name in ("ret_1d", "rv_cc_21d", "beta_ols_60d", "dollar_volume_1d", "iv_index"):
        one = pd.read_parquet(out_dir / f"{name}.parquet")
        two = pd.read_parquet(tmp_path / "processed2" / f"{name}.parquet")
        pd.testing.assert_frame_equal(one, two)
    assert second.calls == {}  # a rerun on the same root needs zero upstream calls


def test_entirely_live_window_returns_no_rows_before_start(tmp_path, monkeypatch):
    # L pinned so the whole request is after it: no stored/live mixing at all.
    monkeypatch.setattr(
        "data_universe.store.source._last_complete_date",
        lambda: pd.Timestamp("2024-06-28"),
    )
    upstream = CountingFake()
    sbs = StoreBackedSource(upstream, root=tmp_path)
    sbs.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-03-29")
    out = sbs.daily_bars_split_adjusted("AAA", "2024-07-01", "2024-07-03")
    pd.testing.assert_frame_equal(
        out, upstream.daily_bars_split_adjusted("AAA", "2024-07-01", "2024-07-03"),
        check_freq=False,
    )
    assert (_row_dates(out, None) >= pd.Timestamp("2024-07-01")).all()
    pd.testing.assert_frame_equal(
        sbs.daily_bars("AAA", "2024-07-01", "2024-07-03"),
        adjust.apply_total_return(
            upstream.daily_bars_split_adjusted("AAA", "2024-07-01", "2024-07-03"),
            adjust.dividend_factors(
                upstream.daily_bars_raw("AAA", "2024-07-01", "2024-07-03")["close"],
                upstream.dividends("AAA", "2024-07-01", "2024-07-03"),
            ),
        ),
        check_freq=False,
    )
