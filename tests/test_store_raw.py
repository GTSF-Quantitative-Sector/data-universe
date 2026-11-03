"""`RawStore` round-trips: parquet fidelity, coverage merges, schema
mismatch, quoted keys, atomic writes, invalidate, status."""
import json

import pandas as pd

from data_universe.sources.fakes import FakePolygonSource
from data_universe.store.raw import STORE_SCHEMA_VERSION, RawStore, _fetch_record


def _record(start, end, status="ok", rows=0, rebased=False, family="fake"):
    return _fetch_record(start, end, status, rows, "Fake", family, rebased)


def test_parquet_round_trip_is_exact_for_bars_and_dividends(tmp_path):
    store = RawStore(tmp_path)
    source = FakePolygonSource()
    bars = source.daily_bars_split_adjusted("AAA", "2024-01-01", "2024-01-31")
    store.replace_all("daily_bars_split_adjusted", "AAA", bars,
                      _record("2024-01-01", "2024-01-31", rows=len(bars)))
    out = store.read("daily_bars_split_adjusted", "AAA")
    pd.testing.assert_frame_equal(out, bars, check_freq=False)

    dividends = source.dividends("AAA", "2024-01-01", "2024-06-01")
    if dividends.empty:
        dividends = pd.DataFrame(
            {
                "ex_dividend_date": [pd.Timestamp("2024-02-07")],
                "cash_amount": [0.25],
                "currency": ["USD"],
                "dividend_type": ["CD"],
            }
        )
    store.replace_all("dividends", "AAA", dividends,
                      _record("2024-01-01", "2024-06-01", rows=len(dividends)))
    out = store.read("dividends", "AAA")
    pd.testing.assert_frame_equal(
        out.reset_index(drop=True), dividends.reset_index(drop=True))


def test_coverage_merges_overlapping_adjacent_and_keeps_only_ok_empty(tmp_path):
    store = RawStore(tmp_path)
    for start, end, status in (
        ("2024-01-01", "2024-01-10", "ok"),
        ("2024-01-11", "2024-01-20", "empty"),
        ("2024-02-01", "2024-02-05", "ok"),
        ("2024-03-01", "2024-03-02", "not_entitled"),
    ):
        store.record("daily_bars_raw", "AAA", _record(start, end, status))
    coverage = store.coverage("daily_bars_raw", "AAA")
    assert coverage == [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-20")),
        (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-05")),
    ]


def test_schema_mismatch_reads_nothing_and_next_upsert_discards(tmp_path):
    store = RawStore(tmp_path)
    store.replace_all(
        "daily_bars_raw", "AAA", FakePolygonSource().daily_bars_raw(
            "AAA", "2024-01-01", "2024-01-31"),
        _record("2024-01-01", "2024-01-31"),
    )
    manifest_path = tmp_path / "daily_bars_raw" / "AAA.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = STORE_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    assert store.read("daily_bars_raw", "AAA") is None
    assert store.coverage("daily_bars_raw", "AAA") == []
    store.upsert_range(
        "daily_bars_raw", "AAA", "2024-01-01", "2024-01-10",
        FakePolygonSource().daily_bars_raw("AAA", "2024-01-01", "2024-01-10"),
        None, _record("2024-01-01", "2024-01-10"),
    )
    assert len(store.read("daily_bars_raw", "AAA")) == len(
        FakePolygonSource().daily_bars_raw("AAA", "2024-01-01", "2024-01-10")
    )  # old Jan 11-31 rows discarded, not merged
    assert store.coverage("daily_bars_raw", "AAA") == [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10"))
    ]
    assert len(store.manifest("daily_bars_raw", "AAA")["fetches"]) == 1


def test_record_on_a_mismatched_manifest_removes_stale_data(tmp_path):
    store = RawStore(tmp_path)
    store.replace_all(
        "daily_bars_raw", "AAA", FakePolygonSource().daily_bars_raw(
            "AAA", "2024-01-01", "2024-01-31"),
        _record("2024-01-01", "2024-01-31"),
    )
    manifest_path = tmp_path / "daily_bars_raw" / "AAA.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = STORE_SCHEMA_VERSION + 1
    manifest_path.write_text(json.dumps(manifest))
    store.record("daily_bars_raw", "AAA",
                 _record("2024-02-01", "2024-02-01", "not_entitled"))
    assert not (tmp_path / "daily_bars_raw" / "AAA.parquet").exists()
    assert store.read("daily_bars_raw", "AAA") is None


def test_colon_key_is_quoted_and_round_trips(tmp_path):
    store = RawStore(tmp_path)
    source = FakePolygonSource()
    vix = source.index_daily_bars("I:VIX", "2024-01-01", "2024-01-10")
    store.replace_all("index_daily_bars", "I:VIX", vix,
                      _record("2024-01-01", "2024-01-10", rows=len(vix)))
    paths = list((tmp_path / "index_daily_bars").iterdir())
    assert all(":" not in p.name for p in paths)
    pd.testing.assert_frame_equal(
        store.read("index_daily_bars", "I:VIX"), vix, check_freq=False)


def test_no_temp_files_left_after_writes(tmp_path):
    store = RawStore(tmp_path)
    bars = FakePolygonSource().daily_bars_raw("AAA", "2024-01-01", "2024-01-10")
    store.upsert_range("daily_bars_raw", "AAA", "2024-01-01", "2024-01-10",
                       bars, None, _record("2024-01-01", "2024-01-10"))
    leftovers = [
        p.name for p in (tmp_path / "daily_bars_raw").iterdir()
        if p.name.endswith(".tmp") or p.name.startswith(".AAA-")
    ]
    assert leftovers == []


def test_upsert_range_replaces_rows_only_inside_the_window(tmp_path):
    store = RawStore(tmp_path)
    source = FakePolygonSource()
    bars = source.daily_bars_raw("AAA", "2024-01-01", "2024-02-29")
    store.replace_all("daily_bars_raw", "AAA", bars,
                      _record("2024-01-01", "2024-02-29", rows=len(bars)))
    shifted = bars.copy()
    shifted["close"] = -1.0
    store.upsert_range("daily_bars_raw", "AAA", "2024-02-01", "2024-02-29",
                       shifted, None, _record("2024-02-01", "2024-02-29"))
    out = store.read("daily_bars_raw", "AAA")
    assert len(out) == len(bars)
    assert (out.loc[out.index < pd.Timestamp("2024-02-01", tz="America/New_York"),
                  "close"] != -1.0).all()
    assert (out.loc[out.index >= pd.Timestamp("2024-02-01", tz="America/New_York"),
                  "close"] == -1.0).all()


def test_invalidate_removes_data_and_manifest(tmp_path):
    store = RawStore(tmp_path)
    bars = FakePolygonSource().daily_bars_raw("AAA", "2024-01-01", "2024-01-10")
    store.replace_all("daily_bars_raw", "AAA", bars,
                      _record("2024-01-01", "2024-01-10"))
    store.invalidate("daily_bars_raw", "AAA")
    assert store.read("daily_bars_raw", "AAA") is None
    assert store.coverage("daily_bars_raw", "AAA") == []
    assert store.manifest("daily_bars_raw", "AAA")["fetches"] == []


def test_status_reports_one_row_per_manifest(tmp_path):
    store = RawStore(tmp_path)
    bars = FakePolygonSource().daily_bars_raw("AAA", "2024-01-01", "2024-01-10")
    store.replace_all("daily_bars_raw", "AAA", bars,
                      _record("2024-01-01", "2024-01-10", rows=len(bars)))
    status = store.status()
    assert list(status.columns) == [
        "dataset", "key", "family", "coverage_start", "coverage_end",
        "n_intervals", "n_fetches", "last_fetched_at", "last_status",
        "schema_version",
    ]
    row = status.iloc[0]
    assert row["dataset"] == "daily_bars_raw" and row["key"] == "AAA"
    assert row["coverage_start"] == pd.Timestamp("2024-01-01")
    assert row["coverage_end"] == pd.Timestamp("2024-01-10")
    assert row["n_intervals"] == 1 and row["n_fetches"] == 1
    assert row["last_status"] == "ok"
    assert row["schema_version"] == STORE_SCHEMA_VERSION
