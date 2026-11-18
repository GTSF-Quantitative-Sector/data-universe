"""Tests for `store.derived.DerivedStore`: versioned per-key parquet +
coverage.json layout, atomic replace_range/invalidate, quoting, locking."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from data_universe.store.derived import DERIVED_SCHEMA_VERSION, DerivedStore


def _frame(dates, values=None, **extra):
    n = len(dates)
    data = {"date": pd.to_datetime(dates)}
    if values is None:
        values = list(range(n))
    data["value"] = values
    data.update(extra)
    return pd.DataFrame(data)


def _manifest(name="feat", fingerprint="abc123", **overrides):
    manifest = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "name": name,
        "kind": "feature",
        "fingerprint": fingerprint,
        "spec": {"schema_version": 1, "name": name},
        "key_column": "ticker",
        "date_column": "date",
        "columns": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "tool_version": "0.13.0",
    }
    manifest.update(overrides)
    return manifest


def _record(start, end, rows=1):
    return {
        "start": start,
        "end": end,
        "rows": rows,
        "computed_at": "2026-01-01T00:00:00+00:00",
        "tool_version": "0.13.0",
        "source": "FakePolygonSource",
    }


def test_replace_range_drops_only_rows_inside_the_window(tmp_path):
    store = DerivedStore(tmp_path)
    store.replace_range(
        "feat", "fp1", "AAPL", "2024-01-10", "2024-01-20",
        _frame(["2024-01-10", "2024-01-15"], [1.0, 2.0]),
        _record("2024-01-10", "2024-01-20"),
    )
    store.replace_range(
        "feat", "fp1", "AAPL", "2024-01-20", "2024-01-31",
        _frame(["2024-01-20", "2024-01-25"], [3.0, 4.0]),
        _record("2024-01-20", "2024-01-31"),
    )
    # Overwrite just the middle: 01-15 and 01-20 rows are dropped, the rest stay.
    store.replace_range(
        "feat", "fp1", "AAPL", "2024-01-14", "2024-01-21",
        _frame(["2024-01-16"], [9.0]),
        _record("2024-01-14", "2024-01-21"),
    )
    frame = store.read("feat", "fp1", "AAPL")
    assert frame["value"].tolist() == [1.0, 9.0, 4.0]
    assert frame["date"].tolist() == list(pd.to_datetime(
        ["2024-01-10", "2024-01-16", "2024-01-25"]))
    coverage = store.coverage("feat", "fp1", "AAPL")
    assert coverage == [
        (pd.Timestamp("2024-01-10"), pd.Timestamp("2024-01-31"))
    ]


def test_replace_range_empty_result_unlinks_the_data_file(tmp_path):
    store = DerivedStore(tmp_path)
    store.replace_range(
        "feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
        _frame(["2024-01-05"]), _record("2024-01-01", "2024-01-31"),
    )
    data_path = tmp_path / "feat" / "fp1" / "AAPL.parquet"
    assert data_path.exists()
    store.replace_range(
        "feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
        _frame([], []), _record("2024-01-01", "2024-01-31", rows=0),
    )
    assert not data_path.exists()
    # The coverage record still landed -- the range was computed, just empty.
    assert len(store.records("feat", "fp1", "AAPL")) == 2


def test_coverage_merges_adjacent_within_one_day(tmp_path):
    store = DerivedStore(tmp_path)
    store.replace_range("f", "fp", "K", "2024-01-01", "2024-01-05",
                        _frame([]), _record("2024-01-01", "2024-01-05"))
    store.replace_range("f", "fp", "K", "2024-01-06", "2024-01-10",
                        _frame([]), _record("2024-01-06", "2024-01-10"))
    store.replace_range("f", "fp", "K", "2024-02-01", "2024-02-05",
                        _frame([]), _record("2024-02-01", "2024-02-05"))
    assert store.coverage("f", "fp", "K") == [
        (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-10")),
        (pd.Timestamp("2024-02-01"), pd.Timestamp("2024-02-05")),
    ]


def test_invalidate_whole_key_removes_data_and_coverage(tmp_path):
    store = DerivedStore(tmp_path)
    store.ensure_dataset("feat", "fp1", _manifest())
    store.replace_range("feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
                        _frame(["2024-01-05"]), _record("2024-01-01", "2024-01-31"))
    assert store.invalidate("feat", "fp1", "AAPL") is True
    assert store.read("feat", "fp1", "AAPL") is None
    assert store.records("feat", "fp1", "AAPL") == []
    assert store.invalidate("feat", "fp1", "AAPL") is False


def test_invalidate_date_range_splits_coverage_records(tmp_path):
    store = DerivedStore(tmp_path)
    store.ensure_dataset("feat", "fp1", _manifest())
    store.replace_range(
        "feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
        _frame(["2024-01-05", "2024-01-15", "2024-01-25"], [1, 2, 3]),
        _record("2024-01-01", "2024-01-31", rows=3),
    )
    assert store.invalidate("feat", "fp1", "AAPL", "2024-01-10", "2024-01-20")
    frame = store.read("feat", "fp1", "AAPL")
    assert frame["value"].tolist() == [1, 3]
    records = store.records("feat", "fp1", "AAPL")
    assert [(r["start"], r["end"]) for r in records] == [
        ("2024-01-01", "2024-01-09"),
        ("2024-01-21", "2024-01-31"),
    ]
    # Metadata is preserved on the split records.
    assert records[0]["tool_version"] == "0.13.0"
    assert records[0]["source"] == "FakePolygonSource"


def test_invalidate_open_ended_range(tmp_path):
    store = DerivedStore(tmp_path)
    store.replace_range("f", "fp", "K", "2024-01-01", "2024-03-31",
                        _frame(["2024-01-10", "2024-03-01"]),
                        _record("2024-01-01", "2024-03-31"))
    store.invalidate("f", "fp", "K", start="2024-02-01")
    frame = store.read("f", "fp", "K")
    assert frame["date"].tolist() == list(pd.to_datetime(["2024-01-10"]))
    records = store.records("f", "fp", "K")
    assert [(r["start"], r["end"]) for r in records] == [("2024-01-01", "2024-01-31")]


def test_keys_unquotes_special_characters(tmp_path):
    store = DerivedStore(tmp_path)
    for key in ("BRK.B", "I:VIX", "plain"):
        store.replace_range("f", "fp", key, "2024-01-01", "2024-01-05",
                            _frame(["2024-01-02"]), _record("2024-01-01", "2024-01-05"))
    assert store.keys("f", "fp") == ["BRK.B", "I:VIX", "plain"]
    assert store.read("f", "fp", "I:VIX") is not None


def test_set_columns_sets_once_then_rejects_differences(tmp_path):
    store = DerivedStore(tmp_path)
    store.ensure_dataset("feat", "fp1", _manifest())
    store.set_columns("feat", "fp1", ["date", "ticker", "value"])
    manifest = store.dataset_manifest("feat", "fp1")
    assert manifest["columns"] == ["date", "ticker", "value"]
    store.set_columns("feat", "fp1", ["date", "ticker", "value"])  # same: fine
    with pytest.raises(ValueError, match="bump the version"):
        store.set_columns("feat", "fp1", ["date", "ticker", "other"])


def test_wrong_schema_version_coverage_reads_as_absent(tmp_path):
    store = DerivedStore(tmp_path)
    store.ensure_dataset("feat", "fp1", _manifest())
    store.replace_range("feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
                        _frame(["2024-01-05"]), _record("2024-01-01", "2024-01-31"))
    coverage_path = tmp_path / "feat" / "fp1" / "AAPL.coverage.json"
    payload = json.loads(coverage_path.read_text())
    payload["schema_version"] = 999
    coverage_path.write_text(json.dumps(payload))
    assert store.records("feat", "fp1", "AAPL") == []
    assert store.coverage("feat", "fp1", "AAPL") == []
    # The data file itself still reads -- only coverage is schema-gated.
    assert store.read("feat", "fp1", "AAPL") is not None


def test_failed_parquet_write_leaves_old_file_and_no_temp_files(tmp_path, monkeypatch):
    store = DerivedStore(tmp_path)
    store.replace_range("feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
                        _frame(["2024-01-05"], [1.0]),
                        _record("2024-01-01", "2024-01-31"))
    data_path = tmp_path / "feat" / "fp1" / "AAPL.parquet"
    before = data_path.read_bytes()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        store.replace_range("feat", "fp1", "AAPL", "2024-02-01", "2024-02-29",
                            _frame(["2024-02-05"], [2.0]),
                            _record("2024-02-01", "2024-02-29"))
    assert data_path.read_bytes() == before
    assert list((tmp_path / "feat" / "fp1").glob("*.tmp")) == []
    # The failed coverage write never landed either.
    assert len(store.records("feat", "fp1", "AAPL")) == 1


def test_versions_names_and_status(tmp_path):
    store = DerivedStore(tmp_path)
    assert store.names() == []
    assert store.status().empty
    store.ensure_dataset("feat", "fp1", _manifest("feat", "fp1"))
    store.ensure_dataset("feat", "fp2", _manifest("feat", "fp2",
                                                  created_at="2026-02-01T00:00:00+00:00"))
    store.ensure_dataset("other", "fp9", _manifest("other", "fp9"))
    store.replace_range("feat", "fp1", "AAPL", "2024-01-01", "2024-01-31",
                        _frame(["2024-01-05", "2024-01-10"]),
                        _record("2024-01-01", "2024-01-31", rows=2))
    store.set_columns("feat", "fp1", ["date", "value"])

    assert store.names() == ["feat", "other"]
    versions = store.versions("feat")
    assert [v["fingerprint"] for v in versions] == ["fp1", "fp2"]

    status = store.status()
    assert list(status.columns) == [
        "name", "kind", "fingerprint", "schema_version", "key",
        "coverage_start", "coverage_end", "n_intervals", "rows",
        "last_computed_at", "tool_version",
    ]
    row = status[(status["name"] == "feat") & (status["fingerprint"] == "fp1")].iloc[0]
    assert row["key"] == "AAPL"
    assert row["kind"] == "feature"
    assert row["schema_version"] == DERIVED_SCHEMA_VERSION
    assert row["coverage_start"] == pd.Timestamp("2024-01-01")
    assert row["coverage_end"] == pd.Timestamp("2024-01-31")
    assert row["n_intervals"] == 1
    assert row["rows"] == 2
    assert row["tool_version"] == "0.13.0"
