from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from data_universe.features import registry
from data_universe.precompute import load_feature, precompute
from data_universe.sources.fakes import FakePolygonSource, FakeRateSource


def test_precompute_writes_one_feature_one_ticker_to_parquet(tmp_path):
    source = FakePolygonSource()
    report = precompute(
        ["AAPL"],
        ["ret_1d"],
        "2024-01-02",
        "2024-01-31",
        out_dir=str(tmp_path),
        source=source,
    )
    assert report.failed == {}
    assert report.succeeded[("AAPL", "ret_1d")] > 0

    out_file = tmp_path / "ret_1d.parquet"
    assert out_file.exists()
    frame = pd.read_parquet(out_file)
    assert list(frame.columns) == ["date", "ticker", "value"]
    assert (frame["ticker"] == "AAPL").all()
    assert frame["value"].notna().all()
    assert len(frame) == report.succeeded[("AAPL", "ret_1d")]


def test_precompute_rerun_upserts_instead_of_duplicating(tmp_path):
    source = FakePolygonSource()
    precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )
    first_len = len(pd.read_parquet(tmp_path / "ret_1d.parquet"))

    # Rerun the exact same window -- should not duplicate rows.
    report = precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )
    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert len(frame) == first_len
    assert report.succeeded[("AAPL", "ret_1d")] == first_len

    # Extend the window -- new dates get merged in, old ones untouched, no dupes.
    precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-02-15", out_dir=str(tmp_path), source=source
    )
    frame2 = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert len(frame2) > first_len
    assert not frame2.duplicated(subset=["date", "ticker"]).any()


def test_precompute_merges_across_tickers_for_same_feature(tmp_path):
    source = FakePolygonSource()
    precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )
    precompute(
        ["MSFT"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )

    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}
    assert not frame.duplicated(subset=["date", "ticker"]).any()


class _RaisingSource:
    """Wraps a real source but raises for one specific ticker, to test isolation."""

    def __init__(self, inner, bad_ticker: str) -> None:
        self._inner = inner
        self._bad_ticker = bad_ticker

    def daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker == self._bad_ticker:
            raise ValueError(f"simulated failure for {ticker}")
        return self._inner.daily_bars(ticker, start, end)

    def index_daily_bars(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        return self._inner.index_daily_bars(ticker, start, end)


def test_precompute_isolates_unregistered_feature_name(tmp_path):
    source = FakePolygonSource()
    report = precompute(
        ["AAPL"], ["not_a_real_feature"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert report.succeeded == {}
    assert "not_a_real_feature" in report.failed[("AAPL", "not_a_real_feature")]
    assert not (tmp_path / "not_a_real_feature.parquet").exists()


def test_precompute_isolates_one_ticker_failure_from_the_rest(tmp_path):
    source = _RaisingSource(FakePolygonSource(), bad_ticker="BADCO")
    report = precompute(
        ["AAPL", "BADCO"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert ("AAPL", "ret_1d") in report.succeeded
    assert "simulated failure for BADCO" in report.failed[("BADCO", "ret_1d")]

    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert set(frame["ticker"]) == {"AAPL"}


def test_load_feature_reads_back_precomputed_values(tmp_path):
    source = FakePolygonSource()
    precompute(
        ["AAPL", "MSFT"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )

    frame = load_feature("ret_1d", out_dir=str(tmp_path))
    assert list(frame.columns) == ["date", "ticker", "value"]
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}

    filtered = load_feature("ret_1d", tickers=["AAPL"], out_dir=str(tmp_path))
    assert set(filtered["ticker"]) == {"AAPL"}


def test_load_feature_raises_a_clear_error_when_nothing_precomputed(tmp_path):
    with pytest.raises(FileNotFoundError, match="ret_1d"):
        load_feature("ret_1d", out_dir=str(tmp_path))


def test_precompute_accepts_a_label_name(tmp_path):
    source = FakePolygonSource()
    report = precompute(
        ["AAPL"], ["fwd_ret_15d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    assert report.failed == {}
    assert report.succeeded[("AAPL", "fwd_ret_15d")] > 0
    frame = pd.read_parquet(tmp_path / "fwd_ret_15d.parquet")
    assert (frame["ticker"] == "AAPL").all()


def test_precompute_forwards_rate_source_to_compute(tmp_path):
    source = FakePolygonSource()
    report = precompute(
        ["AAPL"], ["ret_excess_1d"], "2024-03-01", "2024-03-15",
        out_dir=str(tmp_path), source=source,
        rate_source=FakeRateSource(annual_rate=0.05),
    )
    assert report.failed == {}
    frame = pd.read_parquet(tmp_path / "ret_excess_1d.parquet")
    raw = registry.compute(
        registry.FEATURE_REGISTRY, "ret_1d", "AAPL", "2024-03-01", "2024-03-15", source
    )
    expected = (raw - 0.05 / 252.0).dropna().rename("value")
    written = frame.set_index("date")["value"].sort_index()
    pd.testing.assert_series_equal(written, expected, check_freq=False)


def test_precompute_ret_excess_1d_without_rate_source_fails_loudly(tmp_path):
    report = precompute(
        ["AAPL"], ["ret_excess_1d"], "2024-03-01", "2024-03-15",
        out_dir=str(tmp_path), source=FakePolygonSource(),
    )
    assert ("AAPL", "ret_excess_1d") in report.failed
    assert "rate_source" in report.failed[("AAPL", "ret_excess_1d")]
    assert not (tmp_path / "ret_excess_1d.parquet").exists()


def test_precompute_concurrent_writers_all_land(tmp_path):
    source = FakePolygonSource()
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]

    def _run(ticker):
        return precompute(
            [ticker], ["ret_1d"], "2024-01-02", "2024-01-31",
            out_dir=str(tmp_path), source=source,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(_run, tickers))

    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert set(frame["ticker"]) == set(tickers)
    assert not frame.duplicated(subset=["date", "ticker"]).any()


def test_failed_write_leaves_existing_parquet_and_no_temp_files(tmp_path, monkeypatch):
    source = FakePolygonSource()
    precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31",
        out_dir=str(tmp_path), source=source,
    )
    before = (tmp_path / "ret_1d.parquet").read_bytes()

    def _boom(self, *args, **kwargs):
        raise RuntimeError("simulated disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _boom)
    with pytest.raises(RuntimeError, match="disk full"):
        precompute(
            ["MSFT"], ["ret_1d"], "2024-01-02", "2024-02-15",
            out_dir=str(tmp_path), source=source,
        )
    assert (tmp_path / "ret_1d.parquet").read_bytes() == before
    assert list(tmp_path.glob("*.tmp")) == []
