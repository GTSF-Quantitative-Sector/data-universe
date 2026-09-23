import pandas as pd
import pytest

from data_universe.precompute import load_feature, precompute
from data_universe.sources.fakes import FakePolygonSource


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
