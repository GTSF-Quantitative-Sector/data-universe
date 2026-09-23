import pandas as pd

from data_universe import precompute as pc
from data_universe.sources.fakes import FakePolygonSource


def test_precompute_writes_one_feature_one_ticker_to_parquet(tmp_path):
    source = FakePolygonSource()
    report = pc.precompute(
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
    pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )
    first_len = len(pd.read_parquet(tmp_path / "ret_1d.parquet"))

    # Rerun the exact same window -- should not duplicate rows.
    report = pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )
    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert len(frame) == first_len
    assert report.succeeded[("AAPL", "ret_1d")] == first_len

    # Extend the window -- new dates get merged in, old ones untouched, no dupes.
    pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-02-15", out_dir=str(tmp_path), source=source
    )
    frame2 = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert len(frame2) > first_len
    assert not frame2.duplicated(subset=["date", "ticker"]).any()


def test_precompute_merges_across_tickers_for_same_feature(tmp_path):
    source = FakePolygonSource()
    pc.precompute(
        ["AAPL"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )
    pc.precompute(
        ["MSFT"], ["ret_1d"], "2024-01-02", "2024-01-31", out_dir=str(tmp_path), source=source
    )

    frame = pd.read_parquet(tmp_path / "ret_1d.parquet")
    assert set(frame["ticker"]) == {"AAPL", "MSFT"}
    assert not frame.duplicated(subset=["date", "ticker"]).any()
