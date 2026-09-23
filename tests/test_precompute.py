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
