import pytest

from data_universe.sources import sec_adapter


def test_sp500_raises_helpful_import_error_without_sec_parser():
    # sec_parser is not installed in this environment (it's an optional
    # extra), so this exercises the real failure path, not a mock.
    with pytest.raises(ImportError) as exc_info:
        sec_adapter.sp500(query_date="2019-01-01")
    assert "pip install" in str(exc_info.value)
    assert ".[sec]" in str(exc_info.value)


def test_fundamentals_raises_helpful_import_error_without_sec_parser():
    with pytest.raises(ImportError) as exc_info:
        sec_adapter.fundamentals("AAPL")
    assert "pip install" in str(exc_info.value)


def test_sp500_delegates_to_sec_parser_lookups(monkeypatch):
    calls = []

    class FakeLookups:
        @staticmethod
        def get_sp500_tickers(query_date=None):
            calls.append(query_date)
            return ["AAPL", "MSFT"]

    monkeypatch.setattr(sec_adapter, "_import_sec_parser", lambda: FakeLookups)
    result = sec_adapter.sp500(query_date="2019-01-01")
    assert result == ["AAPL", "MSFT"]
    assert calls == ["2019-01-01"]


def test_sp500_defaults_query_date_to_none(monkeypatch):
    calls = []

    class FakeLookups:
        @staticmethod
        def get_sp500_tickers(query_date=None):
            calls.append(query_date)
            return []

    monkeypatch.setattr(sec_adapter, "_import_sec_parser", lambda: FakeLookups)
    sec_adapter.sp500()
    assert calls == [None]
