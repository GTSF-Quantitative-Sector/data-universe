from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe.cache import reset_cache
from data_universe.sources.edgar import EdgarSource


@pytest.fixture(autouse=True)
def _reset_state():
    reset_cache()
    yield
    reset_cache()


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def test_get_cik_sends_descriptive_user_agent():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    )
    source = EdgarSource(session=session, user_agent="my-org contact@example.org")
    cik = source.get_cik("AAPL")
    assert cik == "320193"
    _, kwargs = session.get.call_args
    assert kwargs["headers"]["User-Agent"] == "my-org contact@example.org"


def test_get_cik_is_case_insensitive():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
    )
    source = EdgarSource(session=session)
    assert source.get_cik("aapl") == "320193"


def test_get_cik_raises_for_unknown_ticker():
    session = Mock()
    session.get.return_value = FakeResponse({"0": {"cik_str": 1, "ticker": "AAPL", "title": "x"}})
    source = EdgarSource(session=session)
    with pytest.raises(ValueError):
        source.get_cik("NOPE")


def _submissions(rows):
    """rows: list of (form, filing_date, acceptance_datetime, items)."""
    return {
        "filings": {
            "recent": {
                "form": [r[0] for r in rows],
                "filingDate": [r[1] for r in rows],
                "acceptanceDateTime": [r[2] for r in rows],
                "items": [r[3] for r in rows],
            }
        }
    }


def test_earnings_8k_filings_keeps_only_item_202():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions(
            [
                ("8-K", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02,9.01"),
                ("8-K", "2024-02-01", "2024-02-01T16:05:00.000Z", "5.02"),
            ]
        )
    )
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert len(filings) == 1
    assert filings.iloc[0]["filing_date"] == pd.Timestamp("2024-01-02")


def test_earnings_8k_filings_excludes_8ka():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions(
            [
                ("8-K/A", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02"),
                ("8-K", "2024-04-01", "2024-04-01T16:05:00.000Z", "2.02"),
            ]
        )
    )
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert len(filings) == 1
    assert filings.iloc[0]["filing_date"] == pd.Timestamp("2024-04-01")


def test_earnings_8k_filings_sorted_by_acceptance_datetime():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions(
            [
                ("8-K", "2024-04-01", "2024-04-01T16:05:00.000Z", "2.02"),
                ("8-K", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02"),
            ]
        )
    )
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert list(filings["filing_date"]) == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-04-01"),
    ]


def test_earnings_8k_filings_empty_when_no_filings():
    session = Mock()
    session.get.return_value = FakeResponse(_submissions([]))
    source = EdgarSource(session=session)
    filings = source.earnings_8k_filings("320193")
    assert filings.empty
    assert list(filings.columns) == ["filing_date", "acceptance_datetime", "form", "items"]


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse(
        _submissions([("8-K", "2024-01-02", "2024-01-02T16:05:00.000Z", "2.02")])
    )
    source = EdgarSource(session=session)
    source.earnings_8k_filings("320193")
    source.earnings_8k_filings("320193")
    assert session.get.call_count == 1
