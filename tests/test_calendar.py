import pandas as pd
import pytest

from data_universe.calendar import next_session, previous_session, sessions, shift_sessions
from data_universe.sources.fakes import FakePolygonSource


@pytest.fixture
def session_index():
    source = FakePolygonSource()
    return sessions(source, "2024-01-01", "2024-01-31")


def test_sessions_is_tz_naive_and_named_date():
    source = FakePolygonSource()
    idx = sessions(source, "2024-01-01", "2024-01-10")
    assert idx.tz is None
    assert idx.name == "date"


def test_sessions_excludes_weekends():
    source = FakePolygonSource()
    idx = sessions(source, "2024-01-01", "2024-01-10")  # 2024-01-06/07 is a Sat/Sun
    assert pd.Timestamp("2024-01-06") not in idx
    assert pd.Timestamp("2024-01-07") not in idx
    assert pd.Timestamp("2024-01-08") in idx


def test_sessions_uses_configured_proxy_ticker():
    source = FakePolygonSource()
    default_proxy = sessions(source, "2024-01-01", "2024-01-10")
    other_proxy = sessions(source, "2024-01-01", "2024-01-10", proxy_ticker="QQQ")
    # Same fake calendar (bdate_range) regardless of ticker -- this just proves
    # the call succeeds with a non-default proxy and returns the same session set.
    assert list(default_proxy) == list(other_proxy)


def test_next_session_returns_next_business_day(session_index):
    friday = pd.Timestamp("2024-01-05")
    result = next_session(session_index, friday)
    assert result == pd.Timestamp("2024-01-08")


def test_next_session_returns_none_past_end_of_calendar(session_index):
    result = next_session(session_index, pd.Timestamp("2024-01-31"))
    assert result is None


def test_previous_session_returns_prior_business_day(session_index):
    monday = pd.Timestamp("2024-01-08")
    result = previous_session(session_index, monday)
    assert result == pd.Timestamp("2024-01-05")


def test_previous_session_returns_none_before_start_of_calendar(session_index):
    result = previous_session(session_index, pd.Timestamp("2024-01-01"))
    assert result is None


def test_shift_sessions_forward(session_index):
    start = pd.Timestamp("2024-01-02")
    result = shift_sessions(session_index, start, 2)
    assert result == pd.Timestamp("2024-01-04")


def test_shift_sessions_backward(session_index):
    start = pd.Timestamp("2024-01-10")
    result = shift_sessions(session_index, start, -2)
    assert result == pd.Timestamp("2024-01-08")


def test_shift_sessions_out_of_range_returns_none(session_index):
    result = shift_sessions(session_index, pd.Timestamp("2024-01-30"), 10)
    assert result is None
