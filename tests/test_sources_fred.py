import traceback
from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources.base import DataSourceError
from data_universe.sources.fred import FredSource


@pytest.fixture(autouse=True)
def _reset_state():
    config.reset()
    reset_cache()
    config.set_fred_key("test-fred-key")
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, json_data):
        self._json_data = json_data

    def json(self):
        return self._json_data

    def raise_for_status(self):
        pass


def test_daily_rate_raises_without_a_key():
    config.set_fred_key(None)
    source = FredSource(session=Mock())
    with pytest.raises(ValueError):
        source.daily_rate()


def test_daily_rate_converts_percent_to_decimal_daily_rate():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"observations": [{"date": "2024-01-02", "value": "5.25"}]}
    )
    source = FredSource(session=session)
    rate = source.daily_rate()
    assert rate.index[0] == pd.Timestamp("2024-01-02")
    assert rate.iloc[0] == pytest.approx(5.25 / 100 / 252)


def test_daily_rate_skips_missing_observations():
    session = Mock()
    session.get.return_value = FakeResponse(
        {
            "observations": [
                {"date": "2024-01-01", "value": "."},
                {"date": "2024-01-02", "value": "5.25"},
            ]
        }
    )
    source = FredSource(session=session)
    rate = source.daily_rate()
    assert len(rate) == 1
    assert rate.index[0] == pd.Timestamp("2024-01-02")


def test_daily_rate_sends_api_key_and_series_id_as_params():
    session = Mock()
    session.get.return_value = FakeResponse({"observations": []})
    source = FredSource(session=session)
    source.daily_rate(series_id="DGS3MO", start="2024-01-01", end="2024-02-01")
    _, kwargs = session.get.call_args
    params = kwargs["params"]
    assert params["api_key"] == "test-fred-key"
    assert params["series_id"] == "DGS3MO"
    assert params["observation_start"] == "2024-01-01"
    assert params["observation_end"] == "2024-02-01"


def test_daily_rate_defaults_to_dtb3():
    session = Mock()
    session.get.return_value = FakeResponse({"observations": []})
    source = FredSource(session=session)
    source.daily_rate()
    _, kwargs = session.get.call_args
    assert kwargs["params"]["series_id"] == "DTB3"


def test_repeat_call_is_served_from_cache():
    session = Mock()
    session.get.return_value = FakeResponse(
        {"observations": [{"date": "2024-01-02", "value": "5.0"}]}
    )
    source = FredSource(session=session)
    source.daily_rate()
    source.daily_rate()
    assert session.get.call_count == 1


def test_daily_rate_http_error_never_leaks_the_api_key():
    # requests' HTTPError message embeds the full request URL, which carries
    # api_key -- it must not survive into the raised error or its chain.
    session = Mock()
    response = Mock()
    response.status_code = 403
    response.raise_for_status.side_effect = requests.HTTPError(
        "403 Client Error: Forbidden for url: "
        "https://api.stlouisfed.org/fred/series/observations?api_key=test-fred-key",
        response=response,
    )
    session.get.return_value = response
    source = FredSource(session=session)
    with pytest.raises(DataSourceError) as excinfo:
        source.daily_rate()
    exc = excinfo.value
    assert "403" in str(exc)
    assert "DTB3" in str(exc)
    assert "test-fred-key" not in str(exc)
    assert exc.__context__ is None
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert "test-fred-key" not in formatted


def test_daily_rate_connection_error_never_leaks_the_api_key():
    session = Mock()
    session.get.side_effect = requests.ConnectionError(
        "Failed to resolve host for url: "
        "https://api.stlouisfed.org/fred/series/observations?api_key=test-fred-key"
    )
    source = FredSource(session=session)
    with pytest.raises(DataSourceError) as excinfo:
        source.daily_rate()
    exc = excinfo.value
    assert "ConnectionError" in str(exc)
    assert "test-fred-key" not in str(exc)
    assert exc.__context__ is None
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert "test-fred-key" not in formatted
