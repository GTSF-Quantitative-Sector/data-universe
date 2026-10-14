import pandas as pd
import pytest

from data_universe import config
from data_universe.membership import store
from data_universe.membership.errors import MembershipValidationError
from data_universe.membership.intervals import membership_intervals
from data_universe.membership.schema import SECURITY_MAP_COLUMNS, TICKER_INTERVALS_COLUMNS


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def _write_tables(tmp_path, ticker_intervals_rows, security_map_rows, tail_changes_rows=None):
    config.set_membership_tables_dir(str(tmp_path))
    store.save_ticker_intervals(
        "sp500", pd.DataFrame(ticker_intervals_rows, columns=TICKER_INTERVALS_COLUMNS)
    )
    store.save_security_map(
        "sp500", pd.DataFrame(security_map_rows, columns=SECURITY_MAP_COLUMNS)
    )
    if tail_changes_rows is not None:
        from data_universe.membership.schema import TAIL_CHANGES_COLUMNS

        store.save_tail_changes(
            "sp500", pd.DataFrame(tail_changes_rows, columns=TAIL_CHANGES_COLUMNS)
        )


def test_single_interval_passes_through(tmp_path):
    _write_tables(
        tmp_path,
        [["sp500", "AAA", "2020-01-01", None]],
        [["sp500", "AAA", "2020-01-01", "AAA-20200101", "new_default", False]],
    )
    result = membership_intervals("sp500")
    assert len(result) == 1
    assert result.iloc[0]["security_id"] == "AAA-20200101"
    assert pd.isna(result.iloc[0]["end_date_excl"])


def test_touching_intervals_for_same_security_id_coalesce(tmp_path):
    # FB -> META: two ticker_intervals rows resolved (by identity, simulated
    # here via security_map) to the SAME security_id must become one row.
    _write_tables(
        tmp_path,
        [
            ["sp500", "FB", "2013-01-01", "2022-06-09"],
            ["sp500", "META", "2022-06-09", None],
        ],
        [
            ["sp500", "FB", "2013-01-01", "META-20130101", "override", True],
            ["sp500", "META", "2022-06-09", "META-20130101", "override", True],
        ],
    )
    result = membership_intervals("sp500")
    assert len(result) == 1
    assert result.iloc[0]["start_date"] == "2013-01-01"
    assert pd.isna(result.iloc[0]["end_date_excl"])


def test_real_gap_for_same_security_id_stays_two_intervals(tmp_path):
    # AMD: left and came back, with a real gap in between. Must NOT coalesce.
    _write_tables(
        tmp_path,
        [
            ["sp500", "AMD", "1996-01-01", "2013-09-23"],
            ["sp500", "AMD", "2017-03-20", None],
        ],
        [
            ["sp500", "AMD", "1996-01-01", "AMD-19960101", "override", True],
            ["sp500", "AMD", "2017-03-20", "AMD-19960101", "override", True],
        ],
    )
    result = membership_intervals("sp500")
    assert len(result) == 2
    assert sorted(result["start_date"]) == ["1996-01-01", "2017-03-20"]


def test_ticker_intervals_row_with_no_security_map_row_raises(tmp_path):
    _write_tables(
        tmp_path,
        [["sp500", "AAA", "2020-01-01", None]],
        [],  # no matching security_map row
    )
    with pytest.raises(MembershipValidationError):
        membership_intervals("sp500")


def test_overlapping_intervals_for_same_security_id_raises(tmp_path):
    _write_tables(
        tmp_path,
        [
            ["sp500", "AAA", "2020-01-01", "2020-06-01"],
            ["sp500", "BBB", "2020-03-01", None],
        ],
        [
            ["sp500", "AAA", "2020-01-01", "DUP-20200101", "override", True],
            ["sp500", "BBB", "2020-03-01", "DUP-20200101", "override", True],
        ],
    )
    with pytest.raises(MembershipValidationError):
        membership_intervals("sp500")
