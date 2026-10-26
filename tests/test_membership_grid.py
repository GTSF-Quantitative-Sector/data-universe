import pandas as pd
import pytest

from data_universe import config
from data_universe.membership import store
from data_universe.membership.errors import (
    MembershipError,
    MembershipStaleError,
    MembershipStaleWarning,
)
from data_universe.membership.grid import mask_like, members_on, membership_grid, status
from data_universe.membership.schema import SECURITY_MAP_COLUMNS, TICKER_INTERVALS_COLUMNS


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


@pytest.fixture
def two_security_tables(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    store.save_ticker_intervals(
        "sp500",
        pd.DataFrame(
            [
                ["sp500", "AAA", "2020-01-01", "2020-01-10"],
                ["sp500", "BBB", "2020-01-05", None],
            ],
            columns=TICKER_INTERVALS_COLUMNS,
        ),
    )
    store.save_security_map(
        "sp500",
        pd.DataFrame(
            [
                ["sp500", "AAA", "2020-01-01", "AAA-20200101", "override", True],
                ["sp500", "BBB", "2020-01-05", "BBB-20200105", "override", True],
            ],
            columns=SECURITY_MAP_COLUMNS,
        ),
    )
    verified_through = pd.Timestamp.now(tz="UTC").date().isoformat()
    store.save_manifest("sp500", {"verified_through": verified_through})
    return tmp_path


def test_grid_membership_is_half_open(two_security_tables):
    sessions = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-05", "2020-01-09", "2020-01-10", "2020-01-11"]
    )
    grid = membership_grid("sp500", sessions)
    aaa = grid["AAA-20200101"]
    assert aaa.loc["2020-01-09"] == True  # noqa: E712 -- last session AAA is a member
    assert aaa.loc["2020-01-10"] == False  # noqa: E712 -- end_date_excl, no longer a member
    bbb = grid["BBB-20200105"]
    assert bbb.loc["2020-01-01"] == False  # noqa: E712 -- before BBB's start_date
    assert bbb.loc["2020-01-11"] == True  # noqa: E712 -- open interval, still a member


def test_members_on_returns_sorted_security_ids(two_security_tables):
    assert members_on("sp500", "2020-01-09") == ["AAA-20200101", "BBB-20200105"]
    assert members_on("sp500", "2020-01-11") == ["BBB-20200105"]


def test_mask_like_matches_features_shape_and_drops_unknown_columns(two_security_tables):
    features = pd.DataFrame(
        {"AAA-20200101": [1.0], "UNKNOWN-ID": [2.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    mask = mask_like(features, "sp500")
    assert list(mask.columns) == list(features.columns)
    assert mask.loc["2020-01-02", "AAA-20200101"] == True  # noqa: E712
    assert mask.loc["2020-01-02", "UNKNOWN-ID"] == False  # noqa: E712


def test_status_reports_fresh_manifest(two_security_tables):
    result = status("sp500")
    assert result["n_securities"] == 2
    assert result["warnings"] == []


def test_status_warns_when_manifest_is_stale(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    store.save_ticker_intervals("sp500", pd.DataFrame(columns=TICKER_INTERVALS_COLUMNS))
    store.save_security_map("sp500", pd.DataFrame(columns=SECURITY_MAP_COLUMNS))
    store.save_manifest("sp500", {"verified_through": "2000-01-01"})
    result = status("sp500")
    assert len(result["warnings"]) == 1
    assert "2000-01-01" in result["warnings"][0]


def test_membership_grid_warns_when_stale(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    store.save_ticker_intervals("sp500", pd.DataFrame(columns=TICKER_INTERVALS_COLUMNS))
    store.save_security_map("sp500", pd.DataFrame(columns=SECURITY_MAP_COLUMNS))
    store.save_manifest("sp500", {"verified_through": "2000-01-01"})
    with pytest.warns(MembershipStaleWarning):
        membership_grid("sp500", pd.DatetimeIndex(["2020-01-01"]))


def test_membership_grid_strict_raises_when_stale(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    store.save_ticker_intervals("sp500", pd.DataFrame(columns=TICKER_INTERVALS_COLUMNS))
    store.save_security_map("sp500", pd.DataFrame(columns=SECURITY_MAP_COLUMNS))
    store.save_manifest("sp500", {"verified_through": "2000-01-01"})
    with pytest.raises(MembershipStaleError):
        membership_grid("sp500", pd.DatetimeIndex(["2020-01-01"]), strict=True)


def test_membership_grid_is_correct_regardless_of_session_input_order(two_security_tables):
    # AAA is a member from 2020-01-01 (inclusive) to 2020-01-10 (exclusive).
    # Passed out of order, a naive searchsorted over an unsorted index would
    # compute nonsense slice positions and silently return wrong values.
    unsorted_sessions = pd.DatetimeIndex(["2020-01-10", "2020-01-05", "2020-01-09"])
    grid = membership_grid("sp500", unsorted_sessions)
    aaa = grid["AAA-20200101"]
    assert list(aaa) == [False, True, True]
    # Row order must match the caller's requested order, not sorted order.
    assert list(grid.index) == list(unsorted_sessions)


def test_membership_grid_handles_duplicate_sessions_in_requested_order(two_security_tables):
    sessions = pd.DatetimeIndex(["2020-01-09", "2020-01-10", "2020-01-09"])
    grid = membership_grid("sp500", sessions)
    assert list(grid["AAA-20200101"]) == [True, False, True]
    assert list(grid.index) == list(sessions)


def test_mask_like_preserves_hourly_index_and_computes_correct_values(two_security_tables):
    # AAA is a member through the close of 2020-01-09 and no longer a member
    # from 2020-01-10 onward (end_date_excl). Hourly timestamps on those two
    # days must each resolve against the right day, not get collapsed onto a
    # single normalized midnight row.
    hourly_index = pd.DatetimeIndex(["2020-01-09 10:00", "2020-01-09 15:30", "2020-01-10 11:00"])
    features = pd.DataFrame({"AAA-20200101": [1.0, 2.0, 3.0]}, index=hourly_index)
    mask = mask_like(features, "sp500")
    assert list(mask.index) == list(hourly_index)
    assert mask.index.equals(features.index)
    assert list(mask["AAA-20200101"]) == [True, True, False]


def test_mask_like_handles_tz_aware_index_without_raising(two_security_tables):
    tz_index = pd.DatetimeIndex(["2020-01-09", "2020-01-10"], tz="America/New_York")
    features = pd.DataFrame({"AAA-20200101": [1.0, 2.0]}, index=tz_index)
    mask = mask_like(features, "sp500")
    assert mask.index.equals(features.index)
    assert str(mask.index.tz) == "America/New_York"
    assert list(mask["AAA-20200101"]) == [True, False]


def test_grid_by_ticker_uses_ticker_columns_with_half_open_membership(two_security_tables):
    sessions = pd.DatetimeIndex(
        ["2020-01-01", "2020-01-09", "2020-01-10", "2020-01-11"]
    )
    grid = membership_grid("sp500", sessions, by="ticker")
    assert list(grid.columns) == ["AAA", "BBB"]
    assert list(grid["AAA"]) == [True, True, False, False]
    assert list(grid["BBB"]) == [False, True, True, True]


def test_members_on_by_ticker_returns_tickers(two_security_tables):
    assert members_on("sp500", "2020-01-09", by="ticker") == ["AAA", "BBB"]
    assert members_on("sp500", "2020-01-11", by="ticker") == ["BBB"]


def test_mask_like_by_ticker_matches_ticker_columns(two_security_tables):
    features = pd.DataFrame(
        {"AAA": [1.0], "NOT-IN-INDEX": [2.0]},
        index=pd.DatetimeIndex(["2020-01-02"]),
    )
    mask = mask_like(features, "sp500", by="ticker")
    assert list(mask.columns) == list(features.columns)
    assert mask.loc["2020-01-02", "AAA"] == True  # noqa: E712
    assert mask.loc["2020-01-02", "NOT-IN-INDEX"] == False  # noqa: E712


def test_mask_like_by_ticker_never_raises_on_zero_matches(two_security_tables):
    features = pd.DataFrame(
        {"ZZZ": [1.0]}, index=pd.DatetimeIndex(["2020-01-02"])
    )
    mask = mask_like(features, "sp500", by="ticker")
    assert list(mask["ZZZ"]) == [False]


def test_mask_like_default_raises_on_ticker_columns(two_security_tables):
    # Ticker columns under by="security_id" match nothing -- an all-False
    # mask is almost certainly a caller mistake, so this fails loudly.
    features = pd.DataFrame(
        {"AAA": [1.0], "BBB": [2.0]}, index=pd.DatetimeIndex(["2020-01-02"])
    )
    with pytest.raises(MembershipError, match='by="ticker"'):
        mask_like(features, "sp500")


def test_invalid_by_raises_value_error(two_security_tables):
    sessions = pd.DatetimeIndex(["2020-01-02"])
    with pytest.raises(ValueError):
        membership_grid("sp500", sessions, by="bogus")
    with pytest.raises(ValueError):
        members_on("sp500", "2020-01-02", by="bogus")
    with pytest.raises(ValueError):
        mask_like(pd.DataFrame({"AAA": [1.0]}, index=sessions), "sp500", by="bogus")
