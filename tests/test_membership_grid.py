import pandas as pd
import pytest

from data_universe import config
from data_universe.membership import store
from data_universe.membership.errors import MembershipStaleError, MembershipStaleWarning
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
