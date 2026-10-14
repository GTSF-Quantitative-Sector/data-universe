import pandas as pd
import pytest

from data_universe import config
from data_universe.membership import store
from data_universe.membership.errors import MembershipError
from data_universe.membership.schema import TICKER_INTERVALS_COLUMNS


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def test_tables_dir_defaults_to_packaged_directory():
    path = store.tables_dir("sp500")
    assert path.parts[-3:] == ("membership", "tables", "sp500")


def test_tables_dir_honors_config_override(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    assert store.tables_dir("sp500") == tmp_path / "sp500"


def test_load_ticker_intervals_returns_empty_frame_with_right_columns_when_file_missing(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    # The index_id's tables_dir itself must exist -- a known index with just
    # this one file missing is a legitimate state (e.g. freshly initialized).
    (tmp_path / "sp500").mkdir(parents=True)
    frame = store.load_ticker_intervals("sp500")
    assert list(frame.columns) == TICKER_INTERVALS_COLUMNS
    assert len(frame) == 0


def test_ticker_intervals_round_trip(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    original = pd.DataFrame(
        [["sp500", "AAA", "2020-01-01", None], ["sp500", "BBB", "2020-01-01", "2020-06-01"]],
        columns=TICKER_INTERVALS_COLUMNS,
    )
    store.save_ticker_intervals("sp500", original)
    reloaded = store.load_ticker_intervals("sp500")
    assert reloaded["ticker"].tolist() == ["AAA", "BBB"]
    assert reloaded["end_date_excl"].iloc[1] == "2020-06-01"


def test_security_map_reviewed_column_round_trips_as_bool(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    from data_universe.membership.schema import SECURITY_MAP_COLUMNS

    original = pd.DataFrame(
        [["sp500", "AAA", "2020-01-01", "AAA-20200101", "new_default", False]],
        columns=SECURITY_MAP_COLUMNS,
    )
    store.save_security_map("sp500", original)
    reloaded = store.load_security_map("sp500")
    assert reloaded["reviewed"].iloc[0] is False or reloaded["reviewed"].iloc[0] == False  # noqa: E712


def test_manifest_round_trip(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    manifest = {"verified_through": "2026-08-18", "commit_sha": "abc123"}
    store.save_manifest("sp500", manifest)
    assert store.load_manifest("sp500") == manifest


def test_load_manifest_returns_empty_dict_when_missing(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    # The index_id's tables_dir itself must exist -- a known index before its
    # manifest.json has been written is a legitimate state.
    (tmp_path / "sp500").mkdir(parents=True)
    assert store.load_manifest("sp500") == {}


def test_load_function_raises_membership_error_when_tables_dir_missing(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    # Nothing creates tmp_path/"dow" -- the whole directory for this
    # index_id is missing, which should error loudly rather than look like
    # a known-but-empty index.
    with pytest.raises(MembershipError, match="dow"):
        store.load_ticker_intervals("dow")


def test_load_manifest_raises_membership_error_when_tables_dir_missing(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    with pytest.raises(MembershipError, match="dow"):
        store.load_manifest("dow")


def test_load_function_still_returns_empty_frame_for_known_index_missing_one_file(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    (tmp_path / "sp500").mkdir(parents=True)
    # ticker_intervals.csv was never written for this known index, but the
    # directory exists -- this must keep returning an empty frame, not raise.
    frame = store.load_ticker_intervals("sp500")
    assert list(frame.columns) == TICKER_INTERVALS_COLUMNS
    assert len(frame) == 0
