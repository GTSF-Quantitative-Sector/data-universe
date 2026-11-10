"""The packaged identity tables reproduce byte-for-byte under --offline, and
the committed resolution actually links the FB -> META rename."""
import shutil

import pytest

from data_universe import config
from data_universe.membership import store
from data_universe.membership.identity import resolve

INDEX = "sp500"


@pytest.fixture()
def packaged_copy(tmp_path):
    config.reset()
    config.set_membership_tables_dir(str(tmp_path))
    packaged = (
        __import__("pathlib").Path(store.__file__).resolve().parent
        / "tables" / INDEX
    )
    if not (packaged / "identity_evidence.csv").exists():
        pytest.skip("identity tables have not been built yet")
    shutil.copytree(packaged, tmp_path / INDEX)
    yield tmp_path / INDEX
    config.reset()


def test_offline_rebuild_reproduces_committed_bytes(packaged_copy):
    before = {p.name: p.read_bytes() for p in packaged_copy.iterdir()}
    result = resolve(INDEX, offline=True)
    assert result.n_massive_fetched == 0 and result.n_sec_fetched == 0
    after = {p.name: p.read_bytes() for p in packaged_copy.iterdir()}
    assert after == before


def test_fb_and_meta_share_one_security_id(packaged_copy):
    security_map = store.load_security_map(INDEX)
    fb = security_map[security_map["ticker"] == "FB"]
    meta = security_map[security_map["ticker"] == "META"]
    assert len(fb) and len(meta)
    assert fb["security_id"].iloc[0] == meta["security_id"].iloc[0]
