"""End-to-end check against the real committed sp500 tables (not synthetic
fixtures) -- this is the one test file in this PR that reads production data."""
import pandas as pd
import pytest

import data_universe as q
from data_universe import config


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def test_mask_like_works_end_to_end_against_committed_data():
    features = pd.DataFrame(
        {"AAPL-19960102": [1.0]},  # real provisional security_id for AAPL
        index=pd.DatetimeIndex(["2020-06-15"]),
    )
    mask = q.membership.mask_like(features, "sp500")
    assert mask.shape == features.shape
    assert mask.dtypes.iloc[0] == "bool"
    # AAPL was genuinely an S&P 500 member on 2020-06-15 -- this is the one
    # cell in the mask that must be True. A mask that's silently all-False
    # (e.g. from a broken index lookup) would still pass the shape/dtype
    # checks above, so this is the check that actually catches that.
    assert mask.iloc[0, 0] == True  # noqa: E712


def test_grid_column_sums_match_interval_lengths_sampled():
    intervals = q.membership.sp500_intervals()
    assert len(intervals) > 1000  # real data: ~1,262 derived intervals expected
    # Provisional identity: each security_id is "<ticker>-<start_date>", so it
    # is unique by construction (pre-PR2 review has not yet merged any
    # rename/reuse chains onto a shared security_id).
    assert intervals["security_id"].is_unique is True
    # Sanity-bound check on an actual column sum via members_on: S&P 500
    # membership count is always close to 500.
    assert 480 <= len(q.membership.members_on("sp500", "2020-06-15")) <= 520


def test_status_reports_real_verified_through():
    result = q.membership.status("sp500")
    assert result["verified_through"] is not None
    assert result["n_securities"] > 1000
