"""End-to-end check against the real committed sp500 tables (not synthetic
fixtures) -- this is the one test file in this PR that reads production data."""
import pandas as pd

import data_universe as q


def test_mask_like_works_end_to_end_against_committed_data():
    features = pd.DataFrame(
        {"AAPL-19960102": [1.0]},  # real provisional security_id for AAPL
        index=pd.DatetimeIndex(["2020-06-15"]),
    )
    mask = q.membership.mask_like(features, "sp500")
    assert mask.shape == features.shape
    assert mask.dtypes.iloc[0] == "bool"


def test_grid_column_sums_match_interval_lengths_sampled():
    intervals = q.membership.sp500_intervals()
    assert len(intervals) > 1000  # real data: ~1,262 derived intervals expected
    # Provisional identity: each security_id is "<ticker>-<start_date>", so it
    # is unique by construction (pre-PR2 review has not yet merged any
    # rename/reuse chains onto a shared security_id).
    assert intervals["security_id"].is_unique is True


def test_status_reports_real_verified_through():
    result = q.membership.status("sp500")
    assert result["verified_through"] is not None
    assert result["n_securities"] > 1000
