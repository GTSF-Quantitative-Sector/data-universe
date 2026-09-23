import numpy as np
import pandas as pd
import pytest

from data_universe.features import core


def _closes(values):
    index = pd.bdate_range("2024-01-02", periods=len(values), name="date")
    return pd.Series(values, index=index, dtype="float64")


def test_log_returns_basic():
    closes = _closes([100.0, 105.0, 100.0])
    result = core.log_returns(closes)
    assert np.isnan(result.iloc[0])
    assert result.iloc[1] == pytest.approx(np.log(105.0 / 100.0))
    assert result.iloc[2] == pytest.approx(np.log(100.0 / 105.0))


def test_realized_vol_cc_annualized_matches_manual_calc():
    rng = np.random.default_rng(0)
    values = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, size=40)))
    closes = _closes(values)
    result = core.realized_vol_cc(closes, window=21)
    returns = core.log_returns(closes)
    expected_last = returns.iloc[-21:].std(ddof=1) * np.sqrt(252)
    assert result.iloc[-1] == pytest.approx(expected_last)
    assert result.iloc[:20].isna().all()  # not enough history for the first 20 rows


def test_realized_vol_parkinson_positive_and_uses_high_low_range():
    index = pd.bdate_range("2024-01-02", periods=25, name="date")
    highs = pd.Series(101.0, index=index)
    lows = pd.Series(99.0, index=index)
    result = core.realized_vol_parkinson(highs, lows, window=21)
    assert (result.dropna() > 0).all()
    flat_highs = pd.Series(100.0, index=index)
    flat_lows = pd.Series(100.0, index=index)
    zero_range = core.realized_vol_parkinson(flat_highs, flat_lows, window=21)
    assert zero_range.dropna().eq(0.0).all()


def test_realized_vol_garman_klass_zero_when_flat():
    index = pd.bdate_range("2024-01-02", periods=25, name="date")
    flat = pd.Series(100.0, index=index)
    result = core.realized_vol_garman_klass(flat, flat, flat, flat, window=21)
    assert result.dropna().eq(0.0).all()


def test_rolling_ols_beta_alpha_recovers_planted_relationship():
    rng = np.random.default_rng(1)
    n = 120
    index = pd.bdate_range("2024-01-02", periods=n, name="date")
    market_returns = pd.Series(rng.normal(0.0003, 0.01, size=n), index=index)
    planted_beta, planted_alpha = 1.5, 0.0002
    noise = pd.Series(rng.normal(0.0, 0.0005, size=n), index=index)
    asset_returns = planted_alpha + planted_beta * market_returns + noise

    beta, alpha = core.rolling_ols_beta_alpha(asset_returns, market_returns, window=60)

    assert beta.iloc[-1] == pytest.approx(planted_beta, abs=0.15)
    assert alpha.iloc[-1] == pytest.approx(planted_alpha, abs=0.001)
    assert beta.iloc[:59].isna().all()
