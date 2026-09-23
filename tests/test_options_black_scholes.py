import math

import pytest

from data_universe.options.black_scholes import implied_vol, price, vega


def test_price_at_expiry_equals_intrinsic_value_call():
    assert price(S=110, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=True) == 10
    assert price(S=90, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=True) == 0


def test_price_at_expiry_equals_intrinsic_value_put():
    assert price(S=90, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=False) == 10
    assert price(S=110, K=100, T=0, r=0.05, q=0.0, sigma=0.2, is_call=False) == 0


def test_call_price_increases_with_volatility():
    low = price(S=100, K=100, T=1, r=0.02, q=0.0, sigma=0.1, is_call=True)
    high = price(S=100, K=100, T=1, r=0.02, q=0.0, sigma=0.4, is_call=True)
    assert high > low


def test_vega_is_positive_for_a_live_option():
    v = vega(S=100, K=100, T=1, r=0.02, q=0.0, sigma=0.2)
    assert v > 0


def test_vega_is_zero_at_expiry():
    assert vega(S=100, K=100, T=0, r=0.02, q=0.0, sigma=0.2) == 0


@pytest.mark.parametrize("moneyness", [0.8, 1.0, 1.2])
@pytest.mark.parametrize("T", [0.05, 0.25, 1.0, 2.0])
@pytest.mark.parametrize("is_call", [True, False])
def test_implied_vol_recovers_planted_sigma(moneyness, T, is_call):
    S = 100.0
    K = S / moneyness
    r, q, sigma = 0.03, 0.01, 0.25
    target_price = price(S, K, T, r, q, sigma, is_call)
    recovered = implied_vol(target_price, S, K, T, r, q, is_call)
    assert recovered == pytest.approx(sigma, abs=1e-4)


def test_implied_vol_returns_nan_when_price_below_no_arbitrage_bound():
    result = implied_vol(target_price=-5.0, S=100, K=100, T=1, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)


def test_implied_vol_returns_nan_when_price_above_no_arbitrage_bound():
    # A call can never be worth more than the (discounted) spot price.
    result = implied_vol(target_price=1000.0, S=100, K=100, T=1, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)


def test_implied_vol_returns_nan_at_zero_time_to_expiry():
    result = implied_vol(target_price=10.0, S=110, K=100, T=0, r=0.02, q=0.0, is_call=True)
    assert math.isnan(result)


def test_implied_vol_never_raises_on_bad_inputs():
    # Should return nan, not raise, even for a degenerate/nonsensical price.
    result = implied_vol(
        target_price=float("inf"), S=100, K=100, T=1, r=0.02, q=0.0, is_call=True
    )
    assert math.isnan(result)
