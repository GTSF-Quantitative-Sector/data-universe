import pandas as pd
import pytest

from data_universe.sources import adjust
from data_universe.sources.base import BAR_COLUMNS, NY_TZ, DataSourceError


def _closes(pairs):
    index = pd.DatetimeIndex([pd.Timestamp(d, tz=NY_TZ) for d, _ in pairs], name="timestamp")
    return pd.Series([c for _, c in pairs], index=index, name="close")


def _bars(pairs, volume=1000.0):
    closes = _closes(pairs)
    return pd.DataFrame(
        {c: closes.values for c in ("open", "high", "low", "close", "vwap")}
        | {"volume": volume},
        index=closes.index,
    )[BAR_COLUMNS]


def _dividends(rows):
    return pd.DataFrame(
        [
            {
                "ex_dividend_date": pd.Timestamp(d),
                "cash_amount": a,
                "currency": cur,
                "dividend_type": "CD",
            }
            for d, a, cur in rows
        ],
        columns=adjust.DIVIDEND_COLUMNS,
    )


PRICES = [("2024-01-02", 100.0), ("2024-01-03", 102.0), ("2024-01-04", 101.0),
          ("2024-01-05", 103.0)]


def test_single_dividend_factor_uses_prior_close():
    factors = adjust.dividend_factors(_closes(PRICES), _dividends([("2024-01-04", 1.0, "USD")]))
    assert factors.index.tolist() == [pd.Timestamp("2024-01-04")]
    assert factors.iloc[0] == pytest.approx(1.0 - 1.0 / 102.0)


def test_same_day_records_are_summed():
    divs = _dividends([("2024-01-04", 1.0, "USD"), ("2024-01-04", 0.5, "USD")])
    factors = adjust.dividend_factors(_closes(PRICES), divs)
    assert factors.iloc[0] == pytest.approx(1.0 - 1.5 / 102.0)


def test_ex_dates_on_first_bar_or_outside_window_are_ignored():
    divs = _dividends(
        [("2024-01-02", 1.0, "USD"), ("2023-12-01", 1.0, "USD"), ("2024-02-01", 1.0, "USD")]
    )
    assert adjust.dividend_factors(_closes(PRICES), divs).empty


def test_ex_date_on_a_missing_session_falls_back_to_last_prior_close():
    closes = _closes([("2024-01-02", 100.0), ("2024-01-03", 102.0), ("2024-01-05", 103.0)])
    factors = adjust.dividend_factors(closes, _dividends([("2024-01-04", 1.0, "")]))
    assert factors.iloc[0] == pytest.approx(1.0 - 1.0 / 102.0)


def test_non_usd_dividend_raises():
    with pytest.raises(DataSourceError):
        adjust.dividend_factors(_closes(PRICES), _dividends([("2024-01-04", 1.0, "CAD")]))


@pytest.mark.parametrize("amount", [102.0, 500.0, -1.0, 0.0])
def test_factor_outside_unit_interval_raises(amount):
    with pytest.raises(DataSourceError):
        adjust.dividend_factors(_closes(PRICES), _dividends([("2024-01-04", amount, "USD")]))


def test_apply_total_return_scales_prices_before_ex_date_only():
    bars = _bars(PRICES)
    factors = adjust.dividend_factors(bars["close"], _dividends([("2024-01-04", 1.0, "USD")]))
    result = adjust.apply_total_return(bars, factors)
    f = 1.0 - 1.0 / 102.0
    assert result["close"].tolist() == pytest.approx([100.0 * f, 102.0 * f, 101.0, 103.0])
    assert result["open"].tolist() == pytest.approx(result["close"].tolist())
    assert result["volume"].tolist() == bars["volume"].tolist()
    assert bars["close"].iloc[0] == 100.0  # input untouched


def test_two_dividends_compound():
    bars = _bars(PRICES)
    divs = _dividends([("2024-01-03", 1.0, "USD"), ("2024-01-05", 2.0, "USD")])
    result = adjust.apply_total_return(bars, adjust.dividend_factors(bars["close"], divs))
    f1, f2 = 1.0 - 1.0 / 100.0, 1.0 - 2.0 / 101.0
    assert result["close"].tolist() == pytest.approx(
        [100.0 * f1 * f2, 102.0 * f2, 101.0 * f2, 103.0]
    )


def test_apply_total_return_to_intraday_bars_uses_new_york_session_date():
    stamps = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-03 15:00", tz=NY_TZ), pd.Timestamp("2024-01-04 09:00", tz=NY_TZ)],
        name="timestamp",
    )
    hourly = pd.DataFrame({c: [10.0, 20.0] for c in BAR_COLUMNS}, index=stamps)
    factors = pd.Series([0.5], index=pd.DatetimeIndex([pd.Timestamp("2024-01-04")]))
    result = adjust.apply_total_return(hourly, factors)
    assert result["close"].tolist() == [5.0, 20.0]


def test_no_dividends_leaves_bars_unchanged():
    bars = _bars(PRICES)
    factors = adjust.dividend_factors(bars["close"], adjust.empty_dividends())
    pd.testing.assert_frame_equal(adjust.apply_total_return(bars, factors), bars)
