"""Concrete registered features for the beta, VRP, and PEAD projects. See
`PLAN.md` §7 for the project-level rationale behind each one.
"""
from __future__ import annotations

import pandas as pd

from data_universe.features import core
from data_universe.features.registry import register
from data_universe.options.iv30 import iv30

_PROXY_TICKER = "SPY"


def _daily_frame(ticker: str, start: str, end: str, source) -> pd.DataFrame:
    """Daily bars for `ticker`, re-indexed to a tz-naive date index."""
    bars = source.daily_bars(ticker, start, end)
    frame = bars.copy()
    frame.index = bars.index.tz_localize(None).normalize()
    frame.index.name = "date"
    return frame


@register("ret_1d", warmup_trading_days=1, project="beta", description="1-day simple return")
def ret_1d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_frame(ticker, start, end, source)["close"]
    return closes.pct_change().rename("ret_1d")


@register(
    "ret_excess_1d",
    warmup_trading_days=1,
    project="beta",
    description="1-day return minus the daily risk-free rate (0 if no rate_source given)",
)
def ret_excess_1d(
    ticker: str, start: str, end: str, source, rate_source=None, **kwargs
) -> pd.Series:
    raw = ret_1d(ticker, start, end, source)
    if rate_source is None:
        return raw.rename("ret_excess_1d")
    rate = rate_source.daily_rate(start=start, end=end).reindex(raw.index).fillna(0.0)
    return (raw - rate).rename("ret_excess_1d")


@register(
    "ret_21d", warmup_trading_days=21, project="beta", description="21-trading-day simple return"
)
def ret_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_frame(ticker, start, end, source)["close"]
    return closes.pct_change(21).rename("ret_21d")


@register(
    "rv_cc_21d",
    warmup_trading_days=21,
    project="vrp",
    description="Annualized 21d close-to-close realized volatility",
)
def rv_cc_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_frame(ticker, start, end, source)["close"]
    return core.realized_vol_cc(closes, window=21).rename("rv_cc_21d")


@register(
    "rv_parkinson_21d",
    warmup_trading_days=21,
    project="vrp",
    description="Annualized 21d Parkinson realized volatility",
)
def rv_parkinson_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    frame = _daily_frame(ticker, start, end, source)
    return core.realized_vol_parkinson(frame["high"], frame["low"], window=21).rename(
        "rv_parkinson_21d"
    )


@register(
    "rv_gk_21d",
    warmup_trading_days=21,
    project="vrp",
    description="Annualized 21d Garman-Klass realized volatility",
)
def rv_gk_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    frame = _daily_frame(ticker, start, end, source)
    return core.realized_vol_garman_klass(
        frame["open"], frame["high"], frame["low"], frame["close"], window=21
    ).rename("rv_gk_21d")


@register("dollar_volume_1d", warmup_trading_days=0, project="pead", description="close * volume")
def dollar_volume_1d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    frame = _daily_frame(ticker, start, end, source)
    return (frame["close"] * frame["volume"]).rename("dollar_volume_1d")


@register(
    "adv_usd_20d",
    warmup_trading_days=20,
    project="pead",
    description="20-trading-day rolling average dollar volume",
)
def adv_usd_20d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    dollar_volume = dollar_volume_1d(ticker, start, end, source)
    return dollar_volume.rolling(20).mean().rename("adv_usd_20d")


def _beta_alpha(ticker: str, start: str, end: str, source, window: int):
    asset_closes = _daily_frame(ticker, start, end, source)["close"]
    market_closes = _daily_frame(_PROXY_TICKER, start, end, source)["close"]
    asset_returns = core.log_returns(asset_closes)
    market_returns = core.log_returns(market_closes).reindex(asset_returns.index)
    return core.rolling_ols_beta_alpha(asset_returns, market_returns, window)


@register(
    "beta_ols_60d",
    warmup_trading_days=61,
    project="beta",
    description="60d rolling OLS beta vs. SPY",
)
def beta_ols_60d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    beta, _ = _beta_alpha(ticker, start, end, source, window=60)
    return beta.rename("beta_ols_60d")


@register(
    "beta_ols_90d",
    warmup_trading_days=91,
    project="beta",
    description="90d rolling OLS beta vs. SPY",
)
def beta_ols_90d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    beta, _ = _beta_alpha(ticker, start, end, source, window=90)
    return beta.rename("beta_ols_90d")


@register(
    "alpha_ols_60d",
    warmup_trading_days=61,
    project="beta",
    description="60d rolling OLS alpha vs. SPY",
)
def alpha_ols_60d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    _, alpha = _beta_alpha(ticker, start, end, source, window=60)
    return alpha.rename("alpha_ols_60d")


@register(
    "alpha_ols_90d",
    warmup_trading_days=91,
    project="beta",
    description="90d rolling OLS alpha vs. SPY",
)
def alpha_ols_90d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    _, alpha = _beta_alpha(ticker, start, end, source, window=90)
    return alpha.rename("alpha_ols_90d")


@register(
    "iv_index",
    warmup_trading_days=0,
    project="vrp",
    description=(
        "Implied-vol index level: VIX/100 for SPY, else options.iv30 (NotImplementedError today)"
    ),
)
def iv_index(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    if ticker != _PROXY_TICKER:
        # Single-name IV30 is not implemented yet -- see options/iv30.py.
        iv30(ticker, end)
    frame = source.index_daily_bars("I:VIX", start, end)
    closes = frame["close"] / 100.0
    closes.index = closes.index.tz_localize(None).normalize()
    return closes.rename("iv_index")


@register(
    "vrp_signal_rv21",
    warmup_trading_days=21,
    project="vrp",
    description="iv_index minus trailing rv_cc_21d (ex-ante VRP signal)",
)
def vrp_signal_rv21(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    iv = iv_index(ticker, start, end, source)
    rv = rv_cc_21d(ticker, start, end, source)
    return (iv - rv).rename("vrp_signal_rv21")


FEATURE_NAMES = [
    "ret_1d",
    "ret_excess_1d",
    "ret_21d",
    "rv_cc_21d",
    "rv_parkinson_21d",
    "rv_gk_21d",
    "dollar_volume_1d",
    "adv_usd_20d",
    "beta_ols_60d",
    "beta_ols_90d",
    "alpha_ols_60d",
    "alpha_ols_90d",
    "iv_index",
    "vrp_signal_rv21",
]
