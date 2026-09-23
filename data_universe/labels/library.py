"""Concrete registered labels. Every function name here starts with `fwd_`
(enforced by `labels.registry.register`) as a loud reminder that these look
forward and must never be fed into a model as an input feature.

Important limitation shared by every label here: `compute()` only fetches
data through the requested `end` (never beyond it -- see `PLAN.md` §6), so
rows near the end of any requested range are genuinely NaN for lack of
future data, not a bug.
"""
from __future__ import annotations

import pandas as pd

from data_universe.features.library import iv_index, rv_cc_21d
from data_universe.labels.registry import register


def _daily_closes(ticker: str, start: str, end: str, source) -> pd.Series:
    bars = source.daily_bars(ticker, start, end)
    closes = bars["close"].copy()
    closes.index = bars.index.tz_localize(None).normalize()
    return closes


@register(
    "fwd_rv_cc_21d",
    warmup_trading_days=0,
    project="vrp",
    description="Realized close-to-close vol over the 21 trading days AFTER each date",
)
def fwd_rv_cc_21d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    trailing = rv_cc_21d(ticker, start, end, source)
    return trailing.shift(-21).rename("fwd_rv_cc_21d")


@register(
    "fwd_vrp_expost",
    warmup_trading_days=0,
    project="vrp",
    description="iv_index minus fwd_rv_cc_21d (realized, ex-post VRP)",
)
def fwd_vrp_expost(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    iv = iv_index(ticker, start, end, source)
    fwd_rv = fwd_rv_cc_21d(ticker, start, end, source)
    return (iv - fwd_rv).rename("fwd_vrp_expost")


@register(
    "fwd_ret_15d",
    warmup_trading_days=0,
    project="pead",
    description="Simple return from each date to 15 trading days later",
)
def fwd_ret_15d(ticker: str, start: str, end: str, source, **kwargs) -> pd.Series:
    closes = _daily_closes(ticker, start, end, source)
    return (closes.shift(-15) / closes - 1.0).rename("fwd_ret_15d")


LABEL_NAMES = ["fwd_rv_cc_21d", "fwd_vrp_expost", "fwd_ret_15d"]
