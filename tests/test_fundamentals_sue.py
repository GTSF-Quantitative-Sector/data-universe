"""Tests for `fundamentals.sue` -- first-reported XBRL EPS, split factors,
derived Q4s, and the point-in-time SUE series. All offline with planted
concept facts and `FakeMarket` price fakes."""
import numpy as np
import pandas as pd
import pytest

from data_universe.events.earnings import EVENT_COLUMNS, build_pead_events
from data_universe.fundamentals.sue import (
    QUARTER_COLUMNS,
    SUE_EVENT_COLUMNS,
    attach_sue,
    quarterly_sue,
)
from data_universe.sources.base import DataSourceError
from data_universe.sources.edgar import CONCEPT_COLUMNS
from data_universe.sources.fakes import FakeMarket, FakePolygonSource
from tests.test_events_earnings import FakeEdgarSource, FakeOptionsSource


def _fact(end, val, filed, start=None, accn="a1", form="10-Q", fy=None, fp=None,
          frame=None):
    start = start or (pd.Timestamp(end) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    return {
        "start": start, "end": end, "val": val, "accn": accn, "fy": fy, "fp": fp,
        "form": form, "filed": filed, "frame": frame,
    }


class FakeConceptEdgar:
    """`company_concept` answers from planted fact lists keyed by tag."""

    def __init__(self, facts_by_tag=None, raises=False):
        self._facts = facts_by_tag or {}
        self._raises = raises
        self.calls = []

    def company_concept(self, cik, tag, unit="USD/shares", taxonomy="us-gaap"):
        self.calls.append((cik, tag, unit, taxonomy))
        if self._raises:
            raise DataSourceError("boom")
        frame = pd.DataFrame(
            [dict(e) for e in self._facts.get(tag, [])], columns=CONCEPT_COLUMNS
        )
        for column in ("start", "end", "filed"):
            frame[column] = pd.to_datetime(frame[column])
        frame["val"] = frame["val"].astype("float64")
        frame["fy"] = frame["fy"].astype("Int64")
        return frame.sort_values(
            ["filed", "end", "accn"], na_position="last"
        ).reset_index(drop=True)


class EdgarWithConcept(FakeEdgarSource):
    """The earnings-test EDGAR fake plus a `company_concept`."""

    def __init__(self, *args, facts_by_tag=None, raises=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._concept = FakeConceptEdgar(facts_by_tag, raises=raises)

    def company_concept(self, *args, **kwargs):
        return self._concept.company_concept(*args, **kwargs)


def _series_facts(eps_list, end0="2019-03-31", step=91, filed_lag=45, scale=1.0,
                  tag="EarningsPerShareDiluted", start_offset=90):
    """First-reported quarterly facts: ends `step` days apart, each filed
    `filed_lag` days later, duration `start_offset` days."""
    end0 = pd.Timestamp(end0)
    return [
        _fact(
            (end0 + pd.Timedelta(days=step * i)).strftime("%Y-%m-%d"),
            eps * scale,
            (end0 + pd.Timedelta(days=step * i + filed_lag)).strftime("%Y-%m-%d"),
            start=(end0 + pd.Timedelta(days=step * i - start_offset)).strftime(
                "%Y-%m-%d"
            ),
        )
        for i, eps in enumerate(eps_list)
    ]


_EPS_16 = [1.0, 1.1, 1.2, 0.9, 1.3, 1.5, 0.8, 1.6,
           1.7, 1.9, 1.1, 2.0, 1.4, 1.6, 1.8, 2.2]


def _sue_of(quarters, i):
    return quarters.iloc[i]["sue"], quarters.iloc[i]["sue_n"]


def _expected_d(eps, i):
    return eps[i] - eps[i - 4] if i >= 4 else np.nan


def test_first_reported_ignores_later_comparatives_and_amendments():
    facts = _series_facts([1.0], end0="2020-03-31")
    facts += [
        # same period re-reported a year later (comparative) and as a 10-Q/A
        _fact("2020-03-31", 1.25, "2021-04-30", accn="a2"),
        _fact("2020-03-31", 1.5, "2020-08-01", accn="a3", form="10-Q/A"),
    ]
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    assert len(quarters) == 1
    assert quarters.iloc[0]["eps_diluted"] == 1.0
    assert quarters.iloc[0]["filed"] == pd.Timestamp("2020-05-15")


def test_same_filed_date_tag_tie_prefers_diluted():
    shared = dict(start="2020-01-01", end="2020-03-31", filed="2020-05-15")
    facts = {
        "EarningsPerShareDiluted": [_fact(**{**shared, "val": 1.0, "accn": "b"})],
        "EarningsPerShareBasicAndDiluted": [
            _fact(**{**shared, "val": 9.9, "accn": "a"})
        ],
    }
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar(facts), FakePolygonSource())
    assert len(quarters) == 1
    assert quarters.iloc[0]["eps_diluted"] == 1.0
    assert quarters.iloc[0]["tag"] == "EarningsPerShareDiluted"


def test_basic_and_diluted_only_filer_works():
    facts = {"EarningsPerShareBasicAndDiluted": _series_facts(_EPS_16)}
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar(facts), FakePolygonSource())
    assert len(quarters) == 16
    assert (quarters["tag"] == "EarningsPerShareBasicAndDiluted").all()


def test_no_facts_gives_empty_typed_frame():
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar({}), FakePolygonSource())
    assert quarters.empty
    assert list(quarters.columns) == QUARTER_COLUMNS
    assert str(quarters["period_end"].dtype) == "datetime64[ns]"
    assert str(quarters["sue_n"].dtype) == "int64"


def test_sue_math_is_exact_on_a_clean_16_quarter_series():
    facts = {"EarningsPerShareDiluted": _series_facts(_EPS_16)}
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar(facts), FakePolygonSource())
    assert len(quarters) == 16
    # quarter 15: d = eps15 - eps11; prior diffs = quarters 7..14 (8 of them).
    d_q = _EPS_16[15] - _EPS_16[11]
    prior = [_EPS_16[p] - _EPS_16[p - 4] for p in range(7, 15)]
    sue, sue_n = _sue_of(quarters, 15)
    assert sue_n == 8
    assert sue == pytest.approx(d_q / np.std(prior, ddof=1))
    row = quarters.iloc[15]
    assert row["eps_yoy_diff"] == pytest.approx(d_q)
    assert row["sue_std"] == pytest.approx(np.std(prior, ddof=1))


def test_min_obs_boundary_5_is_nan_6_is_finite():
    facts = {"EarningsPerShareDiluted": _series_facts(_EPS_16)}
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar(facts), FakePolygonSource())
    sue9, n9 = _sue_of(quarters, 9)   # prior diffs = quarters 4..8 -> 5
    sue10, n10 = _sue_of(quarters, 10)  # prior diffs = quarters 4..9 -> 6
    assert n9 == 5 and np.isnan(sue9)
    assert n10 == 6 and np.isfinite(sue10)


def test_identical_prior_diffs_give_std_zero_and_nan_sue():
    eps = [1.0 + 0.5 * i for i in range(16)]
    facts = {"EarningsPerShareDiluted": _series_facts(eps)}
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar(facts), FakePolygonSource())
    row = quarters.iloc[15]
    assert row["sue_n"] == 8
    assert row["sue_std"] == 0.0
    assert np.isnan(row["sue"])


def test_split_invariance_sue_equals_the_no_split_control():
    # quarter 5's filed date itself -- a filing ON the split date counts as
    # already post-split.
    split_day = _series_facts(_EPS_16)[5]["filed"]
    market = FakeMarket(splits={"AAA": [(split_day, 4.0)]})
    split_source = FakePolygonSource(market)
    control = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": _series_facts(_EPS_16)}),
        FakePolygonSource(),
    )
    # Same economy: filings before the split date report 4x the values; the
    # one filed exactly on the split date is already on the new basis.
    scaled = []
    for fact in _series_facts(_EPS_16):
        fact = dict(fact)
        fact["val"] *= 4.0 if fact["filed"] < split_day else 1.0
        scaled.append(fact)
    split_quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": scaled}),
        split_source,
    )
    assert len(split_quarters) == len(control)
    np.testing.assert_allclose(
        split_quarters["sue"].to_numpy(), control["sue"].to_numpy(), atol=1e-9
    )
    # Own-basis values: quarter 4 was reported pre-split at 4x; quarter 5 was
    # filed exactly on the split date and counts as already post-split.
    assert split_quarters.iloc[4]["eps_diluted"] == pytest.approx(1.3 * 4.0)
    assert split_quarters.iloc[5]["eps_diluted"] == pytest.approx(1.5)
    # Own-basis outputs: pre-split quarters' eps_yoy_diff is 4x the control's.
    for i in range(len(control)):
        expected = control["eps_yoy_diff"].iloc[i]
        expected_std = control["sue_std"].iloc[i]
        if not np.isfinite(expected):
            continue
        if control["filed"].iloc[i] < pd.Timestamp(split_day):
            assert split_quarters["eps_yoy_diff"].iloc[i] == pytest.approx(
                4.0 * expected
            )
            if np.isfinite(expected_std):
                assert split_quarters["sue_std"].iloc[i] == pytest.approx(
                    4.0 * expected_std
                )
        else:
            assert split_quarters["eps_yoy_diff"].iloc[i] == pytest.approx(
                expected
            )
            if np.isfinite(expected_std):
                assert split_quarters["sue_std"].iloc[i] == pytest.approx(
                    expected_std
                )


def _fy_facts(annual_eps, q_eps, fy_end="2020-12-31", fy_filed="2021-02-01",
              fy_days=364):
    """One FY row plus direct Q1..Q3 rows tiling the fiscal year: 90-day
    quarters with 1-day gaps, so the derived Q4 is `fy_days - 273` days."""
    end = pd.Timestamp(fy_end)
    start = end - pd.Timedelta(days=fy_days)
    facts = [
        _fact(fy_end, annual_eps, fy_filed, start=start.strftime("%Y-%m-%d"),
              form="10-K")
    ]
    for i, eps in enumerate(q_eps):
        q_end = start + pd.Timedelta(days=91 * (i + 1) - 1)
        q_start = start + pd.Timedelta(days=91 * i + (1 if i else 0))
        facts.append(
            _fact(
                q_end.strftime("%Y-%m-%d"), eps,
                (q_end + pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
                start=q_start.strftime("%Y-%m-%d"), form="10-Q",
            )
        )
    return facts


def test_derived_q4_is_fy_minus_q1_q2_q3():
    facts = _fy_facts(4.0, [1.0, 1.0, 1.0])
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    q4 = quarters.iloc[-1]
    assert q4["derived"] == True  # noqa: E712
    assert q4["eps_diluted"] == pytest.approx(1.0)
    assert q4["filed"] == pd.Timestamp("2021-02-01")
    assert q4["form"] == "10-K"
    assert q4["period_end"] == pd.Timestamp("2020-12-31")


def test_direct_q4_wins_over_derivation():
    facts = _fy_facts(4.0, [1.0, 1.0, 1.0])
    # a direct Q4 ending on the fiscal year end, filed before the 10-K
    facts.append(
        _fact("2020-12-31", 1.5, "2021-01-20",
              start="2020-10-01", accn="q4", form="10-Q")
    )
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    q4 = quarters.iloc[-1]
    assert q4["derived"] == False  # noqa: E712
    assert q4["eps_diluted"] == 1.5


def test_derived_q4_handles_a_mid_year_split():
    split_day = "2020-09-01"
    market = FakeMarket(splits={"AAA": [(split_day, 4.0)]})
    source = FakePolygonSource(market)
    facts = _fy_facts(4.0, [4.0, 4.0, 1.0])  # Q1/Q2 reported pre-split at 4x
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}), source,
    )
    q4 = quarters.iloc[-1]
    assert q4["derived"] == True  # noqa: E712
    # FY(4.0 post-split) - (1 + 1 + 1 adjusted) = 1.0 on today's basis.
    assert q4["eps_diluted"] == pytest.approx(1.0)


def test_broken_tiling_skips_the_derived_q4():
    facts = _fy_facts(4.0, [1.0, 1.0, 1.0])[0:1] + _fy_facts(4.0, [1.0, 1.0, 1.0])[1::2]
    # keep FY + Q1 + Q3 (drop Q2)
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    assert (quarters["period_end"] == pd.Timestamp("2020-12-31")).sum() == 0
    assert not quarters["derived"].any()


def test_52_53_week_year_derives_a_98_day_q4():
    facts = _fy_facts(4.0, [1.0, 1.0, 1.0], fy_end="2021-01-02",
                      fy_filed="2021-02-05", fy_days=371)
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    q4 = quarters.iloc[-1]
    assert q4["derived"] == True  # noqa: E712
    assert (q4["period_end"] - q4["period_start"]).days == 98


def test_lag4_matches_a_371_day_gap():
    # One 98-day step among 91-day steps -> a lag-4 gap of exactly 371 days,
    # inside [350, 380], still matches.
    ends = []
    end = pd.Timestamp("2020-03-31")
    for step in (91, 91, 91, 98, 91, 91, 91, 91):
        end = end + pd.Timedelta(days=step)
        ends.append(end)
    eps = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    facts = [
        _fact(e.strftime("%Y-%m-%d"), v, (e + pd.Timedelta(days=45)).strftime("%Y-%m-%d"))
        for e, v in zip(ends, eps)
    ]
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    row = quarters[quarters["period_end"] == ends[4]].iloc[0]
    assert row["eps_yoy_diff"] == pytest.approx(5.0 - 1.0)


def test_lag4_missing_quarter_makes_the_diff_nan():
    facts = _series_facts(_EPS_16)
    del facts[11]  # no quarter ends ~364 days before quarter 15's
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    last = quarters[quarters["period_end"] == quarters["period_end"].max()].iloc[0]
    assert np.isnan(last["eps_yoy_diff"])


def test_prior_quarter_filed_after_filed_q_is_excluded():
    eps = list(_EPS_16)
    facts = _series_facts(eps)
    # Quarter 14's XBRL filing lands after quarter 15's -- invisible to V_15.
    late = dict(facts[14])
    late["filed"] = (
        pd.Timestamp(facts[15]["filed"]) + pd.Timedelta(days=1)
    ).strftime("%Y-%m-%d")
    facts[14] = late
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    _, sue_n = _sue_of(quarters, 15)
    assert sue_n == 7  # the visible prior diffs are quarters 7..13


def test_missing_factor_makes_components_nan_without_error():
    facts = {"EarningsPerShareDiluted": _series_facts(_EPS_16)}
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar(facts), FakePolygonSource(),
        history_start="2099-01-01",  # after every filed date -> no factors
    )
    assert np.isnan(quarters["sue"]).all()


class _FactorGapSource(FakePolygonSource):
    """Drops raw and split-adjusted bars over a >10-day window."""

    def __init__(self, drop_start, drop_days, market=None):
        super().__init__(market)
        self._drop_start = pd.Timestamp(drop_start)
        self._drop_days = drop_days

    def _drop(self, bars):
        dates = bars.index.tz_localize(None).normalize()
        keep = (dates < self._drop_start) | (
            dates >= self._drop_start + pd.Timedelta(days=self._drop_days)
        )
        return bars.loc[keep]

    def daily_bars_raw(self, ticker, start, end):
        return self._drop(super().daily_bars_raw(ticker, start, end))

    def daily_bars_split_adjusted(self, ticker, start, end):
        return self._drop(super().daily_bars_split_adjusted(ticker, start, end))


def test_bars_gap_over_ten_days_makes_that_factor_nan():
    facts = _series_facts(_EPS_16)
    filed_15 = pd.Timestamp(facts[15]["filed"])
    source = _FactorGapSource((filed_15 - pd.Timedelta(days=12)).strftime("%Y-%m-%d"), 13)
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}), source
    )
    # Only quarter 15's factor is missing -> its own-basis outputs go NaN.
    assert np.isnan(quarters.iloc[15]["eps_yoy_diff"])
    assert np.isfinite(quarters.iloc[14]["eps_yoy_diff"])


def _events_frame(dates):
    return pd.DataFrame(
        {"ticker": ["AAA"] * len(dates), "event_date": pd.to_datetime(dates)}
    )


def _quarters_frame(ends, fileds):
    rows = []
    for end, filed in zip(ends, fileds):
        rows.append(
            {
                "cik": "1", "period_start": pd.Timestamp(end) - pd.Timedelta(days=90),
                "period_end": pd.Timestamp(end), "filed": pd.Timestamp(filed),
                "accn": "a", "form": "10-Q", "tag": "EarningsPerShareDiluted",
                "eps_diluted": 1.5, "derived": False, "eps_yoy_diff": 0.3,
                "sue_std": 0.2, "sue_n": 8, "sue": 1.5,
            }
        )
    return pd.DataFrame(rows, columns=QUARTER_COLUMNS)


def test_attach_sue_matches_a_recent_quarter_filed_after_the_event():
    events = _events_frame(["2024-04-30"])
    quarters = _quarters_frame(["2024-04-05"], ["2024-05-05"])  # filed 5d after
    out = attach_sue(events, quarters)
    row = out.iloc[0]
    assert row["eps_period_end"] == pd.Timestamp("2024-04-05")
    assert row["eps_filed"] == pd.Timestamp("2024-05-05")
    assert row["eps_diluted"] == 1.5
    assert row["eps_derived"] == False  # noqa: E712
    assert row["sue_n"] == 8
    assert row["sue"] == 1.5


def test_attach_sue_unmatched_when_filed_too_early_or_period_too_old():
    events = _events_frame(["2024-04-30"])
    quarters = _quarters_frame(["2024-04-20"], ["2024-04-22"])  # filed 8d before
    out = attach_sue(events, quarters)
    assert pd.isna(out.iloc[0]["eps_period_end"])
    quarters = _quarters_frame(["2024-01-19"], ["2024-05-01"])  # 101 days before
    out = attach_sue(events, quarters)
    assert pd.isna(out.iloc[0]["eps_period_end"])


def test_attach_sue_picks_the_latest_candidate_and_reuses_it():
    events = _events_frame(["2024-04-30", "2024-05-02"])
    quarters = _quarters_frame(
        ["2024-01-31", "2024-04-05"], ["2024-05-01", "2024-05-05"]
    )
    out = attach_sue(events, quarters)
    assert (out["eps_period_end"] == pd.Timestamp("2024-04-05")).all()


def test_attach_sue_derived_q4_and_dtypes_and_empty_inputs():
    events = _events_frame(["2024-04-30"])
    quarters = _quarters_frame(["2024-04-05"], ["2024-05-05"])
    quarters["derived"] = True
    out = attach_sue(events, quarters)
    assert out.iloc[0]["eps_derived"] == True  # noqa: E712
    assert list(out.columns[-8:]) == SUE_EVENT_COLUMNS
    assert str(out["eps_derived"].dtype) == "boolean"
    assert str(out["sue_n"].dtype) == "Int64"
    assert str(out["eps_period_end"].dtype) == "datetime64[ns]"
    empty_events = attach_sue(events.iloc[0:0], quarters)
    assert list(empty_events.columns) == list(events.columns) + SUE_EVENT_COLUMNS
    out = attach_sue(events, quarters.iloc[0:0])
    assert pd.isna(out.iloc[0]["sue"])
    assert out.iloc[0]["eps_derived"] is pd.NA or pd.isna(out.iloc[0]["eps_derived"])


def test_build_pead_events_include_sue_appends_columns_identical_events():
    facts = _series_facts(_EPS_16, end0="2020-03-31")
    edgar = EdgarWithConcept(
        filing_dates=["2024-02-01T20:15:00.000Z", "2024-05-02T20:15:00.000Z"],
        facts_by_tag={"EarningsPerShareDiluted": facts},
    )
    price_source = FakePolygonSource()
    base = build_pead_events(
        "AAPL", edgar, price_source, FakeOptionsSource(),
        calendar_start="2023-10-01", calendar_end="2024-06-01",
    )
    with_sue = build_pead_events(
        "AAPL", edgar, price_source, FakeOptionsSource(), include_sue=True,
        calendar_start="2023-10-01", calendar_end="2024-06-01",
    )
    pd.testing.assert_frame_equal(with_sue[base.columns], base)
    assert list(with_sue.columns[-8:]) == SUE_EVENT_COLUMNS
    # Row 0 (the 2024-02-01 filing) matches quarter 15 (end 2023-12-26, first
    # filed 2024-02-09); row 1 (2024-05-02) is 128 days past it -> unmatched.
    quarters = quarterly_sue(
        "AAPL", "1", edgar, price_source, history_start="2020-03-10"
    )
    row0 = with_sue.iloc[0]
    assert row0["eps_period_end"] == pd.Timestamp("2023-12-26")
    assert row0["eps_filed"] == pd.Timestamp("2024-02-09")
    assert row0["sue"] == pytest.approx(quarters.iloc[15]["sue"])
    assert np.isfinite(row0["sue"])
    row1 = with_sue.iloc[1]
    assert pd.isna(row1["eps_period_end"])
    assert np.isnan(row1["sue"])


def test_include_sue_false_makes_no_concept_requests():
    edgar = EdgarWithConcept(facts_by_tag={"EarningsPerShareDiluted": _series_facts(_EPS_16)})
    build_pead_events(
        "AAPL", edgar, FakePolygonSource(), FakeOptionsSource(),
        calendar_start="2023-10-01", calendar_end="2024-06-01",
    )
    assert edgar._concept.calls == []


def test_empty_filings_with_include_sue_have_all_columns():
    class _Empty(EdgarWithConcept):
        def earnings_8k_filings(self, cik):
            return pd.DataFrame(
                columns=["filing_date", "acceptance_datetime", "form", "items"]
            )

    out = build_pead_events(
        "AAPL", _Empty(), FakePolygonSource(), FakeOptionsSource(),
        include_sue=True, calendar_start="2023-10-01", calendar_end="2024-06-01",
    )
    assert out.empty
    assert list(out.columns) == EVENT_COLUMNS + SUE_EVENT_COLUMNS


def test_derived_q4_missing_part_factor_gives_nan_not_wrong_value():
    # Q1/Q2 were filed before history_start, so their factors are NaN; the
    # derived Q4 must be NaN, not FY - (Q2+Q3).
    facts = _fy_facts(4.0, [1.0, 1.0, 1.0])
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(), history_start="2020-06-01",
    )
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4["derived"] == True  # noqa: E712
    assert np.isnan(q4["eps_diluted"])


def test_float_noise_std_is_reported_as_zero_and_sue_nan():
    eps = [1.0 + 0.1 * i for i in range(16)]  # diffs all ~0.4 +/- float noise
    facts = {"EarningsPerShareDiluted": _series_facts(eps)}
    quarters = quarterly_sue("AAA", "1", FakeConceptEdgar(facts), FakePolygonSource())
    row = quarters.iloc[15]
    assert row["sue_n"] == 8
    assert row["sue_std"] == 0.0
    assert np.isnan(row["sue"])
    assert row["eps_yoy_diff"] == pytest.approx(0.4)


def test_direct_quarters_are_deduped_before_deriving_q4():
    # A later comparative of Q2 with a start shifted by a day survives the
    # exact (start, end) first-reported rule; the end-cluster dedupe must run
    # on direct quarters BEFORE Q4 derivation, or the FY sees 4 parts and the
    # Q4 is skipped entirely.
    facts = _fy_facts(4.0, [1.0, 1.0, 1.0])
    q2 = facts[2]
    facts.append(
        _fact(
            q2["end"], 1.0, "2021-08-10",
            start=(pd.Timestamp(q2["start"]) - pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            ),
            accn="dup",
        )
    )
    quarters = quarterly_sue(
        "AAA", "1", FakeConceptEdgar({"EarningsPerShareDiluted": facts}),
        FakePolygonSource(),
    )
    assert quarters["period_end"].is_unique
    q4 = quarters[quarters["period_end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert q4["derived"] == True  # noqa: E712
    assert q4["eps_diluted"] == pytest.approx(1.0)
    assert q4["filed"] == pd.Timestamp("2021-02-01")
