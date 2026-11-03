"""Tests for `build_pead_panel` -- the point-in-time, index-wide PEAD event
table (v0.7.0). All offline: membership tables are written to `tmp_path`, and
the EDGAR/price/options sources are fakes."""
import numpy as np
import pandas as pd
import pytest

from data_universe import config
from data_universe.events.earnings import (
    EVENT_COLUMNS,
    build_pead_panel,
    massive_cik_resolver,
)
from data_universe.fundamentals.sue import SUE_EVENT_COLUMNS
from data_universe.membership import store
from data_universe.membership.errors import (
    MembershipError,
    MembershipStaleError,
    MembershipStaleWarning,
)
from data_universe.membership.schema import TICKER_INTERVALS_COLUMNS
from data_universe.sources.base import DataSourceError
from data_universe.sources.edgar import CONCEPT_COLUMNS
from data_universe.sources.fakes import FakePolygonSource
from tests.test_events_earnings import FakeOptionsSource

INDEX = "sp500"


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


class FakeEdgarSource:
    """ticker -> CIK map (ValueError for unknown tickers) plus per-CIK filing
    lists, so one source can serve several tickers and CIKs."""

    def __init__(self, cik_by_ticker, filings_by_cik):
        self._cik_by_ticker = cik_by_ticker
        self._filings_by_cik = filings_by_cik

    def get_cik(self, ticker):
        if ticker not in self._cik_by_ticker:
            raise ValueError(f"Ticker {ticker!r} not found in SEC company_tickers.json")
        return self._cik_by_ticker[ticker]

    def earnings_8k_filings(self, cik):
        rows = [
            {
                "filing_date": pd.Timestamp(d[:10]),
                "acceptance_datetime": pd.Timestamp(d),
                "form": "8-K",
                "items": "2.02,9.01",
            }
            for d in self._filings_by_cik.get(str(int(cik)), [])
        ]
        return pd.DataFrame(rows, columns=["filing_date", "acceptance_datetime", "form", "items"])


class FailingBarsSource(FakePolygonSource):
    """`daily_bars` raises DataSourceError for one ticker (everyone else and
    the SPY calendar behave normally)."""

    def __init__(self, fail_ticker, market=None):
        super().__init__(market)
        self._fail_ticker = fail_ticker

    def daily_bars(self, ticker, start, end):
        if ticker == self._fail_ticker:
            raise DataSourceError(f"boom {ticker}")
        return super().daily_bars(ticker, start, end)


def _save_tables(tmp_path, intervals, verified_through=None):
    config.set_membership_tables_dir(str(tmp_path))
    store.save_ticker_intervals(
        INDEX, pd.DataFrame(intervals, columns=TICKER_INTERVALS_COLUMNS)
    )
    store.save_manifest(
        INDEX,
        {
            "verified_through": (
                verified_through or pd.Timestamp.now(tz="UTC").date().isoformat()
            )
        },
    )


def _panel(start, end, edgar, **kwargs):
    return build_pead_panel(
        start,
        end,
        edgar_source=edgar,
        price_source=kwargs.pop("price_source", None) or FakePolygonSource(),
        options_source=kwargs.pop("options_source", None) or FakeOptionsSource(),
        **kwargs,
    )


def test_point_in_time_membership_uses_pre_announcement_session(tmp_path):
    _save_tables(
        tmp_path,
        [
            [INDEX, "AAA", "2024-03-01", None],          # joins mid-range
            [INDEX, "BBB", "2024-01-01", "2024-02-02"],  # leaves mid-range
            [INDEX, "FFF", "2024-06-01", None],          # joins too late for its filing
        ],
    )
    edgar = FakeEdgarSource(
        {"AAA": "1111111", "BBB": "2222222", "FFF": "6666666"},
        {
            # AAA: the Feb event precedes the join (pre_announcement 01-31) ->
            # dropped; the May event is kept.
            "1111111": ["2024-02-01T20:15:00.000Z", "2024-05-02T19:15:00.000Z"],
            # BBB leaves at end_date_excl 02-02: a pre-open filing that day is
            # kept (pre_announcement is the prior session, 02-01), a post-close
            # filing is dropped (pre_announcement is day0 = 02-02 itself).
            "2222222": ["2024-02-02T12:00:00.000Z", "2024-02-02T21:00:00.000Z"],
            "6666666": ["2024-02-01T20:15:00.000Z"],
        },
    )
    report = _panel("2024-01-01", "2024-06-01", edgar)

    assert report.succeeded == {"AAA": 1, "BBB": 1, "FFF": 0}
    aaa = report.events[report.events["ticker"] == "AAA"]
    assert list(aaa["event_date"]) == [pd.Timestamp("2024-05-02")]
    bbb = report.events[report.events["ticker"] == "BBB"].iloc[0]
    assert bbb["event_date"] == pd.Timestamp("2024-02-02")
    assert bbb["announce_timing"] == "pre_open"  # the post-close filing was dropped


def test_event_date_range_filter_is_inclusive_at_start(tmp_path):
    _save_tables(tmp_path, [[INDEX, "CCC", "2023-01-01", None]])
    edgar = FakeEdgarSource(
        {"CCC": "3333333"},
        {
            "3333333": [
                "2023-12-29T20:15:00.000Z",  # event_date 2023-12-29 -- before start
                "2024-01-01T20:15:00.000Z",  # event_date == start -- kept
                "2024-05-30T19:15:00.000Z",  # event_date 2024-05-30 -- kept
            ]
        },
    )
    report = _panel("2024-01-01", "2024-05-31", edgar)
    assert report.succeeded["CCC"] == 2
    assert list(report.events["event_date"]) == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-05-30"),
    ]
    assert report.events["event_date"].max() <= pd.Timestamp("2024-05-31")


def test_one_tickers_failure_does_not_stop_the_panel(tmp_path):
    _save_tables(
        tmp_path,
        [
            [INDEX, "AAA", "2024-03-01", None],
            [INDEX, "DDD", "2023-01-01", None],  # unknown CIK
            [INDEX, "EEE", "2023-01-01", None],  # bars source raises
        ],
    )
    edgar = FakeEdgarSource(
        {"AAA": "1111111", "EEE": "5555555"},  # no DDD -> get_cik raises
        {
            "1111111": ["2024-05-02T19:15:00.000Z"],
            "5555555": ["2024-02-01T20:15:00.000Z"],
        },
    )
    report = _panel(
        "2024-01-01", "2024-06-01", edgar, price_source=FailingBarsSource("EEE")
    )
    assert report.succeeded == {"AAA": 1}
    assert set(report.failed) == {"DDD", "EEE"}
    assert report.failed["DDD"].startswith("ValueError")
    assert report.failed["EEE"] == "DataSourceError: boom EEE"
    assert list(report.events["ticker"]) == ["AAA"]


def test_tickers_subset_restricts_candidates_and_reports_non_members(tmp_path):
    _save_tables(
        tmp_path,
        [
            [INDEX, "AAA", "2024-03-01", None],
            [INDEX, "BBB", "2024-01-01", "2024-02-02"],
        ],
    )
    edgar = FakeEdgarSource(
        {"AAA": "1111111", "BBB": "2222222"},
        {"1111111": ["2024-05-02T19:15:00.000Z"], "2222222": ["2024-02-02T12:00:00.000Z"]},
    )
    report = _panel("2024-01-01", "2024-06-01", edgar, tickers=["AAA", "ZZZ"])
    assert set(report.succeeded) == {"AAA"}
    assert report.failed["ZZZ"] == (
        "not a member of sp500 between 2024-01-01 and 2024-06-01"
    )
    assert list(report.events["ticker"]) == ["AAA"]

    # Requesting only a non-member yields an empty frame with EVENT_COLUMNS.
    report = _panel("2024-01-01", "2024-06-01", edgar, tickers=["ZZZ"])
    assert report.events.empty
    assert list(report.events.columns) == EVENT_COLUMNS
    assert report.events["pre_is_stale"].dtype == pd.BooleanDtype()


def test_cik_overrides_label_events_explicit(tmp_path):
    _save_tables(tmp_path, [[INDEX, "AAA", "2024-03-01", None]])
    edgar = FakeEdgarSource(
        {"AAA": "1111111"}, {"7777777": ["2024-05-02T19:15:00.000Z"]}
    )
    report = _panel(
        "2024-01-01",
        "2024-06-01",
        edgar,
        cik_overrides={"AAA": "0007777777"},
    )
    assert report.succeeded == {"AAA": 1}
    assert list(report.events["cik"]) == ["7777777"]  # normalized
    assert list(report.events["cik_source"]) == ["explicit"]


def test_cik_resolver_is_called_per_interval_and_its_label_recorded(tmp_path):
    _save_tables(tmp_path, [[INDEX, "AAA", "2024-03-01", None]])
    calls = []

    def resolver(ticker, as_of):
        calls.append((ticker, as_of))
        return ("8888888", "massive_asof")

    edgar = FakeEdgarSource(
        {"AAA": "1111111"}, {"8888888": ["2024-05-02T19:15:00.000Z"]}
    )
    report = _panel("2024-01-01", "2024-06-01", edgar, cik_resolver=resolver)
    # as_of = max(interval start, last session before `start`=2024-01-01)
    assert calls == [("AAA", "2024-03-01")]
    assert list(report.events["cik"]) == ["8888888"]
    assert list(report.events["cik_source"]) == ["massive_asof"]


def test_cik_resolver_returning_none_falls_through_to_sec_current(tmp_path):
    _save_tables(tmp_path, [[INDEX, "AAA", "2024-03-01", None]])
    edgar = FakeEdgarSource(
        {"AAA": "1111111"}, {"1111111": ["2024-05-02T19:15:00.000Z"]}
    )
    report = _panel(
        "2024-01-01", "2024-06-01", edgar, cik_resolver=lambda ticker, as_of: None
    )
    assert list(report.events["cik"]) == ["1111111"]
    assert list(report.events["cik_source"]) == ["sec_current"]


def test_ticker_with_two_cik_intervals_keeps_each_ciks_events_in_its_own_interval(
    tmp_path,
):
    _save_tables(
        tmp_path,
        [
            [INDEX, "REN", "2024-01-01", "2024-03-01"],
            [INDEX, "REN", "2024-03-01", None],
        ],
    )

    def resolver(ticker, as_of):
        return ("1111111", "asof") if as_of < "2024-03-01" else ("2222222", "asof")

    edgar = FakeEdgarSource(
        {"REN": "1111111"},
        {
            # The Feb event falls in the first interval and is kept under CIK
            # 1111111; the May filing under that same CIK lands outside the
            # interval and is dropped.
            "1111111": ["2024-02-01T20:15:00.000Z", "2024-05-02T19:15:00.000Z"],
            "2222222": ["2024-05-02T19:15:00.000Z"],
        },
    )
    report = _panel("2024-01-01", "2024-06-01", edgar, cik_resolver=resolver)
    assert report.succeeded == {"REN": 2}
    by_date = report.events.set_index("event_date")
    assert by_date.loc[pd.Timestamp("2024-02-01"), "cik"] == "1111111"
    assert by_date.loc[pd.Timestamp("2024-05-02"), "cik"] == "2222222"


def test_panel_output_is_sorted_by_event_date_then_ticker_with_event_columns(
    tmp_path,
):
    _save_tables(
        tmp_path,
        [[INDEX, "AAA", "2023-01-01", None], [INDEX, "BBB", "2023-01-01", None]],
    )
    edgar = FakeEdgarSource(
        {"AAA": "1111111", "BBB": "2222222"},
        {
            "1111111": ["2024-05-02T19:15:00.000Z", "2024-02-01T20:15:00.000Z"],
            "2222222": ["2024-05-02T11:00:00.000Z"],
        },
    )
    report = _panel("2024-01-01", "2024-06-01", edgar)
    assert list(report.events.columns) == EVENT_COLUMNS
    keys = list(zip(report.events["event_date"], report.events["ticker"]))
    assert keys == [
        (pd.Timestamp("2024-02-01"), "AAA"),
        (pd.Timestamp("2024-05-02"), "AAA"),
        (pd.Timestamp("2024-05-02"), "BBB"),
    ]
    assert keys == sorted(keys)
    assert report.events["pre_is_stale"].dtype == pd.BooleanDtype()


def test_stale_manifest_warns_and_strict_raises(tmp_path):
    _save_tables(
        tmp_path, [[INDEX, "AAA", "2023-01-01", None]], verified_through="2000-01-01"
    )
    edgar = FakeEdgarSource({"AAA": "1111111"}, {"1111111": ["2024-05-02T19:15:00.000Z"]})
    with pytest.warns(MembershipStaleWarning):
        report = _panel("2024-01-01", "2024-06-01", edgar)
    assert report.succeeded["AAA"] == 1
    with pytest.raises(MembershipStaleError):
        _panel("2024-01-01", "2024-06-01", edgar, strict=True)


def test_unknown_index_id_propagates_membership_error(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    edgar = FakeEdgarSource({"AAA": "1111111"}, {})
    with pytest.raises(MembershipError):
        _panel("2024-01-01", "2024-06-01", edgar, index_id="bogus")


def test_start_after_end_raises_value_error(tmp_path):
    _save_tables(tmp_path, [])
    edgar = FakeEdgarSource({}, {})
    with pytest.raises(ValueError):
        _panel("2024-06-01", "2024-01-01", edgar)


def test_panel_events_carry_the_v2_columns(tmp_path):
    _save_tables(tmp_path, [[INDEX, "AAA", "2023-01-01", None]])
    edgar = FakeEdgarSource({"AAA": "1111111"}, {"1111111": ["2024-05-02T19:15:00.000Z"]})
    report = _panel("2024-01-01", "2024-06-01", edgar)
    row = report.events.iloc[0]
    assert row["announce_timing"] == "market_hours"
    assert row["pre_announcement_session"] == pd.Timestamp("2024-05-01")
    assert np.isfinite(row["implied_move"])
    assert row["cik_source"] == "sec_current"


# --- v0.8.0: include_sue --------------------------------------------------------

def _fact(end, val, filed, start=None):
    start = start or (pd.Timestamp(end) - pd.Timedelta(days=90)).strftime("%Y-%m-%d")
    return {
        "start": start, "end": end, "val": val, "accn": "a", "fy": None,
        "fp": None, "form": "10-Q", "filed": filed, "frame": None,
    }


class EdgarWithConcept(FakeEdgarSource):
    """The panel EDGAR fake plus a `company_concept`, keyed by CIK."""

    def __init__(self, cik_by_ticker, filings_by_cik, facts_by_cik=None,
                 raise_ciks=None):
        super().__init__(cik_by_ticker, filings_by_cik)
        self._facts = facts_by_cik or {}
        self._raise_ciks = {str(int(c)) for c in (raise_ciks or set())}

    def company_concept(self, cik, tag, unit="USD/shares", taxonomy="us-gaap"):
        if str(int(cik)) in self._raise_ciks:
            raise DataSourceError(f"boom {cik}")
        facts = self._facts.get(str(int(cik)), {}).get(tag, [])
        frame = pd.DataFrame([dict(e) for e in facts], columns=CONCEPT_COLUMNS)
        for column in ("start", "end", "filed"):
            frame[column] = pd.to_datetime(frame[column])
        frame["val"] = frame["val"].astype("float64")
        frame["fy"] = frame["fy"].astype("Int64")
        return frame.sort_values(
            ["filed", "end", "accn"], na_position="last"
        ).reset_index(drop=True)


_SUE_FACTS = {
    "1111111": {
        "EarningsPerShareDiluted": [_fact("2024-03-31", 1.5, "2024-05-05")],
    },
}


def test_panel_include_sue_appends_columns_and_matches(tmp_path):
    edgar = EdgarWithConcept(
        {"AAA": "1111111"},
        {"1111111": ["2024-05-02T19:15:00.000Z"]},
        facts_by_cik=_SUE_FACTS,
    )
    _save_tables(tmp_path, [[INDEX, "AAA", "2023-01-01", None]])
    report = _panel("2024-01-01", "2024-06-01", edgar, include_sue=True)
    assert list(report.events.columns) == EVENT_COLUMNS + SUE_EVENT_COLUMNS
    assert report.sue_failed == {}
    row = report.events.iloc[0]
    assert row["eps_period_end"] == pd.Timestamp("2024-03-31")
    assert row["eps_filed"] == pd.Timestamp("2024-05-05")
    assert row["eps_diluted"] == 1.5
    assert str(report.events["eps_derived"].dtype) == "boolean"
    assert str(report.events["sue_n"].dtype) == "Int64"


def test_panel_include_sue_failure_keeps_events_and_records_it(tmp_path):
    edgar = EdgarWithConcept(
        {"AAA": "1111111", "EEE": "6666666"},
        {
            "1111111": ["2024-05-02T19:15:00.000Z"],
            "6666666": ["2024-05-02T19:15:00.000Z"],
        },
        facts_by_cik=_SUE_FACTS,
        raise_ciks={"6666666"},
    )
    _save_tables(
        tmp_path,
        [[INDEX, "AAA", "2023-01-01", None], [INDEX, "EEE", "2023-01-01", None]],
    )
    report = _panel("2024-01-01", "2024-06-01", edgar, include_sue=True)
    assert report.succeeded == {"AAA": 1, "EEE": 1}
    assert report.sue_failed == {"EEE": "DataSourceError: boom 6666666"}
    eee = report.events[report.events["ticker"] == "EEE"].iloc[0]
    assert np.isnan(eee["sue"])
    assert pd.isna(eee["eps_period_end"])
    aaa = report.events[report.events["ticker"] == "AAA"].iloc[0]
    assert aaa["eps_diluted"] == 1.5


def test_panel_without_include_sue_has_exact_event_columns(tmp_path):
    edgar = EdgarWithConcept(
        {"AAA": "1111111"},
        {"1111111": ["2024-05-02T19:15:00.000Z"]},
        facts_by_cik=_SUE_FACTS,
    )
    _save_tables(tmp_path, [[INDEX, "AAA", "2023-01-01", None]])
    report = _panel("2024-01-01", "2024-06-01", edgar)
    assert list(report.events.columns) == EVENT_COLUMNS
    assert report.sue_failed == {}


def test_empty_panel_with_include_sue_has_all_columns(tmp_path):
    edgar = EdgarWithConcept({"AAA": "1111111"}, {})
    _save_tables(tmp_path, [[INDEX, "AAA", "2023-01-01", None]])
    report = _panel("2024-01-01", "2024-06-01", edgar, include_sue=True)
    assert report.events.empty
    assert list(report.events.columns) == EVENT_COLUMNS + SUE_EVENT_COLUMNS
    assert str(report.events["sue_n"].dtype) == "Int64"
    assert str(report.events["eps_derived"].dtype) == "boolean"


# --- v0.10.0: massive_as_of CIK resolver + default wiring ------------------------


class FakeMassiveOptions(FakeOptionsSource):
    """Stands in for `MassiveOptionsSource` when the panel constructs its
    default sources: serves CIKs from a class-level map and records
    `as_of_cik(ticker, date)` calls."""

    ciks = {}
    last_instance = None

    def __init__(self):
        super().__init__()
        self.as_of_calls = []
        FakeMassiveOptions.last_instance = self

    def as_of_cik(self, ticker, date):
        self.as_of_calls.append((ticker, date))
        return self.ciks.get(ticker)


def test_massive_cik_resolver_labels_hits_and_falls_through():
    class _FakeAsOf:
        def as_of_cik(self, ticker, date):
            return {"FB": "1326801"}.get(ticker)

    resolve = massive_cik_resolver(_FakeAsOf())
    assert resolve("FB", "2020-06-01") == ("1326801", "massive_as_of")
    assert resolve("GOOGL", "2015-01-02") is None


def test_default_sources_wire_the_massive_as_of_resolver(tmp_path, monkeypatch):
    _save_tables(tmp_path, [[INDEX, "AAA", "2024-01-01", None]])
    edgar = FakeEdgarSource(
        {"AAA": "1111111"}, {"7777777": ["2024-05-02T19:15:00.000Z"]}
    )
    FakeMassiveOptions.ciks = {"AAA": "7777777"}
    monkeypatch.setattr(
        "data_universe.events.earnings.MassiveOptionsSource", FakeMassiveOptions
    )
    report = build_pead_panel("2024-04-15", "2024-05-15", edgar_source=edgar,
                              tickers=["AAA"])
    row = report.events.iloc[0]
    assert row["cik"] == "7777777"
    assert row["cik_source"] == "massive_as_of"
    # as_of = max(interval start, last session before `start`) = 2024-04-12
    assert FakeMassiveOptions.last_instance.as_of_calls == [("AAA", "2024-04-12")]


def test_default_resolver_falls_through_to_sec_current(tmp_path, monkeypatch):
    _save_tables(tmp_path, [[INDEX, "AAA", "2024-01-01", None]])
    edgar = FakeEdgarSource(
        {"AAA": "1111111"}, {"1111111": ["2024-05-02T19:15:00.000Z"]}
    )
    FakeMassiveOptions.ciks = {}  # as_of_cik unknown -> the SEC map wins
    monkeypatch.setattr(
        "data_universe.events.earnings.MassiveOptionsSource", FakeMassiveOptions
    )
    report = build_pead_panel("2024-04-15", "2024-05-15", edgar_source=edgar,
                              tickers=["AAA"])
    row = report.events.iloc[0]
    assert row["cik"] == "1111111"
    assert row["cik_source"] == "sec_current"


def test_an_explicit_resolver_beats_the_massive_default(tmp_path, monkeypatch):
    _save_tables(tmp_path, [[INDEX, "AAA", "2024-01-01", None]])
    edgar = FakeEdgarSource(
        {"AAA": "1111111"}, {"8888888": ["2024-05-02T19:15:00.000Z"]}
    )
    FakeMassiveOptions.ciks = {"AAA": "7777777"}
    monkeypatch.setattr(
        "data_universe.events.earnings.MassiveOptionsSource", FakeMassiveOptions
    )
    report = build_pead_panel(
        "2024-04-15", "2024-05-15", edgar_source=edgar, tickers=["AAA"],
        cik_resolver=lambda ticker, as_of: ("8888888", "custom"),
    )
    row = report.events.iloc[0]
    assert row["cik"] == "8888888"
    assert row["cik_source"] == "custom"
    # the default resolver was never consulted
    assert FakeMassiveOptions.last_instance.as_of_calls == []
