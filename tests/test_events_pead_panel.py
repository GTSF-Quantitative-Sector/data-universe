"""Tests for `build_pead_panel` -- the point-in-time, index-wide PEAD event
table (v0.7.0). All offline: membership tables are written to `tmp_path`, and
the EDGAR/price/options sources are fakes."""
import numpy as np
import pandas as pd
import pytest

from data_universe import config
from data_universe.events.earnings import (
    PANEL_EVENT_COLUMNS,
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
from data_universe.membership.schema import (
    CIK_HISTORY_COLUMNS,
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TICKER_HISTORY_COLUMNS,
    TICKER_INTERVALS_COLUMNS,
)
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

    assert report.succeeded == {"AAA-20240301": 1, "BBB-20240101": 1, "FFF-20240601": 0}
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
    assert report.succeeded["CCC-20230101"] == 2
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
    assert report.succeeded == {"AAA-20240301": 1}
    assert set(report.failed) == {"DDD-20230101", "EEE-20230101"}
    assert report.failed["DDD-20230101"].startswith("ValueError")
    assert report.failed["EEE-20230101"] == "DataSourceError: boom EEE"
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
    assert set(report.succeeded) == {"AAA-20240301"}
    assert report.failed["ZZZ"] == (
        "not a member of sp500 between 2024-01-01 and 2024-06-01"
    )
    assert list(report.events["ticker"]) == ["AAA"]

    # Requesting only a non-member yields an empty frame with EVENT_COLUMNS.
    report = _panel("2024-01-01", "2024-06-01", edgar, tickers=["ZZZ"])
    assert report.events.empty
    assert list(report.events.columns) == PANEL_EVENT_COLUMNS
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
    assert report.succeeded == {"AAA-20240301": 1}
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
    assert report.succeeded == {"REN-20240101": 1, "REN-20240301": 1}
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
    assert list(report.events.columns) == PANEL_EVENT_COLUMNS
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
    assert report.succeeded["AAA-20230101"] == 1
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
    assert list(report.events.columns) == PANEL_EVENT_COLUMNS + SUE_EVENT_COLUMNS
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
    assert report.succeeded == {"AAA-20230101": 1, "EEE-20230101": 1}
    assert report.sue_failed == {"EEE-20230101": "DataSourceError: boom 6666666"}
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
    assert list(report.events.columns) == PANEL_EVENT_COLUMNS
    assert report.sue_failed == {}


def test_empty_panel_with_include_sue_has_all_columns(tmp_path):
    edgar = EdgarWithConcept({"AAA": "1111111"}, {})
    _save_tables(tmp_path, [[INDEX, "AAA", "2023-01-01", None]])
    report = _panel("2024-01-01", "2024-06-01", edgar, include_sue=True)
    assert report.events.empty
    assert list(report.events.columns) == PANEL_EVENT_COLUMNS + SUE_EVENT_COLUMNS
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

    def as_of_record(self, ticker, date):
        self.as_of_calls.append((ticker, date))
        cik = self.ciks.get(ticker)
        return {"cik": cik, "name": "X"} if cik else None


def test_massive_cik_resolver_labels_hits_and_falls_through():
    class _FakeAsOf:
        def as_of_record(self, ticker, date):
            rec = {"FB": {"cik": "1326801", "name": "Meta Platforms"}}
            return rec.get(ticker)

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


# --- Phase 4b: the identity path (cik_history / ticker_history) ----------------


def _save_identity_tables(tmp_path, intervals, smap, thist, chist):
    """ticker_intervals + the identity tables a security-level panel reads."""
    _save_tables(tmp_path, intervals)
    store.save_security_map(
        INDEX, pd.DataFrame(smap, columns=SECURITY_MAP_COLUMNS))
    store.save_ticker_history(
        INDEX, pd.DataFrame(thist, columns=TICKER_HISTORY_COLUMNS))
    store.save_cik_history(
        INDEX, pd.DataFrame(chist, columns=CIK_HISTORY_COLUMNS))
    ids = sorted({row[3] for row in smap})
    store.save_security_master(INDEX, pd.DataFrame(
        [[sid, "", smap[0][1], "", "", "", ""] for sid in ids],
        columns=SECURITY_MASTER_COLUMNS))


def _mdt_tables(tmp_path):
    """One MDT interval/security with a mid-interval CIK switch."""
    _save_identity_tables(
        tmp_path,
        [[INDEX, "MDT", "1996-01-02", None]],
        [[INDEX, "MDT", "1996-01-02", "MDT-19960102", "massive", False]],
        [["MDT-19960102", "MDT", "1996-01-02", "", "membership"]],
        [["MDT-19960102", "64670", "1996-01-02", "2015-01-28", "massive"],
         ["MDT-19960102", "1613103", "2015-01-28", "", "massive"]],
    )


def test_identity_mid_interval_cik_switch_keeps_both_sides(tmp_path):
    """a. Old-CIK filings before AND 20 days after the switch (inside the
    100-day slack) are kept with the old CIK; new-CIK filings keep theirs;
    next_earnings_session is continuous across the switch."""
    _mdt_tables(tmp_path)
    edgar = FakeEdgarSource(
        {"MDT": "1613103"},
        {"64670": ["2014-10-20T20:15:00.000Z", "2015-02-10T20:15:00.000Z"],
         "1613103": ["2015-05-15T20:15:00.000Z"]},
    )
    report = _panel("2014-10-01", "2015-06-30", edgar)
    assert report.succeeded == {"MDT-19960102": 3}
    rows = report.events.sort_values("event_date")
    assert list(rows["cik"]) == ["64670", "64670", "1613103"]
    assert list(rows["cik_source"]) == ["cik_history"] * 3
    assert list(rows["next_earnings_session"].dropna()) == list(
        rows["event_date"].iloc[1:])


def test_identity_old_cik_filing_past_slack_is_dropped(tmp_path):
    """b. A filing of the old CIK 150 days after the switch is outside the
    +100d window and dropped."""
    _mdt_tables(tmp_path)
    edgar = FakeEdgarSource(
        {"MDT": "1613103"},
        {"64670": ["2015-07-01T20:15:00.000Z"],
         "1613103": ["2015-05-15T20:15:00.000Z"]},
    )
    report = _panel("2014-10-01", "2015-09-30", edgar)
    assert list(report.events["cik"]) == ["1613103"]


def test_identity_rename_uses_the_symbol_at_filing_time(tmp_path):
    """c. FB-era filings carry ticker FB, META-era ones META, under one
    security_id."""
    _save_identity_tables(
        tmp_path,
        [[INDEX, "FB", "2013-12-23", "2022-10-27"],
         [INDEX, "META", "2022-10-27", None]],
        [[INDEX, "FB", "2013-12-23", "FB-20131223", "massive", False],
         [INDEX, "META", "2022-10-27", "FB-20131223", "figi_link", False]],
        [["FB-20131223", "FB", "2013-12-23", "2022-06-09", "membership"],
         ["FB-20131223", "META", "2022-06-09", "", "membership"]],
        [["FB-20131223", "1326801", "2013-12-23", "", "massive"]],
    )
    edgar = FakeEdgarSource(
        {"META": "1326801"},
        {"1326801": ["2022-05-25T20:15:00.000Z", "2022-07-15T20:15:00.000Z"]},
    )
    report = _panel("2022-04-01", "2022-11-01", edgar)
    assert report.succeeded == {"FB-20131223": 2}
    rows = report.events.sort_values("event_date")
    assert list(rows["ticker"]) == ["FB", "META"]
    assert set(rows["security_id"]) == {"FB-20131223"}


def test_identity_same_day_filings_dedupe_to_the_home_cik(tmp_path):
    """d. Co-filed duplicates under both CIKs give one event; a same-day
    different-accession pair keeps the home-CIK filing."""
    _mdt_tables(tmp_path)
    edgar = FakeEdgarSource(
        {"MDT": "1613103"},
        {"64670": ["2015-02-10T20:15:00.000Z"],
         "1613103": ["2015-02-10T20:15:00.000Z"]},
    )
    report = _panel("2014-10-01", "2015-06-30", edgar)
    assert len(report.events) == 1
    assert report.events["cik"].iloc[0] == "1613103"
    # same day, different accessions -> still one event, the home one
    edgar = FakeEdgarSource(
        {"MDT": "1613103"},
        {"64670": ["2015-02-10T18:00:00.000Z"],
         "1613103": ["2015-02-10T21:30:00.000Z"]},
    )
    report = _panel("2014-10-01", "2015-06-30", edgar)
    assert len(report.events) == 1
    assert report.events["cik"].iloc[0] == "1613103"


def test_identity_tickers_filter_matches_history_and_misses(tmp_path):
    """e. tickers=["META"] selects the FB security (META in its
    ticker_history); tickers=["NOPE"] lands in failed."""
    _save_identity_tables(
        tmp_path,
        [[INDEX, "FB", "2013-12-23", "2022-10-27"],
         [INDEX, "META", "2022-10-27", None]],
        [[INDEX, "FB", "2013-12-23", "FB-20131223", "massive", False],
         [INDEX, "META", "2022-10-27", "FB-20131223", "figi_link", False]],
        [["FB-20131223", "FB", "2013-12-23", "2022-06-09", "membership"],
         ["FB-20131223", "META", "2022-06-09", "", "membership"]],
        [["FB-20131223", "1326801", "2013-12-23", "", "massive"]],
    )
    edgar = FakeEdgarSource(
        {"META": "1326801"}, {"1326801": ["2022-05-25T20:15:00.000Z"]})
    report = _panel("2022-04-01", "2022-11-01", edgar, tickers=["META"])
    assert report.succeeded == {"FB-20131223": 1}
    assert report.events["ticker"].iloc[0] == "FB"
    report = _panel("2022-04-01", "2022-11-01", edgar, tickers=["NOPE"])
    assert report.failed["NOPE"].startswith("not a member")


def test_identity_cik_override_labels_events_explicit(tmp_path):
    """f. A cik_overrides entry on the security's ticker collapses the
    cik rows to one explicit CIK."""
    _mdt_tables(tmp_path)
    edgar = FakeEdgarSource(
        {"MDT": "999"},
        {"999": ["2014-10-20T20:15:00.000Z", "2015-05-15T20:15:00.000Z"]},
    )
    report = _panel(
        "2014-10-01", "2015-06-30", edgar, cik_overrides={"MDT": "999"})
    assert set(report.events["cik"]) == {"999"}
    assert set(report.events["cik_source"]) == {"explicit"}


def test_identity_cik_gaps_reported_for_empty_cik_rows(tmp_path):
    """g. A cik row with >=180d of membership overlap and no home filing
    lands in cik_gaps."""
    _mdt_tables(tmp_path)
    edgar = FakeEdgarSource(
        {"MDT": "1613103"},
        {"64670": ["2014-10-20T20:15:00.000Z"]},  # nothing for 1613103
    )
    report = _panel("2014-10-01", "2015-09-30", edgar)
    gaps = report.cik_gaps["MDT-19960102"]
    assert "1613103" in gaps and "no 8-K Item 2.02" in gaps


def test_identity_sue_uses_predecessor_ciks(tmp_path, monkeypatch):
    """h. include_sue calls quarterly_sue with the security's other
    cik_history CIKs as predecessor_ciks."""
    _mdt_tables(tmp_path)
    calls = {}

    def spy(symbol, cik, edgar_source, price_source, **kwargs):
        calls["symbol"] = symbol
        calls["cik"] = cik
        calls["predecessor_ciks"] = kwargs.get("predecessor_ciks")
        return pd.DataFrame(columns=QUARTER_COLUMNS)

    from data_universe.fundamentals.sue import QUARTER_COLUMNS
    monkeypatch.setattr(
        "data_universe.events.earnings.quarterly_sue", spy)
    edgar = FakeEdgarSource(
        {"MDT": "1613103"}, {"1613103": ["2015-05-15T20:15:00.000Z"]})
    _panel("2015-01-01", "2015-06-30", edgar, include_sue=True)
    assert calls["symbol"] == "MDT"
    assert calls["cik"] == "1613103"
    assert calls["predecessor_ciks"] == ["64670"]


def test_identity_tickers_filter_before_rename_returns_old_era(tmp_path):
    """tickers=["META"] with a window entirely before the rename still
    selects FB-20131223 and returns its FB-ticker events."""
    _save_identity_tables(
        tmp_path,
        [[INDEX, "FB", "2013-12-23", "2022-10-27"],
         [INDEX, "META", "2022-10-27", None]],
        [[INDEX, "FB", "2013-12-23", "FB-20131223", "massive", False],
         [INDEX, "META", "2022-10-27", "FB-20131223", "figi_link", False]],
        [["FB-20131223", "FB", "2013-12-23", "2022-06-09", "membership"],
         ["FB-20131223", "META", "2022-06-09", "", "membership"]],
        [["FB-20131223", "1326801", "2013-12-23", "", "massive"]],
    )
    edgar = FakeEdgarSource(
        {"META": "1326801"}, {"1326801": ["2015-05-13T20:15:00.000Z"]})
    report = _panel("2015-04-01", "2015-06-30", edgar, tickers=["META"])
    assert report.succeeded == {"FB-20131223": 1}
    assert report.events["ticker"].iloc[0] == "FB"
    assert report.events["security_id"].iloc[0] == "FB-20131223"


def test_identity_reused_ticker_selects_every_security(tmp_path):
    """A reused ticker selects every security that ever used it and has
    candidate pieces in the window (e.g. two Dow companies)."""
    _save_identity_tables(
        tmp_path,
        [[INDEX, "DOW", "1996-01-02", "2017-09-01"],
         [INDEX, "DOW", "2019-04-02", None]],
        [[INDEX, "DOW", "1996-01-02", "DOW-19960102", "massive", False],
         [INDEX, "DOW", "2019-04-02", "DOW-20190402", "massive", False]],
        [["DOW-19960102", "DOW", "1996-01-02", "2017-09-01", "membership"],
         ["DOW-20190402", "DOW", "2019-04-02", "", "membership"]],
        [["DOW-19960102", "29915", "1996-01-02", "2017-09-01", "massive"],
         ["DOW-20190402", "1751788", "2019-04-02", "", "massive"]],
    )
    edgar = FakeEdgarSource(
        {"DOW": "1751788"},
        {"29915": ["2017-02-14T21:00:00.000Z"],
         "1751788": ["2020-02-05T21:00:00.000Z"]},
    )
    report = _panel("2016-10-01", "2020-06-30", edgar, tickers=["DOW"])
    assert set(report.succeeded) == {"DOW-19960102", "DOW-20190402"}
    assert set(report.events["security_id"]) == {"DOW-19960102", "DOW-20190402"}
