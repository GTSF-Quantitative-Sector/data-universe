"""`membership.identity` offline: probing, CIK segments, linking, overrides,
evidence round-trips -- all against local stubs, no network."""

import pandas as pd
import pytest
import requests

from data_universe import config
from data_universe.membership import store
from data_universe.membership.errors import MembershipError, MembershipValidationError
from data_universe.membership.identity import resolve
from data_universe.membership.schema import (
    IDENTITY_OVERRIDES_COLUMNS,
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TICKER_INTERVALS_COLUMNS,
)

INDEX = "sp500"


class StubReference:
    """(ticker, date) -> details dict, or None for a 404. `switches`:
    {ticker: [(date, cik), ...]} -- cik is the one in effect from date on."""

    def __init__(self, records=None, switches=None, tickers_for_cik=None):
        self.records = records or {}
        self.switches = switches or {}
        self._tfc = tickers_for_cik or {}
        self.calls = []

    def tickers_for_cik(self, cik, date):
        self.calls.append(("#cik", str(cik), date))
        return self._tfc.get((str(cik), date), [])

    def _switch_cik(self, ticker, date):
        cik = None
        for d, c in self.switches.get(ticker, []):
            if pd.Timestamp(date) >= pd.Timestamp(d):
                cik = c
        return cik

    def ticker_details(self, ticker, date=None):
        self.calls.append((ticker, date))
        key = (ticker, date)
        if key in self.records:
            return self.records[key]
        if ticker in self.switches:
            cik = self._switch_cik(ticker, date)
            if cik is None:
                return None
            return {"cik": cik}
        return None


class _Resp:
    def __init__(self, code):
        self.status_code = code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(response=self)


class _Resp404Ref(StubReference):
    """Reference that raises a real 404 HTTPError for missing keys."""

    def ticker_details(self, ticker, date=None):
        self.calls.append((ticker, date))
        out = super().ticker_details(ticker, date)
        if out is None:
            raise requests.HTTPError(response=_Resp(404))
        return out


class FakeEdgar:
    """registrant_summary: per-cik overrides, else an ok summary so probed
    ciks are trusted. Set `.summaries[cik] = None` for a 404."""

    def __init__(self, summaries=None):
        self.summaries = summaries or {}
        self.calls = []

    def registrant_summary(self, cik):
        self.calls.append(cik)
        if str(cik) in self.summaries:
            return self.summaries[str(cik)]
        return {"name": "ANY REGISTRANT", "former_names": [],
                "first_filed": "1996-01-02", "has_item_202": True}


@pytest.fixture(autouse=True)
def _dir(tmp_path):
    config.reset()
    config.set_membership_tables_dir(str(tmp_path))
    yield
    config.reset()


def _tables(tmp_path, intervals, overrides=None, verified="2024-11-01"):
    store.save_ticker_intervals(
        INDEX, pd.DataFrame(intervals, columns=TICKER_INTERVALS_COLUMNS))
    store.save_manifest(INDEX, {"verified_through": verified})
    if overrides:
        pd.DataFrame(overrides, columns=IDENTITY_OVERRIDES_COLUMNS).to_csv(
            tmp_path / INDEX / "identity_overrides.csv", index=False)
    return tmp_path / INDEX


def _resolve(reference, edgar=None, **kw):
    return resolve(INDEX, reference_source=reference,
                   edgar_source=edgar or FakeEdgar(), **kw)


def _evidence(cik="", cf="", sc=""):
    return {"cik": cik, "composite_figi": cf, "share_class_figi": sc}


def test_fb_meta_figi_link_shares_one_id(tmp_path):
    _tables(tmp_path, [
        [INDEX, "FB", "2013-12-23", "2022-10-27"],
        [INDEX, "META", "2022-10-27", None],
    ])
    ref = StubReference(records={
        ("FB", "2013-12-23"): _evidence("1326801", "FIGIFB"),
        ("FB", "2022-10-26"): _evidence("1326801", "FIGIFB"),
        ("META", "2022-10-27"): _evidence("1326801", "FIGIFB"),
        ("META", "2024-11-01"): _evidence("1326801", "FIGIFB"),
    })
    result = _resolve(ref)
    sm = store.load_security_map(INDEX)
    ids = set(sm["security_id"])
    assert ids == {"FB-20131223"}
    fb_evidence = sm[sm["ticker"] == "FB"]["evidence"].iloc[0]
    meta_evidence = sm[sm["ticker"] == "META"]["evidence"].iloc[0]
    assert fb_evidence == "massive" and meta_evidence == "figi_link"
    th = store.load_ticker_history(INDEX)
    assert len(th) == 2
    ch = store.load_cik_history(INDEX)
    assert len(ch) == 1 and ch["cik"].iloc[0] == "1326801"
    assert result.links["figi"] == 1


def test_amd_gap_same_figi_merges(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AMD", "1996-01-02", "2004-01-01"],
        [INDEX, "AMD", "2005-01-01", None],
    ])
    rec = _evidence("2488", "FIGIAMD")
    ref = StubReference(records={
        ("AMD", "2003-09-11"): rec, ("AMD", "2003-12-31"): rec,
        ("AMD", "2005-01-01"): rec, ("AMD", "2024-11-01"): rec,
    })
    _resolve(ref)
    sm = store.load_security_map(INDEX)
    assert set(sm["security_id"]) == {"AMD-19960102"}
    th = store.load_ticker_history(INDEX)
    assert len(th) == 1  # merged across the gap


def test_dow_reuse_different_ciks_stays_separate(tmp_path):
    _tables(tmp_path, [
        [INDEX, "DOW", "2016-01-04", "2019-04-02"],
        [INDEX, "DOW", "2019-04-02", None],
    ])
    ref = StubReference(records={
        ("DOW", "2016-01-04"): _evidence("29915"),
        ("DOW", "2019-04-01"): _evidence("29915"),
        ("DOW", "2019-04-02"): _evidence("1751788"),
        ("DOW", "2024-11-01"): _evidence("1751788"),
    })
    result = _resolve(ref)
    sm = store.load_security_map(INDEX)
    assert len(set(sm["security_id"])) == 2
    assert result.review.empty


def test_touching_same_cik_no_figi_unique_links(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAA", "2010-01-01", "2015-01-01"],
        [INDEX, "BBB", "2015-01-01", None],
    ])
    ref = StubReference(records={
        ("AAA", "2010-01-01"): _evidence("111"),
        ("AAA", "2014-12-31"): _evidence("111"),
        ("BBB", "2015-01-01"): _evidence("111"),
        ("BBB", "2024-11-01"): _evidence("111"),
    })
    result = _resolve(ref)
    assert set(store.load_security_map(INDEX)["security_id"]) == {"AAA-20100101"}
    assert result.links["cik"] == 1


def test_ambiguous_cik_link_when_two_candidates(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAA", "2010-01-01", "2015-01-01"],
        [INDEX, "BBB", "2015-01-01", "2020-01-01"],
        [INDEX, "CCC", "2015-01-01", None],
    ])
    rec = _evidence("111")
    ref = StubReference(records={
        ("AAA", "2010-01-01"): rec, ("AAA", "2014-12-31"): rec,
        ("BBB", "2015-01-01"): rec, ("BBB", "2019-12-31"): rec,
        ("CCC", "2015-01-01"): rec, ("CCC", "2024-11-01"): rec,
    })
    result = _resolve(ref)
    ambiguous = result.review[result.review["kind"] == "ambiguous_cik_link"]
    assert len(ambiguous) == 2  # AAA->BBB and AAA->CCC


def test_same_cik_different_figis_is_a_mismatch(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAA", "2010-01-01", "2015-01-01"],
        [INDEX, "BBB", "2015-01-01", None],
    ])
    ref = StubReference(records={
        ("AAA", "2010-01-01"): _evidence("111", "F1"),
        ("AAA", "2014-12-31"): _evidence("111", "F1"),
        ("BBB", "2015-01-01"): _evidence("111", "F2"),
        ("BBB", "2024-11-01"): _evidence("111", "F2"),
    })
    result = _resolve(ref)
    assert len(set(store.load_security_map(INDEX)["security_id"])) == 2
    assert (result.review["kind"] == "same_cik_figi_mismatch").any()


def test_gap_same_cik_missing_figi_is_rejoin_unconfirmed(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAA", "2010-01-01", "2012-01-01"],
        [INDEX, "AAA", "2015-01-01", None],
    ])
    rec = _evidence("111")
    ref = StubReference(records={
        ("AAA", "2010-01-01"): rec, ("AAA", "2011-12-31"): rec,
        ("AAA", "2015-01-01"): rec, ("AAA", "2024-11-01"): rec,
    })
    result = _resolve(ref)
    assert len(set(store.load_security_map(INDEX)["security_id"])) == 2
    assert (result.review["kind"] == "rejoin_unconfirmed").any()


def test_out_of_coverage_and_all_404_are_new_default(tmp_path):
    _tables(tmp_path, [
        [INDEX, "OLD", "1996-01-02", "2000-01-01"],
        [INDEX, "GONE", "2010-01-01", "2011-01-01"],
    ])
    result = _resolve(StubReference())  # everything 404s
    sm = store.load_security_map(INDEX)
    assert (sm["evidence"] == "new_default").all()
    snf = result.review[result.review["kind"] == "symbol_not_found"]
    assert len(snf) == 1 and snf["ticker"].iloc[0] == "GONE"
    assert "2010-01-01" in snf["detail"].iloc[0]


def test_first_fact_walks_offsets(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]])
    ref = StubReference(records={
        ("AAA", "2010-01-08"): _evidence("111"),
        ("AAA", "2010-12-31"): _evidence("111"),
    })
    _resolve(ref)
    assert ("AAA", "2010-01-08") in ref.calls
    assert ("AAA", "2010-01-31") not in ref.calls


def test_figi_only_fact_when_no_cik(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]])
    rec = _evidence("", "FIGI")
    ref = StubReference(records={
        ("AAA", "2010-01-01"): rec, ("AAA", "2010-12-31"): rec,
    })
    _resolve(ref)
    sm = store.load_security_map(INDEX)
    assert sm["evidence"].iloc[0] == "massive"
    assert store.load_cik_history(INDEX).empty


def test_cik_switch_found_to_the_day(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2014-06-01", None]])
    ref = StubReference(switches={"AAA": [("2014-06-01", "111"), ("2015-01-28", "222")]})
    result = _resolve(ref)
    ch = store.load_cik_history(INDEX)
    assert list(ch["cik"]) == ["111", "222"]
    assert ch["valid_from"].tolist()[1] == "2015-01-28"
    assert ch["valid_to_excl"].tolist()[0] == "2015-01-28"
    assert (result.review["kind"] == "cik_change_within_interval").sum() == 1


def test_two_switches_give_three_segments(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2014-01-01", None]])
    ref = StubReference(switches={"AAA": [
        ("2014-01-01", "111"), ("2016-02-01", "222"), ("2018-03-01", "333")]})
    _resolve(ref)
    ch = store.load_cik_history(INDEX)
    assert list(ch["cik"]) == ["111", "222", "333"]


def test_404_stretch_after_switch_uses_first_known_date(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2014-01-01", None]])
    records = {}
    # 404s right after the true switch until 2015-02-10
    ref = StubReference(records=records,
                        switches={"AAA": [("2014-01-01", "111"),
                                          ("2015-02-10", "222")]})
    _resolve(ref)
    ch = store.load_cik_history(INDEX)
    assert ch["valid_from"].tolist()[1] == "2015-02-10"


def test_cik_starts_late_flags_and_manual_cik_clears(tmp_path):
    _tables(tmp_path, [
        [INDEX, "DIS", "1996-01-02", None],
    ])
    ref = StubReference(records={
        ("DIS", "2003-09-11"): _evidence("1744489"),
        ("DIS", "2024-11-01"): _evidence("1744489"),
    })
    edgar = FakeEdgar({"1744489": {"name": "Walt Disney", "first_filed": "2018-06-25"}})
    result = _resolve(ref, edgar)
    assert (result.review["kind"] == "cik_starts_late").any()


def test_manual_cik_override_cuts_and_suppresses(tmp_path):
    _tables(tmp_path, [
        [INDEX, "DIS", "1996-01-02", None],
    ], overrides=[
        ["cik", "DIS", "1996-01-02", "", "", "1001039", "1996-01-02", "2019-03-20", ""],
    ])
    ref = StubReference(records={
        ("DIS", "2003-09-11"): _evidence("1744489"),
        ("DIS", "2024-11-01"): _evidence("1744489"),
    })
    edgar = FakeEdgar({
        "1744489": {"name": "Walt Disney", "first_filed": "2018-06-25"},
        "1001039": {"name": "DISNEY", "first_filed": "1996-01-02"},
    })
    result = _resolve(ref, edgar)
    ch = store.load_cik_history(INDEX)
    assert set(ch["source"]) == {"manual", "massive"}
    assert not (result.review["kind"] == "cik_starts_late").any()


def test_unlink_splits_and_link_joins(tmp_path):
    _tables(tmp_path, [
        [INDEX, "FB", "2013-12-23", "2022-10-27"],
        [INDEX, "META", "2022-10-27", None],
        [INDEX, "AAA", "2010-01-01", "2011-01-01"],
        [INDEX, "BBB", "2011-01-01", None],
    ], overrides=[
        ["unlink", "FB", "2013-12-23", "META", "2022-10-27", "", "", "", ""],
        ["link", "AAA", "2010-01-01", "BBB", "2011-01-01", "", "", "", ""],
    ])
    rec = _evidence("1326801", "FIGIFB")
    ref = StubReference(records={
        ("FB", "2013-12-23"): rec, ("FB", "2022-10-26"): rec,
        ("META", "2022-10-27"): rec, ("META", "2024-11-01"): rec,
    })
    _resolve(ref)
    sm = store.load_security_map(INDEX)
    assert len(set(sm["security_id"])) == 3  # FB and META stay apart
    meta = sm[sm["ticker"] == "META"].iloc[0]
    assert meta["reviewed"] in (True, "True", "true")
    assert sm[sm["ticker"] == "BBB"]["security_id"].iloc[0] == "AAA-20100101"


def test_unlink_contradicted_by_transitive_links_raises(tmp_path):
    _tables(tmp_path, [
        [INDEX, "A", "2010-01-01", "2011-01-01"],
        [INDEX, "B", "2011-01-01", "2012-01-01"],
        [INDEX, "C", "2012-01-01", None],
    ], overrides=[
        ["link", "A", "2010-01-01", "B", "2011-01-01", "", "", "", ""],
        ["link", "B", "2011-01-01", "C", "2012-01-01", "", "", "", ""],
        ["unlink", "A", "2010-01-01", "C", "2012-01-01", "", "", "", ""],
    ])
    with pytest.raises(MembershipValidationError):
        _resolve(StubReference())


def test_symbol_override_changes_probe_symbol_and_history(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAMRQ", "2013-10-01", "2014-01-01"],
    ], overrides=[
        ["symbol", "AAMRQ", "2013-10-01", "", "", "AAL", "2003-09-11", "2013-11-01", ""],
    ])
    ref = StubReference(records={
        ("AAL", "2013-10-01"): _evidence("620520"),
        ("AAL", "2013-10-31"): _evidence("620520"),
    })
    _resolve(ref)
    sm = store.load_security_map(INDEX)
    assert sm["evidence"].iloc[0] == "massive"
    # in-window dates probe as AAL, post-window dates as AAMRQ (404)
    assert ("AAL", "2013-10-01") in ref.calls
    assert ("AAMRQ", "2013-12-31") in ref.calls
    th = store.load_ticker_history(INDEX)
    assert set(th["source"]) == {"membership", "manual"}
    assert "AAL" in set(th["ticker"])


def test_unknown_interval_and_ack(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]], overrides=[
        ["link", "AAA", "2010-01-01", "NOPE", "2001-01-01", "", "", "", ""],
    ])
    with pytest.raises(MembershipValidationError):
        _resolve(StubReference())
    # now ack a review row instead
    _tables(tmp_path, [[INDEX, "GONE", "2010-01-01", "2011-01-01"]], overrides=[
        ["ack", "", "", "", "", "symbol_not_found", "", "", ""],
    ])
    # ack needs the (ticker, start_date) of the review row it suppresses
    pd.DataFrame([
        ["ack", "GONE", "2010-01-01", "", "", "symbol_not_found", "", "", ""],
    ], columns=IDENTITY_OVERRIDES_COLUMNS).to_csv(
        tmp_path / INDEX / "identity_overrides.csv", index=False)
    result = _resolve(StubReference())
    assert not (result.review["kind"] == "symbol_not_found").any()


def test_offline_rebuild_and_evidence_pruning(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAA", "2010-01-01", "2011-01-01"],
    ])
    ref = StubReference(records={
        ("AAA", "2010-01-01"): _evidence("111"),
        ("AAA", "2010-12-31"): _evidence("111"),
    })
    _resolve(ref)
    files_before = {p.name: p.read_bytes() for p in (tmp_path / INDEX).iterdir()}
    # offline: same bytes, zero new lookups
    ref2 = StubReference()
    _resolve(ref2, offline=True)
    assert ref2.calls == []
    assert {p.name: p.read_bytes() for p in (tmp_path / INDEX).iterdir()} == files_before
    # deleting an evidence row breaks the offline rebuild
    ev = store.load_identity_evidence(INDEX)
    store.save_identity_evidence(INDEX, ev.iloc[1:])
    with pytest.raises(MembershipError):
        _resolve(ref2, offline=True)


def test_dry_run_writes_nothing_and_refetch_is_once(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]])
    ref = StubReference(records={
        ("AAA", "2010-01-01"): _evidence("111"),
        ("AAA", "2010-12-31"): _evidence("111"),
    })
    _resolve(ref, dry_run=True)
    assert not (tmp_path / INDEX / "security_map.csv").exists()
    _resolve(ref)
    n_before = len(ref.calls)
    _resolve(ref, refetch=True)
    assert len(ref.calls) > n_before  # refetch re-queries
    # unused evidence rows are pruned on write
    ev = store.load_identity_evidence(INDEX)
    assert len(ev) >= 2


def test_security_master_and_changed_ids(tmp_path):
    _tables(tmp_path, [
        [INDEX, "FB", "2013-12-23", "2022-10-27"],
        [INDEX, "META", "2022-10-27", None],
    ])
    ref = StubReference(records={
        ("FB", "2013-12-23"): _evidence("1326801", "FIGIFB", "SCFB"),
        ("FB", "2022-10-26"): _evidence("1326801", "FIGIFB", "SCFB"),
        ("META", "2022-10-27"): _evidence("1326801", "FIGIFB", "SCFB"),
        ("META", "2024-11-01"): _evidence("1326801", "FIGIFB", "SCFB"),
    })
    edgar = FakeEdgar({"1326801": {"name": "Meta Platforms", "first_filed": "2004-01-01"}})
    result = _resolve(ref, edgar)
    master = store.load_security_master(INDEX)
    row = master.iloc[0]
    assert row["name"] == "Meta Platforms"
    assert row["latest_ticker"] == "META"
    assert row["cik"] == "1326801"
    assert row["composite_figi"] == "FIGIFB"
    assert result.changed_ids == [("META", "2022-10-27", "META-20221027", "FB-20131223")]


def test_history_helpers(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]])
    ref = StubReference(records={
        ("AAA", "2010-01-01"): _evidence("111"),
        ("AAA", "2010-12-31"): _evidence("111"),
    })
    _resolve(ref)
    from data_universe.membership.history import cik_segments, symbol_segments
    segs = cik_segments(INDEX, "AAA-20100101", "2009-01-01", "2012-06-01")
    assert segs["valid_from"].min() == pd.Timestamp("2009-01-01")
    assert segs["valid_to_excl"].max() == pd.Timestamp("2012-06-02")
    assert "extended" in set(segs["source"])
    sym = symbol_segments(INDEX, "AAA-20100101", "2010-01-01", "2010-12-31")
    assert set(sym["ticker"]) == {"AAA"}
    with pytest.raises(MembershipError):
        cik_segments(INDEX, "NOPE-00000000", "2010-01-01", "2010-02-01")
    with pytest.raises(ValueError):
        cik_segments(INDEX, "AAA-20100101", "2012-01-01", "2011-01-01")
    empty = symbol_segments(INDEX, "AAA-20100101", "1980-01-01", "1990-01-01")
    assert len(empty)  # extended fills the whole window


def test_link_would_overlap_is_reviewed_not_applied(tmp_path):
    # A and C overlap; both share a FIGI with B at the same touch date, so
    # linking A->B then C->B would put overlapping A and C in one component.
    _tables(tmp_path, [
        [INDEX, "A", "2010-01-01", "2013-06-30"],
        [INDEX, "C", "2011-06-01", "2013-06-30"],
        [INDEX, "B", "2013-07-01", "2014-01-01"],
    ])
    rec = _evidence("1", "F")
    ref = StubReference(records={
        ("A", "2010-01-01"): rec, ("A", "2013-06-29"): rec,
        ("C", "2011-06-01"): rec, ("C", "2013-06-29"): rec,
        ("B", "2013-07-01"): rec, ("B", "2013-12-31"): rec,
    })
    result = _resolve(ref)
    assert (result.review["kind"] == "link_would_overlap").any()
    sm = store.load_security_map(INDEX)
    assert len(set(sm["security_id"])) == 2  # C kept its own id


def test_manual_link_that_would_overlap_raises(tmp_path):
    _tables(tmp_path, [
        [INDEX, "A", "2010-01-01", None],
        [INDEX, "B", "2011-01-01", None],
    ], overrides=[
        ["link", "A", "2010-01-01", "B", "2011-01-01", "", "", "", ""],
    ])
    with pytest.raises(MembershipValidationError):
        _resolve(StubReference())


# --- R-series: trusted cik, splits, discovery, R1/R3/R7 ------------------------


def test_r1_one_sided_cik_still_gets_a_segment(tmp_path):
    """First fact 404 + last fact with cik -> one segment over the interval."""
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]])
    ref = StubReference(records={
        ("AAA", "2010-12-31"): _evidence("111"),
    })
    _resolve(ref)
    ch = store.load_cik_history(INDEX)
    assert len(ch) == 1 and ch["cik"].iloc[0] == "111"
    assert ch["valid_from"].iloc[0] == "2010-01-01"
    # FIGI-only first fact + cik last fact -> one segment too
    _tables(tmp_path, [[INDEX, "BBB", "2010-01-01", "2011-01-01"]])
    ref = StubReference(records={
        ("BBB", "2010-01-01"): _evidence("", "FIGI"),
        ("BBB", "2010-12-31"): _evidence("222"),
    })
    _resolve(ref)
    ch = store.load_cik_history(INDEX)
    assert len(ch) == 1 and ch["cik"].iloc[0] == "222"


def test_r4_untrusted_cik_is_absent_and_reviewed(tmp_path):
    """A record whose CIK fails the SEC name check is treated as absent and
    reviewed once per (interval, cik)."""
    _tables(tmp_path, [[INDEX, "XRAY", "2010-01-01", "2011-01-01"]])
    rec = {"cik": "313616", "name": "DENTSPLY INTL INC",
           "composite_figi": "", "share_class_figi": ""}
    ref = StubReference(records={
        ("XRAY", "2010-01-01"): rec, ("XRAY", "2010-12-31"): rec})
    edgar = FakeEdgar({"313616": {
        "name": "DANAHER CORP /DE/", "former_names": [],
        "first_filed": "1996-01-02", "has_item_202": True}})
    result = _resolve(ref, edgar)
    assert store.load_cik_history(INDEX).empty
    rows = result.review[result.review["kind"] == "cik_name_mismatch"]
    assert len(rows) == 1
    assert "DANAHER" in rows["detail"].iloc[0]


def test_r4_former_name_keeps_rename_trusted(tmp_path):
    rec = {"cik": "1326801", "name": "Facebook, Inc.",
           "composite_figi": "F", "share_class_figi": ""}
    _tables(tmp_path, [[INDEX, "FB", "2013-12-23", "2014-01-01"]])
    ref = StubReference(records={
        ("FB", "2013-12-23"): rec, ("FB", "2013-12-31"): rec})
    edgar = FakeEdgar({"1326801": {
        "name": "Meta Platforms, Inc.", "former_names": ["FACEBOOK INC"],
        "first_filed": "2004-02-01", "has_item_202": True}})
    _resolve(ref, edgar)
    ch = store.load_cik_history(INDEX)
    assert len(ch) == 1 and ch["cik"].iloc[0] == "1326801"


def test_r6_split_creates_two_securities(tmp_path):
    _tables(tmp_path, [[INDEX, "CB", "1996-01-02", None]], overrides=[
        ["split", "CB", "1996-01-02", "", "", "", "2016-01-19", "", ""],
    ])
    ref = StubReference(records={
        ("CB", "2003-09-11"): _evidence("20171"),
        ("CB", "2016-01-18"): _evidence("20171"),
        ("CB", "2016-01-19"): _evidence("896159"),
        ("CB", "2024-11-01"): _evidence("896159"),
    })
    result = _resolve(ref)
    sm = store.load_security_map(INDEX)
    assert set(sm["security_id"]) == {"CB-19960102", "CB-20160119"}
    assert sm["reviewed"].all()
    mi = __import__("data_universe.membership.intervals", fromlist=["x"])
    mem = mi.membership_intervals(INDEX)
    assert len(mem) == 2
    ch = store.load_cik_history(INDEX)
    assert set(ch["cik"]) == {"20171", "896159"}
    assert result.n_unresolved == 0


def test_r6_split_bounds_raise(tmp_path):
    _tables(tmp_path, [[INDEX, "CB", "1996-01-02", "2000-01-01"]], overrides=[
        ["split", "CB", "1996-01-02", "", "", "", "2000-01-01", "", ""],
    ])
    with pytest.raises(MembershipValidationError):
        _resolve(StubReference())
    _tables(tmp_path, [[INDEX, "CB", "1996-01-02", "2000-01-01"]], overrides=[
        ["split", "CB", "1996-01-02", "", "", "", "1996-01-02", "", ""],
    ])
    with pytest.raises(MembershipValidationError):
        _resolve(StubReference())


def test_r6_link_to_split_part(tmp_path):
    """An interval ending at the split date, with part 2's FIGI, links to
    part 2 only."""
    _tables(tmp_path, [
        [INDEX, "X", "1996-01-02", None],
        [INDEX, "W", "2014-06-01", "2016-01-19"],
    ], overrides=[
        ["split", "X", "1996-01-02", "", "", "", "2016-01-19", "", ""],
    ])
    ref = StubReference(records={
        ("X", "2003-09-11"): _evidence("1", "F1"),
        ("X", "2016-01-18"): _evidence("1", "F1"),
        ("X", "2016-01-19"): _evidence("2", "F2"),
        ("X", "2024-11-01"): _evidence("2", "F2"),
        ("W", "2014-06-01"): _evidence("2", "F2"),
        ("W", "2016-01-18"): _evidence("2", "F2"),
    })
    _resolve(ref)
    sm = store.load_security_map(INDEX)
    w_id = sm[sm["ticker"] == "W"]["security_id"].iloc[0]
    part1 = sm[(sm["ticker"] == "X") & (sm["start_date"] == "1996-01-02")]
    part2 = sm[(sm["ticker"] == "X") & (sm["start_date"] == "2016-01-19")]
    assert part2["security_id"].iloc[0] == w_id  # W links into part 2
    assert part1["security_id"].iloc[0] == "X-19960102"


def test_r7_pair_reviews_suppressed_for_overridden_pairs(tmp_path):
    _tables(tmp_path, [
        [INDEX, "AAA", "2010-01-01", "2015-01-01"],
        [INDEX, "BBB", "2015-01-01", None],
    ], overrides=[
        ["link", "AAA", "2010-01-01", "BBB", "2015-01-01", "", "", "", ""],
    ])
    ref = StubReference(records={
        ("AAA", "2010-01-01"): _evidence("111", "F1"),
        ("AAA", "2014-12-31"): _evidence("111", "F1"),
        ("BBB", "2015-01-01"): _evidence("111", "F2"),
        ("BBB", "2024-11-01"): _evidence("111", "F2"),
    })
    result = _resolve(ref)
    assert result.review.empty  # figi-mismatch pair review suppressed


def test_r5_case_a_backward_chain(tmp_path):
    """MHP -> MHFI -> SPGI: the walk goes from the upstream takeover back;
    tickers_for_cik is only consulted right before each switch."""
    _tables(tmp_path, [[INDEX, "SPGI", "1996-01-02", None]])
    tfc = {
        ("64040", "2016-02-29"): ["MHFI", "DLRpC"],  # preferred filtered out
        ("64040", "2013-05-13"): ["MHP"],
    }
    ref = StubReference(
        switches={
            "MHP": [("2000-01-01", "64040")],   # c's symbol throughout coverage
            "MHFI": [("2013-05-14", "64040")],  # MHP -> MHFI rename
            "SPGI": [("2016-03-01", "64040")],  # upstream takeover
        },
        tickers_for_cik=tfc)
    result = _resolve(ref)
    th = store.load_ticker_history(INDEX).set_index("ticker")
    assert (th.loc["MHP", "valid_from"], th.loc["MHP", "valid_to_excl"],
            th.loc["MHP", "source"]) == ("1996-01-02", "2013-05-14", "massive")
    assert (th.loc["MHFI", "valid_from"], th.loc["MHFI", "valid_to_excl"],
            th.loc["MHFI", "source"]) == ("2013-05-14", "2016-03-01", "massive")
    spgi = th.loc["SPGI"]
    assert (spgi["valid_from"], spgi["source"]) == ("2016-03-01", "membership")
    ch = store.load_cik_history(INDEX)
    assert len(ch) == 1 and ch["cik"].iloc[0] == "64040"
    assert not (result.review["kind"] == "symbol_not_found").any()


def test_r5_case_a_found_from_the_end(tmp_path):
    """ANTM-like: tickers_for_cik is empty at lo -- the chain still resolves
    because the walk starts at the upstream takeover date."""
    _tables(tmp_path, [[INDEX, "ANTM", "2002-07-25", None]])
    tfc = {
        ("1156039", "2003-09-11"): [],       # vendor gap at lo
        ("1156039", "2014-09-30"): ["WLP"],
    }
    ref = StubReference(
        switches={
            "WLP": [("2000-01-01", "1156039")],
            "ANTM": [("2014-10-01", "1156039")],
        },
        tickers_for_cik=tfc)
    _resolve(ref)
    th = store.load_ticker_history(INDEX).set_index("ticker")
    assert (th.loc["WLP", "valid_from"], th.loc["WLP", "valid_to_excl"],
            th.loc["WLP", "source"]) == ("2002-07-25", "2014-10-01", "massive")
    antm = th.loc["ANTM"]
    assert (antm["valid_from"], antm["source"]) == ("2014-10-01", "membership")


def test_r5_guard_touching_interval_is_allowed(tmp_path):
    """A candidate's own index interval ending exactly at t_start does not
    trip the guard (touching is allowed)."""
    _tables(tmp_path, [
        [INDEX, "WLP", "1996-01-02", "2003-09-11"],
        [INDEX, "ANTM", "2002-07-25", None],
    ])
    tfc = {
        ("1156039", "2003-09-11"): [],
        ("1156039", "2014-09-30"): ["WLP"],
    }
    ref = StubReference(
        switches={
            "WLP": [("2000-01-01", "1156039")],
            "ANTM": [("2014-10-01", "1156039")],
        },
        tickers_for_cik=tfc)
    result = _resolve(ref)
    th = store.load_ticker_history(INDEX)
    assert "WLP" in set(th["ticker"])
    assert not (result.review["kind"] == "symbol_discovery_ambiguous").any()


def test_r5_partial_chain_keeps_found_segments(tmp_path):
    """The earliest list coming back empty ends the walk but keeps the
    segments already found; the uncovered prefix stays S/membership."""
    _tables(tmp_path, [[INDEX, "SPGI", "1996-01-02", None]])
    tfc = {
        ("64040", "2016-02-29"): ["MHFI"],
        ("64040", "2013-05-13"): [],  # vendor gap ends the walk
    }
    ref = StubReference(
        switches={
            "MHFI": [("2013-05-14", "64040")],
            "SPGI": [("2016-03-01", "64040")],
        },
        tickers_for_cik=tfc)
    _resolve(ref)
    th = store.load_ticker_history(INDEX)
    mhfi = th[th["ticker"] == "MHFI"].iloc[0]
    assert (mhfi["valid_from"], mhfi["valid_to_excl"], mhfi["source"]) == \
        ("2013-05-14", "2016-03-01", "massive")
    spgi = th[th["ticker"] == "SPGI"].sort_values("valid_from")
    assert list(spgi["valid_from"]) == ["1996-01-02", "2016-03-01"]
    assert set(spgi["source"]) == {"membership"}


def test_r5_manual_symbol_override_suppresses_discovery(tmp_path):
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]], overrides=[
        ["symbol", "AAA", "2010-01-01", "", "", "ALT", "2000-01-01", "", ""],
    ])
    ref = StubReference(records={
        ("ALT", "2010-01-01"): _evidence("111"),
        ("AAA", "2010-12-31"): _evidence("111"),
    })
    result = _resolve(ref)
    th = store.load_ticker_history(INDEX)
    assert "manual" in set(th["source"])
    assert not (result.review["kind"] == "symbol_discovery_ambiguous").any()
    assert ("#cik", "111", "2010-01-01") not in ref.calls  # no discovery probes


def test_r5_case_b_produces_symbol_reassigned(tmp_path):
    """first c1, last c2: the line changes company -- propose overrides."""
    _tables(tmp_path, [[INDEX, "AAA", "2010-01-01", "2011-01-01"]])
    records = {
        ("AAA", "2010-01-01"): _evidence("111"),
        ("AAA", "2010-06-01"): _evidence("222"),
        ("AAA", "2010-12-31"): _evidence("222"),
        ("BBB", "2010-01-01"): _evidence("222"),
    }
    tfc = {("222", "2010-01-01"): ["BBB"]}
    ref = StubReference(records=records, tickers_for_cik=tfc)
    result = _resolve(ref)
    rows = result.review[result.review["kind"] == "symbol_reassigned"]
    assert len(rows) == 1
    assert "BBB" in rows["detail"].iloc[0]
    ch = store.load_cik_history(INDEX)  # data otherwise unchanged
    assert len(ch) == 2


def test_r3_history_open_end_round_trip(tmp_path):
    """An open-ended last history row reloads from CSV (NaN, not "") and
    still extends -- a window across FB->META gives FB then META rows."""
    _tables(tmp_path, [
        [INDEX, "FB", "2013-12-23", "2022-10-27"],
        [INDEX, "META", "2022-10-27", None],
    ])
    ref = StubReference(records={
        ("FB", "2013-12-23"): _evidence("1326801", "FIGIFB"),
        ("FB", "2022-10-26"): _evidence("1326801", "FIGIFB"),
        ("META", "2022-10-27"): _evidence("1326801", "FIGIFB"),
        ("META", "2024-11-01"): _evidence("1326801", "FIGIFB"),
    })
    _resolve(ref)
    # round-trip through the committed CSV so the open end reloads as NaN
    from data_universe.membership.history import symbol_segments
    segs = symbol_segments(INDEX, "FB-20131223", "2022-10-01", "2023-01-01")
    assert list(segs["ticker"]) == ["FB", "META"]
    assert segs["valid_to_excl"].max() == pd.Timestamp("2023-01-02")
    assert "extended" not in set(segs["source"])


def test_r6_refresh_keeps_split_rows(tmp_path):
    """build_tables keeps security_map rows whose start lies strictly inside
    a derived interval (a split sub-interval), and drops them when the
    interval disappears."""
    from data_universe.membership import refresh
    existing_map = pd.DataFrame(
        [[INDEX, "CB", "1996-01-02", "CB-19960102", "massive", False],
         [INDEX, "CB", "2016-01-19", "CB-20160119", "massive", True]],
        columns=SECURITY_MAP_COLUMNS)
    ti, sm, master, dropped = refresh.build_tables(
        snapshots=[("1996-01-02", {"CB"}), ("2020-01-01", {"CB", "AAA"})],
        security_map=existing_map,
        security_master=pd.DataFrame(columns=SECURITY_MASTER_COLUMNS),
    )
    kept = set(zip(sm["ticker"], sm["start_date"]))
    assert ("CB", "2016-01-19") in kept  # sub-interval row survives


def test_r8_resolver_name_check():
    """massive_cik_resolver falls back to sec_current when the record's name
    mismatches the SEC registrant; a former-name match stays massive."""
    from data_universe.events.earnings import massive_cik_resolver

    class Src:
        def __init__(self, record):
            self.record = record

        def as_of_record(self, ticker, date):
            return self.record

        def as_of_cik(self, ticker, date):
            return str(int(self.record["cik"])) if self.record else None

    class Ed:
        def __init__(self, summary):
            self.summary = summary

        def registrant_summary(self, cik):
            return self.summary

    # XRAY-like mismatch -> None (sec_current fallback)
    rec = {"cik": "313616", "name": "DENTSPLY INTL INC"}
    res = massive_cik_resolver(Src(rec), Ed({"name": "DANAHER CORP /DE/"}))
    assert res("XRAY", "2015-01-01") is None
    # FB-like former-name match -> massive_as_of
    rec = {"cik": "1326801", "name": "Facebook, Inc."}
    res = massive_cik_resolver(Src(rec), Ed({
        "name": "Meta Platforms, Inc.", "former_names": ["FACEBOOK INC"]}))
    assert res("FB", "2020-06-01") == ("1326801", "massive_as_of")
    # summary 404 -> None
    res = massive_cik_resolver(Src(rec), Ed(None))
    assert res("FB", "2020-06-01") is None
    # no edgar -> legacy behavior
    res = massive_cik_resolver(Src(rec))
    assert res("FB", "2020-06-01") == ("1326801", "massive_as_of")
