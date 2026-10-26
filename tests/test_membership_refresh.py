import datetime as dt
import hashlib

import pandas as pd
import pytest
import requests

from data_universe import config
from data_universe.membership import intervals, refresh, store
from data_universe.membership.errors import MembershipValidationError
from data_universe.membership.schema import (
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TICKER_INTERVALS_COLUMNS,
)

SHA = "a" * 40
NOW = dt.datetime(2026, 10, 26, 18, 30, tzinfo=dt.timezone.utc)
RAW_URL = (
    f"https://raw.githubusercontent.com/fja05680/sp500/{SHA}/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20%28Updated%29.csv"
)
BASE = [f"T{i:03d}" for i in range(500)]


@pytest.fixture(autouse=True)
def _reset_config():
    config.reset()
    yield
    config.reset()


def _csv(rows) -> bytes:
    lines = ["date,tickers"] + [f'{date},"{",".join(sorted(members))}"' for date, members in rows]
    return ("\n".join(lines) + "\n").encode()


def _upstream() -> bytes:
    """T000 leaves on 2020-02-03 (NEW replaces it) and rejoins on 2020-03-02 (T499 leaves)."""
    return _csv(
        [
            ("2020-01-02", BASE),
            ("2020-02-03", BASE[1:] + ["NEW"]),
            ("2020-03-02", BASE[:-1] + ["NEW"]),
        ]
    )


class _Response:
    def __init__(self, status_code=200, json_data=None, content=b""):
        self.status_code = status_code
        self._json_data = json_data
        self.content = content

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


class _Session:
    def __init__(self, csv_bytes: bytes):
        self.csv_bytes = csv_bytes
        self.urls = []

    def get(self, url, headers=None, timeout=None):
        self.urls.append(url)
        if url.startswith("https://api.github.com/repos/fja05680/sp500/commits/"):
            commit = {"committer": {"date": "2026-09-07T14:56:49Z"}}
            return _Response(json_data={"sha": SHA, "commit": commit})
        if url == RAW_URL:
            return _Response(content=self.csv_bytes)
        return _Response(status_code=404)


def _tables_dir(tmp_path):
    config.set_membership_tables_dir(str(tmp_path))
    return tmp_path / "sp500"


def test_parse_snapshots_reads_upstream_format():
    raw = b'\xef\xbb\xbfdate,tickers\r\n2020-01-02,"AAA,BBB"\r\n2020-02-03,"BBB, CCC"\r\n'
    assert refresh.parse_snapshots(raw) == [
        ("2020-01-02", {"AAA", "BBB"}),
        ("2020-02-03", {"BBB", "CCC"}),
    ]


@pytest.mark.parametrize(
    "raw",
    [
        b'day,tickers\n2020-01-02,"AAA"\n',
        b'date,tickers\n2020-02-03,"AAA"\n2020-01-02,"AAA"\n',
        b'date,tickers\n2020-01-02,"AAA"\n2020-01-02,"BBB"\n',
        b'date,tickers\n2020-13-01,"AAA"\n',
        b'date,tickers\n20200102,"AAA"\n',
        b'date,tickers\n2020-01-02,"AAA,,BBB"\n',
        b'date,tickers\n2020-01-02,"AAA",extra\n',
        b"date,tickers\n",
    ],
)
def test_parse_snapshots_rejects_malformed_files(raw):
    with pytest.raises(MembershipValidationError):
        refresh.parse_snapshots(raw)


def test_refresh_writes_validated_tables_and_manifest(tmp_path):
    target = _tables_dir(tmp_path)
    raw = _upstream()
    session = _Session(raw)

    result = refresh.refresh(session=session, now=NOW)

    assert session.urls == [
        "https://api.github.com/repos/fja05680/sp500/commits/HEAD",
        RAW_URL,
    ]
    assert result.written and result.commit_sha == SHA
    assert result.n_intervals_before == 0 and result.n_intervals_after == 502
    assert sorted(path.name for path in target.iterdir()) == sorted(refresh.WRITTEN_FILES)

    ticker_intervals = store.load_ticker_intervals("sp500").set_index(["ticker", "start_date"])
    assert ticker_intervals.loc[("T000", "2020-01-02"), "end_date_excl"] == "2020-02-03"
    assert pd.isna(ticker_intervals.loc[("T000", "2020-03-02"), "end_date_excl"])
    assert ticker_intervals.loc[("T499", "2020-01-02"), "end_date_excl"] == "2020-03-02"
    assert pd.isna(ticker_intervals.loc[("NEW", "2020-02-03"), "end_date_excl"])

    security_map = store.load_security_map("sp500").set_index(["ticker", "start_date"])
    assert security_map.loc[("T000", "2020-03-02"), "security_id"] == "T000-20200302"
    assert security_map.loc[("NEW", "2020-02-03"), "security_id"] == "NEW-20200203"
    assert set(security_map["evidence"]) == {"new_default"}
    assert not security_map["reviewed"].any()
    master = store.load_security_master("sp500").set_index("security_id")
    assert master.loc["NEW-20200203", "latest_ticker"] == "NEW"
    assert len(master) == 502

    assert store.load_manifest("sp500") == {
        "n_derived_intervals": 502,
        "n_snapshot_rows": 3,
        "retrieved_at": "2026-10-26T18:30:00+00:00",
        "tool_version": refresh.TOOL_VERSION,
        "upstream_commit_date": "2026-09-07T14:56:49Z",
        "upstream_commit_sha": SHA,
        "upstream_filename": refresh.UPSTREAM_FILENAME,
        "upstream_last_snapshot_date": "2020-03-02",
        "upstream_repo": "fja05680/sp500",
        "upstream_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_through": "2026-10-26",
    }
    assert len(intervals.membership_intervals("sp500")) == 502


def test_refresh_keeps_reviewed_mappings_and_reports_dropped_ones(tmp_path):
    target = _tables_dir(tmp_path)
    store.save_ticker_intervals(
        "sp500",
        pd.DataFrame(
            [["sp500", "GONE", "2019-01-02", None], ["sp500", "T000", "2020-01-02", None]],
            columns=TICKER_INTERVALS_COLUMNS,
        ),
    )
    store.save_security_map(
        "sp500",
        pd.DataFrame(
            [
                ["sp500", "GONE", "2019-01-02", "GONE-20190102", "override", True],
                ["sp500", "T000", "2020-01-02", "T000-20200102", "new_default", False],
                ["sp500", "T000", "2020-03-02", "T000-20200102", "override", True],
            ],
            columns=SECURITY_MAP_COLUMNS,
        ),
    )
    store.save_security_master(
        "sp500",
        pd.DataFrame(
            [["T000-20200102", "Tee Zero Inc", "T000", "123", None, None, "rejoined 2020"]],
            columns=SECURITY_MASTER_COLUMNS,
        ),
    )

    result = refresh.refresh(session=_Session(_upstream()), now=NOW)

    assert result.removed == [("GONE", "2019-01-02")]
    assert ("T000", "2020-01-02", None, "2020-02-03") in result.end_changed
    assert result.n_reviewed_kept == 1
    assert result.dropped_reviewed == [("GONE", "2019-01-02", "GONE-20190102")]
    summary = refresh.format_summary(result)
    assert "WARNING: dropped reviewed mapping GONE from 2019-01-02 -> GONE-20190102" in summary

    security_map = store.load_security_map("sp500").set_index(["ticker", "start_date"])
    rejoin = security_map.loc[("T000", "2020-03-02")]
    assert (rejoin["security_id"], rejoin["evidence"], rejoin["reviewed"]) == (
        "T000-20200102",
        "override",
        True,
    )
    assert "GONE" not in set(security_map.index.get_level_values("ticker"))
    master = store.load_security_master("sp500").set_index("security_id")
    assert master.loc["T000-20200102", "name"] == "Tee Zero Inc"
    assert master.loc["T000-20200102", "notes"] == "rejoined 2020"
    assert "T000-20200302" not in master.index and "GONE-20190102" not in master.index

    t000 = intervals.membership_intervals("sp500").query("security_id == 'T000-20200102'")
    assert list(t000["start_date"]) == ["2020-01-02", "2020-03-02"]
    assert t000["end_date_excl"].iloc[0] == "2020-02-03"
    assert pd.isna(t000["end_date_excl"].iloc[1])
    assert (target / "manifest.json").exists()


def test_refresh_rejects_overlapping_mapping_without_touching_committed_tables(tmp_path):
    target = _tables_dir(tmp_path)
    store.save_security_map(
        "sp500",
        pd.DataFrame(
            [
                ["sp500", "T001", "2020-01-02", "SHARED", "override", True],
                ["sp500", "NEW", "2020-02-03", "SHARED", "override", True],
            ],
            columns=SECURITY_MAP_COLUMNS,
        ),
    )
    before = (target / "security_map.csv").read_bytes()

    with pytest.raises(MembershipValidationError):
        refresh.refresh(session=_Session(_upstream()), now=NOW)

    assert sorted(path.name for path in target.iterdir()) == ["security_map.csv"]
    assert (target / "security_map.csv").read_bytes() == before


def test_refresh_dry_run_writes_nothing(tmp_path):
    target = _tables_dir(tmp_path)
    result = refresh.refresh(dry_run=True, session=_Session(_upstream()), now=NOW)
    assert not result.written
    assert len(result.added) == 502
    assert not target.exists()
    summary = refresh.format_summary(result, max_listed=2)
    assert "... and 500 more" in summary
    assert "dry run: nothing written" in summary


def test_refresh_rejects_member_count_outside_expected_range(tmp_path):
    target = _tables_dir(tmp_path)
    raw = _csv([("2020-01-02", BASE[:10])])
    with pytest.raises(MembershipValidationError, match="10 members"):
        refresh.refresh(session=_Session(raw), now=NOW)
    assert not target.exists()


def test_refresh_resolves_an_explicit_ref(tmp_path):
    _tables_dir(tmp_path)
    session = _Session(_upstream())
    refresh.refresh(ref="master", dry_run=True, session=session, now=NOW)
    assert session.urls[0] == "https://api.github.com/repos/fja05680/sp500/commits/master"


def test_main_prints_a_summary(tmp_path, monkeypatch, capsys):
    _tables_dir(tmp_path)
    monkeypatch.setattr(refresh.requests, "Session", lambda: _Session(_upstream()))
    assert refresh.main(["--dry-run"]) == 0
    output = capsys.readouterr().out
    assert f"upstream: fja05680/sp500@{SHA}" in output
    assert "intervals: 0 -> 502 (502 new" in output
