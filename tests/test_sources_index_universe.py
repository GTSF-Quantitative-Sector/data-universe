from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config
from data_universe.cache import reset_cache
from data_universe.sources import index_universe as iu
from data_universe.sources.massive import TICKER_COLUMNS


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch):
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.delenv("POLYGON_API_KEY", raising=False)
    config.reset()
    reset_cache()
    yield
    config.reset()
    reset_cache()


# --- HTML fixtures shaped like the real Wikipedia tables -------------------------------------


def _table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = ""
    for row in rows:
        body += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return (
        '<table class="wikitable sortable">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    )


def _page(*tables):
    return "<html><body>" + "".join(tables) + "</body></html>"


# A change-log table with a two-row header, like the ones under the Nasdaq-100 and S&P 500 lists.
_CHANGES_TABLE = (
    '<table class="wikitable"><thead>'
    '<tr><th rowspan="2">Date</th><th colspan="2">Added</th><th colspan="2">Removed</th>'
    '<th rowspan="2">Reason</th></tr>'
    "<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr></thead>"
    "<tbody><tr><td>June 22, 2026</td><td>ALAB</td><td>Astera Labs</td><td>CHTR</td>"
    "<td>Charter</td><td>Reconstitution</td></tr></tbody></table>"
)

_SP500_HEADERS = [
    "Symbol",
    "Security",
    "GICS Sector",
    "GICS Sub-Industry",
    "Headquarters Location",
    "Date added",
    "CIK",
    "Founded",
]


def _sp500_row(symbol, name, sector, sub_industry="Building Products"):
    return [symbol, name, sector, sub_industry, "HQ", "2020-01-01", "0000000001", "1900"]


def _sp500_rows(count=503):
    rows = [
        _sp500_row("MMM[a]", "3M", "Industrials"),  # footnote marker on the ticker cell
        _sp500_row("BRK.B", "Berkshire Hathaway", "Financials"),
        _sp500_row("BF.B", "Brown-Forman", "Consumer Staples"),
        _sp500_row("GOOGL", "Alphabet Inc. (Class A)", "Communication Services"),
        _sp500_row("GOOG", "Alphabet Inc. (Class C)", "Communication Services"),
    ]
    for i in range(count - len(rows)):
        rows.append(_sp500_row(f"T{i:03d}", f"Company {i}", "Industrials"))
    return rows


_DOW_TICKERS = [
    "MMM", "GOOGL", "AXP", "AMGN", "AMZN", "AAPL", "BA", "CAT", "CVX", "CSCO",
    "KO", "DIS", "GS", "HD", "HON", "IBM", "JNJ", "JPM", "MCD", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "CRM", "SHW", "TRV", "UNH", "V", "WMT",
]  # fmt: skip


def _dow_rows():
    rows = []
    for ticker in _DOW_TICKERS:
        rows.append([f"Co {ticker}", "NYSE", ticker, "Industrials", "1999-01-01", ""])
    return rows


def _nasdaq100_rows(count=101):
    rows = [
        ["GOOGL", "Alphabet Inc. (Class A)", "Technology", "Software"],
        ["GOOG", "Alphabet Inc. (Class C)", "Technology", "Software"],
        ["AAPL", "Apple Inc.", "Technology", "Computer Hardware"],
        ["MSFT", "Microsoft", "Technology", "Software"],
    ]
    for i in range(count - len(rows)):
        rows.append([f"N{i:03d}", f"Nasdaq Co {i}", "Health Care", "Biotechnology"])
    return rows


_DOW_HEADERS = ["Company", "Exchange", "Symbol", "Sector", "Date added", "Notes"]
_NASDAQ_ICB_HEADERS = ["Ticker", "Company", "ICB Industry", "ICB Subsector"]
_NASDAQ_GICS_HEADERS = ["Ticker", "Company", "GICS Sector", "GICS Sub-Industry"]


class FakeResponse:
    def __init__(self, text="", status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# --- clean_ticker ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("BRK.B", "BRK.B"),
        ("BRK-B", "BRK.B"),
        ("  aapl ", "AAPL"),
        ("MMM[1]", "MMM"),
        ("NASDAQ: AAPL", "AAPL"),
        ("\xa0", ""),
        (float("nan"), ""),
    ],
)
def test_clean_ticker(raw, expected):
    assert iu.clean_ticker(raw) == expected


# --- parse_constituents ----------------------------------------------------------------------


def test_parse_sp500_normalizes_tickers_and_reads_sector():
    html = _page(_table(_SP500_HEADERS, _sp500_rows()))
    frame = iu.parse_constituents("sp500", html, "https://example.test/sp500")
    assert list(frame.columns) == iu.UNIVERSE_COLUMNS
    assert len(frame) == 503
    tickers = frame["ticker"].tolist()
    assert tickers[0] == "MMM"  # footnote marker stripped
    assert "BRK.B" in tickers and "BF.B" in tickers
    assert "GOOGL" in tickers and "GOOG" in tickers  # both share classes kept
    row = frame.iloc[1]
    assert row["index"] == "sp500"
    assert row["name"] == "Berkshire Hathaway"
    assert row["sector"] == "Financials"
    assert row["source"] == "https://example.test/sp500"


def test_parse_dow_uses_symbol_and_sector_columns():
    frame = iu.parse_constituents("dow", _page(_table(_DOW_HEADERS, _dow_rows())))
    assert len(frame) == 30
    assert frame["ticker"].tolist() == _DOW_TICKERS
    assert frame.iloc[0]["name"] == "Co MMM"
    assert frame.iloc[0]["sector"] == "Industrials"


def test_parse_dow_accepts_the_older_industry_header():
    headers = ["Company", "Exchange", "Symbol", "Industry", "Date added", "Notes"]
    frame = iu.parse_constituents("dow", _page(_table(headers, _dow_rows())))
    assert frame.iloc[0]["sector"] == "Industrials"


def test_parse_nasdaq100_with_icb_headers():
    html = _page(_table(_NASDAQ_ICB_HEADERS, _nasdaq100_rows()))
    frame = iu.parse_constituents("nasdaq100", html)
    assert len(frame) == 101
    assert frame.iloc[0]["ticker"] == "GOOGL"
    assert frame.iloc[0]["sector"] == "Technology"


def test_parse_nasdaq100_with_gics_headers():
    html = _page(_table(_NASDAQ_GICS_HEADERS, _nasdaq100_rows()))
    frame = iu.parse_constituents("nasdaq100", html)
    assert len(frame) == 101
    assert frame.iloc[2]["sector"] == "Technology"


def test_parse_skips_decoy_tables_before_and_after_the_real_one():
    small_decoy = _table(["Ticker", "Company"], [["ZZZ", "Decoy Inc."]])
    real = _table(_NASDAQ_ICB_HEADERS, _nasdaq100_rows())
    html = _page(small_decoy, _CHANGES_TABLE, real, _CHANGES_TABLE)
    frame = iu.parse_constituents("nasdaq100", html)
    assert len(frame) == 101
    assert "ZZZ" not in frame["ticker"].tolist()


def test_parse_drops_duplicate_tickers():
    rows = _nasdaq100_rows()
    rows[5] = list(rows[2])  # repeat AAPL; the frame should keep the first one only
    frame = iu.parse_constituents("nasdaq100", _page(_table(_NASDAQ_ICB_HEADERS, rows)))
    assert frame["ticker"].tolist().count("AAPL") == 1
    assert len(frame) == 100


def test_parse_wrong_row_count_raises_with_page_summary():
    html = _page(_table(_SP500_HEADERS, _sp500_rows(count=40)))
    with pytest.raises(iu.UniverseParseError, match="layout may have changed"):
        iu.parse_constituents("sp500", html, "https://example.test/sp500")


def test_parse_missing_ticker_column_raises():
    headers = ["Company", "Exchange", "Sector", "Date added", "Notes"]
    rows = []
    for ticker in _DOW_TICKERS:
        rows.append([f"Co {ticker}", "NYSE", "Industrials", "1999-01-01", ""])
    with pytest.raises(iu.UniverseParseError):
        iu.parse_constituents("dow", _page(_table(headers, rows)))


def test_parse_page_without_tables_raises():
    with pytest.raises(iu.UniverseParseError, match="no HTML tables"):
        iu.parse_constituents("dow", "<html><body><p>Blocked</p></body></html>")


def test_parse_malformed_ticker_raises():
    rows = _dow_rows()
    rows[3][2] = "not a ticker"
    with pytest.raises(iu.UniverseParseError, match="unexpected ticker"):
        iu.parse_constituents("dow", _page(_table(_DOW_HEADERS, rows)))


def test_parse_unknown_index_raises():
    with pytest.raises(ValueError):
        iu.parse_constituents("russell2000", "<html></html>")


# --- IndexUniverseSource ---------------------------------------------------------------------


def test_source_fetches_the_right_url_with_a_user_agent():
    session = Mock()
    session.get.return_value = FakeResponse(_page(_table(_DOW_HEADERS, _dow_rows())))
    source = iu.IndexUniverseSource(session=session, user_agent="test-agent (me@example.org)")
    frame = source.dow()
    assert len(frame) == 30
    url = session.get.call_args[0][0]
    assert url == "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies"
    assert session.get.call_args[1]["headers"]["User-Agent"] == "test-agent (me@example.org)"
    assert frame.iloc[0]["source"] == url


def test_source_sp500_and_nasdaq100_helpers():
    session = Mock()
    session.get.side_effect = [
        FakeResponse(_page(_table(_SP500_HEADERS, _sp500_rows()))),
        FakeResponse(_page(_table(_NASDAQ_ICB_HEADERS, _nasdaq100_rows()))),
    ]
    source = iu.IndexUniverseSource(session=session)
    assert len(source.sp500()) == 503
    assert len(source.nasdaq100()) == 101
    assert "List_of_S%26P_500_companies" in session.get.call_args_list[0][0][0]
    assert session.get.call_args_list[1][0][0].endswith("/Nasdaq-100")


def test_source_http_error_propagates():
    session = Mock()
    session.get.return_value = FakeResponse("", status_code=403)
    with pytest.raises(RuntimeError):
        iu.IndexUniverseSource(session=session).sp500()


def test_source_unknown_index_raises():
    with pytest.raises(ValueError):
        iu.IndexUniverseSource(session=Mock()).constituents("nasdaq_listed")


# --- Massive reference: nasdaq_listed and attach_massive_reference ----------------------------


def _reference(*rows):
    records = []
    for ticker, exchange, ticker_type in rows:
        record = {column: None for column in TICKER_COLUMNS}
        record.update(
            {
                "ticker": ticker,
                "name": f"{ticker} Inc.",
                "market": "stocks",
                "type": ticker_type,
                "primary_exchange": exchange,
                "active": True,
                "cik": f"cik-{ticker}",
                "composite_figi": f"figi-{ticker}",
            }
        )
        records.append(record)
    return pd.DataFrame(records, columns=TICKER_COLUMNS)


_REFERENCE = _reference(
    ("AAPL", "XNAS", "CS"),
    ("MSFT", "XNAS", "CS"),
    ("QQQ", "XNAS", "ETF"),
    ("PDD", "XNAS", "ADRC"),
    ("JPM", "XNYS", "CS"),
)


def test_nasdaq_listed_keeps_only_nasdaq_common_stock():
    frame = iu.nasdaq_listed(_REFERENCE)
    assert list(frame.columns) == iu.UNIVERSE_COLUMNS
    assert frame["ticker"].tolist() == ["AAPL", "MSFT"]
    assert frame.iloc[0]["index"] == "nasdaq_listed"
    assert "XNAS" in frame.iloc[0]["source"]


def test_nasdaq_listed_with_no_type_filter_keeps_etfs_and_adrs():
    frame = iu.nasdaq_listed(_REFERENCE, ticker_type=None)
    assert frame["ticker"].tolist() == ["AAPL", "MSFT", "QQQ", "PDD"]


def _universe_row(index, ticker, name=None):
    return {
        "index": index,
        "ticker": ticker,
        "name": name or ticker,
        "sector": None,
        "source": "s",
        "as_of": "d",
    }


def test_attach_massive_reference_flags_and_fills():
    universe = pd.DataFrame(
        [_universe_row("dow", "AAPL", "Apple"), _universe_row("dow", "GONE", "Gone Corp")]
    )
    result = iu.attach_massive_reference(universe, _REFERENCE)
    assert result["in_massive"].tolist() == [True, False]
    assert result.iloc[0]["primary_exchange"] == "XNAS"
    assert result.iloc[0]["cik"] == "cik-AAPL"
    assert pd.isna(result.iloc[1]["primary_exchange"])
    assert "in_massive" not in universe.columns  # the input frame is left alone


# --- collect_universes -----------------------------------------------------------------------


class FakeWiki:
    """Stands in for IndexUniverseSource so no HTTP happens."""

    def __init__(self):
        self.calls = []

    def constituents(self, index):
        self.calls.append(index)
        tickers = {"sp500": ["AAPL", "JPM"], "dow": ["AAPL", "GONE"], "nasdaq100": ["MSFT"]}
        records = []
        for ticker in tickers[index]:
            records.append(_universe_row(index, ticker))
        return pd.DataFrame(records, columns=iu.UNIVERSE_COLUMNS)


def _fake_massive(reference=_REFERENCE):
    massive = Mock()
    massive.list_tickers.return_value = reference
    return massive


def test_collect_universes_with_massive_pulls_reference_once():
    massive = _fake_massive()
    universes = iu.collect_universes(massive=massive, wiki=FakeWiki())
    assert list(universes) == list(iu.ALL_INDEXES)
    massive.list_tickers.assert_called_once_with(market="stocks", active=True)
    assert universes["dow"]["in_massive"].tolist() == [True, False]
    assert universes["nasdaq_listed"]["ticker"].tolist() == ["AAPL", "MSFT"]
    assert universes["sp500"].iloc[1]["primary_exchange"] == "XNYS"


def test_collect_universes_without_massive_has_no_reference_columns():
    universes = iu.collect_universes(iu.WIKI_INDEXES, wiki=FakeWiki())
    assert list(universes) == list(iu.WIKI_INDEXES)
    assert "in_massive" not in universes["sp500"].columns


def test_collect_universes_nasdaq_listed_requires_massive():
    with pytest.raises(ValueError, match="massive"):
        iu.collect_universes(["nasdaq_listed"], wiki=FakeWiki())


def test_collect_universes_rejects_unknown_index():
    with pytest.raises(ValueError, match="russell"):
        iu.collect_universes(["russell"], wiki=FakeWiki())


def test_collect_universes_only_fetches_requested_indexes():
    wiki = FakeWiki()
    iu.collect_universes(["dow"], wiki=wiki)
    assert wiki.calls == ["dow"]


def test_collect_universes_save_writes_parquet(tmp_path):
    iu.collect_universes(["dow", "sp500"], wiki=FakeWiki(), save=True, out_dir=str(tmp_path))
    names = sorted(p.name for p in tmp_path.iterdir())
    assert len(names) == 2
    assert names[0].startswith("dow_") and names[1].startswith("sp500_")


def test_all_tickers_is_a_sorted_union():
    universes = iu.collect_universes(iu.WIKI_INDEXES, wiki=FakeWiki())
    assert iu.all_tickers(universes) == ["AAPL", "GONE", "JPM", "MSFT"]


# --- save / load -----------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path):
    frame = FakeWiki().constituents("sp500")
    path = iu.save_universe(frame, "sp500", str(tmp_path))
    assert path.endswith(".parquet")
    loaded = iu.load_universe("sp500", str(tmp_path))
    assert loaded["ticker"].tolist() == ["AAPL", "JPM"]


def test_load_universe_returns_the_newest_file(tmp_path):
    newest = FakeWiki().constituents("sp500")
    iu.save_universe(newest, "sp500", str(tmp_path))
    older = newest.iloc[:1]
    older.to_parquet(tmp_path / "sp500_2020-01-01.parquet", index=False)
    assert iu.load_universe("sp500", str(tmp_path))["ticker"].tolist() == ["AAPL", "JPM"]


def test_load_universe_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        iu.load_universe("sp500", str(tmp_path))


def test_save_defaults_to_the_configured_data_dir(tmp_path):
    config.set_data_dir(str(tmp_path))
    path = iu.save_universe(FakeWiki().constituents("dow"), "dow")
    assert path.startswith(str(tmp_path / "raw" / "universe"))


# --- command line ----------------------------------------------------------------------------


def test_main_skip_massive_collects_the_three_wikipedia_indexes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(iu, "IndexUniverseSource", FakeWiki)
    code = iu.main(
        ["--skip-massive", "--out-dir", str(tmp_path), "--config", str(tmp_path / "none.yaml")]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "sp500: 2 tickers" in output
    assert "nasdaq_listed" not in output
    assert len(list(tmp_path.glob("*.parquet"))) == 3


def test_main_with_massive_reports_tickers_missing_from_massive(tmp_path, monkeypatch, capsys):
    fake_massive = _fake_massive()
    monkeypatch.setattr(iu, "IndexUniverseSource", FakeWiki)
    monkeypatch.setattr(iu, "MassiveSource", lambda: fake_massive)
    code = iu.main(["--out-dir", str(tmp_path), "--config", str(tmp_path / "none.yaml")])
    assert code == 0
    fake_massive.ticker_details.assert_called_once_with("AAPL")
    output = capsys.readouterr().out
    assert "not found in Massive: GONE" in output
    assert "nasdaq_listed: 2 tickers" in output


def test_main_without_a_key_fails_cleanly(tmp_path, capsys):
    code = iu.main(["--out-dir", str(tmp_path), "--config", str(tmp_path / "none.yaml")])
    assert code == 1
    assert "MASSIVE_API_KEY" in capsys.readouterr().err
