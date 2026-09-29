"""Collect index universes -- S&P 500, Dow 30, Nasdaq-100, all Nasdaq-listed stocks -- and
cross-check them against Massive (formerly Polygon.io) reference data.

Where each list comes from:

- `sp500`, `dow`, `nasdaq100`: the constituent tables on Wikipedia. Massive publishes index
  *prices* (I:SPX, I:DJI, I:NDX) but no constituent lists, so membership has to come from
  elsewhere. Wikipedia is free and keyed by ticker, but it is not an official source.
- `nasdaq_listed`: every active common stock whose primary exchange is Nasdaq (MIC "XNAS"),
  straight from Massive's reference data. This is *not* the Nasdaq-100.

**All of these are today's membership.** Using them to pick a backtest universe over past
dates introduces survivorship bias. For point-in-time S&P 500 use
`data_universe.sources.sec_adapter.sp500(query_date=...)` instead.

Tickers are stored in Massive's format: share classes use a dot (`BRK.B`, `BF.B`). Sector
labels are whatever the source page uses -- GICS for the S&P 500, Wikipedia's own "Sector"
for the Dow, ICB for the Nasdaq-100 -- so they are not comparable across indexes.

Every parsed table is checked against an expected row count, and the parser raises
`UniverseParseError` rather than return a partial list if Wikipedia's layout changes.

Usage:

    python -m data_universe.sources.index_universe                  # all four, with Massive
    python -m data_universe.sources.index_universe --skip-massive   # constituents only

or from Python:

    from data_universe.sources.index_universe import IndexUniverseSource, collect_universes
    from data_universe.sources.massive import MassiveSource

    universes = collect_universes(massive=MassiveSource(), save=True)
    universes["sp500"]["ticker"].tolist()
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import requests

from data_universe import config
from data_universe.sources.base import DataSourceError
from data_universe.sources.massive import MassiveSource

WIKI_INDEXES = ("sp500", "dow", "nasdaq100")
ALL_INDEXES = WIKI_INDEXES + ("nasdaq_listed",)

UNIVERSE_COLUMNS = ["index", "ticker", "name", "sector", "source", "as_of"]
REFERENCE_FIELDS = ["primary_exchange", "type", "cik", "composite_figi", "share_class_figi"]

_WIKI_URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "dow": "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}

# (min, max) plausible row count of each constituents table. Wide enough for normal
# membership changes (503 stocks in the S&P 500, 101 in the Nasdaq-100 because of dual share
# classes), tight enough to reject the wrong table or a half-parsed page.
_EXPECTED_ROWS = {"sp500": (490, 515), "dow": (30, 30), "nasdaq100": (95, 110)}

# Wikipedia has renamed these headers before (the Nasdaq-100 table moved from GICS to ICB
# sector columns), so each field accepts several header names, in order of preference.
_TICKER_HEADERS = ("symbol", "ticker")
_NAME_HEADERS = ("security", "company")
_SECTOR_HEADERS = ("gics sector", "sector", "icb industry", "industry")

_TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z0-9]{1,2})?$")

# Wikimedia asks automated clients to send a User-Agent that identifies them and gives a
# contact. Replace the placeholder address with a real one for regular use.
_DEFAULT_USER_AGENT = "data_universe (contact: gtsf-quant@example.org)"


class UniverseParseError(DataSourceError):
    """Raised when a constituents page doesn't look the way the parser expects."""


def _strip_footnotes(text: str) -> str:
    """Remove Wikipedia footnote markers such as "[1]" or "[a]"."""
    return re.sub(r"\[.*?\]", "", text)


def _normalize_header(column) -> str:
    return _strip_footnotes(str(column)).strip().lower()


def _find_column(headers: List[str], candidates: Sequence[str]) -> Optional[int]:
    """Position of the first header matching any candidate (candidates in preference order)."""
    for candidate in candidates:
        for position in range(len(headers)):
            if headers[position] == candidate:
                return position
    return None


def clean_ticker(raw) -> str:
    """Normalize a scraped ticker cell to Massive's format.

    Drops footnote markers, any "NASDAQ:" style prefix and whitespace, uppercases, and
    turns a dash into a dot ("BRK-B" -> "BRK.B"). Returns "" for an empty cell.
    """
    if pd.isna(raw):
        return ""
    text = _strip_footnotes(str(raw).replace("\xa0", " "))
    text = text.split(":")[-1].strip().upper()
    return text.replace("-", ".")


def _cell_text(table: pd.DataFrame, row: int, column: Optional[int]) -> Optional[str]:
    if column is None:
        return None
    value = table.iloc[row, column]
    if pd.isna(value):
        return None
    text = _strip_footnotes(str(value)).strip()
    if text == "":
        return None
    return text


def _describe_tables(tables: List[pd.DataFrame]) -> str:
    """Short summary of what was on the page, for error messages."""
    parts = []
    for table in tables:
        if isinstance(table.columns, pd.MultiIndex):
            parts.append(f"{len(table)} rows x multi-level header")
        else:
            headers = [_normalize_header(c) for c in table.columns]
            parts.append(f"{len(table)} rows {headers[:4]}")
    return "; ".join(parts)


def parse_constituents(index: str, html: str, source_url: str = "") -> pd.DataFrame:
    """Parse a Wikipedia page's constituents table into a universe frame.

    Picks the one table that has a ticker column ("Symbol" or "Ticker") and a row count in
    the expected range for `index`, so decoy tables (change logs, annual returns, ...) on
    the same page are skipped.

    Args:
        index: One of `WIKI_INDEXES`.
        html: The page's HTML.
        source_url: Recorded in the `source` column.

    Returns:
        A frame with the columns in `UNIVERSE_COLUMNS`, one row per ticker, in page order.

    Raises:
        UniverseParseError: If no table matches, or a ticker cell is malformed.
    """
    if index not in _EXPECTED_ROWS:
        raise ValueError(f"Unknown Wikipedia index {index!r}; expected one of {WIKI_INDEXES}")
    low, high = _EXPECTED_ROWS[index]

    try:
        tables = pd.read_html(StringIO(html), flavor="lxml")
    except ValueError as exc:  # pandas raises ValueError when the page has no tables
        raise UniverseParseError(f"{index}: no HTML tables found on {source_url}") from exc

    table = None
    headers: List[str] = []
    for candidate in tables:
        if isinstance(candidate.columns, pd.MultiIndex):
            continue
        candidate_headers = [_normalize_header(c) for c in candidate.columns]
        if _find_column(candidate_headers, _TICKER_HEADERS) is None:
            continue
        if low <= len(candidate) <= high:
            table = candidate
            headers = candidate_headers
            break
    if table is None:
        raise UniverseParseError(
            f"{index}: found no table with a Symbol/Ticker column and {low}-{high} rows on "
            f"{source_url}. Wikipedia's layout may have changed. Tables seen: "
            f"{_describe_tables(tables)}"
        )

    ticker_position = _find_column(headers, _TICKER_HEADERS)
    name_position = _find_column(headers, _NAME_HEADERS)
    sector_position = _find_column(headers, _SECTOR_HEADERS)
    as_of = dt.date.today().isoformat()

    records = []
    seen = set()
    for row in range(len(table)):
        ticker = clean_ticker(table.iloc[row, ticker_position])
        if ticker == "":
            continue
        if not _TICKER_PATTERN.match(ticker):
            raise UniverseParseError(f"{index}: unexpected ticker {ticker!r} in row {row}")
        if ticker in seen:
            continue
        seen.add(ticker)
        records.append(
            {
                "index": index,
                "ticker": ticker,
                "name": _cell_text(table, row, name_position),
                "sector": _cell_text(table, row, sector_position),
                "source": source_url,
                "as_of": as_of,
            }
        )
    return pd.DataFrame(records, columns=UNIVERSE_COLUMNS)


class IndexUniverseSource:
    """Current S&P 500, Dow 30 and Nasdaq-100 constituents, scraped from Wikipedia.

    Args:
        session: An injectable `requests.Session` (for tests). Defaults to a fresh
            pooled session.
        user_agent: Identifies this client to Wikimedia. Use a real contact address.
    """

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        user_agent: str = _DEFAULT_USER_AGENT,
    ) -> None:
        self._session = session or requests.Session()
        self._user_agent = user_agent

    def _fetch_html(self, url: str) -> str:
        response = self._session.get(url, headers={"User-Agent": self._user_agent}, timeout=30)
        response.raise_for_status()
        return response.text

    def constituents(self, index: str) -> pd.DataFrame:
        """Fetch and parse one index. `index` is "sp500", "dow" or "nasdaq100"."""
        if index not in _WIKI_URLS:
            raise ValueError(f"Unknown Wikipedia index {index!r}; expected one of {WIKI_INDEXES}")
        url = _WIKI_URLS[index]
        return parse_constituents(index, self._fetch_html(url), url)

    def sp500(self) -> pd.DataFrame:
        """Current S&P 500 constituents (about 503 tickers; some companies list two classes)."""
        return self.constituents("sp500")

    def dow(self) -> pd.DataFrame:
        """Current Dow Jones Industrial Average constituents (30 tickers)."""
        return self.constituents("dow")

    def nasdaq100(self) -> pd.DataFrame:
        """Current Nasdaq-100 constituents (about 101 tickers; some companies list two classes)."""
        return self.constituents("nasdaq100")


def nasdaq_listed(reference: pd.DataFrame, ticker_type: str = "CS") -> pd.DataFrame:
    """Every active stock whose primary exchange is Nasdaq, from a Massive reference frame.

    This is the whole exchange, not the Nasdaq-100. It is derived from a frame returned by
    `MassiveSource.list_tickers(market="stocks")` so that one paginated pull can feed both this
    and `attach_massive_reference`.

    Args:
        reference: A frame from `MassiveSource.list_tickers`.
        ticker_type: Massive ticker type to keep. "CS" is common stock, which leaves out ETFs,
            ADRs ("ADRC"), warrants and preferred shares. Pass None to keep every type.

    Returns:
        A frame with the columns in `UNIVERSE_COLUMNS`.
    """
    as_of = dt.date.today().isoformat()
    source = "massive:/v3/reference/tickers (primary_exchange=XNAS"
    source += f", type={ticker_type})" if ticker_type else ")"

    records = []
    for row in range(len(reference)):
        if reference.iloc[row]["primary_exchange"] != "XNAS":
            continue
        if ticker_type and reference.iloc[row]["type"] != ticker_type:
            continue
        records.append(
            {
                "index": "nasdaq_listed",
                "ticker": str(reference.iloc[row]["ticker"]).upper(),
                "name": reference.iloc[row]["name"],
                "sector": None,
                "source": source,
                "as_of": as_of,
            }
        )
    return pd.DataFrame(records, columns=UNIVERSE_COLUMNS)


def attach_massive_reference(universe: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Add Massive reference data to a universe frame.

    Adds `in_massive` (True if Massive lists the ticker as active) and the columns in
    `REFERENCE_FIELDS` (missing values where Massive doesn't list the ticker). A False
    `in_massive` on a real constituent usually means a symbol-format mismatch or a very recent
    listing change.

    Args:
        universe: A frame with a `ticker` column.
        reference: A frame from `MassiveSource.list_tickers(market="stocks")`.

    Returns:
        A copy of `universe` with the extra columns.
    """
    position_by_ticker = {}
    for row in range(len(reference)):
        position_by_ticker[str(reference.iloc[row]["ticker"]).upper()] = row

    in_massive = []
    columns = {}
    for field in REFERENCE_FIELDS:
        columns[field] = []

    for row in range(len(universe)):
        position = position_by_ticker.get(universe.iloc[row]["ticker"])
        if position is None:
            in_massive.append(False)
            for field in REFERENCE_FIELDS:
                columns[field].append(None)
        else:
            in_massive.append(True)
            for field in REFERENCE_FIELDS:
                columns[field].append(reference.iloc[position][field])

    result = universe.copy()
    result["in_massive"] = in_massive
    for field in REFERENCE_FIELDS:
        result[field] = columns[field]
    return result


def collect_universes(
    indexes: Sequence[str] = ALL_INDEXES,
    massive: Optional[MassiveSource] = None,
    wiki: Optional[IndexUniverseSource] = None,
    save: bool = False,
    out_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Collect several universes in one call.

    If `massive` is given, its active-stock reference list is pulled once and used to add
    Massive reference columns to every universe and to build `nasdaq_listed`. Without it,
    only the Wikipedia-based indexes can be collected and they carry no Massive columns.

    Args:
        indexes: Names from `ALL_INDEXES`.
        massive: A `MassiveSource`. Required if "nasdaq_listed" is requested.
        wiki: An `IndexUniverseSource` (injectable for tests). Defaults to a fresh one.
        save: If True, write each universe to parquet with `save_universe`.
        out_dir: Directory for saved files; see `save_universe`.

    Returns:
        A dict mapping index name to its universe frame.
    """
    for name in indexes:
        if name not in ALL_INDEXES:
            raise ValueError(f"Unknown index {name!r}; expected one of {ALL_INDEXES}")
    if "nasdaq_listed" in indexes and massive is None:
        raise ValueError(
            "'nasdaq_listed' comes from Massive reference data; pass massive=MassiveSource()"
        )

    reference = None
    if massive is not None:
        reference = massive.list_tickers(market="stocks", active=True)
    if wiki is None:
        wiki = IndexUniverseSource()

    universes = {}
    for name in indexes:
        if name == "nasdaq_listed":
            frame = nasdaq_listed(reference)
        else:
            frame = wiki.constituents(name)
        if reference is not None:
            frame = attach_massive_reference(frame, reference)
        universes[name] = frame
        if save:
            save_universe(frame, name, out_dir)
    return universes


def all_tickers(universes: Dict[str, pd.DataFrame]) -> List[str]:
    """Sorted, de-duplicated union of tickers across universes, e.g. as input to `precompute`."""
    tickers = set()
    for frame in universes.values():
        for ticker in frame["ticker"].tolist():
            tickers.add(ticker)
    return sorted(tickers)


def _default_universe_dir() -> Path:
    return Path(config.get_data_dir()) / "raw" / "universe"


def save_universe(frame: pd.DataFrame, name: str, out_dir: Optional[str] = None) -> str:
    """Write a universe to `<out_dir>/<name>_<YYYY-MM-DD>.parquet` and return the path.

    Args:
        frame: A universe frame.
        name: Index name, e.g. "sp500".
        out_dir: Destination directory. Defaults to `<config.get_data_dir()>/raw/universe`.
    """
    directory = Path(out_dir) if out_dir else _default_universe_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}_{dt.date.today().isoformat()}.parquet"
    frame.to_parquet(path, index=False)
    return str(path)


def load_universe(name: str, out_dir: Optional[str] = None) -> pd.DataFrame:
    """Read the most recent saved universe called `name` (see `save_universe`).

    Raises:
        FileNotFoundError: If no saved file exists for `name`.
    """
    directory = Path(out_dir) if out_dir else _default_universe_dir()
    matches = sorted(directory.glob(f"{name}_*.parquet"))  # ISO dates sort chronologically
    if not matches:
        raise FileNotFoundError(f"No saved universe named {name!r} in {directory}")
    return pd.read_parquet(matches[-1])


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: collect universes, save them, print a short summary."""
    parser = argparse.ArgumentParser(
        description="Collect S&P 500, Dow, Nasdaq-100 and Nasdaq-listed universes."
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        choices=ALL_INDEXES,
        default=None,
        help="Universes to collect. Default: all four (or the three Wikipedia ones with "
        "--skip-massive).",
    )
    parser.add_argument("--out-dir", default=None, help="Where to write parquet files.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--skip-massive",
        action="store_true",
        help="Don't call Massive: constituents only, no reference columns, no nasdaq_listed.",
    )
    args = parser.parse_args(argv)

    config.load_yaml(args.config)
    indexes = args.indexes
    if indexes is None:
        indexes = WIKI_INDEXES if args.skip_massive else ALL_INDEXES

    try:
        massive = None
        if not args.skip_massive:
            massive = MassiveSource()
            # Cheap single call so a bad key or base URL fails now, not after a long pull.
            massive.ticker_details("AAPL")
        universes = collect_universes(indexes, massive=massive, save=True, out_dir=args.out_dir)
    except (ValueError, DataSourceError, requests.RequestException) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for name, frame in universes.items():
        print(f"{name}: {len(frame)} tickers")
        if "in_massive" in frame.columns:
            missing = []
            for row in range(len(frame)):
                if not frame.iloc[row]["in_massive"]:
                    missing.append(frame.iloc[row]["ticker"])
            if missing:
                print(f"  not found in Massive: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
