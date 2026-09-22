"""Optional wrapper around sec_parser
(https://github.com/GTSF-Quantitative-Sector/sec_parser) for fundamentals
and point-in-time S&P 500 membership. Install with `pip install ".[sec]"`
to enable this module -- `sec_parser` is imported lazily inside each
function, never at module import time, so importing this module never fails
just because the optional dependency isn't installed.
"""
from __future__ import annotations

from typing import Optional

_INSTALL_HINT = (
    "sec_parser is not installed. Install it with `pip install '.[sec]'` "
    "to use data_universe.sources.sec_adapter."
)


def _import_sec_parser():
    """Import and return sec_parser's `lookups` module, or raise a helpful ImportError."""
    try:
        from sec import lookups
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return lookups


def sp500(query_date: Optional[str] = None) -> list:
    """Point-in-time S&P 500 constituent tickers, delegating to sec_parser.

    Args:
        query_date: Date "YYYY-MM-DD" to look up membership as of, or None
            for the current membership. **Never pass None for historical
            research** -- it returns today's membership, which introduces
            survivorship bias into any backtest over past dates.

    Returns:
        List of ticker symbols.

    Raises:
        ImportError: If `sec_parser` isn't installed.
    """
    lookups = _import_sec_parser()
    return lookups.get_sp500_tickers(query_date=query_date)


def fundamentals(ticker: str):
    """Return a sec_parser `stock.Stock` for fundamentals access
    (`get_ebit`, `get_revenue`, `get_wacc`, etc. -- see the sec_parser README
    for its full API).

    Args:
        ticker: Stock ticker symbol.

    Raises:
        ImportError: If `sec_parser` isn't installed.
    """
    try:
        from sec import stock
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return stock.Stock(ticker)
