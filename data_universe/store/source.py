"""`StoreBackedSource`: a `DataSource` that serves daily datasets from the
durable `RawStore`, fetching each upstream window once (or once per process
for the still-open part of the window) and never again.

Requested windows split at `_last_complete_date()` L (a session counts as
complete from 06:00 ET the next calendar day): the part <= L is read from
the store -- fetching the hull of the missing sub-intervals in ONE upstream
call when coverage has gaps -- and the part after L is passed through to the
upstream's normal (cached) methods, never persisted. Stored datasets are the
four daily ones in `STORED_DATASETS`; `dividends` is stored only on the raw
cash basis (`adjust.DIVIDEND_CASH_BASIS == "raw"`), and `hourly_bars` plus
every other attribute delegate to the upstream.

Two invariants keep basis-dependent series (`daily_bars_split_adjusted`)
correct across splits:

- When a store fetch wouldn't overlap an existing stored row, the fetch is
  extended to the nearest stored row so the fresh frame overlaps; an overlap
  close that differs by >1e-6 relative means a split rebased the vendor's
  history, and the whole coverage is refetched once with `rebased=True`.
- The same check runs on the live part: it starts at the last stored row
  <= L (not L + 1 day), so a split effective inside the live window is
  caught the same way before the stored and live parts are concatenated.

Upstream fetches for the store bypass the `@cached` decorator (the store is
the durable layer -- no double storage, no stale disk-cache hits). The
manifest's `family` stamps the upstream source family; a read against a
different family raises `DataSourceError` rather than serving foreign data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_universe.sources import adjust, base
from data_universe.sources.base import NY_TZ, DataSource, DataSourceError, EntitlementError
from data_universe.store.raw import RawStore, _fetch_record, _row_dates

# dataset == upstream method name; value = the manifest's row-date column
# (None = derive row dates from the tz-aware index).
STORED_DATASETS = {
    "daily_bars_split_adjusted": None,
    "daily_bars_raw": None,
    "index_daily_bars": None,
    "dividends": "ex_dividend_date",
}

# Datasets whose values change basis when a later split lands: split-adjusted
# closes are restated, raw closes and dividends are not.
_BASIS_DATASETS = {"daily_bars_split_adjusted"}
_REBASE_TOL = 1e-6


def _last_complete_date() -> pd.Timestamp:
    """The last session complete from an upstream point of view: sessions are
    treated as complete from 06:00 ET the next calendar day, so this is
    (now ET - 6h)'s date minus one day."""
    et_now = pd.Timestamp.now(tz=NY_TZ) - pd.Timedelta(hours=6)
    return et_now.tz_localize(None).normalize() - pd.Timedelta(days=1)


def _missing_intervals(coverage, s, e):
    """The gaps inside [s, e] not covered by merged `coverage` intervals."""
    gaps = []
    cursor = s
    for c_start, c_end in coverage:
        if c_end < cursor:
            continue
        if c_start > e:
            break
        if c_start > cursor:
            gaps.append((cursor, min(c_start - pd.Timedelta(days=1), e)))
        cursor = max(cursor, c_end + pd.Timedelta(days=1))
    if cursor <= e:
        gaps.append((cursor, e))
    return gaps


class StoreBackedSource(adjust.TotalReturnMixin, DataSource):
    """Daily datasets served from a durable `RawStore` under `root`.

    Args:
        upstream: The underlying `DataSource` (PolygonSource, MassiveSource,
            or a fake sharing their interface).
        root: Store root directory; None uses `<config.get_data_dir()>/raw`.
    """

    def __init__(self, upstream, root=None) -> None:
        self.upstream = upstream
        self.store = RawStore(root)
        self.family = getattr(upstream, "store_family", None) or type(upstream).__name__

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        upstream = self.__dict__.get("upstream")
        if upstream is None:
            raise AttributeError(name)
        return getattr(upstream, name)

    def daily_bars_split_adjusted(self, ticker, start, end):
        return self._stored("daily_bars_split_adjusted", ticker, start, end)

    def daily_bars_raw(self, ticker, start, end):
        return self._stored("daily_bars_raw", ticker, start, end)

    def index_daily_bars(self, ticker, start, end):
        return self._stored("index_daily_bars", ticker, start, end)

    def dividends(self, ticker, start, end):
        # Stored only on the raw cash basis: other bases make amounts
        # basis-dependent, which is unhandled until the probe answers.
        if adjust.DIVIDEND_CASH_BASIS != "raw":
            return self.upstream.dividends(ticker, start, end)
        return self._stored("dividends", ticker, start, end)

    def hourly_bars(self, ticker, start, end):
        return self.upstream.hourly_bars(ticker, start, end)

    def _upstream_call(self, dataset):
        """The dataset's fetch callable WITHOUT the `@cached` layer: the
        store is the durable layer, so double storage and stale disk-cache
        hits are avoided by unwrapping `__wrapped__` when present."""
        method = getattr(type(self.upstream), dataset)
        unwrapped = getattr(method, "__wrapped__", None)
        if unwrapped is not None:
            return lambda key, s, e: unwrapped(self.upstream, key, s, e)
        bound = getattr(self.upstream, dataset)
        return lambda key, s, e: bound(key, s, e)

    def _check_family(self, dataset, key):
        """Every access verifies the manifest's family matches ours -- a
        store written by a different source family must never be served."""
        manifest = self.store.manifest(dataset, key)
        foreign = {
            record.get("family")
            for record in manifest.get("fetches", [])
            if record.get("family") is not None and record.get("family") != self.family
        }
        if foreign:
            raise DataSourceError(
                f"Raw store at {self.store.root} holds {sorted(foreign)} data for "
                f"{dataset}/{key}, but this source is family {self.family!r} -- "
                "use a separate root per vendor family"
            )

    def _stored(self, dataset, ticker, start, end):
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        last_complete = _last_complete_date()
        stored_end = min(e, last_complete)
        date_column = STORED_DATASETS[dataset]
        self._check_family(dataset, ticker)

        stored_part = None
        if s <= stored_end:
            coverage = self.store.coverage(dataset, ticker)
            if _missing_intervals(coverage, s, stored_end):
                with self.store.lock(dataset, ticker):
                    coverage = self.store.coverage(dataset, ticker)
                    missing = _missing_intervals(coverage, s, stored_end)
                    if missing:
                        a = min(gap[0] for gap in missing)
                        b = max(gap[1] for gap in missing)
                        self._fetch_into_store(dataset, ticker, a, b)
            stored_part = self._slice(self.store.read(dataset, ticker), s, stored_end,
                                      date_column)

        live = None
        if e > last_complete:
            # The overlap row comes from the stored-part slice, so the
            # extension happens only when the window spans stored + live
            # parts -- the only case where two bases could mix.
            overlap_date = None
            if dataset in _BASIS_DATASETS and stored_part is not None:
                overlap_date = _row_dates(stored_part, date_column).max()
            live_floor = max(s, last_complete + pd.Timedelta(days=1))
            live_start = (
                min(live_floor, overlap_date)
                if overlap_date is not None
                else live_floor
            )
            live = getattr(self.upstream, dataset)(
                ticker, live_start.strftime("%Y-%m-%d"), end
            )
            if overlap_date is not None and len(live):
                live_dates = _row_dates(live, date_column)
                overlap_rows = live_dates == overlap_date
                if overlap_rows.any() and self._close_differs(
                    float(live["close"].iloc[overlap_rows.nonzero()[0][0]]),
                    self._stored_close(dataset, ticker, overlap_date, date_column),
                ):
                    with self.store.lock(dataset, ticker):
                        self._rebase(dataset, ticker, overlap_date, last_complete,
                                     date_column)
                    stored_part = self._slice(
                        self.store.read(dataset, ticker), s, stored_end, date_column
                    )
            if len(live):
                live_dates = _row_dates(live, date_column)
                live = live.iloc[live_dates >= live_floor]

        parts = [p for p in (stored_part, live) if p is not None and len(p)]
        if not parts:
            return (
                adjust.empty_dividends() if dataset == "dividends" else base.empty_bars()
            )
        result = pd.concat(parts) if len(parts) > 1 else parts[0]
        if date_column is not None:
            result = result.reset_index(drop=True)
        return result

    def _slice(self, frame, s, e, date_column):
        if frame is None or not len(frame):
            return None
        dates = _row_dates(frame, date_column)
        kept = frame.iloc[(dates >= s) & (dates <= e)]
        return kept if len(kept) else None

    @staticmethod
    def _close_differs(fresh_close, stored_close):
        if stored_close is None or not np.isfinite(stored_close):
            return False
        return abs(fresh_close / stored_close - 1.0) > _REBASE_TOL

    def _stored_close(self, dataset, ticker, date, date_column):
        stored = self.store.read(dataset, ticker)
        if stored is None or not len(stored) or "close" not in stored.columns:
            return None
        dates = _row_dates(stored, date_column)
        matches = dates == date
        if not matches.any():
            return None
        return float(stored["close"].iloc[matches.nonzero()[0][0]])

    def _fetch_into_store(self, dataset, key, a, b):
        """Fetch [a, b] from the upstream ONCE (uncached) and upsert it,
        handling overlap-extension and full rebases for basis datasets."""
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        date_column = STORED_DATASETS[dataset]
        fetch = self._upstream_call(dataset)
        fetch_a, fetch_b = a, b

        stored = self.store.read(dataset, key)
        if dataset in _BASIS_DATASETS and stored is not None and len(stored):
            stored_dates = _row_dates(stored, date_column)
            inside = stored_dates[(stored_dates >= a) & (stored_dates <= b)]
            if not len(inside):
                before = stored_dates[stored_dates < a]
                after = stored_dates[stored_dates > b]
                if len(before):
                    fetch_a = min(fetch_a, before.max())
                elif len(after):
                    fetch_b = max(fetch_b, after.min())

        try:
            fresh = fetch(key, fetch_a.strftime("%Y-%m-%d"), fetch_b.strftime("%Y-%m-%d"))
        except EntitlementError:
            self.store.record(
                dataset, key,
                _fetch_record(fetch_a, fetch_b, "not_entitled", 0,
                              type(self.upstream).__name__, self.family, False),
            )
            raise

        if dataset in _BASIS_DATASETS and stored is not None and len(stored):
            stored_dates = _row_dates(stored, date_column)
            if len(fresh):
                fresh_dates = _row_dates(fresh, date_column)
                common = sorted(set(fresh_dates) & set(stored_dates))
                for date in common:
                    fresh_close = float(
                        fresh["close"].iloc[(fresh_dates == date).nonzero()[0][0]]
                    )
                    stored_close = float(
                        stored["close"].iloc[(stored_dates == date).nonzero()[0][0]]
                    )
                    if self._close_differs(fresh_close, stored_close):
                        coverage = self.store.coverage(dataset, key)
                        rebase_a = min(coverage[0][0], fetch_a)
                        rebase_b = max(coverage[-1][1], fetch_b)
                        rebased = fetch(
                            key,
                            rebase_a.strftime("%Y-%m-%d"),
                            rebase_b.strftime("%Y-%m-%d"),
                        )
                        rebased_dates = _row_dates(rebased, date_column)
                        rows = int(
                            ((rebased_dates >= rebase_a) & (rebased_dates <= rebase_b)).sum()
                        )
                        self.store.replace_all(
                            dataset, key, rebased,
                            _fetch_record(rebase_a, rebase_b, "ok",
                                          rows,
                                          type(self.upstream).__name__,
                                          self.family, True),
                        )
                        return

        fresh_dates = _row_dates(fresh, date_column) if len(fresh) else None
        in_range = (
            int(((fresh_dates >= fetch_a) & (fresh_dates <= fetch_b)).sum())
            if fresh_dates is not None
            else 0
        )
        self.store.upsert_range(
            dataset, key, fetch_a, fetch_b, fresh, date_column,
            _fetch_record(fetch_a, fetch_b, "ok" if in_range else "empty",
                          in_range, type(self.upstream).__name__, self.family, False),
        )

    def _rebase(self, dataset, key, overlap_date, last_complete, date_column):
        """Live-part rebase: a split effective inside the live window made
        the overlap close differ, so refetch the whole coverage (capped at
        the last complete date) and replace it."""
        coverage = self.store.coverage(dataset, key)
        if not coverage:
            return
        fetch = self._upstream_call(dataset)
        rebase_a = coverage[0][0]
        rebase_b = min(max(coverage[-1][1], overlap_date), last_complete)
        rebased = fetch(key, rebase_a.strftime("%Y-%m-%d"), rebase_b.strftime("%Y-%m-%d"))
        self.store.replace_all(
            dataset, key, rebased,
            _fetch_record(rebase_a, rebase_b, "ok", len(rebased),
                          type(self.upstream).__name__, self.family, True),
        )
