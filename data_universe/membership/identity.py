"""Resolve one-id-per-interval security_ids into securities.

Upstream Massive ticker details `?date=` gives the as-of record -- CIK,
composite FIGI, share-class FIGI and registrant name -- so FB-era membership
intervals link to META's, a leave-and-rejoin merges into one security, and
reused tickers (two different Dow companies) stay apart. Every lookup lands
in committed evidence tables (`identity_evidence.csv`, `cik_evidence.csv`,
`cik_ticker_evidence.csv`), so an `--offline` rebuild reproduces the
committed files byte for byte with zero network.

    python -m data_universe.membership.identity              # build
    python -m data_universe.membership.identity --offline    # rebuild, no fetch
    python -m data_universe.membership.identity --dry-run    # report, write nothing
    python -m data_universe.membership.identity --refetch    # refetch every key once

A Massive record's CIK is only TRUSTED when its registrant name matches the
SEC name (or a former name) for the CIK it carries -- old records sometimes
carry another company's CIK (XRAY -> Danaher). Intervals where upstream
folded two companies into one ticker line are cut by `split` overrides into
sub-intervals, each resolved on its own. `symbol` discovery finds the
historical symbol a CIK traded under via `tickers_for_cik` when upstream uses
a retroactive symbol (SPGI <- MHFI, BKNG <- PCLN).

`identity_review.csv` is the human triage queue (see REVIEW_KINDS); answers
go in `identity_overrides.csv` (link/unlink/cik/symbol/split/ack), which only
humans edit -- every row is validated and applied on the next run. See
`tables/sp500/README.md` and `OPEN_QUESTIONS.md` items 24, 34-40.
"""
from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

from data_universe import config
from data_universe.membership import intervals as _intervals_mod
from data_universe.membership import store
from data_universe.membership.errors import MembershipError, MembershipValidationError
from data_universe.membership.names import names_match
from data_universe.membership.schema import (
    CIK_EVIDENCE_COLUMNS,
    CIK_HISTORY_COLUMNS,
    CIK_TICKER_EVIDENCE_COLUMNS,
    IDENTITY_EVIDENCE_COLUMNS,
    IDENTITY_REVIEW_COLUMNS,
    SECURITY_MAP_COLUMNS,
    SECURITY_MASTER_COLUMNS,
    TICKER_HISTORY_COLUMNS,
)

INDEX_ID = "sp500"
TOOL_VERSION = "membership-identity-1"
COVERAGE_START = "2003-09-11"  # CIKs appear in Massive ?date= records from here
PROBE_OFFSETS_DAYS = (0, 7, 30, 90)
MAX_CIK_SWITCHES = 3
MAX_SYMBOL_SWITCHES = 3
CIK_LATE_SLACK_DAYS = 30
CHECKPOINT_EVERY = 200
_REBASE_FILES = (
    "security_map.csv", "security_master.csv", "ticker_history.csv",
    "cik_history.csv", "identity_review.csv", "identity_evidence.csv",
    "cik_evidence.csv", "cik_ticker_evidence.csv", "manifest.json",
)

REVIEW_KINDS = (
    "symbol_not_found",
    "cik_change_within_interval",
    "too_many_cik_changes",
    "ambiguous_cik_link",
    "same_cik_figi_mismatch",
    "rejoin_unconfirmed",
    "link_would_overlap",
    "cik_starts_late",
    "cik_name_mismatch",
    "symbol_discovery_ambiguous",
    "symbol_reassigned",
    "cik_no_earnings_filings",
)

_OPEN = pd.Timestamp.max.tz_localize(None)


# --- small value helpers ------------------------------------------------------


def _ts(value) -> Optional[pd.Timestamp]:
    """Parse a possibly-empty/NaN date value; None/""/NaT -> None."""
    if value is None or (isinstance(value, str) and not value):
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return None if pd.isna(ts) else ts


def _iso_day(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def _now_utc(now=None) -> pd.Timestamp:
    return now if now is not None else pd.Timestamp.now(tz="UTC")


def _norm_row(row: dict) -> dict:
    """CSV empty cells load as NaN -- normalize so "" semantics (no
    cik/figi/name) survive the offline round-trip."""
    return {
        k: ("" if isinstance(v, float) and pd.isna(v) else v)
        for k, v in dict(row).items()
    }


# --- union-find ---------------------------------------------------------------


class _UF:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class _Interval:
    """One (sub-)interval: a ticker_intervals row, or the piece of a row cut
    by a `split` override. `key` is (ticker, sub_start)."""

    key: Tuple[str, str]          # (ticker, sub_start "YYYY-MM-DD")
    ticker: str
    start: pd.Timestamp
    end_excl: Optional[pd.Timestamp]
    lo: pd.Timestamp              # probe window, clipped to COVERAGE_START
    hi: Optional[pd.Timestamp]    # None when the window is empty
    split: bool = False           # part of a `split`-cut interval
    first: Optional[dict] = None
    last: Optional[dict] = None
    first_tried: tuple = ("",)
    last_tried: tuple = ("",)
    sym_segments: tuple = ()      # discovered (symbol, vf, vt) spans
    sym_takeover: Optional[pd.Timestamp] = None  # discovered s (S's takeover)

    @property
    def end(self) -> pd.Timestamp:
        return self.end_excl if self.end_excl is not None else _OPEN

    def overlaps(self, other) -> bool:
        return self.start < other.end and other.start < self.end


# --- overrides ----------------------------------------------------------------


def _parse_overrides(frame: pd.DataFrame,
                     ticker_intervals: pd.DataFrame) -> dict:
    """Validate identity_overrides rows into {action: [row...]} plus
    `split_dates`: {(ticker, interval_start): [split Timestamps]}.

    Splits are parsed FIRST -- the valid keys for every other action are the
    interval starts plus the split dates on that ticker's intervals (a
    cik/symbol/link row may name a sub-interval like (AGN, 2015-03-18)).
    Raises MembershipValidationError naming the offending row."""
    out = {"link": [], "unlink": [], "cik": [], "symbol": [], "ack": [],
           "split": []}
    interval_ends = {
        (r["ticker"], r["start_date"]): _ts(r["end_date_excl"])
        for _, r in ticker_intervals.iterrows()
    }

    # pass 1: splits -- (ticker, start_date) is an existing ticker interval;
    # valid_from = d with start < d < end_excl (or open)
    split_dates = {}
    for _, row in frame[frame["action"] == "split"].iterrows():
        key = (row["ticker"], row["start_date"])
        if key not in interval_ends:
            raise MembershipValidationError(
                f"identity_overrides: split on unknown interval {dict(row)}")
        d = _ts(row["valid_from"])
        start = pd.Timestamp(row["start_date"])
        end = interval_ends[key] or _OPEN
        if d is None or not (start < d < end):
            raise MembershipValidationError(
                f"identity_overrides: split date out of bounds {dict(row)}")
        split_dates.setdefault(key, []).append(d)
        out["split"].append(row)
    for key in split_dates:
        split_dates[key] = sorted(set(split_dates[key]))

    valid_keys = set(interval_ends)
    for (ticker, _s), dates in split_dates.items():
        for d in dates:
            valid_keys.add((ticker, _iso_day(d)))

    # pass 2: every other action
    for _, row in frame.iterrows():
        where = dict(row)
        action = row["action"]
        if action == "split":
            continue
        if action not in out:
            raise MembershipValidationError(
                f"identity_overrides: bad action {where}")
        if (row["ticker"], row["start_date"]) not in valid_keys:
            raise MembershipValidationError(
                f"identity_overrides: unknown interval {where}")
        if action in ("link", "unlink"):
            other = (row["other_ticker"], row["other_start_date"])
            if other not in valid_keys or other == (row["ticker"], row["start_date"]):
                raise MembershipValidationError(
                    f"identity_overrides: {action} needs a different existing "
                    f"interval {where}")
        elif action == "cik":
            value = str(row["value"])
            if not value.isdigit():
                raise MembershipValidationError(
                    f"identity_overrides: cik value {where}")
            _check_valid_range(row)
        elif action == "symbol":
            if not str(row["value"]):
                raise MembershipValidationError(
                    f"identity_overrides: symbol {where}")
            _check_valid_range(row)
        elif action == "ack":
            if row["value"] not in REVIEW_KINDS:
                raise MembershipValidationError(
                    f"identity_overrides: ack kind {where}")
        out[action].append(row)
    _check_symbol_overlap(out["symbol"])
    out["split_dates"] = split_dates
    return out


def _check_valid_range(row):
    valid_from = _ts(row["valid_from"])
    if valid_from is None:
        raise MembershipValidationError(
            f"identity_overrides: missing valid_from {dict(row)}")
    valid_to = _ts(row["valid_to_excl"])
    if valid_to is not None and valid_to <= valid_from:
        raise MembershipValidationError(
            f"identity_overrides: valid_to_excl <= valid_from {dict(row)}"
        )


def _check_symbol_overlap(rows):
    by_interval = {}
    for row in rows:
        key = (row["ticker"], row["start_date"])
        span = (_ts(row["valid_from"]), _ts(row["valid_to_excl"]) or _OPEN)
        for prev in by_interval.get(key, []):
            if span[0] < prev[1] and prev[0] < span[1]:
                raise MembershipValidationError(
                    f"identity_overrides: overlapping symbol rows for {key}"
                )
        by_interval.setdefault(key, []).append(span)


def _symbol_for(interval_key, symbol_rows, ticker):
    """Curried `_symbol_at`: (date) -> the probe symbol for that date."""
    def sym(date):
        return _symbol_at(interval_key, symbol_rows, ticker, date)

    return sym


def _symbol_at(interval_key, symbol_rows, ticker, date) -> str:
    """The override symbol covering `date` on this interval, else `ticker`."""
    for row in symbol_rows:
        if (row["ticker"], row["start_date"]) != interval_key:
            continue
        vf, vt = _ts(row["valid_from"]), _ts(row["valid_to_excl"]) or _OPEN
        if vf <= date < vt:
            return str(row["value"])
    return ticker


def _probe_symbol_for(interval, symbol_rows):
    """The interval's probe-symbol function: manual override > discovered
    segments > upstream ticker."""
    manual = _symbol_for(interval.key, symbol_rows, interval.ticker)

    def sym(date):
        via_manual = manual(date)
        if via_manual != interval.ticker:
            return via_manual
        for ticker, vf, vt in interval.sym_segments:
            if vf <= date < vt:
                return ticker
        return interval.ticker

    return sym


# --- evidence -----------------------------------------------------------------


class _Evidence:
    """Evidence stores: (ticker, date) -> Massive record, cik -> SEC summary,
    (cik, date) -> tickers-for-cik list -- backed by the committed CSVs with
    online fetching layered on top."""

    def __init__(self, massive_frame, sec_frame, cik_ticker_frame,
                 reference_source, edgar_source,
                 offline, refetch, progress, checkpoint_fn=None):
        self.stored_massive = {
            (r["ticker"], r["date"]): _norm_row(r)
            for _, r in massive_frame.iterrows()
        }
        self.stored_sec = {
            str(r["cik"]): _norm_row(r) for _, r in sec_frame.iterrows()
        }
        self.stored_cik_ticker = {
            (str(r["cik"]), r["date"]): _norm_row(r)
            for _, r in cik_ticker_frame.iterrows()
        }
        self.new_massive = {}
        self.new_sec = {}
        self.new_cik_ticker = {}
        self.consulted_massive = set()
        self.consulted_sec = set()
        self.consulted_cik_ticker = set()
        self.reference = reference_source
        self.edgar = edgar_source
        self.offline = offline
        self.refetch = refetch
        self.progress = progress
        self.checkpoint_fn = checkpoint_fn
        self.n_massive_fetched = 0
        self.n_massive_reused = 0
        self.n_sec_fetched = 0
        self.n_cik_ticker_fetched = 0
        self._checkpoint_tick = 0

    # ---- consulted/all row sets (for pruned / union writes)

    def consulted_massive_rows(self):
        return [r for k in sorted(self.consulted_massive)
                for r in (self.new_massive.get(k)
                          or self.stored_massive.get(k),) if r is not None]

    def consulted_sec_rows(self):
        return [r for k in sorted(self.consulted_sec, key=lambda x: int(x))
                for r in (self.new_sec.get(k) or self.stored_sec.get(k),)
                if r is not None]

    def consulted_cik_ticker_rows(self):
        return [r for k in sorted(self.consulted_cik_ticker,
                                 key=lambda x: (int(x[0]), x[1]))
                for r in (self.new_cik_ticker.get(k)
                          or self.stored_cik_ticker.get(k),) if r is not None]

    def all_massive_rows(self):
        merged = dict(self.stored_massive)
        merged.update(self.new_massive)
        return list(merged.values())

    def all_sec_rows(self):
        merged = dict(self.stored_sec)
        merged.update(self.new_sec)
        return list(merged.values())

    def all_cik_ticker_rows(self):
        merged = dict(self.stored_cik_ticker)
        merged.update(self.new_cik_ticker)
        return list(merged.values())

    # ---- massive (ticker, date)

    def massive(self, ticker, date) -> dict:
        """Evidence record for (ticker, date) -- stored row, else a fresh
        fetch (online only)."""
        key = (ticker, _iso_day(date))
        self.consulted_massive.add(key)
        if not self.refetch:
            stored = self.stored_massive.get(key)
            if stored is not None:
                self.n_massive_reused += 1
                return dict(stored)
        if self.offline and key not in self.new_massive:
            raise MembershipError(
                f"offline rebuild is missing evidence for {ticker} @ {key[1]} -- "
                "rerun online or restore the evidence row"
            )
        row = self.new_massive.get(key)
        if row is None:
            row = self._fetch_massive(ticker, key[1])
            self.new_massive[key] = row
        return dict(row)

    def _fetch_massive(self, ticker, date_str) -> dict:
        if self.reference is None:
            raise MembershipError(
                "identity run needs a reference_source for Massive lookups "
                f"({ticker} @ {date_str} missing from evidence)"
            )
        method = getattr(type(self.reference), "ticker_details", None)
        unwrapped = getattr(method, "__wrapped__", None)
        try:
            if unwrapped is not None:
                results = unwrapped(self.reference, ticker, date_str)
            else:
                results = self.reference.ticker_details(ticker, date=date_str)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                results = None
            else:
                raise
        self.n_massive_fetched += 1
        self._checkpoint_tick += 1
        fetched_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
        if not results:
            row = {"status": "not_found", "cik": "", "name": "",
                   "composite_figi": "", "share_class_figi": ""}
        else:
            cik = results.get("cik")
            row = {
                "status": "ok",
                "cik": str(int(cik)) if cik else "",
                "name": results.get("name") or "",
                "composite_figi": results.get("composite_figi") or "",
                "share_class_figi": results.get("share_class_figi") or "",
            }
        record = {
            "ticker": ticker, "date": date_str, "fetched_at": fetched_at, **row,
        }
        if self.progress and self.n_massive_fetched % 100 == 0:
            self.progress(f"massive evidence: {self.n_massive_fetched} fetched")
        if (self.checkpoint_fn
                and self._checkpoint_tick % CHECKPOINT_EVERY == 0):
            self.checkpoint_fn()
        return record

    # ---- sec (cik)

    def sec(self, cik) -> dict:
        """SEC registrant summary for `cik` (unpadded digits)."""
        cik = str(cik)
        self.consulted_sec.add(cik)
        if not self.refetch:
            stored = self.stored_sec.get(cik)
            if stored is not None:
                return dict(stored)
        if self.offline and cik not in self.new_sec:
            raise MembershipError(
                f"offline rebuild is missing SEC evidence for CIK {cik}"
            )
        row = self.new_sec.get(cik)
        if row is None:
            row = self._fetch_sec(cik)
            self.new_sec[cik] = row
        return dict(row)

    def _fetch_sec(self, cik) -> dict:
        if self.edgar is None:
            raise MembershipError(
                f"identity run needs an edgar_source for CIK {cik}"
            )
        summary = self.edgar.registrant_summary(cik)
        self.n_sec_fetched += 1
        fetched_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S")
        if summary is None:
            row = {"status": "not_found", "name": "", "former_names": "",
                   "first_filed": "", "has_item_202": ""}
        else:
            row = {
                "status": "ok",
                "name": summary.get("name") or "",
                "former_names": " | ".join(summary.get("former_names") or []),
                "first_filed": summary.get("first_filed") or "",
                "has_item_202": "true" if summary.get("has_item_202") else "false",
            }
        if self.progress and self.n_sec_fetched % 100 == 0:
            self.progress(f"sec evidence: {self.n_sec_fetched} fetched")
        return {
            "cik": cik, "fetched_at": fetched_at, **row,
        }

    # ---- tickers-for-cik (cik, date)

    def cik_tickers(self, cik, date) -> list:
        """Ticker symbols CIK `cik` carried on `date` (Massive
        tickers_for_cik), with the same evidence semantics."""
        key = (str(cik), _iso_day(date))
        self.consulted_cik_ticker.add(key)
        if not self.refetch:
            stored = self.stored_cik_ticker.get(key)
            if stored is not None:
                return stored["tickers"].split() if stored["tickers"] else []
        if self.offline and key not in self.new_cik_ticker:
            raise MembershipError(
                f"offline rebuild is missing tickers-for-CIK evidence for "
                f"{key[0]} @ {key[1]}"
            )
        row = self.new_cik_ticker.get(key)
        if row is None:
            row = self._fetch_cik_ticker(*key)
            self.new_cik_ticker[key] = row
        return row["tickers"].split() if row["tickers"] else []

    def _fetch_cik_ticker(self, cik, date_str) -> dict:
        if self.reference is None:
            raise MembershipError(
                "identity run needs a reference_source for tickers_for_cik "
                f"({cik} @ {date_str} missing from evidence)"
            )
        method = getattr(type(self.reference), "tickers_for_cik", None)
        unwrapped = getattr(method, "__wrapped__", None)
        if unwrapped is not None:
            tickers = unwrapped(self.reference, cik, date_str)
        else:
            tickers = self.reference.tickers_for_cik(cik, date_str)
        self.n_cik_ticker_fetched += 1
        return {
            "cik": cik, "date": date_str,
            "tickers": " ".join(tickers or []),
            "fetched_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%S"),
        }


# --- trusted cik ----------------------------------------------------------------


def _trusted_cik(record, interval_key, evidence, review_rows, mismatches):
    """A record's cik is trusted iff status ok, cik non-empty, the SEC
    summary for it is ok, and the record's registrant name matches the SEC
    name or a former name. Untrusted -> None (treated as absent
    everywhere); the mismatch is reviewed once per (interval key, cik)."""
    if record["status"] != "ok" or not record.get("cik"):
        return None
    cik = record["cik"]
    summary = evidence.sec(cik)
    sec_names = ([summary["name"]] if summary.get("name") else []) + [
        n for n in (summary.get("former_names") or "").split(" | ") if n
    ]
    if summary["status"] == "ok" and names_match(record.get("name"), sec_names):
        return cik
    if (interval_key, cik) not in mismatches:
        mismatches.add((interval_key, cik))
        former = f" (formerly {summary['former_names']})" \
            if summary.get("former_names") else ""
        review_rows.append({
            "kind": "cik_name_mismatch",
            "ticker": interval_key[0], "start_date": interval_key[1],
            "other_ticker": "", "other_start_date": "",
            "first_date": "", "last_date": "",
            "detail": (
                f"Massive names CIK {cik} '{record.get('name', '')}' but SEC "
                f"has '{summary.get('name', '')}'{former} -- CIK ignored"
            ),
        })
    return None


# --- probing ------------------------------------------------------------------


def _probe_fact(evidence, symbol_fn, start_d, direction, offsets, boundary,
                interval_key, review_rows, mismatches):
    """The first ok record walking `direction` from `start_d` through
    `offsets`, preferring records whose cik is TRUSTED. Returns
    (fact_or_None, tried_dates)."""
    ok_facts = []
    tried = []
    for off in offsets:
        d = start_d + pd.Timedelta(days=off) * direction
        if direction > 0 and d > boundary:
            break
        if direction < 0 and d < boundary:
            break
        record = evidence.massive(symbol_fn(d), _iso_day(d))
        tried.append(_iso_day(d))
        if record["status"] == "ok":
            trusted = _trusted_cik(
                record, interval_key, evidence, review_rows, mismatches)
            fact = {
                "date": _iso_day(d),
                "symbol": symbol_fn(d),
                "cik": trusted,
                "composite_figi": record["composite_figi"] or None,
                "share_class_figi": record["share_class_figi"] or None,
            }
            if fact["cik"]:
                return fact, tried
            ok_facts.append(fact)
    return (ok_facts[0] if ok_facts else None), tried


# --- symbol discovery ------------------------------------------------------------


def _cik_switch_date(evidence, symbol, lo, hi, c, interval_key,
                     review_rows, mismatches):
    """First date in (lo, hi] where massive(symbol, d) has trusted cik == c
    (unknown/untrusted counts as False). Plain bisection."""
    lo_d, hi_d = lo, hi
    while (hi_d - lo_d).days > 1:
        mid = lo_d + pd.Timedelta(days=(hi_d - lo_d).days // 2)
        record = evidence.massive(symbol, _iso_day(mid))
        trusted = _trusted_cik(
            record, interval_key, evidence, review_rows, mismatches)
        if trusted == c:
            hi_d = mid
        else:
            lo_d = mid
    return hi_d


def _find_symbol(evidence, c, d, exclude, interval, all_intervals,
                 review_rows, mismatches):
    """Steps 1-4 of Case-A discovery at date `d`: candidate tickers for CIK
    `c` (clean symbols, not excluded), keep those whose massive record that
    day has trusted cik == c, unique-or-figi match, guard against another
    interval's ticker. Returns (symbol_or_None, reason) where reason is
    None, "none", "ambiguous" or "guard"."""
    cands = [t for t in evidence.cik_tickers(c, _iso_day(d))
             if t not in exclude and re.fullmatch(r"[A-Z0-9.]+", t)]
    kept = []
    for t in cands:
        record = evidence.massive(t, _iso_day(d))
        if _trusted_cik(record, interval.key, evidence,
                        review_rows, mismatches) == c:
            kept.append((t, record))
    if not kept:
        return None, "none"
    if len(kept) == 1:
        chosen = kept[0][0]
    else:
        figi = interval.last["composite_figi"]
        matches = [t for t, rec in kept
                   if figi and rec["composite_figi"] == figi]
        if len(matches) == 1:
            chosen = matches[0]
        else:
            review_rows.append({
                "kind": "symbol_discovery_ambiguous",
                "ticker": interval.ticker, "start_date": interval.key[1],
                "other_ticker": "", "other_start_date": "",
                "first_date": _iso_day(d), "last_date": "",
                "detail": (
                    f"{len(kept)} candidates for CIK {c} at {_iso_day(d)}: "
                    f"{[t for t, _ in kept]} -- figi match "
                    f"{len(matches)} not 1"
                ),
            })
            return None, "ambiguous"
    for other in all_intervals:
        if (other.key != interval.key and other.ticker == chosen
                and other.start <= d < other.end):
            review_rows.append({
                "kind": "symbol_discovery_ambiguous",
                "ticker": interval.ticker, "start_date": interval.key[1],
                "other_ticker": other.ticker, "other_start_date": other.key[1],
                "first_date": _iso_day(d), "last_date": "",
                "detail": f"{chosen} is {other.ticker}'s ticker from "
                          f"{other.key[1]}",
            })
            return None, "guard"
    return chosen, None


def _discover(evidence, interval, all_intervals, review_rows, mismatches):
    """Case A: the upstream symbol's record has no trusted cik but the last
    fact has trusted c -- walk BACKWARDS from the upstream takeover date to
    the symbols c traded under before it. Vendor gaps in tickers_for_cik end
    the walk but keep the segments found so far.
    Returns ([(sym, vf, vt)...earliest-first], takeover s) or None."""
    lo = interval.lo
    c = interval.last["cik"]
    s = _cik_switch_date(evidence, interval.ticker, lo,
                         pd.Timestamp(interval.last["date"]), c,
                         interval.key, review_rows, mismatches)
    boundary = s
    used = {interval.ticker}
    segments = []
    while len(segments) < MAX_SYMBOL_SWITCHES:
        prev_d = boundary - pd.Timedelta(days=1)
        if prev_d < lo:
            break
        T, _why = _find_symbol(evidence, c, prev_d, used, interval,
                               all_intervals, review_rows, mismatches)
        if T is None:  # vendor gap / no earlier symbol: keep what we have
            break
        if _trusted_cik(evidence.massive(T, _iso_day(lo)), interval.key,
                        evidence, review_rows, mismatches) == c:
            t_start = lo
        else:
            t_start = _cik_switch_date(
                evidence, T, lo, prev_d, c, interval.key,
                review_rows, mismatches)
        clash = next((j for j in all_intervals
                      if j.key != interval.key and j.ticker == T
                      and j.start < boundary and t_start < j.end), None)
        if clash is not None:
            review_rows.append({
                "kind": "symbol_discovery_ambiguous",
                "ticker": interval.ticker, "start_date": interval.key[1],
                "other_ticker": clash.ticker, "other_start_date": clash.key[1],
                "first_date": _iso_day(t_start), "last_date": _iso_day(boundary),
                "detail": f"{T} is {clash.ticker}'s ticker during "
                          f"[{_iso_day(t_start)}, {_iso_day(boundary)})",
            })
            break
        segments.insert(0, (T, t_start, boundary))
        used.add(T)
        boundary = t_start
        if t_start == lo:
            break
    if not segments:
        return None
    # extrapolate before coverage: the earliest found symbol covers I.start
    if segments[0][1] == lo:
        segments[0] = (segments[0][0], interval.start, segments[0][2])
    return segments, s


def _discover_reassign(evidence, interval, all_intervals, review_rows,
                       mismatches, sec_name):
    """Case B (proposal only): first fact trusted c1, last trusted c2 != c1.
    If c2's historical symbol is findable, propose a `symbol`/`split`
    override as a symbol_reassigned review row."""
    c1, c2 = interval.first["cik"], interval.last["cik"]
    first_date = pd.Timestamp(interval.first["date"])
    T, why = _find_symbol(evidence, c2, first_date, {interval.ticker},
                          interval, all_intervals, review_rows, mismatches)
    if T is None and why != "guard":
        return
    if T is not None:
        s = _cik_switch_date(evidence, interval.ticker, first_date,
                             pd.Timestamp(interval.last["date"]), c2,
                             interval.key, review_rows, mismatches)
        detail = (
            f"{interval.ticker} belonged to CIK {c1} ({sec_name(c1)}) at "
            f"{interval.first['date']}; CIK {c2} ({sec_name(c2)}) traded as "
            f"{T} then and took {interval.ticker} on {_iso_day(s)} -- add "
            f"`symbol {T}` for [{_iso_day(interval.start)}, {_iso_day(s)}) "
            f"if the line is {c2}'s, or `split` at {_iso_day(s)} if the "
            "line changes company"
        )
    else:
        detail = (
            f"{interval.ticker} belonged to CIK {c1} ({sec_name(c1)}) at "
            f"{interval.first['date']} but CIK {c2} ({sec_name(c2)})'s "
            "candidate symbol is another interval's ticker (guard failed)"
        )
    review_rows.append({
        "kind": "symbol_reassigned",
        "ticker": interval.ticker, "start_date": interval.key[1],
        "other_ticker": "", "other_start_date": "",
        "first_date": interval.first["date"],
        "last_date": interval.last["date"],
        "detail": detail,
    })


# --- resolve ------------------------------------------------------------------


@dataclass
class IdentityResult:
    index_id: str
    tables_dir: str
    as_of: str
    written: bool
    n_intervals: int = 0
    n_in_coverage: int = 0
    n_massive_fetched: int = 0
    n_massive_reused: int = 0
    n_sec_fetched: int = 0
    n_cik_ticker_fetched: int = 0
    n_securities_before: int = 0
    n_securities_after: int = 0
    links: Dict[str, int] = field(default_factory=dict)
    n_unresolved: int = 0
    n_cik_rows: int = 0
    n_cik_changes: int = 0
    n_ticker_rows: int = 0
    n_discovered: int = 0
    changed_ids: List[Tuple[str, str, str, str]] = field(default_factory=list)
    review: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=IDENTITY_REVIEW_COLUMNS)
    )


def resolve(index_id: str = INDEX_ID, *, reference_source=None, edgar_source=None,
            offline: bool = False, dry_run: bool = False, refetch: bool = False,
            now=None, progress=None) -> IdentityResult:
    """Rebuild the identity tables for `index_id` from committed evidence
    plus (online) fresh lookups. `--offline` rebuilds from committed
    evidence only; `dry_run` fetches into memory but writes nothing."""
    manifest = store.load_manifest(index_id)
    if offline:
        as_of = (
            manifest.get("identity_as_of")
            or manifest.get("verified_through")
            or _now_utc(now).strftime("%Y-%m-%d")
        )
    else:
        as_of = manifest.get("verified_through") or _now_utc(now).strftime("%Y-%m-%d")

    if not offline:
        if reference_source is None:
            from data_universe.sources.massive import MassiveSource
            reference_source = MassiveSource()
        if edgar_source is None:
            from data_universe.sources.edgar import EdgarSource
            edgar_source = EdgarSource()

    massive_evidence = store.load_identity_evidence(index_id)
    sec_evidence = store.load_cik_evidence(index_id)
    cik_ticker_evidence = store.load_cik_ticker_evidence(index_id)

    evidence = _Evidence(
        massive_evidence, sec_evidence, cik_ticker_evidence,
        reference_source, edgar_source, offline, refetch, progress,
    )

    def _write_union_evidence():
        """Crash-safe checkpoint: stored + new evidence (not pruned)."""
        if dry_run:
            return
        store.save_identity_evidence(index_id, pd.DataFrame(
            evidence.all_massive_rows(), columns=IDENTITY_EVIDENCE_COLUMNS))
        store.save_cik_evidence(index_id, pd.DataFrame(
            evidence.all_sec_rows(), columns=CIK_EVIDENCE_COLUMNS))
        store.save_cik_ticker_evidence(index_id, pd.DataFrame(
            evidence.all_cik_ticker_rows(), columns=CIK_TICKER_EVIDENCE_COLUMNS))

    evidence.checkpoint_fn = _write_union_evidence

    ticker_intervals = store.load_ticker_intervals(index_id)
    old_map = store.load_security_map(index_id)
    old_master = store.load_security_master(index_id)
    overrides = _parse_overrides(
        store.load_identity_overrides(index_id), ticker_intervals)

    try:
        coverage_start = pd.Timestamp(COVERAGE_START)
        review_rows = []
        mismatches = set()
        sec_names = {}

        def sec_name(cik):
            if not cik:
                return ""
            if cik not in sec_names:
                sec_names[cik] = evidence.sec(cik)["name"]
            return sec_names[cik]

        # R6: cut `split` intervals into sub-intervals keyed (ticker, sub_start)
        intervals = []
        for _, row in ticker_intervals.sort_values("start_date").iterrows():
            ticker = row["ticker"]
            start = pd.Timestamp(row["start_date"])
            end_excl = _ts(row["end_date_excl"])
            splits = overrides["split_dates"].get((ticker, row["start_date"]), [])
            bounds = [start] + list(splits)
            for i, sub_start in enumerate(bounds):
                sub_end = bounds[i + 1] if i + 1 < len(bounds) else end_excl
                lo = max(sub_start, coverage_start)
                hi = (sub_end - pd.Timedelta(days=1)) if sub_end is not None \
                    else _ts(as_of)
                intervals.append(_Interval(
                    key=(ticker, _iso_day(sub_start)),
                    ticker=ticker, start=sub_start, end_excl=sub_end,
                    lo=lo, hi=hi, split=len(bounds) > 1,
                ))

        by_key = {iv.key: iv for iv in intervals}

        # phase 1: probe endpoint facts under (manual > S) symbols
        for iv in intervals:
            if iv.lo > iv.hi:
                continue
            sym = _symbol_for(iv.key, overrides["symbol"], iv.ticker)
            iv.first, iv.first_tried = _probe_fact(
                evidence, sym, iv.lo, 1, PROBE_OFFSETS_DAYS, iv.hi,
                iv.key, review_rows, mismatches)
            iv.last, iv.last_tried = _probe_fact(
                evidence, sym, iv.hi, -1, PROBE_OFFSETS_DAYS, iv.lo,
                iv.key, review_rows, mismatches)

        # phase 2: symbol discovery + re-probe; then symbol_not_found
        n_discovered = 0
        for iv in intervals:
            if iv.lo > iv.hi or iv.first is None and iv.last is None:
                continue
            has_manual_symbol = any(
                (r["ticker"], r["start_date"]) == iv.key
                for r in overrides["symbol"])
            if has_manual_symbol:
                continue
            first_cik = iv.first["cik"] if iv.first else None
            last_cik = iv.last["cik"] if iv.last else None
            if not first_cik and last_cik:
                found = _discover(evidence, iv, intervals,
                                  review_rows, mismatches)
                if found is not None:
                    iv.sym_segments, iv.sym_takeover = found
                    sym = _probe_symbol_for(iv, overrides["symbol"])
                    iv.first, iv.first_tried = _probe_fact(
                        evidence, sym, iv.lo, 1, PROBE_OFFSETS_DAYS, iv.hi,
                        iv.key, review_rows, mismatches)
                    n_discovered += 1
            elif first_cik and last_cik and first_cik != last_cik:
                _discover_reassign(evidence, iv, intervals,
                                   review_rows, mismatches, sec_name)

        for iv in intervals:
            if iv.lo > iv.hi:
                continue
            if iv.first is None or iv.last is None:
                sides = []
                if iv.first is None:
                    sides.append("first")
                if iv.last is None:
                    sides.append("last")
                review_rows.append({
                    "kind": "symbol_not_found",
                    "ticker": iv.ticker, "start_date": iv.key[1],
                    "other_ticker": "", "other_start_date": "",
                    "first_date": _iso_day(iv.lo), "last_date": _iso_day(iv.hi),
                    "detail": (
                        f"{'/'.join(sides)} fact missing; tried "
                        f"{list(iv.first_tried) + list(iv.last_tried)}"
                    ),
                })

        # D4: CIK segments per interval
        interval_segments = {}
        for iv in intervals:
            interval_segments[iv.key] = _cik_segments(
                evidence, iv, overrides["symbol"], review_rows, mismatches,
                sec_name)

        # D6: linking
        edges, review_rows = _link(intervals, interval_segments, overrides,
                                 review_rows, sec_name)
        uf = _UF()
        applied_edges = []
        for kind, a, b in edges:
            iv_a, iv_b = by_key[a], by_key[b]
            if uf.find(a) == uf.find(b):
                continue
            comp_a = [k for k in by_key if uf.find(k) == uf.find(a)]
            comp_b = [k for k in by_key if uf.find(k) == uf.find(b)]
            clash = any(by_key[ka].overlaps(by_key[kb])
                        for ka in comp_a for kb in comp_b)
            if clash:
                if kind == "manual":
                    raise MembershipValidationError(
                        f"manual link of {a} and {b} would overlap intervals"
                    )
                review_rows.append({
                    "kind": "link_would_overlap",
                    "ticker": iv_a.ticker, "start_date": a[1],
                    "other_ticker": iv_b.ticker, "other_start_date": b[1],
                    "first_date": "", "last_date": "",
                    "detail": f"{kind} edge {a} -> {b} skipped: components "
                              "would overlap",
                })
                continue
            uf.union(a, b)
            applied_edges.append((kind, a, b))

        # component ids: earliest (start, ticker) in each component
        components = {}
        for iv in intervals:
            components.setdefault(uf.find(iv.key), []).append(iv)
        security_id_of = {}
        for members in components.values():
            earliest = min(members, key=lambda iv: (iv.start, iv.ticker))
            sid = f"{earliest.ticker}-{earliest.start.strftime('%Y%m%d')}"
            for iv in members:
                security_id_of[iv.key] = sid

        connect_kind = {}
        for kind, a, b in applied_edges:
            connect_kind.setdefault(b, kind)
        overridden_intervals = set()
        for r in overrides["link"] + overrides["unlink"]:
            overridden_intervals.add((r["ticker"], r["start_date"]))
            overridden_intervals.add((r["other_ticker"], r["other_start_date"]))

        # D7: security_map -- one row per sub-interval
        map_rows = []
        for iv in sorted(intervals, key=lambda x: (x.ticker, x.start)):
            key = iv.key
            sid = security_id_of[key]
            if key in connect_kind:
                evidence_label = {"manual": "manual_link", "figi": "figi_link",
                                  "cik": "cik_link"}[connect_kind[key]]
            elif iv.first is not None or iv.last is not None:
                evidence_label = "massive"
            else:
                evidence_label = "new_default"
            reviewed = iv.split or key in overridden_intervals
            map_rows.append({
                "index_id": index_id, "ticker": iv.ticker,
                "start_date": key[1], "security_id": sid,
                "evidence": evidence_label, "reviewed": reviewed,
            })
        security_map = pd.DataFrame(map_rows, columns=SECURITY_MAP_COLUMNS)

        # sanity: unlink pair that ended in one component raises
        for row in overrides["unlink"]:
            a = (row["ticker"], row["start_date"])
            b = (row["other_ticker"], row["other_start_date"])
            if security_id_of[a] == security_id_of[b]:
                raise MembershipValidationError(
                    f"unlink override for {a} / {b} contradicts transitive links"
                )

        members_by_sid = _members_by_security(security_id_of, intervals)

        # D8: cik_history per security -- massive segments, manual cuts
        sec_segments = {}
        for iv in intervals:
            for seg in interval_segments[iv.key]:
                sec_segments.setdefault(security_id_of[iv.key], []).append(seg)
        cik_rows = []
        for sid, ivs in members_by_sid.items():
            merged = _merge_segments(sec_segments.get(sid, []))
            rows = [[c, vf, vt, "massive"] for vf, vt, c in merged]
            rows = _cut_manual(
                rows, overrides["cik"], {iv.key for iv in ivs}, "cik")
            for c, vf, vt, src in rows:
                cik_rows.append({
                    "security_id": sid, "cik": c, "valid_from": vf,
                    "valid_to_excl": vt, "source": src,
                })
        cik_history = pd.DataFrame(cik_rows, columns=CIK_HISTORY_COLUMNS)

        # D9: ticker_history per security -- membership rows (+ discovered
        # massive rows), manual symbol cuts
        ticker_rows = []
        for sid, ivs in members_by_sid.items():
            segs = []
            for iv in ivs:
                for sym_t, vf, vt in iv.sym_segments:
                    segs.append((sym_t, vf, vt, "massive"))
                if iv.sym_takeover is not None:
                    pieces = [(iv.start, iv.sym_takeover)]
                    for _t, vf, vt in iv.sym_segments:
                        pieces = [p for ps, pe in pieces
                                  for p in _subtract(ps, pe, vf, vt)]
                    for ps, pe in pieces:
                        segs.append((iv.ticker, ps, pe, "membership"))
                    segs.append(
                        (iv.ticker, iv.sym_takeover, iv.end_excl, "membership"))
                else:
                    segs.append((iv.ticker, iv.start, iv.end_excl, "membership"))
            merged = _merge_named(segs)
            rows = [[t, vf, vt, src] for t, vf, vt, src in merged]
            rows = _cut_manual(
                rows, overrides["symbol"], {iv.key for iv in ivs}, "symbol")
            for t, vf, vt, src in rows:
                ticker_rows.append({
                    "security_id": sid, "ticker": t, "valid_from": vf,
                    "valid_to_excl": vt, "source": src,
                })
        ticker_history = pd.DataFrame(ticker_rows, columns=TICKER_HISTORY_COLUMNS)

        # D10: security_master
        master_rows = []
        old_master_by_id = {r["security_id"]: r for _, r in old_master.iterrows()}
        for sid in sorted(set(security_id_of.values())):
            members = sorted(members_by_sid[sid], key=lambda iv: (iv.start, iv.ticker))
            latest_iv = max(members, key=lambda iv: (iv.start, iv.ticker))
            sid_cik_rows = cik_history[cik_history["security_id"] == sid]
            last_cik = sid_cik_rows["cik"].iloc[-1] if len(sid_cik_rows) else ""
            comp_figi = share_figi = ""
            for iv in sorted(members, key=lambda i: (i.start, i.ticker), reverse=True):
                for fact in (iv.last, iv.first):
                    if fact is None:
                        continue
                    if not comp_figi and fact["composite_figi"]:
                        comp_figi = fact["composite_figi"]
                    if not share_figi and fact["share_class_figi"]:
                        share_figi = fact["share_class_figi"]
                if comp_figi and share_figi:
                    break
            name = sec_name(last_cik) if last_cik else ""
            old = old_master_by_id.get(sid)
            master_rows.append({
                "security_id": sid, "name": name or "",
                "latest_ticker": latest_iv.ticker, "cik": last_cik or "",
                "composite_figi": comp_figi, "share_class_figi": share_figi,
                "notes": ("" if old is None else (old["notes"] or "")),
            })
        security_master = pd.DataFrame(master_rows, columns=SECURITY_MASTER_COLUMNS)

        # D11: cik_starts_late + dormant-CIK reviews
        review_rows += _cik_flag_reviews(
            cik_history, evidence, sec_name, intervals, security_id_of)

        # apply ack overrides
        ack = {(r["value"], r["ticker"], r["start_date"])
               for r in overrides["ack"]}
        review_rows = [
            r for r in review_rows
            if (r["kind"], r["ticker"], r["start_date"]) not in ack
        ]
        review = pd.DataFrame(review_rows, columns=IDENTITY_REVIEW_COLUMNS)

        # changed ids vs the old map
        old_ids = {(r["ticker"], r["start_date"]): r["security_id"]
                   for _, r in old_map.iterrows()}
        changed_ids = []
        for key, sid in sorted(security_id_of.items(),
                               key=lambda x: (x[0][1], x[0][0])):
            old = old_ids.get(key, key[0] + "-" + key[1].replace("-", ""))
            if old != sid:
                changed_ids.append((key[0], key[1], old, sid))

        n_unresolved = int((security_map["evidence"] == "new_default").sum())
        n_in_coverage = sum(
            1 for iv in intervals
            if iv.first is not None or iv.last is not None
        )
        result = IdentityResult(
            index_id=index_id, tables_dir=str(store.tables_dir(index_id)),
            as_of=as_of, written=not dry_run,
            n_intervals=len(intervals),
            n_in_coverage=n_in_coverage,
            n_massive_fetched=evidence.n_massive_fetched,
            n_massive_reused=evidence.n_massive_reused,
            n_sec_fetched=evidence.n_sec_fetched,
            n_cik_ticker_fetched=evidence.n_cik_ticker_fetched,
            n_securities_before=old_map["security_id"].nunique()
            if len(old_map) else 0,
            n_securities_after=security_map["security_id"].nunique(),
            links={
                "manual": sum(1 for e in applied_edges if e[0] == "manual"),
                "figi": sum(1 for e in applied_edges if e[0] == "figi"),
                "cik": sum(1 for e in applied_edges if e[0] == "cik"),
            },
            n_unresolved=n_unresolved,
            n_cik_rows=len(cik_history),
            n_cik_changes=int(
                (review["kind"] == "cik_change_within_interval").sum()),
            n_ticker_rows=len(ticker_history),
            n_discovered=n_discovered,
            changed_ids=changed_ids,
            review=review,
        )

        if not dry_run:
            _write(
                index_id, security_map, security_master, ticker_history,
                cik_history, review, evidence, manifest, as_of, result,
                offline=offline,
            )
        return result
    except BaseException:
        # on error, flush the union of stored + new evidence so a
        # rerun resumes from the checkpointed tables
        _write_union_evidence()
        raise


# --- D4: CIK segments ----------------------------------------------------------


def _cik_segments(evidence, interval, symbol_rows, review_rows, mismatches,
                  sec_name):
    """[(start, end_excl, cik)] for one interval -- one segment over the
    interval when only one side has a trusted cik (or both agree),
    bisected segments when both have differing trusted ciks."""
    first, last = interval.first, interval.last
    ciks = [f["cik"] for f in (first, last) if f and f["cik"]]
    if not ciks:
        return []
    if len(ciks) == 1 or first["cik"] == last["cik"]:
        return [(interval.start, interval.end_excl, ciks[0])]

    sym = _probe_symbol_for(interval, symbol_rows)
    probed = {pd.Timestamp(first["date"]): first["cik"],
              pd.Timestamp(last["date"]): last["cik"]}
    switches = _bisect_cik(
        evidence, sym, pd.Timestamp(first["date"]), first["cik"],
        pd.Timestamp(last["date"]), last["cik"], probed, interval,
        review_rows, mismatches, sec_name, depth=0,
    )
    segs = []
    cik, seg_start = first["cik"], interval.start
    for switch_date, c_before, c_after in switches:
        segs.append((seg_start, switch_date, cik))
        seg_start, cik = switch_date, c_after
    segs.append((seg_start, interval.end_excl, cik))
    return segs


def _bisect_cik(evidence, sym, lo_date, c_lo, hi_date, c_hi, probed,
                interval, review_rows, mismatches, sec_name, depth):
    """Bisect calendar days between lo_date (cik c_lo) and hi_date (cik c_hi)
    to find switch dates; returns [(switch_date, c_before, c_hi)] ordered."""
    lo, hi = lo_date, hi_date
    while (hi - lo).days > 1:
        mid = lo + pd.Timedelta(days=(hi - lo).days // 2)
        record = evidence.massive(sym(mid), _iso_day(mid))
        cik = _trusted_cik(
            record, interval.key, evidence, review_rows, mismatches)
        probed[mid] = cik
        if cik == c_hi:
            hi = mid
        else:
            lo = mid
    s = hi  # first date whose cik equals c_hi
    known_before = sorted(
        (d, c) for d, c in probed.items() if d < s and c)
    if known_before:
        d_before, c_before = known_before[-1]
    else:
        d_before, c_before = lo_date, c_lo
    switches = []
    if c_before == c_lo:
        switches.append((s, c_lo, c_hi))
        _cik_change_review(review_rows, interval, c_lo, c_hi, s,
                           known_before, sec_name)
    elif depth < MAX_CIK_SWITCHES - 1:
        switches += _bisect_cik(
            evidence, sym, lo_date, c_lo, d_before, c_before, probed,
            interval, review_rows, mismatches, sec_name, depth + 1)
        switches.append((s, c_before, c_hi))
        _cik_change_review(review_rows, interval, c_before, c_hi, s,
                           known_before, sec_name)
    else:
        review_rows.append({
            "kind": "too_many_cik_changes", "ticker": interval.ticker,
            "start_date": interval.key[1], "other_ticker": "",
            "other_start_date": "", "first_date": _iso_day(d_before),
            "last_date": _iso_day(s),
            "detail": f">={MAX_CIK_SWITCHES} CIK changes within the interval",
        })
        switches.append((s, c_lo, c_hi))
    return switches


def _cik_change_review(review_rows, interval, c_a, c_b, s, known_before, sec_name):
    first_date = (max(known_before, key=lambda x: x[0])[0]
                  if known_before else pd.Timestamp(s))
    review_rows.append({
        "kind": "cik_change_within_interval", "ticker": interval.ticker,
        "start_date": interval.key[1], "other_ticker": "",
        "other_start_date": "", "first_date": _iso_day(first_date),
        "last_date": _iso_day(s),
        "detail": (
            f"CIK {c_a} ({sec_name(c_a)}) -> {c_b} ({sec_name(c_b)}) "
            f"on {_iso_day(s)}"
        ),
    })


# --- D6: linking ---------------------------------------------------------------


def _link(intervals, interval_segments, overrides, review_rows, sec_name):
    """Collect (kind, a_key, b_key) edges: manual links, then figi edges,
    then cik edges, each group sorted by (B.start, B.ticker, A.start,
    A.ticker). `a` is the earlier interval by (start, ticker). Pair reviews
    are skipped for pairs named in a link/unlink override, either order."""
    by_key = {iv.key: iv for iv in intervals}
    overridden_pairs = set()
    for r in overrides["unlink"]:
        overridden_pairs.add(frozenset(((r["ticker"], r["start_date"]),
                                        (r["other_ticker"], r["other_start_date"]))))
    for r in overrides["link"]:
        overridden_pairs.add(frozenset(((r["ticker"], r["start_date"]),
                                        (r["other_ticker"], r["other_start_date"]))))

    def not_overridden(a, b):
        return frozenset((a, b)) not in overridden_pairs

    def not_unlinked(a, b):
        pair = frozenset((a, b))
        for r in overrides["unlink"]:
            if pair == frozenset(((r["ticker"], r["start_date"]),
                                  (r["other_ticker"], r["other_start_date"]))):
                return False
        return True

    def ordered(a_key, b_key):
        """Return the pair (earlier, later) by (start, ticker)."""
        iv_a, iv_b = by_key[a_key], by_key[b_key]
        return (a_key, b_key) if (iv_a.start, iv_a.ticker) < (iv_b.start, iv_b.ticker) \
            else (b_key, a_key)

    edges = []
    for row in overrides["link"]:
        a = (row["ticker"], row["start_date"])
        b = (row["other_ticker"], row["other_start_date"])
        edges.append(("manual", *ordered(a, b)))

    # figi edges: A.end_excl <= B.start, same non-null composite figi
    by_last_figi = {}
    for iv in intervals:
        if iv.last and iv.last["composite_figi"] and iv.end_excl is not None:
            by_last_figi.setdefault(iv.last["composite_figi"], []).append(iv)
    figi_edges = set()
    for b in intervals:
        if not (b.first and b.first["composite_figi"]):
            continue
        for a in by_last_figi.get(b.first["composite_figi"], []):
            if a.end_excl <= b.start and not_unlinked(a.key, b.key):
                figi_edges.add((a.key, b.key))
    edges += [("figi", a, b) for a, b in sorted(
        figi_edges, key=lambda p: (by_key[p[1]].start, by_key[p[1]].ticker,
                                   by_key[p[0]].start, by_key[p[0]].ticker))]

    # cik edges: touching only, same cik, unique on both sides within the date
    by_touch_date = {}
    for a in intervals:
        if a.end_excl is not None and a.last and a.last["cik"]:
            by_touch_date.setdefault(a.end_excl, []).append(a)
    by_start_date = {}
    for b in intervals:
        if b.first and b.first["cik"]:
            by_start_date.setdefault(b.start, []).append(b)
    for touch, ends in by_touch_date.items():
        starts = by_start_date.get(touch, [])
        pairs = []
        for a in ends:
            for b in starts:
                if a is b or not not_unlinked(a.key, b.key):
                    continue
                if (a.key, b.key) in figi_edges:
                    continue
                if a.last["cik"] != b.first["cik"]:
                    continue
                pairs.append((a, b))
        a_candidates = {}
        b_candidates = {}
        mismatched = set()
        for a, b in pairs:
            a_candidates.setdefault(a.key, []).append(b.key)
            b_candidates.setdefault(b.key, []).append(a.key)
            fa, fb = a.last, b.first
            cf_diff = (fa["composite_figi"] and fb["composite_figi"]
                       and fa["composite_figi"] != fb["composite_figi"])
            sc_diff = (fa["share_class_figi"] and fb["share_class_figi"]
                       and fa["share_class_figi"] != fb["share_class_figi"])
            if (cf_diff or sc_diff) and not_overridden(a.key, b.key):
                mismatched.add((a.key, b.key))
                review_rows.append({
                    "kind": "same_cik_figi_mismatch",
                    "ticker": a.ticker, "start_date": a.key[1],
                    "other_ticker": b.ticker, "other_start_date": b.key[1],
                    "first_date": "", "last_date": "",
                    "detail": (
                        f"same CIK {a.last['cik']} at touch {a.key}->{b.key} "
                        f"but figis differ ({fa['composite_figi']}/"
                        f"{fa['share_class_figi']} vs {fb['composite_figi']}/"
                        f"{fb['share_class_figi']})"
                    ),
                })
        for a, b in pairs:
            a_key, b_key = a.key, b.key
            if (a_key, b_key) in mismatched:
                continue
            if len(a_candidates[a_key]) == 1 and len(b_candidates[b_key]) == 1:
                edges.append(("cik", a_key, b_key))
            elif not_overridden(a_key, b_key):
                review_rows.append({
                    "kind": "ambiguous_cik_link",
                    "ticker": a.ticker, "start_date": a_key[1],
                    "other_ticker": b.ticker, "other_start_date": b_key[1],
                    "first_date": "", "last_date": "",
                    "detail": (
                        f"{len(a_candidates[a_key])}x{len(b_candidates[b_key])} "
                        f"candidates share CIK {a.last['cik']} at {_iso_day(touch)}"
                    ),
                })

    # gap review: A.end < B.start, same cik, a figi missing, no figi edge
    by_end = sorted(
        (iv for iv in intervals if iv.end_excl is not None and iv.last
         and iv.last["cik"]),
        key=lambda iv: iv.end_excl,
    )
    for b in intervals:
        if not (b.first and b.first["cik"]):
            continue
        candidates = [
            a for a in by_end
            if a.last["cik"] == b.first["cik"] and a.end_excl < b.start
            and (a.key, b.key) not in figi_edges and not_unlinked(a.key, b.key)
            and not_overridden(a.key, b.key)
            and (not a.last["composite_figi"] or not b.first["composite_figi"])
        ]
        if candidates:
            a = max(candidates, key=lambda x: x.end_excl)
            review_rows.append({
                "kind": "rejoin_unconfirmed",
                "ticker": a.ticker, "start_date": a.key[1],
                "other_ticker": b.ticker, "other_start_date": b.key[1],
                "first_date": _iso_day(a.end_excl), "last_date": _iso_day(b.start),
                "detail": (
                    f"same CIK {a.last['cik']} across a gap "
                    f"({_iso_day(a.end_excl)} -> {_iso_day(b.start)}) "
                    "but a composite FIGI is missing"
                ),
            })

    # manual first, then figi, then cik (sorted within each group)
    order = {"manual": 0, "figi": 1, "cik": 2}
    edges.sort(key=lambda e: (
        order[e[0]], by_key[e[2]].start, by_key[e[2]].ticker,
        by_key[e[1]].start, by_key[e[1]].ticker))
    return edges, review_rows


# --- merge/cut helpers ----------------------------------------------------------


def _merge_segments(segs):
    """Merge consecutive same-cik (start, end, cik) segments -- touching or
    across a gap -- into [(vf_str, vt_str, cik)]."""
    out = []
    for start, end, cik in sorted(segs, key=lambda x: x[0]):
        if out and out[-1][2] == cik:
            if end is None:
                out[-1][1] = None
            elif out[-1][1] is not None:
                out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end, cik])
    return [[_iso_day(a), (_iso_day(b) if b is not None else ""), c]
            for a, b, c in out]


def _merge_named(segs):
    """Merge consecutive same-ticker AND same-source
    (ticker, start, end, source) rows -- touching or across a gap -- into
    [(ticker, vf_str, vt_str, source)]."""
    out = []
    for ticker, start, end, src in sorted(segs, key=lambda x: x[1]):
        if out and out[-1][0] == ticker and out[-1][3] == src:
            out[-1][2] = end
        else:
            out.append([ticker, start, end, src])
    return [[t, _iso_day(a), (_iso_day(b) if b is not None else ""), src]
            for t, a, b, src in out]


def _subtract(ps, pe, mvf, mvt):
    """Subtract [mvf, mvt) from [ps, pe); yields the remaining pieces."""
    if mvt <= ps or mvf >= pe:
        yield (ps, pe)
        return
    if ps < mvf:
        yield (ps, mvf)
    if mvt < pe:
        yield (mvt, pe)


def _cut_manual(rows, manual_rows, sid_keys, kind):
    """Cut manual cik/symbol overrides into `[[value, vf, vt, src]]` rows:
    remove each applicable row's [valid_from, valid_to_excl) from existing
    rows (splitting partial overlaps, dropping covered rows), then insert
    the manual row. Applicable = the override's interval belongs to this
    security (`sid_keys`). Overlapping manual rows for one security raise."""
    applicable = [
        m for m in manual_rows if (m["ticker"], m["start_date"]) in sid_keys
    ]
    if not applicable:
        return rows
    spans = [
        (_ts(m["valid_from"]), _ts(m["valid_to_excl"]) or _OPEN, str(m["value"]))
        for m in applicable
    ]
    for i, (vf, vt, _) in enumerate(spans):
        for ovf, ovt, _ in spans[i + 1:]:
            if vf < ovt and ovf < vt:
                raise MembershipValidationError(
                    f"overlapping manual {kind} rows for security containing "
                    f"{applicable[0]['ticker']}/{applicable[0]['start_date']}"
                )
    out = []
    for value, rvf_s, rvt_s, src in rows:
        pieces = [(_ts(rvf_s), _ts(rvt_s) or _OPEN)]
        for mvf, mvt, _ in spans:
            pieces = [
                piece for ps, pe in pieces
                for piece in _subtract(ps, pe, mvf, mvt)
            ]
        for ps, pe in pieces:
            out.append([value, _iso_day(ps),
                        "" if pe == _OPEN else _iso_day(pe), src])
    for mvf, mvt, mvalue in spans:
        out.append([mvalue, _iso_day(mvf),
                    "" if mvt == _OPEN else _iso_day(mvt), "manual"])
    out.sort(key=lambda r: r[1])
    return out


def _members_by_security(security_id_of, intervals):
    out = {}
    for iv in intervals:
        out.setdefault(security_id_of[iv.key], []).append(iv)
    return out


# --- D11 late-CIK review --------------------------------------------------------


def _cik_flag_reviews(cik_history, evidence, sec_name, intervals, security_id_of):
    """cik_starts_late (the registrant's first filing postdates the segment)
    and cik_no_earnings_filings (the CIK never filed an Item 2.02 -- dormant
    or subsidiary registrant carrying the index line)."""
    rows = []
    members = _members_by_security(security_id_of, intervals)
    coverage = pd.Timestamp(COVERAGE_START)
    for _, row in cik_history.iterrows():
        if row["source"] != "massive" or not row["cik"]:
            continue
        summary = evidence.sec(row["cik"])
        sid = row["security_id"]
        vf = pd.Timestamp(row["valid_from"])
        check = max(vf, coverage)
        vt = _ts(row["valid_to_excl"])
        members_list = sorted(members.get(sid, []), key=lambda iv: iv.start)
        holder = next((iv for iv in members_list
                       if iv.start <= check < iv.end), members_list[0] if members_list else None)
        if summary.get("has_item_202") == "false":
            rows.append({
                "kind": "cik_no_earnings_filings",
                "ticker": holder.ticker if holder else "",
                "start_date": holder.key[1] if holder else "",
                "other_ticker": "", "other_start_date": "",
                "first_date": row["valid_from"],
                "last_date": row["valid_to_excl"],
                "detail": (
                    f"CIK {row['cik']} ({summary['name']}) has never filed an "
                    "8-K Item 2.02 -- likely a dormant or subsidiary "
                    "registrant; set a `cik` override for "
                    f"[{row['valid_from']}, {row['valid_to_excl']})"
                ),
            })
        first_filed = _ts(summary["first_filed"]) if summary["first_filed"] else None
        if first_filed is None:
            continue
        extends = vt is None or vt > check
        if not (extends and first_filed > check + pd.Timedelta(days=CIK_LATE_SLACK_DAYS)):
            continue
        rows.append({
            "kind": "cik_starts_late", "ticker": holder.ticker if holder else "",
            "start_date": holder.key[1] if holder else "",
            "other_ticker": "", "other_start_date": "",
            "first_date": _iso_day(check), "last_date": _iso_day(first_filed),
            "detail": (
                f"CIK {row['cik']} ({summary['name']}) first filed "
                f"{_iso_day(first_filed)}; segment starts {row['valid_from']} -- "
                "a predecessor CIK is likely needed before "
                f"{_iso_day(first_filed)}"
            ),
        })
    return rows


# --- D12: write -----------------------------------------------------------------


def _write(index_id, security_map, security_master, ticker_history, cik_history,
           review, evidence, manifest, as_of, result, offline=False):
    """Stage the tables next to a copy of the tables dir, validate, install."""
    target = store.tables_dir(index_id)
    previous = config.get_membership_tables_dir()
    consulted_massive = evidence.consulted_massive_rows()
    consulted_sec = evidence.consulted_sec_rows()
    consulted_cik_ticker = evidence.consulted_cik_ticker_rows()
    with tempfile.TemporaryDirectory() as staging_root:
        staged = Path(staging_root) / index_id
        if target.is_dir():
            shutil.copytree(target, staged)
        staged.mkdir(parents=True, exist_ok=True)
        config.set_membership_tables_dir(staging_root)
        try:
            store.save_security_map(index_id, security_map)
            store.save_security_master(index_id, security_master)
            store.save_ticker_history(index_id, ticker_history)
            store.save_cik_history(index_id, cik_history)
            store.save_identity_review(index_id, review)
            store.save_identity_evidence(
                index_id, pd.DataFrame(
                    consulted_massive, columns=IDENTITY_EVIDENCE_COLUMNS))
            store.save_cik_evidence(
                index_id, pd.DataFrame(consulted_sec, columns=CIK_EVIDENCE_COLUMNS))
            store.save_cik_ticker_evidence(
                index_id, pd.DataFrame(
                    consulted_cik_ticker, columns=CIK_TICKER_EVIDENCE_COLUMNS))
            new_manifest = dict(manifest)
            new_manifest.update({
                "identity_tool_version": TOOL_VERSION,
                "identity_built_at": (
                    # offline rebuilds must reproduce committed bytes
                    manifest.get("identity_built_at")
                    if offline
                    else pd.Timestamp.now(tz="UTC").isoformat()
                ),
                "identity_as_of": as_of,
                "identity_coverage_start": COVERAGE_START,
                "identity_n_intervals": result.n_intervals,
                "identity_n_in_coverage": result.n_in_coverage,
                "identity_n_securities": result.n_securities_after,
                "identity_n_unresolved": result.n_unresolved,
                "identity_n_review": len(result.review),
            })
            store.save_manifest(index_id, new_manifest)
            _validate_staged(index_id)
        finally:
            config.set_membership_tables_dir(previous)
        target.mkdir(parents=True, exist_ok=True)
        for name in _REBASE_FILES:
            shutil.copyfile(staged / name, target / name)


def _validate_staged(index_id):
    _intervals_mod.membership_intervals(index_id)
    master_ids = set(store.load_security_master(index_id)["security_id"])
    map_ids = set(store.load_security_map(index_id)["security_id"])
    missing = map_ids - master_ids
    if missing:
        raise MembershipValidationError(
            f"security_map ids missing from security_master: {sorted(missing)[:5]}"
        )
    for name, loader in (("ticker_history", store.load_ticker_history),
                         ("cik_history", store.load_cik_history)):
        frame = loader(index_id)
        for sid, group in frame.groupby("security_id"):
            ends = pd.to_datetime(group["valid_to_excl"], errors="coerce")
            starts = pd.to_datetime(group["valid_from"])
            order = starts.argsort(kind="stable")
            starts, ends = starts.iloc[order], ends.iloc[order]
            prev_end = None
            for s, e in zip(starts, ends):
                if prev_end is not None and s < prev_end:
                    raise MembershipValidationError(
                        f"{name}: overlapping rows for {sid} at {s.date()}"
                    )
                prev_end = e if pd.notna(e) else _OPEN


# --- summary / CLI ---------------------------------------------------------------


def format_summary(result: IdentityResult, max_listed: int = 25) -> str:
    lines = [
        f"identity({result.index_id}) as_of={result.as_of} "
        f"written={result.written}",
        f"intervals {result.n_intervals} ({result.n_in_coverage} in coverage); "
        f"securities {result.n_securities_before} -> {result.n_securities_after}",
        f"links: manual={result.links.get('manual', 0)} "
        f"figi={result.links.get('figi', 0)} cik={result.links.get('cik', 0)}; "
        f"unresolved {result.n_unresolved}; "
        f"symbol discovery applied to {result.n_discovered} interval(s)",
        f"evidence: massive {result.n_massive_fetched} fetched "
        f"({result.n_massive_reused} reused), sec {result.n_sec_fetched} "
        f"fetched, tickers-for-cik {result.n_cik_ticker_fetched} fetched",
        f"cik_history {result.n_cik_rows} rows "
        f"({result.n_cik_changes} mid-interval changes); "
        f"ticker_history {result.n_ticker_rows} rows",
        f"review {len(result.review)} rows"
        + (": " + ", ".join(
            f"{k}={n}" for k, n in
            result.review["kind"].value_counts().sort_index().items())
           if len(result.review) else ""),
    ]
    if result.changed_ids:
        lines.append(f"changed ids ({len(result.changed_ids)}, showing "
                     f"{min(max_listed, len(result.changed_ids))}):")
        for ticker, start, old, new in result.changed_ids[:max_listed]:
            lines.append(f"  {ticker} {start}: {old} -> {new}")
    if len(result.review):
        lines.append(f"review rows (showing {min(max_listed, len(result.review))}):")
        for _, row in result.review.head(max_listed).iterrows():
            lines.append(f"  {row['kind']} {row['ticker']} {row['start_date']}: "
                         f"{row['detail']}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m data_universe.membership.identity")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--offline", action="store_true",
                       help="rebuild from committed evidence only")
    group.add_argument("--refetch", action="store_true",
                       help="refetch every consulted key once")
    parser.add_argument("--dry-run", action="store_true",
                        help="report changes without writing anything")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--max-listed", type=int, default=25)
    args = parser.parse_args(argv)

    config.load_yaml(args.config)
    result = resolve(
        INDEX_ID,
        offline=args.offline,
        dry_run=args.dry_run,
        refetch=args.refetch,
        progress=print,
    )
    print(format_summary(result, args.max_listed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
