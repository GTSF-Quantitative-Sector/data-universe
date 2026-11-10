# S&P 500 point-in-time membership tables

Vendored from [fja05680/sp500](https://github.com/fja05680/sp500), file
`S&P 500 Historical Components & Changes (Updated).csv`: one row per membership-change date,
`date,tickers`, where `tickers` is that day's full comma-separated member list. Coverage starts
at the first upstream row (1996-01-02). `manifest.json` records exactly which upstream commit
and file bytes the committed tables were built from.

## Files

| File | Contents |
|---|---|
| `ticker_intervals.csv` | `index_id, ticker, start_date, end_date_excl`: one row per run of consecutive upstream snapshots that list the ticker. `start_date` is the first snapshot date listing it, `end_date_excl` the first snapshot date that no longer does (empty: still a member). A ticker is a member on session `t` when `start_date <= t < end_date_excl`. |
| `security_map.csv` | `index_id, ticker, start_date, security_id, evidence, reviewed`: the security each interval belongs to. `evidence` is `manual_link`, `figi_link`, `cik_link` (how the interval joined its security), `massive` (Massive evidence exists but it starts a component), or `new_default` (unresolved). `reviewed` marks intervals named in a link/unlink override. |
| `security_master.csv` | One row per `security_id`: `name` (SEC registrant name), `latest_ticker`, `cik` (last cik_history row), `composite_figi`, `share_class_figi`, `notes`. |
| `ticker_history.csv` | `security_id, ticker, valid_from, valid_to_excl, source`: which symbol each security traded under. `source` is `membership`, `massive` (a symbol discovered via `tickers_for_cik`) or `manual` (a `symbol` override cut). |
| `cik_history.csv` | `security_id, cik, valid_from, valid_to_excl, source`: which SEC CIK each security filed under (mid-interval re-registrations become multiple rows). `source` is `massive` or `manual`. |
| `identity_evidence.csv` | `ticker, date, status, cik, name, composite_figi, share_class_figi, fetched_at`: every Massive `?date=` lookup the resolver consulted. |
| `cik_ticker_evidence.csv` | `cik, date, tickers, fetched_at`: every `tickers_for_cik` lookup (symbol discovery). |
| `cik_evidence.csv` | `cik, status, name, former_names, first_filed, has_item_202, fetched_at`: every SEC submissions summary consulted. |
| `identity_review.csv` | `kind, ticker, start_date, other_ticker, other_start_date, first_date, last_date, detail`: the human triage queue (symbol_not_found, cik_change_within_interval, too_many_cik_changes, ambiguous_cik_link, same_cik_figi_mismatch, rejoin_unconfirmed, link_would_overlap, cik_starts_late, cik_name_mismatch, symbol_discovery_ambiguous, symbol_reassigned, cik_no_earnings_filings). |
| `identity_overrides.csv` | `action, ticker, start_date, other_ticker, other_start_date, value, valid_from, valid_to_excl, note`: human-authored overrides (`link`, `unlink`, `cik`, `symbol`, `split`, `ack`). Tools only ever read this file. |
| `tail_changes.csv` | Reserved for changes newer than upstream. Empty, and never written by the refresh tool. |
| `manifest.json` | Provenance: `upstream_repo`, `upstream_commit_sha`, `upstream_commit_date`, `upstream_filename`, `upstream_sha256`, `upstream_last_snapshot_date`, row counts, `retrieved_at`, `tool_version`, and `verified_through` (the UTC date the tables were last confirmed against upstream's latest commit). |

## Identity resolution

`python -m data_universe.membership.identity` resolves intervals into securities using
Massive `?date=` ticker details (as-of CIK, composite FIGI, share-class FIGI) plus SEC
submissions summaries. Renames keep the composite FIGI, so FB and META link into one
security (`FB-20131223`); a company that left and rejoined (AMD) merges; two companies that
used the same ticker years apart (the two Dows) stay apart. Ids are `<ticker>-<YYYYMMDD>`
of each security's *first* membership interval.

Evidence is conservative: a FIGI link needs matching non-null composite FIGIs; a CIK-only
link needs touching intervals with matching CIKs, unique candidates on both sides, and no
FIGI contradiction. Anything ambiguous lands in `identity_review.csv` for a human to answer
via `identity_overrides.csv`; events are missed, never mis-attributed. A
record's CIK is used only when its registrant name matches the SEC name (or a
former name) for that CIK; historical symbols are discovered by walking
`tickers_for_cik` backwards from the upstream takeover date; `split` overrides
cut intervals where upstream folded two companies into one line; and a
`cik_no_earnings_filings` row flags any registrant that never filed an Item
2.02 (dormant/subsidiary CIK like DISCA's 1024333 -- fixed by the DISCA/DISCK
`cik` rows to 1437107). Where upstream folds two companies into one line the
`split` action cuts the interval at the change-of-company date; where it lists
the same company twice (CPRI listed from 2016 while KORS was still listed
until 2018-09-19), a split plus a link attaches the real part while the
duplicate part keeps its own id and has no vendor data. The manifest records
`identity_as_of`, the build counts, and `identity_n_unresolved` (still `new_default`).

Coverage: Massive `?date=` has CIKs from 2003-09-11, and upstream uses retroactive symbols
for some delistings, so early intervals and recent delistings can stay unresolved.

Tickers are upstream's symbols. For some delisted companies that is the last symbol they
traded under rather than the one in use at the time (for example `AAMRQ`, AMR Corp's
post-bankruptcy symbol, appears from 1996), and share classes use a dot (`BRK.B`, `BF.B`).
Check the symbols against your price vendor before joining.

## Refreshing

`data_universe.membership.status("sp500")` and `membership_grid()` warn once
`verified_through` is more than `membership_max_staleness_days` (default 45) days old, and
`membership_grid(strict=True)` raises. To refresh, from a repo checkout:

```bash
python -m data_universe.membership.refresh --dry-run   # what would change
python -m data_universe.membership.refresh             # upstream's latest commit
python -m data_universe.membership.refresh --ref SHA   # a specific upstream commit
```

The tool downloads the upstream file at a pinned commit and checks it: the header must be
exactly `date,tickers`, dates must be valid and strictly ascending, there must be no empty
tickers, and the latest snapshot must list 480 to 520 members. It derives the intervals and
keeps every existing `security_map`/`security_master` row whose interval still exists, so
reviewed mappings survive. New intervals get `new_default` ids. Rows whose interval disappeared
upstream are dropped, and a dropped reviewed row is reported as a warning. The tables are
validated in a staging directory (unmapped or overlapping intervals fail the run) before
`ticker_intervals.csv`, `security_map.csv`, `security_master.csv` and `manifest.json` are
replaced. Afterwards, review `git diff`, run `pytest`, bump the package version and commit.

To audit the committed tables, rebuild them at the manifest's `upstream_commit_sha` into a
scratch directory and compare `ticker_intervals.csv` byte for byte. For the identity tables,
`python -m data_universe.membership.identity --offline` rebuilds purely from the committed
evidence files and must leave `git status` clean -- that is the committed resolution's own
audit.

```bash
DATA_UNIVERSE_MEMBERSHIP_TABLES_DIR=/tmp/sp500-audit \
    python -m data_universe.membership.refresh --ref <upstream_commit_sha>
for f in ticker_intervals.csv security_map.csv security_master.csv; do
    cmp /tmp/sp500-audit/sp500/$f data_universe/membership/tables/sp500/$f
done
```
