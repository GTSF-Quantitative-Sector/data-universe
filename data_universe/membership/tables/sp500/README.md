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
| `security_map.csv` | `index_id, ticker, start_date, security_id, evidence, reviewed`: the security each interval belongs to. |
| `security_master.csv` | One row per `security_id`: `name, latest_ticker, cik, composite_figi, share_class_figi, notes` (only `latest_ticker` is filled in today). |
| `ticker_history.csv`, `tail_changes.csv` | Reserved for identity resolution and for changes newer than upstream. Empty, and never written by the refresh tool. |
| `manifest.json` | Provenance: `upstream_repo`, `upstream_commit_sha`, `upstream_commit_date`, `upstream_filename`, `upstream_sha256`, `upstream_last_snapshot_date`, row counts, `retrieved_at`, `tool_version`, and `verified_through` (the UTC date the tables were last confirmed against upstream's latest commit). |

## Provisional identity

Identity resolution has not been done yet. Every interval maps to its own default
`<ticker>-<YYYYMMDD>` security_id (`evidence=new_default`, `reviewed=False`), so a renamed
company (FB, then META), a company that left and rejoined (AMD), and two companies that used
the same ticker years apart (AAL) all appear as separate securities. These ids will change
once reviewed mappings land; don't persist them. For panels keyed by ticker, use
`by="ticker"` in `membership_grid`, `mask_like` and `members_on`.

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
reviewed mappings survive. New intervals get provisional ids. Rows whose interval disappeared
upstream are dropped, and a dropped reviewed row is reported as a warning. The tables are
validated in a staging directory (unmapped or overlapping intervals fail the run) before
`ticker_intervals.csv`, `security_map.csv`, `security_master.csv` and `manifest.json` are
replaced. Afterwards, review `git diff`, run `pytest`, bump the package version and commit.

To audit the committed tables, rebuild them at the manifest's `upstream_commit_sha` into a
scratch directory and compare. No output from `cmp` means a file matches byte for byte.
`ticker_intervals.csv` must always match. `security_map.csv` and `security_master.csv` match
only while every mapping is still the provisional default, because a scratch rebuild has no
reviewed rows to keep.

```bash
DATA_UNIVERSE_MEMBERSHIP_TABLES_DIR=/tmp/sp500-audit \
    python -m data_universe.membership.refresh --ref <upstream_commit_sha>
for f in ticker_intervals.csv security_map.csv security_master.csv; do
    cmp /tmp/sp500-audit/sp500/$f data_universe/membership/tables/sp500/$f
done
```
