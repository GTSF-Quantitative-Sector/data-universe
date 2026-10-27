# Roadmap

A prioritized plan for improving the data `data_universe` fetches, driven by the PEAD
research (earnings announcements + options) and the GNN ranker (a large, point-in-time
S&P 500 universe, 2015 to now). Phase 1 shipped as v0.6.0 — see
`docs/design/2026-10-27-v0.6.0-data-correctness-and-hardening.md` for its design and
behavior changes.

## Phases

| Phase | Theme | Main consumers | Size | Depends on |
|---|---|---|---|---|
| 0 | Merge v0.5.0 (`ethan` → `main`); consider merging Mark's `RESEARCH_CONTEXT.md` | all | S (user) | none |
| **1** | **Correctness + hardening — shipped as v0.6.0** | PEAD, GNN | L | 0 |
| 2 | PEAD event table v2 (additive columns) | PEAD | M | 1 + probe |
| 3 | Durable raw store + bulk ingest + ingest validation | GNN, PEAD | XL | 1 + probe |
| 4 | Identity resolution (`membership/identity.py`) | GNN, PEAD | L | 1 + probe (3 helps) |
| 5 | Precompute v2 (panels, versioning, tombstones) | all | M | 3 |
| 6 | GNN datasets (return panels, open-to-open labels, correlation edges) | GNN | L | 3, 4, 5 |
| 7 | New datasets: XBRL SUE, earnings calendar/estimates, news, short interest | PEAD, GNN | M each | 1 + probe; parallel with 3–6 |
| 8 | Later: VRP IV30, Beta intraday/Kalman | VRP, Beta | L | 3 |

### Phase 1 — done (v0.6.0)

Entitlement/shape probe (`python -m data_universe.probe`); true total-return bars plus
raw bars (`sources/adjust.py`); the PEAD raw-strike fix; option chain filters and the
expired/active union; `MassiveOptionsSource`; retries + `EntitlementError`; EDGAR
throttle/retries/User-Agent. **Gated:** don't merge to `main` until the probe confirms
`dividend_cash_basis` (`OPEN_QUESTIONS.md` item 20).

### Phase 2 — PEAD event table v2

Keep every existing column (including `day0` semantics) and append:

- `acceptance_time_et` and `announce_timing` ∈ {`pre_open` (<09:30), `market_hours`,
  `post_close` (≥16:00)}.
- `pre_announcement_session` (last session whose close precedes the news) and
  `reaction_session` (first session whose close reflects it); weekend/holiday filings
  map to the surrounding sessions.
- The day-0 straddle values currently computed and dropped: `strike`, `expiry`,
  `call_ticker`, `put_ticker`, `call_price`, `put_price`, `straddle_price`, `spot_raw`.
- A second straddle priced at `pre_announcement_session` (`pre_*` columns) and
  `implied_move = pre_straddle_price / pre_spot_raw`.
- `ret_reaction` (total-return close at `reaction_session` / `pre_announcement_session`
  − 1) and `surprise_ratio = |ret_reaction| / implied_move`.
- As-of CIK lookup for renamed/delisted tickers (Massive ticker details as of a date,
  or ticker events — whichever the probe confirms), with a `cik_source` column.
- `build_pead_panel(index_id="sp500", start, end)` with per-ticker failure isolation
  and point-in-time membership (`by="ticker"` until Phase 4).

### Phase 3 — durable raw store + bulk ingest

- `data_universe/store/` with partitioned parquet under `<data_dir>/raw/`: split-adjusted
  and raw daily bars, dividends, splits, option contracts, option daily aggregates.
- A coverage manifest per (dataset, key): range, `fetched_at`, source, status
  (`ok`/`empty`/`not_entitled`), tool version.
- `StoreBackedSource(DataSource)` fetches only missing ranges, stores completed sessions
  only (never a partial "today"). Total-return anchoring moves to the store's last
  date; only ratios matter, so consumers see no change.
- Flat-file ingest if the probe shows entitlement: ~2,950 files per dataset for
  2015-now instead of millions of REST calls.
- Ingest validation flags (never fixes): OHLC sanity, duplicate stamps, gaps vs the SPY
  calendar, jumps with no split record, stale runs.
- Acceptance: precompute over 500 tickers × all features makes O(tickers) upstream
  calls; a rerun makes zero.

### Phase 4 — identity resolution

- Auto-link membership intervals by CIK and composite FIGI (Massive details/events +
  SEC `tickers`/`formerNames`): same CIK + touching intervals → one security (FB→META);
  same CIK with a gap → leave-and-rejoin (AMD); different CIK under one ticker →
  different securities (AAL).
- Fill `security_master`/`ticker_history` (a (security, date) → vendor-symbol map —
  fixes upstream-symbol cases like AAMRQ).
- Only conflicts go to a human review queue; `reviewed=True` rows already survive a
  refresh. Then drop the "provisional" security_id warnings.

### Phase 5 — precompute v2

Multi-column panel outputs (edge lists, event tables); version metadata (code + input
dataset + parameter hash); skip completed work; explicit invalidation; NaN tombstones so
a recomputed NaN deletes a stale value; per-feature kwargs.

### Phase 6 — GNN datasets

Point-in-time total-return panels (date × security, NaN outside membership);
next-open-to-open forward labels with `label_end` (parity with xsec-gnn-ranker);
close-to-close forward labels at 1/5/21/63 sessions; `momentum_20/60`, unannualized
`vol_20`; gap-aware 60-session rolling correlations + shared-observation counts saved
as sparse top-N edges per snapshot; SIC codes as a stand-in for historical GICS.

### Phase 7 — new datasets

- **7a XBRL `companyfacts` (free):** first-reported quarterly diluted EPS keyed by the
  earliest `filed` date (restatements can't leak in); SUE =
  (EPS_q − EPS_{q−4}) / std of last 8 differences.
- **7b Benzinga earnings/estimates (if entitled):** announcement date/time, consensus
  and actual EPS — a surprise measure and an 8-K timing cross-check.
- **7c Massive news:** daily article counts per ticker keyed by `published_utc`.
- **7d Short interest / short volume (if entitled):** date each value by
  **publication**, not settlement.

### Phase 8 — later

IV30 (raw spot + dividends, maturity-matched rates, options store, American-exercise
policy, quality flags); minute bars for strictly regular-hours hourly bars; Kalman
states with checkpoints.

## Immediate next steps (user actions)

1. Merge v0.5.0 to `main`.
2. Run `python -m data_universe.probe` with real keys and share the JSON — it resolves
   `OPEN_QUESTIONS.md` items 4, 5, 10, 15, 20 and 22 and scopes Phases 2–4 and 7.
   If `dividend_cash_basis` comes back `split_adjusted`, flip
   `adjust.DIVIDEND_CASH_BASIS` (the fakes follow automatically).
3. **Don't merge v0.6.0 until the dividend basis is confirmed** — a wrong basis is a
   silent error.

## Risks

- Dividend cash basis is unverified — mitigated by one constant plus the probe gate.
- Each bars window costs 3 cached requests until Phase 3.
- Adjusted levels depend on the window by design; every current feature is
  scale-invariant or uses raw prices.
- Every return-based value shifts slightly vs v0.5.0 — recompute precomputed outputs.
- Non-USD dividends raise on purpose (rare in the S&P 500; loud beats silently wrong).
- Unverified vendor semantics (the `expired` flag, as-of details, ticker events) are
  confined to the probe, not hard dependencies.
