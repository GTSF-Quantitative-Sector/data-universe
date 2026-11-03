# Roadmap

A prioritized plan for improving the data `data_universe` fetches, driven by the PEAD
research (earnings announcements + options) and the GNN ranker (a large, point-in-time
S&P 500 universe, 2015 to now). Phase 1 shipped as v0.6.0 — see
`docs/design/2026-10-27-v0.6.0-data-correctness-and-hardening.md` for its design and
behavior changes. Phase 2 shipped as v0.7.0. Phase 7a shipped as v0.8.0.

## Phases

| Phase | Theme | Main consumers | Size | Depends on |
|---|---|---|---|---|
| 0 | Merge v0.5.0 (`ethan` → `main`); consider merging Mark's `RESEARCH_CONTEXT.md` | all | S (user) | none |
| **1** | **Correctness + hardening — shipped as v0.6.0** | PEAD, GNN | L | 0 |
| **2** | **PEAD event table v2 — shipped as v0.7.0** | PEAD | M | 1 (+ probe for the as-of CIK resolver) |
| 3 | (3a shipped as v0.9.0)  Durable raw store + bulk ingest + ingest validation | GNN, PEAD | XL | 1 + probe |
| 4 | Identity resolution (`membership/identity.py`) | GNN, PEAD | L | 1 + probe (3 helps) |
| 5 | Precompute v2 (panels, versioning, tombstones) | all | M | 3 |
| 6 | GNN datasets (return panels, open-to-open labels, correlation edges) | GNN | L | 3, 4, 5 |
| 7 | New datasets: XBRL SUE, earnings calendar/estimates, news, short interest (7a shipped as v0.8.0) | PEAD, GNN | M each | 1 + probe; parallel with 3–6 |
| 8 | Later: VRP IV30, Beta intraday/Kalman | VRP, Beta | L | 3 |

### Phase 1 — done (v0.6.0)

Entitlement/shape probe (`python -m data_universe.probe`); true total-return bars plus
raw bars (`sources/adjust.py`); the PEAD raw-strike fix; option chain filters and the
expired/active union (replaced in v0.9.1 by one `expired=false` query); `MassiveOptionsSource`; retries + `EntitlementError`; EDGAR
throttle/retries/User-Agent. **Gate satisfied:** the 2026-11-03 probe confirmed
`dividend_cash_basis = "raw"` (`OPEN_QUESTIONS.md` item 20).

### Phase 2 — done (v0.7.0)

Every v1 column kept (including `day0` semantics), with the v2 columns appended:
`acceptance_time_et`/`announce_timing` (clock-based `pre_open`/`market_hours`/
`post_close`), `pre_announcement_session`/`reaction_session` (weekend/holiday filings
map to the surrounding sessions), the exported day-0 straddle (`strike`…`spot_raw`),
the pre-announcement straddle (`pre_*`, same expiry rule — nearest expiry strictly
after `day0`), `implied_move`, `ret_reaction`, `surprise_ratio`, and
`cik`/`cik_source`. New columns can only go missing, never skip a row.
`build_pead_panel(start, end, index_id="sp500")` adds point-in-time membership at the
close of `pre_announcement_session` (ticker-keyed until Phase 4), per-ticker failure
isolation, a `tickers` subset, `cik_overrides`, and the `cik_resolver` hook.

**Not shipped:** the as-of CIK lookup itself — the `cik_resolver` hook is in place but
unfilled until the probe's `fb_as_of_cik_is_meta`/ticker-events check confirms the
endpoint (`OPEN_QUESTIONS.md` item 24); panel CIKs default to SEC's current map
(`cik_source="sec_current"`). The 2026-11-03 probe answered item 24 (ticker details
`?date=`), so the resolver is now the next code step.

### Phase 3 — durable raw store + bulk ingest

- **3a shipped as v0.9.0**: `data_universe/store/` — parquet + JSON manifest per
  (dataset, key) under `<data_dir>/raw/` for split-adjusted and raw daily bars,
  index daily bars and dividends (splits/options datasets deferred), each fetch
  recorded with range, `fetched_at`, source, status (`ok`/`empty`/`not_entitled`),
  tool version. `StoreBackedSource(DataSource)` fetches only missing ranges (hull,
  one call), stores completed sessions only (complete from 06:00 ET the next day),
  rebases on splits, and passes the still-open tail straight through. Total-return
  anchoring stays window-end, so outputs are identical.
- Flat-file ingest if the probe shows entitlement: ~2,950 files per dataset for
  2015-now instead of millions of REST calls.
- Validation flags shipped in v0.9.0 (3a) — never fixes: OHLC sanity, duplicate stamps, gaps vs the SPY
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

- **7a XBRL `companyfacts` (free) — shipped as v0.8.0** (`data_universe/fundamentals/sue.py`
  via `companyconcept`, plus `include_sue` on `build_pead_events`/`build_pead_panel`):
  first-reported quarterly diluted EPS keyed by the earliest `filed` date
  (restatements can't leak in); SUE = (EPS_q − EPS_{q−4}) / std of last 8 differences.
  The probe's `aapl_eps_concept` check confirmed the first-reported EPS assumption
  on 2026-11-03.
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

The probe ran 2026-11-03 against the Massive host (report:
`data/probe/probe_20261103T192344Z.json`, gitignored). Headline findings:

- `dividend_cash_basis = "raw"` — matches `adjust.DIVIDEND_CASH_BASIS`; the
  v0.6.0+ merge gate is **satisfied**.
- `expired` on options contracts is relative to `as_of` (the empty-chain bug;
  fixed in v0.9.1); strikes are raw; option open-close pricing works on 2019
  contracts.
- `adjusted=true` is split-only; hourly bars reach 2015 and minute bars work.
- Aggregates are keyed by the symbol in use at the time (FB has 2020 bars,
  META none); ticker details `?date=` gives the as-of CIK (Meta) — the
  `cik_resolver` endpoint.
- `I:VIX` is **not entitled** on this plan; SEC checks all pass including the
  companyconcept first-reported finding; flat files need S3 creds and FRED
  needs a key (both skipped).

1. Merge v0.6.0+ (`v0.6.0` through `v0.9.1`) to `main`.
2. ~~Implement a `cik_resolver` for `build_pead_panel`~~ — **shipped in v0.10.0**
   (`massive_cik_resolver` over `/v3/reference/tickers/{ticker}?date=`, the
   default when the panel constructs Massive sources). Predecessor-CIK gaps
   remain — see `OPEN_QUESTIONS.md` item 34.
3. Optional re-probe: set a FRED key and/or S3 flat-file credentials +
   `pip install -e ".[s3]"` to cover the skipped groups.
4. Set a real SEC User-Agent contact (the placeholder is polite but generic).

## Risks

- Dividend cash basis confirmed `raw` by the 2026-11-03 probe; a new vendor would need the same check.
- Each bars window costs 3 cached requests unless a `StoreBackedSource` (v0.9.0)
  wraps the source — then only the still-open tail and first-time ranges hit upstream.
- Adjusted levels depend on the window by design; every current feature is
  scale-invariant or uses raw prices.
- Every return-based value shifts slightly vs v0.5.0 — recompute precomputed outputs.
- Non-USD dividends raise on purpose (rare in the S&P 500; loud beats silently wrong).
- Unverified vendor semantics (the `expired` flag, as-of details, ticker events) are
  confined to the probe, not hard dependencies.
- SUE split factors depend on symbol continuity (bars are fetched by the current
  symbol) until Phase 4's (security, date) → vendor-symbol map; `OPEN_QUESTIONS.md`
  item 27.
