# Contracts

This library defines its own frame/schema conventions where the original prompt didn't pin
one down — most notably the PEAD event table and the precompute output format. Other project
teams (vrp-research in particular, per `PLAN.md` §9 item 8) should reconcile against these
schemas rather than assume a different one; if a schema here doesn't match what your project
already uses, raise it rather than silently diverging.

## Bar frames

Source: `data_universe/sources/base.py`.

- Index: tz-aware `America/New_York` `DatetimeIndex`, name `timestamp`.
- Columns: `open, high, low, close, volume, vwap`, all `float64`.
- Daily bars are stamped at midnight of the session date.
- Intraday bars are stamped at the **start** of their window (the vendor convention for
  Polygon/Massive aggregates) and filtered to regular trading hours by window overlap:
  `filter_regular_hours` keeps a bar iff `[stamp, stamp + bar_minutes)` intersects
  9:30-16:00 ET. `hourly_bars` therefore returns stamps 09:00–15:00 — and the 09:00 bar
  includes 9:00-9:30 pre-market trades (an accepted convention, not a bug).
- **Three bar flavors per window** (`PolygonSource`/`MassiveSource`; see
  `sources/adjust.py`):
  - `daily_bars` / `hourly_bars` — **total-return** adjusted: splits *and* cash
    dividends. `f_e = 1 − D_e / C(prev)` per ex-date, applied backward so the last bar
    equals the split-adjusted price. Adjusted levels are therefore anchored at the
    window's last bar and depend on the requested range — compare levels only within
    one call; returns and other ratios are window-independent. Spin-offs and stock
    dividends are not adjusted.
  - `daily_bars_split_adjusted` / `hourly_bars_split_adjusted` — splits only (the
    vendor's `adjusted=true`), the v0.5.0 series.
  - `daily_bars_raw` — unadjusted prices and share volume as actually traded. Use this
    wherever a real price level matters: option strikes, price filters, dollar volume.
    A source that doesn't implement it raises `NotImplementedError` when asked.
- Volume is split-adjusted share counts in adjusted frames and raw share counts in raw
  frames (raw volume is *higher* before a split).
- `dividends(ticker, start, end)` returns a frame with columns
  `ex_dividend_date` (tz-naive), `cash_amount` (float), `currency`, `dividend_type`,
  one row per dividend record with an ex-date in `[start, end]`, sorted by ex-date.
- HTTP 401/403 from a vendor raises `EntitlementError` (a `DataSourceError`): bad key
  or a dataset the plan doesn't include — distinct from an empty frame. Transient
  failures (429/5xx/connection/timeout) are retried with doubling backoff; a persistent
  429 raises `RateLimitedError`, a persistent 5xx/transport failure `DataSourceError`.

## Feature / label series

Source: `data_universe/features/registry.py` and `data_universe/labels/registry.py`'s
shared `compute()`.

- Return type: `pd.Series`, indexed by a tz-naive `date` index, sliced to exactly
  `[start, end]` inclusive of the caller's requested range.
- `compute()` pads the underlying data fetch backward by the feature/label's declared
  `warmup_trading_days` before calling the registered function, then slices the result back
  — so the first returned value is never `NaN` purely for lack of history. It can still be
  `NaN` for a real data gap (a source outage, a delisted ticker), which is a different
  situation and is not hidden.
- Every **label** name starts with `fwd_`, enforced at `@register` decoration time
  (`labels/registry.py`). Every **feature** name is guaranteed *not* to start with `fwd_`,
  enforced the same way (`features/registry.py`). A `fwd_`-prefixed label is forward-looking
  by construction and must never be used as a model input feature.
- A label's forward-looking rows near the end of any requested `[start, end]` range are
  genuinely `NaN` — `compute()` never fetches data past the requested `end` — this is a
  documented limitation, not a bug.
- **Risk-free rate (`rate_source`)**: `ret_excess_1d` requires `compute(...,
  rate_source=...)` (e.g. `FredSource()`, `FakeRateSource()`) and raises `ValueError`
  without one. `beta_ols_60d/90d` and `alpha_ols_60d/90d` accept an optional
  `rate_source`: when given, both the asset's and the market's log returns are shifted by
  `log1p(daily_rf)` before the rolling OLS (excess-return regression); when omitted the
  regression runs on raw returns. `precompute(..., **kwargs)` forwards `rate_source` (and
  any other `compute()` kwarg) to every pair.

## PEAD event table

Source: `data_universe/events/earnings.py`'s `build_pead_events(ticker, edgar_source,
price_source, options_source, cik=None, calendar_start="2015-01-01",
calendar_end=None, cik_source=None) -> pd.DataFrame`.

One row per qualifying 8-K Item 2.02 filing, sorted ascending by `event_date`. Columns,
exactly, in this order — the 13 v1 columns are unchanged in name, order, values and
skip rules; the v0.7.0 (v2) columns are appended after them and can never cause a row
to be skipped — anything they can't compute is missing (NaN / NaT / None / `<NA>`):

| Column | Meaning |
|---|---|
| `ticker` | Underlying ticker symbol. |
| `event_date` | The 8-K's SEC `acceptanceDateTime` (ISO UTC) converted to America/New_York, then normalized to a calendar date — an evening filing on ET date D lands on D, not D+1. |
| `pre_session` | The trading session immediately before `day0`. |
| `day0` | `event_date` itself if it's a trading session, else the next one. |
| `day1` | The trading session immediately after `day0`. |
| `sigma_pre_daily` | Daily (not annualized) stdev of the 30 log returns (31 closes) ending at `pre_session`, computed on **total-return** closes. |
| `adv_usd_pre20` | 20-trading-day average dollar volume ending at `pre_session`, **raw** close × raw volume — dollars actually traded. |
| `adv_usd_day0_day1` | Average dollar volume over `day0`/`day1` only (raw basis) — no other post-entry window is computed anywhere in this library. |
| `days_to_expiry` | Calendar days from `day0` to the nearest option expiry strictly after `day0`. The "nearest expiry after" convention is kept as originally specified; any bias this introduces is left for the project team to adjust, not fixed here. |
| `next_earnings_session` | **Realized**, not forecast: the following row's `event_date`, filled in only after sorting. `NaT` for the most recent filing. Never use this as a model input for `day0` itself — it's hindsight. |
| `call_volume` / `put_volume` | Contract volume traded on `day0` for the chosen ATM straddle's call/put leg. |
| `is_stale` | `True` whenever either leg of the straddle traded zero volume on `day0` (a stale daily close) — including an untraded contract, which `PolygonOptionsSource.daily_close` reports as `{"close": NaN, "volume": 0.0}` rather than raising on its HTTP 404. |
| `acceptance_time_et` | The 8-K's `acceptanceDateTime` as a tz-aware `America/New_York` Timestamp (tz-aware input is `tz_convert`ed; a naive value is localized as ET wall time). Note: the 8-K acceptance time is only a proxy for the actual announcement time. |
| `announce_timing` | Clock-based bucket of `acceptance_time_et`: `"pre_open"` before 09:30, `"market_hours"` 09:30 up to (not including) 16:00, `"post_close"` 16:00 or later — including on weekend and holiday dates, which classify the same way. |
| `pre_announcement_session` | The last session whose close precedes the news: `day0` itself for a post-close filing on a session day, else `pre_session`. |
| `reaction_session` | The first session whose close reflects the news: `day1` for a post-close filing on a session day, else `day0`. Weekend/holiday filings therefore map to the surrounding sessions. |
| `strike` / `expiry` | The day-0 straddle's strike and (tz-naive) expiry — the values v1 already computed, now exported. |
| `call_ticker` / `put_ticker` | OCC tickers of the day-0 straddle's legs. |
| `call_price` / `put_price` | Leg closes on `day0`; NaN when the contract didn't trade. |
| `straddle_price` | `call_price + put_price`; NaN propagates. |
| `spot_raw` | The raw day-0 close the ATM strike was chosen against (same value v1 uses as `spot`). |
| `pre_strike` / `pre_expiry` / `pre_call_ticker` / `pre_put_ticker` / `pre_call_price` / `pre_put_price` / `pre_straddle_price` / `pre_spot_raw` | The same fields for a second straddle priced at the close of `pre_announcement_session` on the chain as of that day: same expiry rule as day0 (nearest expiry strictly after `day0`), strike band around `pre_spot_raw` (the raw close there). For a post-close filing on a session day these copy the day-0 values with no extra requests. All missing when `pre_spot_raw` is NaN or ≤ 0, or the chain/straddle can't be priced (the ValueError/KeyError cases v1 already skips on for day0). |
| `pre_days_to_expiry` | Calendar days from `pre_announcement_session` to `pre_expiry`, as a float. |
| `pre_call_volume` / `pre_put_volume` | Pre-leg volumes on `pre_announcement_session`. |
| `pre_is_stale` | Nullable `boolean` — True when either pre leg traded zero volume, `<NA>` when there is no pre straddle. |
| `implied_move` | `pre_straddle_price / pre_spot_raw`; NaN when either is missing. |
| `ret_reaction` | Total-return close-to-close return from `pre_announcement_session` to `reaction_session`, from the same `daily_bars` window the event already fetched; NaN when either bar is missing. |
| `surprise_ratio` | `abs(ret_reaction) / implied_move`, only when both are finite and `implied_move > 0`; else NaN. |
| `cik` | The filing CIK used for this event, normalized to a digit string via `str(int(cik))`. |
| `cik_source` | Where `cik` came from: `"explicit"` (a `cik=` argument or `cik_overrides` — a custom `cik_source=` label is recorded instead when given), `"massive_as_of"` (Massive ticker details `?date=` via `as_of_cik`), `"sec_current"` (SEC's current ticker→CIK map via `edgar_source.get_cik`), or a `cik_resolver` label. Passing `cik_source` without `cik` raises `ValueError`. |

The ATM straddle is chosen with `spot` = the **raw** day-0 close — historical option
strikes are on the raw (as-traded) basis, so a split-adjusted spot would pick the wrong
strike for any event preceding a later split. The chain query itself is filtered
server-side: expirations in `(day0, day0 + 40]` days and strikes within ±20% of spot
(`PolygonOptionsSource.contracts_as_of`'s `expiration_*`/`strike_*` params), as a
single `expired=false` query (the flag is relative to `as_of` — confirmed 2026-11-03). The pre-announcement
straddle runs the same selection through one shared helper, just priced and
chain-fetched at `pre_announcement_session` with its strike band around
`pre_spot_raw`.

A filing too close to either edge of the available calendar/price history to compute (not
enough trading-session lookback, no expiry found, no matching call/put in the chain, or a
ticker bar missing on `day0`/`day1` in either the adjusted or the raw frame — e.g. a halt
while the market stayed open) is
**silently skipped** — it does not appear as a null row, and it does not raise.
Those skip rules are unchanged from v1; the v2 columns only ever go missing,
they never add a new reason to drop an event.

Issue 3 from the original spec review ("weekly ranking lookahead") is out of scope: this
table has no ranking column at all.

### PEAD panel

Source: `data_universe/events/earnings.py`'s `build_pead_panel(start, end=None,
index_id="sp500", *, edgar_source=None, price_source=None, options_source=None,
tickers=None, cik_overrides=None, cik_resolver=None, strict=False) -> PeadPanelReport`.

- `end` defaults to today; `start > end` raises `ValueError`. `edgar_source` defaults
  to `EdgarSource()`; whichever of `price_source`/`options_source` is omitted shares
  one `MassiveOptionsSource()`.
- **Membership rule**: an event is kept iff `start <= event_date <= end` AND the
  ticker was an index member at the close of its `pre_announcement_session` under one
  of the intervals in its resolved-CIK group — `start_date <=
  pre_announcement_session < end_date_excl`, a NaN end meaning an open interval.
  Intervals come from `membership.store.load_ticker_intervals(index_id)` — keyed by
  ticker until Phase 4's identity resolution; an unknown `index_id` raises
  `MembershipError`, and the manifest's staleness check (`strict=`) applies first.
- **CIK resolution**, per candidate interval with `as_of` = `max(interval
  start_date, last session before `start`)`: `cik_overrides[ticker]` (label
  `"explicit"`) → `cik_resolver(ticker, as_of)` if given and non-None —
  defaulting to `massive_cik_resolver` on the shared `MassiveOptionsSource`
  when the price/options sources defaulted (label `"massive_as_of"`,
  `MassiveSource.as_of_cik` → `ticker_details(ticker, date=as_of)["cik"]`;
  404 or a missing `cik` falls through) — →
  `edgar_source.get_cik(ticker)` (label `"sec_current"`). Each ticker's intervals
  group by (normalized CIK, label); each group gets one `build_pead_events` call and
  keeps its events only inside that group's own intervals.
- `PeadPanelReport.events` has exactly `EVENT_COLUMNS`, sorted by (event_date,
  ticker) — an empty frame with those columns when nothing qualifies. `succeeded`
  maps ticker → events kept (0 when none); `failed` maps ticker →
  `"ExcType: message"`. One ticker's failure is caught and recorded, never stopping
  the panel; a `tickers=`-requested ticker with no candidate interval lands in
  `failed`, non-fatally.
- Request cost: each event still costs its own bars window — 3 cached requests
  (split-adjusted bars + raw bars + dividends, the total-return composite's
  primitives; the SPY proxy fetch is shared through the cache) — plus up to 2 chain
  queries and 4 contract closes per event. Each candidate ticker's events are
  computed over the whole `[start − 90 days, end]` calendar and only then filtered
  to membership, so a ticker that was a member for only part of the range still
  pays for its non-member events.

**`include_sue` (v0.8.0):** `build_pead_events(..., include_sue=True)` appends
`eps_period_end, eps_filed, eps_diluted, eps_derived, eps_yoy_diff, sue_std,
sue_n, sue` (datetime64, datetime64, float64, nullable `boolean`, float64 ×2,
nullable `Int64`, float64). Each event matches the `quarterly_sue` quarter
with the latest `period_end` strictly before `event_date`, within 100 days,
and first `filed` no earlier than 7 days before the event (the 10-Q/10-K
usually lands days to weeks after the 8-K) — the filed rule stops a
mid-quarter Item 2.02 from matching the previous quarter whose 10-Q is
months old. Unmatched events get NaT/NaN/`<NA>`. `build_pead_panel`'s
`include_sue` runs `quarterly_sue` once per (CIK, label) group with kept
events; a group's SUE failure lands in `PeadPanelReport.sue_failed`
(ticker → "ExcType: message", first failure wins) and the events are kept
with SUE missing. Cost: 2 SEC `companyconcept` requests + 2 bars requests
per CIK group (plus the per-event costs above). The default stays
`include_sue=False` — identical outputs and request counts to v0.7.0.

## Quarterly EPS / SUE table

Source: `data_universe/fundamentals/sue.py`'s `quarterly_sue(ticker, cik,
edgar_source, price_source, *, history_start=None, window=8, min_obs=6)
-> pd.DataFrame`.

One row per fiscal quarter, sorted by `period_end` — built from SEC XBRL
`companyconcept` facts (us-gaap `EarningsPerShareDiluted`, then
`EarningsPerShareBasicAndDiluted`, unit `USD/shares`).

| Column | Meaning |
|---|---|
| `cik` | The reporting CIK, digit string. |
| `period_start` / `period_end` | The EPS period, from the fact's `start`/`end` (never `fy`/`fp`, which describe the filing). |
| `filed` | The earliest `filed` date among filings disclosing this exact (start, end) period — the first-reported date. Restatements, amendments and later comparatives are filed later and drop out, so restated values can never leak in. |
| `accn` / `form` / `tag` | Accession, form and tag of that first-reporting filing (of the annual filing, for a derived Q4). |
| `eps_diluted` | Diluted EPS on the quarter's own as-filed share basis (float64). |
| `derived` | `True` for a Q4 computed as `FY − (Q1+Q2+Q3)` — approximate (EPS isn't additive). |
| `eps_yoy_diff` | `eps_q − eps_q−4` computed on today's share basis, then rescaled onto the quarter's own basis (float64, NaN when there's no lag-4 quarter). |
| `sue_std` | Sample std (ddof=1) of the `window` most recent prior diffs visible on `filed`, on the quarter's own basis; NaN when fewer than `min_obs` diffs. |
| `sue_n` | The number of prior diffs used (int64). |
| `sue` | `eps_yoy_diff / sue_std` on the adjusted basis — the standardized unexpected earnings; NaN unless the std is finite and above the noise floor (a std ≤ 1e-9 × max(1, max|prior diff|) is float noise, reported as `sue_std` = 0.0). |

**First-reported rule:** for each exact (start, end) period the earliest
`filed` wins (ties: diluted tag, then `accn`); periods are identified by
start/end, never `fy`/`fp`.

**Split-basis rule:** internal math uses `eps_adj = val / factor(filed)` where
`factor` is the raw ÷ split-adjusted close at the filing date (the product of
split ratios effective strictly after it — a split on or before the filing
date counts as already reflected, matching ASC 260 retroactive restatement).
Outputs are multiplied back onto each quarter's own as-filed basis, so stored
values never change when a later split happens; `sue` is basis-free.

**Q4 derivation:** when no direct quarter ends within 7 days of a fiscal
year's end, `FY − (Q1+Q2+Q3)` fills it — only if exactly three *deduplicated*
direct quarters tile the year (the ±7-day end clustering runs on direct
quarters first, so a same-period comparative with a shifted start can't
count as a fourth part; |Q1.start − FY.start| ≤ 7d, consecutive gaps of
1–7 days, and the leftover Q4 period is 65–125 days). `derived=True`,
`filed` = the latest of the four filings.

**SUE / point-in-time view:** SUE_q uses only V_q = {p : filed_p ≤ filed_q} —
exactly what was computable the day EPS_q was first filed. The numerator is
q's own diff; the denominator is the std (ddof=1) of the `window` most
recent earlier diffs inside ~`window` quarters of lookback, when at least
`min_obs` exist.

## Raw store

Source: `data_universe/store/raw.py` (`RawStore`) and
`data_universe/store/source.py` (`StoreBackedSource`), opt-in in v0.9.0 —
defaults unchanged.

**Layout:** under `root` (default `<config.get_data_dir()>/raw`),
`<dataset>/<quote(key, safe="")>.parquet` (e.g. `I%3AVIX.parquet`), the same
stem + `.manifest.json`, and a `.<quoted>.lock` flock target. Datasets:
`daily_bars_split_adjusted`, `daily_bars_raw`, `index_daily_bars`,
`dividends`.

**Manifest:** `{"dataset", "key", "schema_version", "fetches": [record]}` with
`record = {"start", "end", "status" ("ok"|"empty"|"not_entitled"), "rows",
"fetched_at" (UTC ISO), "source" (upstream class), "family", "tool_version",
"rebased"}`. `coverage` is the merged union of ok/empty records (intervals
merge when overlapping or ≤1 day apart); a `schema_version` mismatch reads
as absent everywhere and the next write discards the old rows and records.

**Coverage semantics:** a requested window splits at `L = _last_complete_date()`
(a session counts as complete from 06:00 ET the next calendar day). The part
≤ L is served from the store — gaps inside coverage are filled by fetching
their hull `[a, b]` in ONE upstream call — and the part > L always goes to
upstream normally and is never persisted. `not_entitled` is recorded (not
covered) and re-raised; empty results are covered so they aren't refetched.

**Basis datasets** (`daily_bars_split_adjusted`): if a fetch wouldn't overlap
a stored row, it extends to the nearest stored row, and if an overlap close
differs >1e-6 relative (a split rebased the vendor's history) the whole
coverage is refetched once with `rebased=True`. When a window spans both
parts, the live fetch starts at the stored slice's last row (not L + 1) for
the same check; live rows are always cut to row dates >= max(start, L + 1),
which drops the overlap row before concatenation. Entirely-live windows
(after L) fetch at `start` and never touch the store.

**Dividends:** stored only when `adjust.DIVIDEND_CASH_BASIS == "raw"` — other
bases pass straight through (amounts would be basis-dependent; item 30).

**Family guard:** every access checks the manifest's `family` against the
upstream's (`store_family` attr, else its class name) — a foreign store
raises `DataSourceError` rather than serve another vendor's data.

**Parity:** the same window returns the same frames as the upstream (parquet
index `freq` isn't part of the contract); `daily_bars` still composes
total-return window-end-anchored through `adjust.TotalReturnMixin`.

### Validation flags

Source: `data_universe/store/validate.py` — on-demand flags, never written
back and never used to alter data. `validate_bars(raw, split_adjusted,
sessions)` returns a bool `FLAG_COLUMNS` frame indexed by tz-naive `date`,
only flagged dates:

- `ohlc_bad` — non-finite/non-positive OHLC, high < max(open, close),
  low > min(open, close), low > high, or non-finite/negative volume.
- `duplicate_stamp` — the session date occurs more than once in either frame.
- `calendar_gap` — a session in `sessions` inside [first, last] bar dates has
  no bar (all False when `sessions` is None).
- `jump_no_split` — |log close-to-close| > log(1.4) while the raw/adjusted
  split factor is unchanged (1e-6): an unrecorded split — or a genuine
  crash/large special dividend (false positives by design).
- `stale_run` — every date of a >=5-row run of identical closes.

`validate_ticker(source, ticker, start, end)` uses the source's raw +
split-adjusted bars (sessions default to the calendar ticker's raw dates);
`validate_tickers(source, tickers, start, end)` returns a `ValidationReport`
(`flags`: `ticker, date, *FLAG_COLUMNS`; `failed`: ticker -> error,
per-ticker isolated) and fetches the calendar once per call.

**Cost:** each (dataset, key, range) costs one upstream fetch the first
time it is requested by ANY process sharing the root and zero afterwards;
the still-open tail costs one normal (cached) call per request.

## Precompute parquet output

Source: `data_universe/precompute.py`.

- Path: `<out_dir>/<feature_or_label_name>.parquet`. Default `out_dir`:
  `<config.get_data_dir()>/processed`.
- Format: long, columns exactly `date, ticker, value`.
- Rows with a `NaN` value are dropped before writing — never written to disk.
- Rerunning `precompute()` for a ticker/date range already present **upserts** on
  `(date, ticker)`: existing rows for keys not touched by the new run are preserved; keys
  touched by the new run are replaced, never duplicated.
- `precompute(tickers, feature_names, start, end, out_dir=None, source=None, **kwargs) ->
  PrecomputeReport`: `feature_names` may name anything in `FEATURE_REGISTRY` or
  `LABEL_REGISTRY`; `**kwargs` are forwarded to `compute()` for every pair (e.g.
  `rate_source=FredSource()`). One `(ticker, feature_name)` pair's failure (unregistered
  name, a source error) is caught and recorded in `PrecomputeReport.failed[(ticker, name)]
  = str(error)`; it never stops the rest of the batch. Successful pairs are recorded in
  `PrecomputeReport.succeeded[(ticker, name)] = <rows written>`.
- Writes are safe for concurrent writers: on POSIX the read-merge-write holds an
  exclusive `fcntl.flock` on `<out_dir>/.<name>.parquet.lock`, and the result is written
  to a temp file in the same directory then `os.replace`'d into place — a failed write
  leaves the previous file byte-identical and no temp files behind. On Windows there is
  no lock (atomic write still applies).
- `load_feature(name, tickers=None, out_dir=None) -> pd.DataFrame` reads the file back,
  optionally filtered to a ticker subset. Raises `FileNotFoundError` if `name` was never
  precomputed to that `out_dir`.

## Membership grid

Source: `data_universe/membership/grid.py` (wrapped by `data_universe.membership`).

- `membership_grid(index_id, sessions, strict=False, by="security_id") -> pd.DataFrame`:
  one row per requested session, one column per member key, `bool`. `grid.loc[t, k]` is
  True iff `start_date <= t < end_date_excl` for `k`'s interval — membership at the close
  of session `t` (half-open intervals; `end_date_excl` is the first non-member date).
- `by="security_id"` (default): columns are provisional `<ticker>-<YYYYMMDD>` ids that
  will change when identity resolution lands — **don't persist them**. `by="ticker"`:
  columns are upstream's ticker symbol for each membership interval (see
  `data_universe/membership/tables/sp500/README.md`, "Provisional identity"); renames
  like FB → META are NOT linked.
- `mask_like(features, index_id, by=...)`: same index/columns as `features`; unknown
  columns are all-False. With `by="security_id"` and zero matching columns it raises
  `MembershipError` (almost certainly ticker columns — pass `by="ticker"`); with
  `by="ticker"` zero matches legitimately means non-members and never raises.
- `members_on(index_id, date, by=...) -> list`: sorted member keys at the close of `date`.
- Staleness: `status(index_id)` reports `verified_through`/`age_days`/`n_securities`/
  `warnings`; once `age_days` exceeds `membership_max_staleness_days` (default 45) the
  grid warns (`MembershipStaleWarning`) or, with `strict=True`, raises
  `MembershipStaleError`.

## Config precedence

Source: `data_universe/config.py`.

Precedence: **environment variable > `config/config.yaml` > hardcoded default.**

| Env var | Internal key |
|---|---|
| `POLYGON_API_KEY` | `polygon_key` |
| `POLYGON_BASE_URL` | `polygon_base_url` |
| `POLYGON_FLATFILES_ENDPOINT` | `polygon_flatfiles_endpoint` |
| `POLYGON_S3_ACCESS_KEY` | `polygon_s3_access_key` |
| `POLYGON_S3_SECRET_KEY` | `polygon_s3_secret_key` |
| `MASSIVE_API_KEY` | `massive_key` |
| `MASSIVE_BASE_URL` | `massive_base_url` |
| `FRED_API_KEY` | `fred_key` |
| `DATA_UNIVERSE_DATA_DIR` | `data_dir` |
| `DATA_UNIVERSE_CACHE_SIZE` | `cache_size` |
| `DATA_UNIVERSE_CACHE_DISK_DIR` | `cache_disk_dir` |
| `DATA_UNIVERSE_CACHE_DISK_SIZE_LIMIT` | `cache_disk_size_limit` |
| `DATA_UNIVERSE_MEMBERSHIP_MAX_STALENESS_DAYS` | `membership_max_staleness_days` |
| `DATA_UNIVERSE_MEMBERSHIP_TABLES_DIR` | `membership_tables_dir` |
| `DATA_UNIVERSE_USER_AGENT` | `user_agent` |

The Polygon API key is held only in memory and sent as an `Authorization: Bearer` header —
never interpolated into a URL, never logged.

## Known gaps other teams should not build against yet

- **`Ticker`** — not built. Nothing in `data_universe/__init__.py` exports it.
- **`FlatFilesSource`** — not built. The flat-files env vars above are read by `config.py`
  but nothing consumes them.
- **Single-name IV30** (`vrp_signal_rv21` and `fwd_vrp_expost` for any ticker other than
  `"SPY"`) — always raises `NotImplementedError` via the `options/iv30.py` stub. Both are
  registered (satisfying "provide interfaces and stubs") but unusable for a non-SPY ticker
  until IV30 lands.
- **`I:VIX` indices access** — `iv_index`'s SPY path depends on Polygon indices-aggregates
  entitlement, which the 2026-11-03 probe found is **not** on the current plan (HTTP 403),
  so it raises `EntitlementError`. See `OPEN_QUESTIONS.md` item 4.

See [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) for the full list of unresolved assumptions
behind every schema above.
