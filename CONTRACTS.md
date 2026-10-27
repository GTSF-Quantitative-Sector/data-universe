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
calendar_end=None) -> pd.DataFrame`.

One row per qualifying 8-K Item 2.02 filing, sorted ascending by `event_date`. Columns,
exactly, in this order:

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

The ATM straddle is chosen with `spot` = the **raw** day-0 close — historical option
strikes are on the raw (as-traded) basis, so a split-adjusted spot would pick the wrong
strike for any event preceding a later split. The chain query itself is filtered
server-side: expirations in `(day0, day0 + 40]` days and strikes within ±20% of spot
(`PolygonOptionsSource.contracts_as_of`'s `expiration_*`/`strike_*` params), with an
`expired`+`active` union when the window can reach today.

A filing too close to either edge of the available calendar/price history to compute (not
enough trading-session lookback, no expiry found, no matching call/put in the chain, or a
ticker bar missing on `day0`/`day1` in either the adjusted or the raw frame — e.g. a halt
while the market stayed open) is
**silently skipped** — it does not appear as a null row, and it does not raise.

Issue 3 from the original spec review ("weekly ranking lookahead") is out of scope: this
table has no ranking column at all.

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
  entitlement, which is unconfirmed. See `OPEN_QUESTIONS.md` item 4.

See [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) for the full list of unresolved assumptions
behind every schema above.
