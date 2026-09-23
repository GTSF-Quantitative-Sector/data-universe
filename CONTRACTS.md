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
- Intraday bars are filtered to regular trading hours (9:30-16:00 ET) unless the caller
  explicitly opts out.
- Prices are split- and dividend-adjusted unless a function's name says "raw" or
  "unadjusted".

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

## PEAD event table

Source: `data_universe/events/earnings.py`'s `build_pead_events(ticker, edgar_source,
price_source, options_source, cik=None, calendar_start="2015-01-01",
calendar_end=None) -> pd.DataFrame`.

One row per qualifying 8-K Item 2.02 filing, sorted ascending by `event_date`. Columns,
exactly, in this order:

| Column | Meaning |
|---|---|
| `ticker` | Underlying ticker symbol. |
| `event_date` | The 8-K's SEC acceptance timestamp, normalized to a session date. |
| `pre_session` | The trading session immediately before `day0`. |
| `day0` | `event_date` itself if it's a trading session, else the next one. |
| `day1` | The trading session immediately after `day0`. |
| `sigma_pre_daily` | Daily (not annualized) stdev of log returns, 30 **trading** days ending at `pre_session`. |
| `adv_usd_pre20` | 20-trading-day average dollar volume ending at `pre_session`. |
| `adv_usd_day0_day1` | Average dollar volume over `day0`/`day1` only — no other post-entry window is computed anywhere in this library. |
| `days_to_expiry` | Calendar days from `day0` to the nearest option expiry strictly after `day0`. The "nearest expiry after" convention is kept as originally specified; any bias this introduces is left for the project team to adjust, not fixed here. |
| `next_earnings_session` | **Realized**, not forecast: the following row's `event_date`, filled in only after sorting. `NaT` for the most recent filing. Never use this as a model input for `day0` itself — it's hindsight. |
| `call_volume` / `put_volume` | Contract volume traded on `day0` for the chosen ATM straddle's call/put leg. |
| `is_stale` | `True` whenever either leg of the straddle traded zero volume on `day0` (a stale daily close). |

A filing too close to either edge of the available calendar/price history to compute (not
enough trading-session lookback, no expiry found, no matching call/put in the chain) is
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
- `precompute(tickers, feature_names, start, end, out_dir=None, source=None) ->
  PrecomputeReport`: `feature_names` may name anything in `FEATURE_REGISTRY` or
  `LABEL_REGISTRY`. One `(ticker, feature_name)` pair's failure (unregistered name, a source
  error) is caught and recorded in `PrecomputeReport.failed[(ticker, name)] = str(error)`;
  it never stops the rest of the batch. Successful pairs are recorded in
  `PrecomputeReport.succeeded[(ticker, name)] = <rows written>`.
- `load_feature(name, tickers=None, out_dir=None) -> pd.DataFrame` reads the file back,
  optionally filtered to a ticker subset. Raises `FileNotFoundError` if `name` was never
  precomputed to that `out_dir`.

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
| `FRED_API_KEY` | `fred_key` |
| `DATA_UNIVERSE_DATA_DIR` | `data_dir` |
| `DATA_UNIVERSE_CACHE_SIZE` | `cache_size` |

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
