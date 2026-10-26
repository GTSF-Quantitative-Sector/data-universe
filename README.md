# data_universe

GTSF Quant Sector shared data and feature library. Built for three research projects —
Intraday Beta Estimation (Kalman filter), VRP research, and PEAD with options-implied
surprise — so each stops re-writing its own Polygon/EDGAR/FRED plumbing.

## Install

```bash
pip install -e .

# optional extras:
pip install -e ".[sec]"   # sec_parser fundamentals / S&P 500 membership (alternative to the built-in membership/)
pip install -e ".[s3]"    # boto3, reserved for the not-yet-built flat-files source
pip install -e ".[dev]"   # pytest, ruff
```

## Quickstart

```python
import data_universe as q
from data_universe.sources.fakes import FakePolygonSource  # no API key needed

source = FakePolygonSource()

# Compute a registered feature directly against a source:
from data_universe.features.registry import compute as compute_feature
beta = compute_feature(
    q.FEATURE_REGISTRY, "beta_ols_60d", "AAPL", "2024-01-01", "2024-06-01", source
)

# Or batch-compute to parquet and read it back later:
report = q.precompute(
    ["AAPL", "MSFT"], ["rv_cc_21d", "beta_ols_60d"], "2024-01-01", "2024-06-01",
    out_dir="data/processed", source=source,
)
panel = q.load_feature("beta_ols_60d", tickers=["AAPL"])

# With a real Polygon key instead of the fake:
q.config.set_polygon_key("YOUR_KEY")
q.config.set_project("beta")  # tags cache access-log rows
q.get_cache().stats()
```

`q.Ticker` from the original design sketch — a thin `aapl.feature(...)`/`aapl.daily(...)`
façade over the pieces above — has not been built yet; see `OPEN_QUESTIONS.md`. Until then,
call `features.registry.compute()` / `labels.registry.compute()` directly, or use
`precompute()`/`load_feature()` for batch work.

## What's here

- **`config.py`** — global configuration: `set_polygon_key`, `set_project`, `set_data_dir`,
  `set_cache_size`, `load_yaml`, and their `get_*` counterparts. Env var > YAML file >
  hardcoded default precedence (see "Configuration" below).
- **`cache.py`** — a shared, two-tier (in-memory, optionally disk-backed) LRU `Cache` with
  per-access logging. `get_cache()` returns the process-wide singleton, whose disk tier
  turns on when `cache_disk_dir` is configured, so colocated processes share persisted
  entries across restarts; the `@cached("namespace")` decorator wraps `DataSource`
  methods; `simulate(log, sizes)` replays a logged access sequence against candidate cache
  sizes to report hit rate.
- **`calendar.py`** — trading-session helpers (`sessions`, `next_session`, `previous_session`,
  `shift_sessions`) derived from a market proxy's own daily bars, no separate holiday
  calendar.
- **`sources/`** — `PolygonSource` (stock/index daily & hourly aggregates),
  `PolygonOptionsSource` (options contract reference + per-contract daily closes),
  `FredSource` (risk-free rate, default series `DTB3`), `EdgarSource` (8-K Item 2.02
  earnings-announcement dates + CIK lookup), `sec_adapter` (optional `sec_parser` wrapper for
  `sp500(query_date=...)` and `fundamentals(ticker)`), and `fakes.FakePolygonSource` /
  `fakes.FakeMarket` — a deterministic synthetic market with a planted beta/alpha, used by
  the whole test suite and available to anyone without an API key.
- **`sources/massive.py`** — `MassiveSource`, `PolygonSource` pointed at Massive (Polygon.io's
  new name, host `api.massive.com`): the same bars methods plus `list_tickers` and
  `ticker_details` reference data. Reads `MASSIVE_API_KEY`, falling back to the Polygon key.
- **`sources/index_universe.py`** — collects the S&P 500, Dow 30, Nasdaq-100 and all
  Nasdaq-listed stocks, cross-checks them against Massive, and saves them to parquet. See
  "Index universes" below.
- **`options/`** — `occ` (OCC ticker parse/build), `black_scholes` (price, vega,
  implied_vol), `chain` (`as_of`, `atm_strike`, `nearest_expiry_after`, `straddle`), and
  `iv30` — a documented `NotImplementedError` stub (owner: vrp-research data track).
- **`features/`** — pure math in `core.py` (log returns, close-to-close/Parkinson/
  Garman-Klass realized vol, rolling OLS beta/alpha) plus the registry (`registry.py`,
  `FEATURE_REGISTRY`, `compute()`) and 14 concrete registered features in `library.py`:
  `ret_1d`, `ret_excess_1d`, `ret_21d`, `rv_cc_21d`, `rv_parkinson_21d`, `rv_gk_21d`,
  `dollar_volume_1d`, `adv_usd_20d`, `beta_ols_60d`, `beta_ols_90d`, `alpha_ols_60d`,
  `alpha_ols_90d`, `iv_index`, `vrp_signal_rv21`. `ret_excess_1d` requires
  `compute(..., rate_source=...)` (e.g. `FredSource()` or `FakeRateSource()`); the
  `beta_*`/`alpha_*` features run an excess-return regression when `rate_source` is
  given and a raw-return regression without it.
- **`labels/`** — the same registry shape (`registry.py`, `LABEL_REGISTRY`), enforcing that
  every name starts with `fwd_`, and 3 concrete labels in `library.py`: `fwd_rv_cc_21d`,
  `fwd_vrp_expost`, `fwd_ret_15d`.
- **`events/earnings.py`** — `build_pead_events(...)`: one row per qualifying 8-K Item 2.02
  filing, combining `EdgarSource`, price bars, and the options chain into the PEAD event
  table (schema in `CONTRACTS.md`).
- **`membership/`** — point-in-time S&P 500 membership from vendored
  [fja05680/sp500](https://github.com/fja05680/sp500) tables: `sp500_intervals`,
  `membership_grid`, `mask_like`, `members_on`, `status`, keyed by
  `by="security_id"` (provisional `<ticker>-<YYYYMMDD>` ids — don't persist them) or
  `by="ticker"` (upstream's symbol per interval; renames not linked). `status()`/the
  grid warn once the data is more than 45 days past `verified_through`
  (`strict=True` raises). Refresh monthly or after index changes with
  `python -m data_universe.membership.refresh [--ref REF] [--dry-run]`; see
  `MAINTENANCE.md` item 2.
- **`precompute.py`** — `precompute(tickers, feature_names, start, end, out_dir=None,
  source=None, **kwargs)` batch-computes to parquet with upsert-on-rerun, per-pair
  failure isolation (`PrecomputeReport`), and an flocked + atomic write so concurrent
  writers can't corrupt a file; extra `**kwargs` (e.g. `rate_source=FredSource()`)
  are forwarded to `compute()` for every pair. `load_feature(name, tickers=None,
  out_dir=None)` reads the result back.

## Index universes

```bash
python -m data_universe.sources.index_universe                 # S&P 500, Dow, Nasdaq-100, Nasdaq-listed
python -m data_universe.sources.index_universe --skip-massive  # constituents only, no API key needed
```

```python
import data_universe as q
from data_universe.sources.index_universe import collect_universes, all_tickers, load_universe
from data_universe.sources.massive import MassiveSource

universes = collect_universes(massive=MassiveSource(), save=True)   # saves to <data_dir>/raw/universe
universes["dow"][["ticker", "name", "in_massive", "primary_exchange"]]
q.precompute(all_tickers(universes), ["rv_cc_21d"], "2024-01-01", "2024-06-01")
load_universe("sp500")                                              # newest saved copy
```

`sp500`, `dow` and `nasdaq100` come from Wikipedia (Massive has index prices but no
constituent lists); `nasdaq_listed` is every active Nasdaq-listed common stock from Massive
reference data, which is not the Nasdaq-100. Massive is called once for the active-stock
list (roughly a dozen paginated requests) and adds `in_massive`, `primary_exchange`, `type`,
`cik` and FIGI columns to each universe. **These are today's memberships**: for backtests
use `q.membership` (below); `q.universe.sp500(query_date=...)` via the optional
`sec_parser` dependency is the alternative. To pull bars from the Massive host, use
`MassiveSource` (it has the same `daily_bars`/`hourly_bars` methods as `PolygonSource`).
See `OPEN_QUESTIONS.md` items 15-17.

## Point-in-time S&P 500 membership

```python
panel = q.load_feature("ret_1d").pivot(index="date", columns="ticker", values="value")
in_index = q.membership.mask_like(panel, "sp500", by="ticker")  # True where a member
in_index.loc["2020-06-15", "AAPL"]                              # True
q.membership.members_on("sp500", "2020-06-15", by="ticker")     # ~500 tickers
```

`by="ticker"` keys columns on upstream's ticker symbol for each membership interval
(renames like FB → META are *not* linked); `by="security_id"` (the default) keys on
provisional `<ticker>-<YYYYMMDD>` ids that will change when identity resolution lands —
don't persist them. Tables and refresh tool live in `data_universe/membership/tables/sp500`;
see `MAINTENANCE.md` item 2.

## Frame conventions

- **Bar frames** (`sources/base.py`): tz-aware `America/New_York` `DatetimeIndex` named
  `timestamp`; columns `open, high, low, close, volume, vwap`, all `float64`. Daily bars are
  stamped at midnight of the session date; intraday bars are stamped at the **start** of
  their window and filtered to regular trading hours by window overlap — a bar survives iff
  `[stamp, stamp + bar_minutes)` intersects 9:30-16:00 ET, so `hourly_bars` keeps stamps
  09:00–15:00 (the 09:00 bar includes 9:00–9:30 pre-market trades — accepted convention;
  see `OPEN_QUESTIONS.md` item 19).
- **Feature/label series**: tz-naive `DatetimeIndex` named `date`, one value per session.
- Prices are split- and dividend-adjusted by default; a function whose name doesn't say
  "raw" or "unadjusted" is always adjusted.

## Configuration

Precedence: **environment variable > `config/config.yaml` > hardcoded default.**

| Env var | Purpose |
|---|---|
| `POLYGON_API_KEY` | Polygon.io API key (sent as `Authorization: Bearer`, never in a URL) |
| `POLYGON_BASE_URL` | Polygon REST base URL (default `https://api.polygon.io`) |
| `POLYGON_FLATFILES_ENDPOINT` | S3-compatible flat-files endpoint (read by `config.py`; no consumer built yet — see "Not yet implemented") |
| `POLYGON_S3_ACCESS_KEY` | S3 access key for flat-files (same caveat) |
| `POLYGON_S3_SECRET_KEY` | S3 secret key for flat-files (same caveat) |
| `MASSIVE_API_KEY` | Massive API key (Bearer header). Falls back to `POLYGON_API_KEY` if unset |
| `MASSIVE_BASE_URL` | Massive REST base URL (default `https://api.massive.com`) |
| `FRED_API_KEY` | FRED API key, used by `sources.fred.FredSource` |
| `DATA_UNIVERSE_DATA_DIR` | Root directory for raw/interim/processed data and cache logs |
| `DATA_UNIVERSE_CACHE_SIZE` | Default `Cache` capacity (entry count) |
| `DATA_UNIVERSE_CACHE_DISK_DIR` | Directory for the disk-backed cache tier. Unset keeps the cache in-memory only (default) |
| `DATA_UNIVERSE_CACHE_DISK_SIZE_LIMIT` | Byte size limit for the disk tier. Unset uses `diskcache`'s own default |
| `DATA_UNIVERSE_MEMBERSHIP_MAX_STALENESS_DAYS` | Days past `verified_through` before membership grids/status warn (default 45; `strict=True` raises) |
| `DATA_UNIVERSE_MEMBERSHIP_TABLES_DIR` | Directory to read membership tables from instead of the packaged `data_universe/membership/tables` (default unset) |

Copy `config/config.example.yaml` to `config/config.yaml` (gitignored) and fill in real
values, or call `q.config.load_yaml("config/config.yaml")` explicitly. An environment
variable already set always wins over the YAML file.

## Not yet implemented

- **`Ticker`** (`data_universe/ticker.py`) — not built; `__init__.py` doesn't export it.
  Workaround: call `features.registry.compute()` / `labels.registry.compute()` directly, or
  use `precompute()`/`load_feature()`. See `OPEN_QUESTIONS.md` item 11.
- **`FlatFilesSource`** (`data_universe/sources/flatfiles.py`) — not built; no S3 bulk
  download or `download_options_day_aggs()` exists yet, despite the flat-files env vars
  already being read by `config.py`. See `OPEN_QUESTIONS.md` item 12.

## Testing

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

The full suite runs against `sources/fakes.py`'s deterministic `FakeMarket`/
`FakePolygonSource`, never live APIs, so CI needs no secrets
(see `.github/workflows/ci.yml`).

## See also

- [`PLAN.md`](PLAN.md) — the original design spec this library implements.
- [`CONTRACTS.md`](CONTRACTS.md) — frame/schema reference for downstream project teams.
- [`MAINTENANCE.md`](MAINTENANCE.md) — recurring upkeep tasks, including the standing
  version-bump gap.
- [`OPEN_QUESTIONS.md`](OPEN_QUESTIONS.md) — every unverified assumption and known gap, with
  current stance and next step.
