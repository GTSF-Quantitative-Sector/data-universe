# data_universe

GTSF Quant Sector shared data and feature library. Built for three research projects —
Intraday Beta Estimation (Kalman filter), VRP research, and PEAD with options-implied
surprise — so each stops re-writing its own Polygon/EDGAR/FRED plumbing.

## Install

```bash
pip install -e .

# optional extras:
pip install -e ".[sec]"   # sec_parser fundamentals / point-in-time S&P 500 membership
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
- **`cache.py`** — a shared in-memory LRU `Cache` with per-access logging. `get_cache()`
  returns the process-wide singleton; the `@cached("namespace")` decorator wraps `DataSource`
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
- **`options/`** — `occ` (OCC ticker parse/build), `black_scholes` (price, vega,
  implied_vol), `chain` (`as_of`, `atm_strike`, `nearest_expiry_after`, `straddle`), and
  `iv30` — a documented `NotImplementedError` stub (owner: vrp-research data track).
- **`features/`** — pure math in `core.py` (log returns, close-to-close/Parkinson/
  Garman-Klass realized vol, rolling OLS beta/alpha) plus the registry (`registry.py`,
  `FEATURE_REGISTRY`, `compute()`) and 14 concrete registered features in `library.py`:
  `ret_1d`, `ret_excess_1d`, `ret_21d`, `rv_cc_21d`, `rv_parkinson_21d`, `rv_gk_21d`,
  `dollar_volume_1d`, `adv_usd_20d`, `beta_ols_60d`, `beta_ols_90d`, `alpha_ols_60d`,
  `alpha_ols_90d`, `iv_index`, `vrp_signal_rv21`.
- **`labels/`** — the same registry shape (`registry.py`, `LABEL_REGISTRY`), enforcing that
  every name starts with `fwd_`, and 3 concrete labels in `library.py`: `fwd_rv_cc_21d`,
  `fwd_vrp_expost`, `fwd_ret_15d`.
- **`events/earnings.py`** — `build_pead_events(...)`: one row per qualifying 8-K Item 2.02
  filing, combining `EdgarSource`, price bars, and the options chain into the PEAD event
  table (schema in `CONTRACTS.md`).
- **`precompute.py`** — `precompute(tickers, feature_names, start, end, out_dir=None,
  source=None)` batch-computes to parquet with upsert-on-rerun and per-pair failure
  isolation (`PrecomputeReport`), and `load_feature(name, tickers=None, out_dir=None)` reads
  the result back.

## Frame conventions

- **Bar frames** (`sources/base.py`): tz-aware `America/New_York` `DatetimeIndex` named
  `timestamp`; columns `open, high, low, close, volume, vwap`, all `float64`. Daily bars are
  stamped at midnight of the session date; intraday bars are filtered to regular trading
  hours (9:30-16:00 ET) unless the caller opts out.
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
| `FRED_API_KEY` | FRED API key, used by `sources.fred.FredSource` |
| `DATA_UNIVERSE_DATA_DIR` | Root directory for raw/interim/processed data and cache logs |
| `DATA_UNIVERSE_CACHE_SIZE` | Default `Cache` capacity (entry count) |

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
