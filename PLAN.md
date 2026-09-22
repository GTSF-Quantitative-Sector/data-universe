# PLAN.md — GTSF Quant shared data + feature library

Status: **Step 1 deliverable. Stopping here for review before any code is written**, per the
task instructions ("Explore the repo, read this prompt fully, and write PLAN.md ... Do not
write code yet.").

## 0. What I looked at before writing this

- This repo (`data-universe` on GitHub, org `GTSF-Quantitative-Sector`): brand new, empty,
  zero commits, no existing files besides `.git`. Nothing to preserve or migrate.
- `GTSF-Quantitative-Sector/sec_parser` (cloned to a scratch dir and read in full for
  `stock.py`, `lookups.py`, `processor.py`, `constants.py`, `__init__.py`, `README.md`,
  and both data `README.md` files under `sec/data/`). Conventions confirmed there and
  mirrored below:
  - `setup.py` has a bare `VERSION = "1.93"` string constant; README repeats "you must
    update the version number... pip will not update the library" verbatim.
  - A module-level global + setter: `constants.POLYGON_KEY = None`, `set_polygon_key(key)`
    mutates a module global. No class, no env var fallback.
  - Every public method takes `query_date: str = None` (`YYYY-MM-DD`, `None` = latest) and
    docstrings in Google/Args-Returns style stating types and defaults.
  - `Stock.get_price` / `get_aggs` / `get_rsi` are `async def` using `aiohttp` with
    `TCPConnector(ssl=False)` — this is the thing our spec explicitly tells us to fix
    (sync `requests.Session`, TLS verification on).
  - `lookups.get_sp500_tickers(query_date=None)` reads a manually-maintained
    `sp500_historical.csv`; `sec/data/sp500/README.md` is a one-paragraph maintenance note
    ("steps below should be completed monthly... link... date of last update"). Our
    `MAINTENANCE.md` entries will follow that exact shape.
  - `packages=["sec"]` is listed explicitly in `setup.py`, not discovered — I'll mirror
    that (explicit list, not `find_packages()`), for parity with the reference repo.
  - `sec/__init__.py` is empty — sec_parser does **not** re-export a public API at the
    package root; everything is `from sec import stock, constants, lookups`. Our spec
    explicitly asks for `__init__.py` to do public API re-exports and `__version__`, so
    we deliberately diverge here (documented in the README's "how this differs from
    sec_parser" section) rather than silently copying the empty-`__init__` pattern.

I could not fetch live Polygon/FRED/EDGAR API docs from this environment, so every
endpoint path, field name, and base URL below is either taken directly from the prompt's
explicit instructions, or marked `[UNVERIFIED]` and written up in `OPEN_QUESTIONS.md`
instead of guessed into working code.

## 1. Package name

The repo is `data-universe`, which is not a valid Python identifier (hyphen). Proposed
package name: **`data_universe`**. Install stays `pip install
git+https://github.com/GTSF-Quantitative-Sector/data-universe.git`; the importable name
is `import data_universe as q`, matching the README example in the prompt (`import
<package> as q`). All paths below use `data_universe/` as the package root.

## 2. File layout

```
data_universe/
  __init__.py            public API re-exports, __version__
  config.py              set_polygon_key, set_project, base URLs, data dir, cache size
  cache.py               LRU cache, access log, @cached decorator, simulate()
  calendar.py            trading sessions derived from market-proxy daily bars
  ticker.py              Ticker object API
  precompute.py          batch compute to parquet, merge on rerun, failure isolation
  sources/
    __init__.py
    base.py              DataSource ABC, Bars/frame conventions, exceptions
    polygon.py            PolygonSource: stock/index aggregates, pagination, 429 backoff
    polygon_options.py    PolygonOptionsSource: contract reference, per-contract daily aggs
    flatfiles.py          FlatFilesSource: S3 bulk download, download_options_day_aggs()
    fred.py                FredSource: risk-free rate (DTB3)
    edgar.py               EdgarSource: 8-K Item 2.02 dates, CIK lookup
    sec_adapter.py         optional wrapper around sec_parser (fundamentals, S&P membership)
    fakes.py               deterministic fake sources for tests / no-key usage
  options/
    __init__.py
    occ.py                 OCC ticker parse/build
    black_scholes.py       price, vega, implied_vol
    chain.py               as_of, atm_strike, nearest_expiry_after, straddle
    iv30.py                STUB — NotImplementedError, owner: vrp-research data track (Lionel)
  features/
    __init__.py
    core.py                pure math: returns, vol estimators, rolling OLS beta/alpha
    registry.py            FEATURE_REGISTRY, @register, compute()
    library.py              registered feature functions
  labels/
    __init__.py
    registry.py             LABEL_REGISTRY, @register (fwd_ prefix enforced)
    library.py               registered label functions (fwd_rv_cc_21d, vrp_expost, fwd_ret_15d)
  events/
    __init__.py
    earnings.py              PEAD event table builder
config/
  config.example.yaml
data/                      gitignored: raw/, interim/, processed/, cache_logs/
tests/
  conftest.py
  test_cache.py
  test_config.py
  test_calendar.py
  test_sources_polygon.py
  test_sources_polygon_options.py
  test_sources_flatfiles.py
  test_sources_fred.py
  test_sources_edgar.py
  test_sources_sec_adapter.py
  test_options_occ.py
  test_options_black_scholes.py
  test_options_chain.py
  test_features_core.py
  test_features_registry.py
  test_features_no_lookahead.py   # generic test run over every registered feature
  test_labels_registry.py
  test_events_earnings.py
  test_precompute.py
CONTRACTS.md
MAINTENANCE.md
README.md
PLAN.md
OPEN_QUESTIONS.md
setup.py
requirements.txt
.gitignore
.github/workflows/ci.yml
```

No changes to the layout given in the prompt other than filling in the package name and
adding `__init__.py` files for each sub-package (needed for `pip install -e .` to work;
not a structural change).

## 3. Public API (`data_universe/__init__.py`)

```python
from data_universe import config
from data_universe.cache import get_cache, simulate
from data_universe.ticker import Ticker
from data_universe.precompute import precompute, load_feature
from data_universe.sources import sec_adapter as universe   # q.universe.sp500(query_date=...)
from data_universe.features.registry import FEATURE_REGISTRY
from data_universe.labels.registry import LABEL_REGISTRY

__version__ = "0.1.0"   # kept in sync with setup.py's VERSION; bump both on every change
```

Matches every line of the prompt's README example:

```python
import data_universe as q
q.config.set_polygon_key("YOUR_KEY")
q.config.set_project("pead")
aapl = q.Ticker("AAPL")
aapl.daily(start=..., end=...)
aapl.hourly(start=..., end=...)
aapl.feature("beta_ols_60d", start=..., end=...)
aapl.earnings_events()
q.universe.sp500(query_date="2019-01-01")
q.precompute(["AAPL","MSFT"], ["rv_cc_21d","beta_ols_60d"], "2020-01-01", "2024-12-31")
q.load_feature("beta_ols_60d", tickers=["AAPL"])
q.get_cache().stats()
q.simulate(q.get_cache().access_log(), [16, 64, 256, 1024])
```

### `config.py`

```python
def set_polygon_key(key: str) -> None: ...
def set_project(name: str) -> None: ...          # tags every subsequent cache access
def set_polygon_base_url(url: str) -> None: ...  # POLYGON_BASE_URL, default https://api.polygon.io
def set_data_dir(path: str) -> None: ...
def load_yaml(path: str = "config/config.yaml") -> None: ...  # env vars still override after load
```

Env vars: `POLYGON_API_KEY`, `POLYGON_BASE_URL`, `POLYGON_FLATFILES_ENDPOINT`,
`POLYGON_S3_ACCESS_KEY`, `POLYGON_S3_SECRET_KEY`, `FRED_API_KEY`, `DATA_UNIVERSE_DATA_DIR`,
`DATA_UNIVERSE_CACHE_SIZE`. Precedence: env var > `config/config.yaml` > hardcoded default.
The API key is stored only in memory and sent as `Authorization: Bearer <key>` — never
interpolated into a URL.

### `cache.py`

```python
class Cache:
    def __init__(self, capacity: int = 256): ...
    def get_or_load(self, namespace: str, key: tuple, loader: Callable[[], Any]) -> Any: ...
    def clear(self) -> None: ...
    def stats(self) -> dict: ...          # size, hits, misses, evictions, hit_rate, bytes_held
    def access_log(self) -> pd.DataFrame: ...
    def flush_log(self, path: str | None = None) -> str: ...  # -> data/cache_logs/*.parquet

def get_cache() -> Cache: ...             # module-level singleton
def cached(namespace: str): ...           # decorator: namespace = f"{ClassName}.{namespace}"
def simulate(log: pd.DataFrame, sizes: list[int]) -> pd.DataFrame: ...
```

`Cache` is a small `Protocol`/ABC (`get_or_load`, `clear`, `stats`) so a byte-bounded or
disk-backed implementation can be swapped in later. The default implementation is the
`OrderedDict` + `threading.Lock` LRU: the lock guards dict mutation only; the actual
`loader()` call happens outside the lock, so a double in-flight load for the same key on a
cache miss is possible and accepted (documented limitation, matches "keep it simple").

### `ticker.py`

```python
class Ticker:
    def __init__(self, symbol: str): ...
    def daily(self, start: str, end: str) -> pd.DataFrame: ...
    def hourly(self, start: str, end: str) -> pd.DataFrame: ...
    def feature(self, name: str, start: str, end: str) -> pd.Series: ...
    def label(self, name: str, start: str, end: str) -> pd.Series: ...
    def earnings_events(self) -> pd.DataFrame: ...
```

`Ticker` holds no cache of its own — it is a thin façade over `sources.polygon.PolygonSource`
and the registries. The `@cached` decorator lives on the source methods (`PolygonSource.
daily_bars`, etc.) so namespace separation ("class name in the namespace") is automatic and
two `Ticker("AAPL")` instances share one cache entry.

### `precompute.py`

```python
def precompute(tickers: list[str], feature_names: list[str], start: str, end: str,
                out_dir: str | None = None) -> "PrecomputeReport": ...
def load_feature(name: str, tickers: list[str] | None = None) -> pd.DataFrame: ...

@dataclass
class PrecomputeReport:
    succeeded: dict[tuple[str, str], int]   # (ticker, feature) -> rows written
    failed: dict[tuple[str, str], str]      # (ticker, feature) -> error message
```

Output layout: `data/processed/<feature_name>.parquet`, long format
`date, ticker, value`. Rerunning `precompute` for a ticker/date range already present
upserts on `(date, ticker)` rather than duplicating rows. One ticker's failure (e.g. a
missing filing, a 404 from Polygon) is caught, recorded in `PrecomputeReport.failed`, and
does not stop the batch.

## 4. Frame conventions (`sources/base.py`, restated in README)

- Bar frames: tz-aware `DatetimeIndex` (`America/New_York`), index name `timestamp`;
  columns `open, high, low, close, volume, vwap`, all `float64`. Daily bars stamped at
  midnight of the session date. Intraday bars filtered to regular trading hours (9:30–16:00
  ET) unless a function is explicitly named `_extended` or takes `regular_hours=False`.
- Daily feature/label frames: tz-naive `DatetimeIndex` named `date` (session date), one
  column per feature/label, or long format (`date, ticker, value`) for multi-ticker output
  — precompute output and the PEAD/VRP panels use long format; single-ticker `Ticker.
  feature()` returns a `pd.Series` indexed by `date`.
- Prices are split- and dividend-adjusted by default; a function whose name doesn't say
  "raw" or "unadjusted" is always adjusted.

## 5. The cache — design specifics

- `OrderedDict[key] -> CacheEntry(value, bytes, inserted_at)`, one `threading.Lock` guarding
  moves/evictions/inserts. Default capacity 256, configurable via
  `DATA_UNIVERSE_CACHE_SIZE` or `config.set_cache_size(n)`.
- Key = `(namespace, normalized_args)`. Normalization: positional args + sorted kwargs
  items; lists sorted+tuple'd; dicts turned into sorted tuple-of-pairs; `datetime`/`date`
  objects and any string date turned into ISO date strings, so `"2024-01-01"` and
  `datetime(2024,1,1)` collide on purpose.
- `namespace = f"{type(self).__name__}.{decorator_namespace}"` — set inside the `@cached`
  decorator by inspecting `self` on the wrapped bound method, so `FakePolygonSource` and
  `PolygonSource` never collide even if given the same `@cached("daily_bars")`.
- Hit path: `copy.deepcopy` for `dict`/`list` leaves, `.copy()` for `pd.DataFrame`/
  `pd.Series` (cheap, and correct for "notebook can't corrupt the cache").
  Non-DataFrame/Series scalars are returned as-is (immutable by convention: floats, tuples).
- Access log columns: `timestamp, namespace, key_repr, hit, bytes, load_seconds, project`.
  `project` comes from `config.set_project(name)`, stored as a module-level current value
  (same "global mutable config" pattern as sec_parser's `POLYGON_KEY`) — not thread-local,
  since this library is not expected to run multiple projects concurrently in one process;
  flagged as a documented limitation, not solved with `contextvars` for this EOW pass.
- `cache.flush_log(path=None)` writes parquet to `data/cache_logs/access_log_<ts>.parquet`
  by default.
- `simulate(log, sizes)` replays the logged sequence of `(namespace, key, bytes,
  load_seconds)` through an in-memory-only LRU of each requested size (no real loads) and
  returns a `pd.DataFrame` with one row per size: `hit_rate`, `load_seconds_saved`.

## 6. Features vs. labels — enforcement

- `features/registry.py` and `labels/registry.py` are separate `dict`-based registries.
  Both `@register(name, warmup, project, description)` decorators validate the `fwd_`
  prefix rule at *registration time* (module import), so a naming mistake fails on
  `import data_universe`, not only when a mutation test runs.
- `compute(registry, name, ticker, start, end)` is shared logic in `features/registry.py`
  (imported by `labels/registry.py`) that: looks up the function, fetches
  `start - warmup_trading_days` through `end` of underlying bars via the calendar module,
  calls the function, and slices the result back to `[start, end]` — so the first returned
  value is never NaN because of insufficient history (it can still be NaN for a real data
  gap, which is different and reported, not hidden).

## 7. Project-specific pieces — plan-level notes

**Beta project.** `beta_ols_60d`/`90d` and `alpha_ols_60d`/`90d` computed in `features/
core.py` as pure functions over aligned excess-return arrays (no I/O), called from
`features/library.py` with data supplied by `Ticker`/`compute()`. VIX comes from
`sources/polygon.py` `index_aggs("I:VIX", ...)` — flagged in `OPEN_QUESTIONS.md` because
indices access may not be included in the Options Business plan tier. Risk-free rate comes
from `sources/fred.py` using series id `DTB3` (configurable), converted from an annualized
T-bill discount/yield quote to a per-trading-day rate — the exact conversion (bank
discount vs. bond-equivalent yield, /252 vs. actual day count) is flagged in
`OPEN_QUESTIONS.md` since the prompt doesn't pin it down and getting it wrong changes every
downstream excess return by a small but real amount.

**VRP project.** `features/core.py` gets close-to-close, Parkinson, and Garman-Klass
realized vol estimators (annualized, `sqrt(252)` scaling, 21-day trailing default window).
`labels/library.py` gets `fwd_rv_cc_21d` (forward 21-trading-day realized vol) and
`vrp_expost` (= `iv_index`/IV30 input minus `fwd_rv_cc_21d`); `vrp_expost` and
`vrp_signal_rv21` both call `options/iv30.py`, which raises `NotImplementedError` for any
ticker until Lionel's IV30 panel exists — so both are registered (satisfying "provide
interfaces and stubs") but unusable until that dependency lands. `iv_index` for SPY only
resolves to `I:VIX`; single-name IV30 raises `NotImplementedError` explicitly, per spec.
Point-in-time universe is `sources/sec_adapter.sp500(query_date=...)`, a pass-through to
sec_parser's `lookups.get_sp500_tickers(query_date=...)` — never today's membership.

**PEAD project.** `events/earnings.py` builds one row per (ticker, earnings event) using
`sources/edgar.py` for 8-K Item 2.02 acceptance timestamps, `sources/polygon.py` for
underlying closes, and `options/chain.py` for the ATM straddle. Every "spec issue to flag"
in section 8 below gets its own named column, never silently resolved.

## 8. Spec issues — flagged with named columns (not resolved)

| # | Issue | Column(s) added | Resolution taken now |
|---|-------|------------------|-----------------------|
| 1 | `sigma_pre` units | `sigma_pre_daily` | Daily stdev of log returns, 30 **trading** days ending at `pre_session` (not annualized). "Trading vs. calendar days" asked in OPEN_QUESTIONS.md. |
| 2 | Liquidity filter lookahead | `adv_usd_pre20`, `adv_usd_day0_day1` | Only these two provided; no post-entry-window average is computed anywhere in the library. |
| 3 | Weekly ranking lookahead | *(none — not built)* | The event table has no "this week's rank" column at all; ranking is explicitly out of scope and flagged. |
| 4 | Straddle expiry bias | `days_to_expiry` | Formula kept as specified (nearest expiry **after** earnings); bias left for the project team to adjust. |
| 5 | Next earnings date | `next_earnings_session` | Column is explicitly documented as "realized date, not one known in advance" in its docstring and in `CONTRACTS.md`. |
| 6 | Stale option closes | `call_volume`, `put_volume`, `is_stale` | `is_stale = (call_volume == 0) or (put_volume == 0)` on the chosen straddle contracts. |

## 9. Conflicts / ambiguities found while reading the prompt

1. **Package/import name.** Repo name isn't a valid identifier. Resolved above as
   `data_universe`; flag only in case a different name is preferred (e.g. `gtsf_quant`).
2. **`config/config.yaml` needs a YAML parser.** The prompt's `requirements.txt` list
   (`pandas`, `numpy`, `requests`, `pyarrow`, `scipy` for core) doesn't include one. I'll
   add `pyyaml` to core requirements — flagged in `OPEN_QUESTIONS.md` in case there's a
   reason to avoid it (e.g. prefer TOML/env-only config).
2b. **FRED needs an API key.** Not mentioned in the config/env var list in the prompt.
   Adding `FRED_API_KEY` / `config.set_fred_key()`, flagged in `OPEN_QUESTIONS.md`.
3. **Options Business tier and NBBO/quotes.** The prompt says to *assume* only daily
   contract closes are available, and separately says full trades/NBBO "may be a separate
   add-on." I'm building strictly against the assume-daily-closes-only interface (hence
   `is_stale`) and not attempting an NBBO/quotes source at all this pass — flagged as a
   question rather than built speculatively.
4. **Indices access for `I:VIX`.** Same as above — built against the interface, actual
   entitlement unconfirmed, flagged.
5. **Polygon endpoint paths/fields I am not confident of** (each written against an
   interface + mocked in tests, each also listed in `OPEN_QUESTIONS.md`, never hardcoded
   from memory into a "confident" call): the index-aggregates endpoint for `I:VIX`; the
   options contract-reference endpoint's exact path/params for an as-of-date chain lookup;
   the options daily-aggregates endpoint path for a single OCC ticker; the flat-files S3
   endpoint hostname (I will not guess a default and will require it be set via
   `POLYGON_FLATFILES_ENDPOINT` with no hardcoded fallback until confirmed).
6. **EDGAR 8-K Item 2.02 source.** I'm planning to use SEC's per-company submissions JSON
   (`data.sec.gov/submissions/CIK##########.json`) to find 8-K filings and their
   `primaryDocument`/accession metadata, then a full-text or filing-index fetch to confirm
   Item 2.02 is present (vs. e.g. Item 5.02) and to exclude `8-K/A`. I have not verified the
   exact JSON field that carries "which Items are checked" for a given 8-K accession — flagged.
   `sec_parser` has no 8-K support at all, so nothing to mirror there.
7. **`sigma_pre` "30-day" ambiguity** — trading days vs. calendar days — taken as trading
   days for the implementation (consistent with every other "Nd" window in the spec), and
   asked explicitly in `OPEN_QUESTIONS.md`.
8. **CONTRACTS.md schema** — we don't have vrp-research's; we define our own long-format
   schema (`date, ticker, <value columns>`) and ask them to reconcile, per the prompt's own
   instruction.
9. **`setup.py` package list** — sec_parser lists `packages=["sec"]` explicitly (single
   package, no sub-packages to enumerate). We have sub-packages (`sources`, `options`,
   `features`, `labels`, `events`); I'll enumerate all of them explicitly in `setup.py`
   rather than use `find_packages()`, for the closest parity with the reference repo's
   explicit style, while still covering every sub-package. Flagging only because it's a
   minor deviation (explicit list of 6 vs. sec_parser's list of 1) — no action needed unless
   you'd rather I use `find_packages()`.
10. **Hourly hourly-bar needs for the beta project** — the prompt says "Hourly variants if
    hourly bars are available," but doesn't say whether Polygon's plan tier includes
    intraday minute/hour aggregates or how far back. Built against the interface
    (`Ticker.hourly()`, `beta_ols_60d` computed on either daily or hourly bars via a
    `timeframe` argument), actual availability flagged as a question for the beta team.

## 10. Phase plan (unchanged from the prompt, restated for tracking)

1. PLAN.md — **you are here, stop for review**
2. Skeleton: layout, `config.py`, `setup.py`, CI, `.gitignore`, `sources/fakes.py`. `pytest`
   runs (near-empty), `ruff` passes.
3. Cache: full implementation + tests (eviction order, hits/misses, copy isolation,
   namespace separation, `simulate` hit-rate-increases-with-size).
4. Sources: Polygon stocks/indices, FRED, EDGAR, sec_parser adapter, calendar — all tested
   with mocked HTTP, no network in CI.
5. Options toolkit: OCC, Black-Scholes/IV, chain helpers, `iv30.py` stub.
6. Features, labels, PEAD event table + the generic no-lookahead test over the whole
   registry.
7. Precompute to parquet with merge-on-rerun and per-ticker failure report.
8. Docs: README, CONTRACTS.md, MAINTENANCE.md, OPEN_QUESTIONS.md.

Each phase ends with `pytest` and `ruff check` both green before moving to the next.

---

**Waiting for your review of this plan before writing any code.** If you'd like changes —
different package name, a different config-file format, dropping `pyyaml`, using
`find_packages()`, anything above — say so and I'll fold it in before Phase 2.
