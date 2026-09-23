# Phase 8 — Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the four docs `PLAN.md` §10 item 8 calls for — `README.md`, `CONTRACTS.md`,
`MAINTENANCE.md`, `OPEN_QUESTIONS.md` — that accurately describe the library **as it exists
today**, including the three layout gaps discovered while writing them (below), rather than
restating `PLAN.md`'s aspirational file layout as if it were all built.

**Architecture:** Each doc is a single Markdown file at the repo root (README.md,
CONTRACTS.md, MAINTENANCE.md, OPEN_QUESTIONS.md), plus one missing config example file
(`config/config.example.yaml`) that README/CONTRACTS both reference. No production code
changes. "Tests" for this phase are accuracy checks — every function signature, file path,
and feature/label name quoted in a doc is verified against the actual source file before the
task is marked done, and the full `pytest`/`ruff` suite is re-run at the end to confirm the
new files (config.example.yaml in particular) don't break anything.

**Tech Stack:** Markdown, YAML (`config.example.yaml`).

**Spec:** `PLAN.md` in full (this phase documents the whole library), specifically §2 (file
layout), §3 (public API), §4 (frame conventions), §5 (cache design), §6 (features/labels
enforcement), §7 (project-specific pieces), §8 (spec issues table), §9 (conflicts/
ambiguities), §10 item 8.

## Global Constraints

- Every claim in every doc must be verifiable against the actual current source code in this
  repo — never restate a `PLAN.md` aspiration as fact if the code doesn't back it up yet.
- Three layout items `PLAN.md` §2 calls for do not exist in the codebase as of this phase.
  Every doc that would otherwise imply they exist must instead say plainly that they don't,
  and point at `OPEN_QUESTIONS.md` for the tracking entry:
  1. `data_universe/ticker.py` (`Ticker` class) — never built in Phases 1-7. `__init__.py`
     does not import or export it.
  2. `data_universe/sources/flatfiles.py` (`FlatFilesSource`, S3 bulk download) — never built.
  3. `config/config.example.yaml` — never created. This phase creates it (Task 2), since
     `config.load_yaml()`'s default path (`config/config.yaml`) and its env-var-precedence
     behavior deserve a real example to point at.
- `__version__` in `data_universe/__init__.py` and `VERSION` in `setup.py` are both still
  `"0.1.0"` despite 7 phases of changes — `PLAN.md` §0 requires bumping both "on every
  change" (mirroring sec_parser's convention) and this has not happened once. `MAINTENANCE.md`
  must call this out as the standing rule going forward; this plan does not itself bump the
  version (no code changed, doc-only phase), but flags the accumulated gap in
  `OPEN_QUESTIONS.md`.
- Ruff: `line-length = 100`, `select = ["E", "F", "I"]` (`ruff.toml`) — doesn't apply to
  Markdown/YAML, but run `.venv/bin/ruff check .` and `.venv/bin/pytest -q` after Task 2 (the
  only task touching a file ruff/pytest could care about) and again at the end.
- Every commit ends with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

### Source-of-truth files each task must read before writing (do not guess from PLAN.md alone)

```
data_universe/__init__.py            actual public API surface
data_universe/config.py              actual config functions, env vars, YAML keys
data_universe/cache.py               actual Cache/get_cache/cached/simulate signatures
data_universe/calendar.py            actual sessions/next_session/previous_session/shift_sessions
data_universe/sources/base.py        actual DataSource ABC, bar frame schema, exceptions
data_universe/sources/polygon.py     actual PolygonSource methods
data_universe/sources/polygon_options.py   actual PolygonOptionsSource methods
data_universe/sources/fred.py        actual FredSource.daily_rate
data_universe/sources/edgar.py       actual EdgarSource methods
data_universe/sources/sec_adapter.py actual sp500()/fundamentals()
data_universe/sources/fakes.py       actual FakeMarket/FakePolygonSource
data_universe/options/occ.py         actual OccContract/parse/build
data_universe/options/black_scholes.py     actual price/vega/implied_vol
data_universe/options/chain.py       actual as_of/atm_strike/nearest_expiry_after/straddle
data_universe/options/iv30.py        actual stub + owner note
data_universe/features/core.py       actual pure-math functions
data_universe/features/registry.py   actual register()/compute()/FeatureSpec
data_universe/labels/registry.py     actual register()/compute()
data_universe/features/library.py    actual FEATURE_NAMES list (14 names)
data_universe/labels/library.py      actual LABEL_NAMES list (3 names)
data_universe/events/earnings.py     actual build_pead_events() + _EVENT_COLUMNS
data_universe/precompute.py          actual precompute()/load_feature()/PrecomputeReport
setup.py                             actual VERSION, install_requires, extras_require
ruff.toml, requirements.txt, .github/workflows/ci.yml
```

---

### Task 1: `README.md`

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: everything under "Source-of-truth files" above.
- Produces: nothing consumed by a later task, but Task 3 (CONTRACTS.md) and Task 5
  (OPEN_QUESTIONS.md) must not contradict what this task states about what's implemented.

- [ ] **Step 1: Read every file listed under "Source-of-truth files" above that isn't already
  fresh in context.** This is the actual verification step for this doc-only phase — there is
  no failing-test step, because there's no code under test; accuracy against source is the
  substitute discipline.

- [ ] **Step 2: Write `README.md`** with these sections, in order:

  1. **Title + one-paragraph description** — "GTSF Quant Sector shared data and feature
     library" (matches `setup.py`'s `DESCRIPTION`), who it's for (the three research
     projects: Intraday Beta Estimation, VRP research, PEAD).
  2. **Install** —
     ```
     pip install -e .
     # optional extras:
     pip install -e ".[sec]"   # sec_parser fundamentals/S&P membership adapter
     pip install -e ".[s3]"    # boto3, reserved for the not-yet-built flat-files source
     pip install -e ".[dev]"   # pytest, ruff
     ```
  3. **Quickstart** — a runnable example using only what actually exists today (no `Ticker`):
     ```python
     import data_universe as q
     from data_universe.sources.fakes import FakePolygonSource  # no API key needed

     source = FakePolygonSource()

     # Compute a registered feature directly against a source:
     from data_universe.features.registry import compute as compute_feature
     beta = compute_feature(q.FEATURE_REGISTRY, "beta_ols_60d", "AAPL",
                             "2024-01-01", "2024-06-01", source)

     # Or batch-compute to parquet and read it back later:
     report = q.precompute(["AAPL", "MSFT"], ["rv_cc_21d", "beta_ols_60d"],
                            "2024-01-01", "2024-06-01", out_dir="data/processed",
                            source=source)
     panel = q.load_feature("beta_ols_60d", tickers=["AAPL"])

     # With a real Polygon key instead of the fake:
     q.config.set_polygon_key("YOUR_KEY")
     q.config.set_project("beta")  # tags cache access-log rows
     q.get_cache().stats()
     ```
     Follow with one sentence: "`q.Ticker` from `PLAN.md`'s original sketch — a thin
     `aapl.feature(...)`/`aapl.daily(...)` façade over the pieces above — has not been built
     yet; see `OPEN_QUESTIONS.md`. Until then, call `features.registry.compute()` /
     `labels.registry.compute()` directly, or use `precompute()`/`load_feature()` for batch
     work."
  4. **What's here** — one subsection per module, each 1-3 sentences, naming the actual
     classes/functions (not paraphrased): `config.py`, `cache.py` (mention `@cached`,
     `simulate()`), `calendar.py`, `sources/` (list `PolygonSource`, `PolygonOptionsSource`,
     `FredSource`, `EdgarSource`, `sec_adapter`, `fakes.FakePolygonSource`/`FakeMarket`),
     `options/` (`occ`, `black_scholes`, `chain`, `iv30` stub), `features/` + `labels/`
     (name all 14 features from `features/library.py`'s `FEATURE_NAMES` and all 3 labels from
     `labels/library.py`'s `LABEL_NAMES`, verbatim), `events/earnings.py`
     (`build_pead_events`), `precompute.py` (`precompute`, `load_feature`,
     `PrecomputeReport`).
  5. **Frame conventions** — summarize `sources/base.py`'s module docstring (bar frame schema:
     tz-aware `America/New_York` `DatetimeIndex` named `timestamp`, columns
     `open,high,low,close,volume,vwap`; daily feature/label series: tz-naive index named
     `date`; adjusted-by-default price convention).
  6. **Configuration** — table of every env var from `config.py`'s `_ENV_VARS`
     (`POLYGON_API_KEY`, `POLYGON_BASE_URL`, `POLYGON_FLATFILES_ENDPOINT`,
     `POLYGON_S3_ACCESS_KEY`, `POLYGON_S3_SECRET_KEY`, `FRED_API_KEY`,
     `DATA_UNIVERSE_DATA_DIR`, `DATA_UNIVERSE_CACHE_SIZE`), precedence
     (env var > `config/config.yaml` > hardcoded default), and a pointer to
     `config/config.example.yaml` (created in Task 2) as the file to copy to
     `config/config.yaml` and fill in.
  7. **Not yet implemented** — bulleted, matching the three items in this plan's Global
     Constraints section verbatim (`Ticker`, `FlatFilesSource`, and — cross this one off,
     since Task 2 builds it — do NOT list `config.example.yaml` here once Task 2 is done;
     list only `Ticker` and `FlatFilesSource`), each with one sentence on the current
     workaround and a pointer to `OPEN_QUESTIONS.md`.
  8. **Testing** —
     ```
     pip install -e ".[dev]"
     pytest
     ruff check .
     ```
     One sentence: the full suite runs against `sources/fakes.py`'s deterministic
     `FakeMarket`/`FakePolygonSource`, never live APIs, so CI needs no secrets (matches
     `.github/workflows/ci.yml`).
  9. **See also** — one-line pointers to `PLAN.md` (original spec), `CONTRACTS.md` (schemas
     for downstream teams), `MAINTENANCE.md`, `OPEN_QUESTIONS.md`.

- [ ] **Step 3: Verify every quoted name against source.** Grep-check the feature/label lists
  and every function name mentioned:

```bash
grep -n "^FEATURE_NAMES" -A 16 data_universe/features/library.py
grep -n "^LABEL_NAMES" data_universe/labels/library.py
grep -n "^_ENV_VARS" -A 10 data_universe/config.py
```

  Confirm the README's lists match these outputs exactly (14 feature names, 3 label names, 8
  env vars). Fix any mismatch before continuing.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: add README covering the library as it actually exists today"
```

---

### Task 2: `config/config.example.yaml`

**Files:**
- Create: `config/config.example.yaml`

**Interfaces:**
- Consumes: `data_universe/config.py`'s `_YAML_KEYS` mapping (the authoritative list of
  top-level keys `load_yaml()` understands).
- Produces: the file `README.md` (Task 1) already points to.

- [ ] **Step 1: Re-read `data_universe/config.py`'s `_YAML_KEYS` dict** to get the exact set
  of top-level keys `load_yaml()` reads (currently: `polygon_api_key`, `polygon_base_url`,
  `polygon_flatfiles_endpoint`, `polygon_s3_access_key`, `polygon_s3_secret_key`,
  `fred_api_key`, `data_dir`, `cache_size`).

- [ ] **Step 2: Write the file**

```yaml
# Copy this file to config/config.yaml and fill in real values.
# config/config.yaml is gitignored -- never commit real credentials.
# Every key here can also be set via an environment variable (see README.md
# "Configuration"); an environment variable always overrides this file.

polygon_api_key: ""
polygon_base_url: "https://api.polygon.io"
polygon_flatfiles_endpoint: ""
polygon_s3_access_key: ""
polygon_s3_secret_key: ""
fred_api_key: ""
data_dir: "data"
cache_size: 256
```

- [ ] **Step 3: Verify it loads without error**

```bash
mkdir -p /tmp/du_config_check
cp config/config.example.yaml /tmp/du_config_check/config.yaml
.venv/bin/python -c "
from data_universe import config
config.load_yaml('/tmp/du_config_check/config.yaml')
print(config.get_polygon_base_url(), config.get_cache_size(), config.get_data_dir())
"
rm -rf /tmp/du_config_check
```

Expected output: `https://api.polygon.io 256 data` (empty-string keys silently overwrite
defaults with `""`, which is fine for an example file that's never loaded with real values in
CI — this is expected, not a bug to fix).

- [ ] **Step 4: Run the full suite to confirm nothing else changed behavior**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Expected: 224 passed, all checks passed (this task adds a YAML file pytest/ruff don't touch,
so this is a no-op confirmation, not expected to catch anything — run it anyway per the
Global Constraints).

- [ ] **Step 5: Commit**

```bash
git add config/config.example.yaml
git commit -m "docs: add config/config.example.yaml (PLAN.md layout item, never created in Phase 2)"
```

---

### Task 3: `CONTRACTS.md`

**Files:**
- Create: `CONTRACTS.md`

**Interfaces:**
- Consumes: `sources/base.py` (bar frame schema), `features/registry.py`/`labels/registry.py`
  (`compute()` output schema), `precompute.py` (parquet output schema), `events/earnings.py`
  (`_EVENT_COLUMNS`), `config.py` (env vars).

- [ ] **Step 1: Re-read `data_universe/events/earnings.py`'s `_EVENT_COLUMNS` list and
  `data_universe/precompute.py`'s `precompute`/`load_feature` docstrings** to quote their
  schemas exactly.

- [ ] **Step 2: Write `CONTRACTS.md`** — this is the schema-reconciliation doc `PLAN.md` §9
  item 8 asks for ("we define our own long-format schema... and ask [other teams] to
  reconcile"), so open with one sentence framing it that way. Sections:

  1. **Bar frames** (`sources/base.py`): tz-aware `America/New_York` `DatetimeIndex` named
     `timestamp`; columns `open, high, low, close, volume, vwap`, all `float64`; daily bars
     stamped at midnight of the session date; intraday bars filtered to regular trading hours
     (9:30-16:00 ET) unless the caller opts out; prices split/dividend-adjusted unless the
     function name says "raw"/"unadjusted".
  2. **Feature/label series** (`features/registry.py` and `labels/registry.py`'s shared
     `compute()`): a `pd.Series` indexed by tz-naive `date`, sliced to exactly
     `[start, end]` inclusive; NaN only for a real data gap or (for labels) proximity to
     `end`, never for insufficient warmup history. Every label name starts with `fwd_`;
     every feature name is guaranteed not to.
  3. **PEAD event table** (`events/earnings.py`'s `build_pead_events`): one row per qualifying
     8-K Item 2.02 filing, columns exactly as in `_EVENT_COLUMNS` — list all 13 column names
     verbatim with a one-line meaning each (pull the meanings from the module docstring's
     spec-issue table, items 1/2/4/5/6). State explicitly that `next_earnings_session` is a
     realized value known only in hindsight, never a forecast, and that a filing too close to
     a calendar/price-history edge is silently skipped (not raised on, not present as a
     null row).
  4. **Precompute parquet output** (`precompute.py`): `<out_dir>/<feature_or_label_name>.parquet`
     (default `out_dir`: `<config.get_data_dir()>/processed`), long format, columns exactly
     `date, ticker, value`; reruns upsert on `(date, ticker)`; rows with a NaN value are never
     written. `load_feature(name, tickers=None, out_dir=None)` reads this back, raising
     `FileNotFoundError` if `name` was never precomputed to that `out_dir`.
  5. **Config precedence** (`config.py`): env var > `config/config.yaml` > hardcoded default;
     list the 8 env vars from `_ENV_VARS` in a table with their purpose.
  6. **Known gaps other teams should not build against yet** — one line each: `Ticker`
     (not built), `FlatFilesSource` (not built), single-name IV30 / `vrp_signal_rv21` and
     `fwd_vrp_expost` for any ticker other than SPY (raise `NotImplementedError` via the
     `options/iv30.py` stub), indices access for `I:VIX` (entitlement unconfirmed, see
     `OPEN_QUESTIONS.md`).

- [ ] **Step 3: Verify the PEAD column list against source**

```bash
grep -n "_EVENT_COLUMNS" -A 5 data_universe/events/earnings.py
```

  Confirm all 13 names in the doc match this output exactly, in the same order.

- [ ] **Step 4: Commit**

```bash
git add CONTRACTS.md
git commit -m "docs: add CONTRACTS.md schema reference for downstream project teams"
```

---

### Task 4: `MAINTENANCE.md`

**Files:**
- Create: `MAINTENANCE.md`

**Interfaces:**
- Consumes: `setup.py` (`VERSION`), `data_universe/__init__.py` (`__version__`), `PLAN.md` §0
  (the sec_parser version-bump convention this mirrors).

- [ ] **Step 1: Confirm the current version drift** —

```bash
grep VERSION setup.py
grep __version__ data_universe/__init__.py
git log --oneline | wc -l
```

  Both are `"0.1.0"` despite the commit count above 1 — this is the fact Step 2's first entry
  documents.

- [ ] **Step 2: Write `MAINTENANCE.md`**, mirroring sec_parser's one-paragraph-per-task style
  (per `PLAN.md` §0's description of `sec/data/sp500/README.md`). Entries:

  1. **Bump the version on every change.** `setup.py`'s `VERSION` and
     `data_universe/__init__.py`'s `__version__` must be kept equal and incremented on every
     merged change — `pip install -e .` / a fresh `git+https://...` install will not pick up
     code changes otherwise, since pip caches by version. **Status: not currently followed —
     both are still `"0.1.0"` after 7 implementation phases.** Whoever picks this up next
     should bump to `"0.2.0"` (call it out as a real fix, not just a note) and start
     following it from there.
  2. **S&P 500 historical membership.** Point-in-time membership
     (`sources.sec_adapter.sp500(query_date=...)`) is a pass-through to `sec_parser`'s
     `lookups.get_sp500_tickers()`, which reads a manually-maintained
     `sp500_historical.csv` in the `sec_parser` repo, not this one. This repo has no
     maintenance task of its own here — the task (and its "steps below should be completed
     monthly" note) lives in `sec_parser`'s own `sec/data/sp500/README.md`. Just don't
     duplicate that file's maintenance burden into this repo.
  3. **API keys.** Polygon, FRED, and (once built) Polygon S3 flat-files credentials are
     read from environment variables or `config/config.yaml` (gitignored) — never committed,
     never logged (see `config.py`'s module docstring). Rotate them per each project's own
     security policy; this repo has no key-rotation automation.
  4. **Dependency floors.** `install_requires` in `setup.py` currently pins no upper or lower
     bounds beyond `python_requires=">=3.9"`. If a future `pandas`/`numpy` release breaks a
     test, pin the floor that fixes it in `setup.py`, not just in a contributor's local venv.
  5. **CI.** `.github/workflows/ci.yml` runs `pytest` + `ruff check .` on every push/PR against
     Python 3.11, using only `sources/fakes.py` (no live API calls, no secrets required).
     If a new source module is added, keep it testable against a fake/mock the same way —
     don't make CI depend on a real API key.

- [ ] **Step 3: Commit**

```bash
git add MAINTENANCE.md
git commit -m "docs: add MAINTENANCE.md, including the standing version-bump gap"
```

---

### Task 5: `OPEN_QUESTIONS.md`

**Files:**
- Create: `OPEN_QUESTIONS.md`

**Interfaces:**
- Consumes: `PLAN.md` §8 (spec issues table) and §9 (conflicts/ambiguities), plus every
  `OPEN_QUESTIONS.md`-forward-reference already written into source docstrings across the
  codebase (grep for them — Step 1).

- [ ] **Step 1: Collect every existing forward-reference to this file**, so nothing already
  flagged in code gets missed:

```bash
grep -rn "OPEN_QUESTIONS" data_universe/ | grep -v __pycache__
```

  Expected hits include (at least): `sources/polygon.py` (indices access for `I:VIX`),
  `sources/fred.py` (day-count/discount-vs-yield conversion), `sources/edgar.py`
  (submissions JSON shape unverified), `sources/polygon_options.py` (endpoint paths
  unverified), `events/earnings.py`'s docstring cross-references, `PLAN.md` itself.

- [ ] **Step 2: Write `OPEN_QUESTIONS.md`** as a table or one-entry-per-question list, each
  entry with: **Question**, **Why it matters**, **Current stance** (what the code actually
  does today, e.g. "raises `NotImplementedError`" or "assumed X, unverified"), **Owner /
  next step**. Cover, at minimum, every one of these (numbers match `PLAN.md` §9 where that
  section already numbered them, so cross-references stay stable):

  1. Package/import name (`data_universe` vs. e.g. `gtsf_quant`) — resolved as `data_universe`,
     flagged only in case a rename is still wanted.
  2. `pyyaml` added as a core dependency for `config/config.yaml` support — flagged in case a
     different format (TOML/env-only) is preferred.
  2b. `FRED_API_KEY` / `config.set_fred_key()` added; not in the original prompt's env var
      list.
  3. Options Business tier / NBBO — building against daily-close-only; full trades/quotes
     not attempted.
  4. `I:VIX` indices access entitlement unconfirmed — `iv_index` feature (Phase 6) depends on
     it for SPY.
  5. Polygon endpoint paths not re-verified against live docs: stock/index aggregates,
     options contract reference, options daily aggregates, flat-files S3 hostname (no
     hardcoded default — must be set via `POLYGON_FLATFILES_ENDPOINT`, and note that
     `FlatFilesSource` itself was never built, see item 12 below).
  6. EDGAR submissions JSON shape (`filings.recent.items` field carrying which Items are
     checked) not re-verified against live docs.
  7. `sigma_pre` "30-day" window taken as 30 **trading** days, not calendar days — see
     `events/earnings.py`'s `sigma_pre_daily` column.
  7b. FRED `DTB3` day-count/conversion: currently `(value/100)/252`; whether a
      bank-discount-to-bond-equivalent-yield conversion should happen first is unresolved
      (see `sources/fred.py` module docstring).
  8. `CONTRACTS.md`'s schemas are this repo's own definition, not vrp-research's — ask them
     to reconcile (this is why `CONTRACTS.md`, Task 3, exists).
  9. `setup.py` package list is explicit (`packages=[...]`, 6 entries) rather than
     `find_packages()` — minor style deviation from `sec_parser`, flagged only, no action
     needed.
  10. Hourly-bar availability for the beta project's intraday variants — built against the
      `hourly_bars()` interface; actual Polygon plan-tier availability unconfirmed.
  11. **`Ticker` (PLAN.md §3) was never built.** `data_universe/ticker.py` does not exist;
      `__init__.py` does not export `Ticker`. Every PLAN.md README example line using
      `q.Ticker(...)` does not work today. Workaround: call
      `features.registry.compute()`/`labels.registry.compute()` directly with a `source`, or
      use `precompute()`/`load_feature()`. Owner/next step: a future phase, if requested.
  12. **`FlatFilesSource` (`sources/flatfiles.py`, PLAN.md §2) was never built.** No S3 bulk
      download or `download_options_day_aggs()` exists. `POLYGON_FLATFILES_ENDPOINT`,
      `POLYGON_S3_ACCESS_KEY`, `POLYGON_S3_SECRET_KEY` are read by `config.py` but nothing
      consumes them yet.
  13. **Version-bump discipline (PLAN.md §0) has not been followed.** `setup.py`'s `VERSION`
      and `__init__.py`'s `__version__` are both still `"0.1.0"` after 7 implementation
      phases. See `MAINTENANCE.md` item 1.
  14. **`vrp_signal_rv21` / `fwd_vrp_expost` for non-SPY tickers always raise
      `NotImplementedError`** (via the `options/iv30.py` stub) — registered per PLAN.md §6
      ("provide interfaces and stubs") but unusable for any ticker but SPY until IV30 lands.
      Owner: vrp-research data track (per `options/iv30.py`'s own docstring).

- [ ] **Step 3: Cross-check every numbered item against its source file** using the grep from
  Step 1 plus a final read of `PLAN.md` §8/§9, to make sure no existing flagged item was
  dropped and no new claim contradicts the actual code.

- [ ] **Step 4: Commit**

```bash
git add OPEN_QUESTIONS.md
git commit -m "docs: consolidate every flagged spec ambiguity and implementation gap into OPEN_QUESTIONS.md"
```

---

### Task 6: Final cross-doc consistency pass

**Files:**
- Modify (only if an inconsistency is found): `README.md`, `CONTRACTS.md`,
  `MAINTENANCE.md`, `OPEN_QUESTIONS.md`

**Interfaces:**
- Consumes: all four docs from Tasks 1, 3, 4, 5.

- [ ] **Step 1: Read all four docs back-to-back** and check for contradictions: does README's
  "Not yet implemented" list match OPEN_QUESTIONS items 11-12 exactly? Does CONTRACTS.md's
  PEAD column list match the grep output from Task 3 Step 3? Does MAINTENANCE.md's version
  gap match OPEN_QUESTIONS item 13's wording (same fact, don't let them drift)?

- [ ] **Step 2: Fix any inconsistency found inline.** If none is found, this step is a no-op —
  state that explicitly rather than skipping the read.

- [ ] **Step 3: Run the full suite one final time**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Expected: 224 passed, all checks passed — confirms the doc-only phase left the codebase
exactly as Phase 7 left it, plus one new example YAML file.

- [ ] **Step 4: Commit (only if Step 2 made changes)**

```bash
git add README.md CONTRACTS.md MAINTENANCE.md OPEN_QUESTIONS.md
git commit -m "docs: fix cross-doc inconsistencies found in final Phase 8 review"
```

---

## Self-Review Notes (already folded into the tasks above)

- **Spec coverage:** `PLAN.md` §10 item 8's four files are Tasks 1/3/4/5; the missing
  `config/config.example.yaml` layout item is Task 2 (a genuine gap discovered while planning
  this phase, not scope creep — README/CONTRACTS both need to point somewhere real).
- **No placeholders:** every doc task specifies exact section list and exact content to check
  against source, not "document appropriately."
- **Consistency:** Task 6 exists specifically to catch drift between docs written in separate
  tasks (e.g. the "not yet implemented" list appearing in both README and OPEN_QUESTIONS.md
  must agree).
