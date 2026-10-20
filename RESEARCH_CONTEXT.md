# Research context for shared precomputation

Recorded 2026-10-20. This is durable context for extending `data-universe` based on
reviews of three downstream research repositories. This document is an inventory and proposed
implementation direction, not a claim that the pipelines below are complete.

## Objective and evidence

Make `data-universe` fetch shared inputs once, compute expensive derived datasets
once, and serve reusable, versioned panels to the research projects.

Reviewed the following local checkouts. Paths below each repository's section are
relative to that repository; the linked commits identify the reviewed versions.

| Repository | Reviewed commit | Evidence inspected |
|---|---|---|
| [vrp-research](https://github.com/GTSF-Quantitative-Sector/vrp-research) | [d332f92](https://github.com/GTSF-Quantitative-Sector/vrp-research/tree/d332f92) | README, CONTRACTS, research log, all three data notebooks, realized-volatility app, evaluation harness |
| [xsec-gnn-ranker](https://github.com/GTSF-Quantitative-Sector/xsec-gnn-ranker) | [25d15b4](https://github.com/GTSF-Quantitative-Sector/xsec-gnn-ranker/tree/25d15b4) | README, component guidance, contracts, research protocol, plans, data/graph/training code, selected tests |
| [Intraday-Beta-Estimation-Strategy](https://github.com/GTSF-Quantitative-Sector/Intraday-Beta-Estimation-Strategy) | [1198d7a](https://github.com/GTSF-Quantitative-Sector/Intraday-Beta-Estimation-Strategy/tree/1198d7a) | README, both notebooks, sample raw and cleaned CSVs |
| [data-universe](https://github.com/GTSF-Quantitative-Sector/data-universe) | [8f3e157](https://github.com/GTSF-Quantitative-Sector/data-universe/tree/8f3e157) | Sources, options toolkit, registries, precompute, PLAN and OPEN_QUESTIONS |

Notebook source was inspected; no live research pipeline or paid API job was run.
Saved notebook prose/timings are author reports, not independently verified results.

## Findings at a glance

Massive provides market observations and some reference/analytics inputs. The shared
library should turn those inputs into validated, reusable research datasets.

| Project | Main expensive work | Current status |
|---|---|---|
| VRP | Historical contract discovery/prices, IV inversion, constant-maturity ATM IV30 | Notebook prototype with correctness issues; shared IV30 is a stub |
| GNN ranker | Rolling correlations and graph snapshots; later market residualization, graphical lasso and factor betas | Small ETF correlation graph implemented; larger graph methods planned |
| Beta estimation | Repeated Kalman calibration/filter runs and state uncertainty | Daily-data notebook implementations; causal construction needs correction in several paths |

Download shared inputs once, preserve their provenance, precompute expensive
derivatives, and let each project retain its own model fitting and evaluation.

## What Massive supplies versus what we need to derive

The following describes the documented product, not verified account entitlements.

| Input or output | Provider boundary | Shared-library implication |
|---|---|---|
| Equity OHLCV | REST supplies raw or split-adjusted bars; not dividend-adjusted bars [1] | Preserve raw OHLCV and corporate actions; explicitly construct and version the research adjustment convention. |
| Historical option contract metadata | Reference API supports `as_of`, expiration/strike filters, and pagination [2] | Build historical chains from reference data, rather than treating a current snapshot as a historical chain. |
| Historical option prices | Contract OHLCV aggregates are available, with documented history back to June 2, 2014; no qualifying trades means no bar [3] | Store missingness; a daily trade close is not a synchronized bid/ask midpoint. |
| Bulk option daily bars | S3 daily aggregate files cover the market [4] | Prefer bulk ingestion for broad backfills, then partition locally by date/underlying/contract. |
| IV, Greeks, open interest | Option-chain snapshot includes current analytics; its documented query has no historical observation-date parameter [5] | Historical contract IV and constant-maturity ATM IV30 need reconstruction for this workflow. Historical Greeks could be derived later but are not implemented in the research repo. |
| Historical open interest | Massive says historical OI is unavailable; snapshots expose the latest daily value [6] | Archive daily snapshots going forward if needed; price history cannot reconstruct past OI. |
| Trades and NBBO quotes | Separate datasets and plan-dependent access; Business pages distinguish feed expansions and differing history lengths [7] | Verify account access and history per dataset before choosing midpoint-based historical IV. Do not assume quotes cover the entire 2018–2024 study. |
| Treasury yields | Daily yield history across standard maturities is available [10] | Choose and document the rate series, availability time and per-period conversion; the beta notebook currently uses a constant instead. |
| Financial statements | Statements include filing dates, but current endpoints do not preserve every historical filing version [11] | Retrieve original filings when original reported values are needed. Do not attach today's restated figures to an earlier publication date. |
| Technical indicators | Massive offers SMA, EMA, MACD and RSI [12] | These do not replace the exact momentum, volatility, correlation and filter definitions in these projects. |
| RV, forecasts, VRP and research labels | The inspected market-data endpoints do not supply these project-specific constructions | Compute from the inputs using explicit estimator, horizon, adjustment and availability conventions. |
| Correlation networks and dynamic beta | The inspected APIs do not supply these projects' configured graphs, Kalman states or calibrated models | Compute from prices with versioned windows, parameters, fitting cutoffs and universe definitions. |
| Historical index membership, entity continuity, historical GICS, earnings-window flags, delisting returns | These are required research joins, not established by the inspected price/snapshot endpoints | Resolve and validate sources separately. A current constituent list, inactive ticker flag or filing date does not complete these datasets. |
| BEA mappings and customer/supplier relationships | Coverage of the planned graph's required datasets was not established in this audit | Source BEA tables/concordances and the required relationship/segment disclosures separately; availability must be resolved before building the graph. |

The README's broad “options history back to 2014” must be interpreted by dataset:
aggregate/reference coverage does not establish equally long quote history.
The current Business page contains ambiguous language about base versus expanded
trade/quote access. Account-specific access remains unverified.

## Computations actually present in vrp-research

### Historical ATM IV30 prototype — main expensive computation

Source: `src/vrp/data/IVCalculator.ipynb`, code cell 11, `inv_black_scholes`.

1. Fetch underlying daily close.
2. List option contracts expiring 23–37 calendar days after the observation date.
3. Keep strikes within 95–105% of spot.
4. Select nearest-to-spot calls and puts on each side of the 30-day target.
5. Invert Black–Scholes on historical contract closes for contract-level IV.
6. For calls and puts separately, interpolate **total variance** to 30 days:
   `IV30 = sqrt((w*sigma_near^2*T_near + (1-w)*sigma_far^2*T_far) / T30)`.
7. Average call and put IV30; export per-date/per-ticker results to CSV.

This is a prototype, not a validated panel builder. Its annual batch loops over
20 hardcoded names and calendar dates in 2024, with a constant risk-free rate of
5%. The notebook reports about 67 seconds for a week of 20 names and estimates
about an hour for a year. Those figures are not benchmarks for a corrected pipeline.

**Correctness issues to fix before migrating or caching its outputs:**

- Far-call price is assigned `adac_price = bdac_price`, and far-put price is
  `adap_price = bdap_price`; neither far contract's actual price is fetched.
- Underlying bars default to split-adjusted history while strikes describe actual
  historical contracts. Historical option inversion needs consistent spot/strike/
  premium units; raw contemporaneous spot is the natural starting point.
- Contract lookup omits `as_of` and uses a capped number of pages. The code does
  not explicitly select the nearest near and far expirations before choosing
  strikes; row order and proximity to spot can select different maturities.
- Exact 30-day expirations are excluded by strict `<` and `>` tests. Empty
  call/put subsets can also fail before the intended missing-result fallback.
- Requests include the following day and use the first row without validating
  its session. A missing observation-day bar must not become a next-day price.
- The prose mentions midpoint prices, but the inversion actually uses daily
  closes. Spread/staleness validation is absent; `quality` is only `OK` or `Bad`.
- Output `IV_30`, quality spelling and `n_contracts` (all filtered candidates,
  rather than contracts surviving pricing/quality checks) differ from CONTRACTS.
- Dividend inputs, exercise-style handling and American-option approximation
  policy are missing. The shared Black–Scholes solver is explicitly European;
  any use for American contracts must have a recorded approximation policy.

Source: `src/vrp/data/retrieveData.ipynb` also implements raw/adjusted equity pulls,
historical contract lookup using `as_of`, and bracketing-expiry annotation.
It does not finish IV30. It omits contract pagination, and its example passes a
list of underlyings to a function written for one ticker. Its `Adj_Close` is only
the API's split-adjusted close, despite the dividend-adjusted research contract.

### Realized volatility — implemented

Source: `notebooks/realized_volatility.py`.

- `log_return = log(close).diff()`.
- `rv_close_close_21`: rolling 21-observation sample standard deviation of log
  returns (`ddof=1`), annualized by `sqrt(252)`.
- `parkinson_daily_component = log(high/low)^2`.
- `rv_parkinson_21`: square root of the 21-day mean component divided by
  `4*log(2)`, annualized by `sqrt(252)`.

The app reads local CSV `close`; it does not establish the input's adjustment
provenance. These computations are cheap compared with historical options I/O.

### Other implemented derivatives

Source: `IVCalculator.ipynb`, code cell 9, `get_option_chain` (current snapshots).

`dte`, strike distance from spot, intrinsic value, nonnegative-clipped extrinsic
value, volume/OI ratio, contract notional, delta-adjusted exposure, ATM/ITM/OTM
classification. Premium preference is FMV, then midpoint, then close; ATM is
within 1% of spot. These are helper/demo fields, not historical research panels.
Persist price provenance if reusing them; do not mix fallback price types silently.

Source: `src/vrp/eval/harness.py`.

- `forward_returns_from_prices`: simple forward returns for 1, 5, 21, 63 sessions.
- `momentum_signal`: 12–1 momentum, `adj_close[t-21]/adj_close[t-252] - 1`.

These currently group via ticker columns, not stable entity identity. Forward
returns do not implement `delist_ret`; the CSV loader silently uses `close` if
`adj_close` is absent. Those shortcuts do not meet the full research contract.
Rank IC, Newey–West errors, quantile spreads, leave-one-out stability and synthetic
studies are evaluation outputs; keep them downstream of the shared market panels.

### Requested in VRP CONTRACTS, but not yet implemented in the inspected research code

- Garman–Klass and Yang–Zhang RV, forward 21-session RV, and a fitted RV forecast.
- Production `vrp_signal = iv30 - rv_cc_21` (primary), forecast variants, and
  `vrp_expost = iv30 - rv_fwd_21`.
- Point-in-time S&P 500 membership, stable `entity_id`, historical GICS sector,
  earnings in the next 30 calendar days / previous 21 sessions, and delisting returns.
- Fully validated IV30 quality categories and a contract-compliant processed panel.

Preserve the agreed research semantics: sorted `(date, ticker)` panels; date is
normalized and tz-naive; annualized vol is decimal; missing values remain NaN.
Compute longitudinal windows per entity, including ticker changes. Primary trading
score is **negative** VRP. `rv_fwd_21`, `vrp_expost`, and forward returns are labels,
never available-at-t signals. In-sample is 2018–2024; holdout is 2025–2026.

## Computations in xsec-gnn-ranker

### Implemented data and features

- `src/gnns/data/massive.py` retrieves split-adjusted daily bars and dividend
  `historical_adjustment_factor` records. `src/gnns/data/store.py::adjust` multiplies
  OHLC by the factor of the first ex-dividend date after the bar date, leaving volume
  split-adjusted. This follows Massive's documented factor convention [8] and is
  functionality the inspected `data-universe` source lacks.
- `src/gnns/data/features.py` computes `momentum_20`, `momentum_60` and
  `volatility_20` under the current construction config. Momentum is
  `close[t]/close[t-window] - 1`. Volatility is rolling sample standard deviation
  of **simple** one-bar returns, without annualization.
- Windows use a shared bar axis with missing observations represented as nulls.
  Eligibility uses configured history length, then missing feature values receive
  the contemporaneous cross-sectional median. Normalization happens later in
  `src/gnns/train/dataset.py`, across nodes of the same type at each timestamp.
- `src/gnns/data/labels.py` computes next-open-to-future-open returns:
  `open[i+1+h]/open[i+1] - 1`, with `label_end` for purging. Missing entry/exit
  bars produce no label. These labels are not the VRP close-to-close labels.

Data builds download full histories and persist validated Parquet plus a manifest
with build time, code version and file hash. Shared raw inputs and versioned
adjusted views can reduce repeated downloads across universes/projects.

### Implemented rolling correlation graph

Sources: `src/gnns/graphs/sector_etf.py`, `stats.py`, `build.py`.

For each configured snapshot date, the builder pivots recent closes, computes
60 simple returns, and calculates gap-aware pairwise Pearson correlations and
shared-observation counts. Pairs need at least three shared returns and finite
correlation. Each accepted pair produces both directed edges; edge weight is
absolute correlation and the edge attribute retains its sign.

Each training run rebuilds features and snapshots in memory. Snapshot `.npz`
save/load exists in `src/gnns/data/snapshot.py`, but the build pipeline does not
use a persistent snapshot cache. The plan reports roughly four seconds for 818
snapshots with 9–11 ETFs; this is not currently a demonstrated large bottleneck.
At 1,500 nodes a fully connected graph has 2,248,500 directed non-self edges per
snapshot, so both compute and storage need attention before scaling.

Persist raw correlation/count outputs and sparse edge lists where useful. Cache
complete snapshots by construction config, ordered node identities, dataset hash,
timestamp and code version. Separate reusable relationships from node features so
changing a model feature need not rerun correlation estimation.

### Planned expensive constructions — not implemented

Source: `docs/project_overview.md`; only `sector_etf_correlation` is registered in
the inspected builder registry.

- **Static partial-correlation graph:** market-residualize returns, fit graphical
  lasso over rolling 60–90-session windows, derive sparse partial correlations,
  and export top-N neighbor lists. This is the main future graph precompute target.
  Cache by residualization recipe, estimation window, regularization and universe.
- **Macro-sensitivity graph:** rolling regressions of stock returns on changes in
  macro factors, keeping signed coefficients and their magnitudes. Input series,
  transformations and release times must be established first.
- **Supply-chain graph:** BEA industry concordances, segment-revenue-weighted
  commodity exposures, customer/supplier links and multi-hop relationships.
  Cache structural exposures per source release; update dynamic commodity-shock
  components separately. Required source access is unresolved by this review.
- **Optional filings NLP:** exposure extraction can be cached per filing and
  extraction-model version, using the filing's availability time.
- **Learned lead-lag network:** temporal encoders and attention depend on fitted
  weights. Share input windows and causal candidate-neighbor lists; keep learned
  edges and model fitting with the experiment rather than a universal feature table.

The current fixed ticker lists do not implement a historical equity universe or
stable security identity. Intraday 1m/65m support is planned; the data builder only
implements 1d. Keep universe-dependent median filling and normalization out of
universal raw feature caches, or include the universe and policies in the cache key.
The GNN held-out period starts in 2024, independently of VRP's 2025–2026 holdout.

## Computations in Intraday-Beta-Estimation-Strategy

The inspected implementation uses **daily bars**, despite the project's name.
`data_pipeline.ipynb` downloads stocks and SPY for COVID, tariff and 2008 crisis
periods. An hourly-data docstring does not match the function's `timespan='day'`
default. Massive supports intraday bars [9]; moving this workflow intraday still
requires explicit session, timestamp, alignment and annualization policies.

### Implemented estimators

Source: `KF_beta_single_security.ipynb`.

| Output | Construction | Shared precompute role |
|---|---|---|
| Log returns and optional excess returns | `log(close[t]/close[t-1])`, optionally minus constant annual bps divided by periods/year | Reusable inputs, with return/rate conventions recorded |
| Static alpha/beta | OLS against SPY over the selected sample | Calibration/baseline tied to a fitting cutoff; not a universal historical feature |
| Rolling OLS beta | 60-period rolling covariance with SPY divided by SPY variance | Direct overlap with existing shared OLS features |
| Random-walk Kalman alpha/beta | State `[alpha, beta]`, identity transition, diagonal process noise, observation noise | Parameterized sequential feature family |
| Mean-reverting Kalman alpha/beta | Beta transition `mu + phi*(beta-mu)` | Separate estimator with explicit long-run mean and persistence |
| State covariance | A 2×2 posterior covariance matrix per observation | Persist uncertainty and final state for incremental continuation |
| Conditional return estimates/innovations | Prior alpha/beta combined with the current SPY return; observed minus predicted return | Residual features only with correct availability semantics |

The random-walk grid evaluates 4 alpha-noise × 5 beta-noise × 3 observation-noise
scales = **60 combinations**. The mean-reversion grid evaluates 5 persistence ×
4 beta-noise × 2 observation-noise values = **40 combinations**. The notebook also
runs filters across securities, with per-security OLS residual variance used for
observation noise. These are repeated computations worth caching as the universe,
frequency and parameter search grow; no runtime benchmark was performed here.

For fixed parameters, persist the last posterior state and covariance, then update
once per new bar. Replaying all history is unnecessary unless inputs or calibration
change. Calibration/search results must be keyed by training window, not just ticker.
RMSE, R-squared, residual normality/autocorrelation/ARCH tests and plots remain
research diagnostics rather than core shared market datasets.

### Correctness issues before migration

- Cleaning uses `ffill().bfill()` for OHLCV, creating observations outside actual
  coverage and copying volume. Inspected Visa raw data starts 2008-03-19, but its
  cleaned file starts 2007-08-01. GM's raw crisis-period file ends 2009-06-01,
  while cleaning repeats its prices/volume through 2009-06-30. Preserve genuine
  missingness, entity lifetimes and flags instead of caching these synthetic bars.
- Default filter initialization fits OLS across the entire supplied series.
  Validation supplies train+validation; another test path supplies the full sample.
  Those initial states contain future information. Shifting states one bar does
  not remove it. Initialize on a prior calibration window or a documented fixed prior.
- Some observation-noise estimates use full-sample residuals; the mean-reverting
  grid scores the full supplied sample. Keep parameter selection confined to the
  proper calibration/validation periods. A separate OOS mean-reverting routine does
  initialize on training data only; not every notebook path has the same problem.
- The so-called one-step-ahead estimate uses **current** SPY return. It can describe
  conditional stock return before observing the stock return, but is not available
  before that bar starts. Store prior and posterior states with distinct meanings.
- Epoch timestamps are converted without explicit UTC localization, leaving naive
  times such as 04:00. Convert from UTC to exchange time and define availability
  at bar completion before joining these features to other projects.
- The notebook uses split-adjusted closes without a dividend-adjustment step.
  Risk-free settings also change between cells. Record resolved conventions and
  fitting windows rather than relying on mutable notebook globals.

## Cross-project compatibility requirements

| Dimension | VRP | GNN | Beta estimation |
|---|---|---|---|
| Returns | Simple forward labels; log returns for CC RV | Simple feature/graph returns | Log returns, optionally excess |
| Volatility | Annualized, 21-session estimators | Unannualized rolling simple-return standard deviation | Residual variance used for filter observation noise |
| Label entry | Current close | Next open | Conditional current-bar return estimates, not the same forward label |
| Time | Normalized tz-naive session dates | Aware 16:00 ET daily bar-end timestamps | Current CSVs naive; normalization required |
| Missingness | Contract says NaN | Dense axis; universe-dependent median filling | Current forward/backward filling needs correction |

Share primitive inputs and explicit estimator implementations, not ambiguous column
names. Cache raw feature values before construction-specific filling/normalization.
Preserve both session identity and known-at timestamps through adapters. Longitudinal
computations must eventually use stable security identity rather than ticker alone.

## Existing data-universe foundation and gaps

- Already has cached sources, Black–Scholes price/vega/IV inversion, chain helpers,
  CC/Parkinson/GK RV, feature/label registries, and Parquet precompute/load support.
- Rolling OLS alpha/beta at 60 and 90 days already exist. Kalman filters, state
  covariance/checkpoints, pairwise graph outputs and graphical lasso do not.
- `options/iv30.py` always raises. `iv_index` uses VIX/100 for SPY and the stub for
  other tickers. VIX is not the research project's reconstructed single-name ATM
  IV30, so preserve its separate identity rather than treating it as that panel.
- `MassiveSource` adds stock/index bars and ticker reference data, not options
  methods. `PolygonOptionsSource` uses Polygon config directly for options URLs;
  passing a Massive stock source does not supply a complete IV30 input interface.
- No flat-file ingestion, Yang–Zhang estimator, RV forecast, or research-compatible
  multi-column IV30 panel builder exists. Existing forward-return registration is
  15 days, not the research harness's four horizons.
- README and `sources/base.py` promise dividend-adjusted bars, but Polygon/Massive
  implementation only sends `adjusted=true`. This mismatch needs correction.
- `precompute` stores scalar `(date, ticker, value)` rows and drops NaNs. It cannot
  preserve IV30 diagnostics, graph edges or covariance/checkpoint fields as written.
  It recomputes requested pairs
  on rerun, then upserts; it does not skip completed work or track dependency versions.
- Dropping new NaNs before upsert can leave an old cached value when a corrected
  computation now finds that observation invalid. Invalidation needs explicit policy.
- Labels stop at the requested end date and lack future data near the boundary.
  Incremental jobs must backfill newly matured labels without exposing them as signals.

## Proposed precompute order

1. **Durable raw inputs:** bulk option bars where entitled, raw equity bars,
   historical contract metadata, corporate actions, dated rates and entity mapping.
   Record dataset coverage and source timestamps. Separate empty data from denied access.
2. **Historical contract IV:** consistent date/units, actual price for every selected
   contract, price/model/input provenance, solver checks, and quality failures.
   Cache at contract/date/model/input-version granularity so reruns reuse solved IV.
3. **ATM IV30 panel:** explicit near/far expiry choice, exact-target handling,
   call/put and strike policy, total-variance interpolation, diagnostic columns.
4. **Shared RV and labels:** CC/Parkinson/GK first, then Yang–Zhang/forecast once
   specified; 1/5/21/63-session returns and forward RV with entity/delisting handling.
5. **VRP and research joins:** inexpensive arithmetic from persisted IV/RV, with
   point-in-time membership and earnings controls. Keep labels physically/logically
   distinguishable from features and record the date they become available.
6. **Shared rolling relationships:** reuse return panels for OLS and gap-aware
   pairwise correlations/counts; persist sparse edges and neighbor lists. Add
   residualization, graphical lasso and macro betas when their recipes are specified.
7. **Sequential beta states:** implement causal random-walk and mean-reverting
   filters with parameter/calibration provenance, uncertainty and resume checkpoints.
   Keep tuning in the research project while caching reusable fixed-recipe outputs.

This order is a dependency-oriented proposal, not an agreed schedule. Cheap rolling
returns/RV/OLS can land early; the largest expected specialized savings are historical
IV reconstruction, large-universe relationship estimation and repeated Kalman runs.

Extend the existing precompute machinery rather than replacing it wholesale:
support multi-column panels, dependency reuse, partitioned storage, missing/failed
row accounting, resume checkpoints, explicit invalidation and versioned inputs/
formulas. Cheap rolling math can be recomputed over affected windows; avoid repeating
historical API discovery and IV solves. Materialize downstream-compatible panel names
through an adapter, since the shared registry and VRP contract currently differ.

Verify actual dataset entitlements before implementation. No forecast model,
cross-project dividend convention, American-option pricing policy, graph recipe,
filter calibration policy or new shared schema is agreed by this inventory alone.
Resolve the identified data/causality issues before persisting derived outputs;
precomputation should not turn research shortcuts into permanent shared data.

## Provider references (checked 2026-10-20)

1. [Stock split/dividend adjustment](https://massive.com/knowledge-base/article/is-massives-stock-data-adjusted-for-splits-or-dividends).
2. [Historical options contract reference](https://massive.com/docs/rest/options/contracts/all-contracts).
3. [Historical options aggregates](https://massive.com/docs/rest/options/aggregates/custom-bars).
4. [Options daily S3 files](https://massive.com/docs/flat-files/options/day-aggregates).
5. [Option-chain snapshot API](https://massive.com/docs/rest/options/snapshots/option-chain-snapshot).
6. [Options FAQ: historical OI and missing snapshot analytics](https://massive.com/knowledge-base/categories/options).
7. [Business options coverage and expansions](https://massive.com/business-options).
8. [Dividends and historical adjustment factors](https://massive.com/docs/rest/stocks/corporate-actions/dividends).
9. [Stock custom aggregate bars](https://massive.com/docs/rest/stocks/aggregates/custom-bars).
10. [Treasury yield coverage](https://massive.com/blog/release-notes-june-2025).
11. [Financial filing dates and historical-version limitations](https://massive.com/knowledge-base/article/does-massive-provide-the-filing-date-for-any-financial-reports).
12. [Supported technical indicators](https://massive.com/knowledge-base/article/does-massive-offer-any-technical-indicators).
