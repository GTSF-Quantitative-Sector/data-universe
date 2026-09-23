# Maintenance

Recurring upkeep tasks for this repo, one paragraph each — mirrors the style of
`sec_parser`'s own `sec/data/sp500/README.md`.

## 1. Bump the version on every change

`setup.py`'s `VERSION` and `data_universe/__init__.py`'s `__version__` must be kept equal
and incremented on every merged change. `pip install -e .` and
`pip install git+https://github.com/GTSF-Quantitative-Sector/data-universe.git` will not
reliably pick up code changes for consumers who pin by version otherwise.

**Status: not currently followed.** Both are still `"0.1.0"` after 40 commits across 7
implementation phases. Whoever picks this repo up next should bump to `"0.2.0"` as a real
fix — not just note it here again — and start following the rule from that point on.

## 2. S&P 500 historical membership

Point-in-time membership (`sources.sec_adapter.sp500(query_date=...)`) is a pass-through to
`sec_parser`'s `lookups.get_sp500_tickers()`, which reads a manually-maintained
`sp500_historical.csv` **in the `sec_parser` repo, not this one**. This repo has no
maintenance task of its own here — the "steps below should be completed monthly" note and
its instructions live in `sec_parser`'s own `sec/data/sp500/README.md`. Don't duplicate that
file's maintenance burden into this repo; just keep the `sec_parser` dependency
(`pip install ".[sec]"`) reasonably current.

## 3. API keys

Polygon, FRED, and (once `FlatFilesSource` is built — see `OPEN_QUESTIONS.md` item 12) S3
flat-files credentials are read from environment variables or `config/config.yaml`
(gitignored) — never committed, never logged (`config.py`'s module docstring). Rotate them
per each project's own security policy; this repo has no key-rotation automation of its own.

## 4. Dependency floors

`install_requires` in `setup.py` currently pins no upper or lower bounds beyond
`python_requires=">=3.9"`. If a future `pandas`/`numpy`/`scipy` release breaks a test, pin
the floor that fixes it in `setup.py` itself, not just in a contributor's local venv — a
floor pinned only locally doesn't help the next person who installs fresh.

## 5. CI

`.github/workflows/ci.yml` runs `pytest` and `ruff check .` on every push/PR against Python
3.11, using only `sources/fakes.py` — no live API calls, no secrets required. If a new source
module is added, keep it testable against a fake/mock the same way; don't make CI depend on
a real API key.
