"""Entitlement and response-shape probe: call each upstream endpoint this library relies
on (or plans to) once, with your real keys, and record what the plan actually includes.

    python -m data_universe.probe                           # every group
    python -m data_universe.probe --only stocks options     # some groups
    python -m data_universe.probe --host polygon --out /tmp/probe.json

Opt-in and informational -- never run in CI, always exits 0 (unless the arguments are
bad). It writes a JSON report (default `<data_dir>/probe/probe_<UTC timestamp>.json`) and
prints a table. It resolves `OPEN_QUESTIONS.md` items 4, 5, 10, 15, 20 and 22, and scopes
the roadmap's later phases (flat files, as-of CIK lookup, new datasets).

Each check records `status`, one of:
`ok` (data returned), `empty` (200 with no rows), `not_entitled` (401/403),
`not_found` (404 on a documented path), `path_unverified` (404 on a path this module has
not confirmed against vendor docs -- may simply be the wrong path), `rate_limited`,
`error`, `skipped_no_key`, `skipped_missing_dep`, `skipped_dependency` (an earlier check
it needs returned nothing usable).

Derived `findings` answer specific questions -- e.g. `dividend_cash_basis` decides
`sources.adjust.DIVIDEND_CASH_BASIS`. Only paths already used by this package or long
documented by the vendor are probed; endpoints for new datasets (Benzinga earnings, short
interest, Treasury yields, ...) are added only once their exact paths are confirmed in
Massive's docs (`OPEN_QUESTIONS.md` item 23).

Secrets: request headers are never recorded, endpoints are recorded as path templates
(no query strings), and FRED goes through `FredSource`, which strips its URL-borne key
from errors.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable, Dict, List, Optional, Sequence

import pandas as pd
import requests

import data_universe
from data_universe import config
from data_universe.sources import edgar
from data_universe.sources.base import NY_TZ, DataSourceError, RateLimitedError
from data_universe.sources.http import get_with_retries

GROUPS = ("stocks", "reference", "indices", "options", "flatfiles", "edgar", "fred", "newdata")
FLATFILES_BUCKET = "flatfiles"
FLATFILES_PREFIXES = {
    "flatfiles_stocks_day_aggs": "us_stocks_sip/day_aggs_v1/2024/01/",
    "flatfiles_options_day_aggs": "us_options_opra/day_aggs_v1/2024/01/",
}
META_CIK = 1326801
AAPL_2019_DIVIDEND = {"ex_date": "2019-08-09", "raw": 0.77, "split_adjusted": 0.1925}


@dataclass
class CheckResult:
    check: str
    group: str
    endpoint: str
    path_verified: bool = True
    status: str = "error"
    http_status: Optional[int] = None
    rows: Optional[int] = None
    detail: str = ""
    findings: dict = field(default_factory=dict)


# --- pure helpers (unit-tested) -----------------------------------------------------


def classify(http_status: int, rows: int, path_verified: bool) -> str:
    """Map an HTTP status and row count to a probe status."""
    if http_status in (401, 403):
        return "not_entitled"
    if http_status == 404:
        return "not_found" if path_verified else "path_unverified"
    if http_status == 429:
        return "rate_limited"
    if http_status >= 400:
        return "error"
    return "ok" if rows > 0 else "empty"


def count_rows(payload) -> int:
    """Rows in a vendor JSON payload: `len(results)` for a list, 1 for a non-empty dict
    `results` or a non-empty payload without `results` (e.g. `/v1/open-close`)."""
    if not isinstance(payload, dict):
        return 0
    if "results" in payload:
        results = payload["results"]
        if isinstance(results, list):
            return len(results)
        return 1 if results else 0
    return 1 if payload else 0


def closes_by_date(agg_rows: Sequence[dict]) -> Dict[str, float]:
    """`{"YYYY-MM-DD": close}` from aggregate rows (`t` in epoch ms)."""
    out = {}
    for row in agg_rows or []:
        stamp = pd.Timestamp(row["t"], unit="ms", tz="UTC").tz_convert(NY_TZ)
        out[stamp.strftime("%Y-%m-%d")] = float(row["c"])
    return out


def split_only_finding(adjusted: Dict[str, float], raw: Dict[str, float], dividends) -> dict:
    """Does the vendor's `adjusted=true` leave dividends out? Compares the adjusted and
    raw close ratio across the first ex-dividend date inside both series."""
    dates = sorted(set(adjusted) & set(raw))
    ex_dates = sorted(
        d["ex_dividend_date"] for d in dividends or [] if d.get("ex_dividend_date") in dates
    )
    ex_dates = [e for e in ex_dates if dates.index(e) > 0]
    if not ex_dates:
        return {"value": None, "detail": "no ex-dividend date inside the fetched window"}
    ex_date = ex_dates[0]
    prev = dates[dates.index(ex_date) - 1]
    adjusted_ratio = adjusted[ex_date] / adjusted[prev]
    raw_ratio = raw[ex_date] / raw[prev]
    return {
        "value": abs(adjusted_ratio - raw_ratio) < 1e-9,
        "ex_date": ex_date,
        "adjusted_ratio": adjusted_ratio,
        "raw_ratio": raw_ratio,
    }


def split_applied_finding(
    adjusted: Dict[str, float], raw: Dict[str, float], splits, on_date: str
) -> dict:
    """Is `adjusted=true` split-adjusted? raw/adjusted on `on_date` (before the split)
    should equal the split ratio from the splits endpoint."""
    after = [s for s in splits or [] if s.get("execution_date", "") > on_date]
    if not after or on_date not in adjusted or on_date not in raw:
        return {"value": None, "detail": "missing split record or bars"}
    split = min(after, key=lambda s: s["execution_date"])
    expected = float(split["split_to"]) / float(split["split_from"])
    observed = raw[on_date] / adjusted[on_date]
    return {
        "value": abs(observed / expected - 1.0) < 0.01,
        "expected_ratio": expected,
        "observed_ratio": observed,
        "split_execution_date": split["execution_date"],
    }


def dividend_basis_finding(dividends) -> dict:
    """Is `cash_amount` as declared (`raw`) or adjusted for later splits? Uses AAPL's
    2019-08-09 dividend: $0.77 as declared, $0.1925 after the 2020 4-for-1 split."""
    rows = dividends or []
    has_factor = any("historical_adjustment_factor" in row for row in rows)
    match = [r for r in rows if r.get("ex_dividend_date") == AAPL_2019_DIVIDEND["ex_date"]]
    basis = "unknown"
    amount = None
    if match:
        amount = float(match[0].get("cash_amount", float("nan")))
        for name in ("raw", "split_adjusted"):
            if abs(amount - AAPL_2019_DIVIDEND[name]) < 1e-6:
                basis = name
    return {
        "value": basis,
        "aapl_2019_08_09_cash_amount": amount,
        "has_historical_adjustment_factor": has_factor,
    }


def strikes_raw_finding(contracts, raw_close: Optional[float]) -> dict:
    """Are historical option strikes on the raw (as-traded) basis?"""
    strikes = [float(c["strike_price"]) for c in contracts or [] if "strike_price" in c]
    if not strikes or not raw_close:
        return {"value": None, "detail": "no contracts or no raw close"}
    ratio = median(strikes) / raw_close
    return {"value": 0.5 <= ratio <= 2.0, "median_strike_over_raw_close": ratio}


def nearest_call(contracts, spot: float) -> Optional[str]:
    """OCC ticker of the call whose strike is nearest `spot`."""
    calls = [c for c in contracts or [] if c.get("contract_type") == "call"]
    if not calls:
        return None
    return min(calls, key=lambda c: abs(float(c["strike_price"]) - spot))["ticker"]


def fb_cik_finding(details) -> dict:
    """Does `/v3/reference/tickers/FB?date=2020-06-01` resolve to Meta's CIK?"""
    cik = (details or {}).get("cik")
    try:
        value = int(cik) == META_CIK
    except (TypeError, ValueError):
        value = None
    return {"value": value, "cik": cik, "name": (details or {}).get("name")}


# --- runner ---------------------------------------------------------------------------


class Probe:
    """Runs the checks. Every network object is injectable for tests."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        host: str = "massive",
        edgar_session: Optional[requests.Session] = None,
        fred_source=None,
        s3_client_factory: Optional[Callable] = None,
        max_retries: int = 2,
    ) -> None:
        self.session = session or requests.Session()
        self.edgar_session = edgar_session or self.session
        self.host = host
        self.base_url = (
            config.get_massive_base_url() if host == "massive" else config.get_polygon_base_url()
        )
        self.key = (
            (config.get_massive_key() or config.get_polygon_key())
            if host == "massive"
            else config.get_polygon_key()
        )
        self.fred_source = fred_source
        self.s3_client_factory = s3_client_factory
        self.max_retries = max_retries
        self.results: List[CheckResult] = []
        self.payloads: Dict[str, object] = {}

    # one request -> one CheckResult
    def _record(self, result: CheckResult, payload=None) -> CheckResult:
        self.results.append(result)
        self.payloads[result.check] = payload
        return result

    def _http(
        self,
        check: str,
        group: str,
        url: str,
        endpoint: str,
        params: Optional[dict],
        headers: Optional[dict],
        path_verified: bool,
        throttle=None,
    ) -> CheckResult:
        result = CheckResult(check, group, endpoint, path_verified)
        try:
            response = get_with_retries(
                self.edgar_session if group == "edgar" else self.session,
                url,
                params=params,
                headers=headers,
                timeout=30,
                max_retries=self.max_retries,
                backoff_seconds=1.0,
                throttle=throttle,
            )
        except RateLimitedError:
            result.status, result.http_status = "rate_limited", 429
            return self._record(result)
        except (DataSourceError, requests.RequestException) as exc:
            result.status, result.detail = "error", type(exc).__name__ + ": " + str(exc)[:200]
            return self._record(result)
        result.http_status = response.status_code
        payload = None
        if response.status_code < 400:
            try:
                payload = response.json()
            except ValueError:
                result.status, result.detail = "error", "response was not JSON"
                return self._record(result)
        result.rows = count_rows(payload) if payload is not None else 0
        result.status = classify(response.status_code, result.rows, path_verified)
        return self._record(result, payload)

    def vendor(
        self, check: str, group: str, path: str, params=None, path_verified: bool = True
    ) -> CheckResult:
        endpoint = path.split("?")[0]
        if not self.key:
            return self._record(
                CheckResult(check, group, endpoint, path_verified, status="skipped_no_key")
            )
        return self._http(
            check,
            group,
            f"{self.base_url}{path}",
            endpoint,
            params,
            {"Authorization": f"Bearer {self.key}"},
            path_verified,
        )

    def results_of(self, check: str):
        payload = self.payloads.get(check)
        return payload.get("results") if isinstance(payload, dict) else None

    def _aggs(self, check, group, ticker, timespan, start, end, adjusted=True):
        return self.vendor(
            check,
            group,
            f"/v2/aggs/ticker/{ticker}/range/1/{timespan}/{start}/{end}",
            {"adjusted": "true" if adjusted else "false", "sort": "asc", "limit": 50000},
        )

    def _attach(self, check: str, name: str, finding: dict) -> None:
        for result in self.results:
            if result.check == check:
                result.findings[name] = finding

    # --- groups ---------------------------------------------------------------------

    def run_stocks(self):
        g = "stocks"
        self._aggs("aapl_daily_split_adjusted", g, "AAPL", "day", "2024-08-01", "2024-08-20")
        self._aggs("aapl_daily_raw", g, "AAPL", "day", "2024-08-01", "2024-08-20", False)
        self._aggs("nvda_daily_split_adjusted", g, "NVDA", "day", "2024-06-03", "2024-06-14")
        self._aggs("nvda_daily_raw", g, "NVDA", "day", "2024-06-03", "2024-06-14", False)
        self._aggs("aapl_hourly_2024", g, "AAPL", "hour", "2024-01-02", "2024-01-02")
        self._aggs("aapl_hourly_2015", g, "AAPL", "hour", "2015-01-02", "2015-01-02")
        self._aggs("aapl_minute_2024", g, "AAPL", "minute", "2024-01-02", "2024-01-02")
        self._aggs("fb_daily_2020", g, "FB", "day", "2020-01-02", "2020-01-03")
        self._aggs("meta_daily_2020", g, "META", "day", "2020-01-02", "2020-01-03")

    def run_reference(self):
        g = "reference"
        self.vendor("aapl_dividends", g, "/v3/reference/dividends",
                    {"ticker": "AAPL", "limit": 1000})
        self.vendor("aapl_splits", g, "/v3/reference/splits", {"ticker": "AAPL"})
        self.vendor("nvda_splits", g, "/v3/reference/splits", {"ticker": "NVDA"})
        self.vendor("aapl_ticker_details", g, "/v3/reference/tickers/AAPL")
        self.vendor("fb_ticker_details_2020", g, "/v3/reference/tickers/FB",
                    {"date": "2020-06-01"})
        self.vendor("meta_ticker_events", g, "/vX/reference/tickers/META/events",
                    path_verified=False)

    def run_indices(self):
        self._aggs("vix_daily", "indices", "I:VIX", "day", "2024-01-02", "2024-01-05")

    def run_options(self):
        g = "options"
        self._aggs("aapl_daily_raw_2019", g, "AAPL", "day", "2019-07-01", "2019-07-01", False)
        self.vendor(
            "aapl_contracts_as_of_2019",
            g,
            "/v3/reference/options/contracts",
            {
                "underlying_ticker": "AAPL",
                "as_of": "2019-07-01",
                "expired": "true",
                "expiration_date.lte": "2019-08-16",
                "limit": 1000,
            },
        )
        raw_close = closes_by_date(self.results_of("aapl_daily_raw_2019")).get("2019-07-01")
        contracts = self.results_of("aapl_contracts_as_of_2019")
        self._attach(
            "aapl_contracts_as_of_2019",
            "options_strikes_are_raw",
            strikes_raw_finding(contracts, raw_close),
        )
        contract = nearest_call(contracts, raw_close) if raw_close else None
        if contract is None:
            for check, endpoint in (
                ("option_open_close", "/v1/open-close/{contract}/2019-07-01"),
                ("option_daily_range_aggs", "/v2/aggs/ticker/{contract}/range/1/day/..."),
            ):
                self._record(CheckResult(
                    check, g, endpoint,
                    status="skipped_no_key" if not self.key else "skipped_dependency",
                    detail="no contract to price (contracts or raw close unavailable)",
                ))
            return
        self.vendor("option_open_close", g, f"/v1/open-close/{contract}/2019-07-01",
                    {"adjusted": "true"})
        self._aggs("option_daily_range_aggs", g, contract, "day", "2019-06-01", "2019-07-19")
        for result in self.results[-2:]:
            result.endpoint = result.endpoint.replace(contract, "{contract}")
            result.detail = f"contract {contract}"

    def run_flatfiles(self):
        g = "flatfiles"
        endpoint_url = config.get_polygon_flatfiles_endpoint()
        access = config.get_polygon_s3_access_key()
        secret = config.get_polygon_s3_secret_key()
        factory = self.s3_client_factory
        if factory is None:
            try:
                import boto3
            except ImportError:
                boto3 = None
            if boto3 is not None:
                def factory(**kwargs):
                    return boto3.client("s3", **kwargs)
        for check, prefix in FLATFILES_PREFIXES.items():
            result = CheckResult(check, g, f"s3://{FLATFILES_BUCKET}/{prefix}", False)
            if factory is None:
                result.status = "skipped_missing_dep"
                result.detail = 'pip install -e ".[s3]"'
            elif not (endpoint_url and access and secret):
                result.status = "skipped_no_key"
            else:
                try:
                    client = factory(
                        endpoint_url=endpoint_url,
                        aws_access_key_id=access,
                        aws_secret_access_key=secret,
                    )
                    listing = client.list_objects_v2(
                        Bucket=FLATFILES_BUCKET, Prefix=prefix, MaxKeys=3
                    )
                    keys = [o["Key"] for o in listing.get("Contents", [])]
                    result.rows = len(keys)
                    result.status = "ok" if keys else "empty"
                    result.findings["sample_keys"] = keys
                except Exception as exc:  # noqa: BLE001 - botocore error classes vary
                    text = type(exc).__name__ + ": " + str(exc)[:200]
                    denied = "AccessDenied" in text or "403" in text
                    result.status = "not_entitled" if denied else "error"
                    result.detail = text
            self._record(result)

    def run_edgar(self):
        g = "edgar"
        headers = {"User-Agent": edgar.EdgarSource()._resolved_user_agent()}
        self._http("aapl_submissions", g, "https://data.sec.gov/submissions/CIK0000320193.json",
                   "/submissions/CIK##########.json", None, headers, True, edgar._throttle)
        self._http(
            "aapl_companyfacts",
            g,
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
            "/api/xbrl/companyfacts/CIK##########.json",
            None,
            headers,
            True,
            edgar._throttle,
        )
        facts = self.payloads.get("aapl_companyfacts")
        eps = []
        if isinstance(facts, dict):
            eps = (
                facts.get("facts", {}).get("us-gaap", {}).get("EarningsPerShareDiluted", {})
                .get("units", {}).get("USD/shares", [])
            )
        self._attach("aapl_companyfacts", "diluted_eps_has_filed_dates", {
            "value": bool(eps) and all("filed" in e for e in eps),
            "n_facts": len(eps),
        })

    def run_fred(self):
        g = "fred"
        for series in ("DTB3", "DTB4WK", "DGS1MO"):
            result = CheckResult(f"fred_{series.lower()}", g, "/fred/series/observations")
            if not config.get_fred_key():
                result.status = "skipped_no_key"
                self._record(result)
                continue
            if self.fred_source is None:
                from data_universe.sources.fred import FredSource

                self.fred_source = FredSource()
            try:
                series_values = self.fred_source.daily_rate(
                    series_id=series, start="2024-01-02", end="2024-01-10"
                )
                result.rows = len(series_values)
                result.status = "ok" if len(series_values) else "empty"
            except (DataSourceError, ValueError) as exc:
                result.status, result.detail = "error", str(exc)[:200]
            self._record(result)

    def run_newdata(self):
        self.vendor("aapl_news", "newdata", "/v2/reference/news", {"ticker": "AAPL", "limit": 1})

    def derive(self) -> None:
        """Cross-check findings that need several checks' payloads."""
        dividends = self.results_of("aapl_dividends")
        if "aapl_daily_split_adjusted" in self.payloads and dividends is not None:
            self._attach("aapl_daily_split_adjusted", "adjusted_true_is_split_only",
                         split_only_finding(
                             closes_by_date(self.results_of("aapl_daily_split_adjusted")),
                             closes_by_date(self.results_of("aapl_daily_raw")),
                             dividends,
                         ))
        if "nvda_daily_split_adjusted" in self.payloads and "nvda_splits" in self.payloads:
            self._attach("nvda_daily_split_adjusted", "split_adjustment_applied",
                         split_applied_finding(
                             closes_by_date(self.results_of("nvda_daily_split_adjusted")),
                             closes_by_date(self.results_of("nvda_daily_raw")),
                             self.results_of("nvda_splits"),
                             "2024-06-07",
                         ))
        if dividends is not None:
            self._attach("aapl_dividends", "dividend_cash_basis",
                         dividend_basis_finding(dividends))
        if "fb_ticker_details_2020" in self.payloads:
            self._attach("fb_ticker_details_2020", "fb_as_of_cik_is_meta",
                         fb_cik_finding(self.results_of("fb_ticker_details_2020")))
        for check in ("fb_daily_2020", "meta_daily_2020"):
            if check in self.payloads:
                rows = self.results_of(check) or []
                self._attach(check, "has_2020_bars", {"value": bool(rows)})

    def run(self, groups: Sequence[str] = GROUPS) -> dict:
        for group in GROUPS:
            if group in groups:
                getattr(self, f"run_{group}")()
        self.derive()
        return self.report()

    def report(self) -> dict:
        counts: Dict[str, int] = {}
        for result in self.results:
            counts[result.status] = counts.get(result.status, 0) + 1
        return {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "host": self.host,
            "base_url": self.base_url,
            "package_version": data_universe.__version__,
            "summary": counts,
            "checks": [asdict(r) for r in self.results],
        }


def format_table(report: dict) -> str:
    """Human-readable summary of a probe report."""
    lines = [
        f"data_universe {report['package_version']} probe against {report['base_url']}",
        f"{'check':34} {'status':20} {'http':>5} {'rows':>6}  findings",
    ]
    for check in report["checks"]:
        findings = ", ".join(
            f"{name}={value.get('value')!r}" for name, value in check["findings"].items()
            if isinstance(value, dict) and "value" in value
        )
        http = "" if check["http_status"] is None else str(check["http_status"])
        rows = "" if check["rows"] is None else str(check["rows"])
        lines.append(f"{check['check']:34} {check['status']:20} {http:>5} {rows:>6}  {findings}")
    lines.append("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(report["summary"].items())))
    return "\n".join(lines)


def default_output_path() -> Path:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(config.get_data_dir()) / "probe" / f"probe_{stamp}.json"


def main(argv: Optional[Sequence[str]] = None, probe: Optional[Probe] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m data_universe.probe",
        description="Probe which upstream datasets your keys/plan include.",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Config YAML to load.")
    parser.add_argument("--out", default=None, help="Where to write the JSON report.")
    parser.add_argument("--host", choices=("massive", "polygon"), default="massive")
    parser.add_argument("--only", nargs="+", choices=GROUPS, default=None, help="Groups to run.")
    args = parser.parse_args(argv)

    config.load_yaml(args.config)
    probe = probe or Probe(host=args.host)
    report = probe.run(args.only or GROUPS)
    out = Path(args.out) if args.out else default_output_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(format_table(report))
    print(f"report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
