import json
from unittest.mock import Mock

import pandas as pd
import pytest

from data_universe import config, probe
from data_universe.cache import reset_cache
from data_universe.sources import edgar

SECRET_MASSIVE = "SENTINEL-MASSIVE-KEY-123"
SECRET_FRED = "SENTINEL-FRED-KEY-456"
SECRET_S3 = "SENTINEL-S3-SECRET-789"


@pytest.fixture(autouse=True)
def _reset_state(monkeypatch, tmp_path):
    config.reset()
    reset_cache()
    config.set_data_dir(str(tmp_path))
    monkeypatch.setattr(edgar, "_SEC_MIN_INTERVAL_SECONDS", 0.0)
    monkeypatch.setattr(edgar, "_warned_placeholder", True)
    monkeypatch.setattr("data_universe.sources.http.time.sleep", lambda s: None)
    yield
    config.reset()
    reset_cache()


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}

    def json(self):
        return self._payload


def _agg(day, close):
    ms = int(pd.Timestamp(day, tz="America/New_York").timestamp() * 1000)
    return {"t": ms, "o": close, "h": close, "l": close, "c": close, "v": 1.0}


def _router(routes):
    """`routes`: list of (substring, adjusted-or-None, response). First match wins."""

    def get(url, params=None, headers=None, timeout=None):
        for needle, adjusted, response in routes:
            if needle in url and (adjusted is None or (params or {}).get("adjusted") == adjusted):
                return response
        return FakeResponse(404)

    session = Mock()
    session.get.side_effect = get
    return session


# --- pure helpers ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,rows,verified,expected",
    [
        (200, 3, True, "ok"),
        (200, 0, True, "empty"),
        (401, 0, True, "not_entitled"),
        (403, 0, True, "not_entitled"),
        (404, 0, True, "not_found"),
        (404, 0, False, "path_unverified"),
        (429, 0, True, "rate_limited"),
        (400, 0, True, "error"),
    ],
)
def test_classify(status, rows, verified, expected):
    assert probe.classify(status, rows, verified) == expected


def test_count_rows():
    assert probe.count_rows({"results": [1, 2]}) == 2
    assert probe.count_rows({"results": {"cik": "1"}}) == 1
    assert probe.count_rows({"results": []}) == 0
    assert probe.count_rows({"close": 1.0}) == 1
    assert probe.count_rows({}) == 0


def test_split_only_finding_detects_dividend_free_adjustment():
    raw = {"2024-08-09": 216.0, "2024-08-12": 217.0}
    divs = [{"ex_dividend_date": "2024-08-12", "cash_amount": 0.25}]
    assert probe.split_only_finding(dict(raw), raw, divs)["value"] is True
    total_return = {"2024-08-09": 216.0 * (1 - 0.25 / 216.0), "2024-08-12": 217.0}
    assert probe.split_only_finding(total_return, raw, divs)["value"] is False
    assert probe.split_only_finding(raw, raw, [])["value"] is None


def test_split_applied_finding():
    splits = [{"execution_date": "2024-06-10", "split_from": 1, "split_to": 10}]
    finding = probe.split_applied_finding(
        {"2024-06-07": 120.0}, {"2024-06-07": 1200.0}, splits, "2024-06-07"
    )
    assert finding["value"] is True
    assert finding["expected_ratio"] == 10.0


@pytest.mark.parametrize("amount,basis", [(0.77, "raw"), (0.1925, "split_adjusted"), (1.0,
                                          "unknown")])
def test_dividend_basis_finding(amount, basis):
    rows = [{"ex_dividend_date": "2019-08-09", "cash_amount": amount}]
    assert probe.dividend_basis_finding(rows)["value"] == basis


def test_dividend_basis_finding_reports_vendor_factor_field():
    rows = [{"ex_dividend_date": "2024-01-01", "historical_adjustment_factor": 0.99}]
    finding = probe.dividend_basis_finding(rows)
    assert finding["value"] == "unknown"
    assert finding["has_historical_adjustment_factor"] is True


def test_strikes_raw_finding_and_nearest_call():
    contracts = [
        {"ticker": "O:A1", "strike_price": 195.0, "contract_type": "call"},
        {"ticker": "O:A2", "strike_price": 205.0, "contract_type": "call"},
        {"ticker": "O:A3", "strike_price": 200.0, "contract_type": "put"},
    ]
    assert probe.strikes_raw_finding(contracts, 202.0)["value"] is True
    assert probe.strikes_raw_finding(contracts, 50.0)["value"] is False
    assert probe.nearest_call(contracts, 204.0) == "O:A2"


def test_fb_cik_finding():
    assert probe.fb_cik_finding({"cik": "0001326801"})["value"] is True
    assert probe.fb_cik_finding({"cik": "0000000001"})["value"] is False
    assert probe.fb_cik_finding({})["value"] is None


# --- runner ------------------------------------------------------------------------------


def _vendor_routes():
    return [
        ("/v2/aggs/ticker/AAPL/range/1/day/2024-08-01", "true",
         FakeResponse(200, {"results": [_agg("2024-08-09", 216.0), _agg("2024-08-12", 217.0)]})),
        ("/v2/aggs/ticker/AAPL/range/1/day/2024-08-01", "false",
         FakeResponse(200, {"results": [_agg("2024-08-09", 216.0), _agg("2024-08-12", 217.0)]})),
        ("/v2/aggs/ticker/NVDA/range/1/day/", "true",
         FakeResponse(200, {"results": [_agg("2024-06-07", 120.0)]})),
        ("/v2/aggs/ticker/NVDA/range/1/day/", "false",
         FakeResponse(200, {"results": [_agg("2024-06-07", 1200.0)]})),
        ("/range/1/hour/2015-01-02", None, FakeResponse(200, {"results": []})),
        ("/range/1/minute/", None, FakeResponse(403)),
        ("/v2/aggs/ticker/AAPL/range/1/day/2019-07-01", None,
         FakeResponse(200, {"results": [_agg("2019-07-01", 202.0)]})),
        ("/v2/aggs/ticker/O:", None, FakeResponse(200, {"results": [_agg("2019-07-01", 5.0)]})),
        ("/v2/aggs/", None, FakeResponse(200, {"results": [_agg("2024-01-02", 1.0)]})),
        ("/v3/reference/dividends", None, FakeResponse(200, {"results": [
            {"ex_dividend_date": "2024-08-12", "cash_amount": 0.25},
            {"ex_dividend_date": "2019-08-09", "cash_amount": 0.77},
        ]})),
        ("/v3/reference/splits", None, FakeResponse(200, {"results": [
            {"execution_date": "2024-06-10", "split_from": 1, "split_to": 10},
        ]})),
        ("/v3/reference/tickers/FB", None,
         FakeResponse(200, {"results": {"cik": "0001326801", "name": "Meta Platforms"}})),
        ("/v3/reference/tickers/AAPL", None, FakeResponse(200, {"results": {"cik": "0000320193"}})),
        ("/v3/reference/options/contracts", None, FakeResponse(200, {"results": [
            {"ticker": "O:AAPL190719C00200000", "strike_price": 200.0, "contract_type": "call"},
            {"ticker": "O:AAPL190719P00200000", "strike_price": 200.0, "contract_type": "put"},
        ]})),
        ("/v1/open-close/O:AAPL190719C00200000/2019-07-01", None,
         FakeResponse(200, {"close": 5.1, "volume": 100})),
        ("/v2/reference/news", None, FakeResponse(200, {"results": [{"id": "x"}]})),
        ("data.sec.gov/submissions/", None, FakeResponse(200, {"cik": "320193"})),
        ("companyfacts", None, FakeResponse(200, {"facts": {"us-gaap": {
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                {"val": 1.5, "filed": "2024-08-02"}]}}}}})),
    ]


def _by_check(report):
    return {c["check"]: c for c in report["checks"]}


def test_full_run_statuses_and_findings():
    config.set_massive_key(SECRET_MASSIVE)
    session = _router(_vendor_routes())
    report = probe.Probe(session=session).run(
        ["stocks", "reference", "indices", "options", "edgar", "newdata"]
    )
    checks = _by_check(report)
    assert checks["aapl_daily_split_adjusted"]["status"] == "ok"
    assert checks["aapl_minute_2024"]["status"] == "not_entitled"
    assert checks["aapl_hourly_2015"]["status"] == "empty"
    assert checks["meta_ticker_events"]["status"] == "path_unverified"
    split_only = checks["aapl_daily_split_adjusted"]["findings"]["adjusted_true_is_split_only"]
    assert split_only["value"] is True
    nvda = checks["nvda_daily_split_adjusted"]["findings"]["split_adjustment_applied"]
    assert nvda["value"] is True
    basis = checks["aapl_dividends"]["findings"]["dividend_cash_basis"]
    assert basis["value"] == "raw"
    assert checks["fb_ticker_details_2020"]["findings"]["fb_as_of_cik_is_meta"]["value"] is True
    strikes = checks["aapl_contracts_as_of_2019"]["findings"]["options_strikes_are_raw"]
    assert strikes["value"] is True
    assert checks["option_open_close"]["status"] == "ok"
    assert checks["option_open_close"]["endpoint"] == "/v1/open-close/{contract}/2019-07-01"
    assert checks["aapl_companyfacts"]["findings"]["diluted_eps_has_filed_dates"]["value"]
    assert checks["aapl_news"]["status"] == "ok"
    assert report["summary"]["ok"] >= 10


def test_no_keys_means_every_vendor_check_is_skipped():
    report = probe.Probe(session=Mock()).run(
        ["stocks", "reference", "indices", "options", "fred", "newdata"]
    )
    statuses = {c["status"] for c in report["checks"]}
    assert statuses == {"skipped_no_key"}


def test_flatfiles_skips_without_dependency_or_credentials(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_boto3(name, *args, **kwargs):
        if name == "boto3":
            raise ImportError("no boto3")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_boto3)
    report = probe.Probe(session=Mock()).run(["flatfiles"])
    assert {c["status"] for c in report["checks"]} == {"skipped_missing_dep"}

    report = probe.Probe(session=Mock(), s3_client_factory=lambda **kw: None).run(["flatfiles"])
    assert {c["status"] for c in report["checks"]} == {"skipped_no_key"}


def test_flatfiles_listing_and_access_denied():
    config.set_polygon_flatfiles_endpoint("https://files.example.test")
    config.set_polygon_s3_access_key("access")
    config.set_polygon_s3_secret_key(SECRET_S3)

    class Client:
        def list_objects_v2(self, Bucket, Prefix, MaxKeys):
            if "options" in Prefix:
                raise RuntimeError("An error occurred (AccessDenied) when calling ListObjectsV2")
            return {"Contents": [{"Key": Prefix + "2024-01-02.csv.gz"}]}

    report = probe.Probe(session=Mock(), s3_client_factory=lambda **kw: Client()).run(
        ["flatfiles"]
    )
    checks = _by_check(report)
    assert checks["flatfiles_stocks_day_aggs"]["status"] == "ok"
    assert checks["flatfiles_options_day_aggs"]["status"] == "not_entitled"
    assert SECRET_S3 not in json.dumps(report)


def test_fred_checks_use_the_fred_source():
    config.set_fred_key(SECRET_FRED)
    fred = Mock()
    fred.daily_rate.return_value = pd.Series([0.0002, 0.0002])
    report = probe.Probe(session=Mock(), fred_source=fred).run(["fred"])
    assert {c["status"] for c in report["checks"]} == {"ok"}
    assert [c[1]["series_id"] for c in fred.daily_rate.call_args_list] == [
        "DTB3", "DTB4WK", "DGS1MO",
    ]


def test_main_writes_report_and_never_leaks_secrets(tmp_path, capsys):
    config.set_massive_key(SECRET_MASSIVE)
    config.set_fred_key(SECRET_FRED)
    fred = Mock()
    fred.daily_rate.return_value = pd.Series([0.0002])
    runner = probe.Probe(session=_router(_vendor_routes()), fred_source=fred)
    out = tmp_path / "probe.json"
    code = probe.main(
        ["--config", str(tmp_path / "missing.yaml"), "--out", str(out)], probe=runner
    )
    assert code == 0
    text = out.read_text()
    stdout = capsys.readouterr().out
    report = json.loads(text)
    assert report["package_version"]
    assert "report written to" in stdout
    for secret in (SECRET_MASSIVE, SECRET_FRED):
        assert secret not in text
        assert secret not in stdout
    assert all("?" not in c["endpoint"] for c in report["checks"])
