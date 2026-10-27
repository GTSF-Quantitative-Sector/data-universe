"""Shared HTTP GET with bounded retries, used by the Polygon/Massive and EDGAR sources.

Retries transient failures -- HTTP 429/500/502/503/504 and `requests.ConnectionError`/
`requests.Timeout` -- with doubling backoff (`backoff_seconds * 2**(attempt - 1)`), and
hands every other response back to the caller, which decides how to raise (e.g.
`EntitlementError` for a vendor's 401/403). `time.sleep` is looked up at call time so tests
can monkeypatch it.

Error messages name only the URL *path* -- never the query string or headers -- so a key
that travels in a query parameter can't leak into logs or tracebacks.
"""
from __future__ import annotations

import time
from typing import Callable, Optional
from urllib.parse import urlsplit

import requests

from data_universe.sources.base import DataSourceError, RateLimitedError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


def url_path(url: str) -> str:
    """The path component of `url` (no scheme, host, query or fragment)."""
    return urlsplit(url).path


def get_with_retries(
    session: requests.Session,
    url: str,
    *,
    params: Optional[dict] = None,
    headers: Optional[dict] = None,
    timeout: float = 30,
    max_retries: int = 5,
    backoff_seconds: float = 1.0,
    throttle: Optional[Callable[[], None]] = None,
) -> requests.Response:
    """GET `url`, retrying transient failures up to `max_retries` times.

    Args:
        session: The `requests.Session` to issue the request with.
        url: Request URL.
        params: Query parameters.
        headers: Request headers.
        timeout: Per-attempt timeout in seconds.
        max_retries: Retries after the first attempt before giving up.
        backoff_seconds: Base backoff, doubled on each retry.
        throttle: Optional zero-argument callable invoked before every attempt
            (e.g. a rate limiter).

    Returns:
        The response for any non-retryable status code.

    Raises:
        RateLimitedError: HTTP 429 persisted past `max_retries`.
        DataSourceError: A 5xx status or connection/timeout error persisted past
            `max_retries`.
    """
    attempt = 0
    while True:
        if throttle is not None:
            throttle()
        failure = None
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            failure = type(exc).__name__
        else:
            if response.status_code not in RETRY_STATUSES:
                return response
            failure = f"HTTP {response.status_code}"
        attempt += 1
        if attempt > max_retries:
            if failure == "HTTP 429":
                raise RateLimitedError(f"Rate limit exceeded for {url_path(url)}")
            raise DataSourceError(
                f"GET {url_path(url)} failed after {max_retries} retries: {failure}"
            )
        time.sleep(backoff_seconds * (2 ** (attempt - 1)))
