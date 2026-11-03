"""Point-in-time SUE (standardized unexpected earnings) from SEC XBRL
company-concept EPS facts.

Pipeline: `EdgarSource.company_concept` gives one entry per filing that
disclosed a period's EPS -- the same (start, end) period appears once per
disclosing filing. Keeping the earliest `filed` per period recovers the
first-reported (as-filed) value; later comparatives, amendments and recasts
drop out. Values are on the share basis at filing time, so `eps_adj =
val / factor(filed)` rescales every first-reported value to today's share
basis, where `factor` is the raw / split-adjusted close ratio as of the
filing date (the product of split ratios effective strictly after it --
a split effective on or before the filing date is treated as already
reflected, matching ASC 260 retroactive restatement). SUE's year-over-year
diff and std are computed on the adjusted basis (so a split mid-history
doesn't corrupt the series) but the OUTPUT `eps_yoy_diff`/`sue_std`/
`eps_diluted` are multiplied back onto each quarter's own as-filed basis, so
stored values never change when a later split happens.

The `sue` for quarter q uses only the point-in-time view V_q = {p :
filed_p <= filed_q} -- exactly what was computable on the day EPS_q was
first filed. A Q4 that no filing reports directly is derived as
FY - (Q1+Q2+Q3) (approximate -- EPS isn't additive -- and flagged by
`derived`). The `frames` API is never used: it would return restated
(latest-filed) values, which is lookahead.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")  # priority order
EPS_UNIT = "USD/shares"
SUE_WINDOW = 8
SUE_MIN_OBS = 6
QUARTER_COLUMNS = [
    "cik", "period_start", "period_end", "filed", "accn", "form", "tag",
    "eps_diluted", "derived", "eps_yoy_diff", "sue_std", "sue_n", "sue",
]
SUE_EVENT_COLUMNS = [
    "eps_period_end", "eps_filed", "eps_diluted", "eps_derived",
    "eps_yoy_diff", "sue_std", "sue_n", "sue",
]

# Duration means (end - start).days. The wide quarter range covers 12-, 13-,
# 14-, 16- and 17-week quarters (e.g. Costco, Kroger).
_QUARTER_DAYS = (65, 125)
_ANNUAL_DAYS = (340, 380)
_LAG4_DAYS = (350, 380)
_SAME_QUARTER_DAYS = 7
_FACTOR_MAX_GAP_DAYS = 10
_MATCH_MAX_DAYS = 100
_MATCH_FILED_SLACK_DAYS = 7
# A prior-diff std at or below this fraction of the largest |diff| (floored
# at 1.0) is float noise, not a real dispersion -- report 0.0, so `sue`
# comes out NaN instead of ~1e15.
_STD_REL_TOL = 1e-9


def _empty_quarters() -> pd.DataFrame:
    frame = pd.DataFrame(columns=QUARTER_COLUMNS)
    for column in ("period_start", "period_end", "filed"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("eps_diluted", "eps_yoy_diff", "sue_std", "sue"):
        frame[column] = frame[column].astype("float64")
    frame["derived"] = frame["derived"].astype("bool")
    frame["sue_n"] = frame["sue_n"].astype("int64")
    return frame


def _factor_series(price_source, ticker, start: str, end: str) -> pd.Series:
    """raw close / split-adjusted close on sessions present in both frames,
    kept only when finite and positive -- the product of split ratios
    effective strictly after each date."""
    raw = price_source.daily_bars_raw(ticker, start, end)
    adjusted = price_source.daily_bars_split_adjusted(ticker, start, end)
    raw_close = raw["close"].copy()
    raw_close.index = raw.index.tz_localize(None).normalize()
    adj_close = adjusted["close"].copy()
    adj_close.index = adjusted.index.tz_localize(None).normalize()
    common = raw_close.index.intersection(adj_close.index)
    factors = (raw_close.loc[common] / adj_close.loc[common]).sort_index()
    return factors[np.isfinite(factors) & (factors > 0)]


def _factor_at(factors: pd.Series, filed) -> float:
    """The factor at the last factor date <= `filed`, or NaN when that date
    is more than `_FACTOR_MAX_GAP_DAYS` earlier (or there is none)."""
    if factors.empty:
        return np.nan
    position = factors.index.searchsorted(filed, side="right") - 1
    if position < 0:
        return np.nan
    date = factors.index[position]
    if date < filed - pd.Timedelta(days=_FACTOR_MAX_GAP_DAYS):
        return np.nan
    return float(factors.iloc[position])


def quarterly_sue(
    ticker,
    cik,
    edgar_source,
    price_source,
    *,
    history_start=None,
    window: int = SUE_WINDOW,
    min_obs: int = SUE_MIN_OBS,
) -> pd.DataFrame:
    """The point-in-time SUE table for one CIK: one row per fiscal quarter.

    Args:
        ticker: Ticker used for the split-factor price bars.
        cik: SEC CIK of the reporting company.
        edgar_source: Object with `.company_concept(cik, tag, unit=...)`.
        price_source: Object with `.daily_bars_raw`/`.daily_bars_split_adjusted`.
        history_start: Start of the split-factor bars window, "YYYY-MM-DD";
            defaults to 10 days before the earliest first-reported filing.
        window: How many prior quarterly diffs form the SUE std (default 8).
        min_obs: Minimum diffs needed for a finite std (default 6).

    Returns:
        A `QUARTER_COLUMNS` frame sorted by `period_end` -- empty, correctly
        typed, when neither EPS tag has facts (not an error).
    """
    cik = str(int(cik))
    frames = []
    for tag in EPS_TAGS:
        concepts = edgar_source.company_concept(cik, tag, unit=EPS_UNIT)
        if len(concepts):
            frames.append(concepts.assign(tag=tag))
    if not frames:
        return _empty_quarters()
    facts = pd.concat(frames, ignore_index=True).dropna(
        subset=["val", "start", "end", "filed"]
    )
    if facts.empty:
        return _empty_quarters()

    # First-reported value per exact (start, end) period: earliest filed,
    # then EPS_TAGS priority, then accn. Restatements, amendments and
    # comparatives are filed later, so they drop out.
    tag_rank = {tag: rank for rank, tag in enumerate(EPS_TAGS)}
    facts = facts.assign(_tag_rank=facts["tag"].map(tag_rank)).sort_values(
        ["filed", "_tag_rank", "accn"], na_position="last", kind="stable"
    )
    first = facts.drop_duplicates(subset=["start", "end"], keep="first").copy()

    # Split factors from one pair of bars requests.
    filed_min = first["filed"].min()
    filed_max = first["filed"].max()
    start = (
        pd.Timestamp(history_start)
        if history_start is not None
        else filed_min - pd.Timedelta(days=10)
    )
    end = filed_max
    factors = (
        _factor_series(
            price_source,
            ticker,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        if start <= end
        else pd.Series(dtype="float64")
    )
    first["eps_adj"] = [
        row["val"] / _factor_at(factors, row["filed"]) for _, row in first.iterrows()
    ]
    first["duration"] = (first["end"] - first["start"]).dt.days

    directs = first[first["duration"].between(*_QUARTER_DAYS)]
    annuals = first[first["duration"].between(*_ANNUAL_DAYS)]

    quarters = []
    for _, row in directs.iterrows():
        row = row.copy()
        row["derived"] = False
        quarters.append({"row": row, "derived": False})
    quarters.sort(key=lambda q: q["row"]["end"])
    # Dedupe direct quarters before deriving Q4s: a later comparative whose
    # start is shifted by a day survives the exact (start, end) rule and
    # would otherwise count as a fourth FY part (skipping the Q4) or pull the
    # derived filed date a year late.
    kept_directs = _dedupe_cluster(quarters)
    deduped_directs = (
        pd.DataFrame([q["row"] for q in kept_directs])
        if kept_directs
        else directs.iloc[0:0]
    )

    # Cluster once more over kept directs + derived rows, so two drifting FY
    # facts' two derived Q4s still collapse to the earliest-filed one.
    all_quarters = kept_directs + _derived_q4_rows(deduped_directs, annuals)
    all_quarters.sort(key=lambda q: q["row"]["end"])
    kept = _dedupe_cluster(all_quarters)

    return _sue_columns(cik, kept, factors, window, min_obs)


def _dedupe_cluster(quarters):
    """Keep one row per end cluster: a quarter ending within
    `_SAME_QUARTER_DAYS` of an already-kept quarter joins its cluster; each
    cluster keeps its earliest-filed row (tie: duration closest to 91 days).
    Output period_ends are unique."""
    kept = []
    cluster = []
    for quarter in quarters:
        if cluster and (
            quarter["row"]["end"] - cluster[-1]["row"]["end"]
        ).days <= _SAME_QUARTER_DAYS:
            cluster.append(quarter)
        else:
            if cluster:
                kept.append(_pick_cluster(cluster))
            cluster = [quarter]
    if cluster:
        kept.append(_pick_cluster(cluster))
    return kept


def _derived_q4_rows(directs, annuals):
    """FY - (Q1+Q2+Q3) Q4s for annual rows with no direct Q4. Every tiling
    check must pass or the Q4 is skipped silently (not an error); YTD 6- and
    9-month facts are ignored."""
    derived = []
    for _, annual in annuals.iterrows():
        year_start, year_end = annual["start"], annual["end"]
        near_end = (directs["end"] - year_end).abs() <= pd.Timedelta(
            days=_SAME_QUARTER_DAYS
        )
        if near_end.any():
            continue  # a direct quarter already ends this fiscal year
        parts = directs[
            (directs["start"] >= year_start - pd.Timedelta(days=_SAME_QUARTER_DAYS))
            & (directs["end"] <= year_end - pd.Timedelta(days=60))
        ].sort_values("end")
        if len(parts) != 3:
            continue
        if abs((parts.iloc[0]["start"] - year_start).days) > _SAME_QUARTER_DAYS:
            continue
        gaps_ok = all(
            0 < (parts.iloc[i + 1]["start"] - parts.iloc[i]["end"]).days
            <= _SAME_QUARTER_DAYS
            for i in range(2)
        )
        if not gaps_ok:
            continue
        q4_start = parts.iloc[2]["end"] + pd.Timedelta(days=1)
        if not (_QUARTER_DAYS[0] <= (year_end - q4_start).days <= _QUARTER_DAYS[1]):
            continue
        row = annual.copy()
        row["start"] = q4_start
        # NaN propagates: a missing split factor on any part makes the derived
        # Q4 missing (never silently wrong), while the row itself is kept.
        row["eps_adj"] = annual["eps_adj"] - parts["eps_adj"].sum(skipna=False)
        row["val"] = np.nan  # no as-filed value; eps_diluted uses eps_adj * factor
        row["filed"] = max(parts["filed"].max(), annual["filed"])
        row["duration"] = (row["end"] - row["start"]).days
        row["derived"] = True
        derived.append({"row": row, "derived": True})
    return derived


def _pick_cluster(cluster):
    return min(
        cluster,
        key=lambda q: (q["row"]["filed"], abs(q["row"]["duration"] - 91)),
    )


def _sue_columns(cik, kept, factors, window, min_obs) -> pd.DataFrame:
    """Attach the point-in-time SUE columns and emit QUARTER_COLUMNS.

    For each quarter q the view is V_q = {p : filed_p <= filed_q} -- exactly
    what was computable the day EPS_q was first filed. Internal math runs on
    the adjusted (today's) basis; the reported eps_yoy_diff/sue_std/eps_diluted
    are multiplied back onto each quarter's own as-filed basis via
    factor(filed_q). A prior-diff std at or below `_STD_REL_TOL` times the
    largest |diff| (floored at 1.0) is float noise, reported as `sue_std` 0.0
    so `sue` is NaN.
    """
    periods = [q["row"] for q in kept]
    lookback = int(window * 365.25 / 4) + 20

    def diff_of(period, view):
        """eps_adj_p - eps_adj of p's lag-4 quarter inside `view`."""
        earlier = [
            other
            for other in view
            if _LAG4_DAYS[0] <= (period["end"] - other["end"]).days <= _LAG4_DAYS[1]
        ]
        if not earlier:
            return np.nan
        lag = min(earlier, key=lambda other: abs((period["end"] - other["end"]).days - 364))
        if not (np.isfinite(period["eps_adj"]) and np.isfinite(lag["eps_adj"])):
            return np.nan
        return period["eps_adj"] - lag["eps_adj"]

    rows = []
    for period in periods:
        filed_q = period["filed"]
        view = [p for p in periods if p["filed"] <= filed_q]
        d_map = {p["end"]: diff_of(p, view) for p in view}
        d_q = d_map[period["end"]]
        prior = [
            (e, d_map[e])
            for e in sorted(d_map)
            if e < period["end"] and (period["end"] - e).days <= lookback
            and np.isfinite(d_map[e])
        ]
        prior = prior[-window:]
        sue_n = len(prior)
        sue_std = (
            float(np.std([d for _, d in prior], ddof=1)) if sue_n >= min_obs else np.nan
        )
        if np.isfinite(sue_std) and sue_std <= _STD_REL_TOL * max(
            1.0, max(abs(d) for _, d in prior)
        ):
            sue_std = 0.0  # economically identical diffs -- noise, not signal
        sue = (
            d_q / sue_std
            if np.isfinite(d_q) and np.isfinite(sue_std) and sue_std > 0
            else np.nan
        )
        factor = _factor_at(factors, filed_q)
        eps_diluted = (
            period["val"]
            if np.isfinite(period["val"])
            else period["eps_adj"] * factor
        )
        rows.append(
            {
                "cik": cik,
                "period_start": period["start"],
                "period_end": period["end"],
                "filed": filed_q,
                "accn": period["accn"],
                "form": period["form"],
                "tag": period["tag"],
                "eps_diluted": eps_diluted,
                "derived": period["derived"],
                "eps_yoy_diff": d_q * factor if np.isfinite(d_q) else np.nan,
                "sue_std": sue_std * factor if np.isfinite(sue_std) else np.nan,
                "sue_n": sue_n,
                "sue": sue,
            }
        )

    frame = pd.DataFrame(rows, columns=QUARTER_COLUMNS)
    if frame.empty:
        return _empty_quarters()
    for column in ("period_start", "period_end", "filed"):
        frame[column] = pd.to_datetime(frame[column]).astype("datetime64[ns]")
    for column in ("eps_diluted", "eps_yoy_diff", "sue_std", "sue"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    frame["derived"] = frame["derived"].astype("bool")
    frame["sue_n"] = frame["sue_n"].astype("int64")
    return frame.sort_values("period_end").reset_index(drop=True)


def coerce_sue_dtypes(frame: pd.DataFrame) -> pd.DataFrame:
    """Cast the `SUE_EVENT_COLUMNS` to their contract dtypes: datetime64 x2,
    float64, nullable `boolean`, float64 x2, nullable `Int64`, float64."""
    out = frame.copy()
    for column in ("eps_period_end", "eps_filed"):
        out[column] = pd.to_datetime(out[column]).astype("datetime64[ns]")
    for column in ("eps_diluted", "eps_yoy_diff", "sue_std", "sue"):
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    out["eps_derived"] = out["eps_derived"].astype("boolean")
    out["sue_n"] = out["sue_n"].astype("Int64")
    return out


def attach_sue(events: pd.DataFrame, quarters: pd.DataFrame) -> pd.DataFrame:
    """A copy of `events` with `SUE_EVENT_COLUMNS` appended.

    An event on `event_date` E matches the quarter with the latest
    `period_end` among those ending before E, at most `_MATCH_MAX_DAYS`
    earlier, and first `filed` no earlier than `_MATCH_FILED_SLACK_DAYS`
    before E (the 10-Q/10-K usually lands days to weeks after the 8-K) --
    the filed rule stops a mid-quarter Item 2.02 from matching the previous
    quarter, whose 10-Q is already months old.
    """
    out = events.copy()
    for column in SUE_EVENT_COLUMNS:
        out[column] = pd.NA
    if len(out) and len(quarters):
        for idx, event in out.iterrows():
            event_date = event["event_date"]
            candidates = quarters[
                (quarters["period_end"] < event_date)
                & ((event_date - quarters["period_end"]).dt.days <= _MATCH_MAX_DAYS)
                & (
                    quarters["filed"]
                    >= event_date - pd.Timedelta(days=_MATCH_FILED_SLACK_DAYS)
                )
            ]
            if candidates.empty:
                continue
            match = candidates.loc[candidates["period_end"].idxmax()]
            out.loc[idx, "eps_period_end"] = match["period_end"]
            out.loc[idx, "eps_filed"] = match["filed"]
            out.loc[idx, "eps_diluted"] = match["eps_diluted"]
            out.loc[idx, "eps_derived"] = bool(match["derived"])
            out.loc[idx, "eps_yoy_diff"] = match["eps_yoy_diff"]
            out.loc[idx, "sue_std"] = match["sue_std"]
            out.loc[idx, "sue_n"] = match["sue_n"]
            out.loc[idx, "sue"] = match["sue"]
    return coerce_sue_dtypes(out)
