"""Vendor-name check: does a Massive `?date=` record's registrant name match
the SEC registrant for the CIK it carries? Old records sometimes carry
another company's CIK under the right name (XRAY -> Danaher's 313616);
`names_match` is the tolerant comparison that keeps real renames
(FB -> "Meta Platforms") while rejecting wrong-CIK records.

Deliberately heuristic -- it must accept historical name quirks (abbreviations,
"AND"/"&", spaced-out initials like "F M C", hyphenated vs merged words like
MCGRAW-HILL / MCGRAWHILL, one-char typos like ROHM AND HASS / ROHM & HAAS,
"Series A Common Stock" security names) but still reject genuinely different
registrants. Each name is tokenized twice -- hyphens removed and hyphens as
spaces -- and a pair matches if any variant pair matches. False positives are
still possible; they become `cik_name_mismatch` review rows triaged by
hand (a missed CIK, never a wrong one).
"""
from __future__ import annotations

import re

SUFFIX_TOKENS = frozenset({
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "COS",
    "LTD", "LIMITED", "PLC", "LLC", "LP", "NV", "SA", "AG", "SE", "BV",
    "THE", "CLASS", "CL", "COM", "COMMON", "STOCK", "STK", "SHS", "SHARES",
    "ORD", "ORDINARY", "NEW", "DE", "DEL", "ADR", "ADS", "HOLDINGS",
    "HOLDING", "HLDGS", "HLDG", "GROUP", "GRP", "AND", "A", "B", "C",
    "SERIES",
})
ABBREVIATIONS = {
    "INTL": "INTERNATIONAL", "SVC": "SERVICE", "SVCS": "SERVICES",
    "MFG": "MANUFACTURING", "MKT": "MARKET", "MKTS": "MARKETS",
    "FINL": "FINANCIAL", "NATL": "NATIONAL", "TECH": "TECHNOLOGIES",
    "SYS": "SYSTEMS", "PPTYS": "PROPERTIES", "COMMUNS": "COMMUNICATIONS",
    "CENTY": "CENTURY", "INVST": "INVESTMENT", "MGMT": "MANAGEMENT",
    "ENRGY": "ENERGY", "GP": "GROUP", "INDS": "INDUSTRIES",
    "PETE": "PETROLEUM", "PHARMS": "PHARMACEUTICALS", "HLTH": "HEALTH",
    "ELEC": "ELECTRIC", "PWR": "POWER", "INSTRS": "INSTRUMENTS",
}


def _join_initials(tokens):
    """Join runs of >=2 consecutive single-character tokens (spaced-out
    initials like "F M C" -> "FMC"); a lone single letter is untouched."""
    out, i = [], 0
    while i < len(tokens):
        if len(tokens[i]) == 1:
            j = i
            while j < len(tokens) and len(tokens[j]) == 1:
                j += 1
            run = tokens[i:j]
            out.extend(run) if len(run) == 1 else out.append("".join(run))
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def name_tokens(name, hyphen_as_space=False):
    """Normalize `name` to its significant tokens. Hyphens are removed by
    default ("MCGRAW-HILL" -> "MCGRAWHILL"); `hyphen_as_space` splits
    ("CL-A" -> "CL" + "A") -- `names_match` tries both."""
    s = (name or "").upper().replace("&", " ")
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"\([^)]*$", " ", s)
    s = re.sub(r"/[A-Z]{2,3}/?", " ", s)
    s = s.replace("-", " " if hyphen_as_space else "")
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    toks = _join_initials(s.split())
    return [
        t for t in (ABBREVIATIONS.get(t, t) for t in toks)
        if t not in SUFFIX_TOKENS
    ]


def _edit1(a, b):
    """Levenshtein distance between a and b (small DP, cap the strip)."""
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return abs(la - lb)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[lb]


def _token_match(a, b):
    return (
        a == b
        or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))
        or (len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4])
        or (len(a) >= 4 and len(b) >= 4 and _edit1(a, b) <= 1)
    )


def _pair_match(v, s):
    vj = "".join(v)
    sj = "".join(s)
    if min(len(vj), len(sj)) >= 4 and (vj in sj or sj in vj):
        return True
    matched = sum(1 for t in v if any(_token_match(t, u) for u in s))
    return matched >= min(2, len(v), len(s))


def names_match(vendor_name, sec_names) -> bool:
    v_variants = {tuple(name_tokens(vendor_name)),
                  tuple(name_tokens(vendor_name, hyphen_as_space=True))}
    v_variants = {v for v in v_variants if v}
    if not v_variants:
        return True
    secs = [t for n in sec_names if n
            for t in (name_tokens(n), name_tokens(n, hyphen_as_space=True))]
    secs = {tuple(t) for t in secs if t}
    if not secs:
        return True
    return any(
        _pair_match(v, s) for v in v_variants for s in secs
    )
