"""Table-driven `names_match` cases -- the vendor-name check that decides
whether a Massive record's CIK is trusted."""
import pytest

from data_universe.membership.names import name_tokens, names_match

MATCH = [
    ("INTL BUSINESS MACHINES CORP", ["INTERNATIONAL BUSINESS MACHINES CORP"]),
    ("Facebook, Inc. Class A", ["Meta Platforms, Inc.", "FACEBOOK INC"]),
    ("PUB SVC ENTERPRISE GROUP INC", ["PUBLIC SERVICE ENTERPRISE GROUP INC"]),
    ("BERKSHIRE HATHWY INC(HLDG CO)B", ["BERKSHIRE HATHAWAY INC"]),
    ("MCGRAW-HILL FINANCIAL INC COM", ["S&P Global Inc.", "MCGRAW HILL FINANCIAL INC"]),
    ("L-3 COMMUNICATIONS HLDGS INC", ["L 3 COMMUNICATIONS HOLDINGS INC"]),
    ("TWENTY-FIRST CENTY FOX INC CL A COM STK  (DE)", ["TWENTY-FIRST CENTURY FOX, INC."]),
    ("AT&T INC", ["AT&T INC."]),
    ("", ["X"]),
    ("APARTMENT INVST & MGMT CO", ["APARTMENT INVESTMENT & MANAGEMENT CO"]),
    ("CR Bard Inc.", ["BARD C R INC (PRED)"]),
    ("CONSTELLATION ENRGY GP(HLDG CO", ["CONSTELLATION ENERGY GROUP INC"]),
    ("DR HORTON INC", ["HORTON D R INC /DE/"]),
    ("Discovery, Inc. Series A Common Stock", ["DISCOVERY COMMUNICATIONS INC"]),
    ("F M C Technologies Inc", ["FMC TECHNOLOGIES INC"]),
    ("M B N A CORP", ["MBNA CORP"]),
    ("MARATHON PETE CORP COM STK (DE)", ["Marathon Petroleum Corp"]),
    ("NABORS INDS LTD (NEW) BERMUDA", ["NABORS INDUSTRIES LTD"]),
    ("P M C SIERRA INC", ["PMC SIERRA INC"]),
    ("ROHM AND HASS COMPANY", ["ROHM & HAAS CO"]),
    ('SCRIPPS CO (E.W.) CL-A "NEW"', ["E.W. SCRIPPS Co"]),
    ("T-ROWE PRICE GROUP INC", ["PRICE T ROWE GROUP INC"]),
    ("VERTAS SOFTWARE CORP(DE)", ["VERITAS SOFTWARE CORP /DE/"]),
]

NO_MATCH = [
    ("DENTSPLY INTL INC", ["DANAHER CORP /DE/"]),
    ("L-3 COMMUNICATIONS HLDGS INC", ["JETBLUE AIRWAYS CORP"]),
    ("APOLLO GROUP INC CL A", ["ABIOMED INC"]),
    ("CMS Energy Corporation", ["CONSUMERS ENERGY CO"]),
    ("AMERICAN EXPRESS CO", ["American Airlines Group Inc."]),
    ("YAHOO INC", ["FIELDPOINT PETROLEUM CORP", "ENERGY PRODUCTION CO"]),
    ("ALEXION PHARMACEUTICALS INC", ["Vanda Pharmaceuticals Inc."]),
    ("COMMERCE BANCORP INC.", ["Vulcan Materials CO", "Virginia Holdco, Inc."]),
    ("EMC CORPORATION", ["VMWARE LLC", "VMWARE, INC."]),
    ("LEVEL 3 COMMUNICATIONS INC NEW",
     ["EXPEDITORS INTERNATIONAL OF WASHINGTON INC"]),
    ("MAY DEPARTMENT STORES CO", ["MILLENNIUM CHEMICALS INC"]),
    ("MEREDITH CORP", ["KOCH INDUSTRIES LLC"]),
    ("MONSTER WORLDWIDE INC", ["MORGAN STANLEY"]),
    ("NATIONAL CITY CORP", ["FPIC INSURANCE GROUP INC"]),
    ("PETSMART INC", ["FEDERATED HERMES, INC."]),
    ("R R DONNELLEY & SONS CO", ["GameStop Corp.", "GSC Holdings Corp."]),
    ("SYMANTEC CORP", ["NEXLAND INC"]),
    ("TENET HEALTHCARE CORPORATION", ["HOCKEY CO"]),
    ("WACHOVIA CORP 2ND NEW", ["RYDEX ETF TRUST"]),
]


@pytest.mark.parametrize("vendor,secs", MATCH)
def test_names_match_hits(vendor, secs):
    assert names_match(vendor, secs) is True


@pytest.mark.parametrize("vendor,secs", NO_MATCH)
def test_names_match_misses(vendor, secs):
    assert names_match(vendor, secs) is False


def test_name_tokens_strips_suffixes_and_abbreviates():
    assert name_tokens("PUB SVC ENTERPRISE GROUP INC") == [
        "PUB", "SERVICE", "ENTERPRISE"]
    assert name_tokens("MCGRAW-HILL FINANCIAL INC COM") == [
        "MCGRAWHILL", "FINANCIAL"]
