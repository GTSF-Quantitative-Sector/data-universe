from data_universe.membership.derive import derive_ticker_intervals


def test_simple_add_and_remove():
    snapshots = [
        ("2020-01-01", {"AAA", "BBB"}),
        ("2020-02-01", {"AAA", "BBB", "CCC"}),  # CCC added
        ("2020-03-01", {"AAA", "CCC"}),          # BBB removed
    ]
    result = sorted(derive_ticker_intervals(snapshots))
    assert result == sorted(
        [
            ("AAA", "2020-01-01", None),
            ("BBB", "2020-01-01", "2020-03-01"),
            ("CCC", "2020-02-01", None),
        ]
    )


def test_ticker_reused_after_a_gap_is_two_intervals():
    # Same ticker leaves and rejoins -- derive doesn't know or care whether
    # it's the same company (that's identity.py's job in PR2); it just
    # records two intervals.
    snapshots = [
        ("1996-01-01", {"AAL"}),
        ("1997-01-01", set()),
        ("2015-01-01", {"AAL"}),
        ("2024-01-01", set()),
    ]
    result = derive_ticker_intervals(snapshots)
    aal_intervals = sorted(t for t in result if t[0] == "AAL")
    assert aal_intervals == [
        ("AAL", "1996-01-01", "1997-01-01"),
        ("AAL", "2015-01-01", "2024-01-01"),
    ]


def test_same_day_replacement_produces_adjacent_intervals():
    snapshots = [
        ("2013-01-01", {"FB"}),
        ("2022-06-09", {"META"}),  # FB removed and META added same snapshot
    ]
    result = sorted(derive_ticker_intervals(snapshots))
    assert result == [("FB", "2013-01-01", "2022-06-09"), ("META", "2022-06-09", None)]


def test_no_change_row_produces_no_extra_interval():
    snapshots = [
        ("2020-01-01", {"AAA"}),
        ("2020-02-01", {"AAA"}),  # identical row
        ("2020-03-01", set()),
    ]
    result = derive_ticker_intervals(snapshots)
    assert result == [("AAA", "2020-01-01", "2020-03-01")]
