import warnings

from data_universe.membership.errors import (
    MembershipError,
    MembershipStaleError,
    MembershipStaleWarning,
    MembershipValidationError,
)


def test_stale_error_is_a_membership_error():
    assert issubclass(MembershipStaleError, MembershipError)


def test_validation_error_is_a_membership_error():
    assert issubclass(MembershipValidationError, MembershipError)


def test_stale_warning_is_a_warning_not_an_exception():
    assert issubclass(MembershipStaleWarning, Warning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.warn("too old", MembershipStaleWarning)
        assert len(caught) == 1
        assert issubclass(caught[0].category, MembershipStaleWarning)
