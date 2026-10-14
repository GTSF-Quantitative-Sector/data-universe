"""Exceptions and warnings for data_universe.membership.

Not a subclass of `data_universe.sources.base.DataSourceError` -- membership
is not a `DataSource` (no `daily_bars`/`hourly_bars`), so sharing that
hierarchy would be misleading.
"""
from __future__ import annotations


class MembershipError(Exception):
    """Base class for all data_universe.membership errors."""


class MembershipValidationError(MembershipError):
    """A committed table violates an invariant (overlapping intervals for one
    security, a ticker_intervals row with no matching security_map row,
    a duplicate security_id, a member count outside the expected range)."""


class MembershipStaleError(MembershipError):
    """Raised by `membership_grid(strict=True)` when the data is older than
    `config.get_membership_max_staleness_days()`."""


class MembershipStaleWarning(UserWarning):
    """Emitted by `membership_grid(strict=False)` (the default) in the same
    situation `MembershipStaleError` would otherwise raise for."""
