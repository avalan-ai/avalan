"""Reject a sync-or-awaitable protocol return type."""

from collections.abc import Awaitable


def reject_sync_or_awaitable() -> Awaitable[int] | int:
    """Keep one deliberately ambiguous async return type for the fixture."""
    return 1
