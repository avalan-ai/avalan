"""Reject cast-based patch type erasure."""

from typing import cast


def reject_cast(value: object) -> int:
    """Keep a deliberately forbidden cast for the fixture."""
    return cast(int, value)
