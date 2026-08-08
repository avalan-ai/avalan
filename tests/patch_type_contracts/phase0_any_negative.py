"""Reject explicit Any at the patch strict boundary."""

from typing import Any


def reject_any(value: Any) -> None:
    """Keep a deliberately forbidden dynamic value for the fixture."""
    assert value is not None
