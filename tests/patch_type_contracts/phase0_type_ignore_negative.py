"""Reject ignored type errors at the patch strict boundary."""


def reject_ignored_error() -> None:
    """Keep a deliberately forbidden ignored assignment for the fixture."""
    value: int = "not-an-int"  # type: ignore[assignment]
    assert value
