"""Define tool-cycle limit settings."""

from typing import Any, Literal, TypeAlias, cast

UNLIMITED_TOOL_CYCLES: Literal["unlimited"] = "unlimited"
MaximumToolCycles: TypeAlias = int | Literal["unlimited"]


def validate_maximum_tool_cycles(value: Any) -> MaximumToolCycles:
    """Return a valid maximum tool cycle setting.

    Args:
        value: Candidate maximum tool cycle setting.

    Returns:
        Validated maximum tool cycle setting.
    """
    assert (
        value == UNLIMITED_TOOL_CYCLES or type(value) is int and value > 0
    ), "maximum_tool_cycles must be a positive integer or 'unlimited'"
    return cast(MaximumToolCycles, value)
