"""Reject interchange of opaque public request and continuation refs."""

from avalan import (
    InputContinuationRef,
    InputController,
    InputRequestRef,
    inspect_input,
)


async def inspect_swapped(
    controller: InputController,
    request_id: InputRequestRef,
    continuation_id: InputContinuationRef,
) -> None:
    """Swap the request and continuation references."""
    await inspect_input(
        controller,
        continuation_id,
        request_id,
    )
