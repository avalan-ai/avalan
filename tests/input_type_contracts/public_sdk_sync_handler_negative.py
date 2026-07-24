"""Reject a synchronous handler through public imports."""

from avalan import (
    AttachedInputContext,
    AttachedInputDetached,
    AttachedInputOutcome,
    create_attached_input_runtime,
)


def detach(
    context: AttachedInputContext,
) -> AttachedInputOutcome:
    """Return a typed outcome through an invalid synchronous callback."""
    _ = context
    return AttachedInputDetached()


async def construct_sync_handler() -> None:
    """Pass a synchronous callback to the async runtime factory."""
    await create_attached_input_runtime(detach)
