"""Reject untyped public resolution payloads."""

from avalan import (
    InputContinuationRef,
    InputController,
    InputRequestRef,
    ResolutionIdempotencyKey,
    resolve_input,
)


async def resolve_untyped(
    controller: InputController,
    request_id: InputRequestRef,
    continuation_id: InputContinuationRef,
    idempotency_key: ResolutionIdempotencyKey,
) -> None:
    """Pass a dictionary where the public API requires a typed submission."""
    await resolve_input(
        controller,
        request_id,
        continuation_id,
        {"answers": []},
        idempotency_key=idempotency_key,
    )
