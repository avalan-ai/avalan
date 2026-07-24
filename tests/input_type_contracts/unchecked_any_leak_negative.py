"""Fail when public interaction projections are not statically typed."""

from avalan.event import Event, InteractionLifecyclePayload
from avalan.interaction import (
    InputRequest,
    StateRevision,
    mark_request_pending,
)


def reject_unchecked_result(request: InputRequest) -> None:
    """Assign a typed transition result to an impossible target."""
    result: str = mark_request_pending(
        request,
        expected_state_revision=StateRevision(0),
    )
    print(result)


def reject_unchecked_event(payload: InteractionLifecyclePayload) -> None:
    """Assign typed event projections to impossible string targets."""
    unchecked_payload: str = Event.from_interaction_lifecycle(payload).payload
    unchecked_lifecycle: str = Event.from_interaction_lifecycle(
        payload
    ).interaction_lifecycle_payload
    print(unchecked_payload, unchecked_lifecycle)
