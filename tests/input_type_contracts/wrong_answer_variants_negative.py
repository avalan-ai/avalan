"""Reject untyped resolution variants at the public transition boundary."""

from avalan.interaction import InputRequest, StateRevision, resolve_request


def reject_wrong_variants(request: InputRequest) -> None:
    """Exercise strict resolution-variant arguments."""
    resolve_request(
        request,
        "answered",
        expected_state_revision=StateRevision(0),
    )
    resolve_request(
        request,
        1,
        expected_state_revision=StateRevision(0),
    )
