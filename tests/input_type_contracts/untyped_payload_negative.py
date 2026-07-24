"""Reject dictionary payloads at the typed resolution boundary."""

from avalan.interaction import InputRequest, StateRevision, resolve_request


def reject_untyped_payload(request: InputRequest) -> None:
    """Exercise the typed resolution requirement."""
    payload: dict[str, object] = {"status": "answered"}
    resolve_request(
        request,
        payload,
        expected_state_revision=StateRevision(0),
    )
