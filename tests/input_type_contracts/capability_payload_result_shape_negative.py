"""Reject untyped capability payloads and wrong-shaped call results."""

from collections.abc import Mapping

from avalan.model import (
    CorrelatedCapabilityResult,
    ModelCapabilityCatalog,
    TaskInputCapabilityCall,
)


def reject_untyped_call(
    catalog: ModelCapabilityCatalog, payload: object
) -> None:
    """Reject an untyped provider call at the catalog boundary."""
    catalog.decode_call(payload)


def reject_wrong_model_result(
    catalog: ModelCapabilityCatalog,
    call: TaskInputCapabilityCall,
    payload: Mapping[str, object],
) -> None:
    """Reject a dictionary in place of a typed input model result."""
    catalog.project_result(call, payload)


def reject_wrong_correlated_payload(payload: list[object]) -> None:
    """Reject a non-mapping correlated result payload."""
    CorrelatedCapabilityResult(
        call_id="call-1",
        canonical_name="request_user_input",
        provider_name="request_user_input",
        payload=payload,
    )
