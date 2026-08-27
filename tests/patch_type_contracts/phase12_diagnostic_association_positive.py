"""Assert planned Phase 12 deliveries are detached typed data."""

from typing import assert_type

from avalan.patch.diagnostic_association import (
    DiagnosticApprovalReceipt,
    DiagnosticAssociation,
    DiagnosticCapability,
    FormatterFixerResult,
    RemediationPatchAuthorization,
    RemediationPatchRequester,
    diagnostic_retention_kind,
)
from avalan.patch.domain import PatchRequestId
from avalan.patch.durable_store import DurableRetentionKind


def assert_data_only_diagnostic_deliveries(
    capability_bytes: bytes,
    receipt_bytes: bytes,
    association_bytes: bytes,
) -> None:
    """Keep delivery values as bytes without a service or approval method."""
    assert_type(DiagnosticCapability(capability_bytes), DiagnosticCapability)
    assert_type(
        DiagnosticApprovalReceipt(receipt_bytes), DiagnosticApprovalReceipt
    )
    assert_type(
        DiagnosticAssociation(association_bytes), DiagnosticAssociation
    )


async def assert_remediation_starts_a_new_patch_request(
    requester: RemediationPatchRequester,
    authorization: RemediationPatchAuthorization,
    result: FormatterFixerResult,
) -> None:
    """Require separate remediation authority to request a new patch."""
    assert_type(
        await requester.begin_new_patch_request(authorization, result),
        PatchRequestId,
    )


def assert_diagnostic_retention_kind() -> None:
    """Name diagnostics retention without any retention write authority."""
    assert_type(diagnostic_retention_kind(), DurableRetentionKind)
