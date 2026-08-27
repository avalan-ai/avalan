"""Reject public host creation, self approval, and authority swaps."""

import avalan.patch.diagnostic_association as diagnostic_association
from avalan.patch.diagnostic_association import (
    DiagnosticApprovalReceipt,
    DiagnosticAssociation,
    FormatterFixerResult,
    RemediationPatchRequester,
)
from avalan.patch.domain import PatchRequestId


def reject_public_chosen_policy_factory() -> object:
    """Attempt to construct a diagnostic host with a caller policy."""
    return diagnostic_association.create_diagnostic_association_service


def reject_public_self_approval(
    receipt: DiagnosticApprovalReceipt,
) -> object:
    """Attempt to approve through a delivered data-only receipt."""
    return receipt.approve


async def reject_association_as_remediation_authority(
    requester: RemediationPatchRequester,
    association: DiagnosticAssociation,
    result: FormatterFixerResult,
) -> PatchRequestId:
    """Turn a diagnostic association into a patch write authority."""
    return await requester.begin_new_patch_request(association, result)
