"""Assert the planned trusted host projection boundary stays typed."""

from typing import assert_type

from avalan.patch.domain import PatchPublicCorrelationId, SequenceNumber
from avalan.patch.durable_outbox import (
    DurableOutboxProjectionReceipt,
    EventManagerDurableOutboxProjection,
)
from avalan.patch.projection import (
    ApproverProjectionAuthority,
    ApproverProjectionBoundary,
    AuditProjectionAuthority,
    AuditProjectionBoundary,
    ModelProjectionAuthority,
    ModelProjectionBoundary,
    PatchProjectionSource,
    ProjectionOutputLimit,
    create_approver_projection_boundary,
    create_audit_projection_boundary,
    create_model_projection_boundary,
)
from avalan.patch.projection_codec import (
    ApproverProjectionDelivery,
    AuditProjectionDelivery,
    ModelProjectionDelivery,
)


def assert_audience_projection_types(
    source: PatchProjectionSource,
    output_limit: ProjectionOutputLimit,
) -> None:
    """Require separate trusted boundaries and detached delivery bytes."""
    model = create_model_projection_boundary(source)
    approver = create_approver_projection_boundary(source)
    audit = create_audit_projection_boundary(source)
    assert_type(model, ModelProjectionBoundary)
    assert_type(approver, ApproverProjectionBoundary)
    assert_type(audit, AuditProjectionBoundary)
    model_authority = model.authority()
    approver_authority = approver.authority()
    audit_authority = audit.authority()
    assert_type(model_authority, ModelProjectionAuthority)
    assert_type(approver_authority, ApproverProjectionAuthority)
    assert_type(audit_authority, AuditProjectionAuthority)
    assert_type(model_authority.correlation_id, PatchPublicCorrelationId)
    assert_type(approver_authority.correlation_id, PatchPublicCorrelationId)
    assert_type(audit_authority.correlation_id, PatchPublicCorrelationId)
    assert_type(
        model.project(model_authority, output_limit), ModelProjectionDelivery
    )
    assert_type(
        approver.project(approver_authority), ApproverProjectionDelivery
    )
    assert_type(audit.project(audit_authority), AuditProjectionDelivery)


async def assert_generic_progress_types(
    projection: EventManagerDurableOutboxProjection,
) -> None:
    """Require generic progress to read only its bound trusted store."""
    assert_type(projection, EventManagerDurableOutboxProjection)
    assert_type(
        await projection.project(SequenceNumber(0), 1),
        DurableOutboxProjectionReceipt,
    )
