"""Reject substitution or public construction of Phase 12 record bytes."""

from avalan.patch.audience_projection import (
    AudienceProjectionAccess,
    AuditRecordAuthority,
    AuditRecordBoundary,
    MetricsRecordAuthority,
    PatchAudienceProjectionSource,
)
from avalan.patch.audience_retention import (
    AudienceRetentionReadAuthority,
    AudienceRetentionReadReceipt,
    AudienceRetentionService,
    AudienceRetentionWriter,
)
from avalan.patch.domain import PatchRetentionRecordId
from avalan.patch.projection_codec import (
    AuditRecordDelivery,
    MetricsRecordDelivery,
)


def reject_metrics_authority_as_audit(
    boundary: AuditRecordBoundary,
    authority: MetricsRecordAuthority,
) -> AuditRecordDelivery:
    """Pass a metrics witness to the audit-only delivery boundary."""
    return boundary.project(authority)


def reject_metrics_delivery_as_audit(
    delivery: MetricsRecordDelivery,
) -> AuditRecordDelivery:
    """Treat detached metric bytes as a distinct audit record type."""
    return delivery


def reject_public_canonical_record_codec() -> AuditRecordDelivery:
    """Construct privileged canonical delivery bytes outside its boundary."""
    return AuditRecordDelivery()


def reject_audit_authority_source(
    authority: AuditRecordAuthority,
) -> None:
    """Read a deliberately unavailable canonical source value."""
    authority.source


def reject_public_projection_source() -> PatchAudienceProjectionSource:
    """Construct canonical source truth without a host witness."""
    return PatchAudienceProjectionSource()


def reject_public_projection_access() -> AudienceProjectionAccess:
    """Construct projection access without a host witness."""
    return AudienceProjectionAccess()


def reject_public_retention_writer() -> AudienceRetentionWriter:
    """Construct retention writer without an authorization witness."""
    return AudienceRetentionWriter()


def reject_public_retention_reader() -> AudienceRetentionReadAuthority:
    """Construct a reader authority without its service-issued witness."""
    return AudienceRetentionReadAuthority()


async def reject_writer_as_retention_reader(
    service: AudienceRetentionService,
    writer: AudienceRetentionWriter,
    retention_id: PatchRetentionRecordId,
) -> AudienceRetentionReadReceipt:
    """Open retained bytes with the wrong service capability type."""
    return await service.open(writer, retention_id)
