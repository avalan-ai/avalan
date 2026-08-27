"""Assert planned Phase 12 data-only audience projections stay typed."""

from typing import assert_type

from avalan.patch.audience_projection import (
    AudienceProjectionAccess,
    AudienceProjectionHost,
    AuditRecordAuthority,
    AuditRecordBoundary,
    MetricsRecordAuthority,
    MetricsRecordBoundary,
    PatchAudienceProjectionSource,
    ServerRecordAuthority,
    ServerRecordBoundary,
    TelemetryRecordAuthority,
    TelemetryRecordBoundary,
    create_audit_record_boundary,
    create_metrics_record_boundary,
    create_server_record_boundary,
    create_telemetry_record_boundary,
)
from avalan.patch.audience_retention import (
    AudienceRetainedValue,
    AudienceRetentionReadAuthority,
    AudienceRetentionReadReceipt,
    AudienceRetentionService,
    AudienceRetentionWriter,
    AudienceRetentionWriteReceipt,
)
from avalan.patch.domain import (
    PatchObserverCorrelationId,
    PatchRetentionRecordId,
)
from avalan.patch.durable_store import (
    DurableRequestAccess,
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionKind,
)
from avalan.patch.policy import SealedPlan
from avalan.patch.projection_codec import (
    AuditRecordDelivery,
    MetricsRecordDelivery,
    ServerRecordDelivery,
    TelemetryRecordDelivery,
)


def assert_audit_retention_projection_types(
    source: PatchAudienceProjectionSource,
) -> None:
    """Require each lower audience to receive only its detached record."""
    audit = create_audit_record_boundary(source)
    metrics = create_metrics_record_boundary(source)
    telemetry = create_telemetry_record_boundary(source)
    server = create_server_record_boundary(source)
    assert_type(audit, AuditRecordBoundary)
    assert_type(metrics, MetricsRecordBoundary)
    assert_type(telemetry, TelemetryRecordBoundary)
    assert_type(server, ServerRecordBoundary)
    audit_authority = audit.authority()
    metrics_authority = metrics.authority()
    telemetry_authority = telemetry.authority()
    server_authority = server.authority()
    assert_type(audit_authority, AuditRecordAuthority)
    assert_type(metrics_authority, MetricsRecordAuthority)
    assert_type(telemetry_authority, TelemetryRecordAuthority)
    assert_type(server_authority, ServerRecordAuthority)
    assert_type(audit.project(audit_authority), AuditRecordDelivery)
    assert_type(metrics.project(metrics_authority), MetricsRecordDelivery)
    assert_type(
        telemetry.project(telemetry_authority), TelemetryRecordDelivery
    )
    assert_type(server.project(server_authority), ServerRecordDelivery)


async def assert_host_projection_source_types(
    host: AudienceProjectionHost,
    access: DurableRequestAccess,
    correlation: PatchObserverCorrelationId,
    plan: SealedPlan,
) -> None:
    """Require store-issued projection construction types."""
    witness = await host.issue_access(access, correlation)
    assert_type(witness, AudienceProjectionAccess)
    assert_type(
        await host.source(plan, witness), PatchAudienceProjectionSource
    )


async def assert_retention_writer_types(
    service: AudienceRetentionService,
    reservation: DurableReservation,
    writer: AudienceRetentionWriter,
    value: AudienceRetainedValue,
) -> None:
    """Require policy-issued writer and receipt types."""
    issued = await service.issue_writer(
        reservation, DurableRetentionKind.AUDIT_PROJECTION
    )
    assert_type(issued, AudienceRetentionWriter | None)
    assert_type(
        await service.retain(writer, value), AudienceRetentionWriteReceipt
    )


async def assert_retention_reader_types(
    service: AudienceRetentionService,
    reservation: DurableReservation,
    access: DurableRetentionAccess,
    reader: AudienceRetentionReadAuthority,
    retention_id: PatchRetentionRecordId,
) -> None:
    """Require exact audience-bound reader authority and result types."""
    issued = await service.issue_read_authority(
        reservation, access, DurableRetentionKind.AUDIT_PROJECTION
    )
    assert_type(issued, AudienceRetentionReadAuthority | None)
    assert_type(
        await service.open(reader, retention_id), AudienceRetentionReadReceipt
    )
