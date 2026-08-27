"""Reject lower-consumer attempts to influence trusted projection delivery."""

from dataclasses import asdict

import avalan.patch.projection_codec as codec
from avalan.patch.durable_outbox import (
    EventManagerDurableOutboxProjection,
)
from avalan.patch.durable_store import DurableOutboxRecord
from avalan.patch.projection import (
    AuditProjectionAuthority,
    ModelProjectionAuthority,
    ModelProjectionBoundary,
    ProjectionOutputLimit,
)
from avalan.patch.projection_codec import (
    AuditProjectionDelivery,
    ModelProjectionDelivery,
)


def reject_cross_audience_authority(
    boundary: ModelProjectionBoundary,
    authority: AuditProjectionAuthority,
    output_limit: ProjectionOutputLimit,
) -> ModelProjectionDelivery:
    """Call the model boundary with a deliberately wrong authority."""
    return boundary.project(authority, output_limit)


def reject_cross_audience_delivery(
    delivery: ModelProjectionDelivery,
) -> AuditProjectionDelivery:
    """Return model bytes as an incompatible audit delivery type."""
    return delivery


def reject_authority_source_access(
    authority: ModelProjectionAuthority,
) -> None:
    """Read the deliberately unavailable privileged source field."""
    authority.source


def reject_delivery_constructor() -> ModelProjectionDelivery:
    """Construct host-delivery bytes without the trusted host path."""
    return ModelProjectionDelivery()


def reject_host_decoder_access() -> None:
    """Read a deliberately absent lower-consumer decoder capability."""
    codec.decode_model_projection


def reject_delivery_mutation(delivery: ModelProjectionDelivery) -> None:
    """Alter an immutable lower delivery byte."""
    delivery[0] = 0


def reject_generic_dataclass_codec(delivery: ModelProjectionDelivery) -> None:
    """Pass primitive delivery bytes to generic dataclass serialization."""
    asdict(delivery)


class ForgedProjection(ModelProjectionDelivery):
    """Attempt to subclass one final detached host delivery."""


async def reject_generic_progress_record_input(
    projection: EventManagerDurableOutboxProjection,
    record: DurableOutboxRecord,
) -> None:
    """Pass a fabricated durable record where a cursor is required."""
    await projection.project(record, 1)
