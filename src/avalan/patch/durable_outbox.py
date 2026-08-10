"""Project dormant durable lifecycle outbox records without settlement writes.

Delivery and audit observers are deliberately best effort.  They are never a
source of mutation truth, never acknowledge durable records, and cannot change
the request, journal, terminal result, or future at-least-once retry.
"""

from asyncio import Lock
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from avalan.patch.domain import SequenceNumber
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurablePatchStore,
    DurableRequestAccess,
    DurableStoreError,
    DurableStoreErrorCode,
)


class DurableOutboxDelivery(Protocol):
    """Deliver one durable lifecycle record to a best-effort projection."""

    async def deliver(self, record: DurableOutboxRecord) -> None:
        """Deliver one content-free stable lifecycle event."""


class DurableAuditProjection(Protocol):
    """Observe one durable lifecycle record without controlling settlement."""

    async def project(self, record: DurableOutboxRecord) -> None:
        """Project one content-free lifecycle event to an audit sink."""


class EventManagerDurableOutboxProjection:
    """Fan out durable records to EventManager without owning request truth.

    The durable store remains the replay authority.  This bounded in-process
    projection suppresses repeated stable identities only while it is alive;
    an EventManager failure releases the identity so the durable outbox can
    retry it later.
    """

    def __init__(
        self,
        event_manager: EventManager,
        *,
        deduplication_limit: int = 8192,
    ) -> None:
        """Bind one fallible generic-event projection with bounded dedupe."""
        if (
            type(event_manager) is not EventManager
            or type(deduplication_limit) is not int
            or not 1 <= deduplication_limit <= 65_536
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        self._event_manager = event_manager
        self._deduplication_limit = deduplication_limit
        self._event_ids: set[str] = set()
        self._event_order: deque[str] = deque()
        self._lock = Lock()

    async def deliver(self, record: DurableOutboxRecord) -> None:
        """Project one record once per live stable event identity."""
        if type(record) is not DurableOutboxRecord:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        event_id = record.event_id.value
        async with self._lock:
            if event_id in self._event_ids:
                return
            self._event_ids.add(event_id)
        try:
            await self._event_manager.trigger(
                Event(
                    type=EventType.TOOL_PROGRESS,
                    payload={
                        "correlation_id": record.correlation_id.value,
                        "event_id": event_id,
                        "lifecycle": record.lifecycle.value,
                        "request_id": record.request_id.value,
                        "sequence": record.sequence.value,
                    },
                )
            )
        except BaseException:
            async with self._lock:
                self._event_ids.discard(event_id)
            raise
        async with self._lock:
            self._event_order.append(event_id)
            while len(self._event_order) > self._deduplication_limit:
                self._event_ids.discard(self._event_order.popleft())


@dataclass(frozen=True, slots=True)
class DurableOutboxProjectionReceipt:
    """Report delivery and audit outcomes without sensitive failure detail."""

    records: tuple[DurableOutboxRecord, ...]
    delivered: int
    delivery_failed: bool
    audit_failed: bool

    def __post_init__(self) -> None:
        """Require accounting aligned with the observed event vector."""
        if (
            type(self.records) is not tuple
            or any(
                type(item) is not DurableOutboxRecord for item in self.records
            )
            or type(self.delivered) is not int
            or not 0 <= self.delivered <= len(self.records)
            or type(self.delivery_failed) is not bool
            or type(self.audit_failed) is not bool
            or (self.delivery_failed and self.delivered == len(self.records))
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


class DurableOutboxProjector:
    """Read and project durable events while isolating delivery failures."""

    def __init__(
        self,
        store: DurablePatchStore,
        delivery: DurableOutboxDelivery,
        audit: DurableAuditProjection,
    ) -> None:
        """Bind non-authoritative delivery and audit projections."""
        if (
            not callable(getattr(store, "outbox", None))
            or not callable(getattr(delivery, "deliver", None))
            or not callable(getattr(audit, "project", None))
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        self._store = store
        self._delivery = delivery
        self._audit = audit

    async def project(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> DurableOutboxProjectionReceipt:
        """Deliver events in order and retain all failures for later retry."""
        if (
            type(access) is not DurableRequestAccess
            or type(after) is not SequenceNumber
            or type(limit) is not int
            or not 1 <= limit <= 1024
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        records = await self._store.outbox(access, after, limit)
        delivered = 0
        delivery_failed = False
        audit_failed = False
        for record in records:
            try:
                await self._delivery.deliver(record)
            except Exception as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                delivery_failed = True
                break
            delivered += 1
            try:
                await self._audit.project(record)
            except Exception as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                audit_failed = True
        return DurableOutboxProjectionReceipt(
            records,
            delivered,
            delivery_failed,
            audit_failed,
        )
