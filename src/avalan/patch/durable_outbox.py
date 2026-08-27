"""Project dormant durable lifecycle outbox records without settlement writes.

Delivery and audit observers are deliberately best effort.  They are never a
source of mutation truth, never acknowledge durable records, and cannot change
the request, journal, terminal result, or future at-least-once retry.
"""

from asyncio import (
    CancelledError,
    Lock,
    Queue,
    QueueEmpty,
    QueueFull,
    Task,
    create_task,
    current_task,
    gather,
    shield,
)
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from json import dumps, loads
from typing import Awaitable, NoReturn, Protocol, final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from avalan.event import (
    Event,
    EventType,
    project_observer_id,
)
from avalan.event.manager import EventManager
from avalan.patch.domain import (
    LifecyclePhase,
    PatchObserverCorrelationId,
    PatchPublicCorrelationId,
    PatchRequestId,
    SequenceNumber,
)
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurablePatchStore,
    DurableRequestAccess,
    DurableStoreError,
    DurableStoreErrorCode,
)
from avalan.patch.projection_codec import GenericToolProgressDelivery

_GENERIC_PROGRESS_SCHEMA_VERSION = 1
_MAX_GENERIC_PROGRESS_DELIVERY_BYTES = 1_024
_GENERIC_PROGRESS_AUDIENCE = "generic_tool_progress"
_ED25519_SIGNATURE_BYTES = 64


class DurableOutboxDelivery(Protocol):
    """Deliver one durable lifecycle record to a best-effort projection."""

    async def deliver(self, record: DurableOutboxRecord) -> None:
        """Deliver one content-free stable lifecycle event."""


class DurableAuditProjection(Protocol):
    """Observe one durable lifecycle record without controlling settlement."""

    async def project(self, record: DurableOutboxRecord) -> None:
        """Project one content-free lifecycle event to an audit sink."""


class _DurableOutboxReader(Protocol):
    """Read canonical durable outbox records for one authorized request."""

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Return canonical records after one durable cursor."""


class GenericToolProgressPhase(StrEnum):
    """Name the two coarse durable mutation-progress phases."""

    SETTLEMENT_PENDING = "settlement_pending"
    SETTLED = "settled"


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _GenericToolProgressAuthority:
    """Authorize one exact generic-progress audience boundary."""

    _issuer: object
    correlation_id: PatchPublicCorrelationId

    def __post_init__(self) -> None:
        """Require one opaque audience-scoped public correlation."""
        if type(self.correlation_id) is not PatchPublicCorrelationId:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __repr__(self) -> str:
        """Render a non-sensitive marker for trusted diagnostics."""
        return "_GenericToolProgressAuthority(<opaque>)"

    def __copy__(self) -> NoReturn:
        """Reject copying an exact audience authority witness."""
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject copying an exact audience authority witness."""
        del memo
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __reduce__(self) -> NoReturn:
        """Reject serializing an exact audience authority witness."""
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific authority serialization."""
        del protocol
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


@final
@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _GenericToolProgressBoundary:
    """Produce verified immutable generic progress only from one outbox."""

    _issuer: object
    _authority: _GenericToolProgressAuthority
    _public_correlation_id: PatchPublicCorrelationId
    _request_id: PatchRequestId
    _correlation_id: PatchObserverCorrelationId
    _signing_key: Ed25519PrivateKey

    def __post_init__(self) -> None:
        """Require one exact request, observer, and signing boundary."""
        if (
            type(self._request_id) is not PatchRequestId
            or type(self._correlation_id) is not PatchObserverCorrelationId
            or type(self._public_correlation_id)
            is not PatchPublicCorrelationId
            or not isinstance(self._signing_key, Ed25519PrivateKey)
            or self._authority._issuer is not self._issuer
            or self._authority.correlation_id
            is not self._public_correlation_id
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def authority(self) -> _GenericToolProgressAuthority:
        """Return this exact boundary's generic-progress authority."""
        return self._authority

    def project(
        self,
        authority: _GenericToolProgressAuthority,
        record: DurableOutboxRecord,
    ) -> GenericToolProgressDelivery:
        """Return one host-verified immutable generic progress delivery."""
        if type(record) is not DurableOutboxRecord:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if (
            type(authority) is not _GenericToolProgressAuthority
            or authority is not self._authority
            or authority._issuer is not self._issuer
            or authority.correlation_id is not self._public_correlation_id
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
        if (
            record.request_id != self._request_id
            or record.correlation_id != self._correlation_id
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
        body = _generic_progress_body(
            self._public_correlation_id,
            record,
        )
        return _verified_generic_progress_delivery(
            _sign_generic_progress_delivery(body, self._signing_key),
            body,
            self._signing_key.public_key(),
        )

    def matches(
        self,
        access: DurableRequestAccess,
        correlation_id: PatchObserverCorrelationId,
    ) -> bool:
        """Return whether exact outbox bindings match this boundary."""
        return (
            type(access) is DurableRequestAccess
            and type(correlation_id) is PatchObserverCorrelationId
            and access.request_id == self._request_id
            and correlation_id == self._correlation_id
        )

    def __copy__(self) -> NoReturn:
        """Reject copying a trusted generic-progress boundary."""
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __deepcopy__(self, memo: dict[int, object]) -> NoReturn:
        """Reject copying a trusted generic-progress boundary."""
        del memo
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __reduce__(self) -> NoReturn:
        """Reject serializing a trusted generic-progress boundary."""
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject protocol-specific boundary serialization."""
        del protocol
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


def _create_generic_tool_progress_boundary(
    access: DurableRequestAccess,
    correlation_id: PatchObserverCorrelationId,
) -> _GenericToolProgressBoundary:
    """Create one trusted request- and observer-bound progress boundary."""
    if (
        type(access) is not DurableRequestAccess
        or type(correlation_id) is not PatchObserverCorrelationId
    ):
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    issuer = object()
    public_correlation_id = PatchPublicCorrelationId.new()
    return _GenericToolProgressBoundary(
        issuer,
        _GenericToolProgressAuthority(issuer, public_correlation_id),
        public_correlation_id,
        access.request_id,
        correlation_id,
        Ed25519PrivateKey.generate(),
    )


def _generic_progress_phase(
    lifecycle: LifecyclePhase,
) -> GenericToolProgressPhase:
    """Map an outbox lifecycle record to its sole coarse public phase."""
    if lifecycle is LifecyclePhase.SETTLEMENT_PENDING:
        return GenericToolProgressPhase.SETTLEMENT_PENDING
    if lifecycle is LifecyclePhase.REQUEST_COMPLETED:
        return GenericToolProgressPhase.SETTLED
    raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)


def _generic_progress_body(
    correlation_id: PatchPublicCorrelationId,
    record: DurableOutboxRecord,
) -> dict[str, str | int]:
    """Return the sole coarse immutable payload permitted to generic tools."""
    return {
        "schema_version": _GENERIC_PROGRESS_SCHEMA_VERSION,
        "audience": _GENERIC_PROGRESS_AUDIENCE,
        "correlation_id": correlation_id.value,
        "delivery_id": project_observer_id(
            record.event_id.value, "delivery_id"
        ),
        "phase": _generic_progress_phase(record.lifecycle).value,
        "count": record.sequence.value,
    }


def _canonical_generic_progress_bytes(value: object) -> bytes:
    """Return canonical bytes for one signed generic-progress envelope."""
    return dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sign_generic_progress_delivery(
    body: dict[str, str | int],
    signing_key: Ed25519PrivateKey,
) -> bytes:
    """Sign one private-host generic-progress body before delivery."""
    signature = signing_key.sign(_canonical_generic_progress_bytes(body)).hex()
    return _canonical_generic_progress_bytes({**body, "signature": signature})


def _verified_generic_progress_delivery(
    value: bytes,
    expected_body: dict[str, str | int],
    public_key: Ed25519PublicKey,
) -> GenericToolProgressDelivery:
    """Verify host-produced bytes without accepting lower-consumer input."""
    if (
        type(value) is not bytes
        or len(value) > _MAX_GENERIC_PROGRESS_DELIVERY_BYTES
        or not isinstance(public_key, Ed25519PublicKey)
    ):
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    try:
        parsed = loads(value.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("delivery is not an object")
        signature = parsed.pop("signature")
        if (
            type(signature) is not str
            or len(signature) != _ED25519_SIGNATURE_BYTES * 2
            or bytes.fromhex(signature).hex() != signature
            or parsed != expected_body
            or _canonical_generic_progress_bytes(
                {
                    **parsed,
                    "signature": signature,
                }
            )
            != value
        ):
            raise ValueError("delivery is not canonical")
        public_key.verify(
            bytes.fromhex(signature),
            _canonical_generic_progress_bytes(parsed),
        )
    except (
        InvalidSignature,
        KeyError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
    ) as error:
        raise DurableStoreError(
            DurableStoreErrorCode.LIFECYCLE_CONFLICT
        ) from error
    return GenericToolProgressDelivery(value)


def _outbox_record_digest(record: DurableOutboxRecord) -> bytes:
    """Return a stable receipt digest for one complete canonical record."""
    return sha256(
        "\0".join(
            (
                record.event_id.value,
                record.request_id.value,
                str(record.sequence.value),
                record.lifecycle.value,
                record.correlation_id.value,
            )
        ).encode("utf-8")
    ).digest()


@dataclass(frozen=True, slots=True)
class _QueuedGenericProgress:
    """Bind one already-verified immutable delivery to its receipt identity."""

    event_id: str
    receipt_digest: bytes
    delivery: GenericToolProgressDelivery


class EventManagerDurableOutboxProjection:
    """Read one trusted durable outbox and fan out verified progress.

    The durable store remains the replay authority. This request- and
    observer-bound bridge signs and verifies immutable generic bytes before
    scheduling a bounded isolated progress channel. Optional EventManager
    events are explicitly non-authoritative; canonical listeners are never
    registered with EventManager. Listener outcome never acknowledges a
    durable record or affects mutation truth.
    """

    def __init__(
        self,
        store: _DurableOutboxReader,
        access: DurableRequestAccess,
        correlation_id: PatchObserverCorrelationId,
        *,
        event_manager: EventManager | None = None,
    ) -> None:
        """Bind one exact trusted store and observer progress projection."""
        if (
            not callable(getattr(store, "outbox", None))
            or type(access) is not DurableRequestAccess
            or type(correlation_id) is not PatchObserverCorrelationId
            or (
                event_manager is not None
                and type(event_manager) is not EventManager
            )
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        self._event_manager = event_manager
        self._store = store
        self._access = access
        self._correlation_id = correlation_id
        self._project_lock = Lock()
        self._project_owner: Task[object] | None = None
        self._after: SequenceNumber | None = None
        self._next_sequence = 0
        self._pending_seen = False
        self._terminal_seen = False
        self._receipts: dict[str, bytes] = {}
        self._queued_ids: set[str] = set()
        self._delivered_ids: set[str] = set()
        self._progress_boundary = _create_generic_tool_progress_boundary(
            access, correlation_id
        )
        self._progress_authority = self._progress_boundary.authority()
        self._queue: Queue[_QueuedGenericProgress] = Queue(maxsize=2)
        self._listeners: set[Callable[[bytes], Awaitable[None]]] = set()
        self._listener_tasks: set[Task[None]] = set()
        self._worker: Task[None] | None = None
        self._close_task: Task[None] | None = None
        self._closed = False
        self._lock = Lock()

    def add_progress_listener(
        self,
        listener: Callable[[bytes], Awaitable[None]],
    ) -> None:
        """Register one isolated canonical-progress observer."""
        if not callable(listener) or self._closed:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        self._listeners.add(listener)

    async def project(
        self,
        after: SequenceNumber,
        limit: int,
    ) -> "DurableOutboxProjectionReceipt":
        """Read, validate, and queue only this store's canonical records."""
        if (
            type(after) is not SequenceNumber
            or type(limit) is not int
            or not 1 <= limit <= 1024
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        caller = current_task()
        if caller is not None and caller is self._project_owner:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        async with self._project_lock:
            self._project_owner = caller
            try:
                return await self._project_page(after, limit)
            finally:
                self._project_owner = None

    async def _project_page(
        self,
        after: SequenceNumber,
        limit: int,
    ) -> "DurableOutboxProjectionReceipt":
        """Accept one complete page at the current trusted cursor."""
        async with self._lock:
            if self._closed or (
                self._after is not None and after != self._after
            ):
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            if self._after is None:
                self._after = after
                self._next_sequence = after.value + 1
            else:
                assert self._after is not None
                after = self._after
        records = await self._store.outbox(self._access, after, limit)
        _validate_canonical_outbox_records(
            self._access, self._correlation_id, after, records
        )
        if len(records) > limit:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        accepted_after = after
        for record in records:
            await self._deliver_record(record)
            accepted_after = record.sequence
        async with self._lock:
            if self._closed:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            self._after = accepted_after
        return DurableOutboxProjectionReceipt(
            records, len(records), False, False
        )

    async def _deliver_record(self, record: DurableOutboxRecord) -> None:
        """Queue one trusted store-read immutable delivery."""
        if type(record) is not DurableOutboxRecord:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        event_id = record.event_id.value
        receipt_digest = _outbox_record_digest(record)
        async with self._lock:
            if self._closed:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            existing = self._receipts.get(event_id)
            if existing is not None:
                if existing != receipt_digest:
                    raise DurableStoreError(
                        DurableStoreErrorCode.LIFECYCLE_CONFLICT
                    )
                if (
                    event_id in self._queued_ids
                    or event_id in self._delivered_ids
                ):
                    return
                self._enqueue_retry(record, receipt_digest)
                return
            self._validate_next_record(record)
            self._enqueue_retry(record, receipt_digest)
            self._receipts[event_id] = receipt_digest
            self._next_sequence += 1
            self._pending_seen = self._pending_seen or (
                record.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
            )
            self._terminal_seen = (
                record.lifecycle is LifecyclePhase.REQUEST_COMPLETED
            )

    async def aclose(self) -> None:
        """Cancel bounded fan-out without touching durable state."""
        caller = current_task()
        async with self._lock:
            caller_is_owned = (
                caller is self._worker or caller in self._listener_tasks
            )
            close_task = self._close_task
            if close_task is None:
                self._closed = True
                worker = self._worker
                self._worker = None
                self._queued_ids.clear()
                while True:
                    try:
                        self._queue.get_nowait()
                        self._queue.task_done()
                    except QueueEmpty:
                        break
                close_task = create_task(
                    self._finish_close(
                        worker,
                        caller if caller_is_owned else None,
                    )
                )
                self._close_task = close_task
            else:
                assert close_task is not None
        if caller_is_owned:
            return
        await shield(close_task)

    async def _finish_close(
        self,
        worker: Task[None] | None,
        protected_task: Task[None] | None,
    ) -> None:
        """Cancel every fan-out task except an active reentrant caller."""
        if (
            worker is not None
            and not worker.done()
            and worker is not protected_task
        ):
            worker.cancel()
            await gather(worker, return_exceptions=True)
        listener_tasks = tuple(
            task
            for task in self._listener_tasks
            if not task.done() and task is not protected_task
        )
        for task in listener_tasks:
            task.cancel()
        if listener_tasks:
            await gather(*listener_tasks, return_exceptions=True)

    def _validate_next_record(self, record: DurableOutboxRecord) -> None:
        """Require the exact contiguous request and observer record next."""
        if (
            record.request_id != self._access.request_id
            or record.correlation_id != self._correlation_id
        ):
            raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)
        if (
            record.sequence.value != self._next_sequence
            or self._terminal_seen
            or record.lifecycle is LifecyclePhase.SETTLEMENT_PENDING
            and self._pending_seen
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        _generic_progress_phase(record.lifecycle)

    def _enqueue_retry(
        self,
        record: DurableOutboxRecord,
        receipt_digest: bytes,
    ) -> None:
        """Queue a retry while retaining its canonical receipt."""
        event_id = record.event_id.value
        try:
            delivery = self._progress_boundary.project(
                self._progress_authority,
                record,
            )
            self._queue.put_nowait(
                _QueuedGenericProgress(event_id, receipt_digest, delivery)
            )
        except QueueFull as error:
            raise DurableStoreError(
                DurableStoreErrorCode.LIFECYCLE_CONFLICT
            ) from error
        self._queued_ids.add(event_id)
        if self._worker is None or self._worker.done():
            self._worker = create_task(self._drain())

    async def _drain(self) -> None:
        """Schedule canonical listeners outside the durable caller."""
        while True:
            queued = await self._queue.get()
            try:
                self._notify_canonical_listeners(queued.delivery)
                self._emit_non_authoritative_event()
                async with self._lock:
                    self._queued_ids.discard(queued.event_id)
                    self._delivered_ids.add(queued.event_id)
            finally:
                self._queue.task_done()
            async with self._lock:
                if self._queue.empty():
                    self._worker = None
                    return

    def _notify_canonical_listeners(
        self,
        delivery: GenericToolProgressDelivery,
    ) -> None:
        """Isolate each verified canonical listener from durable fan-out."""
        for listener in tuple(self._listeners):
            task = create_task(
                self._call_canonical_listener(listener, delivery)
            )
            self._listener_tasks.add(task)
            task.add_done_callback(self._listener_tasks.discard)

    async def _call_canonical_listener(
        self,
        listener: Callable[[bytes], Awaitable[None]],
        delivery: GenericToolProgressDelivery,
    ) -> None:
        """Run one canonical listener without allowing it to affect truth."""
        try:
            await listener(delivery)
        except CancelledError:
            if self._current_task_is_cancelling():
                raise
        except BaseException:
            return

    def _emit_non_authoritative_event(self) -> None:
        """Publish an explicitly non-authoritative EventManager marker."""
        if self._event_manager is None:
            return
        task = create_task(
            self._event_manager.trigger(
                Event(
                    type=EventType.TOOL_PROGRESS,
                    payload={"audience": "non_authoritative_progress"},
                )
            )
        )
        self._listener_tasks.add(task)
        task.add_done_callback(self._listener_tasks.discard)

    async def _release_failed_delivery(
        self,
        queued: _QueuedGenericProgress,
    ) -> None:
        """Allow only an exact later retry after a fan-out failure."""
        async with self._lock:
            if self._receipts.get(queued.event_id) == queued.receipt_digest:
                self._queued_ids.discard(queued.event_id)

    @staticmethod
    def _current_task_is_cancelling() -> bool:
        """Return whether the bridge worker is being closed by its owner."""
        task = current_task()
        return bool(task and task.cancelling())


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
        correlation_id: PatchObserverCorrelationId,
        after: SequenceNumber,
        limit: int,
    ) -> DurableOutboxProjectionReceipt:
        """Deliver events in order and retain all failures for later retry."""
        if (
            type(access) is not DurableRequestAccess
            or type(correlation_id) is not PatchObserverCorrelationId
            or type(after) is not SequenceNumber
            or type(limit) is not int
            or not 1 <= limit <= 1024
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        records = await self._store.outbox(access, after, limit)
        _validate_canonical_outbox_records(
            access,
            correlation_id,
            after,
            records,
        )
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


def _validate_canonical_outbox_records(
    access: DurableRequestAccess,
    correlation_id: PatchObserverCorrelationId,
    after: SequenceNumber,
    records: tuple[DurableOutboxRecord, ...],
) -> None:
    """Reject malformed, reordered, or non-canonical durable event replay."""
    if type(records) is not tuple:
        raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
    expected_sequence = after.value + 1
    event_ids: set[str] = set()
    pending_seen = False
    terminal_seen = False
    for record in records:
        if (
            type(record) is not DurableOutboxRecord
            or record.request_id != access.request_id
            or record.correlation_id != correlation_id
            or record.sequence.value != expected_sequence
            or record.event_id.value in event_ids
        ):
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        if record.lifecycle is LifecyclePhase.SETTLEMENT_PENDING:
            if pending_seen or terminal_seen:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            pending_seen = True
        elif record.lifecycle is LifecyclePhase.REQUEST_COMPLETED:
            if terminal_seen:
                raise DurableStoreError(
                    DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
            terminal_seen = True
        else:
            raise DurableStoreError(DurableStoreErrorCode.LIFECYCLE_CONFLICT)
        expected_sequence += 1
        event_ids.add(record.event_id.value)
