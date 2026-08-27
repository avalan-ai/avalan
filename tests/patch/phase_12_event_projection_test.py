"""Exercise trusted store-backed Phase 12 generic-progress fan-out."""

from asyncio import CancelledError, create_task, gather, run, sleep
from asyncio import Event as Signal
from collections.abc import Callable
from copy import copy, deepcopy
from dataclasses import replace
from json import loads

import pytest
from phase_8_store_test import _correlation, _identity

from avalan.event import Event as AvalanEvent
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.patch import durable_outbox as outbox
from avalan.patch.domain import (
    LifecyclePhase,
    PatchEventId,
    PatchObserverCorrelationId,
    PatchPublicCorrelationId,
    PatchRequestId,
    SequenceNumber,
)
from avalan.patch.durable_outbox import EventManagerDurableOutboxProjection
from avalan.patch.durable_store import (
    DurableOutboxRecord,
    DurableRequestAccess,
    DurableStoreError,
    DurableStoreErrorCode,
)
from avalan.patch.projection_codec import GenericToolProgressDelivery


class _Store:
    """Expose a construction-time trusted outbox read seam for tests."""

    def __init__(self, records: tuple[DurableOutboxRecord, ...]) -> None:
        """Bind the exact records returned by this store seam."""
        self.records = records
        self.calls: list[tuple[SequenceNumber, int]] = []

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Return the bound record vector after consuming trusted access."""
        del access
        self.calls.append((after, limit))
        return self.records


class _PagedStore(_Store):
    """Return only the canonical page after the supplied trusted cursor."""

    async def outbox(
        self,
        access: DurableRequestAccess,
        after: SequenceNumber,
        limit: int,
    ) -> tuple[DurableOutboxRecord, ...]:
        """Return one bounded page while retaining the observed cursor."""
        del access
        self.calls.append((after, limit))
        return tuple(
            record
            for record in self.records
            if record.sequence.value > after.value
        )[:limit]


class _PublicCorrelationSubstitute(PatchPublicCorrelationId):
    """Model a same-interface value that lacks exact audience authority."""


class _RequestAccessSubstitute(DurableRequestAccess):
    """Model a same-interface access value that lacks exact authority."""


class _OutboxRecordSubstitute(DurableOutboxRecord):
    """Model a same-interface record that lacks exact store authority."""


def _access() -> DurableRequestAccess:
    """Return the exact request identity used by all test records."""
    return DurableRequestAccess(
        PatchRequestId("request_" + "a" * 16), _identity("a")
    )


def _record(
    token: str,
    sequence: int,
    lifecycle: LifecyclePhase,
    correlation: PatchObserverCorrelationId,
) -> DurableOutboxRecord:
    """Return one canonical content-free lifecycle record."""
    return DurableOutboxRecord(
        PatchEventId("event_" + token * 16),
        _access().request_id,
        SequenceNumber(sequence),
        lifecycle,
        correlation,
    )


async def _eventually(predicate: Callable[[], bool]) -> None:
    """Yield until a bounded background projection expectation is met."""
    for _ in range(32):
        if predicate():
            return
        await sleep(0)
    raise AssertionError("progress fan-out did not complete")


def _assert_lifecycle_conflict(operation: Callable[[], object]) -> None:
    """Require one defensive operation to reject its invalid input."""
    with pytest.raises(DurableStoreError) as rejected:
        operation()
    assert rejected.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT


def test_store_read_progress_rejects_eventmanager_injection() -> None:
    """Deliver only store-read signed bytes to projection-owned listeners."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        terminal = _record(
            "b", 2, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        store = _Store((pending, terminal))
        manager = EventManager()
        canonical: list[bytes] = []
        ordinary: list[AvalanEvent] = []
        manager.add_listener(ordinary.append, (EventType.TOOL_PROGRESS,))
        projection = EventManagerDurableOutboxProjection(
            store, access, correlation, event_manager=manager
        )

        async def observe(delivery: bytes) -> None:
            """Record one projection-issued canonical delivery."""
            canonical.append(delivery)

        projection.add_progress_listener(observe)
        with pytest.raises(AttributeError):
            await getattr(projection, "deliver")(pending)
        await manager.trigger(
            AvalanEvent(type=EventType.TOOL_PROGRESS, payload=b"forged")
        )
        assert not canonical
        receipt = await projection.project(SequenceNumber(0), 10)
        await _eventually(lambda: len(canonical) == 2)
        assert receipt.records == (pending, terminal)
        assert store.calls == [(SequenceNumber(0), 10)]
        assert [loads(item)["count"] for item in canonical] == [1, 2]
        assert all(
            loads(item)["audience"] == "generic_tool_progress"
            for item in canonical
        )
        await _eventually(lambda: len(ordinary) == 3)
        assert ordinary[0].payload == b"forged"
        assert all(
            event.payload == {"audience": "non_authoritative_progress"}
            for event in ordinary[1:]
        )
        await projection.aclose()
        await manager.aclose()

    run(scenario())


def test_store_backed_progress_advances_trusted_cursor_across_pages() -> None:
    """Advance pending to terminal without accepting caller cursor forgery."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        terminal = _record(
            "b", 2, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        store = _PagedStore((pending, terminal))
        projection = EventManagerDurableOutboxProjection(
            store, access, correlation
        )
        delivered: list[bytes] = []

        async def observe(delivery: bytes) -> None:
            """Record one exact canonical delivery."""
            delivered.append(delivery)

        projection.add_progress_listener(observe)
        initial = SequenceNumber(0)
        first = await projection.project(initial, 1)
        caller_next = SequenceNumber(1)
        second = await projection.project(caller_next, 1)
        empty = await projection.project(SequenceNumber(2), 1)
        await _eventually(lambda: len(delivered) == 2)
        assert first.records == (pending,)
        assert second.records == (terminal,)
        assert empty.records == ()
        assert store.calls[0] == (initial, 1)
        assert store.calls[1][0] is pending.sequence
        assert store.calls[1][0] is not caller_next
        assert store.calls[2][0] is terminal.sequence
        assert [loads(item)["phase"] for item in delivered] == [
            "settlement_pending",
            "settled",
        ]
        with pytest.raises(DurableStoreError) as raised:
            await projection.project(SequenceNumber(1), 10)
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        with pytest.raises(DurableStoreError) as forged:
            await projection.project(SequenceNumber(3), 10)
        assert forged.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        await projection.aclose()

        for records in (
            (_record("c", 2, LifecyclePhase.REQUEST_COMPLETED, correlation),),
            (
                pending,
                replace(terminal, event_id=pending.event_id),
            ),
            (
                pending,
                _record(
                    "d", 2, LifecyclePhase.SETTLEMENT_PENDING, correlation
                ),
            ),
        ):
            invalid = EventManagerDurableOutboxProjection(
                _Store(records), access, correlation
            )
            with pytest.raises(DurableStoreError):
                await invalid.project(SequenceNumber(0), 10)
            assert not invalid._receipts
            assert invalid._after == SequenceNumber(0)
            await invalid.aclose()

    run(scenario())


def test_event_projection_empty_error_and_cross_page_state_are_closed() -> (
    None
):
    """Advance only a complete accepted page and retain empty poll cursors."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        terminal = _record(
            "b", 2, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        store = _PagedStore(())
        projection = EventManagerDurableOutboxProjection(
            store, access, correlation
        )
        initial = SequenceNumber(0)
        assert (await projection.project(initial, 1)).records == ()
        store.records = (pending, terminal)
        assert (await projection.project(SequenceNumber(0), 1)).records == (
            pending,
        )
        assert store.calls[1][0] is initial
        await projection.aclose()

        class FailingStore(_PagedStore):
            """Fail one fetch before returning its unchanged first page."""

            def __init__(self) -> None:
                """Initialize one deterministic fetch failure."""
                super().__init__((pending,))
                self.failed = False

            async def outbox(
                self,
                access: DurableRequestAccess,
                after: SequenceNumber,
                limit: int,
            ) -> tuple[DurableOutboxRecord, ...]:
                """Raise once without allowing cursor advancement."""
                if not self.failed:
                    self.failed = True
                    raise DurableStoreError(
                        DurableStoreErrorCode.LIFECYCLE_CONFLICT
                    )
                return await super().outbox(access, after, limit)

        failing_store = FailingStore()
        retrying = EventManagerDurableOutboxProjection(
            failing_store, access, correlation
        )
        with pytest.raises(DurableStoreError):
            await retrying.project(SequenceNumber(0), 1)
        assert (await retrying.project(SequenceNumber(0), 1)).records == (
            pending,
        )
        await retrying.aclose()

        duplicate_pending = EventManagerDurableOutboxProjection(
            _PagedStore(
                (
                    pending,
                    _record(
                        "c",
                        2,
                        LifecyclePhase.SETTLEMENT_PENDING,
                        correlation,
                    ),
                )
            ),
            access,
            correlation,
        )
        await duplicate_pending.project(SequenceNumber(0), 1)
        with pytest.raises(DurableStoreError) as duplicate:
            await duplicate_pending.project(SequenceNumber(1), 1)
        assert duplicate.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        assert duplicate_pending._after == SequenceNumber(1)
        await duplicate_pending.aclose()

        oversized = EventManagerDurableOutboxProjection(
            _Store((pending, terminal)), access, correlation
        )
        with pytest.raises(DurableStoreError) as over_limit:
            await oversized.project(SequenceNumber(0), 1)
        assert (
            over_limit.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        assert not oversized._receipts
        await oversized.aclose()

        class ClosingStore:
            """Close the projection during one empty trusted fetch."""

            def __init__(self) -> None:
                """Initialize without a bridge until construction completes."""
                self.projection: EventManagerDurableOutboxProjection | None = (
                    None
                )

            async def outbox(
                self,
                access: DurableRequestAccess,
                after: SequenceNumber,
                limit: int,
            ) -> tuple[DurableOutboxRecord, ...]:
                """Close before the empty page can be accepted."""
                del access, after, limit
                assert self.projection is not None
                await self.projection.aclose()
                return ()

        closing_store = ClosingStore()
        closing = EventManagerDurableOutboxProjection(
            closing_store, access, correlation
        )
        closing_store.projection = closing
        with pytest.raises(DurableStoreError) as closed_page:
            await closing.project(SequenceNumber(0), 1)
        assert (
            closed_page.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )

    run(scenario())


def test_event_projection_serializes_concurrent_and_reentrant_pages() -> None:
    """Keep one page owner and reject stale or same-task reentrant reads."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        terminal = _record(
            "b", 2, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        started = Signal()
        release = Signal()

        class BlockingStore(_PagedStore):
            """Pause the first trusted page fetch for a concurrency race."""

            async def outbox(
                self,
                access: DurableRequestAccess,
                after: SequenceNumber,
                limit: int,
            ) -> tuple[DurableOutboxRecord, ...]:
                """Return pages after exposing the first active owner."""
                if not self.calls:
                    started.set()
                    await release.wait()
                return await super().outbox(access, after, limit)

        store = BlockingStore((pending, terminal))
        projection = EventManagerDurableOutboxProjection(
            store, access, correlation
        )
        first = create_task(projection.project(SequenceNumber(0), 1))
        await started.wait()
        stale = create_task(projection.project(SequenceNumber(0), 1))
        await sleep(0)
        assert not stale.done()
        release.set()
        assert (await first).records == (pending,)
        with pytest.raises(DurableStoreError) as rejected:
            await stale
        assert rejected.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        assert len(store.calls) == 1
        assert (await projection.project(SequenceNumber(1), 1)).records == (
            terminal,
        )
        await projection.aclose()

        class ReentrantStore:
            """Attempt one same-task recursive project call from a fetch."""

            def __init__(self) -> None:
                """Initialize without a bridge until construction completes."""
                self.projection: EventManagerDurableOutboxProjection | None = (
                    None
                )

            async def outbox(
                self,
                access: DurableRequestAccess,
                after: SequenceNumber,
                limit: int,
            ) -> tuple[DurableOutboxRecord, ...]:
                """Require recursive page ownership to fail without waiting."""
                del access
                assert self.projection is not None
                with pytest.raises(DurableStoreError) as reentrant:
                    await self.projection.project(after, limit)
                assert (
                    reentrant.value.code
                    is DurableStoreErrorCode.LIFECYCLE_CONFLICT
                )
                return ()

        reentrant_store = ReentrantStore()
        reentrant_projection = EventManagerDurableOutboxProjection(
            reentrant_store, access, correlation
        )
        reentrant_store.projection = reentrant_projection
        assert (
            await reentrant_projection.project(SequenceNumber(0), 1)
        ).records == ()
        await reentrant_projection.aclose()

    run(scenario())


def test_private_signer_cannot_enter_canonical_channel() -> None:
    """Keep fabricated signed bytes and slow listeners non-authoritative."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        store = _Store((pending,))
        started = Signal()
        release = Signal()
        observed: list[bytes] = []
        projection = EventManagerDurableOutboxProjection(
            store, access, correlation
        )

        async def slow_listener(delivery: bytes) -> None:
            """Delay one isolated verified listener without holding project."""
            observed.append(delivery)
            started.set()
            await release.wait()

        projection.add_progress_listener(slow_listener)
        forged_boundary = getattr(
            outbox, "_create_generic_tool_progress_boundary"
        )(access, correlation)
        forged = forged_boundary.project(forged_boundary.authority(), pending)
        assert type(forged) is bytes
        assert not hasattr(projection, "progress_boundary")
        await projection.project(SequenceNumber(0), 1)
        await started.wait()
        assert len(observed) == 1
        release.set()
        await projection.aclose()

    run(scenario())


def test_private_progress_authority_rejects_substitution_and_copying() -> None:
    """Keep generic-progress authority exact, opaque, and non-transferable."""
    access = _access()
    correlation = _correlation("a")
    pending = _record("a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation)
    boundary = outbox._create_generic_tool_progress_boundary(
        access, correlation
    )
    authority = boundary.authority()

    with pytest.raises(DurableStoreError) as invalid_authority:
        outbox._GenericToolProgressAuthority(
            object(),
            _PublicCorrelationSubstitute("public_" + "a" * 16),
        )
    assert (
        invalid_authority.value.code
        is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    )
    assert repr(authority) == "_GenericToolProgressAuthority(<opaque>)"
    _assert_lifecycle_conflict(lambda: copy(authority))
    _assert_lifecycle_conflict(lambda: deepcopy(authority))
    _assert_lifecycle_conflict(authority.__reduce__)
    _assert_lifecycle_conflict(lambda: authority.__reduce_ex__(5))

    with pytest.raises(DurableStoreError) as invalid_boundary:
        outbox._GenericToolProgressBoundary(
            object(),
            authority,
            authority.correlation_id,
            access.request_id,
            correlation,
            boundary._signing_key,
        )
    assert (
        invalid_boundary.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    )
    assert boundary.matches(access, correlation)
    assert not boundary.matches(access, _correlation("b"))

    substitute_authority = outbox._GenericToolProgressAuthority(
        object(), authority.correlation_id
    )
    with pytest.raises(DurableStoreError) as wrong_authority:
        boundary.project(substitute_authority, pending)
    assert wrong_authority.value.code is DurableStoreErrorCode.ACCESS_DENIED
    substituted_record = _OutboxRecordSubstitute(
        pending.event_id,
        pending.request_id,
        pending.sequence,
        pending.lifecycle,
        pending.correlation_id,
    )
    with pytest.raises(DurableStoreError) as substituted_record_error:
        boundary.project(authority, substituted_record)
    assert (
        substituted_record_error.value.code
        is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    )
    with pytest.raises(DurableStoreError) as wrong_record:
        boundary.project(
            authority,
            replace(pending, request_id=PatchRequestId("request_" + "b" * 16)),
        )
    assert wrong_record.value.code is DurableStoreErrorCode.ACCESS_DENIED
    _assert_lifecycle_conflict(lambda: copy(boundary))
    _assert_lifecycle_conflict(lambda: deepcopy(boundary))
    _assert_lifecycle_conflict(boundary.__reduce__)
    _assert_lifecycle_conflict(lambda: boundary.__reduce_ex__(5))


def test_private_progress_signer_rejects_malformed_delivery_inputs() -> None:
    """Reject noncanonical, substituted, and unsupported generic bytes."""
    access = _access()
    correlation = _correlation("a")
    pending = _record("a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation)
    boundary = outbox._create_generic_tool_progress_boundary(
        access, correlation
    )
    body = outbox._generic_progress_body(
        boundary.authority().correlation_id, pending
    )
    signed = outbox._sign_generic_progress_delivery(
        body, boundary._signing_key
    )
    assert (
        outbox._verified_generic_progress_delivery(
            signed, body, boundary._signing_key.public_key()
        )
        == signed
    )

    for delivery, expected in (
        (b"x" * 1_025, body),
        (b"[]", body),
        (signed, {}),
    ):
        with pytest.raises(DurableStoreError) as malformed:
            outbox._verified_generic_progress_delivery(
                delivery, expected, boundary._signing_key.public_key()
            )
        assert malformed.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as invalid_factory_access:
        outbox._create_generic_tool_progress_boundary(
            _RequestAccessSubstitute(access.request_id, access.identity),
            correlation,
        )
    assert (
        invalid_factory_access.value.code
        is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    )
    with pytest.raises(DurableStoreError) as invalid_phase:
        getattr(outbox, "_generic_progress_phase")(object())
    assert invalid_phase.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT


def test_event_projection_rejects_closed_and_conflicting_delivery_state() -> (
    None
):
    """Keep closed, altered, and unauthorized record delivery fail closed."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        projection = EventManagerDurableOutboxProjection(
            _Store((pending,)), access, correlation
        )
        substituted_record = _OutboxRecordSubstitute(
            pending.event_id,
            pending.request_id,
            pending.sequence,
            pending.lifecycle,
            pending.correlation_id,
        )
        with pytest.raises(DurableStoreError) as invalid_record:
            await projection._deliver_record(substituted_record)
        assert (
            invalid_record.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        with pytest.raises(DurableStoreError) as invalid_arguments:
            await projection.project(SequenceNumber(0), 0)
        assert (
            invalid_arguments.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        await projection.project(SequenceNumber(0), 1)
        with pytest.raises(DurableStoreError) as duplicate_conflict:
            await projection._deliver_record(
                replace(pending, lifecycle=LifecyclePhase.REQUEST_COMPLETED)
            )
        assert (
            duplicate_conflict.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        with pytest.raises(DurableStoreError) as wrong_access:
            projection._validate_next_record(
                replace(
                    pending,
                    request_id=PatchRequestId("request_" + "b" * 16),
                )
            )
        assert wrong_access.value.code is DurableStoreErrorCode.ACCESS_DENIED
        with pytest.raises(DurableStoreError) as incorrect_sequence:
            projection._validate_next_record(
                replace(pending, sequence=SequenceNumber(3))
            )
        assert (
            incorrect_sequence.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        await projection.aclose()
        with pytest.raises(DurableStoreError) as closed_listener:
            projection.add_progress_listener(lambda delivery: sleep(0))
        assert (
            closed_listener.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        with pytest.raises(DurableStoreError) as closed_delivery:
            await projection._deliver_record(pending)
        assert (
            closed_delivery.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        with pytest.raises(DurableStoreError) as closed_projection:
            await projection.project(SequenceNumber(0), 1)
        assert (
            closed_projection.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )

    run(scenario())


def test_event_projection_bounds_queue_and_cleans_cancelled_tasks() -> None:
    """Bound queued retry work and cancel worker/listener tasks on close."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        terminal = _record(
            "b", 2, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        saturated = EventManagerDurableOutboxProjection(
            _Store(()), access, correlation
        )
        pending_digest = outbox._outbox_record_digest(pending)
        terminal_digest = outbox._outbox_record_digest(terminal)
        saturated._receipts[pending.event_id.value] = pending_digest
        saturated._enqueue_retry(pending, pending_digest)
        saturated._enqueue_retry(terminal, terminal_digest)
        queued = outbox._QueuedGenericProgress(
            pending.event_id.value,
            pending_digest,
            saturated._progress_boundary.project(
                saturated._progress_authority, pending
            ),
        )
        await saturated._release_failed_delivery(queued)
        with pytest.raises(DurableStoreError) as queue_full:
            saturated._enqueue_retry(pending, pending_digest)
        assert (
            queue_full.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        worker = saturated._worker
        assert worker is not None
        await saturated.aclose()
        assert worker.cancelled()

        closing = EventManagerDurableOutboxProjection(
            _Store(()), access, correlation
        )
        listener_record = _record(
            "c", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        await closing.project(SequenceNumber(0), 1)
        await closing.project(SequenceNumber(0), 1)
        assert closing._next_sequence == 1
        assert not closing._terminal_seen
        started = Signal()

        async def blocking_listener(delivery: bytes) -> None:
            """Wait until close proves listener cancellation is isolated."""
            del delivery
            started.set()
            await Signal().wait()

        closing.add_progress_listener(blocking_listener)
        await closing._deliver_record(listener_record)
        await started.wait()
        await closing.aclose()
        assert not closing._listener_tasks

        retrying = EventManagerDurableOutboxProjection(
            _Store(()), access, correlation
        )
        await retrying.project(SequenceNumber(0), 1)
        await retrying._deliver_record(pending)
        first_queued = retrying._queue.get_nowait()
        retrying._queue.task_done()
        await retrying._release_failed_delivery(
            replace(first_queued, receipt_digest=b"unrelated")
        )
        await retrying._release_failed_delivery(first_queued)
        await retrying._deliver_record(pending)
        assert pending.event_id.value in retrying._queued_ids
        await retrying._deliver_record(pending)
        await retrying.aclose()

    run(scenario())


def test_event_projection_listener_cancellation_and_record_corruption() -> (
    None
):
    """Isolate listener errors and reject corrupted trusted-store records."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        projection = EventManagerDurableOutboxProjection(
            _Store(()), access, correlation
        )
        safe_delivery = GenericToolProgressDelivery(b"safe")

        async def cancelling_listener(delivery: bytes) -> None:
            """Remain pending until the owner cancels the listener task."""
            del delivery
            await Signal().wait()

        listener_task = create_task(
            projection._call_canonical_listener(
                cancelling_listener, safe_delivery
            )
        )
        await sleep(0)
        listener_task.cancel()
        with pytest.raises(CancelledError):
            await listener_task

        async def failing_listener(delivery: bytes) -> None:
            """Fail without influencing durable progress fan-out."""
            del delivery
            raise RuntimeError("listener failed")

        await projection._call_canonical_listener(
            failing_listener, safe_delivery
        )

        async def independently_cancelled_listener(delivery: bytes) -> None:
            """Raise cancellation without asking the bridge owner to stop."""
            del delivery
            raise CancelledError

        await projection._call_canonical_listener(
            independently_cancelled_listener, safe_delivery
        )
        corrupted = _record(
            "b", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        object.__setattr__(corrupted, "lifecycle", object())
        with pytest.raises(DurableStoreError) as malformed_records:
            outbox._validate_canonical_outbox_records(
                access, correlation, SequenceNumber(0), (corrupted,)
            )
        assert (
            malformed_records.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        with pytest.raises(DurableStoreError) as non_tuple_records:
            getattr(outbox, "_validate_canonical_outbox_records")(
                access, correlation, SequenceNumber(0), []
            )
        assert (
            non_tuple_records.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        first_terminal = _record(
            "c", 1, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        second_terminal = _record(
            "d", 2, LifecyclePhase.REQUEST_COMPLETED, correlation
        )
        with pytest.raises(DurableStoreError) as duplicate_terminal:
            outbox._validate_canonical_outbox_records(
                access,
                correlation,
                SequenceNumber(0),
                (first_terminal, second_terminal),
            )
        assert (
            duplicate_terminal.value.code
            is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        )
        await projection.aclose()

    run(scenario())


def test_event_projection_reentrant_listener_close_is_bounded() -> None:
    """Let a canonical listener close its own projection without self-await."""

    async def scenario() -> None:
        access = _access()
        correlation = _correlation("a")
        pending = _record(
            "a", 1, LifecyclePhase.SETTLEMENT_PENDING, correlation
        )
        projection = EventManagerDurableOutboxProjection(
            _Store((pending,)), access, correlation
        )
        listener_returned = Signal()

        async def reentrant_listener(delivery: bytes) -> None:
            """Close the bridge from its own isolated listener task."""
            del delivery
            await projection.aclose()
            listener_returned.set()

        projection.add_progress_listener(reentrant_listener)
        await projection.project(SequenceNumber(0), 1)
        await listener_returned.wait()
        first_close = create_task(projection.aclose())
        second_close = create_task(projection.aclose())
        await gather(first_close, second_close)
        await _eventually(
            lambda: (
                projection._close_task is not None
                and projection._close_task.done()
                and not projection._listener_tasks
            )
        )
        assert projection._closed
        assert projection._worker is None
        assert projection._queue.empty()
        assert not projection._queued_ids
        await projection.aclose()

    run(scenario())
