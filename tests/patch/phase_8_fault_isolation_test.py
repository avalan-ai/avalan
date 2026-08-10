"""Exercise Phase 8 encryption, delivery, and response-loss isolation."""

from asyncio import Event, create_task, run
from dataclasses import replace

import pytest
from phase_8_store_test import (
    _APPROVAL_AUTHORITY,
    _approval,
    _backend,
    _correlation,
    _digest,
    _identity,
    _owner,
    _plan,
    _result,
)

from avalan.event import Event as AvalanEvent
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.patch import durable_outbox as outbox
from avalan.patch.domain import (
    Audience,
    CommitStepState,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    MutationState,
    PatchEventId,
    PatchPendingOperationId,
    PatchRequestId,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    SequenceNumber,
)
from avalan.patch.durable_outbox import (
    DurableOutboxProjectionReceipt,
    DurableOutboxProjector,
    EventManagerDurableOutboxProjection,
)
from avalan.patch.durable_retention import (
    AesGcmDurableRetentionCipher,
    AesGcmDurableRetentionEnvelopeValidator,
    DurableRetentionBinding,
    DurableRetentionKey,
    InMemoryDurableRetentionKeyResolver,
    StaticDurableRetentionAuthorizer,
)
from avalan.patch.durable_store import (
    DurableCommitClaimState,
    DurableJournalCursor,
    DurableOutboxRecord,
    DurablePendingRequest,
    DurableRequestAccess,
    DurableRetentionAccess,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStoreError,
    DurableStoreErrorCode,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)


class _FailingRetentionResolver:
    """Model an unavailable active or historical retention key lookup."""

    async def active_key(self) -> DurableRetentionKey:
        """Fail closed before emitting any ciphertext."""
        raise RuntimeError("test key service unavailable")

    async def read_key(
        self, key_id: PatchRetentionKeyId
    ) -> DurableRetentionKey:
        """Fail closed without revealing whether the key exists."""
        del key_id
        raise RuntimeError("test key service unavailable")


class _BlockedRetentionValidator:
    """Pause validation so terminal settlement can race a retained insert."""

    def __init__(
        self, delegate: AesGcmDurableRetentionEnvelopeValidator
    ) -> None:
        """Bind one genuine envelope validator and deterministic barriers."""
        self._delegate = delegate
        self.entered = Event()
        self.release = Event()

    async def validate(
        self,
        request_id: PatchRequestId,
        record: DurableRetentionRecord,
    ) -> None:
        """Authenticate first, then block before the store's insert lock."""
        await self._delegate.validate(request_id, record)
        self.entered.set()
        await self.release.wait()


class _Delivery:
    """Record stable delivery identities and optionally fail once."""

    def __init__(self, fail_sequence: int | None = None) -> None:
        """Select the first delivery sequence that should fail."""
        self.fail_sequence = fail_sequence
        self.event_ids: list[str] = []

    async def deliver(self, record: DurableOutboxRecord) -> None:
        """Record a stable event identity or inject one transport failure."""
        if record.sequence.value == self.fail_sequence:
            raise RuntimeError("test delivery unavailable")
        self.event_ids.append(record.event_id.value)


class _Audit:
    """Record audit projection calls and optionally isolate failures."""

    def __init__(self, fail: bool = False) -> None:
        """Select whether audit projection should fail for every record."""
        self.fail = fail
        self.event_ids: list[str] = []

    async def project(self, record: DurableOutboxRecord) -> None:
        """Record an audit identity or inject a non-authoritative failure."""
        self.event_ids.append(record.event_id.value)
        if self.fail:
            raise RuntimeError("test audit unavailable")


def _retention_key(token: str) -> DurableRetentionKey:
    """Return one deterministic isolated AES-256-GCM test key."""
    return DurableRetentionKey(
        PatchRetentionKeyId("retention_" + token * 16),
        token.encode("ascii") * 32,
    )


def test_retention_encryption_key_loss_and_cleanup_are_non_authoritative() -> (
    None
):
    """Keep failed key and cleanup work separate from mutation truth."""

    async def scenario() -> None:
        identity = _identity("a")
        first = _retention_key("a")
        second = _retention_key("b")
        resolver = InMemoryDurableRetentionKeyResolver(
            second.key_id,
            {first.key_id: first, second.key_id: second},
        )
        retention_cipher = AesGcmDurableRetentionCipher(resolver)
        backend = InMemoryDurablePatchBackend(
            approval_verifier=_APPROVAL_AUTHORITY,
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.APPROVER,))
            ),
            retention_validator=AesGcmDurableRetentionEnvelopeValidator(
                retention_cipher
            ),
        )
        store = InMemoryDurablePatchStore(backend)
        reservation = await store.reserve(identity, _digest("a"))
        binding = DurableRetentionBinding(
            reservation.request_id,
            PatchRetentionRecordId("retained_" + "a" * 16),
            DurableRetentionKind.PRIVATE_STAGING,
        )
        first_cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(
                first.key_id, {first.key_id: first, second.key_id: second}
            )
        )
        old = await first_cipher.seal(b"private-old", binding)
        rotated = await AesGcmDurableRetentionCipher(resolver).seal(
            b"private-new",
            binding,
        )
        assert old.key_id == first.key_id
        assert rotated.key_id == second.key_id
        assert old.value != rotated.value
        assert (
            await AesGcmDurableRetentionCipher(resolver).open(old, binding)
            == b"private-old"
        )
        assert (
            await AesGcmDurableRetentionCipher(resolver).open(rotated, binding)
            == b"private-new"
        )
        with pytest.raises(DurableStoreError) as wrong_binding:
            await AesGcmDurableRetentionCipher(resolver).open(
                old,
                replace(
                    binding,
                    retention_id=PatchRetentionRecordId(
                        "retained_" + "b" * 16
                    ),
                ),
            )
        assert (
            wrong_binding.value.code is DurableStoreErrorCode.RETENTION_DENIED
        )
        with pytest.raises(DurableStoreError) as unavailable:
            await AesGcmDurableRetentionCipher(
                _FailingRetentionResolver()
            ).seal(b"private", binding)
        assert unavailable.value.code is DurableStoreErrorCode.RETENTION_DENIED

        record = DurableRetentionRecord(
            binding.retention_id,
            binding.kind,
            old.key_id,
            old.value,
            DurableRetentionPolicy(ExpiryTick(20), False),
        )
        await store.put_retention(reservation, record)
        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity),
        )
        before = await store.inspect(access.request)
        assert (
            await store.get_retention(
                access, record.retention_id, ExpiryTick(10)
            )
            == record
        )
        with pytest.raises(TypeError):
            getattr(DurableRetentionAccess, "__init__")(
                access, access.request, Audience.PUBLIC
            )
        assert await store.inspect(access.request) == before
        cleanup = await store.cleanup_retention(ExpiryTick(20))
        assert cleanup.records_deleted == 1
        assert (await store.inspect(access.request)).terminal is None

    run(scenario())


def test_retention_cannot_insert_after_terminal_cleanup_race() -> None:
    """Reject a late retained insert after terminal settlement wins."""

    async def scenario() -> None:
        identity = _identity("d")
        key = _retention_key("d")
        cipher = AesGcmDurableRetentionCipher(
            InMemoryDurableRetentionKeyResolver(key.key_id, {key.key_id: key})
        )
        validator = _BlockedRetentionValidator(
            AesGcmDurableRetentionEnvelopeValidator(cipher)
        )
        backend = InMemoryDurablePatchBackend(
            approval_verifier=_APPROVAL_AUTHORITY,
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.APPROVER,))
            ),
            retention_validator=validator,
        )
        store = InMemoryDurablePatchStore(backend)
        digest = _digest("d")
        reservation = await store.reserve(identity, digest)
        plan = _plan(digest, "d", step_count=1)
        await store.persist_plan(reservation, plan)
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(identity, digest, plan, "d"),
            _owner("d"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert claim.lease is not None
        retention_id = PatchRetentionRecordId("retained_" + "d" * 16)
        encrypted = await cipher.seal(
            b"terminal-race-ciphertext",
            DurableRetentionBinding(
                reservation.request_id,
                retention_id,
                DurableRetentionKind.SEALED_PLAN,
            ),
        )
        retained = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.SEALED_PLAN,
            encrypted.key_id,
            encrypted.value,
            DurableRetentionPolicy(ExpiryTick(100), True),
        )
        insertion = create_task(store.put_retention(reservation, retained))
        await validator.entered.wait()
        journal = await store.append_step(
            claim.lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await store.append_step(
            claim.lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        await store.settle(
            claim.lease,
            journal.cursor,
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            _correlation("d"),
            ExpiryTick(12),
        )
        validator.release.set()
        with pytest.raises(DurableStoreError) as denied:
            await insertion
        assert denied.value.code is DurableStoreErrorCode.RETENTION_DENIED
        with pytest.raises(DurableStoreError) as denied:
            await store.get_retention(
                DurableRetentionAccess(
                    DurableRequestAccess(reservation.request_id, identity)
                ),
                retained.retention_id,
                ExpiryTick(13),
            )
        assert denied.value.code is DurableStoreErrorCode.RETENTION_DENIED

    run(scenario())


def test_outbox_delivery_and_audit_failures_preserve_terminal_truth() -> None:
    """Keep failed observers retryable without rewriting a terminal record."""

    async def scenario() -> None:
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        identity = _identity("b")
        digest = _digest("b")
        reservation = await store.reserve(identity, digest)
        plan = _plan(digest, "b", step_count=1)
        await store.persist_plan(reservation, plan)
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(identity, digest, plan, "b"),
            _owner("b"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert claim.state is DurableCommitClaimState.OWNER
        assert claim.lease is not None
        journal = await store.append_step(
            claim.lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await store.append_step(
            claim.lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        pending = DurablePendingRequest(
            PatchPendingOperationId("pending_" + "b" * 16),
            _correlation("b"),
            DurationTicks(5),
        )
        await store.suspend(claim.lease, pending, ExpiryTick(12))
        terminal = await store.settle(
            claim.lease,
            journal.cursor,
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            pending.correlation_id,
            ExpiryTick(13),
        )
        access = DurableRequestAccess(reservation.request_id, identity)
        before = await store.inspect(access)
        failed_delivery = _Delivery(fail_sequence=1)
        failed = await DurableOutboxProjector(
            store, failed_delivery, _Audit()
        ).project(access, SequenceNumber(0), 10)
        assert failed.delivery_failed
        assert failed.delivered == 0
        assert await store.inspect(access) == before

        delivery = _Delivery()
        audit = _Audit(fail=True)
        receipt = await DurableOutboxProjector(store, delivery, audit).project(
            access, SequenceNumber(0), 10
        )
        assert not receipt.delivery_failed
        assert receipt.audit_failed
        assert receipt.delivered == 2
        assert len(set(delivery.event_ids)) == 2
        assert await store.inspect(access) == before
        replay = await DurableOutboxProjector(
            store, _Delivery(), _Audit()
        ).project(access, SequenceNumber(0), 10)
        assert replay.records[-1] == terminal.outbox
        assert await store.inspect(access) == before

    run(scenario())


def test_event_manager_projection_deduplicates_stable_outbox_identity() -> (
    None
):
    """Project duplicate durable delivery without creating terminal truth."""

    async def scenario() -> None:
        backend = _backend()
        store = InMemoryDurablePatchStore(backend)
        identity = _identity("e")
        digest = _digest("e")
        reservation = await store.reserve(identity, digest)
        plan = _plan(digest, "e", step_count=1)
        await store.persist_plan(reservation, plan)
        claim = await store.claim_commit(
            reservation,
            plan,
            _approval(identity, digest, plan, "e"),
            _owner("e"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert claim.lease is not None
        journal = await store.append_step(
            claim.lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await store.append_step(
            claim.lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        terminal = await store.settle(
            claim.lease,
            journal.cursor,
            _result(reservation.request_id, plan, MutationState.COMMITTED),
            _correlation("e"),
            ExpiryTick(12),
        )
        access = DurableRequestAccess(reservation.request_id, identity)
        before = await store.inspect(access)
        event_manager = EventManager()
        observed: list[AvalanEvent] = []
        event_manager.add_listener(
            observed.append,
            (EventType.TOOL_PROGRESS,),
        )
        projection = EventManagerDurableOutboxProjection(event_manager)
        try:
            await projection.deliver(terminal.outbox)
            await projection.deliver(terminal.outbox)
            history = event_manager.history
            assert len(history) == 1
            assert history[0].payload is None
            assert len(observed) == 1
            assert observed[0].payload == {
                "correlation_id": terminal.outbox.correlation_id.value,
                "event_id": terminal.outbox.event_id.value,
                "lifecycle": terminal.outbox.lifecycle.value,
                "request_id": reservation.request_id.value,
                "sequence": terminal.outbox.sequence.value,
            }
            assert await store.inspect(access) == before
            events = await store.outbox(access, SequenceNumber(0), 10)
            assert events == (terminal.outbox,)
        finally:
            await event_manager.aclose()

    run(scenario())


def test_durable_outbox_rejects_bad_records_and_preserves_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound event deduplication and preserve interruption from observers."""
    with pytest.raises(DurableStoreError) as raised:
        getattr(outbox, "EventManagerDurableOutboxProjection")(object())
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        DurableOutboxProjectionReceipt((), 1, False, False)
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        getattr(outbox, "DurableOutboxProjector")(
            object(), _Delivery(), _Audit()
        )
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

    async def scenario() -> None:
        request_id = PatchRequestId("request_" + "f" * 16)
        correlation = _correlation("f")
        first = DurableOutboxRecord(
            PatchEventId("event_" + "a" * 16),
            request_id,
            SequenceNumber(1),
            LifecyclePhase.SETTLEMENT_PENDING,
            correlation,
        )
        second = DurableOutboxRecord(
            PatchEventId("event_" + "b" * 16),
            request_id,
            SequenceNumber(2),
            LifecyclePhase.REQUEST_COMPLETED,
            correlation,
        )
        manager = EventManager()
        projection = EventManagerDurableOutboxProjection(
            manager, deduplication_limit=1
        )
        with pytest.raises(DurableStoreError) as raised:
            await getattr(projection, "deliver")(object())
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        original_trigger = manager.trigger

        async def unavailable(event: AvalanEvent) -> None:
            """Fail one event-manager delivery without retaining identity."""
            del event
            raise RuntimeError("event manager unavailable")

        monkeypatch.setattr(manager, "trigger", unavailable)
        with pytest.raises(RuntimeError):
            await projection.deliver(first)
        monkeypatch.setattr(manager, "trigger", original_trigger)
        await projection.deliver(first)
        await projection.deliver(second)
        await projection.deliver(first)
        assert len(manager.history) == 3

        class Store:
            """Return one fixed durable event without settlement authority."""

            async def outbox(
                self,
                access: DurableRequestAccess,
                after: SequenceNumber,
                limit: int,
            ) -> tuple[DurableOutboxRecord, ...]:
                """Return the fixed event after consuming typed parameters."""
                del access, after, limit
                return (first,)

        class InterruptingDelivery:
            """Raise a translated observer interruption during delivery."""

            async def deliver(self, record: DurableOutboxRecord) -> None:
                """Interrupt delivery for the supplied record."""
                del record
                raise RuntimeError("interrupted")

        class InterruptingAudit:
            """Raise a translated observer interruption during audit."""

            async def project(self, record: DurableOutboxRecord) -> None:
                """Interrupt audit for the supplied record."""
                del record
                raise RuntimeError("interrupted")

        access = DurableRequestAccess(request_id, _identity("f"))
        with pytest.raises(DurableStoreError) as raised:
            await getattr(outbox, "DurableOutboxProjector")(
                Store(), _Delivery(), _Audit()
            ).project(access, SequenceNumber(0), 0)
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        monkeypatch.setattr(
            outbox, "KeyboardInterrupt", RuntimeError, raising=False
        )
        with pytest.raises(RuntimeError):
            await getattr(outbox, "DurableOutboxProjector")(
                Store(), InterruptingDelivery(), _Audit()
            ).project(access, SequenceNumber(0), 1)
        with pytest.raises(RuntimeError):
            await getattr(outbox, "DurableOutboxProjector")(
                Store(), _Delivery(), InterruptingAudit()
            ).project(access, SequenceNumber(0), 1)

    run(scenario())


@pytest.mark.parametrize(
    "boundary",
    (
        "private_preparation",
        "commit_claim",
        "commit_started",
        "requested_effect_step",
        "terminal_settlement",
        "outbox_publication",
    ),
)
def test_lost_response_at_each_durable_boundary_attaches_same_identity(
    boundary: str,
) -> None:
    """Ignore one response then recover through the exact reservation only."""

    async def scenario() -> None:
        backend = _backend()
        first = InMemoryDurablePatchStore(backend)
        identity = _identity("c")
        digest = _digest("c")
        reservation = await first.reserve(identity, digest)
        if boundary == "private_preparation":
            await first.persist_plan(reservation, _plan(digest, "c", 1))
            replay = await InMemoryDurablePatchStore(backend).reserve(
                identity, digest
            )
            assert replay.replayed
            assert replay.request_id == reservation.request_id
            return
        plan = _plan(digest, "c", 1)
        await first.persist_plan(reservation, plan)
        claim = await first.claim_commit(
            reservation,
            plan,
            _approval(identity, digest, plan, "c"),
            _owner("c"),
            ExpiryTick(10),
            DurationTicks(20),
            (),
        )
        assert claim.lease is not None
        if boundary in {"commit_claim", "commit_started"}:
            attached = await InMemoryDurablePatchStore(backend).claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, "c"),
                _owner("d"),
                ExpiryTick(11),
                DurationTicks(20),
                (),
            )
            assert attached.state is DurableCommitClaimState.ATTACHED
            assert (
                await first.inspect(
                    DurableRequestAccess(reservation.request_id, identity)
                )
            ).lease == claim.lease
            return
        journal = await first.append_step(
            claim.lease,
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            plan.steps[0].step_id,
            CommitStepState.PLANNED,
            ExpiryTick(11),
        )
        journal = await first.append_step(
            claim.lease,
            journal.cursor,
            plan.steps[0].step_id,
            CommitStepState.COMMITTED,
            ExpiryTick(11),
        )
        result = _result(reservation.request_id, plan, MutationState.COMMITTED)
        terminal = await first.settle(
            claim.lease,
            journal.cursor,
            result,
            _correlation("c"),
            ExpiryTick(12),
        )
        fresh = InMemoryDurablePatchStore(backend)
        replay = await fresh.reserve(identity, digest)
        assert replay.replayed
        settled = await fresh.settle(
            claim.lease,
            journal.cursor,
            result,
            _correlation("c"),
            ExpiryTick(13),
        )
        assert settled == terminal
        events = await fresh.outbox(
            DurableRequestAccess(reservation.request_id, identity),
            SequenceNumber(0),
            10,
        )
        assert events == (terminal.outbox,)

    run(scenario())
