"""Run the Phase 8 durable-store contract against isolated PostgreSQL state."""

from asyncio import Event, create_task, gather, run, sleep, to_thread
from collections.abc import Awaitable, Callable
from dataclasses import replace
from os import environ
from types import SimpleNamespace
from uuid import uuid4

import pytest
from phase_8_store_test import (
    _APPROVAL_AUTHORITY,
    _approval,
    _artifact,
    _correlation,
    _digest,
    _exclusive_recovery_contract,
    _identity,
    _owner,
    _plan,
    _result,
    _worker_transition_lease_parity_contract,
)

from avalan.patch import pgsql_store as pgsql_durable
from avalan.patch.domain import (
    ArtifactState,
    Audience,
    ByteSize,
    CommitStepState,
    DurationTicks,
    ExpiryTick,
    LifecyclePhase,
    MutationState,
    PatchContextId,
    PatchExecutionId,
    PatchGrantId,
    PatchPendingOperationId,
    PatchPlanId,
    PatchRequestId,
    PatchRetentionKeyId,
    PatchRetentionRecordId,
    PatchStepId,
    SequenceNumber,
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
    DurableArtifactState,
    DurableCommitClaimState,
    DurableCommitLease,
    DurableJournal,
    DurableJournalCursor,
    DurablePatchStore,
    DurablePendingAccess,
    DurablePendingRecord,
    DurablePendingRequest,
    DurableRequestAccess,
    DurableReservation,
    DurableRetentionAccess,
    DurableRetentionEnvelopeValidator,
    DurableRetentionKind,
    DurableRetentionPolicy,
    DurableRetentionRecord,
    DurableStepJournalEntry,
    DurableStoreError,
    DurableStoreErrorCode,
    EncryptedRetentionValue,
    InMemoryDurablePatchBackend,
    InMemoryDurablePatchStore,
)
from avalan.patch.pgsql_store import (
    PgsqlDurablePatchStore,
    PgsqlDurablePatchStoreSettings,
)
from avalan.patch.policy import PatchPrincipalId, PatchTenantId, PolicyRouteId
from avalan.pgsql import (
    PgsqlFailureCategory,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_RETENTION_KEY = DurableRetentionKey(
    PatchRetentionKeyId("retention_" + "a" * 16), b"r" * 32
)
_RETENTION_CIPHER = AesGcmDurableRetentionCipher(
    InMemoryDurableRetentionKeyResolver(
        _RETENTION_KEY.key_id,
        {_RETENTION_KEY.key_id: _RETENTION_KEY},
    )
)

pytestmark = pytest.mark.skipif(
    _DSN is None,
    reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
)


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop an isolated Phase 8 schema after all owned pools close."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


async def _run_schema(
    callback: Callable[[str, str], Awaitable[None]],
) -> None:
    """Migrate and tear down one schema around a PostgreSQL contract case."""
    assert _DSN is not None
    schema = "patch_phase8_" + uuid4().hex
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    try:
        await callback(_DSN, schema)
    finally:
        await _drop_schema(_DSN, schema)


async def _store(
    dsn: str,
    schema: str,
    retention_validator: DurableRetentionEnvelopeValidator | None = None,
    retention_cipher: AesGcmDurableRetentionCipher | None = None,
) -> PgsqlDurablePatchStore:
    """Open one independent bounded store client against the test schema."""
    store = PgsqlDurablePatchStore.from_settings(
        PgsqlDurablePatchStoreSettings(
            dsn=dsn,
            schema=schema,
            pool_minimum=1,
            pool_maximum=2,
        ),
        approval_verifier=_APPROVAL_AUTHORITY,
        retention_authorizer=StaticDurableRetentionAuthorizer(
            frozenset((Audience.APPROVER,))
        ),
        retention_validator=(
            AesGcmDurableRetentionEnvelopeValidator(
                _RETENTION_CIPHER
                if retention_cipher is None
                else retention_cipher
            )
            if retention_validator is None
            else retention_validator
        ),
    )
    await store.open()
    return store


class _BlockedRetentionValidator:
    """Block an authenticated retention insert before its SQL transaction."""

    def __init__(self, delegate: DurableRetentionEnvelopeValidator) -> None:
        """Bind one genuine validator and deterministic synchronization."""
        self._delegate = delegate
        self.entered = Event()
        self.release = Event()

    async def validate(
        self,
        request_id: PatchRequestId,
        record: DurableRetentionRecord,
    ) -> None:
        """Validate the envelope, then pause before the durable write."""
        await self._delegate.validate(request_id, record)
        self.entered.set()
        await self.release.wait()


class _RotatingRetentionKeyResolver:
    """Rotate live writes while retaining exact historical decrypt keys."""

    def __init__(
        self,
        active_key: DurableRetentionKey,
        keys: tuple[DurableRetentionKey, ...],
    ) -> None:
        """Bind one nonempty exact-version key ring for an isolated test."""
        if (
            type(active_key) is not DurableRetentionKey
            or type(keys) is not tuple
            or not keys
            or any(type(item) is not DurableRetentionKey for item in keys)
            or len({item.key_id for item in keys}) != len(keys)
            or active_key.key_id not in {item.key_id for item in keys}
        ):
            raise AssertionError("retention key ring is invalid")
        self._active_key_id = active_key.key_id
        self._keys = {item.key_id: item for item in keys}

    def rotate(self, key_id: PatchRetentionKeyId) -> None:
        """Select one existing key for subsequent versioned envelopes."""
        if type(key_id) is not PatchRetentionKeyId or key_id not in self._keys:
            raise AssertionError("retention key is unavailable")
        self._active_key_id = key_id

    async def active_key(self) -> DurableRetentionKey:
        """Return the currently rotated write key without exposing bytes."""
        return self._keys[self._active_key_id]

    async def read_key(
        self, key_id: PatchRetentionKeyId
    ) -> DurableRetentionKey:
        """Return only the exact retained historical key version."""
        if type(key_id) is not PatchRetentionKeyId or key_id not in self._keys:
            raise AssertionError("retention key is unavailable")
        return self._keys[key_id]


async def _semantic_outcome(
    store: DurablePatchStore,
    token: str,
) -> tuple[
    bool,
    DurableStoreErrorCode,
    DurableCommitClaimState,
    int,
    MutationState,
    tuple[tuple[int, LifecyclePhase], ...],
]:
    """Run the portable Phase 8 semantic contract without target execution."""
    identity = _identity(token)
    digest = _digest(token)
    reservation = await store.reserve(identity, digest)
    duplicate = await store.reserve(identity, digest)
    with pytest.raises(DurableStoreError) as conflicting:
        await store.reserve(identity, _digest("a" if token != "a" else "b"))
    plan = _plan(digest, token, step_count=1)
    await store.persist_plan(reservation, plan)
    wrong_route_identity = replace(
        identity, route_id=PolicyRouteId("route-forged-" + token)
    )
    with pytest.raises(DurableStoreError) as route_denied:
        await store.claim_commit(
            reservation,
            plan,
            _approval(wrong_route_identity, digest, plan, token),
            _owner(token),
            ExpiryTick(10),
            DurationTicks(30),
            (),
        )
    assert route_denied.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH
    claim = await store.claim_commit(
        reservation,
        plan,
        _approval(identity, digest, plan, token),
        _owner(token),
        ExpiryTick(10),
        DurationTicks(30),
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
    pending = DurablePendingRequest(
        PatchPendingOperationId("pending_" + token * 16),
        _correlation(token),
        DurationTicks(5),
    )
    await store.suspend(claim.lease, pending, ExpiryTick(12))
    access = DurablePendingAccess(
        DurableRequestAccess(reservation.request_id, identity),
        pending.pending_operation_id,
        pending.correlation_id,
    )
    terminal = await store.settle(
        claim.lease,
        journal.cursor,
        _result(reservation.request_id, plan, MutationState.COMMITTED),
        pending.correlation_id,
        ExpiryTick(13),
    )
    assert terminal.pending_operation_id == pending.pending_operation_id
    assert await store.inspect_pending(access) == terminal
    wrong_handle = DurablePendingAccess(
        access.request,
        PatchPendingOperationId(
            "pending_" + ("e" if token != "e" else "d") * 16
        ),
        pending.correlation_id,
    )
    with pytest.raises(DurableStoreError) as denied:
        await store.inspect_pending(wrong_handle)
    assert denied.value.code is DurableStoreErrorCode.ACCESS_DENIED
    events = await store.outbox(access.request, SequenceNumber(0), 10)
    return (
        duplicate.replayed,
        conflicting.value.code,
        claim.state,
        terminal.outbox.sequence.value,
        terminal.result.truth.mutation_state,
        tuple((event.sequence.value, event.lifecycle) for event in events),
    )


async def _schema_exists(dsn: str, schema: str) -> bool:
    """Return whether the exact isolated schema remains after teardown."""
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT EXISTS ("
                    "SELECT 1 FROM information_schema.schemata "
                    "WHERE schema_name = %s"
                    ") AS exists",
                    (schema,),
                )
                row = await cursor.fetchone()
                assert row is not None
                return bool(row["exists"])


def test_pgsql_store_races_claim_and_reconciles_after_pool_restart() -> None:
    """Race durable ownership, restart pools, and preserve one terminal."""

    async def scenario(dsn: str, schema: str) -> None:
        first = await _store(dsn, schema)
        second = await _store(dsn, schema)
        try:
            identity = _identity("a")
            digest = _digest("a")
            reserved = await gather(
                first.reserve(identity, digest),
                second.reserve(identity, digest),
            )
            assert {item.replayed for item in reserved} == {False, True}
            reservation = reserved[0]
            with pytest.raises(DurableStoreError) as conflicting:
                await second.reserve(identity, _digest("f"))
            assert (
                conflicting.value.code
                is DurableStoreErrorCode.IDEMPOTENCY_CONFLICT
            )
            plan = _plan(digest, "a", step_count=1)
            await first.persist_plan(reservation, plan)
            approval = _approval(identity, digest, plan, "a")
            forged_claims = (
                replace(
                    approval,
                    grant_id=PatchGrantId("grant_" + "f" * 16),
                ),
                replace(
                    approval,
                    fingerprint_digest=_digest("e"),
                ),
                replace(
                    approval,
                    context_id=PatchContextId("context_" + "f" * 16),
                ),
                replace(
                    approval,
                    reviewers=(PatchPrincipalId("reviewer-forged"),),
                ),
            )
            for forged in forged_claims:
                with pytest.raises(DurableStoreError) as denied:
                    await first.claim_commit(
                        reservation,
                        plan,
                        forged,
                        _owner("a"),
                        ExpiryTick(10),
                        DurationTicks(30),
                        (),
                    )
                assert (
                    denied.value.code
                    is DurableStoreErrorCode.APPROVAL_MISMATCH
                )
            claims = await gather(
                first.claim_commit(
                    reservation,
                    plan,
                    approval,
                    _owner("a"),
                    ExpiryTick(10),
                    DurationTicks(30),
                    (),
                ),
                second.claim_commit(
                    reservation,
                    plan,
                    _approval(identity, digest, plan, "a"),
                    _owner("b"),
                    ExpiryTick(10),
                    DurationTicks(30),
                    (),
                ),
            )
            assert {item.state for item in claims} == {
                DurableCommitClaimState.OWNER,
                DurableCommitClaimState.ATTACHED,
            }
            owner = next(
                item
                for item in claims
                if item.state is DurableCommitClaimState.OWNER
            )
            assert owner.lease is not None
            planned = await gather(
                first.append_step(
                    owner.lease,
                    DurableJournalCursor(
                        reservation.request_id, SequenceNumber(0)
                    ),
                    plan.steps[0].step_id,
                    CommitStepState.PLANNED,
                    ExpiryTick(11),
                ),
                second.append_step(
                    owner.lease,
                    DurableJournalCursor(
                        reservation.request_id, SequenceNumber(0)
                    ),
                    plan.steps[0].step_id,
                    CommitStepState.PLANNED,
                    ExpiryTick(11),
                ),
                return_exceptions=True,
            )
            journals = tuple(
                item for item in planned if not isinstance(item, Exception)
            )
            failures = tuple(
                item for item in planned if isinstance(item, Exception)
            )
            assert len(journals) == 1
            assert len(failures) == 1
            assert isinstance(failures[0], DurableStoreError)
            assert failures[0].code is DurableStoreErrorCode.JOURNAL_CONFLICT
            journal = journals[0]
            assert isinstance(journal, DurableJournal)
            journal = await first.append_step(
                owner.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            pending = DurablePendingRequest(
                PatchPendingOperationId("pending_" + "a" * 16),
                _correlation("a"),
                DurationTicks(5),
            )
            await first.suspend(owner.lease, pending, ExpiryTick(12))
            await first.aclose()
            await second.aclose()
            restarted = await _store(dsn, schema)
            try:
                access = DurablePendingAccess(
                    DurableRequestAccess(reservation.request_id, identity),
                    pending.pending_operation_id,
                    pending.correlation_id,
                )
                current_pending = await restarted.inspect_pending(access)
                assert isinstance(current_pending, DurablePendingRecord)
                assert current_pending.request_id == reservation.request_id
                terminal = await restarted.settle(
                    owner.lease,
                    journal.cursor,
                    _result(
                        reservation.request_id, plan, MutationState.COMMITTED
                    ),
                    pending.correlation_id,
                    ExpiryTick(13),
                )
                assert await restarted.await_terminal(access) == terminal
                events = await restarted.outbox(
                    access.request, SequenceNumber(0), 10
                )
                assert tuple(item.sequence for item in events) == (
                    SequenceNumber(1),
                    SequenceNumber(2),
                )
            finally:
                await restarted.aclose()
        finally:
            await first.aclose()
            await second.aclose()

    run(_run_schema(scenario))


def test_pgsql_store_rotates_retention_and_fences_expired_owner() -> None:
    """Read versioned ciphertext and reject a stale owner after replacement."""

    async def scenario(dsn: str, schema: str) -> None:
        store = await _store(dsn, schema)
        try:
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
                DurationTicks(10),
                (),
            )
            assert claim.lease is not None
            policy = DurableRetentionPolicy(ExpiryTick(100), True)
            first_id = PatchRetentionRecordId("retained_" + "a" * 16)
            second_id = PatchRetentionRecordId("retained_" + "b" * 16)
            first_value = await _RETENTION_CIPHER.seal(
                b"key-one",
                DurableRetentionBinding(
                    reservation.request_id,
                    first_id,
                    DurableRetentionKind.SEALED_PLAN,
                ),
            )
            second_value = await _RETENTION_CIPHER.seal(
                b"key-two",
                DurableRetentionBinding(
                    reservation.request_id,
                    second_id,
                    DurableRetentionKind.REVIEW_ARTIFACT,
                ),
            )
            records = (
                DurableRetentionRecord(
                    first_id,
                    DurableRetentionKind.SEALED_PLAN,
                    first_value.key_id,
                    first_value.value,
                    policy,
                ),
                DurableRetentionRecord(
                    second_id,
                    DurableRetentionKind.REVIEW_ARTIFACT,
                    second_value.key_id,
                    second_value.value,
                    policy,
                ),
            )
            for record in records:
                await store.put_retention(reservation, record)
            access = DurableRetentionAccess(
                DurableRequestAccess(reservation.request_id, identity)
            )
            with pytest.raises(TypeError):
                getattr(DurableRetentionAccess, "__init__")(
                    access, access.request, Audience.PUBLIC
                )
            invalid_id = PatchRetentionRecordId("retained_" + "c" * 16)
            wrong_binding = await _RETENTION_CIPHER.seal(
                b"wrong-binding",
                DurableRetentionBinding(
                    reservation.request_id,
                    first_id,
                    DurableRetentionKind.SEALED_PLAN,
                ),
            )
            invalid_records = (
                DurableRetentionRecord(
                    invalid_id,
                    DurableRetentionKind.SEALED_PLAN,
                    wrong_binding.key_id,
                    wrong_binding.value,
                    policy,
                ),
                DurableRetentionRecord(
                    invalid_id,
                    DurableRetentionKind.SEALED_PLAN,
                    PatchRetentionKeyId("retention_" + "b" * 16),
                    first_value.value,
                    policy,
                ),
                DurableRetentionRecord(
                    invalid_id,
                    DurableRetentionKind.SEALED_PLAN,
                    first_value.key_id,
                    EncryptedRetentionValue(b"plaintext-is-not-an-envelope"),
                    policy,
                ),
            )
            for invalid in invalid_records:
                with pytest.raises(DurableStoreError) as denied:
                    await store.put_retention(reservation, invalid)
                assert (
                    denied.value.code is DurableStoreErrorCode.RETENTION_DENIED
                )
            assert (
                await store.get_retention(
                    access, records[0].retention_id, ExpiryTick(11)
                )
                == records[0]
            )
            replacement = await store.replace_expired_owner(
                reservation,
                claim.lease,
                _owner("c"),
                ExpiryTick(20),
                DurationTicks(10),
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.append_step(
                    claim.lease,
                    DurableJournalCursor(
                        reservation.request_id, SequenceNumber(0)
                    ),
                    plan.steps[0].step_id,
                    CommitStepState.PLANNED,
                    ExpiryTick(21),
                )
            assert raised.value.code is DurableStoreErrorCode.FENCED
            with pytest.raises(DurableStoreError) as raised:
                await store.append_artifact(
                    claim.lease,
                    DurableJournalCursor(
                        reservation.request_id, SequenceNumber(0)
                    ),
                    _artifact("c"),
                    DurableArtifactState.REMOVED,
                    ExpiryTick(21),
                )
            assert raised.value.code is DurableStoreErrorCode.FENCED
            with pytest.raises(DurableStoreError) as raised:
                await store.settle(
                    claim.lease,
                    DurableJournalCursor(
                        reservation.request_id, SequenceNumber(0)
                    ),
                    _result(
                        reservation.request_id, plan, MutationState.COMMITTED
                    ),
                    _correlation("b"),
                    ExpiryTick(21),
                )
            assert raised.value.code is DurableStoreErrorCode.FENCED
            assert await store.is_current_fence(replacement, ExpiryTick(21))
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_pgsql_expired_reaped_owner_remains_exclusive_until_settlement() -> (
    None
):
    """Run the shared exclusive-recovery contract across two SQL pools."""

    async def scenario(dsn: str, schema: str) -> None:
        first = await _store(dsn, schema)
        second = await _store(dsn, schema)
        try:
            await _exclusive_recovery_contract(first, second)
        finally:
            await first.aclose()
            await second.aclose()

    run(_run_schema(scenario))


def test_pgsql_worker_transitions_require_the_exact_renewed_lease() -> None:
    """Run renewed worker transition parity through PostgreSQL."""

    async def scenario(dsn: str, schema: str) -> None:
        store = await _store(dsn, schema)
        try:
            await _worker_transition_lease_parity_contract(store)
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_pgsql_retention_validation_cannot_outlive_terminal_cleanup() -> None:
    """Reject a validated late retained insert after another pool settles."""

    async def scenario(dsn: str, schema: str) -> None:
        validator = _BlockedRetentionValidator(
            AesGcmDurableRetentionEnvelopeValidator(_RETENTION_CIPHER)
        )
        retention_store = await _store(dsn, schema, validator)
        settlement_store = await _store(dsn, schema)
        try:
            identity = _identity("f")
            digest = _digest("f")
            reservation = await settlement_store.reserve(identity, digest)
            plan = _plan(digest, "f", step_count=1)
            await settlement_store.persist_plan(reservation, plan)
            claim = await settlement_store.claim_commit(
                reservation,
                plan,
                _approval(identity, digest, plan, "f"),
                _owner("f"),
                ExpiryTick(10),
                DurationTicks(30),
                (),
            )
            assert claim.lease is not None
            retention_id = PatchRetentionRecordId("retained_" + "f" * 16)
            sealed = await _RETENTION_CIPHER.seal(
                b"retention-terminal-race",
                DurableRetentionBinding(
                    reservation.request_id,
                    retention_id,
                    DurableRetentionKind.SEALED_PLAN,
                ),
            )
            retained = DurableRetentionRecord(
                retention_id,
                DurableRetentionKind.SEALED_PLAN,
                sealed.key_id,
                sealed.value,
                DurableRetentionPolicy(ExpiryTick(100), True),
            )
            insertion = create_task(
                retention_store.put_retention(reservation, retained)
            )
            await validator.entered.wait()
            journal = await settlement_store.append_step(
                claim.lease,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            )
            journal = await settlement_store.append_step(
                claim.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            await settlement_store.settle(
                claim.lease,
                journal.cursor,
                _result(reservation.request_id, plan, MutationState.COMMITTED),
                _correlation("f"),
                ExpiryTick(12),
            )
            validator.release.set()
            with pytest.raises(DurableStoreError) as denied:
                await insertion
            assert denied.value.code is DurableStoreErrorCode.RETENTION_DENIED
            with pytest.raises(DurableStoreError) as denied:
                await settlement_store.get_retention(
                    DurableRetentionAccess(
                        DurableRequestAccess(reservation.request_id, identity)
                    ),
                    retained.retention_id,
                    ExpiryTick(13),
                )
            assert denied.value.code is DurableStoreErrorCode.RETENTION_DENIED
        finally:
            validator.release.set()
            await retention_store.aclose()
            await settlement_store.aclose()

    run(_run_schema(scenario))


def test_pgsql_retention_key_rotation_reads_exact_historical_versions() -> (
    None
):
    """Read authorized K1 and K2 records without changing durable identity."""

    async def scenario(dsn: str, schema: str) -> None:
        first_key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "e" * 16), b"e" * 32
        )
        second_key = DurableRetentionKey(
            PatchRetentionKeyId("retention_" + "f" * 16), b"f" * 32
        )
        resolver = _RotatingRetentionKeyResolver(
            first_key, (first_key, second_key)
        )
        cipher = AesGcmDurableRetentionCipher(resolver)
        store = await _store(dsn, schema, retention_cipher=cipher)
        try:
            identity = _identity("e")
            digest = _digest("e")
            reservation = await store.reserve(identity, digest)
            plan = _plan(digest, "e", step_count=1)
            await store.persist_plan(reservation, plan)
            first_id = PatchRetentionRecordId("retained_" + "e" * 16)
            second_id = PatchRetentionRecordId("retained_" + "f" * 16)
            first_value = await cipher.seal(
                b"retained-under-key-one",
                DurableRetentionBinding(
                    reservation.request_id,
                    first_id,
                    DurableRetentionKind.SEALED_PLAN,
                ),
            )
            resolver.rotate(second_key.key_id)
            second_value = await cipher.seal(
                b"retained-under-key-two",
                DurableRetentionBinding(
                    reservation.request_id,
                    second_id,
                    DurableRetentionKind.REVIEW_ARTIFACT,
                ),
            )
            first = DurableRetentionRecord(
                first_id,
                DurableRetentionKind.SEALED_PLAN,
                first_value.key_id,
                first_value.value,
                DurableRetentionPolicy(ExpiryTick(100), False),
            )
            second = DurableRetentionRecord(
                second_id,
                DurableRetentionKind.REVIEW_ARTIFACT,
                second_value.key_id,
                second_value.value,
                DurableRetentionPolicy(ExpiryTick(100), False),
            )
            await store.put_retention(reservation, first)
            await store.put_retention(reservation, second)
            access = DurableRetentionAccess(
                DurableRequestAccess(reservation.request_id, identity)
            )
            assert first.key_id == first_key.key_id
            assert second.key_id == second_key.key_id
            assert (
                await store.get_retention(
                    access, first.retention_id, ExpiryTick(10)
                )
                == first
            )
            assert (
                await store.get_retention(
                    access, second.retention_id, ExpiryTick(10)
                )
                == second
            )
            snapshot = await store.inspect(access.request)
            assert snapshot.reservation.identity == identity
            assert snapshot.reservation.canonical_digest == digest
            assert snapshot.plan == plan
            assert snapshot.terminal is None
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_pgsql_expiry_cleanup_keeps_minimum_terminal_retry_truth() -> None:
    """Delete private review and plan values but retain terminal evidence."""

    async def scenario(dsn: str, schema: str) -> None:
        store = await _store(dsn, schema)
        try:
            identity = _identity("d")
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
                DurationTicks(30),
                (),
            )
            assert claim.lease is not None
            policy = DurableRetentionPolicy(ExpiryTick(20), False)
            retained: list[DurableRetentionRecord] = []
            for token, kind in (
                ("e", DurableRetentionKind.SEALED_PLAN),
                ("f", DurableRetentionKind.REVIEW_ARTIFACT),
            ):
                retention_id = PatchRetentionRecordId("retained_" + token * 16)
                value = await _RETENTION_CIPHER.seal(
                    ("private-" + token).encode("ascii"),
                    DurableRetentionBinding(
                        reservation.request_id,
                        retention_id,
                        kind,
                    ),
                )
                record = DurableRetentionRecord(
                    retention_id,
                    kind,
                    value.key_id,
                    value.value,
                    policy,
                )
                await store.put_retention(reservation, record)
                retained.append(record)
            journal = await store.append_step(
                claim.lease,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
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
                _correlation("d"),
                ExpiryTick(12),
            )
            access = DurableRetentionAccess(
                DurableRequestAccess(reservation.request_id, identity)
            )
            cleanup = await store.cleanup_retention(ExpiryTick(20))
            assert cleanup.records_deleted == len(retained)
            assert cleanup.bytes_deleted == ByteSize(
                sum(record.value.size().value for record in retained)
            )
            replay = await store.reserve(identity, digest)
            assert replay.replayed
            assert replay.request_id == reservation.request_id
            snapshot = await store.inspect(access.request)
            assert snapshot.reservation == reservation
            assert snapshot.plan == plan
            assert snapshot.journal == journal
            assert snapshot.terminal == terminal
            events = await store.outbox(access.request, SequenceNumber(0), 10)
            assert events == (terminal.outbox,)
            for record in retained:
                with pytest.raises(DurableStoreError) as denied:
                    await store.get_retention(
                        access, record.retention_id, ExpiryTick(20)
                    )
                assert (
                    denied.value.code is DurableStoreErrorCode.RETENTION_DENIED
                )
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_pgsql_terminal_truth_cannot_relabel_leaked_or_unknown_artifacts() -> (
    None
):
    """Reject a cleaned terminal result that contradicts durable artifacts."""

    async def scenario(dsn: str, schema: str) -> None:
        store = await _store(dsn, schema)
        try:
            for token, state in (
                ("c", DurableArtifactState.LEAKED),
                ("d", DurableArtifactState.UNKNOWN),
            ):
                identity = _identity(token)
                digest = _digest(token)
                reservation = await store.reserve(identity, digest)
                plan = _plan(digest, token, step_count=1)
                artifact = _artifact(token)
                await store.persist_plan(reservation, plan)
                claim = await store.claim_commit(
                    reservation,
                    plan,
                    _approval(identity, digest, plan, token),
                    _owner(token),
                    ExpiryTick(10),
                    DurationTicks(20),
                    (artifact,),
                )
                assert claim.lease is not None
                snapshot = await store.inspect(
                    DurableRequestAccess(reservation.request_id, identity)
                )
                journal = await store.append_step(
                    claim.lease,
                    snapshot.journal.cursor,
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
                journal = await store.append_artifact(
                    claim.lease,
                    journal.cursor,
                    artifact,
                    DurableArtifactState.PRESENT,
                    ExpiryTick(11),
                )
                journal = await store.append_artifact(
                    claim.lease,
                    journal.cursor,
                    artifact,
                    state,
                    ExpiryTick(11),
                )
                with pytest.raises(DurableStoreError) as denied:
                    await store.settle(
                        claim.lease,
                        journal.cursor,
                        _result(
                            reservation.request_id,
                            plan,
                            MutationState.COMMITTED,
                            ArtifactState.CLEANED,
                        ),
                        _correlation(token),
                        ExpiryTick(12),
                    )
                assert (
                    denied.value.code
                    is DurableStoreErrorCode.TERMINAL_CONFLICT
                )
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_semantic_contract_matches_memory_and_postgresql() -> None:
    """Run one closed durable contract against both supported stores."""

    async def scenario(dsn: str, schema: str) -> None:
        memory = InMemoryDurablePatchStore(
            InMemoryDurablePatchBackend(
                approval_verifier=_APPROVAL_AUTHORITY,
            )
        )
        expected = await _semantic_outcome(memory, "e")
        store = await _store(dsn, schema)
        try:
            assert await _semantic_outcome(store, "f") == expected
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_pgsql_pending_authority_liveness_and_schema_teardown() -> None:
    """Deny every foreign pending handle without blocking unrelated work."""
    schema_names: list[str] = []

    async def scenario(dsn: str, schema: str) -> None:
        schema_names.append(schema)
        store = await _store(dsn, schema)
        try:
            identity = _identity("d")
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
                DurationTicks(30),
                (),
            )
            assert claim.lease is not None
            journal = await store.append_step(
                claim.lease,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
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
                PatchPendingOperationId("pending_" + "d" * 16),
                _correlation("d"),
                DurationTicks(5),
            )
            await store.suspend(claim.lease, pending, ExpiryTick(12))
            access = DurablePendingAccess(
                DurableRequestAccess(reservation.request_id, identity),
                pending.pending_operation_id,
                pending.correlation_id,
            )
            wrong_identities = (
                replace(identity, tenant_id=PatchTenantId("tenant-x")),
                replace(
                    identity,
                    principal_id=PatchPrincipalId("principal-x"),
                ),
                replace(
                    identity,
                    execution_id=PatchExecutionId("execution_" + "f" * 16),
                ),
                replace(identity, route_id=PolicyRouteId("route-x")),
            )
            wrong_accesses = tuple(
                DurablePendingAccess(
                    DurableRequestAccess(reservation.request_id, candidate),
                    pending.pending_operation_id,
                    pending.correlation_id,
                )
                for candidate in wrong_identities
            ) + (
                DurablePendingAccess(
                    DurableRequestAccess(
                        PatchRequestId("request_" + "f" * 16), identity
                    ),
                    pending.pending_operation_id,
                    pending.correlation_id,
                ),
                DurablePendingAccess(
                    access.request,
                    PatchPendingOperationId("pending_" + "f" * 16),
                    pending.correlation_id,
                ),
                DurablePendingAccess(
                    access.request,
                    pending.pending_operation_id,
                    _correlation("f"),
                ),
            )
            for wrong in wrong_accesses:
                with pytest.raises(DurableStoreError) as denied:
                    await store.inspect_pending(wrong)
                assert denied.value.code is DurableStoreErrorCode.ACCESS_DENIED
                with pytest.raises(DurableStoreError) as denied:
                    await store.await_terminal(wrong)
                assert denied.value.code is DurableStoreErrorCode.ACCESS_DENIED
                if wrong.request != access.request:
                    with pytest.raises(DurableStoreError) as denied:
                        await store.request_cancellation(wrong.request)
                    assert (
                        denied.value.code
                        is DurableStoreErrorCode.ACCESS_DENIED
                    )
                    with pytest.raises(DurableStoreError) as denied:
                        await store.outbox(
                            wrong.request, SequenceNumber(0), 10
                        )
                    assert (
                        denied.value.code
                        is DurableStoreErrorCode.ACCESS_DENIED
                    )
            assert isinstance(
                await store.inspect_pending(access), DurablePendingRecord
            )

            blocker = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
            async with blocker:
                async with blocker.connection() as connection:
                    async with connection.transaction():
                        async with connection.cursor() as cursor:
                            await cursor.execute(
                                'SELECT "request_id" FROM '
                                f"{quote_pgsql_identifier(schema)}."
                                '"patch_durable_requests" '
                                'WHERE "request_id" = %s FOR UPDATE',
                                (reservation.request_id.value,),
                            )
                            unrelated = await store.reserve(
                                _identity("e"), _digest("e")
                            )
                            assert not unrelated.replayed
        finally:
            await store.aclose()

    run(_run_schema(scenario))
    assert schema_names
    assert _DSN is not None
    assert not run(_schema_exists(_DSN, schema_names[0]))


def test_pgsql_durable_helpers_reject_malformed_persistence_values() -> None:
    """Fail closed before SQL for malformed durable persistence witnesses."""
    with pytest.raises(DurableStoreError) as raised:
        PgsqlDurablePatchStoreSettings(dsn="")
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        PgsqlDurablePatchStoreSettings(
            dsn="postgresql://example.invalid/db",
            pool_minimum=2,
            pool_maximum=1,
        )
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        getattr(PgsqlDurablePatchStore, "__init__")(object(), object())
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._require_exact(object(), PatchRequestId)
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    artifact = _artifact("a")
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._require_artifacts((artifact, artifact))
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._lease_expiry(ExpiryTick(2**63 - 1), DurationTicks(1))
    assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
    digest = _digest("a")
    plan = _plan(digest, step_count=1)
    assert pgsql_durable._decode_plan(pgsql_durable._encode_plan(plan)) == plan
    malformed_plan = _plan(digest, "b", step_count=1)
    object.__setattr__(
        malformed_plan,
        "plan_id",
        type("MalformedPlanId", (), {"value": "plan\x1fseparator"})(),
    )
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._encode_plan(malformed_plan)
    assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
    for malformed in (b"not-a-plan", b"\xff"):
        with pytest.raises(DurableStoreError) as raised:
            pgsql_durable._decode_plan(malformed)
        assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
    assert (
        pgsql_durable._row_bytes({"value": memoryview(b"bytes")}, "value")
        == b"bytes"
    )
    assert pgsql_durable._row_optional_str({"value": None}, "value") is None
    assert pgsql_durable._row_optional_int({"value": None}, "value") is None
    for reader in (
        pgsql_durable._row_str,
        pgsql_durable._row_int,
        pgsql_durable._row_bool,
        pgsql_durable._row_bytes,
    ):
        with pytest.raises(DurableStoreError) as raised:
            reader({"value": object()}, "value")
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._row_optional_int({"value": 0}, "value")
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    assert pgsql_durable._step_transition(None, CommitStepState.PLANNED)
    assert not pgsql_durable._step_transition(
        CommitStepState.COMMITTED, CommitStepState.UNKNOWN
    )
    assert pgsql_durable._artifact_transition(
        DurableArtifactState.INTENDED, DurableArtifactState.PRESENT
    )
    assert not pgsql_durable._artifact_transition(
        DurableArtifactState.REMOVED, DurableArtifactState.LEAKED
    )
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._require_plan(None)
    assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._validate_approval(
            DurableReservation(
                PatchRequestId("request_" + "a" * 16),
                _identity("a"),
                digest,
                False,
            ),
            plan,
            _approval(_identity("a"), digest, plan, "a", expires_at=1),
            ExpiryTick(1),
        )
    assert raised.value.code is DurableStoreErrorCode.APPROVAL_EXPIRED

    class Cursor:
        """Return configured rows for one closed helper-boundary test."""

        def __init__(self, rows: tuple[dict[str, object] | None, ...]) -> None:
            """Store the finite response sequence for fetches."""
            self._rows = iter(rows)

        async def execute(self, statement: str, parameters: object) -> None:
            """Accept helper SQL without connecting to a database."""
            del statement, parameters

        async def fetchone(self) -> dict[str, object] | None:
            """Return the next configured row."""
            return next(self._rows)

    async def scenario() -> None:
        reservation = DurableReservation(
            PatchRequestId("request_" + "a" * 16),
            _identity("a"),
            digest,
            False,
        )
        mismatched: dict[str, object] = {
            "request_id": "request_" + "b" * 16,
            "canonical_digest": digest.value,
        }
        with pytest.raises(DurableStoreError) as raised:
            await getattr(pgsql_durable, "_select_reservation_for_update")(
                Cursor((mismatched,)), reservation
            )
        assert raised.value.code is DurableStoreErrorCode.INVALID_RESERVATION
        with pytest.raises(DurableStoreError) as raised:
            await getattr(pgsql_durable, "_advance_journal")(
                Cursor((None,)),
                {"request_id": reservation.request_id.value},
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(1)
                ),
            )
        assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
        with pytest.raises(DurableStoreError) as raised:
            await getattr(pgsql_durable, "_terminal")(
                Cursor(()), {"terminal_result": None}
            )
        assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT

    run(scenario())
    incomplete_lease = {
        "owner_id": "owner_" + "a" * 16,
        "domain_id": None,
        "lease_expires_at": None,
    }
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._lease_from_row(incomplete_lease)
    assert raised.value.code is DurableStoreErrorCode.FENCED
    broken_pending = {
        "pending_operation_id": "pending_" + "a" * 16,
        "pending_correlation_id": None,
        "pending_next_check_after": None,
        "pending_event_cursor": None,
        "owner_id": None,
    }
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._pending_from_row(broken_pending)
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._retention_from_row({"ciphertext": None})
    assert raised.value.code is DurableStoreErrorCode.RETENTION_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._retention_from_row(
            {
                "ciphertext": b"opaque",
                "ciphertext_digest": "forged",
            }
        )
    assert raised.value.code is DurableStoreErrorCode.RETENTION_CONFLICT
    lease = DurableCommitLease(
        PatchRequestId("request_" + "a" * 16),
        plan.domain_id,
        _owner("a"),
        SequenceNumber(1),
        ExpiryTick(10),
    )
    with pytest.raises(DurableStoreError) as raised:
        run(
            PgsqlDurablePatchStore(
                type("Pool", (), {"connection": lambda self: None})(),
                owns_database=False,
            ).renew_lease(lease, ExpiryTick(9), DurationTicks(1))
        )
    assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
    with pytest.raises(DurableStoreError) as raised:
        run(
            PgsqlDurablePatchStore(
                type("Pool", (), {"connection": lambda self: None})(),
                owns_database=False,
            ).replace_expired_owner(
                DurableReservation(
                    PatchRequestId("request_" + "a" * 16),
                    _identity("a"),
                    digest,
                    False,
                ),
                lease,
                lease.owner_id,
                ExpiryTick(10),
                DurationTicks(1),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED
    with pytest.raises(DurableStoreError) as raised:
        run(
            PgsqlDurablePatchStore(
                type("Pool", (), {"connection": lambda self: None})(),
                owns_database=False,
            ).outbox(
                DurableRequestAccess(lease.request_id, _identity("a")),
                SequenceNumber(0),
                0,
            )
        )
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._require_lease(
            {
                "request_id": lease.request_id.value,
                "owner_id": lease.owner_id.value,
                "domain_id": lease.domain_id.value,
                "fence": lease.fence.value,
                "lease_expires_at": lease.expires_at.value,
                "lifecycle": LifecyclePhase.COMMIT_STARTED.value,
            },
            lease,
            ExpiryTick(10),
        )
    assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
    with pytest.raises(DurableStoreError) as raised:
        run(
            getattr(pgsql_durable, "_advance_journal")(
                Cursor(()),
                {"request_id": lease.request_id.value},
                DurableJournalCursor(lease.request_id, SequenceNumber(8192)),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
    unknown = DurableJournal(
        DurableJournalCursor(lease.request_id, SequenceNumber(1)),
        (
            DurableStepJournalEntry(
                DurableJournalCursor(lease.request_id, SequenceNumber(1)),
                plan.steps[0].step_id,
                plan.steps[0].lineage_id,
                CommitStepState.UNKNOWN,
            ),
        ),
        (),
    )
    assert (
        pgsql_durable._journal_mutation_state(unknown, plan)
        is MutationState.INDETERMINATE
    )
    journal = DurableJournal(
        DurableJournalCursor(lease.request_id, SequenceNumber(1)),
        (
            DurableStepJournalEntry(
                DurableJournalCursor(lease.request_id, SequenceNumber(1)),
                plan.steps[0].step_id,
                plan.steps[0].lineage_id,
                CommitStepState.NOT_COMMITTED,
            ),
        ),
        (),
    )
    assert (
        pgsql_durable._journal_mutation_state(journal, plan)
        is MutationState.NOT_COMMITTED
    )
    partial_plan = _plan(digest, "c", step_count=2)
    partial = DurableJournal(
        DurableJournalCursor(lease.request_id, SequenceNumber(2)),
        (
            DurableStepJournalEntry(
                DurableJournalCursor(lease.request_id, SequenceNumber(1)),
                partial_plan.steps[0].step_id,
                partial_plan.steps[0].lineage_id,
                CommitStepState.COMMITTED,
            ),
            DurableStepJournalEntry(
                DurableJournalCursor(lease.request_id, SequenceNumber(2)),
                partial_plan.steps[1].step_id,
                partial_plan.steps[1].lineage_id,
                CommitStepState.NOT_COMMITTED,
            ),
        ),
        (),
    )
    assert (
        pgsql_durable._journal_mutation_state(partial, partial_plan)
        is MutationState.PARTIALLY_COMMITTED
    )
    planned = DurableJournal(
        DurableJournalCursor(lease.request_id, SequenceNumber(1)),
        (
            DurableStepJournalEntry(
                DurableJournalCursor(lease.request_id, SequenceNumber(1)),
                plan.steps[0].step_id,
                plan.steps[0].lineage_id,
                CommitStepState.PLANNED,
            ),
        ),
        (),
    )
    with pytest.raises(DurableStoreError) as raised:
        pgsql_durable._journal_mutation_state(planned, plan)
    assert raised.value.code is DurableStoreErrorCode.JOURNAL_INCOMPLETE


def test_pgsql_durable_pool_lifecycle_is_idempotent_and_closed() -> None:
    """Open and close only an owned durable pool without SQL side effects."""

    class Pool:
        """Record owned pool lifecycle calls without exposing a connection."""

        def __init__(self) -> None:
            """Initialize empty lifecycle counters."""
            self.opened = 0
            self.closed = 0

        def connection(self) -> None:
            """Satisfy the durable store's database shape check."""
            return None

        async def open(self) -> None:
            """Record an owned pool open."""
            self.opened += 1

        async def aclose(self) -> None:
            """Record an owned pool close."""
            self.closed += 1

    async def scenario() -> None:
        pool = Pool()
        store = getattr(pgsql_durable, "PgsqlDurablePatchStore")(
            pool, owns_database=True
        )
        assert store.database is pool
        await store.open()
        await store.open()
        assert pool.opened == 1
        await store.aclose()
        await store.aclose()
        assert pool.closed == 1
        with pytest.raises(DurableStoreError) as raised:
            await store.open()
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
        context_pool = Pool()
        async with getattr(pgsql_durable, "PgsqlDurablePatchStore")(
            context_pool, owns_database=True
        ) as context_store:
            assert context_store.database is context_pool
        assert context_pool.opened == context_pool.closed == 1
        unowned_pool = Pool()
        unowned = getattr(pgsql_durable, "PgsqlDurablePatchStore")(
            unowned_pool, owns_database=False
        )
        await unowned.open()
        await unowned.aclose()
        assert unowned_pool.closed == 0

    with pytest.raises(DurableStoreError) as raised:
        getattr(PgsqlDurablePatchStore, "from_settings")(object())
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
    run(scenario())


def test_pgsql_durable_transaction_rethrows_and_translates_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve control flow while classifying database callback failures."""

    async def scenario(dsn: str, schema: str) -> None:
        unopened = PgsqlDurablePatchStore(
            PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn)),
            owns_database=False,
        )

        async def unused(_: object) -> None:
            """Provide an unreachable callback for lifecycle validation."""

        with pytest.raises(DurableStoreError) as raised:
            await unopened._transaction("unopened", unused)
        assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

        store = await _store(dsn, schema)
        try:

            async def durable_failure(_: object) -> None:
                """Raise a semantic error that must retain its exact code."""
                raise DurableStoreError(DurableStoreErrorCode.ACCESS_DENIED)

            with pytest.raises(DurableStoreError) as raised:
                await store._transaction("durable", durable_failure)
            assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED

            async def interrupt(_: object) -> None:
                """Raise an interrupt that bypasses database classification."""
                raise KeyboardInterrupt

            with pytest.raises(KeyboardInterrupt):
                await store._transaction("interrupt", interrupt)

            async def database_failure(_: object) -> None:
                """Raise one driver failure for translation assertions."""
                raise RuntimeError("database failure")

            monkeypatch.setattr(
                pgsql_durable,
                "classify_pgsql_error",
                lambda error, *, operation: SimpleNamespace(
                    category=PgsqlFailureCategory.UNIQUE_CONFLICT
                ),
            )
            with pytest.raises(DurableStoreError) as raised:
                await store._transaction("unique", database_failure)
            assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT

            monkeypatch.setattr(
                pgsql_durable,
                "classify_pgsql_error",
                lambda error, *, operation: SimpleNamespace(
                    category=PgsqlFailureCategory.UNKNOWN
                ),
            )
            with pytest.raises(DurableStoreError) as raised:
                await store._transaction("unknown", database_failure)
            assert (
                raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
            )
        finally:
            await store.aclose()

    run(_run_schema(scenario))


def test_pgsql_durable_public_cas_races_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model stale SQL CAS outcomes without weakening public invariants."""

    class Cursor:
        """Return configured SQL rows while recording closed CAS attempts."""

        def __init__(self, rows: tuple[dict[str, object] | None, ...]) -> None:
            """Bind finite SQL results for one simulated transaction."""
            self._rows = iter(rows)

        async def execute(self, statement: str, parameters: object) -> None:
            """Accept one parameterized adapter statement."""
            del statement, parameters

        async def fetchone(self) -> dict[str, object] | None:
            """Return the next configured database result."""
            return next(self._rows)

    async def noop(*_: object) -> None:
        """Model a successful lock helper without an external database."""

    async def transaction(
        operation: str,
        callback: Callable[[object], Awaitable[object]],
    ) -> object:
        """Execute one public operation against its configured CAS cursor."""
        del operation
        return await callback(cursor)

    identity = _identity("a")
    digest = _digest("a")
    reservation = DurableReservation(
        PatchRequestId("request_" + "a" * 16), identity, digest, False
    )
    plan = _plan(digest, "a", step_count=1)
    approval = _approval(identity, digest, plan, "a")
    lease = DurableCommitLease(
        reservation.request_id,
        plan.domain_id,
        _owner("a"),
        SequenceNumber(1),
        ExpiryTick(20),
    )
    store = PgsqlDurablePatchStore(
        type("Pool", (), {"connection": lambda self: None})(),
        approval_verifier=_APPROVAL_AUTHORITY,
    )
    monkeypatch.setattr(store, "_transaction", transaction)

    row = {
        "request_id": reservation.request_id.value,
        "plan_payload": pgsql_durable._encode_plan(plan),
        "lifecycle": LifecyclePhase.COMMIT_STARTED.value,
        "owner_id": lease.owner_id.value,
        "domain_id": lease.domain_id.value,
        "fence": lease.fence.value,
        "lease_expires_at": lease.expires_at.value,
        "pending_operation_id": None,
        "event_cursor": 0,
    }

    async def select_reservation(
        _: object, __: DurableReservation
    ) -> dict[str, object]:
        """Return the current row selected by a simulated request lock."""
        return row

    monkeypatch.setattr(
        pgsql_durable, "_select_reservation_for_update", select_reservation
    )

    row["plan_payload"] = None
    row["lifecycle"] = LifecyclePhase.PLANNED.value
    cursor = Cursor(())
    with pytest.raises(DurableStoreError) as raised:
        run(store.persist_plan(reservation, plan))
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

    row["plan_payload"] = pgsql_durable._encode_plan(plan)
    row["lifecycle"] = LifecyclePhase.RECEIVED.value
    row["owner_id"] = None
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.claim_commit(
                reservation,
                plan,
                approval,
                lease.owner_id,
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

    row["lifecycle"] = LifecyclePhase.PLANNED.value
    monkeypatch.setattr(pgsql_durable, "_lock_domain", noop)

    async def next_fence(_: object, __: object) -> SequenceNumber:
        """Return the next monotonic fence for a valid claim simulation."""
        return SequenceNumber(1)

    monkeypatch.setattr(pgsql_durable, "_advance_domain_fence", next_fence)
    cursor = Cursor(({"grant_id": approval.grant_id.value}, None))
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.claim_commit(
                reservation,
                plan,
                approval,
                lease.owner_id,
                ExpiryTick(10),
                DurationTicks(10),
                (),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED

    async def select_lease(
        _: object, __: DurableCommitLease, ___: ExpiryTick
    ) -> dict[str, object]:
        """Model a lease that passed its pre-CAS lock predicate."""
        return row

    monkeypatch.setattr(
        pgsql_durable, "_select_lease_for_update", select_lease
    )
    cursor = Cursor((None,))
    with pytest.raises(DurableStoreError) as raised:
        run(store.renew_lease(lease, ExpiryTick(11), DurationTicks(20)))
    assert raised.value.code is DurableStoreErrorCode.FENCED
    cursor = Cursor((row,))
    renewed = run(store.renew_lease(lease, ExpiryTick(11), DurationTicks(20)))
    assert renewed.expires_at == ExpiryTick(31)

    row.update(
        owner_id=lease.owner_id.value,
        domain_id=lease.domain_id.value,
        fence=lease.fence.value,
        lease_expires_at=lease.expires_at.value,
        lifecycle=LifecyclePhase.COMMIT_STARTED.value,
    )
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.replace_expired_owner(
                reservation,
                replace(lease, fence=SequenceNumber(2)),
                _owner("b"),
                ExpiryTick(20),
                DurationTicks(10),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.replace_expired_owner(
                reservation,
                lease,
                _owner("b"),
                ExpiryTick(19),
                DurationTicks(10),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.LEASE_EXPIRED
    row["lifecycle"] = LifecyclePhase.PLANNED.value
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.replace_expired_owner(
                reservation,
                lease,
                _owner("b"),
                ExpiryTick(20),
                DurationTicks(10),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED
    row["lifecycle"] = LifecyclePhase.COMMIT_STARTED.value
    cursor = Cursor((None,))
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.replace_expired_owner(
                reservation,
                lease,
                _owner("b"),
                ExpiryTick(20),
                DurationTicks(10),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED

    row["lifecycle"] = LifecyclePhase.PLANNED.value
    row["owner_id"] = None
    cursor = Cursor(())
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.suspend(
                lease,
                DurablePendingRequest(
                    PatchPendingOperationId("pending_" + "a" * 16),
                    _correlation("a"),
                    DurationTicks(5),
                ),
                ExpiryTick(11),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT

    row["lifecycle"] = LifecyclePhase.COMMIT_STARTED.value
    cursor = Cursor((None,))
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.suspend(
                lease,
                DurablePendingRequest(
                    PatchPendingOperationId("pending_" + "a" * 16),
                    _correlation("a"),
                    DurationTicks(5),
                ),
                ExpiryTick(11),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED

    access = DurableRequestAccess(reservation.request_id, identity)

    async def select_access(
        _: object, __: DurableRequestAccess
    ) -> dict[str, object]:
        """Return one deliberately lease-free commit-start record."""
        return row

    monkeypatch.setattr(
        pgsql_durable, "_select_access_for_update", select_access
    )
    row["owner_id"] = None
    cursor = Cursor(())
    with pytest.raises(DurableStoreError) as raised:
        run(store.request_cancellation(access))
    assert raised.value.code is DurableStoreErrorCode.FENCED

    row["owner_id"] = lease.owner_id.value
    marker = object()

    async def snapshot(_: object, __: object) -> object:
        """Return a marker after the cancellation CAS accepts its row."""
        return marker

    monkeypatch.setattr(pgsql_durable, "_require_current_domain_fence", noop)
    monkeypatch.setattr(pgsql_durable, "_snapshot", snapshot)
    cursor = Cursor((row,))
    assert run(store.request_cancellation(access)) is marker

    async def select_request(
        _: object, __: PatchRequestId
    ) -> dict[str, object]:
        """Return one current row before a simulated settle CAS race."""
        return {
            "lifecycle": LifecyclePhase.COMMIT_STARTED.value,
            "event_cursor": 0,
        }

    def current_lease(_: object, __: object, ___: object) -> None:
        """Accept the exact synthetic lease for the settle race."""

    def current_cursor(_: object, __: object) -> None:
        """Accept the exact synthetic journal cursor for the settle race."""

    def current_plan(_: object) -> object:
        """Return the expected immutable plan for the settle race."""
        return plan

    async def journal(_: object, __: object, ___: object) -> DurableJournal:
        """Return an empty journal replaced by direct state witnesses."""
        return DurableJournal(
            DurableJournalCursor(reservation.request_id, SequenceNumber(0)),
            (),
            (),
        )

    async def artifact_state(
        _: object, __: object, ___: object
    ) -> ArtifactState:
        """Return the exact artifact truth presented by the result."""
        return ArtifactState.ABSENT

    monkeypatch.setattr(
        pgsql_durable, "_select_request_for_update", select_request
    )
    monkeypatch.setattr(pgsql_durable, "_require_lease", current_lease)
    monkeypatch.setattr(pgsql_durable, "_require_cursor", current_cursor)
    monkeypatch.setattr(pgsql_durable, "_require_plan", current_plan)
    monkeypatch.setattr(pgsql_durable, "_journal", journal)
    monkeypatch.setattr(
        pgsql_durable,
        "_journal_mutation_state",
        lambda _, __: MutationState.COMMITTED,
    )
    monkeypatch.setattr(
        pgsql_durable, "_journal_artifact_state", artifact_state
    )
    monkeypatch.setattr(pgsql_durable, "_pending_from_row", lambda _: None)
    cursor = Cursor((None,))
    with pytest.raises(DurableStoreError) as raised:
        run(
            store.settle(
                lease,
                DurableJournalCursor(
                    reservation.request_id, SequenceNumber(0)
                ),
                _result(reservation.request_id, plan, MutationState.COMMITTED),
                _correlation("a"),
                ExpiryTick(11),
            )
        )
    assert raised.value.code is DurableStoreErrorCode.FENCED


def test_pgsql_durable_retention_boundaries_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject bounded retention races, expiry, and empty authorization."""

    class Cursor:
        """Return configured SQL rows for one retention CAS simulation."""

        def __init__(self, rows: tuple[dict[str, object] | None, ...]) -> None:
            """Bind the finite sequence of SQL results."""
            self._rows = iter(rows)

        async def execute(self, statement: str, parameters: object) -> None:
            """Accept one parameterized retention statement."""
            del statement, parameters

        async def fetchone(self) -> dict[str, object] | None:
            """Return the next configured SQL result."""
            return next(self._rows)

    class EmptyAuthorizer:
        """Deny every read by returning no valid audience set."""

        async def audiences_for(
            self, identity: object, kind: object
        ) -> frozenset[Audience]:
            """Return no audience after accepting the closed parameters."""
            del identity, kind
            return frozenset()

    async def scenario() -> None:
        identity = _identity("a")
        digest = _digest("a")
        reservation = DurableReservation(
            PatchRequestId("request_" + "a" * 16), identity, digest, False
        )
        retention_id = PatchRetentionRecordId("retained_" + "a" * 16)
        sealed = await _RETENTION_CIPHER.seal(
            b"retention-value",
            DurableRetentionBinding(
                reservation.request_id,
                retention_id,
                DurableRetentionKind.SEALED_PLAN,
            ),
        )
        record = DurableRetentionRecord(
            retention_id,
            DurableRetentionKind.SEALED_PLAN,
            sealed.key_id,
            sealed.value,
            DurableRetentionPolicy(ExpiryTick(20), False),
        )
        request_row: dict[str, object] = {
            "lifecycle": LifecyclePhase.PLANNED.value
        }
        identity_row: dict[str, object] = {
            "tenant_id": identity.tenant_id.value,
            "principal_id": identity.principal_id.value,
            "execution_id": identity.execution_id.value,
            "route_id": identity.route_id.value,
            "retransmission_key": identity.retransmission_key.value,
        }

        def retention_row(
            retained: DurableRetentionRecord,
            *,
            kind: DurableRetentionKind | None = None,
        ) -> dict[str, object]:
            """Encode one retention row without exposing plaintext."""
            return {
                "retention_id": retained.retention_id.value,
                "kind": (retained.kind if kind is None else kind).value,
                "key_id": retained.key_id.value,
                "ciphertext": retained.value._ciphertext,
                "ciphertext_digest": retained.value.digest().value,
                "expires_at": retained.policy.expires_at.value,
                "delete_on_terminal": retained.policy.delete_on_terminal,
            }

        cursor = Cursor(())

        async def transaction(
            operation: str,
            callback: Callable[[object], Awaitable[object]],
        ) -> object:
            """Run callbacks against the configured retention CAS cursor."""
            del operation
            return await callback(cursor)

        async def select_reservation(
            _: object, __: DurableReservation
        ) -> dict[str, object]:
            """Return one valid unlocked request for retention validation."""
            return request_row

        async def select_access(
            _: object, __: DurableRequestAccess
        ) -> dict[str, object]:
            """Return one authenticated retention request identity."""
            return identity_row

        validator = AesGcmDurableRetentionEnvelopeValidator(_RETENTION_CIPHER)
        select_access_original = pgsql_durable._select_access_for_update
        store = PgsqlDurablePatchStore(
            type("Pool", (), {"connection": lambda self: None})(),
            retention_authorizer=StaticDurableRetentionAuthorizer(
                frozenset((Audience.APPROVER,))
            ),
            retention_validator=validator,
        )
        monkeypatch.setattr(store, "_transaction", transaction)
        monkeypatch.setattr(
            pgsql_durable, "_select_reservation_for_update", select_reservation
        )
        monkeypatch.setattr(
            pgsql_durable, "_select_access_for_update", select_access
        )

        cursor = Cursor(({"record_count": 128},))
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(reservation, record)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT

        cursor = Cursor(({"record_count": 0}, {"byte_count": 4_194_304}))
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(reservation, record)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_LIMIT

        cursor = Cursor(
            (
                {"record_count": 0},
                {"byte_count": 0},
                None,
                retention_row(
                    record, kind=DurableRetentionKind.REVIEW_ARTIFACT
                ),
            )
        )
        with pytest.raises(DurableStoreError) as raised:
            await store.put_retention(reservation, record)
        assert raised.value.code is DurableStoreErrorCode.RETENTION_CONFLICT

        access = DurableRetentionAccess(
            DurableRequestAccess(reservation.request_id, identity)
        )
        cursor = Cursor((retention_row(record),))
        with pytest.raises(DurableStoreError) as raised:
            await store.get_retention(
                access, retention_id, record.policy.expires_at
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

        empty_authorizer_store = PgsqlDurablePatchStore(
            type("Pool", (), {"connection": lambda self: None})(),
            retention_authorizer=EmptyAuthorizer(),
            retention_validator=validator,
        )
        monkeypatch.setattr(
            empty_authorizer_store, "_transaction", transaction
        )
        cursor = Cursor((retention_row(record),))
        with pytest.raises(DurableStoreError) as raised:
            await empty_authorizer_store.get_retention(
                access, retention_id, ExpiryTick(10)
            )
        assert raised.value.code is DurableStoreErrorCode.RETENTION_DENIED

        async def unexpected_identity(
            _: object, __: DurableRequestAccess
        ) -> dict[str, object]:
            """Raise a non-lifecycle error that access selection preserves."""
            raise DurableStoreError(DurableStoreErrorCode.APPROVAL_MISMATCH)

        monkeypatch.setattr(
            pgsql_durable, "_select_identity_for_update", unexpected_identity
        )
        with pytest.raises(DurableStoreError) as raised:
            await getattr(select_access_original, "__call__")(
                object(), access.request
            )
        assert raised.value.code is DurableStoreErrorCode.APPROVAL_MISMATCH

    run(scenario())


def test_pgsql_store_rejects_conflicting_public_lifecycle_requests() -> None:
    """Reject invalid plan, journal, pending, and terminal transitions."""

    async def scenario(dsn: str, schema: str) -> None:
        store = await _store(dsn, schema)
        try:
            identity = _identity("f")
            digest = _digest("f")
            reservation = await store.reserve(identity, digest)
            with pytest.raises(DurableStoreError) as raised:
                await store.persist_plan(reservation, _plan(_digest("e"), "e"))
            assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
            plan = _plan(digest, "f", step_count=1)
            await store.persist_plan(reservation, plan)
            with pytest.raises(DurableStoreError) as raised:
                await store.persist_plan(reservation, _plan(digest, "e"))
            assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
            with pytest.raises(DurableStoreError) as raised:
                await store.request_cancellation(
                    DurableRequestAccess(reservation.request_id, identity)
                )
            assert (
                raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.claim_commit(
                    reservation,
                    _plan(digest, "e"),
                    _approval(identity, digest, plan, "f"),
                    _owner("f"),
                    ExpiryTick(10),
                    DurationTicks(20),
                    (),
                )
            assert raised.value.code is DurableStoreErrorCode.PLAN_MISMATCH
            approval = _approval(identity, digest, plan, "f")
            claim = await store.claim_commit(
                reservation,
                plan,
                approval,
                _owner("f"),
                ExpiryTick(10),
                DurationTicks(20),
                (),
            )
            assert claim.lease is not None
            duplicate_identity = _identity("e")
            duplicate_digest = _digest("e")
            duplicate_reservation = await store.reserve(
                duplicate_identity, duplicate_digest
            )
            duplicate_plan = _plan(duplicate_digest, "e", step_count=1)
            await store.persist_plan(duplicate_reservation, duplicate_plan)
            duplicate_approval = _APPROVAL_AUTHORITY.seal(
                replace(
                    _approval(
                        duplicate_identity,
                        duplicate_digest,
                        duplicate_plan,
                        "e",
                    ),
                    grant_id=approval.grant_id,
                )
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.claim_commit(
                    duplicate_reservation,
                    duplicate_plan,
                    duplicate_approval,
                    _owner("e"),
                    ExpiryTick(10),
                    DurationTicks(20),
                    (),
                )
            assert raised.value.code is DurableStoreErrorCode.APPROVAL_CONSUMED
            cursor = DurableJournalCursor(
                reservation.request_id, SequenceNumber(0)
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.append_step(
                    claim.lease,
                    cursor,
                    PatchStepId("step_" + "e" * 16),
                    CommitStepState.PLANNED,
                    ExpiryTick(11),
                )
            assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
            with pytest.raises(DurableStoreError) as raised:
                await store.append_artifact(
                    claim.lease,
                    cursor,
                    _artifact("f"),
                    DurableArtifactState.PRESENT,
                    ExpiryTick(11),
                )
            assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
            journal = await store.append_step(
                claim.lease,
                cursor,
                plan.steps[0].step_id,
                CommitStepState.PLANNED,
                ExpiryTick(11),
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.append_step(
                    claim.lease,
                    journal.cursor,
                    plan.steps[0].step_id,
                    CommitStepState.PLANNED,
                    ExpiryTick(11),
                )
            assert raised.value.code is DurableStoreErrorCode.JOURNAL_CONFLICT
            journal = await store.append_step(
                claim.lease,
                journal.cursor,
                plan.steps[0].step_id,
                CommitStepState.COMMITTED,
                ExpiryTick(11),
            )
            pending = DurablePendingRequest(
                PatchPendingOperationId("pending_" + "f" * 16),
                _correlation("f"),
                DurationTicks(5),
            )
            current = await store.suspend(claim.lease, pending, ExpiryTick(12))
            assert (
                await store.suspend(claim.lease, pending, ExpiryTick(12))
                == current
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.suspend(
                    claim.lease,
                    replace(
                        pending,
                        pending_operation_id=PatchPendingOperationId(
                            "pending_" + "e" * 16
                        ),
                    ),
                    ExpiryTick(12),
                )
            assert (
                raised.value.code is DurableStoreErrorCode.LIFECYCLE_CONFLICT
            )
            result = _result(
                reservation.request_id, plan, MutationState.COMMITTED
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.settle(
                    claim.lease,
                    journal.cursor,
                    replace(
                        result,
                        plan_id=PatchPlanId("plan_" + "e" * 16),
                    ),
                    pending.correlation_id,
                    ExpiryTick(13),
                )
            assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT
            with pytest.raises(DurableStoreError) as raised:
                await store.settle(
                    claim.lease,
                    journal.cursor,
                    result,
                    _correlation("e"),
                    ExpiryTick(13),
                )
            assert raised.value.code is DurableStoreErrorCode.ACCESS_DENIED
            awaiting_terminal = create_task(
                store.await_terminal(
                    DurablePendingAccess(
                        DurableRequestAccess(reservation.request_id, identity),
                        pending.pending_operation_id,
                        pending.correlation_id,
                    )
                )
            )
            await sleep(0)
            terminal = await store.settle(
                claim.lease,
                journal.cursor,
                result,
                pending.correlation_id,
                ExpiryTick(13),
            )
            assert await awaiting_terminal == terminal
            assert (
                await store.settle(
                    claim.lease,
                    journal.cursor,
                    result,
                    pending.correlation_id,
                    ExpiryTick(14),
                )
                == terminal
            )
            with pytest.raises(DurableStoreError) as raised:
                await store.settle(
                    claim.lease,
                    journal.cursor,
                    _result(
                        reservation.request_id,
                        plan,
                        MutationState.INDETERMINATE,
                    ),
                    pending.correlation_id,
                    ExpiryTick(14),
                )
            assert raised.value.code is DurableStoreErrorCode.TERMINAL_CONFLICT
            replay_claim = await store.claim_commit(
                reservation,
                plan,
                approval,
                _owner("f"),
                ExpiryTick(14),
                DurationTicks(20),
                (),
            )
            assert replay_claim.state is DurableCommitClaimState.TERMINAL
        finally:
            await store.aclose()

    run(_run_schema(scenario))
