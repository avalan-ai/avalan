"""Exercise the durable conversation store against real PostgreSQL."""

from asyncio import CancelledError, gather, to_thread
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field, replace
from datetime import timedelta
from os import environ
from typing import cast
from uuid import uuid4

import pytest
from durable_codec_test import _continuation_reference
from phase2_fixtures import NOW, authority
from psycopg.errors import CheckViolation
from store_conformance_test import (
    _atomic_commit,
    _child_candidate,
    _claimed_record,
    _execution_reservation,
    _execution_stage,
    _outbox_target,
    _prepare_atomic,
    _seed,
    _stored_atomic_commit,
)

import avalan.conversation as conversation
from avalan.conversation.store import StoreBoundaryHook
from avalan.pgsql import (
    PgsqlConnection,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        _DSN is None,
        reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run durable conformance on asyncio only."""
    return "asyncio"


def _key(
    key_id: str = "conversation-key-1",
    *,
    revision: int = 1,
    status: conversation.ConversationKeyStatus = (
        conversation.ConversationKeyStatus.CURRENT
    ),
    material: bytes = b"1" * 32,
) -> conversation.ConversationDataKey:
    return conversation.ConversationDataKey(
        key_id=key_id,
        revision=revision,
        status=status,
        key_bytes=material,
    )


def _resolver(
    *scopes: conversation.AuthorityScope,
) -> conversation.InMemoryConversationKeyResolver:
    selected = scopes or (authority(),)
    return conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): (_key(),) for scope in selected}
    )


def _limit_policy(**changes: int) -> conversation.PgsqlConversationStorePolicy:
    limits = conversation.StoreLimits()
    return conversation.PgsqlConversationStorePolicy(
        limits=replace(limits, **changes)
    )


def _assert_single_limit_result(results: list[object]) -> None:
    successes = tuple(
        result for result in results if not isinstance(result, BaseException)
    )
    failures = tuple(
        result for result in results if isinstance(result, BaseException)
    )
    assert len(successes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], conversation.ConversationLimitError)


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(PsycopgPoolSettings(dsn=dsn))
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


@dataclass(slots=True)
class _PgsqlHarness:
    dsn: str
    schema: str
    resolver: conversation.InMemoryConversationKeyResolver
    clock: conversation.DeterministicFakeClock
    stores: list[conversation.PgsqlConversationStore] = field(
        default_factory=list
    )

    def store(
        self,
        *,
        resolver: conversation.ConversationKeyResolver | None = None,
        clock: conversation.ConversationClock | None = None,
        policy: conversation.PgsqlConversationStorePolicy | None = None,
        boundary_hook: StoreBoundaryHook | None = None,
        fault_hook: conversation.PgsqlConversationFaultHook | None = None,
    ) -> conversation.PgsqlConversationStore:
        """Return one closed store over this isolated migrated schema."""
        store = conversation.PgsqlConversationStore.from_settings(
            conversation.PgsqlConversationStoreSettings(
                dsn=self.dsn,
                schema=self.schema,
                pool_minimum=1,
                pool_maximum=2,
            ),
            key_resolver=resolver or self.resolver,
            cipher=conversation.AesGcmConversationCipher(),
            policy=policy or conversation.PgsqlConversationStorePolicy(),
            clock=clock or self.clock,
            boundary_hook=boundary_hook,
            fault_hook=fault_hook,
        )
        self.stores.append(store)
        return store


class _FailOnceCloseDatabase:
    def __init__(self, database: PsycopgAsyncDatabase) -> None:
        self.database = database
        self.close_calls = 0

    def connection(self) -> AbstractAsyncContextManager[PgsqlConnection]:
        return cast(
            AbstractAsyncContextManager[PgsqlConnection],
            self.database.connection(),
        )

    async def open(self) -> None:
        await self.database.open()

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_calls == 1:
            raise RuntimeError("injected pool close failure")
        await self.database.aclose()


@pytest.fixture
async def pgsql_harness() -> AsyncIterator[_PgsqlHarness]:
    """Yield one real isolated schema and close every owned pool."""
    assert _DSN is not None
    schema = f"conv_p3_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    harness = _PgsqlHarness(
        dsn=_DSN,
        schema=schema,
        resolver=_resolver(),
        clock=conversation.DeterministicFakeClock(NOW),
    )
    try:
        yield harness
    finally:
        for store in harness.stores:
            await store.close()
        await _drop_schema(_DSN, schema)


async def test_pgsql_matches_core_store_lifecycle(
    pgsql_harness: _PgsqlHarness,
) -> None:
    """Match create, stage, heads, paging, deletion, and retention."""
    store = pgsql_harness.store()
    await store.open()
    scope, _engine, root, first_result = await _seed(
        cast(conversation.ConversationStore, store),
        suffix="pgsql-core",
    )
    checkpoint_id = root.checkpoint.identity.checkpoint_id
    assert await store.load(checkpoint_id, scope) == root.checkpoint
    assert await store.authorize(checkpoint_id, scope) == root.checkpoint

    candidate = _child_candidate(
        root.checkpoint,
        first_result,
        suffix="pgsql-staged-child",
    )
    rolled_back = await store.stage(candidate)
    async with rolled_back:
        pass
    with pytest.raises(conversation.ConversationStorageError):
        await rolled_back.__aenter__()

    committed_unit = await store.stage(candidate)
    async with committed_unit:
        child = await committed_unit.commit()
    with pytest.raises(conversation.ConversationStorageError):
        await committed_unit.commit()
    assert child.identity.parent_checkpoint_id == checkpoint_id
    assert await store.branch_count(checkpoint_id, scope) == 1

    root_head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("pgsql-main"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=checkpoint_id,
    )
    child_head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("pgsql-child"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=child.identity.checkpoint_id,
    )
    await store.create_head(root_head, scope)
    await store.create_head(child_head, scope)
    assert await store.load_head(root_head.head_id, scope) == root_head
    page = await store.list_checkpoints(scope, cursor=None, limit=1)
    assert len(page.checkpoints) == 1
    assert page.next_cursor is not None
    remainder = await store.list_checkpoints(
        scope,
        cursor=page.next_cursor,
        limit=10,
    )
    assert len(remainder.checkpoints) == 1
    assert remainder.next_cursor is None

    assert root.result is not None
    public_response_id = root.result.public_response_id
    assert await store.retrieve(public_response_id, scope) == root.result
    tombstone = await store.tombstone(
        public_response_id,
        scope,
        NOW + timedelta(minutes=1),
    )
    assert tombstone.lifecycle is conversation.CheckpointLifecycle.TOMBSTONED
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.tombstone(
            public_response_id,
            scope,
            NOW + timedelta(minutes=2),
        )
    await store.delete(
        public_response_id,
        scope,
        NOW + timedelta(minutes=2),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(checkpoint_id, scope)
    assert await store.load(child.identity.checkpoint_id, scope) == child

    collected = await store.garbage_collect(limit=100)
    assert collected.deleted_payloads > 0
    first_sweep = await store.sweep(NOW + timedelta(hours=2), limit=1)
    second_sweep = await store.sweep(NOW + timedelta(hours=2), limit=1)
    assert first_sweep.expired == 1
    assert second_sweep.deleted == 1
    assert (await store.garbage_collect(limit=100)).deleted_payloads > 0

    assert (
        await store.inspect_close()
    ).disposition is conversation.StoreCloseDisposition.OPEN
    await store.close()
    with pytest.raises(conversation.ConversationStorageError):
        await store.load(child.identity.checkpoint_id, scope)


async def test_pgsql_payload_references_reject_identity_reassignment(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Reject cross-checkpoint, cross-authority, and cross-lane swaps."""
    record_property("conversation_acceptance_evidence", "security")
    second_scope = authority("principal-other", tenant="tenant-other")
    resolver = _resolver(authority(), second_scope)
    store = pgsql_harness.store(resolver=resolver)
    await store.open()
    first_commit = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, store),
        _atomic_commit("payload-owner-first"),
    )
    second_commit = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, store),
        _atomic_commit("payload-owner-second", scope=second_scope),
    )
    first = await store.commit_atomic(first_commit)
    second = await store.commit_atomic(second_commit)
    first_id = first.checkpoint.identity.checkpoint_id
    second_id = second.checkpoint.identity.checkpoint_id
    second_authority = str(conversation.authority_digest(second_scope))
    first_lane = str(first_commit.output_candidates[0].lane_id)

    async with store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT "payload_id"
                FROM "conversation_checkpoint_payload_refs"
                WHERE "checkpoint_id" = %s AND "payload_kind" = 'checkpoint'
                """,
                (first_id,),
            )
            first_row = await cursor.fetchone()
            await cursor.execute(
                """
                SELECT "payload_id"
                FROM "conversation_checkpoint_payload_refs"
                WHERE "checkpoint_id" = %s AND "payload_kind" = 'checkpoint'
                """,
                (second_id,),
            )
            second_row = await cursor.fetchone()
            assert first_row is not None
            assert second_row is not None
            first_payload_id = first_row["payload_id"]
            second_payload_id = second_row["payload_id"]

            for assignment, value in (
                ('"payload_id" = %s', second_payload_id),
                ('"authority_digest" = %s', second_authority),
                ('"lane_id" = %s', first_lane),
            ):
                with pytest.raises(CheckViolation):
                    await cursor.execute(
                        'UPDATE "conversation_checkpoint_payload_refs" '
                        f"SET {assignment} "
                        'WHERE "checkpoint_id" = %s '
                        "AND \"payload_kind\" = 'checkpoint'",
                        (value, first_id),
                    )

            await cursor.execute(
                """
                SELECT "payload_id", "reference_count"
                FROM "conversation_encrypted_payloads"
                WHERE "payload_id" IN (%s, %s)
                ORDER BY "payload_id"
                """,
                (first_payload_id, second_payload_id),
            )
            refcounts = await cursor.fetchall()
            assert len(refcounts) == 2
            assert {row["reference_count"] for row in refcounts} == {1}

            await cursor.execute(
                'ALTER TABLE "conversation_checkpoint_payload_refs" '
                "DISABLE TRIGGER ALL"
            )
            await cursor.execute(
                """
                UPDATE "conversation_checkpoint_payload_refs"
                SET "authority_digest" = %s
                WHERE "checkpoint_id" = %s AND "payload_kind" = 'checkpoint'
                """,
                (second_authority, first_id),
            )
            await cursor.execute(
                'ALTER TABLE "conversation_checkpoint_payload_refs" '
                "ENABLE TRIGGER ALL"
            )

    with pytest.raises(conversation.ConversationStorageError):
        await store.load(first_id, authority())


async def test_pgsql_write_key_generation_rejects_stale_writer(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep a newer current generation when a stale process writes."""
    record_property("conversation_acceptance_evidence", "security")
    scope = authority()
    digest = conversation.authority_digest(scope)
    original = _key()
    initial_resolver = conversation.InMemoryConversationKeyResolver(
        {digest: (original,)}
    )
    initial_store = pgsql_harness.store(resolver=initial_resolver)
    await initial_store.open()
    first = await initial_store.create(
        _atomic_commit("key-generation-first").candidate
    )

    grace = replace(original, status=conversation.ConversationKeyStatus.GRACE)
    current = _key(
        "conversation-key-2",
        revision=2,
        material=b"2" * 32,
    )
    current_resolver = conversation.InMemoryConversationKeyResolver(
        {digest: (grace, current)}
    )
    current_store = pgsql_harness.store(resolver=current_resolver)
    await current_store.open()
    second = await current_store.create(
        _atomic_commit("key-generation-second").candidate
    )

    stale_resolver = conversation.InMemoryConversationKeyResolver(
        {digest: (original,)}
    )
    stale_store = pgsql_harness.store(resolver=stale_resolver)
    await stale_store.open()
    stale_candidate = _atomic_commit("key-generation-stale").candidate
    with pytest.raises(conversation.ConversationKeyPolicyError):
        await stale_store.create(stale_candidate)

    assert (
        await current_store.load(first.identity.checkpoint_id, scope) == first
    )
    assert (
        await current_store.load(second.identity.checkpoint_id, scope)
        == second
    )
    async with current_store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT "current_generation", "current_key_id",
                       "current_key_revision"
                FROM "conversation_key_authorities"
                WHERE "authority_digest" = %s
                """,
                (str(digest),),
            )
            authority_row = await cursor.fetchone()
            await cursor.execute(
                """
                SELECT COUNT(*)::BIGINT AS "record_count"
                FROM "conversation_encrypted_payloads"
                WHERE "checkpoint_id" = %s
                """,
                (stale_candidate.checkpoint.identity.checkpoint_id,),
            )
            failed_row = await cursor.fetchone()
    assert authority_row == {
        "current_generation": 2,
        "current_key_id": "conversation-key-2",
        "current_key_revision": 2,
    }
    assert failed_row is not None
    assert failed_row["record_count"] == 0


async def test_pgsql_atomic_idempotency_and_outbox_recovery(
    pgsql_harness: _PgsqlHarness,
) -> None:
    """Fence execution, commit once, and recover publication exactly once."""
    store = pgsql_harness.store()
    await store.open()
    base = _atomic_commit("pgsql-outbox")
    prepared = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, store),
        base,
    )
    reservation = _execution_reservation(base)
    leased = await store.inspect_idempotency_settlement(
        prepared.idempotency,
        prepared.owner_token,
    )
    assert leased.disposition is (
        conversation.IdempotencySettlementDisposition.LEASED
    )
    assert (
        await store.reserve_idempotency(
            base.idempotency,
            execution=reservation,
        )
    ).disposition is conversation.IdempotencyDisposition.FENCED
    conflicting = replace(
        base.idempotency,
        request_digest=conversation.CanonicalRequestDigest("different"),
    )
    assert (
        await store.reserve_idempotency(conflicting)
    ).disposition is conversation.IdempotencyDisposition.CONFLICT

    receipt = await store.commit_atomic(prepared)
    assert receipt.outbox is not None
    replay = await store.reserve_idempotency(
        base.idempotency,
        execution=reservation,
    )
    assert replay.disposition is (
        conversation.IdempotencyDisposition.REPLAY_COMMITTED
    )
    assert replay.checkpoint_id == receipt.checkpoint.identity.checkpoint_id

    target = _outbox_target(receipt.outbox)
    missing = replace(target, intent_id="missing-intent")
    assert (await store.claim_outbox(missing)).disposition is (
        conversation.OutboxClaimDisposition.NOT_FOUND_OR_UNAUTHORIZED
    )
    claimed = _claimed_record(await store.claim_outbox(target))
    assert claimed.lease_owner is not None
    assert (await store.claim_outbox(target)).disposition is (
        conversation.OutboxClaimDisposition.ACTIVELY_LEASED
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.release_outbox(target, "wrong-owner")
    await store.release_outbox(target, claimed.lease_owner)
    reclaimed = _claimed_record(await store.claim_outbox(target))
    assert reclaimed.lease_owner is not None
    await store.acknowledge_outbox(target, reclaimed.lease_owner)
    await store.acknowledge_outbox(target, reclaimed.lease_owner)
    await store.release_outbox(target, reclaimed.lease_owner)
    assert (await store.claim_outbox(target)).disposition is (
        conversation.OutboxClaimDisposition.ALREADY_PUBLISHED
    )

    second = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, store),
        _atomic_commit("pgsql-recovery"),
    )
    second_receipt = await store.commit_atomic(second)
    assert second_receipt.outbox is not None
    worker = store.create_outbox_recovery_worker(authority())
    batch = await worker.claim(limit=10)
    assert batch.disposition is conversation.OutboxRecoveryDisposition.CLAIMED
    assert len(batch.records) == 1
    await worker.release(batch.records[0])
    replay_batch = await worker.claim(limit=10)
    assert len(replay_batch.records) == 1
    await worker.acknowledge(replay_batch.records[0])
    assert (await worker.claim(limit=10)).disposition is (
        conversation.OutboxRecoveryDisposition.EMPTY
    )

    pruned = await store.prune(NOW + timedelta(hours=1), limit=10)
    assert pruned.outbox_records == 2
    assert pruned.idempotency_records == 0


async def test_pgsql_known_no_dispatch_retries_and_cleanup_converges(
    pgsql_harness: _PgsqlHarness,
) -> None:
    """Retry known-safe failures while fencing ambiguous ownership."""
    store = pgsql_harness.store()
    await store.open()
    identity = _atomic_commit("pgsql-idempotency").idempotency
    first = await store.reserve_idempotency(identity)
    assert first.owner_token is not None
    await store.fence_idempotency(
        identity,
        first.owner_token,
        ambiguous=False,
    )
    retried = await store.reserve_idempotency(identity)
    assert retried.disposition is conversation.IdempotencyDisposition.EXECUTE
    assert retried.owner_token is not None
    await store.rollback_attempt(retried.owner_token)
    settled = await store.abandon_idempotency(
        identity,
        retried.owner_token,
        ambiguous=False,
    )
    assert settled.disposition is (
        conversation.IdempotencySettlementDisposition.SETTLED
    )
    assert (
        await store.inspect_idempotency_settlement(
            identity,
            retried.owner_token,
        )
    ).disposition is conversation.IdempotencySettlementDisposition.SETTLED
    assert (
        await store.reconcile_idempotency(
            identity,
            retried.owner_token,
            ambiguous=False,
        )
    ).disposition is conversation.IdempotencySettlementDisposition.SETTLED

    ambiguous = await store.reserve_idempotency(identity)
    assert ambiguous.owner_token is not None
    await store.abandon_idempotency(
        identity,
        ambiguous.owner_token,
        ambiguous=True,
    )
    assert (
        await store.reserve_idempotency(identity)
    ).disposition is conversation.IdempotencyDisposition.FENCED
    with pytest.raises(conversation.ConversationConflictError):
        await store.reconcile_idempotency(
            identity,
            "wrong-owner",
            ambiguous=True,
        )
    await store.reconcile_idempotency(
        identity,
        ambiguous.owner_token,
        ambiguous=True,
    )


async def test_pgsql_ambiguous_dispatch_reconciliation_survives_restart_race(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Resolve one durable fence without moving an unrelated named head."""
    record_property("conversation_acceptance_evidence", "database")
    store = pgsql_harness.store()
    await store.open()
    ordinary = _atomic_commit("pgsql-reserved-quarantine-prefix").candidate
    assert type(ordinary) is conversation.OutwardTurnCheckpointCandidate
    spoofed_checkpoint = conversation.with_checkpoint_integrity(
        replace(
            ordinary.checkpoint,
            identity=replace(
                ordinary.checkpoint.identity,
                checkpoint_id=conversation.CheckpointId(
                    "quarantine-reserved-prefix-spoof"
                ),
            ),
            integrity=None,
        )
    )
    with pytest.raises(conversation.ConversationValidationError):
        await store.create(
            conversation.OutwardTurnCheckpointCandidate(
                checkpoint=spoofed_checkpoint,
                public_response_id=ordinary.public_response_id,
            )
        )
    root = await store.create(_atomic_commit("pgsql-ambiguity-head").candidate)
    head_id = conversation.NamedHeadId("pgsql-ambiguity-head")
    await store.create_head(
        conversation.NamedHeadSnapshot(
            head_id=head_id,
            revision=conversation.NamedHeadRevision(0),
            checkpoint_id=root.identity.checkpoint_id,
        ),
        authority(),
    )
    head_before = await store.load_head(head_id, authority())
    identity = _atomic_commit("pgsql-ambiguity-restart").idempotency
    with pytest.raises(conversation.ConversationValidationError):
        await store.reconcile_ambiguous_dispatch(
            cast(
                conversation.AmbiguousDispatchReconciliationRequest,
                object(),
            )
        )
    reservation = await store.reserve_idempotency(identity)
    assert reservation.owner_token is not None
    in_progress_request = conversation.AmbiguousDispatchReconciliationRequest(
        authority=authority(),
        operation=identity.operation,
        idempotency_key=identity.key,
        resolution=conversation.AmbiguousDispatchResolution.RETAIN_FENCE,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await store.reconcile_ambiguous_dispatch(in_progress_request)
    await store.abandon_idempotency(
        identity,
        reservation.owner_token,
        ambiguous=True,
    )
    await store.close()

    restarted = pgsql_harness.store()
    await restarted.open()
    assert (await restarted.reserve_idempotency(identity)).disposition is (
        conversation.IdempotencyDisposition.FENCED
    )
    request_value = conversation.AmbiguousDispatchReconciliationRequest(
        authority=authority(),
        operation=identity.operation,
        idempotency_key=identity.key,
        resolution=conversation.AmbiguousDispatchResolution.RETAIN_FENCE,
    )
    retained = await restarted.reconcile_ambiguous_dispatch(request_value)
    assert retained.disposition is (
        conversation.AmbiguousDispatchReconciliationDisposition.FENCE_RETAINED
    )
    assert (await restarted.reserve_idempotency(identity)).disposition is (
        conversation.IdempotencyDisposition.FENCED
    )

    confirmation = replace(
        request_value,
        resolution=(
            conversation.AmbiguousDispatchResolution.CONFIRMED_NO_DISPATCH
        ),
    )
    raced = await gather(
        restarted.reconcile_ambiguous_dispatch(confirmation),
        restarted.reconcile_ambiguous_dispatch(confirmation),
    )
    assert {result.disposition for result in raced} == {
        conversation.AmbiguousDispatchReconciliationDisposition.RESOLVED_NO_DISPATCH,
        conversation.AmbiguousDispatchReconciliationDisposition.ALREADY_RESOLVED_NO_DISPATCH,
    }
    repeated = await restarted.reconcile_ambiguous_dispatch(confirmation)
    assert repeated.disposition is (
        conversation.AmbiguousDispatchReconciliationDisposition.ALREADY_RESOLVED_NO_DISPATCH
    )
    concealed = await restarted.reconcile_ambiguous_dispatch(
        replace(
            confirmation,
            authority=replace(authority(), principal_id="wrong-principal"),
        )
    )
    assert concealed.disposition is (
        conversation.AmbiguousDispatchReconciliationDisposition.NOT_FOUND_OR_UNAUTHORIZED
    )
    assert await restarted.load_head(head_id, authority()) == head_before
    retry = await restarted.reserve_idempotency(identity)
    assert retry.disposition is conversation.IdempotencyDisposition.EXECUTE
    assert retry.owner_token is not None
    await restarted.abandon_idempotency(
        identity,
        retry.owner_token,
        ambiguous=False,
    )


async def test_pgsql_checkpoint_capacity_is_global_under_race(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Permit only one concurrent checkpoint at the global maximum."""
    record_property("conversation_acceptance_evidence", "database")
    policy = _limit_policy(max_checkpoints=1)
    first_store = pgsql_harness.store(policy=policy)
    second_store = pgsql_harness.store(policy=policy)
    await first_store.open()
    await second_store.open()
    results = list(
        await gather(
            first_store.create(
                _atomic_commit("capacity-checkpoint-a").candidate
            ),
            second_store.create(
                _atomic_commit("capacity-checkpoint-b").candidate
            ),
            return_exceptions=True,
        )
    )
    assert len(results) == 2
    _assert_single_limit_result(results)
    async with first_store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                'SELECT COUNT(*)::BIGINT AS "record_count" '
                'FROM "conversation_checkpoints"'
            )
            row = await cursor.fetchone()
    assert row == {"record_count": 1}


async def test_pgsql_idempotency_capacity_is_global_under_race(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Permit only one concurrent in-flight idempotency reservation."""
    record_property("conversation_acceptance_evidence", "database")
    policy = _limit_policy(max_idempotency_records=1, max_in_flight=1)
    first_store = pgsql_harness.store(policy=policy)
    second_store = pgsql_harness.store(policy=policy)
    await first_store.open()
    await second_store.open()
    first = _atomic_commit("capacity-idempotency-a")
    second = _atomic_commit("capacity-idempotency-b")
    results = list(
        await gather(
            first_store.reserve_idempotency(first.idempotency),
            second_store.reserve_idempotency(second.idempotency),
            return_exceptions=True,
        )
    )
    assert len(results) == 2
    _assert_single_limit_result(results)
    async with first_store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                'SELECT COUNT(*)::BIGINT AS "record_count" '
                'FROM "conversation_idempotency"'
            )
            row = await cursor.fetchone()
    assert row == {"record_count": 1}


async def test_pgsql_execution_staging_capacity_is_global_under_race(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Permit only one concurrent durable execution staging record."""
    record_property("conversation_acceptance_evidence", "database")
    policy = _limit_policy(max_staged_execution_records=1)
    first_store = pgsql_harness.store(policy=policy)
    second_store = pgsql_harness.store(policy=policy)
    await first_store.open()
    await second_store.open()
    first = _atomic_commit("capacity-staging-a")
    second = _atomic_commit("capacity-staging-b")
    first_reservation = await first_store.reserve_idempotency(
        first.idempotency,
        execution=_execution_reservation(first),
    )
    second_reservation = await second_store.reserve_idempotency(
        second.idempotency,
        execution=_execution_reservation(second),
    )
    assert first_reservation.owner_token is not None
    assert second_reservation.owner_token is not None
    results = list(
        await gather(
            first_store.stage_execution(
                _execution_stage(
                    first,
                    first.output_candidates[0],
                    first_reservation.owner_token,
                )
            ),
            second_store.stage_execution(
                _execution_stage(
                    second,
                    second.output_candidates[0],
                    second_reservation.owner_token,
                )
            ),
            return_exceptions=True,
        )
    )
    assert len(results) == 2
    _assert_single_limit_result(results)
    async with first_store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                'SELECT COUNT(*)::BIGINT AS "record_count" '
                'FROM "conversation_execution_staging"'
            )
            row = await cursor.fetchone()
    assert row == {"record_count": 1}


async def test_pgsql_head_capacity_is_global_under_race(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Permit only one concurrent named head at the global maximum."""
    record_property("conversation_acceptance_evidence", "database")
    policy = _limit_policy(max_heads=1)
    first_store = pgsql_harness.store(policy=policy)
    second_store = pgsql_harness.store(policy=policy)
    await first_store.open()
    await second_store.open()
    first_checkpoint = await first_store.create(
        _atomic_commit("capacity-head-checkpoint-a").candidate
    )
    second_checkpoint = await first_store.create(
        _atomic_commit("capacity-head-checkpoint-b").candidate
    )
    first_head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("capacity-head-a"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=first_checkpoint.identity.checkpoint_id,
    )
    second_head = conversation.NamedHeadSnapshot(
        head_id=conversation.NamedHeadId("capacity-head-b"),
        revision=conversation.NamedHeadRevision(0),
        checkpoint_id=second_checkpoint.identity.checkpoint_id,
    )
    results = list(
        await gather(
            first_store.create_head(first_head, authority()),
            second_store.create_head(second_head, authority()),
            return_exceptions=True,
        )
    )
    assert len(results) == 2
    _assert_single_limit_result(results)
    async with first_store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                'SELECT COUNT(*)::BIGINT AS "record_count" '
                'FROM "conversation_named_heads"'
            )
            row = await cursor.fetchone()
    assert row == {"record_count": 1}


async def test_pgsql_outbox_capacity_is_global_under_race(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Roll back the losing checkpoint when concurrent outbox is full."""
    record_property("conversation_acceptance_evidence", "database")
    policy = _limit_policy(max_outbox_records=1)
    first_store = pgsql_harness.store(policy=policy)
    second_store = pgsql_harness.store(policy=policy)
    await first_store.open()
    await second_store.open()
    first = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, first_store),
        _atomic_commit("capacity-outbox-a"),
    )
    second = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, second_store),
        _atomic_commit("capacity-outbox-b"),
    )
    results = list(
        await gather(
            first_store.commit_atomic(first),
            second_store.commit_atomic(second),
            return_exceptions=True,
        )
    )
    assert len(results) == 2
    _assert_single_limit_result(results)
    async with first_store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM "conversation_outbox")::BIGINT
                        AS "outbox_count",
                    (SELECT COUNT(*) FROM "conversation_checkpoints")::BIGINT
                        AS "checkpoint_count"
                """)
            row = await cursor.fetchone()
    assert row == {"outbox_count": 1, "checkpoint_count": 1}


async def test_pgsql_stored_tombstone_reconciliation_and_gc(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Retain upstream deletion work until it succeeds, then collect bytes."""
    record_property("conversation_acceptance_evidence", "database")
    store = pgsql_harness.store()
    await store.open()
    commit = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, store),
        _stored_atomic_commit("pgsql-stored-delete"),
    )
    receipt = await store.commit_atomic(commit)
    assert commit.public_response_id is not None
    expected_target = commit.output_candidates[0].upstream_response_id
    assert expected_target is not None
    restored = await store.retrieve(commit.public_response_id, authority())
    assert isinstance(restored.handle, conversation.StoredConversationHandle)
    await store.tombstone(
        commit.public_response_id,
        authority(),
        NOW + timedelta(minutes=1),
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(commit.public_response_id, authority())
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(
            receipt.checkpoint.identity.checkpoint_id,
            authority(),
        )

    first = await store.claim_reconciliation(authority(), limit=10)
    assert len(first) == 1
    assert first[0].upstream_response_id == expected_target
    assert str(expected_target) not in repr(first[0])
    restarted = pgsql_harness.store()
    await restarted.open()
    assert await restarted.claim_reconciliation(authority(), limit=10) == ()
    pgsql_harness.clock.set(NOW + timedelta(seconds=31))
    reclaimed = await restarted.claim_reconciliation(authority(), limit=10)
    assert len(reclaimed) == 1
    assert reclaimed[0].reconciliation_id == first[0].reconciliation_id
    assert reclaimed[0].upstream_response_id == expected_target
    assert reclaimed[0].lease_owner != first[0].lease_owner
    await restarted.acknowledge_reconciliation(
        reclaimed[0],
        succeeded=False,
    )
    with pytest.raises(conversation.ConversationTransitionError):
        await store.delete(
            commit.public_response_id,
            authority(),
            NOW + timedelta(minutes=2),
        )

    await store.garbage_collect(limit=100)
    async with store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT p."reference_count"
                FROM "conversation_checkpoint_payload_refs" AS r
                JOIN "conversation_encrypted_payloads" AS p
                  ON p."payload_id" = r."payload_id"
                WHERE r."checkpoint_id" = %s
                  AND r."payload_kind" = 'deletion_target'
                """,
                (receipt.checkpoint.identity.checkpoint_id,),
            )
            retained = await cursor.fetchone()
    assert retained == {"reference_count": 1}

    retried = await store.claim_reconciliation(authority(), limit=10)
    assert len(retried) == 1
    assert retried[0].upstream_response_id == expected_target
    with pytest.raises(conversation.ConversationConflictError):
        await store.acknowledge_reconciliation(
            replace(retried[0], lease_owner="wrong-owner"),
            succeeded=True,
        )
    await store.acknowledge_reconciliation(retried[0], succeeded=True)
    assert await store.claim_reconciliation(authority(), limit=10) == ()
    await store.delete(
        commit.public_response_id,
        authority(),
        NOW + timedelta(minutes=2),
    )
    assert (await store.garbage_collect(limit=100)).deleted_payloads >= 3
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(
            receipt.checkpoint.identity.checkpoint_id, authority()
        )


async def test_pgsql_stored_expiry_waits_for_upstream_reconciliation(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep expired stored bytes until upstream deletion is acknowledged."""
    record_property("conversation_acceptance_evidence", "database")
    store = pgsql_harness.store()
    await store.open()
    commit = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, store),
        _stored_atomic_commit("pgsql-stored-expiry"),
    )
    receipt = await store.commit_atomic(commit)
    assert commit.public_response_id is not None
    expected_target = commit.output_candidates[0].upstream_response_id
    assert expected_target is not None

    first_sweep = await store.sweep(NOW + timedelta(hours=2), limit=10)
    assert first_sweep.expired == 1
    assert first_sweep.deleted == 0
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.retrieve(commit.public_response_id, authority())
    second_sweep = await store.sweep(
        NOW + timedelta(hours=2, seconds=1),
        limit=10,
    )
    assert second_sweep.deleted == 0
    await store.garbage_collect(limit=100)

    work = await store.claim_reconciliation(authority(), limit=10)
    assert len(work) == 1
    assert work[0].upstream_response_id == expected_target
    async with store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                SELECT p."reference_count"
                FROM "conversation_checkpoint_payload_refs" AS r
                JOIN "conversation_encrypted_payloads" AS p
                  ON p."payload_id" = r."payload_id"
                WHERE r."checkpoint_id" = %s
                  AND r."payload_kind" = 'deletion_target'
                """,
                (receipt.checkpoint.identity.checkpoint_id,),
            )
            retained = await cursor.fetchone()
    assert retained == {"reference_count": 1}

    await store.acknowledge_reconciliation(work[0], succeeded=True)
    final_sweep = await store.sweep(
        NOW + timedelta(hours=2, seconds=2),
        limit=10,
    )
    assert final_sweep.deleted == 1
    assert (await store.garbage_collect(limit=100)).deleted_payloads >= 3


async def test_pgsql_continuation_reference_is_encrypted_and_normalized(
    pgsql_harness: _PgsqlHarness,
) -> None:
    """Restore an exact suspension reference and reject normalized drift."""
    store = pgsql_harness.store()
    await store.open()
    base = _atomic_commit("pgsql-suspension").candidate.checkpoint
    staged = conversation.with_checkpoint_integrity(
        replace(
            base,
            kind=conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
        )
    )
    reference = _continuation_reference()
    candidate = conversation.SuspensionCheckpointCandidate(
        checkpoint=staged,
        continuation=reference,
    )
    committed = await store.create(candidate)
    assert (
        await store.load_continuation_reference(
            committed.identity.checkpoint_id,
            authority(),
        )
        == reference
    )

    async with store.database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute(
                """
                UPDATE conversation_checkpoint_continuations
                SET definition_digest = %s
                WHERE checkpoint_id = %s
                """,
                ("0" * 64, committed.identity.checkpoint_id),
            )
    with pytest.raises(conversation.ConversationStorageError):
        await store.load_continuation_reference(
            committed.identity.checkpoint_id,
            authority(),
        )


@dataclass(slots=True)
class _FaultHook:
    fail_boundary: conversation.PgsqlConversationFaultBoundary | None = None
    fail_operation: str | None = None
    cancellation: bool = False
    points: list[conversation.PgsqlConversationFaultPoint] = field(
        default_factory=list
    )

    async def reach(
        self,
        point: conversation.PgsqlConversationFaultPoint,
    ) -> None:
        self.points.append(point)
        if (
            point.boundary is self.fail_boundary
            and point.operation == self.fail_operation
        ):
            if self.cancellation:
                raise CancelledError
            raise RuntimeError("injected durable boundary failure")


async def test_pgsql_faults_rollback_or_recover_committed_result(
    pgsql_harness: _PgsqlHarness,
) -> None:
    """Distinguish pre-commit rollback from post-commit recovery."""
    before = _FaultHook(
        fail_boundary=conversation.PgsqlConversationFaultBoundary.SQL_AFTER,
        fail_operation="checkpoint_insert",
    )
    rollback_store = pgsql_harness.store(fault_hook=before)
    await rollback_store.open()
    rollback_candidate = _atomic_commit("pgsql-before-commit").candidate
    with pytest.raises(conversation.ConversationStorageError):
        await rollback_store.create(rollback_candidate)
    rollback_store._fault_hook = _FaultHook()
    with pytest.raises(conversation.ConversationAuthorizationError):
        await rollback_store.load(
            rollback_candidate.checkpoint.identity.checkpoint_id,
            authority(),
        )

    after = _FaultHook(
        fail_boundary=conversation.PgsqlConversationFaultBoundary.COMMIT_AFTER,
        fail_operation="checkpoint_create",
    )
    committed_store = pgsql_harness.store(fault_hook=after)
    await committed_store.open()
    committed_candidate = _atomic_commit("pgsql-after-commit").candidate
    with pytest.raises(conversation.ConversationStorageError):
        await committed_store.create(committed_candidate)
    committed_store._fault_hook = _FaultHook()
    restored = await committed_store.load(
        committed_candidate.checkpoint.identity.checkpoint_id,
        authority(),
    )
    assert restored.identity == committed_candidate.checkpoint.identity

    recording = _FaultHook()
    atomic_store = pgsql_harness.store(fault_hook=recording)
    await atomic_store.open()
    atomic = await _prepare_atomic(
        cast(conversation.InMemoryConversationStore, atomic_store),
        _atomic_commit("pgsql-fault-record"),
    )
    await atomic_store.commit_atomic(atomic)
    observed = {point.boundary for point in recording.points}
    assert observed == set(conversation.PgsqlConversationFaultBoundary)


async def test_pgsql_cancellation_before_commit_leaves_no_checkpoint(
    pgsql_harness: _PgsqlHarness,
) -> None:
    """Propagate cancellation and return the transaction connection cleanly."""
    hook = _FaultHook(
        fail_boundary=conversation.PgsqlConversationFaultBoundary.COMMIT_BEFORE,
        fail_operation="checkpoint_create",
        cancellation=True,
    )
    store = pgsql_harness.store(fault_hook=hook)
    await store.open()
    candidate = _atomic_commit("pgsql-cancel").candidate
    with pytest.raises(CancelledError):
        await store.create(candidate)
    store._fault_hook = _FaultHook()
    with pytest.raises(conversation.ConversationAuthorizationError):
        await store.load(
            candidate.checkpoint.identity.checkpoint_id,
            authority(),
        )
    assert (await store.garbage_collect(limit=10)).deleted_payloads == 0


async def test_pgsql_real_close_failure_remains_open_and_retriable(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Keep a real owned PostgreSQL store usable after close failure."""
    record_property("conversation_acceptance_evidence", "database")
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(
            dsn=pgsql_harness.dsn,
            schema=pgsql_harness.schema,
            application_name="avalan-conversation-close-retry",
        )
    )
    failing = _FailOnceCloseDatabase(database)
    store = conversation.PgsqlConversationStore(
        failing,
        key_resolver=pgsql_harness.resolver,
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=True,
    )
    pgsql_harness.stores.append(store)
    await store.open()
    with pytest.raises(RuntimeError, match="pool close failure"):
        await store.close()
    assert (
        await store.inspect_close()
    ).disposition is conversation.StoreCloseDisposition.OPEN
    assert (await store.readiness(authority())).schema_version == 1
    assert (await store.close()).disposition is (
        conversation.StoreCloseDisposition.CLOSED
    )
    assert failing.close_calls == 2
