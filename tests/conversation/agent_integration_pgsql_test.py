"""Exercise Phase 8 agent durability through real PostgreSQL state."""

from asyncio import gather, to_thread
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from os import environ
from uuid import uuid4

import httpx
import pytest
from agent_integration_contract_test import (
    _execution_segment as _contract_execution_segment,
)
from agent_integration_contract_test import (
    _internal_checkpoint as _contract_internal_checkpoint,
)
from native_openai_provider_test import (
    _binding,
    _function_call,
    _message,
    _provider,
    _reasoning,
    _response,
)
from phase2_fixtures import authority, retention
from phase2_fixtures import binding as phase_binding

import avalan
import avalan.conversation as conversation
from avalan.pgsql import (
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task.stores import (
    PgsqlTaskMigrationSettings,
    task_pgsql_upgrade,
)
from avalan.types import JsonValue

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_NOW = datetime(2026, 8, 2, 12, tzinfo=UTC)

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        _DSN is None,
        reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run durable agent integration tests on asyncio only."""
    return "asyncio"


def _key() -> conversation.ConversationDataKey:
    """Return one deterministic test-only conversation data key."""
    return conversation.ConversationDataKey(
        key_id="phase8-agent-key",
        revision=1,
        status=conversation.ConversationKeyStatus.CURRENT,
        key_bytes=b"8" * 32,
    )


def _resolver() -> conversation.InMemoryConversationKeyResolver:
    """Return exact key authority for the Phase 8 test principal."""
    scope = authority()
    return conversation.InMemoryConversationKeyResolver(
        {conversation.authority_digest(scope): (_key(),)}
    )


async def _drop_schema(dsn: str, schema: str) -> None:
    """Drop one isolated migrated PostgreSQL schema."""
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
    """Own one migrated schema and every conversation-store handle."""

    dsn: str
    schema: str
    stores: list[conversation.PgsqlConversationStore] = field(
        default_factory=list
    )

    def store(self) -> conversation.PgsqlConversationStore:
        """Return one fresh durable conversation store handle."""
        store = conversation.PgsqlConversationStore.from_settings(
            conversation.PgsqlConversationStoreSettings(
                dsn=self.dsn,
                schema=self.schema,
                pool_minimum=1,
                pool_maximum=2,
            ),
            key_resolver=_resolver(),
            cipher=conversation.AesGcmConversationCipher(),
            clock=conversation.DeterministicFakeClock(_NOW),
        )
        self.stores.append(store)
        return store


@pytest.fixture
async def pgsql_harness() -> AsyncIterator[_PgsqlHarness]:
    """Yield one real migrated Phase 8 PostgreSQL schema."""
    assert _DSN is not None
    schema = f"conv_phase8_agent_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    harness = _PgsqlHarness(dsn=_DSN, schema=schema)
    try:
        yield harness
    finally:
        for store in harness.stores:
            await store.close()
        await _drop_schema(_DSN, schema)


def _client(
    store: conversation.ConversationStore,
    provider: conversation.NativeOpenAIStatelessProvider,
    *,
    boundary_hook: conversation.CoordinatorBoundaryHook | None = None,
) -> tuple[
    avalan.DirectConversationClient,
    conversation.RunScopedConversationCoordinator,
]:
    """Return one direct client over a durable PostgreSQL coordinator."""
    scope = authority()
    coordinator = conversation.RunScopedConversationCoordinator(
        store=store,
        authority_resolver=conversation.DeterministicFakeAuthorityResolver(
            scope
        ),
        clock=conversation.DeterministicFakeClock(_NOW),
        publisher=conversation.DeterministicFakePublisher(),
        observer=conversation.DeterministicFakeObserver(),
        retry_waiter=conversation.DeterministicFakeRetryWaiter(),
        lanes=(
            conversation.NativeOpenAIConversationLaneRuntime(
                provider=provider
            ),
        ),
        boundary_hook=boundary_hook,
    )
    runtime = avalan.DirectConversationRuntime(
        coordinator=coordinator,
        store=store,
        authority=scope,
        lane=provider.binding,
        retention=retention(),
        id_namespace="phase8-pgsql-tool-boundary",
    )
    return avalan.DirectConversationClient(runtime), coordinator


async def test_pgsql_recovery_admission_is_exact_and_single_owner(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
) -> None:
    """Lease one encrypted PostgreSQL suffix across a fresh store handle."""
    record_property("conversation_acceptance_evidence", "database")
    original = _contract_execution_segment()
    recovery_binding = phase_binding("lane-recovery-admission")
    segment = replace(
        original,
        binding=recovery_binding,
        items=(replace(original.items[0], lane_id=recovery_binding.lane_id),),
    )
    template = replace(
        _contract_internal_checkpoint(segment),
        authority=authority(),
        content=conversation.MultiLaneCheckpointContent(
            visible_transcript=conversation.VisibleTranscript(entries=()),
            lanes=(),
            execution_segments=(segment,),
        ),
        lifecycle=conversation.CheckpointLifecycle.STAGED,
        timestamps=conversation.CheckpointTimestamps(
            created_at=_NOW,
            expires_at=_NOW + timedelta(hours=1),
        ),
        integrity=None,
    )
    staged = conversation.with_checkpoint_integrity(template)
    store = pgsql_harness.store()
    await store.open()
    committed = await store.commit(
        conversation.ExecutionSegmentCheckpointCandidate(checkpoint=staged)
    )
    idempotency = conversation.RequestIdempotencyIdentity(
        authority=authority(),
        operation=conversation.ConversationOperation.CREATE,
        key=segment.idempotency_key,
        request_digest=segment.request_digest,
    )
    execution = conversation.ConversationExecutionReservation(
        idempotency=idempotency,
        identity=conversation.CheckpointIdentity(
            conversation_id=committed.identity.conversation_id,
            logical_turn_id=committed.identity.logical_turn_id,
            execution_segment_id=conversation.ExecutionSegmentId(
                "pgsql-recovery-outward-segment"
            ),
            checkpoint_id=conversation.CheckpointId(
                "pgsql-recovery-outward-checkpoint"
            ),
            branch_id=conversation.ConversationBranchId(
                "pgsql-recovery-outward-branch"
            ),
            sequence=conversation.CheckpointSequence(0),
        ),
        lanes=(
            conversation.ProviderLaneExecutionReservation(
                binding=segment.binding,
                mode=segment.mode,
                scope=conversation.ProviderLaneOutputScope.CURRENT_CALL,
            ),
        ),
    )
    reservation = await store.reserve_idempotency(
        idempotency,
        execution=execution,
    )
    assert reservation.owner_token is not None
    await store.fence_idempotency(
        idempotency,
        reservation.owner_token,
        ambiguous=True,
    )
    await store.close()

    restarted = pgsql_harness.store()
    await restarted.open()
    assert committed.integrity is not None
    admission = conversation.DurableToolRecoveryAdmission(
        checkpoint_id=committed.identity.checkpoint_id,
        checkpoint_integrity=committed.integrity.digest,
        idempotency=idempotency,
        binding=segment.binding,
        action=(conversation.DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT),
        segment_count=1,
    )
    with pytest.raises(conversation.ConversationConflictError):
        await restarted.admit_tool_recovery(
            replace(
                admission,
                checkpoint_integrity=conversation.IntegrityDigest("0" * 64),
            ),
            execution,
        )
    with pytest.raises(conversation.ConversationConflictError):
        await restarted.admit_tool_recovery(
            replace(admission, segment_count=2),
            execution,
        )
    wrong_idempotency = replace(
        idempotency,
        request_digest=conversation.CanonicalRequestDigest("wrong-request"),
    )
    with pytest.raises(conversation.ConversationConflictError):
        await restarted.admit_tool_recovery(
            replace(admission, idempotency=wrong_idempotency),
            replace(execution, idempotency=wrong_idempotency),
        )
    wrong_authority = authority(principal="other-principal")
    unauthorized_idempotency = replace(
        idempotency,
        authority=wrong_authority,
    )
    with pytest.raises(conversation.ConversationAuthorizationError):
        await restarted.admit_tool_recovery(
            replace(admission, idempotency=unauthorized_idempotency),
            replace(execution, idempotency=unauthorized_idempotency),
        )

    outcomes = await gather(
        restarted.admit_tool_recovery(admission, execution),
        restarted.admit_tool_recovery(admission, execution),
        return_exceptions=True,
    )

    leases = tuple(
        outcome
        for outcome in outcomes
        if type(outcome) is conversation.DurableToolRecoveryLease
    )
    conflicts = tuple(
        outcome
        for outcome in outcomes
        if type(outcome) is conversation.ConversationConflictError
    )
    assert len(leases) == 1
    assert len(conflicts) == 1
    with pytest.raises(conversation.ConversationConflictError):
        await restarted.admit_tool_recovery(admission, execution)
    lease = leases[0]
    assert isinstance(lease, conversation.DurableToolRecoveryLease)
    settlement = await restarted.inspect_idempotency_settlement(
        idempotency,
        lease.owner_token,
    )
    assert settlement.disposition is (
        conversation.IdempotencySettlementDisposition.LEASED
    )


@pytest.mark.parametrize(
    (
        "crash_boundary",
        "effect_policy",
        "expected_action",
        "expected_effects",
        "expected_requests",
        "expected_segments",
    ),
    (
        (
            "requested_before_effect",
            conversation.ToolEffectPolicy.IDEMPOTENT,
            conversation.DurableToolRecoveryAction.REEXECUTE_IDEMPOTENT,
            0,
            1,
            1,
        ),
        (
            "effect_before_output",
            conversation.ToolEffectPolicy.FENCED_UNPROTECTED,
            conversation.DurableToolRecoveryAction.REQUIRE_RECONCILIATION,
            1,
            1,
            1,
        ),
        (
            "output_before_resume",
            conversation.ToolEffectPolicy.IDEMPOTENT,
            conversation.DurableToolRecoveryAction.RESUME_PROVIDER,
            1,
            2,
            2,
        ),
        (
            "internal_complete_before_outward",
            conversation.ToolEffectPolicy.IDEMPOTENT,
            conversation.DurableToolRecoveryAction.COMMIT_OUTWARD,
            1,
            2,
            3,
        ),
    ),
)
async def test_pgsql_tool_boundaries_recover_without_duplicate_effect(
    pgsql_harness: _PgsqlHarness,
    record_property: Callable[[str, object], None],
    crash_boundary: str,
    effect_policy: conversation.ToolEffectPolicy,
    expected_action: conversation.DurableToolRecoveryAction,
    expected_effects: int,
    expected_requests: int,
    expected_segments: int,
) -> None:
    """Recover every frozen tool boundary from durable PostgreSQL state."""
    record_property("conversation_acceptance_evidence", "database")
    effects = 0
    reconciliations = 0
    requests = 0
    tool_requests = 0

    class CommitCrashHook:
        async def reach(
            self,
            boundary: conversation.CoordinatorAwaitBoundary,
        ) -> None:
            if (
                crash_boundary == "internal_complete_before_outward"
                and boundary is conversation.CoordinatorAwaitBoundary.COMMIT
            ):
                raise conversation.ConversationCommitError()

    async def lookup(arguments: Mapping[str, JsonValue]) -> str:
        nonlocal effects, tool_requests
        assert arguments == {"value": 1}
        tool_requests += 1
        if crash_boundary == "requested_before_effect" and tool_requests == 1:
            raise RuntimeError("crash before tool effect")
        effects += 1
        if crash_boundary == "effect_before_output" and tool_requests == 1:
            raise RuntimeError("crash after tool effect")
        return "durable-tool-output"

    async def reconcile_effect(
        arguments: Mapping[str, JsonValue],
    ) -> conversation.ToolEffectReconciliation:
        nonlocal reconciliations
        assert arguments == {"value": 1}
        reconciliations += 1
        return conversation.ToolEffectReconciliation(
            applied=True,
            output="durable-tool-output",
        )

    tool = conversation.NativeOpenAIFunctionTool(
        name="lookup",
        parameters={"type": "object"},
        handler=lookup,
        effect_policy=effect_policy,
        reconciliation_handler=(
            reconcile_effect
            if effect_policy
            is conversation.ToolEffectPolicy.FENCED_UNPROTECTED
            else None
        ),
    )
    assert (tool.reconciliation_handler is not None) is (
        effect_policy is conversation.ToolEffectPolicy.FENCED_UNPROTECTED
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        await request.aread()
        if requests == 1:
            return httpx.Response(
                200,
                json=_response(
                    f"{crash_boundary}-request",
                    [
                        _reasoning(
                            f"{crash_boundary}-reasoning-1",
                            f"{crash_boundary}-private-1",
                        ),
                        _function_call(
                            f"{crash_boundary}-call",
                            f"{crash_boundary}-call-id",
                        ),
                    ],
                ),
            )
        if crash_boundary == "output_before_resume" and requests == 2:
            raise httpx.ConnectError(
                "crash before provider resume",
                request=request,
            )
        return httpx.Response(
            200,
            json=_response(
                f"{crash_boundary}-complete",
                [
                    _reasoning(
                        f"{crash_boundary}-reasoning-2",
                        f"{crash_boundary}-private-2",
                    ),
                    _message(
                        f"{crash_boundary}-message",
                        "complete but not outwardly committed",
                    ),
                ],
            ),
        )

    store = pgsql_harness.store()
    await store.open()
    provider = _provider(
        _binding(lane_id=f"lane-{crash_boundary}"),
        handler,
        tools=(tool,),
    )
    client, coordinator = _client(
        store,
        provider,
        boundary_hook=CommitCrashHook(),
    )
    settings = avalan.StatelessConversationSettings()
    request = client._root_request(
        "exercise durable tool boundary",
        settings,
        reset_parent=None,
        idempotency_key=conversation.RequestIdempotencyKey(
            f"{crash_boundary}-recovery-key"
        ),
    )
    try:
        with pytest.raises(conversation.ConversationError):
            await coordinator.execute(request)
    finally:
        await coordinator.close()

    restarted = pgsql_harness.store()
    await restarted.open()
    page = await restarted.list_checkpoints(
        authority(),
        cursor=None,
        limit=10,
    )
    recovery_checkpoint = max(
        (
            candidate
            for candidate in page.checkpoints
            if candidate.content.execution_segments
        ),
        key=lambda candidate: len(candidate.content.execution_segments),
    )
    ordered = recovery_checkpoint.content.execution_segments

    assert len(page.checkpoints) == expected_segments
    assert all(
        checkpoint.kind
        is conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
        for checkpoint in page.checkpoints
    )
    assert conversation.durable_tool_recovery_action(ordered) is (
        expected_action
    )
    assert effects == expected_effects
    assert requests == expected_requests
    restarted_provider = _provider(
        provider.binding,
        handler,
        tools=(tool,),
    )
    _, restarted_coordinator = _client(
        restarted,
        restarted_provider,
    )
    try:
        receipt = await restarted_coordinator.recover_durable_tool_execution(
            recovery_checkpoint.identity.checkpoint_id,
            authority(),
        )
        with pytest.raises(conversation.ConversationConflictError):
            await restarted_coordinator.recover_durable_tool_execution(
                recovery_checkpoint.identity.checkpoint_id,
                authority(),
            )
    finally:
        await restarted_coordinator.close()
    assert receipt.checkpoint.kind is (
        conversation.CheckpointKind.COMPLETED_OUTWARD_TURN
    )
    assert receipt.result is not None
    assert (
        conversation.durable_tool_recovery_action(
            receipt.checkpoint.content.execution_segments
        )
        is conversation.DurableToolRecoveryAction.COMMIT_OUTWARD
    )
    assert effects == 1
    assert reconciliations == (
        1 if crash_boundary == "effect_before_output" else 0
    )
    assert (
        requests
        == {
            "requested_before_effect": 2,
            "effect_before_output": 2,
            "output_before_resume": 3,
            "internal_complete_before_outward": 2,
        }[crash_boundary]
    )
