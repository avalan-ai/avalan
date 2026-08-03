"""Verify conversation and structured-input suspension commit atomically."""

from asyncio import to_thread
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from datetime import timedelta
from os import environ
from pathlib import Path
from sys import path as sys_path
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

sys_path.append(str(Path(__file__).parents[2] / "conversation"))
sys_path.append(str(Path(__file__).parents[2] / "task" / "stores"))

import interaction_pgsql_e2e as restart_support  # noqa: E402
import interaction_pgsql_store_test as durable_support  # noqa: E402
from pgsql_harness import task_pgsql_psycopg_dsn  # noqa: E402
from phase2_fixtures import authority  # noqa: E402
from store_conformance_test import _atomic_commit  # noqa: E402

import avalan.conversation as conversation  # noqa: E402
from avalan.agent import (
    durable_runtime as durable_runtime_module,  # noqa: E402
)
from avalan.agent.continuation import (  # noqa: E402
    AgentConversationContinuationResult,
    DurableAgentContinuationResumer,
    ResolvedAgentConversationContinuation,
)
from avalan.agent.orchestrator import Orchestrator  # noqa: E402
from avalan.event.manager import EventManager  # noqa: E402
from avalan.interaction import (  # noqa: E402
    ContinuationRuntimeResolver,
    DurableInteractionSuspension,
    InputRequiredResult,
    InteractionCorrelation,
    InteractionPolicy,
    PortableContinuation,
    PortableConversationCheckpointReference,
    ResolvedContinuationRuntime,
    ScopedInteractionLookup,
    StateRevision,
    bind_portable_continuation_to_conversation,
    portable_continuation_binding_digest,
)
from avalan.interaction.store import (  # noqa: E402
    CreateInteractionApplied,
    InteractionRecord,
)
from avalan.interaction.stores.pgsql import (  # noqa: E402
    PgsqlDurableTaskCoordinator,
)
from avalan.model.capability import (  # noqa: E402
    CorrelatedCapabilityResult,
    TaskInputCapabilityCall,
)
from avalan.pgsql import (  # noqa: E402
    PgsqlDatabase,
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
    quote_pgsql_identifier,
)
from avalan.task import (  # noqa: E402
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRunPolicy,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskTargetRunner,
    TaskTargetType,
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
    TaskWorkerProcessResult,
    completed_task_target_outcome,
    suspended_task_target_outcome,
)
from avalan.task.context import TaskDurableResumeHandle  # noqa: E402
from avalan.task.durable_agent import TaskDurableAgentRuntime  # noqa: E402
from avalan.task.queues import PgsqlTaskQueue  # noqa: E402
from avalan.task.resume import TaskDurableResumeCoordinator  # noqa: E402
from avalan.task.stores import (  # noqa: E402
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)

_DSN = environ.get("AVALAN_TASK_TEST_POSTGRESQL_DSN")
_NOW = durable_support._NOW
_QUEUE = "conversation-atomic-suspension"

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        _DSN is None,
        reason="AVALAN_TASK_TEST_POSTGRESQL_DSN is not set",
    ),
]


@pytest.fixture
def anyio_backend() -> str:
    """Run the shared transaction proof on asyncio only."""
    return "asyncio"


@dataclass(slots=True)
class _Harness:
    dsn: str
    schema: str
    database: PsycopgAsyncDatabase


async def _drop_schema(dsn: str, schema: str) -> None:
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(dsn=task_pgsql_psycopg_dsn(dsn))
    )
    async with database:
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "DROP SCHEMA IF EXISTS "
                    f"{quote_pgsql_identifier(schema)} CASCADE"
                )


@pytest.fixture
async def atomic_harness() -> AsyncIterator[_Harness]:
    """Yield one migrated schema and one shared production database."""
    assert _DSN is not None
    schema = f"conv_atomic_{uuid4().hex}"
    await to_thread(
        task_pgsql_upgrade,
        PgsqlTaskMigrationSettings(url=_DSN, schema=schema),
    )
    database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(
            dsn=task_pgsql_psycopg_dsn(_DSN),
            schema=schema,
            pool_minimum=1,
            pool_maximum=4,
            application_name="avalan-conversation-atomic-test",
        )
    )
    await database.open()
    try:
        yield _Harness(dsn=_DSN, schema=schema, database=database)
    finally:
        await database.aclose()
        await _drop_schema(_DSN, schema)


class _FailAfterConversationCommit:
    """Inject a downstream failure after conversation SQL is staged."""

    def __init__(
        self,
        unit: conversation.PgsqlConversationUnitOfWork,
    ) -> None:
        self.unit = unit
        self.commit_calls = 0
        self.settle_calls = 0

    @property
    def database(self) -> PgsqlDatabase:
        return self.unit.database

    @property
    def checkpoint_id(self) -> str:
        return self.unit.checkpoint_id

    @property
    def execution_segment_id(self) -> str:
        return self.unit.execution_segment_id

    @property
    def continuation_id(self) -> str | None:
        return self.unit.continuation_id

    @property
    def continuation_state_revision(self) -> int | None:
        return self.unit.continuation_state_revision

    async def commit_in(self, unit: PgsqlUnitOfWork) -> object:
        self.commit_calls += 1
        await self.unit.commit_in(unit)
        raise RuntimeError("injected post-conversation suspension failure")

    def settle_committed(self) -> None:
        self.settle_calls += 1
        self.unit.settle_committed()


class _AtomicSuspensionTarget:
    """Stage one bound conversation checkpoint with durable input state."""

    def __init__(
        self,
        store: conversation.PgsqlConversationStore,
        *,
        suffix: str,
        fail_after_conversation: bool,
    ) -> None:
        self.store = store
        self.suffix = suffix
        self.fail_after_conversation = fail_after_conversation
        self.checkpoint: conversation.ConversationCheckpoint | None = None
        self.reference: conversation.PortableContinuationReference | None = (
            None
        )
        self.unit: conversation.PgsqlConversationUnitOfWork | None = None
        self.failure: _FailAfterConversationCommit | None = None
        self.suspension: DurableInteractionSuspension | None = None

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        del definition, context
        return ()

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        request = durable_support._request(context.execution.run_id)
        base = _atomic_commit(self.suffix).candidate.checkpoint
        parent = await self.store.commit(
            conversation.ExecutionSegmentCheckpointCandidate(
                checkpoint=conversation.with_checkpoint_integrity(
                    replace(
                        base,
                        kind=(
                            conversation.CheckpointKind.INTERNAL_PROVIDER_BOUNDARY
                        ),
                    )
                )
            )
        )
        checkpoint = conversation.with_checkpoint_integrity(
            replace(
                base,
                identity=replace(
                    base.identity,
                    checkpoint_id=conversation.CheckpointId(
                        f"checkpoint-{self.suffix}-suspension"
                    ),
                    execution_segment_id=conversation.ExecutionSegmentId(
                        f"segment-{self.suffix}-suspension"
                    ),
                    sequence=conversation.CheckpointSequence(1),
                    parent_checkpoint_id=parent.identity.checkpoint_id,
                    parent_sequence=parent.identity.sequence,
                ),
                kind=conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
                integrity=None,
            )
        )
        interaction_reference = PortableConversationCheckpointReference(
            checkpoint_id=str(checkpoint.identity.checkpoint_id),
            execution_segment_id=str(checkpoint.identity.execution_segment_id),
        )
        portable = durable_support._portable(request)
        portable = bind_portable_continuation_to_conversation(
            replace(
                portable,
                provider_snapshot=None,
                state_revision=StateRevision(int(request.state_revision) + 1),
            ),
            interaction_reference,
        )
        reference = conversation.PortableContinuationReference(
            continuation_id=portable.continuation_id,
            state_revision=portable.state_revision,
            digest=conversation.ContinuationDigest(
                portable_continuation_binding_digest(portable)
            ),
            definition=portable.definition,
            revision_binding=portable.revision_binding,
        )
        unit = await self.store.stage(
            conversation.SuspensionCheckpointCandidate(
                checkpoint=checkpoint,
                continuation=reference,
            )
        )
        participant: conversation.PgsqlConversationUnitOfWork | (
            _FailAfterConversationCommit
        ) = unit
        if self.fail_after_conversation:
            self.failure = _FailAfterConversationCommit(unit)
            participant = self.failure
        self.checkpoint = checkpoint
        self.reference = reference
        self.unit = unit
        self.suspension = DurableInteractionSuspension(
            command=durable_support._create_command(request),
            continuation=portable,
        )
        return suspended_task_target_outcome(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=True,
            ),
            checkpoint_id=str(checkpoint.identity.checkpoint_id),
            durable=self.suspension,
            conversation_unit=participant,
        )


class _AtomicResumeTarget(TaskTargetRunner):
    """Complete only through one admitted fresh-worker durable dispatch."""

    def __init__(self) -> None:
        self.resume_calls = 0

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        del definition, context
        return ()

    def supports_durable_resume(self, target_type: TaskTargetType) -> bool:
        """Accept only the agent target reconstructed from durable state."""
        return target_type is TaskTargetType.AGENT

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        del context
        raise AssertionError("fresh worker must not restart initial work")

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        assert context.durable_resume is durable_resume
        self.resume_calls += 1
        return completed_task_target_outcome(await durable_resume.dispatch())


def _definition(suffix: str) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name=f"conversation_atomic_{suffix}", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued(_QUEUE),
    )


async def _run_worker(
    harness: _Harness,
    *,
    suffix: str,
    fail_after_conversation: bool,
) -> tuple[
    _AtomicSuspensionTarget,
    TaskWorkerProcessResult,
]:
    database = harness.database
    task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
    queue = PgsqlTaskQueue(database, clock=lambda: _NOW)
    interaction_store = await durable_support._store(database)
    resolver = conversation.InMemoryConversationKeyResolver(
        {
            conversation.authority_digest(authority()): (
                conversation.ConversationDataKey(
                    key_id="conversation-atomic-key",
                    revision=1,
                    status=conversation.ConversationKeyStatus.CURRENT,
                    key_bytes=b"a" * 32,
                ),
            )
        }
    )
    conversation_store = conversation.PgsqlConversationStore(
        database,
        key_resolver=resolver,
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=False,
    )
    await conversation_store.open()
    definition_id = f"conversation-atomic-definition-{suffix}"
    await task_store.register_definition(
        _definition(suffix),
        definition_hash=definition_id,
    )
    await queue.enqueue_run(
        TaskExecutionRequest(
            definition_id=definition_id,
            queue=_QUEUE,
        ),
        queue_name=_QUEUE,
    )
    target = _AtomicSuspensionTarget(
        conversation_store,
        suffix=suffix,
        fail_after_conversation=fail_after_conversation,
    )
    result = await TaskWorker(
        task_store,
        queue,
        target=target,
        worker_id=f"conversation-atomic-worker-{suffix}",
        queue_name=_QUEUE,
        durable_suspension_coordinator=PgsqlDurableTaskCoordinator(
            interaction_store,
            task_store,
        ),
        clock=lambda: _NOW,
    ).process_once()
    return target, result


async def _atomic_counts(database: PsycopgAsyncDatabase) -> dict[str, object]:
    async with database.connection() as connection:
        async with connection.cursor() as cursor:
            await cursor.execute("""
                SELECT
                    (SELECT COUNT(*) FROM "conversation_checkpoints")::BIGINT
                        AS "checkpoint_count",
                    (
                        SELECT COUNT(*)
                        FROM "conversation_checkpoint_payload_refs"
                    )::BIGINT AS "payload_ref_count",
                    (
                        SELECT COUNT(*)
                        FROM "conversation_checkpoint_continuations"
                    )::BIGINT AS "conversation_continuation_count",
                    (SELECT COUNT(*) FROM "interaction_records")::BIGINT
                        AS "interaction_count",
                    (SELECT COUNT(*) FROM "interaction_continuations")::BIGINT
                        AS "interaction_continuation_count",
                    (
                        SELECT COUNT(*) FROM "task_queue_items"
                        WHERE "state" = 'suspended'
                    )::BIGINT AS "suspended_queue_count",
                    (
                        SELECT COUNT(*) FROM "task_runs"
                        WHERE "state" = 'input_required'
                    )::BIGINT AS "input_required_run_count"
                """)
            row = await cursor.fetchone()
    assert row is not None
    return dict(row)


async def test_atomic_suspension_rolls_back_every_durable_surface(
    atomic_harness: _Harness,
    record_property: Callable[[str, object], None],
) -> None:
    """Roll back conversation SQL when downstream suspension fails."""
    record_property("conversation_acceptance_evidence", "database")
    target, result = await _run_worker(
        atomic_harness,
        suffix="rollback",
        fail_after_conversation=True,
    )
    assert getattr(result, "suspension") is None
    assert target.failure is not None
    assert target.failure.commit_calls == 1
    assert target.failure.settle_calls == 0
    assert await _atomic_counts(atomic_harness.database) == {
        "checkpoint_count": 1,
        "payload_ref_count": 1,
        "conversation_continuation_count": 0,
        "interaction_count": 0,
        "interaction_continuation_count": 0,
        "suspended_queue_count": 0,
        "input_required_run_count": 0,
    }


async def test_atomic_suspension_commits_every_durable_surface(
    atomic_harness: _Harness,
    record_property: Callable[[str, object], None],
) -> None:
    """Commit exact conversation, interaction, and task suspension state."""
    record_property("conversation_acceptance_evidence", "database")
    target, result = await _run_worker(
        atomic_harness,
        suffix="success",
        fail_after_conversation=False,
    )
    assert getattr(result, "suspension") is not None
    assert target.checkpoint is not None
    assert target.reference is not None
    assert target.unit is not None
    assert await _atomic_counts(atomic_harness.database) == {
        "checkpoint_count": 2,
        "payload_ref_count": 3,
        "conversation_continuation_count": 1,
        "interaction_count": 1,
        "interaction_continuation_count": 1,
        "suspended_queue_count": 1,
        "input_required_run_count": 1,
    }

    resolver = conversation.InMemoryConversationKeyResolver(
        {
            conversation.authority_digest(authority()): (
                conversation.ConversationDataKey(
                    key_id="conversation-atomic-key",
                    revision=1,
                    status=conversation.ConversationKeyStatus.CURRENT,
                    key_bytes=b"a" * 32,
                ),
            )
        }
    )
    restarted = conversation.PgsqlConversationStore(
        cast(PgsqlDatabase, atomic_harness.database),
        key_resolver=resolver,
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=False,
    )
    await restarted.open()
    restored = await restarted.load(
        target.checkpoint.identity.checkpoint_id,
        authority(),
    )
    restored_reference = await restarted.load_continuation_reference(
        target.checkpoint.identity.checkpoint_id,
        authority(),
    )
    assert restored == conversation.with_checkpoint_integrity(
        replace(
            target.checkpoint,
            lifecycle=conversation.CheckpointLifecycle.COMMITTED,
            timestamps=replace(
                target.checkpoint.timestamps,
                committed_at=target.checkpoint.timestamps.created_at,
            ),
        )
    )
    assert restored_reference == target.reference
    with pytest.raises(conversation.ConversationStorageError):
        await target.unit.commit()


async def test_fresh_worker_applies_atomic_conversation_answer_once(
    atomic_harness: _Harness,
    record_property: Callable[[str, object], None],
) -> None:
    """Resume one reference-only suspension in a fresh durable worker."""
    record_property("conversation_acceptance_evidence", "database")
    target, suspended = await _run_worker(
        atomic_harness,
        suffix="fresh-worker",
        fail_after_conversation=False,
    )
    assert suspended.suspension is not None
    assert target.suspension is not None
    assert target.checkpoint is not None
    suspension_checkpoint = target.checkpoint
    portable = target.suspension.continuation
    assert portable.version == 2
    assert portable.provider_snapshot is None
    assert portable.conversation_checkpoint_reference is not None
    run_id = suspended.suspension.run.run_id

    restarted_database = PsycopgAsyncDatabase(
        PsycopgPoolSettings(
            dsn=task_pgsql_psycopg_dsn(atomic_harness.dsn),
            schema=atomic_harness.schema,
            pool_minimum=1,
            pool_maximum=4,
            application_name="avalan-conversation-resume-test",
        )
    )
    await restarted_database.open()
    restarted_interaction = await durable_support._store(restarted_database)
    restarted_task = PgsqlTaskStore(
        restarted_database,
        clock=lambda: _NOW + timedelta(seconds=3),
    )
    restarted_queue = PgsqlTaskQueue(
        restarted_database,
        clock=lambda: _NOW + timedelta(seconds=3),
    )
    resolver = conversation.InMemoryConversationKeyResolver(
        {
            conversation.authority_digest(authority()): (
                conversation.ConversationDataKey(
                    key_id="conversation-atomic-key",
                    revision=1,
                    status=conversation.ConversationKeyStatus.CURRENT,
                    key_bytes=b"a" * 32,
                ),
            )
        }
    )
    restarted_conversation = conversation.PgsqlConversationStore(
        restarted_database,
        key_resolver=resolver,
        cipher=conversation.AesGcmConversationCipher(),
        owns_database=False,
    )
    await restarted_conversation.open()
    try:
        command = target.suspension.command
        record = await restarted_interaction.lookup_scoped(
            ScopedInteractionLookup(
                actor=command.actor,
                correlation=InteractionCorrelation.from_request(
                    command.request
                ),
            )
        )
        assert isinstance(record, InteractionRecord)
        created = CreateInteractionApplied(
            command=command,
            record=record,
            policy=InteractionPolicy(),
        )
        answer = durable_support._answer(
            created,
            key="phase8-conversation-answer",
        )
        task_coordinator = PgsqlDurableTaskCoordinator(
            restarted_interaction,
            restarted_task,
        )
        resolved = await task_coordinator.resolve_and_requeue(
            answer,
            task_run_id=run_id,
            now=_NOW + timedelta(seconds=1),
        )
        replayed = await task_coordinator.resolve_and_requeue(
            answer,
            task_run_id=run_id,
            now=_NOW + timedelta(seconds=1),
        )
        resolved_record = getattr(resolved.resolution, "record", None)
        replayed_record = getattr(replayed.resolution, "record", None)
        assert isinstance(resolved_record, InteractionRecord)
        assert isinstance(replayed_record, InteractionRecord)
        assert resolved_record == replayed_record
        assert resolved.reentry == replayed.reentry

        applications = 0

        async def resolve_conversation(
            continuation: PortableContinuation,
            expected_digest: str,
        ) -> ResolvedAgentConversationContinuation:
            nonlocal applications
            assert continuation.provider_snapshot is None
            assert continuation.conversation_checkpoint_reference == (
                portable.conversation_checkpoint_reference
            )
            checkpoint = await restarted_conversation.load(
                suspension_checkpoint.identity.checkpoint_id,
                authority(),
            )
            reference = (
                await restarted_conversation.load_continuation_reference(
                    checkpoint.identity.checkpoint_id,
                    authority(),
                )
            )
            assert reference.continuation_id == continuation.continuation_id
            assert int(reference.state_revision) + 1 == int(
                continuation.state_revision
            ), (reference.state_revision, continuation.state_revision)
            assert str(reference.digest) == expected_digest, (
                str(reference.digest),
                expected_digest,
            )
            assert reference.definition == continuation.definition
            assert reference.revision_binding == (
                continuation.revision_binding
            )

            async def apply_result(
                call: TaskInputCapabilityCall,
                result: CorrelatedCapabilityResult,
            ) -> AgentConversationContinuationResult:
                nonlocal applications
                assert call is not None and result is not None
                applications += 1
                identity = checkpoint.identity
                continued = conversation.with_checkpoint_integrity(
                    replace(
                        checkpoint,
                        identity=conversation.CheckpointIdentity(
                            conversation_id=identity.conversation_id,
                            logical_turn_id=identity.logical_turn_id,
                            execution_segment_id=(
                                conversation.ExecutionSegmentId(
                                    "fresh-worker-resumed-segment"
                                )
                            ),
                            checkpoint_id=conversation.CheckpointId(
                                "fresh-worker-resumed-checkpoint"
                            ),
                            branch_id=identity.branch_id,
                            sequence=conversation.CheckpointSequence(
                                identity.sequence + 1
                            ),
                            parent_checkpoint_id=identity.checkpoint_id,
                            parent_sequence=identity.sequence,
                        ),
                        kind=(
                            conversation.CheckpointKind.COMPLETED_OUTWARD_TURN
                        ),
                        lifecycle=conversation.CheckpointLifecycle.STAGED,
                        timestamps=replace(
                            checkpoint.timestamps,
                            created_at=(
                                checkpoint.timestamps.created_at
                                + timedelta(seconds=1)
                            ),
                            committed_at=None,
                        ),
                    )
                )
                committed = await restarted_conversation.commit(
                    conversation.SuspensionContinuationCheckpointCandidate(
                        checkpoint=continued,
                        public_response_id=conversation.PublicResponseId(
                            "fresh-worker-resumed-response"
                        ),
                        suspension_checkpoint_id=(
                            checkpoint.identity.checkpoint_id
                        ),
                    )
                )
                return AgentConversationContinuationResult(
                    checkpoint=committed,
                    output="resumed after one structured answer",
                )

            return ResolvedAgentConversationContinuation(
                checkpoint=checkpoint,
                continuation_reference=reference,
                apply_result=apply_result,
            )

        durable_record = (
            await restarted_interaction.get_task_continuation_record(run_id)
        )
        continuation = durable_record.continuation
        stack = AsyncExitStack()
        await stack.__aenter__()
        orchestrator = MagicMock(spec=Orchestrator)
        orchestrator.event_manager = EventManager()
        stack.push_async_callback(orchestrator.event_manager.aclose)
        executor = durable_runtime_module.TrustedAgentContinuationExecutor(
            orchestrator,
            stager=(
                durable_runtime_module.PortableAgentContinuationStager(
                    clock=lambda: _NOW + timedelta(seconds=3)
                )
            ),
            ownership=(
                durable_runtime_module._TrustedContinuationRuntimeOwnership(
                    stack
                )
            ),
        )
        runtime = ResolvedContinuationRuntime(
            definition=continuation.definition,
            revision_binding=continuation.revision_binding,
            runtime=executor,
            operation=object(),
            model=object(),
            tools=object(),
            capabilities=restart_support._catalog(
                continuation.revision_binding
            ),
            credentials_reloaded_from_trusted_config=True,
        )
        loader = restart_support._ResumeLoader(runtime)

        class _FreshWorkerConversationCoordinator:
            async def resume_structured_input(
                self,
                checkpoint: conversation.ConversationCheckpoint,
                call: TaskInputCapabilityCall,
                result: CorrelatedCapabilityResult,
            ) -> AgentConversationContinuationResult:
                resolved_state = await resolve_conversation(
                    continuation,
                    portable_continuation_binding_digest(continuation),
                )
                assert resolved_state.checkpoint == checkpoint
                return await resolved_state.apply_result(call, result)

        conversation_runtime = TaskDurableAgentRuntime(
            store=restarted_conversation,
            coordinator=_FreshWorkerConversationCoordinator(),
            authority=authority(),
        )
        resumer = DurableAgentContinuationResumer(
            restarted_interaction,
            ContinuationRuntimeResolver(
                loader,
                clock=lambda: _NOW + timedelta(seconds=3),
            ),
            conversation_resolver=conversation_runtime.resolver(),
            clock=lambda: _NOW + timedelta(seconds=3),
        )
        resume_coordinator = TaskDurableResumeCoordinator(
            restarted_interaction,
            resumer,
        )
        resume_target = _AtomicResumeTarget()
        resumed = await TaskWorker(
            restarted_task,
            restarted_queue,
            target=resume_target,
            worker_id="conversation-atomic-fresh-worker",
            queue_name=_QUEUE,
            durable_suspension_coordinator=task_coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: _NOW + timedelta(seconds=3),
        ).process_once()

        assert resumed.completion is not None
        assert resumed.output == "resumed after one structured answer"
        assert resume_target.resume_calls == 1
        assert applications == 1
        assert loader.calls == 1
        child = await restarted_conversation.load(
            conversation.CheckpointId("fresh-worker-resumed-checkpoint"),
            authority(),
        )
        assert child.identity.logical_turn_id == (
            suspension_checkpoint.identity.logical_turn_id
        )
        assert child.identity.parent_checkpoint_id == (
            suspension_checkpoint.identity.checkpoint_id
        )
        assert child.identity.sequence == (
            suspension_checkpoint.identity.sequence + 1
        )
        assert (
            await restarted_queue.claim(
                _QUEUE,
                worker_id="no-duplicate-worker",
                lease_expires_at=_NOW + timedelta(seconds=10),
                now=_NOW + timedelta(seconds=4),
            )
            is None
        )
        assert applications == 1
    finally:
        await restarted_interaction.aclose()
        await restarted_database.aclose()
