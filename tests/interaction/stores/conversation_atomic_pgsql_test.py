"""Verify conversation and structured-input suspension commit atomically."""

from asyncio import to_thread
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, replace
from os import environ
from pathlib import Path
from sys import path as sys_path
from typing import cast
from uuid import uuid4

import pytest

sys_path.append(str(Path(__file__).parents[2] / "conversation"))
sys_path.append(str(Path(__file__).parents[2] / "task" / "stores"))

import interaction_pgsql_store_test as durable_support  # noqa: E402
from pgsql_harness import task_pgsql_psycopg_dsn  # noqa: E402
from phase2_fixtures import authority  # noqa: E402
from store_conformance_test import _atomic_commit  # noqa: E402

import avalan.conversation as conversation  # noqa: E402
from avalan.interaction import (  # noqa: E402
    DurableInteractionSuspension,
    InputRequiredResult,
    PortableConversationCheckpointReference,
    bind_portable_continuation_to_conversation,
    portable_continuation_digest,
)
from avalan.interaction.stores.pgsql import (  # noqa: E402
    PgsqlDurableTaskCoordinator,
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
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
    suspended_task_target_outcome,
)
from avalan.task.queues import PgsqlTaskQueue  # noqa: E402
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
        checkpoint = conversation.with_checkpoint_integrity(
            replace(
                base,
                kind=conversation.CheckpointKind.STRUCTURED_INPUT_SUSPENSION,
            )
        )
        interaction_reference = PortableConversationCheckpointReference(
            checkpoint_id=str(checkpoint.identity.checkpoint_id),
            execution_segment_id=str(checkpoint.identity.execution_segment_id),
        )
        portable = bind_portable_continuation_to_conversation(
            durable_support._portable(request),
            interaction_reference,
        )
        reference = conversation.PortableContinuationReference(
            continuation_id=portable.continuation_id,
            state_revision=portable.state_revision,
            digest=conversation.ContinuationDigest(
                portable_continuation_digest(portable)
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
        return suspended_task_target_outcome(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=True,
            ),
            checkpoint_id=str(checkpoint.identity.checkpoint_id),
            durable=DurableInteractionSuspension(
                command=durable_support._create_command(request),
                continuation=portable,
            ),
            conversation_unit=participant,
        )


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
    object,
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
        "checkpoint_count": 0,
        "payload_ref_count": 0,
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
        "checkpoint_count": 1,
        "payload_ref_count": 2,
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
