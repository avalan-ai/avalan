"""Run durable task recovery through real PostgreSQL worker processes."""

from asyncio import Event, create_task, gather, wait_for
from collections.abc import Mapping
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from sys import path as sys_path
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

sys_path.append(str(Path(__file__).parents[2] / "task"))
sys_path.append(str(Path(__file__).parents[2] / "task" / "stores"))

import interaction_pgsql_store_test as durable_support  # noqa: E402
from pgsql_harness import (  # noqa: E402
    drop_task_pgsql_schema,
    isolated_task_pgsql_schema,
    real_task_pgsql_dsn,
    task_pgsql_psycopg_dsn,
)

from avalan.agent.continuation import (  # noqa: E402
    AgentContinuationEventListener,
    AgentContinuationEventListenerRegistration,
    AgentContinuationResumeCommand,
    DurableAgentContinuationResumer,
)
from avalan.interaction import (  # noqa: E402
    AnswerProvenance,
    BranchId,
    ContinuationClaimOwnerId,
    ContinuationCompletionCommand,
    ContinuationRevisionBinding,
    ContinuationRuntimeResolver,
    ContinuationSnapshot,
    DurableInteractionSuspension,
    ExecutionDefinitionRef,
    InputRequest,
    InputRequiredResult,
    InteractionBranchRegistration,
    InteractionBranchRootLookup,
    InteractionCorrelation,
    InteractionExecutionScope,
    InteractionNotFoundError,
    InteractionPolicy,
    ListInteractionsCommand,
    PortableContinuation,
    PrincipalScope,
    RegisterInteractionBranchCommand,
    RequestState,
    ResolvedContinuationRuntime,
    RunId,
    ScopedInteractionLookup,
    TerminalizeDueInteractionsCommand,
    UserId,
    WaitForInteractionChangeCommand,
)
from avalan.interaction.codec import (  # noqa: E402
    decode_continuation_snapshot,
    encode_continuation_snapshot,
)
from avalan.interaction.continuation import (  # noqa: E402
    derive_continuation_dispatch_id,
    derive_provider_idempotency_key,
)
from avalan.interaction.store import (  # noqa: E402
    CancelInteractionCommand,
    CreateInteractionApplied,
    DueInteractionsApplied,
    InteractionRecord,
    InteractionStoreReplayed,
    ResolveInteractionApplied,
    ResolveInteractionRejected,
    SupersedeInteractionScopeCommand,
    TerminalizeInteractionScopeCommand,
)
from avalan.interaction.stores.pgsql import (  # noqa: E402
    PgsqlDurableTaskCoordinator,
    PgsqlInteractionStore,
    PgsqlInteractionStoreError,
    PgsqlInteractionStorePolicy,
    PgsqlResumptionReconciler,
)
from avalan.model.capability import (  # noqa: E402
    ContinuationSnapshotCodecRegistry,
    ModelCapabilityCatalog,
    ProviderCapabilitySupport,
)
from avalan.pgsql import (  # noqa: E402
    PgsqlUnitOfWork,
    PsycopgAsyncDatabase,
    PsycopgPoolSettings,
)
from avalan.task import (  # noqa: E402
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskClient,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskInputContract,
    TaskInteractionEventType,
    TaskMetadata,
    TaskOutputContract,
    TaskQueueItemState,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetOutcome,
    TaskTargetRunner,
    TaskTargetType,
    TaskValidationContext,
    TaskValidationIssue,
    TaskWorker,
    completed_task_target_outcome,
    suspended_task_target_outcome,
)
from avalan.task.context import TaskDurableResumeHandle  # noqa: E402
from avalan.task.queues import PgsqlTaskQueue  # noqa: E402
from avalan.task.resume import TaskDurableResumeCoordinator  # noqa: E402
from avalan.task.settlement import (  # noqa: E402
    TaskDurableResumeFailure,
)
from avalan.task.stores import (  # noqa: E402
    PgsqlTaskMigrationSettings,
    PgsqlTaskStore,
    task_pgsql_upgrade,
)

_NOW = durable_support._NOW
_QUEUE = "durable-worker-e2e"
_CHECKPOINT = "durable-worker-checkpoint"


class _DurableWorkerTarget(TaskTargetRunner):
    """Suspend initial work and resume only through a durable handle."""

    def __init__(
        self,
        checkpoint_id: str = _CHECKPOINT,
        *,
        branch_id: BranchId | None = None,
        parent_branch_id: BranchId | None = None,
    ) -> None:
        assert (branch_id is None) == (parent_branch_id is None)
        self.initial_calls = 0
        self.resume_calls = 0
        self.suspension: DurableInteractionSuspension | None = None
        self.checkpoint_id = checkpoint_id
        self.branch_id = branch_id
        self.parent_branch_id = parent_branch_id

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        del definition, context
        return ()

    def supports_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        return target_type is TaskTargetType.AGENT

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        self.initial_calls += 1
        request = durable_support._request(context.execution.run_id)
        if self.branch_id is not None:
            assert self.parent_branch_id is not None
            request = replace(
                request,
                origin=replace(
                    request.origin,
                    branch_id=self.branch_id,
                    parent_branch_id=self.parent_branch_id,
                ),
            )
        continuation = durable_support._portable(request)
        snapshot = continuation.provider_snapshot
        assert isinstance(snapshot, ContinuationSnapshot)
        dispatch_id = derive_continuation_dispatch_id(request.continuation_id)
        continuation = replace(
            continuation,
            provider_snapshot=replace(
                snapshot,
                provider_idempotency_key=derive_provider_idempotency_key(
                    request.continuation_id,
                    dispatch_id,
                ),
            ),
        )
        self.suspension = DurableInteractionSuspension(
            command=durable_support._create_command(request),
            continuation=continuation,
        )
        return suspended_task_target_outcome(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=True,
            ),
            checkpoint_id=self.checkpoint_id,
            durable=self.suspension,
        )

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        assert context.durable_resume is durable_resume
        self.resume_calls += 1
        return completed_task_target_outcome(await durable_resume.dispatch())


class _ResumeAdapter:
    """Validate and restore the exact provider replay capsule."""

    def __init__(self) -> None:
        self.imported: list[ContinuationSnapshot] = []

    def validate_continuation_snapshot_call(
        self,
        snapshot: ContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
        expected_provider_name: str,
        expected_arguments: Mapping[str, object],
    ) -> None:
        assert snapshot.revision_binding == expected_binding
        assert (
            snapshot.payload["reserved_capability_call_id"]
            == provider_call_correlation_id
        )
        replay_items = snapshot.payload["replay_items"]
        assert isinstance(replay_items, tuple)
        replay_item = replay_items[0]
        assert isinstance(replay_item, Mapping)
        assert replay_item["id"] == "reasoning-item"
        assert replay_item["encrypted_content"] == "provider-ciphertext"
        assert expected_provider_name == "request_user_input"
        assert expected_arguments["mode"] == "required"

    def import_continuation_snapshot(
        self,
        snapshot: ContinuationSnapshot,
        *,
        expected_binding: ContinuationRevisionBinding,
        provider_call_correlation_id: str,
    ) -> None:
        assert snapshot.revision_binding == expected_binding
        assert (
            snapshot.payload["reserved_capability_call_id"]
            == provider_call_correlation_id
        )
        self.imported.append(snapshot)


class _EventListenerRegistration:
    """Own one no-op PostgreSQL fixture listener."""

    def close(self) -> None:
        """Close the no-op fixture registration."""


class _ResumeExecutor:
    """Return one deterministic provider continuation result."""

    trusted_agent_continuation_executor = True

    def __init__(self) -> None:
        self.commands: list[AgentContinuationResumeCommand] = []

    def register_event_listener(
        self,
        listener: AgentContinuationEventListener,
    ) -> AgentContinuationEventListenerRegistration:
        assert callable(listener)
        return _EventListenerRegistration()

    async def resume_agent_continuation(
        self,
        command: AgentContinuationResumeCommand,
    ) -> object:
        self.commands.append(command)
        return "resumed output"

    async def close_continuation_runtime(self) -> None:
        """Close the deterministic runtime without external resources."""


class _ResumeLoader:
    """Reconstruct a fresh trusted runtime from persisted revisions."""

    trusted_continuation_runtime_loader = True

    def __init__(self, runtime: ResolvedContinuationRuntime) -> None:
        self.runtime = runtime
        self.calls = 0

    async def load_continuation_runtime(
        self,
        definition: ExecutionDefinitionRef,
        revision_binding: ContinuationRevisionBinding,
    ) -> ResolvedContinuationRuntime:
        assert definition == self.runtime.definition
        assert revision_binding == self.runtime.revision_binding
        self.calls += 1
        return self.runtime


class _RuntimeBundle:
    """Own the fresh runtime objects for one competing worker."""

    def __init__(
        self,
        *,
        target: _DurableWorkerTarget,
        adapter: _ResumeAdapter,
        executor: _ResumeExecutor,
        loader: _ResumeLoader,
        worker: TaskWorker,
    ) -> None:
        self.target = target
        self.adapter = adapter
        self.executor = executor
        self.loader = loader
        self.worker = worker


class PgsqlDurableInteractionE2ETest(IsolatedAsyncioTestCase):
    """Exercise worker restart, crash recovery, races, and retention."""

    async def asyncSetUp(self) -> None:
        dsn = real_task_pgsql_dsn()
        assert dsn, "AVALAN_TASK_TEST_POSTGRESQL_DSN is required"
        self.dsn = dsn
        self.schema = isolated_task_pgsql_schema("avalan_interaction_e2e")
        task_pgsql_upgrade(
            PgsqlTaskMigrationSettings(url=dsn, schema=self.schema)
        )
        self.databases: list[PsycopgAsyncDatabase] = []

    async def asyncTearDown(self) -> None:
        for database in reversed(self.databases):
            await database.aclose()
        await drop_task_pgsql_schema(self.dsn, self.schema)

    async def test_coordinator_scopes_isolate_real_corruption(self) -> None:
        async def suspended(
            suffix: str,
            *,
            child_only: bool = False,
        ) -> tuple[
            PsycopgAsyncDatabase,
            PgsqlInteractionStore,
            PgsqlTaskStore,
            PgsqlDurableTaskCoordinator,
            CreateInteractionApplied,
            CreateInteractionApplied | None,
        ]:
            database = await self._database(f"coordinator-{suffix}")
            task_store = PgsqlTaskStore(database, clock=lambda: _NOW)
            queue = PgsqlTaskQueue(database, clock=lambda: _NOW)
            store = await durable_support._store(database)
            coordinator = PgsqlDurableTaskCoordinator(store, task_store)
            queue_name = f"{_QUEUE}-{suffix}"
            definition_id = f"durable-worker-definition-{suffix}"
            await task_store.register_definition(
                replace(
                    _definition(),
                    task=TaskMetadata(
                        name=f"durable_worker_{suffix.replace('-', '_')}",
                        version="1",
                    ),
                    run=TaskRunPolicy.queued(queue_name),
                ),
                definition_hash=definition_id,
            )
            submission = await queue.enqueue_run(
                TaskExecutionRequest(
                    definition_id=definition_id,
                    queue=queue_name,
                ),
                queue_name=queue_name,
            )
            child_branch_id: BranchId | None = None
            parent_branch_id: BranchId | None = None
            unbound_root: CreateInteractionApplied | None = None
            if child_only:
                run_id = RunId(submission.run.run_id)
                root_request = durable_support._request(
                    f"{run_id}-unbound-root"
                )
                root_request = replace(
                    root_request,
                    origin=replace(
                        root_request.origin,
                        run_id=run_id,
                        branch_id=BranchId(f"{suffix}-unbound-root"),
                    ),
                )
                root_result = await store.create_durable(
                    durable_support._create_command(root_request),
                    durable_support._portable(root_request),
                )
                assert isinstance(root_result, CreateInteractionApplied)
                unbound_root = root_result
                terminal_root = await store.cancel(
                    CancelInteractionCommand(
                        actor=root_result.command.actor,
                        correlation=root_result.record.correlation,
                        provenance=AnswerProvenance.HUMAN,
                    )
                )
                assert terminal_root.store_mutation_applied
                parent_branch_id = BranchId(f"{suffix}-middle")
                middle = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=root_result.command.actor,
                        registration=InteractionBranchRegistration(
                            run_id=run_id,
                            branch_id=parent_branch_id,
                            parent_branch_id=(root_request.origin.branch_id),
                            principal=root_request.origin.principal,
                        ),
                    )
                )
                assert middle.store_mutation_applied
                child_branch_id = BranchId(f"{suffix}-child")
                child = await store.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=root_result.command.actor,
                        registration=InteractionBranchRegistration(
                            run_id=run_id,
                            branch_id=child_branch_id,
                            parent_branch_id=parent_branch_id,
                            principal=root_request.origin.principal,
                        ),
                    )
                )
                assert child.store_mutation_applied
            target = _DurableWorkerTarget(
                checkpoint_id=f"{_CHECKPOINT}-{suffix}",
                branch_id=child_branch_id,
                parent_branch_id=parent_branch_id,
            )
            worker = TaskWorker(
                task_store,
                queue,
                target=target,
                worker_id=f"coordinator-worker-{suffix}",
                queue_name=queue_name,
                durable_suspension_coordinator=coordinator,
                clock=lambda: _NOW,
            )
            result = await worker.process_once()
            assert result.suspension is not None, (
                target.initial_calls,
                result.completion,
                result.retry,
                result.abandonment,
            )
            assert result.suspension.run.run_id == submission.run.run_id
            assert target.suspension is not None
            command = target.suspension.command
            projections = await store.list_scoped(
                ListInteractionsCommand(
                    actor=command.actor,
                    scope=InteractionExecutionScope(
                        run_id=command.request.origin.run_id,
                    ),
                )
            )
            record = next(
                (
                    projection
                    for projection in projections
                    if isinstance(projection, InteractionRecord)
                    and projection.request.request_id
                    == command.request.request_id
                ),
                None,
            )
            assert record is not None
            return (
                database,
                store,
                task_store,
                coordinator,
                CreateInteractionApplied(
                    command=command,
                    record=record,
                    policy=InteractionPolicy(),
                ),
                unbound_root,
            )

        async def invoke(
            operation: str,
            coordinator: PgsqlDurableTaskCoordinator,
            interaction: CreateInteractionApplied,
        ) -> object:
            run_id = str(interaction.record.request.origin.run_id)
            scope = InteractionExecutionScope(
                run_id=interaction.record.request.origin.run_id,
            )
            if operation == "resolve":
                return await coordinator.resolve_and_requeue(
                    durable_support._answer(interaction),
                    task_run_id=run_id,
                )
            if operation == "cancel":
                return await coordinator.cancel_suspended_task(
                    TerminalizeInteractionScopeCommand(
                        actor=interaction.command.actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    ),
                    task_run_id=run_id,
                )
            if operation == "supersede":
                return await coordinator.supersede_suspended_task(
                    SupersedeInteractionScopeCommand(
                        actor=interaction.command.actor,
                        scope=scope,
                        provenance=AnswerProvenance.HUMAN,
                    ),
                    task_run_id=run_id,
                )
            assert operation == "trusted_cancel"
            return await coordinator.cancel_input_required_task(
                task_run_id=run_id,
                now=_NOW,
                metadata={},
            )

        async def corrupt(
            database: PsycopgAsyncDatabase,
            request_id: str,
        ) -> bytes:
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
SELECT "ciphertext"
FROM "interaction_records"
WHERE "request_id" = %s
""",
                        (request_id,),
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    ciphertext = cast(bytes, row["ciphertext"])
                    await cursor.execute(
                        """
UPDATE "interaction_records"
SET "ciphertext" = %s
WHERE "request_id" = %s
""",
                        (b"\x00", request_id),
                    )
                    return ciphertext

        async def restore(
            database: PsycopgAsyncDatabase,
            request_id: str,
            ciphertext: bytes,
        ) -> None:
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
UPDATE "interaction_records"
SET "ciphertext" = %s
WHERE "request_id" = %s
""",
                        (ciphertext, request_id),
                    )

        async def request_state(
            database: PsycopgAsyncDatabase,
            request_id: str,
        ) -> RequestState:
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
SELECT "request_state"
FROM "interaction_records"
WHERE "request_id" = %s
""",
                        (request_id,),
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    state = row["request_state"]
                    assert isinstance(state, str)
                    return RequestState(state)

        async def branch_ciphertext(
            database: PsycopgAsyncDatabase,
            run_id: RunId,
            branch_id: BranchId,
        ) -> bytes:
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
SELECT "ciphertext"
FROM "interaction_branches"
WHERE "run_id" = %s
  AND "branch_id" = %s
""",
                        (str(run_id), str(branch_id)),
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    ciphertext = row["ciphertext"]
                    assert isinstance(ciphertext, bytes)
                    return ciphertext

        async def corrupt_branch(
            database: PsycopgAsyncDatabase,
            run_id: RunId,
            branch_id: BranchId,
        ) -> bytes:
            ciphertext = await branch_ciphertext(
                database,
                run_id,
                branch_id,
            )
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
UPDATE "interaction_branches"
SET "ciphertext" = %s
WHERE "run_id" = %s
  AND "branch_id" = %s
""",
                        (b"\x00", str(run_id), str(branch_id)),
                    )
            return ciphertext

        async def restore_branch(
            database: PsycopgAsyncDatabase,
            run_id: RunId,
            branch_id: BranchId,
            ciphertext: bytes,
        ) -> None:
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
UPDATE "interaction_branches"
SET "ciphertext" = %s
WHERE "run_id" = %s
  AND "branch_id" = %s
""",
                        (
                            ciphertext,
                            str(run_id),
                            str(branch_id),
                        ),
                    )

        expected_states = {
            "resolve": RequestState.ANSWERED,
            "cancel": RequestState.CANCELLED,
            "supersede": RequestState.SUPERSEDED,
            "trusted_cancel": RequestState.CANCELLED,
        }
        for operation, expected_state in expected_states.items():
            with self.subTest(operation=operation, corruption="unrelated"):
                (
                    database,
                    store,
                    _task_store,
                    coordinator,
                    interaction,
                    _unbound_root,
                ) = await suspended(f"{operation}-unrelated")
                unrelated_principal = (
                    interaction.record.request.origin.principal
                    if operation == "trusted_cancel"
                    else PrincipalScope(
                        user_id=UserId(f"real-unrelated-{operation}")
                    )
                )
                unrelated_request = durable_support._request(
                    f"real-unrelated-{operation}"
                )
                unrelated_request = replace(
                    unrelated_request,
                    origin=replace(
                        unrelated_request.origin,
                        run_id=(interaction.record.request.origin.run_id),
                        branch_id=BranchId(f"real-unrelated-root-{operation}"),
                        principal=unrelated_principal,
                    ),
                )
                if operation == "trusted_cancel":
                    unrelated = await store.create_durable(
                        durable_support._create_command(unrelated_request),
                        durable_support._portable(unrelated_request),
                    )
                else:
                    unrelated = await store.create(
                        durable_support._create_command(unrelated_request)
                    )
                assert isinstance(unrelated, CreateInteractionApplied)
                healthy: CreateInteractionApplied | None = None
                bound_peer: CreateInteractionApplied | None = None
                healthy_branch_id: BranchId | None = None
                corrupt_branch_id: BranchId | None = None
                healthy_branch_before: bytes | None = None
                corrupt_branch_before: bytes | None = None
                if operation == "trusted_cancel":
                    healthy_request = durable_support._request(
                        "real-unrelated-trusted-cancel-healthy"
                    )
                    healthy_request = replace(
                        healthy_request,
                        origin=replace(
                            healthy_request.origin,
                            run_id=(interaction.record.request.origin.run_id),
                            branch_id=BranchId(
                                "real-unrelated-trusted-cancel-healthy-root"
                            ),
                            principal=unrelated_principal,
                        ),
                    )
                    healthy_result = await store.create_durable(
                        durable_support._create_command(healthy_request),
                        durable_support._portable(healthy_request),
                    )
                    assert isinstance(
                        healthy_result,
                        CreateInteractionApplied,
                    )
                    healthy = healthy_result
                    bound_middle_id = BranchId(
                        "real-trusted-cancel-bound-middle"
                    )
                    bound_middle = await store.register_branch(
                        RegisterInteractionBranchCommand(
                            actor=interaction.command.actor,
                            registration=InteractionBranchRegistration(
                                run_id=(
                                    interaction.record.request.origin.run_id
                                ),
                                branch_id=bound_middle_id,
                                parent_branch_id=(
                                    interaction.record.request.origin.branch_id
                                ),
                                principal=unrelated_principal,
                            ),
                        )
                    )
                    assert bound_middle.store_mutation_applied
                    bound_child_id = BranchId(
                        "real-trusted-cancel-bound-child"
                    )
                    bound_child = await store.register_branch(
                        RegisterInteractionBranchCommand(
                            actor=interaction.command.actor,
                            registration=InteractionBranchRegistration(
                                run_id=(
                                    interaction.record.request.origin.run_id
                                ),
                                branch_id=bound_child_id,
                                parent_branch_id=bound_middle_id,
                                principal=unrelated_principal,
                            ),
                        )
                    )
                    assert bound_child.store_mutation_applied
                    second_bound_request = durable_support._request(
                        "real-trusted-cancel-second-bound"
                    )
                    second_bound_request = replace(
                        second_bound_request,
                        origin=replace(
                            second_bound_request.origin,
                            run_id=(interaction.record.request.origin.run_id),
                            branch_id=bound_child_id,
                            parent_branch_id=bound_middle_id,
                            principal=unrelated_principal,
                        ),
                    )
                    bound_peer = await coordinator.create_pending_interaction(
                        durable_support._create_command(second_bound_request),
                        durable_support._portable(second_bound_request),
                        task_run_id=str(
                            interaction.record.request.origin.run_id
                        ),
                        checkpoint_id=(
                            "real-trusted-cancel-second-bound-checkpoint"
                        ),
                    )
                    healthy_branch_id = BranchId(
                        "real-unrelated-trusted-cancel-healthy-child"
                    )
                    healthy_branch = await store.register_branch(
                        RegisterInteractionBranchCommand(
                            actor=healthy.command.actor,
                            registration=InteractionBranchRegistration(
                                run_id=healthy_request.origin.run_id,
                                branch_id=healthy_branch_id,
                                parent_branch_id=(
                                    healthy_request.origin.branch_id
                                ),
                                principal=healthy_request.origin.principal,
                            ),
                        )
                    )
                    assert healthy_branch.store_mutation_applied
                    corrupt_branch_id = BranchId(
                        "real-unrelated-trusted-cancel-corrupt-child"
                    )
                    corrupt_branch_result = await store.register_branch(
                        RegisterInteractionBranchCommand(
                            actor=unrelated.command.actor,
                            registration=InteractionBranchRegistration(
                                run_id=unrelated_request.origin.run_id,
                                branch_id=corrupt_branch_id,
                                parent_branch_id=(
                                    unrelated_request.origin.branch_id
                                ),
                                principal=unrelated_request.origin.principal,
                            ),
                        )
                    )
                    assert corrupt_branch_result.store_mutation_applied
                    healthy_branch_before = await branch_ciphertext(
                        database,
                        healthy_request.origin.run_id,
                        healthy_branch_id,
                    )
                    corrupt_branch_before = await corrupt_branch(
                        database,
                        unrelated_request.origin.run_id,
                        corrupt_branch_id,
                    )
                unrelated_ciphertext = await corrupt(
                    database,
                    str(unrelated_request.request_id),
                )

                await invoke(operation, coordinator, interaction)

                projection = await store.lookup_scoped(
                    ScopedInteractionLookup(
                        actor=interaction.command.actor,
                        correlation=interaction.record.correlation,
                    )
                )
                assert isinstance(projection, InteractionRecord)
                assert projection.request.state is expected_state
                if bound_peer is not None:
                    assert (
                        await request_state(
                            database,
                            str(bound_peer.record.request.request_id),
                        )
                        is RequestState.CANCELLED
                    )
                if corrupt_branch_id is not None:
                    assert healthy_branch_id is not None
                    assert healthy_branch_before is not None
                    assert corrupt_branch_before is not None
                    assert (
                        await branch_ciphertext(
                            database,
                            unrelated_request.origin.run_id,
                            corrupt_branch_id,
                        )
                        == b"\x00"
                    )
                    assert (
                        await branch_ciphertext(
                            database,
                            unrelated_request.origin.run_id,
                            healthy_branch_id,
                        )
                        == healthy_branch_before
                    )
                with self.assertRaises(PgsqlInteractionStoreError):
                    await store.lookup_scoped(
                        ScopedInteractionLookup(
                            actor=unrelated.command.actor,
                            correlation=unrelated.record.correlation,
                        )
                    )
                await restore(
                    database,
                    str(unrelated_request.request_id),
                    unrelated_ciphertext,
                )
                if corrupt_branch_id is not None:
                    assert corrupt_branch_before is not None
                    await restore_branch(
                        database,
                        unrelated_request.origin.run_id,
                        corrupt_branch_id,
                        corrupt_branch_before,
                    )
                unrelated_projection = await store.lookup_scoped(
                    ScopedInteractionLookup(
                        actor=unrelated.command.actor,
                        correlation=unrelated.record.correlation,
                    )
                )
                assert isinstance(
                    unrelated_projection,
                    InteractionRecord,
                )
                assert (
                    unrelated_projection.request.state is RequestState.PENDING
                )
                if healthy is not None:
                    healthy_projection = await store.lookup_scoped(
                        ScopedInteractionLookup(
                            actor=healthy.command.actor,
                            correlation=healthy.record.correlation,
                        )
                    )
                    assert isinstance(
                        healthy_projection,
                        InteractionRecord,
                    )
                    assert (
                        healthy_projection.request.state
                        is RequestState.PENDING
                    )

            with self.subTest(operation=operation, corruption="targeted"):
                (
                    database,
                    _store,
                    _task_store,
                    coordinator,
                    interaction,
                    _unbound_root,
                ) = await suspended(f"{operation}-targeted")
                request_id = str(interaction.record.request.request_id)
                target_ciphertext = await corrupt(database, request_id)

                with self.assertRaises(PgsqlInteractionStoreError):
                    await invoke(operation, coordinator, interaction)

                await restore(database, request_id, target_ciphertext)

        with self.subTest(
            operation="trusted_cancel",
            ancestry="child_only",
        ):
            suffix = "trusted-cancel-child-only"
            (
                database,
                _store,
                _task_store,
                coordinator,
                interaction,
                unbound_root,
            ) = await suspended(suffix, child_only=True)
            assert unbound_root is not None
            run_id = interaction.record.request.origin.run_id
            parent_branch_id = (
                interaction.record.request.origin.parent_branch_id
            )
            assert parent_branch_id is not None
            child_branch_before = await branch_ciphertext(
                database,
                run_id,
                interaction.record.request.origin.branch_id,
            )
            parent_branch_before = await branch_ciphertext(
                database,
                run_id,
                parent_branch_id,
            )

            completion = await coordinator.cancel_input_required_task(
                task_run_id=str(run_id),
                now=_NOW,
                metadata={},
            )
            replayed = await coordinator.cancel_input_required_task(
                task_run_id=str(run_id),
                now=_NOW,
                metadata={},
            )

            assert completion == replayed
            assert completion.run.state is TaskRunState.CANCELLED
            assert (
                await request_state(
                    database,
                    str(interaction.record.request.request_id),
                )
                is RequestState.CANCELLED
            )
            assert (
                await request_state(
                    database,
                    str(unbound_root.record.request.request_id),
                )
                is RequestState.CANCELLED
            )
            assert (
                await branch_ciphertext(
                    database,
                    run_id,
                    interaction.record.request.origin.branch_id,
                )
                == child_branch_before
            )
            assert (
                await branch_ciphertext(
                    database,
                    run_id,
                    parent_branch_id,
                )
                == parent_branch_before
            )

        with self.subTest(
            operation="trusted_cancel",
            ancestry="tampered_root",
        ):
            suffix = "trusted-cancel-tampered-root"
            (
                database,
                _store,
                task_store,
                coordinator,
                interaction,
                _unbound_root,
            ) = await suspended(suffix, child_only=True)
            run_id = interaction.record.request.origin.run_id
            parent_branch_id = (
                interaction.record.request.origin.parent_branch_id
            )
            assert parent_branch_id is not None
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
UPDATE "interaction_branches"
SET "root_branch_id" = 'tampered-root'
WHERE "run_id" = %s
  AND "branch_id" = %s
""",
                        (str(run_id), str(parent_branch_id)),
                    )

            with self.assertRaises(PgsqlInteractionStoreError):
                await coordinator.cancel_input_required_task(
                    task_run_id=str(run_id),
                    now=_NOW,
                    metadata={},
                )

            assert (
                await request_state(
                    database,
                    str(interaction.record.request.request_id),
                )
                is RequestState.PENDING
            )
            run = await task_store.get_run(str(run_id))
            assert run.state is TaskRunState.INPUT_REQUIRED

    async def test_wait_change_returns_coherent_resolved_projection(
        self,
    ) -> None:
        writer_database = await self._database("wait-change-writer")
        reader_database = await self._database("wait-change-reader")
        writer = await durable_support._store(writer_database)
        request = durable_support._request("real-wait-snapshot")
        created = await writer.create(durable_support._create_command(request))
        assert isinstance(created, CreateInteractionApplied)
        authorizer = durable_support._InterleavingAuthorizer()
        reader = await durable_support._store(
            reader_database,
            authorizer=authorizer,
            store_policy=PgsqlInteractionStorePolicy(
                poll_interval_seconds=0.001,
            ),
        )
        resolutions: list[ResolveInteractionApplied] = []

        async def resolve_during_authorization() -> None:
            result = await writer.resolve(durable_support._answer(created))
            assert isinstance(result, ResolveInteractionApplied)
            resolutions.append(result)

        authorizer.on_inspect = resolve_during_authorization
        projection = await reader.wait_for_change(
            WaitForInteractionChangeCommand(
                actor=created.command.actor,
                correlation=created.record.correlation,
                after_store_revision=created.record.store_revision,
            )
        )

        assert len(resolutions) == 1
        assert isinstance(projection, InteractionRecord)
        assert projection == resolutions[0].record
        assert projection.request.resolution is not None
        await reader.aclose()
        await writer.aclose()

    async def test_partial_resumed_startup_cancel_recovers_after_restart(
        self,
    ) -> None:
        queue_name = f"{_QUEUE}-cancel-startup"
        database = await self._database("cancel-startup-before-crash")
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW,
        )
        queue = PgsqlTaskQueue(
            database,
            clock=lambda: _NOW,
        )
        store = await durable_support._store(database)
        coordinator = PgsqlDurableTaskCoordinator(store, task_store)
        definition_id = "cancel-startup-definition"
        await task_store.register_definition(
            replace(
                _definition(),
                task=TaskMetadata(
                    name="durable_worker_cancel_startup",
                    version="1",
                ),
                run=TaskRunPolicy.queued(queue_name),
            ),
            definition_hash=definition_id,
        )
        submission = await queue.enqueue_run(
            TaskExecutionRequest(
                definition_id=definition_id,
                queue=queue_name,
            ),
            queue_name=queue_name,
        )
        target = _DurableWorkerTarget(
            checkpoint_id="cancel-startup-checkpoint"
        )
        worker = TaskWorker(
            task_store,
            queue,
            target=target,
            worker_id="cancel-startup-initial-worker",
            queue_name=queue_name,
            durable_suspension_coordinator=coordinator,
            clock=lambda: _NOW,
        )
        suspended = await worker.process_once()
        assert suspended.suspension is not None
        assert target.suspension is not None
        command = target.suspension.command
        record = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=command.actor,
                correlation=InteractionCorrelation.from_request(
                    command.request
                ),
            )
        )
        assert isinstance(record, InteractionRecord)
        interaction = CreateInteractionApplied(
            command=command,
            record=record,
            policy=InteractionPolicy(),
        )
        await coordinator.resolve_and_requeue(
            durable_support._answer(interaction),
            task_run_id=submission.run.run_id,
            now=_NOW + timedelta(seconds=1),
        )
        claim = await queue.claim(
            queue_name,
            worker_id="cancel-startup-resumed-worker",
            lease_expires_at=_NOW + timedelta(seconds=5),
            now=_NOW + timedelta(seconds=2),
        )
        assert claim is not None
        ready = await store.get_continuation(command.request.continuation_id)
        dispatch_id = derive_continuation_dispatch_id(
            command.request.continuation_id
        )
        run_claim = claim.run.claim
        assert run_claim is not None
        claimed = await store.claim(
            command.request.continuation_id,
            expected_store_revision=ready.store_revision,
            owner_id=ContinuationClaimOwnerId(run_claim.claim_token),
            lease_expires_at=_NOW + timedelta(seconds=5),
            dispatch_id=dispatch_id,
            provider_idempotency_key=derive_provider_idempotency_key(
                command.request.continuation_id,
                dispatch_id,
            ),
            now=_NOW + timedelta(seconds=3),
        )
        claim_token = claim.queue_item.claim_token
        assert claim_token is not None
        run = await task_store.transition_run(
            submission.run.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claim_token,
        )
        attempt = await task_store.transition_attempt(
            claim.attempt.attempt_id,
            from_states={TaskAttemptState.SUSPENDED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
            claim_token=claim_token,
        )
        previous_segments = await task_store.list_attempt_segments(
            attempt.attempt_id
        )
        assert len(previous_segments) == 1
        active_segment = await task_store.create_attempt_segment(
            attempt.attempt_id,
            claim_token=claim_token,
            resumed_from_segment_id=previous_segments[0].segment_id,
        )
        cancelled = await TaskClient(
            task_store,
            target=durable_support._unused_task_target,
            clock=lambda: _NOW + timedelta(seconds=4),
        ).cancel(run.run_id)
        assert cancelled.state is TaskRunState.CANCEL_REQUESTED

        restart_database = await self._database("cancel-startup-after-crash")
        restarted_store = await durable_support._store(restart_database)
        restarted_task_store = PgsqlTaskStore(
            restart_database,
            clock=lambda: _NOW + timedelta(seconds=6),
        )
        restarted_coordinator = PgsqlDurableTaskCoordinator(
            restarted_store,
            restarted_task_store,
        )
        restarted_queue = PgsqlTaskQueue(
            restart_database,
            clock=lambda: _NOW + timedelta(seconds=6),
            durable_reentry_coordinator=restarted_coordinator,
        )
        self.assertEqual(
            await restarted_queue.abandon_expired(
                queue_name,
                max_attempts=3,
                limit=10,
                now=_NOW + timedelta(seconds=6),
            ),
            (),
        )
        recovered_run = await restarted_task_store.get_run(run.run_id)
        assert recovered_run.last_attempt_id is not None
        recovered_attempt = await restarted_task_store.get_attempt(
            recovered_run.last_attempt_id
        )
        recovered_segments = await restarted_task_store.list_attempt_segments(
            recovered_attempt.attempt_id
        )
        self.assertEqual(recovered_run.state, TaskRunState.CANCELLED)
        self.assertEqual(
            recovered_attempt.state,
            TaskAttemptState.ABANDONED,
        )
        self.assertEqual(
            recovered_segments[0].state,
            TaskAttemptSegmentState.SUSPENDED,
        )
        self.assertEqual(
            recovered_segments[1].segment_id,
            active_segment.segment_id,
        )
        self.assertEqual(
            recovered_segments[1].state,
            TaskAttemptSegmentState.ABANDONED,
        )
        self.assertEqual(
            recovered_segments[1].resumed_from_segment_id,
            recovered_segments[0].segment_id,
        )
        self.assertEqual(
            recovered_segments[0].request_id,
            str(command.request.request_id),
        )
        self.assertEqual(
            recovered_segments[0].continuation_id,
            str(command.request.continuation_id),
        )
        attempt_claim = recovered_attempt.context.claim
        assert attempt_claim is not None
        self.assertEqual(attempt_claim.claim_token, claim_token)
        segment_claim = recovered_segments[1].claim
        assert segment_claim is not None
        self.assertEqual(segment_claim.claim_token, claim_token)
        recovered_continuation = await restarted_store.get_continuation(
            command.request.continuation_id
        )
        self.assertEqual(
            recovered_continuation.provider_snapshot,
            claimed.continuation.provider_snapshot,
        )
        self.assertEqual(
            recovered_continuation.transcript,
            claimed.continuation.transcript,
        )

        async with restart_database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT "state", "claim_token", "lease_expires_at"
FROM "task_queue_items"
WHERE "run_id" = %s
""",
                    (run.run_id,),
                )
                queue_row = await cursor.fetchone()
                await cursor.execute(
                    """
SELECT ARRAY_AGG("status" ORDER BY "outbox_id") AS "statuses"
FROM "interaction_resumption_outbox"
WHERE "task_run_id" = %s
""",
                    (run.run_id,),
                )
                outbox_row = await cursor.fetchone()
                await cursor.execute(
                    """
SELECT COUNT(*) AS "expired_events"
FROM "task_events"
WHERE "run_id" = %s
  AND "event_type" = %s
""",
                    (
                        run.run_id,
                        TaskInteractionEventType.INPUT_EXPIRED.value,
                    ),
                )
                event_row = await cursor.fetchone()
                await cursor.execute(
                    """
SELECT
    (SELECT COUNT(*) FROM "task_run_transitions"
     WHERE "run_id" = %s) AS "run_transitions",
    (SELECT COUNT(*) FROM "task_attempt_transitions"
     WHERE "run_id" = %s) AS "attempt_transitions",
    (SELECT COUNT(*) FROM "task_attempt_segment_transitions"
     WHERE "run_id" = %s) AS "segment_transitions"
""",
                    (run.run_id, run.run_id, run.run_id),
                )
                transition_counts = await cursor.fetchone()
        assert queue_row is not None
        assert outbox_row is not None
        assert event_row is not None
        assert transition_counts is not None
        self.assertEqual(queue_row["state"], TaskQueueItemState.DEAD.value)
        self.assertIsNone(queue_row["claim_token"])
        self.assertIsNone(queue_row["lease_expires_at"])
        self.assertEqual(outbox_row["statuses"], ["dead"])
        self.assertEqual(event_row["expired_events"], 0)

        replay_failure = TaskExecutionResult(
            error={"code": "expired_durable_reentry_claim"}
        )
        replayed = await restarted_coordinator.reconcile_expired_reentry(
            queue_item_id=claim.queue_item.queue_item_id,
            expected_claim_token=claim_token,
            task_run_id=run.run_id,
            result=replay_failure,
            now=_NOW + timedelta(seconds=7),
        )
        assert replayed.completion is not None
        self.assertEqual(replayed.completion.run, recovered_run)
        for restart_number in range(3):
            repeated_database = await self._database(
                f"cancel-startup-repeat-{restart_number}"
            )
            repeated_store = await durable_support._store(repeated_database)
            repeated_task_store = PgsqlTaskStore(
                repeated_database,
                clock=lambda: _NOW + timedelta(seconds=8),
            )
            repeated_coordinator = PgsqlDurableTaskCoordinator(
                repeated_store,
                repeated_task_store,
            )
            repeated_queue = PgsqlTaskQueue(
                repeated_database,
                clock=lambda: _NOW + timedelta(seconds=8),
                durable_reentry_coordinator=repeated_coordinator,
            )
            self.assertEqual(
                await repeated_queue.abandon_expired(
                    queue_name,
                    max_attempts=3,
                    limit=10,
                    now=_NOW + timedelta(seconds=8 + restart_number),
                ),
                (),
            )
        async with restart_database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT
    (SELECT COUNT(*) FROM "task_run_transitions"
     WHERE "run_id" = %s) AS "run_transitions",
    (SELECT COUNT(*) FROM "task_attempt_transitions"
     WHERE "run_id" = %s) AS "attempt_transitions",
    (SELECT COUNT(*) FROM "task_attempt_segment_transitions"
     WHERE "run_id" = %s) AS "segment_transitions"
""",
                    (run.run_id, run.run_id, run.run_id),
                )
                repeated_counts = await cursor.fetchone()
        self.assertEqual(repeated_counts, transition_counts)

    async def test_worker_restart_crash_recovery_and_retention(self) -> None:
        before_restart = await self._database("before-restart")
        task_store = PgsqlTaskStore(
            before_restart,
            clock=lambda: _NOW,
        )
        queue = PgsqlTaskQueue(
            before_restart,
            clock=lambda: _NOW,
        )
        interaction_store = await durable_support._store(before_restart)
        coordinator = PgsqlDurableTaskCoordinator(
            interaction_store,
            task_store,
        )
        await task_store.register_definition(
            _definition(),
            definition_hash="durable-worker-definition",
        )
        submission = await queue.enqueue_run(
            TaskExecutionRequest(
                definition_id="durable-worker-definition",
                queue=_QUEUE,
            ),
            queue_name=_QUEUE,
        )
        initial_target = _DurableWorkerTarget()
        first_worker = TaskWorker(
            task_store,
            queue,
            target=initial_target,
            worker_id="worker-before-restart",
            queue_name=_QUEUE,
            durable_suspension_coordinator=coordinator,
            clock=lambda: _NOW,
        )

        suspended = await first_worker.process_once()

        assert suspended.suspension is not None
        assert suspended.suspension.run.state is TaskRunState.INPUT_REQUIRED
        assert suspended.suspension.attempt.state is TaskAttemptState.SUSPENDED
        assert (
            suspended.suspension.segment.state
            is TaskAttemptSegmentState.SUSPENDED
        )
        assert suspended.suspension.queue_item.claim_token is None
        assert initial_target.initial_calls == 1
        assert initial_target.resume_calls == 0
        assert initial_target.suspension is not None
        staged_command = initial_target.suspension.command
        continuation_id = (
            initial_target.suspension.continuation.continuation_id
        )
        persisted = await interaction_store.get_continuation(continuation_id)
        persisted_snapshot = persisted.provider_snapshot
        assert isinstance(persisted_snapshot, ContinuationSnapshot)
        expected_dispatch_id = derive_continuation_dispatch_id(continuation_id)
        assert (
            persisted_snapshot.provider_idempotency_key
            == derive_provider_idempotency_key(
                continuation_id,
                expected_dispatch_id,
            )
        )

        await interaction_store.aclose()
        await self._close_database(before_restart)

        after_resolution_restart = await self._database(
            "after-resolution-restart"
        )
        restarted_interaction_store = await durable_support._store(
            after_resolution_restart
        )
        restarted_task_store = PgsqlTaskStore(
            after_resolution_restart,
            clock=lambda: _NOW + timedelta(seconds=1),
        )
        restarted_coordinator = PgsqlDurableTaskCoordinator(
            restarted_interaction_store,
            restarted_task_store,
        )
        staged_record = await restarted_interaction_store.lookup_scoped(
            ScopedInteractionLookup(
                actor=staged_command.actor,
                correlation=InteractionCorrelation.from_request(
                    staged_command.request
                ),
            )
        )
        assert isinstance(staged_record, InteractionRecord)
        staged = CreateInteractionApplied(
            command=staged_command,
            record=staged_record,
            policy=InteractionPolicy(),
        )
        resolution = await restarted_coordinator.resolve_and_requeue(
            durable_support._answer(staged),
            task_run_id=submission.run.run_id,
            now=_NOW + timedelta(seconds=1),
        )

        assert resolution.reentry.run.state is TaskRunState.QUEUED
        assert (
            resolution.reentry.attempt.attempt_id
            == suspended.suspension.attempt.attempt_id
        )
        assert (
            resolution.reentry.queue_item.attempts
            == suspended.suspension.queue_item.attempts
        )

        await restarted_interaction_store.aclose()
        await self._close_database(after_resolution_restart)

        after_crash_restart = await self._database("after-crash-restart")
        recovered_store = await durable_support._store(after_crash_restart)
        delivered: list[object] = []

        async def dispatch(record: object) -> None:
            delivered.append(record)

        reconciler = PgsqlResumptionReconciler(
            recovered_store,
            owner_id=ContinuationClaimOwnerId("reconciler-after-crash"),
            dispatcher=dispatch,
            clock=lambda: _NOW + timedelta(seconds=2),
        )

        assert await reconciler.run_once() == 1
        assert len(delivered) == 1
        await recovered_store.aclose()
        await self._close_database(after_crash_restart)

        worker_database_a = await self._database("worker-a")
        worker_database_b = await self._database("worker-b")
        bundle_a = await self._runtime_bundle(
            worker_database_a,
            worker_id="worker-a",
            task_run_id=submission.run.run_id,
        )
        bundle_b = await self._runtime_bundle(
            worker_database_b,
            worker_id="worker-b",
            task_run_id=submission.run.run_id,
        )

        worker_results = await gather(
            bundle_a.worker.process_once(),
            bundle_b.worker.process_once(),
        )

        completions = tuple(
            result.completion
            for result in worker_results
            if result.completion is not None
        )
        assert len(completions) == 1
        completion = completions[0]
        assert completion.run.state is TaskRunState.SUCCEEDED
        assert completion.attempt.state is TaskAttemptState.SUCCEEDED
        assert completion.queue_item.state is TaskQueueItemState.DONE
        assert (
            completion.queue_item.attempts
            == suspended.suspension.queue_item.attempts
        )
        assert (
            sum(bundle.target.resume_calls for bundle in (bundle_a, bundle_b))
            == 1
        )
        assert (
            sum(
                len(bundle.executor.commands)
                for bundle in (bundle_a, bundle_b)
            )
            == 1
        )
        assert sum(bundle.loader.calls for bundle in (bundle_a, bundle_b)) == 1
        assert (
            sum(
                len(bundle.adapter.imported) for bundle in (bundle_a, bundle_b)
            )
            == 1
        )
        assert all(
            bundle.target.initial_calls == 0 for bundle in (bundle_a, bundle_b)
        )
        command = next(
            bundle.executor.commands[0]
            for bundle in (bundle_a, bundle_b)
            if bundle.executor.commands
        )
        assert str(command.correlated_result.call_id) == "input-call"
        assert (
            command.continuation.provider_snapshot
            == initial_target.suspension.continuation.provider_snapshot
        )

        final_store = PgsqlTaskStore(worker_database_a)
        final_run = await final_store.get_run(submission.run.run_id)
        attempts = await final_store.list_attempts(submission.run.run_id)
        segments = await final_store.list_attempt_segments(
            attempts[0].attempt_id
        )
        assert final_run.state is TaskRunState.SUCCEEDED
        assert final_run.claim is None
        assert len(attempts) == 1
        assert attempts[0].attempt_number == 1
        assert tuple(segment.segment_number for segment in segments) == (1, 2)
        assert segments[1].resumed_from_segment_id == segments[0].segment_id

        retention_store = await durable_support._store(worker_database_a)
        swept = await PgsqlDurableTaskCoordinator(
            retention_store,
            final_store,
        ).sweep_retention(
            now=_NOW + timedelta(days=31),
        )
        assert swept.deleted == (continuation_id,)
        with self.assertRaises(InteractionNotFoundError):
            await retention_store.get_continuation(continuation_id)
        await retention_store.aclose()

    async def test_completed_provider_failure_replays_after_connection_loss(
        self,
    ) -> None:
        database_a = await self._database("completed-provider-a")
        database_b = await self._database("completed-provider-b")
        task_store_a = PgsqlTaskStore(
            database_a,
            clock=lambda: _NOW,
        )
        queue_a = PgsqlTaskQueue(
            database_a,
            clock=lambda: _NOW,
        )
        interaction_store_a = await durable_support._store(database_a)
        coordinator_a = PgsqlDurableTaskCoordinator(
            interaction_store_a,
            task_store_a,
        )
        await task_store_a.register_definition(
            _definition(),
            definition_hash="completed-provider-definition",
        )
        submission = await queue_a.enqueue_run(
            TaskExecutionRequest(
                definition_id="completed-provider-definition",
                queue=_QUEUE,
            ),
            queue_name=_QUEUE,
        )
        target = _DurableWorkerTarget()
        initial = await TaskWorker(
            task_store_a,
            queue_a,
            target=target,
            worker_id="completed-provider-initial",
            queue_name=_QUEUE,
            durable_suspension_coordinator=coordinator_a,
            clock=lambda: _NOW,
        ).process_once()
        assert initial.suspension is not None
        assert target.suspension is not None
        interaction_record = await interaction_store_a.lookup_scoped(
            ScopedInteractionLookup(
                actor=target.suspension.command.actor,
                correlation=InteractionCorrelation.from_request(
                    target.suspension.command.request
                ),
            )
        )
        assert isinstance(interaction_record, InteractionRecord)
        await coordinator_a.resolve_and_requeue(
            durable_support._answer(
                CreateInteractionApplied(
                    command=target.suspension.command,
                    record=interaction_record,
                    policy=InteractionPolicy(),
                )
            ),
            task_run_id=submission.run.run_id,
            now=_NOW + timedelta(seconds=1),
        )
        claim = await queue_a.claim(
            _QUEUE,
            worker_id="completed-provider-worker",
            lease_expires_at=_NOW + timedelta(minutes=5),
            now=_NOW + timedelta(seconds=2),
        )
        assert claim is not None
        claim_token = claim.queue_item.claim_token
        assert claim_token is not None
        run = await task_store_a.transition_run(
            claim.run.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claim_token,
        )
        attempt = await task_store_a.transition_attempt(
            claim.attempt.attempt_id,
            from_states={TaskAttemptState.SUSPENDED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
            claim_token=claim_token,
        )
        active_segment = await task_store_a.create_attempt_segment(
            attempt.attempt_id,
            claim_token=claim_token,
            resumed_from_segment_id=initial.suspension.segment.segment_id,
        )
        active_segment = await task_store_a.transition_attempt_segment(
            active_segment.segment_id,
            from_states={TaskAttemptSegmentState.CREATED},
            to_state=TaskAttemptSegmentState.RUNNING,
            reason="started",
            claim_token=claim_token,
        )
        continuation = await interaction_store_a.get_continuation(
            target.suspension.continuation.continuation_id
        )
        owner_id = ContinuationClaimOwnerId(claim_token)
        dispatch_id = derive_continuation_dispatch_id(
            continuation.continuation_id
        )
        receipt = await interaction_store_a.claim(
            continuation.continuation_id,
            expected_store_revision=continuation.store_revision,
            owner_id=owner_id,
            lease_expires_at=_NOW + timedelta(minutes=5),
            dispatch_id=dispatch_id,
            provider_idempotency_key=derive_provider_idempotency_key(
                continuation.continuation_id,
                dispatch_id,
            ),
            now=_NOW + timedelta(seconds=3),
        )
        dispatching = await interaction_store_a.mark_dispatching(
            continuation.continuation_id,
            expected_store_revision=receipt.continuation.store_revision,
            owner_id=owner_id,
            fencing_token=receipt.fencing_token,
            now=_NOW + timedelta(seconds=4),
        )
        dispatched = await interaction_store_a.mark_dispatched(
            continuation.continuation_id,
            expected_store_revision=dispatching.store_revision,
            owner_id=owner_id,
            fencing_token=receipt.fencing_token,
            now=_NOW + timedelta(seconds=5),
        )
        provider_completion = ContinuationCompletionCommand(
            continuation_id=dispatched.continuation_id,
            expected_store_revision=dispatched.store_revision,
            owner_id=owner_id,
            fencing_token=dispatched.fencing_token,
            result_digest="f" * 64,
        )

        async def complete_provider(
            unit: PgsqlUnitOfWork,
        ) -> PortableContinuation:
            return await interaction_store_a._complete_continuation_in_unit(
                unit,
                continuation_id=dispatched.continuation_id,
                expected_store_revision=(
                    provider_completion.expected_store_revision
                ),
                owner_id=provider_completion.owner_id,
                fencing_token=provider_completion.fencing_token,
                result_digest=provider_completion.result_digest,
                now=_NOW + timedelta(seconds=6),
                expected_task_run_id=run.run_id,
            )

        completed_continuation: PortableContinuation = (
            await interaction_store_a._transaction(
                "test_completed_provider_crash",
                complete_provider,
            )
        )
        failure = TaskDurableResumeFailure(
            result=TaskExecutionResult(
                error={"code": "post_provider_processing_failed"}
            )
        )
        first = await coordinator_a.terminalize_completed_resume(
            provider_completion,
            failure,
            queue_item_id=claim.queue_item.queue_item_id,
            claim_token=claim_token,
            segment_id=active_segment.segment_id,
            task_run_id=run.run_id,
            request_id=initial.suspension.request_id,
            checkpoint_id=_CHECKPOINT,
            now=_NOW + timedelta(seconds=7),
        )
        assert first.completed_continuation == completed_continuation
        await interaction_store_a.aclose()
        await self._close_database(database_a)

        interaction_store_b = await durable_support._store(database_b)
        task_store_b = PgsqlTaskStore(database_b)
        queue_b = PgsqlTaskQueue(database_b)
        coordinator_b = PgsqlDurableTaskCoordinator(
            interaction_store_b,
            task_store_b,
        )
        replayed = await coordinator_b.terminalize_completed_resume(
            provider_completion,
            failure,
            queue_item_id=claim.queue_item.queue_item_id,
            claim_token=claim_token,
            segment_id=active_segment.segment_id,
            task_run_id=run.run_id,
            request_id=initial.suspension.request_id,
            checkpoint_id=_CHECKPOINT,
            now=_NOW + timedelta(seconds=8),
        )
        assert replayed == first
        assert replayed.completion.run.state is TaskRunState.FAILED
        assert replayed.completion.attempt.state is TaskAttemptState.FAILED
        assert replayed.completion.queue_item.state is TaskQueueItemState.DEAD
        assert replayed.completion.run.claim is None
        assert replayed.completion.queue_item.claim_token is None
        assert replayed.completion.queue_item.lease_expires_at is None
        final_segments = await task_store_b.list_attempt_segments(
            replayed.completion.attempt.attempt_id
        )
        assert final_segments[-1].state is TaskAttemptSegmentState.FAILED
        assert (
            final_segments[-1].resumed_from_segment_id
            == initial.suspension.segment.segment_id
        )
        no_dispatch_target = _DurableWorkerTarget()
        no_work = await TaskWorker(
            task_store_b,
            queue_b,
            target=no_dispatch_target,
            worker_id="completed-provider-restart",
            queue_name=_QUEUE,
            durable_suspension_coordinator=coordinator_b,
            clock=lambda: _NOW + timedelta(seconds=8),
        ).process_once()
        assert not no_work.processed
        assert no_dispatch_target.initial_calls == 0
        assert no_dispatch_target.resume_calls == 0
        await interaction_store_b.aclose()

    async def test_independent_connections_serialize_resolution_races(
        self,
    ) -> None:
        database_a = await self._database("resolution-race-a")
        database_b = await self._database("resolution-race-b")
        store_a = await durable_support._store(database_a)
        store_b = await durable_support._store(database_b)

        identical_request = durable_support._request("identical-race")
        identical_created = await store_a.create_durable(
            durable_support._create_command(identical_request),
            durable_support._portable(identical_request),
        )
        assert isinstance(identical_created, CreateInteractionApplied)
        identical_command = durable_support._answer(
            identical_created,
            key="identical-race-key",
        )
        identical_results = await gather(
            store_a.resolve(identical_command),
            store_b.resolve(identical_command),
            return_exceptions=True,
        )
        identical_applied = tuple(
            result
            for result in identical_results
            if isinstance(result, ResolveInteractionApplied)
        )
        identical_replayed = tuple(
            result
            for result in identical_results
            if isinstance(result, InteractionStoreReplayed)
        )
        assert len(identical_applied) == 1
        assert len(identical_replayed) == 1
        assert identical_applied[0].record == identical_replayed[0].record

        conflicting_request = durable_support._request("conflicting-race")
        conflicting_created = await store_a.create_durable(
            durable_support._create_command(conflicting_request),
            durable_support._portable(conflicting_request),
        )
        assert isinstance(conflicting_created, CreateInteractionApplied)
        conflicting_results = await gather(
            store_a.resolve(
                durable_support._answer(
                    conflicting_created,
                    key="conflicting-race-a",
                    value=True,
                )
            ),
            store_b.resolve(
                durable_support._answer(
                    conflicting_created,
                    key="conflicting-race-b",
                    value=False,
                )
            ),
            return_exceptions=True,
        )
        winners = tuple(
            result
            for result in conflicting_results
            if isinstance(result, ResolveInteractionApplied)
        )
        losers = tuple(
            result
            for result in conflicting_results
            if isinstance(result, ResolveInteractionRejected)
        )
        assert len(winners) == 1
        assert len(losers) == 1
        record = await store_a.lookup_scoped(
            ScopedInteractionLookup(
                actor=conflicting_created.command.actor,
                correlation=conflicting_created.record.correlation,
            )
        )
        assert isinstance(record, InteractionRecord)
        assert record.request.resolution is not None

        await store_b.aclose()
        await store_a.aclose()

    async def test_scoped_corruption_isolated_in_real_postgresql(self) -> None:
        database = await self._database("scoped-corruption")
        clock = durable_support._Clock()
        store = await durable_support._store(database, clock=clock)
        good_request = durable_support._request("real-scope-good")
        bad_request = durable_support._request("real-scope-bad")
        bad_request = replace(
            bad_request,
            origin=replace(
                bad_request.origin,
                run_id=good_request.origin.run_id,
                branch_id=BranchId("other-root"),
                principal=PrincipalScope(user_id=UserId("other-owner")),
            ),
        )
        good_created = await store.create(
            durable_support._create_command(good_request)
        )
        bad_created = await store.create(
            durable_support._create_command(bad_request)
        )
        assert isinstance(good_created, CreateInteractionApplied)
        assert isinstance(bad_created, CreateInteractionApplied)

        async with database.connection() as connection:
            async with connection.transaction():
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
UPDATE "interaction_records"
SET "ciphertext" = %s
WHERE "request_id" = %s
""",
                        (b"\x00", str(bad_request.request_id)),
                    )
                    await cursor.execute(
                        """
SELECT
    "ciphertext",
    "state_revision",
    "store_revision",
    "updated_at"
FROM "interaction_records"
WHERE "request_id" = %s
""",
                        (str(bad_request.request_id),),
                    )
                    unrelated_before = await cursor.fetchone()
                    assert unrelated_before is not None

        projection = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=good_created.command.actor,
                correlation=good_created.record.correlation,
            )
        )
        assert projection == good_created.record
        resolved = await store.resolve(durable_support._answer(good_created))
        assert isinstance(resolved, ResolveInteractionApplied)

        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT
    "ciphertext",
    "state_revision",
    "store_revision",
    "updated_at"
FROM "interaction_records"
WHERE "request_id" = %s
""",
                    (str(bad_request.request_id),),
                )
                unrelated_after = await cursor.fetchone()
        assert unrelated_after == unrelated_before
        assert (
            await store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=good_created.command.actor,
                    correlation=bad_created.record.correlation,
                )
            )
            is None
        )
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=bad_created.command.actor,
                    correlation=bad_created.record.correlation,
                )
            )

        maintenance_good_request = durable_support._request(
            "real-maintenance-good"
        )
        maintenance_bad_request = durable_support._request(
            "real-maintenance-bad"
        )
        maintenance_good = await store.create_durable(
            durable_support._create_command(maintenance_good_request),
            durable_support._portable(maintenance_good_request),
        )
        maintenance_bad = await store.create_durable(
            durable_support._create_command(maintenance_bad_request),
            durable_support._portable(maintenance_bad_request),
        )
        assert isinstance(maintenance_good, CreateInteractionApplied)
        assert isinstance(maintenance_bad, CreateInteractionApplied)
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
UPDATE "interaction_continuations"
SET "ciphertext" = %s
WHERE "continuation_id" = %s
""",
                    (
                        b"\x00",
                        str(maintenance_bad_request.continuation_id),
                    ),
                )
                await cursor.execute(
                    """
SELECT
    c."ciphertext",
    c."lifecycle_state",
    c."store_revision" AS "continuation_store_revision",
    r."request_state",
    r."state_revision",
    r."store_revision" AS "record_store_revision"
FROM "interaction_continuations" AS c
JOIN "interaction_records" AS r
  ON r."request_id" = c."request_id"
WHERE c."continuation_id" = %s
""",
                    (str(maintenance_bad_request.continuation_id),),
                )
                maintenance_bad_before = await cursor.fetchone()
                assert maintenance_bad_before is not None

        clock.now += timedelta(minutes=10)
        clock.monotonic += 600
        due = await store.terminalize_due(
            TerminalizeDueInteractionsCommand(maximum_results=10)
        )
        assert isinstance(due, DueInteractionsApplied)
        assert tuple(record.request.request_id for record in due.records) == (
            maintenance_good_request.request_id,
        )

        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT
    c."ciphertext",
    c."lifecycle_state",
    c."store_revision" AS "continuation_store_revision",
    r."request_state",
    r."state_revision",
    r."store_revision" AS "record_store_revision"
FROM "interaction_continuations" AS c
JOIN "interaction_records" AS r
  ON r."request_id" = c."request_id"
WHERE c."continuation_id" = %s
""",
                    (str(maintenance_bad_request.continuation_id),),
                )
                maintenance_bad_after = await cursor.fetchone()
        assert maintenance_bad_after == maintenance_bad_before
        await store.aclose()

    async def test_deadline_reads_bound_continuations_in_real_postgresql(
        self,
    ) -> None:
        database = await self._database("deadline-read-continuations")
        clock = durable_support._Clock()
        store = await durable_support._store(database, clock=clock)
        good_request = durable_support._request(
            "real-deadline-read-good",
            continuation_ttl_seconds=120,
        )
        invalid_request = durable_support._request(
            "real-deadline-read-invalid",
            continuation_ttl_seconds=60,
        )
        good = await store.create_durable(
            durable_support._create_command(good_request),
            durable_support._portable(good_request),
        )
        invalid = await store.create_durable(
            durable_support._create_command(invalid_request),
            durable_support._portable(invalid_request),
        )
        assert isinstance(good, CreateInteractionApplied)
        assert isinstance(invalid, CreateInteractionApplied)

        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
UPDATE "interaction_continuations"
SET "ciphertext" = %s
WHERE "continuation_id" = %s
""",
                    (
                        b"\x00",
                        str(invalid_request.continuation_id),
                    ),
                )

        snapshot = await store.next_deadline()

        assert snapshot.deadline is not None
        assert snapshot.deadline.request_id == good_request.request_id
        assert snapshot.deadline.monotonic_deadline == 120.0
        await store.aclose()

    async def test_continuationless_retention_removes_encrypted_scope(
        self,
    ) -> None:
        database = await self._database("continuationless-retention")
        store = await durable_support._store(
            database,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        request = durable_support._request("real-continuationless-retention")
        created = await store.create(durable_support._create_command(request))
        assert isinstance(created, CreateInteractionApplied)
        registered = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=created.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=request.origin.run_id,
                    branch_id=BranchId("real-retained-child"),
                    parent_branch_id=request.origin.branch_id,
                    principal=request.origin.principal,
                ),
            )
        )
        assert registered.store_mutation_applied

        async def encrypted_row_counts() -> tuple[int, int]:
            async with database.connection() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        """
SELECT
    (
        SELECT COUNT(*)
        FROM "interaction_records"
        WHERE "run_id" = %s
          AND OCTET_LENGTH("ciphertext") > 0
    ) AS "record_count",
    (
        SELECT COUNT(*)
        FROM "interaction_branches"
        WHERE "run_id" = %s
          AND OCTET_LENGTH("ciphertext") > 0
    ) AS "branch_count"
""",
                        (str(request.origin.run_id),) * 2,
                    )
                    row = await cursor.fetchone()
                    assert row is not None
                    return (
                        cast(int, row["record_count"]),
                        cast(int, row["branch_count"]),
                    )

        assert await encrypted_row_counts() == (1, 1)
        swept = await store.sweep(now=_NOW + timedelta(days=2))
        assert swept.invalidated == ()
        assert swept.deleted == ()
        assert await encrypted_row_counts() == (0, 0)
        assert (
            await store.lookup_scoped(
                ScopedInteractionLookup(
                    actor=created.command.actor,
                    correlation=created.record.correlation,
                )
            )
            is None
        )
        await store.aclose()

    async def test_branch_roots_use_full_real_ancestry(self) -> None:
        database = await self._database("branch-root-ancestry")
        store = await durable_support._store(database)
        request = durable_support._request("real-branch-root-ancestry")
        created = await store.create(durable_support._create_command(request))
        assert isinstance(created, CreateInteractionApplied)

        async def register(branch_id: str, parent_branch_id: str) -> None:
            result = await store.register_branch(
                RegisterInteractionBranchCommand(
                    actor=created.command.actor,
                    registration=InteractionBranchRegistration(
                        run_id=request.origin.run_id,
                        branch_id=BranchId(branch_id),
                        parent_branch_id=BranchId(parent_branch_id),
                        principal=request.origin.principal,
                    ),
                )
            )
            assert result.store_mutation_applied

        await register("real-child", str(request.origin.branch_id))
        await register("real-sibling", str(request.origin.branch_id))
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT
    "branch_id",
    "ciphertext",
    "store_revision",
    "root_branch_id"
FROM "interaction_branches"
WHERE "run_id" = %s
ORDER BY "branch_id"
""",
                    (str(request.origin.run_id),),
                )
                stable_before = tuple(await cursor.fetchall())

        await register("real-grandchild", "real-child")
        await register("real-great-grandchild", "real-grandchild")
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT
    "branch_id",
    "ciphertext",
    "store_revision",
    "root_branch_id"
FROM "interaction_branches"
WHERE "run_id" = %s
ORDER BY "branch_id"
""",
                    (str(request.origin.run_id),),
                )
                rows = tuple(await cursor.fetchall())
        expected_root = str(request.origin.branch_id)
        assert {
            str(row["branch_id"]): str(row["root_branch_id"]) for row in rows
        } == {
            "real-child": expected_root,
            "real-sibling": expected_root,
            "real-grandchild": expected_root,
            "real-great-grandchild": expected_root,
        }
        stable_after = tuple(
            row
            for row in rows
            if row["branch_id"] in {"real-child", "real-sibling"}
        )
        assert stable_after == stable_before

        other_scope_request = durable_support._request(
            "real-branch-other-scope"
        )
        other_scope_request = replace(
            other_scope_request,
            origin=replace(
                other_scope_request.origin,
                run_id=request.origin.run_id,
                branch_id=BranchId("real-other-root"),
                principal=PrincipalScope(user_id=UserId("real-other-owner")),
            ),
        )
        other_scope = await store.create(
            durable_support._create_command(other_scope_request)
        )
        assert isinstance(other_scope, CreateInteractionApplied)
        duplicate_branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=other_scope.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=request.origin.run_id,
                    branch_id=BranchId("real-child"),
                    parent_branch_id=other_scope_request.origin.branch_id,
                    principal=other_scope_request.origin.principal,
                ),
            )
        )
        assert duplicate_branch.store_mutation_applied
        first_root = await store.lookup_branch_root(
            InteractionBranchRootLookup(
                actor=created.command.actor,
                run_id=request.origin.run_id,
                branch_id=BranchId("real-child"),
            )
        )
        other_root = await store.lookup_branch_root(
            InteractionBranchRootLookup(
                actor=other_scope.command.actor,
                run_id=request.origin.run_id,
                branch_id=BranchId("real-child"),
            )
        )
        assert first_root is not None
        assert other_root is not None
        assert first_root.root_branch_id == request.origin.branch_id
        assert (
            other_root.root_branch_id == other_scope_request.origin.branch_id
        )

        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
UPDATE "interaction_branches"
SET "root_branch_id" = 'tampered-root'
WHERE "run_id" = %s
  AND "branch_id" = 'real-grandchild'
""",
                    (str(request.origin.run_id),),
                )
        with self.assertRaises(PgsqlInteractionStoreError):
            await store.lookup_branch_root(
                InteractionBranchRootLookup(
                    actor=created.command.actor,
                    run_id=request.origin.run_id,
                    branch_id=BranchId("real-great-grandchild"),
                )
            )
        await store.aclose()

    async def test_retention_is_principal_scoped_in_real_postgresql(
        self,
    ) -> None:
        database = await self._database("principal-retention")
        store = await durable_support._store(
            database,
            store_policy=PgsqlInteractionStorePolicy(retention_days=1),
        )
        shared_run = RunId("real-retention-shared")

        def request(suffix: str, owner: str) -> InputRequest:
            value = durable_support._request(f"real-retention-{suffix}")
            return replace(
                value,
                origin=replace(
                    value.origin,
                    run_id=shared_run,
                    branch_id=BranchId(f"real-root-{suffix}"),
                    principal=PrincipalScope(user_id=UserId(owner)),
                ),
            )

        first_request = request("first", "real-first-owner")
        second_request = request("second", "real-second-owner")
        first = await store.create(
            durable_support._create_command(first_request)
        )
        second = await store.create(
            durable_support._create_command(second_request)
        )
        assert isinstance(first, CreateInteractionApplied)
        assert isinstance(second, CreateInteractionApplied)
        for created, suffix in ((first, "first"), (second, "second")):
            branch = await store.register_branch(
                RegisterInteractionBranchCommand(
                    actor=created.command.actor,
                    registration=InteractionBranchRegistration(
                        run_id=shared_run,
                        branch_id=BranchId(f"real-child-{suffix}"),
                        parent_branch_id=BranchId(f"real-root-{suffix}"),
                        principal=created.command.actor.principal,
                    ),
                )
            )
            assert branch.store_mutation_applied

        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT "request_id", "scope_identity_digest"
FROM "interaction_records"
WHERE "run_id" = %s
ORDER BY "request_id"
""",
                    (str(shared_run),),
                )
                scoped_rows = tuple(await cursor.fetchall())
                assert len(scoped_rows) == 2
                assert (
                    len(
                        {
                            str(row["scope_identity_digest"])
                            for row in scoped_rows
                        }
                    )
                    == 2
                )
                await cursor.execute(
                    """
UPDATE "interaction_records"
SET "ciphertext" = %s
WHERE "request_id" = %s
""",
                    (b"\x00", str(first_request.request_id)),
                )
                await cursor.execute(
                    """
UPDATE "interaction_records"
SET "retention_deadline_at" = %s
WHERE "request_id" = %s
""",
                    (
                        _NOW + timedelta(days=10),
                        str(second_request.request_id),
                    ),
                )

        swept = await store.sweep(now=_NOW + timedelta(days=2))
        assert swept.deleted == ()
        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT "request_id"
FROM "interaction_records"
WHERE "run_id" = %s
ORDER BY "request_id"
""",
                    (str(shared_run),),
                )
                record_ids = tuple(
                    str(row["request_id"]) for row in await cursor.fetchall()
                )
                await cursor.execute(
                    """
SELECT "branch_id"
FROM "interaction_branches"
WHERE "run_id" = %s
ORDER BY "branch_id"
""",
                    (str(shared_run),),
                )
                branch_ids = tuple(
                    str(row["branch_id"]) for row in await cursor.fetchall()
                )
        assert record_ids == (str(second_request.request_id),)
        assert branch_ids == ("real-child-second",)
        projection = await store.lookup_scoped(
            ScopedInteractionLookup(
                actor=second.command.actor,
                correlation=second.record.correlation,
            )
        )
        assert projection == second.record

        race_run = RunId("real-retention-last-two-race")
        race_principal = PrincipalScope(
            user_id=UserId("real-retention-race-owner")
        )

        def race_request(suffix: str) -> InputRequest:
            value = durable_support._request(f"real-retention-race-{suffix}")
            return replace(
                value,
                reason=f"Distinct retention race request {suffix}.",
                origin=replace(
                    value.origin,
                    run_id=race_run,
                    branch_id=BranchId(f"real-retention-race-root-{suffix}"),
                    principal=race_principal,
                ),
            )

        race_first_request = race_request("first")
        race_second_request = race_request("second")
        race_first = await store.create(
            durable_support._create_command(race_first_request)
        )
        race_second = await store.create(
            durable_support._create_command(race_second_request)
        )
        assert isinstance(race_first, CreateInteractionApplied)
        assert isinstance(race_second, CreateInteractionApplied)
        race_branch = await store.register_branch(
            RegisterInteractionBranchCommand(
                actor=race_first.command.actor,
                registration=InteractionBranchRegistration(
                    run_id=race_run,
                    branch_id=BranchId("real-retention-race-child"),
                    parent_branch_id=BranchId(
                        "real-retention-race-root-first"
                    ),
                    principal=race_principal,
                ),
            )
        )
        assert race_branch.store_mutation_applied

        first_delete_database = await self._database(
            "retention-last-two-first"
        )
        second_delete_database = await self._database(
            "retention-last-two-second"
        )
        first_ready = Event()
        second_ready = Event()
        release_deletes = Event()

        async def delete_record(
            delete_database: PsycopgAsyncDatabase,
            request_id: str,
            ready: Event,
        ) -> None:
            async with delete_database.connection() as connection:
                async with connection.transaction():
                    async with connection.cursor() as cursor:
                        ready.set()
                        await release_deletes.wait()
                        await cursor.execute(
                            """
DELETE FROM "interaction_records"
WHERE "request_id" = %s
""",
                            (request_id,),
                        )

        first_delete = create_task(
            delete_record(
                first_delete_database,
                str(race_first_request.request_id),
                first_ready,
            )
        )
        second_delete = create_task(
            delete_record(
                second_delete_database,
                str(race_second_request.request_id),
                second_ready,
            )
        )
        await wait_for(
            gather(first_ready.wait(), second_ready.wait()),
            timeout=10,
        )
        release_deletes.set()
        await wait_for(
            gather(first_delete, second_delete),
            timeout=10,
        )

        async with database.connection() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
SELECT
    (
        SELECT COUNT(*)
        FROM "interaction_records"
        WHERE "run_id" = %s
    ) AS "record_count",
    (
        SELECT COUNT(*)
        FROM "interaction_branches"
        WHERE "run_id" = %s
    ) AS "branch_count"
""",
                    (str(race_run), str(race_run)),
                )
                race_counts = await cursor.fetchone()
        assert race_counts is not None
        assert race_counts["record_count"] == 0
        assert race_counts["branch_count"] == 0
        await store.aclose()

    async def _close_database(
        self,
        database: PsycopgAsyncDatabase,
    ) -> None:
        await database.aclose()
        self.databases.remove(database)

    async def _database(
        self,
        application_name: str,
    ) -> PsycopgAsyncDatabase:
        database = PsycopgAsyncDatabase(
            PsycopgPoolSettings(
                dsn=task_pgsql_psycopg_dsn(self.dsn),
                schema=self.schema,
                application_name=application_name,
            )
        )
        await database.open()
        self.databases.append(database)
        return database

    async def _runtime_bundle(
        self,
        database: PsycopgAsyncDatabase,
        *,
        worker_id: str,
        task_run_id: str,
    ) -> _RuntimeBundle:
        interaction_store = await durable_support._store(database)
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: _NOW + timedelta(seconds=3),
        )
        queue = PgsqlTaskQueue(
            database,
            clock=lambda: _NOW + timedelta(seconds=3),
        )
        record = await interaction_store.get_task_continuation_record(
            task_run_id
        )
        continuation = record.continuation
        adapter = _ResumeAdapter()
        executor = _ResumeExecutor()
        runtime = ResolvedContinuationRuntime(
            definition=continuation.definition,
            revision_binding=continuation.revision_binding,
            runtime=executor,
            operation=object(),
            model=adapter,
            tools=object(),
            capabilities=_catalog(continuation.revision_binding),
            credentials_reloaded_from_trusted_config=True,
        )
        loader = _ResumeLoader(runtime)
        resolver = ContinuationRuntimeResolver(
            loader,
            clock=lambda: _NOW + timedelta(seconds=3),
        )
        resumer = DurableAgentContinuationResumer(
            interaction_store,
            resolver,
            clock=lambda: _NOW + timedelta(seconds=3),
        )
        resume_coordinator = TaskDurableResumeCoordinator(
            interaction_store,
            resumer,
        )
        task_coordinator = PgsqlDurableTaskCoordinator(
            interaction_store,
            task_store,
        )
        target = _DurableWorkerTarget()
        worker = TaskWorker(
            task_store,
            queue,
            target=target,
            worker_id=worker_id,
            queue_name=_QUEUE,
            durable_suspension_coordinator=task_coordinator,
            durable_resume_coordinator=resume_coordinator,
            clock=lambda: _NOW + timedelta(seconds=3),
        )
        return _RuntimeBundle(
            target=target,
            adapter=adapter,
            executor=executor,
            loader=loader,
            worker=worker,
        )


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="durable_worker_e2e", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agent.toml"),
        run=TaskRunPolicy.queued(_QUEUE),
    )


def _catalog(
    binding: ContinuationRevisionBinding,
) -> ModelCapabilityCatalog:
    registry = ContinuationSnapshotCodecRegistry("durable-worker-e2e")
    registry.register(
        codec_id="durable-worker-openai-v1",
        revision_binding=binding,
        snapshot_kind="openai.responses.reasoning",
        export_snapshot=encode_continuation_snapshot,
        restore_snapshot=lambda value, expected: decode_continuation_snapshot(
            value,
            expected_binding=expected,
        ),
    )
    return ModelCapabilityCatalog.create(
        support=ProviderCapabilitySupport(
            structured_invocation=True,
            stable_call_ids=True,
            correlated_results=True,
            durable_store=True,
            registered_resumer=True,
            continuation_snapshot_codec_registry=registry,
            continuation_snapshot_codec=registry.reference(
                "durable-worker-openai-v1"
            ),
        ),
        revision_binding=binding,
    )


if __name__ == "__main__":
    main()
