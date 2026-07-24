"""Exercise public durable interaction behavior."""

from asyncio import (
    CancelledError,
    Event,
    all_tasks,
    create_task,
    current_task,
    gather,
    run,
    sleep,
)
from datetime import timedelta
from json import dumps
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))
sys_path.append(str(Path(__file__).parent))
sys_path.append(str(Path(__file__).parents[1]))

import attached_runtime_matrix_test as matrix_support  # noqa: E402
import failure_matrix_task_e2e_test as task_support  # noqa: E402
import interaction_pgsql_store_test as durable_support  # noqa: E402
import mcp_contract_test as mcp_support  # noqa: E402
from input_consumers.public_sdk_consumer import (  # noqa: E402
    complete_durable_run,
)
from pgsql_support import FakePgsqlDatabase  # noqa: E402

from avalan import (  # noqa: E402
    AgentRunCompleted,
    AgentRunInputRequired,
    AnswerProvenance,
    AttachedInputContext,
    ConfirmationAnswer,
    DurableInputPersistenceAccepted,
    DurableInputPersistenceRequest,
    InputAnswerSubmission,
    InputContinuationRef,
    InputInspection,
    InputInspectionRequest,
    InputRequestRef,
    InputRequestView,
    InputResolutionAccepted,
    InputResolutionRequest,
    InputResolutionResult,
    Orchestrator,
    QuestionId,
    ResolutionIdempotencyKey,
    create_attached_input_runtime,
    create_durable_input_integration,
    run_agent,
)
from avalan.agent.execution import (  # noqa: E402
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionInputRequiredError,
    create_agent_execution,
    create_child_interaction_runtime,
)
from avalan.interaction import (  # noqa: E402
    AgentId,
    AnsweredResolution,
    AsyncInteractionBroker,
    ContinuationClaimOwnerId,
    ContinuationDispatchId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputErrorCode,
    InputRequest,
    InputRequiredResult,
    InteractionActor,
    InteractionCorrelation,
    InteractionExecutionScope,
    InteractionPresentationState,
    InteractionRecord,
    ListInteractionsCommand,
    ProviderIdempotencyKey,
    QuestionType,
    RequestState,
    ResolveInteractionApplied,
    ResolveInteractionRejected,
    ScopeCancellationApplied,
    TerminalizeInteractionScopeCommand,
)
from avalan.interaction.codec import encode_input_request  # noqa: E402
from avalan.interaction.continuation import (  # noqa: E402
    encode_portable_continuation,
)
from avalan.interaction.store import (  # noqa: E402
    InteractionStoreReplayed,
    ResolveInteractionCommand,
)
from avalan.interaction.stores.pgsql import (  # noqa: E402
    ContinuationStoreConflictError,
    PgsqlDurableTaskCoordinator,
    PgsqlInteractionStoreError,
)
from avalan.model.response.text import TextGenerationResponse  # noqa: E402
from avalan.model.stream import (  # noqa: E402
    CanonicalStreamItem,
    StreamItemKind,
)
from avalan.task import TaskInteractionEventType  # noqa: E402
from avalan.task.stores import PgsqlTaskStore  # noqa: E402


def test_idempotency_and_staleness() -> None:
    """Prove replay, conflict, expiry, and duplicate-claim behavior."""

    async def exercise() -> tuple[object, ...]:
        database = FakePgsqlDatabase()
        durable_support._seed_running_task(database, "run")
        interaction_store = await durable_support._store(database)
        identifiers = durable_support._Ids()
        task_store = PgsqlTaskStore(
            database,
            clock=lambda: durable_support._NOW,
            id_factory=lambda: identifiers.next("task"),
        )
        coordinator = PgsqlDurableTaskCoordinator(
            interaction_store,
            task_store,
        )
        request = durable_support._request()
        staged = await coordinator.create_and_suspend(
            durable_support._create_command(request),
            durable_support._portable(request),
            queue_item_id="queue-item",
            claim_token="claim-token",
            segment_id="segment",
            task_run_id="run",
            checkpoint_id="checkpoint",
        )
        command = durable_support._answer(staged.interaction)
        first = await coordinator.resolve_and_requeue(
            command,
            task_run_id="run",
        )
        replay = await coordinator.resolve_and_requeue(
            command,
            task_run_id="run",
        )
        conflict = await interaction_store.resolve(
            durable_support._answer(
                staged.interaction,
                value=False,
            )
        )
        assert isinstance(conflict, ResolveInteractionRejected)
        assert conflict.error.code is InputErrorCode.IDEMPOTENCY_CONFLICT
        assert not conflict.store_mutation_applied

        ready = await interaction_store.get_continuation(
            request.continuation_id
        )
        claims = await gather(
            interaction_store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("worker-a"),
                lease_expires_at=durable_support._NOW + timedelta(minutes=2),
                dispatch_id=ContinuationDispatchId("dispatch-a"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=durable_support._NOW + timedelta(seconds=2),
            ),
            interaction_store.claim(
                request.continuation_id,
                expected_store_revision=ready.store_revision,
                owner_id=ContinuationClaimOwnerId("worker-b"),
                lease_expires_at=durable_support._NOW + timedelta(minutes=2),
                dispatch_id=ContinuationDispatchId("dispatch-b"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=durable_support._NOW + timedelta(seconds=2),
            ),
            return_exceptions=True,
        )
        claim_successes = tuple(
            value for value in claims if not isinstance(value, BaseException)
        )
        claim_failures = tuple(
            value for value in claims if isinstance(value, BaseException)
        )
        assert len(claim_successes) == 1
        assert len(claim_failures) == 1
        assert isinstance(
            claim_failures[0],
            ContinuationStoreConflictError,
        )

        expired_request = durable_support._request("expired")
        expired_created = await interaction_store.create_durable(
            durable_support._create_command(expired_request),
            durable_support._portable(expired_request),
        )
        expired_resolution = await interaction_store.resolve(
            durable_support._answer(expired_created)
        )
        assert expired_resolution.record.request.resolution is not None
        expired_ready = await interaction_store.get_continuation(
            expired_request.continuation_id
        )
        expired_code = None
        try:
            await interaction_store.claim(
                expired_request.continuation_id,
                expected_store_revision=expired_ready.store_revision,
                owner_id=ContinuationClaimOwnerId("expired-worker"),
                lease_expires_at=durable_support._NOW + timedelta(minutes=12),
                dispatch_id=ContinuationDispatchId("expired-dispatch"),
                provider_idempotency_key=ProviderIdempotencyKey(
                    "provider-key"
                ),
                now=durable_support._NOW + timedelta(minutes=11),
            )
        except PgsqlInteractionStoreError as error:
            expired_code = error.code

        return (
            first.resolution.record,
            replay.resolution.record,
            conflict.error.code,
            len(claim_successes),
            type(claim_failures[0]),
            expired_code,
            tuple(row["event_type"] for row in database.events.values()),
        )

    (
        first_record,
        replay_record,
        conflict_code,
        claim_count,
        claim_error,
        expired_code,
        event_types,
    ) = run(exercise())
    assert replay_record == first_record
    assert conflict_code is InputErrorCode.IDEMPOTENCY_CONFLICT
    assert claim_count == 1
    assert claim_error is ContinuationStoreConflictError
    assert expired_code is InputErrorCode.EXPIRED
    assert event_types == (
        TaskInteractionEventType.INPUT_REQUIRED.value,
        TaskInteractionEventType.INPUT_RESUMED.value,
    )


class _PublicResponse:
    def __init__(self, value: str) -> None:
        self.value = value

    async def to_str(self) -> str:
        return self.value


def _request_view(request: InputRequest) -> InputRequestView:
    """Project one internal request into its public semantic view."""
    return InputRequestView(
        mode=request.mode,
        reason=request.reason,
        questions=request.questions,
        created_at=request.created_at,
        state=request.state,
        state_revision=request.state_revision,
    )


class _TaskDurableInputBridge:
    """Adapt the durable task harness to the public host bridge contract."""

    def __init__(
        self,
        suspended: task_support._DurableFailureHarness,
    ) -> None:
        self.suspended = suspended
        self.calls: list[str] = []
        self.request_id: InputRequestRef | None = None
        self.continuation_id: InputContinuationRef | None = None
        self.resolver_commands: list[ResolveInteractionCommand] = []

    async def persist_input(
        self,
        request: DurableInputPersistenceRequest,
    ) -> DurableInputPersistenceAccepted:
        """Verify and acknowledge the exact serialized suspension."""
        suspension = self.suspended.target.suspensions[0]
        expected_request_payload = dumps(
            encode_input_request(suspension.command.request),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        assert request.request_payload == expected_request_payload
        assert request.continuation_payload == encode_portable_continuation(
            suspension.continuation
        )
        persisted = await task_support._persisted_request(self.suspended)
        assert persisted == self.suspended.request
        assert persisted.state is RequestState.PENDING
        assert request.request == _request_view(persisted)
        self.calls.append("persist")
        self.request_id = request.request_id
        self.continuation_id = request.continuation_id
        return DurableInputPersistenceAccepted(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            persistence_digest=request.persistence_digest,
        )

    async def inspect_input(
        self,
        request: InputInspectionRequest,
    ) -> InputInspection:
        """Return the exact task-owned pending request projection."""
        self._assert_refs(request.request_id, request.continuation_id)
        persisted = await task_support._persisted_request(self.suspended)
        self.calls.append("inspect")
        return InputInspection(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            request=_request_view(persisted),
            detached_resumption_available=True,
        )

    async def resolve_input(
        self,
        request: InputResolutionRequest,
    ) -> InputResolutionResult:
        """Resolve and requeue the exact task continuation atomically."""
        self._assert_refs(request.request_id, request.continuation_id)
        assert isinstance(request.submission, InputAnswerSubmission)
        persisted = await task_support._persisted_request(self.suspended)
        command = ResolveInteractionCommand(
            actor=self.suspended.target.suspensions[0].command.actor,
            correlation=InteractionCorrelation.from_request(persisted),
            expected_state_revision=persisted.state_revision,
            idempotency_key=request.idempotency_key,
            proposed_resolution=AnsweredResolution(
                request_id=persisted.request_id,
                provenance=request.submission.provenance,
                resolved_at=self.suspended.clock.now,
                answers=request.submission.answers,
            ),
        )
        self.resolver_commands.append(command)
        reentry = await self.suspended.coordinator.resolve_and_requeue(
            command,
            task_run_id=self.suspended.run_id,
            now=self.suspended.clock.now,
        )
        resolution = reentry.resolution
        assert isinstance(
            resolution,
            ResolveInteractionApplied | InteractionStoreReplayed,
        )
        assert resolution.record.request.state is RequestState.ANSWERED
        self.calls.append("resolve")
        return InputResolutionResult(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            resolution=InputResolutionAccepted(
                interaction_state="answered",
                idempotent=isinstance(
                    resolution,
                    InteractionStoreReplayed,
                ),
            ),
        )

    def _assert_refs(
        self,
        request_id: InputRequestRef,
        continuation_id: InputContinuationRef,
    ) -> None:
        """Require the exact references accepted during persistence."""
        assert request_id == self.request_id
        assert continuation_id == self.continuation_id


class _InputRequiredResponse:
    """Raise one exact input-required signal during materialization."""

    def __init__(self, error: ExecutionInputRequiredError) -> None:
        self.error = error

    async def to_str(self) -> str:
        """Raise the input-required signal owned by this response."""
        raise self.error


class _InputRequiredOrchestrator:
    """Return a response that raises one exact input-required signal."""

    def __init__(self, error: ExecutionInputRequiredError) -> None:
        self.error = error

    async def __call__(self, input: object, **kwargs: object) -> object:
        del input, kwargs
        return _InputRequiredResponse(self.error)


class _RequiredOrchestrator:
    async def __call__(self, input: object, **kwargs: object) -> Any:
        del input, kwargs
        request = durable_support._request("public-required")
        error = ExecutionInputRequiredError(
            InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=False,
            ),
            request=request,
        )
        return _InputRequiredResponse(error)


class _AdvisoryOrchestrator:
    async def __call__(
        self,
        input: object,
        **kwargs: object,
    ) -> _PublicResponse:
        del input, kwargs
        return _PublicResponse("continued-after-advisory-timeout")


class _MultiAgentHandler:
    """Drive one invalid sibling answer and one valid child answer."""

    def __init__(self) -> None:
        self.contexts: list[AttachedInputContext] = []
        self.presentation_order: list[QuestionId] = []
        self.reviewer_presented = Event()
        self.submit_wrong_sibling = Event()
        self.wrong_sibling_rejected = Event()
        self.resolve_reviewer = Event()
        self.planner_presented = Event()
        self.planner_cancelled = Event()
        self.active_presentations = 0
        self.maximum_active_presentations = 0

    async def __call__(
        self,
        context: AttachedInputContext,
    ) -> InputAnswerSubmission:
        """Return one externally authored answer for the presented child."""
        question_id = context.request.questions[0].question_id
        self.contexts.append(context)
        self.presentation_order.append(question_id)
        self.active_presentations += 1
        self.maximum_active_presentations = max(
            self.maximum_active_presentations,
            self.active_presentations,
        )
        try:
            if question_id == QuestionId("confirmation"):
                self.reviewer_presented.set()
                if context.validation_error is None:
                    await self.submit_wrong_sibling.wait()
                    answer_id = QuestionId("text")
                else:
                    assert (
                        context.validation_error.code
                        is InputErrorCode.UNKNOWN_QUESTION
                    )
                    self.wrong_sibling_rejected.set()
                    await self.resolve_reviewer.wait()
                    answer_id = question_id
                return InputAnswerSubmission(
                    answers=(
                        ConfirmationAnswer(
                            question_id=answer_id,
                            provenance=AnswerProvenance.HUMAN,
                            value=True,
                        ),
                    ),
                    provenance=AnswerProvenance.HUMAN,
                )

            assert question_id == QuestionId("text")
            assert context.validation_error is None
            self.planner_presented.set()
            try:
                await Event().wait()
            except CancelledError:
                self.planner_cancelled.set()
                raise
            raise AssertionError("planner input must be cancelled")
        finally:
            self.active_presentations -= 1


class _MultiAgentOriginOrchestrator:
    """Run isolated child requests through the public attached runtime."""

    def __init__(self, handler: _MultiAgentHandler) -> None:
        self.handler = handler
        self.parent_origin: ExecutionOrigin | None = None
        self.child_origins: tuple[ExecutionOrigin, ...] = ()
        self.context_labels: tuple[str | None, ...] = ()
        self.provider_calls: tuple[int, ...] = ()
        self.continuation_calls: tuple[int, ...] = ()
        self.tasks_completed = False

    @staticmethod
    def _harness(
        runtime: AttachedInteractionRuntime,
        responses: list[TextGenerationResponse],
    ) -> Any:
        broker = runtime.broker
        assert isinstance(broker, AsyncInteractionBroker)
        harness = matrix_support._Harness(
            broker=broker,
            clock=matrix_support._Clock(),
            handler=runtime.handler,
            responses=responses,
            tool=matrix_support._empty_tool_manager(),
            tool_confirm=None,
        )
        harness.runtime = runtime
        return harness

    @staticmethod
    async def _records(
        broker: AsyncInteractionBroker,
        origin: ExecutionOrigin,
        count: int,
    ) -> tuple[InteractionRecord, ...]:
        for _ in range(100):
            projections = await broker.list(
                ListInteractionsCommand(
                    actor=InteractionActor(principal=origin.principal),
                    scope=InteractionExecutionScope(run_id=origin.run_id),
                )
            )
            records = tuple(
                item
                for item in projections
                if isinstance(item, InteractionRecord)
            )
            if len(records) == count:
                return records
            await sleep(0)
        raise AssertionError("child interactions were not admitted")

    async def __call__(
        self,
        input: object,
        **kwargs: object,
    ) -> _PublicResponse:
        del input
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, AttachedInteractionRuntime)
        broker = runtime.broker
        assert isinstance(broker, AsyncInteractionBroker)
        definition = ExecutionDefinitionRef(
            agent_definition_locator="agent://public-multi",
            agent_definition_revision="agent-r1",
            operation_id="public-multi",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        )
        parent = await create_agent_execution(
            definition=definition,
            agent_id=AgentId("parent"),
            principal=runtime.actor.principal,
            initial_messages=(),
            interaction_runtime=runtime,
        )
        planner_runtime, reviewer_runtime = await gather(
            create_child_interaction_runtime(
                runtime,
                parent_origin=parent.origin,
                context_label="Planner child",
            ),
            create_child_interaction_runtime(
                runtime,
                parent_origin=parent.origin,
                context_label="Reviewer child",
            ),
        )
        assert isinstance(planner_runtime, AttachedInteractionRuntime)
        assert isinstance(reviewer_runtime, AttachedInteractionRuntime)
        planner = self._harness(
            planner_runtime,
            [
                matrix_support._task_input_response(
                    "planner",
                    matrix_support._input_arguments(
                        matrix_support._question(QuestionType.TEXT)
                    ),
                ),
                matrix_support._provider_response(
                    "planner-final",
                    answer="planner-complete",
                ),
            ],
        )
        reviewer = self._harness(
            reviewer_runtime,
            [
                matrix_support._task_input_response(
                    "reviewer",
                    matrix_support._input_arguments(
                        matrix_support._question(QuestionType.CONFIRMATION)
                    ),
                ),
                matrix_support._provider_response(
                    "reviewer-final",
                    answer="reviewer-complete",
                ),
            ],
        )
        unrelated = self._harness(
            runtime,
            [
                matrix_support._provider_response(
                    "unrelated",
                    answer="unrelated-complete",
                )
            ],
        )
        reviewer_response = await reviewer.response()
        planner_response = await planner.response()
        unrelated_response = await unrelated.response()
        reviewer_execution = reviewer_response._execution
        planner_execution = planner_response._execution
        unrelated_execution = unrelated_response._execution
        assert reviewer_execution is not None
        assert planner_execution is not None
        assert unrelated_execution is not None
        self.parent_origin = parent.origin
        self.child_origins = (
            planner_execution.origin,
            reviewer_execution.origin,
        )
        self.context_labels = (
            planner_runtime.context_label,
            reviewer_runtime.context_label,
        )

        reviewer_task = create_task(
            matrix_support._consume(reviewer_response),
            name="public-reviewer-child",
        )
        await self.handler.reviewer_presented.wait()
        planner_task = create_task(
            matrix_support._consume(planner_response),
            name="public-planner-child",
        )
        records = await self._records(broker, parent.origin, 2)
        by_label = {record.request.context_label: record for record in records}
        planner_record = by_label["Planner child"]
        reviewer_record = by_label["Reviewer child"]
        assert (
            planner_record.request.request_id
            != reviewer_record.request.request_id
        )
        assert (
            planner_record.request.continuation_id
            != reviewer_record.request.continuation_id
        )
        assert planner_record.request.state is RequestState.PENDING
        assert reviewer_record.request.state is RequestState.PENDING
        assert (
            planner_record.presentation is InteractionPresentationState.QUEUED
        )
        assert (
            reviewer_record.presentation
            is InteractionPresentationState.PRESENTED
        )

        self.handler.submit_wrong_sibling.set()
        await self.handler.wrong_sibling_rejected.wait()
        records = await self._records(broker, parent.origin, 2)
        assert all(
            record.request.state is RequestState.PENDING for record in records
        )
        assert reviewer.engine_agent.await_count == 1
        assert planner.engine_agent.await_count == 1

        self.handler.resolve_reviewer.set()
        reviewer_items = await reviewer_task
        await self.handler.planner_presented.wait()
        records = await self._records(broker, parent.origin, 2)
        by_label = {record.request.context_label: record for record in records}
        assert (
            by_label["Reviewer child"].request.state is RequestState.ANSWERED
        )
        assert by_label["Planner child"].request.state is RequestState.PENDING
        assert reviewer.engine_agent.await_count == 2
        assert planner.engine_agent.await_count == 1
        reviewer_kinds = tuple(item.kind for item in reviewer_items)
        assert StreamItemKind.MODEL_CONTINUATION_COMPLETED in reviewer_kinds
        assert StreamItemKind.STREAM_COMPLETED in reviewer_kinds

        assert await parent.cancel()
        parent_broker = parent.interaction_broker
        assert parent_broker is not None
        command = TerminalizeInteractionScopeCommand(
            actor=runtime.actor,
            scope=InteractionExecutionScope(
                run_id=parent.origin.run_id,
                branch_id=parent.origin.branch_id,
                include_descendants=True,
            ),
            provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
        )
        cancellation = await parent_broker.cancel_scope(command)
        assert isinstance(cancellation.store_result, ScopeCancellationApplied)
        assert cancellation.store_result.command == command
        await self.handler.planner_cancelled.wait()
        planner_items: list[CanonicalStreamItem] = await planner_task
        assert StreamItemKind.INTERACTION_CANCELLED in tuple(
            item.kind for item in planner_items
        )
        unrelated_items = await matrix_support._consume(unrelated_response)
        assert (
            "".join(
                item.text_delta or ""
                for item in unrelated_items
                if item.kind is StreamItemKind.ANSWER_DELTA
            )
            == "unrelated-complete"
        )
        assert unrelated_execution.origin.run_id != parent.origin.run_id
        assert unrelated_execution.status is AgentExecutionStatus.COMPLETED

        records = await self._records(broker, parent.origin, 2)
        by_label = {record.request.context_label: record for record in records}
        assert (
            by_label["Planner child"].request.state is RequestState.CANCELLED
        )
        assert (
            by_label["Reviewer child"].request.state is RequestState.ANSWERED
        )
        assert reviewer_execution.status is AgentExecutionStatus.COMPLETED
        assert planner_execution.status is AgentExecutionStatus.CANCELLED
        self.provider_calls = (
            planner.engine_agent.await_count,
            reviewer.engine_agent.await_count,
            unrelated.engine_agent.await_count,
        )
        self.continuation_calls = (
            len(planner.contexts) - 1,
            len(reviewer.contexts) - 1,
            len(unrelated.contexts) - 1,
        )
        self.tasks_completed = planner_task.done() and reviewer_task.done()
        return _PublicResponse("children-isolated")


def test_fully_headless_run() -> None:
    """Resolve, requeue, and complete one persisted continuation."""

    async def exercise() -> None:
        suspended = await task_support._durable_failure_harness(
            task_support._confirmation()
        )
        suspension = suspended.target.suspensions[0]
        bridge = _TaskDurableInputBridge(suspended)
        integration = create_durable_input_integration(
            bridge,
            handoff_wait_seconds=1,
        )

        error = ExecutionInputRequiredError(
            InputRequiredResult(
                request_id=suspension.command.request.request_id,
                continuation_id=suspension.command.request.continuation_id,
                detached_resumption_available=True,
            ),
            durable=suspension,
        )

        async def resume_continuation(
            pause: AgentRunInputRequired,
        ) -> AgentRunCompleted[str]:
            assert pause.request_id is not None
            assert pause.continuation_id is not None
            assert len(bridge.resolver_commands) == 1
            command = bridge.resolver_commands[0]
            assert command.correlation.request_id == (
                suspended.request.request_id
            )
            assert command.correlation.continuation_id == (
                suspended.request.continuation_id
            )
            assert str(command.correlation.run_id) == suspended.run_id
            queued = await suspended.client.inspect(suspended.run_id)
            assert queued.run.state.value == "queued"
            resumed = await task_support._resume_harness(suspended)
            processed = await resumed.worker.process_once()
            assert processed.completion is not None
            assert processed.completion.run.run_id == suspended.run_id
            assert processed.completion.run.state.value == "succeeded"
            assert len(resumed.executor.commands) == 1
            resume_command = resumed.executor.commands[0]
            assert resume_command.request.request_id == (
                suspended.request.request_id
            )
            assert resume_command.request.continuation_id == (
                suspended.request.continuation_id
            )
            assert resume_command.continuation.origin.run_id == (
                suspended.request.origin.run_id
            )
            assert suspended.target.domain_side_effects == ["resumed output"]
            return AgentRunCompleted(
                value=cast(str, suspended.target.domain_side_effects[0])
            )

        try:
            result = await complete_durable_run(
                cast(
                    Orchestrator,
                    _InputRequiredOrchestrator(error),
                ),
                "run",
                interaction_runtime=integration.runtime,
                policy=integration.headless_policy,
                controller=integration.controller,
                submission=InputAnswerSubmission(
                    answers=(
                        ConfirmationAnswer(
                            question_id=QuestionId("answer"),
                            provenance=AnswerProvenance.HUMAN,
                            value=True,
                        ),
                    ),
                    provenance=AnswerProvenance.HUMAN,
                ),
                idempotency_key=ResolutionIdempotencyKey("public-e2e-answer"),
                resume_continuation=resume_continuation,
            )
            assert bridge.calls == ["persist", "inspect", "resolve"]
            assert result.pause.detached_resumption_available
            assert result.inspection.detached_resumption_available
            assert result.inspection.request_id == result.pause.request_id
            assert (
                result.inspection.continuation_id
                == result.pause.continuation_id
            )
            assert not result.resolution.idempotent
            assert result.completion.to_str() == "resumed output"
            persisted = await task_support._persisted_request(suspended)
            assert persisted.request_id == suspended.request.request_id
            assert (
                persisted.continuation_id == suspended.request.continuation_id
            )
            final = await suspended.client.inspect(suspended.run_id)
            assert final.run.state.value == "succeeded"
        finally:
            await integration.runtime.aclose()
            await suspended.stack.aclose()
            suspended.temporary.cleanup()

    run(exercise())


def test_required_versus_advisory() -> None:
    """Expose required suspension while advisory execution can complete."""

    async def exercise() -> None:
        required = await run_agent(
            cast(Any, _RequiredOrchestrator()),
            "required",
        )
        advisory = await run_agent(
            cast(Any, _AdvisoryOrchestrator()),
            "advisory",
        )
        assert isinstance(required, AgentRunInputRequired)
        assert not required.detached_resumption_available
        assert required.request_id is None
        assert required.continuation_id is None
        assert isinstance(advisory, AgentRunCompleted)
        assert advisory.to_str() == "continued-after-advisory-timeout"

    run(exercise())


def test_multi_agent_origin() -> None:
    """Isolate real sibling requests, continuations, and parent cleanup."""

    async def exercise() -> None:
        handler = _MultiAgentHandler()
        runtime = await create_attached_input_runtime(handler)
        orchestrator = _MultiAgentOriginOrchestrator(handler)
        try:
            result = await run_agent(
                cast(Any, orchestrator),
                "delegate",
                interaction_runtime=runtime,
            )
            assert isinstance(result, AgentRunCompleted)
            assert result.to_str() == "children-isolated"
            parent = orchestrator.parent_origin
            assert parent is not None
            planner, reviewer = orchestrator.child_origins
            assert planner.run_id == reviewer.run_id == parent.run_id
            assert planner.task_id == reviewer.task_id == parent.task_id
            assert planner.parent_branch_id == parent.branch_id
            assert reviewer.parent_branch_id == parent.branch_id
            assert planner.branch_id != reviewer.branch_id
            assert orchestrator.context_labels == (
                "Planner child",
                "Reviewer child",
            )
            assert handler.presentation_order == (
                [
                    QuestionId("confirmation"),
                    QuestionId("confirmation"),
                    QuestionId("text"),
                ]
            )
            assert handler.maximum_active_presentations == 1
            assert orchestrator.provider_calls == (1, 2, 1)
            assert orchestrator.continuation_calls == (0, 1, 0)
            assert orchestrator.tasks_completed
        finally:
            await runtime.aclose()
        await sleep(0)
        active = current_task()
        assert all(task is active or task.done() for task in all_tasks())

    run(exercise())


def test_mcp_projection() -> None:
    """Run the negotiated inbound, downstream, and durable MCP paths."""
    run(mcp_support._public_projection())
