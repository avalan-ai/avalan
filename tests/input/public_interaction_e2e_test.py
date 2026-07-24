"""Exercise public durable interaction behavior."""

from asyncio import gather, run
from datetime import timedelta
from json import dumps
from pathlib import Path
from sys import path as sys_path
from typing import Any, cast

sys_path.append(str(Path(__file__).parents[1] / "interaction" / "stores"))
sys_path.append(str(Path(__file__).parent))
sys_path.append(str(Path(__file__).parents[1]))

import failure_matrix_task_e2e_test as task_support  # noqa: E402
import interaction_pgsql_store_test as durable_support  # noqa: E402
from input_consumers.public_sdk_consumer import (  # noqa: E402
    complete_durable_run,
)
from pgsql_support import FakePgsqlDatabase  # noqa: E402

from avalan import (  # noqa: E402
    AgentRunCompleted,
    AgentRunInputRequired,
    AnswerProvenance,
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
    create_durable_input_integration,
    run_agent,
)
from avalan.agent.execution import (  # noqa: E402
    ExecutionInputRequiredError,
)
from avalan.interaction import (  # noqa: E402
    AnsweredResolution,
    ContinuationClaimOwnerId,
    ContinuationDispatchId,
    InputErrorCode,
    InputRequest,
    InputRequiredResult,
    InteractionCorrelation,
    ProviderIdempotencyKey,
    RequestState,
    ResolveInteractionApplied,
    ResolveInteractionRejected,
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
