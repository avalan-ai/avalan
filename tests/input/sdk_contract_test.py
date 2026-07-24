"""Exercise the typed asynchronous public SDK contract."""

from asyncio import (
    CancelledError,
    Event,
    create_task,
    gather,
    run,
    sleep,
    wait_for,
)
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from typing import Any, cast

import pytest

from avalan import (
    AgentRunCompleted,
    AgentRunInputRequired,
    AnswerProvenance,
    ConfirmationAnswer,
    InputAnswerSubmission,
    InputAuthorizationError,
    InputDeclineSubmission,
    InputNotFoundError,
    InputResolutionAccepted,
    InputValidationError,
    ResolutionIdempotencyKey,
    inspect_input,
    resolve_input,
    run_agent,
)
from avalan.agent.execution import (
    AttachedInteractionRuntime,
    ExecutionInputRequiredError,
)
from avalan.interaction.broker import InteractionBrokerResult
from avalan.interaction.continuation import (
    ContinuationFencingToken,
    ContinuationStoreRevision,
    PortableContinuation,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import (
    AgentId,
    AnsweredResolution,
    BranchId,
    CapabilityRevision,
    ConfirmationQuestion,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    InputRequestId,
    InputRequiredResult,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    QuestionId,
    RequestState,
    RequirementMode,
    RunId,
    StateRevision,
    StreamSessionId,
    TurnId,
    UserId,
    create_input_request,
)
from avalan.interaction.error import InputErrorCode
from avalan.interaction.handler import (
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerOutcome,
    InputHandlerResolution,
)
from avalan.interaction.headless import DurableHandoffInputPolicy
from avalan.interaction.policy import (
    InteractionActor,
    InteractionPolicy,
    InteractionTime,
)
from avalan.interaction.store import (
    CreateInteractionCommand,
    InteractionCorrelation,
    InteractionRecord,
    InteractionReplayKind,
    InteractionStoreReplayed,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ScopedInteractionLookup,
    apply_candidate_resolution,
    apply_create_interaction,
)
from avalan.sdk import AsyncInputController

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def _origin() -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("run-sdk"),
        turn_id=TurnId("turn-sdk"),
        agent_id=AgentId("agent-sdk"),
        branch_id=BranchId("branch-sdk"),
        model_call_id=ModelCallId("call-sdk"),
        stream_session_id=StreamSessionId("stream-sdk"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://sdk-contract",
            agent_definition_revision="agent-r1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capability-r1",
        ),
        principal=PrincipalScope(),
    )


def _created_request() -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-sdk"),
        continuation_id=ContinuationId("continuation-sdk"),
        origin=_origin(),
        mode=RequirementMode.REQUIRED,
        reason="Choose whether to continue.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        created_at=_NOW,
    )


def _answer(
    provenance: AnswerProvenance = AnswerProvenance.HUMAN,
) -> ConfirmationAnswer:
    return ConfirmationAnswer(
        question_id=QuestionId("confirm"),
        provenance=provenance,
        value=True,
    )


def _suspension() -> DurableInteractionSuspension:
    request = _created_request()
    revision = ContinuationRevisionBinding(
        provider_family=ProviderFamilyName("provider"),
        model_id=ModelId("model"),
        provider_config_revision=ProviderConfigRevision("provider-r1"),
        model_config_revision=ModelConfigRevision("model-r1"),
        capability_revision=CapabilityRevision("capability-r1"),
    )
    continuation = PortableContinuation(
        continuation_id=request.continuation_id,
        request_id=request.request_id,
        origin=request.origin,
        provider_call_id=request.origin.model_call_id,
        provider_call_correlation_id=str(request.origin.model_call_id),
        definition=request.origin.definition,
        operation_cursor=0,
        generation_settings={},
        transcript=(),
        observations=(),
        revision_binding=revision,
        interaction_count=1,
        tool_loop_count=0,
        stream_sequence=0,
        state_revision=StateRevision(0),
        store_revision=ContinuationStoreRevision(0),
        created_at=_NOW,
        updated_at=_NOW,
        expires_at=_NOW + timedelta(days=1),
        fencing_token=ContinuationFencingToken(0),
    )
    return DurableInteractionSuspension(
        command=CreateInteractionCommand(
            actor=InteractionActor(principal=request.origin.principal),
            request=request,
        ),
        continuation=continuation,
    )


class _Clock:
    async def read(self) -> InteractionTime:
        return InteractionTime.from_clock(
            wall_time=_NOW + timedelta(seconds=2),
            monotonic_seconds=2.0,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        del monotonic_deadline
        await Event().wait()


class _Broker:
    def __init__(self, record: InteractionRecord) -> None:
        self.record = record
        self.policy = InteractionPolicy()
        self.observed_at = InteractionTime.from_clock(
            wall_time=_NOW + timedelta(seconds=2),
            monotonic_seconds=2.0,
        )
        self.resolve_calls: list[ResolveInteractionCommand] = []

    async def inspect(
        self,
        query: ScopedInteractionLookup,
    ) -> InteractionRecord | None:
        if (
            query.actor.principal != self.record.request.origin.principal
            or query.correlation != self.record.correlation
        ):
            return None
        return self.record

    async def resolve(
        self,
        command: ResolveInteractionCommand,
    ) -> InteractionBrokerResult:
        self.resolve_calls.append(command)
        if self.record.request.state is RequestState.PENDING:
            applied = apply_candidate_resolution(
                self.record,
                command,
                self.observed_at,
                self.policy,
            )
            assert hasattr(applied, "record")
            self.record = applied.record
            return InteractionBrokerResult(store_result=applied)
        replay = InteractionStoreReplayed(
            command=command,
            record=self.record,
            replay_kind=InteractionReplayKind.SAME_KEY,
        )
        return InteractionBrokerResult(store_result=replay)


class _InputRequiredResponse:
    def __init__(
        self,
        error: ExecutionInputRequiredError,
    ) -> None:
        self.error = error

    async def to_str(self) -> str:
        raise self.error


class _InputRequiredOrchestrator:
    def __init__(
        self,
        error: ExecutionInputRequiredError,
    ) -> None:
        self.error = error

    async def __call__(self, input: object, **kwargs: object) -> Any:
        del input, kwargs
        return _InputRequiredResponse(self.error)


class _EmbeddedRuntimeBroker:
    async def request(self, request: object) -> object:
        del request
        raise AssertionError("embedded isolation harness cannot request input")

    async def cancel_scope(self, command: object) -> object:
        del command
        raise AssertionError("embedded isolation harness cannot cancel input")


class _CompletedResponse:
    def __init__(self, value: str) -> None:
        self.value = value

    async def to_str(self) -> str:
        return self.value


class _ConcurrentEmbeddedOrchestrator:
    def __init__(self) -> None:
        self.entered = 0
        self.release = Event()
        self.runtimes: dict[str, AttachedInteractionRuntime] = {}

    async def __call__(
        self,
        input: object,
        **kwargs: object,
    ) -> _CompletedResponse:
        assert isinstance(input, str)
        runtime = kwargs["interaction_runtime"]
        assert isinstance(runtime, AttachedInteractionRuntime)
        self.runtimes[input] = runtime
        self.entered += 1
        if self.entered == 2:
            self.release.set()
        await self.release.wait()
        outcome = await runtime.handler(
            InputHandlerContext(request=_pending_record().request)
        )
        assert isinstance(outcome, InputHandlerDetached)
        return _CompletedResponse(input)


def _pending_record() -> InteractionRecord:
    request = _created_request()
    return apply_create_interaction(
        CreateInteractionCommand(
            actor=InteractionActor(principal=request.origin.principal),
            request=request,
        ),
        InteractionPolicy(),
    ).record


async def _durable_public_pause(
    broker: _Broker,
    order: list[str],
) -> AgentRunInputRequired:
    suspension = _suspension()

    async def handoff(
        received: DurableInteractionSuspension,
    ) -> InputRequest:
        assert received == suspension
        order.append("persist")
        broker.record = _pending_record()
        return broker.record.request

    async def waiter(seconds: int) -> None:
        assert seconds == 1
        order.append("wait")

    error = ExecutionInputRequiredError(
        InputRequiredResult(
            request_id=suspension.command.request.request_id,
            continuation_id=suspension.command.request.continuation_id,
            detached_resumption_available=True,
        ),
        durable=suspension,
    )
    result = await run_agent(
        cast(Any, _InputRequiredOrchestrator(error)),
        "run",
        headless_policy=DurableHandoffInputPolicy(
            handoff=handoff,
            durable_handoff_wait_seconds=1,
            waiter=waiter,
        ),
    )
    assert isinstance(result, AgentRunInputRequired)
    return result


def test_requirement_input_n_064() -> None:
    """Support async resolve, wait, detach, and cancellation handlers."""

    async def exercise() -> None:
        context = InputHandlerContext(request=_pending_record().request)
        release = Event()
        cancelled = Event()

        async def resolve_now(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            assert received is context
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=received.request.request_id,
                    provenance=AnswerProvenance.HUMAN,
                    resolved_at=_NOW,
                    answers=(_answer(),),
                )
            )

        async def wait_for_external(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            assert received is context
            try:
                await release.wait()
            except CancelledError:
                cancelled.set()
                raise
            return InputHandlerDetached()

        async def detach(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            assert received is context
            return InputHandlerDetached()

        handlers: tuple[InputHandler, ...] = (
            cast(InputHandler, resolve_now),
            cast(InputHandler, wait_for_external),
            cast(InputHandler, detach),
        )
        immediate = await handlers[0](context)
        waiting = create_task(handlers[1](context))
        await sleep(0)
        assert not waiting.done()
        waiting.cancel()
        with pytest.raises(CancelledError):
            await waiting
        assert cancelled.is_set()
        assert isinstance(immediate, InputHandlerResolution)
        assert isinstance(await handlers[2](context), InputHandlerDetached)

    run(exercise())


def test_requirement_input_n_065() -> None:
    """Keep a pending SDK handler from blocking unrelated async work."""

    async def exercise() -> None:
        context = InputHandlerContext(request=_pending_record().request)
        release = Event()
        unrelated: list[str] = []

        async def pending(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            assert received is context
            await release.wait()
            return InputHandlerDetached()

        task = create_task(pending(context))
        await sleep(0)
        unrelated.append("ran")
        assert unrelated == ["ran"]
        assert not task.done()
        release.set()
        assert isinstance(await task, InputHandlerDetached)

    run(exercise())


def test_requirement_input_n_066() -> None:
    """Emit opaque durable refs only after persistence and bounded waiting."""

    async def exercise() -> None:
        broker = _Broker(_pending_record())
        order: list[str] = []
        result = await _durable_public_pause(broker, order)
        assert order == ["persist", "wait"]
        assert result.detached_resumption_available
        assert result.request_id is not None
        assert result.continuation_id is not None

        async def durable_authority(
            correlation: InteractionCorrelation,
        ) -> bool:
            return correlation == broker.record.correlation

        controller = AsyncInputController(
            broker=cast(Any, broker),
            actor=InteractionActor(
                principal=broker.record.request.origin.principal
            ),
            clock=_Clock(),
            durable_authority=durable_authority,
        )
        inspection = await inspect_input(
            controller,
            result.request_id,
            result.continuation_id,
        )
        assert inspection.request.reason == result.request.reason
        assert inspection.detached_resumption_available

        tampered = type(result.request_id)(f"{result.request_id}x")
        with pytest.raises(InputValidationError) as malformed:
            await inspect_input(
                controller,
                tampered,
                result.continuation_id,
            )
        assert malformed.value.code is InputErrorCode.INVALID_FORMAT
        non_ascii = type(result.request_id)(
            "avl-input-v1.non-ascii-\N{SNOWMAN}.checksum"
        )
        with pytest.raises(InputValidationError) as unicode_error:
            await inspect_input(
                controller,
                non_ascii,
                result.continuation_id,
            )
        assert unicode_error.value.code is InputErrorCode.INVALID_FORMAT
        invalid_payload = b"a"
        invalid_checksum = sha256(
            b"avalan.public-input-ref.v1\x00" + invalid_payload
        ).hexdigest()
        malformed_payload = type(result.request_id)(
            f"avl-input-v1.{invalid_payload.decode()}.{invalid_checksum}"
        )
        with pytest.raises(InputValidationError) as payload_error:
            await inspect_input(
                controller,
                malformed_payload,
                result.continuation_id,
            )
        assert payload_error.value.code is InputErrorCode.INVALID_FORMAT

        with pytest.raises(InputNotFoundError):
            await AsyncInputController(
                broker=cast(Any, broker),
                actor=InteractionActor(
                    principal=PrincipalScope(user_id=UserId("other"))
                ),
                clock=_Clock(),
            ).inspect_input(result.request_id, result.continuation_id)

    run(exercise())


def test_requirement_input_n_067() -> None:
    """Allow explicit decline and reject unsafe untyped submissions."""

    async def exercise() -> None:
        broker = _Broker(_pending_record())
        result = await _durable_public_pause(broker, [])
        assert result.request_id is not None
        assert result.continuation_id is not None

        async def durable_authority(
            correlation: InteractionCorrelation,
        ) -> bool:
            return correlation == broker.record.correlation

        async def durable_resolver(
            command: ResolveInteractionCommand,
        ) -> ResolveInteractionApplied | InteractionStoreReplayed:
            store_result = (await broker.resolve(command)).store_result
            assert isinstance(
                store_result,
                ResolveInteractionApplied | InteractionStoreReplayed,
            )
            return store_result

        controller = AsyncInputController(
            broker=cast(Any, broker),
            actor=InteractionActor(principal=PrincipalScope()),
            clock=_Clock(),
            durable_authority=durable_authority,
            durable_resolver=durable_resolver,
        )
        accepted = await resolve_input(
            controller,
            result.request_id,
            result.continuation_id,
            InputDeclineSubmission(),
            idempotency_key=ResolutionIdempotencyKey("decline"),
        )
        assert accepted == InputResolutionAccepted(
            interaction_state="declined",
            idempotent=False,
        )
        replayed = await resolve_input(
            controller,
            result.request_id,
            result.continuation_id,
            InputDeclineSubmission(),
            idempotency_key=ResolutionIdempotencyKey("decline"),
        )
        assert replayed == InputResolutionAccepted(
            interaction_state="declined",
            idempotent=True,
        )
        assert len(broker.resolve_calls) == 2

        with pytest.raises(InputValidationError):
            await controller.resolve_input(
                result.request_id,
                result.continuation_id,
                cast(Any, {"answers": []}),
                idempotency_key=ResolutionIdempotencyKey("unsafe"),
            )
        with pytest.raises(InputValidationError):
            InputAnswerSubmission(
                answers=(_answer(),),
                provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            )

    run(exercise())


def test_durable_resolution_requires_authority_and_atomic_resolver() -> None:
    """Fail closed and route authorized durable resolution atomically."""

    async def exercise() -> None:
        refs_broker = _Broker(_pending_record())
        pause = await _durable_public_pause(refs_broker, [])
        assert pause.request_id is not None
        assert pause.continuation_id is not None
        submission = InputAnswerSubmission(
            answers=(_answer(),),
            provenance=AnswerProvenance.HUMAN,
        )
        key = ResolutionIdempotencyKey("durable-answer")
        resolver_calls: list[ResolveInteractionCommand] = []

        async def resolver(
            command: ResolveInteractionCommand,
        ) -> ResolveInteractionApplied:
            resolver_calls.append(command)
            raise AssertionError("unauthorized resolver invocation")

        unauthorized = _Broker(_pending_record())
        with pytest.raises(InputAuthorizationError):
            await AsyncInputController(
                broker=cast(Any, unauthorized),
                actor=InteractionActor(principal=PrincipalScope()),
                clock=_Clock(),
                durable_resolver=resolver,
            ).resolve_input(
                pause.request_id,
                pause.continuation_id,
                submission,
                idempotency_key=key,
            )
        assert resolver_calls == []
        assert unauthorized.resolve_calls == []

        async def false_authority(
            correlation: InteractionCorrelation,
        ) -> bool:
            assert correlation == unauthorized.record.correlation
            return False

        with pytest.raises(InputAuthorizationError):
            await AsyncInputController(
                broker=cast(Any, unauthorized),
                actor=InteractionActor(principal=PrincipalScope()),
                clock=_Clock(),
                durable_authority=false_authority,
                durable_resolver=resolver,
            ).resolve_input(
                pause.request_id,
                pause.continuation_id,
                submission,
                idempotency_key=key,
            )
        assert resolver_calls == []
        assert unauthorized.resolve_calls == []

        async def true_authority(
            correlation: InteractionCorrelation,
        ) -> bool:
            return correlation == unauthorized.record.correlation

        with pytest.raises(InputValidationError) as unavailable:
            await AsyncInputController(
                broker=cast(Any, unauthorized),
                actor=InteractionActor(principal=PrincipalScope()),
                clock=_Clock(),
                durable_authority=true_authority,
            ).resolve_input(
                pause.request_id,
                pause.continuation_id,
                submission,
                idempotency_key=key,
            )
        assert unavailable.value.code is InputErrorCode.UNAVAILABLE
        assert unauthorized.resolve_calls == []

        async def invalid_authority(
            correlation: InteractionCorrelation,
        ) -> object:
            assert correlation == unauthorized.record.correlation
            return "authorized"

        with pytest.raises(InputValidationError) as invalid_authority_error:
            await AsyncInputController(
                broker=cast(Any, unauthorized),
                actor=InteractionActor(principal=PrincipalScope()),
                clock=_Clock(),
                durable_authority=invalid_authority,  # type: ignore[arg-type]
                durable_resolver=resolver,
            ).resolve_input(
                pause.request_id,
                pause.continuation_id,
                submission,
                idempotency_key=key,
            )
        assert (
            invalid_authority_error.value.code is InputErrorCode.INVALID_TYPE
        )
        assert resolver_calls == []

        async def invalid_resolver(
            command: ResolveInteractionCommand,
        ) -> Any:
            assert command.correlation == unauthorized.record.correlation
            return object()

        with pytest.raises(InputValidationError) as invalid_result:
            await AsyncInputController(
                broker=cast(Any, unauthorized),
                actor=InteractionActor(principal=PrincipalScope()),
                clock=_Clock(),
                durable_authority=true_authority,
                durable_resolver=invalid_resolver,
            ).resolve_input(
                pause.request_id,
                pause.continuation_id,
                submission,
                idempotency_key=key,
            )
        assert invalid_result.value.code is InputErrorCode.INVALID_TYPE
        assert unauthorized.resolve_calls == []

        other_request = replace(
            _created_request(),
            request_id=InputRequestId("other-request"),
            continuation_id=ContinuationId("other-continuation"),
            origin=replace(_origin(), run_id=RunId("other-run")),
        )
        other_record = apply_create_interaction(
            CreateInteractionCommand(
                actor=InteractionActor(
                    principal=other_request.origin.principal
                ),
                request=other_request,
            ),
            InteractionPolicy(),
        ).record
        other_command = ResolveInteractionCommand(
            actor=InteractionActor(principal=PrincipalScope()),
            correlation=other_record.correlation,
            expected_state_revision=other_record.request.state_revision,
            idempotency_key=ResolutionIdempotencyKey("other-answer"),
            proposed_resolution=AnsweredResolution(
                request_id=other_record.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
                answers=(_answer(),),
            ),
        )
        other_applied = apply_candidate_resolution(
            other_record,
            other_command,
            unauthorized.observed_at,
            unauthorized.policy,
        )
        assert isinstance(other_applied, ResolveInteractionApplied)

        async def mismatched_resolver(
            command: ResolveInteractionCommand,
        ) -> ResolveInteractionApplied:
            assert command.correlation == unauthorized.record.correlation
            return other_applied

        with pytest.raises(InputValidationError) as mismatch:
            await AsyncInputController(
                broker=cast(Any, unauthorized),
                actor=InteractionActor(principal=PrincipalScope()),
                clock=_Clock(),
                durable_authority=true_authority,
                durable_resolver=mismatched_resolver,
            ).resolve_input(
                pause.request_id,
                pause.continuation_id,
                submission,
                idempotency_key=key,
            )
        assert mismatch.value.code is InputErrorCode.CORRELATION_MISMATCH
        assert unauthorized.resolve_calls == []

        durable = _Broker(_pending_record())
        requeued: list[InteractionCorrelation] = []

        async def durable_authority(
            correlation: InteractionCorrelation,
        ) -> bool:
            return correlation == durable.record.correlation

        async def durable_resolver(
            command: ResolveInteractionCommand,
        ) -> ResolveInteractionApplied:
            applied = apply_candidate_resolution(
                durable.record,
                command,
                durable.observed_at,
                durable.policy,
            )
            assert isinstance(applied, ResolveInteractionApplied)
            durable.record = applied.record
            requeued.append(command.correlation)
            return applied

        accepted = await AsyncInputController(
            broker=cast(Any, durable),
            actor=InteractionActor(principal=PrincipalScope()),
            clock=_Clock(),
            durable_authority=durable_authority,
            durable_resolver=durable_resolver,
        ).resolve_input(
            pause.request_id,
            pause.continuation_id,
            submission,
            idempotency_key=key,
        )
        assert accepted == InputResolutionAccepted(
            interaction_state="answered",
            idempotent=False,
        )
        assert requeued == [durable.record.correlation]
        assert durable.resolve_calls == []

    run(exercise())


def test_durable_controller_rejects_sync_callbacks() -> None:
    """Reject synchronous authority and resolver bridges at construction."""

    def sync_authority(correlation: InteractionCorrelation) -> bool:
        del correlation
        return True

    def sync_resolver(
        command: ResolveInteractionCommand,
    ) -> ResolveInteractionApplied:
        del command
        raise AssertionError("sync resolver must not run")

    with pytest.raises(InputValidationError) as authority_error:
        AsyncInputController(
            broker=cast(Any, _Broker(_pending_record())),
            actor=InteractionActor(principal=PrincipalScope()),
            clock=_Clock(),
            durable_authority=sync_authority,  # type: ignore[arg-type]
        )
    assert authority_error.value.code is InputErrorCode.INVALID_TYPE
    with pytest.raises(InputValidationError) as resolver_error:
        AsyncInputController(
            broker=cast(Any, _Broker(_pending_record())),
            actor=InteractionActor(principal=PrincipalScope()),
            clock=_Clock(),
            durable_resolver=sync_resolver,  # type: ignore[arg-type]
        )
    assert resolver_error.value.code is InputErrorCode.INVALID_TYPE


def test_multiple_embedded_runs_keep_distinct_handlers() -> None:
    """Keep concurrent public SDK runtimes and handlers isolated by run."""

    async def exercise() -> None:
        first_calls: list[InputHandlerContext] = []
        second_calls: list[InputHandlerContext] = []

        async def first_handler(
            context: InputHandlerContext,
        ) -> InputHandlerOutcome:
            first_calls.append(context)
            return InputHandlerDetached()

        async def second_handler(
            context: InputHandlerContext,
        ) -> InputHandlerOutcome:
            second_calls.append(context)
            return InputHandlerDetached()

        first_runtime = AttachedInteractionRuntime(
            broker=cast(Any, _EmbeddedRuntimeBroker()),
            actor=InteractionActor(principal=PrincipalScope()),
            handler=first_handler,
        )
        second_runtime = AttachedInteractionRuntime(
            broker=cast(Any, _EmbeddedRuntimeBroker()),
            actor=InteractionActor(principal=PrincipalScope()),
            handler=second_handler,
        )
        orchestrator = _ConcurrentEmbeddedOrchestrator()

        first_result, second_result = await wait_for(
            gather(
                run_agent(
                    cast(Any, orchestrator),
                    "first",
                    interaction_runtime=first_runtime,
                ),
                run_agent(
                    cast(Any, orchestrator),
                    "second",
                    interaction_runtime=second_runtime,
                ),
            ),
            timeout=1,
        )

        assert isinstance(first_result, AgentRunCompleted)
        assert isinstance(second_result, AgentRunCompleted)
        assert first_result.to_str() == "first"
        assert second_result.to_str() == "second"
        assert orchestrator.runtimes == {
            "first": first_runtime,
            "second": second_runtime,
        }
        assert first_runtime.handler is first_handler
        assert second_runtime.handler is second_handler
        assert len(first_calls) == 1
        assert len(second_calls) == 1
        assert first_calls[0] is not second_calls[0]

    run(exercise())
