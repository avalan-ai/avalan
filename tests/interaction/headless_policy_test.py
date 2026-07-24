"""Exercise explicit async headless policies and their trust boundary."""

from asyncio import (
    CancelledError,
    Event,
    create_task,
    get_running_loop,
    wait_for,
)
from asyncio import run as asyncio_run
from asyncio import sleep as asyncio_sleep
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from avalan.interaction.broker import (
    AsyncInteractionBroker,
    InteractionBrokerRequest,
)
from avalan.interaction.codec import canonical_resolution_digest
from avalan.interaction.continuation import (
    ContinuationFencingToken,
    ContinuationStoreRevision,
    PortableContinuation,
)
from avalan.interaction.durable import DurableInteractionSuspension
from avalan.interaction.entities import (
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    BranchId,
    CapabilityRevision,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ContinuationRevisionBinding,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    InputRequestId,
    ModelCallId,
    ModelConfigRevision,
    ModelId,
    PrincipalScope,
    ProviderConfigRevision,
    ProviderFamilyName,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    RunId,
    StateRevision,
    StreamSessionId,
    TurnId,
    UserId,
    create_input_request,
)
from avalan.interaction.error import (
    InputErrorCode,
    InputValidationError,
)
from avalan.interaction.handler import (
    _TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN,
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    _new_trusted_policy_input_handler_resolution,
    _TrustedInputHandlerResolution,
    _validate_trusted_input_handler_resolution,
)
from avalan.interaction.headless import (
    AsyncioDurableHandoffWaiter,
    DeclineInputPolicy,
    DurableHandoffInputPolicy,
    ExternalControllerInputPolicy,
    PolicyValueInputPolicy,
    PredeclaredInputPolicy,
    TrustedDefaultInputPolicy,
    UnavailableInputPolicy,
)
from avalan.interaction.policy import (
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionClock,
    InteractionDisclosure,
    InteractionIdFactory,
    InteractionOperation,
    InteractionPolicy,
    InteractionTime,
    TaskInputClassification,
    TaskInputClassificationRequest,
)
from avalan.interaction.state import InputTransitionError
from avalan.interaction.store import (
    _TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN,
    _TRUSTED_POLICY_RESOLVER,
    CreateInteractionCommand,
    InteractionStoreReplayed,
    ResolveInteractionCommand,
    _new_trusted_policy_resolution_command,
    _TrustedPolicyResolutionCommand,
    _validate_candidate_resolution_command,
    _validate_trusted_policy_resolution_command,
    apply_candidate_resolution,
    apply_create_interaction,
    apply_semantic_resolution_replay,
)
from avalan.interaction.stores import MemoryInteractionStoreFactory
from avalan.interaction.stores import pgsql as interaction_pgsql

_NOW = datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


def _definition() -> ExecutionDefinitionRef:
    return ExecutionDefinitionRef(
        agent_definition_locator="agent://headless-policy",
        agent_definition_revision="agent-r1",
        operation_id="operation",
        operation_index=0,
        model_config_reference="model-r1",
        tool_revision="tools-r1",
        capability_revision="capability-r1",
    )


def _origin(run_id: str = "run-1") -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId(f"turn-{run_id}"),
        agent_id=AgentId("agent-1"),
        branch_id=BranchId("branch-1"),
        model_call_id=ModelCallId(f"call-{run_id}"),
        stream_session_id=StreamSessionId(f"stream-{run_id}"),
        definition=_definition(),
        principal=PrincipalScope(),
    )


def _created_request() -> InputRequest:
    return create_input_request(
        request_id=InputRequestId("request-1"),
        continuation_id=ContinuationId("continuation-1"),
        origin=_origin(),
        mode=RequirementMode.REQUIRED,
        reason="Choose whether to continue.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=True,
            ),
        ),
        created_at=_NOW,
    )


def _pending_request() -> InputRequest:
    return replace(
        _created_request(),
        state=RequestState.PENDING,
        state_revision=StateRevision(1),
    )


def _policy_answer() -> ConfirmationAnswer:
    return ConfirmationAnswer(
        question_id=QuestionId("confirm"),
        provenance=AnswerProvenance.POLICY,
        value=True,
    )


def _policy_resolution(
    *,
    request_id: InputRequestId = InputRequestId("request-1"),
    provenance: AnswerProvenance = AnswerProvenance.POLICY,
    answer_provenance: AnswerProvenance = AnswerProvenance.POLICY,
) -> AnsweredResolution:
    return AnsweredResolution(
        request_id=request_id,
        provenance=provenance,
        resolved_at=_NOW,
        answers=(
            replace(
                _policy_answer(),
                provenance=answer_provenance,
            ),
        ),
    )


class _Clock(InteractionClock):
    async def read(self) -> InteractionTime:
        return InteractionTime.from_clock(
            wall_time=_NOW,
            monotonic_seconds=0.0,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        del monotonic_deadline
        await Event().wait()


class _Ids(InteractionIdFactory):
    def __init__(self) -> None:
        self.sequence = 0

    def _next(self, kind: str) -> str:
        self.sequence += 1
        return f"{kind}-{self.sequence}"

    async def new_request_id(self) -> InputRequestId:
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        return ActiveControlLeaseNonce(self._next("lease"))


class _Authorizer:
    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


class _UnusedClassifier:
    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        del request
        raise AssertionError("confirmation input does not need classification")


class _EscalatingHandler:
    def __init__(
        self,
        *,
        provenance: AnswerProvenance = AnswerProvenance.POLICY,
        stop_after_feedback: bool = True,
    ) -> None:
        self.contexts: list[InputHandlerContext] = []
        self.provenance = provenance
        self.stop_after_feedback = stop_after_feedback

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        self.contexts.append(context)
        if context.validation_error is not None and self.stop_after_feedback:
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.HANDLER_UNAVAILABLE
            )
        return InputHandlerResolution(
            resolution=AnsweredResolution(
                request_id=context.request.request_id,
                provenance=self.provenance,
                resolved_at=_NOW,
                answers=(
                    replace(
                        _policy_answer(),
                        provenance=self.provenance,
                    ),
                ),
            )
        )


async def _broker() -> AsyncInteractionBroker:
    policy = InteractionPolicy()
    clock = _Clock()
    ids = _Ids()
    classifier = _UnusedClassifier()
    factory = MemoryInteractionStoreFactory(
        policy=policy,
        clock=clock,
        authorizer=_Authorizer(),
        id_factory=ids,
        classifier=classifier,
    )
    return AsyncInteractionBroker(
        store=await factory.open(),
        clock=clock,
        id_factory=ids,
        policy=policy,
        classifier=classifier,
    )


def _broker_request(
    handler: object,
    *,
    run_id: str,
    default_value: bool | None = True,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=InteractionActor(principal=PrincipalScope()),
        origin=_origin(run_id),
        mode=RequirementMode.REQUIRED,
        reason="Choose whether to continue.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
                default_value=default_value,
            ),
        ),
        handler=handler,  # type: ignore[arg-type]
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


def test_trusted_headless_policies_use_sealed_broker_path() -> None:
    """Commit policy/default provenance without trusting payload labels."""

    async def exercise() -> None:
        broker = await _broker()
        try:
            policy_result = await broker.request(
                _broker_request(
                    PredeclaredInputPolicy(answers=(_policy_answer(),)),
                    run_id="policy",
                )
            )
            policy_delivery = policy_result.delivery
            assert policy_delivery is not None
            policy_resolution = policy_delivery.record.request.resolution
            assert isinstance(policy_resolution, AnsweredResolution)
            assert policy_resolution.provenance is AnswerProvenance.POLICY
            assert all(
                answer.provenance is AnswerProvenance.POLICY
                for answer in policy_resolution.answers
            )

            default_result = await broker.request(
                _broker_request(
                    TrustedDefaultInputPolicy(),
                    run_id="default",
                )
            )
            default_delivery = default_result.delivery
            assert default_delivery is not None
            default_resolution = default_delivery.record.request.resolution
            assert isinstance(default_resolution, AnsweredResolution)
            assert (
                default_resolution.provenance
                is AnswerProvenance.TRUSTED_DEFAULT
            )
        finally:
            await broker.aclose()

    asyncio_run(exercise())


def test_ordinary_handler_cannot_escalate_policy_provenance() -> None:
    """Reject trusted labels from ordinary attached handler outcomes."""

    async def exercise() -> None:
        broker = await _broker()
        handler = _EscalatingHandler()
        try:
            result = await broker.request(
                _broker_request(handler, run_id="escalation")
            )
            delivery = result.delivery
            assert delivery is not None
            assert len(handler.contexts) == 2
            assert handler.contexts[1].validation_error is not None
            assert (
                handler.contexts[1].validation_error.code
                is InputErrorCode.FORBIDDEN
            )
            resolution = delivery.record.request.resolution
            assert resolution is not None
            assert resolution.status is ResolutionStatus.UNAVAILABLE
            assert resolution.provenance not in {
                AnswerProvenance.POLICY,
                AnswerProvenance.TRUSTED_DEFAULT,
            }
        finally:
            await broker.aclose()

    asyncio_run(exercise())


@pytest.mark.parametrize(
    "provenance",
    (
        AnswerProvenance.POLICY,
        AnswerProvenance.TRUSTED_DEFAULT,
    ),
)
def test_repeated_trusted_provenance_escalation_fails_closed(
    provenance: AnswerProvenance,
) -> None:
    """Bound a handler that never corrects forbidden provenance."""

    async def exercise() -> None:
        broker = await _broker()
        handler = _EscalatingHandler(
            provenance=provenance,
            stop_after_feedback=False,
        )
        try:
            result = await wait_for(
                broker.request(
                    _broker_request(handler, run_id="repeated-escalation")
                ),
                timeout=1,
            )
            delivery = result.delivery
            assert delivery is not None
            assert delivery.handler_attempts == 2
            assert len(handler.contexts) == 2
            assert handler.contexts[0].validation_error is None
            correction = handler.contexts[1].validation_error
            assert correction is not None
            assert correction.code is InputErrorCode.FORBIDDEN
            resolution = delivery.record.request.resolution
            assert resolution is not None
            assert resolution.status is ResolutionStatus.UNAVAILABLE
            assert resolution.provenance not in {
                AnswerProvenance.POLICY,
                AnswerProvenance.TRUSTED_DEFAULT,
            }
        finally:
            await broker.aclose()

    asyncio_run(exercise())


def test_policy_variants_are_async_typed_and_explicit() -> None:
    """Exercise policy value, decline, external, and unavailable outcomes."""

    async def exercise() -> None:
        context = InputHandlerContext(request=_pending_request())

        async def provider(
            received: InputHandlerContext,
        ) -> tuple[ConfirmationAnswer, ...]:
            assert received is context
            return (_policy_answer(),)

        policy_value = await PolicyValueInputPolicy(provider=provider)(context)
        trusted = _validate_trusted_input_handler_resolution(policy_value)
        assert isinstance(trusted.resolution, AnsweredResolution)

        declined = await DeclineInputPolicy()(context)
        trusted_decline = _validate_trusted_input_handler_resolution(declined)
        assert trusted_decline.resolution is not None
        assert trusted_decline.resolution.status is ResolutionStatus.DECLINED

        async def controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=received.request.request_id,
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    resolved_at=_NOW,
                    answers=(
                        replace(
                            _policy_answer(),
                            provenance=(AnswerProvenance.EXTERNAL_CONTROLLER),
                        ),
                    ),
                )
            )

        external = await ExternalControllerInputPolicy(controller=controller)(
            context
        )
        assert isinstance(external, InputHandlerResolution)
        assert (
            external.resolution.provenance
            is AnswerProvenance.EXTERNAL_CONTROLLER
        )

        unavailable = await UnavailableInputPolicy()(context)
        assert isinstance(unavailable, InputHandlerDisconnected)
        assert unavailable.reason is InputDisconnectReason.HANDLER_UNAVAILABLE

    asyncio_run(exercise())


def test_durable_handoff_requires_exact_pending_request() -> None:
    """Keep persistence and bounded waiting separate from input expiry."""

    async def exercise() -> None:
        suspension = _suspension()
        expected = replace(
            suspension.command.request,
            state=RequestState.PENDING,
            state_revision=StateRevision(1),
        )
        waits: list[int] = []

        async def handoff(
            received: DurableInteractionSuspension,
        ) -> InputRequest:
            assert received is suspension
            return expected

        async def waiter(seconds: int) -> None:
            waits.append(seconds)

        policy = DurableHandoffInputPolicy(
            handoff=handoff,
            durable_handoff_wait_seconds=17,
            waiter=waiter,
        )
        detached = await policy(
            InputHandlerContext(request=_pending_request())
        )
        assert detached.kind.value == "detached"
        assert await policy.persist(suspension) == expected
        await policy.wait()
        assert waits == [17]
        assert (
            suspension.command.request.continuation_ttl_seconds
            == expected.continuation_ttl_seconds
        )

        async def wrong_handoff(
            received: DurableInteractionSuspension,
        ) -> InputRequest:
            del received
            return replace(expected, request_id=InputRequestId("wrong"))

        wrong = DurableHandoffInputPolicy(
            handoff=wrong_handoff,
            waiter=waiter,
        )
        with pytest.raises(InputValidationError) as raised:
            await wrong.persist(suspension)
        assert raised.value.code is InputErrorCode.CORRELATION_MISMATCH

    asyncio_run(exercise())


def test_durable_handoff_wait_is_wall_clock_bounded_and_cancellable() -> None:
    """Bound an uncooperative waiter without swallowing caller cancellation."""

    async def exercise() -> None:
        async def handoff(
            suspension: DurableInteractionSuspension,
        ) -> InputRequest:
            return replace(
                suspension.command.request,
                state=RequestState.PENDING,
                state_revision=StateRevision(1),
            )

        timed_started = Event()
        timed_cancelled = Event()

        async def ignores_budget(seconds: int) -> None:
            assert seconds == 1
            timed_started.set()
            try:
                await Event().wait()
            except CancelledError:
                timed_cancelled.set()
                return

        timed = DurableHandoffInputPolicy(
            handoff=handoff,
            durable_handoff_wait_seconds=1,
            waiter=ignores_budget,
        )
        started_at = get_running_loop().time()
        await timed.wait()
        elapsed = get_running_loop().time() - started_at
        await asyncio_sleep(0)
        assert timed_started.is_set()
        assert timed_cancelled.is_set()
        assert 0.75 <= elapsed < 1.5

        caller_started = Event()
        caller_cancelled = Event()

        async def caller_waiter(seconds: int) -> None:
            assert seconds == 30
            caller_started.set()
            try:
                await Event().wait()
            except CancelledError:
                caller_cancelled.set()
                raise

        cancellable = DurableHandoffInputPolicy(
            handoff=handoff,
            durable_handoff_wait_seconds=30,
            waiter=caller_waiter,
        )
        task = create_task(cancellable.wait())
        await caller_started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
        await asyncio_sleep(0)
        assert caller_cancelled.is_set()

    asyncio_run(exercise())


def test_policy_cancellation_notifies_async_host_callback() -> None:
    """Deliver cancellation when the containing handler task ends."""

    async def exercise() -> None:
        context = InputHandlerContext(request=_pending_request())
        started = Event()
        release = Event()
        cancelled: list[InputHandlerContext] = []

        async def provider(
            received: InputHandlerContext,
        ) -> tuple[ConfirmationAnswer, ...]:
            assert received is context
            started.set()
            await release.wait()
            return (_policy_answer(),)

        async def cancellation_handler(
            received: InputHandlerContext,
        ) -> None:
            cancelled.append(received)

        policy = PolicyValueInputPolicy(
            provider=provider,
            cancellation_handler=cancellation_handler,
        )
        task = create_task(policy(context))
        await started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
        assert cancelled == [context]
        await UnavailableInputPolicy()._cancelled(context)

    asyncio_run(exercise())


def test_policy_constructors_reject_sync_or_untrusted_values() -> None:
    """Reject sync bridges and externally relabeled policy answers."""

    def sync_provider(
        context: InputHandlerContext,
    ) -> tuple[ConfirmationAnswer, ...]:
        del context
        return (_policy_answer(),)

    with pytest.raises(InputValidationError):
        PolicyValueInputPolicy(provider=sync_provider)  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        DurableHandoffInputPolicy(
            handoff=sync_provider,  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        PredeclaredInputPolicy(
            answers=(
                replace(
                    _policy_answer(),
                    provenance=AnswerProvenance.HUMAN,
                ),
            )
        )
    with pytest.raises(InputValidationError):
        DurableHandoffInputPolicy(
            handoff=sync_provider,  # type: ignore[arg-type]
            durable_handoff_wait_seconds=0,
        )


def test_sealed_outcome_is_not_a_public_handler_resolution() -> None:
    """Keep trusted policy authority out of the ordinary outcome class."""

    async def exercise() -> None:
        outcome = await PredeclaredInputPolicy(answers=(_policy_answer(),))(
            InputHandlerContext(request=_pending_request())
        )
        assert type(outcome) is _TrustedInputHandlerResolution
        assert not isinstance(outcome, InputHandlerResolution)

    asyncio_run(exercise())


def test_sealed_outcome_rejects_forged_authority_and_shapes() -> None:
    """Validate every field before a trusted handler outcome is accepted."""
    resolution = AnsweredResolution(
        request_id=InputRequestId("request-1"),
        provenance=AnswerProvenance.POLICY,
        resolved_at=_NOW,
        answers=(_policy_answer(),),
    )
    with pytest.raises(InputValidationError):
        _TrustedInputHandlerResolution(
            resolution=resolution,
            trusted_default=False,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _TrustedInputHandlerResolution(
            resolution=None,
            trusted_default="yes",  # type: ignore[arg-type]
            _token=_TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN,
        )
    with pytest.raises(InputValidationError):
        _TrustedInputHandlerResolution(
            resolution=resolution,
            trusted_default=True,
            _token=_TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN,
        )
    with pytest.raises(InputValidationError):
        _TrustedInputHandlerResolution(
            resolution=None,
            trusted_default=False,
            _token=_TRUSTED_INPUT_HANDLER_RESOLUTION_TOKEN,
        )
    with pytest.raises(InputValidationError):
        _validate_trusted_input_handler_resolution(object())

    trusted = _new_trusted_policy_input_handler_resolution(resolution)
    assert trusted.kind.value == "resolution"
    object.__setattr__(trusted, "_authority", object())
    with pytest.raises(InputValidationError):
        _validate_trusted_input_handler_resolution(trusted)


def test_headless_validation_rejects_untyped_and_relabelled_callbacks() -> (
    None
):
    """Fail closed on sync callbacks and forged external outcomes."""

    async def exercise() -> None:
        context = InputHandlerContext(request=_pending_request())

        async def invalid_controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            del received
            return object()  # type: ignore[return-value]

        with pytest.raises(InputValidationError):
            await ExternalControllerInputPolicy(controller=invalid_controller)(
                context
            )

        async def trusted_controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            del received
            return await PredeclaredInputPolicy(answers=(_policy_answer(),))(
                context
            )

        with pytest.raises(InputValidationError):
            await ExternalControllerInputPolicy(controller=trusted_controller)(
                context
            )

        async def policy_controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=received.request.request_id,
                    provenance=AnswerProvenance.POLICY,
                    resolved_at=_NOW,
                    answers=(_policy_answer(),),
                )
            )

        with pytest.raises(InputValidationError):
            await ExternalControllerInputPolicy(controller=policy_controller)(
                context
            )

        async def mismatched_controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            return InputHandlerResolution(
                resolution=AnsweredResolution(
                    request_id=received.request.request_id,
                    provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                    resolved_at=_NOW,
                    answers=(
                        replace(
                            _policy_answer(),
                            provenance=AnswerProvenance.HUMAN,
                        ),
                    ),
                )
            )

        with pytest.raises(InputValidationError):
            await ExternalControllerInputPolicy(
                controller=mismatched_controller
            )(context)

        async def disconnected_controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            del received
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.CONTROL_CHANNEL_CLOSED
            )

        disconnected = await ExternalControllerInputPolicy(
            controller=disconnected_controller
        )(context)
        assert isinstance(disconnected, InputHandlerDisconnected)

    asyncio_run(exercise())

    def sync_controller(
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        del context
        return InputHandlerDisconnected(
            reason=InputDisconnectReason.HANDLER_UNAVAILABLE
        )

    def sync_cancellation(context: InputHandlerContext) -> None:
        del context

    with pytest.raises(InputValidationError):
        ExternalControllerInputPolicy(
            controller=sync_controller,  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        UnavailableInputPolicy(
            cancellation_handler=sync_cancellation,  # type: ignore[arg-type]
        )
    with pytest.raises(InputValidationError):
        PredeclaredInputPolicy(answers=[])  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        PredeclaredInputPolicy(answers=(object(),))  # type: ignore[arg-type]


def test_validation_feedback_allows_only_dynamic_retry() -> None:
    """Avoid static retry loops while allowing a policy provider correction."""

    async def exercise() -> None:
        error = InputTransitionError(
            code=InputErrorCode.ANSWER_TYPE_MISMATCH,
            path="answers.confirm",
            message="answer type does not match",
        )
        context = InputHandlerContext(
            request=_pending_request(),
            validation_error=error,
        )
        with pytest.raises(InputValidationError):
            await PredeclaredInputPolicy(answers=(_policy_answer(),))(context)
        with pytest.raises(InputValidationError):
            await TrustedDefaultInputPolicy()(context)

        async def corrected_provider(
            received: InputHandlerContext,
        ) -> tuple[ConfirmationAnswer, ...]:
            assert received.validation_error is error
            return (_policy_answer(),)

        outcome = await PolicyValueInputPolicy(provider=corrected_provider)(
            context
        )
        assert type(outcome) is _TrustedInputHandlerResolution

    asyncio_run(exercise())


def test_external_controller_cancellation_notifies_host() -> None:
    """Forward cancellation from a pending external controller exactly once."""

    async def exercise() -> None:
        context = InputHandlerContext(request=_pending_request())
        started = Event()
        release = Event()
        cancellations: list[InputHandlerContext] = []

        async def controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            started.set()
            await release.wait()
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.HANDLER_UNAVAILABLE
            )

        async def cancelled(received: InputHandlerContext) -> None:
            cancellations.append(received)

        policy = ExternalControllerInputPolicy(
            controller=controller,
            cancellation_handler=cancelled,
        )
        task = create_task(policy(context))
        await started.wait()
        task.cancel()
        with pytest.raises(CancelledError):
            await task
        assert cancellations == [context]

        second_started = Event()

        async def second_controller(
            received: InputHandlerContext,
        ) -> InputHandlerOutcome:
            del received
            second_started.set()
            await Event().wait()
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.HANDLER_UNAVAILABLE
            )

        async def failing_callback(
            received: InputHandlerContext,
        ) -> None:
            del received
            raise RuntimeError("callback failure")

        failing_policy = ExternalControllerInputPolicy(
            controller=second_controller,
            cancellation_handler=failing_callback,
        )
        failing_task = create_task(failing_policy(context))
        await second_started.wait()
        failing_task.cancel()
        with pytest.raises(CancelledError) as raised:
            await failing_task
        assert raised.value.__notes__ == [
            "input cancellation callback failed: RuntimeError"
        ]

    asyncio_run(exercise())


def test_durable_handoff_rejects_invalid_callbacks_and_receipts() -> None:
    """Require async persistence, async waiting, and a typed exact receipt."""

    async def exercise() -> None:
        suspension = _suspension()

        async def invalid_handoff(
            received: DurableInteractionSuspension,
        ) -> InputRequest:
            del received
            return object()  # type: ignore[return-value]

        async def waiter(seconds: int) -> None:
            del seconds

        policy = DurableHandoffInputPolicy(
            handoff=invalid_handoff,
            waiter=waiter,
        )
        with pytest.raises(InputValidationError):
            await policy.persist(suspension)
        with pytest.raises(InputValidationError):
            await policy.persist(object())  # type: ignore[arg-type]

        real_waiter = AsyncioDurableHandoffWaiter()
        wait_task = create_task(real_waiter(1))
        await asyncio_sleep(0)
        wait_task.cancel()
        with pytest.raises(CancelledError):
            await wait_task

    asyncio_run(exercise())

    async def valid_handoff(
        suspension: DurableInteractionSuspension,
    ) -> InputRequest:
        return replace(
            suspension.command.request,
            state=RequestState.PENDING,
            state_revision=StateRevision(1),
        )

    def sync_waiter(seconds: int) -> None:
        del seconds

    with pytest.raises(InputValidationError):
        DurableHandoffInputPolicy(
            handoff=valid_handoff,
            waiter=sync_waiter,  # type: ignore[arg-type]
        )


def test_trusted_policy_resolver_round_trips_pgsql_record_codec() -> None:
    """Persist sealed policy authority without relabeling it as a principal."""
    encoded = interaction_pgsql._encode_resolver(_TRUSTED_POLICY_RESOLVER)
    assert encoded == {"kind": "trusted_policy"}
    assert (
        interaction_pgsql._decode_resolver(encoded) is _TRUSTED_POLICY_RESOLVER
    )


def test_trusted_policy_store_command_is_sealed_and_strict() -> None:
    """Reject every attempt to mint or mutate trusted policy authority."""
    request = _created_request()
    actor = InteractionActor(principal=request.origin.principal)
    created = apply_create_interaction(
        CreateInteractionCommand(actor=actor, request=request),
        InteractionPolicy(),
    ).record
    correlation = created.correlation
    key = ResolutionIdempotencyKey("policy-key")
    resolution = _policy_resolution()

    with pytest.raises(InputValidationError):
        _TrustedPolicyResolutionCommand(
            actor=actor,
            correlation=correlation,
            expected_state_revision=created.request.state_revision,
            idempotency_key=key,
            proposed_resolution=resolution,
            _token=object(),
        )
    with pytest.raises(InputValidationError):
        _TrustedPolicyResolutionCommand(
            actor=actor,
            correlation=correlation,
            expected_state_revision=created.request.state_revision,
            idempotency_key=key,
            proposed_resolution=object(),  # type: ignore[arg-type]
            _token=_TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN,
        )
    with pytest.raises(InputValidationError):
        _TrustedPolicyResolutionCommand(
            actor=actor,
            correlation=correlation,
            expected_state_revision=created.request.state_revision,
            idempotency_key=key,
            proposed_resolution=_policy_resolution(
                request_id=InputRequestId("wrong-request")
            ),
            _token=_TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN,
        )
    with pytest.raises(InputValidationError):
        _TrustedPolicyResolutionCommand(
            actor=actor,
            correlation=correlation,
            expected_state_revision=created.request.state_revision,
            idempotency_key=key,
            proposed_resolution=_policy_resolution(
                provenance=AnswerProvenance.HUMAN,
                answer_provenance=AnswerProvenance.HUMAN,
            ),
            _token=_TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN,
        )
    with pytest.raises(InputValidationError):
        _TrustedPolicyResolutionCommand(
            actor=actor,
            correlation=correlation,
            expected_state_revision=created.request.state_revision,
            idempotency_key=key,
            proposed_resolution=_policy_resolution(
                answer_provenance=AnswerProvenance.HUMAN
            ),
            _token=_TRUSTED_POLICY_RESOLUTION_COMMAND_TOKEN,
        )

    command = _new_trusted_policy_resolution_command(
        actor=actor,
        correlation=correlation,
        expected_state_revision=created.request.state_revision,
        idempotency_key=key,
        proposed_resolution=resolution,
    )
    assert command.resolution_digest == canonical_resolution_digest(resolution)
    assert _validate_trusted_policy_resolution_command(command) is command
    assert _validate_candidate_resolution_command(command) is command
    with pytest.raises(InputValidationError):
        _validate_trusted_policy_resolution_command(object())

    object.__setattr__(command, "_authority", object())
    with pytest.raises(InputValidationError):
        _validate_trusted_policy_resolution_command(command)


def test_policy_candidate_uses_sealed_resolver_on_commit_and_replay() -> None:
    """Commit and replay policy provenance without principal relabeling."""
    policy = InteractionPolicy()
    request = _created_request()
    actor = InteractionActor(principal=request.origin.principal)
    pending = apply_create_interaction(
        CreateInteractionCommand(actor=actor, request=request),
        policy,
    ).record
    resolution = _policy_resolution()
    command = _new_trusted_policy_resolution_command(
        actor=actor,
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("policy-key-1"),
        proposed_resolution=resolution,
    )
    observed_at = InteractionTime.from_clock(
        wall_time=_NOW,
        monotonic_seconds=0.0,
    )
    applied = apply_candidate_resolution(
        pending,
        command,
        observed_at,
        policy,
    )
    assert applied.record.resolved_by is _TRUSTED_POLICY_RESOLVER
    assert (
        applied.record.request.resolution is not None
        and applied.record.request.resolution.provenance
        is AnswerProvenance.POLICY
    )

    wrong_actor_command = _new_trusted_policy_resolution_command(
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("other-user"))
        ),
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("wrong-actor"),
        proposed_resolution=resolution,
    )
    with pytest.raises(InputValidationError) as wrong_actor:
        apply_candidate_resolution(
            pending,
            wrong_actor_command,
            observed_at,
            policy,
        )
    assert wrong_actor.value.code is InputErrorCode.FORBIDDEN

    replay_command = _new_trusted_policy_resolution_command(
        actor=actor,
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("policy-key-2"),
        proposed_resolution=resolution,
    )
    replay = apply_semantic_resolution_replay(
        applied.record,
        replay_command,
    )
    assert isinstance(replay, InteractionStoreReplayed)
    assert replay.store_mutation_applied
    assert replay.record.resolved_by is _TRUSTED_POLICY_RESOLVER

    wrong_replay_actor = _new_trusted_policy_resolution_command(
        actor=InteractionActor(
            principal=PrincipalScope(user_id=UserId("other-user"))
        ),
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("policy-key-3"),
        proposed_resolution=resolution,
    )
    with pytest.raises(InputValidationError):
        apply_semantic_resolution_replay(
            replay.record,
            wrong_replay_actor,
        )

    public = ResolveInteractionCommand(
        actor=actor,
        correlation=pending.correlation,
        expected_state_revision=pending.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey("human-key"),
        proposed_resolution=_policy_resolution(
            provenance=AnswerProvenance.HUMAN,
            answer_provenance=AnswerProvenance.HUMAN,
        ),
    )
    assert _validate_candidate_resolution_command(public) is public

    with pytest.raises(InputValidationError):
        replace(applied.record, resolved_by=actor.principal)
