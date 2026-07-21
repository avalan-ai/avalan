"""Exercise the concrete broker requirements owned by Phase 2."""

from asyncio import (
    CancelledError,
    Event,
    Future,
    create_task,
    gather,
    get_running_loop,
    run,
)
from collections.abc import Callable
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta

from avalan.interaction import (
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    BranchId,
    CancelInteractionApplied,
    CancelInteractionCommand,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    CreateInteractionApplied,
    DeclinedResolution,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputAnswer,
    InputDisconnectReason,
    InputErrorCode,
    InputHandler,
    InputHandlerContext,
    InputHandlerDetached,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputRequestId,
    InputResumer,
    InputResumptionNotification,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionClock,
    InteractionCorrelation,
    InteractionDelivery,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionObserverEvent,
    InteractionOperation,
    InteractionPolicy,
    InteractionPresentationState,
    InteractionRecord,
    InteractionReplayKind,
    InteractionRequestAuthorizationTarget,
    InteractionRequestResult,
    InteractionStoreReplayed,
    InteractionTime,
    ListInteractionsCommand,
    ModelCallId,
    ParticipantId,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionDecisionStage,
    ResolutionIdempotencyKey,
    ResolutionStatus,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    RunId,
    ScopedInteractionLookup,
    ScopeSupersessionApplied,
    SessionId,
    StreamSessionId,
    SupersedeInteractionScopeCommand,
    TaskId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TenantId,
    TextAnswer,
    TurnId,
    UserId,
)
from avalan.interaction.stores import MemoryInteractionStoreFactory

_NOW = datetime(2026, 7, 21, 12, 0, tzinfo=UTC)


async def _yield_once() -> None:
    """Yield one explicit scheduler turn without a timer."""
    loop = get_running_loop()
    ready = loop.create_future()
    loop.call_soon(ready.set_result, None)
    await ready


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    turns: int = 200,
) -> None:
    """Wait a bounded number of scheduler turns for one condition."""
    for _ in range(turns):
        if predicate():
            return
        await _yield_once()
    raise AssertionError("event-loop condition did not become true")


class _Clock(InteractionClock):
    """Provide coherent observations and manually advanced deadline waits."""

    def __init__(self) -> None:
        self.wall_time = _NOW
        self.monotonic_seconds = 0.0
        self.wait_calls: list[float] = []
        self._waiters: list[tuple[float, Future[None]]] = []

    async def read(self) -> InteractionTime:
        """Return the current trusted time."""
        return InteractionTime.from_clock(
            wall_time=self.wall_time,
            monotonic_seconds=self.monotonic_seconds,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait for explicit advancement to the requested deadline."""
        self.wait_calls.append(monotonic_deadline)
        if self.monotonic_seconds >= monotonic_deadline:
            return
        future = get_running_loop().create_future()
        entry = (monotonic_deadline, future)
        self._waiters.append(entry)
        try:
            await future
        except CancelledError:
            raise
        finally:
            if entry in self._waiters:
                self._waiters.remove(entry)

    def advance(self, seconds: float) -> None:
        """Advance both time domains and release reached waiters."""
        assert seconds >= 0.0
        self.wall_time += timedelta(seconds=seconds)
        self.monotonic_seconds += seconds
        for deadline, future in tuple(self._waiters):
            if self.monotonic_seconds >= deadline and not future.done():
                future.set_result(None)


class _IdFactory(InteractionIdFactory):
    """Mint deterministic broker-owned opaque identifiers."""

    def __init__(self) -> None:
        self.sequence = 0
        self.request_ids: list[InputRequestId] = []
        self.continuation_ids: list[ContinuationId] = []

    def _next(self, kind: str) -> str:
        self.sequence += 1
        return f"contract-{kind}-{self.sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return one request identifier."""
        value = InputRequestId(self._next("request"))
        self.request_ids.append(value)
        return value

    async def new_continuation_id(self) -> ContinuationId:
        """Return one continuation identifier."""
        value = ContinuationId(self._next("continuation"))
        self.continuation_ids.append(value)
        return value

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return one idempotency key."""
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return one active-control nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _Classifier(TaskInputClassifier):
    """Allow normalized values under one trusted policy binding."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.sequence = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return an exact echoed allow classification."""
        self.sequence += 1
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"classification-{self.sequence}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Authorizer:
    """Authorize exact operations and retain their immutable targets."""

    def __init__(self) -> None:
        self.calls: list[
            tuple[
                InteractionActor,
                InteractionOperation,
                InteractionAuthorizationTarget,
            ]
        ] = []
        self.denied_operations: set[InteractionOperation] = set()

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return disclosure bound to the exact authorization input."""
        self.calls.append((actor, operation, target))
        allowed = operation not in self.denied_operations
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=allowed,
            disclosure=(
                InteractionDisclosure.FULL
                if allowed
                else InteractionDisclosure.NONE
            ),
        )


class _Resumer(InputResumer):
    """Record every broker-owned continuation attempt."""

    def __init__(self) -> None:
        self.notifications: list[InputResumptionNotification] = []
        self.called = Event()

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Record one committed continuation notification."""
        self.notifications.append(notification)
        self.called.set()


class _DetachedHandler(InputHandler):
    """Present once and explicitly hand durable handling back to the broker."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return the only typed detached outcome."""
        self.contexts.append(context)
        return InputHandlerDetached()


class _DisconnectedHandler(InputHandler):
    """Report loss of an advertised attached input capability."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return one typed capability-loss outcome."""
        self.contexts.append(context)
        return InputHandlerDisconnected(
            reason=InputDisconnectReason.HANDLER_UNAVAILABLE,
        )


class _GateHandler(InputHandler):
    """Expose presentation state while a typed renderer outcome is pending."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait for release and return a typed detached outcome."""
        self.contexts.append(context)
        self.started.set()
        await self.release.wait()
        return InputHandlerDetached()


class _CorrectionHandler(InputHandler):
    """Submit one meaning-changing answer type, then a valid correction."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return an invalid typed answer before the valid canonical type."""
        self.contexts.append(context)
        answer: InputAnswer
        if len(self.contexts) == 1:
            answer = TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value="replace the confirmation meaning",
            )
        else:
            answer = ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            )
        return InputHandlerResolution(
            resolution=AnsweredResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW + timedelta(days=1),
                answers=(answer,),
            )
        )


class _FailingObserver:
    """Record content-safe events and fail without lifecycle authority."""

    def __init__(self) -> None:
        self.events: list[InteractionObserverEvent] = []

    async def __call__(self, event: InteractionObserverEvent) -> None:
        """Record and fail one best-effort observer delivery."""
        self.events.append(event)
        raise RuntimeError("observer cannot alter broker state")


@dataclass(slots=True)
class _Harness:
    """Hold one concrete broker and its trusted deterministic adapters."""

    broker: AsyncInteractionBroker
    clock: _Clock
    ids: _IdFactory
    authorizer: _Authorizer


async def _harness(
    *,
    observer: _FailingObserver | None = None,
) -> _Harness:
    """Open the public broker over the concrete memory store factory."""
    policy = InteractionPolicy()
    clock = _Clock()
    ids = _IdFactory()
    authorizer = _Authorizer()
    classifier = _Classifier(policy)
    factory = MemoryInteractionStoreFactory(
        policy=policy,
        clock=clock,
        authorizer=authorizer,
        id_factory=ids,
        classifier=classifier,
    )
    broker = AsyncInteractionBroker(
        store=await factory.open(),
        clock=clock,
        id_factory=ids,
        policy=policy,
        classifier=classifier,
        observer=observer,
    )
    return _Harness(
        broker=broker,
        clock=clock,
        ids=ids,
        authorizer=authorizer,
    )


def _principal() -> PrincipalScope:
    return PrincipalScope(
        user_id=UserId("user"),
        tenant_id=TenantId("tenant"),
        participant_id=ParticipantId("participant"),
        session_id=SessionId("session"),
    )


def _actor() -> InteractionActor:
    return InteractionActor(principal=_principal())


def _origin(run_id: str) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId(run_id),
        turn_id=TurnId(f"turn-{run_id}"),
        task_id=TaskId(f"task-{run_id}"),
        agent_id=AgentId("agent"),
        branch_id=BranchId("root"),
        model_call_id=ModelCallId(f"call-{run_id}"),
        stream_session_id=StreamSessionId(f"stream-{run_id}"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://broker-contract",
            agent_definition_revision="revision-1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-1",
            tool_revision="tools-1",
            capability_revision="capabilities-1",
        ),
        principal=_principal(),
    )


def _request(
    handler: InputHandler | None,
    *,
    run_id: str,
    resumer: InputResumer | None = None,
    mode: RequirementMode = RequirementMode.REQUIRED,
    advisory_wait_seconds: int | None = None,
    continuation_ttl_seconds: int = 600,
    reason: str,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=_actor(),
        origin=_origin(run_id),
        mode=mode,
        reason=reason,
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        handler=handler,
        resumer=resumer,
        continuation_ttl_seconds=continuation_ttl_seconds,
        advisory_wait_seconds=advisory_wait_seconds,
    )


def _correlation(
    request_id: InputRequestId,
    continuation_id: ContinuationId,
    origin: ExecutionOrigin,
) -> InteractionCorrelation:
    return InteractionCorrelation(
        request_id=request_id,
        continuation_id=continuation_id,
        run_id=origin.run_id,
        turn_id=origin.turn_id,
        task_id=origin.task_id,
        agent_id=origin.agent_id,
        branch_id=origin.branch_id,
        model_call_id=origin.model_call_id,
    )


async def _inspect(
    broker: AsyncInteractionBroker,
    correlation: InteractionCorrelation,
) -> InteractionRecord:
    """Return one authorized full projection from the concrete store."""
    projection = await broker.inspect(
        ScopedInteractionLookup(actor=_actor(), correlation=correlation)
    )
    assert isinstance(projection, InteractionRecord)
    return projection


def _decline(
    record: InteractionRecord,
    key: str,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=DeclinedResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(hours=1),
        ),
    )


def _answer(
    record: InteractionRecord,
    key: str,
    *,
    value: bool,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW + timedelta(hours=1),
            answers=(
                ConfirmationAnswer(
                    question_id=QuestionId("confirm"),
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                ),
            ),
        ),
    )


def test_requirement_input_n_019() -> None:
    """Keep the complete current interaction lifecycle broker-owned."""

    async def exercise() -> tuple[object, ...]:
        observer = _FailingObserver()
        harness = await _harness(observer=observer)
        handler = _DetachedHandler()
        resumer = _Resumer()
        request = _request(
            handler,
            run_id="n019",
            resumer=resumer,
            reason="Broker ownership contract.",
        )
        try:
            result = await harness.broker.request(request)
            assert isinstance(result, InteractionRequestResult)
            assert isinstance(result.create_result, CreateInteractionApplied)
            assert isinstance(result.delivery, InteractionDelivery)
            delivery = result.delivery
            assert harness.ids.request_ids == [delivery.correlation.request_id]
            assert harness.ids.continuation_ids == [
                delivery.correlation.continuation_id
            ]
            assert delivery.record.request.origin == request.origin
            assert delivery.record.correlation == delivery.correlation
            assert delivery.record.request.created_at == _NOW
            assert delivery.record.presentation is (
                InteractionPresentationState.DETACHED
            )
            assert delivery.record.request.state is RequestState.PENDING
            assert len(handler.contexts) == 1

            persisted = await _inspect(
                harness.broker,
                delivery.correlation,
            )
            assert persisted == delivery.record
            listed = await harness.broker.list(
                ListInteractionsCommand(
                    actor=_actor(),
                    scope=InteractionExecutionScope(
                        run_id=request.origin.run_id,
                    ),
                )
            )
            assert listed == (persisted,)

            command = _decline(persisted, "n019-key")
            resolved = await harness.broker.resolve(command)
            assert isinstance(
                resolved.store_result,
                ResolveInteractionApplied,
            )
            assert len(resumer.notifications) == 1
            replay = await harness.broker.resolve(command)
            assert isinstance(replay.store_result, InteractionStoreReplayed)
            assert replay.store_result.replay_kind is (
                InteractionReplayKind.SAME_KEY
            )
            assert len(resumer.notifications) == 1

            await _wait_until(lambda: bool(observer.events))
            assert tuple(item.name for item in fields(observer.events[0])) == (
                "kind",
                "request_id",
                "status",
                "schedule_revision",
                "dropped_events",
            )
            assert (
                await _inspect(
                    harness.broker,
                    delivery.correlation,
                )
                == resolved.store_result.record
            )

            correction_handler = _CorrectionHandler()
            corrected = await harness.broker.request(
                _request(
                    correction_handler,
                    run_id="n019-correction",
                    reason="Available capability with correction.",
                )
            )
            assert isinstance(corrected.delivery, InteractionDelivery)
            assert corrected.delivery.handler_attempts == 2
            assert len(correction_handler.contexts) == 2
            assert correction_handler.contexts[0].validation_error is None
            correction_error = correction_handler.contexts[1].validation_error
            assert correction_error is not None
            assert correction_error.code is InputErrorCode.ANSWER_TYPE_MISMATCH
            correction_resolution = (
                corrected.delivery.record.request.resolution
            )
            assert isinstance(correction_resolution, AnsweredResolution)
            assert correction_resolution.status is ResolutionStatus.ANSWERED

            unavailable = await harness.broker.request(
                _request(
                    None,
                    run_id="n019-unavailable",
                    reason="No attached capability is available.",
                )
            )
            assert isinstance(unavailable.delivery, InteractionDelivery)
            assert unavailable.delivery.handler_attempts == 0
            unavailable_resolution = (
                unavailable.delivery.record.request.resolution
            )
            assert unavailable_resolution is not None
            assert (
                unavailable_resolution.status is ResolutionStatus.UNAVAILABLE
            )

            disconnected_handler = _DisconnectedHandler()
            disconnected = await harness.broker.request(
                _request(
                    disconnected_handler,
                    run_id="n019-disconnected",
                    reason="Advertised capability is lost.",
                )
            )
            assert isinstance(disconnected.delivery, InteractionDelivery)
            assert disconnected.delivery.handler_attempts == 1
            assert len(disconnected_handler.contexts) == 1
            disconnected_resolution = (
                disconnected.delivery.record.request.resolution
            )
            assert disconnected_resolution is not None
            assert (
                disconnected_resolution.status is ResolutionStatus.UNAVAILABLE
            )

            cancel_resumer = _Resumer()
            cancellable = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n019-cancel",
                    resumer=cancel_resumer,
                    reason="Broker-owned cancellation.",
                )
            )
            assert isinstance(cancellable.delivery, InteractionDelivery)
            cancelled = await harness.broker.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=cancellable.delivery.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=(
                        cancellable.delivery.record.request.state_revision
                    ),
                )
            )
            assert isinstance(cancelled.store_result, CancelInteractionApplied)
            await cancel_resumer.called.wait()
            assert len(cancel_resumer.notifications) == 1
            cancelled_resolution = (
                cancelled.store_result.record.request.resolution
            )
            assert cancelled_resolution is not None
            assert cancelled_resolution.status is ResolutionStatus.CANCELLED

            timeout_resumer = _Resumer()
            timed = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n019-timeout",
                    resumer=timeout_resumer,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=5,
                    reason="Broker-owned advisory timeout.",
                )
            )
            expiry_resumer = _Resumer()
            expiring = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n019-expiry",
                    resumer=expiry_resumer,
                    continuation_ttl_seconds=60,
                    reason="Broker-owned absolute expiry.",
                )
            )
            assert isinstance(timed.delivery, InteractionDelivery)
            assert isinstance(expiring.delivery, InteractionDelivery)
            await _wait_until(lambda: 5.0 in harness.clock.wait_calls)
            harness.clock.advance(5)
            await timeout_resumer.called.wait()
            await _wait_until(lambda: 60.0 in harness.clock.wait_calls)
            harness.clock.advance(55)
            await expiry_resumer.called.wait()
            timed_record = await _inspect(
                harness.broker,
                timed.delivery.correlation,
            )
            expiry_record = await _inspect(
                harness.broker,
                expiring.delivery.correlation,
            )
            timed_resolution = timed_record.request.resolution
            expiry_resolution = expiry_record.request.resolution
            assert timed_resolution is not None
            assert expiry_resolution is not None
            assert timed_resolution.status is ResolutionStatus.TIMED_OUT
            assert expiry_resolution.status is ResolutionStatus.EXPIRED
            assert len(timeout_resumer.notifications) == 1
            assert len(expiry_resumer.notifications) == 1

            observer_fields = tuple(
                item.name for item in fields(observer.events[0])
            )
            assert observer_fields == (
                "kind",
                "request_id",
                "status",
                "schedule_revision",
                "dropped_events",
            )
            return (
                delivery.record.request.state,
                replay.store_result.replay_kind,
                corrected.delivery.handler_attempts,
                correction_resolution.status,
                unavailable_resolution.status,
                disconnected_resolution.status,
                cancelled_resolution.status,
                timed_resolution.status,
                expiry_resolution.status,
                len(resumer.notifications),
                len(cancel_resumer.notifications),
                len(timeout_resumer.notifications),
                len(expiry_resumer.notifications),
                observer_fields,
            )
        finally:
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (
        RequestState.PENDING,
        InteractionReplayKind.SAME_KEY,
        2,
        ResolutionStatus.ANSWERED,
        ResolutionStatus.UNAVAILABLE,
        ResolutionStatus.UNAVAILABLE,
        ResolutionStatus.CANCELLED,
        ResolutionStatus.TIMED_OUT,
        ResolutionStatus.EXPIRED,
        1,
        1,
        1,
        1,
        (
            "kind",
            "request_id",
            "status",
            "schedule_revision",
            "dropped_events",
        ),
    )


def test_requirement_input_n_020() -> None:
    """Limit an attached renderer to presentation and typed outcomes."""

    async def exercise() -> tuple[object, ...]:
        harness = await _harness()
        handler = _GateHandler()
        resumer = _Resumer()
        request = _request(
            handler,
            run_id="n020",
            resumer=resumer,
            reason="Renderer presentation boundary.",
        )
        task = create_task(harness.broker.request(request))
        try:
            await handler.started.wait()
            assert len(harness.ids.request_ids) == 1
            correlation = _correlation(
                harness.ids.request_ids[0],
                harness.ids.continuation_ids[0],
                request.origin,
            )
            presented = await _inspect(harness.broker, correlation)
            assert presented.presentation is (
                InteractionPresentationState.PRESENTED
            )
            assert presented.request.state is RequestState.PENDING
            assert tuple(
                item.name for item in fields(handler.contexts[0])
            ) == (
                "request",
                "validation_error",
            )
            assert handler.contexts[0].request == presented.request
            assert handler.contexts[0].validation_error is None

            handler.release.set()
            result = await task
            assert isinstance(result.delivery, InteractionDelivery)
            assert result.delivery.handler_attempts == 1
            assert result.delivery.record.presentation is (
                InteractionPresentationState.DETACHED
            )
            assert result.delivery.record.request.state is RequestState.PENDING
            assert resumer.notifications == []
            return (
                presented.presentation,
                result.delivery.handler_attempts,
                result.delivery.record.presentation,
                result.delivery.record.request.state,
                len(resumer.notifications),
            )
        finally:
            handler.release.set()
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (
        InteractionPresentationState.PRESENTED,
        1,
        InteractionPresentationState.DETACHED,
        RequestState.PENDING,
        0,
    )


def test_requirement_input_n_021() -> None:
    """Reject renderer attempts to redefine question meaning or lifecycle."""

    async def exercise() -> tuple[object, ...]:
        harness = await _harness()
        handler = _CorrectionHandler()
        request = _request(
            handler,
            run_id="n021",
            reason="Immutable request meaning.",
        )
        try:
            result = await harness.broker.request(request)
            assert isinstance(result.delivery, InteractionDelivery)
            assert result.delivery.handler_attempts == 2
            assert len(handler.contexts) == 2
            first, second = handler.contexts
            assert first.request == second.request
            assert first.request.origin == request.origin
            assert first.request.reason == request.reason
            assert first.request.questions == request.questions
            assert first.request.state is RequestState.PENDING
            assert second.request.state is RequestState.PENDING
            assert first.validation_error is None
            assert second.validation_error is not None
            assert second.validation_error.code is (
                InputErrorCode.ANSWER_TYPE_MISMATCH
            )

            record = result.delivery.record
            assert record.request.state is RequestState.ANSWERED
            assert record.request.resolution is not None
            assert record.request.resolution.resolved_at == _NOW
            assert isinstance(record.request.resolution, AnsweredResolution)
            assert len(record.request.resolution.answers) == 1
            answer = record.request.resolution.answers[0]
            assert isinstance(answer, ConfirmationAnswer)
            assert answer.value is True
            return (
                result.delivery.handler_attempts,
                second.validation_error.code,
                record.request.state,
                record.request.resolution.status,
                record.request.resolution.resolved_at,
                answer.value,
            )
        finally:
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (
        2,
        InputErrorCode.ANSWER_TYPE_MISMATCH,
        RequestState.ANSWERED,
        ResolutionStatus.ANSWERED,
        _NOW,
        True,
    )


def test_requirement_input_n_090() -> None:
    """Bind full trusted scope and commit only the first valid resolution."""

    async def exercise() -> tuple[object, ...]:
        harness = await _harness()
        resumer = _Resumer()
        request = _request(
            _DetachedHandler(),
            run_id="n090",
            resumer=resumer,
            reason="First valid resolution wins.",
        )
        try:
            requested = await harness.broker.request(request)
            assert isinstance(requested.delivery, InteractionDelivery)
            record = requested.delivery.record
            correlation = record.correlation
            assert record.request.origin.principal == _principal()
            assert correlation.run_id == request.origin.run_id
            assert correlation.turn_id == request.origin.turn_id
            assert correlation.task_id == request.origin.task_id
            assert correlation.agent_id == request.origin.agent_id
            assert correlation.branch_id == request.origin.branch_id
            assert correlation.model_call_id == request.origin.model_call_id

            start = Event()

            async def submit(
                command: ResolveInteractionCommand,
            ) -> InteractionBrokerResult:
                await start.wait()
                return await harness.broker.resolve(command)

            answer_task = create_task(
                submit(_answer(record, "n090-answer", value=True))
            )
            decline_task = create_task(
                submit(_decline(record, "n090-decline"))
            )
            start.set()
            broker_results = await gather(answer_task, decline_task)
            store_results = tuple(
                result.store_result for result in broker_results
            )
            applied_count = sum(
                isinstance(result, ResolveInteractionApplied)
                for result in store_results
            )
            rejected_count = sum(
                isinstance(result, ResolveInteractionRejected)
                for result in store_results
            )
            assert applied_count == 1
            assert rejected_count == 1
            await resumer.called.wait()
            assert len(resumer.notifications) == 1
            terminal = await _inspect(harness.broker, correlation)
            assert terminal.request.resolution is not None
            assert terminal.request.resolution.status in {
                ResolutionStatus.ANSWERED,
                ResolutionStatus.DECLINED,
            }
            assert terminal.request.state_revision == (
                record.request.state_revision + 1
            )
            return (
                applied_count,
                rejected_count,
                terminal.request.state_revision
                - record.request.state_revision,
                len(resumer.notifications),
                terminal.request.resolution.status
                in {
                    ResolutionStatus.ANSWERED,
                    ResolutionStatus.DECLINED,
                },
            )
        finally:
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (1, 1, 1, 1, True)


def test_requirement_input_n_091() -> None:
    """Authorize semantic replay without disclosing denied store state."""

    async def exercise() -> tuple[object, ...]:
        harness = await _harness()
        resumer = _Resumer()
        request = _request(
            _DetachedHandler(),
            run_id="n091",
            resumer=resumer,
            reason="Semantic replay.",
        )
        try:
            requested = await harness.broker.request(request)
            assert isinstance(requested.delivery, InteractionDelivery)
            pending = requested.delivery.record
            authorization_target = InteractionRequestAuthorizationTarget(
                request_id=pending.request.request_id,
                origin=pending.request.origin,
            )
            expected_authorization_call = (
                _actor(),
                InteractionOperation.RESOLVE,
                authorization_target,
            )

            first_command = _decline(pending, "n091-first")
            harness.authorizer.calls.clear()
            first = await harness.broker.resolve(first_command)
            assert isinstance(first.store_result, ResolveInteractionApplied)
            first_authorization_calls = tuple(
                call
                for call in harness.authorizer.calls
                if call[1] is InteractionOperation.RESOLVE
            )
            assert first_authorization_calls == (expected_authorization_call,)
            await resumer.called.wait()
            assert len(resumer.notifications) == 1

            replay_command = _decline(pending, "n091-second")
            harness.authorizer.calls.clear()
            replay = await harness.broker.resolve(replay_command)
            assert isinstance(replay.store_result, InteractionStoreReplayed)
            replay_authorization_calls = tuple(
                call
                for call in harness.authorizer.calls
                if call[1] is InteractionOperation.RESOLVE
            )
            assert replay_authorization_calls == (expected_authorization_call,)
            assert replay.store_result.replay_kind is (
                InteractionReplayKind.SEMANTIC_NEW_KEY
            )
            assert replay.store_result.store_mutation_applied
            assert replay.store_result.record.request == (
                first.store_result.record.request
            )
            assert tuple(
                entry.key
                for entry in replay.store_result.record.idempotency_ledger
            ) == (
                ResolutionIdempotencyKey("n091-first"),
                ResolutionIdempotencyKey("n091-second"),
            )
            assert len(resumer.notifications) == 1
            assert replay.resumer_failed is False

            denied_command = _decline(pending, "n091-denied")
            harness.authorizer.denied_operations.add(
                InteractionOperation.RESOLVE
            )
            harness.authorizer.calls.clear()
            denied = await harness.broker.resolve(denied_command)
            denied_authorization_calls = tuple(
                call
                for call in harness.authorizer.calls
                if call[1] is InteractionOperation.RESOLVE
            )
            assert denied_authorization_calls == (expected_authorization_call,)
            assert isinstance(denied.store_result, ResolveInteractionRejected)
            assert denied.store_result.command == denied_command
            assert denied.store_result.error.code is InputErrorCode.FORBIDDEN
            assert denied.store_result.decision_stage is (
                ResolutionDecisionStage.AUTHORIZATION
            )
            assert not denied.store_result.store_mutation_applied
            denied_fields = tuple(
                item.name for item in fields(denied.store_result)
            )
            assert denied_fields == (
                "command",
                "error",
                "decision_stage",
                "store_mutation_applied",
                "kind",
            )
            assert "record" not in denied_fields
            assert "previous" not in denied_fields
            assert len(resumer.notifications) == 1

            harness.authorizer.denied_operations.clear()
            stored = await _inspect(harness.broker, pending.correlation)
            assert stored == replay.store_result.record
            assert tuple(entry.key for entry in stored.idempotency_ledger) == (
                ResolutionIdempotencyKey("n091-first"),
                ResolutionIdempotencyKey("n091-second"),
            )
            return (
                replay.store_result.replay_kind,
                len(stored.idempotency_ledger),
                denied.store_result.error.code,
                denied.store_result.decision_stage,
                denied.store_result.store_mutation_applied,
                denied_fields,
                len(resumer.notifications),
                tuple(
                    len(calls)
                    for calls in (
                        first_authorization_calls,
                        replay_authorization_calls,
                        denied_authorization_calls,
                    )
                ),
            )
        finally:
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (
        InteractionReplayKind.SEMANTIC_NEW_KEY,
        2,
        InputErrorCode.FORBIDDEN,
        ResolutionDecisionStage.AUTHORIZATION,
        False,
        (
            "command",
            "error",
            "decision_stage",
            "store_mutation_applied",
            "kind",
        ),
        1,
        (1, 1, 1),
    )


def test_requirement_input_n_092() -> None:
    """Reject conflicting key reuse without a second continuation attempt."""

    async def exercise() -> tuple[object, ...]:
        harness = await _harness()
        resumer = _Resumer()
        request = _request(
            _DetachedHandler(),
            run_id="n092",
            resumer=resumer,
            reason="Conflicting later resolution.",
        )
        try:
            requested = await harness.broker.request(request)
            assert isinstance(requested.delivery, InteractionDelivery)
            pending = requested.delivery.record
            first = await harness.broker.resolve(
                _answer(pending, "n092-key", value=True)
            )
            assert isinstance(first.store_result, ResolveInteractionApplied)
            await resumer.called.wait()
            assert len(resumer.notifications) == 1

            conflict = await harness.broker.resolve(
                _answer(pending, "n092-key", value=False)
            )
            assert isinstance(
                conflict.store_result,
                ResolveInteractionRejected,
            )
            assert conflict.store_result.error.code is (
                InputErrorCode.IDEMPOTENCY_CONFLICT
            )
            assert not conflict.store_result.store_mutation_applied
            assert len(resumer.notifications) == 1
            stored = await _inspect(harness.broker, pending.correlation)
            assert stored == first.store_result.record
            assert stored.request.resolution is not None
            assert isinstance(stored.request.resolution, AnsweredResolution)
            answer = stored.request.resolution.answers[0]
            assert isinstance(answer, ConfirmationAnswer)
            assert answer.value is True
            return (
                stored.request.resolution.status,
                conflict.store_result.error.code,
                conflict.store_result.store_mutation_applied,
                answer.value,
                len(resumer.notifications),
            )
        finally:
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (
        ResolutionStatus.ANSWERED,
        InputErrorCode.IDEMPOTENCY_CONFLICT,
        False,
        True,
        1,
    )


def test_requirement_input_n_093() -> None:
    """Reject replies after timeout, expiry, cancellation, or supersession."""

    async def exercise() -> tuple[object, ...]:
        harness = await _harness()
        timeout_resumer = _Resumer()
        expiry_resumer = _Resumer()
        cancel_resumer = _Resumer()
        supersede_resumer = _Resumer()
        try:
            timed_request = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n093-timeout",
                    resumer=timeout_resumer,
                    mode=RequirementMode.ADVISORY,
                    advisory_wait_seconds=5,
                    reason="Advisory timeout.",
                )
            )
            expiry_request = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n093-expiry",
                    resumer=expiry_resumer,
                    continuation_ttl_seconds=60,
                    reason="Absolute expiry.",
                )
            )
            cancel_request = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n093-cancel",
                    resumer=cancel_resumer,
                    reason="Explicit cancellation.",
                )
            )
            supersede_request = await harness.broker.request(
                _request(
                    _DetachedHandler(),
                    run_id="n093-supersede",
                    resumer=supersede_resumer,
                    reason="Execution supersession.",
                )
            )
            deliveries = (
                timed_request.delivery,
                expiry_request.delivery,
                cancel_request.delivery,
                supersede_request.delivery,
            )
            assert all(
                isinstance(delivery, InteractionDelivery)
                for delivery in deliveries
            )
            timed = deliveries[0]
            expiring = deliveries[1]
            cancelling = deliveries[2]
            superseding = deliveries[3]
            assert isinstance(timed, InteractionDelivery)
            assert isinstance(expiring, InteractionDelivery)
            assert isinstance(cancelling, InteractionDelivery)
            assert isinstance(superseding, InteractionDelivery)

            cancelled = await harness.broker.cancel(
                CancelInteractionCommand(
                    actor=_actor(),
                    correlation=cancelling.correlation,
                    provenance=AnswerProvenance.HUMAN,
                    expected_state_revision=(
                        cancelling.record.request.state_revision
                    ),
                )
            )
            assert isinstance(cancelled.store_result, CancelInteractionApplied)
            superseded = await harness.broker.supersede(
                SupersedeInteractionScopeCommand(
                    actor=_actor(),
                    scope=InteractionExecutionScope(
                        run_id=superseding.record.request.origin.run_id,
                    ),
                    provenance=AnswerProvenance.HUMAN,
                )
            )
            assert isinstance(
                superseded.store_result,
                ScopeSupersessionApplied,
            )

            await _wait_until(
                lambda: 5.0 in harness.clock.wait_calls,
            )
            harness.clock.advance(5)
            await timeout_resumer.called.wait()
            await _wait_until(
                lambda: 60.0 in harness.clock.wait_calls,
            )
            harness.clock.advance(55)
            await expiry_resumer.called.wait()

            cases = (
                (
                    timed.record,
                    ResolutionStatus.TIMED_OUT,
                    timeout_resumer,
                    "n093-timeout-key",
                ),
                (
                    expiring.record,
                    ResolutionStatus.EXPIRED,
                    expiry_resumer,
                    "n093-expiry-key",
                ),
                (
                    cancelling.record,
                    ResolutionStatus.CANCELLED,
                    cancel_resumer,
                    "n093-cancel-key",
                ),
                (
                    superseding.record,
                    ResolutionStatus.SUPERSEDED,
                    supersede_resumer,
                    "n093-supersede-key",
                ),
            )
            observed_statuses: list[ResolutionStatus] = []
            observed_errors: list[InputErrorCode] = []
            observed_notifications: list[int] = []
            for pending, status, resumer, key in cases:
                stale = await harness.broker.resolve(_decline(pending, key))
                assert isinstance(
                    stale.store_result,
                    ResolveInteractionRejected,
                )
                assert stale.store_result.error.code is (
                    InputErrorCode.ALREADY_RESOLVED
                )
                assert not stale.store_result.store_mutation_applied
                assert len(resumer.notifications) == 1
                terminal = await _inspect(
                    harness.broker,
                    pending.correlation,
                )
                assert terminal.request.resolution is not None
                assert terminal.request.resolution.status is status
                observed_statuses.append(terminal.request.resolution.status)
                observed_errors.append(stale.store_result.error.code)
                observed_notifications.append(len(resumer.notifications))
            return (
                tuple(observed_statuses),
                tuple(observed_errors),
                tuple(observed_notifications),
            )
        finally:
            await harness.broker.aclose()

    summary = run(exercise())
    assert summary == (
        (
            ResolutionStatus.TIMED_OUT,
            ResolutionStatus.EXPIRED,
            ResolutionStatus.CANCELLED,
            ResolutionStatus.SUPERSEDED,
        ),
        (
            InputErrorCode.ALREADY_RESOLVED,
            InputErrorCode.ALREADY_RESOLVED,
            InputErrorCode.ALREADY_RESOLVED,
            InputErrorCode.ALREADY_RESOLVED,
        ),
        (1, 1, 1, 1),
    )
