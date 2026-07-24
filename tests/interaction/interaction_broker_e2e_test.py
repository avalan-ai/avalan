"""Exercise complete broker flows over the public in-memory store surface."""

from asyncio import Event, create_task, get_running_loop, run
from datetime import UTC, datetime

from avalan.interaction import (
    ActiveControlLeaseNonce,
    AgentId,
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    BranchId,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputAnswer,
    InputErrorCode,
    InputHandlerContext,
    InputHandlerResolution,
    InputRequestId,
    InputResumer,
    InputResumptionNotification,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBrokerRequest,
    InteractionClock,
    InteractionDisclosure,
    InteractionExecutionScope,
    InteractionIdFactory,
    InteractionOperation,
    InteractionPolicy,
    InteractionRecord,
    InteractionTime,
    ListInteractionsCommand,
    ModelCallId,
    PrincipalScope,
    QuestionId,
    RegisterInteractionBranchCommand,
    RequestState,
    RequirementMode,
    ResolutionIdempotencyKey,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResumeInputContinuation,
    RunId,
    ScopedInteractionLookup,
    StreamSessionId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TextAnswer,
    TextQuestion,
    TurnId,
    UserId,
    WaitForInteractionChangeCommand,
)
from avalan.interaction.handler import InputHandler, InputHandlerOutcome
from avalan.interaction.stores import MemoryInteractionStoreFactory

_NOW = datetime(2026, 7, 21, 16, 0, tzinfo=UTC)


async def _yield_once() -> None:
    """Yield one deterministic event-loop turn without a timer."""
    ready = get_running_loop().create_future()
    get_running_loop().call_soon(ready.set_result, None)
    await ready


class _Clock(InteractionClock):
    """Provide one fixed coherent observation for non-deadline E2E flows."""

    async def read(self) -> InteractionTime:
        """Return one trusted fixed observation."""
        return InteractionTime.from_clock(
            wall_time=_NOW,
            monotonic_seconds=0.0,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Remain cancellable because these flows never reach a deadline."""
        assert monotonic_deadline >= 0.0
        await Event().wait()


class _IdFactory(InteractionIdFactory):
    """Mint deterministic identities and transport keys."""

    def __init__(self) -> None:
        self.sequence = 0

    def _next(self, kind: str) -> str:
        self.sequence += 1
        return f"broker-e2e-{kind}-{self.sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return one request identity."""
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        """Return one continuation identity."""
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return one resolution transport key."""
        return ResolutionIdempotencyKey(self._next("resolution"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return one active-control nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _Classifier(TaskInputClassifier):
    """Allow every typed answer with exact binding echoes."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self.policy = policy
        self.sequence = 0

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return one deterministic allow decision."""
        self.sequence += 1
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self.policy.task_input_classifier_id,
            classification_id=f"broker-e2e-classification-{self.sequence}",
            policy_revision=self.policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Authorizer:
    """Grant full disclosure while preserving exact authorization echoes."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Authorize one exact actor, operation, and target."""
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


class _Resumer(InputResumer):
    """Record exact continuation notifications."""

    def __init__(self) -> None:
        self.notifications: list[InputResumptionNotification] = []
        self.called = Event()

    async def __call__(
        self,
        notification: InputResumptionNotification,
    ) -> None:
        """Record one post-commit callback."""
        self.notifications.append(notification)
        self.called.set()


class _CorrectionHandler(InputHandler):
    """Expose barriers around invalid input and its typed correction."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []
        self.first_presented = Event()
        self.submit_invalid = Event()
        self.correction_presented = Event()
        self.submit_valid = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return an invalid answer, then the corrected answer."""
        self.contexts.append(context)
        if len(self.contexts) == 1:
            self.first_presented.set()
            await self.submit_invalid.wait()
            answer: InputAnswer = TextAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value="not a confirmation",
            )
        else:
            self.correction_presented.set()
            await self.submit_valid.wait()
            answer = ConfirmationAnswer(
                question_id=QuestionId("confirm"),
                provenance=AnswerProvenance.HUMAN,
                value=True,
            )
        return InputHandlerResolution(
            resolution=AnsweredResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
                answers=(answer,),
            )
        )


class _SiblingHandler(InputHandler):
    """Hold branch A while recording every delivered context."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []
        self.presented = Event()
        self.cancelled = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Wait for external resolution and record cancellation."""
        self.contexts.append(context)
        self.presented.set()
        try:
            await Event().wait()
        finally:
            self.cancelled.set()
        raise AssertionError("unreachable after external resolution")


async def _open_broker() -> AsyncInteractionBroker:
    """Open one broker over the real public memory-store factory."""
    policy = InteractionPolicy()
    clock = _Clock()
    ids = _IdFactory()
    classifier = _Classifier(policy)
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


def _principal() -> PrincipalScope:
    return PrincipalScope(user_id=UserId("broker-e2e-owner"))


def _actor() -> InteractionActor:
    return InteractionActor(principal=_principal())


def _origin(
    branch: str,
    *,
    parent: str | None = None,
) -> ExecutionOrigin:
    return ExecutionOrigin(
        run_id=RunId("broker-e2e-run"),
        turn_id=TurnId(f"turn-{branch}"),
        agent_id=AgentId("broker-e2e-agent"),
        branch_id=BranchId(branch),
        parent_branch_id=None if parent is None else BranchId(parent),
        model_call_id=ModelCallId(f"call-{branch}"),
        stream_session_id=StreamSessionId(f"stream-{branch}"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://broker-e2e",
            agent_definition_revision="revision-1",
            operation_id="operation",
            operation_index=0,
            model_config_reference="model-1",
            tool_revision="tools-1",
            capability_revision="capabilities-1",
        ),
        principal=_principal(),
    )


def _confirmation_request(
    handler: InputHandler,
    resumer: InputResumer,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=_actor(),
        origin=_origin("correction"),
        mode=RequirementMode.REQUIRED,
        reason="Confirm the E2E choice.",
        questions=(
            ConfirmationQuestion(
                question_id=QuestionId("confirm"),
                prompt="Continue?",
                required=True,
            ),
        ),
        handler=handler,
        resumer=resumer,
        continuation_ttl_seconds=600,
    )


def _sibling_request(
    branch: str,
    handler: InputHandler,
    resumer: InputResumer,
) -> InteractionBrokerRequest:
    return InteractionBrokerRequest(
        actor=_actor(),
        origin=_origin(branch, parent="root"),
        mode=RequirementMode.REQUIRED,
        reason=f"Provide the isolated {branch} value.",
        questions=(
            TextQuestion(
                question_id=QuestionId(f"question-{branch}"),
                prompt=f"Value for {branch}?",
                required=True,
            ),
        ),
        handler=handler,
        resumer=resumer,
        continuation_ttl_seconds=600,
    )


async def _records(
    broker: AsyncInteractionBroker,
) -> tuple[InteractionRecord, ...]:
    """Return full records for the E2E run through the public list API."""
    projections = await broker.list(
        ListInteractionsCommand(
            actor=_actor(),
            scope=InteractionExecutionScope(run_id=RunId("broker-e2e-run")),
        )
    )
    return tuple(
        projection
        for projection in projections
        if isinstance(projection, InteractionRecord)
    )


async def _wait_for_record_count(
    broker: AsyncInteractionBroker,
    count: int,
) -> tuple[InteractionRecord, ...]:
    """Wait deterministically for public inventory to reach one size."""
    for _ in range(100):
        records = await _records(broker)
        if len(records) == count:
            return records
        await _yield_once()
    raise AssertionError("interaction inventory did not reach expected size")


def _answer_command(
    record: InteractionRecord,
    value: str,
    key: str,
) -> ResolveInteractionCommand:
    return ResolveInteractionCommand(
        actor=_actor(),
        correlation=record.correlation,
        expected_state_revision=record.request.state_revision,
        idempotency_key=ResolutionIdempotencyKey(key),
        proposed_resolution=AnsweredResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_NOW,
            answers=(
                TextAnswer(
                    question_id=record.request.questions[0].question_id,
                    provenance=AnswerProvenance.HUMAN,
                    value=value,
                ),
            ),
        ),
    )


def test_create_attached_correction_resolution_and_wait_e2e() -> None:
    """Create, correct, resolve, resume, and complete a public broker wait."""

    async def exercise() -> None:
        broker = await _open_broker()
        handler = _CorrectionHandler()
        resumer = _Resumer()
        try:
            request_task = create_task(
                broker.request(_confirmation_request(handler, resumer))
            )
            await handler.first_presented.wait()
            (presented,) = await _wait_for_record_count(broker, 1)
            wait_task = create_task(
                broker.wait(
                    WaitForInteractionChangeCommand(
                        actor=_actor(),
                        correlation=presented.correlation,
                        after_store_revision=presented.store_revision,
                    )
                )
            )

            handler.submit_invalid.set()
            await handler.correction_presented.wait()
            assert not wait_task.done()
            correction = handler.contexts[1].validation_error
            assert correction is not None
            assert correction.code is InputErrorCode.ANSWER_TYPE_MISMATCH

            handler.submit_valid.set()
            request_result = await request_task
            completed = await wait_task
            await resumer.called.wait()

            assert request_result.delivery is not None
            assert request_result.delivery.handler_attempts == 2
            assert isinstance(completed, InteractionRecord)
            assert completed == request_result.delivery.record
            assert completed.request.state is RequestState.ANSWERED
            assert len(resumer.notifications) == 1
            assert isinstance(
                resumer.notifications[0].outcome,
                ResumeInputContinuation,
            )
        finally:
            await broker.aclose()

    run(exercise())


def test_concurrent_siblings_resolve_b_first_without_leaking_into_a() -> None:
    """Resolve queued sibling B while A stays pending and data-isolated."""

    async def exercise() -> None:
        broker = await _open_broker()
        handler_a = _SiblingHandler()
        handler_b = _SiblingHandler()
        resumer_a = _Resumer()
        resumer_b = _Resumer()
        try:
            for branch in ("branch-a", "branch-b"):
                registration = await broker.register_branch(
                    RegisterInteractionBranchCommand(
                        actor=_actor(),
                        registration=InteractionBranchRegistration(
                            run_id=RunId("broker-e2e-run"),
                            branch_id=BranchId(branch),
                            parent_branch_id=BranchId("root"),
                            principal=_principal(),
                        ),
                    )
                )
                assert isinstance(
                    registration.store_result,
                    InteractionBranchRegistrationApplied,
                )

            task_a = create_task(
                broker.request(
                    _sibling_request("branch-a", handler_a, resumer_a)
                )
            )
            await handler_a.presented.wait()
            task_b = create_task(
                broker.request(
                    _sibling_request("branch-b", handler_b, resumer_b)
                )
            )
            records = await _wait_for_record_count(broker, 2)
            by_branch = {
                record.request.origin.branch_id: record for record in records
            }
            record_a = by_branch[BranchId("branch-a")]
            record_b = by_branch[BranchId("branch-b")]

            resolved_b = await broker.resolve(
                _answer_command(record_b, "branch-B-answer", "resolve-b")
            )
            assert isinstance(
                resolved_b.store_result,
                ResolveInteractionApplied,
            )
            await resumer_b.called.wait()

            projection_a = await broker.inspect(
                ScopedInteractionLookup(
                    actor=_actor(),
                    correlation=record_a.correlation,
                )
            )
            assert isinstance(projection_a, InteractionRecord)
            assert projection_a.request.state is RequestState.PENDING
            assert resumer_a.notifications == []
            assert not handler_b.presented.is_set()
            assert "branch-B-answer" not in repr(handler_a.contexts)
            assert "branch-B-answer" not in repr(projection_a)

            resolved_a = await broker.resolve(
                _answer_command(record_a, "branch-A-answer", "resolve-a")
            )
            assert isinstance(
                resolved_a.store_result,
                ResolveInteractionApplied,
            )
            await handler_a.cancelled.wait()
            await resumer_a.called.wait()
            result_a = await task_a
            result_b = await task_b
            assert result_a is not None
            assert result_b is not None
        finally:
            await broker.aclose()

    run(exercise())
