"""Exercise attached input across type, outcome, and approval boundaries."""

from asyncio import Event, Queue, create_task
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from json import dumps, loads
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.agent import (
    AgentOperation,
    EngineEnvironment,
    InputType,
    Specification,
)
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    ExecutionIdFactory,
    InteractionLoopLimitError,
    ModelPromptRecord,
)
from avalan.agent.orchestrator import Orchestrator
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    ToolCall,
    ToolManagerSettings,
    ToolNamePolicyMode,
    ToolNamePolicySettings,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.interaction import (
    RESERVED_INPUT_CAPABILITY_NAME,
    ActiveControlLeaseNonce,
    AnsweredResolution,
    AnswerProvenance,
    AsyncInteractionBroker,
    BranchId,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    ContinuationId,
    DeclinedResolution,
    FreeFormOther,
    HandlerLossDisposition,
    InputAnswer,
    InputDisconnectReason,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
    InputQuestion,
    InputRequestId,
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizationTarget,
    InteractionClock,
    InteractionCorrelation,
    InteractionDisclosure,
    InteractionIdFactory,
    InteractionOperation,
    InteractionPolicy,
    InteractionTime,
    ModelCallId,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalScope,
    QuestionType,
    ResolutionIdempotencyKey,
    RunId,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    StreamSessionId,
    TaskId,
    TaskInputClassification,
    TaskInputClassificationDecision,
    TaskInputClassificationRequest,
    TaskInputClassifier,
    TextAnswer,
    TextQuestion,
    TrustedDefaultResolutionRequest,
    TurnId,
    UserId,
)
from avalan.interaction.stores import MemoryInteractionStoreFactory
from avalan.memory.manager import MemoryManager
from avalan.model.call import ModelCallContext
from avalan.model.capability import ProviderCapabilitySupport
from avalan.model.manager import ModelManager
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager

_NOW = datetime(2026, 7, 22, 15, 0, tzinfo=UTC)


class _Clock(InteractionClock):
    """Expose a deterministic deadline controlled by an explicit event."""

    def __init__(self) -> None:
        self.wall_time = _NOW
        self.monotonic_seconds = 1.0
        self.deadline: float | None = None
        self._deadline_updates: Queue[float] = Queue()
        self._timer_release = Event()

    async def read(self) -> InteractionTime:
        """Return the current coherent trusted time observation."""
        return InteractionTime.from_clock(
            wall_time=self.wall_time,
            monotonic_seconds=self.monotonic_seconds,
        )

    async def wait_until(self, monotonic_deadline: float) -> None:
        """Wait for the test to advance exactly to the requested deadline."""
        self.deadline = monotonic_deadline
        self._deadline_updates.put_nowait(monotonic_deadline)
        await self._timer_release.wait()

    async def advance_to_deadline(self) -> None:
        """Advance trusted wall and monotonic time to the armed deadline."""
        while True:
            deadline = await self._deadline_updates.get()
            elapsed = deadline - self.monotonic_seconds
            if elapsed <= 3_600:
                break
        assert elapsed >= 0
        self.deadline = deadline
        self.monotonic_seconds = deadline
        self.wall_time += timedelta(seconds=elapsed)
        self._timer_release.set()


class _BrokerIds(InteractionIdFactory):
    """Mint deterministic broker-owned identifiers."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next(self, kind: str) -> str:
        self._sequence += 1
        return f"matrix-{kind}-{self._sequence}"

    async def new_request_id(self) -> InputRequestId:
        """Return a request identifier."""
        return InputRequestId(self._next("request"))

    async def new_continuation_id(self) -> ContinuationId:
        """Return a continuation identifier."""
        return ContinuationId(self._next("continuation"))

    async def new_idempotency_key(self) -> ResolutionIdempotencyKey:
        """Return an idempotency key."""
        return ResolutionIdempotencyKey(self._next("key"))

    async def new_active_control_lease_nonce(
        self,
    ) -> ActiveControlLeaseNonce:
        """Return an active-control lease nonce."""
        return ActiveControlLeaseNonce(self._next("lease"))


class _ExecutionIds(ExecutionIdFactory):
    """Mint deterministic execution identifiers."""

    def __init__(self) -> None:
        self._sequence = 0

    def _next(self, kind: str) -> str:
        self._sequence += 1
        return f"matrix-execution-{kind}-{self._sequence}"

    async def new_run_id(self) -> RunId:
        """Return a run identifier."""
        return RunId(self._next("run"))

    async def new_turn_id(self) -> TurnId:
        """Return a turn identifier."""
        return TurnId(self._next("turn"))

    async def new_model_call_id(self) -> ModelCallId:
        """Return a model-call identifier."""
        return ModelCallId(self._next("call"))

    async def new_task_id(self) -> TaskId:
        """Return a task identifier."""
        return TaskId(self._next("task"))

    async def new_branch_id(self) -> BranchId:
        """Return a branch identifier."""
        return BranchId(self._next("branch"))

    async def new_stream_session_id(self) -> StreamSessionId:
        """Return a stream-session identifier."""
        return StreamSessionId(self._next("stream"))


class _Classifier(TaskInputClassifier):
    """Allow each normalized task-input request under the active policy."""

    def __init__(self, policy: InteractionPolicy) -> None:
        self._policy = policy

    async def classify_task_input(
        self,
        request: TaskInputClassificationRequest,
    ) -> TaskInputClassification:
        """Return an exact allow decision for the normalized request."""
        return TaskInputClassification(
            decision=TaskInputClassificationDecision.ALLOW,
            classifier_id=self._policy.task_input_classifier_id,
            classification_id="attached-runtime-matrix",
            policy_revision=self._policy.task_input_policy_revision,
            request_id=request.request_id,
            candidate_digest=request.candidate_digest,
            question_id=request.question_id,
            semantic_type=request.semantic_type,
        )


class _Authorizer:
    """Authorize the test principal for exact broker operations."""

    async def authorize(
        self,
        actor: InteractionActor,
        operation: InteractionOperation,
        target: InteractionAuthorizationTarget,
    ) -> InteractionAuthorizationDecision:
        """Return full disclosure bound to the exact target."""
        return InteractionAuthorizationDecision(
            actor=actor,
            operation=operation,
            target=target,
            allowed=True,
            disclosure=InteractionDisclosure.FULL,
        )


def _human_answer(
    question: InputQuestion,
    *,
    text_value: str = "human text",
) -> InputAnswer:
    """Return one valid human answer for the exact question variant."""
    common = {
        "question_id": question.question_id,
        "provenance": AnswerProvenance.HUMAN,
    }
    if isinstance(question, ConfirmationQuestion):
        return ConfirmationAnswer(**common, value=True)
    if isinstance(question, TextQuestion):
        return TextAnswer(**common, value=text_value)
    if isinstance(question, MultilineTextQuestion):
        return MultilineTextAnswer(**common, value="first line\nsecond line")
    if isinstance(question, SingleSelectionQuestion):
        return SingleSelectionAnswer(
            **common,
            value=SelectedChoice(value=ChoiceValue("blue")),
        )
    assert isinstance(question, MultipleSelectionQuestion)
    return MultipleSelectionAnswer(
        **common,
        values=(
            SelectedChoice(value=ChoiceValue("blue")),
            FreeFormOther(text="custom"),
        ),
    )


class _AnsweringHandler:
    """Answer each attached question with a typed human value."""

    def __init__(self, *, text_value: str = "human text") -> None:
        self.text_value = text_value
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Answer the sole question in the exact broker request."""
        self.contexts.append(context)
        assert len(context.request.questions) == 1
        return InputHandlerResolution(
            resolution=AnsweredResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
                answers=(
                    _human_answer(
                        context.request.questions[0],
                        text_value=self.text_value,
                    ),
                ),
            )
        )


class _DecliningHandler:
    """Decline one attached request as the human principal."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return one explicit human decline."""
        self.contexts.append(context)
        return InputHandlerResolution(
            resolution=DeclinedResolution(
                request_id=context.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_NOW,
            )
        )


class _DisconnectedHandler:
    """Report deterministic attached-channel loss."""

    def __init__(self) -> None:
        self.contexts: list[InputHandlerContext] = []

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Return a typed unavailable-handler disconnect."""
        self.contexts.append(context)
        return InputHandlerDisconnected(
            reason=InputDisconnectReason.HANDLER_UNAVAILABLE
        )


class _BlockingHandler:
    """Remain pending until broker-owned settlement cancels the handler."""

    def __init__(self) -> None:
        self.started = Event()
        self.contexts: list[InputHandlerContext] = []
        self._never = Event()

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        """Expose the request and wait for authoritative settlement."""
        self.contexts.append(context)
        self.started.set()
        await self._never.wait()
        raise AssertionError("authoritative settlement did not cancel handler")


class _ApprovalGate:
    """Hold protected action approval behind a distinct control event."""

    def __init__(self) -> None:
        self.started = Event()
        self.release = Event()
        self.calls: list[ToolCall] = []

    async def __call__(self, call: ToolCall) -> bool:
        """Wait for explicit approval of one exact domain-tool call."""
        self.calls.append(call)
        self.started.set()
        await self.release.wait()
        return True


async def _open_broker(
    policy: InteractionPolicy,
    clock: _Clock,
) -> AsyncInteractionBroker:
    """Open a real broker over the public in-memory store factory."""
    ids = _BrokerIds()
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


def _question(
    kind: QuestionType, *, default: object = None
) -> dict[str, object]:
    """Return one provider-native question for the requested semantic type."""
    question: dict[str, object] = {
        "question_id": kind.value,
        "kind": kind.value,
        "prompt": f"Provide {kind.value} input.",
        "required": True,
        "choices": [],
        "allow_other": False,
    }
    if kind in {
        QuestionType.SINGLE_SELECTION,
        QuestionType.MULTIPLE_SELECTION,
    }:
        question["choices"] = [
            {"value": "blue", "label": "Blue"},
            {"value": "green", "label": "Green"},
        ]
    if kind is QuestionType.MULTIPLE_SELECTION:
        question["allow_other"] = True
        question["constraints"] = {"minimum": 1, "maximum": 3}
    if default is not None:
        question["default_value"] = default
    return question


def _input_arguments(
    question: dict[str, object],
    *,
    mode: str = "required",
) -> dict[str, object]:
    """Return one reserved capability invocation payload."""
    return {
        "mode": mode,
        "reason": "Need explicit task clarification.",
        "questions": [question],
    }


def _provider_response(
    label: str,
    *,
    calls: tuple[ToolCall, ...] = (),
    answer: str | None = None,
) -> TextGenerationResponse:
    """Return one deterministic structured provider stream."""

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        sequence = 0
        common = {
            "stream_session_id": f"provider-stream-{label}",
            "run_id": f"provider-run-{label}",
            "turn_id": f"provider-turn-{label}",
            "provider_family": "openai",
        }
        yield CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=StreamItemKind.STREAM_STARTED,
            channel=StreamChannel.CONTROL,
        )
        sequence += 1
        for call in calls:
            assert call.id is not None
            correlation = StreamItemCorrelation(tool_call_id=str(call.id))
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta=dumps(call.arguments or {}),
                correlation=correlation,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": call.name},
                correlation=correlation,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_DONE,
                channel=StreamChannel.TOOL_CALL,
                correlation=correlation,
            )
            sequence += 1
        if answer is not None:
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=answer,
            )
            sequence += 1
            yield CanonicalStreamItem(
                **common,
                sequence=sequence,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            sequence += 1
        yield CanonicalStreamItem(
            **common,
            sequence=sequence,
            kind=StreamItemKind.STREAM_COMPLETED,
            channel=StreamChannel.CONTROL,
            usage={},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        )
        yield CanonicalStreamItem(
            **common,
            sequence=sequence + 1,
            kind=StreamItemKind.STREAM_CLOSED,
            channel=StreamChannel.CONTROL,
        )

    return TextGenerationResponse(
        lambda **_: items(),
        logger=getLogger(__name__),
        use_async_generator=True,
    )


def _task_input_response(
    label: str,
    arguments: dict[str, object],
) -> TextGenerationResponse:
    """Return a provider response containing one reserved input call."""
    return _provider_response(
        label,
        calls=(
            ToolCall(
                id=f"input-{label}",
                name=RESERVED_INPUT_CAPABILITY_NAME,
                arguments=arguments,
            ),
        ),
    )


def _operation() -> AgentOperation:
    """Return one real orchestrator operation for the fake provider."""
    return AgentOperation(
        specification=Specification(
            role=None,
            goal=None,
            input_type=InputType.TEXT,
        ),
        environment=EngineEnvironment(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="attached-runtime-matrix",
                params={},
            ),
            settings=TransformerEngineSettings(),
        ),
    )


def _empty_tool_manager() -> ToolManager:
    """Return a tool manager with no domain tools."""
    return ToolManager.create_instance(enable_tools=[])


def _protected_tool_manager(
    executions: list[str],
) -> ToolManager:
    """Return one mapped domain tool whose execution is observable."""

    async def protected_action(command: str) -> str:
        """Perform one protected action.

        Args:
            command: Action command to execute.

        Returns:
            Confirmation of the executed command.
        """
        executions.append(command)
        return f"executed:{command}"

    return ToolManager.create_instance(
        available_toolsets=[
            ToolSet(namespace="ops", tools=[protected_action])
        ],
        enable_tools=["ops.protected_action"],
        settings=ToolManagerSettings(
            tool_name_policy=ToolNamePolicySettings(
                mode=ToolNamePolicyMode.SANITIZED,
                map={"ops.protected_action": "protected_action"},
            )
        ),
    )


class _Harness:
    """Drive public orchestration with an attached real interaction broker."""

    def __init__(
        self,
        *,
        broker: AsyncInteractionBroker,
        clock: _Clock,
        handler: Callable[[InputHandlerContext], Any],
        responses: list[TextGenerationResponse],
        tool: ToolManager,
        tool_confirm: Callable[[ToolCall], Any] | None,
        maximum_tool_cycles: int = 24,
    ) -> None:
        self.broker = broker
        self.clock = clock
        self.handler = handler
        self.responses = iter(responses)
        self.tool_confirm = tool_confirm
        self.maximum_tool_cycles = maximum_tool_cycles
        self.contexts: list[ModelCallContext] = []
        self.actor = InteractionActor(
            principal=PrincipalScope(user_id=UserId("matrix-user"))
        )
        self.runtime = AttachedInteractionRuntime(
            broker=broker,
            actor=self.actor,
            handler=cast(Any, handler),
            id_factory=_ExecutionIds(),
        )
        self.operation = _operation()
        events = MagicMock(spec=EventManager)
        events.trigger = AsyncMock()
        events.trigger_stream_item = AsyncMock()
        events.should_emit.return_value = False
        memory = MagicMock(spec=MemoryManager)
        memory.participant_id = None
        memory.permanent_message = None
        self.orchestrator = Orchestrator(
            getLogger(__name__),
            MagicMock(spec=ModelManager),
            memory,
            tool,
            events,
            self.operation,
        )

        async def model_call(
            context: ModelCallContext,
        ) -> TextGenerationResponse:
            self.contexts.append(context)
            execution = context.execution
            assert execution is not None
            assert context.input is not None
            await execution.record_prompt(
                ModelPromptRecord(
                    input=context.input,
                    instructions=None,
                    system_prompt=None,
                    developer_prompt=None,
                )
            )
            return next(self.responses)

        self.engine_agent = AsyncMock(
            spec=EngineAgent,
            side_effect=model_call,
        )
        self.engine_agent.engine = SimpleNamespace(
            model_id="attached-runtime-matrix",
            tokenizer=SimpleNamespace(eos_token="<eos>"),
            provider_capability_support=ProviderCapabilitySupport(
                structured_invocation=True,
                stable_call_ids=True,
                correlated_results=True,
                attached_resolution=True,
            ),
        )
        environment_hash = dumps(asdict(self.operation.environment))
        self.orchestrator._engine_agents[environment_hash] = self.engine_agent

    @classmethod
    async def create(
        cls,
        *,
        handler: Callable[[InputHandlerContext], Any],
        responses: list[TextGenerationResponse],
        policy: InteractionPolicy | None = None,
        clock: _Clock | None = None,
        tool: ToolManager | None = None,
        tool_confirm: Callable[[ToolCall], Any] | None = None,
        maximum_tool_cycles: int = 24,
    ) -> "_Harness":
        """Open a harness with one real attached broker."""
        active_clock = clock or _Clock()
        active_policy = policy or InteractionPolicy()
        broker = await _open_broker(active_policy, active_clock)
        return cls(
            broker=broker,
            clock=active_clock,
            handler=handler,
            responses=responses,
            tool=tool or _empty_tool_manager(),
            tool_confirm=tool_confirm,
            maximum_tool_cycles=maximum_tool_cycles,
        )

    async def response(self) -> Any:
        """Start one attached orchestration response."""
        return await self.orchestrator(
            "Begin the attached task.",
            interaction_runtime=self.runtime,
            tool_confirm=self.tool_confirm,
            maximum_tool_cycles=self.maximum_tool_cycles,
        )

    async def close(self) -> None:
        """Close the real broker and its deterministic background tasks."""
        await self.broker.aclose()


async def _consume(response: Any) -> list[CanonicalStreamItem]:
    """Collect one public orchestrator response."""
    return [item async for item in response]


def _model_result(context: ModelCallContext) -> dict[str, object]:
    """Return the newest correlated task-input result visible to the model."""
    assert isinstance(context.input, list)
    for message in reversed(context.input):
        assert isinstance(message, Message)
        if (
            message.role is MessageRole.TOOL
            and message.name == RESERVED_INPUT_CAPABILITY_NAME
        ):
            assert isinstance(message.content, str)
            envelope = cast(dict[str, object], loads(message.content))
            result = envelope["result"]
            assert isinstance(result, dict)
            return cast(dict[str, object], result)
    raise AssertionError("model continuation has no correlated input result")


def _originating_arguments(context: ModelCallContext) -> dict[str, Any]:
    """Return the newest reserved call arguments visible to the model."""
    assert isinstance(context.input, list)
    for message in reversed(context.input):
        assert isinstance(message, Message)
        if message.role is not MessageRole.ASSISTANT or not message.tool_calls:
            continue
        call = message.tool_calls[-1]
        if call.name == RESERVED_INPUT_CAPABILITY_NAME:
            return call.arguments
    raise AssertionError("model continuation has no originating input call")


class AttachedRuntimeMatrixTest(IsolatedAsyncioTestCase):
    """Require attached-runtime semantics across public orchestration paths."""

    async def test_every_question_type_crosses_the_attached_runtime(
        self,
    ) -> None:
        handler = _AnsweringHandler()
        confirmation = AsyncMock(return_value=False)
        kinds = tuple(QuestionType)
        responses = [
            _task_input_response(
                f"question-{index}",
                _input_arguments(_question(kind)),
            )
            for index, kind in enumerate(kinds)
        ]
        responses.append(_provider_response("question-final", answer="done"))
        harness = await _Harness.create(
            handler=handler,
            responses=responses,
            tool_confirm=confirmation,
        )
        try:
            response = await harness.response()
            await _consume(response)

            self.assertEqual(
                {
                    context.request.questions[0].kind
                    for context in handler.contexts
                },
                set(QuestionType),
            )
            results = tuple(
                _model_result(context) for context in harness.contexts[1:]
            )
            self.assertEqual(len(results), len(QuestionType))
            self.assertTrue(
                all(result["kind"] == "answered" for result in results)
            )
            self.assertEqual(
                {
                    cast(list[dict[str, object]], result["answers"])[0]["kind"]
                    for result in results
                },
                {kind.value for kind in QuestionType},
            )
            assert response._execution is not None
            self.assertEqual(
                response._execution.interaction_count, len(QuestionType)
            )
            self.assertIs(
                response._execution.status, AgentExecutionStatus.COMPLETED
            )
            self.assertEqual(response._tool_cycle_count, 0)
            confirmation.assert_not_awaited()
        finally:
            await harness.close()

    async def test_model_visible_declined_cancelled_and_unavailable_outcomes(
        self,
    ) -> None:
        cases = (
            (
                "declined",
                _DecliningHandler(),
                InteractionPolicy(),
                "human",
                StreamItemKind.INTERACTION_DECLINED,
            ),
            (
                "cancelled",
                _DisconnectedHandler(),
                InteractionPolicy(
                    attached_loss_without_resumer=(
                        HandlerLossDisposition.CANCEL_REQUEST
                    )
                ),
                "external_controller",
                StreamItemKind.INTERACTION_CANCELLED,
            ),
            (
                "unavailable",
                _DisconnectedHandler(),
                InteractionPolicy(),
                "external_controller",
                StreamItemKind.INTERACTION_UNAVAILABLE,
            ),
        )
        for kind, handler, policy, provenance, terminal_kind in cases:
            with self.subTest(kind=kind):
                confirmation = AsyncMock(return_value=True)
                harness = await _Harness.create(
                    handler=handler,
                    policy=policy,
                    tool_confirm=confirmation,
                    responses=[
                        _task_input_response(
                            f"outcome-{kind}",
                            _input_arguments(
                                _question(QuestionType.CONFIRMATION)
                            ),
                        ),
                        _provider_response(
                            f"outcome-{kind}-final",
                            answer="settled",
                        ),
                    ],
                )
                try:
                    response = await harness.response()
                    items = await _consume(response)
                    result = _model_result(harness.contexts[-1])

                    self.assertEqual(result["kind"], kind)
                    self.assertEqual(result["provenance"], provenance)
                    self.assertEqual(
                        sum(item.kind is terminal_kind for item in items),
                        1,
                    )
                    self.assertEqual(response._tool_cycle_count, 0)
                    confirmation.assert_not_awaited()
                finally:
                    await harness.close()

    async def test_human_answer_provenance_is_preserved_separately(
        self,
    ) -> None:
        handler = _AnsweringHandler(text_value="human supplied")
        harness = await _Harness.create(
            handler=handler,
            responses=[
                _task_input_response(
                    "human-provenance",
                    _input_arguments(_question(QuestionType.TEXT)),
                ),
                _provider_response("human-provenance-final", answer="done"),
            ],
        )
        try:
            response = await harness.response()
            await _consume(response)
            result = _model_result(harness.contexts[-1])
            answers = cast(list[dict[str, object]], result["answers"])

            self.assertEqual(result["provenance"], "human")
            self.assertEqual(answers[0]["provenance"], "human")
            self.assertEqual(answers[0]["value"], "human supplied")
        finally:
            await harness.close()

    async def test_trusted_default_uses_broker_owned_provenance(
        self,
    ) -> None:
        handler = _BlockingHandler()
        harness = await _Harness.create(
            handler=handler,
            responses=[
                _task_input_response(
                    "trusted-default",
                    _input_arguments(
                        _question(QuestionType.CONFIRMATION, default=True)
                    ),
                ),
                _provider_response("trusted-default-final", answer="done"),
            ],
        )
        try:
            response = await harness.response()
            consumption = create_task(_consume(response))
            await handler.started.wait()
            request = handler.contexts[0].request
            await harness.broker.resolve_trusted_default(
                TrustedDefaultResolutionRequest(
                    actor=harness.actor,
                    correlation=InteractionCorrelation.from_request(request),
                    expected_state_revision=request.state_revision,
                )
            )
            await consumption
            result = _model_result(harness.contexts[-1])
            answers = cast(list[dict[str, object]], result["answers"])

            self.assertEqual(result["kind"], "answered")
            self.assertEqual(result["provenance"], "trusted_default")
            self.assertEqual(answers[0]["provenance"], "trusted_default")
            self.assertIs(answers[0]["value"], True)
        finally:
            await harness.close()

    async def test_timeout_preserves_policy_provenance_and_declared_default(
        self,
    ) -> None:
        handler = _BlockingHandler()
        clock = _Clock()
        harness = await _Harness.create(
            handler=handler,
            clock=clock,
            responses=[
                _task_input_response(
                    "policy-timeout",
                    _input_arguments(
                        _question(QuestionType.CONFIRMATION, default=False),
                        mode="advisory",
                    ),
                ),
                _provider_response("policy-timeout-final", answer="done"),
            ],
        )
        try:
            response = await harness.response()
            consumption = create_task(_consume(response))
            await handler.started.wait()
            await clock.advance_to_deadline()
            items = await consumption
            result = _model_result(harness.contexts[-1])
            arguments = _originating_arguments(harness.contexts[-1])
            questions = cast(list[dict[str, object]], arguments["questions"])

            self.assertEqual(result["kind"], "timed_out")
            self.assertEqual(result["provenance"], "policy")
            self.assertNotIn("answers", result)
            self.assertIs(questions[0]["default_value"], False)
            seen = handler.contexts[0].request.questions[0]
            self.assertIsInstance(seen, ConfirmationQuestion)
            self.assertIs(
                cast(ConfirmationQuestion, seen).default_value, False
            )
            self.assertEqual(
                sum(
                    item.kind is StreamItemKind.INTERACTION_TIMED_OUT
                    for item in items
                ),
                1,
            )
        finally:
            await harness.close()

    async def test_equivalent_input_reaches_its_bound_without_tool_cycles(
        self,
    ) -> None:
        handler = _AnsweringHandler()
        confirmation = AsyncMock(return_value=True)
        arguments = _input_arguments(_question(QuestionType.CONFIRMATION))
        harness = await _Harness.create(
            handler=handler,
            tool_confirm=confirmation,
            maximum_tool_cycles=1,
            responses=[
                _task_input_response(f"equivalent-{index}", arguments)
                for index in range(MAXIMUM_EQUIVALENT_INPUT_REQUESTS + 1)
            ],
        )
        try:
            response = await harness.response()
            with self.assertRaises(InteractionLoopLimitError):
                await _consume(response)

            assert response._execution is not None
            self.assertEqual(
                response._execution.interaction_count,
                MAXIMUM_EQUIVALENT_INPUT_REQUESTS,
            )
            self.assertEqual(
                len(handler.contexts), MAXIMUM_EQUIVALENT_INPUT_REQUESTS
            )
            self.assertEqual(response._tool_cycle_count, 0)
            self.assertEqual(response._tool_cycle_signatures, set())
            confirmation.assert_not_awaited()
        finally:
            await harness.close()

    async def test_clarification_words_do_not_approve_protected_action(
        self,
    ) -> None:
        for index, phrase in enumerate(("yes", "continue", "do it")):
            with self.subTest(phrase=phrase):
                executions: list[str] = []
                handler = _AnsweringHandler(text_value=phrase)
                confirmation = AsyncMock(return_value=False)
                harness = await _Harness.create(
                    handler=handler,
                    tool=_protected_tool_manager(executions),
                    tool_confirm=confirmation,
                    responses=[
                        _task_input_response(
                            f"clarification-{index}",
                            _input_arguments(_question(QuestionType.TEXT)),
                        ),
                        _provider_response(
                            f"protected-{index}",
                            calls=(
                                ToolCall(
                                    id=f"protected-call-{index}",
                                    name="protected_action",
                                    arguments={"command": "deploy"},
                                ),
                            ),
                        ),
                        _provider_response(
                            f"protected-{index}-final",
                            answer="not executed",
                        ),
                    ],
                )
                try:
                    response = await harness.response()
                    await _consume(response)
                    result = _model_result(harness.contexts[1])
                    answers = cast(
                        list[dict[str, object]],
                        result["answers"],
                    )

                    self.assertEqual(answers[0]["value"], phrase)
                    self.assertEqual(executions, [])
                    confirmation.assert_awaited_once()
                    confirmed_call = confirmation.await_args.args[0]
                    self.assertEqual(
                        confirmed_call.name, "ops.protected_action"
                    )
                    self.assertNotEqual(
                        confirmed_call.name,
                        RESERVED_INPUT_CAPABILITY_NAME,
                    )
                    self.assertFalse(response._tool_confirm_all)
                finally:
                    await harness.close()

    async def test_protected_action_waits_for_separate_explicit_approval(
        self,
    ) -> None:
        executions: list[str] = []
        handler = _AnsweringHandler(text_value="yes")
        approval = _ApprovalGate()
        harness = await _Harness.create(
            handler=handler,
            tool=_protected_tool_manager(executions),
            tool_confirm=approval,
            responses=[
                _task_input_response(
                    "approval-clarification",
                    _input_arguments(_question(QuestionType.TEXT)),
                ),
                _provider_response(
                    "approval-protected",
                    calls=(
                        ToolCall(
                            id="approval-protected-call",
                            name="protected_action",
                            arguments={"command": "deploy"},
                        ),
                    ),
                ),
                _provider_response("approval-final", answer="executed"),
            ],
        )
        try:
            response = await harness.response()
            consumption = create_task(_consume(response))
            await approval.started.wait()

            self.assertEqual(executions, [])
            self.assertEqual(len(approval.calls), 1)
            self.assertEqual(approval.calls[0].name, "ops.protected_action")

            approval.release.set()
            await consumption
            self.assertEqual(executions, ["deploy"])
        finally:
            await harness.close()
