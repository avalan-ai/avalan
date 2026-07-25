"""Cover response convergence branches with deterministic behavior probes."""

from asyncio import (
    CancelledError,
    Future,
    Task,
    create_task,
    get_running_loop,
    sleep,
)
from asyncio import (
    Event as AsyncioEvent,
)
from datetime import UTC, datetime
from logging import getLogger
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    ExecutionTerminatedError,
    create_agent_execution,
)
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
    _ToolExecutionOutcome,
)
from avalan.cli import CommandAbortException
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    TransformerEngineSettings,
)
from avalan.interaction.broker import (
    InteractionBroker,
    InteractionRequestResult,
)
from avalan.interaction.entities import (
    AgentId,
    ConfirmationQuestion,
    ContinuationId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputModelResult,
    InputRequest,
    InputRequestId,
    InputUnavailableResult,
    PrincipalScope,
    QuestionId,
    RequestState,
    RequirementMode,
    ResolutionStatus,
    TerminateInputContinuation,
    TextQuestion,
    UserId,
)
from avalan.interaction.error import InputErrorCode, InputValidationError
from avalan.interaction.policy import (
    InteractionActor,
    InteractionPolicy,
    InteractionTime,
    RuntimeInteractionClock,
    TaskInputCapabilityState,
)
from avalan.interaction.store import (
    CreateInteractionApplied,
    CreateInteractionRejected,
)
from avalan.model.call import ModelCallContext
from avalan.model.capability import (
    ModelCapabilityCatalog,
    ModelCapabilityValidationError,
    TaskInputCapabilityCall,
)
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)


def _operation() -> AgentOperation:
    """Return one minimal text-generation operation."""
    return AgentOperation(
        specification=Specification(instructions="respond exactly"),
        environment=EngineEnvironment(
            engine_uri=EngineUri(
                host=None,
                port=None,
                user=None,
                password=None,
                vendor=None,
                model_id="coverage-model",
                params={},
            ),
            settings=TransformerEngineSettings(),
        ),
    )


def _text_response(text: str = "") -> TextGenerationResponse:
    """Return a real non-stream provider response."""

    def output(**_: object) -> str:
        return text

    return TextGenerationResponse(
        output,
        logger=getLogger(),
        use_async_generator=False,
    )


def _response() -> OrchestratorResponse:
    """Return a fully initialized response with controlled dependencies."""
    operation = _operation()
    message = Message(role=MessageRole.USER, content="hello")
    agent = MagicMock(spec=EngineAgent)
    agent.engine = SimpleNamespace(model_id="coverage-model", tokenizer=None)
    context = ModelCallContext(
        specification=operation.specification,
        input=message,
    )
    return OrchestratorResponse(
        message,
        _text_response(),
        agent,
        operation,
        {},
        context,
        enable_tool_parsing=False,
    )


def _task_input_call() -> TaskInputCapabilityCall:
    """Return a controlled reserved-call boundary value."""
    call = MagicMock(spec=TaskInputCapabilityCall)
    call.call_id = "input-call"
    call.provider_name = "request_user_input"
    call.arguments = {}
    call.mode = "required"
    call.reason = "Need a decision."
    call.questions = (
        ConfirmationQuestion(
            question_id=QuestionId("continue"),
            prompt="Continue?",
            required=True,
        ),
    )
    return cast(TaskInputCapabilityCall, call)


def _canonical_item(
    kind: StreamItemKind,
    *,
    text_delta: str | None = None,
    tool_call_id: str | None = None,
) -> CanonicalStreamItem:
    """Return one canonical item for direct state-transition tests."""
    return CanonicalStreamItem(
        stream_session_id="stream",
        run_id="run",
        turn_id="turn",
        sequence=0,
        kind=kind,
        channel=(
            StreamChannel.ANSWER
            if kind is StreamItemKind.ANSWER_DELTA
            else StreamChannel.TOOL_CALL
        ),
        text_delta=text_delta,
        correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
    )


def _pending_broker_result(
    request: InputRequest,
) -> InteractionRequestResult:
    """Return the minimum trusted-shape result consumed by response logic."""
    created = MagicMock(spec=CreateInteractionApplied)
    created.record.request = request
    delivery = SimpleNamespace(record=SimpleNamespace(request=request))
    return cast(
        InteractionRequestResult,
        SimpleNamespace(create_result=created, delivery=delivery),
    )


class OrchestratorResponseIterationCoverageTest(IsolatedAsyncioTestCase):
    """Exercise iterator and cancellation convergence branches."""

    async def test_aclose_delegates_after_completed_execution(self) -> None:
        response = _response()
        response._execution = cast(
            AgentExecution,
            SimpleNamespace(status=AgentExecutionStatus.COMPLETED),
        )
        close = AsyncMock()

        with patch.object(response._response, "aclose", close):
            await response.aclose()

        close.assert_awaited_once_with()

    async def test_terminal_iterator_waits_for_pending_tool_batch(
        self,
    ) -> None:
        response = _response()
        response._response_iterator = cast(AsyncIterator[Any], object())
        response._canonical_stream_terminal = StreamTerminalOutcome.COMPLETED
        gate = AsyncioEvent()

        async def finish_batch() -> list[Any]:
            await gate.wait()
            return []

        task = create_task(finish_batch())
        response._pending_tool_batch_task = cast(Task[Any], task)
        get_running_loop().call_soon(gate.set)

        with self.assertRaises(StopAsyncIteration):
            await response.__anext__()

        self.assertIsNone(response._pending_tool_batch_task)

    async def test_iterator_preserves_stop_without_canonical_item(
        self,
    ) -> None:
        response = _response()
        response._response_iterator = cast(AsyncIterator[Any], object())
        next_item = AsyncMock(side_effect=StopAsyncIteration())

        with patch.object(response, "_next_item", next_item):
            with self.assertRaises(StopAsyncIteration):
                await response.__anext__()

        next_item.assert_awaited_once_with()

    async def test_drained_reserved_call_starts_attached_interaction(
        self,
    ) -> None:
        response = _response()
        call = _task_input_call()
        response._response_iterator = cast(AsyncIterator[Any], object())
        response._response_drained = True
        response._staged_tool_batch_present = True
        response._task_input_call = call
        response._execution = cast(
            AgentExecution,
            SimpleNamespace(interaction_runtime=object()),
        )
        propagate = AsyncMock()
        start = AsyncMock()

        with (
            patch.object(
                response,
                "_propagate_cancellation_to_pending_work",
                propagate,
            ),
            patch.object(response, "_finish_active_model_continuation"),
            patch.object(response, "_drain_tool_call_batch", return_value=[]),
            patch.object(response, "_start_task_input", start),
        ):
            await response._next_item()

        propagate.assert_awaited_once_with()
        start.assert_awaited_once_with(call)
        self.assertIsNone(response._task_input_call)

    async def test_inactive_stream_input_installs_unavailable_response(
        self,
    ) -> None:
        """Replace a live stream when new task input is inactive."""

        async def provider_items() -> AsyncIterator[object]:
            yield object()

        response = _response()
        call = _task_input_call()
        broker = SimpleNamespace(
            request=AsyncMock(),
            cancel_scope=AsyncMock(),
        )
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("coverage-user"))
            ),
            handler=AsyncMock(),
            policy=InteractionPolicy(
                capability_state=TaskInputCapabilityState.DORMANT
            ),
        )
        execution = MagicMock(spec=AgentExecution)
        execution.interaction_runtime = runtime
        execution.begin_interaction = AsyncMock()
        response._execution = cast(AgentExecution, execution)
        response._response_iterator = provider_items()
        canonical = _canonical_item(
            StreamItemKind.TOOL_CALL_DONE,
            tool_call_id="input-call",
        )
        replacement = _text_response("unavailable")
        resume = AsyncMock(return_value=replacement)

        with (
            patch.object(
                response,
                "_propagate_cancellation_to_pending_work",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_canonical_item_from_response_item",
                return_value=canonical,
            ),
            patch.object(
                response,
                "_process_canonical_response_item",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_classify_completed_task_input_boundary",
                return_value=call,
            ),
            patch.object(response._response, "aclose", AsyncMock()),
            patch.object(response, "_finish_active_model_continuation"),
            patch.object(
                response,
                "_resume_unavailable_task_input",
                resume,
            ),
            patch.object(
                response,
                "_begin_tool_call_lifecycle_response",
            ) as begin_lifecycle,
        ):
            await response._next_item()

        self.assertIs(response._response, replacement)
        self.assertFalse(response._response_drained)
        execution.begin_interaction.assert_awaited_once()
        resume.assert_awaited_once()
        begin_lifecycle.assert_called_once_with()

    async def test_stream_cleanup_finishes_state_before_provider_error(
        self,
    ) -> None:
        response = _response()
        provider = AsyncMock(side_effect=RuntimeError("provider close failed"))
        pending = AsyncMock()
        finalize = AsyncMock()

        with (
            patch.object(response, "_cancel_pending_tool_batch", AsyncMock()),
            patch.object(response, "_cancel_provider_response", provider),
            patch.object(response, "_cancel_pending_interaction", pending),
            patch.object(response, "_finalize_execution", finalize),
            patch.object(
                response,
                "_discard_untrusted_response_tool_call_batch",
            ),
            patch.object(response, "_finish_canonical_stream") as finish,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "provider close failed",
            ):
                await response._run_stream_cancellation_cleanup()

        pending.assert_awaited_once_with()
        finalize.assert_awaited_once_with(StreamItemKind.STREAM_CANCELLED)
        finish.assert_called_once_with(StreamItemKind.STREAM_CANCELLED)

    async def test_completed_tool_outcome_records_execution_messages(
        self,
    ) -> None:
        response = _response()
        response._response_iterator = cast(AsyncIterator[Any], object())
        call = ToolCall(id="call-1", name="lookup", arguments={})
        result = ToolCallResult(
            id="call-1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result={"ok": True},
        )
        context = cast(ToolCallContext, MagicMock(spec=ToolCallContext))
        response._tool_result_outcomes.put(
            _ToolExecutionOutcome(
                call=call,
                context=context,
                planned_index=0,
                result=result,
                history_recorded=True,
            )
        )
        execution = MagicMock(spec=AgentExecution)
        execution.record_messages = AsyncMock()
        response._execution = cast(AgentExecution, execution)
        observation = Message(role=MessageRole.TOOL, content="observed")

        with (
            patch.object(
                response,
                "_propagate_cancellation_to_pending_work",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_provider_facing_tool_outcome",
                return_value=result,
            ),
            patch.object(
                response,
                "_provider_facing_tool_call",
                return_value=call,
            ),
            patch.object(
                response,
                "_tool_observation_messages",
                return_value=[observation],
            ),
            patch.object(
                response,
                "_should_continue_tool_cycle",
                return_value=False,
            ),
            patch.object(response, "_finalize_execution", AsyncMock()),
            patch.object(response, "_finish_canonical_stream"),
        ):
            with self.assertRaises(StopAsyncIteration):
                await response._next_item()

        execution.record_messages.assert_awaited_once_with((observation,))

    async def test_abort_batch_with_execution_is_consumed(self) -> None:
        response = _response()

        async def abort() -> list[_ToolExecutionOutcome]:
            raise CommandAbortException()

        task = create_task(abort())
        await sleep(0)
        response._pending_tool_batch_task = task
        response._execution = cast(
            AgentExecution,
            MagicMock(spec=AgentExecution),
        )
        finalize = AsyncMock()

        with patch.object(response, "_finalize_execution", finalize):
            await response._consume_pending_tool_batch(task)

        finalize.assert_awaited_once_with(StreamItemKind.STREAM_CANCELLED)
        self.assertIsNone(response._pending_tool_batch_task)


class OrchestratorResponseInteractionCoverageTest(IsolatedAsyncioTestCase):
    """Exercise attached-interaction admission and polling boundaries."""

    async def test_start_task_input_requires_interaction_runtime(self) -> None:
        response = _response()
        response._execution = None

        with self.assertRaisesRegex(
            RuntimeError,
            "^task input requires an explicit interaction runtime$",
        ):
            await response._start_task_input(_task_input_call())

    async def test_sensitive_input_is_rejected_before_ledger_reservation(
        self,
    ) -> None:
        response = _response()
        broker = SimpleNamespace(
            request=AsyncMock(),
            cancel_scope=AsyncMock(),
        )
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("coverage-user"))
            ),
            handler=AsyncMock(),
        )
        execution = MagicMock(spec=AgentExecution)
        execution.interaction_runtime = runtime
        execution.begin_interaction = AsyncMock()
        response._execution = cast(AgentExecution, execution)
        call = _task_input_call()
        cast(Any, call).questions = (
            TextQuestion(
                question_id=QuestionId("login-code"),
                prompt="Enter your sign-in code.",
                required=True,
            ),
        )

        with self.assertRaises(InputValidationError) as error:
            await response._start_task_input(call)

        self.assertIs(error.exception.code, InputErrorCode.PROHIBITED_INPUT)
        execution.begin_interaction.assert_not_awaited()
        broker.request.assert_not_awaited()
        self.assertIsNone(response._pending_interaction_task)

    async def test_start_task_input_abandons_failed_scheduling(self) -> None:
        response = _response()
        call = _task_input_call()
        broker = SimpleNamespace(
            request=AsyncMock(),
            cancel_scope=AsyncMock(),
        )
        runtime = AttachedInteractionRuntime(
            broker=cast(InteractionBroker, broker),
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("coverage-user"))
            ),
            handler=AsyncMock(),
        )
        execution = MagicMock(spec=AgentExecution)
        execution.interaction_runtime = runtime
        execution.interaction_broker = broker
        execution.origin = object()
        execution.begin_interaction = AsyncMock()
        execution.abandon_interaction = AsyncMock()
        execution.status = AgentExecutionStatus.PREPARING_INPUT
        response._execution = cast(AgentExecution, execution)
        response._canonical_items = [
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                text_delta="provider preface",
            )
        ]

        def fail_scheduling(coroutine: Any, **_: object) -> None:
            coroutine.close()
            raise RuntimeError("task scheduling failed")

        with (
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.InteractionBrokerRequest",
                return_value=object(),
            ),
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.create_task",
                side_effect=fail_scheduling,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "task scheduling failed",
            ):
                await response._start_task_input(call)

        execution.begin_interaction.assert_awaited_once()
        reserved_message = execution.begin_interaction.await_args.args[2]
        self.assertEqual(reserved_message.role, MessageRole.ASSISTANT)
        self.assertEqual(reserved_message.content, "provider preface")
        self.assertEqual(reserved_message.tool_calls[0].id, "input-call")
        execution.abandon_interaction.assert_awaited_once_with()
        self.assertIsNone(response._pending_interaction_call)
        self.assertEqual(response._pending_interaction_assistant_text, "")
        self.assertFalse(response._pending_interaction_published)

    async def test_poll_caller_cancellation_cleans_local_waiters(self) -> None:
        response = _response()
        never = AsyncioEvent()
        pending = create_task(never.wait())
        response._pending_interaction_task = cast(Task[Any], pending)

        async def check_cancellation() -> None:
            await never.wait()

        response._cancellation_checker = check_cancellation
        cancel_pending = AsyncMock()

        try:
            with (
                patch(
                    "avalan.agent.orchestrator.response."
                    "orchestrator_response.wait",
                    AsyncMock(side_effect=CancelledError()),
                ),
                patch.object(
                    response,
                    "_cancel_pending_interaction",
                    cancel_pending,
                ),
            ):
                with self.assertRaises(CancelledError):
                    await response._poll_pending_interaction()
        finally:
            pending.cancel()

        cancel_pending.assert_awaited_once_with()

    async def test_poll_session_cancellation_cancels_interaction(self) -> None:
        response = _response()
        loop = get_running_loop()
        pending: Future[Any] = loop.create_future()
        item_available: Future[Any] = loop.create_future()
        cancellation: Future[Any] = loop.create_future()
        cancellation.set_result(None)
        response._pending_interaction_task = cast(Task[Any], pending)
        response._cancellation_checker = AsyncMock()
        cancel_pending = AsyncMock()
        scheduled = iter((item_available, cancellation))

        def schedule(coroutine: Any, **_: object) -> Future[Any]:
            coroutine.close()
            return next(scheduled)

        with (
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.create_task",
                side_effect=schedule,
            ),
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.wait",
                AsyncMock(return_value=({cancellation}, set())),
            ),
            patch.object(
                response,
                "_cancel_pending_interaction",
                cancel_pending,
            ),
        ):
            with self.assertRaises(CancelledError):
                await response._poll_pending_interaction()

        self.assertTrue(item_available.cancelled())
        cancel_pending.assert_awaited_once_with()
        pending.cancel()

    async def test_poll_item_notification_cancels_session_watcher(
        self,
    ) -> None:
        response = _response()
        loop = get_running_loop()
        pending: Future[Any] = loop.create_future()
        item_available: Future[Any] = loop.create_future()
        cancellation: Future[Any] = loop.create_future()
        item_available.set_result(True)
        response._pending_interaction_task = cast(Task[Any], pending)
        response._cancellation_checker = AsyncMock()
        scheduled = iter((item_available, cancellation))

        def schedule(coroutine: Any, **_: object) -> Future[Any]:
            coroutine.close()
            return next(scheduled)

        with (
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.create_task",
                side_effect=schedule,
            ),
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.wait",
                AsyncMock(return_value=({item_available}, set())),
            ),
        ):
            await response._poll_pending_interaction()

        self.assertTrue(cancellation.cancelled())
        self.assertFalse(pending.done())
        pending.cancel()

    async def test_finish_task_input_prioritizes_run_cancellation(
        self,
    ) -> None:
        response = _response()
        pending_task = cast(Task[Any], object())
        pending_call = _task_input_call()
        response._pending_interaction_task = pending_task
        response._pending_interaction_call = pending_call
        response._pending_interaction_assistant_text = "preface"
        response._pending_interaction_published = True
        checker = AsyncMock(side_effect=CancelledError())
        response._cancellation_checker = checker

        with self.assertRaises(CancelledError):
            await response._finish_task_input(
                cast(InteractionRequestResult, object()),
                raise_on_noncompletion=False,
            )

        checker.assert_awaited_once_with()
        self.assertIs(response._pending_interaction_task, pending_task)
        self.assertIs(response._pending_interaction_call, pending_call)
        self.assertEqual(
            response._pending_interaction_assistant_text,
            "preface",
        )
        self.assertTrue(response._pending_interaction_published)

    async def test_finish_task_input_rejects_failed_admission(self) -> None:
        response = _response()
        execution = MagicMock(spec=AgentExecution)
        execution.abandon_interaction = AsyncMock()
        response._execution = cast(AgentExecution, execution)
        response._pending_interaction_task = cast(Task[Any], object())
        response._pending_interaction_call = _task_input_call()
        response._pending_interaction_assistant_text = "preface"
        response._pending_interaction_published = True
        result = cast(
            InteractionRequestResult,
            SimpleNamespace(create_result=object(), delivery=None),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "interaction admission was rejected",
        ):
            await response._finish_task_input(
                result,
                raise_on_noncompletion=False,
            )

        execution.abandon_interaction.assert_awaited_once_with()
        self.assertIsNone(response._pending_interaction_task)
        self.assertIsNone(response._pending_interaction_call)
        self.assertEqual(response._pending_interaction_assistant_text, "")
        self.assertFalse(response._pending_interaction_published)

    async def test_capability_loss_resumes_with_unavailable_result(
        self,
    ) -> None:
        response = _response()
        execution = MagicMock(spec=AgentExecution)
        response._execution = cast(AgentExecution, execution)
        created = InputRequest(
            request_id=InputRequestId("unavailable-request"),
            continuation_id=ContinuationId("unavailable-continuation"),
            origin=cast(
                ExecutionOrigin,
                MagicMock(spec=ExecutionOrigin),
            ),
            mode=RequirementMode.REQUIRED,
            reason="Need a decision.",
            questions=(
                ConfirmationQuestion(
                    question_id=QuestionId("continue"),
                    prompt="Continue?",
                    required=True,
                ),
            ),
            created_at=datetime(2099, 7, 24, tzinfo=UTC),
        )
        continued = _text_response("continued")
        resume = AsyncMock(return_value=continued)
        publish = AsyncMock()
        append = AsyncMock()

        with (
            patch.object(response, "_publish_interaction_wait", publish),
            patch.object(response, "_append_interaction_terminal", append),
            patch.object(response, "_resume_after_task_input", resume),
        ):
            result = await response._resume_unavailable_task_input(
                _task_input_call(),
                assistant_text="provider preface",
                created=created,
            )

        self.assertIs(result, continued)
        assert publish.await_args is not None
        assert append.await_args is not None
        assert resume.await_args is not None
        pending = publish.await_args.args[0]
        terminal = append.await_args.args[0]
        self.assertIs(pending.state, RequestState.PENDING)
        self.assertIs(terminal.state, RequestState.UNAVAILABLE)
        self.assertIsNotNone(terminal.resolution)
        assert terminal.resolution is not None
        self.assertEqual(
            terminal.resolution.resolved_at,
            created.created_at,
        )
        model_result = resume.await_args.args[2]
        self.assertIsInstance(model_result, InputUnavailableResult)
        self.assertEqual(model_result.request_id, created.request_id)

    async def test_capability_loss_without_request_uses_trusted_clock(
        self,
    ) -> None:
        """Use durable runtime time for synthesized capability loss."""
        response = _response()
        trusted_time = datetime(2099, 8, 25, tzinfo=UTC)
        clock_reads = 0

        def wall_clock() -> datetime:
            nonlocal clock_reads
            clock_reads += 1
            return trusted_time

        async def unused_stager(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError((args, kwargs))

        principal = PrincipalScope(user_id=UserId("durable-owner"))
        runtime = DurableInteractionRuntime(
            actor=InteractionActor(principal=principal),
            stager=unused_stager,
            clock=RuntimeInteractionClock(wall_clock),
        )
        execution = await create_agent_execution(
            definition=ExecutionDefinitionRef(
                agent_definition_locator="file:///trusted/agent.toml",
                agent_definition_revision="agent-r1",
                operation_id="operation",
                operation_index=0,
                model_config_reference="model",
                tool_revision="tools-r1",
                capability_revision="capabilities-r1",
            ),
            agent_id=AgentId("durable-agent"),
            principal=principal,
            initial_messages=(
                Message(role=MessageRole.USER, content="hello"),
            ),
            interaction_runtime=runtime,
        )
        response._execution = execution
        call = _task_input_call()
        object.__setattr__(call, "mode", RequirementMode.REQUIRED)
        continued = _text_response("continued")
        resume = AsyncMock(return_value=continued)
        publish = AsyncMock()
        append = AsyncMock()

        with (
            patch.object(response, "_publish_interaction_wait", publish),
            patch.object(response, "_append_interaction_terminal", append),
            patch.object(response, "_resume_after_task_input", resume),
        ):
            result = await response._resume_unavailable_task_input(
                call,
                assistant_text="provider preface",
            )

        self.assertIs(result, continued)
        self.assertIsNone(execution.interaction_broker)
        self.assertIs(execution.interaction_clock, runtime.clock)
        self.assertEqual(clock_reads, 1)
        assert publish.await_args is not None
        assert append.await_args is not None
        pending = publish.await_args.args[0]
        terminal = append.await_args.args[0]
        self.assertEqual(pending.created_at, trusted_time)
        self.assertIsNotNone(terminal.resolution)
        assert terminal.resolution is not None
        self.assertEqual(terminal.resolution.resolved_at, trusted_time)

    async def test_capability_loss_uses_attached_broker_time(self) -> None:
        """Use the attached broker when no runtime clock is exposed."""
        response = _response()
        trusted_time = datetime(2099, 8, 26, tzinfo=UTC)
        observed_at = InteractionTime.from_clock(
            wall_time=trusted_time,
            monotonic_seconds=1,
        )
        read_time = AsyncMock(return_value=observed_at)
        execution = SimpleNamespace(
            interaction_clock=None,
            interaction_broker=SimpleNamespace(read_time=read_time),
            origin=MagicMock(spec=ExecutionOrigin),
        )
        response._execution = cast(AgentExecution, execution)
        call = _task_input_call()
        object.__setattr__(call, "mode", RequirementMode.REQUIRED)
        continued = _text_response("continued")
        publish = AsyncMock()

        with (
            patch.object(response, "_publish_interaction_wait", publish),
            patch.object(
                response,
                "_append_interaction_terminal",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_resume_after_task_input",
                AsyncMock(return_value=continued),
            ),
        ):
            result = await response._resume_unavailable_task_input(
                call,
                assistant_text="provider preface",
            )

        self.assertIs(result, continued)
        read_time.assert_awaited_once_with()
        assert publish.await_args is not None
        self.assertEqual(publish.await_args.args[0].created_at, trusted_time)

    async def test_capability_loss_rejects_missing_or_invalid_time(
        self,
    ) -> None:
        """Fail closed without a genuine clock or broker observation."""
        call = _task_input_call()
        object.__setattr__(call, "mode", RequirementMode.REQUIRED)
        cases = (
            (
                None,
                "capability loss requires a trusted interaction clock",
            ),
            (
                SimpleNamespace(read_time=AsyncMock(return_value=object())),
                "trusted interaction clock returned invalid time",
            ),
        )
        for broker, message in cases:
            response = _response()
            response._execution = cast(
                AgentExecution,
                SimpleNamespace(
                    interaction_clock=None,
                    interaction_broker=broker,
                ),
            )
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    await response._resume_unavailable_task_input(
                        call,
                        assistant_text="provider preface",
                    )

    async def test_attached_input_returns_immediate_unavailable_response(
        self,
    ) -> None:
        """Return the immediate policy response without awaiting a task."""
        response = _response()
        response._execution = cast(
            AgentExecution,
            SimpleNamespace(
                interaction_runtime=object.__new__(AttachedInteractionRuntime)
            ),
        )
        replacement = _text_response("unavailable")
        start = AsyncMock(return_value=replacement)

        with patch.object(response, "_start_task_input", start):
            result = await response._run_attached_task_input(
                _task_input_call()
            )

        self.assertIs(result, replacement)
        start.assert_awaited_once()

    async def test_finish_unavailable_admission_resumes_model(self) -> None:
        """Map a broker availability rejection to a model continuation."""
        response = _response()
        call = _task_input_call()
        created = MagicMock(spec=InputRequest)
        rejected = MagicMock(spec=CreateInteractionRejected)
        rejected.error.code = InputErrorCode.UNAVAILABLE
        rejected.command.request = created
        response._pending_interaction_task = cast(Task[Any], object())
        response._pending_interaction_call = call
        response._pending_interaction_assistant_text = "provider preface"
        continued = _text_response("continued")
        resume = AsyncMock(return_value=continued)

        with patch.object(
            response,
            "_resume_unavailable_task_input",
            resume,
        ):
            result = await response._finish_task_input(
                cast(
                    InteractionRequestResult,
                    SimpleNamespace(create_result=rejected, delivery=None),
                ),
                raise_on_noncompletion=False,
            )

        self.assertIs(result, continued)
        resume.assert_awaited_once_with(
            call,
            assistant_text="provider preface",
            created=created,
        )

    async def _finish_terminated(
        self,
        *,
        raise_on_noncompletion: bool,
    ) -> tuple[OrchestratorResponse, TerminateInputContinuation]:
        response = _response()
        request = MagicMock(spec=InputRequest)
        request.state = RequestState.CANCELLED
        request.request_id = InputRequestId("terminated-request")
        outcome = TerminateInputContinuation(
            request_id=request.request_id,
            status=ResolutionStatus.CANCELLED,
        )
        execution = MagicMock(spec=AgentExecution)
        execution.record_interaction_termination = AsyncMock()
        response._execution = cast(AgentExecution, execution)
        response._pending_interaction_task = cast(Task[Any], object())
        response._pending_interaction_call = _task_input_call()
        response._pending_interaction_published = True

        with (
            patch.object(
                response,
                "_publish_interaction_wait",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_append_interaction_terminal",
                AsyncMock(),
            ),
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.project_resolution_to_model",
                return_value=outcome,
            ),
            patch.object(response, "_finish_canonical_stream") as finish,
        ):
            result = await response._finish_task_input(
                _pending_broker_result(cast(InputRequest, request)),
                raise_on_noncompletion=raise_on_noncompletion,
            )

        self.assertIsNone(result)
        execution.record_interaction_termination.assert_awaited_once_with(
            request,
            outcome,
        )
        finish.assert_called_once_with(StreamItemKind.STREAM_CANCELLED)
        return response, outcome

    async def test_finish_terminated_returns_for_streaming_consumer(
        self,
    ) -> None:
        response, _ = await self._finish_terminated(
            raise_on_noncompletion=False,
        )

        self.assertTrue(response._execution_terminated)
        self.assertFalse(response._pending_interaction_published)

    async def test_finish_terminated_raises_for_materialized_consumer(
        self,
    ) -> None:
        with self.assertRaises(ExecutionTerminatedError):
            await self._finish_terminated(raise_on_noncompletion=True)

    async def test_append_terminal_rejects_nonterminal_request(self) -> None:
        response = _response()
        request = MagicMock(spec=InputRequest)
        request.state = RequestState.CREATED

        with self.assertRaisesRegex(
            RuntimeError,
            "nonterminal interaction",
        ):
            await response._append_interaction_terminal(
                cast(InputRequest, request)
            )

    async def test_resume_rejects_already_applied_continuation(self) -> None:
        response = _response()
        call = _task_input_call()
        request = cast(InputRequest, MagicMock(spec=InputRequest))
        result = cast(InputModelResult, MagicMock(spec=InputModelResult))
        correlated = MagicMock()
        correlated.local_message.return_value = Message(
            role=MessageRole.TOOL,
            content="decision",
        )
        capability = MagicMock(spec=ModelCapabilityCatalog)
        capability.project_result.return_value = correlated
        execution = MagicMock(spec=AgentExecution)
        execution.record_interaction_result = AsyncMock(return_value=False)
        response._capability_catalog = cast(ModelCapabilityCatalog, capability)
        response._execution = cast(AgentExecution, execution)

        with self.assertRaisesRegex(
            RuntimeError,
            "already applied",
        ):
            await response._resume_after_task_input(
                call,
                request,
                result,
                assistant_text="preface",
            )

    async def test_resume_records_cancelled_model_continuation(self) -> None:
        response = _response()
        call = _task_input_call()
        request = cast(InputRequest, MagicMock(spec=InputRequest))
        result = cast(InputModelResult, MagicMock(spec=InputModelResult))
        correlated = MagicMock()
        correlated.local_message.return_value = Message(
            role=MessageRole.TOOL,
            content="decision",
        )
        capability = MagicMock(spec=ModelCapabilityCatalog)
        capability.project_result.return_value = correlated
        execution = MagicMock(spec=AgentExecution)
        execution.record_interaction_result = AsyncMock(return_value=True)
        execution.messages = (Message(role=MessageRole.USER, content="hello"),)
        response._capability_catalog = cast(ModelCapabilityCatalog, capability)
        response._execution = cast(AgentExecution, execution)
        child = ModelCallContext(
            specification=response._operation.specification,
            input=response._input,
            execution_origin=cast(
                Any,
                SimpleNamespace(model_call_id="model-turn"),
            ),
        )
        append = MagicMock(return_value=object())

        with (
            patch.object(response, "_new_tool_context", return_value=None),
            patch.object(
                response,
                "_make_child_context",
                AsyncMock(return_value=child),
            ),
            patch.object(
                response,
                "_append_canonical_model_continuation",
                append,
            ),
            patch.object(
                response,
                "_trigger_canonical_observability_event",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_await_with_session_cancellation",
                AsyncMock(side_effect=CancelledError()),
            ),
        ):
            with self.assertRaises(CancelledError):
                await response._resume_after_task_input(
                    call,
                    request,
                    result,
                    assistant_text="preface",
                )

        self.assertEqual(append.call_count, 2)
        self.assertEqual(
            append.call_args_list[-1].args,
            (StreamItemKind.MODEL_CONTINUATION_CANCELLED, "model-turn"),
        )

    async def test_unpublished_interaction_needs_no_canonical_cancel(
        self,
    ) -> None:
        response = _response()
        request = MagicMock(spec=InputRequest)
        request.request_id = InputRequestId("unpublished-request")

        await response._append_interaction_cancellation_if_open(
            cast(InputRequest, request)
        )

        self.assertEqual(response.canonical_items, ())


class OrchestratorResponseBoundaryCoverageTest(IsolatedAsyncioTestCase):
    """Exercise reserved-call, context, transcript, and terminal guards."""

    def test_reserved_call_validation_failure_restages_batch(self) -> None:
        response = _response()
        execution = MagicMock(spec=AgentExecution)
        execution.interaction_runtime = object()
        capability = MagicMock(spec=ModelCapabilityCatalog)
        capability.canonical_name.side_effect = ModelCapabilityValidationError(
            "invalid_provider_call",
            "provider name is invalid",
        )
        response._execution = cast(AgentExecution, execution)
        response._capability_catalog = cast(ModelCapabilityCatalog, capability)
        call = ToolCall(id="call-1", name="request_user_input", arguments={})
        response._calls.put(call)
        item = _canonical_item(
            StreamItemKind.TOOL_CALL_DONE,
            tool_call_id="call-1",
        )

        result = response._classify_completed_task_input_boundary(item)

        self.assertIsNone(result)
        self.assertIs(response._calls.get(), call)

    async def test_child_context_can_reuse_current_execution_origin(
        self,
    ) -> None:
        response = _response()
        origin = object()
        execution = MagicMock(spec=AgentExecution)
        execution.origin = origin
        execution.advance_model_turn = AsyncMock()
        response._execution = cast(AgentExecution, execution)

        child = await response._make_child_context(
            Message(role=MessageRole.USER, content="continue"),
            advance_turn=False,
        )

        self.assertIs(child.execution_origin, origin)
        execution.advance_model_turn.assert_not_awaited()

    def test_canonical_answer_text_ignores_other_channels(self) -> None:
        response = _response()
        response._canonical_items = [
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                text_delta="first",
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                tool_call_id="ignored-call",
            ),
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                text_delta=" second",
            ),
        ]

        self.assertEqual(response._canonical_answer_text(), "first second")

    async def test_finalize_execution_rejects_nonterminal_kind(self) -> None:
        response = _response()
        execution = MagicMock(spec=AgentExecution)
        execution.status = AgentExecutionStatus.RUNNING
        response._execution = cast(AgentExecution, execution)
        response._execution_finalized = False

        with self.assertRaisesRegex(
            ValueError,
            "unsupported execution terminal kind",
        ):
            await response._finalize_execution(StreamItemKind.ANSWER_DONE)

        self.assertFalse(response._execution_finalized)
