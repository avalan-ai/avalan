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
    ExecutionTerminatedError,
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
from avalan.interaction.broker import InteractionRequestResult
from avalan.interaction.entities import (
    InputModelResult,
    InputRequest,
    InputRequestId,
    RequestState,
    ResolutionStatus,
    TerminateInputContinuation,
)
from avalan.interaction.store import CreateInteractionApplied
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
    call.questions = ("Continue?",)
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

    async def test_start_task_input_requires_attached_runtime(self) -> None:
        response = _response()
        response._execution = None

        with self.assertRaisesRegex(
            RuntimeError,
            "explicitly attached runtime",
        ):
            await response._start_task_input(_task_input_call())

    async def test_start_task_input_abandons_failed_scheduling(self) -> None:
        response = _response()
        call = _task_input_call()
        runtime = SimpleNamespace(actor=object(), handler=AsyncMock())
        broker = SimpleNamespace(request=MagicMock(return_value=object()))
        execution = MagicMock(spec=AgentExecution)
        execution.interaction_runtime = runtime
        execution.interaction_broker = broker
        execution.origin = object()
        execution.begin_interaction = AsyncMock()
        execution.abandon_interaction = AsyncMock()
        response._execution = cast(AgentExecution, execution)
        response._canonical_items = [
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                text_delta="provider preface",
            )
        ]

        with (
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.InteractionBrokerRequest",
                return_value=object(),
            ),
            patch(
                "avalan.agent.orchestrator.response."
                "orchestrator_response.create_task",
                side_effect=RuntimeError("task scheduling failed"),
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
