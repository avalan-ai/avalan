"""Cover response cleanup and ownership failure boundaries."""

from asyncio import (
    CancelledError,
    Event,
    Task,
    create_task,
    gather,
    get_running_loop,
)
from collections.abc import AsyncIterator
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    UuidExecutionIdFactory,
)
from avalan.agent.orchestrator.response import orchestrator_response
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
    AgentId,
    BranchId,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputRequest,
    ModelCallId,
    PrincipalScope,
    RunId,
    StreamSessionId,
    TaskId,
    TurnId,
    UserId,
)
from avalan.model.call import ModelCallContext
from avalan.model.engine import Engine
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamProviderEvent,
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
    """Return one real non-stream provider response."""

    def output(**_: object) -> str:
        return text

    return TextGenerationResponse(
        output,
        logger=getLogger(),
        use_async_generator=False,
    )


def _response() -> OrchestratorResponse:
    """Return a response with controlled dependencies."""
    operation = _operation()
    message = Message(role=MessageRole.USER, content="hello")
    engine = create_autospec(Engine, instance=True, spec_set=True)
    engine.configure_mock(model_id="coverage-model", tokenizer=None)
    agent = create_autospec(EngineAgent, instance=True, spec_set=True)
    agent.configure_mock(engine=engine)
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


def _execution() -> AgentExecution:
    """Return one real running execution with stable identities."""
    origin = ExecutionOrigin(
        run_id=RunId("cleanup-run"),
        turn_id=TurnId("cleanup-turn"),
        task_id=TaskId("cleanup-task"),
        agent_id=AgentId("cleanup-agent"),
        branch_id=BranchId("cleanup-branch"),
        parent_branch_id=None,
        model_call_id=ModelCallId("cleanup-model-call"),
        stream_session_id=StreamSessionId("cleanup-stream"),
        definition=ExecutionDefinitionRef(
            agent_definition_locator="agent://cleanup-coverage",
            agent_definition_revision="agent-r1",
            operation_id="cleanup-operation",
            operation_index=0,
            model_config_reference="model-r1",
            tool_revision="tools-r1",
            capability_revision="capabilities-r1",
        ),
        principal=PrincipalScope(user_id=UserId("cleanup-user")),
    )
    return AgentExecution(
        origin=origin,
        id_factory=UuidExecutionIdFactory(),
        initial_messages=(),
    )


async def _failed_tool_batch_task(
    error: BaseException,
) -> Task[list[_ToolExecutionOutcome]]:
    """Return one observed, completed task retaining its failure."""

    async def fail() -> list[_ToolExecutionOutcome]:
        raise error

    completed = Event()
    task = create_task(fail(), name="failed-tool-batch")
    task.add_done_callback(lambda _: completed.set())
    await completed.wait()
    return task


async def _pending_tool_batch(release: Event) -> list[_ToolExecutionOutcome]:
    """Wait until the test releases one real tool batch."""
    await release.wait()
    return []


async def _pending_interaction(release: Event) -> InteractionRequestResult:
    """Wait until the test releases one real interaction task."""
    await release.wait()
    raise CancelledError


async def _wait_until_released(release: Event, started: Event) -> None:
    """Signal task startup and wait for explicit release."""
    started.set()
    await release.wait()


async def _resist_one_cancellation(release: Event, started: Event) -> None:
    """Remain pending after one cancellation until explicitly released."""
    started.set()
    try:
        await release.wait()
    except CancelledError:
        await release.wait()


def _cleanup_note(error: BaseException) -> str:
    """Return the production cleanup note for one secondary failure."""
    return (
        f"post-provider cleanup failure: {error.__class__.__name__}: {error}"
    )


def _stream_started_item() -> CanonicalStreamItem:
    """Return one valid provider stream-start item."""
    return CanonicalStreamItem(
        stream_session_id="provider-stream",
        run_id="provider-run",
        turn_id="provider-turn",
        sequence=0,
        kind=StreamItemKind.STREAM_STARTED,
        channel=StreamChannel.CONTROL,
    )


def _blocking_stream_response(
    started: Event,
    release: Event,
) -> TextGenerationResponse:
    """Return a real provider stream that blocks after its first item."""

    async def items() -> AsyncIterator[CanonicalStreamItem]:
        yield _stream_started_item()
        started.set()
        await release.wait()

    return TextGenerationResponse(
        lambda **_: items(),
        logger=getLogger(),
        use_async_generator=True,
    )


class OrchestratorResponseCleanupDeadlineCoverageTest(IsolatedAsyncioTestCase):
    """Exercise bounded cleanup and task-observation branches."""

    def test_completed_provider_event_is_not_appended(self) -> None:
        response = _response()
        before = response.canonical_items

        result = response._append_canonical_provider_event_item(
            StreamProviderEvent(kind=StreamItemKind.STREAM_COMPLETED)
        )

        self.assertIsNone(result)
        self.assertEqual(response.canonical_items, before)

    async def test_tool_continuation_cancellation_keeps_cleanup_failure_note(
        self,
    ) -> None:
        response = _response()
        response.__aiter__()
        call = ToolCall(id="call-1", name="lookup", arguments={})
        result = ToolCallResult(
            id="call-1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result={"ok": True},
        )
        response._tool_result_outcomes.put(
            _ToolExecutionOutcome(
                call=call,
                context=ToolCallContext(),
                planned_index=0,
                result=result,
                history_recorded=True,
            )
        )
        observation = Message(role=MessageRole.TOOL, content="observed")
        child_context = ModelCallContext(
            specification=response._operation.specification,
            input=response._input,
        )
        cleanup_failure = RuntimeError("continuation cleanup failed")

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
                return_value=True,
            ),
            patch.object(response, "_new_tool_context", return_value=None),
            patch.object(
                response,
                "_make_child_context",
                AsyncMock(return_value=child_context),
            ),
            patch.object(
                response,
                "_append_canonical_model_continuation",
                return_value=None,
            ),
            patch.object(
                response,
                "_raise_if_cancelled",
                AsyncMock(side_effect=CancelledError("caller cancelled")),
            ),
            patch.object(
                response,
                "_finalize_execution",
                AsyncMock(side_effect=cleanup_failure),
            ),
        ):
            with self.assertRaises(CancelledError) as raised:
                await response._next_item()

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_pending_batch_caller_cancellation_notes_cleanup_failure(
        self,
    ) -> None:
        response = _response()
        release = Event()
        pending = create_task(
            _pending_tool_batch(release),
            name="pending-tool-batch",
        )
        response._pending_tool_batch_task = pending
        cleanup_failure = RuntimeError("batch cleanup failed")

        try:
            with patch.object(
                response,
                "_cancel_pending_tool_batch",
                AsyncMock(side_effect=cleanup_failure),
            ):
                caller = create_task(
                    response._await_pending_tool_batch(),
                    name="pending-batch-caller",
                )
                get_running_loop().call_soon(caller.cancel)
                with self.assertRaises(CancelledError) as raised:
                    await caller
        finally:
            release.set()
            await pending

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_cancel_task_deadline_rejects_uncooperative_task(
        self,
    ) -> None:
        release = Event()
        started = Event()
        task = create_task(
            _resist_one_cancellation(release, started),
            name="uncooperative-cleanup",
        )
        await started.wait()

        try:
            with (
                patch.object(
                    orchestrator_response,
                    "_CLEANUP_TIMEOUT_SECONDS",
                    0.0,
                ),
                self.assertRaisesRegex(
                    TimeoutError,
                    "owned stage cleanup exceeded",
                ),
            ):
                await OrchestratorResponse._cancel_task_with_deadline(
                    task,
                    "owned stage",
                )
        finally:
            release.set()
            await task

        self.assertFalse(task.cancelled())

    async def test_cleanup_task_deadline_cancels_unfinished_task(self) -> None:
        release = Event()
        started = Event()
        task = create_task(
            _wait_until_released(release, started),
            name="unfinished-retryable-cleanup",
        )
        await started.wait()

        with (
            patch.object(
                orchestrator_response,
                "_CLEANUP_TIMEOUT_SECONDS",
                0.0,
            ),
            self.assertRaisesRegex(
                TimeoutError,
                "retryable stage cleanup exceeded",
            ),
        ):
            await OrchestratorResponse._await_cleanup_task_with_deadline(
                task,
                "retryable stage",
            )

        await gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())

    async def test_cleanup_observer_ignores_cancelled_task(self) -> None:
        release = Event()
        started = Event()
        task = create_task(
            _wait_until_released(release, started),
            name="cancelled-observed-cleanup",
        )
        await started.wait()
        task.cancel()
        await gather(task, return_exceptions=True)

        OrchestratorResponse._observe_cleanup_task(task)

        self.assertTrue(task.cancelled())

    async def test_cleanup_observer_ignores_pending_task(self) -> None:
        release = Event()
        started = Event()
        task = create_task(
            _wait_until_released(release, started),
            name="pending-observed-cleanup",
        )
        await started.wait()

        try:
            OrchestratorResponse._observe_cleanup_task(task)
            self.assertFalse(task.done())
        finally:
            task.cancel()
            await gather(task, return_exceptions=True)

    async def test_execution_cleanup_deadline_clears_cancelled_task(
        self,
    ) -> None:
        response = _response()
        response._execution = _execution()
        release = Event()
        started = Event()

        async def settle() -> tuple[BaseException, ...]:
            await _wait_until_released(release, started)
            return ()

        task = create_task(settle(), name="execution-cleanup")
        response._execution_cleanup_task = task
        await started.wait()

        with (
            patch.object(
                orchestrator_response,
                "_CLEANUP_TIMEOUT_SECONDS",
                0.0,
            ),
            self.assertRaisesRegex(
                TimeoutError,
                "execution terminalization cleanup exceeded",
            ),
        ):
            await response._settle_execution_with_deadline(cancelled=True)

        await gather(task, return_exceptions=True)
        self.assertTrue(task.cancelled())
        self.assertIsNone(response._execution_cleanup_task)

    async def test_close_provider_is_idempotent_after_cleanup(self) -> None:
        response = _response()
        response._provider_cleanup_complete = True
        close = AsyncMock()

        with patch.object(response._response, "aclose", close):
            await response._close_provider_response()

        close.assert_not_awaited()


class OrchestratorResponseFailureAggregationCoverageTest(
    IsolatedAsyncioTestCase
):
    """Exercise cleanup aggregation without replacing primary failures."""

    async def test_cancellation_cleanup_aggregates_independent_failures(
        self,
    ) -> None:
        response = _response()
        batch_failure = RuntimeError("batch cancellation failed")
        finalize_failure = ValueError("execution cancellation failed")
        terminal_failure = LookupError("terminal cancellation failed")

        with (
            patch.object(
                response,
                "_cancel_pending_tool_batch",
                AsyncMock(side_effect=batch_failure),
            ),
            patch.object(
                response,
                "_cancel_provider_response",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_cancel_pending_interaction",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_finalize_execution",
                AsyncMock(side_effect=finalize_failure),
            ),
            patch.object(
                response,
                "_discard_untrusted_response_tool_call_batch",
            ),
            patch.object(
                response,
                "_finish_canonical_stream",
                side_effect=terminal_failure,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await response._run_stream_cancellation_cleanup()

        self.assertIs(raised.exception, batch_failure)
        self.assertEqual(
            getattr(batch_failure, "__notes__", ()),
            [
                _cleanup_note(finalize_failure),
                _cleanup_note(terminal_failure),
            ],
        )

    async def test_error_cleanup_attempts_every_stage(self) -> None:
        response = _response()
        failures = (
            RuntimeError("batch close failed"),
            ValueError("provider close failed"),
            LookupError("execution error settlement failed"),
            OSError("interaction close failed"),
            AssertionError("terminal error failed"),
        )

        with (
            patch.object(
                response,
                "_cancel_pending_tool_batch",
                AsyncMock(side_effect=failures[0]),
            ),
            patch.object(
                response,
                "_close_provider_response",
                AsyncMock(side_effect=failures[1]),
            ),
            patch.object(
                response,
                "_finalize_execution",
                AsyncMock(side_effect=failures[2]),
            ),
            patch.object(
                response,
                "_cancel_pending_interaction",
                AsyncMock(side_effect=failures[3]),
            ),
            patch.object(
                response,
                "_discard_untrusted_response_tool_call_batch",
            ),
            patch.object(
                response,
                "_finish_canonical_stream",
                side_effect=failures[4],
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await response._run_stream_error_cleanup()

        self.assertIs(raised.exception, failures[0])
        self.assertEqual(
            getattr(failures[0], "__notes__", ()),
            [_cleanup_note(failure) for failure in failures[1:]],
        )

    def test_cleanup_failure_is_not_attached_to_itself(self) -> None:
        primary_failure = RuntimeError("primary")

        OrchestratorResponse._attach_cleanup_failures(
            primary_failure,
            [primary_failure],
        )

        self.assertEqual(getattr(primary_failure, "__notes__", ()), ())

    def test_completion_guard_rejects_unsettled_success(self) -> None:
        response = _response()

        with self.assertRaisesRegex(
            RuntimeError,
            "execution did not complete successfully",
        ):
            response._raise_if_completion_lost()

    def test_terminal_guard_rejects_errored_stream(self) -> None:
        response = _response()
        response._canonical_stream_terminal = StreamTerminalOutcome.ERRORED

        with self.assertRaisesRegex(
            RuntimeError,
            "execution failed before completion",
        ):
            response._raise_if_terminal_failure()

    async def test_continuation_install_failure_is_settled(self) -> None:
        response = _response()
        continuation_response = _text_response("continued")
        install_failure = RuntimeError("handoff acknowledgement failed")
        settle = AsyncMock()
        acknowledge = MagicMock(side_effect=install_failure)

        with (
            patch.object(
                response._engine_agent,
                "acknowledge_provider_handoff",
                acknowledge,
            ),
            patch.object(
                response,
                "_settle_continuation_handoff_failure",
                settle,
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await response._install_continuation_response(
                    continuation_response,
                    "continuation-1",
                    activate=False,
                )

        self.assertIs(raised.exception, install_failure)
        settle.assert_awaited_once_with(
            continuation_response,
            install_failure,
        )

    async def test_handoff_settlement_captures_all_result_shapes(
        self,
    ) -> None:
        response = _response()
        continuation_response = _text_response("continued")
        cancel_failure = RuntimeError("provider cancel failed")
        settle_failure = ValueError("execution settle failed")
        cancel = AsyncMock(side_effect=cancel_failure)
        close = AsyncMock()
        execution = _execution()
        settle_provider_exit = AsyncMock(return_value=(settle_failure,))
        response._execution = execution
        drain = AsyncMock(return_value=())
        primary_failure = CancelledError("handoff cancelled")

        with (
            patch.object(continuation_response, "cancel", cancel),
            patch.object(continuation_response, "aclose", close),
            patch.object(
                execution,
                "settle_provider_exit",
                settle_provider_exit,
            ),
            patch.object(
                response._engine_agent,
                "drain_pending_provider_cleanups",
                drain,
            ),
        ):
            await response._settle_continuation_handoff_failure(
                continuation_response,
                primary_failure,
            )

        settle_provider_exit.assert_awaited_once_with(cancelled=True)
        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [_cleanup_note(cancel_failure), _cleanup_note(settle_failure)],
        )

    async def test_stream_failure_respects_cancelled_execution(self) -> None:
        response = _response()
        execution = _execution()
        await execution.cancel()
        response._execution = execution

        with patch.object(
            response,
            "_provider_response_cleanup_is_complete",
            return_value=True,
        ):
            await response._settle_stream_failure(RuntimeError("provider"))

        self.assertTrue(response._provider_cleanup_complete)

    async def test_keyboard_interrupt_notes_cleanup_failure(self) -> None:
        response = _response()
        primary_failure = KeyboardInterrupt("interrupted")
        cleanup_failure = RuntimeError("cancellation convergence failed")

        with patch.object(
            response,
            "_converge_stream_cancellation",
            AsyncMock(side_effect=cleanup_failure),
        ):
            await response._settle_stream_failure(primary_failure)

        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_stream_failure_collects_all_cleanup_failures(self) -> None:
        response = _response()
        primary_failure = RuntimeError("provider failed")
        finalize_failure = ValueError("execution finalize failed")
        interaction_failure = LookupError("interaction cancel failed")
        terminal_failure = OSError("terminal projection failed")

        with (
            patch.object(
                response,
                "_finalize_execution",
                AsyncMock(side_effect=finalize_failure),
            ),
            patch.object(
                response,
                "_provider_response_cleanup_is_complete",
                return_value=False,
            ),
            patch.object(
                response,
                "_cancel_pending_interaction",
                AsyncMock(side_effect=interaction_failure),
            ),
            patch.object(
                response,
                "_discard_untrusted_response_tool_call_batch",
            ),
            patch.object(
                response,
                "_finish_canonical_stream",
                side_effect=terminal_failure,
            ),
        ):
            await response._settle_stream_failure(primary_failure)

        self.assertEqual(
            getattr(primary_failure, "__notes__", ()),
            [
                _cleanup_note(finalize_failure),
                _cleanup_note(interaction_failure),
                _cleanup_note(terminal_failure),
            ],
        )

    async def test_cancelled_tool_batch_notes_finalize_failure(self) -> None:
        response = _response()
        task = await _failed_tool_batch_task(CancelledError("batch cancelled"))
        response._pending_tool_batch_task = task
        cleanup_failure = RuntimeError("cancel finalization failed")

        with patch.object(
            response,
            "_finalize_execution",
            AsyncMock(side_effect=cleanup_failure),
        ):
            with self.assertRaises(CancelledError) as raised:
                await response._consume_pending_tool_batch(task)

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_aborted_tool_batch_notes_finalize_failure(self) -> None:
        response = _response()
        task = await _failed_tool_batch_task(CommandAbortException())
        response._pending_tool_batch_task = task
        cleanup_failure = RuntimeError("abort finalization failed")

        with patch.object(
            response,
            "_finalize_execution",
            AsyncMock(side_effect=cleanup_failure),
        ):
            with self.assertRaises(CommandAbortException) as raised:
                await response._consume_pending_tool_batch(task)

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_failed_tool_batch_notes_finalize_failure(self) -> None:
        response = _response()
        task = await _failed_tool_batch_task(ValueError("batch failed"))
        response._pending_tool_batch_task = task
        cleanup_failure = RuntimeError("error finalization failed")

        with patch.object(
            response,
            "_finalize_execution",
            AsyncMock(side_effect=cleanup_failure),
        ):
            with self.assertRaises(ValueError) as raised:
                await response._consume_pending_tool_batch(task)

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )


class OrchestratorResponseInteractionCleanupCoverageTest(
    IsolatedAsyncioTestCase
):
    """Exercise interaction polling and branch cleanup failures."""

    async def test_poll_caller_cancellation_notes_cleanup_failure(
        self,
    ) -> None:
        response = _response()
        release = Event()
        pending = create_task(
            _pending_interaction(release),
            name="pending-interaction",
        )
        response._pending_interaction_task = pending
        cleanup_failure = RuntimeError("interaction cleanup failed")

        try:
            with patch.object(
                response,
                "_cancel_pending_interaction",
                AsyncMock(side_effect=cleanup_failure),
            ):
                caller = create_task(
                    response._poll_pending_interaction(),
                    name="interaction-poll-caller",
                )
                get_running_loop().call_soon(caller.cancel)
                with self.assertRaises(CancelledError) as raised:
                    await caller
        finally:
            release.set()
            await gather(pending, return_exceptions=True)

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_poll_session_cancellation_notes_cleanup_failure(
        self,
    ) -> None:
        response = _response()
        release = Event()
        pending = create_task(
            _pending_interaction(release),
            name="session-pending-interaction",
        )
        response._pending_interaction_task = pending

        async def cancellation_checker() -> None:
            raise CancelledError("session cancelled")

        response._cancellation_checker = cancellation_checker
        cleanup_failure = RuntimeError("session interaction cleanup failed")

        try:
            with patch.object(
                response,
                "_cancel_pending_interaction",
                AsyncMock(side_effect=cleanup_failure),
            ):
                with self.assertRaises(CancelledError) as raised:
                    await response._poll_pending_interaction()
        finally:
            release.set()
            await gather(pending, return_exceptions=True)

        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )

    async def test_pending_interaction_task_cleanup_failure_is_raised(
        self,
    ) -> None:
        response = _response()
        release = Event()
        task = create_task(
            _pending_interaction(release),
            name="cleanup-pending-interaction",
        )
        response._pending_interaction_task = task
        cleanup_failure = RuntimeError("pending interaction task failed")

        try:
            with patch.object(
                response,
                "_cancel_task_with_deadline",
                AsyncMock(side_effect=cleanup_failure),
            ):
                with self.assertRaises(RuntimeError) as raised:
                    await response._cancel_pending_interaction()
        finally:
            release.set()
            await gather(task, return_exceptions=True)

        self.assertIs(raised.exception, cleanup_failure)

    async def test_interaction_cleanup_aggregates_branch_failures(
        self,
    ) -> None:
        response = _response()
        request = create_autospec(
            InputRequest,
            instance=True,
            spec_set=True,
        )
        runtime = create_autospec(
            AttachedInteractionRuntime,
            instance=True,
            spec_set=True,
        )
        execution = create_autospec(
            AgentExecution,
            instance=True,
            spec_set=True,
        )
        execution.configure_mock(
            interaction_runtime=runtime,
            pending_request=request,
            status=AgentExecutionStatus.RUNNING,
        )
        response._execution = execution
        settle_failure = RuntimeError("execution settlement failed")
        append_failure = ValueError("interaction cancellation failed")
        terminal_failure = LookupError("terminal cancellation failed")

        with (
            patch.object(
                response,
                "_settle_execution_with_deadline",
                AsyncMock(side_effect=settle_failure),
            ),
            patch.object(
                response,
                "_append_interaction_cancellation_if_open",
                AsyncMock(side_effect=append_failure),
            ),
            patch.object(
                response,
                "_finish_canonical_stream",
                side_effect=terminal_failure,
            ),
            patch.object(
                response,
                "_finalize_interaction_cleanup",
                AsyncMock(),
            ),
        ):
            with self.assertRaises(RuntimeError) as raised:
                await response._cancel_pending_interaction()

        self.assertIs(raised.exception, settle_failure)
        self.assertEqual(
            getattr(settle_failure, "__notes__", ()),
            [_cleanup_note(append_failure), _cleanup_note(terminal_failure)],
        )

    async def test_response_collection_cancellation_notes_cleanup_failure(
        self,
    ) -> None:
        response = _response()
        started = Event()
        release = Event()
        provider_response = _blocking_stream_response(started, release)
        cleanup_failure = RuntimeError("active response cleanup failed")

        try:
            with (
                patch.object(
                    response,
                    "_cancel_active_model_continuation_response",
                    AsyncMock(side_effect=cleanup_failure),
                ),
                patch.object(
                    response,
                    "_finish_active_model_continuation",
                ) as finish,
            ):
                collection = create_task(
                    response._response_text_and_calls(provider_response),
                    name="response-collection",
                )
                await started.wait()
                collection.cancel()
                with self.assertRaises(CancelledError) as raised:
                    await collection
        finally:
            release.set()
            await provider_response.aclose()

        finish.assert_called_once_with(
            StreamItemKind.MODEL_CONTINUATION_CANCELLED
        )
        self.assertEqual(
            getattr(raised.exception, "__notes__", ()),
            [_cleanup_note(cleanup_failure)],
        )
