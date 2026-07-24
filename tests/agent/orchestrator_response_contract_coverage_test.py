"""Exercise fail-closed orchestrator response contract boundaries."""

from asyncio import Task, create_task
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import AsyncMock, MagicMock, patch

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.execution import (
    AgentExecution,
    AgentExecutionStatus,
    DurableInteractionRuntime,
)
from avalan.agent.orchestrator.response import orchestrator_response
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    TransformerEngineSettings,
)
from avalan.event.manager import EventManager
from avalan.interaction import (
    DurableInteractionSuspension,
    InteractionActor,
    PrincipalScope,
    RequestState,
    UserId,
)
from avalan.model.call import ModelCallContext
from avalan.model.capability import TaskInputCapabilityCall
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import StreamItemKind


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


def _text_response() -> TextGenerationResponse:
    """Return one real non-stream provider response."""

    def output(**_: object) -> str:
        return ""

    return TextGenerationResponse(
        output,
        logger=getLogger(),
        use_async_generator=False,
    )


def _response(
    *,
    event_manager: EventManager | None = None,
) -> OrchestratorResponse:
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
        event_manager=event_manager,
    )


def _task_input_call() -> TaskInputCapabilityCall:
    """Return one controlled reserved task-input call."""
    call = MagicMock(spec=TaskInputCapabilityCall)
    call.call_id = "input-call"
    call.provider_name = "request_user_input"
    call.arguments = {}
    call.mode = "required"
    call.reason = "Need a decision."
    call.questions = ("Continue?",)
    return cast(TaskInputCapabilityCall, call)


def _durable_suspension(
    *,
    actor: object,
    request: object,
    continuation: object = None,
) -> DurableInteractionSuspension:
    """Return an exact-type staging result with controlled raw fields."""
    durable = object.__new__(DurableInteractionSuspension)
    object.__setattr__(
        durable,
        "command",
        SimpleNamespace(actor=actor, request=request),
    )
    object.__setattr__(durable, "continuation", continuation)
    return durable


class OrchestratorResponseContractCoverageTest(TestCase):
    """Exercise synchronous response contract validation."""

    def test_event_manager_property_returns_exact_owner(self) -> None:
        event_manager = MagicMock(spec=EventManager)
        response = _response(event_manager=event_manager)

        self.assertIs(response.event_manager, event_manager)

    def test_durable_staging_rejects_invalid_result_actor_and_request(
        self,
    ) -> None:
        actor = object()
        origin = object()
        request_spec = SimpleNamespace(
            actor=actor,
            origin=origin,
            mode="required",
            reason="Need a decision.",
            questions=("Continue?",),
            continuation_ttl_seconds=60,
            advisory_wait_seconds=None,
        )
        staging = cast(Any, object())

        with self.assertRaises(TypeError):
            OrchestratorResponse._validate_durable_staging(
                cast(Any, request_spec),
                cast(DurableInteractionSuspension, object()),
                staging=staging,
            )

        request = SimpleNamespace(
            origin=origin,
            mode="required",
            reason="Need a decision.",
            questions=("Continue?",),
            continuation_ttl_seconds=60,
            advisory_wait_seconds=None,
            state=RequestState.CREATED,
        )
        with self.assertRaises(RuntimeError):
            OrchestratorResponse._validate_durable_staging(
                cast(Any, request_spec),
                _durable_suspension(
                    actor=object(),
                    request=request,
                ),
                staging=staging,
            )

        request.reason = "Changed in staging."
        with self.assertRaises(RuntimeError):
            OrchestratorResponse._validate_durable_staging(
                cast(Any, request_spec),
                _durable_suspension(
                    actor=actor,
                    request=request,
                ),
                staging=staging,
            )

    def test_durable_staging_context_requires_catalog(self) -> None:
        response = _response()
        response._capability_catalog = None

        with self.assertRaises(RuntimeError):
            response._durable_staging_context(
                _task_input_call(),
                cast(AgentExecution, object()),
            )

    def test_durable_staging_context_rejects_snapshot_model_call_drift(
        self,
    ) -> None:
        response = _response()
        adapter = SimpleNamespace(
            export_continuation_snapshot=MagicMock(
                return_value=SimpleNamespace(model_call_id="other-call")
            ),
            validate_continuation_snapshot_call=MagicMock(),
        )
        response._response = cast(
            TextGenerationResponse,
            SimpleNamespace(continuation_snapshot_adapter=adapter),
        )
        response._capability_catalog = cast(
            Any,
            SimpleNamespace(
                revision_binding=object(),
                support=SimpleNamespace(
                    continuation_snapshot_codec_registry=object(),
                    continuation_snapshot_codec=object(),
                ),
            ),
        )
        execution = cast(
            AgentExecution,
            SimpleNamespace(
                origin=SimpleNamespace(model_call_id="expected-call")
            ),
        )

        with self.assertRaises(RuntimeError):
            response._durable_staging_context(
                _task_input_call(),
                execution,
            )


class OrchestratorResponseAsyncContractCoverageTest(IsolatedAsyncioTestCase):
    """Exercise asynchronous task-input and cleanup contract guards."""

    async def test_durable_task_input_must_suspend_execution(self) -> None:
        response = _response()
        response._tool_context = object()
        response._task_input_call = _task_input_call()
        response._execution = cast(
            AgentExecution,
            SimpleNamespace(
                interaction_runtime=object.__new__(DurableInteractionRuntime)
            ),
        )

        with (
            patch.object(
                response,
                "_response_text_and_calls",
                AsyncMock(return_value=("preface", [])),
            ),
            patch.object(
                response,
                "_start_task_input",
                AsyncMock(),
            ) as start,
            self.assertRaises(AssertionError),
        ):
            await response._react(response._response)
        start.assert_awaited_once()

    async def test_durable_staging_preserves_cleanup_failure(self) -> None:
        response = _response()
        primary = RuntimeError("staging failed")
        cleanup = ValueError("provider cleanup failed")

        async def fail_staging(*_args: object, **_kwargs: object) -> object:
            raise primary

        runtime = DurableInteractionRuntime(
            actor=InteractionActor(
                principal=PrincipalScope(user_id=UserId("coverage-user"))
            ),
            stager=fail_staging,
        )
        execution = SimpleNamespace(
            interaction_runtime=runtime,
            origin=object(),
            begin_interaction=AsyncMock(),
            abandon_interaction=AsyncMock(),
            status=AgentExecutionStatus.RUNNING,
        )
        response._execution = cast(AgentExecution, execution)

        with (
            patch.object(
                orchestrator_response,
                "InteractionBrokerRequest",
                return_value=object(),
            ),
            patch.object(
                response,
                "_durable_staging_context",
                return_value=object(),
            ),
            patch.object(
                response._response,
                "aclose",
                AsyncMock(side_effect=cleanup),
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await response._start_task_input(_task_input_call())

        self.assertIs(raised.exception, primary)
        self.assertEqual(
            getattr(primary, "__notes__", ()),
            [
                "post-provider cleanup failure: "
                "ValueError: provider cleanup failed"
            ],
        )

    async def test_cancel_without_runtime_clears_completed_pending_task(
        self,
    ) -> None:
        response = _response()

        async def complete() -> None:
            return None

        task = create_task(complete())
        await task
        response._pending_interaction_task = cast(Task[Any], task)
        response._pending_interaction_call = _task_input_call()
        response._pending_interaction_assistant_text = "preface"
        response._pending_interaction_published = True
        response._execution = None

        with patch.object(
            response,
            "_cancel_task_with_deadline",
            AsyncMock(),
        ):
            await response._cancel_pending_interaction()

        self.assertIsNone(response._pending_interaction_task)
        self.assertIsNone(response._pending_interaction_call)
        self.assertEqual(response._pending_interaction_assistant_text, "")
        self.assertFalse(response._pending_interaction_published)

    async def test_cancel_with_runtime_preserves_task_cleanup_failure(
        self,
    ) -> None:
        response = _response()

        async def complete() -> None:
            return None

        task = create_task(complete())
        await task
        response._pending_interaction_task = cast(Task[Any], task)
        response._execution = cast(
            AgentExecution,
            SimpleNamespace(
                interaction_runtime=object(),
                pending_request=None,
            ),
        )
        response._active_interaction_request = None
        response._interaction_cleanup_complete = True
        cleanup = RuntimeError("pending interaction cleanup failed")

        with (
            patch.object(
                response,
                "_settle_execution_with_deadline",
                AsyncMock(return_value=()),
            ),
            patch.object(
                response,
                "_execution_terminal_kind",
                return_value=StreamItemKind.STREAM_CANCELLED,
            ),
            patch.object(response, "_finish_canonical_stream"),
            patch.object(
                response,
                "_finalize_interaction_cleanup",
                AsyncMock(),
            ),
            patch.object(
                response,
                "_cancel_task_with_deadline",
                AsyncMock(side_effect=cleanup),
            ),
            self.assertRaises(RuntimeError) as raised,
        ):
            await response._cancel_pending_interaction()

        self.assertIs(raised.exception, cleanup)
