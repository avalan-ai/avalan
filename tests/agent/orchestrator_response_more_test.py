from logging import getLogger
from queue import Queue
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    Input,
    Message,
    MessageRole,
    Token,
)
from avalan.event import Event, EventType
from avalan.model.call import ModelCallContext
from avalan.model.response.text import TextGenerationResponse
from avalan.tool.manager import ToolManager


class _DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.tokenizer = MagicMock()


def _dummy_operation() -> AgentOperation:
    env = EngineEnvironment(
        engine_uri=EngineUri(
            host=None,
            port=None,
            user=None,
            password=None,
            vendor=None,
            model_id="m",
            params={},
        ),
        settings=None,
    )
    spec = Specification(role="assistant", goal=None)
    return AgentOperation(specification=spec, environment=env)


def _empty_response() -> TextGenerationResponse:
    async def gen():
        if False:
            yield ""  # pragma: no cover - never executed

    return TextGenerationResponse(
        lambda: gen(), logger=getLogger(), use_async_generator=True
    )


def _make_response(
    input_value: Input,
    response: TextGenerationResponse,
    agent: EngineAgent,
    operation: AgentOperation,
    engine_args: dict,
    **kwargs,
) -> OrchestratorResponse:
    context = ModelCallContext(
        specification=operation.specification,
        input=input_value,
        engine_args=dict(engine_args),
    )
    return OrchestratorResponse(
        input_value,
        response,
        agent,
        operation,
        engine_args,
        context,
        **kwargs,
    )


class OrchestratorResponseMoreCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_parser_queue_precedence(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
        )
        resp.__aiter__()
        resp._parser_queue.put("queued")
        self.assertEqual(await resp.__anext__(), "queued")

    async def test_flush_after_stop(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = MagicMock(spec=ToolManager)
        tool.is_empty = False
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
            tool=tool,
        )
        resp._tool_parser = MagicMock()
        flushed = [Event(type=EventType.END), Token(id=1, token="x")]
        resp._tool_parser.flush = AsyncMock(return_value=flushed)
        resp.__aiter__()
        self.assertEqual(await resp.__anext__(), flushed[1])
        self.assertEqual(resp._tool_process_events.get_nowait(), flushed[0])

    async def test_flush_queues_diagnostic_after_plain_tokens(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = MagicMock(spec=ToolManager)
        tool.is_empty = False
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
            tool=tool,
        )
        resp._tool_parser = MagicMock()
        diagnostic = Event(type=EventType.TOOL_DIAGNOSTIC)
        token = Token(id=1, token="x")
        resp._tool_parser.flush = AsyncMock(return_value=[diagnostic, token])
        resp.__aiter__()

        self.assertEqual(await resp.__anext__(), token)
        self.assertEqual(await resp.__anext__(), diagnostic)

    async def test_emit_parsed_event(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = MagicMock(spec=ToolManager)
        tool.is_empty = False
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
            tool=tool,
        )
        resp.__aiter__()
        event = Event(type=EventType.END)
        resp._tool_parser = MagicMock()
        resp._tool_parser.push = AsyncMock(return_value=[event])
        self.assertIs(await resp._emit("text"), event)

    async def test_tool_parser_disabled(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = MagicMock(spec=ToolManager)
        tool.is_empty = False
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )
        self.assertIsNone(resp._tool_parser)

    async def test_iteration_uses_bounded_owned_staging_queues(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
        )

        resp.__aiter__()

        self.assertEqual(
            resp._parser_queue.maxsize,
            OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
        )
        self.assertEqual(
            resp._calls.maxsize,
            OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
        )
        self.assertEqual(
            resp._tool_call_events.maxsize,
            OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
        )
        self.assertEqual(
            resp._tool_process_events.maxsize,
            OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
        )
        self.assertEqual(
            resp._tool_result_events.maxsize,
            OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
        )

    async def test_staging_queue_overflow_fails_without_blocking(self):
        queue: Queue[object] = Queue(maxsize=1)
        first = object()
        second = object()

        OrchestratorResponse._put_staging_item(queue, first, "parser item")

        with self.assertRaisesRegex(
            RuntimeError,
            "Orchestrator parser item queue is full.",
        ):
            OrchestratorResponse._put_staging_item(
                queue,
                second,
                "parser item",
            )

        self.assertIs(queue.get_nowait(), first)
