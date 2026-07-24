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
)
from avalan.event import EventType
from avalan.model.call import ModelCallContext
from avalan.model.response.text import TextGenerationResponse
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamProviderEvent,
)
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


def _canonical_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    data: object | None = None,
) -> CanonicalStreamItem:
    channel = (
        StreamChannel.ANSWER
        if kind in {StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DONE}
        else StreamChannel.CONTROL
    )
    return CanonicalStreamItem(
        stream_session_id="more-stream",
        run_id="more-run",
        turn_id="more-turn",
        sequence=sequence,
        kind=kind,
        channel=channel,
        text_delta=text_delta,
        data=data,
    )


def _make_response(
    input_value: Input,
    response: TextGenerationResponse,
    agent: EngineAgent,
    operation: AgentOperation,
    engine_args: dict,
    **kwargs,
) -> OrchestratorResponse:
    kwargs.setdefault(
        "enable_tool_parsing", kwargs.get("capability") is not None
    )
    context = ModelCallContext(
        specification=operation.specification,
        input=input_value,
        capability=kwargs.get("capability"),
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
        queued = _canonical_item(
            StreamItemKind.ANSWER_DELTA,
            0,
            text_delta="queued",
        )
        resp._parser_queue.put(queued)
        self.assertIs(
            (await resp.__anext__()).kind, StreamItemKind.STREAM_STARTED
        )
        item = await resp.__anext__()
        self.assertIs(item.kind, StreamItemKind.ANSWER_DELTA)
        self.assertEqual(item.text_delta, "queued")

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
        flushed = [
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                0,
                text_delta="x",
            )
        ]
        resp._tool_parser.flush = AsyncMock(side_effect=[flushed, []])
        resp.__aiter__()
        items = []
        while True:
            try:
                items.append(await resp.__anext__())
            except StopAsyncIteration:
                break

        self.assertEqual(
            [item.text_delta for item in items if item.text_delta],
            ["x"],
        )

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
        diagnostic = StreamProviderEvent(
            kind=StreamItemKind.STREAM_DIAGNOSTIC,
            data={"event_type": EventType.TOOL_DIAGNOSTIC.value},
        )
        item = _canonical_item(
            StreamItemKind.ANSWER_DELTA,
            0,
            text_delta="x",
        )
        resp._tool_parser.flush = AsyncMock(
            side_effect=[[diagnostic, item], []]
        )
        resp.__aiter__()

        items = []
        while True:
            try:
                items.append(await resp.__anext__())
            except StopAsyncIteration:
                break

        self.assertIn(
            StreamItemKind.STREAM_DIAGNOSTIC,
            [item.kind for item in items],
        )
        self.assertIn("x", [item.text_delta for item in items])

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
        resp._tool_parser = MagicMock()
        resp._tool_parser.push = AsyncMock(
            return_value=[
                _canonical_item(
                    StreamItemKind.STREAM_DIAGNOSTIC,
                    0,
                    data={"code": "parser.push"},
                )
            ]
        )
        await resp._process_canonical_response_item(
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                0,
                text_delta="text",
            )
        )
        self.assertIn(
            StreamItemKind.STREAM_DIAGNOSTIC,
            [item.kind for item in resp.canonical_items],
        )

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
            resp._tool_result_outcomes.maxsize,
            OrchestratorResponse._MAXIMUM_STAGING_QUEUE_ITEMS,
        )
        self.assertEqual(resp._canonical_tool_call_lifecycles, {})

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
