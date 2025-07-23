from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.agent import EngineEnvironment, Operation, Specification
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    Token,
)
from avalan.event import Event, EventType
from avalan.tool.manager import ToolManager
from avalan.model.response.text import TextGenerationResponse
from avalan.agent.engine import EngineAgent
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock


class _DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.tokenizer = MagicMock()


def _dummy_operation() -> Operation:
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
    return Operation(specification=spec, environment=env)


def _empty_response() -> TextGenerationResponse:
    async def gen():
        if False:
            yield ""  # pragma: no cover - never executed

    return TextGenerationResponse(lambda: gen(), use_async_generator=True)


class OrchestratorResponseMoreCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_parser_queue_precedence(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = OrchestratorResponse(
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
        resp = OrchestratorResponse(
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

    async def test_emit_parsed_event(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = MagicMock(spec=ToolManager)
        tool.is_empty = False
        resp = OrchestratorResponse(
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
