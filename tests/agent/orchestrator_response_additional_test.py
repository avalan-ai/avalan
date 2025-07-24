from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.agent import EngineEnvironment, EngineOperation, Specification
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    ToolCall,
    ToolCallResult,
    Token,
    TransformerEngineSettings,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from avalan.tool.manager import ToolManager
from avalan.cli import CommandAbortException
from avalan.model import TextGenerationResponse
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


class _DummyEngine:
    def __init__(self) -> None:
        self.model_id = "m"
        self.tokenizer = MagicMock()


def _dummy_operation() -> EngineOperation:
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
        settings=TransformerEngineSettings(),
    )
    spec = Specification(role="assistant", goal=None)
    return EngineOperation(specification=spec, environment=env)


def _dummy_response(async_gen: bool = True) -> TextGenerationResponse:
    async def output_gen():
        yield "a"
        yield Token(id=5, token="b")

    def output_fn():
        return output_gen()

    return TextGenerationResponse(output_fn, use_async_generator=async_gen)


class OrchestratorResponseAdditionalCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_tool_process_queue(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
        )
        resp.__aiter__()
        event = Event(type=EventType.TOOL_PROCESS, payload=None)
        resp._tool_process_events.put(event)
        result = await resp.__anext__()
        self.assertEqual(result, event)
        self.assertEqual(resp._tool_call_events.get_nowait(), event)

    async def test_tool_call_confirm_all(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def tool_exec(call, context):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="ok",
            )

        tool.side_effect = tool_exec

        call = ToolCall(id=uuid4(), name="calc", arguments=None)
        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=lambda c: "a",
        )
        resp.__aiter__()
        resp._tool_call_events.put(
            Event(type=EventType.TOOL_PROCESS, payload=[call])
        )
        result = await resp.__anext__()
        self.assertEqual(result.type, EventType.TOOL_RESULT)
        self.assertTrue(resp._tool_confirm_all)
        tool.assert_awaited_once()
        self.assertGreater(event_manager.trigger.await_count, 0)

    async def test_tool_call_async_confirm(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def exec_tool(call, context):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="ok",
            )

        tool.side_effect = exec_tool

        async def confirm(call):
            return "y"

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            tool=tool,
            tool_confirm=confirm,
        )
        resp.__aiter__()
        resp._calls.put(ToolCall(id=uuid4(), name="t", arguments=None))
        result = await resp.__anext__()
        self.assertEqual(result.type, EventType.TOOL_RESULT)

    async def test_tool_confirm_abort(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            tool=tool,
            tool_confirm=lambda c: "n",
        )
        resp.__aiter__()
        resp._calls.put(ToolCall(id=uuid4(), name="calc", arguments=None))
        with self.assertRaises(CommandAbortException):
            await resp.__anext__()

    async def test_result_processing(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        inner_response = _dummy_response()
        agent.return_value = inner_response
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        resp.__aiter__()
        result = ToolCallResult(
            id=uuid4(),
            call=ToolCall(id=uuid4(), name="tool", arguments=None),
            name="tool",
            arguments=None,
            result="1",
        )
        resp._tool_result_events.put(
            Event(type=EventType.TOOL_RESULT, payload={"result": result})
        )
        event = await resp.__anext__()
        self.assertEqual(event.type, EventType.TOOL_MODEL_RESPONSE)
        agent.assert_awaited_once()

    async def test_emit_token_and_process(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        agent.engine.tokenizer.encode.return_value = [5]
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        resp.__aiter__()
        await resp._emit(Token(id=5, token="x"))
        event_manager.trigger.assert_awaited()
        process_event = Event(type=EventType.TOOL_PROCESS, payload=None)
        returned = await resp._emit(process_event)
        self.assertEqual(returned, process_event)
        other = Event(type=EventType.END)
        self.assertIs(await resp._emit(other), other)
