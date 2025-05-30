import unittest
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock

from avalan.agent import Operation, Specification, EngineEnvironment
from avalan.agent.orchestrator.response import (
    OrchestratorResponse,
    ToolAwareResponse,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.entities import (
    EngineUri,
    ToolCall,
    ToolCallResult,
    TransformerEngineSettings,
)
from avalan.agent.engine import EngineAgent
from avalan.tool.manager import ToolManager


class _DummyEngine:
    def __init__(self):
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
        settings=TransformerEngineSettings(),
    )
    spec = Specification(role="assistant", goal=None)
    return Operation(specification=spec, environment=env)


def _dummy_response(text: str = "ok") -> TextGenerationResponse:
    return TextGenerationResponse(lambda: text, use_async_generator=False)


class OrchestratorResponseInitTestCase(unittest.TestCase):
    def setUp(self):
        self.engine = _DummyEngine()
        self.agent = MagicMock(spec=EngineAgent)
        self.agent.engine = self.engine
        self.operation = _dummy_operation()

    def test_init_without_event_and_tool(self):
        resp = _dummy_response()
        orch = OrchestratorResponse(resp, self.agent, self.operation, {"a": 1})
        self.assertIs(orch._engine_agent, self.agent)
        self.assertIs(orch._operation, self.operation)
        self.assertEqual(orch._engine_args, {"a": 1})
        self.assertIsNone(orch._event_manager)
        self.assertIsNone(orch._tool)
        self.assertEqual(len(orch._responses), 1)
        self.assertIsInstance(orch._responses[0], ToolAwareResponse)
        self.assertEqual(orch._index, 0)
        self.assertFalse(orch._finished)

    def test_init_with_event_and_tool(self):
        resp = _dummy_response()
        event_manager = MagicMock(spec=EventManager)
        tool = MagicMock(spec=ToolManager)
        orch = OrchestratorResponse(
            resp,
            self.agent,
            self.operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        self.assertIs(orch._event_manager, event_manager)
        self.assertIs(orch._tool, tool)
        self.assertIsInstance(orch._responses[0], ToolAwareResponse)


class OrchestratorWrapResponseTestCase(unittest.TestCase):
    def test_wrap_response(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _dummy_response()
        orch = OrchestratorResponse(resp, agent, operation, {})
        new_resp = _dummy_response("x")
        wrapped = orch._wrap_response(new_resp)
        self.assertIsInstance(wrapped, ToolAwareResponse)
        self.assertIs(wrapped._event_manager, orch._event_manager)
        self.assertEqual(wrapped._model_id, engine.model_id)
        self.assertIs(wrapped._tokenizer, engine.tokenizer)


class OrchestratorOnTokenTestCase(IsolatedAsyncioTestCase):
    async def test_on_token_with_tool(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        tool = MagicMock(spec=ToolManager)
        tool.has_tool_call.return_value = True
        call = ToolCall(name="calc")
        result = ToolCallResult(call=call, name="calc", result="2")
        tool.return_value = ([call], [result])
        agent.return_value = _dummy_response("next")
        orch = OrchestratorResponse(
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        await orch._on_token("<tool_call>{}</tool_call>")
        called_types = [
            c.args[0].type for c in event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.TOOL_PROCESS, called_types)
        self.assertIn(EventType.TOOL_EXECUTE, called_types)
        self.assertIn(EventType.TOOL_RESULT, called_types)
        agent.assert_awaited_once()
        self.assertEqual(len(orch._responses), 2)


class OrchestratorResponseIterationTestCase(IsolatedAsyncioTestCase):
    async def test_async_iteration(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        resp = _dummy_response()
        orch = OrchestratorResponse(
            resp,
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        self.assertIs(orch.__aiter__(), orch)
        first = await orch.__anext__()
        self.assertIs(first, orch._responses[0])
        with self.assertRaises(StopAsyncIteration):
            await orch.__anext__()
        self.assertTrue(
            any(
                c.args[0].type == EventType.END
                for c in event_manager.trigger.await_args_list
            )
        )


if __name__ == "__main__":
    unittest.main()
