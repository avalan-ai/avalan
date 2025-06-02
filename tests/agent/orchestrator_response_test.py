from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.agent import Operation, Specification, EngineEnvironment
from avalan.entities import (
    EngineUri,
    Message,
    MessageRole,
    TransformerEngineSettings,
    Token,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.agent.engine import EngineAgent
from avalan.model import TextGenerationResponse

from unittest import IsolatedAsyncioTestCase
from dataclasses import dataclass
from avalan.tool.manager import ToolManager
from avalan.entities import ToolCall, ToolCallResult
from unittest.mock import AsyncMock, MagicMock


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


def _dummy_response(async_gen=True):
    async def output_gen():
        yield "a"
        yield Token(id=5, token="b")

    def output_fn():
        return output_gen()

    return TextGenerationResponse(output_fn, use_async_generator=async_gen)


class OrchestratorResponseIterationTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_emits_events_and_end(self):
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [42]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
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

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(tokens, ["a", Token(id=5, token="b")])
        calls = event_manager.trigger.await_args_list
        self.assertTrue(any(c.args[0].type == EventType.END for c in calls))
        self.assertTrue(
            any(c.args[0].type == EventType.STREAM_END for c in calls)
        )
        token_events = [
            c.args[0]
            for c in calls
            if c.args[0].type == EventType.TOKEN_GENERATED
        ]
        self.assertEqual(len(token_events), 2)
        self.assertEqual(
            token_events[0].payload,
            {"token_id": 42, "model_id": "m", "token": "a", "step": 0},
        )
        self.assertEqual(
            token_events[1].payload,
            {"token_id": 5, "model_id": "m", "token": "b", "step": 1},
        )


@dataclass
class Example:
    value: str


def _string_response(text: str, *, async_gen: bool = False, inputs=None):
    def output_fn(*args, **kwargs):
        if async_gen:

            async def gen():
                for ch in text:
                    yield ch

            return gen()
        return text

    return TextGenerationResponse(
        output_fn,
        use_async_generator=async_gen,
        inputs=inputs or {"input_ids": [[1, 2, 3]]},
    )


class OrchestratorResponseMethodsTestCase(IsolatedAsyncioTestCase):
    async def test_counts_and_conversions(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _string_response('{"value": "ok"}', async_gen=False),
            agent,
            operation,
            {},
        )

        self.assertEqual(resp.input_token_count, 3)
        self.assertEqual(await resp.to_str(), '{"value": "ok"}')
        self.assertEqual(await resp.to_json(), '{"value": "ok"}')
        result = await resp.to(Example)
        self.assertEqual(result, Example(value="ok"))


class OrchestratorResponseEventTestCase(IsolatedAsyncioTestCase):
    async def test_event_manager_callback(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        await resp.to_str()
        event_manager.trigger.assert_awaited_once()
        self.assertEqual(
            event_manager.trigger.await_args.args[0].type, EventType.STREAM_END
        )


class OrchestratorResponseToolCallTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_with_tool_call(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def outer_gen():
            for ch in "call":
                yield ch

        outer_response = TextGenerationResponse(
            lambda: outer_gen(), use_async_generator=True
        )

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = (
            lambda text: [ToolCall(name="calc", arguments=None)]
            if text == "call"
            else None
        )

        async def tool_exec(call):
            return ToolCallResult(
                call=call, name=call.name, arguments=call.arguments, result="2"
            )

        tool.side_effect = tool_exec

        async def inner_gen():
            yield "r"

        inner_response = TextGenerationResponse(
            lambda: inner_gen(), use_async_generator=True
        )
        agent.return_value = inner_response

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        items = []
        async for item in resp:
            items.append(item)

        types = [getattr(i, "type", None) for i in items]
        self.assertIn(EventType.TOOL_PROCESS, types)
        self.assertIn(EventType.TOOL_RESULT, types)
        self.assertEqual(items[0], "c")
        self.assertEqual(items[1], "a")
        self.assertEqual(items[2], "l")
        self.assertEqual(items[3], "l")
        self.assertEqual(items[-1], "r")

        agent.assert_awaited_once()
        spec_arg, messages = agent.await_args.args
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].content, "hi")
        self.assertEqual(messages[1].role, MessageRole.TOOL)
        self.assertEqual(messages[1].name, "calc")
        self.assertEqual(messages[1].content, "2")

        triggered = [
            c.args[0].type for c in event_manager.trigger.await_args_list
        ]
        self.assertIn(EventType.TOOL_PROCESS, triggered)
        self.assertIn(EventType.TOOL_EXECUTE, triggered)
        self.assertIn(EventType.TOOL_RESULT, triggered)


class OrchestratorResponseNoToolTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_without_tool(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen():
            yield "h"
            yield "i"

        response = TextGenerationResponse(
            lambda: gen(), use_async_generator=True
        )

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(tokens, ["h", "i"])
