from asyncio import wait_for
from collections.abc import AsyncIterator
from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Message,
    MessageRole,
    ReasoningSettings,
    ReasoningToken,
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolCallToken,
    ToolFormat,
    Token,
    TokenDetail,
    TransformerEngineSettings,
)
from avalan.event import EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser
from dataclasses import dataclass
from io import StringIO
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4


class _DummyEngine:
    def __init__(self):
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
        settings=TransformerEngineSettings(),
    )
    spec = Specification(role="assistant", goal=None)
    return AgentOperation(specification=spec, environment=env)


def _dummy_response(async_gen=True):
    async def output_gen():
        yield "a"
        yield Token(id=5, token="b")

    def output_fn(**_):
        return output_gen()

    settings = GenerationSettings()
    return TextGenerationResponse(
        output_fn,
        logger=getLogger(),
        use_async_generator=async_gen,
        generation_settings=settings,
        settings=settings,
    )


def _complex_response():
    async def gen():
        rp = ReasoningParser(
            reasoning_settings=ReasoningSettings(), logger=getLogger()
        )
        tm = MagicMock()
        tm.is_potential_tool_call.return_value = True
        tm.get_calls.return_value = None
        base_parser = ToolCallParser()
        tm.tool_call_status.side_effect = base_parser.tool_call_status
        tp = ToolCallResponseParser(tm, None)

        sequence = [
            "X",
            "<think>",
            "ra",
            "rb",
            "</think>",
            "Y",
            "<tool_call>",
            "foo",
            "bar",
            "</tool_call>",
            "Z",
        ]

        for s in sequence:
            items = await rp.push(s)
            for item in items:
                parsed = (
                    await tp.push(item) if isinstance(item, str) else [item]
                )
                for p in parsed:
                    if isinstance(p, str):
                        if p == "</think>":
                            yield TokenDetail(id=3, token=p, probability=0.5)
                        elif p in {"X", "Y"}:
                            yield Token(id=1, token=p)
                        elif p == "<think>" or p == "Z":
                            yield p
                    elif isinstance(p, ToolCallToken):
                        if p.token == "</tool_call>":
                            yield TokenDetail(
                                id=4, token=p.token, probability=0.5
                            )
                        else:
                            yield p
                    else:
                        yield p

    settings = GenerationSettings()
    return TextGenerationResponse(
        lambda **_: gen(),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=settings,
        settings=settings,
    )


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
            {
                "token_id": 42,
                "token_type": "str",
                "model_id": "m",
                "token": "a",
                "step": 0,
            },
        )
        self.assertEqual(
            token_events[1].payload,
            {
                "token_id": 5,
                "token_type": "Token",
                "model_id": "m",
                "token": "b",
                "step": 1,
            },
        )

    async def test_harmony_streaming_handles_split_prefix(self) -> None:
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [1]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def gen() -> AsyncIterator[str]:
            yield "<|start|>assistant"
            yield "<|channel|>commentary to=mytool <|message|>{}<|call|>"

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        tool_manager = MagicMock(spec=ToolManager)
        tool_manager.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        tool_manager.tool_call_status.side_effect = (
            base_parser.tool_call_status
        )
        tool_manager.get_calls.side_effect = base_parser
        tool_manager.is_empty = False

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool_manager,
        )

        iterator = resp.__aiter__()
        first = await wait_for(iterator.__anext__(), 1)
        second = await wait_for(iterator.__anext__(), 1)
        self.assertIsInstance(first, ToolCallToken)
        self.assertIsInstance(second, ToolCallToken)


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
        logger=getLogger(),
        use_async_generator=async_gen,
        inputs=inputs or {"input_ids": [[1, 2, 3]]},
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
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
        self.assertEqual(resp.output_token_count, 0)
        self.assertTrue(resp.can_think)
        self.assertFalse(resp.is_thinking)
        resp.set_thinking(True)
        self.assertTrue(resp.is_thinking)
        self.assertEqual(await resp.to_str(), '{"value": "ok"}')
        self.assertEqual(resp.output_token_count, len('{"value": "ok"}'))
        self.assertEqual(await resp.to_json(), '{"value": "ok"}')
        result = await resp.to(Example)
        self.assertEqual(result, Example(value="ok"))
        resp.set_thinking(False)
        self.assertFalse(resp.is_thinking)


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


class OrchestratorResponseNoToolTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_without_tool(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen():
            yield "h"
            yield "i"

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
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


class OrchestratorResponseToStrTestCase(IsolatedAsyncioTestCase):
    async def test_to_str_without_tool_call(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen():
            yield "o"
            yield "k"

        settings = GenerationSettings()
        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        TextGenerationResponse._buffer = StringIO()

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        result = await resp.to_str()
        self.assertEqual(result, "ok")

    async def test_to_str_with_tool_call(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def outer_gen():
            yield "c"
            yield "a"
            yield "l"
            yield "l"

        settings = GenerationSettings()
        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [ToolCall(id=uuid4(), name="calc", arguments=None)]
            if text == "call"
            else None
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="2",
            )

        tool.side_effect = tool_exec

        async def inner_gen():
            yield "r"

        inner_settings = GenerationSettings()
        inner_response = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=inner_settings,
            settings=inner_settings,
        )
        agent.return_value = inner_response

        TextGenerationResponse._buffer = StringIO()

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        result = await resp.to_str()
        self.assertEqual(result, "r")
        agent.assert_awaited_once()
        tool.assert_awaited_once()


class OrchestratorResponseContextTestCase(IsolatedAsyncioTestCase):
    async def test_tool_context_ids(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        tool = MagicMock(spec=ToolManager)
        tool.is_empty = True

        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        aid = uuid4()
        pid = uuid4()
        sid = uuid4()

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            tool=tool,
            event_manager=event_manager,
            agent_id=aid,
            participant_id=pid,
            session_id=sid,
        )
        resp.__aiter__()

        self.assertEqual(resp._tool_context.agent_id, aid)
        self.assertEqual(resp._tool_context.participant_id, pid)
        self.assertEqual(resp._tool_context.session_id, sid)
        self.assertEqual(resp._tool_context.calls, [])


class OrchestratorResponseParsedTokensTestCase(IsolatedAsyncioTestCase):
    async def test_mixed_tokens(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        resp = OrchestratorResponse(
            Message(role=MessageRole.USER, content="hi"),
            _complex_response(),
            agent,
            operation,
            {},
        )

        tokens = []
        async for t in resp:
            tokens.append(t)

        self.assertEqual(
            len([t for t in tokens if isinstance(t, ReasoningToken)]),
            4,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, ToolCallToken)]),
            3,
        )
        self.assertEqual(
            len([t for t in tokens if isinstance(t, TokenDetail)]),
            1,
        )
        self.assertGreaterEqual(
            len([t for t in tokens if type(t) is Token]),
            2,
        )
        self.assertEqual(len([t for t in tokens if isinstance(t, str)]), 1)
