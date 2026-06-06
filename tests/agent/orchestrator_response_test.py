from asyncio import wait_for
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import StringIO
from json import loads
from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.entities import (
    EngineUri,
    GenerationSettings,
    Input,
    Message,
    MessageRole,
    MessageToolCall,
    ReasoningSettings,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallResult,
    ToolCallToken,
    ToolFormat,
    TransformerEngineSettings,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.tool import ToolSet
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser


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


def _make_response(
    input_value: Input,
    response: TextGenerationResponse,
    agent: EngineAgent,
    operation: AgentOperation,
    engine_args: dict,
    **kwargs,
) -> OrchestratorResponse:
    agent_id = kwargs.get("agent_id")
    participant_id = kwargs.get("participant_id")
    session_id = kwargs.get("session_id")
    context = ModelCallContext(
        specification=operation.specification,
        input=input_value,
        engine_args=dict(engine_args),
        agent_id=agent_id,
        participant_id=participant_id,
        session_id=session_id,
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

        resp = _make_response(
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

        resp = _make_response(
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

    async def test_harmony_streaming_emits_flush_tool_event(self) -> None:
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [1]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def gen() -> AsyncIterator[str]:
            yield (
                "<|start|>assistant<|channel|>commentary "
                "to=functions.browser.open <|constrain|>json<|message|>"
                '{"url":"https://example.com"}'
            )

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
        tool_manager.tool_format = ToolFormat.HARMONY
        tool_manager.is_empty = False

        resp = _make_response(
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
        self.assertEqual(second.type, EventType.TOOL_PROCESS)
        call = second.payload[0]
        self.assertEqual(call.name, "browser.open")
        self.assertEqual(call.arguments, {"url": "https://example.com"})


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

        resp = _make_response(
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

        resp = _make_response(
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

        resp = _make_response(
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

        resp = _make_response(
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

        resp = _make_response(
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

    async def test_to_str_with_structured_tool_call_token(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "2 + 2"},
        )

        async def outer_gen():
            yield ToolCallToken(token="2 + 2", call=None)
            yield ToolCallToken(token="", call=call)

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
        tool.get_calls.return_value = None

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="4",
            )

        tool.side_effect = tool_exec

        async def inner_gen():
            yield "4"

        inner_response = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        agent.return_value = inner_response

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        result = await resp.to_str()

        self.assertEqual(result, "4")
        tool.assert_awaited_once()
        tool.get_calls.assert_called_once_with("4")
        context = agent.await_args.args[0]
        assert isinstance(context.input, list)
        self.assertEqual(context.input[-2].tool_calls[0].id, "call1")
        self.assertEqual(context.input[-1].content, "4")

    async def test_to_str_returns_anchored_diagnostic_to_model(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def known(value: str) -> str:
            """Return the provided value.

            Args:
                value: Value to return.

            Returns:
                Provided value.
            """
            return value

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[known])]
        )
        call = ToolCall(
            id="call1",
            name="missing",
            arguments={"value": "x"},
        )

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        agent.return_value = _string_response("recovered", async_gen=False)

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        result = await resp.to_str()

        self.assertEqual(result, "recovered")
        context = agent.await_args.args[0]
        assert isinstance(context.input, list)
        assistant_message = context.input[-2]
        diagnostic_message = context.input[-1]
        self.assertEqual(assistant_message.tool_calls[0].id, "call1")
        self.assertEqual(assistant_message.tool_calls[0].name, "missing")
        self.assertEqual(diagnostic_message.role, MessageRole.TOOL)
        self.assertIsNone(diagnostic_message.tool_call_result)
        self.assertIsNone(diagnostic_message.tool_call_error)
        self.assertIsNotNone(diagnostic_message.tool_call_diagnostic)
        diagnostic = diagnostic_message.tool_call_diagnostic
        assert diagnostic is not None
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.UNKNOWN_TOOL,
        )
        self.assertEqual(diagnostic.call_id, "call1")
        payload = loads(str(diagnostic_message.content))
        self.assertEqual(payload["code"], "tool.unknown")
        self.assertEqual(payload["requested_name"], "missing")

    async def test_to_str_stops_after_consecutive_non_executed_cycles(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def known(value: str) -> str:
            """Return the provided value.

            Args:
                value: Value to return.

            Returns:
                Provided value.
            """
            return value

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[known])]
        )
        call = ToolCall(
            id="call1",
            name="missing",
            arguments={"value": "x"},
        )

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        async def inner_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        inner_response = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        agent.return_value = inner_response

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        resp._MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES = 1

        result = await resp.to_str()

        self.assertEqual(result, "")
        agent.assert_awaited_once()
        self.assertEqual(resp._call_history, [])

    async def test_to_str_does_not_rerun_model_without_tool_observation(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={})
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [call] if text == "call" else None
        )
        tool.return_value = None

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("call", async_gen=False),
            agent,
            operation,
            {},
            tool=tool,
        )

        result = await resp.to_str()

        self.assertEqual(result, "call")
        agent.assert_not_awaited()
        tool.assert_awaited_once()
        self.assertEqual(resp._call_history, [])

    async def test_repeated_tool_attempt_returns_guard_diagnostic(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"value": 1})
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="ok",
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("call", async_gen=False),
            agent,
            operation,
            {},
            tool=tool,
        )
        context = ToolCallContext(input=resp._input, calls=[])

        first = await resp._execute_tool_call(
            call,
            context,
            confirm=False,
        )
        second = await resp._execute_tool_call(
            call,
            context,
            confirm=False,
        )

        self.assertIsInstance(first, ToolCallResult)
        self.assertIsInstance(second, ToolCallDiagnostic)
        assert isinstance(second, ToolCallDiagnostic)
        self.assertEqual(second.code, ToolCallDiagnosticCode.REPEATED_CALL)
        self.assertEqual(second.stage, ToolCallDiagnosticStage.GUARD)
        tool.assert_awaited_once()

    async def test_iteration_stops_without_tool_observation(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        resp.__aiter__()
        resp._tool_result_events.put(
            Event(type=EventType.TOOL_RESULT, payload={"result": None})
        )

        with self.assertRaises(StopAsyncIteration):
            await resp.__anext__()

        agent.assert_not_awaited()
        self.assertTrue(resp._finished)
        self.assertEqual(
            event_manager.trigger.await_args.args[0].type,
            EventType.END,
        )

    async def test_tool_cycle_guard_rejects_duplicate_and_maximum_cycles(
        self,
    ):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"value": 1})
        result = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="ok",
        )
        messages = OrchestratorResponse._tool_observation_messages(
            result,
            json_output=False,
        )
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("call", async_gen=False),
            agent,
            operation,
            {},
        )

        self.assertTrue(resp._should_continue_tool_cycle(messages, [result]))
        self.assertFalse(resp._should_continue_tool_cycle(messages, [result]))

        limited = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("call", async_gen=False),
            agent,
            operation,
            {},
        )
        limited._MAXIMUM_TOOL_CYCLES = 0
        self.assertFalse(
            limited._should_continue_tool_cycle(messages, [result])
        )

    async def test_null_tool_result_projects_empty_observation(self):
        call = ToolCall(id="call1", name="calc", arguments={})
        result = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=None,
        )

        messages = OrchestratorResponse._tool_observation_messages(
            result,
            json_output=False,
        )

        self.assertEqual(messages[-1].content, "")

    async def test_unanchored_diagnostic_uses_recovery_message(self):
        diagnostic = ToolCallDiagnostic(
            id="diag1",
            code=ToolCallDiagnosticCode.MALFORMED_CALL,
            stage=ToolCallDiagnosticStage.PARSE,
            message="Could not parse tool call.",
        )

        messages = OrchestratorResponse._tool_observation_messages(
            diagnostic,
            json_output=False,
        )

        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0].role, MessageRole.ASSISTANT)
        self.assertIs(messages[0].tool_call_diagnostic, diagnostic)
        self.assertIsNone(messages[0].tool_calls)
        payload = loads(str(messages[0].content))
        self.assertEqual(payload["code"], "tool_call.malformed")

    async def test_iteration_carries_tool_messages_to_nested_tool_call(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        first_call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "25 * 2"},
        )
        second_call = ToolCall(
            id="call2",
            name="calc",
            arguments={"expression": "50 * 2"},
        )

        async def outer_gen():
            yield ToolCallToken(token="<tool_call />", call=first_call)

        async def first_inner_gen():
            yield ToolCallToken(token="<tool_call />", call=second_call)

        async def second_inner_gen():
            yield "done"

        settings = GenerationSettings()
        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        first_inner_response = TextGenerationResponse(
            lambda **_: first_inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        second_inner_response = TextGenerationResponse(
            lambda **_: second_inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        agent.side_effect = [first_inner_response, second_inner_response]

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def tool_exec(call, context: ToolCallContext):
            result = "50" if call.id == "call1" else "100"
            return ToolCallResult(
                id=f"{call.id}-result",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=result,
            )

        tool.side_effect = tool_exec

        resp = _make_response(
            [
                Message(role=MessageRole.USER, content="previous"),
                Message(role=MessageRole.ASSISTANT, content="25"),
                Message(role=MessageRole.USER, content="and that times two?"),
            ],
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        tokens = []
        async for token in resp:
            tokens.append(token)

        self.assertIn("done", tokens)
        self.assertEqual(agent.await_count, 2)
        self.assertEqual(tool.await_count, 2)

        first_context = agent.await_args_list[0].args[0]
        second_context = agent.await_args_list[1].args[0]
        assert isinstance(first_context.input, list)
        assert isinstance(second_context.input, list)
        self.assertEqual(
            [message.role for message in first_context.input],
            [
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.TOOL,
            ],
        )
        self.assertEqual(
            [message.role for message in second_context.input],
            [
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.USER,
                MessageRole.ASSISTANT,
                MessageRole.TOOL,
                MessageRole.ASSISTANT,
                MessageRole.TOOL,
            ],
        )
        first_tool_call = second_context.input[3].tool_calls
        second_tool_call = second_context.input[5].tool_calls
        self.assertEqual(
            first_tool_call,
            [
                MessageToolCall(
                    id="call1",
                    name="calc",
                    arguments={"expression": "25 * 2"},
                )
            ],
        )
        self.assertEqual(
            second_tool_call,
            [
                MessageToolCall(
                    id="call2",
                    name="calc",
                    arguments={"expression": "50 * 2"},
                )
            ],
        )


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

        resp = _make_response(
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

        async def cancel() -> None:
            return None

        resp.set_cancellation_checker(cancel)
        resp.__aiter__()

        self.assertEqual(resp._tool_context.agent_id, aid)
        self.assertEqual(resp._tool_context.participant_id, pid)
        self.assertEqual(resp._tool_context.session_id, sid)
        self.assertEqual(resp._tool_context.calls, [])
        self.assertIs(resp._tool_context.cancellation_checker, cancel)

        self.assertEqual(resp._context.agent_id, aid)
        self.assertEqual(resp._context.participant_id, pid)
        self.assertEqual(resp._context.session_id, sid)

    async def test_child_context_inherits_identifiers(self):
        agent = MagicMock(spec=EngineAgent)
        operation = _dummy_operation()

        aid = uuid4()
        pid = uuid4()
        sid = uuid4()

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            agent_id=aid,
            participant_id=pid,
            session_id=sid,
        )

        parent = resp._context
        child = resp._make_child_context(
            Message(role=MessageRole.USER, content="hello")
        )

        self.assertEqual(child.agent_id, aid)
        self.assertEqual(child.participant_id, pid)
        self.assertEqual(child.session_id, sid)
        self.assertIs(child.parent, parent)


class OrchestratorResponseParsedTokensTestCase(IsolatedAsyncioTestCase):
    async def test_mixed_tokens(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        resp = _make_response(
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
