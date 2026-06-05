from logging import getLogger
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
)
from avalan.cli import CommandAbortException
from avalan.entities import (
    EngineUri,
    Input,
    Message,
    MessageRole,
    Token,
    ToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    TransformerEngineSettings,
)
from avalan.event import Event, EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.stream import TextGenerationSingleStream
from avalan.task.usage import (
    UsageSource,
    usage_observation_from_response,
    usage_observations_from_response,
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
        settings=TransformerEngineSettings(),
    )
    spec = Specification(role="assistant", goal=None)
    return AgentOperation(specification=spec, environment=env)


def _dummy_response(async_gen: bool = True) -> TextGenerationResponse:
    async def output_gen():
        yield "a"
        yield Token(id=5, token="b")

    def output_fn():
        return output_gen()

    return TextGenerationResponse(
        output_fn, logger=getLogger(), use_async_generator=async_gen
    )


def _usage_response(text: str, usage: object) -> TextGenerationResponse:
    return TextGenerationResponse(
        TextGenerationSingleStream(text, usage=usage),
        logger=getLogger(),
        use_async_generator=False,
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


class OrchestratorResponseAdditionalCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_usage_returns_none_without_provider_usage(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
        )

        self.assertIsNone(response.usage)

    async def test_usage_returns_single_provider_usage(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        usage = {"input_tokens": 1, "total_tokens": 2}
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _usage_response("answer", usage),
            agent,
            operation,
            {},
        )

        self.assertEqual(response.usage, usage)

    async def test_provider_usage_survives_to_str_tool_loop(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        outer_response = _usage_response(
            "call",
            {
                "input_tokens": 3,
                "cached_input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 5,
                "provider_family": "openai",
            },
        )
        inner_response = _usage_response(
            "answer",
            {
                "input_tokens": 4,
                "cache_creation_input_tokens": 2,
                "output_tokens": 6,
                "reasoning_tokens": 1,
                "total_tokens": 10,
                "provider_family": "openai",
            },
        )
        agent.return_value = inner_response
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [ToolCall(id=uuid4(), name="calc", arguments=None)]
            if text == "call"
            else None
        )

        async def exec_tool(call, context):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="ok",
            )

        tool.side_effect = exec_tool
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        output = await response.to_str()
        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(output, "answer")
        self.assertEqual(len(response.usage_responses), 2)
        self.assertIsInstance(response.usage, tuple)
        self.assertEqual(len(observations), 2)
        self.assertTrue(
            all(
                observation.source == UsageSource.EXACT
                for observation in observations
            )
        )
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.totals.input_tokens, 7)
        self.assertEqual(aggregate.totals.cached_input_tokens, 1)
        self.assertEqual(aggregate.totals.cache_creation_input_tokens, 2)
        self.assertEqual(aggregate.totals.output_tokens, 8)
        self.assertEqual(aggregate.totals.reasoning_tokens, 1)
        self.assertEqual(aggregate.totals.total_tokens, 15)

    async def test_malformed_wrapper_usage_does_not_hide_valid_child_usage(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        outer_response = _usage_response(
            "call",
            {
                "input_tokens": "private prompt",
                "cached_input_tokens": True,
                "output_tokens": -1,
                "total_tokens": 1.5,
                "provider_family": "private-provider",
            },
        )
        inner_response = _usage_response(
            "answer",
            {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )
        agent.return_value = inner_response
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [ToolCall(id=uuid4(), name="calc", arguments=None)]
            if text == "call"
            else None
        )

        async def exec_tool(call, context):
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="ok",
            )

        tool.side_effect = exec_tool
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        await response.to_str()
        observations = usage_observations_from_response(response)
        aggregate = usage_observation_from_response(response)

        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0].source, UsageSource.ESTIMATED)
        self.assertEqual(observations[0].totals.input_tokens, 0)
        self.assertEqual(observations[0].totals.output_tokens, 4)
        self.assertIsNone(observations[0].totals.total_tokens)
        self.assertEqual(observations[0].metadata, {})
        self.assertEqual(observations[1].source, UsageSource.EXACT)
        self.assertEqual(observations[1].totals.input_tokens, 0)
        self.assertEqual(observations[1].totals.output_tokens, 0)
        self.assertEqual(observations[1].totals.total_tokens, 0)
        self.assertEqual(observations[1].metadata, {})
        self.assertIsNotNone(aggregate)
        assert aggregate is not None
        self.assertEqual(aggregate.source, UsageSource.ESTIMATED)
        self.assertEqual(aggregate.totals.input_tokens, 0)
        self.assertEqual(aggregate.totals.output_tokens, 4)
        self.assertEqual(aggregate.totals.total_tokens, 0)
        rendered = str(observations) + str(aggregate)
        self.assertNotIn("private prompt", rendered)
        self.assertNotIn("private-provider", rendered)

    async def test_react_uses_explicit_output(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _dummy_response()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        result = await resp._react(response, output="forced")

        self.assertEqual(result, "forced")

    async def test_response_text_and_calls_skips_events(self):
        class Response:
            is_async_generator = True

            def __aiter__(self):
                return self.output_gen()

            async def output_gen(self):
                yield "a"
                yield Event(type=EventType.TOOL_DETECT)
                yield Token(id=7, token="b")

        text, calls = await OrchestratorResponse._response_text_and_calls(
            Response()
        )

        self.assertEqual(text, "ab")
        self.assertEqual(calls, [])

    async def test_tool_process_queue(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
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

    async def test_tool_call_token_emits_event(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id=uuid4(), name="calc", arguments=None)

        async def output_gen():
            yield ToolCallToken(token="c", call=call)

        response = TextGenerationResponse(
            output_gen, logger=getLogger(), use_async_generator=True
        )

        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        iterator = resp.__aiter__()
        first = await iterator.__anext__()
        self.assertIsInstance(first, ToolCallToken)

        second = await iterator.__anext__()
        self.assertEqual(second.type, EventType.TOOL_PROCESS)
        self.assertEqual(second.payload, [call])

        third = await iterator.__anext__()
        self.assertEqual(third.type, EventType.TOOL_RESULT)

        calls = [c.args[0] for c in event_manager.trigger.await_args_list]
        self.assertTrue(any(c.type == EventType.TOOL_PROCESS for c in calls))

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
        resp = _make_response(
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

        resp = _make_response(
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
        resp = _make_response(
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
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        initial_context = resp._context
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
        child_context = agent.await_args_list[0].args[0]
        self.assertIs(child_context.parent, initial_context)
        self.assertIs(child_context.root_parent, initial_context)
        self.assertIs(resp._context, child_context)

    async def test_emit_token_and_process(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        agent.engine.tokenizer.encode.return_value = [5]
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
        await resp._emit(Token(id=5, token="x"))
        event_manager.trigger.assert_awaited()
        process_event = Event(type=EventType.TOOL_PROCESS, payload=None)
        returned = await resp._emit(process_event)
        self.assertEqual(returned, process_event)
        other = Event(type=EventType.END)
        self.assertIs(await resp._emit(other), other)

    async def test_tool_call_error_message(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _dummy_response()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def tool_exec(call, context):
            return ToolCallError(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                error=ValueError("boom"),
                message="boom",
            )

        tool.side_effect = tool_exec

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        resp.__aiter__()
        call = ToolCall(id=uuid4(), name="fail", arguments={})
        resp._tool_call_events.put(
            Event(type=EventType.TOOL_PROCESS, payload=[call])
        )

        event = await resp.__anext__()
        self.assertEqual(event.type, EventType.TOOL_RESULT)

        model_event = await resp.__anext__()
        self.assertEqual(model_event.type, EventType.TOOL_MODEL_RESPONSE)

        context = agent.await_args_list[0].args[0]
        assert isinstance(context.input, list)
        self.assertEqual(
            context.input[2].tool_call_error.message,
            "boom",
        )
