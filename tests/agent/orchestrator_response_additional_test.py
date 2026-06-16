from logging import getLogger
from typing import cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    LegacyToolEventShim,
    OrchestratorResponse,
    classify_legacy_tool_event_shim,
    legacy_tool_event_shim_inventory,
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
from avalan.event import TOOL_TYPES, Event, EventPayloadKind, EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    TextGenerationSingleStream,
    stream_channel_for_kind,
    validate_canonical_stream_items,
    validate_tool_lifecycle_items,
)
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


def _empty_response() -> TextGenerationResponse:
    async def output_gen():
        for token in ():
            yield token

    def output_fn():
        return output_gen()

    return TextGenerationResponse(
        output_fn, logger=getLogger(), use_async_generator=True
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
    async def test_stream_item_projection_confines_legacy_surfaces(self):
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
        call = ToolCall(id="call-1", name="calc", arguments={"x": 1})
        tool_token = ToolCallToken(token='{"x":1}', call=call)
        event = Event(type=EventType.TOOL_PROCESS, payload=[call])
        canonical_item = CanonicalStreamItem(
            stream_session_id="stream",
            run_id="run",
            turn_id="turn",
            sequence=7,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="answer",
        )
        projection_item = StreamConsumerProjection.from_item(canonical_item)

        string_projection = response._stream_item_projection("raw", 3)
        event_projection = response._stream_item_projection(event, 4)
        tool_projection = response._stream_item_projection(tool_token, 5)
        semantic_projection = response._stream_item_projection(
            projection_item,
            6,
        )

        assert string_projection.canonical_item is not None
        self.assertEqual(
            string_projection.canonical_item.kind,
            StreamItemKind.ANSWER_DELTA,
        )
        self.assertEqual(string_projection.canonical_item.text_delta, "raw")
        self.assertEqual(string_projection.parser_text, "raw")
        self.assertEqual(string_projection.legacy_token, "raw")
        self.assertIs(event_projection.event, event)
        self.assertIsNone(event_projection.canonical_item)
        self.assertIs(tool_projection.legacy_tool_call_token, tool_token)
        assert tool_projection.canonical_item is not None
        self.assertEqual(
            tool_projection.canonical_item.kind,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        )
        self.assertEqual(
            tool_projection.canonical_item.correlation.tool_call_id,
            "call-1",
        )
        self.assertEqual(
            tool_projection.canonical_item.data,
            {"name": "calc", "arguments": {"x": 1}},
        )
        self.assertTrue(semantic_projection.canonical_source)
        assert semantic_projection.canonical_item is not None
        self.assertEqual(semantic_projection.canonical_item, canonical_item)

    async def test_stream_item_projection_rejects_unsupported_item(self):
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

        with self.assertRaises(AssertionError):
            response._stream_item_projection(object(), 0)

    async def test_projection_reuses_recorded_tool_lifecycle_item(self):
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
        call = ToolCall(id="call-1", name="calc", arguments={"x": 1})
        token = ToolCallToken(token='{"x":1}', call=call)
        response._append_canonical_item(StreamItemKind.STREAM_STARTED)
        response._record_streamed_tool_call_token(token)
        items_before = response.canonical_items

        response._append_canonical_projection_item(token)
        response._append_canonical_projection_item(
            Event(type=EventType.TOOL_PROCESS, payload=[call])
        )

        self.assertEqual(response.canonical_items, items_before)
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            ],
        )

    async def test_iteration_projects_raw_response_event_through_emit(self):
        class Response:
            is_async_generator = True

            def __aiter__(self):
                return self.output_gen()

            async def output_gen(self):
                yield Event(type=EventType.TOOL_DIAGNOSTIC)

        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            cast(TextGenerationResponse, Response()),
            agent,
            operation,
            {},
        )

        item = await response.__aiter__().__anext__()

        assert isinstance(item, Event)
        self.assertEqual(item.type, EventType.TOOL_DIAGNOSTIC)

    async def test_legacy_tool_event_shim_inventory_covers_tool_events(self):
        inventory = legacy_tool_event_shim_inventory()

        self.assertEqual(
            {shim.event_type for shim in inventory},
            TOOL_TYPES,
        )
        self.assertEqual(
            len(inventory), len({shim.event_type for shim in inventory})
        )
        for shim in inventory:
            with self.subTest(event_type=shim.event_type):
                self.assertIs(
                    classify_legacy_tool_event_shim(shim.event_type),
                    shim,
                )
                self.assertIs(
                    shim.canonical_channel,
                    stream_channel_for_kind(shim.canonical_kind),
                )
                self.assertTrue(shim.owner)
                self.assertTrue(shim.removal_condition)

        with self.assertRaises(StreamValidationError):
            classify_legacy_tool_event_shim(EventType.END)
        with self.assertRaises(AssertionError):
            classify_legacy_tool_event_shim("tool_result")  # type: ignore[arg-type]

    async def test_legacy_tool_event_shim_rejects_malformed_entries(self):
        invalid_entries = (
            lambda: LegacyToolEventShim(
                event_type=EventType.END,
                canonical_kind=StreamItemKind.STREAM_DIAGNOSTIC,
                canonical_channel=StreamChannel.CONTROL,
                owner="agent.orchestrator.response",
                removal_condition="done",
            ),
            lambda: LegacyToolEventShim(
                event_type=EventType.TOOL_RESULT,
                canonical_kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                canonical_channel=StreamChannel.CONTROL,
                owner="agent.orchestrator.response",
                removal_condition="done",
            ),
            lambda: LegacyToolEventShim(
                event_type=EventType.TOOL_RESULT,
                canonical_kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                canonical_channel=StreamChannel.TOOL_EXECUTION,
                owner="",
                removal_condition="done",
            ),
            lambda: LegacyToolEventShim(
                event_type=EventType.TOOL_RESULT,
                canonical_kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
                canonical_channel=StreamChannel.TOOL_EXECUTION,
                owner="agent.orchestrator.response",
                removal_condition="",
            ),
        )

        for build_entry in invalid_entries:
            with self.subTest(build_entry=build_entry):
                with self.assertRaises(AssertionError):
                    build_entry()

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
        response._model_responses = []
        response._response = MagicMock(
            input_token_count=None,
            output_token_count=None,
            usage=None,
        )
        self.assertEqual(response._canonical_usage(), {})

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
        validate_canonical_stream_items(response.canonical_items)
        terminal_usage = response.canonical_items[-2].usage
        self.assertIsNotNone(terminal_usage)
        assert terminal_usage is not None
        self.assertEqual(terminal_usage["source"], UsageSource.EXACT.value)
        self.assertEqual(terminal_usage["totals"]["input_tokens"], 7)
        self.assertEqual(terminal_usage["totals"]["output_tokens"], 8)
        self.assertEqual(terminal_usage["totals"]["total_tokens"], 15)

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

    async def test_canonical_completion_usage_uses_sanitized_estimate(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        outer_response = _usage_response(
            "call",
            {
                "input_tokens": "private prompt",
                "provider_family": "private-provider",
            },
        )
        inner_response = _usage_response(
            "answer",
            {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        )
        agent.return_value = inner_response
        call = ToolCall(id=uuid4(), name="calc", arguments=None)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [call] if text == "call" else None
        )
        tool.return_value = ToolCallResult(
            id=uuid4(),
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="ok",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=MagicMock(trigger=AsyncMock()),
            tool=tool,
        )

        await response.to_str()

        terminal_usage = response.canonical_items[-2].usage
        self.assertIsNotNone(terminal_usage)
        assert terminal_usage is not None
        self.assertEqual(terminal_usage["source"], UsageSource.ESTIMATED.value)
        self.assertEqual(terminal_usage["totals"]["input_tokens"], 1)
        self.assertEqual(terminal_usage["totals"]["output_tokens"], 6)
        rendered = str(terminal_usage)
        self.assertNotIn("private prompt", rendered)
        self.assertNotIn("private-provider", rendered)

    async def test_canonical_items_carry_agent_correlation(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        agent_id = uuid4()
        participant_id = uuid4()
        session_id = uuid4()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(async_gen=False),
            agent,
            operation,
            {},
            agent_id=agent_id,
            participant_id=participant_id,
            session_id=session_id,
        )

        await response.to_str()

        validate_canonical_stream_items(response.canonical_items)
        self.assertTrue(response.canonical_items)
        for item in response.canonical_items:
            self.assertEqual(item.stream_session_id, str(session_id))
            self.assertEqual(item.run_id, str(agent_id))
            self.assertEqual(item.turn_id, str(participant_id))
            self.assertEqual(item.correlation.task_id, str(agent_id))

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

        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [42]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        text, calls = await response._response_text_and_calls(
            cast(TextGenerationResponse, Response())
        )
        token_events = [
            call.args[0]
            for call in event_manager.trigger.await_args_list
            if call.args[0].type == EventType.TOKEN_GENERATED
        ]

        self.assertEqual(text, "ab")
        self.assertEqual(calls, [])
        self.assertEqual(
            [event.payload["token"] for event in token_events],
            ["a", "b"],
        )
        self.assertEqual(
            [event.observability.kind for event in token_events],
            [
                EventPayloadKind.CANONICAL_STREAM,
                EventPayloadKind.CANONICAL_STREAM,
            ],
        )
        self.assertEqual(
            [
                cast(
                    dict[str, object],
                    event.observability.data["summary"],
                )["text_delta_length"]
                for event in token_events
            ],
            [1, 1],
        )
        self.assertEqual(
            [event.observability.data["sequence"] for event in token_events],
            [0, 1],
        )

    async def test_response_text_and_calls_reads_semantic_answer_items(self):
        async def output_gen():
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="semantic",
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        semantic_response = TextGenerationResponse(
            output_gen,
            logger=getLogger(),
            use_async_generator=True,
        )
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

        text, calls = await response._response_text_and_calls(
            semantic_response
        )

        self.assertEqual(text, "semantic")
        self.assertEqual(calls, [])

    async def test_streaming_iteration_projects_semantic_answer_items(self):
        async def output_gen():
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="semantic",
            )
            yield CanonicalStreamItem(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
                sequence=2,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )

        semantic_response = TextGenerationResponse(
            output_gen,
            logger=getLogger(),
            use_async_generator=True,
        )
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            semantic_response,
            agent,
            operation,
            {},
        )
        response.__aiter__()

        item = await response.__anext__()

        self.assertIsInstance(item, Token)
        self.assertEqual(item.token, "semantic")

    async def test_tool_process_queue(self):
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
        tool_items = tuple(
            item
            for item in resp.canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            }
        )
        validated_items = validate_tool_lifecycle_items(
            tool_items,
            planned_tool_call_ids=[str(call.id)],
        )

        calls = [c.args[0] for c in event_manager.trigger.await_args_list]
        self.assertTrue(any(c.type == EventType.TOOL_PROCESS for c in calls))
        self.assertEqual(
            [item.kind for item in validated_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ],
        )
        self.assertEqual(
            {item.correlation.tool_call_id for item in validated_items},
            {str(call.id)},
        )

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
            _empty_response(),
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
            _empty_response(),
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
            _empty_response(),
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
        event_manager.should_emit.return_value = True
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
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
        event_manager.should_emit.return_value = True
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
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
        self.assertIsNone(returned)
        self.assertIs(await resp.__anext__(), process_event)
        other = Event(type=EventType.END)
        self.assertIs(await resp._emit(other), other)

    async def test_token_event_policy_fallbacks(self):
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
            event_manager=None,
        )

        self.assertFalse(resp._should_emit_token_generated_event())
        self.assertFalse(resp._should_enrich_token_ids())

        resp._event_manager = cast(EventManager, object())
        self.assertTrue(resp._should_emit_token_generated_event())

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
            _empty_response(),
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
