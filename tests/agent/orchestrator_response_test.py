from asyncio import CancelledError, create_task, sleep, wait_for
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import StringIO
from json import loads
from logging import getLogger
from typing import Any, cast
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
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    ToolFormat,
    ToolManagerSettings,
    TransformerEngineSettings,
)
from avalan.event import Event, EventPayloadKind, EventType
from avalan.event.manager import EventManager
from avalan.model import TextGenerationResponse
from avalan.model.call import ModelCallContext
from avalan.model.response.parsers.reasoning import ReasoningParser
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemKind,
    StreamTerminalOutcome,
    validate_canonical_stream_items,
    validate_tool_lifecycle_items,
)
from avalan.tool import Tool, ToolSet
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


class _SkippingToolParser:
    async def push(self, _: str) -> list[Token | TokenDetail | Event]:
        return []

    async def flush(self) -> list[Token | TokenDetail | Event]:
        return []


class _ToolProcessOnlyParser:
    def __init__(self, call: ToolCall) -> None:
        self._call = call
        self._emitted = False

    async def push(self, _: str) -> list[Token | TokenDetail | Event]:
        if self._emitted:
            return []
        self._emitted = True
        return [Event(type=EventType.TOOL_PROCESS, payload=[self._call])]

    async def flush(self) -> list[Token | TokenDetail | Event]:
        return []


class _FlushItemsParser:
    def __init__(self) -> None:
        self._flushed = False

    async def push(self, _: str) -> list[Token | TokenDetail | Event]:
        return []

    async def flush(self) -> list[Token | TokenDetail | Event]:
        if self._flushed:
            return []
        self._flushed = True
        return [
            Token(id=9, token="x"),
            Event(type=EventType.TOOL_DIAGNOSTIC),
        ]


class _UnsupportedFlushEventParser:
    async def push(self, _: str) -> list[Token | TokenDetail | Event]:
        return []

    async def flush(self) -> list[Token | TokenDetail | Event]:
        return [Event(type=EventType.END)]


class _PushDiagnosticParser:
    async def push(self, _: str) -> list[Token | TokenDetail | Event]:
        return [Event(type=EventType.TOOL_DIAGNOSTIC)]

    async def flush(self) -> list[Token | TokenDetail | Event]:
        return []


class OrchestratorResponseIterationTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_emits_events_and_end(self):
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [42]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        event_manager.enrich_token_ids = True

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
        self.assertIs(
            token_events[0].observability.kind,
            EventPayloadKind.CANONICAL_STREAM,
        )
        self.assertEqual(
            token_events[0].observability.data["kind"],
            StreamItemKind.ANSWER_DELTA.value,
        )
        self.assertEqual(
            token_events[0].observability.data["channel"],
            "answer",
        )
        self.assertEqual(
            token_events[0].observability.data["summary"],
            {"text_delta_length": 1},
        )
        self.assertNotIn("token", token_events[0].observability.data)
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
        self.assertEqual(
            token_events[1].observability.data["summary"],
            {"text_delta_length": 1, "metadata_keys": ["token_id"]},
        )

    async def test_consumer_projections_align_token_event_sequences(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        event_manager.enrich_token_ids = False

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        projections = [
            projection
            async for projection in resp.consumer_projections(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
            )
        ]
        answer_sequences = [
            projection.sequence
            for projection in projections
            if projection.kind is StreamItemKind.ANSWER_DELTA
        ]
        token_events = [
            call.args[0]
            for call in event_manager.trigger.await_args_list
            if call.args[0].type is EventType.TOKEN_GENERATED
        ]
        event_sequences = [
            event.observability.data["sequence"] for event in token_events
        ]

        self.assertEqual(answer_sequences, [1, 2])
        self.assertEqual(event_sequences, answer_sequences)

    async def test_consumer_projections_align_canonical_answer_event_sequence(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True

        async def gen() -> AsyncIterator[CanonicalStreamItem]:
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=36,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
            )
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=37,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta="answer",
            )
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=38,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
            )
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=39,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage={"source": "test"},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yield CanonicalStreamItem(
                stream_session_id="provider-stream",
                run_id="provider-run",
                turn_id="provider-turn",
                sequence=40,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
            )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        projections = [
            projection
            async for projection in resp.consumer_projections(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
            )
        ]
        answer_sequences = [
            projection.sequence
            for projection in projections
            if projection.kind is StreamItemKind.ANSWER_DELTA
        ]
        token_events = [
            call.args[0]
            for call in event_manager.trigger.await_args_list
            if call.args[0].type is EventType.TOKEN_GENERATED
        ]
        event_sequences = [
            event.observability.data["sequence"] for event in token_events
        ]

        self.assertEqual(answer_sequences, [1])
        self.assertEqual(event_sequences, answer_sequences)

    async def test_projection_rejects_mismatched_token_event_sequence(
        self,
    ) -> None:
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
        resp._last_token_event_canonical_item = CanonicalStreamItem(
            stream_session_id=resp._canonical_stream_session_id,
            run_id=resp._canonical_run_id,
            turn_id=resp._canonical_turn_id,
            sequence=1,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta="same",
        )

        self.assertIsNone(
            resp._matching_token_event_canonical_item(
                ReasoningToken(token="same")
            )
        )
        self.assertIsNone(resp._last_token_event_canonical_item)

    async def test_to_str_emits_streamed_token_events(self):
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [42]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        event_manager.enrich_token_ids = False

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        self.assertEqual(await resp.to_str(), "ab")
        token_events = [
            call.args[0]
            for call in event_manager.trigger.await_args_list
            if call.args[0].type == EventType.TOKEN_GENERATED
        ]
        self.assertEqual(len(token_events), 2)
        self.assertEqual(
            [event.payload["token"] for event in token_events],
            ["a", "b"],
        )
        self.assertEqual(
            [event.payload["step"] for event in token_events],
            [0, 1],
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
        event_sequences = [
            event.observability.data["sequence"] for event in token_events
        ]
        canonical_sequences = {item.sequence for item in resp.canonical_items}
        self.assertEqual(event_sequences, [1, 2])
        self.assertTrue(canonical_sequences.isdisjoint(event_sequences))
        self.assertEqual(
            [item.kind for item in resp.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        engine.tokenizer.encode.assert_not_called()

    async def test_token_events_skip_tokenizer_when_enrichment_disabled(self):
        class LazyTokenizerEngine:
            model_id = "m"

            @property
            def tokenizer(self) -> object:
                raise AssertionError("tokenizer should be opt-in")

        agent = MagicMock(spec=EngineAgent)
        agent.engine = LazyTokenizerEngine()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        event_manager.enrich_token_ids = False

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        item = await resp._emit_token_generated_event("a")

        self.assertIsNotNone(item)
        event = event_manager.trigger.await_args.args[0]
        self.assertIs(event.type, EventType.TOKEN_GENERATED)
        self.assertIsNone(event.payload["token_id"])
        self.assertEqual(event.payload["token"], "a")

    async def test_iteration_skips_token_events_without_subscriber(self):
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [42]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = False

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        self.assertEqual(
            [t async for t in resp],
            ["a", Token(id=5, token="b")],
        )
        calls = event_manager.trigger.await_args_list
        self.assertFalse(
            any(c.args[0].type == EventType.TOKEN_GENERATED for c in calls)
        )
        engine.tokenizer.encode.assert_not_called()

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


class OrchestratorResponseCanonicalLifecycleTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_skips_parser_empty_chunks_without_recursion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen() -> AsyncIterator[str]:
            for _ in range(1200):
                yield "{"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _SkippingToolParser()
        )

        iterator = orchestrated.__aiter__()
        with self.assertRaises(StopAsyncIteration):
            await wait_for(iterator.__anext__(), 1)

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(orchestrated._step, 1200)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_parser_tool_process_only_item_advances_lifecycle(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})

        async def gen() -> AsyncIterator[str]:
            yield "<tool>"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = None
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _ToolProcessOnlyParser(call)
        )

        iterator = orchestrated.__aiter__()
        first = await wait_for(iterator.__anext__(), 1)
        second = await wait_for(iterator.__anext__(), 1)
        with self.assertRaises(StopAsyncIteration):
            await wait_for(iterator.__anext__(), 1)

        self.assertEqual(first.type, EventType.TOOL_PROCESS)
        self.assertEqual(second.type, EventType.TOOL_RESULT)
        tool.assert_awaited_once()
        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data["code"],
            "orchestrator.tool_cycle.empty_observation",
        )

    async def test_iteration_tool_continuation_without_event_manager(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})

        async def gen() -> AsyncIterator[str]:
            yield "<tool>"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _ToolProcessOnlyParser(call)
        )

        items: list[Any] = []
        iterator = orchestrated.__aiter__()
        while True:
            try:
                items.append(await wait_for(iterator.__anext__(), 1))
            except StopAsyncIteration:
                break

        self.assertEqual(
            [getattr(item, "type", None) for item in items[:3]],
            [
                EventType.TOOL_PROCESS,
                EventType.TOOL_RESULT,
                EventType.TOOL_MODEL_RESPONSE,
            ],
        )
        self.assertEqual(len(items), 3)
        tool.assert_awaited_once()
        agent.assert_awaited_once()
        context = agent.await_args.args[0]
        assert isinstance(context.input, list)
        self.assertEqual(context.input[-1].content, '"4"')
        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_parser_flush_items_are_returned_before_completion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen() -> AsyncIterator[str]:
            yield "{"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _FlushItemsParser()
        )

        iterator = orchestrated.__aiter__()
        token = await wait_for(iterator.__anext__(), 1)
        diagnostic = await wait_for(iterator.__anext__(), 1)
        with self.assertRaises(StopAsyncIteration):
            await wait_for(iterator.__anext__(), 1)

        self.assertEqual(token, Token(id=9, token="x"))
        self.assertEqual(diagnostic.type, EventType.TOOL_DIAGNOSTIC)
        validate_canonical_stream_items(orchestrated.canonical_items)

    async def test_parser_flush_rejects_non_process_control_event(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen() -> AsyncIterator[str]:
            yield "{"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _UnsupportedFlushEventParser()
        )

        with self.assertRaises(AssertionError):
            await wait_for(orchestrated.__aiter__().__anext__(), 1)

    async def test_emit_queues_parser_diagnostic_event(self) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        orchestrated.__aiter__()
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _PushDiagnosticParser()
        )

        item = await orchestrated._emit("{")

        assert isinstance(item, Event)
        self.assertEqual(item.type, EventType.TOOL_DIAGNOSTIC)

    async def test_iteration_records_tool_and_continuation_lifecycle(
        self,
    ) -> None:
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

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def inner_gen() -> AsyncIterator[str]:
            yield "4"

        inner_response = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        agent.return_value = inner_response

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        items = [item async for item in response]

        self.assertEqual(
            [getattr(item, "type", None) for item in items],
            [
                None,
                EventType.TOOL_PROCESS,
                EventType.TOOL_RESULT,
                EventType.TOOL_MODEL_RESPONSE,
                None,
            ],
        )
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )
        self.assertEqual(
            canonical_items[1].text_delta, '{"expression": "2 + 2"}'
        )
        self.assertEqual(
            canonical_items[6].correlation.model_continuation_id,
            canonical_items[7].correlation.model_continuation_id,
        )

    async def test_iteration_records_live_tool_output_before_completion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(
            id="call1",
            name="shell.cat",
            arguments={"path": "file.txt"},
        )

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def execute(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolCallResult:
            assert context.stream_event is not None
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="alpha\n",
                    metadata={"bytes": 6},
                )
            )
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDERR,
                    content="warn\n",
                    metadata={"bytes": 5},
                )
            )
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.PROGRESS,
                    content="half",
                    progress=0.5,
                )
            )
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.LOG,
                    content="trace\n",
                    metadata={"logger": "tool"},
                )
            )
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="alpha\n",
            )

        tool.side_effect = execute

        async def inner_gen() -> AsyncIterator[str]:
            if False:
                yield ""

        agent.return_value = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        items = [item async for item in response]

        self.assertEqual(
            [getattr(item, "type", None) for item in items],
            [
                None,
                EventType.TOOL_PROCESS,
                EventType.TOOL_RESULT,
                EventType.TOOL_MODEL_RESPONSE,
            ],
        )
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        output_items = [
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT
        ]
        self.assertEqual(
            [
                (item.data["category"], item.text_delta)
                for item in output_items
            ],
            [
                ("stdout", "alpha\n"),
                ("stderr", "warn\n"),
                ("log", "trace\n"),
            ],
        )
        self.assertEqual(output_items[0].data["metadata"], {"bytes": 6})
        self.assertEqual(output_items[2].data["metadata"], {"logger": "tool"})
        progress_item = canonical_items[7]
        self.assertEqual(
            progress_item.data,
            {
                "category": "progress",
                "content": "half",
                "progress": 0.5,
                "metadata": {},
            },
        )
        terminal_item = canonical_items[9]
        self.assertIs(
            terminal_item.kind, StreamItemKind.TOOL_EXECUTION_COMPLETED
        )
        self.assertTrue(
            all(
                item.sequence < terminal_item.sequence
                for item in (*output_items, progress_item)
            )
        )
        context = agent.await_args.args[0]
        assert isinstance(context.input, list)
        self.assertEqual(context.input[-2].tool_calls[0].id, "call1")
        self.assertEqual(context.input[-1].role, MessageRole.TOOL)
        self.assertEqual(loads(str(context.input[-1].content)), "alpha\n")

    async def test_tool_stream_events_after_terminal_are_ignored(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        result = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="1",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
        )
        response._append_canonical_tool_call_ready(call)
        response._append_canonical_tool_execution_started(call)
        emit = response._make_tool_stream_event_callback(call)

        await emit(
            ToolExecutionStreamEvent(
                kind=ToolExecutionStreamKind.LOG,
                content="before",
            )
        )
        response._append_canonical_tool_execution_terminal(call, result)
        await emit(
            ToolExecutionStreamEvent(
                kind=ToolExecutionStreamKind.STDOUT,
                content="after",
            )
        )
        response._append_canonical_tool_execution_error(
            call,
            RuntimeError("late"),
        )

        validate_tool_lifecycle_items(response.canonical_items)
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ],
        )
        self.assertEqual(response.canonical_items[-2].text_delta, "before")

    async def test_duplicate_tool_lifecycle_boundaries_are_idempotent(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "2 + 2"},
        )
        result = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
        )

        response._append_canonical_tool_call_ready(call)
        response._append_canonical_tool_call_ready(call)
        response._append_canonical_tool_execution_started(call)
        response._append_canonical_tool_execution_started(call)
        response._append_canonical_tool_execution_terminal(call, result)
        response._append_canonical_tool_execution_started(call)
        response._append_canonical_tool_execution_terminal(call, result)

        validate_tool_lifecycle_items(response.canonical_items)
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ],
        )

    async def test_iteration_accumulates_streamed_tool_call_arguments(
        self,
    ) -> None:
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

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token='{"expression"', call=call)
            yield ToolCallToken(token=':"2 + 2"}', call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def inner_gen() -> AsyncIterator[str]:
            yield "4"

        inner_response = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        agent.return_value = inner_response

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        items = [item async for item in response]

        self.assertEqual(
            [getattr(item, "type", None) for item in items],
            [
                None,
                None,
                EventType.TOOL_PROCESS,
                EventType.TOOL_RESULT,
                EventType.TOOL_MODEL_RESPONSE,
                None,
            ],
        )
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        tool_call_items = [
            item
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
        ]
        self.assertEqual(
            [item.kind for item in tool_call_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        self.assertEqual(
            [item.text_delta for item in tool_call_items[:2]],
            ['{"expression"', ':"2 + 2"}'],
        )
        self.assertEqual(
            [item.correlation.tool_call_id for item in tool_call_items],
            ["call1", "call1", "call1", "call1"],
        )
        self.assertEqual(
            tool_call_items[2].data,
            {"name": "calc", "arguments": {"expression": "2 + 2"}},
        )

    async def test_streamed_tool_call_token_boundaries_queue_prior_call(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        first_call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        second_call = ToolCall(id="call2", name="search", arguments={"q": 2})
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        response.__aiter__()

        response._record_streamed_tool_call_token(
            ToolCallToken(token='{"x":1}', call=first_call)
        )
        response._record_streamed_tool_call_token(
            ToolCallToken(token='{"q":2}', call=second_call)
        )

        self.assertFalse(response._tool_process_events.empty())
        queued = response._tool_process_events.get()
        self.assertEqual(queued.type, EventType.TOOL_PROCESS)
        self.assertEqual(queued.payload, [first_call])
        self.assertIs(response._pending_tool_call, second_call)

    async def test_tool_execution_exception_records_error_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.side_effect = RuntimeError("boom")

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(RuntimeError):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-3:]],
            [
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(canonical_items[-2].terminal_outcome, "errored")

    async def test_tool_execution_cancellation_records_cancel_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.side_effect = CancelledError()

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(CancelledError):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-3:]],
            [
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_cancellation_checker_stops_queued_tool_before_execution(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        checker = AsyncMock(side_effect=CancelledError())
        orchestrated.set_cancellation_checker(checker)
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(CancelledError):
            await iterator.__anext__()

        tool.assert_not_awaited()
        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertFalse(
            any(
                trigger.args[0].type is EventType.TOOL_EXECUTE
                for trigger in event_manager.trigger.await_args_list
            )
        )

    async def test_tool_confirmation_denial_records_cancel_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments=None)

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=lambda _: "n",
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(CommandAbortException):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            [item.kind for item in canonical_items],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-3:]],
            [
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data["code"],
            ToolCallDiagnosticCode.USER_REJECTED.value,
        )
        self.assertEqual(
            canonical_items[-3].data["stage"],
            ToolCallDiagnosticStage.CONFIRM.value,
        )

    async def test_async_tool_confirmation_denial_records_diagnostic(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def confirm(_: ToolCall) -> str:
            return "n"

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(CommandAbortException):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-3:]],
            [
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data["code"],
            ToolCallDiagnosticCode.USER_REJECTED.value,
        )

    async def test_tool_confirmation_exception_records_error_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        def confirm(_: ToolCall) -> str:
            raise RuntimeError("confirm failed")

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaisesRegex(RuntimeError, "confirm failed"):
            await iterator.__anext__()

        tool.assert_not_awaited()
        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-3:]],
            [
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data,
            {"error_type": "RuntimeError", "message": "confirm failed"},
        )

    async def test_async_tool_confirmation_cancellation_records_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def confirm(_: ToolCall) -> str:
            raise CancelledError()

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(CancelledError):
            await iterator.__anext__()

        tool.assert_not_awaited()
        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-3:]],
            [
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data["code"],
            ToolCallDiagnosticCode.CANCELLED.value,
        )
        self.assertEqual(
            canonical_items[-3].data["stage"],
            ToolCallDiagnosticStage.CONFIRM.value,
        )

    async def test_tool_confirmation_allows_all_from_coroutine(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def confirm(_: ToolCall) -> str:
            return "a"

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        await iterator.__anext__()

        self.assertTrue(orchestrated._tool_confirm_all)
        validate_tool_lifecycle_items(orchestrated.canonical_items)
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            [item.kind for item in orchestrated.canonical_items],
        )

    async def test_model_continuation_exception_records_error_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.side_effect = RuntimeError("model failed")
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(RuntimeError):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data,
            {"error_type": "RuntimeError", "message": "model failed"},
        )

    async def test_model_continuation_stream_exception_records_error_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def inner_gen() -> AsyncIterator[str]:
            yield "partial"
            raise RuntimeError("stream failed")

        agent.return_value = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        for _ in range(5):
            await iterator.__anext__()
        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data,
            {"error_type": "RuntimeError", "message": "stream failed"},
        )

    async def test_model_continuation_cancellation_records_cancel_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.side_effect = CancelledError()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        await iterator.__anext__()
        await iterator.__anext__()
        await iterator.__anext__()
        with self.assertRaises(CancelledError):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_model_continuation_stream_cancellation_records_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def inner_gen() -> AsyncIterator[str]:
            yield "partial"
            raise CancelledError()

        agent.return_value = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        iterator = orchestrated.__aiter__()

        for _ in range(5):
            await iterator.__anext__()
        with self.assertRaises(CancelledError):
            await iterator.__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_response_cancellation_records_cancel_terminal(self) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen() -> AsyncIterator[str]:
            raise CancelledError()
            yield "unreachable"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        with self.assertRaises(CancelledError):
            await orchestrated.__aiter__().__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )

    async def test_response_exception_records_error_terminal(self) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen() -> AsyncIterator[str]:
            raise ValueError("bad stream")
            yield "unreachable"

        response = TextGenerationResponse(
            lambda **_: gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        with self.assertRaises(ValueError):
            await orchestrated.__aiter__().__anext__()

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].data,
            {"error_type": "ValueError", "message": "bad stream"},
        )

    async def test_stream_terminal_recording_is_idempotent(self) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
        )

        response.__aiter__()
        with self.assertRaises(StopAsyncIteration):
            await response.__anext__()
        response._finish_canonical_stream(StreamItemKind.STREAM_ERRORED)

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_stream_terminal_blocks_late_semantic_items(self) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
        )
        call = ToolCall(
            id="call-1",
            name="late",
            arguments={"value": 1},
        )

        response._append_canonical_item(StreamItemKind.STREAM_STARTED)
        response._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
        emit = response._make_tool_stream_event_callback(call)

        await emit(
            ToolExecutionStreamEvent(
                kind=ToolExecutionStreamKind.STDOUT,
                content="late output",
            )
        )
        response._append_canonical_tool_call_ready(call)
        response._append_canonical_tool_execution_started(call)
        response._append_canonical_tool_execution_error(
            call,
            RuntimeError("late error"),
        )
        response._append_canonical_model_continuation(
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            "continuation-1",
        )
        response._append_canonical_guard_diagnostic(
            code="orchestrator.late",
            message="late diagnostic",
            details={},
        )
        self.assertIsNone(
            response._append_canonical_item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                data={"message": "late"},
            )
        )
        self.assertIsNone(
            response._append_canonical_item(StreamItemKind.STREAM_CLOSED)
        )

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[1].terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )

    async def test_tool_terminal_mapping_covers_terminal_outcomes(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
        )
        response.__aiter__()
        error_call = ToolCall(id="error-call", name="calc", arguments={})
        diagnostic_call = ToolCall(
            id="diagnostic-call", name="missing", arguments={}
        )
        cancel_call = ToolCall(
            id="cancel-call", name="cancelled", arguments={}
        )
        none_call = ToolCall(id="none-call", name="noop", arguments={})

        response._append_canonical_tool_call_ready(error_call)
        response._append_canonical_tool_execution_started(error_call)
        response._append_canonical_tool_execution_terminal(
            error_call,
            ToolCallError(
                id="error1",
                call=error_call,
                name=error_call.name,
                arguments=error_call.arguments,
                error="failed",
                message="failed",
            ),
        )
        response._append_canonical_tool_call_ready(diagnostic_call)
        response._append_canonical_tool_execution_started(diagnostic_call)
        response._append_canonical_tool_execution_terminal(
            diagnostic_call,
            ToolCallDiagnostic(
                id="diagnostic1",
                call_id=diagnostic_call.id,
                requested_name=diagnostic_call.name,
                code=ToolCallDiagnosticCode.UNKNOWN_TOOL,
                stage=ToolCallDiagnosticStage.RESOLVE,
                message="Unknown tool.",
            ),
        )
        response._append_canonical_tool_call_ready(cancel_call)
        response._append_canonical_tool_execution_started(cancel_call)
        response._append_canonical_tool_execution_terminal(
            cancel_call,
            ToolCallDiagnostic(
                id="cancel1",
                call_id=cancel_call.id,
                requested_name=cancel_call.name,
                code=ToolCallDiagnosticCode.CANCELLED,
                stage=ToolCallDiagnosticStage.GUARD,
                message="Tool call was cancelled.",
            ),
        )
        response._append_canonical_tool_call_ready(none_call)
        response._append_canonical_tool_execution_started(none_call)
        response._append_canonical_tool_execution_terminal(none_call, None)
        response._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.TOOL_EXECUTION_ERROR,
            }
        ]
        self.assertEqual(
            terminal_kinds,
            [
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ],
        )

    async def test_to_str_keeps_tool_calls_serial_by_default(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        active = 0
        peak_active = 0

        async def tracked(name: str, delay: float) -> str:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await sleep(delay)
            active -= 1
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first", "delay": 0.001},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second", "delay": 0.001},
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(peak_active, 1)
        validate_canonical_stream_items(response.canonical_items)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_to_str_runs_parallel_safe_tools_with_limit(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        active = 0
        peak_active = 0

        async def tracked(name: str, delay: float) -> str:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await sleep(delay)
            active -= 1
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(
                parallel_tool_calls=True,
                maximum_parallel_tool_calls=2,
            ),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first", "delay": 0.02},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second", "delay": 0.001},
            ),
            ToolCall(
                id="call-3",
                name="tracked",
                arguments={"name": "third", "delay": 0.001},
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(peak_active, 2)
        context = agent.await_args.args[0]
        tool_contents = [
            message.content
            for message in context.input
            if isinstance(message, Message)
            and message.role is MessageRole.TOOL
        ]
        self.assertEqual(tool_contents, ["first", "second", "third"])
        terminal_ids = [
            item.correlation.tool_call_id
            for item in response.canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids[:2], ["call-2", "call-1"])
        validate_canonical_stream_items(response.canonical_items)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_parallel_tool_error_cancels_siblings_before_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        started: list[str] = []
        cancelled: list[str] = []

        async def tracked(name: str) -> str:
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        async def execute_call(
            call: ToolCall,
            _context: ToolCallContext,
            **_: object,
        ) -> ToolCallResult:
            started.append(str(call.id))
            try:
                await sleep(0.001 if call.id == "call-1" else 1)
            except CancelledError:
                cancelled.append(str(call.id))
                raise
            if call.id == "call-1":
                raise RuntimeError("boom")
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=call.arguments["name"],
            )

        cast(Any, tool).execute_call = execute_call

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await response.to_str()

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertCountEqual(started, ["call-1", "call-2"])
        self.assertEqual(cancelled, ["call-2"])
        terminal_items = [
            item
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
            }
        ]
        self.assertEqual(
            [
                (item.correlation.tool_call_id, item.kind)
                for item in terminal_items
            ],
            [
                ("call-1", StreamItemKind.TOOL_EXECUTION_ERROR),
                ("call-2", StreamItemKind.TOOL_EXECUTION_CANCELLED),
            ],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_ERRORED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_parallel_tool_error_cancels_unstarted_siblings(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=MagicMock(spec=ToolManager),
            enable_tool_parsing=False,
        )
        response._append_canonical_item(StreamItemKind.STREAM_STARTED)

        async def lifecycle(
            call: ToolCall,
            **_: object,
        ) -> object:
            if call.id == "call-1":
                error = RuntimeError("boom")
                response._append_canonical_tool_call_ready(call)
                response._append_canonical_tool_execution_started(call)
                response._append_canonical_tool_execution_error(call, error)
                raise error
            await sleep(1)
            raise AssertionError("unreachable")

        cast(Any, response)._execute_tool_call_with_lifecycle = lifecycle

        with self.assertRaisesRegex(RuntimeError, "boom"):
            await response._execute_tool_call_batch(
                calls,
                confirm=False,
                abort_on_reject=False,
                emit_ready=True,
            )

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        terminal_items = [
            item
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
            }
        ]
        self.assertEqual(
            [
                (item.correlation.tool_call_id, item.kind)
                for item in terminal_items
            ],
            [
                ("call-1", StreamItemKind.TOOL_EXECUTION_ERROR),
                ("call-2", StreamItemKind.TOOL_EXECUTION_CANCELLED),
            ],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_ERRORED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_parallel_tool_cancellation_cancels_siblings_before_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        started: list[str] = []
        cancelled: list[str] = []

        async def tracked(name: str) -> str:
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        async def execute_call(
            call: ToolCall,
            _context: ToolCallContext,
            **_: object,
        ) -> ToolCallResult:
            started.append(str(call.id))
            if call.id == "call-1":
                while len(started) < 2:
                    await sleep(0)
                raise CancelledError()
            try:
                await sleep(1)
            except CancelledError:
                cancelled.append(str(call.id))
                raise
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=call.arguments["name"],
            )

        cast(Any, tool).execute_call = execute_call

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        with self.assertRaises(CancelledError):
            await response.to_str()

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        terminal_items = [
            item
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
            }
        ]
        self.assertEqual(
            [
                (item.correlation.tool_call_id, item.kind)
                for item in terminal_items
            ],
            [
                ("call-1", StreamItemKind.TOOL_EXECUTION_CANCELLED),
                ("call-2", StreamItemKind.TOOL_EXECUTION_CANCELLED),
            ],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_CANCELLED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_parallel_batch_cancellation_marks_unstarted_tools(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=MagicMock(spec=ToolManager),
            enable_tool_parsing=False,
        )
        response._append_canonical_item(StreamItemKind.STREAM_STARTED)

        async def lifecycle(
            _call: ToolCall,
            **_: object,
        ) -> object:
            await sleep(1)
            raise AssertionError("unreachable")

        cast(Any, response)._execute_tool_call_with_lifecycle = lifecycle

        task = create_task(
            response._execute_tool_call_batch(
                calls,
                confirm=False,
                abort_on_reject=False,
                emit_ready=True,
            )
        )
        await sleep(0)
        task.cancel()
        with self.assertRaises(CancelledError):
            await task

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        terminal_items = [
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_CANCELLED
        ]
        self.assertEqual(
            [item.correlation.tool_call_id for item in terminal_items],
            ["call-1", "call-2"],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_CANCELLED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_parallel_pre_execution_cancellation_marks_siblings(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        response._append_canonical_item(StreamItemKind.STREAM_STARTED)
        checks = 0

        async def cancel_first_check() -> None:
            nonlocal checks
            checks += 1
            if checks == 1:
                raise CancelledError()
            await sleep(1)

        response.set_cancellation_checker(cancel_first_check)

        with self.assertRaises(CancelledError):
            await response._execute_tool_call_batch(
                calls,
                confirm=False,
                abort_on_reject=False,
                emit_ready=True,
            )

        tool.assert_not_awaited()
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        terminal_items = [
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_CANCELLED
        ]
        self.assertEqual(
            [item.correlation.tool_call_id for item in terminal_items],
            ["call-1", "call-2"],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_CANCELLED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_parallel_wait_cancellation_cleans_up_running_tools(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        started: list[str] = []
        cancelled: list[str] = []

        async def tracked(name: str) -> str:
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        async def execute_call(
            call: ToolCall,
            _context: ToolCallContext,
            **_: object,
        ) -> ToolCallResult:
            started.append(str(call.id))
            try:
                await sleep(1)
            except CancelledError:
                cancelled.append(str(call.id))
                raise
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=call.arguments["name"],
            )

        cast(Any, tool).execute_call = execute_call

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        task = create_task(response.to_str())
        while len(started) < 2:
            await sleep(0)
        task.cancel()
        with self.assertRaises(CancelledError):
            await task

        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertCountEqual(cancelled, ["call-1", "call-2"])
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_CANCELLED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_parallel_execution_keeps_unsafe_tools_serial(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        active = 0
        peak_active = 0

        async def tracked(name: str, delay: float) -> str:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await sleep(delay)
            active -= 1
            return name

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first", "delay": 0.001},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second", "delay": 0.001},
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(peak_active, 1)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_to_str_assigns_distinct_ids_to_anonymous_tool_calls(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        async def tracked(name: str) -> str:
            return name

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        calls = [
            ToolCall(
                id=None,
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id=None,
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        canonical_items = response.canonical_items
        ready_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(
            ready_ids,
            ["orchestrator-tool-call-1", "orchestrator-tool-call-2"],
        )
        self.assertEqual(len(set(ready_ids)), 2)
        terminal_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids, ready_ids)
        child_context = agent.await_args.args[0]
        continuation_messages = cast(list[Message], child_context.input)
        observed_tool_call_ids = [
            message.tool_calls[0].id
            for message in continuation_messages
            if message.tool_calls
        ]
        self.assertEqual(observed_tool_call_ids, ready_ids)
        self.assertNotIn("None", observed_tool_call_ids)
        tool_contents = [
            message.content
            for message in continuation_messages
            if message.role is MessageRole.TOOL
        ]
        self.assertEqual(tool_contents, ["first", "second"])
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_to_str_merges_anonymous_tool_call_fragments(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        observed: list[str] = []

        async def tracked(name: str) -> str:
            observed.append(name)
            return name

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        fragments = [
            (
                '{"name": "fir',
                ToolCall(
                    id=None,
                    name="tracked",
                    arguments={"name": "fir"},
                ),
            ),
            (
                'st"}',
                ToolCall(
                    id=None,
                    name="tracked",
                    arguments={"name": "first"},
                ),
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for token, call in fragments:
                yield ToolCallToken(token=token, call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(observed, ["first"])
        canonical_items = response.canonical_items
        ready_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(ready_ids, ["orchestrator-tool-call-1"])
        argument_deltas = [
            item.text_delta
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        ]
        self.assertEqual(argument_deltas, ['{"name": "fir', 'st"}'])
        terminal_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids, ready_ids)
        child_context = agent.await_args.args[0]
        continuation_messages = cast(list[Message], child_context.input)
        tool_contents = [
            message.content
            for message in continuation_messages
            if message.role is MessageRole.TOOL
        ]
        self.assertEqual(tool_contents, ["first"])
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_to_str_keeps_same_name_anonymous_tool_calls_distinct(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        observed: list[str] = []

        async def tracked(name: str) -> str:
            observed.append(name)
            return name

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        chunks = [
            (
                '{"name":"first"}',
                ToolCall(
                    id=None,
                    name="tracked",
                    arguments={"name": "first"},
                ),
            ),
            (
                '{"name":"second"}',
                ToolCall(
                    id=None,
                    name="tracked",
                    arguments={"name": "second"},
                ),
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for token, call in chunks:
                yield ToolCallToken(token=token, call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(observed, ["first", "second"])
        canonical_items = response.canonical_items
        ready_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(
            ready_ids,
            ["orchestrator-tool-call-1", "orchestrator-tool-call-2"],
        )
        argument_deltas = [
            (item.correlation.tool_call_id, item.text_delta)
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        ]
        self.assertEqual(
            argument_deltas,
            [
                ("orchestrator-tool-call-1", '{"name":"first"}'),
                ("orchestrator-tool-call-2", '{"name":"second"}'),
            ],
        )
        terminal_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids, ready_ids)
        child_context = agent.await_args.args[0]
        tool_contents = [
            message.content
            for message in cast(list[Message], child_context.input)
            if message.role is MessageRole.TOOL
        ]
        self.assertEqual(tool_contents, ["first", "second"])
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_to_str_keeps_different_anonymous_tool_names_distinct(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        observed: list[str] = []

        async def first_tool(value: str) -> str:
            observed.append(f"first:{value}")
            return value

        async def second_tool(value: str) -> str:
            observed.append(f"second:{value}")
            return value

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[first_tool, second_tool])],
            enable_tools=["first_tool", "second_tool"],
            settings=ToolManagerSettings(),
        )
        fragments = [
            (
                '{"value": "a"}',
                ToolCall(
                    id=None,
                    name="first_tool",
                    arguments={"value": "a"},
                ),
            ),
            (
                '{"value": "b"}',
                ToolCall(
                    id=None,
                    name="second_tool",
                    arguments={"value": "b"},
                ),
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for token, call in fragments:
                yield ToolCallToken(token=token, call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(observed, ["first:a", "second:b"])
        canonical_items = response.canonical_items
        ready_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(
            ready_ids,
            ["orchestrator-tool-call-1", "orchestrator-tool-call-2"],
        )
        terminal_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids, ready_ids)
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_parallel_confirmation_denial_prevents_fanout(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []
        confirmed: list[str] = []

        async def tracked(name: str) -> str:
            executed.append(name)
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        def confirm(call: ToolCall) -> str:
            confirmed.append(str(call.id))
            return "n"

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(confirmed, ["call-1"])
        self.assertEqual(executed, [])
        diagnostics = [
            item
            for item in response.canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        ]
        cancellations = [
            item
            for item in response.canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_CANCELLED
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(len(cancellations), 1)
        self.assertEqual(
            [item.data["code"] for item in diagnostics],
            [ToolCallDiagnosticCode.USER_REJECTED.value],
        )
        validate_canonical_stream_items(response.canonical_items)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_parallel_confirmation_mixed_denial_reports_full_batch(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []
        decisions = {"call-1": "y", "call-2": "n"}
        confirmed: list[str] = []

        async def tracked(name: str) -> str:
            executed.append(name)
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        def confirm(call: ToolCall) -> str:
            confirmed.append(str(call.id))
            return decisions[str(call.id)]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(confirmed, ["call-1", "call-2"])
        self.assertEqual(executed, [])
        canonical_items = response.canonical_items
        self.assertEqual(
            [
                (
                    item.correlation.tool_call_id,
                    item.kind,
                    item.data.get("code") if item.data else None,
                )
                for item in canonical_items
                if item.kind
                in (
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                )
            ],
            [
                (
                    "call-1",
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                    None,
                ),
                (
                    "call-2",
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    ToolCallDiagnosticCode.USER_REJECTED.value,
                ),
            ],
        )
        child_context = agent.await_args.args[0]
        tool_messages = [
            message
            for message in cast(list[Message], child_context.input)
            if message.role is MessageRole.TOOL
        ]
        self.assertEqual(
            [
                message.tool_call_diagnostic.code
                for message in tool_messages
                if message.tool_call_diagnostic is not None
            ],
            [
                ToolCallDiagnosticCode.CANCELLED,
                ToolCallDiagnosticCode.USER_REJECTED,
            ],
        )
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_parallel_confirmation_cancellation_prevents_fanout(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []
        confirmed: list[str] = []

        async def tracked(name: str) -> str:
            executed.append(name)
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        async def confirm(call: ToolCall) -> str:
            confirmed.append(str(call.id))
            if call.id == "call-2":
                raise CancelledError()
            return "y"

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )

        with self.assertRaises(CancelledError):
            await response.to_str()

        self.assertEqual(confirmed, ["call-1", "call-2"])
        self.assertEqual(executed, [])
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [
                (
                    item.correlation.tool_call_id,
                    item.kind,
                    item.data.get("code") if item.data else None,
                )
                for item in canonical_items
                if item.kind
                in (
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                )
            ],
            [
                ("call-1", StreamItemKind.TOOL_EXECUTION_CANCELLED, None),
                (
                    "call-2",
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                    ToolCallDiagnosticCode.CANCELLED.value,
                ),
            ],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_CANCELLED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_single_confirmation_denial_records_diagnostic(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []
        confirmed: list[str] = []

        async def tracked(name: str) -> str:
            executed.append(name)
            return name

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id="call-1",
            name="tracked",
            arguments={"name": "first"},
        )

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            yield ToolCallToken(token="", call=call)

        def confirm(call: ToolCall) -> str:
            confirmed.append(str(call.id))
            return "n"

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(confirmed, ["call-1"])
        self.assertEqual(executed, [])
        diagnostics = [
            item
            for item in response.canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].data["code"],
            ToolCallDiagnosticCode.USER_REJECTED.value,
        )
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_parallel_async_confirm_all_allows_fanout(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        active = 0
        peak_active = 0

        async def tracked(name: str, delay: float) -> str:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await sleep(delay)
            active -= 1
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first", "delay": 0.001},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second", "delay": 0.001},
            ),
        ]

        async def confirm(_: ToolCall) -> str:
            return "a"

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertTrue(response._tool_confirm_all)
        self.assertEqual(peak_active, 2)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_parallel_confirm_y_allows_fanout(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        active = 0
        peak_active = 0

        async def tracked(name: str, delay: float) -> str:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await sleep(delay)
            active -= 1
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first", "delay": 0.001},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second", "delay": 0.001},
            ),
        ]

        async def outer_gen() -> AsyncIterator[ToolCallToken]:
            for call in calls:
                yield ToolCallToken(token="", call=call)

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            TextGenerationResponse(
                lambda **_: outer_gen(),
                logger=getLogger(),
                use_async_generator=True,
                generation_settings=GenerationSettings(),
                settings=GenerationSettings(),
            ),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=lambda _: "y",
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(peak_active, 2)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_iteration_parallel_results_emit_before_continuation(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        active = 0
        peak_active = 0

        async def tracked(name: str, delay: float) -> str:
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            await sleep(delay)
            active -= 1
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(
                parallel_tool_calls=True,
                maximum_parallel_tool_calls=2,
            ),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first", "delay": 0.01},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second", "delay": 0.001},
            ),
            ToolCall(
                id="call-3",
                name="tracked",
                arguments={"name": "third", "delay": 0.001},
            ),
        ]

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )
        response.__aiter__()
        for call in calls:
            response._append_canonical_tool_call_ready(call)
            response._put_staging_item(response._calls, call, "tool call")

        items = [
            await response._next_item(),
            await response._next_item(),
            await response._next_item(),
        ]
        model_event = await response._next_item()

        result_events = [
            item
            for item in items
            if isinstance(item, Event) and item.type is EventType.TOOL_RESULT
        ]
        self.assertEqual(len(result_events), 3)
        self.assertEqual(peak_active, 2)
        self.assertEqual(
            [
                event.payload["call"].id
                for event in result_events
                if event.payload is not None
            ],
            ["call-1", "call-2", "call-3"],
        )
        assert isinstance(model_event, Event)
        self.assertEqual(model_event.type, EventType.TOOL_MODEL_RESPONSE)
        agent.assert_awaited_once()
        child_context = agent.await_args.args[0]
        continuation_messages = cast(list[Message], child_context.input)
        self.assertEqual(
            [
                message.tool_calls[0].id
                for message in continuation_messages
                if message.tool_calls
            ],
            ["call-1", "call-2", "call-3"],
        )
        self.assertEqual(
            [
                message.content
                for message in continuation_messages
                if message.role is MessageRole.TOOL
            ],
            ['"first"', '"second"', '"third"'],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in response.canonical_items
                if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
            ],
            ["call-2", "call-1", "call-3"],
        )
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_iteration_parallel_denial_aborts_before_fanout(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []

        async def tracked(name: str) -> str:
            executed.append(name)
            return name

        setattr(tracked, "parallel_safe", True)
        setattr(tracked, "side_effecting", False)
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(parallel_tool_calls=True),
        )
        calls = [
            ToolCall(
                id="call-1",
                name="tracked",
                arguments={"name": "first"},
            ),
            ToolCall(
                id="call-2",
                name="tracked",
                arguments={"name": "second"},
            ),
        ]

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=lambda _: "n",
            enable_tool_parsing=False,
        )
        response.__aiter__()
        for call in calls:
            response._append_canonical_tool_call_ready(call)
            response._put_staging_item(response._calls, call, "tool call")

        with self.assertRaises(CommandAbortException):
            await response._next_item()

        self.assertEqual(executed, [])
        self.assertEqual(
            response.canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )
        validate_tool_lifecycle_items(response.canonical_items)


@dataclass
class Example:
    value: str


def _empty_text_response() -> TextGenerationResponse:
    async def gen() -> AsyncIterator[str]:
        if False:
            yield ""

    return TextGenerationResponse(
        lambda **_: gen(),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )


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

    async def test_to_json_records_canonical_completion(self):
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

        self.assertEqual(await resp.to_json(), '{"value": "ok"}')

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_to_json_records_canonical_error(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        async def gen() -> AsyncIterator[str]:
            raise RuntimeError("boom")
            if False:
                yield ""

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

        with self.assertRaises(RuntimeError):
            await resp.to_json()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )
        self.assertEqual(
            canonical_items[-2].data,
            {"error_type": "RuntimeError", "message": "boom"},
        )


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
        events = [
            call.args[0] for call in event_manager.trigger.await_args_list
        ]
        self.assertFalse(
            any(event.type is EventType.TOOL_DIAGNOSTIC for event in events)
        )

    async def test_to_str_preserves_continuation_text_matching_tool_request(
        self,
    ):
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
        call = ToolCall(id=uuid4(), name="calc", arguments=None)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [call] if text == "call" else None
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
            yield "c"
            yield "a"
            yield "l"
            yield "l"
            yield "b"
            yield "a"
            yield "c"
            yield "k"

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

        self.assertEqual(result, "callback")
        agent.assert_awaited_once()
        tool.assert_awaited_once()
        validate_canonical_stream_items(resp.canonical_items)
        validate_tool_lifecycle_items(resp.canonical_items)

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
            yield ToolCallToken(token="2 + ", call=None)
            yield ToolCallToken(token="2", call=None)
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
        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            [
                item.text_delta
                for item in canonical_items
                if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            ],
            ["2 + ", "2"],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in canonical_items
                if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            ],
            ["call1", "call1"],
        )
        self.assertEqual(
            [item.correlation.tool_call_id for item in canonical_items[1:6]],
            ["call1", "call1", "call1", "call1", "call1"],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )

    async def test_to_str_structured_tool_call_without_event_manager(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "2 + 2"},
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
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await resp.to_str(), "done")
        tool.assert_awaited_once()
        tool.get_calls.assert_called_once_with("done")
        agent.assert_awaited_once()
        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_to_str_continuation_error_without_event_manager(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.side_effect = RuntimeError("model failed")
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            tool=tool,
        )

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data,
            {"error_type": "RuntimeError", "message": "model failed"},
        )

    async def test_to_str_preserves_streamed_structured_tool_arguments(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []

        async def tracked(name: str) -> str:
            executed.append(name)
            return name

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        first_call = ToolCall(
            id="call1",
            name="tracked",
            arguments={"name": "first"},
        )
        second_call = ToolCall(
            id="call2",
            name="tracked",
            arguments={"name": "second"},
        )

        async def outer_gen():
            yield ToolCallToken(token='{"name"', call=first_call)
            yield ToolCallToken(token=':"first"}', call=first_call)
            yield ToolCallToken(token='{"name":"second"}', call=second_call)

        settings = GenerationSettings()
        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=settings,
            settings=settings,
        )
        agent.return_value = _string_response("done", async_gen=True)

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await resp.to_str(), "done")
        self.assertEqual(executed, ["first", "second"])
        canonical_items = resp.canonical_items
        argument_items = [
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        ]
        self.assertEqual(
            [
                (item.correlation.tool_call_id, item.text_delta)
                for item in argument_items
            ],
            [
                ("call1", '{"name"'),
                ("call1", ':"first"}'),
                ("call2", '{"name":"second"}'),
            ],
        )
        completed_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(completed_ids, ["call1", "call2"])
        child_context = agent.await_args.args[0]
        tool_messages = [
            message.content
            for message in cast(list[Message], child_context.input)
            if message.role is MessageRole.TOOL
        ]
        self.assertEqual(tool_messages, ["first", "second"])
        self.assertFalse(
            any(
                item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
                and item.data
                and item.data.get("code")
                == ToolCallDiagnosticCode.REPEATED_CALL.value
                for item in canonical_items
            )
        )
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_to_str_tool_exception_records_error_terminal(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.side_effect = RuntimeError("boom")

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        with self.assertRaises(RuntimeError):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].data,
            {"error_type": "RuntimeError", "message": "boom"},
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.ERRORED,
        )

    async def test_to_str_tool_cancellation_records_cancel_terminal(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.side_effect = CancelledError()

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        with self.assertRaises(CancelledError):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )

    async def test_to_str_tool_manager_cancellation_stops_continuation(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        cancel_during_tool = False

        class CancellingTool(Tool):
            def __init__(self) -> None:
                super().__init__()
                self.__name__ = "cancellable"

            async def __call__(self, context: ToolCallContext) -> str:
                nonlocal cancel_during_tool
                cancel_during_tool = True
                assert context.cancellation_checker is not None
                await context.cancellation_checker()
                return "done"

        call = ToolCall(id="call1", name="cancellable", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[CancellingTool()])],
            enable_tools=["cancellable"],
        )
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        async def checker() -> None:
            if cancel_during_tool:
                raise CancelledError()

        resp.set_cancellation_checker(checker)

        with self.assertRaises(CancelledError):
            await resp.to_str()

        agent.assert_not_awaited()
        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[5].data["code"],
            ToolCallDiagnosticCode.CANCELLED.value,
        )
        self.assertFalse(
            any(
                trigger.args[0].type is EventType.TOOL_MODEL_RUN
                for trigger in event_manager.trigger.await_args_list
            )
        )

    async def test_to_str_model_continuation_exception_records_error_terminal(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.side_effect = RuntimeError("model failed")
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        with self.assertRaises(RuntimeError):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data,
            {"error_type": "RuntimeError", "message": "model failed"},
        )

    async def test_to_str_continuation_stream_error_records_terminal(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def inner_gen() -> AsyncIterator[str]:
            yield "partial"
            raise RuntimeError("stream failed")

        agent.return_value = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[-3].data,
            {"error_type": "RuntimeError", "message": "stream failed"},
        )

    async def test_to_str_model_continuation_cancellation_records_terminal(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.side_effect = CancelledError()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        with self.assertRaises(CancelledError):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_to_str_continuation_stream_cancellation_records_terminal(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        async def inner_gen() -> AsyncIterator[str]:
            yield "partial"
            raise CancelledError()

        agent.return_value = TextGenerationResponse(
            lambda **_: inner_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        with self.assertRaises(CancelledError):
            await resp.to_str()

        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-4:]],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_to_str_cancellation_checker_stops_pending_continuation(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})
        cancel_after_tool = False

        async def outer_gen():
            yield ToolCallToken(token="", call=call)

        outer_response = TextGenerationResponse(
            lambda **_: outer_gen(),
            logger=getLogger(),
            use_async_generator=True,
            generation_settings=GenerationSettings(),
            settings=GenerationSettings(),
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.return_value = None

        async def execute(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolCallResult:
            nonlocal cancel_after_tool
            cancel_after_tool = True
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="4",
            )

        tool.side_effect = execute

        async def checker() -> None:
            if cancel_after_tool:
                raise CancelledError()

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        resp.set_cancellation_checker(checker)

        with self.assertRaises(CancelledError):
            await resp.to_str()

        agent.assert_not_awaited()
        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            [item.kind for item in canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertFalse(
            any(
                trigger.args[0].type is EventType.TOOL_MODEL_RUN
                for trigger in event_manager.trigger.await_args_list
            )
        )

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
        events = [
            call.args[0] for call in event_manager.trigger.await_args_list
        ]
        diagnostic_events = [
            event
            for event in events
            if event.type is EventType.TOOL_DIAGNOSTIC
        ]
        result_events = [
            event for event in events if event.type is EventType.TOOL_RESULT
        ]
        self.assertEqual(len(diagnostic_events), 1)
        self.assertEqual(len(result_events), 1)
        diagnostic_event = diagnostic_events[0]
        diagnostic_payload = diagnostic_event.payload
        result_payload = result_events[0].payload
        assert diagnostic_payload is not None
        assert result_payload is not None
        self.assertEqual(diagnostic_payload["call"], call)
        self.assertIs(diagnostic_payload["diagnostic"], diagnostic)
        self.assertEqual(diagnostic_payload["diagnostics"], [diagnostic])
        self.assertIs(diagnostic_payload["result"], diagnostic)
        self.assertIs(result_payload["result"], diagnostic)

    async def test_to_str_returns_empty_tool_name_diagnostic_to_model(self):
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
            name="",
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
        self.assertEqual(assistant_message.tool_calls[0].name, "tool")
        self.assertEqual(diagnostic_message.role, MessageRole.TOOL)
        self.assertEqual(diagnostic_message.name, "tool")
        self.assertIsNone(diagnostic_message.tool_call_result)
        self.assertIsNone(diagnostic_message.tool_call_error)
        self.assertIsNotNone(diagnostic_message.tool_call_diagnostic)
        diagnostic = diagnostic_message.tool_call_diagnostic
        assert diagnostic is not None
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.MALFORMED_CALL,
        )
        self.assertEqual(
            diagnostic.stage,
            ToolCallDiagnosticStage.RESOLVE,
        )
        self.assertEqual(diagnostic.call_id, "call1")
        self.assertIsNone(diagnostic.requested_name)
        payload = loads(str(diagnostic_message.content))
        self.assertEqual(payload["code"], "tool_call.malformed")
        self.assertNotIn("requested_name", payload)
        validate_tool_lifecycle_items(resp.canonical_items)
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in resp.canonical_items],
        )

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
        diagnostics = [
            item
            for item in resp.canonical_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].data["code"],
            "orchestrator.tool_cycle.non_executed_limit_exceeded",
        )
        self.assertEqual(
            diagnostics[0].data["stage"],
            ToolCallDiagnosticStage.GUARD.value,
        )

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

    async def test_repeated_empty_tool_attempt_returns_guard_diagnostic(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="", arguments={})
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = None

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

        self.assertIsNone(first)
        self.assertIsInstance(second, ToolCallDiagnostic)
        assert isinstance(second, ToolCallDiagnostic)
        self.assertEqual(second.code, ToolCallDiagnosticCode.REPEATED_CALL)
        self.assertEqual(second.stage, ToolCallDiagnosticStage.GUARD)
        self.assertIsNone(second.requested_name)
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
        self.assertEqual(
            resp.canonical_items[-3].data["code"],
            "orchestrator.tool_cycle.empty_observation",
        )
        validate_canonical_stream_items(resp.canonical_items)

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
        self.assertEqual(
            resp.canonical_items[-1].data["code"],
            "orchestrator.tool_cycle.duplicate_observation",
        )
        self.assertEqual(
            resp.canonical_items[-1].data["stage"],
            ToolCallDiagnosticStage.GUARD.value,
        )

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
        self.assertEqual(
            limited.canonical_items[-1].data["code"],
            "orchestrator.tool_cycle.limit_exceeded",
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
