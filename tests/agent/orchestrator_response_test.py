from asyncio import CancelledError, create_task, sleep, wait_for
from collections.abc import AsyncIterator
from dataclasses import dataclass
from io import StringIO
from json import dumps, loads
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
    _ToolExecutionOutcome,
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
from avalan.model.response.parsers.tool import ToolCallResponseParser
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    stream_channel_for_kind,
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


def _canonical_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    correlation: StreamItemCorrelation | None = None,
) -> CanonicalStreamItem:
    outcomes = {
        StreamItemKind.STREAM_COMPLETED: StreamTerminalOutcome.COMPLETED,
        StreamItemKind.STREAM_ERRORED: StreamTerminalOutcome.ERRORED,
        StreamItemKind.STREAM_CANCELLED: StreamTerminalOutcome.CANCELLED,
    }
    return CanonicalStreamItem(
        stream_session_id="provider-stream",
        run_id="provider-run",
        turn_id="provider-turn",
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        text_delta=text_delta,
        data=cast(Any, data),
        usage=cast(Any, usage),
        terminal_outcome=terminal_outcome or outcomes.get(kind),
        correlation=correlation or StreamItemCorrelation(),
    )


def _canonical_answer_items(
    *text_deltas: str,
    usage: object | None = None,
) -> tuple[CanonicalStreamItem, ...]:
    items = [_canonical_item(StreamItemKind.STREAM_STARTED, 0)]
    sequence = 1
    for text_delta in text_deltas:
        items.append(
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                sequence,
                text_delta=text_delta,
            )
        )
        sequence += 1
    if text_deltas:
        items.append(_canonical_item(StreamItemKind.ANSWER_DONE, sequence))
        sequence += 1
    items.append(
        _canonical_item(
            StreamItemKind.STREAM_COMPLETED,
            sequence,
            usage=usage or {},
        )
    )
    sequence += 1
    items.append(_canonical_item(StreamItemKind.STREAM_CLOSED, sequence))
    return tuple(items)


def _canonical_answer_items_with_usage_completed(
    *text_deltas: str,
    usage: object,
) -> tuple[CanonicalStreamItem, ...]:
    items = [_canonical_item(StreamItemKind.STREAM_STARTED, 0)]
    sequence = 1
    for text_delta in text_deltas:
        items.append(
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                sequence,
                text_delta=text_delta,
            )
        )
        sequence += 1
    if text_deltas:
        items.append(_canonical_item(StreamItemKind.ANSWER_DONE, sequence))
        sequence += 1
    items.append(
        _canonical_item(
            StreamItemKind.USAGE_COMPLETED,
            sequence,
            usage=usage,
        )
    )
    sequence += 1
    items.append(_canonical_item(StreamItemKind.STREAM_COMPLETED, sequence))
    sequence += 1
    items.append(_canonical_item(StreamItemKind.STREAM_CLOSED, sequence))
    return tuple(items)


def _canonical_tool_call_items(
    call: ToolCall,
    *argument_deltas: str,
    usage: object | None = None,
    tool_call_id: str | None = None,
) -> tuple[CanonicalStreamItem, ...]:
    resolved_tool_call_id = tool_call_id or str(call.id)
    correlation = StreamItemCorrelation(tool_call_id=resolved_tool_call_id)
    deltas = (
        argument_deltas
        if argument_deltas or call.arguments is None
        else (dumps(call.arguments),)
    )
    items = [_canonical_item(StreamItemKind.STREAM_STARTED, 0)]
    sequence = 1
    for text_delta in deltas:
        items.append(
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                sequence,
                text_delta=text_delta,
                correlation=correlation,
            )
        )
        sequence += 1
    items.append(
        _canonical_item(
            StreamItemKind.TOOL_CALL_READY,
            sequence,
            data={"name": call.name, "arguments": call.arguments or {}},
            correlation=correlation,
        )
    )
    sequence += 1
    items.append(
        _canonical_item(
            StreamItemKind.TOOL_CALL_DONE,
            sequence,
            correlation=correlation,
        )
    )
    sequence += 1
    items.append(
        _canonical_item(
            StreamItemKind.STREAM_COMPLETED,
            sequence,
            usage=usage or {},
        )
    )
    sequence += 1
    items.append(_canonical_item(StreamItemKind.STREAM_CLOSED, sequence))
    return tuple(items)


def _canonical_tool_call_stream_items(
    *chunks: ToolCall | tuple[str, ToolCall],
    usage: object | None = None,
) -> tuple[CanonicalStreamItem, ...]:
    items = [_canonical_item(StreamItemKind.STREAM_STARTED, 0)]
    sequence = 1
    anonymous_ids: dict[int, str] = {}
    anonymous_index = 0
    active_call: ToolCall | None = None
    active_tool_call_id: str | None = None
    active_has_delta = False

    def resolved_id(call: ToolCall) -> str:
        nonlocal anonymous_index
        if call.id is not None:
            return str(call.id)
        key = id(call)
        tool_call_id = anonymous_ids.get(key)
        if tool_call_id is None:
            anonymous_index += 1
            tool_call_id = f"orchestrator-tool-call-{anonymous_index}"
            anonymous_ids[key] = tool_call_id
        return tool_call_id

    def anonymous_fragment_continues(call: ToolCall) -> bool:
        if (
            active_call is None
            or active_call.id is not None
            or call.id is not None
            or active_tool_call_id is None
            or active_call.name != call.name
            or not isinstance(active_call.arguments, dict)
            or not isinstance(call.arguments, dict)
        ):
            return False
        for key, value in active_call.arguments.items():
            next_value = call.arguments.get(key)
            if (
                isinstance(value, str)
                and isinstance(next_value, str)
                and next_value.startswith(value)
            ):
                continue
            if next_value == value:
                continue
            return False
        return True

    def append_ready_done() -> None:
        nonlocal active_call, active_tool_call_id, sequence
        nonlocal active_has_delta
        assert active_call is not None
        assert active_tool_call_id is not None
        correlation = StreamItemCorrelation(tool_call_id=active_tool_call_id)
        if not active_has_delta and active_call.arguments is not None:
            items.append(
                _canonical_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    sequence,
                    text_delta=dumps(active_call.arguments),
                    correlation=correlation,
                )
            )
            sequence += 1
        items.append(
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                sequence,
                data={
                    "name": active_call.name,
                    "arguments": active_call.arguments or {},
                },
                correlation=correlation,
            )
        )
        sequence += 1
        items.append(
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                sequence,
                correlation=correlation,
            )
        )
        sequence += 1
        active_call = None
        active_tool_call_id = None
        active_has_delta = False

    for chunk in chunks:
        if isinstance(chunk, ToolCall):
            text_delta = ""
            call = chunk
        else:
            text_delta, call = chunk
        tool_call_id = (
            active_tool_call_id
            if text_delta and anonymous_fragment_continues(call)
            else resolved_id(call)
        )
        assert tool_call_id is not None
        if (
            active_tool_call_id is not None
            and active_tool_call_id != tool_call_id
        ):
            append_ready_done()
        active_call = call
        active_tool_call_id = tool_call_id
        if text_delta:
            items.append(
                _canonical_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    sequence,
                    text_delta=text_delta,
                    correlation=StreamItemCorrelation(
                        tool_call_id=tool_call_id
                    ),
                )
            )
            sequence += 1
            active_has_delta = True

    if active_tool_call_id is not None:
        append_ready_done()

    items.append(
        _canonical_item(
            StreamItemKind.STREAM_COMPLETED,
            sequence,
            usage=usage or {},
        )
    )
    sequence += 1
    items.append(_canonical_item(StreamItemKind.STREAM_CLOSED, sequence))
    return tuple(items)


def _tool_call_response(
    *chunks: ToolCall | tuple[str, ToolCall],
) -> TextGenerationResponse:
    return _response_from_items(*_canonical_tool_call_stream_items(*chunks))


def _partial_answer_exception_response(
    text_delta: str,
    exception: BaseException,
) -> TextGenerationResponse:
    async def gen() -> AsyncIterator[CanonicalStreamItem]:
        yield _canonical_item(StreamItemKind.STREAM_STARTED, 0)
        yield _canonical_item(
            StreamItemKind.ANSWER_DELTA,
            1,
            text_delta=text_delta,
        )
        raise exception

    return TextGenerationResponse(
        lambda **_: gen(),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )


def _response_from_items(
    *items: CanonicalStreamItem,
    settings: GenerationSettings | None = None,
) -> TextGenerationResponse:
    async def gen() -> AsyncIterator[CanonicalStreamItem]:
        for item in items:
            yield item

    resolved_settings = settings or GenerationSettings()
    return TextGenerationResponse(
        lambda **_: gen(),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=resolved_settings,
        settings=resolved_settings,
    )


async def _collect_stream_items(
    response: OrchestratorResponse,
) -> list[CanonicalStreamItem]:
    items: list[CanonicalStreamItem] = []
    iterator = response.__aiter__()
    while True:
        try:
            items.append(await wait_for(iterator.__anext__(), 1))
        except StopAsyncIteration:
            return items


async def _drain_until_exception(
    iterator: OrchestratorResponse,
    exception_type: type[BaseException],
) -> list[CanonicalStreamItem]:
    items: list[CanonicalStreamItem] = []
    while True:
        try:
            items.append(await wait_for(iterator.__anext__(), 1))
        except exception_type:
            return items
        except StopAsyncIteration as exc:
            raise AssertionError(
                f"{exception_type.__name__} was not raised."
            ) from exc


def _answer_text(items: list[CanonicalStreamItem]) -> str:
    return "".join(
        item.text_delta or ""
        for item in items
        if item.kind is StreamItemKind.ANSWER_DELTA
    )


def _dummy_response(async_gen=True):
    async def output_gen():
        for item in _canonical_answer_items("a", "b"):
            yield item

    def output_fn(**_):
        return output_gen()

    settings = GenerationSettings()
    response = TextGenerationResponse(
        output_fn,
        logger=getLogger(),
        use_async_generator=async_gen,
        generation_settings=settings,
        settings=settings,
    )
    return response


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
    return _response_from_items(
        *_canonical_answer_items("X", "Y", "Z"),
        settings=GenerationSettings(
            reasoning=ReasoningSettings(enabled=False)
        ),
    )


class _SkippingToolParser:
    async def push(self, _: str) -> list[StreamProviderEvent]:
        return []

    async def flush(self) -> list[StreamProviderEvent]:
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

    async def push(self, _: str) -> list[object]:
        return []

    async def flush(self) -> list[object]:
        if self._flushed:
            return []
        self._flushed = True
        return [
            _canonical_item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                0,
                data={"code": "parser.flush"},
            ),
        ]


class _UnsupportedFlushTokenParser:
    async def push(self, _: str) -> list[object]:
        return []

    async def flush(self) -> list[object]:
        return [Token(id=9, token="x")]


class _PushDiagnosticParser:
    async def push(self, _: str) -> list[StreamProviderEvent]:
        return [
            StreamProviderEvent(
                kind=StreamItemKind.STREAM_DIAGNOSTIC,
                data={"event_type": EventType.TOOL_DIAGNOSTIC.value},
            )
        ]

    async def flush(self) -> list[StreamProviderEvent]:
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

        items = await _collect_stream_items(resp)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(_answer_text(items), "ab")
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
                "token_type": "CanonicalStreamItem",
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
                "token_id": 42,
                "token_type": "CanonicalStreamItem",
                "model_id": "m",
                "token": "b",
                "step": 1,
            },
        )
        self.assertEqual(
            token_events[1].observability.data["summary"],
            {"text_delta_length": 1},
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

    async def test_consumer_projections_drain_error_terminal_items(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _partial_answer_exception_response("a", RuntimeError("boom")),
            agent,
            operation,
            {},
        )
        projections = []

        with self.assertRaisesRegex(RuntimeError, "boom"):
            async for projection in resp.consumer_projections(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
            ):
                projections.append(projection)

        self.assertEqual(
            [projection.kind for projection in projections],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_consumer_projections_drain_cancel_terminal_items(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _partial_answer_exception_response("a", CancelledError()),
            agent,
            operation,
            {},
        )
        projections = []

        with self.assertRaises(CancelledError):
            async for projection in resp.consumer_projections(
                stream_session_id="stream",
                run_id="run",
                turn_id="turn",
            ):
                projections.append(projection)

        self.assertEqual(
            [projection.kind for projection in projections],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    async def test_token_event_uses_canonical_item_sequence(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        item = _canonical_item(
            StreamItemKind.ANSWER_DELTA,
            1,
            text_delta="same",
        )

        await resp._emit_token_generated_event(item)

        event = event_manager.trigger.await_args.args[0]
        self.assertEqual(event.observability.data["sequence"], 1)
        self.assertEqual(
            event.observability.data["kind"],
            StreamItemKind.ANSWER_DELTA.value,
        )

    async def test_to_str_consumes_canonical_answer_without_token_events(self):
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
            [item.kind for item in resp.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
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

        await resp._emit_token_generated_event(
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                1,
                text_delta="a",
            )
        )

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

        items = await _collect_stream_items(resp)
        self.assertEqual(_answer_text(items), "ab")
        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
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

        settings = GenerationSettings()
        response = _response_from_items(
            *_canonical_answer_items(
                "<|start|>assistant",
                "<|channel|>commentary to=mytool <|message|>{}<|call|>",
            ),
            settings=settings,
        )

        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        tool_manager = AsyncMock(spec=ToolManager)
        tool_manager.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        tool_manager.tool_call_status.side_effect = (
            base_parser.tool_call_status
        )
        tool_manager.get_calls.side_effect = base_parser
        tool_manager.tool_format = ToolFormat.HARMONY
        tool_manager.is_empty = False
        tool_manager.return_value = None

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool_manager,
        )

        items = await _collect_stream_items(resp)

        self.assertEqual(_answer_text(items), "")
        self.assertIn(
            StreamItemKind.TOOL_CALL_READY,
            [item.kind for item in items],
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "mytool", "arguments": {}})
        validate_canonical_stream_items(resp.canonical_items)

    async def test_harmony_streaming_emits_flush_tool_event(self) -> None:
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [1]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        settings = GenerationSettings()
        response = _response_from_items(
            *_canonical_answer_items(
                "<|start|>assistant<|channel|>commentary "
                "to=functions.browser.open <|constrain|>json<|message|>"
                '{"url":"https://example.com"}'
            ),
            settings=settings,
        )

        base_parser = ToolCallParser(tool_format=ToolFormat.HARMONY)
        tool_manager = AsyncMock(spec=ToolManager)
        tool_manager.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        tool_manager.tool_call_status.side_effect = (
            base_parser.tool_call_status
        )
        tool_manager.get_calls.side_effect = base_parser
        tool_manager.tool_format = ToolFormat.HARMONY
        tool_manager.is_empty = False
        tool_manager.return_value = None

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool_manager,
        )

        items = await _collect_stream_items(resp)

        self.assertEqual(_answer_text(items), "")
        self.assertIn(
            StreamItemKind.TOOL_CALL_READY,
            [item.kind for item in items],
        )
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            ready.data,
            {
                "name": "browser.open",
                "arguments": {"url": "https://example.com"},
            },
        )
        validate_canonical_stream_items(resp.canonical_items)

    async def test_real_tool_parser_text_tool_call_stays_canonical(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        tool_text = (
            '<tool_call>{"name": "calc", "arguments": {"x": 1}}</tool_call>'
        )
        response = _response_from_items(*_canonical_answer_items(tool_text))
        base_parser = ToolCallParser()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        tool.tool_call_status.side_effect = base_parser.tool_call_status
        tool.get_calls.side_effect = base_parser
        tool.tool_format = None

        async def execute(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolCallResult:
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={"ok": True},
            )

        tool.side_effect = execute
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            tool=tool,
        )

        items = await _collect_stream_items(resp)

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertFalse(any(isinstance(item, Event) for item in items))
        kinds = [item.kind for item in items]
        self.assertIn(StreamItemKind.TOOL_CALL_ARGUMENT_DELTA, kinds)
        self.assertIn(StreamItemKind.TOOL_CALL_READY, kinds)
        self.assertIn(StreamItemKind.TOOL_EXECUTION_COMPLETED, kinds)
        self.assertIn(StreamItemKind.MODEL_CONTINUATION_STARTED, kinds)
        self.assertEqual(_answer_text(items), "done")
        self.assertNotIn("<tool_call", _answer_text(items))
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "calc", "arguments": {"x": 1}})
        executed_call = tool.await_args.args[0]
        self.assertEqual(executed_call.name, "calc")
        self.assertEqual(executed_call.arguments, {"x": 1})
        agent.assert_awaited_once()
        validate_canonical_stream_items(resp.canonical_items)
        validate_tool_lifecycle_items(resp.canonical_items)


class OrchestratorResponseCanonicalLifecycleTestCase(IsolatedAsyncioTestCase):
    async def test_iteration_skips_parser_empty_chunks_without_recursion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        response = _response_from_items(
            *_canonical_answer_items(*("{" for _ in range(1200)))
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

        items = await _collect_stream_items(orchestrated)

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in items
                    if item.kind is StreamItemKind.ANSWER_DELTA
                ]
            ),
            1200,
        )
        self.assertEqual(orchestrated._step, 1200)

    async def test_parser_tool_process_item_is_rejected(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})

        response = _response_from_items(*_canonical_answer_items("x"))
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _ToolProcessOnlyParser(call)
        )
        iterator = orchestrated.__aiter__()

        items = await _drain_until_exception(iterator, StreamValidationError)

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertIn(
            StreamItemKind.ANSWER_DELTA,
            [item.kind for item in orchestrated.canonical_items],
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

        response = _tool_call_response(call)
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

        items = await _collect_stream_items(orchestrated)

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(_answer_text(items), "done")
        tool.assert_awaited_once()
        agent.assert_awaited_once()
        context = agent.await_args.args[0]
        assert isinstance(context.input, list)
        self.assertEqual(context.input[-1].content, '"4"')
        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            canonical_items[0].kind, StreamItemKind.STREAM_STARTED
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_COMPLETED, StreamItemKind.STREAM_CLOSED],
        )
        self.assertIn(
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            [item.kind for item in canonical_items],
        )
        self.assertIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )

    async def test_non_tool_usage_completed_is_terminal_usage_only(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        usage = {
            "input_tokens": 1,
            "output_tokens": 2,
            "total_tokens": 3,
        }
        response = _response_from_items(
            *_canonical_answer_items_with_usage_completed(
                "done",
                usage=usage,
            )
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        items = await _collect_stream_items(orchestrated)

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        kinds = [item.kind for item in items]
        self.assertNotIn(StreamItemKind.USAGE_COMPLETED, kinds)
        terminal_item = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_COMPLETED
        )
        self.assertIsNotNone(terminal_item.usage)
        assert terminal_item.usage is not None
        self.assertEqual(
            terminal_item.usage["totals"]["input_tokens"],
            usage["input_tokens"],
        )
        self.assertEqual(
            terminal_item.usage["totals"]["output_tokens"],
            usage["output_tokens"],
        )
        self.assertEqual(
            terminal_item.usage["totals"]["total_tokens"],
            usage["total_tokens"],
        )
        validate_canonical_stream_items(orchestrated.canonical_items)

    async def test_inner_usage_completed_waits_for_outer_completion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _response_from_items(
            *_canonical_answer_items_with_usage_completed(
                "done",
                usage={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
            )
        )
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = _response_from_items(
            _canonical_item(StreamItemKind.STREAM_STARTED, 0),
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta=dumps(call.arguments),
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                2,
                data={"name": call.name, "arguments": call.arguments},
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                3,
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.USAGE_COMPLETED,
                4,
                usage={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            ),
            _canonical_item(StreamItemKind.STREAM_COMPLETED, 5),
            _canonical_item(StreamItemKind.STREAM_CLOSED, 6),
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

        items = await _collect_stream_items(orchestrated)

        kinds = [item.kind for item in items]
        self.assertNotIn(StreamItemKind.USAGE_COMPLETED, kinds)
        self.assertLess(
            kinds.index(StreamItemKind.TOOL_EXECUTION_COMPLETED),
            kinds.index(StreamItemKind.STREAM_COMPLETED),
        )
        validate_canonical_stream_items(orchestrated.canonical_items)
        terminal_usage = orchestrated.canonical_items[-2].usage
        self.assertIsNotNone(terminal_usage)
        assert terminal_usage is not None
        self.assertEqual(terminal_usage["totals"]["input_tokens"], 3)
        self.assertEqual(terminal_usage["totals"]["output_tokens"], 4)
        self.assertEqual(terminal_usage["totals"]["total_tokens"], 7)

    async def test_inner_done_items_wait_for_outer_completion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        correlation = StreamItemCorrelation(tool_call_id="call1")
        agent.return_value = _response_from_items(
            _canonical_item(StreamItemKind.STREAM_STARTED, 0),
            _canonical_item(
                StreamItemKind.REASONING_DELTA,
                1,
                text_delta="inner reasoning",
            ),
            _canonical_item(StreamItemKind.REASONING_DONE, 2),
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                3,
                text_delta="done",
            ),
            _canonical_item(StreamItemKind.ANSWER_DONE, 4),
            _canonical_item(
                StreamItemKind.USAGE_COMPLETED,
                5,
                usage={
                    "input_tokens": 4,
                    "output_tokens": 5,
                    "total_tokens": 9,
                },
            ),
            _canonical_item(StreamItemKind.STREAM_COMPLETED, 6),
            _canonical_item(StreamItemKind.STREAM_CLOSED, 7),
        )
        response = _response_from_items(
            _canonical_item(StreamItemKind.STREAM_STARTED, 0),
            _canonical_item(
                StreamItemKind.REASONING_DELTA,
                1,
                text_delta="outer reasoning",
            ),
            _canonical_item(StreamItemKind.REASONING_DONE, 2),
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                3,
                text_delta="call",
            ),
            _canonical_item(StreamItemKind.ANSWER_DONE, 4),
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                5,
                text_delta=dumps(call.arguments),
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                6,
                data={"name": call.name, "arguments": call.arguments},
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                7,
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.USAGE_COMPLETED,
                8,
                usage={
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "total_tokens": 3,
                },
            ),
            _canonical_item(StreamItemKind.STREAM_COMPLETED, 9),
            _canonical_item(StreamItemKind.STREAM_CLOSED, 10),
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

        items = await _collect_stream_items(orchestrated)

        kinds = [item.kind for item in items]
        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertNotIn(StreamItemKind.USAGE_COMPLETED, kinds)
        self.assertEqual(kinds.count(StreamItemKind.ANSWER_DONE), 1)
        self.assertEqual(kinds.count(StreamItemKind.REASONING_DONE), 1)
        self.assertEqual(kinds.count(StreamItemKind.STREAM_COMPLETED), 1)
        self.assertEqual(
            kinds[-2:],
            [StreamItemKind.STREAM_COMPLETED, StreamItemKind.STREAM_CLOSED],
        )
        self.assertLess(
            kinds.index(StreamItemKind.TOOL_EXECUTION_COMPLETED),
            kinds.index(StreamItemKind.STREAM_COMPLETED),
        )
        self.assertLess(
            kinds.index(StreamItemKind.MODEL_CONTINUATION_COMPLETED),
            kinds.index(StreamItemKind.STREAM_COMPLETED),
        )
        self.assertLess(
            kinds.index(StreamItemKind.MODEL_CONTINUATION_COMPLETED),
            kinds.index(StreamItemKind.ANSWER_DONE),
        )
        self.assertLess(
            kinds.index(StreamItemKind.MODEL_CONTINUATION_COMPLETED),
            kinds.index(StreamItemKind.REASONING_DONE),
        )
        validate_canonical_stream_items(orchestrated.canonical_items)
        validate_tool_lifecycle_items(orchestrated.canonical_items)
        terminal_usage = orchestrated.canonical_items[-2].usage
        self.assertIsNotNone(terminal_usage)
        assert terminal_usage is not None
        self.assertEqual(terminal_usage["totals"]["input_tokens"], 5)
        self.assertEqual(terminal_usage["totals"]["output_tokens"], 7)
        self.assertEqual(terminal_usage["totals"]["total_tokens"], 12)

    async def test_parser_flush_items_are_returned_before_completion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        response = _response_from_items(*_canonical_answer_items("{"))
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

        items = await _collect_stream_items(orchestrated)

        self.assertEqual(_answer_text(items), "{")
        self.assertIn(
            StreamItemKind.STREAM_DIAGNOSTIC,
            [item.kind for item in items],
        )
        validate_canonical_stream_items(orchestrated.canonical_items)

    async def test_parser_flush_rejects_non_process_control_event(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        response = _response_from_items(*_canonical_answer_items("{"))
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        orchestrated._tool_parser = cast(
            ToolCallResponseParser, _UnsupportedFlushTokenParser()
        )

        iterator = orchestrated.__aiter__()
        await _drain_until_exception(iterator, StreamValidationError)
        self.assertEqual(
            [item.kind for item in orchestrated.canonical_items[-2:]],
            [StreamItemKind.STREAM_ERRORED, StreamItemKind.STREAM_CLOSED],
        )
        validate_canonical_stream_items(orchestrated.canonical_items)

    async def test_parser_diagnostic_event_appends_canonical_payload(
        self,
    ) -> None:
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

        await orchestrated._process_canonical_response_item(
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                0,
                text_delta="{",
            )
        )

        self.assertEqual(
            [item.kind for item in orchestrated.canonical_items[-1:]],
            [StreamItemKind.STREAM_DIAGNOSTIC],
        )
        self.assertEqual(
            orchestrated.canonical_items[-1].data,
            {"event_type": EventType.TOOL_DIAGNOSTIC.value},
        )

    async def test_provider_diagnostic_event_appends_canonical_payload(
        self,
    ) -> None:
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
        provider_event = StreamProviderEvent(
            kind=StreamItemKind.STREAM_DIAGNOSTIC,
            data={
                "code": "tool_call.malformed",
                "message": "Malformed tool call.",
                "tool_call_id": "parser-tool-call-1",
                "diagnostics": [
                    {
                        "id": "diagnostic-1",
                        "code": "tool_call.malformed",
                        "stage": "parse",
                        "status": "non_executed",
                        "message": "Malformed tool call.",
                        "retryable": False,
                        "details": {"raw": "bad"},
                        "started_at": "2026-01-01T00:00:00+00:00",
                        "finished_at": "2026-01-01T00:00:01+00:00",
                    }
                ],
            },
            correlation=StreamItemCorrelation(
                tool_call_id="parser-tool-call-1"
            ),
            visibility=StreamVisibility.DIAGNOSTIC,
        )

        orchestrated._queue_parser_output(provider_event)

        item = orchestrated.canonical_items[-1]
        self.assertIs(item.kind, StreamItemKind.STREAM_DIAGNOSTIC)
        self.assertEqual(item.correlation.tool_call_id, "parser-tool-call-1")
        self.assertEqual(item.data, provider_event.data)

    async def test_iteration_records_tool_and_continuation_lifecycle(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _empty_text_response()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "2 + 2"},
        )

        outer_response = _tool_call_response(call)

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        inner_response = _string_response("4", async_gen=True)
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

        items = await _collect_stream_items(response)

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )
        self.assertEqual(_answer_text(items), "4")
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(
            canonical_items[0].kind, StreamItemKind.STREAM_STARTED
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_COMPLETED, StreamItemKind.STREAM_CLOSED],
        )
        self.assertIn(
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            [item.kind for item in canonical_items],
        )
        self.assertIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.COMPLETED,
        )
        self.assertEqual(
            canonical_items[1].text_delta, '{"expression": "2 + 2"}'
        )
        self.assertEqual(
            [
                item.correlation.model_continuation_id
                for item in canonical_items
                if item.kind
                in {
                    StreamItemKind.MODEL_CONTINUATION_STARTED,
                    StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                }
            ],
            [
                canonical_items[6].correlation.model_continuation_id,
                canonical_items[6].correlation.model_continuation_id,
            ],
        )

    async def test_iteration_drains_multi_tool_continuation_before_next_model(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _empty_text_response()
        operation = _dummy_operation()
        calls = [
            ToolCall(
                id="call1",
                name="calc",
                arguments={"expression": "4 + 6"},
            ),
            ToolCall(
                id="call2",
                name="calc",
                arguments={"expression": "10 * 5"},
            ),
            ToolCall(
                id="call3",
                name="calc",
                arguments={"expression": "50 / 2"},
            ),
        ]

        outer_response = _tool_call_response(calls[0])

        inner_response = _tool_call_response(calls[1], calls[2])
        final_response = _string_response("25", async_gen=True)
        agent.side_effect = [inner_response, final_response]

        async def execute(
            call: ToolCall,
            _context: ToolCallContext,
            **_kwargs: Any,
        ) -> ToolCallResult:
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=str(call.arguments),
            )

        tool = AsyncMock(spec=ToolManager, side_effect=execute)
        tool.is_empty = False

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        items = await _collect_stream_items(response)

        self.assertEqual(agent.await_count, 2)
        self.assertEqual(tool.await_count, 3)
        self.assertEqual(_answer_text(items), "25")
        canonical_items = response.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        continuation_items = [
            item
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
            }
        ]
        self.assertEqual(
            [item.kind for item in continuation_items],
            [
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            ],
        )

    async def test_iteration_records_live_tool_output_before_completion(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _empty_text_response()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(
            id="call1",
            name="shell.cat",
            arguments={"path": "file.txt"},
        )

        outer_response = _tool_call_response(call)

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

        agent.return_value = _empty_text_response()

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

        items = await _collect_stream_items(response)

        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
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
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "2 + 2"},
        )

        outer_response = _tool_call_response(
            ('{"expression"', call),
            (':"2 + 2"}', call),
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

        inner_response = _string_response("4", async_gen=True)
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

        items = await _collect_stream_items(response)

        self.assertEqual(_answer_text(items), "4")
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

    async def test_queue_parser_output_rejects_legacy_tool_call_token(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _dummy_response(),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        response.__aiter__()

        with self.assertRaises(StreamValidationError):
            response._queue_parser_output(
                ToolCallToken(token='{"x":1}', call=call)
            )
        self.assertTrue(response._pending_tool_call_ready_items.empty())

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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, RuntimeError)

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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, CancelledError)

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
        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, CancelledError)

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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, CommandAbortException)

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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, CommandAbortException)

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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, RuntimeError)

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
        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, CancelledError)

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
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})

        response = _tool_call_response(call)
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
        await _collect_stream_items(orchestrated)

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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, RuntimeError)

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

        response = _tool_call_response(call)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        agent.return_value = _partial_answer_exception_response(
            "partial",
            RuntimeError("stream failed"),
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

        await _drain_until_exception(iterator, RuntimeError)

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            }
        ]
        self.assertEqual(
            terminal_kinds[-3:],
            [
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_item = next(
            item
            for item in canonical_items
            if item.kind is StreamItemKind.MODEL_CONTINUATION_ERROR
        )
        self.assertEqual(
            error_item.data,
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

        response = _tool_call_response(call)
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

        await _drain_until_exception(iterator, CancelledError)

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            }
        ]
        self.assertEqual(
            terminal_kinds[-3:],
            [
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

        response = _tool_call_response(call)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="4",
        )

        agent.return_value = _partial_answer_exception_response(
            "partial",
            CancelledError(),
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

        await _drain_until_exception(iterator, CancelledError)

        canonical_items = orchestrated.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertNotIn(
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            }
        ]
        self.assertEqual(
            terminal_kinds[-3:],
            [
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

        await _drain_until_exception(
            orchestrated.__aiter__(),
            CancelledError,
        )

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

        await _drain_until_exception(orchestrated.__aiter__(), ValueError)

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
        await _collect_stream_items(response)
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*fragments),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*chunks),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*fragments),
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

        def confirm(call: ToolCall) -> str:
            confirmed.append(str(call.id))
            return "n"

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        def confirm(call: ToolCall) -> str:
            confirmed.append(str(call.id))
            return "n"

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(*calls),
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

        while True:
            try:
                await wait_for(response.__anext__(), 1)
            except StopAsyncIteration:
                break

        self.assertEqual(peak_active, 2)
        events = [
            call.args[0] for call in event_manager.trigger.await_args_list
        ]
        result_positions = [
            index
            for index, event in enumerate(events)
            if event.type is EventType.TOOL_RESULT
        ]
        model_response_positions = [
            index
            for index, event in enumerate(events)
            if event.type is EventType.TOOL_MODEL_RESPONSE
        ]
        result_events = [
            event for event in events if event.type is EventType.TOOL_RESULT
        ]
        self.assertEqual(len(result_events), 3)
        self.assertEqual(len(model_response_positions), 1)
        self.assertLess(max(result_positions), model_response_positions[0])
        self.assertEqual(
            [
                event.payload["call"].id
                for event in result_events
                if event.payload is not None
            ],
            ["call-2", "call-1", "call-3"],
        )
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

        await _drain_until_exception(response, CommandAbortException)

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
    return _response_from_items(*_canonical_answer_items())


def _string_response(text: str, *, async_gen: bool = False, inputs=None):
    def output_fn(*args, **kwargs):
        if async_gen:

            async def gen() -> AsyncIterator[CanonicalStreamItem]:
                for item in _canonical_answer_items(*text):
                    yield item

            return gen()
        return text

    response = TextGenerationResponse(
        output_fn,
        logger=getLogger(),
        use_async_generator=async_gen,
        inputs=inputs or {"input_ids": [[1, 2, 3]]},
        generation_settings=GenerationSettings(),
        settings=GenerationSettings(),
    )
    return response


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

        response = _string_response("hi", async_gen=True)

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            response,
            agent,
            operation,
            {},
        )

        items = await _collect_stream_items(resp)

        self.assertEqual(_answer_text(items), "hi")
        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )


class OrchestratorResponseToStrTestCase(IsolatedAsyncioTestCase):
    async def test_to_str_without_tool_call(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        response = _string_response("ok", async_gen=True)

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

        outer_response = _string_response("call", async_gen=True)

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

        inner_response = _string_response("r", async_gen=True)
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

        outer_response = _string_response("call", async_gen=True)
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

        inner_response = _string_response("callback", async_gen=True)
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

    async def test_to_str_posthoc_tool_detection_defers_usage_completed(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        outer_response = _response_from_items(
            *_canonical_answer_items_with_usage_completed(
                "call",
                usage={
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
        )
        inner_response = _response_from_items(
            *_canonical_answer_items_with_usage_completed(
                "done",
                usage={
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "total_tokens": 5,
                },
            )
        )
        agent.return_value = inner_response
        call = ToolCall(id="call-1", name="calc", arguments=None)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [call] if text == "call" else None
        )

        async def tool_exec(call, context: ToolCallContext):
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="2",
            )

        tool.side_effect = tool_exec
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            tool=tool,
        )

        result = await resp.to_str()

        self.assertEqual(result, "done")
        validate_canonical_stream_items(resp.canonical_items)
        validate_tool_lifecycle_items(resp.canonical_items)
        self.assertNotIn(
            StreamItemKind.USAGE_COMPLETED,
            [item.kind for item in resp.canonical_items],
        )
        terminal_usage = resp.canonical_items[-2].usage
        self.assertIsNotNone(terminal_usage)
        assert terminal_usage is not None
        self.assertEqual(terminal_usage["totals"]["input_tokens"], 3)
        self.assertEqual(terminal_usage["totals"]["output_tokens"], 4)
        self.assertEqual(terminal_usage["totals"]["total_tokens"], 7)

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

        outer_response = _tool_call_response(
            ("2 + ", call),
            ("2", call),
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

        inner_response = _string_response("4", async_gen=True)
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
        tool.get_calls.assert_any_call("4")
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
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                StreamItemKind.ANSWER_DONE,
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

        outer_response = _tool_call_response(call)
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
        tool.get_calls.assert_any_call("done")
        agent.assert_awaited_once()
        canonical_items = resp.canonical_items
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)
        self.assertEqual(_answer_text(list(canonical_items)), "done")
        self.assertEqual(
            canonical_items[0].kind, StreamItemKind.STREAM_STARTED
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_COMPLETED, StreamItemKind.STREAM_CLOSED],
        )

    async def test_to_str_continuation_error_without_event_manager(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.side_effect = RuntimeError("model failed")
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={})

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(
            ('{"name"', first_call),
            (':"first"}', first_call),
            ('{"name":"second"}', second_call),
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
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

        agent.return_value = _partial_answer_exception_response(
            "partial",
            RuntimeError("stream failed"),
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
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            }
        ]
        self.assertEqual(
            terminal_kinds[-3:],
            [
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        error_item = next(
            item
            for item in canonical_items
            if item.kind is StreamItemKind.MODEL_CONTINUATION_ERROR
        )
        self.assertEqual(
            error_item.data,
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

        outer_response = _tool_call_response(call)
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
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            }
        ]
        self.assertEqual(
            terminal_kinds[-3:],
            [
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

        outer_response = _tool_call_response(call)
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

        agent.return_value = _partial_answer_exception_response(
            "partial",
            CancelledError(),
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
        terminal_kinds = [
            item.kind
            for item in canonical_items
            if item.kind
            in {
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            }
        ]
        self.assertEqual(
            terminal_kinds[-3:],
            [
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
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

        outer_response = _tool_call_response(call)
        inner_response = _tool_call_response(call)
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
            _empty_text_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )
        resp.__aiter__()
        resp._tool_result_outcomes.put(
            _ToolExecutionOutcome(
                call=ToolCall(id="call-1", name="tool", arguments=None),
                context=ToolCallContext(),
                event=Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": None},
                ),
                planned_index=0,
                result=None,
            )
        )

        while True:
            try:
                await wait_for(resp.__anext__(), 1)
            except StopAsyncIteration:
                break

        agent.assert_not_awaited()
        self.assertTrue(resp._finished)
        self.assertIn(
            EventType.END,
            [
                call.args[0].type
                for call in event_manager.trigger.await_args_list
            ],
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

        outer_response = _tool_call_response(("<tool_call />", first_call))
        first_inner_response = _tool_call_response(
            (
                "<tool_call />",
                second_call,
            )
        )
        second_inner_response = _string_response("done", async_gen=True)
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

        items = await _collect_stream_items(resp)

        self.assertEqual(_answer_text(items), "done")
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
    async def test_complex_canonical_response_is_public_canonical(self):
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

        items = await _collect_stream_items(resp)

        self.assertEqual(
            len(
                [
                    item
                    for item in items
                    if item.kind is StreamItemKind.REASONING_DELTA
                ]
            ),
            0,
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in items
                    if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
                ]
            ),
            0,
        )
        self.assertEqual(
            len(
                [
                    item
                    for item in items
                    if item.kind is StreamItemKind.ANSWER_DELTA
                ]
            ),
            3,
        )
        self.assertTrue(
            all(isinstance(item, CanonicalStreamItem) for item in items)
        )

    async def test_queue_parser_output_rejects_legacy_outputs(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
        )

        legacy_outputs = (
            "legacy",
            Token(token="legacy"),
            TokenDetail(id=1, token="legacy", probability=0.5),
            Event(type=EventType.TOOL_PROCESS),
            Event(type=EventType.END),
        )
        for legacy_output in legacy_outputs:
            with self.subTest(legacy_output=type(legacy_output).__qualname__):
                with self.assertRaises(StreamValidationError):
                    resp._queue_parser_output(legacy_output)

    async def test_queue_parser_output_accepts_consumer_projection(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
        )
        item = _canonical_item(
            StreamItemKind.ANSWER_DELTA,
            0,
            text_delta="canonical",
        )

        resp._queue_parser_output(StreamConsumerProjection.from_item(item))

        self.assertEqual(
            [item.kind for item in resp.canonical_items],
            [StreamItemKind.ANSWER_DELTA],
        )
        self.assertEqual(resp.canonical_items[0].text_delta, "canonical")
