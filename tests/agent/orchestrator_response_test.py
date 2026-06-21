from asyncio import (
    CancelledError,
    Future,
    create_task,
    get_running_loop,
    sleep,
    wait_for,
)
from asyncio import (
    Event as AsyncioEvent,
)
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass
from io import StringIO
from json import dumps, loads
from logging import getLogger
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from sqlalchemy import create_engine, text

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    OrchestratorResponse,
    _ToolExecutionOutcome,
)
from avalan.cli import CommandAbortException
from avalan.entities import (
    TOOL_DISPLAY_PROJECTOR_METADATA_KEY,
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
    ToolCapabilities,
    ToolDescriptor,
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
from avalan.model.nlp.text.vendor.openai import OpenAIStream
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
from avalan.tool.database import DatabaseToolSet, DatabaseToolSettings
from avalan.tool.display import (
    TOOL_DISPLAY_PROJECTION_METADATA_KEY,
    ToolDisplayProjection,
)
from avalan.tool.manager import ToolManager
from avalan.tool.parser import ToolCallParser


class _DummyEngine:
    def __init__(self):
        self.model_id = "m"
        self.tokenizer = MagicMock()


class _AsyncIter:
    def __init__(self, items: list[object]) -> None:
        self._iter = iter(items)

    def __aiter__(self) -> "_AsyncIter":
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._iter)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _dummy_create_async_engine(dsn: str, **_: Any) -> object:
    engine = create_engine(dsn)

    class DummyAsyncConn:
        def __init__(self, conn: Any) -> None:
            self.conn = conn

        async def exec_driver_sql(
            self, sql: str, *args: Any, **kwargs: Any
        ) -> Any:
            result = self.conn.exec_driver_sql(sql, *args, **kwargs)
            if not result.returns_rows:
                self.conn.commit()
            return result

        async def run_sync(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            return fn(self.conn, *args, **kwargs)

    class DummyConnCtx:
        def __init__(self, sync_engine: Any) -> None:
            self.engine = sync_engine
            self.conn: Any | None = None

        async def __aenter__(self) -> DummyAsyncConn:
            self.conn = self.engine.connect()
            return DummyAsyncConn(self.conn)

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: object | None,
        ) -> bool:
            assert self.conn is not None
            self.conn.close()
            return False

    class DummyAsyncEngine:
        def __init__(self, sync_engine: Any) -> None:
            self.engine = sync_engine

        def begin(self) -> DummyConnCtx:
            return DummyConnCtx(self.engine)

        @property
        def sync_engine(self) -> Any:
            return self.engine

        async def dispose(self) -> None:
            self.engine.dispose()

    return DummyAsyncEngine(engine)


class _ConfirmationAwaitable:
    def __init__(self, value: bool, awaited: list[bool]) -> None:
        self._value = value
        self._awaited = awaited

    def __await__(self) -> Generator[Any, None, bool]:
        async def resolve() -> bool:
            self._awaited.append(True)
            return self._value

        return resolve().__await__()


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


def _openai_completed_response(text: str) -> TextGenerationResponse:
    def output_fn(**_: object) -> OpenAIStream:
        return OpenAIStream(
            _AsyncIter(
                [
                    SimpleNamespace(
                        type="response.completed",
                        response=SimpleNamespace(
                            output=[
                                SimpleNamespace(
                                    type="message",
                                    content=[
                                        SimpleNamespace(
                                            type="output_text",
                                            text=text,
                                        )
                                    ],
                                )
                            ]
                        ),
                    )
                ]
            )
        )

    settings = GenerationSettings()
    return TextGenerationResponse(
        output_fn,
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=settings,
        settings=settings,
    )


class _RawFixtureResponse:
    is_async_generator = True
    input_token_count = 0
    output_token_count = 0
    usage = None
    can_think = False
    is_thinking = False

    def __init__(self, *items: object) -> None:
        self._items = items

    def add_done_callback(self, _: object) -> None:
        return None

    def set_thinking(self, _: bool) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[object]:
        return self._output_gen()

    async def _output_gen(self) -> AsyncIterator[object]:
        for item in self._items:
            yield item


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
        self.assertIs(
            token_events[0].observability.kind,
            EventPayloadKind.CANONICAL_STREAM,
        )
        self.assertEqual(
            token_events[0].payload,
            token_events[0].observability.data,
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
            {
                "text_delta_length": 1,
                "model_id": "m",
                "step": 0,
                "token_id": 42,
            },
        )
        self.assertNotIn("token", token_events[0].observability.data)
        self.assertEqual(
            token_events[1].observability.data["summary"],
            {
                "text_delta_length": 1,
                "model_id": "m",
                "step": 1,
                "token_id": 42,
            },
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
        self.assertEqual(event.payload, event.observability.data)
        self.assertEqual(event.payload["summary"]["text_delta_length"], 1)
        self.assertEqual(event.payload["summary"]["model_id"], "m")
        self.assertEqual(event.payload["summary"]["step"], 0)
        self.assertNotIn("token", event.payload)
        self.assertNotIn("token_id", event.payload["summary"])

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

    async def test_harmony_final_channel_marker_stays_out_of_answer_text(
        self,
    ) -> None:
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
                '{"url":"https://example.com"}',
                "<|channel|>final<|message|>done",
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

        self.assertEqual(_answer_text(items), "done")
        self.assertFalse(
            any(
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta
                and "<|channel|>final<|message|>" in item.text_delta
                for item in items
            )
        )
        self.assertIn(
            StreamItemKind.TOOL_CALL_READY,
            [item.kind for item in items],
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

    async def test_to_str_with_database_tool_and_json_schema_completion(
        self,
    ) -> None:
        with TemporaryDirectory() as tmp:
            dsn = f"sqlite:///{tmp}/db.sqlite"
            engine = create_engine(dsn)
            with engine.begin() as conn:
                conn.execute(
                    text("CREATE TABLE authors(id INTEGER, name TEXT)")
                )
                conn.execute(text("INSERT INTO authors VALUES (1, 'Author')"))
            engine.dispose()

            with patch(
                "avalan.tool.database.create_async_engine",
                _dummy_create_async_engine,
            ):
                async with DatabaseToolSet(
                    DatabaseToolSettings(dsn=dsn),
                    namespace="database",
                ) as toolset:
                    tool = ToolManager.create_instance(
                        available_toolsets=[toolset],
                        enable_tools=["database.run"],
                    )
                    agent = AsyncMock(spec=EngineAgent)
                    agent.engine = _DummyEngine()
                    agent.return_value = _openai_completed_response(
                        '{"answer":"Author"}'
                    )
                    operation = _dummy_operation()
                    response = _tool_call_response(
                        ToolCall(
                            id="call1",
                            name="database.run",
                            arguments={"sql": "SELECT name FROM authors"},
                        )
                    )
                    engine_args = {
                        "response_format": {
                            "type": "json_schema",
                            "name": "answer",
                            "schema": {"type": "object"},
                            "strict": True,
                        }
                    }
                    orchestrated = _make_response(
                        Message(role=MessageRole.USER, content="hi"),
                        response,
                        agent,
                        operation,
                        engine_args,
                        tool=tool,
                        enable_tool_parsing=False,
                    )

                    output = await orchestrated.to_str()

        self.assertEqual(output, '{"answer":"Author"}')
        agent.assert_awaited_once()
        continuation_context = agent.await_args.args[0]
        self.assertEqual(
            continuation_context.engine_args["response_format"]["type"],
            "json_schema",
        )
        continuation_input = continuation_context.input
        assert isinstance(continuation_input, list)
        last_message = continuation_input[-1]
        assert isinstance(last_message, Message)
        self.assertIn("Author", str(last_message.content))
        validate_canonical_stream_items(orchestrated.canonical_items)
        validate_tool_lifecycle_items(orchestrated.canonical_items)


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

    async def test_iteration_preserves_json_schema_for_tool_continuation(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="database.run", arguments={})
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        }

        async def continue_agent(
            context: ModelCallContext,
        ) -> TextGenerationResponse:
            if context.engine_args.get("response_format") == response_format:
                return _openai_completed_message_response('{"answer":"4"}')
            return _empty_text_response()

        agent.side_effect = continue_agent
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result={"rows": [{"answer": 4}]},
        )
        orchestrated = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {"response_format": response_format},
            tool=tool,
            enable_tool_parsing=False,
        )

        items = await _collect_stream_items(orchestrated)

        self.assertEqual(_answer_text(items), '{"answer":"4"}')
        agent.assert_awaited_once()
        context = agent.await_args.args[0]
        self.assertEqual(
            context.engine_args["response_format"],
            response_format,
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

    async def test_finish_stream_closes_active_model_continuation(
        self,
    ) -> None:
        cases = (
            (
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            ),
            (
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.MODEL_CONTINUATION_ERROR,
            ),
            (
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
            ),
        )
        for stream_terminal, continuation_terminal in cases:
            with self.subTest(stream_terminal=stream_terminal):
                engine = _DummyEngine()
                agent = AsyncMock(spec=EngineAgent)
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
                continuation_id = str(uuid4())
                response._append_canonical_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_STARTED,
                    continuation_id,
                )
                response._set_active_model_continuation(continuation_id)

                response._finish_canonical_stream(
                    stream_terminal,
                    data=(
                        {"message": "boom"}
                        if stream_terminal is StreamItemKind.STREAM_ERRORED
                        else None
                    ),
                )

                kinds = [item.kind for item in response.canonical_items]
                self.assertLess(
                    kinds.index(continuation_terminal),
                    kinds.index(stream_terminal),
                )
                self.assertIsNone(response._active_model_continuation_id)
                validate_canonical_stream_items(response.canonical_items)

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
                    metadata={
                        "bytes": 6,
                        TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                            "action": "stream",
                            "target": "stdout",
                            "summary": "Streaming stdout.",
                        },
                    },
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
                    metadata={
                        TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                            "action": "stream",
                            "target": "progress",
                            "summary": "Halfway done.",
                            "progress": 0.5,
                        },
                    },
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
        stdout_projection = cast(
            dict[str, object],
            output_items[0].metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        self.assertEqual(stdout_projection["action"], "stream")
        self.assertEqual(stdout_projection["target"], "stdout")
        self.assertEqual(output_items[0].data["metadata"]["bytes"], 6)
        self.assertEqual(output_items[2].data["metadata"], {"logger": "tool"})
        progress_item = canonical_items[7]
        progress_projection = cast(
            dict[str, object],
            progress_item.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        self.assertEqual(progress_projection["target"], "progress")
        self.assertEqual(progress_projection["progress"], 0.5)
        self.assertEqual(
            progress_item.data,
            {
                "category": "progress",
                "content": "half",
                "progress": 0.5,
                "metadata": {
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                        "action": "stream",
                        "target": "progress",
                        "summary": "Halfway done.",
                        "progress": 0.5,
                    },
                },
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

    async def test_iteration_yields_parallel_live_output_while_running(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _empty_text_response()
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        loop = get_running_loop()
        live_emitted: Future[None] = loop.create_future()
        release_tools: Future[None] = loop.create_future()

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
            context: ToolCallContext,
            **_: object,
        ) -> ToolCallResult:
            assert context.stream_event is not None
            arguments = cast(dict[str, str], call.arguments)
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content=f"{arguments['name']}\n",
                )
            )
            if not live_emitted.done():
                live_emitted.set_result(None)
            await release_tools
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=arguments["name"],
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
        iterator = response.__aiter__()
        items: list[CanonicalStreamItem] = []

        async def collect_first_live_output() -> CanonicalStreamItem:
            while True:
                item = await iterator.__anext__()
                items.append(item)
                if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
                    return item

        output_task = create_task(collect_first_live_output())
        try:
            await wait_for(live_emitted, 1)
            output_item = await wait_for(output_task, 1)
            kinds_before_release = [
                item.kind for item in response.canonical_items
            ]
        finally:
            if not release_tools.done():
                release_tools.set_result(None)

        self.assertIn(output_item.text_delta, {"first\n", "second\n"})
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            kinds_before_release,
        )
        agent.assert_not_awaited()

        while True:
            try:
                items.append(await wait_for(iterator.__anext__(), 1))
            except StopAsyncIteration:
                break

        kinds = [item.kind for item in items]
        output_index = kinds.index(StreamItemKind.TOOL_EXECUTION_OUTPUT)
        completed_index = kinds.index(StreamItemKind.TOOL_EXECUTION_COMPLETED)
        continuation_index = kinds.index(
            StreamItemKind.MODEL_CONTINUATION_STARTED
        )
        self.assertLess(output_index, completed_index)
        self.assertLess(completed_index, continuation_index)
        self.assertIsNone(response._pending_tool_batch_task)
        validate_canonical_stream_items(response.canonical_items)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_iteration_yields_live_tool_output_while_running(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})
        live_emitted = AsyncioEvent()
        release_tool = AsyncioEvent()

        async def continue_model(
            _: ModelCallContext,
        ) -> TextGenerationResponse:
            return _empty_text_response()

        agent.side_effect = continue_model

        async def execute(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolCallResult:
            assert context.stream_event is not None
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="live\n",
                )
            )
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.PROGRESS,
                    content="half",
                    progress=0.5,
                )
            )
            live_emitted.set()
            await release_tool.wait()
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="done",
            )

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.side_effect = execute

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        iterator = response.__aiter__()
        items: list[CanonicalStreamItem] = []
        while not any(
            item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS
            for item in items
        ):
            items.append(await wait_for(iterator.__anext__(), 1))

        self.assertTrue(live_emitted.is_set())
        self.assertFalse(release_tool.is_set())
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            [item.kind for item in items],
        )
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            [item.kind for item in response.canonical_items],
        )
        agent.assert_not_awaited()

        release_tool.set()
        while True:
            try:
                items.append(await wait_for(iterator.__anext__(), 1))
            except StopAsyncIteration:
                break

        kinds = [item.kind for item in items]
        output_index = kinds.index(StreamItemKind.TOOL_EXECUTION_OUTPUT)
        progress_index = kinds.index(StreamItemKind.TOOL_EXECUTION_PROGRESS)
        completed_index = kinds.index(StreamItemKind.TOOL_EXECUTION_COMPLETED)
        continuation_index = kinds.index(
            StreamItemKind.MODEL_CONTINUATION_STARTED
        )
        self.assertLess(output_index, completed_index)
        self.assertLess(progress_index, completed_index)
        self.assertLess(completed_index, continuation_index)
        self.assertIsNone(response._pending_tool_batch_task)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_iteration_yields_live_tool_output_before_tool_error(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={})
        live_emitted = AsyncioEvent()
        release_tool = AsyncioEvent()

        async def execute(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolCallResult:
            assert context.stream_event is not None
            await context.stream_event(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content="before error\n",
                )
            )
            live_emitted.set()
            await release_tool.wait()
            raise RuntimeError("boom")

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.side_effect = execute

        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        iterator = response.__aiter__()
        items: list[CanonicalStreamItem] = []
        while not any(
            item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT for item in items
        ):
            items.append(await wait_for(iterator.__anext__(), 1))

        self.assertTrue(live_emitted.is_set())
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in response.canonical_items],
        )
        agent.assert_not_awaited()

        release_tool.set()
        with self.assertRaises(RuntimeError):
            while True:
                items.append(await wait_for(iterator.__anext__(), 1))

        error_items = [
            item
            for item in response.canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        ]
        self.assertEqual(len(error_items), 1)
        terminal_items = [
            item
            for item in response.canonical_items
            if item.correlation.tool_call_id == "call1"
            and item.kind
            in {
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                StreamItemKind.TOOL_EXECUTION_ERROR,
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
            }
        ]
        self.assertEqual(
            [item.kind for item in terminal_items],
            [StreamItemKind.TOOL_EXECUTION_ERROR],
        )
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in items],
        )
        self.assertIsNone(response._pending_tool_batch_task)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_iteration_tool_context_cancellation_records_cancel_terminal(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        cancel_during_tool = False

        class CancellableTool(Tool):
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
        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[CancellableTool()])],
            enable_tools=["cancellable"],
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            enable_tool_parsing=False,
        )

        async def checker() -> None:
            if cancel_during_tool:
                raise CancelledError()

        response.set_cancellation_checker(checker)

        with self.assertRaises(CancelledError):
            await _collect_stream_items(response)

        agent.assert_not_awaited()
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
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                StreamItemKind.STREAM_CANCELLED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(
            canonical_items[5].data["code"],
            ToolCallDiagnosticCode.CANCELLED.value,
        )
        self.assertEqual(
            canonical_items[-2].terminal_outcome,
            StreamTerminalOutcome.CANCELLED,
        )
        self.assertFalse(
            any(
                trigger.args[0].type is EventType.TOOL_MODEL_RUN
                for trigger in event_manager.trigger.await_args_list
            )
        )

    async def test_tool_stream_events_after_terminal_are_ignored(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
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
            event_manager=event_manager,
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
        await emit(
            ToolExecutionStreamEvent(
                kind=ToolExecutionStreamKind.PROGRESS,
                content="half",
                progress=0.5,
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
        events = [
            call.args[0] for call in event_manager.trigger.await_args_list
        ]
        self.assertEqual(
            [event.type for event in events],
            [EventType.TOOL_PROGRESS, EventType.TOOL_PROGRESS],
        )
        output_payload = events[0].payload
        progress_payload = events[1].payload
        assert output_payload is not None
        self.assertEqual(
            output_payload["kind"],
            StreamItemKind.TOOL_EXECUTION_OUTPUT.value,
        )
        self.assertEqual(
            output_payload["correlation"],
            {"tool_call_id": "call1"},
        )
        self.assertEqual(
            output_payload["summary"],
            {
                "text_delta_length": 6,
                "data_keys": ["category", "content", "metadata"],
            },
        )
        self.assertNotIn("before", repr(output_payload))
        assert progress_payload is not None
        self.assertEqual(
            progress_payload["kind"],
            StreamItemKind.TOOL_EXECUTION_PROGRESS.value,
        )
        self.assertEqual(
            progress_payload["summary"],
            {
                "data_keys": [
                    "category",
                    "content",
                    "metadata",
                    "progress",
                ],
            },
        )
        self.assertNotIn("half", repr(progress_payload))
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ],
        )
        self.assertEqual(response.canonical_items[-3].text_delta, "before")

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
        executed_call = tool.await_args.args[0]
        self.assertEqual(
            executed_call.arguments,
            {"expression": "2 + 2"},
        )

    async def test_iteration_waits_for_tool_call_done_before_execution(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = cast(
            TextGenerationResponse,
            _RawFixtureResponse(
                _canonical_item(StreamItemKind.STREAM_STARTED, 0),
                _canonical_item(
                    StreamItemKind.TOOL_CALL_READY,
                    1,
                    data={"name": "calc", "arguments": {"x": 1}},
                    correlation=correlation,
                ),
                _canonical_item(StreamItemKind.STREAM_COMPLETED, 2),
                _canonical_item(StreamItemKind.STREAM_CLOSED, 3),
            ),
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

        tool.assert_not_awaited()
        agent.assert_not_awaited()
        kinds = [item.kind for item in items]
        self.assertIn(StreamItemKind.TOOL_CALL_READY, kinds)
        self.assertNotIn(StreamItemKind.TOOL_EXECUTION_STARTED, kinds)
        diagnostic = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(
            diagnostic.data["code"],
            "orchestrator.tool_call.ready_without_done",
        )
        self.assertEqual(diagnostic.correlation.tool_call_id, "call1")
        validate_canonical_stream_items(orchestrated.canonical_items)

    async def test_iteration_argument_deltas_override_ready_arguments(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def execute(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolCallResult:
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="ok",
            )

        tool.side_effect = execute
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = _response_from_items(
            _canonical_item(StreamItemKind.STREAM_STARTED, 0),
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta='{"expression":"delta"}',
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                2,
                data={"name": "calc", "arguments": {"expression": "ready"}},
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                3,
                correlation=correlation,
            ),
            _canonical_item(StreamItemKind.STREAM_COMPLETED, 4, usage={}),
            _canonical_item(StreamItemKind.STREAM_CLOSED, 5),
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

        self.assertEqual(_answer_text(items), "done")
        executed_call = tool.await_args.args[0]
        self.assertEqual(
            executed_call.arguments,
            {"expression": "delta"},
        )
        ready_item = next(
            item
            for item in orchestrated.canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(
            ready_item.data,
            {"name": "calc", "arguments": {"expression": "ready"}},
        )
        validate_canonical_stream_items(orchestrated.canonical_items)
        validate_tool_lifecycle_items(orchestrated.canonical_items)

    async def test_iteration_raw_argument_deltas_use_ready_arguments(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def execute(
            call: ToolCall,
            _context: ToolCallContext,
        ) -> ToolCallResult:
            return ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="ok",
            )

        tool.side_effect = execute
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = _response_from_items(
            _canonical_item(StreamItemKind.STREAM_STARTED, 0),
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta="(4 + 6) * 5 / 2",
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                2,
                data={
                    "name": "calc",
                    "arguments": {"input": "(4 + 6) * 5 / 2"},
                },
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                3,
                correlation=correlation,
            ),
            _canonical_item(StreamItemKind.STREAM_COMPLETED, 4, usage={}),
            _canonical_item(StreamItemKind.STREAM_CLOSED, 5),
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

        self.assertEqual(_answer_text(items), "done")
        executed_call = tool.await_args.args[0]
        self.assertEqual(
            executed_call.arguments,
            {"input": "(4 + 6) * 5 / 2"},
        )
        self.assertNotIn(
            StreamItemKind.STREAM_DIAGNOSTIC,
            [item.kind for item in orchestrated.canonical_items],
        )
        validate_canonical_stream_items(orchestrated.canonical_items)
        validate_tool_lifecycle_items(orchestrated.canonical_items)

    async def test_iteration_malformed_argument_deltas_emit_diagnostic(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = _response_from_items(
            _canonical_item(StreamItemKind.STREAM_STARTED, 0),
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta='{"x":',
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                2,
                data={"name": "calc", "arguments": {"x": 1}},
                correlation=correlation,
            ),
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                3,
                correlation=correlation,
            ),
            _canonical_item(StreamItemKind.STREAM_COMPLETED, 4, usage={}),
            _canonical_item(StreamItemKind.STREAM_CLOSED, 5),
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

        tool.assert_not_awaited()
        diagnostics = [
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].data["code"],
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
        )
        self.assertEqual(diagnostics[0].correlation.tool_call_id, "call1")
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            [item.kind for item in items],
        )
        validate_canonical_stream_items(orchestrated.canonical_items)

    async def test_malformed_tool_lifecycle_data_emits_diagnostics(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        cases: tuple[
            tuple[
                StreamItemKind,
                str | None,
                object | None,
                str,
            ],
            ...,
        ] = (
            (
                StreamItemKind.TOOL_CALL_READY,
                None,
                None,
                "orchestrator.tool_call.missing_ready_data",
            ),
            (
                StreamItemKind.TOOL_CALL_READY,
                None,
                {"arguments": {}},
                "orchestrator.tool_call.missing_name",
            ),
            (
                StreamItemKind.TOOL_CALL_READY,
                None,
                {"name": "calc", "arguments": []},
                ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
            ),
        )

        for kind, text_delta, data, expected_code in cases:
            with self.subTest(code=expected_code):
                response = _make_response(
                    Message(role=MessageRole.USER, content="hi"),
                    _empty_text_response(),
                    agent,
                    operation,
                    {},
                    enable_tool_parsing=False,
                )
                correlation = StreamItemCorrelation(
                    provider_request_id="provider-request-1",
                    model_continuation_id="continuation-1",
                    tool_call_id="call1",
                    task_id="task-1",
                )
                response._append_canonical_response_item(
                    _canonical_item(
                        kind,
                        1,
                        text_delta=text_delta,
                        data=data,
                        correlation=correlation,
                    )
                )
                if kind is StreamItemKind.TOOL_CALL_READY:
                    response._append_canonical_response_item(
                        _canonical_item(
                            StreamItemKind.TOOL_CALL_DONE,
                            2,
                            correlation=correlation,
                        )
                    )

                diagnostic = next(
                    item
                    for item in response.canonical_items
                    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                )
                self.assertEqual(diagnostic.data["code"], expected_code)
                self.assertEqual(diagnostic.correlation.tool_call_id, "call1")
                self.assertTrue(response._calls.empty())

    async def test_malformed_canonical_lifecycle_missing_id_keeps_correlation(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        correlation = StreamItemCorrelation(
            provider_request_id="provider-request-1",
            model_continuation_id="continuation-1",
            flow_run_id="flow-1",
            node_id="node-1",
            parent_sequence=3,
            protocol_item_id="protocol-1",
            task_id="task-1",
            artifact_id="artifact-1",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_text_response(),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )

        response._append_canonical_tool_call_lifecycle_item(
            StreamItemKind.TOOL_CALL_READY,
            data={"name": "calc", "arguments": {"x": 1}},
            correlation=correlation,
        )

        diagnostic = next(
            item
            for item in response.canonical_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(
            diagnostic.data["code"],
            "orchestrator.tool_call.missing_id",
        )
        self.assertTrue(
            diagnostic.correlation.tool_call_id.startswith(
                "orchestrator-tool-call-diagnostic-"
            )
        )
        self.assertEqual(
            diagnostic.correlation.provider_request_id,
            "provider-request-1",
        )
        self.assertEqual(
            diagnostic.correlation.model_continuation_id,
            "continuation-1",
        )
        self.assertEqual(diagnostic.correlation.flow_run_id, "flow-1")
        self.assertEqual(diagnostic.correlation.node_id, "node-1")
        self.assertEqual(diagnostic.correlation.parent_sequence, 3)
        self.assertEqual(
            diagnostic.correlation.protocol_item_id,
            "protocol-1",
        )
        self.assertEqual(diagnostic.correlation.task_id, "task-1")
        self.assertEqual(diagnostic.correlation.artifact_id, "artifact-1")
        self.assertTrue(response._calls.empty())

    async def test_invalid_tool_lifecycle_order_diagnostics_are_specific(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        ready_data = {"name": "calc", "arguments": {}}

        def make_response() -> OrchestratorResponse:
            return _make_response(
                Message(role=MessageRole.USER, content="hi"),
                _empty_text_response(),
                agent,
                operation,
                {},
                enable_tool_parsing=False,
            )

        def diagnostic_codes(
            response: OrchestratorResponse,
        ) -> list[str]:
            return [
                item.data["code"]
                for item in response.canonical_items
                if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
            ]

        response = make_response()
        correlation = StreamItemCorrelation(tool_call_id="call-after-ready")
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                1,
                data=ready_data,
                correlation=correlation,
            )
        )
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                2,
                text_delta="{}",
                correlation=correlation,
            )
        )

        self.assertEqual(
            diagnostic_codes(response),
            ["orchestrator.tool_call.argument_after_ready"],
        )

        response = make_response()
        correlation = StreamItemCorrelation(tool_call_id="call-after-done")
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                1,
                data=ready_data,
                correlation=correlation,
            )
        )
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                2,
                correlation=correlation,
            )
        )
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                3,
                text_delta="{}",
                correlation=correlation,
            )
        )

        self.assertEqual(
            diagnostic_codes(response),
            ["orchestrator.tool_call.argument_after_done"],
        )

        response = make_response()
        correlation = StreamItemCorrelation(tool_call_id="ready-after-done")
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                1,
                data=ready_data,
                correlation=correlation,
            )
        )
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                2,
                correlation=correlation,
            )
        )
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                3,
                data=ready_data,
                correlation=correlation,
            )
        )

        self.assertEqual(
            diagnostic_codes(response),
            ["orchestrator.tool_call.ready_after_done"],
        )

    async def test_tool_lifecycle_append_after_terminal_is_ignored(
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
            enable_tool_parsing=False,
        )
        response._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)
        item_count = len(response.canonical_items)

        appended = response._append_canonical_tool_call_lifecycle_item(
            StreamItemKind.TOOL_CALL_READY,
            data={"name": "calc", "arguments": {}},
            correlation=StreamItemCorrelation(tool_call_id="call1"),
        )

        self.assertIsNone(appended)
        self.assertEqual(len(response.canonical_items), item_count)

    async def test_completed_tool_lifecycle_queues_once(self) -> None:
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
            enable_tool_parsing=False,
        )
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_READY,
                1,
                data={"name": "calc", "arguments": {}},
                correlation=correlation,
            )
        )
        response._append_canonical_response_item(
            _canonical_item(
                StreamItemKind.TOOL_CALL_DONE,
                2,
                correlation=correlation,
            )
        )
        state = response._canonical_tool_call_lifecycles["call1"]

        response._queue_completed_canonical_tool_call("call1", state)

        self.assertEqual(response._calls.qsize(), 1)

    async def test_tool_call_from_lifecycle_fallback_branches(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        def make_response() -> OrchestratorResponse:
            return _make_response(
                Message(role=MessageRole.USER, content="hi"),
                _empty_text_response(),
                agent,
                operation,
                {},
                enable_tool_parsing=False,
            )

        cases: tuple[
            tuple[str, object, tuple[str, ...], str],
            ...,
        ] = (
            (
                "missing-ready-data",
                "not-ready-data",
                (),
                "orchestrator.tool_call.missing_ready_data",
            ),
            (
                "missing-name",
                {"arguments": {}},
                (),
                "orchestrator.tool_call.missing_name",
            ),
            (
                "decoded-non-object",
                {"name": "calc", "arguments": {}},
                ("[]",),
                ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
            ),
            (
                "ready-arguments-non-object",
                {"name": "calc", "arguments": []},
                (),
                ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
            ),
        )

        for call_id, ready_data, argument_deltas, expected_code in cases:
            with self.subTest(call_id=call_id):
                response = make_response()
                state = response._canonical_tool_call_lifecycle(call_id)
                state.ready_item = _canonical_item(
                    StreamItemKind.TOOL_CALL_READY,
                    1,
                    data=ready_data,
                    correlation=StreamItemCorrelation(tool_call_id=call_id),
                )
                state.argument_deltas.extend(argument_deltas)

                self.assertIsNone(
                    response._tool_call_from_canonical_lifecycle(
                        call_id,
                        state,
                    )
                )
                diagnostic = next(
                    item
                    for item in response.canonical_items
                    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                )
                self.assertEqual(diagnostic.data["code"], expected_code)
                self.assertEqual(diagnostic.correlation.tool_call_id, call_id)

        response = make_response()
        state = response._canonical_tool_call_lifecycle("call-no-arguments")
        state.ready_item = _canonical_item(
            StreamItemKind.TOOL_CALL_READY,
            1,
            data={"name": "calc"},
            correlation=StreamItemCorrelation(
                tool_call_id="call-no-arguments"
            ),
        )

        call = response._tool_call_from_canonical_lifecycle(
            "call-no-arguments",
            state,
        )

        self.assertIsNotNone(call)
        assert call is not None
        self.assertEqual(call.id, "call-no-arguments")
        self.assertEqual(call.name, "calc")
        self.assertIsNone(call.arguments)

    async def test_iteration_waits_until_tool_call_done_before_execution(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="1",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _response_from_items(*_canonical_tool_call_items(call)),
            agent,
            operation,
            {},
            tool=tool,
            tool_confirm=lambda confirmed: "y",
            enable_tool_parsing=False,
        )
        iterator = response.__aiter__()

        ready_items: list[CanonicalStreamItem] = []
        while True:
            item = await iterator.__anext__()
            ready_items.append(item)
            if item.kind is StreamItemKind.TOOL_CALL_READY:
                break

        tool.assert_not_awaited()
        self.assertTrue(response._calls.empty())
        self.assertEqual(ready_items[-1].data["arguments"], {"x": 1})

        while True:
            item = await iterator.__anext__()
            if item.kind is StreamItemKind.TOOL_CALL_DONE:
                break

        tool.assert_not_awaited()
        self.assertFalse(response._calls.empty())

        while True:
            item = await iterator.__anext__()
            if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
                break

        tool.assert_awaited_once()
        executed_call = tool.await_args.args[0]
        self.assertEqual(executed_call.arguments, {"x": 1})

    async def test_malformed_tool_call_arguments_emit_diagnostic(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            cast(
                TextGenerationResponse,
                _RawFixtureResponse(
                    _canonical_item(StreamItemKind.STREAM_STARTED, 0),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        1,
                        text_delta='{"x":',
                        correlation=correlation,
                    ),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_READY,
                        2,
                        data={"name": "calc", "arguments": {"x": 1}},
                        correlation=correlation,
                    ),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_DONE,
                        3,
                        correlation=correlation,
                    ),
                    _canonical_item(StreamItemKind.STREAM_COMPLETED, 4),
                ),
            ),
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        items = await _collect_stream_items(response)

        tool.assert_not_awaited()
        diagnostics = [
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].data["code"],
            ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
        )
        self.assertEqual(diagnostics[0].correlation.tool_call_id, "call1")
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            [item.kind for item in items],
        )
        validate_canonical_stream_items(response.canonical_items)

    async def test_incomplete_tool_call_ready_emits_diagnostic(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        correlation = StreamItemCorrelation(tool_call_id="call1")
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            cast(
                TextGenerationResponse,
                _RawFixtureResponse(
                    _canonical_item(StreamItemKind.STREAM_STARTED, 0),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_READY,
                        1,
                        data={"name": "calc", "arguments": {"x": 1}},
                        correlation=correlation,
                    ),
                    _canonical_item(StreamItemKind.STREAM_COMPLETED, 2),
                ),
            ),
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        items = await _collect_stream_items(response)

        tool.assert_not_awaited()
        diagnostics = [
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertEqual(
            diagnostics[0].data["code"],
            "orchestrator.tool_call.ready_without_done",
        )
        self.assertEqual(diagnostics[0].correlation.tool_call_id, "call1")
        self.assertEqual(
            [
                item.kind
                for item in items
                if item.correlation.tool_call_id == "call1"
            ],
            [
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        validate_canonical_stream_items(response.canonical_items)

    async def test_incomplete_tool_call_delta_emits_missing_ready_diagnostic(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        correlation = StreamItemCorrelation(
            provider_request_id="provider-request-1",
            model_continuation_id="continuation-1",
            tool_call_id="call1",
            task_id="task-1",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            cast(
                TextGenerationResponse,
                _RawFixtureResponse(
                    _canonical_item(StreamItemKind.STREAM_STARTED, 0),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        1,
                        text_delta='{"x":1}',
                        correlation=correlation,
                    ),
                    _canonical_item(StreamItemKind.STREAM_COMPLETED, 2),
                ),
            ),
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        items = await _collect_stream_items(response)

        tool.assert_not_awaited()
        agent.assert_not_awaited()
        diagnostic = next(
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        )
        self.assertEqual(
            diagnostic.data["code"],
            "orchestrator.tool_call.missing_ready",
        )
        self.assertEqual(diagnostic.correlation.tool_call_id, "call1")
        self.assertEqual(
            diagnostic.correlation.provider_request_id,
            "provider-request-1",
        )
        self.assertEqual(
            diagnostic.correlation.model_continuation_id,
            "continuation-1",
        )
        self.assertEqual(diagnostic.correlation.task_id, "task-1")
        self.assertEqual(
            [
                item.kind
                for item in items
                if item.correlation.tool_call_id == "call1"
            ],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        )
        closing = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_DONE
        )
        self.assertEqual(
            closing.metadata["tool_call.close_reason"],
            "malformed",
        )
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            [item.kind for item in items],
        )
        validate_canonical_stream_items(response.canonical_items)

    async def test_invalid_tool_call_lifecycles_do_not_execute(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        for code, lifecycle_items in (
            (
                "orchestrator.tool_call.duplicate_ready",
                (
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                ),
            ),
            (
                "orchestrator.tool_call.done_before_ready",
                (
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_DONE,
                ),
            ),
        ):
            with self.subTest(code=code):
                tool.reset_mock()
                correlation = StreamItemCorrelation(
                    provider_request_id="provider-request-1",
                    model_continuation_id="continuation-1",
                    tool_call_id="call1",
                    task_id="task-1",
                )
                sequence = 1
                items = [_canonical_item(StreamItemKind.STREAM_STARTED, 0)]
                for kind in lifecycle_items:
                    kwargs: dict[str, Any] = {"correlation": correlation}
                    if kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
                        kwargs["text_delta"] = '{"x":1}'
                    if kind is StreamItemKind.TOOL_CALL_READY:
                        kwargs["data"] = {
                            "name": "calc",
                            "arguments": {"x": 1},
                        }
                    items.append(_canonical_item(kind, sequence, **kwargs))
                    sequence += 1
                items.append(
                    _canonical_item(StreamItemKind.STREAM_COMPLETED, sequence)
                )
                response = _make_response(
                    Message(role=MessageRole.USER, content="hi"),
                    cast(
                        TextGenerationResponse,
                        _RawFixtureResponse(*items),
                    ),
                    agent,
                    operation,
                    {},
                    tool=tool,
                    enable_tool_parsing=False,
                )

                collected = await _collect_stream_items(response)

                tool.assert_not_awaited()
                diagnostics = [
                    item
                    for item in collected
                    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                ]
                self.assertTrue(
                    any(item.data["code"] == code for item in diagnostics)
                )
                self.assertTrue(
                    all(
                        item.correlation.tool_call_id == "call1"
                        for item in diagnostics
                    )
                )
                self.assertTrue(
                    all(
                        item.correlation.provider_request_id
                        == "provider-request-1"
                        for item in diagnostics
                    )
                )
                self.assertTrue(
                    all(
                        item.correlation.model_continuation_id
                        == "continuation-1"
                        for item in diagnostics
                    )
                )
                self.assertTrue(
                    all(
                        item.correlation.task_id == "task-1"
                        for item in diagnostics
                    )
                )
                self.assertNotIn(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    [item.kind for item in collected],
                )
                validate_canonical_stream_items(response.canonical_items)

    async def test_to_str_duplicate_done_executes_queued_call_once(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _string_response("done", async_gen=True)
        operation = _dummy_operation()
        correlation = StreamItemCorrelation(tool_call_id="call1")
        executed: list[int] = []

        async def tracked(x: int) -> str:
            executed.append(x)
            return "ok"

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(id="call1", name="tracked", arguments={"x": 1})
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            cast(
                TextGenerationResponse,
                _RawFixtureResponse(
                    _canonical_item(StreamItemKind.STREAM_STARTED, 0),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_READY,
                        1,
                        data={"name": call.name, "arguments": call.arguments},
                        correlation=correlation,
                    ),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_DONE,
                        2,
                        correlation=correlation,
                    ),
                    _canonical_item(
                        StreamItemKind.TOOL_CALL_DONE,
                        3,
                        correlation=correlation,
                    ),
                    _canonical_item(StreamItemKind.STREAM_COMPLETED, 4),
                ),
            ),
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")

        self.assertEqual(executed, [1])
        agent.assert_awaited_once()
        diagnostics = [
            item
            for item in response.canonical_items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(
            diagnostics[0].data["code"],
            "orchestrator.tool_call.duplicate_done",
        )
        self.assertEqual(diagnostics[0].correlation.tool_call_id, "call1")
        self.assertEqual(
            [
                item.kind
                for item in response.canonical_items
                if item.correlation.tool_call_id == "call1"
                and item.kind is StreamItemKind.TOOL_CALL_DONE
            ],
            [StreamItemKind.TOOL_CALL_DONE],
        )
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            [item.kind for item in response.canonical_items],
        )
        validate_canonical_stream_items(response.canonical_items)

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
        self.assertTrue(response._calls.empty())

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
        self.assertEqual(
            canonical_items[-3].data["code"],
            ToolCallDiagnosticCode.CANCELLED.value,
        )
        self.assertEqual(
            canonical_items[-3].data["stage"],
            ToolCallDiagnosticStage.DISPATCH.value,
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
        self.assertEqual(
            canonical_items[5].data["code"],
            ToolCallDiagnosticCode.CANCELLED.value,
        )
        self.assertEqual(
            canonical_items[5].data["stage"],
            ToolCallDiagnosticStage.GUARD.value,
        )
        self.assertFalse(
            any(
                trigger.args[0].type is EventType.TOOL_EXECUTE
                for trigger in event_manager.trigger.await_args_list
            )
        )

    async def test_iteration_ready_precedes_confirmation_denial(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call1", name="calc", arguments={"x": 1})
        response = _tool_call_response(call)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        confirmation_snapshots: list[list[StreamItemKind]] = []
        orchestrated: OrchestratorResponse

        def confirm(confirmed_call: ToolCall) -> str:
            self.assertEqual(confirmed_call.id, "call1")
            confirmation_snapshots.append(
                [
                    item.kind
                    for item in orchestrated.canonical_items
                    if item.correlation.tool_call_id == "call1"
                ]
            )
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
        prefix: list[CanonicalStreamItem] = []
        while True:
            item = await iterator.__anext__()
            prefix.append(item)
            if item.kind is StreamItemKind.TOOL_CALL_READY:
                break

        self.assertEqual(confirmation_snapshots, [])
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            [item.kind for item in prefix],
        )

        await _drain_until_exception(iterator, CommandAbortException)

        tool.assert_not_awaited()
        agent.assert_not_awaited()
        self.assertEqual(
            confirmation_snapshots,
            [
                [
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    StreamItemKind.TOOL_CALL_READY,
                    StreamItemKind.TOOL_CALL_DONE,
                ]
            ],
        )
        canonical_items = orchestrated.canonical_items
        self.assertEqual(
            [
                item.kind
                for item in canonical_items
                if item.correlation.tool_call_id == "call1"
            ],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_ERROR,
            ],
        )
        rejection = next(
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        )
        self.assertEqual(
            rejection.data["code"],
            ToolCallDiagnosticCode.USER_REJECTED.value,
        )
        self.assertEqual(
            [item.kind for item in canonical_items[-2:]],
            [StreamItemKind.STREAM_CANCELLED, StreamItemKind.STREAM_CLOSED],
        )
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            [item.kind for item in canonical_items],
        )
        self.assertFalse(
            any(
                trigger.args[0].type is EventType.TOOL_EXECUTE
                for trigger in event_manager.trigger.await_args_list
            )
        )
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

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
        self.assertFalse(
            any(
                call.args[0].type is EventType.TOOL_MODEL_RESPONSE
                for call in event_manager.trigger.await_args_list
            )
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
        self.assertFalse(
            any(
                call.args[0].type is EventType.TOOL_MODEL_RESPONSE
                for call in event_manager.trigger.await_args_list
            )
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
        self.assertFalse(
            any(
                call.args[0].type is EventType.TOOL_MODEL_RESPONSE
                for call in event_manager.trigger.await_args_list
            )
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

    async def test_to_str_remaps_reused_source_tool_call_id(self) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        observed: list[str] = []

        async def tracked(value: str) -> str:
            observed.append(value)
            return value

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        first_call = ToolCall(
            id="call1",
            name="tracked",
            arguments={"value": "first"},
        )
        second_call = ToolCall(
            id="call1",
            name="tracked",
            arguments={"value": "second"},
        )

        agent.side_effect = [
            _tool_call_response(second_call),
            _string_response("done", async_gen=True),
        ]
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(first_call),
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
        self.assertEqual(ready_ids, ["call1", "orchestrator-tool-call-1"])
        terminal_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids, ready_ids)
        child_contexts = [
            call_args.args[0] for call_args in agent.await_args_list
        ]
        observed_tool_call_ids = [
            message.tool_calls[0].id
            for context in child_contexts
            for message in cast(list[Message], context.input)
            if message.tool_calls
        ]
        self.assertEqual(
            observed_tool_call_ids,
            ["call1", "call1", "orchestrator-tool-call-1"],
        )
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_to_str_reserves_diagnostic_only_tool_call_id(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        observed: list[str] = []

        async def tracked(value: str) -> str:
            observed.append(value)
            return value

        tool = ToolManager.create_instance(
            available_toolsets=[ToolSet(tools=[tracked])],
            enable_tools=["tracked"],
            settings=ToolManagerSettings(),
        )
        first_call = ToolCall(
            id="call2",
            name="tracked",
            arguments={"value": "first"},
        )
        second_call = ToolCall(
            id="call1",
            name="tracked",
            arguments={"value": "second"},
        )
        first_call_correlation = StreamItemCorrelation(tool_call_id="call2")
        first_response = cast(
            TextGenerationResponse,
            _RawFixtureResponse(
                _canonical_item(StreamItemKind.STREAM_STARTED, 0),
                _canonical_item(
                    StreamItemKind.TOOL_CALL_DONE,
                    1,
                    correlation=StreamItemCorrelation(tool_call_id="call1"),
                ),
                _canonical_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    2,
                    text_delta=dumps(first_call.arguments),
                    correlation=first_call_correlation,
                ),
                _canonical_item(
                    StreamItemKind.TOOL_CALL_READY,
                    3,
                    data={
                        "name": first_call.name,
                        "arguments": first_call.arguments,
                    },
                    correlation=first_call_correlation,
                ),
                _canonical_item(
                    StreamItemKind.TOOL_CALL_DONE,
                    4,
                    correlation=first_call_correlation,
                ),
                _canonical_item(StreamItemKind.STREAM_COMPLETED, 5),
                _canonical_item(StreamItemKind.STREAM_CLOSED, 6),
            ),
        )

        agent.side_effect = [
            _tool_call_response(second_call),
            _string_response("done", async_gen=True),
        ]
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            first_response,
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
        diagnostic_codes = [
            item.data["code"]
            for item in canonical_items
            if (
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                and item.correlation.tool_call_id == "call1"
            )
        ]
        self.assertIn(
            "orchestrator.tool_call.done_before_ready",
            diagnostic_codes,
        )
        ready_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        ]
        self.assertEqual(ready_ids, ["call2", "orchestrator-tool-call-1"])
        self.assertNotIn("call1", ready_ids)
        terminal_ids = [
            item.correlation.tool_call_id
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        ]
        self.assertEqual(terminal_ids, ready_ids)
        child_contexts = [
            call_args.args[0] for call_args in agent.await_args_list
        ]
        observed_tool_call_ids = [
            message.tool_calls[0].id
            for context in child_contexts
            for message in cast(list[Message], context.input)
            if message.tool_calls
        ]
        self.assertEqual(
            observed_tool_call_ids,
            ["call2", "call2", "orchestrator-tool-call-1"],
        )
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

    async def test_parallel_text_detected_confirmation_sees_ready_done(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        executed: list[str] = []
        confirmation_snapshots: list[dict[str, list[StreamItemKind]]] = []
        execution_seen_at_confirmation: list[bool] = []

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
        cast(Any, tool).get_calls = MagicMock(
            side_effect=lambda text: calls if text == "tools" else None
        )
        response: OrchestratorResponse

        def confirm(call: ToolCall) -> str:
            self.assertEqual(call.id, "call-1")
            confirmation_snapshots.append(
                {
                    call_id: [
                        item.kind
                        for item in response.canonical_items
                        if item.correlation.tool_call_id == call_id
                    ]
                    for call_id in ("call-1", "call-2")
                }
            )
            execution_seen_at_confirmation.append(
                any(
                    item.kind
                    in {
                        StreamItemKind.TOOL_EXECUTION_STARTED,
                        StreamItemKind.TOOL_EXECUTION_COMPLETED,
                        StreamItemKind.TOOL_EXECUTION_ERROR,
                        StreamItemKind.TOOL_EXECUTION_CANCELLED,
                    }
                    for item in response.canonical_items
                )
            )
            return "n"

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("tools", async_gen=True),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=confirm,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")

        self.assertEqual(executed, [])
        self.assertEqual(execution_seen_at_confirmation, [False])
        self.assertEqual(
            confirmation_snapshots,
            [
                {
                    "call-1": [
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        StreamItemKind.TOOL_CALL_READY,
                        StreamItemKind.TOOL_CALL_DONE,
                    ],
                    "call-2": [
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        StreamItemKind.TOOL_CALL_READY,
                        StreamItemKind.TOOL_CALL_DONE,
                    ],
                }
            ],
        )
        self.assertEqual(
            [
                (item.correlation.tool_call_id, item.kind)
                for item in response.canonical_items
                if item.kind
                in {
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                }
            ],
            [
                ("call-1", StreamItemKind.TOOL_EXECUTION_STARTED),
                ("call-1", StreamItemKind.TOOL_EXECUTION_ERROR),
                ("call-2", StreamItemKind.TOOL_EXECUTION_STARTED),
                ("call-2", StreamItemKind.TOOL_EXECUTION_CANCELLED),
            ],
        )
        validate_canonical_stream_items(response.canonical_items)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_parallel_text_detected_confirmation_snapshots_for_outcomes(
        self,
    ) -> None:
        expected_snapshot = {
            "call-1": [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
            "call-2": [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            ],
        }
        cases: tuple[
            tuple[
                str,
                dict[str, str | BaseException],
                type[BaseException] | None,
                list[str],
                bool,
            ],
            ...,
        ] = (
            ("accept-all", {"call-1": "a"}, None, ["call-1"], True),
            (
                "all-accept",
                {"call-1": "y", "call-2": "y"},
                None,
                ["call-1", "call-2"],
                True,
            ),
            (
                "exception",
                {"call-1": "y", "call-2": RuntimeError("confirm failed")},
                RuntimeError,
                ["call-1", "call-2"],
                False,
            ),
            (
                "cancellation",
                {"call-1": "y", "call-2": CancelledError()},
                CancelledError,
                ["call-1", "call-2"],
                False,
            ),
        )

        for (
            label,
            decisions,
            expected_exception,
            expected_confirmed,
            should_execute,
        ) in cases:
            with self.subTest(label=label):
                engine = _DummyEngine()
                agent = AsyncMock(spec=EngineAgent)
                agent.engine = engine
                agent.return_value = _string_response("done", async_gen=True)
                operation = _dummy_operation()
                event_manager = MagicMock(spec=EventManager)
                event_manager.trigger = AsyncMock()
                executed: list[str] = []
                confirmed: list[str] = []
                confirmation_snapshots: list[
                    dict[str, list[StreamItemKind]]
                ] = []
                execution_seen_at_confirmation: list[bool] = []

                async def tracked(name: str, delay: float) -> str:
                    executed.append(name)
                    await sleep(delay)
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
                cast(Any, tool).get_calls = MagicMock(
                    side_effect=lambda text: calls if text == "tools" else None
                )
                response: OrchestratorResponse

                def confirm(call: ToolCall) -> str:
                    call_id = str(call.id)
                    confirmed.append(call_id)
                    confirmation_snapshots.append(
                        {
                            snapshot_call_id: [
                                item.kind
                                for item in response.canonical_items
                                if (
                                    item.correlation.tool_call_id
                                    == snapshot_call_id
                                )
                            ]
                            for snapshot_call_id in ("call-1", "call-2")
                        }
                    )
                    execution_seen_at_confirmation.append(
                        any(
                            item.kind
                            in {
                                StreamItemKind.TOOL_EXECUTION_STARTED,
                                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                                StreamItemKind.TOOL_EXECUTION_ERROR,
                                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                            }
                            for item in response.canonical_items
                        )
                    )
                    decision = decisions[call_id]
                    if isinstance(decision, BaseException):
                        raise decision
                    return decision

                response = _make_response(
                    Message(role=MessageRole.USER, content="hi"),
                    _string_response("tools", async_gen=True),
                    agent,
                    operation,
                    {},
                    event_manager=event_manager,
                    tool=tool,
                    tool_confirm=confirm,
                    enable_tool_parsing=False,
                )

                if expected_exception is None:
                    self.assertEqual(await response.to_str(), "done")
                else:
                    with self.assertRaises(expected_exception):
                        await response.to_str()

                self.assertEqual(confirmed, expected_confirmed)
                self.assertEqual(
                    execution_seen_at_confirmation,
                    [False for _ in expected_confirmed],
                )
                self.assertEqual(
                    confirmation_snapshots,
                    [expected_snapshot for _ in expected_confirmed],
                )
                if should_execute:
                    self.assertCountEqual(executed, ["first", "second"])
                    self.assertTrue(
                        response._tool_confirm_all or label == "all-accept"
                    )
                else:
                    self.assertEqual(executed, [])
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
                    ToolCallDiagnosticCode.CANCELLED.value,
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
                (
                    "call-1",
                    StreamItemKind.TOOL_EXECUTION_CANCELLED,
                    ToolCallDiagnosticCode.CANCELLED.value,
                ),
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

    async def test_single_confirmation_accepts_true(self) -> None:
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
        call = ToolCall(
            id="call-1",
            name="tracked",
            arguments={"name": "first"},
        )

        agent.return_value = _string_response("done", async_gen=True)
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=lambda _: True,
            enable_tool_parsing=False,
        )

        self.assertEqual(await response.to_str(), "done")
        self.assertEqual(executed, ["first"])
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in response.canonical_items],
        )
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            [item.kind for item in response.canonical_items],
        )
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_single_confirmation_accepts_future(self) -> None:
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
        call = ToolCall(
            id="call-1",
            name="tracked",
            arguments={"name": "first"},
        )

        def confirm(_: ToolCall) -> Future[bool]:
            future: Future[bool] = get_running_loop().create_future()
            future.set_result(True)
            return future

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
        self.assertEqual(executed, ["first"])
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in response.canonical_items],
        )
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            [item.kind for item in response.canonical_items],
        )
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_single_confirmation_accepts_custom_awaitable(
        self,
    ) -> None:
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        awaited: list[bool] = []
        executed: list[str] = []

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

        def confirm(_: ToolCall) -> _ConfirmationAwaitable:
            return _ConfirmationAwaitable(True, awaited)

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
        self.assertEqual(awaited, [True])
        self.assertEqual(executed, ["first"])
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in response.canonical_items],
        )
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            [item.kind for item in response.canonical_items],
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

    async def test_parallel_confirmation_accepts_true_and_future(
        self,
    ) -> None:
        for confirmation_kind in ("true", "future"):
            with self.subTest(confirmation_kind=confirmation_kind):
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

                def confirm(call: ToolCall) -> bool | Future[bool]:
                    confirmed.append(str(call.id))
                    if confirmation_kind == "true":
                        return True
                    future: Future[bool] = get_running_loop().create_future()
                    future.set_result(True)
                    return future

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
                self.assertCountEqual(executed, ["first", "second"])
                self.assertNotIn(
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    [item.kind for item in response.canonical_items],
                )
                self.assertEqual(
                    [
                        item.correlation.tool_call_id
                        for item in response.canonical_items
                        if (
                            item.kind
                            is StreamItemKind.TOOL_EXECUTION_COMPLETED
                        )
                    ],
                    ["call-1", "call-2"],
                )
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
        model_run_events = [
            event for event in events if event.type is EventType.TOOL_MODEL_RUN
        ]
        model_response_events = [
            event
            for event in events
            if event.type is EventType.TOOL_MODEL_RESPONSE
        ]
        result_events = [
            event for event in events if event.type is EventType.TOOL_RESULT
        ]
        self.assertEqual(len(result_events), 3)
        self.assertEqual(len(model_response_positions), 1)
        self.assertEqual(len(model_run_events), 1)
        self.assertEqual(len(model_response_events), 1)
        self.assertLess(max(result_positions), model_response_positions[0])
        self.assertEqual(
            model_run_events[0].payload,
            model_run_events[0].observability.data,
        )
        self.assertEqual(
            model_response_events[0].payload,
            model_response_events[0].observability.data,
        )
        self.assertEqual(
            model_run_events[0].payload["kind"],
            StreamItemKind.MODEL_CONTINUATION_STARTED.value,
        )
        self.assertEqual(
            model_response_events[0].payload["kind"],
            StreamItemKind.MODEL_CONTINUATION_COMPLETED.value,
        )
        self.assertNotIn("messages", model_run_events[0].payload)
        self.assertNotIn("response", model_response_events[0].payload)
        self.assertEqual(
            [
                event.observability.data["correlation"]["tool_call_id"]
                for event in result_events
            ],
            ["call-2", "call-1", "call-3"],
        )
        for event in result_events:
            self.assertEqual(event.payload, event.observability.data)
            self.assertNotIn("call", event.payload)
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


def _openai_completed_message_response(text: str) -> TextGenerationResponse:
    async def events() -> AsyncIterator[object]:
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                output=[
                    SimpleNamespace(
                        type="message",
                        content=[
                            SimpleNamespace(
                                type="output_text",
                                text=text,
                            )
                        ],
                    )
                ],
            ),
        )

    settings = GenerationSettings()
    return TextGenerationResponse(
        lambda **_: OpenAIStream(events()),
        logger=getLogger(),
        use_async_generator=True,
        generation_settings=settings,
        settings=settings,
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
            enable_tool_parsing=False,
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

    async def test_canonical_tool_items_include_display_projection_metadata(
        self,
    ) -> None:
        async def projected_calc(expression: str) -> str:
            return "4" if expression == "2 + 2" else expression

        def project_display(*items: object, **kwargs: object) -> object | None:
            outcome = kwargs.get("outcome")
            if outcome is None and len(items) > 1:
                outcome = items[1]
            if isinstance(outcome, ToolCallResult):
                return ToolDisplayProjection(
                    action="finish",
                    target=outcome.name,
                    summary="Calculated result.",
                    status="completed",
                    outcome="result",
                )
            call = kwargs.get("call")
            if not isinstance(call, ToolCall) and items:
                call = items[0]
            if not isinstance(call, ToolCall):
                return None
            return ToolDisplayProjection(
                action="calculate",
                target=call.name,
                summary="Calculate expression.",
            )

        setattr(projected_calc, "tool_display_projector", project_display)
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        manager = ToolManager.create_instance(
            enable_tools=["projected_calc"],
            available_toolsets=[ToolSet(tools=[projected_calc])],
            settings=ToolManagerSettings(),
        )
        call = ToolCall(
            id="call1",
            name="projected_calc",
            arguments={"expression": "2 + 2"},
        )
        agent.return_value = _string_response("done", async_gen=True)
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {},
            tool=manager,
        )

        result = await resp.to_str()

        self.assertEqual(result, "done")
        canonical_items = resp.canonical_items
        ready = next(
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        started = next(
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED
        )
        terminal = next(
            item
            for item in canonical_items
            if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
        )
        ready_projection = cast(
            dict[str, object],
            ready.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        start_projection = cast(
            dict[str, object],
            started.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        terminal_projection = cast(
            dict[str, object],
            terminal.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )

        self.assertEqual(ready_projection["action"], "calculate")
        self.assertEqual(start_projection["action"], "calculate")
        self.assertEqual(terminal_projection["action"], "finish")
        self.assertEqual(terminal_projection["status"], "completed")
        self.assertEqual(
            terminal.data,
            {
                "name": "projected_calc",
                "result": "4",
                "arguments": {"expression": "2 + 2"},
            },
        )
        validate_canonical_stream_items(canonical_items)
        validate_tool_lifecycle_items(canonical_items)

    async def test_ready_tool_item_keeps_existing_display_projection_metadata(
        self,
    ) -> None:
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
        metadata = {
            TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                "action": "existing",
                "summary": "Already projected.",
            }
        }

        item = resp._append_canonical_tool_call_lifecycle_item(
            StreamItemKind.TOOL_CALL_READY,
            data={
                "id": "ready-call",
                "name": "plain",
                "arguments": {"value": "ok"},
            },
            correlation=StreamItemCorrelation(tool_call_id="ready-call"),
            metadata=metadata,
        )

        projection = cast(
            dict[str, object],
            item.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        self.assertEqual(projection["action"], "existing")
        self.assertEqual(projection["summary"], "Already projected.")

    async def test_ready_tool_item_keeps_metadata_for_unprojectable_data(
        self,
    ) -> None:
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
        metadata = {"source": "fixture"}

        non_mapping = resp._tool_call_ready_display_metadata(
            "ready-call-1",
            "not a mapping",
            metadata,
        )
        missing_name = resp._tool_call_ready_display_metadata(
            "ready-call-2",
            {"id": "ready-call-2", "arguments": {"value": "ok"}},
            metadata,
        )

        self.assertEqual(non_mapping, metadata)
        self.assertEqual(missing_name, metadata)

    async def test_call_projection_accepts_keyword_only_projector(
        self,
    ) -> None:
        async def keyword_calc(expression: str) -> str:
            return expression

        invocations: list[tuple[object, ...]] = []

        def project_display(
            *items: object,
            call: ToolCall,
        ) -> ToolDisplayProjection:
            invocations.append(items)
            return ToolDisplayProjection(
                action="keyword",
                target=call.name,
                summary="Keyword-only projection.",
            )

        setattr(keyword_calc, "tool_display_projector", project_display)
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        manager = ToolManager.create_instance(
            enable_tools=["keyword_calc"],
            available_toolsets=[ToolSet(tools=[keyword_calc])],
            settings=ToolManagerSettings(),
        )
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            tool=manager,
        )
        call = ToolCall(
            id="keyword-call",
            name="keyword_calc",
            arguments={"expression": "2 + 2"},
        )

        resp._append_canonical_tool_execution_started(call)

        projection = cast(
            dict[str, object],
            resp.canonical_items[-1].metadata[
                TOOL_DISPLAY_PROJECTION_METADATA_KEY
            ],
        )
        self.assertEqual(projection["action"], "keyword")
        self.assertEqual(projection["target"], "keyword_calc")
        self.assertEqual(invocations, [()])

    async def test_call_projection_does_not_retry_after_projector_declines(
        self,
    ) -> None:
        async def guarded_calc(expression: str) -> str:
            return expression

        invocations: list[tuple[object, ...]] = []

        def project_display(
            *items: object,
            call: ToolCall | None = None,
        ) -> ToolDisplayProjection | None:
            invocations.append(items)
            if items:
                return None
            assert call is not None
            return ToolDisplayProjection(
                action="unexpected",
                target=call.name,
            )

        setattr(guarded_calc, "tool_display_projector", project_display)
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        manager = ToolManager.create_instance(
            enable_tools=["guarded_calc"],
            available_toolsets=[ToolSet(tools=[guarded_calc])],
            settings=ToolManagerSettings(),
        )
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            tool=manager,
        )
        call = ToolCall(
            id="guarded-call",
            name="guarded_calc",
            arguments={"expression": "2 + 2"},
        )

        resp._append_canonical_tool_execution_started(call)

        projection = cast(
            dict[str, object],
            resp.canonical_items[-1].metadata[
                TOOL_DISPLAY_PROJECTION_METADATA_KEY
            ],
        )
        self.assertEqual(len(invocations), 1)
        self.assertEqual(projection["action"], "call")

    async def test_call_projection_uses_fallback_for_unusable_projector(
        self,
    ) -> None:
        async def projected_tool(value: str) -> str:
            return value

        class BadSignatureToolDescriptor(ToolDescriptor):
            @property
            def display_projector(self) -> object:
                raise AssertionError("bad projector")

        class MissingDescribeToolManager:
            is_empty = False

        class NonCallableDescribeToolManager:
            is_empty = False
            describe_tool_call = "not callable"

        class RaisingDescribeToolManager:
            is_empty = False

            def describe_tool_call(self, call: ToolCall) -> ToolDescriptor:
                raise RuntimeError("boom")

        class InvalidDescribeToolManager:
            is_empty = False

            def describe_tool_call(self, call: ToolCall) -> object:
                return object()

        class AwaitableDescriptor:
            closed = False

            def __await__(self) -> Generator[None, None, object]:
                yield None
                return object()

            def close(self) -> None:
                self.closed = True

        class AwaitableDescribeToolManager:
            is_empty = False

            def __init__(self) -> None:
                self.descriptor = AwaitableDescriptor()

            def describe_tool_call(self, call: ToolCall) -> object:
                return self.descriptor

        class BadSignatureProjector:
            @property
            def __signature__(self) -> object:
                raise ValueError("bad signature")

            def __call__(self, call: ToolCall) -> ToolDisplayProjection:
                return ToolDisplayProjection(
                    action="bad-signature",
                    target=call.name,
                )

        def requires_unavailable_argument(
            *,
            missing: str,
        ) -> ToolDisplayProjection:
            return ToolDisplayProjection(action=missing)

        descriptor = BadSignatureToolDescriptor(
            name="projected_tool",
            namespace=None,
            schema={},
            capabilities=ToolCapabilities(),
            metadata={
                TOOL_DISPLAY_PROJECTOR_METADATA_KEY: projected_tool,
            },
        )
        call = ToolCall(
            id="fallback-call",
            name="projected_tool",
            arguments={"value": "ok"},
        )

        for manager in (
            MissingDescribeToolManager(),
            NonCallableDescribeToolManager(),
            RaisingDescribeToolManager(),
            InvalidDescribeToolManager(),
            AwaitableDescribeToolManager(),
        ):
            with self.subTest(manager=type(manager).__name__):
                resp = _make_response(
                    Message(role=MessageRole.USER, content="hi"),
                    _string_response("hi", async_gen=False),
                    MagicMock(spec=EngineAgent),
                    _dummy_operation(),
                    {},
                    tool=cast(Any, manager),
                )

                resp._append_canonical_tool_execution_started(call)

                projection = cast(
                    dict[str, object],
                    resp.canonical_items[-1].metadata[
                        TOOL_DISPLAY_PROJECTION_METADATA_KEY
                    ],
                )
                self.assertEqual(projection["action"], "call")
                if isinstance(manager, AwaitableDescribeToolManager):
                    self.assertTrue(manager.descriptor.closed)

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            MagicMock(spec=EngineAgent),
            _dummy_operation(),
            {},
        )
        self.assertIsNone(
            resp._project_tool_display(
                descriptor,
                call,
                call=call,
            )
        )
        keyword_only_descriptor = ToolDescriptor(
            name="projected_tool",
            metadata={
                TOOL_DISPLAY_PROJECTOR_METADATA_KEY: (
                    requires_unavailable_argument
                ),
            },
        )
        self.assertIsNone(
            resp._project_tool_display(
                keyword_only_descriptor,
                call,
                call=call,
            )
        )
        bad_signature_descriptor = ToolDescriptor(
            name="projected_tool",
            metadata={
                TOOL_DISPLAY_PROJECTOR_METADATA_KEY: BadSignatureProjector(),
            },
        )

        projection = resp._project_tool_display(
            bad_signature_descriptor,
            call,
            call=call,
        )

        self.assertIsInstance(projection, ToolDisplayProjection)
        assert isinstance(projection, ToolDisplayProjection)
        self.assertEqual(projection.action, "bad-signature")

    async def test_call_projection_accepts_tool_manager_subclass(self) -> None:
        async def subclass_calc(expression: str) -> str:
            return expression

        def project_display(call: ToolCall) -> ToolDisplayProjection:
            return ToolDisplayProjection(
                action="subclass",
                target=call.name,
                summary="Subclass manager projection.",
            )

        class SubclassToolManager(ToolManager):
            pass

        setattr(subclass_calc, "tool_display_projector", project_display)
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        manager = SubclassToolManager(
            enable_tools=["subclass_calc"],
            available_toolsets=[ToolSet(tools=[subclass_calc])],
            parser=ToolCallParser(),
            settings=ToolManagerSettings(),
        )
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            tool=manager,
        )
        call = ToolCall(
            id="subclass-call",
            name="subclass_calc",
            arguments={"expression": "2 + 2"},
        )

        resp._append_canonical_tool_execution_started(call)

        projection = cast(
            dict[str, object],
            resp.canonical_items[-1].metadata[
                TOOL_DISPLAY_PROJECTION_METADATA_KEY
            ],
        )
        self.assertEqual(projection["action"], "subclass")
        self.assertEqual(projection["target"], "subclass_calc")

    async def test_terminal_projection_does_not_retry_after_projector_declines(
        self,
    ) -> None:
        async def guarded_result(expression: str) -> str:
            return expression

        invocations: list[tuple[object, ...]] = []

        def project_display(
            *items: object,
            call: ToolCall | None = None,
            outcome: ToolCallResult | None = None,
        ) -> ToolDisplayProjection | None:
            invocations.append(items)
            if items:
                return None
            assert call is not None
            assert outcome is not None
            return ToolDisplayProjection(
                action="unexpected",
                target=outcome.name,
            )

        setattr(guarded_result, "tool_display_projector", project_display)
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        manager = ToolManager.create_instance(
            enable_tools=["guarded_result"],
            available_toolsets=[ToolSet(tools=[guarded_result])],
            settings=ToolManagerSettings(),
        )
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("hi", async_gen=False),
            agent,
            operation,
            {},
            tool=manager,
        )
        call = ToolCall(
            id="guarded-result",
            name="guarded_result",
            arguments={"expression": "2 + 2"},
        )

        completed = resp._append_canonical_tool_execution_terminal(
            call,
            ToolCallResult(
                id="result1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="4",
            ),
        )

        assert completed is not None
        projection = cast(
            dict[str, object],
            completed.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        self.assertEqual(len(invocations), 1)
        self.assertEqual(projection["action"], "finish")

    async def test_canonical_terminal_projection_metadata_uses_fallbacks(
        self,
    ) -> None:
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
        complete_call = ToolCall(
            id="complete-call",
            name="plain",
            arguments={"value": "ok"},
        )
        error_call = ToolCall(id="error-call", name="plain", arguments={})
        cancel_call = ToolCall(id="cancel-call", name="plain", arguments={})

        resp._append_canonical_item(StreamItemKind.STREAM_STARTED)
        resp._append_canonical_tool_execution_started(complete_call)
        completed = resp._append_canonical_tool_execution_terminal(
            complete_call,
            ToolCallResult(
                id="result1",
                call=complete_call,
                name=complete_call.name,
                arguments=complete_call.arguments,
                result={"ok": True},
            ),
        )
        resp._append_canonical_tool_execution_started(error_call)
        resp._append_canonical_tool_execution_error(
            error_call,
            RuntimeError("boom"),
        )
        errored = resp.canonical_items[-1]
        resp._append_canonical_tool_execution_started(cancel_call)
        resp._append_canonical_tool_execution_cancelled(cancel_call)
        cancelled = resp.canonical_items[-1]
        resp._finish_canonical_stream(
            StreamItemKind.STREAM_ERRORED,
            data={"message": "done"},
        )

        assert completed is not None
        completed_projection = cast(
            dict[str, object],
            completed.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        error_projection = cast(
            dict[str, object],
            errored.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        cancel_projection = cast(
            dict[str, object],
            cancelled.metadata[TOOL_DISPLAY_PROJECTION_METADATA_KEY],
        )
        self.assertEqual(completed_projection["action"], "finish")
        self.assertEqual(completed_projection["status"], "completed")
        self.assertEqual(error_projection["status"], "error")
        self.assertEqual(error_projection["severity"], "error")
        self.assertEqual(cancel_projection["action"], "skip")
        self.assertEqual(
            cancel_projection["outcome"],
            ToolCallDiagnosticCode.CANCELLED.value,
        )
        validate_canonical_stream_items(resp.canonical_items)

    async def test_to_str_preserves_json_schema_for_tool_continuation(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id=uuid4(), name="database.run", arguments=None)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {"type": "object"},
                "strict": True,
            },
        }

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.get_calls.side_effect = lambda text: (
            [call] if text == "call" else None
        )

        async def tool_exec(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolCallResult:
            return ToolCallResult(
                id=uuid4(),
                call=call,
                name=call.name,
                arguments=call.arguments,
                result={"rows": [{"answer": 4}]},
            )

        async def continue_agent(
            context: ModelCallContext,
        ) -> TextGenerationResponse:
            if context.engine_args.get("response_format") == response_format:
                return _openai_completed_message_response('{"answer":"4"}')
            return _empty_text_response()

        tool.side_effect = tool_exec
        agent.side_effect = continue_agent
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _string_response("call", async_gen=True),
            agent,
            operation,
            {"response_format": response_format},
            tool=tool,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()

        self.assertEqual(result, '{"answer":"4"}')
        agent.assert_awaited_once()
        context = agent.await_args.args[0]
        self.assertEqual(
            context.engine_args["response_format"],
            response_format,
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
        base_parser = ToolCallParser()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        tool.tool_call_status.side_effect = base_parser.tool_call_status
        tool.get_calls.side_effect = lambda text: (
            [call] if text == "call" else None
        )
        tool.tool_format = None

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
            enable_tool_parsing=False,
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

    async def test_to_str_remaps_reused_posthoc_tool_call_id(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        outer_response = _string_response("first-call", async_gen=True)
        second_response = _string_response("second-call", async_gen=True)
        final_response = _string_response("done", async_gen=True)
        agent.side_effect = [second_response, final_response]
        first_call = ToolCall(
            id="call1",
            name="calc",
            arguments={"x": 1},
        )
        second_call = ToolCall(
            id="call1",
            name="calc",
            arguments={"x": 2},
        )
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.parallel_tool_calls = False
        tool.get_calls.side_effect = lambda text: {
            "first-call": [first_call],
            "second-call": [second_call],
        }.get(text)

        async def tool_exec(
            call: ToolCall,
            context: ToolCallContext,
        ) -> ToolCallResult:
            arguments = cast(dict[str, int], call.arguments)
            return ToolCallResult(
                id=f"result-{arguments['x']}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=str(arguments["x"]),
            )

        tool.side_effect = tool_exec
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            outer_response,
            agent,
            operation,
            {},
            tool=tool,
            enable_tool_parsing=False,
        )

        result = await resp.to_str()

        self.assertEqual(result, "done")
        self.assertEqual(tool.await_count, 2)
        self.assertEqual(agent.await_count, 2)
        self.assertEqual(
            [call_args.args[0].id for call_args in tool.await_args_list],
            ["call1", "orchestrator-tool-call-1"],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in resp.canonical_items
                if item.kind is StreamItemKind.TOOL_CALL_READY
            ],
            ["call1", "orchestrator-tool-call-1"],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in resp.canonical_items
                if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED
            ],
            ["call1", "orchestrator-tool-call-1"],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in resp.canonical_items
                if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
            ],
            ["call1", "orchestrator-tool-call-1"],
        )
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

        outer_response = _tool_call_response(
            ('{"expression":"2 + ', call),
            ('2"}', call),
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
            ['{"expression":"2 + ', '2"}'],
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
        self.assertEqual(
            diagnostic_payload, diagnostic_event.observability.data
        )
        self.assertEqual(result_payload, result_events[0].observability.data)
        self.assertEqual(
            diagnostic_payload["correlation"]["tool_call_id"],
            "call1",
        )
        self.assertEqual(
            result_payload["correlation"]["tool_call_id"],
            "call1",
        )
        self.assertEqual(
            diagnostic_payload["kind"],
            StreamItemKind.TOOL_EXECUTION_ERROR.value,
        )
        self.assertEqual(
            result_payload["kind"],
            StreamItemKind.TOOL_EXECUTION_ERROR.value,
        )
        self.assertNotIn("call", diagnostic_payload)
        self.assertNotIn("result", result_payload)

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
        inner_call = ToolCall(
            id="call2",
            name="missing",
            arguments={"value": "x"},
        )

        outer_response = _tool_call_response(call)
        inner_response = _tool_call_response(inner_call)
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
        signature = resp._call_signature(call)
        self.assertEqual(second.code, ToolCallDiagnosticCode.REPEATED_CALL)
        self.assertEqual(second.stage, ToolCallDiagnosticStage.GUARD)
        self.assertEqual(second.details["signature"], signature)
        self.assertIn(signature, second.details["attempted_call_signatures"])
        resp._append_canonical_tool_execution_started(call)
        resp._append_canonical_tool_execution_terminal(call, second)
        terminal = resp.canonical_items[-1]
        self.assertIs(terminal.kind, StreamItemKind.TOOL_EXECUTION_ERROR)
        self.assertEqual(terminal.data["details"]["signature"], signature)
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
        limited._tool_cycle_count = 24
        self.assertFalse(
            limited._should_continue_tool_cycle(messages, [result])
        )
        self.assertEqual(
            limited.canonical_items[-1].data["code"],
            "orchestrator.tool_cycle.limit_exceeded",
        )
        self.assertEqual(
            limited.canonical_items[-1].data["details"]["maximum_cycles"], 24
        )

    async def test_iteration_stops_at_configured_maximum_tool_cycles(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        first_call = ToolCall(id="call1", name="calc", arguments={"value": 1})
        second_call = ToolCall(id="call2", name="calc", arguments={"value": 2})
        agent.return_value = _tool_call_response(second_call)
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def execute(
            call: ToolCall, _: ToolCallContext
        ) -> ToolCallResult:
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=f"ok-{call.id}",
            )

        tool.side_effect = execute
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(first_call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            maximum_tool_cycles=1,
        )

        items = []
        async for item in resp:
            items.append(item)

        self.assertFalse(
            any(item.kind is StreamItemKind.ANSWER_DELTA for item in items)
        )
        self.assertEqual(agent.await_count, 1)
        self.assertEqual(tool.await_count, 2)
        diagnostics = [
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertEqual(
            diagnostics[-1].data["code"],
            "orchestrator.tool_cycle.limit_exceeded",
        )
        self.assertEqual(diagnostics[-1].data["details"]["maximum_cycles"], 1)
        validate_canonical_stream_items(items)

    async def test_iteration_allows_configured_extra_tool_cycles(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        first_call = ToolCall(id="call1", name="calc", arguments={"value": 1})
        second_call = ToolCall(id="call2", name="calc", arguments={"value": 2})
        agent.side_effect = [
            _tool_call_response(second_call),
            _response_from_items(
                *_canonical_answer_items("final json"),
            ),
        ]
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False

        async def execute(
            call: ToolCall, _: ToolCallContext
        ) -> ToolCallResult:
            return ToolCallResult(
                id=f"result-{call.id}",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=f"ok-{call.id}",
            )

        tool.side_effect = execute
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(first_call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            maximum_tool_cycles=2,
        )

        items = []
        async for item in resp:
            items.append(item)

        self.assertTrue(
            any(
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta == "final json"
                for item in items
            )
        )
        self.assertEqual(agent.await_count, 2)
        self.assertEqual(tool.await_count, 2)
        self.assertFalse(
            any(
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                and item.data
                and item.data.get("code")
                == "orchestrator.tool_cycle.limit_exceeded"
                for item in items
            )
        )
        validate_canonical_stream_items(items)

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

        outer_response = _tool_call_response(
            (
                dumps(first_call.arguments),
                first_call,
            )
        )
        first_inner_response = _tool_call_response(
            (
                dumps(second_call.arguments),
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

    async def test_iteration_allows_reused_provider_call_id_across_turns(
        self,
    ):
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
            id="call1",
            name="calc",
            arguments={"expression": "50 * 2"},
        )

        agent.side_effect = [
            _tool_call_response((dumps(second_call.arguments), second_call)),
            _string_response("done", async_gen=True),
        ]

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        observed_arguments: list[dict[str, Any] | None] = []

        async def tool_exec(
            call: ToolCall,
            _: ToolCallContext,
        ) -> ToolCallResult:
            observed_arguments.append(call.arguments)
            result = "50" if call.arguments == first_call.arguments else "100"
            return ToolCallResult(
                id=f"{len(observed_arguments)}-result",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result=result,
            )

        tool.side_effect = tool_exec

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response((dumps(first_call.arguments), first_call)),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        items = await _collect_stream_items(resp)

        self.assertEqual(_answer_text(items), "done")
        self.assertEqual(
            observed_arguments,
            [first_call.arguments, second_call.arguments],
        )
        self.assertEqual(tool.await_count, 2)
        self.assertEqual(agent.await_count, 2)
        diagnostic_codes = [
            item.data["code"]
            for item in resp.canonical_items
            if (
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                and isinstance(item.data, dict)
            )
        ]
        self.assertNotIn(
            "orchestrator.tool_call.argument_after_ready",
            diagnostic_codes,
        )
        self.assertNotIn(
            "orchestrator.tool_call.invalid_lifecycle",
            diagnostic_codes,
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in resp.canonical_items
                if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
            ],
            ["call1", "orchestrator-tool-call-1"],
        )
        child_contexts = [
            call_args.args[0] for call_args in agent.await_args_list
        ]
        tool_call_observations = [
            [
                message.tool_calls[0].id
                for message in cast(list[Message], context.input)
                if message.tool_calls
            ]
            for context in child_contexts
        ]
        tool_result_observations = [
            [
                message.content
                for message in cast(list[Message], context.input)
                if message.role is MessageRole.TOOL
            ]
            for context in child_contexts
        ]
        self.assertEqual(
            tool_call_observations,
            [["call1"], ["call1", "orchestrator-tool-call-1"]],
        )
        self.assertEqual(
            tool_result_observations,
            [['"50"'], ['"50"', '"100"']],
        )
        validate_canonical_stream_items(resp.canonical_items)
        validate_tool_lifecycle_items(resp.canonical_items)

    async def test_iteration_repeated_tool_skip_continues_to_model(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()

        call = ToolCall(
            id="call1",
            name="calc",
            arguments={"expression": "25 * 2"},
        )
        repeated_call = ToolCall(
            id="call2",
            name="calc",
            arguments={"expression": "25 * 2"},
        )
        agent.side_effect = [
            _tool_call_response(repeated_call),
            _string_response("done", async_gen=True),
        ]

        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        tool.return_value = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="50",
        )

        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _tool_call_response(call),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        items = await _collect_stream_items(resp)

        self.assertEqual(_answer_text(items), "done")
        tool.assert_awaited_once()
        self.assertEqual(agent.await_count, 2)

        first_context = agent.await_args_list[0].args[0]
        second_context = agent.await_args_list[1].args[0]
        assert isinstance(first_context.input, list)
        assert isinstance(second_context.input, list)
        first_tool_messages = [
            message
            for message in first_context.input
            if message.role is MessageRole.TOOL
        ]
        second_tool_messages = [
            message
            for message in second_context.input
            if message.role is MessageRole.TOOL
        ]
        self.assertEqual(len(first_tool_messages), 1)
        self.assertEqual(len(second_tool_messages), 2)
        self.assertIsNone(second_tool_messages[-1].tool_call_result)
        diagnostic = second_tool_messages[-1].tool_call_diagnostic
        self.assertIsNotNone(diagnostic)
        assert diagnostic is not None
        self.assertEqual(
            diagnostic.code,
            ToolCallDiagnosticCode.REPEATED_CALL,
        )
        payload = loads(str(second_tool_messages[-1].content))
        self.assertEqual(payload["code"], "tool_call.repeated")

        validate_canonical_stream_items(resp.canonical_items)
        validate_tool_lifecycle_items(resp.canonical_items)


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
            {"max_new_tokens": 5, "tool_choice": "mcp.call"},
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
        self.assertEqual(child.engine_args["max_new_tokens"], 5)
        self.assertNotIn("tool_choice", child.engine_args)


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
