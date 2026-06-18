from asyncio import CancelledError, create_task, sleep
from collections.abc import AsyncIterator
from json import dumps
from logging import getLogger
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from avalan.agent import AgentOperation, EngineEnvironment, Specification
from avalan.agent.engine import EngineAgent
from avalan.agent.orchestrator.response.orchestrator_response import (
    LegacyToolEventShim,
    OrchestratorResponse,
    _ToolExecutionOutcome,
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
    TokenDetail,
    ToolCall,
    ToolCallContext,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    TransformerEngineSettings,
)
from avalan.event import TOOL_TYPES, Event, EventPayloadKind, EventType
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
from avalan.tool.parser import ToolCallParser


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


def _canonical_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    correlation: StreamItemCorrelation | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
) -> CanonicalStreamItem:
    outcomes = {
        StreamItemKind.STREAM_COMPLETED: StreamTerminalOutcome.COMPLETED,
        StreamItemKind.STREAM_ERRORED: StreamTerminalOutcome.ERRORED,
        StreamItemKind.STREAM_CANCELLED: StreamTerminalOutcome.CANCELLED,
    }
    return CanonicalStreamItem(
        stream_session_id="additional-stream",
        run_id="additional-run",
        turn_id="additional-turn",
        sequence=sequence,
        kind=kind,
        channel=stream_channel_for_kind(kind),
        text_delta=text_delta,
        data=cast(Any, data),
        usage=cast(Any, usage),
        correlation=correlation or StreamItemCorrelation(),
        terminal_outcome=terminal_outcome or outcomes.get(kind),
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
    return tuple(items)


def _canonical_tool_call_items(
    call: ToolCall,
    *,
    tool_call_id: str | None = None,
) -> tuple[CanonicalStreamItem, ...]:
    resolved_tool_call_id = tool_call_id or str(call.id)
    correlation = StreamItemCorrelation(tool_call_id=resolved_tool_call_id)
    arguments = call.arguments or {}
    items = [_canonical_item(StreamItemKind.STREAM_STARTED, 0)]
    sequence = 1
    if arguments:
        items.append(
            CanonicalStreamItem(
                stream_session_id="additional-stream",
                run_id="additional-run",
                turn_id="additional-turn",
                sequence=sequence,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta=dumps(arguments),
                correlation=correlation,
            )
        )
        sequence += 1
    items.append(
        CanonicalStreamItem(
            stream_session_id="additional-stream",
            run_id="additional-run",
            turn_id="additional-turn",
            sequence=sequence,
            kind=StreamItemKind.TOOL_CALL_READY,
            channel=StreamChannel.TOOL_CALL,
            data={"name": call.name, "arguments": arguments},
            correlation=correlation,
        )
    )
    sequence += 1
    items.append(
        CanonicalStreamItem(
            stream_session_id="additional-stream",
            run_id="additional-run",
            turn_id="additional-turn",
            sequence=sequence,
            kind=StreamItemKind.TOOL_CALL_DONE,
            channel=StreamChannel.TOOL_CALL,
            correlation=correlation,
        )
    )
    sequence += 1
    items.append(
        _canonical_item(StreamItemKind.STREAM_COMPLETED, sequence, usage={})
    )
    return tuple(items)


def _response_from_items(
    *items: CanonicalStreamItem,
    async_gen: bool = True,
) -> TextGenerationResponse:
    async def output_gen():
        for item in items:
            yield item

    def output_fn():
        return output_gen()

    return TextGenerationResponse(
        output_fn, logger=getLogger(), use_async_generator=async_gen
    )


def _dummy_response(async_gen: bool = True) -> TextGenerationResponse:
    if not async_gen:
        return TextGenerationResponse(
            lambda: "ab", logger=getLogger(), use_async_generator=False
        )
    return _response_from_items(
        *_canonical_answer_items("a", "b"), async_gen=async_gen
    )


def _empty_response() -> TextGenerationResponse:
    return _response_from_items()


class _LegacyFixtureResponse:
    is_async_generator = True
    input_token_count = 0
    output_token_count = 0
    usage = None

    def __init__(self, *items: object) -> None:
        self._items = items

    def add_done_callback(self, _: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[object]:
        return self._output_gen()

    async def _output_gen(self) -> AsyncIterator[object]:
        for item in self._items:
            yield item


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


_LEGACY_PUBLIC_STREAM_TYPES = (Token, TokenDetail, ToolCallToken, Event)


async def _collect_items(
    response: OrchestratorResponse,
    *,
    limit: int = 100,
) -> list[CanonicalStreamItem]:
    items: list[CanonicalStreamItem] = []
    iterator = response.__aiter__()
    for _ in range(limit):
        try:
            item = await iterator.__anext__()
        except StopAsyncIteration:
            return items
        assert isinstance(item, CanonicalStreamItem)
        assert not isinstance(item, _LEGACY_PUBLIC_STREAM_TYPES)
        items.append(item)
    raise AssertionError("orchestrator response did not finish")


async def _collect_until_kind(
    response: OrchestratorResponse,
    kind: StreamItemKind,
    *,
    limit: int = 100,
) -> list[CanonicalStreamItem]:
    items: list[CanonicalStreamItem] = []
    for _ in range(limit):
        item = await response.__anext__()
        assert isinstance(item, CanonicalStreamItem)
        assert not isinstance(item, _LEGACY_PUBLIC_STREAM_TYPES)
        items.append(item)
        if item.kind is kind:
            return items
    raise AssertionError(f"orchestrator response did not emit {kind.value}")


class OrchestratorResponseAdditionalCoverageTestCase(IsolatedAsyncioTestCase):
    async def test_response_item_normalization_is_canonical_only(self):
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

        self.assertEqual(
            response._canonical_item_from_response_item(canonical_item),
            canonical_item,
        )
        self.assertEqual(
            response._canonical_item_from_response_item(projection_item),
            canonical_item,
        )
        for legacy_item in (
            "raw",
            Token(token="raw"),
            TokenDetail(id=1, token="raw", probability=0.5),
            ToolCallToken(token="raw"),
            Event(type=EventType.TOOL_PROCESS),
            Event(type=EventType.END),
            object(),
        ):
            with self.subTest(legacy_item=type(legacy_item).__qualname__):
                with self.assertRaises(StreamValidationError):
                    response._canonical_item_from_response_item(legacy_item)

    async def test_direct_iteration_rejects_legacy_first_item(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()

        for legacy_item in (
            "legacy",
            Token(token="legacy"),
            TokenDetail(id=1, token="legacy", probability=0.5),
            ToolCallToken(token="legacy"),
            Event(type=EventType.TOOL_PROCESS),
            Event(type=EventType.END),
        ):
            with self.subTest(legacy_item=type(legacy_item).__qualname__):
                legacy_response = cast(
                    TextGenerationResponse,
                    _LegacyFixtureResponse(legacy_item),
                )
                response = _make_response(
                    Message(role=MessageRole.USER, content="hi"),
                    legacy_response,
                    agent,
                    operation,
                    {},
                )
                iterator = response.__aiter__()

                first = await iterator.__anext__()
                self.assertIs(first.kind, StreamItemKind.STREAM_STARTED)
                with self.assertRaises(StreamValidationError):
                    await iterator.__anext__()

    async def test_provider_events_are_staged_as_canonical_lifecycle(self):
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
        ready_event = StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_READY,
            correlation=StreamItemCorrelation(
                provider_request_id="provider-request-1",
                model_continuation_id="continuation-1",
                tool_call_id="call-1",
                task_id="task-1",
            ),
            data={"name": "calc", "arguments": {"x": 1}},
        )
        argument_correlation = StreamItemCorrelation(
            provider_request_id="provider-request-1",
            model_continuation_id="continuation-1",
            tool_call_id="call-1",
            task_id="task-1",
        )

        response.__aiter__()
        response._queue_parser_output(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta='{"x":1}',
                correlation=argument_correlation,
            )
        )
        response._queue_parser_output(ready_event)
        lifecycle_items = [
            item
            for item in response.canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
        ]
        self.assertEqual(
            [item.kind for item in lifecycle_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
            ],
        )
        self.assertEqual(lifecycle_items[0].text_delta, '{"x":1}')
        self.assertEqual(
            lifecycle_items[0].correlation,
            argument_correlation,
        )
        self.assertEqual(
            lifecycle_items[1].correlation,
            ready_event.correlation,
        )
        self.assertTrue(response._calls.empty())

        done_correlation = StreamItemCorrelation(
            provider_request_id="provider-request-1",
            model_continuation_id="continuation-1",
            tool_call_id="call-1",
            task_id="task-1",
        )
        response._queue_parser_output(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=done_correlation,
            )
        )

        self.assertTrue(response._parser_queue.empty())
        self.assertFalse(response._calls.empty())
        lifecycle_items = [
            item
            for item in response.canonical_items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
            }
        ]
        self.assertEqual(
            lifecycle_items[-1].kind,
            StreamItemKind.TOOL_CALL_DONE,
        )
        self.assertEqual(
            lifecycle_items[-1].correlation,
            done_correlation,
        )

    async def test_parser_flush_bridges_provider_ready_and_done_events(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
        )
        response._tool_parser = AsyncMock()
        response._tool_parser.flush.return_value = [
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_READY,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
                data={"name": "calc", "arguments": {"x": 1}},
            ),
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            ),
        ]
        response.__aiter__()

        items = await _collect_until_kind(
            response,
            StreamItemKind.TOOL_CALL_DONE,
        )
        item = items[-1]

        self.assertIs(item.kind, StreamItemKind.TOOL_CALL_DONE)
        self.assertEqual(item.correlation.tool_call_id, "call-1")
        ready = next(
            item
            for item in items
            if item.kind is StreamItemKind.TOOL_CALL_READY
        )
        self.assertEqual(ready.data, {"name": "calc", "arguments": {"x": 1}})
        self.assertTrue(all(not isinstance(item, Event) for item in items))

    async def test_provider_flush_lifecycle_executes_with_full_correlation(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _response_from_items(
            *_canonical_answer_items("done")
        )
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        full_correlation = StreamItemCorrelation(
            provider_request_id="provider-request-1",
            model_continuation_id="continuation-1",
            tool_call_id="provider-call-1",
            flow_run_id="flow-1",
            node_id="node-1",
            parent_sequence=0,
            protocol_item_id="protocol-1",
            task_id="task-1",
            artifact_id="artifact-1",
        )

        async def execute_tool(
            call: ToolCall,
            _: ToolCallContext,
        ) -> ToolCallResult:
            return ToolCallResult(
                id="result-1",
                call=call,
                name=call.name,
                arguments=call.arguments,
                result="3",
            )

        tool.side_effect = execute_tool
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        response._tool_parser = AsyncMock()
        response._tool_parser.flush.side_effect = [
            [
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    text_delta='{"x":3}',
                    correlation=full_correlation,
                ),
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_READY,
                    correlation=full_correlation,
                    data={"name": "calc", "arguments": {"ignored": True}},
                ),
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_DONE,
                    correlation=full_correlation,
                ),
            ],
            [],
        ]

        items = await _collect_items(response)

        tool.assert_awaited_once()
        executed_call = tool.await_args.args[0]
        self.assertEqual(executed_call.id, "provider-call-1")
        self.assertEqual(executed_call.name, "calc")
        self.assertEqual(executed_call.arguments, {"x": 3})
        lifecycle_items = [
            item
            for item in items
            if item.kind
            in {
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            }
        ]
        self.assertEqual(
            [item.kind for item in lifecycle_items],
            [
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
                StreamItemKind.TOOL_CALL_DONE,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ],
        )
        self.assertTrue(
            all(
                item.correlation == full_correlation
                for item in lifecycle_items
            )
        )
        triggered_types = [
            call_args.args[0].type
            for call_args in event_manager.trigger.await_args_list
        ]
        self.assertNotIn(EventType.TOOL_PROCESS, triggered_types)
        validate_canonical_stream_items(response.canonical_items)
        validate_tool_lifecycle_items(response.canonical_items)

    async def test_provider_missing_delta_lifecycle_does_not_execute(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _response_from_items(
            *_canonical_answer_items("done")
        )
        operation = _dummy_operation()
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        correlation = StreamItemCorrelation(
            provider_request_id="provider-request-1",
            model_continuation_id="continuation-1",
            tool_call_id="provider-call-1",
            task_id="task-1",
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
            tool=tool,
        )
        response._tool_parser = AsyncMock()
        response._tool_parser.flush.side_effect = [
            [
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    correlation=correlation,
                ),
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_READY,
                    correlation=correlation,
                    data={"name": "calc", "arguments": {"x": 3}},
                ),
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_DONE,
                    correlation=correlation,
                ),
            ],
            [],
        ]

        items = await _collect_items(response)

        tool.assert_not_awaited()
        diagnostics = [
            item
            for item in items
            if item.kind is StreamItemKind.STREAM_DIAGNOSTIC
        ]
        self.assertTrue(
            any(
                item.data["code"]
                == "orchestrator.tool_call.missing_argument_delta"
                for item in diagnostics
            )
        )
        self.assertNotIn(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            [item.kind for item in items],
        )
        self.assertTrue(
            all(
                item.correlation.provider_request_id == "provider-request-1"
                for item in diagnostics
            )
        )
        validate_canonical_stream_items(response.canonical_items)

    async def test_provider_tool_call_missing_id_emits_diagnostic(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
        )

        first = response._append_canonical_provider_event_item(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_READY,
                data={"name": "calc", "arguments": {"x": 1}},
            )
        )
        second = response._append_canonical_provider_event_item(
            StreamProviderEvent(kind=StreamItemKind.TOOL_CALL_DONE)
        )

        assert first is not None
        assert second is not None
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.STREAM_DIAGNOSTIC,
                StreamItemKind.STREAM_DIAGNOSTIC,
            ],
        )
        self.assertEqual(
            [
                item.correlation.tool_call_id
                for item in response.canonical_items
            ],
            [
                "orchestrator-tool-call-diagnostic-1",
                "orchestrator-tool-call-diagnostic-2",
            ],
        )
        self.assertTrue(
            all(
                item.data["code"] == "orchestrator.tool_call.missing_id"
                for item in response.canonical_items
            )
        )
        self.assertTrue(response._calls.empty())

    async def test_answer_delta_parser_with_no_items_suppresses_raw_delta(
        self,
    ):
        base_parser = ToolCallParser()
        manager = MagicMock()
        manager.tool_format = None
        manager.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.get_calls.side_effect = base_parser
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
        response._tool_parser = ToolCallResponseParser(manager, None)

        await response._process_canonical_response_item(
            _canonical_item(
                StreamItemKind.ANSWER_DELTA,
                0,
                text_delta="<tool_call",
            )
        )

        self.assertEqual([item.kind for item in response.canonical_items], [])

    async def test_response_text_and_calls_rejects_legacy_tool_tokens(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id="call-1", name="calc", arguments=None)
        legacy_response = _LegacyFixtureResponse(
            ToolCallToken(token='{"x":', call=None),
            ToolCallToken(token="1}", call=call),
        )
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            cast(TextGenerationResponse, legacy_response),
            agent,
            operation,
            {},
        )

        with self.assertRaises(StreamValidationError):
            await response._response_text_and_calls(
                cast(TextGenerationResponse, legacy_response)
            )

    async def test_source_tool_lifecycle_items_are_not_duplicated(self):
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
        response._append_canonical_item(StreamItemKind.STREAM_STARTED)
        response._append_canonical_response_item(
            CanonicalStreamItem(
                stream_session_id="inner",
                run_id="inner-run",
                turn_id="inner-turn",
                sequence=0,
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                channel=StreamChannel.TOOL_CALL,
                text_delta='{"x":1}',
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            )
        )
        response._append_canonical_response_item(
            CanonicalStreamItem(
                stream_session_id="inner",
                run_id="inner-run",
                turn_id="inner-turn",
                sequence=1,
                kind=StreamItemKind.TOOL_CALL_READY,
                channel=StreamChannel.TOOL_CALL,
                data={"name": "calc", "arguments": {"x": 1}},
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            )
        )
        items_before = response.canonical_items

        response._append_canonical_tool_call_ready(call)

        self.assertEqual(response.canonical_items, items_before)
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                StreamItemKind.TOOL_CALL_READY,
            ],
        )

    async def test_provider_argument_event_appends_canonical_item(self):
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

        response._append_canonical_provider_event_item(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta='{"x":1}',
                correlation=StreamItemCorrelation(tool_call_id="call-1"),
            )
        )

        items = response.canonical_items
        self.assertEqual(len(items), 1)
        self.assertEqual(
            items[0].kind,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        )
        self.assertEqual(items[0].text_delta, '{"x":1}')
        self.assertEqual(items[0].correlation.tool_call_id, "call-1")
        self.assertEqual(items[0].sequence, 0)

    async def test_provider_argument_event_missing_delta_is_diagnostic(self):
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

        item = response._append_canonical_provider_event_item(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=StreamItemCorrelation(
                    provider_request_id="provider-request-1",
                    model_continuation_id="continuation-1",
                    tool_call_id="call-1",
                    task_id="task-1",
                ),
            )
        )

        assert item is not None
        self.assertIs(item.kind, StreamItemKind.STREAM_DIAGNOSTIC)
        self.assertEqual(
            item.data["code"],
            "orchestrator.tool_call.missing_argument_delta",
        )
        self.assertEqual(item.correlation.tool_call_id, "call-1")
        self.assertEqual(
            item.correlation.provider_request_id,
            "provider-request-1",
        )
        self.assertEqual(
            item.correlation.model_continuation_id,
            "continuation-1",
        )
        self.assertEqual(item.correlation.task_id, "task-1")
        self.assertTrue(response._calls.empty())

    async def test_provider_missing_delta_uses_current_canonical_id(self):
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
        response._append_canonical_tool_call_ready(
            ToolCall(id="call-1", name="calc", arguments={"x": 1})
        )
        response._begin_tool_call_lifecycle_response()

        item = response._append_canonical_provider_event_item(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=StreamItemCorrelation(
                    provider_request_id="provider-request-1",
                    model_continuation_id="continuation-1",
                    tool_call_id="call-1",
                    task_id="task-1",
                ),
            )
        )

        assert item is not None
        self.assertIs(item.kind, StreamItemKind.STREAM_DIAGNOSTIC)
        self.assertEqual(
            item.data["code"],
            "orchestrator.tool_call.missing_argument_delta",
        )
        self.assertEqual(
            item.correlation.tool_call_id,
            "orchestrator-tool-call-1",
        )
        self.assertEqual(
            item.data["tool_call_id"],
            "orchestrator-tool-call-1",
        )
        self.assertEqual(
            item.correlation.provider_request_id,
            "provider-request-1",
        )
        self.assertEqual(
            item.correlation.model_continuation_id,
            "continuation-1",
        )
        self.assertEqual(item.correlation.task_id, "task-1")
        self.assertTrue(response._calls.empty())

    async def test_finish_canonical_stream_closes_open_reasoning_channel(self):
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

        response._append_canonical_item(StreamItemKind.STREAM_STARTED)
        response._append_canonical_item(
            StreamItemKind.REASONING_DELTA,
            text_delta="thinking",
        )
        response._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)

        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        validate_canonical_stream_items(response.canonical_items)

    async def test_iteration_rejects_raw_response_event_after_start(self):
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

        iterator = response.__aiter__()
        item = await iterator.__anext__()

        self.assertIs(item.kind, StreamItemKind.STREAM_STARTED)
        with self.assertRaises(StreamValidationError):
            await iterator.__anext__()

    async def test_legacy_tool_event_shim_inventory_covers_tool_events(self):
        inventory = legacy_tool_event_shim_inventory()

        self.assertEqual(
            {shim.event_type for shim in inventory},
            TOOL_TYPES - {EventType.TOOL_PROCESS},
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
        with self.assertRaises(StreamValidationError):
            classify_legacy_tool_event_shim(EventType.TOOL_PROCESS)
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

    async def test_response_text_and_calls_rejects_legacy_events(self):
        class Response:
            is_async_generator = True

            def __aiter__(self):
                return self.output_gen()

            async def output_gen(self):
                yield Event(type=EventType.TOOL_DETECT)

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

        with self.assertRaises(StreamValidationError):
            await response._response_text_and_calls(
                cast(TextGenerationResponse, Response())
            )

    async def test_response_text_and_calls_emits_token_events_for_answers(
        self,
    ):
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
            _response_from_items(*_canonical_answer_items("a", "b")),
            agent,
            operation,
            {},
            event_manager=event_manager,
        )

        text, calls = await response._response_text_and_calls(
            response._response
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

    async def test_response_text_and_calls_custom_parser_keeps_answer_text(
        self,
    ):
        engine = _DummyEngine()
        engine.tokenizer.encode.return_value = [42]
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _response_from_items(*_canonical_answer_items("a", "b")),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        response._tool_parser = AsyncMock()
        response._tool_parser.push.return_value = []
        response._tool_parser.flush.return_value = []

        text, calls = await response._response_text_and_calls(
            response._response
        )

        self.assertEqual(text, "ab")
        self.assertEqual(calls, [])
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DELTA],
        )
        response._tool_parser.push.assert_not_called()

    async def test_response_text_and_calls_flush_appends_parser_items(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _response_from_items(*_canonical_answer_items("a")),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        response._tool_parser = AsyncMock()
        response._tool_parser.push.return_value = []
        response._tool_parser.flush.return_value = [
            _canonical_item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                0,
                data={"code": "parser.flush"},
            )
        ]

        text, calls = await response._response_text_and_calls(
            response._response
        )

        self.assertEqual(text, "a")
        self.assertEqual(calls, [])
        self.assertEqual(
            [item.kind for item in response.canonical_items],
            [
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.STREAM_DIAGNOSTIC,
            ],
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

    async def test_response_text_and_calls_parses_raw_tool_syntax(self):
        base_parser = ToolCallParser()
        manager = MagicMock()
        manager.tool_format = None
        manager.is_potential_tool_call.side_effect = (
            base_parser.is_potential_tool_call
        )
        manager.tool_call_status.side_effect = base_parser.tool_call_status
        manager.get_calls.side_effect = base_parser
        tool_text = (
            'before <tool_call>{"name": "calc", '
            '"arguments": {"x": 1}}</tool_call> after'
        )
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _response_from_items(*_canonical_answer_items(tool_text)),
            agent,
            operation,
            {},
            enable_tool_parsing=False,
        )
        response._tool_parser = ToolCallResponseParser(manager, None)

        text, calls = await response._response_text_and_calls(
            response._response
        )

        self.assertEqual(text, "before  after")
        self.assertNotIn("<tool_call", text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "calc")
        self.assertEqual(calls[0].arguments, {"x": 1})
        self.assertNotIn(
            "<tool_call",
            "".join(
                item.text_delta or ""
                for item in response.canonical_items
                if item.kind is StreamItemKind.ANSWER_DELTA
            ),
        )

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
            semantic_response,
            agent,
            operation,
            {},
        )

        items = await _collect_items(response)

        self.assertEqual(
            [item.kind for item in items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        self.assertEqual(items[1].text_delta, "semantic")
        self.assertEqual(
            [item.sequence for item in items],
            list(range(len(items))),
        )
        self.assertEqual(
            {item.stream_session_id for item in items},
            {items[0].stream_session_id},
        )
        self.assertNotEqual(items[1].stream_session_id, "stream")

    async def test_direct_iteration_returns_canonical_tool_continuation_items(
        self,
    ):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        agent.return_value = _response_from_items(
            *_canonical_answer_items("ok")
        )
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        event_manager.should_emit.return_value = True
        tool = AsyncMock(spec=ToolManager)
        tool.is_empty = False
        call = ToolCall(id="call-1", name="calc", arguments={"x": 1})

        async def execute_tool(
            executed_call: ToolCall,
            _: ToolCallContext,
        ) -> ToolCallResult:
            return ToolCallResult(
                id="result-1",
                call=executed_call,
                name=executed_call.name,
                arguments=executed_call.arguments,
                result="1",
            )

        tool.side_effect = execute_tool
        response = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _response_from_items(*_canonical_tool_call_items(call)),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )

        items = await _collect_items(response)

        kinds = [item.kind for item in items]
        self.assertEqual(kinds[0], StreamItemKind.STREAM_STARTED)
        self.assertEqual(
            kinds[-2:],
            [StreamItemKind.STREAM_COMPLETED, StreamItemKind.STREAM_CLOSED],
        )
        self.assertIn(StreamItemKind.TOOL_CALL_READY, kinds)
        self.assertIn(StreamItemKind.TOOL_EXECUTION_COMPLETED, kinds)
        self.assertIn(StreamItemKind.MODEL_CONTINUATION_STARTED, kinds)
        self.assertIn(StreamItemKind.MODEL_CONTINUATION_COMPLETED, kinds)
        self.assertIn(StreamItemKind.ANSWER_DELTA, kinds)
        self.assertLess(
            kinds.index(StreamItemKind.TOOL_EXECUTION_COMPLETED),
            kinds.index(StreamItemKind.MODEL_CONTINUATION_STARTED),
        )
        self.assertLess(
            kinds.index(StreamItemKind.MODEL_CONTINUATION_COMPLETED),
            kinds.index(StreamItemKind.STREAM_COMPLETED),
        )
        self.assertEqual(
            [item.sequence for item in items],
            list(range(len(items))),
        )
        self.assertEqual(
            {item.stream_session_id for item in items},
            {items[0].stream_session_id},
        )
        self.assertNotIn(
            "additional-stream",
            {item.stream_session_id for item in items},
        )
        tool.assert_awaited_once()
        agent.assert_awaited_once()
        triggered_types = [
            call_args.args[0].type
            for call_args in event_manager.trigger.await_args_list
        ]
        self.assertNotIn(EventType.TOOL_PROCESS, triggered_types)
        self.assertIn(EventType.TOOL_RESULT, triggered_types)
        self.assertIn(EventType.TOOL_MODEL_RESPONSE, triggered_types)

    async def test_queue_parser_output_rejects_tool_process_event(self):
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
        call = ToolCall(id="call-1", name="calc", arguments=None)
        event = Event(type=EventType.TOOL_PROCESS, payload=[call])

        with self.assertRaises(StreamValidationError):
            resp._queue_parser_output(event)
        self.assertTrue(resp._calls.empty())

    async def test_legacy_flush_items_with_tool_manager_never_execute(self):
        engine = _DummyEngine()
        agent = AsyncMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        event_manager = MagicMock(spec=EventManager)
        event_manager.trigger = AsyncMock()
        call = ToolCall(id="call-1", name="calc", arguments={"x": 1})

        for legacy_item in (
            Event(type=EventType.TOOL_PROCESS, payload=[call]),
            ToolCallToken(token='{"x":1}', call=call),
        ):
            with self.subTest(legacy_item=type(legacy_item).__qualname__):
                tool = AsyncMock(spec=ToolManager)
                tool.is_empty = False
                response = _make_response(
                    Message(role=MessageRole.USER, content="hi"),
                    _empty_response(),
                    agent,
                    operation,
                    {},
                    event_manager=event_manager,
                    tool=tool,
                )
                response._tool_parser = AsyncMock()
                response._tool_parser.flush.return_value = [legacy_item]
                iterator = response.__aiter__()

                first = await iterator.__anext__()
                self.assertIs(first.kind, StreamItemKind.STREAM_STARTED)
                with self.assertRaises(StreamValidationError):
                    await iterator.__anext__()

                tool.assert_not_awaited()
                self.assertTrue(response._calls.empty())
                triggered_types = [
                    call_args.args[0].type
                    for call_args in event_manager.trigger.await_args_list
                ]
                self.assertNotIn(EventType.TOOL_EXECUTE, triggered_types)
                self.assertNotIn(EventType.TOOL_PROCESS, triggered_types)
                event_manager.trigger.reset_mock()

    async def test_queue_parser_output_rejects_tool_process_without_calls(
        self,
    ):
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

        with self.assertRaises(StreamValidationError):
            resp._queue_parser_output(
                Event(type=EventType.TOOL_PROCESS, payload=None)
            )
        self.assertTrue(resp._calls.empty())

    async def test_direct_iteration_rejects_legacy_tool_call_token(self):
        engine = _DummyEngine()
        agent = MagicMock(spec=EngineAgent)
        agent.engine = engine
        operation = _dummy_operation()
        call = ToolCall(id=uuid4(), name="calc", arguments=None)

        response = cast(
            TextGenerationResponse,
            _LegacyFixtureResponse(ToolCallToken(token="c", call=call)),
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
        self.assertIs(first.kind, StreamItemKind.STREAM_STARTED)
        self.assertFalse(isinstance(first, _LEGACY_PUBLIC_STREAM_TYPES))
        with self.assertRaises(StreamValidationError):
            await iterator.__anext__()
        event_manager.trigger.assert_not_awaited()

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
            _response_from_items(*_canonical_tool_call_items(call)),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
            tool_confirm=lambda c: "a",
        )
        resp.__aiter__()
        items = await _collect_until_kind(
            resp,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
        )
        self.assertTrue(all(not isinstance(item, Event) for item in items))
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
        items = await _collect_until_kind(
            resp,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
        )
        self.assertTrue(all(not isinstance(item, Event) for item in items))

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
            while True:
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
        resp._tool_result_outcomes.put(
            _ToolExecutionOutcome(
                call=result.call,
                context=ToolCallContext(),
                event=Event(
                    type=EventType.TOOL_RESULT,
                    payload={"result": result},
                ),
                planned_index=0,
                result=result,
            )
        )
        items = await _collect_until_kind(
            resp,
            StreamItemKind.MODEL_CONTINUATION_STARTED,
        )
        self.assertTrue(all(not isinstance(item, Event) for item in items))
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
        await resp._process_canonical_response_item(
            _canonical_item(StreamItemKind.ANSWER_DELTA, 0, text_delta="x")
        )
        event_manager.trigger.assert_awaited()
        for parser_event in (
            Event(type=EventType.TOOL_PROCESS, payload=None),
            Event(type=EventType.END),
        ):
            with self.subTest(event_type=parser_event.type):
                with self.assertRaises(StreamValidationError):
                    resp._queue_parser_output(parser_event)

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

        call = ToolCall(id=uuid4(), name="fail", arguments={})
        resp = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _response_from_items(*_canonical_tool_call_items(call)),
            agent,
            operation,
            {},
            event_manager=event_manager,
            tool=tool,
        )
        resp.__aiter__()

        items = await _collect_until_kind(
            resp,
            StreamItemKind.MODEL_CONTINUATION_STARTED,
        )
        self.assertIn(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            [item.kind for item in items],
        )

        context = agent.await_args_list[0].args[0]
        assert isinstance(context.input, list)
        self.assertEqual(
            context.input[2].tool_call_error.message,
            "boom",
        )

    async def test_provider_event_usage_done_and_terminal_handling(self):
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
        resp._append_canonical_item(StreamItemKind.STREAM_STARTED)

        self.assertIsNone(
            resp._append_canonical_provider_event_item(
                StreamProviderEvent(kind=StreamItemKind.ANSWER_DONE)
            )
        )
        self.assertIsNone(
            resp._append_canonical_provider_event_item(
                StreamProviderEvent(
                    kind=StreamItemKind.USAGE_COMPLETED,
                    usage={"input_tokens": 1},
                )
            )
        )
        self.assertIsNone(
            resp._append_canonical_provider_event_item(
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_ERRORED,
                    data={"message": "boom"},
                )
            )
        )
        self.assertEqual(
            [item.kind for item in resp.canonical_items[-2:]],
            [StreamItemKind.STREAM_ERRORED, StreamItemKind.STREAM_CLOSED],
        )
        validate_canonical_stream_items(resp.canonical_items)

    async def test_response_terminal_error_item_finalizes_stream(self):
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
        resp._append_canonical_item(StreamItemKind.STREAM_STARTED)

        self.assertIsNone(
            resp._append_canonical_response_item(
                _canonical_item(
                    StreamItemKind.STREAM_ERRORED,
                    0,
                    data={"message": "boom"},
                )
            )
        )
        self.assertEqual(
            [item.kind for item in resp.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.STREAM_ERRORED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )
        validate_canonical_stream_items(resp.canonical_items)

    async def test_execute_tool_call_without_manager_returns_none(self):
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
        context = ToolCallContext(input=Message(role=MessageRole.USER))

        result = await resp._execute_tool_call(
            ToolCall(id="call-1", name="calc", arguments={}),
            context,
            confirm=True,
        )

        self.assertIsNone(result)

    async def test_anonymous_tool_call_helpers_assign_stable_ids(self):
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
        call = ToolCall(id=None, name="calc", arguments={"x": 1})

        tool_call_id = resp._canonical_tool_call_id(call)
        self.assertEqual(resp._canonical_tool_call_id(call), tool_call_id)
        self.assertTrue(tool_call_id.startswith("orchestrator-tool-call-"))
        canonical_call = resp._tool_call_with_canonical_id(call)
        self.assertEqual(canonical_call.id, tool_call_id)
        resp._append_canonical_tool_call_argument_delta(call, '{"x":')

        self.assertIn(
            tool_call_id,
            resp._canonical_tool_call_argument_delta_ids,
        )
        self.assertEqual(
            resp.canonical_items[-1].kind,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        )
        self.assertEqual(
            resp.canonical_items[-1].correlation.tool_call_id,
            tool_call_id,
        )

        fresh_call = ToolCall(id=None, name="calc", arguments={"x": 2})
        canonical_fresh_call = resp._tool_call_with_canonical_id(fresh_call)
        self.assertIsNot(canonical_fresh_call, fresh_call)
        self.assertIsNotNone(canonical_fresh_call.id)
        self.assertNotEqual(canonical_fresh_call.id, tool_call_id)
        self.assertTrue(
            str(canonical_fresh_call.id).startswith("orchestrator-tool-call-")
        )
        self.assertTrue(resp._should_execute_staged_tool_call(fresh_call))

        assigned_call = ToolCall(
            id="assigned-call",
            name="calc",
            arguments=None,
        )
        resp._canonical_tool_call_ids_by_object[id(assigned_call)] = (
            "assigned-call"
        )
        self.assertIs(
            resp._tool_call_with_canonical_id(assigned_call),
            assigned_call,
        )

    async def test_tool_call_lifecycle_defensive_diagnostics(self):
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

        cases: tuple[tuple[str, str], ...] = (
            ("missing-delta", "orchestrator.tool_call.missing_argument_delta"),
            (
                "argument-after-ready",
                "orchestrator.tool_call.argument_after_ready",
            ),
            (
                "argument-after-done",
                "orchestrator.tool_call.argument_after_done",
            ),
            ("ready-after-done", "orchestrator.tool_call.ready_after_done"),
        )
        for tool_call_id, expected_code in cases:
            with self.subTest(code=expected_code):
                if tool_call_id == "argument-after-ready":
                    resp._append_canonical_tool_call_lifecycle_item(
                        StreamItemKind.TOOL_CALL_READY,
                        data={"name": "calc", "arguments": {}},
                        correlation=StreamItemCorrelation(
                            tool_call_id=tool_call_id
                        ),
                    )
                    resp._append_canonical_tool_call_lifecycle_item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        text_delta="{}",
                        correlation=StreamItemCorrelation(
                            tool_call_id=tool_call_id
                        ),
                    )
                elif tool_call_id == "argument-after-done":
                    state = resp._canonical_tool_call_lifecycle(tool_call_id)
                    state.done = True
                    resp._append_canonical_tool_call_lifecycle_item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        text_delta="{}",
                        correlation=StreamItemCorrelation(
                            tool_call_id=tool_call_id
                        ),
                    )
                elif tool_call_id == "ready-after-done":
                    state = resp._canonical_tool_call_lifecycle(tool_call_id)
                    state.done = True
                    resp._append_canonical_tool_call_lifecycle_item(
                        StreamItemKind.TOOL_CALL_READY,
                        data={},
                        correlation=StreamItemCorrelation(
                            tool_call_id=tool_call_id
                        ),
                    )
                else:
                    resp._append_canonical_tool_call_lifecycle_item(
                        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                        correlation=StreamItemCorrelation(
                            tool_call_id=tool_call_id
                        ),
                    )

                self.assertTrue(
                    any(
                        item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                        and item.data["code"] == expected_code
                        and item.correlation.tool_call_id == tool_call_id
                        for item in resp.canonical_items
                    )
                )

        terminal = _make_response(
            Message(role=MessageRole.USER, content="hi"),
            _empty_response(),
            agent,
            operation,
            {},
        )
        terminal._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)
        self.assertIsNone(
            terminal._append_canonical_tool_call_lifecycle_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta="{}",
                correlation=StreamItemCorrelation(tool_call_id="after-close"),
            )
        )

    async def test_tool_call_lifecycle_reconstruction_defensive_paths(self):
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

        bad_data = resp._canonical_tool_call_lifecycle("bad-data")
        bad_data.ready_item = _canonical_item(
            StreamItemKind.TOOL_CALL_READY,
            0,
            data="bad",
            correlation=StreamItemCorrelation(tool_call_id="bad-data"),
        )
        self.assertIsNone(
            resp._tool_call_from_canonical_lifecycle("bad-data", bad_data)
        )

        missing_name = resp._canonical_tool_call_lifecycle("missing-name")
        missing_name.ready_item = _canonical_item(
            StreamItemKind.TOOL_CALL_READY,
            1,
            data={"arguments": {}},
            correlation=StreamItemCorrelation(tool_call_id="missing-name"),
        )
        self.assertIsNone(
            resp._tool_call_from_canonical_lifecycle(
                "missing-name",
                missing_name,
            )
        )

        non_object_delta = resp._canonical_tool_call_lifecycle(
            "non-object-delta"
        )
        non_object_delta.argument_deltas.append("[]")
        non_object_delta.ready_item = _canonical_item(
            StreamItemKind.TOOL_CALL_READY,
            2,
            data={"name": "calc", "arguments": {}},
            correlation=StreamItemCorrelation(tool_call_id="non-object-delta"),
        )
        self.assertIsNone(
            resp._tool_call_from_canonical_lifecycle(
                "non-object-delta",
                non_object_delta,
            )
        )

        ready_list = resp._canonical_tool_call_lifecycle("ready-list")
        ready_list_arguments = resp._tool_call_arguments_from_lifecycle(
            "ready-list",
            ready_list,
            ready_data={"arguments": []},
        )
        self.assertIsNot(
            ready_list_arguments,
            None,
        )
        self.assertNotIsInstance(ready_list_arguments, dict)

        no_arguments = resp._canonical_tool_call_lifecycle("no-arguments")
        self.assertIsNone(
            resp._tool_call_arguments_from_lifecycle(
                "no-arguments",
                no_arguments,
                ready_data={},
            )
        )
        self.assertTrue(
            any(
                item.kind is StreamItemKind.STREAM_DIAGNOSTIC
                and item.correlation.tool_call_id
                in {
                    "bad-data",
                    "missing-name",
                    "non-object-delta",
                    "ready-list",
                }
                for item in resp.canonical_items
            )
        )

    async def test_tool_call_id_allocator_defensive_paths(self):
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

        self.assertTrue(
            resp._should_execute_staged_tool_call(
                ToolCall(id=None, name="calc", arguments={})
            )
        )

        anonymous = ToolCall(id=None, name="calc", arguments={})
        canonical_anonymous = resp._tool_call_with_canonical_id(anonymous)
        self.assertEqual(canonical_anonymous.id, "orchestrator-tool-call-1")

        assigned = ToolCall(id="assigned-call", name="calc", arguments={})
        resp._canonical_tool_call_ids_by_object[id(assigned)] = "assigned-call"
        self.assertIs(resp._tool_call_with_canonical_id(assigned), assigned)

        invalid = resp._canonical_tool_call_lifecycle("invalid-call")
        invalid.invalid = True
        resp._queue_completed_canonical_tool_call("invalid-call", invalid)
        invalid.queued = True
        resp._queue_completed_canonical_tool_call("invalid-call", invalid)
        self.assertTrue(resp._calls.empty())

    async def test_invalid_staged_calls_skip_execution_batch(self) -> None:
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
        invalid = resp._canonical_tool_call_lifecycle("invalid-call")
        invalid.done = True
        invalid.invalid = True
        resp._calls.put(
            ToolCall(id="invalid-call", name="calc", arguments=None)
        )
        resp._response_drained = True

        self.assertIsNone(await resp._next_item())
        self.assertTrue(resp._calls.empty())

    async def test_pending_tool_batch_task_completion_is_consumed(
        self,
    ) -> None:
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
        resp._canonical_yield_index = len(resp.canonical_items)
        resp._canonical_item_available.clear()
        call = ToolCall(id="call1", name="calc", arguments=None)
        result = ToolCallResult(
            id="result1",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result="ok",
        )

        async def complete_batch() -> list[_ToolExecutionOutcome]:
            await sleep(0)
            return [
                _ToolExecutionOutcome(
                    call=call,
                    context=ToolCallContext(),
                    event=Event(type=EventType.TOOL_RESULT),
                    planned_index=0,
                    result=result,
                )
            ]

        task = create_task(complete_batch())
        resp._pending_tool_batch_task = task

        await resp._await_pending_tool_batch()

        self.assertIsNone(resp._pending_tool_batch_task)
        self.assertFalse(resp._tool_result_outcomes.empty())

    async def test_pending_tool_batch_cancellation_cleans_up_task(
        self,
    ) -> None:
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

        async def slow_batch() -> list[_ToolExecutionOutcome]:
            await sleep(1)
            return []

        pending_batch = create_task(slow_batch())
        resp._pending_tool_batch_task = pending_batch
        waiter = create_task(resp._await_pending_tool_batch())
        await sleep(0)

        waiter.cancel()
        with self.assertRaises(CancelledError):
            await waiter

        self.assertIsNone(resp._pending_tool_batch_task)
        self.assertTrue(pending_batch.cancelled())
