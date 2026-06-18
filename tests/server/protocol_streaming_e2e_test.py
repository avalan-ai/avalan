import asyncio
from argparse import Namespace
from asyncio import Event as AsyncEvent
from collections.abc import AsyncIterator
from dataclasses import replace
from json import loads
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call
from uuid import uuid4

from a2a import types as a2a_types
from streaming_trace_fixtures import (
    TERMINAL_ERROR_DATA,
    async_items,
    terminal_outcome_trace,
)

from avalan.cli.commands import model as model_cmds
from avalan.entities import MessageRole
from avalan.event import Event, EventType
from avalan.flow.stream import canonical_flow_event_listener
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    iter_stream_consumer_projections,
)
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    A2AStreamEventConverter,
)
from avalan.server.a2a.store import TaskStore
from avalan.server.entities import ChatCompletionRequest, ChatMessage
from avalan.server.routers import mcp as mcp_router


class _CanonicalResponse:
    input_token_count = 0
    output_token_count = 0
    can_think = False
    is_thinking = False

    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        return _iter_items(self._items)

    def set_thinking(self, value: bool) -> None:
        self.is_thinking = value


class _TrackedDirectCanonicalResponse:
    input_token_count = 0
    output_token_count = 0
    can_think = False
    is_thinking = False

    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = items
        self._index = 0
        self.cancel_count = 0
        self.close_count = 0

    def __aiter__(self) -> "_TrackedDirectCanonicalResponse":
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        if self._index == len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item

    async def cancel(self) -> None:
        self.cancel_count += 1

    async def aclose(self) -> None:
        self.close_count += 1

    def set_thinking(self, value: bool) -> None:
        self.is_thinking = value


class _SyncingOrchestrator:
    def __init__(self) -> None:
        self.synced = False

    async def sync_messages(self) -> None:
        self.synced = True


class _LegacyRejectionResponse:
    input_token_count = 0
    output_token_count = 0

    def __aiter__(self) -> AsyncIterator[object]:
        return self._items()

    async def _items(self) -> AsyncIterator[object]:
        yield "legacy text"


async def _iter_items(
    items: tuple[CanonicalStreamItem, ...],
) -> AsyncIterator[CanonicalStreamItem]:
    for item in items:
        yield item


def _stream_item(
    sequence: int,
    kind: StreamItemKind,
    **kwargs: object,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
        sequence=sequence,
        kind=kind,
        channel=_channel(kind),
        **kwargs,
    )


def _channel(kind: StreamItemKind) -> StreamChannel:
    if kind in {
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
    }:
        return StreamChannel.ANSWER
    if kind in {
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.REASONING_DONE,
    }:
        return StreamChannel.REASONING
    if kind in {
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
    }:
        return StreamChannel.TOOL_CALL
    if kind in {
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
    }:
        return StreamChannel.TOOL_EXECUTION
    if kind is StreamItemKind.USAGE_COMPLETED:
        return StreamChannel.USAGE
    return StreamChannel.CONTROL


def _canonical_items(
    flow_item: CanonicalStreamItem,
) -> tuple[CanonicalStreamItem, ...]:
    call = StreamItemCorrelation(tool_call_id="call-1")
    usage = {
        "input_text_tokens": 2,
        "output_text_tokens": 5,
        "total_tokens": 7,
    }
    return (
        _stream_item(0, StreamItemKind.STREAM_STARTED),
        replace(flow_item, sequence=1),
        _stream_item(2, StreamItemKind.REASONING_DELTA, text_delta="plan"),
        _stream_item(3, StreamItemKind.REASONING_DONE),
        _stream_item(
            4,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation=call,
            text_delta='{"query":"',
        ),
        _stream_item(
            5,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation=call,
            text_delta='docs"}',
        ),
        _stream_item(6, StreamItemKind.TOOL_CALL_READY, correlation=call),
        _stream_item(7, StreamItemKind.TOOL_CALL_DONE, correlation=call),
        _stream_item(
            8,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            correlation=call,
            metadata={"tool_name": "search"},
        ),
        _stream_item(
            9,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=call,
            text_delta="live",
            data={"category": "stdout", "content": "live"},
            metadata={"tool_name": "search"},
        ),
        _stream_item(
            10,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=call,
            text_delta=" warning",
            data={"category": "stderr", "content": " warning"},
            metadata={"tool_name": "search"},
        ),
        _stream_item(
            11,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=call,
            text_delta=" trace",
            data={"category": "log", "content": " trace"},
            metadata={"tool_name": "search"},
        ),
        _stream_item(
            12,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            correlation=call,
            data={"category": "progress", "content": "50%", "progress": 0.5},
            metadata={"tool_name": "search"},
        ),
        _stream_item(
            13,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            correlation=call,
        ),
        _stream_item(14, StreamItemKind.ANSWER_DELTA, text_delta="final "),
        _stream_item(15, StreamItemKind.ANSWER_DELTA, text_delta="answer"),
        _stream_item(16, StreamItemKind.ANSWER_DONE),
        _stream_item(17, StreamItemKind.USAGE_COMPLETED, usage=usage),
        _stream_item(
            18,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    )


def test_same_canonical_stream_projects_through_protocols() -> None:
    asyncio.run(_run_same_canonical_stream_projects_through_protocols())


async def _run_same_canonical_stream_projects_through_protocols() -> None:
    flow_events: list[Event] = []
    flow_listener = canonical_flow_event_listener(
        flow_events.append,
        stream_session_id="stream-1",
        run_id="run-1",
        turn_id="turn-1",
    )
    result = flow_listener(
        Event(
            type=EventType.FLOW_NODE_STARTED,
            payload={
                "flow_id": "flow-1",
                "node": "node-1",
                "status": "started",
            },
        )
    )
    if result is not None:
        await result

    assert len(flow_listener.items) == 1
    assert len(flow_listener.ui_items) == 1
    flow_item = flow_listener.items[0]
    assert flow_item.kind is StreamItemKind.FLOW_EVENT
    assert flow_item.correlation.flow_run_id == "flow-1"
    assert flow_item.correlation.node_id == "node-1"

    items = _canonical_items(flow_item)
    await _assert_mcp_projection(items)
    await _assert_a2a_projection(items)


def test_terminal_outcome_traces_project_through_protocols() -> None:
    asyncio.run(_run_terminal_outcome_traces_project_through_protocols())


async def _run_terminal_outcome_traces_project_through_protocols() -> None:
    for outcome in (
        StreamTerminalOutcome.COMPLETED,
        StreamTerminalOutcome.ERRORED,
        StreamTerminalOutcome.CANCELLED,
    ):
        trace = terminal_outcome_trace(outcome)
        await _assert_mcp_terminal_outcome_projection(trace.items, outcome)
        await _assert_a2a_terminal_outcome_projection(trace.items, outcome)


def test_default_protocol_routes_legacy_rejection_first_item() -> None:
    asyncio.run(_run_default_protocol_routes_legacy_rejection_first_item())


async def _run_default_protocol_routes_legacy_rejection_first_item() -> None:
    await _assert_mcp_legacy_rejection_first_item()
    await _assert_a2a_legacy_rejection_first_item()


def test_protocol_routes_close_direct_sources_once() -> None:
    asyncio.run(_run_protocol_routes_close_direct_sources_once())


async def _run_protocol_routes_close_direct_sources_once() -> None:
    items = (
        _stream_item(0, StreamItemKind.STREAM_STARTED),
        _stream_item(1, StreamItemKind.ANSWER_DELTA, text_delta="ok"),
        _stream_item(2, StreamItemKind.ANSWER_DONE),
        _stream_item(
            3,
            StreamItemKind.STREAM_COMPLETED,
            usage={"output_tokens": 1},
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    )

    mcp_response = _TrackedDirectCanonicalResponse(items)
    mcp_payloads = await _collect_mcp_payloads(
        mcp_response,
        "direct-source-close",
    )
    assert any("result" in payload for payload in mcp_payloads)
    assert mcp_response.cancel_count == 0
    assert mcp_response.close_count == 1

    a2a_response = _TrackedDirectCanonicalResponse(items)
    store = TaskStore()
    task_id = "direct-source-close"
    await store.create_task(
        task_id,
        model="test-model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    _ = [event async for event in translator.run_stream(a2a_response)]
    task = await store.get_task(task_id)
    assert task["status"] == "completed"
    assert a2a_response.cancel_count == 0
    assert a2a_response.close_count == 1


def test_lossy_cli_frames_do_not_drop_lossless_public_surfaces() -> None:
    asyncio.run(_run_lossy_cli_frames_do_not_drop_lossless_public_surfaces())


async def _run_lossy_cli_frames_do_not_drop_lossless_public_surfaces() -> None:
    usage = {"output_tokens": 2}
    items = (
        _stream_item(0, StreamItemKind.STREAM_STARTED),
        _stream_item(1, StreamItemKind.ANSWER_DELTA, text_delta="A"),
        _stream_item(2, StreamItemKind.ANSWER_DELTA, text_delta="B"),
        _stream_item(3, StreamItemKind.ANSWER_DONE),
        _stream_item(4, StreamItemKind.USAGE_COMPLETED, usage=usage),
        _stream_item(
            5,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    )

    sdk_items = [
        item
        async for item in iter_stream_consumer_projections(_iter_items(items))
    ]
    assert [item.sequence for item in sdk_items] == list(range(len(items)))
    assert [item.kind for item in sdk_items] == [item.kind for item in items]
    assert [item.text_delta for item in sdk_items[1:3]] == ["A", "B"]

    await _assert_stdout_projection(items)
    await _assert_lossy_cli_projection(items)
    await _assert_simple_mcp_projection(items)
    await _assert_simple_a2a_projection(items)


async def _assert_stdout_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    console = MagicMock()
    await model_cmds.token_generation(
        args=Namespace(skip_display_reasoning_time=False),
        console=console,
        theme=MagicMock(),
        logger=MagicMock(),
        orchestrator=None,
        event_stats=None,
        lm=MagicMock(),
        input_string="i",
        response=_CanonicalResponse(items),
        display_tokens=0,
        dtokens_pick=0,
        with_stats=False,
        tool_events_limit=2,
        refresh_per_second=2,
    )

    assert console.print.call_args_list == [
        call("A", end=""),
        call("B", end=""),
    ]


async def _assert_lossy_cli_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    yielded_frames: list[str] = []
    closed_frames: list[str] = []

    async def fake_tokens(
        *parameters: object, **_: object
    ) -> AsyncIterator[tuple[object | None, str]]:
        state = cast(Any, parameters[0])
        answer_text = "".join(state.answer_text_tokens)
        try:
            yielded_frames.append(f"first:{answer_text}")
            yield None, f"frame:{answer_text}:first"
            yielded_frames.append(f"second:{answer_text}")
            yield None, f"frame:{answer_text}:second"
        finally:
            closed_frames.append(answer_text)

    theme = MagicMock()
    theme.token_frames = MagicMock(side_effect=fake_tokens)
    args = Namespace(
        skip_display_reasoning_time=False,
        display_time_to_n_token=1,
        display_pause=0,
        start_thinking=False,
        display_probabilities=False,
        display_probabilities_maximum=1.0,
        display_probabilities_sample_minimum=0.0,
        record=False,
    )

    await model_cmds._token_stream(
        args=args,
        console=MagicMock(width=80),
        live=MagicMock(),
        group=None,
        tokens_group_index=None,
        theme=theme,
        logger=MagicMock(),
        orchestrator=None,
        event_stats=None,
        lm=SimpleNamespace(
            model_id="m",
            tokenizer_config=None,
            input_token_count=lambda _: 1,
        ),
        input_string="i",
        response=_CanonicalResponse(items),
        display_tokens=1,
        dtokens_pick=1,
        refresh_per_second=1000,
        stop_signal=None,
        tool_events_limit=2,
        with_stats=True,
    )

    assert yielded_frames == ["first:AB", "second:AB"]
    assert closed_frames == ["AB"]


async def _assert_simple_mcp_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    request_model = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        stream=True,
    )
    payloads: list[dict[str, object]] = []
    async for chunk in mcp_router._stream_mcp_response(
        request_id="req-simple",
        request_model=request_model,
        response=_CanonicalResponse(items),
        response_id=uuid4(),
        timestamp=1,
        progress_token="progress-simple",
        orchestrator=_SyncingOrchestrator(),
        logger=getLogger("test.protocol.mcp.simple"),
        resource_store=mcp_router.MCPResourceStore(),
        base_path="/mcp",
        cancel_event=AsyncEvent(),
    ):
        payloads.extend(
            loads(part) for part in chunk.decode("utf-8").splitlines() if part
        )

    progress = [
        payload["params"]["progress"]
        for payload in payloads
        if payload.get("method") == "notifications/progress"
    ]
    assert [
        item["delta"]
        for item in progress
        if item.get("type") == "answer.delta"
    ] == ["A", "B"]
    result_payloads = [payload for payload in payloads if "result" in payload]
    assert len(result_payloads) == 1
    result = result_payloads[0]["result"]
    assert isinstance(result, dict)
    assert result["content"] == [{"type": "text", "text": "AB"}]


async def _assert_simple_a2a_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    store = TaskStore()
    task_id = "lossy-frame-isolation"
    await store.create_task(
        task_id,
        model="test-model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    _ = [event async for event in translator.run_stream(_iter_items(items))]

    assert translator.text == "AB"
    task = await store.get_task(task_id)
    assert task["status"] == "completed"
    assert task["artifacts"][0]["content"] == [
        {"type": "text", "text": "A"},
        {"type": "text", "text": "B"},
    ]


async def _collect_mcp_payloads(
    response: (
        _CanonicalResponse
        | _LegacyRejectionResponse
        | _TrackedDirectCanonicalResponse
    ),
    request_id: str,
) -> tuple[dict[str, object], ...]:
    request_model = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        stream=True,
    )
    payloads: list[dict[str, object]] = []
    async for chunk in mcp_router._stream_mcp_response(
        request_id=request_id,
        request_model=request_model,
        response=response,
        response_id=uuid4(),
        timestamp=1,
        progress_token=f"progress-{request_id}",
        orchestrator=_SyncingOrchestrator(),
        logger=getLogger(f"test.protocol.mcp.{request_id}"),
        resource_store=mcp_router.MCPResourceStore(),
        base_path="/mcp",
        cancel_event=AsyncEvent(),
    ):
        for part in chunk.decode("utf-8").splitlines():
            if not part:
                continue
            payload = loads(part)
            assert isinstance(payload, dict)
            payloads.append(cast(dict[str, object], payload))
    return tuple(payloads)


async def _assert_mcp_terminal_outcome_projection(
    items: tuple[CanonicalStreamItem, ...],
    outcome: StreamTerminalOutcome,
) -> None:
    payloads = await _collect_mcp_payloads(
        _CanonicalResponse(items),
        f"terminal-{outcome.value}",
    )
    progress_types = [
        _mcp_progress(payload).get("type")
        for payload in payloads
        if payload.get("method") == "notifications/progress"
    ]

    if outcome is StreamTerminalOutcome.COMPLETED:
        assert "answer.completed" in progress_types
        result_payloads = [
            payload for payload in payloads if "result" in payload
        ]
        assert len(result_payloads) == 1
        result = cast(dict[str, object], result_payloads[0]["result"])
        assert result["content"] == [{"type": "text", "text": "ok"}]
        structured = cast(dict[str, object], result["structuredContent"])
        usage = cast(dict[str, object], structured["usage"])
        assert usage["total_tokens"] == 3
        assert not [payload for payload in payloads if "error" in payload]
        return

    expected_progress = (
        "stream.errored"
        if outcome is StreamTerminalOutcome.ERRORED
        else "stream.cancelled"
    )
    expected_error_message = (
        str(TERMINAL_ERROR_DATA["message"])
        if outcome is StreamTerminalOutcome.ERRORED
        else "Request cancelled"
    )
    assert expected_progress in progress_types
    error_payloads = [payload for payload in payloads if "error" in payload]
    assert len(error_payloads) == 1
    error = cast(dict[str, object], error_payloads[0]["error"])
    assert error["message"] == expected_error_message
    assert not [payload for payload in payloads if "result" in payload]


async def _assert_a2a_terminal_outcome_projection(
    items: tuple[CanonicalStreamItem, ...],
    outcome: StreamTerminalOutcome,
) -> None:
    expected_statuses = {
        StreamTerminalOutcome.COMPLETED: "completed",
        StreamTerminalOutcome.ERRORED: "failed",
        StreamTerminalOutcome.CANCELLED: "canceled",
    }
    expected_states = {
        StreamTerminalOutcome.COMPLETED: a2a_types.TaskState.completed,
        StreamTerminalOutcome.ERRORED: a2a_types.TaskState.failed,
        StreamTerminalOutcome.CANCELLED: a2a_types.TaskState.canceled,
    }
    store = TaskStore()
    task_id = f"terminal-{outcome.value}"
    await store.create_task(
        task_id,
        model="test-model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    converter = A2AStreamEventConverter(task_id, store)

    status_updates: list[a2a_types.TaskStatusUpdateEvent] = []
    async for raw_event in translator.run_stream(async_items(items)):
        converted = await converter.convert(raw_event)
        if not isinstance(converted, dict) or "result" not in converted:
            continue
        response = (
            a2a_types.SendStreamingMessageSuccessResponse.model_validate(
                converted
            )
        )
        if isinstance(response.result, a2a_types.TaskStatusUpdateEvent):
            status_updates.append(response.result)

    task = await store.get_task(task_id)
    assert task["status"] == expected_statuses[outcome]
    if outcome is StreamTerminalOutcome.ERRORED:
        assert task["error"] == str(TERMINAL_ERROR_DATA["message"])
    else:
        assert task["error"] is None
    assert status_updates
    assert status_updates[-1].status.state is expected_states[outcome]
    assert status_updates[-1].final is True


async def _assert_mcp_legacy_rejection_first_item() -> None:
    payloads = await _collect_mcp_payloads(
        _LegacyRejectionResponse(),
        "legacy-rejection",
    )
    error_payloads = [payload for payload in payloads if "error" in payload]
    assert len(error_payloads) == 1
    error = cast(dict[str, object], error_payloads[0]["error"])
    assert error["message"] == "An internal server error occurred."
    assert not [payload for payload in payloads if "result" in payload]


async def _assert_a2a_legacy_rejection_first_item() -> None:
    store = TaskStore()
    task_id = "legacy-rejection"
    await store.create_task(
        task_id,
        model="test-model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    events: list[dict[str, object]] = []

    try:
        async for event in translator.run_stream(_LegacyRejectionResponse()):
            events.append(cast(dict[str, object], event))
    except StreamValidationError as exc:
        assert "unsupported legacy A2A stream item" in str(exc)
    else:
        raise AssertionError("A2A accepted a legacy-only stream source")

    task = await store.get_task(task_id)
    assert task["status"] == "failed"
    assert "unsupported legacy A2A stream item" in str(task["error"])
    assert any(
        event.get("event") == "task.status.changed"
        and cast(dict[str, object], event["data"]).get("status") == "failed"
        for event in events
    )


async def _assert_mcp_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    response = _CanonicalResponse(items)
    orchestrator = _SyncingOrchestrator()
    request_model = ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role=MessageRole.USER, content="hi")],
        stream=True,
    )

    payloads: list[dict[str, object]] = []
    async for chunk in mcp_router._stream_mcp_response(
        request_id="req-1",
        request_model=request_model,
        response=response,
        response_id=uuid4(),
        timestamp=1,
        progress_token="progress-1",
        orchestrator=orchestrator,
        logger=getLogger("test.protocol.mcp"),
        resource_store=mcp_router.MCPResourceStore(),
        base_path="/mcp",
        cancel_event=AsyncEvent(),
    ):
        payloads.extend(
            loads(part) for part in chunk.decode("utf-8").splitlines() if part
        )

    assert orchestrator.synced is True
    progress = [
        payload["params"]["progress"]
        for payload in payloads
        if payload.get("method") == "notifications/progress"
    ]
    assert [item["type"] for item in progress if "type" in item].count(
        "answer.completed"
    ) == 1
    assert any(
        item.get("type") == "answer.delta" and item.get("delta") == "final "
        for item in progress
    )
    messages = [
        payload["params"]["message"]
        for payload in payloads
        if payload.get("method") == "notifications/message"
    ]
    assert {"type": "reasoning", "delta": "plan"} in messages
    assert any(
        message.get("type") == "tool.input_delta"
        and message.get("toolCallId") == "call-1"
        for message in messages
    )
    resources = [
        payload
        for payload in payloads
        if payload.get("method") == "notifications/resources/updated"
    ]
    tool_result_index = next(
        index
        for index, payload in enumerate(payloads)
        if (
            payload.get("method") == "notifications/message"
            and _mcp_message(payload).get("type") == "tool.result"
        )
    )
    live_resources = [
        index
        for index, payload in enumerate(payloads)
        if (
            payload.get("method") == "notifications/resources/updated"
            and "delta" in _mcp_resource(payload)
        )
    ]
    closed_resources = [
        payload
        for payload in resources
        if _mcp_resource(payload).get("closed") is True
    ]
    assert len(live_resources) == 4
    assert len(closed_resources) == 4
    assert all(index < tool_result_index for index in live_resources)
    result_payloads = [payload for payload in payloads if "result" in payload]
    assert len(result_payloads) == 1
    result = result_payloads[0]["result"]
    assert isinstance(result, dict)
    assert result["content"] == [{"type": "text", "text": "final answer"}]
    structured = result["structuredContent"]
    assert isinstance(structured, dict)
    assert structured["reasoning"] == "plan"
    assert structured["usage"] == {
        "input_text_tokens": 2,
        "output_text_tokens": 5,
        "total_tokens": 7,
    }
    tool_call = structured["toolCalls"][0]
    assert tool_call["id"] == "call-1"
    assert tool_call["name"] == "search"
    assert tool_call["arguments"] == '{"query":"docs"}'
    assert [resource["name"] for resource in tool_call["resources"]] == [
        "stdout",
        "stderr",
        "logs",
        "progress",
    ]


async def _assert_a2a_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    store = TaskStore()
    task_id = "protocol-stream"
    await store.create_task(
        task_id,
        model="test-model",
        instructions=None,
        input_messages=[],
        metadata={},
    )
    translator = A2AResponseTranslator(task_id, store)
    converter = A2AStreamEventConverter(task_id, store)

    status_updates: list[a2a_types.TaskStatusUpdateEvent] = []
    artifact_updates: list[a2a_types.TaskArtifactUpdateEvent] = []
    async for raw_event in translator.run_stream(_iter_items(items)):
        converted = await converter.convert(raw_event)
        if not isinstance(converted, dict) or "result" not in converted:
            continue
        response = (
            a2a_types.SendStreamingMessageSuccessResponse.model_validate(
                converted
            )
        )
        if isinstance(response.result, a2a_types.TaskStatusUpdateEvent):
            status_updates.append(response.result)
        if isinstance(response.result, a2a_types.TaskArtifactUpdateEvent):
            artifact_updates.append(response.result)

    assert translator.text == "final answer"
    task = await store.get_task(task_id)
    artifacts = {artifact["id"]: artifact for artifact in task["artifacts"]}
    assert task["status"] == "completed"
    assert artifacts["reasoning"]["content"][0]["text"] == "plan"
    assert artifacts["answer"]["content"] == [
        {"type": "text", "text": "final "},
        {"type": "text", "text": "answer"},
    ]
    tool_content = artifacts["call-1"]["content"]
    assert artifacts["call-1"]["kind"] == "tool_execution"
    assert tool_content[0]["text"] == '{"query":"'
    assert tool_content[1]["text"] == 'docs"}'
    terminal_index = next(
        index
        for index, item in enumerate(tool_content)
        if item.get("type") == "tool_terminal"
    )
    live_tool_content = [
        item
        for item in tool_content[:terminal_index]
        if item.get("type") in {"tool_output", "progress"}
    ]
    assert [
        item.get("category", item.get("progress", {}).get("category"))
        for item in live_tool_content
    ] == ["stdout", "stderr", "log", "progress"]
    assert tool_content[terminal_index]["status"] == "completed"
    assert artifact_updates
    assert status_updates[-1].status.state is a2a_types.TaskState.completed
    assert status_updates[-1].final is True


def _mcp_message(payload: dict[str, object]) -> dict[str, object]:
    params = cast(dict[str, object], payload["params"])
    return cast(dict[str, object], params["message"])


def _mcp_progress(payload: dict[str, object]) -> dict[str, object]:
    params = cast(dict[str, object], payload["params"])
    return cast(dict[str, object], params["progress"])


def _mcp_resource(payload: dict[str, object]) -> dict[str, object]:
    params = cast(dict[str, object], payload["params"])
    resources = cast(list[dict[str, object]], params["resources"])
    return resources[0]
