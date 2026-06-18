import asyncio
from asyncio import Event as AsyncEvent
from collections.abc import AsyncIterator
from json import loads
from logging import getLogger
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

from a2a import types as a2a_types
from streaming_trace_fixtures import (
    TERMINAL_ERROR_DATA,
    TERMINAL_TRACE_USAGE,
    TOOL_CALL_ID,
    async_items,
    canonical_item,
    canonical_stream_trace,
    stream_channel,
    terminal_outcome_trace,
)

from avalan.cli.commands import model as model_cmds
from avalan.entities import MessageRole
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    accumulate_canonical_stream_items,
    iter_stream_consumer_projections,
)
from avalan.server.a2a.router import (
    A2AResponseTranslator,
    A2AStreamEventConverter,
)
from avalan.server.a2a.store import TaskStore
from avalan.server.entities import ChatCompletionRequest, ChatMessage
from avalan.server.routers import chat, responses
from avalan.server.routers import mcp as mcp_router
from avalan.server.routers.streaming import ProtocolStreamAccumulator


class _CanonicalResponse:
    input_token_count = 0
    output_token_count = 0
    can_think = False
    is_thinking = False

    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        return async_items(self._items)

    def set_thinking(self, value: bool) -> None:
        self.is_thinking = value

    async def to_str(self) -> str:
        return accumulate_canonical_stream_items(self._items).answer_text


class _SyncingOrchestrator:
    def __init__(self) -> None:
        self.synced = False

    async def sync_messages(self) -> None:
        self.synced = True


def test_canonical_trace_conforms_across_public_stream_surfaces() -> None:
    asyncio.run(_run_canonical_trace_conforms_across_public_stream_surfaces())


async def _run_canonical_trace_conforms_across_public_stream_surfaces() -> (
    None
):
    trace = canonical_stream_trace()
    projections = tuple(
        [
            projection
            async for projection in iter_stream_consumer_projections(
                async_items(trace.items)
            )
        ]
    )
    accumulator = accumulate_canonical_stream_items(trace.items)

    assert await _CanonicalResponse(trace.items).to_str() == "final answer"
    _assert_sdk_projection(projections)
    _assert_stdout_projection(projections)
    await _assert_cli_projection(trace.items)
    _assert_flow_projection(projections)
    _assert_chat_projection(projections, accumulator.final_usage)
    _assert_responses_projection(projections)
    await _assert_mcp_projection(trace.items)
    await _assert_a2a_projection(trace.items)


def test_canonical_golden_trace_locks_item_semantics() -> None:
    trace = canonical_stream_trace()

    assert tuple(_canonical_semantics(item) for item in trace.items) == (
        _semantic_entry(0, StreamItemKind.STREAM_STARTED),
        _semantic_entry(
            1,
            StreamItemKind.FLOW_EVENT,
            correlation={"flow_run_id": "flow-1", "node_id": "node-1"},
            data={
                "flow_id": "flow-1",
                "node": "node-1",
                "status": "started",
            },
            metadata={"event_type": "flow_node_started"},
        ),
        _semantic_entry(
            2,
            StreamItemKind.REASONING_DELTA,
            text_delta="plan",
            visibility=StreamVisibility.PRIVATE,
        ),
        _semantic_entry(
            3,
            StreamItemKind.REASONING_DONE,
            visibility=StreamVisibility.PRIVATE,
        ),
        _semantic_entry(
            4,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation={"tool_call_id": TOOL_CALL_ID},
            text_delta='{"query":"',
            data={"name": "search", "arguments": {"query": "docs"}},
        ),
        _semantic_entry(
            5,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation={"tool_call_id": TOOL_CALL_ID},
            text_delta='docs"}',
            data={"name": "search", "arguments": {"query": "docs"}},
        ),
        _semantic_entry(
            6,
            StreamItemKind.TOOL_CALL_READY,
            correlation={"tool_call_id": TOOL_CALL_ID},
            data={"name": "search", "arguments": {"query": "docs"}},
        ),
        _semantic_entry(
            7,
            StreamItemKind.TOOL_CALL_DONE,
            correlation={"tool_call_id": TOOL_CALL_ID},
        ),
        _semantic_entry(
            8,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            correlation={"tool_call_id": TOOL_CALL_ID},
            metadata={"tool_name": "search"},
        ),
        _semantic_entry(
            9,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation={"tool_call_id": TOOL_CALL_ID},
            text_delta="live",
            data={"category": "stdout", "content": "live"},
            metadata={"tool_name": "search"},
        ),
        _semantic_entry(
            10,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation={"tool_call_id": TOOL_CALL_ID},
            text_delta=" warning",
            data={"category": "stderr", "content": " warning"},
            metadata={"tool_name": "search"},
        ),
        _semantic_entry(
            11,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation={"tool_call_id": TOOL_CALL_ID},
            text_delta=" trace",
            data={"category": "log", "content": " trace"},
            metadata={"tool_name": "search"},
        ),
        _semantic_entry(
            12,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            correlation={"tool_call_id": TOOL_CALL_ID},
            data={
                "category": "progress",
                "content": "50%",
                "progress": 0.5,
            },
            metadata={"tool_name": "search"},
        ),
        _semantic_entry(
            13,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            correlation={"tool_call_id": TOOL_CALL_ID},
            data={"result": "live warning trace"},
            metadata={"tool_name": "search"},
        ),
        _semantic_entry(
            14,
            StreamItemKind.ANSWER_DELTA,
            text_delta="final ",
        ),
        _semantic_entry(
            15,
            StreamItemKind.ANSWER_DELTA,
            text_delta="answer",
        ),
        _semantic_entry(16, StreamItemKind.ANSWER_DONE),
        _semantic_entry(
            17,
            StreamItemKind.USAGE_COMPLETED,
            usage={
                "input_tokens": 2,
                "output_tokens": 5,
                "total_tokens": 7,
            },
        ),
        _semantic_entry(
            18,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
        _semantic_entry(19, StreamItemKind.STREAM_CLOSED),
    )


def test_terminal_outcome_traces_project_protocol_terminal_events() -> None:
    expected = {
        StreamTerminalOutcome.COMPLETED: (None, "response.completed"),
        StreamTerminalOutcome.ERRORED: (
            "chat.completion.failed",
            "response.failed",
        ),
        StreamTerminalOutcome.CANCELLED: (
            "chat.completion.cancelled",
            "response.cancelled",
        ),
    }

    for outcome, (chat_event_name, response_event_name) in expected.items():
        trace = terminal_outcome_trace(outcome)
        projections = asyncio.run(_collect_consumer_projections(trace.items))
        accumulator = accumulate_canonical_stream_items(trace.items)
        protocol_accumulator = ProtocolStreamAccumulator()
        for projection in projections:
            protocol_accumulator.add(projection)
        snapshot = protocol_accumulator.snapshot()
        terminal = next(
            projection
            for projection in projections
            if projection.is_stream_terminal
        )

        assert accumulator.final_usage == TERMINAL_TRACE_USAGE
        assert snapshot.usage == TERMINAL_TRACE_USAGE
        assert snapshot.terminal_outcome is outcome
        assert snapshot.terminal_succeeded is (
            outcome is StreamTerminalOutcome.COMPLETED
        )
        assert terminal.sequence == 4
        assert terminal.terminal_outcome is outcome

        chat_message = chat._chat_terminal_event(
            "chatcmpl-terminal",
            1,
            "test-model",
            terminal,
        )
        if chat_event_name is None:
            assert chat_message is None
        else:
            assert chat_message is not None
            assert _sse_event_name(chat_message) == chat_event_name
            chat_payload = _sse_event_data(chat_message)
            assert chat_payload["type"] == chat_event_name
            assert chat_payload["sequence_number"] == 4
            if outcome is StreamTerminalOutcome.ERRORED:
                assert chat_payload["error"] == TERMINAL_ERROR_DATA

        response_events = responses._terminal_response_events(terminal)
        assert len(response_events) == 1
        response_event = response_events[0]
        assert response_event.event == response_event_name
        assert response_event.data["type"] == response_event_name
        assert response_event.data["sequence_number"] == 4
        if outcome is StreamTerminalOutcome.ERRORED:
            assert response_event.data["error"] == TERMINAL_ERROR_DATA


def test_canonical_trace_rejects_content_after_terminal() -> None:
    trace = canonical_stream_trace()
    invalid_items = trace.items[:-1] + (
        canonical_item(19, StreamItemKind.ANSWER_DELTA, text_delta="late"),
    )

    async def collect() -> None:
        _ = [
            projection
            async for projection in iter_stream_consumer_projections(
                async_items(invalid_items)
            )
        ]

    try:
        asyncio.run(collect())
    except StreamValidationError as exc:
        assert "semantic stream item emitted after terminal outcome" in str(
            exc
        )
    else:
        raise AssertionError("late stream content was accepted")


def test_canonical_trace_rejects_missing_terminal() -> None:
    trace = canonical_stream_trace()
    invalid_items = trace.items[:-2]

    async def collect() -> None:
        _ = [
            projection
            async for projection in iter_stream_consumer_projections(
                async_items(invalid_items)
            )
        ]

    try:
        asyncio.run(collect())
    except StreamValidationError as exc:
        assert "stream missing terminal outcome" in str(exc)
    else:
        raise AssertionError("unterminated stream was accepted")


def test_public_projection_helpers_reject_unsupported_items() -> None:
    for project, message in (
        (
            lambda: model_cmds._stream_projection(object()),  # type: ignore[arg-type]
            "unsupported CLI stream item",
        ),
    ):
        try:
            project()
        except StreamValidationError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"{message} was accepted")


def _assert_sdk_projection(
    projections: tuple[StreamConsumerProjection, ...],
) -> None:
    assert {projection.stream_session_id for projection in projections} == {
        "conformance-stream"
    }
    assert {projection.run_id for projection in projections} == {
        "conformance-run"
    }
    assert {projection.turn_id for projection in projections} == {
        "conformance-turn"
    }
    assert [projection.sequence for projection in projections] == list(
        range(len(projections))
    )
    assert projections[-2].terminal_outcome is StreamTerminalOutcome.COMPLETED
    assert projections[-1].kind is StreamItemKind.STREAM_CLOSED
    _assert_tool_live_projection_order(projections)


def _assert_tool_live_projection_order(
    projections: tuple[StreamConsumerProjection, ...],
) -> None:
    tool_items = [
        projection
        for projection in projections
        if projection.channel is StreamChannel.TOOL_EXECUTION
    ]
    terminal_index = next(
        index
        for index, item in enumerate(tool_items)
        if item.kind is StreamItemKind.TOOL_EXECUTION_COMPLETED
    )
    live_items = tool_items[1:terminal_index]

    assert [item.kind for item in live_items] == [
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
    ]
    assert [_stream_item_data(item)["category"] for item in live_items] == [
        "stdout",
        "stderr",
        "log",
        "progress",
    ]
    assert all(
        item.sequence < tool_items[terminal_index].sequence
        for item in live_items
    )


def _assert_stdout_projection(
    projections: tuple[StreamConsumerProjection, ...],
) -> None:
    text = "".join(
        model_cmds._stream_text(projection) or ""
        for projection in projections
        if projection.channel is StreamChannel.ANSWER
    )
    assert text == "final answer"
    assert any(model_cmds._is_reasoning_stream_item(p) for p in projections)
    assert any(model_cmds._is_tool_stream_item(p) for p in projections)


async def _assert_cli_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    render_items = [
        projection
        async for projection in model_cmds._stream_render_projections(
            _CanonicalResponse(items),
            stream_session_id="cli-conformance-stream",
            run_id="cli-conformance-run",
            turn_id="cli-conformance-turn",
        )
    ]
    projections = tuple(render_items)
    assert [projection.sequence for projection in projections] == [
        item.sequence for item in items
    ]
    assert [projection.kind for projection in projections] == [
        item.kind for item in items
    ]
    assert [projection.channel for projection in projections] == [
        item.channel for item in items
    ]

    accumulator = ProtocolStreamAccumulator()
    for projection in projections:
        accumulator.add(projection)
    snapshot = accumulator.snapshot()
    assert snapshot.answer_text == "final answer"
    assert snapshot.reasoning_text == "plan"
    assert snapshot.tool_call_arguments[TOOL_CALL_ID] == '{"query":"docs"}'
    assert (
        snapshot.tool_execution_outputs[TOOL_CALL_ID] == "live warning trace"
    )
    assert snapshot.usage == {
        "input_tokens": 2,
        "output_tokens": 5,
        "total_tokens": 7,
    }
    assert snapshot.terminal_outcome is StreamTerminalOutcome.COMPLETED
    assert len(snapshot.flow_items) == 1
    assert snapshot.control_items[-2].kind is StreamItemKind.STREAM_COMPLETED
    assert snapshot.control_items[-1].kind is StreamItemKind.STREAM_CLOSED

    captured_frames: list[dict[str, object]] = []

    async def fake_tokens(*parameters: object, **_: object):
        state = cast(Any, parameters[0])
        captured_frames.append(
            {
                "thinking_text_tokens": list(state.reasoning_text_tokens),
                "tool_text_tokens": list(state.tool_text_tokens),
                "answer_text_tokens": list(state.answer_text_tokens),
                "total_tokens": state.total_tokens,
            }
        )
        yield None, "frame"

    args = SimpleNamespace(
        skip_display_reasoning_time=False,
        display_time_to_n_token=1,
        display_pause=0,
        start_thinking=False,
        display_probabilities=False,
        display_probabilities_maximum=0.0,
        display_probabilities_sample_minimum=0.0,
        display_answer_height_expand=False,
        display_answer_height=12,
        record=False,
    )
    theme = MagicMock()
    theme.token_frames = MagicMock(side_effect=fake_tokens)
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
            model_id="test-model",
            tokenizer_config=None,
            input_token_count=MagicMock(return_value=1),
        ),
        input_string="hi",
        response=_CanonicalResponse(items),
        display_tokens=0,
        dtokens_pick=0,
        refresh_per_second=1000,
        stop_signal=None,
        tool_events_limit=2,
        with_stats=True,
    )
    assert captured_frames
    assert captured_frames[-1]["thinking_text_tokens"] == ["plan"]
    assert captured_frames[-1]["tool_text_tokens"] == [
        '{"query":"',
        'docs"}',
        "live",
        " warning",
        " trace",
    ]
    assert captured_frames[-1]["answer_text_tokens"] == [
        "final ",
        "answer",
    ]
    assert captured_frames[-1]["total_tokens"] == 3


def _assert_flow_projection(
    projections: tuple[StreamConsumerProjection, ...],
) -> None:
    accumulator = ProtocolStreamAccumulator()
    for projection in projections:
        accumulator.add(projection)
    snapshot = accumulator.snapshot()
    assert len(snapshot.flow_items) == 1
    flow_item = snapshot.flow_items[0]
    assert flow_item.correlation.flow_run_id == "flow-1"
    assert flow_item.correlation.node_id == "node-1"
    assert flow_item.data == {
        "flow_id": "flow-1",
        "node": "node-1",
        "status": "started",
    }
    assert flow_item.metadata["event_type"] == "flow_node_started"


def _assert_chat_projection(
    projections: tuple[StreamConsumerProjection, ...],
    usage: object | None,
) -> None:
    text = "".join(
        chat._stream_text(projection) or "" for projection in projections
    )
    assert text == "final answer"
    ignored_text = [
        chat._stream_text(projection)
        for projection in projections
        if projection.channel
        in {
            StreamChannel.REASONING,
            StreamChannel.TOOL_CALL,
            StreamChannel.TOOL_EXECUTION,
            StreamChannel.FLOW,
            StreamChannel.USAGE,
            StreamChannel.CONTROL,
        }
    ]
    assert all(text is None for text in ignored_text)
    chat_usage = chat._chat_usage(usage)
    assert chat_usage is not None
    assert chat_usage.model_dump() == {
        "prompt_tokens": 2,
        "completion_tokens": 5,
        "total_tokens": 7,
    }
    assert (
        chat._chat_terminal_event(
            "chatcmpl-test",
            1,
            "test-model",
            StreamTerminalOutcome.COMPLETED,
        )
        is None
    )


def _assert_responses_projection(
    projections: tuple[StreamConsumerProjection, ...],
) -> None:
    event_names: list[str] = []
    tool_events: list[responses._ResponsesSSEEvent] = []
    adapter = responses._ResponsesSSEProjectionAdapter()
    for sequence, projection in enumerate(projections):
        event_names.extend(
            _sse_event_name(event) for event in adapter.switch(projection)
        )
        response_events = responses._token_to_sse_events(
            projection,
            sequence,
            adapter.active_tool_call_id,
        )
        tool_events.extend(
            event
            for event in response_events
            if event.event.startswith("response.tool_execution.")
        )
        event_names.extend(event.event for event in response_events)

    event_names.extend(_sse_event_name(event) for event in adapter.close())
    event_names.extend(
        event.event
        for event in responses._terminal_response_events(
            StreamTerminalOutcome.COMPLETED
        )
    )

    assert "response.reasoning_text.delta" in event_names
    assert event_names.count("response.function_call_arguments.delta") == 2
    assert "response.tool_execution.output" in event_names
    assert "response.tool_execution.progress" in event_names
    assert "response.usage.completed" in event_names
    assert event_names[-1] == "response.completed"

    tool_event_names = [event.event for event in tool_events]
    completed_index = tool_event_names.index(
        "response.tool_execution.completed"
    )
    assert tool_event_names[:completed_index] == [
        "response.tool_execution.started",
        "response.tool_execution.output",
        "response.tool_execution.output",
        "response.tool_execution.output",
        "response.tool_execution.progress",
    ]
    assert [
        event.data.get("data", {}).get("category")
        for event in tool_events[1:completed_index]
    ] == ["stdout", "stderr", "log", "progress"]


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
        logger=getLogger("test.streaming.conformance.mcp"),
        resource_store=mcp_router.MCPResourceStore(),
        base_path="/mcp",
        cancel_event=AsyncEvent(),
    ):
        payloads.extend(
            loads(part) for part in chunk.decode("utf-8").splitlines() if part
        )

    assert orchestrator.synced is True
    tool_result_index = next(
        index
        for index, payload in enumerate(payloads)
        if (
            payload.get("method") == "notifications/message"
            and _mcp_message(payload).get("type") == "tool.result"
        )
    )
    live_resource_indexes = [
        index
        for index, payload in enumerate(payloads)
        if (
            payload.get("method") == "notifications/resources/updated"
            and "delta" in _mcp_resource(payload)
        )
    ]
    assert len(live_resource_indexes) == 4
    assert all(index < tool_result_index for index in live_resource_indexes)
    result_payloads = [payload for payload in payloads if "result" in payload]
    assert len(result_payloads) == 1
    result = result_payloads[0]["result"]
    assert isinstance(result, dict)
    assert result["content"] == [{"type": "text", "text": "final answer"}]
    structured = result["structuredContent"]
    assert isinstance(structured, dict)
    assert structured["reasoning"] == "plan"
    assert structured["toolCalls"][0]["id"] == TOOL_CALL_ID
    assert structured["toolCalls"][0]["arguments"] == '{"query":"docs"}'
    assert [
        resource["name"]
        for resource in structured["toolCalls"][0]["resources"]
    ] == ["stdout", "stderr", "logs", "progress"]
    assert structured["usage"] == {
        "input_text_tokens": 2,
        "output_text_tokens": 5,
        "total_tokens": 7,
    }


async def _assert_a2a_projection(
    items: tuple[CanonicalStreamItem, ...],
) -> None:
    store = TaskStore()
    task_id = "conformance-task"
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
    assert artifacts[TOOL_CALL_ID]["kind"] == "tool_execution"
    tool_content = artifacts[TOOL_CALL_ID]["content"]
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
    assert artifact_updates
    assert status_updates[-1].status.state is a2a_types.TaskState.completed
    assert status_updates[-1].final is True


def _sse_event_name(message: str) -> str:
    return message.split("\n", maxsplit=1)[0].split(": ", maxsplit=1)[1]


def _sse_event_data(message: str) -> dict[str, object]:
    for line in message.splitlines():
        if line.startswith("data: "):
            payload = loads(line.split(": ", maxsplit=1)[1])
            assert isinstance(payload, dict)
            return cast(dict[str, object], payload)
    raise AssertionError("SSE message did not contain data")


async def _collect_consumer_projections(
    items: tuple[CanonicalStreamItem, ...],
) -> tuple[StreamConsumerProjection, ...]:
    return tuple(
        [
            projection
            async for projection in iter_stream_consumer_projections(
                async_items(items)
            )
        ]
    )


def _canonical_semantics(item: CanonicalStreamItem) -> tuple[object, ...]:
    return (
        item.sequence,
        item.kind,
        item.channel,
        item.visibility,
        item.correlation.to_trace_dict(),
        item.text_delta,
        item.data,
        item.usage,
        item.terminal_outcome,
        item.metadata,
    )


def _semantic_entry(
    sequence: int,
    kind: StreamItemKind,
    *,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    correlation: dict[str, object] | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    metadata: dict[str, object] | None = None,
) -> tuple[object, ...]:
    return (
        sequence,
        kind,
        stream_channel(kind),
        visibility,
        correlation or {},
        text_delta,
        data,
        usage,
        terminal_outcome,
        metadata or {},
    )


def _stream_item_data(item: StreamConsumerProjection) -> dict[str, object]:
    data = item.data
    assert isinstance(data, dict)
    return cast(dict[str, object], data)


def _mcp_message(payload: dict[str, object]) -> dict[str, object]:
    params = cast(dict[str, object], payload["params"])
    return cast(dict[str, object], params["message"])


def _mcp_resource(payload: dict[str, object]) -> dict[str, object]:
    params = cast(dict[str, object], payload["params"])
    resources = cast(list[dict[str, object]], params["resources"])
    return resources[0]
