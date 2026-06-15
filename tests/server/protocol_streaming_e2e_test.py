import asyncio
from asyncio import Event as AsyncEvent
from collections.abc import AsyncIterator
from dataclasses import replace
from json import loads
from logging import getLogger
from uuid import uuid4

from a2a import types as a2a_types

from avalan.entities import MessageRole
from avalan.event import Event, EventType
from avalan.flow.stream import canonical_flow_event_listener
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
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

    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        return _iter_items(self._items)


class _SyncingOrchestrator:
    def __init__(self) -> None:
        self.synced = False

    async def sync_messages(self) -> None:
        self.synced = True


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
            text_delta=" update",
            data={"category": "stdout", "content": " update"},
            metadata={"tool_name": "search"},
        ),
        _stream_item(
            11,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            correlation=call,
        ),
        _stream_item(12, StreamItemKind.ANSWER_DELTA, text_delta="final "),
        _stream_item(13, StreamItemKind.ANSWER_DELTA, text_delta="answer"),
        _stream_item(14, StreamItemKind.ANSWER_DONE),
        _stream_item(15, StreamItemKind.USAGE_COMPLETED, usage=usage),
        _stream_item(
            16,
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
    assert len(resources) == 2
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
    assert len(tool_call["resources"]) == 1
    assert tool_call["resources"][0]["name"] == "stdout"


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
    assert tool_content[2]["text"] == "live"
    assert tool_content[3]["text"] == " update"
    assert tool_content[4]["status"] == "completed"
    assert artifact_updates
    assert status_updates[-1].status.state is a2a_types.TaskState.completed
    assert status_updates[-1].final is True
