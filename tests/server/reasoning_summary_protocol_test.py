from asyncio import Event as AsyncEvent
from asyncio import run
from collections.abc import AsyncIterator
from contextlib import contextmanager
from json import loads
from logging import getLogger
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import patch
from uuid import uuid4

from avalan.entities import MessageRole
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamVisibility,
)
from avalan.server.a2a import router as a2a_router
from avalan.server.a2a.router import A2AResponseTranslator
from avalan.server.entities import ChatCompletionRequest, ChatMessage
from avalan.server.routers import mcp as mcp_router

_STREAM_SESSION_ID = "protocol-summary-stream"
_RUN_ID = "protocol-summary-run"
_TURN_ID = "protocol-summary-turn"
_TOOL_CALL_ID = "protocol-tool-call"
_SUMMARY_TEXT = "private-summary"
_NATIVE_TEXT = "private-native"
_TOOL_TEXT = "tool-only"
_ANSWER_TEXT = "public-answer"


class _CanonicalResponse:
    input_token_count = 3
    output_token_count = 5

    def __init__(self, items: tuple[CanonicalStreamItem, ...]) -> None:
        self._items = items

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        return _iter_items(self._items)


class _SyncingOrchestrator:
    def __init__(self) -> None:
        self.synced = 0

    async def sync_messages(self) -> None:
        self.synced += 1


class _A2AUpdater:
    def __init__(self) -> None:
        self.artifacts: list[dict[str, object]] = []
        self.completed = 0

    async def add_artifact(self, parts: object, **kwargs: object) -> None:
        self.artifacts.append({"parts": parts, **kwargs})

    async def update_status(
        self, state: object, metadata: object = None
    ) -> None:
        _ = state, metadata

    async def complete(self) -> None:
        self.completed += 1

    async def cancel(self) -> None:
        raise AssertionError("completed trace must not cancel")

    async def failed(self) -> None:
        raise AssertionError("completed trace must not fail")


def _item(
    sequence: int,
    kind: StreamItemKind,
    *,
    text_delta: str | None = None,
    correlation: StreamItemCorrelation | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    representation: StreamReasoningRepresentation | None = None,
    ordinal: int | None = None,
) -> CanonicalStreamItem:
    if kind in (
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.REASONING_DONE,
    ):
        channel = StreamChannel.REASONING
    elif kind in (StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DONE):
        channel = StreamChannel.ANSWER
    elif kind.name.startswith("TOOL_CALL"):
        channel = StreamChannel.TOOL_CALL
    elif kind.name.startswith("TOOL_EXECUTION"):
        channel = StreamChannel.TOOL_EXECUTION
    elif kind is StreamItemKind.USAGE_COMPLETED:
        channel = StreamChannel.USAGE
    else:
        channel = StreamChannel.CONTROL
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=kind,
        channel=channel,
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=cast(Any, data),
        usage=cast(Any, usage),
        terminal_outcome=terminal_outcome,
        visibility=(
            StreamVisibility.PRIVATE
            if kind
            in (
                StreamItemKind.REASONING_DELTA,
                StreamItemKind.REASONING_DONE,
            )
            else StreamVisibility.PUBLIC
        ),
        reasoning_representation=representation,
        segment_instance_ordinal=ordinal,
    )


def _canonical_trace() -> tuple[CanonicalStreamItem, ...]:
    tool = StreamItemCorrelation(tool_call_id=_TOOL_CALL_ID)
    return (
        _item(0, StreamItemKind.STREAM_STARTED),
        _item(
            1,
            StreamItemKind.REASONING_DELTA,
            text_delta=_SUMMARY_TEXT,
            correlation=StreamItemCorrelation(
                protocol_item_id="summary-item",
                provider_output_index=0,
                model_continuation_id="continuation-1",
            ),
            representation=StreamReasoningRepresentation.SUMMARY,
            ordinal=0,
        ),
        _item(
            2,
            StreamItemKind.REASONING_DELTA,
            text_delta=_NATIVE_TEXT,
            correlation=StreamItemCorrelation(
                protocol_item_id="native-item",
                provider_output_index=1,
                model_continuation_id="continuation-1",
            ),
            representation=StreamReasoningRepresentation.NATIVE_TEXT,
            ordinal=1,
        ),
        _item(3, StreamItemKind.REASONING_DONE),
        _item(
            4,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation=tool,
            text_delta='{"query":"docs"}',
        ),
        _item(
            5,
            StreamItemKind.TOOL_CALL_READY,
            correlation=tool,
            data={"name": "search", "arguments": {"query": "docs"}},
        ),
        _item(6, StreamItemKind.TOOL_CALL_DONE, correlation=tool),
        _item(
            7,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            correlation=tool,
            data={"name": "search"},
        ),
        _item(
            8,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=tool,
            text_delta=_TOOL_TEXT,
            data={"category": "stdout", "content": _TOOL_TEXT},
        ),
        _item(
            9,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            correlation=tool,
            data={"result": _TOOL_TEXT},
        ),
        _item(10, StreamItemKind.ANSWER_DELTA, text_delta=_ANSWER_TEXT),
        _item(11, StreamItemKind.ANSWER_DONE),
        _item(
            12,
            StreamItemKind.USAGE_COMPLETED,
            usage={
                "input_text_tokens": 3,
                "output_text_tokens": 5,
                "total_tokens": 8,
            },
        ),
        _item(
            13,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
    )


async def _iter_items(
    items: tuple[CanonicalStreamItem, ...],
) -> AsyncIterator[CanonicalStreamItem]:
    for item in items:
        yield item


async def _project_mcp(
    items: tuple[CanonicalStreamItem, ...],
) -> tuple[dict[str, object], ...]:
    orchestrator = _SyncingOrchestrator()
    payloads: list[dict[str, object]] = []
    async for chunk in mcp_router._stream_mcp_response(
        request_id="protocol-summary-request",
        request_model=ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role=MessageRole.USER, content="question")],
            stream=True,
        ),
        response=cast(Any, _CanonicalResponse(items)),
        response_id=uuid4(),
        timestamp=1,
        progress_token="protocol-summary-progress",
        orchestrator=cast(Any, orchestrator),
        logger=getLogger("test.reasoning-summary.protocol.mcp"),
        resource_store=mcp_router.MCPResourceStore(),
        base_path="/mcp",
        cancel_event=AsyncEvent(),
    ):
        payloads.extend(
            cast(dict[str, object], loads(line))
            for line in chunk.decode("utf-8").splitlines()
            if line
        )
    assert orchestrator.synced == 1
    return tuple(payloads)


@contextmanager
def _fake_a2a_types() -> Iterator[None]:
    real_import_module = a2a_router.import_module
    fake_pb2 = SimpleNamespace(
        Part=lambda **kwargs: SimpleNamespace(**kwargs),
        TaskState=SimpleNamespace(TASK_STATE_WORKING="working"),
    )

    def import_module(name: str) -> object:
        if name == "a2a.types.a2a_pb2":
            return fake_pb2
        return real_import_module(name)

    with patch.object(a2a_router, "import_module", import_module):
        yield


async def _project_a2a(
    items: tuple[CanonicalStreamItem, ...],
) -> _A2AUpdater:
    updater = _A2AUpdater()
    with _fake_a2a_types():
        translator = A2AResponseTranslator(updater)
        for item in items:
            await translator.process(item)
        await translator.finish()
    assert updater.completed == 1
    return updater


def _mcp_result(payloads: tuple[dict[str, object], ...]) -> dict[str, object]:
    results = [
        payload["result"] for payload in payloads if "result" in payload
    ]
    assert len(results) == 1
    return cast(dict[str, object], results[0])


def _artifact_text(events: list[dict[str, object]]) -> str:
    return "".join(
        cast(str, getattr(part, "text", ""))
        for event in events
        for part in cast(list[object], event["parts"])
    )


def test_protocols_never_promote_summary_to_answer() -> None:
    async def exercise() -> None:
        items = _canonical_trace()
        mcp_result = _mcp_result(await _project_mcp(items))
        a2a = await _project_a2a(items)

        assert mcp_result["content"] == [
            {"type": "text", "text": _ANSWER_TEXT}
        ]
        mcp_answer = repr(mcp_result["content"])
        assert _SUMMARY_TEXT not in mcp_answer
        assert _NATIVE_TEXT not in mcp_answer
        assert _TOOL_TEXT not in mcp_answer
        structured = cast(dict[str, object], mcp_result["structuredContent"])
        assert structured["usage"] == {
            "input_text_tokens": 3,
            "output_text_tokens": 5,
            "total_tokens": 8,
        }
        assert _TOOL_TEXT in repr(structured["toolCalls"])
        assert _SUMMARY_TEXT not in repr(structured["toolCalls"])
        assert _NATIVE_TEXT not in repr(structured["toolCalls"])

        answer_events = [
            event
            for event in a2a.artifacts
            if event["artifact_id"] == "answer"
        ]
        assert _artifact_text(answer_events) == _ANSWER_TEXT
        a2a_answer = repr(answer_events)
        assert _SUMMARY_TEXT not in a2a_answer
        assert _NATIVE_TEXT not in a2a_answer
        assert _TOOL_TEXT not in a2a_answer
        tool_events = [
            event
            for event in a2a.artifacts
            if event["artifact_id"] != "answer"
            and not str(event["artifact_id"]).startswith("reasoning-")
        ]
        assert _TOOL_TEXT in repr(tool_events)
        assert _SUMMARY_TEXT not in repr(tool_events)
        assert _NATIVE_TEXT not in repr(tool_events)
        assert a2a.completed == 1

    run(exercise())


def test_mcp_and_a2a_preserve_representation() -> None:
    async def exercise() -> None:
        items = _canonical_trace()
        mcp_result = _mcp_result(await _project_mcp(items))
        a2a = await _project_a2a(items)

        structured = cast(dict[str, object], mcp_result["structuredContent"])
        mcp_segments = cast(
            list[dict[str, object]], structured["reasoningSegments"]
        )
        assert [segment["representation"] for segment in mcp_segments] == [
            "summary",
            "native_text",
        ]
        assert [segment["text"] for segment in mcp_segments] == [
            _SUMMARY_TEXT,
            _NATIVE_TEXT,
        ]

        reasoning_events = [
            event
            for event in a2a.artifacts
            if str(event["artifact_id"]).startswith("reasoning-")
            and cast(list[object], event["parts"])
        ]
        assert [
            cast(dict[str, object], event["metadata"])["representation"]
            for event in reasoning_events
        ] == ["summary", "native_text"]
        assert [_artifact_text([event]) for event in reasoning_events] == [
            _SUMMARY_TEXT,
            _NATIVE_TEXT,
        ]

    run(exercise())
