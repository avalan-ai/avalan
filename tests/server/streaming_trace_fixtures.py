from collections.abc import AsyncIterator

from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamGoldenTrace,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamVisibility,
)

STREAM_SESSION_ID = "conformance-stream"
RUN_ID = "conformance-run"
TURN_ID = "conformance-turn"
TOOL_CALL_ID = "call-1"
TERMINAL_ERROR_DATA: dict[str, object] = {
    "code": "upstream_error",
    "message": "upstream failed",
}
TERMINAL_CANCELLED_DATA: dict[str, object] = {
    "reason": "client_cancelled",
}
TERMINAL_TRACE_USAGE: dict[str, object] = {
    "input_tokens": 1,
    "output_tokens": 2,
    "total_tokens": 3,
}


def canonical_stream_trace() -> StreamGoldenTrace:
    return StreamGoldenTrace(
        name="streaming-conformance-reasoning-tool-flow",
        items=(
            canonical_item(0, StreamItemKind.STREAM_STARTED),
            canonical_item(
                1,
                StreamItemKind.FLOW_EVENT,
                correlation=StreamItemCorrelation(
                    flow_run_id="flow-1",
                    node_id="node-1",
                ),
                data={
                    "flow_id": "flow-1",
                    "node": "node-1",
                    "status": "started",
                },
                metadata={"event_type": "flow_node_started"},
            ),
            canonical_item(
                2,
                StreamItemKind.REASONING_DELTA,
                text_delta="plan",
                visibility=StreamVisibility.PRIVATE,
            ),
            canonical_item(
                3,
                StreamItemKind.REASONING_DONE,
                visibility=StreamVisibility.PRIVATE,
            ),
            canonical_item(
                4,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                text_delta='{"query":"',
                data={"name": "search", "arguments": {"query": "docs"}},
            ),
            canonical_item(
                5,
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                text_delta='docs"}',
                data={"name": "search", "arguments": {"query": "docs"}},
            ),
            canonical_item(
                6,
                StreamItemKind.TOOL_CALL_READY,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                data={"name": "search", "arguments": {"query": "docs"}},
            ),
            canonical_item(
                7,
                StreamItemKind.TOOL_CALL_DONE,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
            ),
            canonical_item(
                8,
                StreamItemKind.TOOL_EXECUTION_STARTED,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                metadata={"tool_name": "search"},
            ),
            canonical_item(
                9,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                text_delta="live",
                data={"category": "stdout", "content": "live"},
                metadata={"tool_name": "search"},
            ),
            canonical_item(
                10,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                text_delta=" warning",
                data={"category": "stderr", "content": " warning"},
                metadata={"tool_name": "search"},
            ),
            canonical_item(
                11,
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                text_delta=" trace",
                data={"category": "log", "content": " trace"},
                metadata={"tool_name": "search"},
            ),
            canonical_item(
                12,
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                data={
                    "category": "progress",
                    "content": "50%",
                    "progress": 0.5,
                },
                metadata={"tool_name": "search"},
            ),
            canonical_item(
                13,
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                correlation=StreamItemCorrelation(tool_call_id=TOOL_CALL_ID),
                data={"result": "live warning trace"},
                metadata={"tool_name": "search"},
            ),
            canonical_item(
                14,
                StreamItemKind.ANSWER_DELTA,
                text_delta="final ",
            ),
            canonical_item(
                15,
                StreamItemKind.ANSWER_DELTA,
                text_delta="answer",
            ),
            canonical_item(16, StreamItemKind.ANSWER_DONE),
            canonical_item(
                17,
                StreamItemKind.USAGE_COMPLETED,
                usage={
                    "input_tokens": 2,
                    "output_tokens": 5,
                    "total_tokens": 7,
                },
            ),
            canonical_item(
                18,
                StreamItemKind.STREAM_COMPLETED,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            canonical_item(19, StreamItemKind.STREAM_CLOSED),
        ),
        description=(
            "Canonical trace with answer, reasoning, tool, flow, usage, "
            "and terminal items."
        ),
    )


def terminal_outcome_trace(
    outcome: StreamTerminalOutcome,
) -> StreamGoldenTrace:
    assert isinstance(outcome, StreamTerminalOutcome)
    terminal_data: object | None
    if outcome is StreamTerminalOutcome.ERRORED:
        terminal_kind = StreamItemKind.STREAM_ERRORED
        terminal_data = TERMINAL_ERROR_DATA
    elif outcome is StreamTerminalOutcome.CANCELLED:
        terminal_kind = StreamItemKind.STREAM_CANCELLED
        terminal_data = TERMINAL_CANCELLED_DATA
    else:
        terminal_kind = StreamItemKind.STREAM_COMPLETED
        terminal_data = None

    return StreamGoldenTrace(
        name=f"streaming-terminal-{outcome.value}",
        items=(
            canonical_item(0, StreamItemKind.STREAM_STARTED),
            canonical_item(1, StreamItemKind.ANSWER_DELTA, text_delta="ok"),
            canonical_item(2, StreamItemKind.ANSWER_DONE),
            canonical_item(
                3,
                StreamItemKind.USAGE_COMPLETED,
                usage=TERMINAL_TRACE_USAGE,
            ),
            canonical_item(
                4,
                terminal_kind,
                data=terminal_data,
                terminal_outcome=outcome,
            ),
            canonical_item(5, StreamItemKind.STREAM_CLOSED),
        ),
        description=f"Canonical trace ending with {outcome.value}.",
    )


def canonical_item(
    sequence: int,
    kind: StreamItemKind,
    *,
    correlation: StreamItemCorrelation | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    metadata: dict[str, object] | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=STREAM_SESSION_ID,
        run_id=RUN_ID,
        turn_id=TURN_ID,
        sequence=sequence,
        kind=kind,
        channel=stream_channel(kind),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
        metadata=metadata or {},  # type: ignore[arg-type]
        visibility=visibility,
    )


def stream_channel(kind: StreamItemKind) -> StreamChannel:
    if kind in (StreamItemKind.ANSWER_DELTA, StreamItemKind.ANSWER_DONE):
        return StreamChannel.ANSWER
    if kind in (StreamItemKind.REASONING_DELTA, StreamItemKind.REASONING_DONE):
        return StreamChannel.REASONING
    if kind in (
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
    ):
        return StreamChannel.TOOL_CALL
    if kind in (
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
    ):
        return StreamChannel.TOOL_EXECUTION
    if kind is StreamItemKind.FLOW_EVENT:
        return StreamChannel.FLOW
    if kind is StreamItemKind.USAGE_COMPLETED:
        return StreamChannel.USAGE
    return StreamChannel.CONTROL


async def async_items(
    items: tuple[CanonicalStreamItem, ...],
) -> AsyncIterator[CanonicalStreamItem]:
    for item in items:
        yield item
