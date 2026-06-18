from collections.abc import AsyncIterator
from json import loads
from unittest import IsolatedAsyncioTestCase

from avalan.cli.commands import model as model_cmds
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamGoldenTrace,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    accumulate_canonical_stream_items,
    iter_stream_consumer_projections,
)
from avalan.server.routers import chat, responses

_STREAM_SESSION_ID = "consumer-stream"
_RUN_ID = "consumer-run"
_TURN_ID = "consumer-turn"
_TOOL_CALL_ID = "call-1"


def _item(
    sequence: int,
    kind: StreamItemKind,
    *,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    correlation: StreamItemCorrelation | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=_STREAM_SESSION_ID,
        run_id=_RUN_ID,
        turn_id=_TURN_ID,
        sequence=sequence,
        kind=kind,
        channel=(
            StreamChannel.TOOL_EXECUTION
            if kind.name.startswith("TOOL_EXECUTION")
            else (
                StreamChannel.TOOL_CALL
                if kind.name.startswith("TOOL_CALL")
                else (
                    StreamChannel.USAGE
                    if kind.name.startswith("USAGE")
                    else (
                        StreamChannel.ANSWER
                        if kind.name.startswith("ANSWER")
                        else (
                            StreamChannel.REASONING
                            if kind.name.startswith("REASONING")
                            else StreamChannel.CONTROL
                        )
                    )
                )
            )
        ),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
        visibility=visibility,
    )


def _golden_items() -> tuple[CanonicalStreamItem, ...]:
    tool_correlation = StreamItemCorrelation(tool_call_id=_TOOL_CALL_ID)
    return (
        _item(0, StreamItemKind.STREAM_STARTED),
        _item(1, StreamItemKind.ANSWER_DELTA, text_delta="lead "),
        _item(
            2,
            StreamItemKind.REASONING_DELTA,
            text_delta="plan",
            visibility=StreamVisibility.PRIVATE,
        ),
        _item(
            3,
            StreamItemKind.REASONING_DONE,
            visibility=StreamVisibility.PRIVATE,
        ),
        _item(
            4,
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            correlation=tool_correlation,
            text_delta='{"expression":"2+2"}',
            data={
                "name": "math.calculator",
                "arguments": {"expression": "2+2"},
            },
        ),
        _item(
            5,
            StreamItemKind.TOOL_CALL_READY,
            correlation=tool_correlation,
            data={
                "name": "math.calculator",
                "arguments": {"expression": "2+2"},
            },
        ),
        _item(6, StreamItemKind.TOOL_CALL_DONE, correlation=tool_correlation),
        _item(
            7,
            StreamItemKind.TOOL_EXECUTION_STARTED,
            correlation=tool_correlation,
        ),
        _item(
            8,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            correlation=tool_correlation,
            text_delta="4\n",
            data={"category": "stdout"},
        ),
        _item(
            9,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            correlation=tool_correlation,
            data={"category": "progress", "content": "50%", "progress": 0.5},
        ),
        _item(
            10,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            correlation=tool_correlation,
            data={"result": "4"},
        ),
        _item(11, StreamItemKind.ANSWER_DELTA, text_delta="tail"),
        _item(12, StreamItemKind.ANSWER_DONE),
        _item(
            13,
            StreamItemKind.USAGE_COMPLETED,
            usage={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        ),
        _item(
            14,
            StreamItemKind.STREAM_COMPLETED,
            terminal_outcome=StreamTerminalOutcome.COMPLETED,
        ),
        _item(15, StreamItemKind.STREAM_CLOSED),
    )


async def _async_items(
    items: tuple[CanonicalStreamItem, ...],
) -> AsyncIterator[CanonicalStreamItem]:
    for item in items:
        yield item


def _responses_event_names(
    projections: tuple[StreamConsumerProjection, ...],
) -> list[str]:
    adapter = responses._ResponsesSSEProjectionAdapter()
    names: list[str] = []
    for sequence, projection in enumerate(projections):
        names.extend(
            event.split("\n", maxsplit=1)[0].split(": ", maxsplit=1)[1]
            for event in adapter.switch(projection)
        )
        names.extend(
            event.event
            for event in responses._token_to_sse_events(
                projection,
                sequence,
                adapter.active_tool_call_id,
            )
        )
    names.extend(
        event.split("\n", maxsplit=1)[0].split(": ", maxsplit=1)[1]
        for event in adapter.close()
    )
    names.extend(
        event.event
        for event in responses._terminal_response_events(
            StreamTerminalOutcome.COMPLETED
        )
    )
    return names


class PrimaryConsumerProjectionGoldenTestCase(IsolatedAsyncioTestCase):
    async def test_primary_consumers_project_same_canonical_trace(
        self,
    ) -> None:
        items = _golden_items()
        trace = StreamGoldenTrace(
            name="primary-consumer-reasoning-tool-golden",
            items=items,
        )
        projections = tuple(
            [
                projection
                async for projection in iter_stream_consumer_projections(
                    _async_items(trace.items)
                )
            ]
        )

        accumulator = accumulate_canonical_stream_items(trace.items)
        self.assertEqual(accumulator.answer_text, "lead tail")
        self.assertEqual(accumulator.reasoning_text, "plan")
        self.assertEqual(
            accumulator.tool_call_arguments[_TOOL_CALL_ID],
            '{"expression":"2+2"}',
        )
        self.assertEqual(
            accumulator.tool_execution_outputs[_TOOL_CALL_ID], "4\n"
        )
        self.assertEqual(
            [projection.sequence for projection in projections],
            list(range(len(projections))),
        )
        self.assertIs(projections[2].visibility, StreamVisibility.PRIVATE)
        self.assertIs(projections[3].visibility, StreamVisibility.PRIVATE)

        stdout_text = "".join(
            model_cmds._stream_text(projection) or ""
            for projection in projections
            if projection.channel is StreamChannel.ANSWER
        )
        self.assertEqual(stdout_text, "lead tail")
        self.assertTrue(
            any(model_cmds._is_reasoning_stream_item(p) for p in projections)
        )
        self.assertTrue(
            any(model_cmds._is_tool_stream_item(p) for p in projections)
        )

        chat_text = "".join(
            chat._stream_text(chat._stream_projection(item, item.sequence))
            or ""
            for item in trace.items
        )
        self.assertEqual(chat_text, "lead tail")
        self.assertEqual(
            chat._chat_usage(accumulator.final_usage).model_dump(),
            {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        )

        response_events = _responses_event_names(projections)
        self.assertEqual(
            response_events,
            [
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.tool_execution.started",
                "response.tool_execution.output",
                "response.tool_execution.progress",
                "response.tool_execution.completed",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.usage.completed",
                "response.completed",
            ],
        )
        tool_delta = loads(
            responses._token_to_sse(projections[4], 4)[0].split(
                "data: ", maxsplit=1
            )[1]
        )
        self.assertEqual(tool_delta["id"], _TOOL_CALL_ID)
        tool_payload = loads(tool_delta["delta"])
        self.assertEqual(tool_payload["name"], "math.calculator")

    async def test_primary_consumer_projection_rejects_late_content(
        self,
    ) -> None:
        items = (
            _item(0, StreamItemKind.STREAM_STARTED),
            _item(1, StreamItemKind.ANSWER_DELTA, text_delta="ok"),
            _item(2, StreamItemKind.ANSWER_DONE),
            _item(
                3,
                StreamItemKind.STREAM_COMPLETED,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            _item(4, StreamItemKind.ANSWER_DELTA, text_delta="late"),
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            _ = [
                projection
                async for projection in iter_stream_consumer_projections(
                    _async_items(items)
                )
            ]
