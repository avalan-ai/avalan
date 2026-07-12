from collections.abc import AsyncIterator, Sized
from dataclasses import FrozenInstanceError
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.display_reducer import (
    CliStreamSnapshotReducer,
    default_cli_stream_clock,
    iter_cli_canonical_stream_snapshots,
    iter_cli_stream_snapshots,
)
from avalan.cli.display_safety import MAX_SUMMARY_CHARS
from avalan.cli.display_snapshot import CliStreamSnapshot
from avalan.event import Event, EventType
from avalan.model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    canonical_item_from_consumer_projection,
    validate_canonical_stream_items,
)
from avalan.tool.display import TOOL_DISPLAY_PROJECTION_METADATA_KEY


def _config(**overrides: object) -> CliStreamDisplayConfig:
    values = {
        "quiet": False,
        "stats": True,
        "display_tools": True,
        "display_events": True,
        "display_tools_events": 3,
        "record": False,
        "interactive": True,
        "refresh_per_second": 10,
        "answer_height": 12,
        "answer_height_expand": False,
        "display_tokens": 2,
        "display_pause": 0,
        "display_probabilities": True,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": 2,
        "display_reasoning_time": True,
    }
    values.update(overrides)
    return CliStreamDisplayConfig(**values)


def _projection(
    kind: StreamItemKind,
    sequence: int,
    *,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    metadata: dict[str, object] | None = None,
    tool_call_id: str | None = None,
    model_continuation_id: str | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    provider_family: str | None = None,
    provider_event_type: str | None = None,
) -> StreamConsumerProjection:
    return StreamConsumerProjection(
        stream_session_id="session",
        run_id="run",
        turn_id="turn",
        sequence=sequence,
        kind=kind,
        channel=_channel(kind),
        correlation=StreamItemCorrelation(
            model_continuation_id=model_continuation_id,
            tool_call_id=tool_call_id,
        ),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
        visibility=(
            StreamVisibility.PRIVATE
            if kind is StreamItemKind.REASONING_DELTA
            else StreamVisibility.PUBLIC
        ),
        reasoning_representation=(
            StreamReasoningRepresentation.NATIVE_TEXT
            if kind is StreamItemKind.REASONING_DELTA
            else None
        ),
        segment_instance_ordinal=(
            0 if kind is StreamItemKind.REASONING_DELTA else None
        ),
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
        provider_family=provider_family,
        provider_event_type=provider_event_type,
    )


def _channel(kind: StreamItemKind) -> StreamChannel:
    if kind in (
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.ANSWER_DONE,
    ):
        return StreamChannel.ANSWER
    if kind in (
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.REASONING_DONE,
    ):
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
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    ):
        return StreamChannel.TOOL_EXECUTION
    if kind is StreamItemKind.FLOW_EVENT:
        return StreamChannel.FLOW
    if kind in (
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    ):
        return StreamChannel.USAGE
    return StreamChannel.CONTROL


class FakeClock:
    def __init__(self, *values: float) -> None:
        self._values = list(values)
        self.calls = 0

    def __call__(self) -> float:
        value = self._values[self.calls]
        self.calls += 1
        return value


class AsyncCanonicalSource:
    def __init__(self, *items: CanonicalStreamItem) -> None:
        self._items = items
        self.iterations = 0
        self.yields = 0

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        self.iterations += 1
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[CanonicalStreamItem]:
        for item in self._items:
            self.yields += 1
            yield item


def _canonical(projection: StreamConsumerProjection) -> CanonicalStreamItem:
    return canonical_item_from_consumer_projection(projection)


class DisplayReducerTestCase(TestCase):
    def test_default_clock_returns_float(self) -> None:
        self.assertIsInstance(default_cli_stream_clock(), float)

    def test_reduces_canonical_trace_to_display_snapshot(self) -> None:
        clock = FakeClock(
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            20.0,
            21.0,
            22.0,
            23.0,
        )
        reducer = CliStreamSnapshotReducer(_config(), clock=clock)
        tool_id = "tool-1"
        projections = (
            _projection(StreamItemKind.STREAM_STARTED, 0),
            _projection(
                StreamItemKind.REASONING_DELTA,
                1,
                text_delta="think",
                metadata={"token_id": 7, "probability": 0.25},
            ),
            _projection(
                StreamItemKind.ANSWER_DELTA,
                2,
                text_delta="hi",
                metadata={"token_id": 8, "probability": 0.5},
            ),
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                3,
                text_delta='{"x":',
                tool_call_id=tool_id,
            ),
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                4,
                text_delta=" 1}",
                tool_call_id=tool_id,
            ),
            _projection(
                StreamItemKind.TOOL_CALL_READY,
                5,
                data={"name": "calc"},
                tool_call_id=tool_id,
            ),
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                6,
                tool_call_id=tool_id,
                provider_family="local",
            ),
            _projection(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                7,
                text_delta="stdout",
                tool_call_id=tool_id,
            ),
            _projection(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                8,
                data={"progress": 0.5},
                tool_call_id=tool_id,
            ),
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                9,
                data={"result": 2},
                tool_call_id=tool_id,
            ),
            _projection(
                StreamItemKind.FLOW_EVENT,
                10,
                data={"state": "done"},
                metadata={"secret": "hidden"},
            ),
            _projection(
                StreamItemKind.STREAM_DIAGNOSTIC,
                11,
                text_delta="note",
            ),
            _projection(
                StreamItemKind.USAGE_UPDATE,
                12,
                usage={
                    "input_tokens": 3,
                    "output_tokens": 4,
                    "reasoning_usage_tokens": 1,
                    "total_tokens": 10,
                },
            ),
            _projection(
                StreamItemKind.STREAM_COMPLETED,
                13,
                usage={"cached_input_tokens": 2},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        snapshot = reducer.snapshot()
        self.assertEqual(snapshot.build_stats.snapshots_built, 1)
        for projection in projections:
            snapshot = reducer.reduce_projection(projection)

        self.assertEqual(snapshot.answer_text, "hi")
        self.assertEqual(snapshot.reasoning_text, "think")
        self.assertEqual(snapshot.tool_call_request_text, '{"x": 1}')
        self.assertEqual(snapshot.token_counts.answer_tokens, 1)
        self.assertEqual(snapshot.token_counts.reasoning_tokens, 1)
        self.assertEqual(snapshot.token_counts.tool_call_tokens, 2)
        self.assertEqual(snapshot.token_counts.display_tokens, 2)
        self.assertEqual(snapshot.token_counts.total_tokens, 10)
        self.assertEqual(snapshot.token_counts.input_tokens, 3)
        self.assertEqual(snapshot.token_counts.cached_input_tokens, 2)
        self.assertEqual(snapshot.token_counts.output_tokens, 4)
        self.assertEqual(snapshot.token_counts.reasoning_usage_tokens, 1)
        self.assertEqual(snapshot.timing.started_at, 10.0)
        self.assertEqual(snapshot.timing.updated_at, 23.0)
        self.assertEqual(snapshot.timing.finished_at, 23.0)
        self.assertEqual(snapshot.timing.elapsed_seconds, 13.0)
        self.assertEqual(snapshot.timing.first_token_seconds, 1.0)
        self.assertEqual(snapshot.timing.reasoning_seconds, 1.0)
        self.assertEqual(snapshot.timing.time_to_n_token_seconds, 2.0)
        self.assertTrue(snapshot.terminal.completed)
        self.assertEqual(snapshot.terminal.outcome, "completed")
        self.assertEqual(snapshot.terminal.sequence, 13)
        self.assertEqual(snapshot.active_tools, ())
        self.assertEqual(snapshot.completed_tools[0].tool_call_id, tool_id)
        self.assertEqual(snapshot.completed_tools[0].name, "calc")
        self.assertEqual(snapshot.completed_tools[0].status, "completed")
        self.assertEqual(snapshot.completed_tools[0].elapsed_seconds, 3.0)
        self.assertEqual(snapshot.tool_results[0].name, "calc")
        self.assertEqual(snapshot.tool_results[0].status, "result")
        self.assertEqual(snapshot.tool_results[0].arguments_count, 1)
        self.assertIn('"result": 2', snapshot.tool_results[0].result_summary)
        self.assertEqual(
            [event.event_type for event in snapshot.events],
            ["flow.event", "stream.diagnostic"],
        )
        self.assertIn("<redacted>", snapshot.events[0].observability_summary)
        self.assertIn('"text": "note"', snapshot.events[1].payload_summary)
        self.assertEqual(
            [token.token_id for token in snapshot.display_tokens], [7, 8]
        )
        self.assertEqual(
            [summary.kind for summary in snapshot.usage_summaries],
            ["usage.update", "stream.completed"],
        )
        self.assertEqual(clock.calls, len(projections))

    def test_delayed_calculator_progress_snapshots_lifecycle(self) -> None:
        clock = FakeClock(1.0, 2.0, 3.0, 5.0, 7.0, 8.0)
        reducer = CliStreamSnapshotReducer(_config(), clock=clock)
        tool_id = "calculator-call"

        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta='{"expression": "(4 + 6) * 5 / 2"}',
                tool_call_id=tool_id,
            )
        )
        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_CALL_READY,
                2,
                data={"name": "math.calculator"},
                tool_call_id=tool_id,
            )
        )
        start_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                3,
                data={"name": "math.calculator"},
                tool_call_id=tool_id,
            )
        )
        first_progress_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                4,
                data={"name": "math.calculator", "progress": 0.25},
                tool_call_id=tool_id,
            )
        )
        second_progress_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                5,
                data={"name": "math.calculator", "progress": 0.75},
                tool_call_id=tool_id,
            )
        )
        completed_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                6,
                data={"name": "math.calculator", "result": 25},
                tool_call_id=tool_id,
            )
        )

        self.assertEqual(len(start_snapshot.active_tools), 1)
        self.assertEqual(
            start_snapshot.active_tools[0].name, "math.calculator"
        )
        self.assertEqual(start_snapshot.active_tools[0].status, "active")
        self.assertEqual(start_snapshot.active_tools[0].started_at, 3.0)
        self.assertIsNone(start_snapshot.active_tools[0].updated_at)
        self.assertEqual(start_snapshot.completed_tools, ())
        self.assertEqual(start_snapshot.tool_results, ())

        self.assertEqual(len(first_progress_snapshot.active_tools), 1)
        self.assertEqual(
            first_progress_snapshot.active_tools[0].name, "math.calculator"
        )
        self.assertEqual(
            first_progress_snapshot.active_tools[0].started_at, 3.0
        )
        self.assertEqual(
            first_progress_snapshot.active_tools[0].updated_at, 5.0
        )
        self.assertEqual(first_progress_snapshot.completed_tools, ())
        self.assertEqual(first_progress_snapshot.tool_results, ())

        self.assertEqual(len(second_progress_snapshot.active_tools), 1)
        self.assertEqual(
            second_progress_snapshot.active_tools[0].name, "math.calculator"
        )
        self.assertEqual(
            second_progress_snapshot.active_tools[0].started_at, 3.0
        )
        self.assertLess(
            second_progress_snapshot.active_tools[0].started_at,
            second_progress_snapshot.active_tools[0].updated_at,
        )
        self.assertEqual(
            second_progress_snapshot.active_tools[0].updated_at, 7.0
        )
        self.assertGreater(
            second_progress_snapshot.active_tools[0].updated_at,
            first_progress_snapshot.active_tools[0].updated_at,
        )
        self.assertEqual(second_progress_snapshot.completed_tools, ())
        self.assertEqual(second_progress_snapshot.tool_results, ())

        self.assertEqual(completed_snapshot.active_tools, ())
        self.assertEqual(len(completed_snapshot.completed_tools), 1)
        self.assertEqual(
            completed_snapshot.completed_tools[0].name, "math.calculator"
        )
        self.assertEqual(
            completed_snapshot.completed_tools[0].status, "completed"
        )
        self.assertEqual(
            completed_snapshot.completed_tools[0].elapsed_seconds, 5.0
        )
        self.assertEqual(completed_snapshot.completed_tools[0].started_at, 3.0)
        self.assertEqual(completed_snapshot.completed_tools[0].updated_at, 7.0)
        self.assertEqual(len(completed_snapshot.tool_results), 1)
        self.assertEqual(
            completed_snapshot.tool_results[0].name, "math.calculator"
        )
        self.assertEqual(completed_snapshot.tool_results[0].status, "result")
        self.assertEqual(
            completed_snapshot.tool_results[0].elapsed_seconds, 5.0
        )
        self.assertEqual(completed_snapshot.tool_results[0].arguments_count, 1)
        self.assertIn("25", completed_snapshot.tool_results[0].result_summary)
        self.assertEqual(clock.calls, 6)

    def test_tool_start_projection_appears_in_active_snapshot(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0),
        )
        projection_payload = {
            "action": "search",
            "target": "src/avalan",
            "summary": "Search source files.",
        }

        snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                data={"name": "shell.rg"},
                metadata={
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: projection_payload
                },
                tool_call_id="tool-1",
            )
        )

        self.assertEqual(len(snapshot.active_tools), 1)
        display_projection = snapshot.active_tools[0].display_projection
        assert display_projection is not None
        self.assertEqual(display_projection.action, "search")
        self.assertEqual(display_projection.target, "src/avalan")
        self.assertEqual(display_projection.summary, "Search source files.")

    def test_terminal_projection_appears_in_completed_and_result_snapshots(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0, 2.0),
        )
        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                data={"name": "math.calculator"},
                tool_call_id="tool-1",
            )
        )
        terminal_payload = {
            "action": "finish",
            "target": "math.calculator",
            "summary": "Calculated result.",
            "status": "completed",
            "outcome": "result",
        }

        snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                1,
                data={"name": "math.calculator", "result": 25},
                metadata={
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: terminal_payload
                },
                tool_call_id="tool-1",
            )
        )

        self.assertEqual(snapshot.active_tools, ())
        self.assertEqual(len(snapshot.completed_tools), 1)
        completed_projection = snapshot.completed_tools[0].display_projection
        result_projection = snapshot.tool_results[0].display_projection
        assert completed_projection is not None
        assert result_projection is not None
        self.assertEqual(completed_projection.action, "finish")
        self.assertEqual(result_projection.outcome, "result")
        self.assertIn("25", snapshot.tool_results[0].result_summary)

    def test_error_and_cancelled_terminal_projections_are_retained(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0, 2.0),
        )
        error_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                0,
                data={"name": "lookup", "message": "bad"},
                metadata={
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                        "action": "finish",
                        "target": "lookup",
                        "status": "error",
                        "outcome": "error",
                        "severity": "error",
                    }
                },
                tool_call_id="tool-error",
            )
        )
        cancelled_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                1,
                data={"name": "lookup", "message": "cancelled"},
                metadata={
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                        "action": "skip",
                        "target": "lookup",
                        "status": "cancelled",
                        "outcome": "tool_call.cancelled",
                    }
                },
                tool_call_id="tool-cancelled",
            )
        )

        error_projection = error_snapshot.completed_tools[0].display_projection
        cancelled_projection = cancelled_snapshot.completed_tools[
            1
        ].display_projection
        assert error_projection is not None
        assert cancelled_projection is not None
        self.assertEqual(error_projection.status, "error")
        self.assertEqual(cancelled_projection.status, "cancelled")
        self.assertEqual(
            cancelled_snapshot.tool_results[1].display_projection,
            cancelled_projection,
        )

    def test_corrupt_projection_metadata_does_not_break_reduction(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0, 2.0),
        )
        start_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                data={"name": "lookup"},
                metadata={
                    TOOL_DISPLAY_PROJECTION_METADATA_KEY: {
                        "action": 42,
                    }
                },
                tool_call_id="tool-1",
            )
        )
        final_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                1,
                data={"name": "lookup", "result": {"ok": True}},
                metadata={TOOL_DISPLAY_PROJECTION_METADATA_KEY: "bad"},
                tool_call_id="tool-1",
            )
        )

        self.assertIsNone(start_snapshot.active_tools[0].display_projection)
        self.assertIsNone(final_snapshot.completed_tools[0].display_projection)
        self.assertIsNone(final_snapshot.tool_results[0].display_projection)
        self.assertIn("ok", final_snapshot.tool_results[0].result_summary)

    def test_side_channel_events_are_summaries_and_do_not_change_terminal(
        self,
    ) -> None:
        clock = FakeClock(1.0, 2.0, 3.0)
        reducer = CliStreamSnapshotReducer(_config(), clock=clock)

        reducer.reduce_projection(
            _projection(StreamItemKind.STREAM_STARTED, 0)
        )
        reducer.reduce_event(
            Event(type=EventType.TOKEN_GENERATED, payload={"token": "x"})
        )
        reducer.reduce_event(
            Event(type=EventType.TOOL_RESULT, payload={"result": "x"})
        )
        reducer.reduce_event(Event(type=EventType.START, payload={"ok": True}))
        terminal = reducer.reduce_projection(
            _projection(
                StreamItemKind.STREAM_COMPLETED,
                1,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        )
        late = reducer.reduce_event(
            Event(
                type=EventType.MODEL_EXECUTE_AFTER,
                payload={"secret": "hidden", "value": "ok"},
                started=4.0,
                finished=5.0,
                elapsed=1.0,
            )
        )
        closed = reducer.reduce_projection(
            _projection(StreamItemKind.STREAM_CLOSED, 2)
        )
        duplicate_terminal = reducer.reduce_projection(
            _projection(
                StreamItemKind.STREAM_CANCELLED,
                3,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            )
        )

        self.assertEqual(
            [event.event_type for event in terminal.events], ["start"]
        )
        self.assertEqual(
            [event.event_type for event in late.events],
            ["start", "model_execute_after"],
        )
        self.assertEqual(late.terminal.sequence, terminal.terminal.sequence)
        self.assertEqual(late.timing.finished_at, terminal.timing.finished_at)
        self.assertEqual(
            closed.timing.finished_at, terminal.timing.finished_at
        )
        self.assertEqual(
            duplicate_terminal.terminal.sequence, terminal.terminal.sequence
        )
        self.assertEqual(clock.calls, 2)

    def test_side_channel_tool_events_follow_tool_display_config(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=True),
        )
        tool_call = {
            "id": "tool-1",
            "name": "search",
            "arguments": {"api_key": "secret", "query": "weather"},
        }

        reducer.reduce_event(
            Event(
                type=EventType.TOOL_PROCESS,
                payload=[tool_call],
            )
        )
        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": tool_call},
                started=1.0,
            )
        )
        reducer.reduce_event(Event(type=EventType.START, payload={"ok": True}))
        reducer.reduce_event(
            Event(type=EventType.TOKEN_GENERATED, payload={"token": "x"})
        )
        snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_RESULT,
                payload={
                    "result": {
                        "call": tool_call,
                        "result": {"token": "secret", "value": "sunny"},
                    }
                },
                started=1.0,
                finished=2.0,
                elapsed=1.0,
            )
        )

        self.assertEqual(
            [event.event_type for event in snapshot.tool_events],
            ["tool_process", "tool_execute", "tool_result"],
        )
        self.assertEqual(snapshot.tool_events[0].tool_call_id, "tool-1")
        self.assertEqual(snapshot.tool_events[0].name, "search")
        self.assertIn(
            "<redacted>", snapshot.tool_events[0].payload_summary or ""
        )
        self.assertIn(
            "<redacted>", snapshot.tool_events[1].payload_summary or ""
        )
        self.assertEqual(
            [event.event_type for event in snapshot.events], ["start"]
        )

        tools_only = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=False),
        )
        tools_only.reduce_event(
            Event(type=EventType.START, payload={"ok": True})
        )
        tools_only_snapshot = tools_only.reduce_event(
            Event(type=EventType.TOOL_EXECUTE, payload={"call": tool_call})
        )

        self.assertEqual(
            [event.event_type for event in tools_only_snapshot.tool_events],
            ["tool_execute"],
        )
        self.assertEqual(tools_only_snapshot.events, ())

        events_only = CliStreamSnapshotReducer(
            _config(display_tools=False, display_events=True),
        )
        events_only.reduce_event(
            Event(type=EventType.TOOL_EXECUTE, payload={"call": tool_call})
        )
        events_only_snapshot = events_only.reduce_event(
            Event(type=EventType.START, payload={"ok": True})
        )

        self.assertEqual(events_only_snapshot.tool_events, ())
        self.assertEqual(
            [event.event_type for event in events_only_snapshot.events],
            ["start"],
        )

    def test_side_channel_tool_diagnostic_event_is_summarized(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=True),
        )

        snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={
                    "diagnostic": {
                        "call_id": "tool-1",
                        "canonical_name": "search",
                        "code": "tool.invalid",
                        "message": "bad",
                    }
                },
            )
        )

        self.assertEqual(
            [event.event_type for event in snapshot.tool_events],
            ["tool_diagnostic"],
        )
        self.assertEqual(snapshot.tool_events[0].tool_call_id, "tool-1")
        self.assertEqual(snapshot.tool_events[0].name, "search")

    def test_side_channel_tool_event_ids_are_normalized_for_dedupe(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=True),
            clock=FakeClock(1.0),
        )

        event_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": 42, "name": "lookup"}},
            )
        )
        canonical_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="42",
            )
        )

        self.assertEqual(event_snapshot.tool_events[0].tool_call_id, "42")
        self.assertEqual(canonical_snapshot.tool_events, ())

    def test_canonical_tool_projection_dedupes_side_channel_tool_event(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=True),
            clock=FakeClock(1.0, 2.0, 3.0),
        )
        tool_call = {"id": "tool-1", "name": "search", "arguments": {}}
        other_tool_call = {
            "id": "tool-2",
            "name": "lookup",
            "arguments": {},
        }

        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": other_tool_call},
            )
        )
        event_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": {"call": tool_call, "result": "legacy"}},
            )
        )
        start_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="tool-1",
            )
        )
        reducer.reduce_event(
            Event(
                type=EventType.TOOL_RESULT,
                payload={"result": {"call": tool_call, "result": "legacy"}},
            )
        )
        final_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                1,
                data={"result": "done"},
                tool_call_id="tool-1",
            )
        )

        self.assertEqual(
            [event.event_type for event in event_snapshot.tool_events],
            ["tool_execute", "tool_result"],
        )
        self.assertEqual(
            [event.tool_call_id for event in start_snapshot.tool_events],
            ["tool-2"],
        )
        self.assertEqual(
            [event.tool_call_id for event in final_snapshot.tool_events],
            ["tool-2"],
        )
        self.assertEqual(len(final_snapshot.completed_tools), 1)
        self.assertEqual(len(final_snapshot.tool_results), 1)

    def test_apply_projection_reports_side_channel_tool_event_removal(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=True),
            clock=FakeClock(1.0),
        )
        reducer.apply_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={
                    "call": {
                        "id": "tool-1",
                        "name": "search",
                    }
                },
            )
        )
        before = reducer.snapshot()

        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.TOOL_CALL_READY,
                0,
                tool_call_id="tool-1",
            )
        )
        after = reducer.snapshot()

        self.assertTrue(changed)
        self.assertEqual(
            [event.tool_call_id for event in before.tool_events],
            ["tool-1"],
        )
        self.assertEqual(after.tool_events, ())

    def test_apply_projection_reports_stats_projection_metadata(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(stats=True),
            clock=FakeClock(1.0),
        )

        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.TOOL_CALL_READY,
                0,
                metadata={"visible": "yes"},
                tool_call_id="tool-1",
            )
        )
        snapshot = reducer.snapshot()

        self.assertTrue(changed)
        self.assertEqual(len(snapshot.projection_metadata_summaries), 1)
        self.assertIn(
            "visible",
            snapshot.projection_metadata_summaries[0].metadata_summary or "",
        )

    def test_canonical_tool_projection_dedupes_batched_tool_events(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools=True, display_events=True),
            clock=FakeClock(1.0, 2.0),
        )
        tool_call = {"id": "tool-1", "name": "search", "arguments": {}}
        other_tool_call = {
            "id": "tool-2",
            "name": "lookup",
            "arguments": {},
        }

        batch_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_PROCESS,
                payload=[tool_call, other_tool_call],
            )
        )
        diagnostic_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_DIAGNOSTIC,
                payload={
                    "diagnostics": [
                        {
                            "call_id": "tool-1",
                            "canonical_name": "search",
                            "code": "tool.invalid",
                            "message": "bad",
                        },
                        {
                            "call_id": "tool-2",
                            "canonical_name": "lookup",
                            "code": "tool.invalid",
                            "message": "bad",
                        },
                    ]
                },
            )
        )
        start_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="tool-1",
            )
        )
        final_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                1,
                data={"result": "done"},
                tool_call_id="tool-1",
            )
        )

        self.assertEqual(
            [event.event_type for event in batch_snapshot.tool_events],
            ["tool_process"],
        )
        self.assertEqual(
            [event.tool_call_id for event in diagnostic_snapshot.tool_events],
            ["tool-1|tool-2", "tool-1|tool-2"],
        )
        self.assertEqual(start_snapshot.tool_events, ())
        self.assertEqual(final_snapshot.tool_events, ())

    def test_tool_event_dedupe_index_tracks_retained_duplicate_ids(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools_events=2),
            clock=FakeClock(1.0),
        )
        tool_call = {"id": "tool-1", "name": "search"}

        reducer.reduce_event(
            Event(type=EventType.TOOL_EXECUTE, payload={"call": tool_call})
        )
        reducer.reduce_event(
            Event(type=EventType.TOOL_RESULT, payload={"call": tool_call})
        )
        retained_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-2", "name": "lookup"}},
            )
        )
        canonical_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="tool-1",
            )
        )

        self.assertEqual(
            [event.tool_call_id for event in retained_snapshot.tool_events],
            ["tool-1", "tool-2"],
        )
        self.assertEqual(
            [event.tool_call_id for event in canonical_snapshot.tool_events],
            ["tool-2"],
        )

    def test_tool_event_dedupe_index_ignores_canonically_removed_middle(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools_events=3),
            clock=FakeClock(1.0, 2.0),
        )

        reducer.reduce_event(
            Event(
                type=EventType.TOOL_PROCESS,
                payload=[
                    {"id": "tool-1", "name": "search"},
                    {"id": "tool-2", "name": "lookup"},
                ],
            )
        )
        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-3", "name": "middle"}},
            )
        )
        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-4", "name": "tail"}},
            )
        )
        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="tool-3",
            )
        )
        retained_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-5", "name": "new"}},
            )
        )
        canonical_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                1,
                tool_call_id="tool-1",
            )
        )

        self.assertEqual(
            [event.tool_call_id for event in retained_snapshot.tool_events],
            ["tool-1|tool-2", "tool-4", "tool-5"],
        )
        self.assertEqual(
            [event.tool_call_id for event in canonical_snapshot.tool_events],
            ["tool-4", "tool-5"],
        )

    def test_tool_event_dedupe_index_ignores_canonically_removed_head(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools_events=2),
            clock=FakeClock(1.0, 2.0),
        )

        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-1", "name": "head"}},
            )
        )
        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-2", "name": "tail"}},
            )
        )
        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="tool-1",
            )
        )
        retained_snapshot = reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-3", "name": "new"}},
            )
        )
        canonical_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                1,
                tool_call_id="tool-2",
            )
        )

        self.assertEqual(
            [event.tool_call_id for event in retained_snapshot.tool_events],
            ["tool-2", "tool-3"],
        )
        self.assertEqual(
            [event.tool_call_id for event in canonical_snapshot.tool_events],
            ["tool-3"],
        )

    def test_tool_history_limit_zero_keeps_non_tool_events(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools_events=0),
            clock=FakeClock(1.0, 2.0, 3.0, 4.0),
        )
        tool_id = "tool-1"

        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id=tool_id,
            )
        )
        reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                1,
                data={"value": "done"},
                tool_call_id=tool_id,
            )
        )
        reducer.reduce_projection(
            _projection(
                StreamItemKind.FLOW_EVENT,
                2,
                data={"state": "done"},
            )
        )
        snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.STREAM_DIAGNOSTIC,
                3,
                text_delta="detail",
                data={"code": "x"},
            )
        )

        self.assertEqual(snapshot.completed_tools, ())
        self.assertEqual(snapshot.tool_results, ())
        self.assertEqual(
            [event.event_type for event in snapshot.events],
            ["flow.event", "stream.diagnostic"],
        )
        self.assertIn(
            '"data": {"code": "x"}',
            snapshot.events[1].payload_summary,
        )
        self.assertEqual(snapshot.build_stats.dropped_completed_tools, 1)
        self.assertEqual(snapshot.build_stats.dropped_tool_results, 0)

    def test_tool_history_limit_zero_drops_side_channel_tool_events(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(display_tools_events=0),
        )

        reducer.reduce_event(
            Event(
                type=EventType.TOOL_EXECUTE,
                payload={"call": {"id": "tool-1", "name": "search"}},
            )
        )
        snapshot = reducer.reduce_event(
            Event(type=EventType.START, payload={"ok": True})
        )

        self.assertEqual(snapshot.tool_events, ())
        self.assertEqual(
            [event.event_type for event in snapshot.events], ["start"]
        )
        self.assertEqual(snapshot.build_stats.dropped_tool_events, 1)

    def test_tool_error_cancel_and_hidden_tool_display_paths(self) -> None:
        visible = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0, 2.0, 3.0),
        )
        visible.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                0,
                data={"message": "bad", "tool_name": "error_lookup"},
                tool_call_id="error-tool",
            )
        )
        visible.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                1,
                tool_call_id="cancel-tool",
            )
        )
        visible_snapshot = visible.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                2,
                text_delta="orphan",
                tool_call_id="orphan-tool",
            )
        )

        self.assertEqual(
            visible_snapshot.completed_tools[0].name, "error_lookup"
        )
        self.assertEqual(
            [tool.status for tool in visible_snapshot.completed_tools],
            ["error", "cancelled"],
        )
        self.assertEqual(
            [result.status for result in visible_snapshot.tool_results],
            ["error", "error"],
        )
        self.assertIn(
            "tool_execution.cancelled",
            visible_snapshot.tool_results[1].result_summary,
        )

        hidden = CliStreamSnapshotReducer(
            _config(display_tools=False),
            clock=FakeClock(1.0, 2.0),
        )
        hidden.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="hidden-tool",
            )
        )
        hidden_snapshot = hidden.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                1,
                text_delta="hidden",
                tool_call_id="hidden-tool",
            )
        )

        self.assertEqual(hidden_snapshot.active_tools, ())
        self.assertEqual(hidden_snapshot.completed_tools, ())

    def test_usage_non_mapping_and_bool_counts_do_not_update_counts(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0, 2.0),
        )

        reducer.reduce_projection(
            _projection(StreamItemKind.USAGE_UPDATE, 0, usage=["raw"])
        )
        snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.USAGE_UPDATE,
                1,
                usage={"input_tokens": True, "output_tokens": 2},
            )
        )

        self.assertIsNone(snapshot.token_counts.input_tokens)
        self.assertEqual(snapshot.token_counts.output_tokens, 2)

    def test_apply_projection_defers_snapshot_materialization(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(*[float(index) for index in range(10_001)]),
        )

        for index in range(10_000):
            changed = reducer.apply_projection(
                _projection(
                    StreamItemKind.ANSWER_DELTA,
                    index,
                    text_delta="x",
                    metadata={"token_id": index} if index == 0 else None,
                )
            )
            self.assertTrue(changed)

        self.assertFalse(reducer.terminal_completed)
        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.STREAM_COMPLETED,
                10_000,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        )
        self.assertTrue(changed)
        self.assertTrue(reducer.terminal_completed)
        before_duplicate = reducer.snapshot()
        duplicate_terminal = reducer.apply_projection(
            _projection(
                StreamItemKind.STREAM_COMPLETED,
                10_000,
                usage={
                    "input_tokens": 99,
                    "output_tokens": 999,
                    "total_tokens": 1098,
                },
                metadata={"duplicate": True},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        )
        self.assertFalse(duplicate_terminal)
        after_duplicate = reducer.snapshot()

        self.assertEqual(
            after_duplicate.token_counts,
            before_duplicate.token_counts,
        )
        self.assertEqual(
            after_duplicate.display_tokens,
            before_duplicate.display_tokens,
        )
        self.assertEqual(
            after_duplicate.usage_summaries,
            before_duplicate.usage_summaries,
        )
        self.assertEqual(
            after_duplicate.projection_metadata_summaries,
            before_duplicate.projection_metadata_summaries,
        )

        snapshot = after_duplicate
        self.assertEqual(snapshot.answer_text, "x" * 10_000)
        self.assertEqual(snapshot.token_counts.display_tokens, 1)
        self.assertEqual(snapshot.build_stats.snapshots_built, 2)
        self.assertEqual(snapshot.build_stats.answer_chunks, 10_000)
        self.assertEqual(snapshot.build_stats.text_materializations, 6)

    def test_apply_projection_duplicate_terminal_is_noop(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(1.0),
        )
        reducer.apply_projection(
            _projection(
                StreamItemKind.STREAM_COMPLETED,
                0,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
        )

        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.STREAM_CANCELLED,
                1,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            )
        )

        self.assertFalse(changed)

    def test_apply_event_token_generated_is_noop(self) -> None:
        reducer = CliStreamSnapshotReducer(_config())

        changed = reducer.apply_event(Event(type=EventType.TOKEN_GENERATED))
        snapshot = reducer.snapshot()

        self.assertFalse(changed)
        self.assertEqual(snapshot.events, ())
        self.assertEqual(snapshot.tool_events, ())

    def test_apply_event_tool_progress_is_noop(self) -> None:
        reducer = CliStreamSnapshotReducer(_config(display_tools=True))

        changed = reducer.apply_event(
            Event(
                type=EventType.TOOL_PROGRESS,
                payload={"kind": "tool_execution.output"},
            )
        )
        snapshot = reducer.snapshot()

        self.assertFalse(changed)
        self.assertEqual(snapshot.tool_events, ())

    def test_apply_projection_ignores_invisible_tool_arguments(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(stats=False, display_tools=True),
            clock=FakeClock(1.0, 2.0),
        )

        argument_changed = reducer.apply_projection(
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                0,
                text_delta='{"x": 1}',
                tool_call_id="tool-1",
            )
        )
        start_changed = reducer.apply_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                1,
                tool_call_id="tool-1",
            )
        )

        self.assertFalse(argument_changed)
        self.assertTrue(start_changed)

    def test_model_continuation_tracks_active_tool_panel_status(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(stats=False, display_tools=True),
        )

        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                0,
                model_continuation_id="continuation-1",
            )
        )
        snapshot = reducer.snapshot()

        self.assertTrue(changed)
        self.assertEqual(len(snapshot.active_model_continuations), 1)
        self.assertEqual(
            snapshot.active_model_continuations[0].model_continuation_id,
            "continuation-1",
        )
        self.assertEqual(snapshot.tool_events, ())

        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED,
                1,
                model_continuation_id="continuation-1",
            )
        )
        snapshot = reducer.snapshot()

        self.assertTrue(changed)
        self.assertEqual(snapshot.active_model_continuations, ())

    def test_usage_reads_nested_cached_and_reasoning_token_details(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(_config(stats=True))

        reducer.apply_projection(
            _projection(
                StreamItemKind.USAGE_COMPLETED,
                0,
                usage={
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": 6},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": 7},
                    "total_tokens": 30,
                },
            )
        )
        snapshot = reducer.snapshot()

        self.assertEqual(snapshot.token_counts.input_tokens, 10)
        self.assertEqual(snapshot.token_counts.cached_input_tokens, 6)
        self.assertEqual(snapshot.token_counts.output_tokens, 20)
        self.assertEqual(snapshot.token_counts.reasoning_usage_tokens, 7)
        self.assertEqual(snapshot.token_counts.total_tokens, 30)

    def test_usage_reads_chat_compatible_nested_token_details(self) -> None:
        reducer = CliStreamSnapshotReducer(_config(stats=True))

        reducer.apply_projection(
            _projection(
                StreamItemKind.USAGE_COMPLETED,
                0,
                usage={
                    "prompt_tokens_details": {"cached_tokens": 2},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                },
            )
        )
        snapshot = reducer.snapshot()

        self.assertEqual(snapshot.token_counts.cached_input_tokens, 2)
        self.assertEqual(snapshot.token_counts.reasoning_usage_tokens, 3)

    def test_usage_ignores_non_integer_nested_token_details(self) -> None:
        reducer = CliStreamSnapshotReducer(_config(stats=True))

        reducer.apply_projection(
            _projection(
                StreamItemKind.USAGE_COMPLETED,
                0,
                usage={
                    "input_tokens": 10,
                    "input_tokens_details": {"cached_tokens": True},
                    "output_tokens": 20,
                    "output_tokens_details": {"reasoning_tokens": "7"},
                    "total_tokens": 30,
                },
            )
        )
        snapshot = reducer.snapshot()

        self.assertEqual(snapshot.token_counts.input_tokens, 10)
        self.assertIsNone(snapshot.token_counts.cached_input_tokens)
        self.assertEqual(snapshot.token_counts.output_tokens, 20)
        self.assertIsNone(snapshot.token_counts.reasoning_usage_tokens)
        self.assertEqual(snapshot.token_counts.total_tokens, 30)

    def test_apply_projection_change_matrix_for_hidden_surfaces(self) -> None:
        projections = {
            "answer": _projection(
                StreamItemKind.ANSWER_DELTA,
                0,
                text_delta="a",
            ),
            "reasoning": _projection(
                StreamItemKind.REASONING_DELTA,
                0,
                text_delta="r",
            ),
            "tool_argument": _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                0,
                text_delta='{"x": 1}',
                tool_call_id="tool-1",
            ),
            "tool_execution": _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                0,
                tool_call_id="tool-1",
            ),
            "model_continuation": _projection(
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                0,
            ),
            "event": _projection(StreamItemKind.FLOW_EVENT, 0),
            "usage": _projection(
                StreamItemKind.USAGE_UPDATE,
                0,
                usage={"input_tokens": 1},
            ),
            "terminal": _projection(
                StreamItemKind.STREAM_COMPLETED,
                0,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        }
        cases = (
            (
                "quiet",
                _config(
                    quiet=True,
                    stats=True,
                    display_tools=True,
                    display_events=True,
                ),
                {
                    "answer": True,
                    "reasoning": False,
                    "tool_argument": False,
                    "tool_execution": False,
                    "model_continuation": False,
                    "event": False,
                    "usage": False,
                    "terminal": True,
                },
            ),
            (
                "default",
                _config(
                    stats=False,
                    display_tools=False,
                    display_events=False,
                ),
                {
                    "answer": True,
                    "reasoning": False,
                    "tool_argument": False,
                    "tool_execution": False,
                    "model_continuation": False,
                    "event": False,
                    "usage": False,
                    "terminal": True,
                },
            ),
            (
                "stats",
                _config(
                    stats=True,
                    display_tools=False,
                    display_events=False,
                ),
                {
                    "answer": True,
                    "reasoning": True,
                    "tool_argument": True,
                    "tool_execution": False,
                    "model_continuation": False,
                    "event": False,
                    "usage": True,
                    "terminal": True,
                },
            ),
            (
                "tools",
                _config(
                    stats=False,
                    display_tools=True,
                    display_events=False,
                ),
                {
                    "answer": True,
                    "reasoning": False,
                    "tool_argument": False,
                    "tool_execution": True,
                    "model_continuation": True,
                    "event": False,
                    "usage": False,
                    "terminal": True,
                },
            ),
            (
                "events",
                _config(
                    stats=False,
                    display_tools=False,
                    display_events=True,
                ),
                {
                    "answer": True,
                    "reasoning": False,
                    "tool_argument": False,
                    "tool_execution": False,
                    "model_continuation": False,
                    "event": True,
                    "usage": False,
                    "terminal": True,
                },
            ),
            (
                "stderr",
                _config(
                    stats=True,
                    display_tools=True,
                    display_events=True,
                    interactive=False,
                ),
                {
                    "answer": True,
                    "reasoning": True,
                    "tool_argument": True,
                    "tool_execution": True,
                    "model_continuation": True,
                    "event": True,
                    "usage": True,
                    "terminal": True,
                },
            ),
        )

        for label, config, expected in cases:
            for projection_name, projection in projections.items():
                with self.subTest(label=label, projection=projection_name):
                    reducer = CliStreamSnapshotReducer(
                        config,
                        clock=FakeClock(1.0),
                    )
                    self.assertEqual(
                        reducer.apply_projection(projection),
                        expected[projection_name],
                    )

    def test_apply_projection_reports_display_token_before_metadata(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(stats=True, display_tokens=2),
            clock=FakeClock(1.0),
        )

        changed = reducer.apply_projection(
            _projection(
                StreamItemKind.REASONING_DELTA,
                0,
                text_delta="r",
                metadata={"token_id": 7},
            )
        )
        snapshot = reducer.snapshot()

        self.assertTrue(changed)
        self.assertEqual(snapshot.display_tokens[0].token_id, 7)

    def test_apply_event_bounds_side_channel_tool_indexes(self) -> None:
        reducer = CliStreamSnapshotReducer(_config(display_tools_events=3))

        for index in range(10):
            changed = reducer.apply_event(
                Event(
                    type=EventType.TOOL_EXECUTE,
                    payload={
                        "call": {
                            "id": f"tool-{index}",
                            "name": "search",
                        }
                    },
                )
            )
            self.assertTrue(changed)

        event_order = cast(
            Sized,
            getattr(reducer, "_side_channel_tool_event_order"),
        )
        event_counts = cast(
            dict[str, int],
            getattr(reducer, "_side_channel_tool_event_counts"),
        )
        snapshot = reducer.snapshot()

        self.assertLessEqual(len(event_order), 3)
        self.assertLessEqual(sum(event_counts.values()), 3)
        self.assertEqual(
            [event.tool_call_id for event in snapshot.tool_events],
            ["tool-7", "tool-8", "tool-9"],
        )

    def test_tool_argument_display_state_is_bounded_and_cleaned(self) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(*[float(index) for index in range(42)]),
        )
        tool_id = "tool-1"

        empty_changed = reducer.apply_projection(
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                0,
                text_delta="",
                tool_call_id=tool_id,
            )
        )
        self.assertFalse(empty_changed)

        for index in range(20):
            reducer.apply_projection(
                _projection(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    index + 1,
                    text_delta="x" * 50,
                    tool_call_id=tool_id,
                )
            )

        states = cast(
            dict[str, object],
            getattr(reducer, "_tool_argument_state"),
        )
        materialize = getattr(states[tool_id], "materialize")
        self.assertTrue(callable(materialize))
        retained = cast(str, materialize())

        self.assertLessEqual(len(retained), MAX_SUMMARY_CHARS)
        start_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                21,
                tool_call_id=tool_id,
            )
        )
        self.assertLessEqual(
            len(start_snapshot.active_tools[0].arguments_summary or ""),
            MAX_SUMMARY_CHARS + 2,
        )

        sensitive = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(*[float(index) for index in range(10)]),
        )
        sensitive.apply_projection(
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                0,
                text_delta="api_key",
                tool_call_id="sensitive-tool",
            )
        )
        sensitive.apply_projection(
            _projection(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                text_delta="=" + ("x" * MAX_SUMMARY_CHARS),
                tool_call_id="sensitive-tool",
            )
        )
        sensitive_states = cast(
            dict[str, object],
            getattr(sensitive, "_tool_argument_state"),
        )
        sensitive_materialize = getattr(
            sensitive_states["sensitive-tool"],
            "materialize",
        )
        self.assertTrue(callable(sensitive_materialize))
        self.assertEqual(cast(str, sensitive_materialize()), "<redacted>")

        final_snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                22,
                data={"result": "ok"},
                tool_call_id=tool_id,
            )
        )
        states = cast(
            dict[str, object],
            getattr(reducer, "_tool_argument_state"),
        )

        self.assertNotIn(tool_id, states)
        self.assertEqual(final_snapshot.tool_results[0].arguments_count, 1)

    def test_split_sensitive_tool_argument_marker_redacts_after_truncation(
        self,
    ) -> None:
        reducer = CliStreamSnapshotReducer(
            _config(),
            clock=FakeClock(*[float(index) for index in range(6)]),
        )
        tool_id = "sensitive-tool"

        for index, chunk in enumerate(
            (
                "api",
                "_",
                "key",
                "=" + ("x" * MAX_SUMMARY_CHARS),
            )
        ):
            reducer.apply_projection(
                _projection(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    index,
                    text_delta=chunk,
                    tool_call_id=tool_id,
                )
            )

        states = cast(
            dict[str, object],
            getattr(reducer, "_tool_argument_state"),
        )
        materialize = getattr(states[tool_id], "materialize")
        self.assertTrue(callable(materialize))
        self.assertEqual(cast(str, materialize()), "<redacted>")

        snapshot = reducer.reduce_projection(
            _projection(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                4,
                tool_call_id=tool_id,
            )
        )
        self.assertIn(
            "<redacted>",
            snapshot.active_tools[0].arguments_summary or "",
        )

    def test_reduce_projection_uses_canonical_helper_validation(self) -> None:
        reducer = CliStreamSnapshotReducer(_config())

        with self.assertRaises(AssertionError):
            reducer.reduce_projection("bad")  # type: ignore[arg-type]

    def test_late_content_fails_in_canonical_validation_path(self) -> None:
        items = (
            _projection(StreamItemKind.STREAM_STARTED, 0),
            _projection(
                StreamItemKind.STREAM_COMPLETED,
                1,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            _projection(
                StreamItemKind.ANSWER_DELTA,
                2,
                text_delta="late",
            ),
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "semantic stream item emitted after terminal outcome",
        ):
            validate_canonical_stream_items(
                tuple(
                    canonical_item_from_consumer_projection(item)
                    for item in items
                )
            )


class DisplayReducerAsyncTestCase(IsolatedAsyncioTestCase):
    async def test_iter_cli_canonical_stream_snapshots_reduces_full_trace(
        self,
    ) -> None:
        tool_id = "tool-1"
        source = AsyncCanonicalSource(
            _canonical(_projection(StreamItemKind.STREAM_STARTED, 0)),
            _canonical(
                _projection(
                    StreamItemKind.REASONING_DELTA,
                    1,
                    text_delta="plan",
                    metadata={
                        "token_id": 10,
                        "probability": 0.2,
                        "tokens": [
                            {"token": "draft", "probability": 0.1},
                            {"token": 7, "probability": 0.9},
                        ],
                    },
                )
            ),
            _canonical(_projection(StreamItemKind.REASONING_DONE, 2)),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    3,
                    text_delta='{"query":',
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    4,
                    text_delta=' "weather"}',
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_CALL_READY,
                    5,
                    data={"name": "search"},
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_CALL_DONE,
                    6,
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_EXECUTION_STARTED,
                    7,
                    tool_call_id=tool_id,
                    provider_family="local",
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    8,
                    text_delta="stdout",
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_EXECUTION_PROGRESS,
                    9,
                    data={"progress": 0.5},
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    10,
                    data={"answer": "sunny"},
                    tool_call_id=tool_id,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.ANSWER_DELTA,
                    11,
                    text_delta="Done",
                    metadata={"token_id": 11, "probability": 0.7},
                )
            ),
            _canonical(_projection(StreamItemKind.ANSWER_DONE, 12)),
            _canonical(
                _projection(
                    StreamItemKind.USAGE_COMPLETED,
                    13,
                    usage={
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "reasoning_tokens": 1,
                        "total_tokens": 4,
                    },
                    provider_family="local",
                    provider_event_type="usage.final",
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.STREAM_COMPLETED,
                    14,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                )
            ),
        )

        snapshots = [
            snapshot
            async for snapshot in iter_cli_canonical_stream_snapshots(
                source,
                _config(display_time_to_n_token=2),
                clock=FakeClock(
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                    111.0,
                    112.0,
                    113.0,
                    114.0,
                ),
            )
        ]
        final = snapshots[-1]

        self.assertEqual(source.iterations, 1)
        self.assertEqual(source.yields, 15)
        self.assertEqual(len(snapshots), 15)
        self.assertEqual(final.reasoning_text, "plan")
        self.assertEqual(final.answer_text, "Done")
        self.assertEqual(final.tool_call_request_text, '{"query": "weather"}')
        self.assertEqual(final.completed_tools[0].tool_call_id, tool_id)
        self.assertEqual(final.completed_tools[0].name, "search")
        self.assertEqual(final.completed_tools[0].elapsed_seconds, 3.0)
        self.assertEqual(final.tool_results[0].arguments_count, 1)
        self.assertIn(
            '"answer": "sunny"', final.tool_results[0].result_summary
        )
        self.assertEqual(final.token_counts.input_tokens, 2)
        self.assertEqual(final.token_counts.output_tokens, 1)
        self.assertEqual(final.token_counts.reasoning_usage_tokens, 1)
        self.assertEqual(final.token_counts.total_tokens, 4)
        self.assertEqual(final.timing.first_token_seconds, 1.0)
        self.assertEqual(final.timing.reasoning_seconds, 10.0)
        self.assertEqual(final.timing.time_to_n_token_seconds, 11.0)
        self.assertEqual(final.timing.elapsed_seconds, 14.0)
        self.assertTrue(final.terminal.completed)
        self.assertEqual(
            [token.token_id for token in final.display_tokens], [10, 11]
        )
        self.assertEqual(final.display_tokens[0].candidates[0].text, "draft")
        self.assertEqual(final.usage_summaries[0].kind, "usage.completed")
        self.assertEqual(final.usage_summaries[0].provider_family, "local")

        with self.assertRaises(FrozenInstanceError):
            final.terminal.completed = False

    async def test_iter_cli_canonical_stream_snapshots_keeps_output_on_error(
        self,
    ) -> None:
        snapshots: list[CliStreamSnapshot] = []
        source = AsyncCanonicalSource(
            _canonical(_projection(StreamItemKind.STREAM_STARTED, 0)),
            _canonical(
                _projection(
                    StreamItemKind.STREAM_COMPLETED,
                    1,
                    usage={"total_tokens": 0},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                )
            ),
            _canonical(
                _projection(
                    StreamItemKind.STREAM_CANCELLED,
                    2,
                    terminal_outcome=StreamTerminalOutcome.CANCELLED,
                )
            ),
        )

        with self.assertRaisesRegex(
            StreamValidationError,
            "duplicate stream terminal item",
        ):
            async for snapshot in iter_cli_canonical_stream_snapshots(
                source,
                _config(),
            ):
                snapshots.append(snapshot)

        self.assertEqual(len(snapshots), 2)
        self.assertTrue(snapshots[-1].terminal.completed)
        self.assertEqual(snapshots[-1].terminal.outcome, "completed")

    async def test_iter_cli_stream_snapshots_drains_one_ordered_source(
        self,
    ) -> None:
        yielded: list[str] = []

        async def source():
            yielded.append("start")
            yield _projection(StreamItemKind.STREAM_STARTED, 0)
            yielded.append("event")
            yield Event(type=EventType.START, payload={"value": "ok"})
            yielded.append("answer")
            yield _projection(
                StreamItemKind.ANSWER_DELTA,
                1,
                text_delta="a",
            )
            yielded.append("terminal")
            yield _projection(
                StreamItemKind.STREAM_COMPLETED,
                2,
                usage={},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            )
            yielded.append("drained")

        snapshots = [
            snapshot
            async for snapshot in iter_cli_stream_snapshots(
                source(),
                _config(),
                clock=FakeClock(1.0, 2.0, 3.0),
            )
        ]

        self.assertEqual(
            yielded, ["start", "event", "answer", "terminal", "drained"]
        )
        self.assertEqual(len(snapshots), 4)
        self.assertEqual(snapshots[1].events[0].event_type, "start")
        self.assertEqual(snapshots[2].answer_text, "a")
        self.assertTrue(snapshots[3].terminal.completed)

    async def test_iter_cli_canonical_stream_snapshots_consumes_once(
        self,
    ) -> None:
        source = AsyncCanonicalSource(
            _canonical(_projection(StreamItemKind.STREAM_STARTED, 0)),
            _canonical(
                _projection(
                    StreamItemKind.ANSWER_DELTA,
                    1,
                    text_delta="a",
                )
            ),
            _canonical(_projection(StreamItemKind.ANSWER_DONE, 2)),
            _canonical(
                _projection(
                    StreamItemKind.STREAM_COMPLETED,
                    3,
                    usage={},
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                )
            ),
        )

        snapshots = [
            snapshot
            async for snapshot in iter_cli_canonical_stream_snapshots(
                source,
                _config(),
                clock=FakeClock(1.0, 2.0, 3.0, 4.0),
            )
        ]

        self.assertEqual(source.iterations, 1)
        self.assertEqual(source.yields, 4)
        self.assertEqual(snapshots[1].answer_text, "a")
        self.assertTrue(snapshots[-1].terminal.completed)

    async def test_iter_cli_canonical_stream_snapshots_validation_errors(
        self,
    ) -> None:
        cases = (
            (
                (
                    _canonical(_projection(StreamItemKind.STREAM_STARTED, 0)),
                    _canonical(
                        _projection(
                            StreamItemKind.STREAM_COMPLETED,
                            1,
                            usage={},
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        )
                    ),
                    _canonical(
                        _projection(
                            StreamItemKind.STREAM_CANCELLED,
                            2,
                            terminal_outcome=(StreamTerminalOutcome.CANCELLED),
                        )
                    ),
                ),
                "duplicate stream terminal item",
            ),
            (
                (
                    _canonical(_projection(StreamItemKind.STREAM_STARTED, 0)),
                    _canonical(
                        _projection(
                            StreamItemKind.STREAM_COMPLETED,
                            1,
                            usage={},
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        )
                    ),
                    _canonical(
                        _projection(
                            StreamItemKind.ANSWER_DELTA,
                            2,
                            text_delta="late",
                        )
                    ),
                ),
                "semantic stream item emitted after terminal outcome",
            ),
            (
                (
                    _canonical(_projection(StreamItemKind.STREAM_STARTED, 0)),
                    _canonical(
                        _projection(
                            StreamItemKind.STREAM_COMPLETED,
                            2,
                            usage={},
                            terminal_outcome=(StreamTerminalOutcome.COMPLETED),
                        )
                    ),
                ),
                "lossless consumer stream sequence gap",
            ),
        )

        for items, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(StreamValidationError, message):
                    [
                        snapshot
                        async for snapshot in (
                            iter_cli_canonical_stream_snapshots(
                                AsyncCanonicalSource(*items),
                                _config(),
                            )
                        )
                    ]

    async def test_iter_cli_canonical_stream_snapshots_rejects_noncanonical(
        self,
    ) -> None:
        async def bad_items() -> AsyncIterator[object]:
            yield "bad"

        with self.assertRaises(AssertionError):
            [
                snapshot
                async for snapshot in iter_cli_canonical_stream_snapshots(
                    bad_items(),  # type: ignore[arg-type]
                    _config(),
                )
            ]
