from dataclasses import FrozenInstanceError
from importlib import import_module
from sys import modules
from unittest import TestCase

from avalan.cli.display import CliStreamDisplayConfig
from avalan.cli.display_snapshot import (
    CliAppendOnlyTextBuffer,
    CliBoundedHistoryBuffer,
    CliBoundedTextBuffer,
    CliStreamSnapshotBuilder,
    display_token_snapshot,
    retention_from_config,
)
from avalan.entities import (
    Token,
    TokenDetail,
    ToolCall,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallResult,
)
from avalan.event import Event, EventType
from avalan.model.stream import (
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamTerminalOutcome,
)


def _config(**overrides: object) -> CliStreamDisplayConfig:
    values = {
        "quiet": False,
        "stats": False,
        "display_tools": False,
        "display_events": False,
        "display_tools_events": 2,
        "record": False,
        "interactive": True,
        "refresh_per_second": 10,
        "answer_height": 12,
        "answer_height_expand": False,
        "display_tokens": 0,
        "display_pause": 0,
        "display_probabilities": False,
        "display_probabilities_maximum": 0.8,
        "display_probabilities_sample_minimum": 0.1,
        "display_time_to_n_token": None,
        "display_reasoning_time": True,
    }
    values.update(overrides)
    return CliStreamDisplayConfig(**values)


def _answer_projection(
    *,
    sequence: int,
    text: str = "tok",
    metadata: dict[str, object] | None = None,
) -> StreamConsumerProjection:
    return StreamConsumerProjection(
        stream_session_id="session",
        run_id="run",
        turn_id="turn",
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        correlation=StreamItemCorrelation(),
        text_delta=text,
        metadata={} if metadata is None else metadata,
    )


class DisplaySnapshotBufferTestCase(TestCase):
    def test_text_buffer_materializes_only_on_demand(self) -> None:
        buffer = CliAppendOnlyTextBuffer()

        buffer.append("")
        buffer.append("a")

        self.assertEqual(buffer.chunk_count, 1)
        self.assertEqual(buffer.character_count, 1)
        self.assertEqual(buffer.materialization_count, 0)
        self.assertEqual(buffer.materialize(), "a")
        self.assertEqual(buffer.materialization_count, 1)

    def test_bounded_text_buffer_retains_tail_and_redacts_dropped_sensitive(
        self,
    ) -> None:
        buffer = CliBoundedTextBuffer(limit=5)

        buffer.append("")
        buffer.append("abc")
        buffer.append("def")
        retained = buffer.materialize()

        self.assertEqual(buffer.chunk_count, 2)
        self.assertEqual(buffer.character_count, 6)
        self.assertEqual(retained, "bcdef")
        self.assertEqual(buffer.materialization_count, 1)

        zero = CliBoundedTextBuffer(limit=0)
        zero.append("abc")
        self.assertEqual(zero.materialize(), "")

        sensitive = CliBoundedTextBuffer(limit=8)
        sensitive.append("api_key")
        sensitive.append("=" + ("x" * 20))
        self.assertEqual(sensitive.materialize(), "<redacted>")

    def test_bounded_text_buffer_detects_split_sensitive_marker(self) -> None:
        buffer = CliBoundedTextBuffer(limit=8)

        buffer.append("api")
        buffer.append("_")
        buffer.append("key")
        buffer.append("=" + ("x" * 20))

        self.assertEqual(buffer.materialize(), "<redacted>")

    def test_history_buffer_bounds_and_counts(self) -> None:
        buffer = CliBoundedHistoryBuffer[int](2)

        buffer.append(1)
        buffer.append(2)
        buffer.append(3)

        self.assertEqual(buffer.limit, 2)
        self.assertEqual(buffer.item_count, 2)
        self.assertEqual(buffer.dropped_count, 1)
        self.assertEqual(buffer.snapshot(), (2, 3))
        self.assertEqual(buffer.materialization_count, 1)

        empty = CliBoundedHistoryBuffer[int](0)
        empty.append(1)
        self.assertEqual(empty.snapshot(), ())
        self.assertEqual(empty.dropped_count, 1)


class DisplaySnapshotBuilderTestCase(TestCase):
    def test_importing_snapshot_module_does_not_load_theme_modules(
        self,
    ) -> None:
        watched_modules = (
            "avalan.cli.display_snapshot",
            "avalan.cli.theme_registry",
            "avalan.cli.theme",
            "avalan.cli.theme.basic",
            "avalan.cli.theme.fancy",
        )
        saved_modules = {
            name: module
            for name in watched_modules
            if (module := modules.get(name)) is not None
        }

        try:
            for name in watched_modules:
                modules.pop(name, None)

            import_module("avalan.cli.display_snapshot")

            for name in watched_modules[1:]:
                self.assertNotIn(name, modules)
        finally:
            for name in watched_modules:
                modules.pop(name, None)
            modules.update(saved_modules)

    def test_builder_materializes_safe_full_snapshot(self) -> None:
        config = _config(
            stats=True,
            display_tools=True,
            display_events=True,
            display_tools_events=2,
            record=True,
            display_tokens=2,
            display_probabilities=True,
            display_time_to_n_token=32,
        )
        builder = CliStreamSnapshotBuilder(config)
        arguments = {"password": "secret", "query": "weather"}
        result_payload = {"token": "secret", "value": "sunny"}
        event_payload = {"api_key": "secret", "state": "done"}
        metadata = {"authorization": "secret", "token_id": 9}
        usage = {"input_tokens": 3, "output_tokens": 4, "secret": "hidden"}
        call = ToolCall(id="call-1", name="search", arguments=arguments)
        result = ToolCallResult(
            id="call-1",
            call=call,
            name="search",
            arguments=arguments,
            result=result_payload,
        )
        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            call_id="call-1",
            requested_name="search",
            canonical_name="search",
            code=ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
            stage=ToolCallDiagnosticStage.VALIDATE,
            message="bad\ninput",
            details={"api_key": "secret"},
            retryable=True,
            duration_ms=4,
        )
        projection = StreamConsumerProjection(
            stream_session_id="session",
            run_id="run",
            turn_id="turn",
            sequence=5,
            kind=StreamItemKind.USAGE_UPDATE,
            channel=StreamChannel.USAGE,
            correlation=StreamItemCorrelation(),
            usage=usage,
            data={"provider_payload": {"secret": "hidden"}},
            metadata=metadata,
            provider_family="openai",
            provider_event_type="usage.delta",
        )

        builder.append_answer_text("he")
        builder.append_answer_text("llo")
        builder.append_reasoning_text("why")
        builder.append_tool_call_request_text('{"query": "weather"}')
        builder.update_token_counts(input_tokens=3, output_tokens=4)
        builder.update_token_counts(reasoning_usage_tokens=1, total_tokens=10)
        builder.update_timing(
            started_at=1.0,
            updated_at=2.0,
            finished_at=3.0,
            elapsed_seconds=2.0,
            first_token_seconds=0.2,
            reasoning_seconds=0.5,
            time_to_n_token_seconds=0.7,
        )
        builder.set_terminal(
            completed=True,
            outcome=StreamTerminalOutcome.COMPLETED,
            sequence=9,
            error={"secret": "hidden"},
        )
        builder.add_display_token(
            TokenDetail(
                id=7,
                token="[answer]\x1b[31m",
                probability=0.25,
                step=3,
                probability_distribution="softmax",
                tokens=[Token(id=None, token="alt", probability=0.1)],
            ),
            sequence=1,
        )
        builder.add_display_token_from_projection(
            _answer_projection(
                sequence=2,
                text="next",
                metadata={"token_id": 8, "probability": 0.5},
            )
        )
        builder.add_display_token_from_projection(
            _answer_projection(sequence=10, metadata={})
        )
        builder.add_active_tool(
            tool_call_id="call-1",
            name="search",
            arguments=arguments,
            provider_name="provider",
            sequence=3,
            started_at=1.2,
        )
        builder.complete_tool(
            tool_call_id="call-1",
            elapsed_seconds=0.4,
            sequence=4,
        )
        builder.add_tool_result(result, sequence=5, elapsed_seconds=0.4)
        builder.add_tool_diagnostic(diagnostic, sequence=6)
        builder.add_event(
            Event(
                type=EventType.MODEL_EXECUTE_AFTER,
                payload=event_payload,
                started=1,
                finished=2,
                elapsed=1,
            ),
            sequence=7,
        )
        builder.add_event(
            Event(type=EventType.TOOL_RESULT, payload={"result": "hidden"}),
            sequence=8,
        )
        builder.add_event_summary(
            event_type="custom",
            payload={"secret": "hidden", "value": "ok"},
            observability={"password": "hidden"},
            sequence=9,
        )
        builder.add_projection_summary(projection)

        arguments["password"] = "changed"
        result_payload["token"] = "changed"
        event_payload["api_key"] = "changed"
        metadata["authorization"] = "changed"
        usage["secret"] = "changed"
        snapshot = builder.snapshot()

        self.assertEqual(snapshot.answer_text, "hello")
        self.assertEqual(snapshot.reasoning_text, "why")
        self.assertEqual(
            snapshot.tool_call_request_text, '{"query": "weather"}'
        )
        self.assertTrue(snapshot.display.show_probabilities)
        self.assertTrue(snapshot.display.record_enabled)
        self.assertTrue(snapshot.display.show_timing)
        self.assertEqual(snapshot.token_counts.input_tokens, 3)
        self.assertEqual(snapshot.token_counts.output_tokens, 4)
        self.assertEqual(snapshot.token_counts.reasoning_usage_tokens, 1)
        self.assertEqual(snapshot.token_counts.total_tokens, 10)
        self.assertEqual(snapshot.timing.elapsed_seconds, 2.0)
        self.assertEqual(snapshot.terminal.outcome, "completed")
        self.assertIn("<redacted>", snapshot.terminal.error_summary or "")
        self.assertEqual(snapshot.active_tools, ())
        self.assertEqual(snapshot.completed_tools[0].name, "search")
        self.assertIn(
            "<redacted>", snapshot.completed_tools[0].arguments_summary or ""
        )
        self.assertIn("<redacted>", snapshot.tool_results[0].result_summary)
        self.assertEqual(snapshot.tool_results[0].arguments_count, 2)
        self.assertEqual(
            snapshot.tool_diagnostics[0].code, "tool_call.arguments_invalid"
        )
        self.assertIn("\\n", snapshot.tool_diagnostics[0].message)
        self.assertIn(
            "<redacted>", snapshot.tool_diagnostics[0].details_summary or ""
        )
        self.assertEqual(
            [event.event_type for event in snapshot.events],
            ["model_execute_after", "custom"],
        )
        self.assertNotIn("changed", str(snapshot.events))
        self.assertEqual(snapshot.display_tokens[0].token_id, 7)
        self.assertEqual(snapshot.display_tokens[0].candidates[0].text, "alt")
        self.assertEqual(snapshot.display_tokens[1].token_id, 8)
        self.assertIn("<redacted>", snapshot.usage_summaries[0].usage_summary)
        self.assertIn(
            "<redacted>",
            snapshot.projection_metadata_summaries[0].metadata_summary or "",
        )
        self.assertEqual(snapshot.build_stats.snapshots_built, 1)
        self.assertEqual(snapshot.build_stats.answer_chunks, 2)
        self.assertEqual(snapshot.build_stats.text_materializations, 3)
        self.assertEqual(snapshot.build_stats.history_materializations, 8)
        self.assertEqual(snapshot.build_stats.retained_display_tokens, 2)

        with self.assertRaises(FrozenInstanceError):
            snapshot.answer_text = "mutated"

    def test_nested_snapshot_objects_are_immutable(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(
                stats=True,
                display_tools=True,
                display_events=True,
                display_tokens=1,
                display_probabilities=True,
            )
        )

        builder.add_active_tool(tool_call_id="call-1", name="search")
        builder.add_event_summary(event_type="custom", payload={"value": "ok"})
        builder.add_display_token(
            TokenDetail(
                id=7,
                token="answer",
                probability=0.4,
                tokens=[Token(id=8, token="candidate", probability=0.2)],
            )
        )
        snapshot = builder.snapshot()

        with self.assertRaises(FrozenInstanceError):
            snapshot.display.show_tools = False
        with self.assertRaises(FrozenInstanceError):
            snapshot.retention.internal_tool_history_limit = 0
        with self.assertRaises(FrozenInstanceError):
            snapshot.token_counts.total_tokens = 0
        with self.assertRaises(FrozenInstanceError):
            snapshot.timing.elapsed_seconds = 0.1
        with self.assertRaises(FrozenInstanceError):
            snapshot.terminal.completed = True
        with self.assertRaises(FrozenInstanceError):
            snapshot.active_tools[0].name = "mutated"
        with self.assertRaises(FrozenInstanceError):
            snapshot.events[0].payload_summary = "mutated"
        with self.assertRaises(FrozenInstanceError):
            snapshot.display_tokens[0].candidates[0].text = "mutated"
        with self.assertRaises(FrozenInstanceError):
            snapshot.build_stats.snapshots_built = 0

    def test_snapshots_are_isolated_after_builder_mutation(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(stats=True, display_tools=True, display_events=True)
        )
        arguments = {"query": "first", "api_key": "secret"}
        event_payload = {"state": "first"}

        builder.append_answer_text("a")
        builder.add_active_tool(
            tool_call_id="call-1",
            name="search",
            arguments=arguments,
            sequence=1,
        )
        builder.add_event_summary(
            event_type="first",
            payload=event_payload,
            sequence=2,
        )
        first = builder.snapshot()

        arguments["query"] = "mutated"
        event_payload["state"] = "mutated"
        builder.append_answer_text("b")
        builder.complete_tool(tool_call_id="call-1", sequence=3)
        builder.add_tool_result(
            {
                "id": "call-1",
                "name": "search",
                "arguments": arguments,
                "result": {"state": "second"},
            },
            sequence=4,
        )
        builder.add_event_summary(
            event_type="second",
            payload={"state": "second"},
            sequence=5,
        )
        second = builder.snapshot()

        self.assertEqual(first.answer_text, "a")
        self.assertEqual(second.answer_text, "ab")
        self.assertEqual(first.active_tools[0].tool_call_id, "call-1")
        self.assertEqual(first.completed_tools, ())
        self.assertIn("first", first.active_tools[0].arguments_summary or "")
        self.assertNotIn(
            "mutated", first.active_tools[0].arguments_summary or ""
        )
        self.assertEqual(
            [event.event_type for event in first.events], ["first"]
        )
        self.assertIn("first", first.events[0].payload_summary or "")
        self.assertNotIn("mutated", first.events[0].payload_summary or "")
        self.assertEqual(second.active_tools, ())
        self.assertEqual(second.completed_tools[0].tool_call_id, "call-1")
        self.assertEqual(second.tool_results[0].tool_call_id, "call-1")

    def test_active_tool_replacement_updates_completion_summary(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(display_tools=True, display_tools_events=2)
        )

        builder.add_active_tool(
            tool_call_id="call-1",
            name="first",
            arguments={"step": "old"},
            provider_name="provider-a",
            sequence=1,
            started_at=1.0,
        )
        builder.add_active_tool(
            tool_call_id="call-1",
            name="second",
            arguments={"step": "new"},
            provider_name="provider-b",
            sequence=2,
            started_at=2.0,
        )
        active_snapshot = builder.snapshot()
        builder.complete_tool(tool_call_id="call-1", elapsed_seconds=0.5)
        final_snapshot = builder.snapshot()

        self.assertEqual(len(active_snapshot.active_tools), 1)
        self.assertEqual(active_snapshot.active_tools[0].name, "second")
        self.assertIn(
            "new", active_snapshot.active_tools[0].arguments_summary or ""
        )
        self.assertEqual(
            active_snapshot.active_tools[0].provider_name, "provider-b"
        )
        self.assertEqual(active_snapshot.active_tools[0].sequence, 2)
        self.assertEqual(active_snapshot.active_tools[0].started_at, 2.0)
        self.assertEqual(final_snapshot.active_tools, ())
        self.assertEqual(final_snapshot.completed_tools[0].name, "second")
        self.assertIn(
            "new", final_snapshot.completed_tools[0].arguments_summary or ""
        )
        self.assertEqual(
            final_snapshot.completed_tools[0].provider_name, "provider-b"
        )
        self.assertEqual(final_snapshot.completed_tools[0].sequence, 2)
        self.assertEqual(final_snapshot.completed_tools[0].started_at, 2.0)
        self.assertEqual(
            final_snapshot.completed_tools[0].elapsed_seconds, 0.5
        )

    def test_active_tool_snapshots_are_isolated_after_updates(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(display_tools=True, display_tools_events=2)
        )

        builder.add_active_tool(
            tool_call_id="call-1",
            name="search",
            arguments={"query": "first"},
            provider_name="provider-a",
            sequence=1,
            started_at=1.0,
        )
        first = builder.snapshot()
        builder.update_active_tool(
            tool_call_id="call-1",
            name="lookup",
            arguments={"api_key": "secret", "query": "second"},
            provider_name="provider-b",
            sequence=2,
            updated_at=2.0,
        )
        second = builder.snapshot()

        self.assertEqual(first.active_tools[0].name, "search")
        self.assertEqual(first.active_tools[0].provider_name, "provider-a")
        self.assertEqual(first.active_tools[0].sequence, 1)
        self.assertIn("first", first.active_tools[0].arguments_summary or "")
        self.assertNotIn(
            "second", first.active_tools[0].arguments_summary or ""
        )
        self.assertEqual(second.active_tools[0].name, "lookup")
        self.assertEqual(second.active_tools[0].provider_name, "provider-b")
        self.assertEqual(second.active_tools[0].sequence, 2)
        self.assertEqual(second.active_tools[0].updated_at, 2.0)
        self.assertIn(
            "<redacted>", second.active_tools[0].arguments_summary or ""
        )

    def test_retention_disables_unrequested_tool_event_and_stats_history(
        self,
    ) -> None:
        builder = CliStreamSnapshotBuilder(_config())
        call = ToolCall(id="call-1", name="search")
        diagnostic = ToolCallDiagnostic(
            id="diag-1",
            code=ToolCallDiagnosticCode.CANCELLED,
            stage=ToolCallDiagnosticStage.GUARD,
            message="cancelled",
        )

        builder.add_active_tool(tool_call_id="call-1", name="search")
        builder.complete_tool(tool_call_id="call-1")
        builder.add_tool_result(
            ToolCallResult(
                id="call-1",
                call=call,
                name="search",
                result={"value": "ok"},
            )
        )
        builder.add_tool_diagnostic(diagnostic)
        builder.add_event_summary(event_type="custom", payload={"value": "ok"})
        builder.add_usage_summary({"input_tokens": 1})
        builder.add_projection_summary(
            _answer_projection(sequence=1, metadata={"token_id": 1})
        )
        builder.add_display_token(Token(id=1, token="x", probability=True))
        snapshot = builder.snapshot()

        self.assertFalse(snapshot.display.show_tools)
        self.assertEqual(snapshot.active_tools, ())
        self.assertEqual(snapshot.completed_tools, ())
        self.assertEqual(snapshot.tool_results, ())
        self.assertEqual(snapshot.tool_diagnostics, ())
        self.assertEqual(snapshot.events, ())
        self.assertEqual(snapshot.usage_summaries, ())
        self.assertEqual(snapshot.projection_metadata_summaries, ())
        self.assertEqual(snapshot.display_tokens, ())

    def test_custom_zero_history_limits_drop_enabled_histories(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(
                stats=True,
                display_tools=True,
                display_events=True,
                display_tokens=1,
            ),
            event_history_limit=0,
            projection_summary_limit=0,
        )
        projection = StreamConsumerProjection(
            stream_session_id="session",
            run_id="run",
            turn_id="turn",
            sequence=1,
            kind=StreamItemKind.USAGE_UPDATE,
            channel=StreamChannel.USAGE,
            correlation=StreamItemCorrelation(),
            usage={"input_tokens": 1},
            data={"provider_payload": {"token": "secret"}},
            metadata={"authorization": "secret"},
        )

        builder.complete_tool(tool_call_id="call-1", name="search")
        builder.add_event(Event(type=EventType.START, payload={"ok": True}))
        builder.add_projection_summary(projection)
        builder.add_display_token(Token(id=1, token="x", probability=0.2))
        snapshot = builder.snapshot()

        self.assertTrue(snapshot.display.show_events)
        self.assertTrue(snapshot.display.show_stats)
        self.assertEqual(snapshot.retention.event_history_limit, 0)
        self.assertEqual(snapshot.retention.usage_summary_history_limit, 0)
        self.assertEqual(snapshot.completed_tools[0].tool_call_id, "call-1")
        self.assertEqual(snapshot.events, ())
        self.assertEqual(snapshot.usage_summaries, ())
        self.assertEqual(snapshot.projection_metadata_summaries, ())
        self.assertEqual(snapshot.display_tokens[0].token_id, 1)

    def test_display_tools_events_zero_keeps_only_active_tools(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(display_tools=True, display_tools_events=0)
        )

        builder.add_active_tool(tool_call_id="call-1", name="search")
        active_snapshot = builder.snapshot()
        builder.complete_tool(tool_call_id="call-1")
        builder.add_tool_result({"result": "ok"})
        final_snapshot = builder.snapshot()

        self.assertEqual(len(active_snapshot.active_tools), 1)
        self.assertEqual(active_snapshot.completed_tools, ())
        self.assertEqual(final_snapshot.active_tools, ())
        self.assertEqual(final_snapshot.completed_tools, ())
        self.assertEqual(final_snapshot.tool_results, ())
        self.assertEqual(final_snapshot.build_stats.dropped_completed_tools, 1)

    def test_tool_call_request_text_is_sanitized_in_snapshots(self) -> None:
        builder = CliStreamSnapshotBuilder(_config())

        builder.append_tool_call_request_text(
            '{"query": "weather", "api_key": "secret"}'
        )
        snapshot = builder.snapshot()

        self.assertIn("<redacted>", snapshot.tool_call_request_text)
        self.assertIn("weather", snapshot.tool_call_request_text)
        self.assertNotIn("secret", snapshot.tool_call_request_text)

    def test_timing_updates_preserve_omitted_fields(self) -> None:
        builder = CliStreamSnapshotBuilder(_config())

        builder.update_timing(
            started_at=1.0,
            first_token_seconds=0.2,
            reasoning_seconds=0.4,
        )
        builder.update_timing(updated_at=2.0)
        snapshot = builder.snapshot()

        self.assertEqual(snapshot.timing.started_at, 1.0)
        self.assertEqual(snapshot.timing.updated_at, 2.0)
        self.assertEqual(snapshot.timing.first_token_seconds, 0.2)
        self.assertEqual(snapshot.timing.reasoning_seconds, 0.4)

    def test_projection_display_token_metadata_is_sanitized(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(
                stats=True,
                display_tokens=1,
                display_probabilities=True,
            )
        )

        builder.add_display_token_from_projection(
            _answer_projection(
                sequence=3,
                text="\x1b[31manswer",
                metadata={
                    "token_id": 9,
                    "probability": 0.4,
                    "tokens": [
                        {
                            "token_id": 10,
                            "token": "candidate\x1b[0m",
                            "probability": 0.2,
                        },
                        {"token": 7, "probability": 0.8},
                        "invalid",
                    ],
                },
            )
        )
        snapshot = builder.snapshot()

        self.assertEqual(snapshot.display_tokens[0].sequence, 3)
        self.assertEqual(snapshot.display_tokens[0].text, "answer")
        self.assertEqual(snapshot.display_tokens[0].probability, 0.4)
        self.assertEqual(len(snapshot.display_tokens[0].candidates), 1)
        self.assertEqual(
            snapshot.display_tokens[0].candidates[0].text, "candidate"
        )
        self.assertEqual(
            snapshot.display_tokens[0].candidates[0].probability, 0.2
        )

    def test_repeated_appends_materialize_once_per_snapshot_and_bound_history(
        self,
    ) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(
                stats=True,
                display_tools=True,
                display_events=True,
                display_tools_events=2,
                display_tokens=2,
            )
        )

        for index in range(50):
            builder.append_answer_text("x")
            builder.complete_tool(
                tool_call_id=f"tool-{index}",
                name=f"tool-{index}",
                sequence=index,
            )
            builder.add_event_summary(
                event_type=f"event-{index}",
                payload={"index": index},
                sequence=index,
            )
            builder.add_display_token(
                Token(id=index, token=f"token-{index}", probability=0.1),
                sequence=index,
            )

        snapshot = builder.snapshot()

        self.assertEqual(snapshot.answer_text, "x" * 50)
        self.assertEqual(snapshot.build_stats.answer_chunks, 50)
        self.assertEqual(snapshot.build_stats.answer_characters, 50)
        self.assertEqual(snapshot.build_stats.text_materializations, 3)
        self.assertEqual(snapshot.build_stats.history_materializations, 8)
        self.assertEqual(snapshot.build_stats.retained_completed_tools, 2)
        self.assertEqual(snapshot.build_stats.retained_events, 4)
        self.assertEqual(snapshot.build_stats.retained_display_tokens, 2)
        self.assertEqual(snapshot.build_stats.dropped_completed_tools, 48)
        self.assertEqual(snapshot.build_stats.dropped_events, 46)
        self.assertEqual(snapshot.build_stats.dropped_display_tokens, 48)
        self.assertEqual(
            [tool.tool_call_id for tool in snapshot.completed_tools],
            ["tool-48", "tool-49"],
        )
        self.assertEqual(
            [event.event_type for event in snapshot.events],
            ["event-46", "event-47", "event-48", "event-49"],
        )

    def test_large_answer_stream_materializes_only_on_snapshot(self) -> None:
        builder = CliStreamSnapshotBuilder(_config())

        for _ in range(10_000):
            builder.append_answer_text("x")

        snapshot = builder.snapshot()

        self.assertEqual(snapshot.answer_text, "x" * 10_000)
        self.assertEqual(snapshot.build_stats.snapshots_built, 1)
        self.assertEqual(snapshot.build_stats.answer_chunks, 10_000)
        self.assertEqual(snapshot.build_stats.answer_characters, 10_000)
        self.assertEqual(snapshot.build_stats.text_materializations, 3)
        self.assertEqual(snapshot.build_stats.history_materializations, 8)

    def test_finite_and_unbounded_tool_history_are_bounded(self) -> None:
        finite = CliStreamSnapshotBuilder(
            _config(display_tools=True, display_tools_events=1)
        )
        call = ToolCall(id="call-1", name="search", arguments={"x": 1})
        error = ToolCallError(
            id="call-1",
            call=call,
            name="search",
            error={"type": "bad"},
            message="failed",
        )
        no_call = {"result": "orphan"}
        canonical_only = {
            "canonical_name": "canonical",
            "arguments": "x",
            "result": "ok",
        }
        requested_only = {"requested_name": "requested", "result": "ok"}

        finite.complete_tool(tool_call_id="a", name="first")
        finite.complete_tool(tool_call_id="b", name="second")
        finite.add_tool_result(error)
        finite.add_tool_result(no_call)
        finite.add_tool_result(canonical_only)
        finite.add_tool_result(requested_only)
        finite.add_tool_diagnostic(
            ToolCallDiagnostic(
                id="diag-1",
                code=ToolCallDiagnosticCode.CANCELLED,
                stage=ToolCallDiagnosticStage.GUARD,
                message="cancelled",
            )
        )
        finite.add_tool_diagnostic(
            ToolCallDiagnostic(
                id="diag-2",
                call_id="call-1",
                code=ToolCallDiagnosticCode.TIMEOUT,
                stage=ToolCallDiagnosticStage.DISPATCH,
                message="timeout",
            )
        )
        finite_snapshot = finite.snapshot()

        self.assertEqual(finite_snapshot.completed_tools[0].name, "second")
        self.assertEqual(finite_snapshot.tool_results[0].name, "requested")
        self.assertEqual(
            finite_snapshot.tool_diagnostics[0].diagnostic_id, "diag-2"
        )
        self.assertEqual(
            finite_snapshot.build_stats.dropped_completed_tools, 1
        )
        self.assertEqual(finite_snapshot.build_stats.dropped_tool_results, 3)
        self.assertEqual(
            finite_snapshot.build_stats.dropped_tool_diagnostics, 1
        )

        unbounded = CliStreamSnapshotBuilder(
            _config(display_tools=True, display_tools_events=None),
            unlimited_tool_history_limit=2,
        )
        for index in range(3):
            unbounded.complete_tool(
                tool_call_id=f"tool-{index}", name=f"tool-{index}"
            )
        unbounded_snapshot = unbounded.snapshot()

        self.assertIsNone(
            unbounded_snapshot.retention.visible_tool_history_limit
        )
        self.assertEqual(
            unbounded_snapshot.retention.internal_tool_history_limit, 2
        )
        self.assertEqual(
            [tool.tool_call_id for tool in unbounded_snapshot.completed_tools],
            ["tool-1", "tool-2"],
        )

        hard_capped = CliStreamSnapshotBuilder(
            _config(display_tools=True, display_tools_events=None),
        )
        for index in range(260):
            hard_capped.complete_tool(
                tool_call_id=f"tool-{index}",
                name=f"tool-{index}",
            )
        hard_capped_snapshot = hard_capped.snapshot()

        self.assertEqual(
            hard_capped_snapshot.retention.internal_tool_history_limit,
            256,
        )
        self.assertEqual(
            hard_capped_snapshot.build_stats.retained_completed_tools,
            256,
        )
        self.assertEqual(
            hard_capped_snapshot.build_stats.dropped_completed_tools,
            4,
        )
        self.assertEqual(
            hard_capped_snapshot.completed_tools[0].tool_call_id,
            "tool-4",
        )

    def test_events_and_projection_summaries_are_bounded(self) -> None:
        builder = CliStreamSnapshotBuilder(
            _config(stats=True, display_events=True),
            projection_summary_limit=1,
        )
        projection = StreamConsumerProjection(
            stream_session_id="session",
            run_id="run",
            turn_id="turn",
            sequence=2,
            kind=StreamItemKind.USAGE_UPDATE,
            channel=StreamChannel.USAGE,
            correlation=StreamItemCorrelation(),
            usage={"input_tokens": 1},
            metadata={},
        )

        for index in range(5):
            builder.add_event_summary(
                event_type=f"event-{index}",
                payload=None,
                observability=None,
            )
        builder.add_event_summary(
            event_type="tool_result", payload={"value": "x"}
        )
        builder.add_event(Event(type=EventType.START), sequence=7)
        builder.add_projection_summary(_answer_projection(sequence=1))
        builder.add_projection_summary(projection)
        snapshot = builder.snapshot()

        self.assertEqual(
            [event.event_type for event in snapshot.events],
            ["event-2", "event-3", "event-4", "start"],
        )
        self.assertEqual(snapshot.build_stats.dropped_events, 2)
        self.assertEqual(len(snapshot.usage_summaries), 1)
        self.assertEqual(snapshot.projection_metadata_summaries, ())

    def test_config_retention_and_canonical_validation(self) -> None:
        quiet_retention = retention_from_config(_config(quiet=True))
        none_probability = display_token_snapshot(
            Token(id="token-id", token="x", probability=True)
        )

        self.assertFalse(quiet_retention.show_tools)
        self.assertEqual(quiet_retention.internal_tool_history_limit, 0)
        self.assertEqual(none_probability.probability, None)

        with self.assertRaises(AssertionError):
            StreamConsumerProjection(
                stream_session_id="session",
                run_id="run",
                turn_id="turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.USAGE,
                correlation=StreamItemCorrelation(),
                text_delta="bad",
            )
