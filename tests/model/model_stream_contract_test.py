from asyncio import run
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, cast
from unittest import TestCase
from unittest.mock import patch

from avalan.entities import (
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
)
from avalan.model.stream import (
    CanonicalStreamAccumulator,
    CanonicalStreamItem,
    StreamBackpressurePolicy,
    StreamCancellationDrainPolicy,
    StreamCancellationPropagation,
    StreamCancellationPropagationTarget,
    StreamChannel,
    StreamGoldenTrace,
    StreamItemCorrelation,
    StreamItemKind,
    StreamLegacySurface,
    StreamLegacySurfaceClassification,
    StreamLegacySurfaceInventoryEntry,
    StreamPerformanceBudget,
    StreamRetentionPolicy,
    StreamRuntimeContract,
    StreamSessionLifecycle,
    StreamTerminalOutcome,
    StreamToolLifecycleContract,
    StreamToolObservation,
    StreamValidationError,
    StreamVisibility,
    TextGenerationSingleStream,
    TextGenerationStream,
    accumulate_canonical_stream_items,
    assemble_tool_observations,
    canonical_item_from_token,
    classify_legacy_stream_surface,
    is_stream_terminal_kind,
    is_tool_execution_terminal_kind,
    legacy_stream_surface_inventory,
    stream_channel_for_kind,
    stream_terminal_outcome_for_kind,
    validate_canonical_stream_items,
    validate_stream_runtime_contract,
    validate_tool_lifecycle_items,
)


def _item(
    kind: StreamItemKind,
    sequence: int,
    *,
    stream_session_id: str = "stream-1",
    run_id: str = "run-1",
    turn_id: str = "turn-1",
    channel: StreamChannel | None = None,
    correlation: StreamItemCorrelation | None = None,
    text_delta: str | None = None,
    data: object | None = None,
    usage: object | None = None,
    terminal_outcome: StreamTerminalOutcome | None = None,
    visibility: StreamVisibility = StreamVisibility.PUBLIC,
    metadata: dict[str, object] | None = None,
    provider_payload: object | None = None,
    provider_family: str | None = None,
    provider_event_type: str | None = None,
    timestamp: datetime | None = None,
) -> CanonicalStreamItem:
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=kind,
        channel=channel or stream_channel_for_kind(kind),
        correlation=correlation or StreamItemCorrelation(),
        text_delta=text_delta,
        data=data,  # type: ignore[arg-type]
        usage=usage,  # type: ignore[arg-type]
        terminal_outcome=terminal_outcome,
        visibility=visibility,
        metadata={} if metadata is None else metadata,  # type: ignore[arg-type]
        provider_payload=provider_payload,  # type: ignore[arg-type]
        provider_family=provider_family,
        provider_event_type=provider_event_type,
        timestamp=timestamp,
    )


def _stream_completed(sequence: int) -> CanonicalStreamItem:
    return _item(
        StreamItemKind.STREAM_COMPLETED,
        sequence,
        terminal_outcome=StreamTerminalOutcome.COMPLETED,
    )


def _stream_errored(sequence: int) -> CanonicalStreamItem:
    return _item(
        StreamItemKind.STREAM_ERRORED,
        sequence,
        terminal_outcome=StreamTerminalOutcome.ERRORED,
    )


def _tool_item(
    kind: StreamItemKind,
    sequence: int,
    *,
    tool_call_id: str = "tool-1",
    text_delta: str | None = None,
    data: object | None = None,
) -> CanonicalStreamItem:
    return _item(
        kind,
        sequence,
        correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        text_delta=text_delta,
        data=data,
    )


class _StreamProbe(TextGenerationStream):
    def __call__(
        self, *args: object, **kwargs: object
    ) -> AsyncIterator[Token | TokenDetail | str]:
        return TextGenerationStream.__call__(self, *args, **kwargs)

    async def __anext__(self) -> Token | TokenDetail | str:
        return await TextGenerationStream.__anext__(self)


async def _single_token_generator() -> (
    AsyncIterator[Token | TokenDetail | str]
):
    yield "token"


class StreamContractTestCase(TestCase):
    def test_taxonomy_maps_every_kind_to_channel_and_terminal_outcome(
        self,
    ) -> None:
        self.assertEqual(
            {channel.value for channel in StreamChannel},
            {
                "answer",
                "reasoning",
                "tool_call",
                "tool_execution",
                "flow",
                "usage",
                "control",
            },
        )
        self.assertEqual(
            {outcome.value for outcome in StreamTerminalOutcome},
            {"completed", "errored", "cancelled"},
        )
        self.assertEqual(
            {visibility.value for visibility in StreamVisibility},
            {"public", "private", "redacted", "diagnostic"},
        )

        for kind in StreamItemKind:
            self.assertIsInstance(stream_channel_for_kind(kind), StreamChannel)

        self.assertIs(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_COMPLETED),
            StreamTerminalOutcome.COMPLETED,
        )
        self.assertIs(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_ERRORED),
            StreamTerminalOutcome.ERRORED,
        )
        self.assertIs(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_CANCELLED),
            StreamTerminalOutcome.CANCELLED,
        )
        self.assertIsNone(
            stream_terminal_outcome_for_kind(StreamItemKind.STREAM_CLOSED)
        )
        self.assertTrue(is_stream_terminal_kind(StreamItemKind.STREAM_ERRORED))
        self.assertFalse(
            is_stream_terminal_kind(StreamItemKind.TOOL_EXECUTION_ERROR)
        )
        self.assertTrue(
            is_tool_execution_terminal_kind(
                StreamItemKind.TOOL_EXECUTION_CANCELLED
            )
        )
        self.assertFalse(
            is_tool_execution_terminal_kind(StreamItemKind.STREAM_CANCELLED)
        )

        with self.assertRaises(AssertionError):
            stream_channel_for_kind("answer.delta")  # type: ignore[arg-type]

    def test_valid_trace_fixture_serializes_contract_fields(self) -> None:
        timestamp = datetime(2026, 1, 2, 3, 4, 5)
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        full_correlation = StreamItemCorrelation(
            provider_request_id="request-1",
            model_continuation_id="continuation-1",
            tool_call_id="tool-1",
            flow_run_id="flow-1",
            node_id="node-1",
            parent_sequence=1,
            protocol_item_id="protocol-1",
            task_id="task-1",
            artifact_id="artifact-1",
        )
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                1,
                correlation=StreamItemCorrelation(
                    model_continuation_id="continuation-1",
                    parent_sequence=0,
                ),
            ),
            _item(
                StreamItemKind.REASONING_DELTA,
                2,
                text_delta="plan",
                visibility=StreamVisibility.PRIVATE,
            ),
            _item(StreamItemKind.REASONING_DONE, 3),
            _item(
                StreamItemKind.ANSWER_DELTA,
                4,
                correlation=full_correlation,
                text_delta="answer",
                data={"chunk": 1},
                metadata={"provider": "fixture"},
                provider_payload={"native": True},
                provider_family="openai",
                provider_event_type="response.output_text.delta",
                timestamp=timestamp,
            ),
            _item(StreamItemKind.ANSWER_DONE, 5),
            _item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                6,
                correlation=tool,
                text_delta='{"expression"',
            ),
            _item(
                StreamItemKind.TOOL_CALL_READY,
                7,
                correlation=tool,
                data={"name": "math.calculator"},
            ),
            _item(StreamItemKind.TOOL_CALL_DONE, 8, correlation=tool),
            _item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                9,
                correlation=tool,
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                10,
                correlation=tool,
                text_delta="4",
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                11,
                correlation=tool,
                data={"step": 1},
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                12,
                correlation=tool,
                data={"result": 4},
            ),
            _item(StreamItemKind.MODEL_CONTINUATION_COMPLETED, 13),
            _item(
                StreamItemKind.FLOW_EVENT,
                14,
                correlation=StreamItemCorrelation(
                    flow_run_id="flow-1", node_id="node-1"
                ),
                data={"state": "completed"},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                15,
                text_delta="redacted detail",
                data={"code": "stream.note"},
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.USAGE_UPDATE,
                16,
                usage={"input_tokens": 2},
            ),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                17,
                usage={"input_tokens": 2, "output_tokens": 1},
            ),
            _stream_completed(18),
            _item(StreamItemKind.STREAM_CLOSED, 19),
        )

        validated = validate_canonical_stream_items(items)
        self.assertEqual(validated, items)
        self.assertTrue(items[18].is_stream_terminal)
        self.assertTrue(items[12].is_tool_execution_terminal)

        trace = StreamGoldenTrace(
            name="contract-fixture",
            description="Canonical stream contract fixture",
            items=items,
        )
        fixture = trace.to_fixture()

        self.assertEqual(fixture["format_version"], 1)
        self.assertEqual(fixture["name"], "contract-fixture")
        self.assertEqual(fixture["description"], trace.description)
        fixture_items = fixture["items"]
        self.assertIsInstance(fixture_items, list)
        answer = fixture_items[4]  # type: ignore[index]
        self.assertEqual(answer["kind"], "answer.delta")
        self.assertEqual(answer["text_delta"], "answer")
        self.assertEqual(answer["timestamp"], "2026-01-02T03:04:05")
        self.assertEqual(
            answer["correlation"],
            full_correlation.to_trace_dict(),
        )

    def test_completed_stream_allows_atomic_final_usage(self) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                1,
                usage={"input_tokens": 1, "output_tokens": 1},
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )

        self.assertEqual(validate_canonical_stream_items(items), items)

    def test_error_and_cancel_are_terminal_without_final_usage(self) -> None:
        for kind, outcome in (
            (StreamItemKind.STREAM_ERRORED, StreamTerminalOutcome.ERRORED),
            (
                StreamItemKind.STREAM_CANCELLED,
                StreamTerminalOutcome.CANCELLED,
            ),
        ):
            with self.subTest(kind=kind):
                items = (
                    _item(StreamItemKind.STREAM_STARTED, 0),
                    _item(kind, 1, terminal_outcome=outcome),
                )
                self.assertEqual(validate_canonical_stream_items(items), items)

    def test_session_lifecycle_defaults_and_validation(self) -> None:
        lifecycle = StreamSessionLifecycle()

        self.assertTrue(lifecycle.single_use)
        self.assertTrue(lifecycle.cancellable)
        self.assertTrue(lifecycle.closeable)
        self.assertTrue(lifecycle.cleanup_owned)
        self.assertFalse(StreamSessionLifecycle(single_use=False).single_use)
        with self.assertRaises(AssertionError):
            StreamSessionLifecycle(cancellable="yes")  # type: ignore[arg-type]

    def test_runtime_contract_defaults_and_validation(self) -> None:
        contract = StreamRuntimeContract()

        self.assertIs(
            contract.backpressure_policy, StreamBackpressurePolicy.BLOCK
        )
        self.assertIs(
            contract.cancellation_drain_policy,
            StreamCancellationDrainPolicy.DRAIN_BUFFERED,
        )
        self.assertTrue(contract.close_after_terminal)
        self.assertTrue(contract.cancellation_as_terminal)
        self.assertTrue(contract.buffered_items_may_drain_after_cancellation)
        self.assertIs(validate_stream_runtime_contract(contract), contract)
        self.assertEqual(
            contract.retention_policy.accumulator_item_limit, 4096
        )
        self.assertEqual(contract.performance_budget.max_queue_depth, 64)
        self.assertEqual(
            contract.cancellation_propagation.targets,
            tuple(StreamCancellationPropagationTarget),
        )

        discard = StreamRuntimeContract(
            cancellation_drain_policy=(
                StreamCancellationDrainPolicy.DISCARD_BUFFERED
            ),
            retention_policy=StreamRetentionPolicy(
                replay_history_item_limit=0,
                ui_buffer_item_limit=0,
                metrics_history_item_limit=0,
                event_history_item_limit=0,
                mcp_resource_item_limit=0,
                a2a_task_record_item_limit=0,
                flow_history_item_limit=0,
            ),
            performance_budget=StreamPerformanceBudget(
                time_to_first_item_ms=1,
                cancellation_latency_ms=1,
                close_latency_ms=1,
                max_queue_depth=1,
                max_memory_bytes=1,
                per_item_overhead_us=1,
            ),
            cancellation_propagation=StreamCancellationPropagation(
                targets=(
                    StreamCancellationPropagationTarget.CONSUMER,
                    StreamCancellationPropagationTarget.STREAM_SESSION,
                )
            ),
        )
        self.assertFalse(discard.buffered_items_may_drain_after_cancellation)

        invalid_values = (
            lambda: StreamRetentionPolicy(accumulator_item_limit=0),
            lambda: StreamRetentionPolicy(accumulator_item_limit=True),  # type: ignore[arg-type]
            lambda: StreamRetentionPolicy(replay_history_item_limit=-1),
            lambda: StreamRetentionPolicy(active_session_lossless=False),
            lambda: StreamPerformanceBudget(max_queue_depth=0),
            lambda: StreamRuntimeContract(backpressure_policy="block"),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(
                cancellation_drain_policy="drain_buffered",  # type: ignore[arg-type]
            ),
            lambda: StreamRuntimeContract(retention_policy=object()),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(performance_budget=object()),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(
                cancellation_propagation=object(),  # type: ignore[arg-type]
            ),
            lambda: StreamCancellationPropagation(targets=[]),  # type: ignore[arg-type]
            lambda: StreamCancellationPropagation(targets=()),
            lambda: StreamCancellationPropagation(
                targets=(StreamCancellationPropagationTarget.CONSUMER,) * 2,
            ),
            lambda: StreamCancellationPropagation(
                targets=(cast(Any, "consumer"),)
            ),
            lambda: StreamCancellationPropagation(idempotent=False),
            lambda: StreamCancellationPropagation(
                starts_no_new_work_after_terminal=False
            ),
            lambda: StreamRuntimeContract(close_after_terminal="yes"),  # type: ignore[arg-type]
            lambda: StreamRuntimeContract(close_after_terminal=False),
            lambda: StreamRuntimeContract(cancellation_as_terminal=False),
            lambda: validate_stream_runtime_contract(cast(Any, None)),
        )

        for build_value in invalid_values:
            with self.subTest(build_value=build_value):
                with self.assertRaises(AssertionError):
                    build_value()

    def test_tool_lifecycle_contract_defaults_and_validation(self) -> None:
        contract = StreamToolLifecycleContract()

        self.assertTrue(contract.stable_tool_call_id_required)
        self.assertTrue(contract.confirmation_required_before_parallel_fanout)
        self.assertTrue(contract.side_effecting_tools_serial_by_default)
        self.assertTrue(contract.terminal_exactly_once)
        self.assertTrue(contract.terminal_idempotent_for_accumulation)

        invalid_values = (
            lambda: StreamToolLifecycleContract(observation_order="emission"),  # type: ignore[arg-type]
            lambda: StreamToolLifecycleContract(
                stable_tool_call_id_required=False
            ),
            lambda: StreamToolLifecycleContract(
                confirmation_required_before_parallel_fanout=False
            ),
            lambda: StreamToolLifecycleContract(
                side_effecting_tools_serial_by_default=False
            ),
            lambda: StreamToolLifecycleContract(terminal_exactly_once=False),
            lambda: StreamToolLifecycleContract(
                terminal_idempotent_for_accumulation=False
            ),
            lambda: StreamToolObservation(
                tool_call_id="",
                arguments="",
                output="",
                terminal_kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
            ),
            lambda: StreamToolObservation(
                tool_call_id="tool-1",
                arguments="",
                output="",
                terminal_kind=StreamItemKind.ANSWER_DELTA,
            ),
        )

        for build_value in invalid_values:
            with self.subTest(build_value=build_value):
                with self.assertRaises(AssertionError):
                    build_value()

    def test_item_validation_rejects_malformed_fields(self) -> None:
        bad_items = (
            lambda: _item(
                StreamItemKind.ANSWER_DELTA,
                0,
                channel=StreamChannel.REASONING,
                text_delta="x",
            ),
            lambda: _item(StreamItemKind.ANSWER_DELTA, 0),
            lambda: _item(StreamItemKind.ANSWER_DONE, 0, text_delta="x"),
            lambda: _item(
                StreamItemKind.TOOL_CALL_READY,
                0,
                correlation=StreamItemCorrelation(),
            ),
            lambda: _item(StreamItemKind.USAGE_COMPLETED, 0),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                usage={"input_tokens": 1},
            ),
            lambda: _item(StreamItemKind.STREAM_ERRORED, 0),
            lambda: _item(
                StreamItemKind.STREAM_ERRORED,
                0,
                terminal_outcome=StreamTerminalOutcome.CANCELLED,
            ),
            lambda: _item(
                StreamItemKind.STREAM_CLOSED,
                0,
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                stream_session_id="",
            ),
            lambda: _item(StreamItemKind.STREAM_STARTED, -1),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                metadata=[],  # type: ignore[arg-type]
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                provider_family="",
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                provider_event_type="",
            ),
            lambda: _item(
                StreamItemKind.STREAM_STARTED,
                0,
                timestamp="now",  # type: ignore[arg-type]
            ),
        )

        for build_item in bad_items:
            with self.subTest(build_item=build_item):
                with self.assertRaises(AssertionError):
                    build_item()

    def test_correlation_rejects_malformed_fields(self) -> None:
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(provider_request_id="")
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(node_id=1)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(parent_sequence="0")  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamItemCorrelation(parent_sequence=-1)

    def test_golden_trace_rejects_malformed_metadata(self) -> None:
        valid_items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _stream_errored(1),
        )

        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="", items=valid_items)
        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="x", format_version=0, items=valid_items)
        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="x", items=list(valid_items))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            StreamGoldenTrace(name="x", description="", items=valid_items)
        with self.assertRaises(StreamValidationError):
            StreamGoldenTrace(name="x", items=valid_items[:1])

    def test_sequence_validator_rejects_terminal_and_order_errors(
        self,
    ) -> None:
        cases = (
            (),
            (_item(StreamItemKind.STREAM_STARTED, 0),),
            (_item(StreamItemKind.STREAM_STARTED, 0), _stream_errored(0)),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_ERRORED,
                    1,
                    run_id="run-2",
                    terminal_outcome=StreamTerminalOutcome.ERRORED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_DIAGNOSTIC,
                    1,
                    correlation=StreamItemCorrelation(parent_sequence=1),
                    text_delta="diagnostic",
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(
                    StreamItemKind.STREAM_CANCELLED,
                    2,
                    terminal_outcome=StreamTerminalOutcome.CANCELLED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.STREAM_DIAGNOSTIC, 2, text_delta="late"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.STREAM_CLOSED, 1),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.STREAM_CLOSED, 2),
                _item(StreamItemKind.STREAM_CLOSED, 3),
            ),
            (_item(StreamItemKind.STREAM_STARTED, 0), _stream_completed(1)),
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(StreamValidationError):
                    validate_canonical_stream_items(items)

    def test_sequence_validator_rejects_channel_boundary_errors(self) -> None:
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        cases = (
            (
                _item(StreamItemKind.ANSWER_DONE, 0),
                _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="late"),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.REASONING_DONE, 0),
                _item(StreamItemKind.REASONING_DELTA, 1, text_delta="late"),
                _stream_errored(2),
            ),
            (
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    0,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_UPDATE,
                    1,
                    usage={"input_tokens": 2},
                ),
                _stream_errored(2),
            ),
            (
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    0,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _stream_errored(2),
            ),
            (
                _item(StreamItemKind.TOOL_CALL_DONE, 0, correlation=tool),
                _item(StreamItemKind.TOOL_CALL_READY, 1, correlation=tool),
                _stream_errored(2),
            ),
            (
                _item(
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    0,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    1,
                    correlation=tool,
                    text_delta="late",
                ),
                _stream_errored(2),
            ),
            (
                _item(
                    StreamItemKind.TOOL_EXECUTION_COMPLETED,
                    0,
                    correlation=tool,
                ),
                _item(
                    StreamItemKind.TOOL_EXECUTION_ERROR,
                    1,
                    correlation=tool,
                ),
                _stream_errored(2),
            ),
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(StreamValidationError):
                    validate_canonical_stream_items(items)

    def test_tool_lifecycle_assembles_planned_order_observations(
        self,
    ) -> None:
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                1,
                tool_call_id="tool-2",
                text_delta='{"city"',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                2,
                tool_call_id="tool-2",
                text_delta=':"Paris"}',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                3,
                tool_call_id="tool-2",
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                4,
                tool_call_id="tool-2",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                5,
                tool_call_id="tool-2",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                6,
                tool_call_id="tool-2",
                text_delta="city ",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                7,
                tool_call_id="tool-2",
                data={"percent": 50},
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                8,
                tool_call_id="tool-2",
                data={"message": "failed"},
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                9,
                tool_call_id="tool-1",
                text_delta='{"expression"',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                10,
                tool_call_id="tool-1",
                text_delta=':"2+2"}',
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                11,
                tool_call_id="tool-1",
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                12,
                tool_call_id="tool-1",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                13,
                tool_call_id="tool-1",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                14,
                tool_call_id="tool-1",
                text_delta="4",
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                15,
                tool_call_id="tool-1",
                data={"result": 4},
            ),
        )

        validated = validate_tool_lifecycle_items(
            items, planned_tool_call_ids=("tool-1", "tool-2")
        )
        observations = assemble_tool_observations(
            validated, planned_tool_call_ids=("tool-1", "tool-2")
        )

        self.assertEqual(validated, items)
        self.assertEqual(
            [item.tool_call_id for item in observations], ["tool-1", "tool-2"]
        )
        self.assertEqual(observations[0].arguments, '{"expression":"2+2"}')
        self.assertEqual(observations[0].output, "4")
        self.assertIs(
            observations[0].terminal_kind,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
        )
        self.assertEqual(observations[0].terminal_data, {"result": 4})
        self.assertEqual(observations[1].arguments, '{"city":"Paris"}')
        self.assertEqual(observations[1].output, "city ")
        self.assertIs(
            observations[1].terminal_kind,
            StreamItemKind.TOOL_EXECUTION_ERROR,
        )
        self.assertEqual(observations[1].terminal_data, {"message": "failed"})

    def test_tool_lifecycle_allows_empty_tool_stream(self) -> None:
        self.assertEqual(validate_tool_lifecycle_items(()), ())
        self.assertEqual(assemble_tool_observations(()), ())

    def test_tool_lifecycle_rejects_malformed_planned_order(self) -> None:
        invalid_values = (
            lambda: validate_tool_lifecycle_items(
                (), planned_tool_call_ids=("",)
            ),
            lambda: validate_tool_lifecycle_items(
                (), planned_tool_call_ids=("tool-1", "tool-1")
            ),
            lambda: validate_tool_lifecycle_items(
                (), planned_tool_call_ids=(cast(Any, 1),)
            ),
        )

        for validate in invalid_values:
            with self.subTest(validate=validate):
                with self.assertRaises(AssertionError):
                    validate()

    def test_tool_lifecycle_rejects_invalid_transitions(self) -> None:
        complete_prefix = (
            _tool_item(
                StreamItemKind.TOOL_CALL_READY,
                0,
            ),
            _tool_item(
                StreamItemKind.TOOL_CALL_DONE,
                1,
            ),
            _tool_item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                2,
            ),
        )
        cases = (
            (
                _tool_item(
                    StreamItemKind.TOOL_CALL_READY,
                    0,
                    tool_call_id="tool-2",
                ),
            ),
            (
                _tool_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    0,
                    text_delta="late",
                ),
                _tool_item(StreamItemKind.TOOL_CALL_READY, 1),
                _tool_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    2,
                    text_delta="later",
                ),
            ),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_CALL_READY, 1),
            ),
            (_tool_item(StreamItemKind.TOOL_CALL_DONE, 0),),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_CALL_DONE, 1),
                _tool_item(StreamItemKind.TOOL_CALL_DONE, 2),
            ),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_EXECUTION_STARTED, 1),
            ),
            complete_prefix
            + (_tool_item(StreamItemKind.TOOL_EXECUTION_STARTED, 3),),
            (
                _tool_item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    0,
                    text_delta="early",
                ),
            ),
            (_tool_item(StreamItemKind.TOOL_EXECUTION_PROGRESS, 0),),
            (_tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 0),),
            complete_prefix
            + (
                _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
                _tool_item(StreamItemKind.TOOL_EXECUTION_ERROR, 4),
            ),
            complete_prefix
            + (
                _tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),
                _tool_item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    4,
                    text_delta="late",
                ),
            ),
            (
                _tool_item(
                    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    0,
                    text_delta="partial",
                ),
            ),
            (_tool_item(StreamItemKind.TOOL_CALL_READY, 0),),
            (
                _tool_item(StreamItemKind.TOOL_CALL_READY, 0),
                _tool_item(StreamItemKind.TOOL_CALL_DONE, 1),
            ),
            complete_prefix,
        )

        for items in cases:
            with self.subTest(items=items):
                with self.assertRaises(StreamValidationError):
                    validate_tool_lifecycle_items(items)

        with self.assertRaises(StreamValidationError):
            validate_tool_lifecycle_items(
                complete_prefix
                + (_tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),),
                planned_tool_call_ids=("tool-2",),
            )
        with self.assertRaises(StreamValidationError):
            validate_tool_lifecycle_items(
                complete_prefix
                + (_tool_item(StreamItemKind.TOOL_EXECUTION_COMPLETED, 3),),
                planned_tool_call_ids=("tool-1", "tool-2"),
            )

    def test_accumulator_separates_answer_from_other_channels(self) -> None:
        tool = StreamItemCorrelation(tool_call_id="tool-1")
        usage = {"input_tokens": 3, "output_tokens": 2}
        items = (
            _item(StreamItemKind.STREAM_STARTED, 0),
            _item(StreamItemKind.ANSWER_DELTA, 1, text_delta="Hello"),
            _item(StreamItemKind.REASONING_DELTA, 2, text_delta="think"),
            _item(StreamItemKind.REASONING_DONE, 3),
            _item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                4,
                correlation=tool,
                text_delta='{"city"',
            ),
            _item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                5,
                correlation=tool,
                text_delta=':"Paris"}',
            ),
            _item(StreamItemKind.TOOL_CALL_READY, 6, correlation=tool),
            _item(StreamItemKind.TOOL_CALL_DONE, 7, correlation=tool),
            _item(
                StreamItemKind.TOOL_EXECUTION_STARTED,
                8,
                correlation=tool,
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                9,
                correlation=tool,
                text_delta="tool ",
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_OUTPUT,
                10,
                correlation=tool,
                text_delta="output",
            ),
            _item(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                11,
                correlation=tool,
            ),
            _item(StreamItemKind.ANSWER_DELTA, 12, text_delta=" world"),
            _item(StreamItemKind.ANSWER_DONE, 13),
            _item(
                StreamItemKind.FLOW_EVENT,
                14,
                data={"node": "done"},
            ),
            _item(
                StreamItemKind.STREAM_DIAGNOSTIC,
                15,
                text_delta="internal",
                visibility=StreamVisibility.DIAGNOSTIC,
            ),
            _item(
                StreamItemKind.USAGE_UPDATE,
                16,
                usage={"input_tokens": 3},
            ),
            _item(StreamItemKind.USAGE_COMPLETED, 17, usage=usage),
            _stream_completed(18),
            _item(StreamItemKind.STREAM_CLOSED, 19),
        )

        accumulator = accumulate_canonical_stream_items(items)

        self.assertEqual(accumulator.items, items)
        self.assertEqual(accumulator.answer_text, "Hello world")
        self.assertEqual(accumulator.reasoning_text, "think")
        self.assertEqual(
            accumulator.tool_call_arguments, {"tool-1": '{"city":"Paris"}'}
        )
        self.assertEqual(
            accumulator.tool_execution_outputs, {"tool-1": "tool output"}
        )
        self.assertEqual(accumulator.diagnostics, (items[15],))
        self.assertEqual(accumulator.flow_items, (items[14],))
        self.assertEqual(accumulator.usage_items, (items[16], items[17]))
        self.assertEqual(
            accumulator.control_items,
            (items[0], items[15], items[18], items[19]),
        )
        self.assertEqual(accumulator.final_usage, usage)
        self.assertIs(
            accumulator.terminal_outcome, StreamTerminalOutcome.COMPLETED
        )
        self.assertTrue(accumulator.closed)

        copy = accumulator.tool_call_arguments
        copy["tool-1"] = "changed"
        self.assertEqual(
            accumulator.tool_call_arguments, {"tool-1": '{"city":"Paris"}'}
        )

    def test_accumulator_allows_atomic_completion_usage(self) -> None:
        usage: object = {}
        accumulator = CanonicalStreamAccumulator()
        returned = accumulator.add_many(
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_COMPLETED,
                    1,
                    usage=usage,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            )
        )

        self.assertIs(returned, accumulator)
        self.assertEqual(accumulator.final_usage, usage)
        self.assertEqual(accumulator.validate_complete(), accumulator.items)

    def test_accumulator_rejects_incremental_invalid_sequences(self) -> None:
        cases = (
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.ANSWER_DELTA, 0, text_delta="again"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_ERRORED,
                    1,
                    turn_id="turn-2",
                    terminal_outcome=StreamTerminalOutcome.ERRORED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.STREAM_COMPLETED,
                    1,
                    terminal_outcome=StreamTerminalOutcome.COMPLETED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.ANSWER_DELTA, 2, text_delta="late"),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(
                    StreamItemKind.STREAM_CANCELLED,
                    2,
                    terminal_outcome=StreamTerminalOutcome.CANCELLED,
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(StreamItemKind.STREAM_CLOSED, 1),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _stream_errored(1),
                _item(StreamItemKind.STREAM_CLOSED, 2),
                _item(StreamItemKind.STREAM_CLOSED, 3),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    2,
                    usage={"input_tokens": 1},
                ),
            ),
            (
                _item(StreamItemKind.STREAM_STARTED, 0),
                _item(
                    StreamItemKind.USAGE_COMPLETED,
                    1,
                    usage={"input_tokens": 1},
                ),
                _item(
                    StreamItemKind.USAGE_UPDATE,
                    2,
                    usage={"input_tokens": 2},
                ),
            ),
        )

        for items in cases:
            with self.subTest(items=items):
                accumulator = CanonicalStreamAccumulator()
                with self.assertRaises(StreamValidationError):
                    accumulator.add_many(items)

    def test_accumulator_rejects_bad_item_and_incomplete_stream(self) -> None:
        accumulator = CanonicalStreamAccumulator()

        with self.assertRaises(AssertionError):
            accumulator.add("bad")  # type: ignore[arg-type]

        accumulator.add(_item(StreamItemKind.STREAM_STARTED, 0))
        with self.assertRaises(StreamValidationError):
            accumulator.validate_complete()

    def test_text_generation_stream_base_contract(self) -> None:
        probe = _StreamProbe()

        with self.assertRaises(AssertionError):
            probe.__aiter__()
        with self.assertRaises(NotImplementedError):
            probe()
        with self.assertRaises(NotImplementedError):
            run(probe.__anext__())

        probe._generator = _single_token_generator()
        self.assertIs(probe.__aiter__(), probe)

    def test_single_stream_iterates_content_and_resets(self) -> None:
        stream = TextGenerationSingleStream(
            Token(token="one"),
            provider_family="openai",
            usage={"output_tokens": 1},
        )

        self.assertEqual(stream.content, Token(token="one"))
        self.assertEqual(stream.provider_family, "openai")
        self.assertEqual(stream.usage, {"output_tokens": 1})
        self.assertIs(stream(), stream)
        self.assertEqual(run(stream.__anext__()), Token(token="one"))
        with self.assertRaises(StopAsyncIteration):
            run(stream.__anext__())

        self.assertIs(stream.__aiter__(), stream)
        self.assertEqual(run(stream.__anext__()), Token(token="one"))

    def test_single_stream_final_text_uses_canonical_accumulator(
        self,
    ) -> None:
        stream = TextGenerationSingleStream(
            TokenDetail(id=1, token="answer", probability=0.5),
            usage={"output_tokens": 1},
        )

        self.assertEqual(stream.final_text, "answer")
        self.assertEqual(run(stream.to_str()), "answer")
        self.assertEqual(stream.accumulator.answer_text, "answer")
        self.assertEqual(
            [item.kind for item in stream.canonical_items],
            [
                StreamItemKind.STREAM_STARTED,
                StreamItemKind.ANSWER_DELTA,
                StreamItemKind.ANSWER_DONE,
                StreamItemKind.STREAM_COMPLETED,
                StreamItemKind.STREAM_CLOSED,
            ],
        )

    def test_legacy_surface_inventory_classifies_current_shapes(
        self,
    ) -> None:
        inventory = legacy_stream_surface_inventory()

        self.assertEqual(
            {entry.surface for entry in inventory},
            set(StreamLegacySurface),
        )
        self.assertEqual(
            len(inventory), len({entry.surface for entry in inventory})
        )

        shim_surfaces = {
            StreamLegacySurface.STRING,
            StreamLegacySurface.TOKEN,
            StreamLegacySurface.TOKEN_DETAIL,
            StreamLegacySurface.REASONING_TOKEN,
            StreamLegacySurface.TOOL_CALL_TOKEN,
        }
        for entry in inventory:
            with self.subTest(surface=entry.surface):
                self.assertIs(
                    classify_legacy_stream_surface(entry.surface), entry
                )
                self.assertTrue(entry.owner)
                self.assertTrue(entry.removal_condition)
                if entry.surface in shim_surfaces:
                    self.assertIs(
                        entry.classification,
                        (
                            StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                        ),
                    )
                    self.assertEqual(
                        entry.ingestion_shim, "canonical_item_from_token"
                    )
                    assert entry.canonical_kind is not None
                    self.assertIs(
                        entry.canonical_channel,
                        stream_channel_for_kind(entry.canonical_kind),
                    )
                else:
                    self.assertIs(
                        entry.classification,
                        StreamLegacySurfaceClassification.MIGRATE_LATER,
                    )
                    self.assertIsNone(entry.ingestion_shim)
                    self.assertIsNone(entry.canonical_kind)
                    self.assertIsNone(entry.canonical_channel)

        with self.assertRaises(AssertionError):
            classify_legacy_stream_surface("Token")  # type: ignore[arg-type]
        with patch("avalan.model.stream._LEGACY_STREAM_SURFACE_INVENTORY", ()):
            with self.assertRaises(StreamValidationError):
                classify_legacy_stream_surface(StreamLegacySurface.TOKEN)

    def test_legacy_surface_inventory_rejects_malformed_entries(self) -> None:
        invalid_entries = (
            lambda: StreamLegacySurfaceInventoryEntry(
                surface="Token",  # type: ignore[arg-type]
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="done",
                ingestion_shim="canonical_item_from_token",
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel=StreamChannel.ANSWER,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification="temporary",  # type: ignore[arg-type]
                owner="model.stream",
                removal_condition="done",
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="",
                removal_condition="done",
                ingestion_shim="canonical_item_from_token",
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel=StreamChannel.ANSWER,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="",
                ingestion_shim="canonical_item_from_token",
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel=StreamChannel.ANSWER,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="done",
                ingestion_shim="",
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel=StreamChannel.ANSWER,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="done",
                ingestion_shim="canonical_item_from_token",
                canonical_kind="answer.delta",  # type: ignore[arg-type]
                canonical_channel=StreamChannel.ANSWER,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="done",
                ingestion_shim="canonical_item_from_token",
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel="answer",  # type: ignore[arg-type]
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="done",
                ingestion_shim="canonical_item_from_token",
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel=StreamChannel.REASONING,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.TOKEN,
                classification=(
                    StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
                ),
                owner="model.stream",
                removal_condition="done",
                ingestion_shim=None,
                canonical_kind=StreamItemKind.ANSWER_DELTA,
                canonical_channel=StreamChannel.ANSWER,
            ),
            lambda: StreamLegacySurfaceInventoryEntry(
                surface=StreamLegacySurface.EVENT,
                classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
                owner="event",
                removal_condition="done",
                ingestion_shim="canonical_item_from_token",
            ),
        )

        for build_entry in invalid_entries:
            with self.subTest(build_entry=build_entry):
                with self.assertRaises(AssertionError):
                    build_entry()

    def test_legacy_token_canonical_projection_separates_channels(
        self,
    ) -> None:
        answer = canonical_item_from_token(
            Token(token="a"),
            1,
            stream_session_id="shim-stream",
            run_id="shim-run",
            turn_id="shim-turn",
        )
        detail = canonical_item_from_token(
            TokenDetail(id=7, token=" detail", probability=0.9),
            2,
            stream_session_id="shim-stream",
            run_id="shim-run",
            turn_id="shim-turn",
        )
        reasoning = canonical_item_from_token(
            ReasoningToken(token="r"),
            3,
            stream_session_id="shim-stream",
            run_id="shim-run",
            turn_id="shim-turn",
        )
        tool = canonical_item_from_token(
            ToolCallToken(token='{"x":', call=None),
            4,
            stream_session_id="shim-stream",
            run_id="shim-run",
            turn_id="shim-turn",
        )
        call_tool = canonical_item_from_token(
            ToolCallToken(
                token='"y"}',
                call=ToolCall(id="call-1", name="math", arguments={}),
            ),
            5,
            stream_session_id="shim-stream",
            run_id="shim-run",
            turn_id="shim-turn",
        )
        string_answer = canonical_item_from_token(
            "s",
            6,
            stream_session_id="shim-stream",
            run_id="shim-run",
            turn_id="shim-turn",
        )

        self.assertIs(answer.kind, StreamItemKind.ANSWER_DELTA)
        self.assertIs(detail.kind, StreamItemKind.ANSWER_DELTA)
        self.assertIs(reasoning.kind, StreamItemKind.REASONING_DELTA)
        self.assertIs(reasoning.visibility, StreamVisibility.PRIVATE)
        self.assertIs(tool.kind, StreamItemKind.TOOL_CALL_ARGUMENT_DELTA)
        self.assertEqual(tool.correlation.tool_call_id, "legacy-tool-call")
        self.assertEqual(call_tool.correlation.tool_call_id, "call-1")
        self.assertEqual(string_answer.text_delta, "s")
        self.assertEqual(
            [
                item.sequence
                for item in (answer, detail, reasoning, tool, call_tool)
            ],
            [1, 2, 3, 4, 5],
        )
        self.assertEqual(
            {item.stream_session_id for item in (answer, detail, reasoning)},
            {"shim-stream"},
        )

        trace_items = (
            _item(
                StreamItemKind.STREAM_STARTED,
                0,
                stream_session_id="shim-stream",
                run_id="shim-run",
                turn_id="shim-turn",
            ),
            answer,
            detail,
            reasoning,
            tool,
            call_tool,
            string_answer,
            _item(
                StreamItemKind.ANSWER_DONE,
                7,
                stream_session_id="shim-stream",
                run_id="shim-run",
                turn_id="shim-turn",
            ),
            _item(
                StreamItemKind.USAGE_COMPLETED,
                8,
                stream_session_id="shim-stream",
                run_id="shim-run",
                turn_id="shim-turn",
                usage={"output_tokens": 5},
            ),
            _item(
                StreamItemKind.STREAM_COMPLETED,
                9,
                stream_session_id="shim-stream",
                run_id="shim-run",
                turn_id="shim-turn",
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
            ),
        )
        accumulator = accumulate_canonical_stream_items(trace_items)

        self.assertEqual(accumulator.answer_text, "a details")
        self.assertEqual(accumulator.reasoning_text, "r")
        self.assertEqual(
            accumulator.tool_call_arguments,
            {"legacy-tool-call": '{"x":', "call-1": '"y"}'},
        )
        self.assertEqual(accumulator.final_usage, {"output_tokens": 5})

        with self.assertRaises(AssertionError):
            canonical_item_from_token("bad", -1)
        with self.assertRaises(AssertionError):
            canonical_item_from_token(object(), 0)  # type: ignore[arg-type]
