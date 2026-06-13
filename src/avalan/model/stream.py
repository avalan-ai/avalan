from ..entities import (
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCallToken,
)
from ..types import LooseJsonValue
from .provider import ProviderFamily, provider_family_value

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, AsyncIterator, cast


class StreamChannel(StrEnum):
    ANSWER = "answer"
    REASONING = "reasoning"
    TOOL_CALL = "tool_call"
    TOOL_EXECUTION = "tool_execution"
    FLOW = "flow"
    USAGE = "usage"
    CONTROL = "control"


class StreamItemKind(StrEnum):
    ANSWER_DELTA = "answer.delta"
    ANSWER_DONE = "answer.done"
    REASONING_DELTA = "reasoning.delta"
    REASONING_DONE = "reasoning.done"
    TOOL_CALL_ARGUMENT_DELTA = "tool_call.argument_delta"
    TOOL_CALL_READY = "tool_call.ready"
    TOOL_CALL_DONE = "tool_call.done"
    TOOL_EXECUTION_STARTED = "tool_execution.started"
    TOOL_EXECUTION_OUTPUT = "tool_execution.output"
    TOOL_EXECUTION_PROGRESS = "tool_execution.progress"
    TOOL_EXECUTION_COMPLETED = "tool_execution.completed"
    TOOL_EXECUTION_ERROR = "tool_execution.error"
    TOOL_EXECUTION_CANCELLED = "tool_execution.cancelled"
    MODEL_CONTINUATION_STARTED = "model_continuation.started"
    MODEL_CONTINUATION_COMPLETED = "model_continuation.completed"
    MODEL_CONTINUATION_ERROR = "model_continuation.error"
    MODEL_CONTINUATION_CANCELLED = "model_continuation.cancelled"
    FLOW_EVENT = "flow.event"
    USAGE_UPDATE = "usage.update"
    USAGE_COMPLETED = "usage.completed"
    STREAM_STARTED = "stream.started"
    STREAM_DIAGNOSTIC = "stream.diagnostic"
    STREAM_COMPLETED = "stream.completed"
    STREAM_ERRORED = "stream.errored"
    STREAM_CANCELLED = "stream.cancelled"
    STREAM_CLOSED = "stream.closed"


class StreamTerminalOutcome(StrEnum):
    COMPLETED = "completed"
    ERRORED = "errored"
    CANCELLED = "cancelled"


class StreamVisibility(StrEnum):
    PUBLIC = "public"
    PRIVATE = "private"
    REDACTED = "redacted"
    DIAGNOSTIC = "diagnostic"


class StreamBackpressurePolicy(StrEnum):
    BLOCK = "block"
    FAIL = "fail"
    CANCEL = "cancel"


class StreamCancellationDrainPolicy(StrEnum):
    DRAIN_BUFFERED = "drain_buffered"
    DISCARD_BUFFERED = "discard_buffered"


class StreamCancellationPropagationTarget(StrEnum):
    CONSUMER = "consumer"
    STREAM_SESSION = "stream_session"
    PROVIDER = "provider"
    LOCAL_GENERATION = "local_generation"
    RUNNING_TOOLS = "running_tools"
    FANOUT_SUBSCRIBERS = "fanout_subscribers"
    CLEANUP_HOOKS = "cleanup_hooks"


class StreamToolObservationOrder(StrEnum):
    PLANNED_CALL_ORDER = "planned_call_order"


class StreamLegacySurface(StrEnum):
    STRING = "str"
    TOKEN = "Token"
    TOKEN_DETAIL = "TokenDetail"
    REASONING_TOKEN = "ReasoningToken"
    TOOL_CALL_TOKEN = "ToolCallToken"
    EVENT = "Event"


class StreamLegacySurfaceClassification(StrEnum):
    REMOVE_NOW = "remove_now"
    MIGRATE_LATER = "migrate_later"
    TEMPORARY_INGESTION_SHIM = "temporary_ingestion_shim"


class StreamValidationError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamLegacySurfaceInventoryEntry:
    surface: StreamLegacySurface
    classification: StreamLegacySurfaceClassification
    owner: str
    removal_condition: str
    ingestion_shim: str | None = None
    canonical_kind: StreamItemKind | None = None
    canonical_channel: StreamChannel | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.surface, StreamLegacySurface)
        assert isinstance(
            self.classification,
            StreamLegacySurfaceClassification,
        )
        _assert_non_empty_string(self.owner, "owner")
        _assert_non_empty_string(self.removal_condition, "removal_condition")
        if self.ingestion_shim is not None:
            _assert_non_empty_string(self.ingestion_shim, "ingestion_shim")
        if self.canonical_kind is not None:
            assert isinstance(self.canonical_kind, StreamItemKind)
        if self.canonical_channel is not None:
            assert isinstance(self.canonical_channel, StreamChannel)
        if (
            self.canonical_kind is not None
            and self.canonical_channel is not None
        ):
            assert self.canonical_channel is stream_channel_for_kind(
                self.canonical_kind
            )

        if (
            self.classification
            is StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
        ):
            assert self.ingestion_shim is not None
            assert self.canonical_kind is not None
            assert self.canonical_channel is not None
        else:
            assert self.ingestion_shim is None


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamItemCorrelation:
    provider_request_id: str | None = None
    model_continuation_id: str | None = None
    tool_call_id: str | None = None
    flow_run_id: str | None = None
    node_id: str | None = None
    parent_sequence: int | None = None
    protocol_item_id: str | None = None
    task_id: str | None = None
    artifact_id: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("provider_request_id", self.provider_request_id),
            ("model_continuation_id", self.model_continuation_id),
            ("tool_call_id", self.tool_call_id),
            ("flow_run_id", self.flow_run_id),
            ("node_id", self.node_id),
            ("protocol_item_id", self.protocol_item_id),
            ("task_id", self.task_id),
            ("artifact_id", self.artifact_id),
        ):
            if value is not None:
                assert isinstance(value, str), f"{field_name} must be a string"
                assert value.strip(), f"{field_name} must not be empty"
        if self.parent_sequence is not None:
            assert isinstance(
                self.parent_sequence, int
            ), "parent_sequence must be an integer"
            assert (
                self.parent_sequence >= 0
            ), "parent_sequence must not be negative"

    def to_trace_dict(self) -> dict[str, object]:
        result: dict[str, object] = {}
        for field_name, value in (
            ("provider_request_id", self.provider_request_id),
            ("model_continuation_id", self.model_continuation_id),
            ("tool_call_id", self.tool_call_id),
            ("flow_run_id", self.flow_run_id),
            ("node_id", self.node_id),
            ("parent_sequence", self.parent_sequence),
            ("protocol_item_id", self.protocol_item_id),
            ("task_id", self.task_id),
            ("artifact_id", self.artifact_id),
        ):
            if value is not None:
                result[field_name] = value
        return result


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamSessionLifecycle:
    single_use: bool = True
    cancellable: bool = True
    closeable: bool = True
    cleanup_owned: bool = True

    def __post_init__(self) -> None:
        for field_name, value in (
            ("single_use", self.single_use),
            ("cancellable", self.cancellable),
            ("closeable", self.closeable),
            ("cleanup_owned", self.cleanup_owned),
        ):
            assert isinstance(value, bool), f"{field_name} must be a boolean"


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamRetentionPolicy:
    accumulator_item_limit: int = 4096
    replay_history_item_limit: int = 1024
    ui_buffer_item_limit: int = 1024
    metrics_history_item_limit: int = 2048
    event_history_item_limit: int = 2048
    mcp_resource_item_limit: int = 512
    a2a_task_record_item_limit: int = 512
    flow_history_item_limit: int = 1024
    active_session_lossless: bool = True

    def __post_init__(self) -> None:
        _assert_positive_int(
            self.accumulator_item_limit, "accumulator_item_limit"
        )
        for field_name, value in (
            ("replay_history_item_limit", self.replay_history_item_limit),
            ("ui_buffer_item_limit", self.ui_buffer_item_limit),
            ("metrics_history_item_limit", self.metrics_history_item_limit),
            ("event_history_item_limit", self.event_history_item_limit),
            ("mcp_resource_item_limit", self.mcp_resource_item_limit),
            ("a2a_task_record_item_limit", self.a2a_task_record_item_limit),
            ("flow_history_item_limit", self.flow_history_item_limit),
        ):
            _assert_non_negative_int(value, field_name)
        assert isinstance(
            self.active_session_lossless, bool
        ), "active_session_lossless must be a boolean"
        assert self.active_session_lossless


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamPerformanceBudget:
    time_to_first_item_ms: int = 5000
    cancellation_latency_ms: int = 1000
    close_latency_ms: int = 1000
    max_queue_depth: int = 64
    max_memory_bytes: int = 16 * 1024 * 1024
    per_item_overhead_us: int = 250

    def __post_init__(self) -> None:
        for field_name, value in (
            ("time_to_first_item_ms", self.time_to_first_item_ms),
            ("cancellation_latency_ms", self.cancellation_latency_ms),
            ("close_latency_ms", self.close_latency_ms),
            ("max_queue_depth", self.max_queue_depth),
            ("max_memory_bytes", self.max_memory_bytes),
            ("per_item_overhead_us", self.per_item_overhead_us),
        ):
            _assert_positive_int(value, field_name)


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamCancellationPropagation:
    targets: tuple[StreamCancellationPropagationTarget, ...] = (
        StreamCancellationPropagationTarget.CONSUMER,
        StreamCancellationPropagationTarget.STREAM_SESSION,
        StreamCancellationPropagationTarget.PROVIDER,
        StreamCancellationPropagationTarget.LOCAL_GENERATION,
        StreamCancellationPropagationTarget.RUNNING_TOOLS,
        StreamCancellationPropagationTarget.FANOUT_SUBSCRIBERS,
        StreamCancellationPropagationTarget.CLEANUP_HOOKS,
    )
    idempotent: bool = True
    starts_no_new_work_after_terminal: bool = True

    def __post_init__(self) -> None:
        assert isinstance(self.targets, tuple), "targets must be a tuple"
        assert self.targets, "targets must not be empty"
        seen: set[StreamCancellationPropagationTarget] = set()
        for target in self.targets:
            assert isinstance(target, StreamCancellationPropagationTarget)
            assert target not in seen, "targets must be unique"
            seen.add(target)
        for field_name, value in (
            ("idempotent", self.idempotent),
            (
                "starts_no_new_work_after_terminal",
                self.starts_no_new_work_after_terminal,
            ),
        ):
            assert isinstance(value, bool), f"{field_name} must be a boolean"
            assert value, f"{field_name} must be true"


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamRuntimeContract:
    backpressure_policy: StreamBackpressurePolicy = (
        StreamBackpressurePolicy.BLOCK
    )
    cancellation_drain_policy: StreamCancellationDrainPolicy = (
        StreamCancellationDrainPolicy.DRAIN_BUFFERED
    )
    retention_policy: StreamRetentionPolicy = field(
        default_factory=StreamRetentionPolicy
    )
    performance_budget: StreamPerformanceBudget = field(
        default_factory=StreamPerformanceBudget
    )
    cancellation_propagation: StreamCancellationPropagation = field(
        default_factory=StreamCancellationPropagation
    )
    close_after_terminal: bool = True
    cancellation_as_terminal: bool = True

    def __post_init__(self) -> None:
        assert isinstance(self.backpressure_policy, StreamBackpressurePolicy)
        assert isinstance(
            self.cancellation_drain_policy, StreamCancellationDrainPolicy
        )
        assert isinstance(self.retention_policy, StreamRetentionPolicy)
        assert isinstance(self.performance_budget, StreamPerformanceBudget)
        assert isinstance(
            self.cancellation_propagation,
            StreamCancellationPropagation,
        )
        assert isinstance(
            self.close_after_terminal, bool
        ), "close_after_terminal must be a boolean"
        assert isinstance(
            self.cancellation_as_terminal, bool
        ), "cancellation_as_terminal must be a boolean"
        assert self.close_after_terminal
        assert self.cancellation_as_terminal

    @property
    def buffered_items_may_drain_after_cancellation(self) -> bool:
        return (
            self.cancellation_drain_policy
            is StreamCancellationDrainPolicy.DRAIN_BUFFERED
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamToolLifecycleContract:
    observation_order: StreamToolObservationOrder = (
        StreamToolObservationOrder.PLANNED_CALL_ORDER
    )
    stable_tool_call_id_required: bool = True
    confirmation_required_before_parallel_fanout: bool = True
    side_effecting_tools_serial_by_default: bool = True
    terminal_exactly_once: bool = True
    terminal_idempotent_for_accumulation: bool = True

    def __post_init__(self) -> None:
        assert isinstance(self.observation_order, StreamToolObservationOrder)
        assert (
            self.observation_order
            is StreamToolObservationOrder.PLANNED_CALL_ORDER
        )
        for field_name, value in (
            (
                "stable_tool_call_id_required",
                self.stable_tool_call_id_required,
            ),
            (
                "confirmation_required_before_parallel_fanout",
                self.confirmation_required_before_parallel_fanout,
            ),
            (
                "side_effecting_tools_serial_by_default",
                self.side_effecting_tools_serial_by_default,
            ),
            ("terminal_exactly_once", self.terminal_exactly_once),
            (
                "terminal_idempotent_for_accumulation",
                self.terminal_idempotent_for_accumulation,
            ),
        ):
            assert isinstance(value, bool), f"{field_name} must be a boolean"
            assert value, f"{field_name} must be true"


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamToolObservation:
    tool_call_id: str
    arguments: str
    output: str
    terminal_kind: StreamItemKind
    terminal_data: LooseJsonValue | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.tool_call_id, str)
        assert self.tool_call_id.strip()
        assert isinstance(self.arguments, str)
        assert isinstance(self.output, str)
        assert self.terminal_kind in _TOOL_EXECUTION_TERMINAL_KINDS


@dataclass(frozen=True, kw_only=True, slots=True)
class CanonicalStreamItem:
    stream_session_id: str
    run_id: str
    turn_id: str
    sequence: int
    kind: StreamItemKind
    channel: StreamChannel
    correlation: StreamItemCorrelation = field(
        default_factory=StreamItemCorrelation
    )
    text_delta: str | None = None
    data: LooseJsonValue | None = None
    usage: LooseJsonValue | None = None
    terminal_outcome: StreamTerminalOutcome | None = None
    visibility: StreamVisibility = StreamVisibility.PUBLIC
    metadata: dict[str, LooseJsonValue] = field(default_factory=dict)
    provider_payload: LooseJsonValue | None = None
    provider_family: str | None = None
    provider_event_type: str | None = None
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("stream_session_id", self.stream_session_id),
            ("run_id", self.run_id),
            ("turn_id", self.turn_id),
        ):
            assert isinstance(value, str), f"{field_name} must be a string"
            assert value.strip(), f"{field_name} must not be empty"
        assert isinstance(self.sequence, int), "sequence must be an integer"
        assert self.sequence >= 0, "sequence must not be negative"
        assert isinstance(self.kind, StreamItemKind)
        assert isinstance(self.channel, StreamChannel)
        assert self.channel is stream_channel_for_kind(self.kind)
        assert isinstance(self.correlation, StreamItemCorrelation)
        assert isinstance(self.visibility, StreamVisibility)
        assert isinstance(self.metadata, dict), "metadata must be a dict"
        if self.text_delta is not None:
            assert isinstance(self.text_delta, str)
        if self.provider_family is not None:
            assert isinstance(self.provider_family, str)
            assert self.provider_family.strip()
        if self.provider_event_type is not None:
            assert isinstance(self.provider_event_type, str)
            assert self.provider_event_type.strip()
        if self.timestamp is not None:
            assert isinstance(self.timestamp, datetime)
        self._validate_kind_payload()

    @property
    def is_stream_terminal(self) -> bool:
        return is_stream_terminal_kind(self.kind)

    @property
    def is_tool_execution_terminal(self) -> bool:
        return is_tool_execution_terminal_kind(self.kind)

    def to_trace_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "stream_session_id": self.stream_session_id,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "channel": self.channel.value,
            "visibility": self.visibility.value,
        }
        correlation = self.correlation.to_trace_dict()
        if correlation:
            result["correlation"] = correlation
        terminal_outcome = (
            self.terminal_outcome.value if self.terminal_outcome else None
        )
        timestamp = self.timestamp.isoformat() if self.timestamp else None
        for field_name, value in (
            ("text_delta", self.text_delta),
            ("data", self.data),
            ("usage", self.usage),
            ("terminal_outcome", terminal_outcome),
            ("metadata", self.metadata or None),
            ("provider_payload", self.provider_payload),
            ("provider_family", self.provider_family),
            ("provider_event_type", self.provider_event_type),
            ("timestamp", timestamp),
        ):
            if value is not None:
                result[field_name] = value
        return result

    def _validate_kind_payload(self) -> None:
        if self.kind in _TEXT_DELTA_KINDS:
            assert self.text_delta is not None
        elif self.text_delta is not None:
            assert self.kind is StreamItemKind.STREAM_DIAGNOSTIC

        if self.kind in _TOOL_CORRELATED_KINDS:
            assert self.correlation.tool_call_id is not None

        if self.usage is not None:
            assert self.kind in _USAGE_KINDS or (
                self.kind is StreamItemKind.STREAM_COMPLETED
            )
        elif self.kind in _USAGE_KINDS:
            raise AssertionError("usage items must carry usage")

        expected_outcome = stream_terminal_outcome_for_kind(self.kind)
        if expected_outcome is None:
            assert self.terminal_outcome is None
        else:
            assert self.terminal_outcome is expected_outcome


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamGoldenTrace:
    name: str
    items: tuple[CanonicalStreamItem, ...]
    format_version: int = 1
    description: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.name, str), "name must be a string"
        assert self.name.strip(), "name must not be empty"
        assert isinstance(
            self.format_version, int
        ), "format_version must be an integer"
        assert self.format_version > 0, "format_version must be positive"
        assert isinstance(self.items, tuple), "items must be a tuple"
        validate_canonical_stream_items(self.items)
        if self.description is not None:
            assert isinstance(self.description, str)
            assert self.description.strip()

    def to_fixture(self) -> dict[str, object]:
        fixture: dict[str, object] = {
            "format_version": self.format_version,
            "name": self.name,
            "items": [item.to_trace_dict() for item in self.items],
        }
        if self.description is not None:
            fixture["description"] = self.description
        return fixture


_STREAM_KIND_CHANNELS: dict[StreamItemKind, StreamChannel] = {
    StreamItemKind.ANSWER_DELTA: StreamChannel.ANSWER,
    StreamItemKind.ANSWER_DONE: StreamChannel.ANSWER,
    StreamItemKind.REASONING_DELTA: StreamChannel.REASONING,
    StreamItemKind.REASONING_DONE: StreamChannel.REASONING,
    StreamItemKind.TOOL_CALL_ARGUMENT_DELTA: StreamChannel.TOOL_CALL,
    StreamItemKind.TOOL_CALL_READY: StreamChannel.TOOL_CALL,
    StreamItemKind.TOOL_CALL_DONE: StreamChannel.TOOL_CALL,
    StreamItemKind.TOOL_EXECUTION_STARTED: StreamChannel.TOOL_EXECUTION,
    StreamItemKind.TOOL_EXECUTION_OUTPUT: StreamChannel.TOOL_EXECUTION,
    StreamItemKind.TOOL_EXECUTION_PROGRESS: StreamChannel.TOOL_EXECUTION,
    StreamItemKind.TOOL_EXECUTION_COMPLETED: StreamChannel.TOOL_EXECUTION,
    StreamItemKind.TOOL_EXECUTION_ERROR: StreamChannel.TOOL_EXECUTION,
    StreamItemKind.TOOL_EXECUTION_CANCELLED: StreamChannel.TOOL_EXECUTION,
    StreamItemKind.MODEL_CONTINUATION_STARTED: StreamChannel.CONTROL,
    StreamItemKind.MODEL_CONTINUATION_COMPLETED: StreamChannel.CONTROL,
    StreamItemKind.MODEL_CONTINUATION_ERROR: StreamChannel.CONTROL,
    StreamItemKind.MODEL_CONTINUATION_CANCELLED: StreamChannel.CONTROL,
    StreamItemKind.FLOW_EVENT: StreamChannel.FLOW,
    StreamItemKind.USAGE_UPDATE: StreamChannel.USAGE,
    StreamItemKind.USAGE_COMPLETED: StreamChannel.USAGE,
    StreamItemKind.STREAM_STARTED: StreamChannel.CONTROL,
    StreamItemKind.STREAM_DIAGNOSTIC: StreamChannel.CONTROL,
    StreamItemKind.STREAM_COMPLETED: StreamChannel.CONTROL,
    StreamItemKind.STREAM_ERRORED: StreamChannel.CONTROL,
    StreamItemKind.STREAM_CANCELLED: StreamChannel.CONTROL,
    StreamItemKind.STREAM_CLOSED: StreamChannel.CONTROL,
}
_STREAM_TERMINAL_OUTCOMES: dict[StreamItemKind, StreamTerminalOutcome] = {
    StreamItemKind.STREAM_COMPLETED: StreamTerminalOutcome.COMPLETED,
    StreamItemKind.STREAM_ERRORED: StreamTerminalOutcome.ERRORED,
    StreamItemKind.STREAM_CANCELLED: StreamTerminalOutcome.CANCELLED,
}
_TEXT_DELTA_KINDS = frozenset(
    {
        StreamItemKind.ANSWER_DELTA,
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
    }
)
_TOOL_CORRELATED_KINDS = frozenset(
    {
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    }
)
_TOOL_CALL_KINDS = frozenset(
    {
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
    }
)
_TOOL_EXECUTION_TERMINAL_KINDS = frozenset(
    {
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    }
)
_TOOL_EXECUTION_KINDS = frozenset(
    {
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        *tuple(_TOOL_EXECUTION_TERMINAL_KINDS),
    }
)
_TOOL_RELATED_KINDS = _TOOL_CALL_KINDS | _TOOL_EXECUTION_KINDS
_USAGE_KINDS = frozenset(
    {
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    }
)


@dataclass(slots=True)
class _ToolLifecycleState:
    ready: bool = False
    call_done: bool = False
    execution_started: bool = False
    terminal_kind: StreamItemKind | None = None
    terminal_data: LooseJsonValue | None = None
    argument_deltas: list[str] = field(default_factory=list)
    output_deltas: list[str] = field(default_factory=list)


def _assert_positive_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_non_negative_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"


def _assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def stream_channel_for_kind(kind: StreamItemKind) -> StreamChannel:
    assert isinstance(kind, StreamItemKind)
    return _STREAM_KIND_CHANNELS[kind]


def stream_terminal_outcome_for_kind(
    kind: StreamItemKind,
) -> StreamTerminalOutcome | None:
    assert isinstance(kind, StreamItemKind)
    return _STREAM_TERMINAL_OUTCOMES.get(kind)


_LEGACY_STREAM_SURFACE_INVENTORY: tuple[
    StreamLegacySurfaceInventoryEntry, ...
] = (
    StreamLegacySurfaceInventoryEntry(
        surface=StreamLegacySurface.STRING,
        classification=(
            StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
        ),
        owner="model.stream",
        removal_condition=(
            "Canonical item iteration replaces string stream output."
        ),
        ingestion_shim="canonical_item_from_token",
        canonical_kind=StreamItemKind.ANSWER_DELTA,
        canonical_channel=StreamChannel.ANSWER,
    ),
    StreamLegacySurfaceInventoryEntry(
        surface=StreamLegacySurface.TOKEN,
        classification=(
            StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
        ),
        owner="model.stream",
        removal_condition=(
            "Canonical item iteration replaces Token stream output."
        ),
        ingestion_shim="canonical_item_from_token",
        canonical_kind=StreamItemKind.ANSWER_DELTA,
        canonical_channel=StreamChannel.ANSWER,
    ),
    StreamLegacySurfaceInventoryEntry(
        surface=StreamLegacySurface.TOKEN_DETAIL,
        classification=(
            StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
        ),
        owner="model.stream",
        removal_condition=(
            "Canonical item metadata replaces TokenDetail stream output."
        ),
        ingestion_shim="canonical_item_from_token",
        canonical_kind=StreamItemKind.ANSWER_DELTA,
        canonical_channel=StreamChannel.ANSWER,
    ),
    StreamLegacySurfaceInventoryEntry(
        surface=StreamLegacySurface.REASONING_TOKEN,
        classification=(
            StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
        ),
        owner="model.stream",
        removal_condition=(
            "Canonical reasoning channel replaces ReasoningToken output."
        ),
        ingestion_shim="canonical_item_from_token",
        canonical_kind=StreamItemKind.REASONING_DELTA,
        canonical_channel=StreamChannel.REASONING,
    ),
    StreamLegacySurfaceInventoryEntry(
        surface=StreamLegacySurface.TOOL_CALL_TOKEN,
        classification=(
            StreamLegacySurfaceClassification.TEMPORARY_INGESTION_SHIM
        ),
        owner="model.stream",
        removal_condition=(
            "Canonical tool-call items replace ToolCallToken output."
        ),
        ingestion_shim="canonical_item_from_token",
        canonical_kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        canonical_channel=StreamChannel.TOOL_CALL,
    ),
    StreamLegacySurfaceInventoryEntry(
        surface=StreamLegacySurface.EVENT,
        classification=StreamLegacySurfaceClassification.MIGRATE_LATER,
        owner="event",
        removal_condition=(
            "Event streaming is projected through canonical observability."
        ),
    ),
)


def legacy_stream_surface_inventory() -> (
    tuple[StreamLegacySurfaceInventoryEntry, ...]
):
    return _LEGACY_STREAM_SURFACE_INVENTORY


def classify_legacy_stream_surface(
    surface: StreamLegacySurface,
) -> StreamLegacySurfaceInventoryEntry:
    assert isinstance(surface, StreamLegacySurface)
    for entry in _LEGACY_STREAM_SURFACE_INVENTORY:
        if entry.surface is surface:
            return entry
    raise StreamValidationError("unknown legacy stream surface")


def is_stream_terminal_kind(kind: StreamItemKind) -> bool:
    return stream_terminal_outcome_for_kind(kind) is not None


def is_tool_execution_terminal_kind(kind: StreamItemKind) -> bool:
    assert isinstance(kind, StreamItemKind)
    return kind in _TOOL_EXECUTION_TERMINAL_KINDS


def validate_stream_runtime_contract(
    contract: StreamRuntimeContract,
) -> StreamRuntimeContract:
    assert isinstance(contract, StreamRuntimeContract)
    return contract


def validate_tool_lifecycle_items(
    items: Iterable[CanonicalStreamItem],
    *,
    planned_tool_call_ids: Iterable[str] | None = None,
) -> tuple[CanonicalStreamItem, ...]:
    result, _, _ = _collect_tool_lifecycle_states(items, planned_tool_call_ids)
    return result


def assemble_tool_observations(
    items: Iterable[CanonicalStreamItem],
    *,
    planned_tool_call_ids: Iterable[str] | None = None,
) -> tuple[StreamToolObservation, ...]:
    _, states, observation_order = _collect_tool_lifecycle_states(
        items, planned_tool_call_ids
    )
    observations: list[StreamToolObservation] = []
    for tool_call_id in observation_order:
        state = states[tool_call_id]
        terminal_kind = state.terminal_kind
        assert terminal_kind is not None
        observations.append(
            StreamToolObservation(
                tool_call_id=tool_call_id,
                arguments="".join(state.argument_deltas),
                output="".join(state.output_deltas),
                terminal_kind=terminal_kind,
                terminal_data=state.terminal_data,
            )
        )
    return tuple(observations)


def _collect_tool_lifecycle_states(
    items: Iterable[CanonicalStreamItem],
    planned_tool_call_ids: Iterable[str] | None,
) -> tuple[
    tuple[CanonicalStreamItem, ...],
    dict[str, _ToolLifecycleState],
    tuple[str, ...],
]:
    result = tuple(items)
    planned_order = _validated_tool_call_id_order(planned_tool_call_ids)
    planned_ids = set(planned_order)
    order: list[str] = []
    states: dict[str, _ToolLifecycleState] = {}

    for item in result:
        if item.kind not in _TOOL_RELATED_KINDS:
            continue
        tool_call_id = item.correlation.tool_call_id
        assert tool_call_id is not None
        if planned_ids and tool_call_id not in planned_ids:
            raise StreamValidationError("unexpected tool call id")
        if tool_call_id not in states:
            states[tool_call_id] = _ToolLifecycleState()
            order.append(tool_call_id)
        _transition_tool_lifecycle_state(item, states[tool_call_id])

    for tool_call_id, state in states.items():
        _validate_complete_tool_lifecycle(tool_call_id, state)
    for tool_call_id in planned_order:
        if tool_call_id not in states:
            raise StreamValidationError("planned tool call missing")

    observation_order = planned_order or tuple(order)
    return result, states, observation_order


def _validated_tool_call_id_order(
    tool_call_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    if tool_call_ids is None:
        return ()
    order = tuple(tool_call_ids)
    seen: set[str] = set()
    for tool_call_id in order:
        assert isinstance(tool_call_id, str)
        assert tool_call_id.strip()
        assert tool_call_id not in seen
        seen.add(tool_call_id)
    return order


def _transition_tool_lifecycle_state(
    item: CanonicalStreamItem,
    state: _ToolLifecycleState,
) -> None:
    if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        if state.ready:
            raise StreamValidationError(
                "tool-call argument emitted after ready"
            )
        assert item.text_delta is not None
        state.argument_deltas.append(item.text_delta)
    elif item.kind is StreamItemKind.TOOL_CALL_READY:
        if state.ready:
            raise StreamValidationError("duplicate tool-call ready item")
        state.ready = True
    elif item.kind is StreamItemKind.TOOL_CALL_DONE:
        if state.call_done:
            raise StreamValidationError("duplicate tool-call done item")
        if not state.ready:
            raise StreamValidationError("tool-call done before ready")
        state.call_done = True
    elif item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
        if state.execution_started:
            raise StreamValidationError("duplicate tool execution start")
        if not state.call_done:
            raise StreamValidationError(
                "tool execution started before tool-call done"
            )
        state.execution_started = True
    elif item.kind in (
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
    ):
        _validate_tool_execution_live_item(state)
        if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
            assert item.text_delta is not None
            state.output_deltas.append(item.text_delta)
    elif item.kind in _TOOL_EXECUTION_TERMINAL_KINDS:
        if state.terminal_kind is not None:
            raise StreamValidationError("duplicate tool execution terminal")
        if not state.execution_started:
            raise StreamValidationError("tool execution terminal before start")
        state.terminal_kind = item.kind
        state.terminal_data = item.data


def _validate_tool_execution_live_item(
    state: _ToolLifecycleState,
) -> None:
    if state.terminal_kind is not None:
        raise StreamValidationError(
            "tool execution item emitted after terminal item"
        )
    if not state.execution_started:
        raise StreamValidationError("tool execution item before start")


def _validate_complete_tool_lifecycle(
    tool_call_id: str,
    state: _ToolLifecycleState,
) -> None:
    if not state.ready:
        raise StreamValidationError(f"tool call {tool_call_id} missing ready")
    if not state.call_done:
        raise StreamValidationError(f"tool call {tool_call_id} missing done")
    if not state.execution_started:
        raise StreamValidationError(
            f"tool call {tool_call_id} missing execution start"
        )
    if state.terminal_kind is None:
        raise StreamValidationError(
            f"tool call {tool_call_id} missing execution terminal"
        )


def validate_canonical_stream_items(
    items: Iterable[CanonicalStreamItem],
) -> tuple[CanonicalStreamItem, ...]:
    result = tuple(items)
    if not result:
        raise StreamValidationError("stream must contain at least one item")

    session_id = result[0].stream_session_id
    run_id = result[0].run_id
    turn_id = result[0].turn_id
    last_sequence: int | None = None
    terminal: StreamTerminalOutcome | None = None
    closed = False
    answer_done = False
    reasoning_done = False
    usage_completed = False
    tool_call_done: set[str] = set()
    tool_execution_terminal: set[str] = set()

    for item in result:
        _validate_sequence_identity(item, session_id, run_id, turn_id)
        last_sequence = _validate_sequence_order(item, last_sequence)
        _validate_parent_sequence(item)

        if closed:
            raise StreamValidationError("stream item emitted after closed")
        outcome = stream_terminal_outcome_for_kind(item.kind)
        if terminal is not None:
            if item.kind is StreamItemKind.STREAM_CLOSED:
                closed = True
                continue
            if outcome is not None:
                raise StreamValidationError("duplicate stream terminal item")
            else:
                raise StreamValidationError(
                    "semantic stream item emitted after terminal outcome"
                )

        if item.kind is StreamItemKind.STREAM_CLOSED:
            raise StreamValidationError("stream closed before terminal")

        answer_done = _validate_answer_boundary(item, answer_done)
        reasoning_done = _validate_reasoning_boundary(item, reasoning_done)
        _validate_tool_call_boundary(item, tool_call_done)
        _validate_tool_execution_boundary(item, tool_execution_terminal)

        if item.kind is StreamItemKind.USAGE_COMPLETED:
            if usage_completed:
                raise StreamValidationError("duplicate completed usage item")
            usage_completed = True
        elif usage_completed and item.channel is StreamChannel.USAGE:
            raise StreamValidationError("usage item emitted after final usage")

        if outcome is not None:
            if (
                outcome is StreamTerminalOutcome.COMPLETED
                and not usage_completed
                and item.usage is None
            ):
                raise StreamValidationError(
                    "completed stream missing final usage"
                )
            terminal = outcome

    if terminal is None:
        raise StreamValidationError("stream missing terminal outcome")
    return result


def _validate_sequence_identity(
    item: CanonicalStreamItem,
    session_id: str,
    run_id: str,
    turn_id: str,
) -> None:
    if (
        item.stream_session_id != session_id
        or item.run_id != run_id
        or item.turn_id != turn_id
    ):
        raise StreamValidationError("stream identity changed")


def _validate_sequence_order(
    item: CanonicalStreamItem,
    last_sequence: int | None,
) -> int:
    if last_sequence is not None and item.sequence <= last_sequence:
        raise StreamValidationError("stream sequence must increase")
    return item.sequence


def _validate_parent_sequence(item: CanonicalStreamItem) -> None:
    parent_sequence = item.correlation.parent_sequence
    if parent_sequence is not None and parent_sequence >= item.sequence:
        raise StreamValidationError("parent sequence must precede item")


def _validate_answer_boundary(
    item: CanonicalStreamItem,
    answer_done: bool,
) -> bool:
    if item.channel is StreamChannel.ANSWER and answer_done:
        raise StreamValidationError("answer item emitted after answer done")
    return answer_done or item.kind is StreamItemKind.ANSWER_DONE


def _validate_reasoning_boundary(
    item: CanonicalStreamItem,
    reasoning_done: bool,
) -> bool:
    if item.channel is StreamChannel.REASONING and reasoning_done:
        raise StreamValidationError(
            "reasoning item emitted after reasoning done"
        )
    return reasoning_done or item.kind is StreamItemKind.REASONING_DONE


def _validate_tool_call_boundary(
    item: CanonicalStreamItem,
    done_tool_call_ids: set[str],
) -> None:
    if item.kind not in _TOOL_CALL_KINDS:
        return
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    if tool_call_id in done_tool_call_ids:
        raise StreamValidationError(
            "tool-call item emitted after tool-call done"
        )
    if item.kind is StreamItemKind.TOOL_CALL_DONE:
        done_tool_call_ids.add(tool_call_id)


def _validate_tool_execution_boundary(
    item: CanonicalStreamItem,
    terminal_tool_call_ids: set[str],
) -> None:
    if item.channel is not StreamChannel.TOOL_EXECUTION:
        return
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    if tool_call_id in terminal_tool_call_ids:
        raise StreamValidationError(
            "tool execution item emitted after terminal item"
        )
    if item.kind in _TOOL_EXECUTION_TERMINAL_KINDS:
        terminal_tool_call_ids.add(tool_call_id)


class CanonicalStreamAccumulator:
    def __init__(self) -> None:
        self._items: list[CanonicalStreamItem] = []
        self._answer_text: list[str] = []
        self._reasoning_text: list[str] = []
        self._tool_call_arguments: dict[str, list[str]] = {}
        self._tool_execution_outputs: dict[str, list[str]] = {}
        self._diagnostics: list[CanonicalStreamItem] = []
        self._flow_items: list[CanonicalStreamItem] = []
        self._usage_items: list[CanonicalStreamItem] = []
        self._control_items: list[CanonicalStreamItem] = []
        self._final_usage: LooseJsonValue | None = None
        self._session_id: str | None = None
        self._run_id: str | None = None
        self._turn_id: str | None = None
        self._last_sequence: int | None = None
        self._terminal_outcome: StreamTerminalOutcome | None = None
        self._closed = False
        self._answer_done = False
        self._reasoning_done = False
        self._usage_completed = False
        self._tool_call_done: set[str] = set()
        self._tool_execution_terminal: set[str] = set()

    @property
    def items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._items)

    @property
    def answer_text(self) -> str:
        return "".join(self._answer_text)

    @property
    def reasoning_text(self) -> str:
        return "".join(self._reasoning_text)

    @property
    def tool_call_arguments(self) -> dict[str, str]:
        return {
            tool_call_id: "".join(deltas)
            for tool_call_id, deltas in self._tool_call_arguments.items()
        }

    @property
    def tool_execution_outputs(self) -> dict[str, str]:
        return {
            tool_call_id: "".join(deltas)
            for tool_call_id, deltas in self._tool_execution_outputs.items()
        }

    @property
    def diagnostics(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._diagnostics)

    @property
    def flow_items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._flow_items)

    @property
    def usage_items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._usage_items)

    @property
    def control_items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._control_items)

    @property
    def final_usage(self) -> LooseJsonValue | None:
        return self._final_usage

    @property
    def terminal_outcome(self) -> StreamTerminalOutcome | None:
        return self._terminal_outcome

    @property
    def closed(self) -> bool:
        return self._closed

    def add(self, item: CanonicalStreamItem) -> None:
        assert isinstance(item, CanonicalStreamItem)
        self._validate_next(item)
        self._items.append(item)
        self._accumulate(item)

    def add_many(
        self,
        items: Iterable[CanonicalStreamItem],
    ) -> "CanonicalStreamAccumulator":
        for item in items:
            self.add(item)
        return self

    def validate_complete(self) -> tuple[CanonicalStreamItem, ...]:
        return validate_canonical_stream_items(self._items)

    def _validate_next(self, item: CanonicalStreamItem) -> None:
        if self._session_id is None:
            self._session_id = item.stream_session_id
            self._run_id = item.run_id
            self._turn_id = item.turn_id
        else:
            assert self._run_id is not None
            assert self._turn_id is not None
            _validate_sequence_identity(
                item,
                self._session_id,
                self._run_id,
                self._turn_id,
            )

        self._last_sequence = _validate_sequence_order(
            item, self._last_sequence
        )
        _validate_parent_sequence(item)

        if self._closed:
            raise StreamValidationError("stream item emitted after closed")

        outcome = stream_terminal_outcome_for_kind(item.kind)
        if self._terminal_outcome is not None:
            if item.kind is StreamItemKind.STREAM_CLOSED:
                return
            if outcome is not None:
                raise StreamValidationError("duplicate stream terminal item")
            raise StreamValidationError(
                "semantic stream item emitted after terminal outcome"
            )

        if item.kind is StreamItemKind.STREAM_CLOSED:
            raise StreamValidationError("stream closed before terminal")

        self._answer_done = _validate_answer_boundary(item, self._answer_done)
        self._reasoning_done = _validate_reasoning_boundary(
            item, self._reasoning_done
        )
        _validate_tool_call_boundary(item, self._tool_call_done)
        _validate_tool_execution_boundary(item, self._tool_execution_terminal)
        self._validate_usage(item)

        if outcome is not None:
            if (
                outcome is StreamTerminalOutcome.COMPLETED
                and not self._usage_completed
                and item.usage is None
            ):
                raise StreamValidationError(
                    "completed stream missing final usage"
                )
            self._terminal_outcome = outcome

    def _validate_usage(self, item: CanonicalStreamItem) -> None:
        if item.kind is StreamItemKind.USAGE_COMPLETED:
            if self._usage_completed:
                raise StreamValidationError("duplicate completed usage item")
            self._usage_completed = True
        elif self._usage_completed and item.channel is StreamChannel.USAGE:
            raise StreamValidationError("usage item emitted after final usage")

    def _accumulate(self, item: CanonicalStreamItem) -> None:
        if item.kind is StreamItemKind.ANSWER_DELTA:
            assert item.text_delta is not None
            self._answer_text.append(item.text_delta)
        elif item.kind is StreamItemKind.REASONING_DELTA:
            assert item.text_delta is not None
            self._reasoning_text.append(item.text_delta)
        elif item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            assert item.text_delta is not None
            tool_call_id = item.correlation.tool_call_id
            assert tool_call_id is not None
            self._tool_call_arguments.setdefault(tool_call_id, []).append(
                item.text_delta
            )
        elif item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
            assert item.text_delta is not None
            tool_call_id = item.correlation.tool_call_id
            assert tool_call_id is not None
            self._tool_execution_outputs.setdefault(tool_call_id, []).append(
                item.text_delta
            )

        if item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
            self._diagnostics.append(item)
        if item.channel is StreamChannel.FLOW:
            self._flow_items.append(item)
        if item.channel is StreamChannel.USAGE:
            self._usage_items.append(item)
            self._final_usage = item.usage
        if (
            item.kind is StreamItemKind.STREAM_COMPLETED
            and item.usage is not None
        ):
            self._final_usage = item.usage
        if item.channel is StreamChannel.CONTROL:
            self._control_items.append(item)
        if item.kind is StreamItemKind.STREAM_CLOSED:
            self._closed = True


def accumulate_canonical_stream_items(
    items: Iterable[CanonicalStreamItem],
) -> CanonicalStreamAccumulator:
    accumulator = CanonicalStreamAccumulator().add_many(items)
    accumulator.validate_complete()
    return accumulator


class TextGenerationStream(AsyncIterator[Token | TokenDetail | str], ABC):
    _generator: AsyncIterator[Token | TokenDetail | str] | None = None

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Token | TokenDetail | str]:
        raise NotImplementedError()

    @abstractmethod
    async def __anext__(self) -> Token | TokenDetail | str:
        raise NotImplementedError()

    def __aiter__(self) -> AsyncIterator[Token | TokenDetail | str]:
        assert self._generator
        return self


class TextGenerationSingleStream(TextGenerationStream):
    _content: str | Token | TokenDetail
    _consumed: bool = False
    _provider_family: str | None = None
    _usage: object | None = None

    def __init__(
        self,
        content: str | Token | TokenDetail,
        *,
        provider_family: ProviderFamily | str | None = None,
        usage: object | None = None,
    ) -> None:
        self._content = content
        self._provider_family = provider_family_value(provider_family)
        self._usage = usage

    @property
    def content(self) -> str | Token | TokenDetail:
        return self._content

    @property
    def provider_family(self) -> str | None:
        return self._provider_family

    @property
    def usage(self) -> object | None:
        return self._usage

    @property
    def canonical_items(self) -> tuple[CanonicalStreamItem, ...]:
        text = token_text(self._content)
        usage = self._usage if self._usage is not None else {}
        return (
            CanonicalStreamItem(
                stream_session_id="single-stream",
                run_id="single-run",
                turn_id="single-turn",
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                provider_family=self._provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id="single-stream",
                run_id="single-run",
                turn_id="single-turn",
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
                provider_family=self._provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id="single-stream",
                run_id="single-run",
                turn_id="single-turn",
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                provider_family=self._provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id="single-stream",
                run_id="single-run",
                turn_id="single-turn",
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage=cast(Any, usage),
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
                provider_family=self._provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id="single-stream",
                run_id="single-run",
                turn_id="single-turn",
                sequence=4,
                kind=StreamItemKind.STREAM_CLOSED,
                channel=StreamChannel.CONTROL,
                provider_family=self._provider_family,
            ),
        )

    @property
    def accumulator(self) -> CanonicalStreamAccumulator:
        return accumulate_canonical_stream_items(self.canonical_items)

    @property
    def final_text(self) -> str:
        return self.accumulator.answer_text

    async def to_str(self) -> str:
        return self.final_text

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[str | Token | TokenDetail]:
        self._consumed = False
        return self

    def __aiter__(self) -> AsyncIterator[str | Token | TokenDetail]:
        self._consumed = False
        return self

    async def __anext__(self) -> str | Token | TokenDetail:
        if self._consumed:
            raise StopAsyncIteration
        self._consumed = True
        return self._content


def token_text(token: Token | TokenDetail | str) -> str:
    if isinstance(token, str):
        return token
    assert isinstance(token, Token)
    return token.token


def canonical_item_from_token(
    token: Token | TokenDetail | str,
    sequence: int,
    *,
    stream_session_id: str = "legacy-stream",
    run_id: str = "legacy-run",
    turn_id: str = "legacy-turn",
) -> CanonicalStreamItem:
    assert isinstance(sequence, int), "sequence must be an integer"
    assert sequence >= 0, "sequence must not be negative"
    text = token_text(token)
    if isinstance(token, ReasoningToken):
        return CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=sequence,
            kind=StreamItemKind.REASONING_DELTA,
            channel=StreamChannel.REASONING,
            text_delta=text,
            visibility=StreamVisibility.PRIVATE,
        )
    if isinstance(token, ToolCallToken):
        tool_call_id = str(token.call.id) if token.call else "legacy-tool-call"
        return CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=sequence,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
            text_delta=text,
        )
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        text_delta=text,
    )
