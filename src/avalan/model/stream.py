from ..entities import (
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallToken,
)
from ..observability import observability_key_sample
from ..types import LooseJsonValue
from .provider import ProviderFamily, provider_family_value

from abc import ABC, abstractmethod
from asyncio import CancelledError
from collections.abc import AsyncIterable, Awaitable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from inspect import isawaitable
from json import JSONDecodeError, loads
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


class StreamProducerBackend(StrEnum):
    HOSTED = "hosted"
    LOCAL = "local"


class StreamValidationError(ValueError):
    pass


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamProviderCapabilities:
    backend: StreamProducerBackend
    provider_family: ProviderFamily | str | None = None
    supports_reasoning: bool = False
    supports_tool_calls: bool = False
    supports_usage: bool = False
    supports_terminal_events: bool = False
    supports_cancellation: bool = False
    max_queue_depth: int | None = None
    max_item_bytes: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.backend, StreamProducerBackend)
        for field_name, value in (
            ("supports_reasoning", self.supports_reasoning),
            ("supports_tool_calls", self.supports_tool_calls),
            ("supports_usage", self.supports_usage),
            ("supports_terminal_events", self.supports_terminal_events),
            ("supports_cancellation", self.supports_cancellation),
        ):
            assert isinstance(value, bool), f"{field_name} must be a boolean"
        if self.provider_family is not None:
            provider_family_value(self.provider_family)
        if self.max_queue_depth is not None:
            _assert_positive_int(self.max_queue_depth, "max_queue_depth")
        if self.max_item_bytes is not None:
            _assert_positive_int(self.max_item_bytes, "max_item_bytes")

    @property
    def normalized_provider_family(self) -> str | None:
        return provider_family_value(self.provider_family)

    def to_metadata(self) -> dict[str, LooseJsonValue]:
        result: dict[str, LooseJsonValue] = {
            "backend": self.backend.value,
            "supports_reasoning": self.supports_reasoning,
            "supports_tool_calls": self.supports_tool_calls,
            "supports_usage": self.supports_usage,
            "supports_terminal_events": self.supports_terminal_events,
            "supports_cancellation": self.supports_cancellation,
        }
        if self.normalized_provider_family is not None:
            result["provider_family"] = self.normalized_provider_family
        if self.max_queue_depth is not None:
            result["max_queue_depth"] = self.max_queue_depth
        if self.max_item_bytes is not None:
            result["max_item_bytes"] = self.max_item_bytes
        return result


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
class StreamProviderEvent:
    kind: StreamItemKind
    text_delta: str | None = None
    data: LooseJsonValue | None = None
    usage: LooseJsonValue | None = None
    correlation: StreamItemCorrelation = field(
        default_factory=StreamItemCorrelation
    )
    visibility: StreamVisibility = StreamVisibility.PUBLIC
    metadata: dict[str, LooseJsonValue] = field(default_factory=dict)
    provider_payload: LooseJsonValue | None = None
    provider_event_type: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.kind, StreamItemKind)
        assert self.kind is not StreamItemKind.STREAM_STARTED
        assert self.kind is not StreamItemKind.STREAM_CLOSED
        assert isinstance(self.correlation, StreamItemCorrelation)
        assert isinstance(self.visibility, StreamVisibility)
        assert isinstance(self.metadata, dict), "metadata must be a dict"
        if self.text_delta is not None:
            assert isinstance(self.text_delta, str)
        if self.provider_event_type is not None:
            _assert_non_empty_string(
                self.provider_event_type, "provider_event_type"
            )


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
class StreamConsumerProjection:
    stream_session_id: str
    run_id: str
    turn_id: str
    sequence: int
    kind: StreamItemKind
    channel: StreamChannel
    correlation: StreamItemCorrelation
    text_delta: str | None = None
    data: LooseJsonValue | None = None
    usage: LooseJsonValue | None = None
    terminal_outcome: StreamTerminalOutcome | None = None
    visibility: StreamVisibility = StreamVisibility.PUBLIC
    metadata: dict[str, LooseJsonValue] = field(default_factory=dict)
    provider_family: str | None = None
    provider_event_type: str | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("stream_session_id", self.stream_session_id),
            ("run_id", self.run_id),
            ("turn_id", self.turn_id),
        ):
            _assert_non_empty_string(value, field_name)
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
        if self.terminal_outcome is not None:
            assert isinstance(self.terminal_outcome, StreamTerminalOutcome)
            assert self.terminal_outcome is stream_terminal_outcome_for_kind(
                self.kind
            )
        if self.provider_family is not None:
            _assert_non_empty_string(self.provider_family, "provider_family")
        if self.provider_event_type is not None:
            _assert_non_empty_string(
                self.provider_event_type, "provider_event_type"
            )
        self._validate_kind_payload()

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

    @classmethod
    def from_item(
        cls,
        item: CanonicalStreamItem,
    ) -> "StreamConsumerProjection":
        assert isinstance(item, CanonicalStreamItem)
        return cls(
            stream_session_id=item.stream_session_id,
            run_id=item.run_id,
            turn_id=item.turn_id,
            sequence=item.sequence,
            kind=item.kind,
            channel=item.channel,
            correlation=item.correlation,
            text_delta=item.text_delta,
            data=item.data,
            usage=item.usage,
            terminal_outcome=item.terminal_outcome,
            visibility=item.visibility,
            metadata=dict(item.metadata),
            provider_family=item.provider_family,
            provider_event_type=item.provider_event_type,
        )

    @property
    def tool_call_id(self) -> str | None:
        return self.correlation.tool_call_id

    @property
    def is_stream_terminal(self) -> bool:
        return is_stream_terminal_kind(self.kind)


def project_canonical_stream_item(
    item: CanonicalStreamItem,
) -> StreamConsumerProjection:
    return StreamConsumerProjection.from_item(item)


def canonical_item_from_consumer_projection(
    projection: StreamConsumerProjection,
) -> CanonicalStreamItem:
    assert isinstance(projection, StreamConsumerProjection)
    return CanonicalStreamItem(
        stream_session_id=projection.stream_session_id,
        run_id=projection.run_id,
        turn_id=projection.turn_id,
        sequence=projection.sequence,
        kind=projection.kind,
        channel=projection.channel,
        correlation=projection.correlation,
        text_delta=projection.text_delta,
        data=projection.data,
        usage=projection.usage,
        terminal_outcome=projection.terminal_outcome,
        visibility=projection.visibility,
        metadata=dict(projection.metadata),
        provider_family=projection.provider_family,
        provider_event_type=projection.provider_event_type,
    )


def stream_consumer_projection_from_token(
    token: CanonicalStreamItem | StreamConsumerProjection | Token | str,
    sequence: int,
    *,
    stream_session_id: str = "legacy-stream",
    run_id: str = "legacy-run",
    turn_id: str = "legacy-turn",
) -> StreamConsumerProjection:
    assert isinstance(
        token, (CanonicalStreamItem, StreamConsumerProjection, Token, str)
    )
    if isinstance(token, StreamConsumerProjection):
        return token
    if isinstance(token, CanonicalStreamItem):
        return project_canonical_stream_item(token)
    return project_canonical_stream_item(
        canonical_item_from_token(
            token,
            sequence,
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
        )
    )


def stream_projection_text_delta(
    item: CanonicalStreamItem | StreamConsumerProjection,
) -> str | None:
    assert isinstance(item, (CanonicalStreamItem, StreamConsumerProjection))
    projection = (
        project_canonical_stream_item(item)
        if isinstance(item, CanonicalStreamItem)
        else item
    )
    if projection.kind in _TEXT_DELTA_KINDS:
        return projection.text_delta or ""
    return None


def stream_projection_is_reasoning(
    item: CanonicalStreamItem | StreamConsumerProjection,
) -> bool:
    assert isinstance(item, (CanonicalStreamItem, StreamConsumerProjection))
    projection = (
        project_canonical_stream_item(item)
        if isinstance(item, CanonicalStreamItem)
        else item
    )
    return projection.kind is StreamItemKind.REASONING_DELTA


def stream_projection_is_tool_call(
    item: CanonicalStreamItem | StreamConsumerProjection,
) -> bool:
    assert isinstance(item, (CanonicalStreamItem, StreamConsumerProjection))
    projection = (
        project_canonical_stream_item(item)
        if isinstance(item, CanonicalStreamItem)
        else item
    )
    return projection.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA


async def iter_stream_consumer_projections(
    items: AsyncIterable[CanonicalStreamItem],
    *,
    validate_order: bool = True,
) -> AsyncIterator[StreamConsumerProjection]:
    assert isinstance(items, AsyncIterable)
    assert isinstance(validate_order, bool)
    accumulator = CanonicalStreamAccumulator() if validate_order else None
    iterator = items.__aiter__()
    try:
        async for item in iterator:
            if accumulator is not None:
                accumulator.add(item)
            yield project_canonical_stream_item(item)
        if accumulator is not None:
            accumulator.validate_complete()
    finally:
        if iterator is not items:
            await _close_async_iterables(iterator, items)
        else:
            await _close_async_iterable(iterator)


def stream_observability_payload(
    item: CanonicalStreamItem,
) -> dict[str, LooseJsonValue]:
    assert isinstance(item, CanonicalStreamItem)
    payload: dict[str, LooseJsonValue] = {
        "stream_session_id": item.stream_session_id,
        "run_id": item.run_id,
        "turn_id": item.turn_id,
        "sequence": item.sequence,
        "kind": item.kind.value,
        "channel": item.channel.value,
        "visibility": item.visibility.value,
    }
    correlation = item.correlation.to_trace_dict()
    if correlation:
        payload["correlation"] = correlation
    if item.terminal_outcome is not None:
        payload["terminal_outcome"] = item.terminal_outcome.value
    if item.usage is not None:
        payload["usage"] = item.usage
    summary = _stream_observability_summary(item)
    if summary:
        payload["summary"] = summary
    if item.provider_family is not None:
        payload["provider_family"] = item.provider_family
    if item.provider_event_type is not None:
        payload["provider_event_type"] = item.provider_event_type
    return payload


def _stream_observability_summary(
    item: CanonicalStreamItem,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    if item.text_delta is not None:
        summary["text_delta_length"] = len(item.text_delta)
    if isinstance(item.data, dict):
        data_keys, data_keys_truncated = observability_key_sample(item.data)
        summary["data_keys"] = data_keys
        if data_keys_truncated:
            summary["data_key_count"] = len(item.data)
            summary["data_keys_truncated"] = True
    elif item.data is not None:
        summary["data_type"] = type(item.data).__name__
    if item.metadata:
        metadata_keys, metadata_keys_truncated = observability_key_sample(
            item.metadata
        )
        summary["metadata_keys"] = metadata_keys
        if metadata_keys_truncated:
            summary["metadata_key_count"] = len(item.metadata)
            summary["metadata_keys_truncated"] = True
    if item.provider_payload is not None:
        summary["has_provider_payload"] = True
    return summary


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

    for index, item in enumerate(result):
        _validate_stream_start(item, index == 0)
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
        else:
            _validate_post_final_usage_item(item, usage_completed)

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


def _validate_stream_start(
    item: CanonicalStreamItem,
    is_first: bool,
) -> None:
    if is_first:
        if item.kind is not StreamItemKind.STREAM_STARTED:
            raise StreamValidationError(
                "stream must start with stream.started"
            )
    elif item.kind is StreamItemKind.STREAM_STARTED:
        raise StreamValidationError("stream started more than once")


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


def _validate_post_final_usage_item(
    item: CanonicalStreamItem,
    usage_completed: bool,
) -> None:
    if not usage_completed:
        return
    if item.channel is StreamChannel.USAGE:
        raise StreamValidationError("usage item emitted after final usage")
    if (
        item.kind is not StreamItemKind.STREAM_DIAGNOSTIC
        and not item.is_stream_terminal
        and item.kind is not StreamItemKind.STREAM_CLOSED
    ):
        raise StreamValidationError(
            "semantic stream item emitted after final usage"
        )


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
        is_first = self._session_id is None
        _validate_stream_start(item, is_first)

        if is_first:
            self._session_id = item.stream_session_id
            self._run_id = item.run_id
            self._turn_id = item.turn_id
        else:
            assert self._session_id is not None
            assert self._run_id is not None
            assert self._turn_id is not None
            _validate_sequence_identity(
                item,
                self._session_id,
                self._run_id,
                self._turn_id,
            )

        last_sequence = _validate_sequence_order(item, self._last_sequence)
        _validate_parent_sequence(item)

        if self._closed:
            raise StreamValidationError("stream item emitted after closed")

        outcome = stream_terminal_outcome_for_kind(item.kind)
        if self._terminal_outcome is not None:
            if item.kind is StreamItemKind.STREAM_CLOSED:
                self._last_sequence = last_sequence
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
        self._last_sequence = last_sequence

    def _validate_usage(self, item: CanonicalStreamItem) -> None:
        if item.kind is StreamItemKind.USAGE_COMPLETED:
            if self._usage_completed:
                raise StreamValidationError("duplicate completed usage item")
            self._usage_completed = True
        else:
            _validate_post_final_usage_item(item, self._usage_completed)

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


async def normalize_provider_stream(
    events: AsyncIterable[StreamProviderEvent],
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    provider_family: ProviderFamily | str | None = None,
    capabilities: StreamProviderCapabilities | None = None,
    close_after_terminal: bool = True,
) -> AsyncIterator[CanonicalStreamItem]:
    assert isinstance(events, AsyncIterable)
    _assert_non_empty_string(stream_session_id, "stream_session_id")
    _assert_non_empty_string(run_id, "run_id")
    _assert_non_empty_string(turn_id, "turn_id")
    if provider_family is not None:
        provider_family = provider_family_value(provider_family)
    if capabilities is not None:
        assert isinstance(capabilities, StreamProviderCapabilities)
        provider_family = (
            provider_family or capabilities.normalized_provider_family
        )
    assert isinstance(close_after_terminal, bool)

    normalizer = _ProviderStreamNormalizer(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        provider_family=provider_family,
        capabilities=capabilities,
        close_after_terminal=close_after_terminal,
    )
    event_iterator = events.__aiter__()
    provider_closed = False

    async def close_provider_stream() -> None:
        nonlocal provider_closed
        if provider_closed:
            return
        provider_closed = True
        await _close_async_iterable(event_iterator)

    try:
        yield normalizer.started()
        while True:
            try:
                event = await event_iterator.__anext__()
            except StopAsyncIteration:
                break
            items = normalizer.map_event(event)
            should_stop = any(item.is_stream_terminal for item in items)
            for item in items:
                yield item
            if should_stop:
                return
    except CancelledError:
        await close_provider_stream()
        for item in normalizer.cancelled():
            yield item
    except StreamValidationError as exc:
        await close_provider_stream()
        for item in normalizer.errored(exc):
            yield item
    except Exception as exc:
        await close_provider_stream()
        for item in normalizer.errored(exc):
            yield item
    else:
        for item in normalizer.completed():
            yield item
    finally:
        await close_provider_stream()


async def normalize_local_stream(
    tokens: AsyncIterable[Token | TokenDetail | str],
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    provider_family: ProviderFamily | str | None = ProviderFamily.LOCAL,
    capabilities: StreamProviderCapabilities | None = None,
    close_after_terminal: bool = True,
) -> AsyncIterator[CanonicalStreamItem]:
    assert isinstance(tokens, AsyncIterable)
    _assert_non_empty_string(stream_session_id, "stream_session_id")
    _assert_non_empty_string(run_id, "run_id")
    _assert_non_empty_string(turn_id, "turn_id")
    if capabilities is None:
        capabilities = StreamProviderCapabilities(
            backend=StreamProducerBackend.LOCAL,
            provider_family=provider_family,
            supports_reasoning=True,
            supports_tool_calls=True,
            supports_cancellation=True,
            max_queue_depth=StreamPerformanceBudget().max_queue_depth,
        )
    else:
        assert isinstance(capabilities, StreamProviderCapabilities)
        assert capabilities.backend is StreamProducerBackend.LOCAL

    parser = _LocalTextStreamParser()
    pending_tool_call: ToolCall | None = None
    token_iterator = tokens.__aiter__()
    tokens_exhausted = False
    tokens_closed = False

    async def close_tokens() -> None:
        nonlocal tokens_closed
        if tokens_closed or tokens_exhausted:
            return
        tokens_closed = True
        await _close_async_iterable(token_iterator)

    async def events() -> AsyncIterator[StreamProviderEvent]:
        nonlocal pending_tool_call, tokens_exhausted
        try:
            while True:
                try:
                    token = await token_iterator.__anext__()
                except StopAsyncIteration:
                    tokens_exhausted = True
                    break

                if isinstance(token, str):
                    for event in _legacy_tool_call_boundary_events(
                        pending_tool_call
                    ):
                        yield event
                    pending_tool_call = None
                    for event in parser.push(token):
                        yield event
                else:
                    if (
                        isinstance(token, ToolCallToken)
                        and pending_tool_call is not None
                        and token.call is not None
                        and _legacy_tool_call_id(token.call)
                        != _legacy_tool_call_id(pending_tool_call)
                    ):
                        for event in _legacy_tool_call_boundary_events(
                            pending_tool_call
                        ):
                            yield event
                        pending_tool_call = None
                    elif not isinstance(token, ToolCallToken):
                        for event in _legacy_tool_call_boundary_events(
                            pending_tool_call
                        ):
                            yield event
                        pending_tool_call = None

                    item = canonical_item_from_token(token, 0)
                    yield StreamProviderEvent(
                        kind=item.kind,
                        text_delta=item.text_delta,
                        correlation=item.correlation,
                        visibility=item.visibility,
                        metadata=item.metadata,
                    )
                    if (
                        isinstance(token, ToolCallToken)
                        and token.call is not None
                    ):
                        pending_tool_call = token.call
            for event in parser.flush():
                yield event
            for event in _legacy_tool_call_boundary_events(pending_tool_call):
                yield event
        finally:
            await close_tokens()

    provider_stream = normalize_provider_stream(
        events(),
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        provider_family=provider_family,
        capabilities=capabilities,
        close_after_terminal=close_after_terminal,
    )
    try:
        async for item in provider_stream:
            yield item
    except (CancelledError, GeneratorExit):
        await close_tokens()
        await cast(Any, provider_stream).aclose()
        raise


def _legacy_tool_call_boundary_events(
    call: ToolCall | None,
) -> tuple[StreamProviderEvent, ...]:
    if call is None:
        return ()
    call_id = _legacy_tool_call_id(call)
    correlation = StreamItemCorrelation(tool_call_id=call_id)
    return (
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_READY,
            correlation=correlation,
            data={
                "name": call.name,
                "arguments": call.arguments,
            },
        ),
        StreamProviderEvent(
            kind=StreamItemKind.TOOL_CALL_DONE,
            correlation=correlation,
        ),
    )


def _legacy_tool_call_id(call: ToolCall) -> str:
    return str(call.id) if call.id is not None else "legacy-tool-call"


@dataclass(slots=True)
class _LocalTextStreamParser:
    _reasoning_start_tag: str = "<think>"
    _reasoning_end_tag: str = "</think>"
    _tool_start_tag: str = "<tool_call"
    _tool_end_tag: str = "</tool_call>"
    _reasoning_buffer: str = ""
    _reasoning_active: bool = False
    _tool_buffer: str = ""
    _tool_state: str = "outside"
    _tool_call_id: str | None = None
    _tool_call_index: int = 0
    _tool_name: str | None = None
    _tool_argument_deltas: list[str] = field(default_factory=list)

    def push(self, token: str) -> tuple[StreamProviderEvent, ...]:
        assert isinstance(token, str)
        events: list[StreamProviderEvent] = []
        for event in self._push_reasoning(token):
            if event.kind is StreamItemKind.ANSWER_DELTA:
                assert event.text_delta is not None
                events.extend(self._push_tool(event.text_delta))
            else:
                events.append(event)
        return tuple(events)

    def flush(self) -> tuple[StreamProviderEvent, ...]:
        events: list[StreamProviderEvent] = []
        if self._reasoning_buffer:
            text = self._reasoning_buffer
            self._reasoning_buffer = ""
            if self._reasoning_active:
                events.append(self._reasoning_delta(text))
            else:
                events.extend(self._push_tool(text))
        if self._reasoning_active:
            self._reasoning_active = False
            events.append(
                StreamProviderEvent(kind=StreamItemKind.REASONING_DONE)
            )
        events.extend(self._flush_tool())
        return tuple(events)

    def _push_reasoning(self, token: str) -> tuple[StreamProviderEvent, ...]:
        self._reasoning_buffer += token
        events: list[StreamProviderEvent] = []
        while self._reasoning_buffer:
            if self._reasoning_active:
                end_index = self._reasoning_buffer.find(
                    self._reasoning_end_tag
                )
                if end_index != -1:
                    self._append_reasoning_delta(
                        events, self._reasoning_buffer[:end_index]
                    )
                    self._reasoning_buffer = self._reasoning_buffer[
                        end_index + len(self._reasoning_end_tag) :
                    ]
                    self._reasoning_active = False
                    events.append(
                        StreamProviderEvent(kind=StreamItemKind.REASONING_DONE)
                    )
                    continue
                flush_length = self._flushable_prefix_length(
                    self._reasoning_buffer,
                    self._reasoning_end_tag,
                )
                if not flush_length:
                    break
                self._append_reasoning_delta(
                    events, self._reasoning_buffer[:flush_length]
                )
                self._reasoning_buffer = self._reasoning_buffer[flush_length:]
                continue

            start_index = self._reasoning_buffer.find(
                self._reasoning_start_tag
            )
            if start_index != -1:
                events.extend(
                    self._push_tool(self._reasoning_buffer[:start_index])
                )
                self._reasoning_buffer = self._reasoning_buffer[
                    start_index + len(self._reasoning_start_tag) :
                ]
                self._reasoning_active = True
                continue
            flush_length = self._flushable_prefix_length(
                self._reasoning_buffer,
                self._reasoning_start_tag,
            )
            if not flush_length:
                break
            events.extend(
                self._push_tool(self._reasoning_buffer[:flush_length])
            )
            self._reasoning_buffer = self._reasoning_buffer[flush_length:]
        return tuple(events)

    def _push_tool(self, text: str) -> tuple[StreamProviderEvent, ...]:
        if not text:
            return ()
        self._tool_buffer += text
        events: list[StreamProviderEvent] = []
        while self._tool_buffer:
            if self._tool_state == "outside":
                start_index = self._tool_start_index()
                if start_index is not None:
                    self._append_answer_delta(
                        events, self._tool_buffer[:start_index]
                    )
                    self._tool_buffer = self._tool_buffer[start_index:]
                    self._tool_state = "opening"
                    self._tool_call_id = self._next_tool_call_id()
                    continue
                flush_length = self._tool_flushable_prefix_length()
                if not flush_length:
                    break
                self._append_answer_delta(
                    events, self._tool_buffer[:flush_length]
                )
                self._tool_buffer = self._tool_buffer[flush_length:]
                continue

            if self._tool_state == "opening":
                tag_end = self._tool_buffer.find(">")
                if tag_end == -1:
                    break
                opening_tag = self._tool_buffer[: tag_end + 1]
                self._tool_name = self._tool_name_from_opening_tag(opening_tag)
                self._tool_buffer = self._tool_buffer[tag_end + 1 :]
                self._tool_state = "body"
                continue

            end_index = self._tool_buffer.find(self._tool_end_tag)
            if end_index != -1:
                self._append_tool_argument_delta(
                    events, self._tool_buffer[:end_index]
                )
                self._tool_buffer = self._tool_buffer[
                    end_index + len(self._tool_end_tag) :
                ]
                events.extend(self._tool_call_boundary_events())
                self._clear_tool_call()
                continue
            flush_length = self._flushable_prefix_length(
                self._tool_buffer,
                self._tool_end_tag,
            )
            if not flush_length:
                break
            self._append_tool_argument_delta(
                events, self._tool_buffer[:flush_length]
            )
            self._tool_buffer = self._tool_buffer[flush_length:]
        return tuple(events)

    def _flush_tool(self) -> tuple[StreamProviderEvent, ...]:
        events: list[StreamProviderEvent] = []
        if self._tool_state == "outside":
            self._append_answer_delta(events, self._tool_buffer)
            self._tool_buffer = ""
            return tuple(events)

        self._append_tool_argument_delta(events, self._tool_buffer)
        self._tool_buffer = ""
        assert self._tool_call_id is not None
        events.append(
            StreamProviderEvent(
                kind=StreamItemKind.STREAM_DIAGNOSTIC,
                data={
                    "code": "tool_call.malformed",
                    "message": "unterminated tool call",
                    "tool_call_id": self._tool_call_id,
                },
                correlation=StreamItemCorrelation(
                    tool_call_id=self._tool_call_id
                ),
                visibility=StreamVisibility.DIAGNOSTIC,
            )
        )
        self._clear_tool_call()
        return tuple(events)

    def _tool_call_boundary_events(self) -> tuple[StreamProviderEvent, ...]:
        assert self._tool_call_id is not None
        arguments, valid_arguments = self._tool_buffer_arguments()
        if not valid_arguments:
            return (
                StreamProviderEvent(
                    kind=StreamItemKind.STREAM_DIAGNOSTIC,
                    data={
                        "code": "tool_call.malformed",
                        "message": "malformed tool call arguments",
                        "tool_call_id": self._tool_call_id,
                    },
                    correlation=StreamItemCorrelation(
                        tool_call_id=self._tool_call_id
                    ),
                    visibility=StreamVisibility.DIAGNOSTIC,
                ),
            )

        data: dict[str, object] = {
            "name": self._tool_name,
            "arguments": arguments,
        }
        return (
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_READY,
                data=data,
                correlation=StreamItemCorrelation(
                    tool_call_id=self._tool_call_id
                ),
            ),
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=StreamItemCorrelation(
                    tool_call_id=self._tool_call_id
                ),
            ),
        )

    def _tool_buffer_arguments(self) -> tuple[object, bool]:
        text = "".join(self._tool_argument_deltas)
        if not text.strip():
            return {}, True
        try:
            parsed = loads(text)
        except JSONDecodeError:
            return None, False
        if not isinstance(parsed, dict):
            return None, False
        return parsed, True

    def _append_answer_delta(
        self,
        events: list[StreamProviderEvent],
        text: str,
    ) -> None:
        if text:
            events.append(
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=text,
                )
            )

    def _append_reasoning_delta(
        self,
        events: list[StreamProviderEvent],
        text: str,
    ) -> None:
        if text:
            events.append(self._reasoning_delta(text))

    def _reasoning_delta(self, text: str) -> StreamProviderEvent:
        return StreamProviderEvent(
            kind=StreamItemKind.REASONING_DELTA,
            text_delta=text,
            visibility=StreamVisibility.PRIVATE,
        )

    def _append_tool_argument_delta(
        self,
        events: list[StreamProviderEvent],
        text: str,
    ) -> None:
        if not text:
            return
        assert self._tool_call_id is not None
        self._tool_argument_deltas.append(text)
        events.append(
            StreamProviderEvent(
                kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta=text,
                correlation=StreamItemCorrelation(
                    tool_call_id=self._tool_call_id
                ),
            )
        )

    def _next_tool_call_id(self) -> str:
        self._tool_call_index += 1
        return f"local-tool-call-{self._tool_call_index}"

    @staticmethod
    def _tool_name_from_opening_tag(opening_tag: str) -> str | None:
        for quote in ("'", '"'):
            marker = f"name={quote}"
            start = opening_tag.find(marker)
            if start == -1:
                continue
            value_start = start + len(marker)
            value_end = opening_tag.find(quote, value_start)
            if value_end != -1:
                return opening_tag[value_start:value_end]
        return None

    def _tool_start_index(self) -> int | None:
        search_from = 0
        while True:
            start_index = self._tool_buffer.find(
                self._tool_start_tag, search_from
            )
            if start_index == -1:
                return None

            boundary_index = start_index + len(self._tool_start_tag)
            if boundary_index == len(self._tool_buffer):
                return None

            boundary = self._tool_buffer[boundary_index]
            if boundary == ">" or boundary.isspace():
                return start_index

            search_from = start_index + 1

    def _tool_flushable_prefix_length(self) -> int:
        return self._flushable_prefix_length(
            self._tool_buffer,
            self._tool_start_tag,
            keep_full_marker=True,
        )

    @staticmethod
    def _flushable_prefix_length(
        buffer: str,
        marker: str,
        *,
        keep_full_marker: bool = False,
    ) -> int:
        keep = 0
        marker_suffix_length = (
            len(marker) if keep_full_marker else len(marker) - 1
        )
        max_suffix = min(len(buffer), marker_suffix_length)
        for length in range(max_suffix, 0, -1):
            if marker.startswith(buffer[-length:]):
                keep = length
                break
        return len(buffer) - keep

    def _clear_tool_call(self) -> None:
        self._tool_state = "outside"
        self._tool_call_id = None
        self._tool_name = None
        self._tool_argument_deltas.clear()


async def _close_async_iterable(
    events: AsyncIterable[Any],
) -> None:
    aclose = getattr(events, "aclose", None)
    if aclose is None:
        return
    assert callable(aclose)
    result = aclose()
    if isawaitable(result):
        awaited_result = await cast(Awaitable[object], result)
        assert awaited_result is None
    else:
        assert result is None


async def _close_async_iterables(
    *iterables: AsyncIterable[Any],
) -> None:
    errors: list[BaseException] = []
    for iterable in iterables:
        try:
            await _close_async_iterable(iterable)
        except (Exception, CancelledError) as exc:
            errors.append(exc)

    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("stream iterable close failed", errors)


@dataclass(slots=True)
class _ProviderToolCallState:
    ready: bool = False
    done: bool = False
    malformed: bool = False


@dataclass(slots=True)
class _ProviderStreamNormalizer:
    stream_session_id: str
    run_id: str
    turn_id: str
    provider_family: str | None
    capabilities: StreamProviderCapabilities | None
    close_after_terminal: bool
    _sequence: int = 0
    _accumulator: CanonicalStreamAccumulator = field(
        default_factory=CanonicalStreamAccumulator
    )
    _answer_started: bool = False
    _answer_done: bool = False
    _reasoning_started: bool = False
    _reasoning_done: bool = False
    _usage_completed: bool = False
    _tool_call_states: dict[str, _ProviderToolCallState] = field(
        default_factory=dict
    )

    def started(self) -> CanonicalStreamItem:
        metadata: dict[str, LooseJsonValue] = {}
        if self.capabilities is not None:
            metadata["capabilities"] = cast(
                LooseJsonValue, self.capabilities.to_metadata()
            )
        return self._item(
            kind=StreamItemKind.STREAM_STARTED,
            metadata=metadata,
        )

    def map_event(
        self, event: StreamProviderEvent
    ) -> tuple[CanonicalStreamItem, ...]:
        assert isinstance(event, StreamProviderEvent)
        if event.kind is StreamItemKind.STREAM_COMPLETED:
            return self._complete(usage=event.usage, provider_event=event)
        if event.kind is StreamItemKind.STREAM_ERRORED:
            return self._terminal(
                StreamItemKind.STREAM_ERRORED,
                provider_event=event,
            )
        if event.kind is StreamItemKind.STREAM_CANCELLED:
            return self._terminal(
                StreamItemKind.STREAM_CANCELLED,
                provider_event=event,
            )
        if event.kind is StreamItemKind.USAGE_COMPLETED:
            return self._final_usage(event)

        self._track_tool_call_state(event)
        item = self._item(
            kind=event.kind,
            text_delta=event.text_delta,
            data=event.data,
            usage=event.usage,
            correlation=event.correlation,
            visibility=event.visibility,
            metadata=event.metadata,
            provider_payload=event.provider_payload,
            provider_event_type=event.provider_event_type,
        )
        self._track_channel_boundary(item)
        return (item,)

    def completed(self) -> tuple[CanonicalStreamItem, ...]:
        incomplete_tool_call_error = self._incomplete_tool_call_error()
        if incomplete_tool_call_error is not None:
            return self.errored(
                StreamValidationError(incomplete_tool_call_error)
            )
        return self._complete(usage=None, provider_event=None)

    def cancelled(self) -> tuple[CanonicalStreamItem, ...]:
        return self._terminal(StreamItemKind.STREAM_CANCELLED)

    def errored(self, exc: Exception) -> tuple[CanonicalStreamItem, ...]:
        return self._terminal(
            StreamItemKind.STREAM_ERRORED,
            data={
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            },
        )

    def _final_usage(
        self,
        event: StreamProviderEvent,
    ) -> tuple[CanonicalStreamItem, ...]:
        items = list(self._open_channel_done_items())
        item = self._item(
            kind=event.kind,
            usage=event.usage,
            correlation=event.correlation,
            visibility=event.visibility,
            metadata=event.metadata,
            provider_payload=event.provider_payload,
            provider_event_type=event.provider_event_type,
        )
        self._track_channel_boundary(item)
        items.append(item)
        return tuple(items)

    def _complete(
        self,
        *,
        usage: LooseJsonValue | None,
        provider_event: StreamProviderEvent | None,
    ) -> tuple[CanonicalStreamItem, ...]:
        incomplete_tool_call_error = self._incomplete_tool_call_error()
        if incomplete_tool_call_error is not None:
            raise StreamValidationError(incomplete_tool_call_error)
        terminal_usage = usage if self._usage_completed else usage or {}
        return self._terminal(
            StreamItemKind.STREAM_COMPLETED,
            usage=terminal_usage if not self._usage_completed else None,
            provider_event=provider_event,
        )

    def _terminal(
        self,
        kind: StreamItemKind,
        *,
        data: LooseJsonValue | None = None,
        usage: LooseJsonValue | None = None,
        provider_event: StreamProviderEvent | None = None,
    ) -> tuple[CanonicalStreamItem, ...]:
        items = list(self._open_channel_done_items())

        terminal = self._item(
            kind=kind,
            data=data if provider_event is None else provider_event.data,
            usage=usage,
            correlation=(
                None if provider_event is None else provider_event.correlation
            ),
            visibility=(
                StreamVisibility.PUBLIC
                if provider_event is None
                else provider_event.visibility
            ),
            metadata={} if provider_event is None else provider_event.metadata,
            provider_payload=(
                None
                if provider_event is None
                else provider_event.provider_payload
            ),
            provider_event_type=(
                None
                if provider_event is None
                else provider_event.provider_event_type
            ),
            terminal_outcome=stream_terminal_outcome_for_kind(kind),
        )
        items.append(terminal)
        items.extend(self._closed())
        return tuple(items)

    def _open_channel_done_items(self) -> tuple[CanonicalStreamItem, ...]:
        items: list[CanonicalStreamItem] = []
        if self._answer_started and not self._answer_done:
            item = self._item(kind=StreamItemKind.ANSWER_DONE)
            self._track_channel_boundary(item)
            items.append(item)
        if self._reasoning_started and not self._reasoning_done:
            item = self._item(kind=StreamItemKind.REASONING_DONE)
            self._track_channel_boundary(item)
            items.append(item)
        return tuple(items)

    def _closed(self) -> tuple[CanonicalStreamItem, ...]:
        if not self.close_after_terminal:
            return ()
        return (self._item(kind=StreamItemKind.STREAM_CLOSED),)

    def _item(
        self,
        *,
        kind: StreamItemKind,
        text_delta: str | None = None,
        data: LooseJsonValue | None = None,
        usage: LooseJsonValue | None = None,
        correlation: StreamItemCorrelation | None = None,
        visibility: StreamVisibility = StreamVisibility.PUBLIC,
        metadata: dict[str, LooseJsonValue] | None = None,
        provider_payload: LooseJsonValue | None = None,
        provider_event_type: str | None = None,
        terminal_outcome: StreamTerminalOutcome | None = None,
    ) -> CanonicalStreamItem:
        item = CanonicalStreamItem(
            stream_session_id=self.stream_session_id,
            run_id=self.run_id,
            turn_id=self.turn_id,
            sequence=self._sequence,
            kind=kind,
            channel=stream_channel_for_kind(kind),
            correlation=correlation or StreamItemCorrelation(),
            text_delta=text_delta,
            data=data,
            usage=usage,
            terminal_outcome=terminal_outcome,
            visibility=visibility,
            metadata={} if metadata is None else metadata,
            provider_payload=provider_payload,
            provider_family=self.provider_family,
            provider_event_type=provider_event_type,
        )
        self._accumulator.add(item)
        self._sequence += 1
        return item

    def _track_channel_boundary(self, item: CanonicalStreamItem) -> None:
        if item.kind is StreamItemKind.ANSWER_DELTA:
            self._answer_started = True
        elif item.kind is StreamItemKind.ANSWER_DONE:
            self._answer_done = True
        elif item.kind is StreamItemKind.REASONING_DELTA:
            self._reasoning_started = True
        elif item.kind is StreamItemKind.REASONING_DONE:
            self._reasoning_done = True
        elif item.kind is StreamItemKind.USAGE_COMPLETED:
            self._usage_completed = True

    def _track_tool_call_state(self, event: StreamProviderEvent) -> None:
        tool_call_id = event.correlation.tool_call_id
        if (
            event.kind is StreamItemKind.STREAM_DIAGNOSTIC
            and tool_call_id is not None
            and self._is_malformed_tool_call_diagnostic(event)
        ):
            self._tool_call_states.setdefault(
                tool_call_id, _ProviderToolCallState()
            ).malformed = True
            return

        if event.kind not in _TOOL_CALL_KINDS:
            return
        if tool_call_id is None:
            raise StreamValidationError("tool-call item missing tool_call_id")

        state = self._tool_call_states.setdefault(
            tool_call_id, _ProviderToolCallState()
        )
        if state.done:
            raise StreamValidationError(
                "tool-call item emitted after tool-call done"
            )
        if state.malformed:
            raise StreamValidationError(
                "tool-call item emitted after malformed diagnostic"
            )

        if event.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            if state.ready:
                raise StreamValidationError(
                    "tool-call argument emitted after ready"
                )
        elif event.kind is StreamItemKind.TOOL_CALL_READY:
            if state.ready:
                raise StreamValidationError("duplicate tool-call ready item")
            state.ready = True
        elif event.kind is StreamItemKind.TOOL_CALL_DONE:
            if not state.ready:
                raise StreamValidationError("tool-call done before ready")
            state.done = True

    def _incomplete_tool_call_error(self) -> str | None:
        for tool_call_id, state in self._tool_call_states.items():
            if state.done or state.malformed:
                continue
            if not state.ready:
                return f"tool call {tool_call_id} missing ready"
            return f"tool call {tool_call_id} missing done"
        return None

    @staticmethod
    def _is_malformed_tool_call_diagnostic(
        event: StreamProviderEvent,
    ) -> bool:
        data = event.data
        return (
            isinstance(data, dict)
            and data.get("code") == "tool_call.malformed"
        )


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

    def canonical_stream(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: ProviderFamily | str | None = None,
        capabilities: StreamProviderCapabilities | None = None,
        close_after_terminal: bool = True,
    ) -> AsyncIterator[CanonicalStreamItem]:
        return normalize_local_stream(
            self.__aiter__(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=(
                provider_family
                or getattr(self, "provider_family", None)
                or ProviderFamily.LOCAL
            ),
            capabilities=capabilities,
            close_after_terminal=close_after_terminal,
        )


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
    metadata = _token_metadata(token)
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
            metadata=metadata,
        )
    if isinstance(token, ToolCallToken):
        tool_call_id = str(token.call.id) if token.call else "legacy-tool-call"
        data: LooseJsonValue | None = None
        if token.call is not None:
            data = cast(
                LooseJsonValue,
                {
                    "name": token.call.name,
                    "arguments": token.call.arguments,
                },
            )
        return CanonicalStreamItem(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            sequence=sequence,
            kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            channel=StreamChannel.TOOL_CALL,
            correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
            text_delta=text,
            data=data,
            metadata=metadata,
        )
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=StreamItemKind.ANSWER_DELTA,
        channel=StreamChannel.ANSWER,
        text_delta=text,
        metadata=metadata,
    )


def _token_metadata(
    token: Token | TokenDetail | str,
) -> dict[str, LooseJsonValue]:
    if isinstance(token, str):
        return {}
    metadata: dict[str, LooseJsonValue] = {}
    if isinstance(token.id, int) and token.id >= 0:
        metadata["token_id"] = token.id
    if token.probability is not None:
        metadata["probability"] = token.probability
    if isinstance(token, TokenDetail):
        if token.step is not None:
            metadata["step"] = token.step
        if token.probability_distribution is not None:
            metadata["probability_distribution"] = (
                token.probability_distribution
            )
        if token.tokens is not None:
            metadata["tokens"] = [
                _token_metadata(candidate) | {"token": candidate.token}
                for candidate in token.tokens
            ]
    if isinstance(token, ToolCallToken) and token.provider_name is not None:
        metadata["provider_name"] = token.provider_name
    return metadata
