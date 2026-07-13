from ..observability import observability_key_sample
from ..types import LooseJsonValue
from .provider import ProviderFamily, provider_family_value

from abc import ABC, abstractmethod
from asyncio import CancelledError
from collections import deque
from collections.abc import AsyncIterable, Awaitable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
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


class StreamReasoningRepresentation(StrEnum):
    """Identify the provider-authored reasoning text representation."""

    NATIVE_TEXT = "native_text"
    SUMMARY = "summary"


class StreamReasoningSegmentStatus(StrEnum):
    """Identify the retained reasoning segment completion status."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"


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
    TEMPORARY_COMPATIBILITY_SHIM = "temporary_compatibility_shim"


class StreamLegacyBoundaryCategory(StrEnum):
    PRODUCER = "producer"
    SDK_RESPONSE = "sdk_response"
    ORCHESTRATOR = "orchestrator"
    PARSER = "parser"
    EVENTING = "eventing"
    CLI_STDOUT = "cli_stdout"
    CHAT_SSE = "chat_sse"
    RESPONSES_SSE = "responses_sse"
    MCP = "mcp"
    A2A = "a2a"
    FLOW = "flow"
    TEST_FIXTURE = "test_fixture"
    HELPER_ONLY = "helper_only"


class StreamLegacyInventoryScope(StrEnum):
    PRODUCTION_RUNTIME = "production_runtime"
    TEST_FIXTURE = "test_fixture"
    HELPER_ONLY = "helper_only"


class StreamLegacyBoundaryDirection(StrEnum):
    ACCEPTS = "accepts"
    EMITS = "emits"
    PROJECTS = "projects"
    PUBLIC_RETURN_TYPE = "public_return_type"
    CONTROL = "control"


class StreamProducerBackend(StrEnum):
    HOSTED = "hosted"
    LOCAL = "local"


class StreamValidationError(ValueError):
    pass


_LEGACY_TOKEN_STREAM_CLASS = ("avalan.entities", "Token")


def _reject_legacy_token_stream_chunk(chunk: object) -> None:
    for chunk_class in type(chunk).__mro__:
        if (
            chunk_class.__module__ == _LEGACY_TOKEN_STREAM_CLASS[0]
            and chunk_class.__name__ == _LEGACY_TOKEN_STREAM_CLASS[1]
        ):
            raise StreamValidationError("unsupported legacy local stream item")


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamProviderCapabilities:
    backend: StreamProducerBackend
    provider_family: ProviderFamily | str | None = None
    supports_reasoning: bool = False
    supports_reasoning_summary: bool = False
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
            ("supports_reasoning_summary", self.supports_reasoning_summary),
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
            "supports_reasoning_summary": self.supports_reasoning_summary,
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
    categories: tuple[StreamLegacyBoundaryCategory, ...]
    scope: StreamLegacyInventoryScope
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
        _assert_legacy_inventory_metadata(
            self.categories, self.scope, self.classification
        )
        _assert_non_empty_string(self.owner, "owner")
        _assert_non_empty_string(self.removal_condition, "removal_condition")
        if self.ingestion_shim is not None:
            _assert_non_empty_string(self.ingestion_shim, "ingestion_shim")
        if self.canonical_kind is not None:
            assert isinstance(self.canonical_kind, StreamItemKind)
        if self.canonical_channel is not None:
            assert isinstance(self.canonical_channel, StreamChannel)
        assert (self.canonical_kind is None) is (
            self.canonical_channel is None
        )
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
class StreamLegacyClassifierInventoryEntry:
    module: str
    qualname: str
    surfaces: tuple[StreamLegacySurface, ...]
    classification: StreamLegacySurfaceClassification
    category: StreamLegacyBoundaryCategory
    scope: StreamLegacyInventoryScope
    owner: str
    removal_condition: str

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.module, "module")
        _assert_non_empty_string(self.qualname, "qualname")
        assert isinstance(self.surfaces, tuple)
        assert self.surfaces
        assert len(set(self.surfaces)) == len(self.surfaces)
        for surface in self.surfaces:
            assert isinstance(surface, StreamLegacySurface)
        assert isinstance(
            self.classification,
            StreamLegacySurfaceClassification,
        )
        _assert_legacy_inventory_metadata(
            (self.category,), self.scope, self.classification
        )
        _assert_legacy_inventory_namespace(
            self.module, self.qualname, self.scope
        )
        _assert_non_empty_string(self.owner, "owner")
        _assert_non_empty_string(self.removal_condition, "removal_condition")


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamLegacyRuntimeBoundaryInventoryEntry:
    module: str
    qualname: str
    surfaces: tuple[StreamLegacySurface, ...]
    classification: StreamLegacySurfaceClassification
    category: StreamLegacyBoundaryCategory
    scope: StreamLegacyInventoryScope
    directions: tuple[StreamLegacyBoundaryDirection, ...]
    owner: str
    removal_condition: str

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.module, "module")
        _assert_non_empty_string(self.qualname, "qualname")
        assert isinstance(self.surfaces, tuple)
        assert self.surfaces
        assert len(set(self.surfaces)) == len(self.surfaces)
        for surface in self.surfaces:
            assert isinstance(surface, StreamLegacySurface)
        assert isinstance(
            self.classification,
            StreamLegacySurfaceClassification,
        )
        _assert_legacy_inventory_metadata(
            (self.category,), self.scope, self.classification
        )
        _assert_legacy_inventory_namespace(
            self.module, self.qualname, self.scope
        )
        assert isinstance(self.directions, tuple)
        assert self.directions
        assert len(set(self.directions)) == len(self.directions)
        for direction in self.directions:
            assert isinstance(direction, StreamLegacyBoundaryDirection)
        _assert_non_empty_string(self.owner, "owner")
        _assert_non_empty_string(self.removal_condition, "removal_condition")


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamItemCorrelation:
    provider_request_id: str | None = None
    model_continuation_id: str | None = None
    tool_call_id: str | None = None
    flow_run_id: str | None = None
    node_id: str | None = None
    parent_sequence: int | None = None
    protocol_item_id: str | None = None
    provider_output_index: int | None = None
    provider_summary_index: int | None = None
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
        for index_field_name, index_value in (
            ("provider_output_index", self.provider_output_index),
            ("provider_summary_index", self.provider_summary_index),
        ):
            if index_value is not None:
                _assert_non_negative_int(index_value, index_field_name)

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
            ("provider_output_index", self.provider_output_index),
            ("provider_summary_index", self.provider_summary_index),
            ("task_id", self.task_id),
            ("artifact_id", self.artifact_id),
        ):
            if value is not None:
                result[field_name] = value
        return result


_EMPTY_STREAM_ITEM_CORRELATION = StreamItemCorrelation()
REASONING_SEGMENT_BOUNDARY_METADATA_KEY = "reasoning.segment_boundary"


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamReasoningSegment:
    """Describe one retained contiguous reasoning segment."""

    representation: StreamReasoningRepresentation
    segment_instance_ordinal: int
    text: str
    completed: bool
    status: StreamReasoningSegmentStatus
    terminal_outcome: StreamTerminalOutcome | None
    provider_item_id: str | None = None
    output_index: int | None = None
    summary_index: int | None = None
    continuation_id: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.representation, StreamReasoningRepresentation)
        _assert_non_negative_int(
            self.segment_instance_ordinal, "segment_instance_ordinal"
        )
        assert isinstance(self.text, str)
        assert self.text, "reasoning segment text must not be empty"
        assert isinstance(self.completed, bool)
        assert isinstance(self.status, StreamReasoningSegmentStatus)
        if self.terminal_outcome is not None:
            assert isinstance(self.terminal_outcome, StreamTerminalOutcome)
        for field_name, value in (
            ("provider_item_id", self.provider_item_id),
            ("continuation_id", self.continuation_id),
        ):
            if value is not None:
                _assert_non_empty_string(value, field_name)
        for field_name, index_value in (
            ("output_index", self.output_index),
            ("summary_index", self.summary_index),
        ):
            if index_value is not None:
                _assert_non_negative_int(index_value, field_name)
        if self.status is StreamReasoningSegmentStatus.IN_PROGRESS:
            assert not self.completed
            assert self.terminal_outcome is None
        elif self.status is StreamReasoningSegmentStatus.COMPLETED:
            assert self.completed
            assert self.terminal_outcome is StreamTerminalOutcome.COMPLETED
        else:
            assert not self.completed
            assert self.terminal_outcome in (
                StreamTerminalOutcome.ERRORED,
                StreamTerminalOutcome.CANCELLED,
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class StreamReasoningTruncation:
    """Describe deterministic retained reasoning tail truncation."""

    truncated: bool = False
    dropped_segments: int = 0
    dropped_characters: int = 0
    dropped_utf8_bytes: int = 0
    leading_segment_partial: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.truncated, bool)
        assert isinstance(self.leading_segment_partial, bool)
        for field_name, value in (
            ("dropped_segments", self.dropped_segments),
            ("dropped_characters", self.dropped_characters),
            ("dropped_utf8_bytes", self.dropped_utf8_bytes),
        ):
            _assert_non_negative_int(value, field_name)
        has_drops = bool(
            self.dropped_segments
            or self.dropped_characters
            or self.dropped_utf8_bytes
        )
        assert self.truncated is has_drops
        if self.leading_segment_partial:
            assert self.truncated


@dataclass(slots=True)
class StreamReasoningSegmentState:
    """Allocate and validate response-local reasoning segment ordinals."""

    _next_ordinal: int = 0
    _active_identity: tuple[object, ...] | None = None
    _active_ordinal: int | None = None

    def allocate(
        self,
        representation: StreamReasoningRepresentation,
        correlation: StreamItemCorrelation | None = None,
    ) -> int:
        """Return the ordinal for a contiguous reasoning delta.

        Args:
            representation: Provider-authored reasoning representation.
            correlation: Optional provider reasoning identity.

        Returns:
            Zero-based response-local segment instance ordinal.
        """
        assert isinstance(representation, StreamReasoningRepresentation)
        assert correlation is None or isinstance(
            correlation, StreamItemCorrelation
        )
        identity = self._identity(representation, correlation)
        if self._active_identity == identity:
            assert self._active_ordinal is not None
            return self._active_ordinal
        ordinal = self._next_ordinal
        self._next_ordinal += 1
        self._active_identity = identity
        self._active_ordinal = ordinal
        return ordinal

    def complete_segment(self) -> None:
        """Mark the active reasoning segment complete."""
        self._active_identity = None
        self._active_ordinal = None

    @property
    def next_allocation_follows_boundary(self) -> bool:
        """Return whether the next allocation follows a prior segment."""
        return self._next_ordinal > 0 and self._active_identity is None

    def observe(self, event: "StreamProviderEvent") -> None:
        """Validate one provider event against response-local segment state.

        Args:
            event: Provider event to validate.
        """
        assert isinstance(event, StreamProviderEvent)
        if event.kind is not StreamItemKind.REASONING_DELTA:
            self.complete_segment()
            return
        assert event.reasoning_representation is not None
        assert event.segment_instance_ordinal is not None
        self.observe_delta(
            event.reasoning_representation,
            event.correlation,
            event.segment_instance_ordinal,
            follows_completion=(
                event.metadata.get(REASONING_SEGMENT_BOUNDARY_METADATA_KEY)
                == "completed"
            ),
        )

    def observe_delta(
        self,
        representation: StreamReasoningRepresentation,
        correlation: StreamItemCorrelation,
        ordinal: int,
        *,
        follows_completion: bool = False,
    ) -> None:
        """Validate one reasoning delta against the current segment.

        Args:
            representation: Provider-authored reasoning representation.
            correlation: Optional provider reasoning identity fields.
            ordinal: Response-local segment instance ordinal.
            follows_completion: Whether an explicit hidden segment
                completion preceded this delta.
        """
        assert isinstance(representation, StreamReasoningRepresentation)
        assert isinstance(correlation, StreamItemCorrelation)
        _assert_non_negative_int(ordinal, "segment_instance_ordinal")
        assert isinstance(follows_completion, bool)
        if follows_completion:
            self.complete_segment()
        identity = self._identity(representation, correlation)
        if self._active_identity == identity:
            assert self._active_ordinal is not None
            if ordinal == self._active_ordinal:
                return
            raise StreamValidationError(
                "reasoning segment ordinal changed without a boundary"
            )
        elif ordinal != self._next_ordinal:
            raise StreamValidationError(
                "reasoning segment ordinal does not match its boundary"
            )
        self._next_ordinal += 1
        self._active_identity = identity
        self._active_ordinal = ordinal

    @staticmethod
    def _identity(
        representation: StreamReasoningRepresentation,
        correlation: StreamItemCorrelation | None,
    ) -> tuple[object, ...]:
        resolved = correlation or _EMPTY_STREAM_ITEM_CORRELATION
        return (
            representation,
            resolved.protocol_item_id,
            resolved.provider_output_index,
            resolved.provider_summary_index,
            resolved.model_continuation_id,
        )


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
    reasoning_representation: StreamReasoningRepresentation | None = None
    segment_instance_ordinal: int | None = None
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
        _validate_reasoning_fields(
            self.kind,
            self.visibility,
            self.reasoning_representation,
            self.segment_instance_ordinal,
        )


class StreamProviderAdapterError(Exception):
    error: Exception
    provider_payload: LooseJsonValue | None
    provider_event_type: str | None
    safe_data: LooseJsonValue | None

    def __init__(
        self,
        error: Exception,
        *,
        provider_payload: LooseJsonValue | None = None,
        provider_event_type: str | None = None,
        safe_data: LooseJsonValue | None = None,
    ) -> None:
        assert isinstance(error, Exception)
        if provider_event_type is not None:
            _assert_non_empty_string(
                provider_event_type, "provider_event_type"
            )
        super().__init__(
            "Provider adapter rejected an event with safe diagnostics."
            if safe_data is not None
            else str(error)
        )
        self.error = error
        self.provider_payload = provider_payload
        self.provider_event_type = provider_event_type
        self.safe_data = safe_data


class StreamConsumerCancellation(CancelledError):
    """Signal cancellation owned by the local stream consumer."""


class StreamConsumerClosure(Exception):
    """Signal silent closure owned by the local stream consumer."""


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
    reasoning_segment_limit: int = 1024
    reasoning_character_limit: int = 262144
    reasoning_text_byte_limit: int = 1048576
    cli_reasoning_segment_limit: int = 1024
    cli_reasoning_character_limit: int = 262144
    cli_reasoning_text_byte_limit: int = 1048576
    responses_reasoning_item_segment_limit: int = 1024
    responses_reasoning_item_character_limit: int = 262144
    responses_reasoning_item_text_byte_limit: int = 1048576
    mcp_reasoning_segment_limit: int = 512
    mcp_reasoning_character_limit: int = 1048576
    mcp_reasoning_text_byte_limit: int = 1048576
    a2a_reasoning_segment_limit: int = 512
    a2a_reasoning_character_limit: int = 1048576
    a2a_reasoning_text_byte_limit: int = 1048576
    replay_history_item_limit: int = 1024
    openai_replay_reasoning_item_limit: int = 1024
    openai_replay_reasoning_summary_node_limit: int = 4096
    openai_replay_reasoning_summary_character_limit: int = 262144
    openai_replay_reasoning_summary_serialized_byte_limit: int = 1048576
    ui_buffer_item_limit: int = 1024
    metrics_history_item_limit: int = 2048
    event_history_item_limit: int = 2048
    mcp_resource_item_limit: int = 512
    mcp_resource_text_byte_limit: int = 1048576
    a2a_task_record_item_limit: int = 512
    a2a_task_event_byte_limit: int = 1048576
    flow_history_item_limit: int = 1024
    active_session_lossless: bool = True

    def __post_init__(self) -> None:
        _assert_positive_int(
            self.accumulator_item_limit, "accumulator_item_limit"
        )
        _assert_positive_int(
            self.mcp_resource_text_byte_limit,
            "mcp_resource_text_byte_limit",
        )
        _assert_positive_int(
            self.a2a_task_event_byte_limit,
            "a2a_task_event_byte_limit",
        )
        assert (
            self.a2a_task_event_byte_limit >= 2
        ), "a2a_task_event_byte_limit must be at least 2"
        for field_name, value in (
            ("reasoning_segment_limit", self.reasoning_segment_limit),
            ("reasoning_character_limit", self.reasoning_character_limit),
            ("reasoning_text_byte_limit", self.reasoning_text_byte_limit),
            (
                "cli_reasoning_segment_limit",
                self.cli_reasoning_segment_limit,
            ),
            (
                "cli_reasoning_character_limit",
                self.cli_reasoning_character_limit,
            ),
            (
                "cli_reasoning_text_byte_limit",
                self.cli_reasoning_text_byte_limit,
            ),
            (
                "responses_reasoning_item_segment_limit",
                self.responses_reasoning_item_segment_limit,
            ),
            (
                "responses_reasoning_item_character_limit",
                self.responses_reasoning_item_character_limit,
            ),
            (
                "responses_reasoning_item_text_byte_limit",
                self.responses_reasoning_item_text_byte_limit,
            ),
            (
                "mcp_reasoning_segment_limit",
                self.mcp_reasoning_segment_limit,
            ),
            (
                "mcp_reasoning_character_limit",
                self.mcp_reasoning_character_limit,
            ),
            (
                "mcp_reasoning_text_byte_limit",
                self.mcp_reasoning_text_byte_limit,
            ),
            (
                "a2a_reasoning_segment_limit",
                self.a2a_reasoning_segment_limit,
            ),
            (
                "a2a_reasoning_character_limit",
                self.a2a_reasoning_character_limit,
            ),
            (
                "a2a_reasoning_text_byte_limit",
                self.a2a_reasoning_text_byte_limit,
            ),
            ("replay_history_item_limit", self.replay_history_item_limit),
            (
                "openai_replay_reasoning_item_limit",
                self.openai_replay_reasoning_item_limit,
            ),
            (
                "openai_replay_reasoning_summary_node_limit",
                self.openai_replay_reasoning_summary_node_limit,
            ),
            (
                "openai_replay_reasoning_summary_character_limit",
                self.openai_replay_reasoning_summary_character_limit,
            ),
            (
                "openai_replay_reasoning_summary_serialized_byte_limit",
                self.openai_replay_reasoning_summary_serialized_byte_limit,
            ),
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
class StreamPerformanceBudgetReconciliation:
    baseline_budget: StreamPerformanceBudget = field(
        default_factory=StreamPerformanceBudget
    )
    enforced_budget: StreamPerformanceBudget = field(
        default_factory=StreamPerformanceBudget
    )
    equivalence_harness_passed: bool = True
    benchmark_source: str = "specs/streaming/BENCHMARKS.md"

    def __post_init__(self) -> None:
        assert isinstance(self.baseline_budget, StreamPerformanceBudget)
        assert isinstance(self.enforced_budget, StreamPerformanceBudget)
        assert isinstance(
            self.equivalence_harness_passed, bool
        ), "equivalence_harness_passed must be a boolean"
        _assert_non_empty_string(self.benchmark_source, "benchmark_source")
        loosened = self.loosened_metrics
        tightened = self.tightened_metrics
        assert not loosened, "enforced budget must not loosen baseline"
        if tightened:
            assert self.equivalence_harness_passed

    @property
    def tightened_metrics(self) -> tuple[str, ...]:
        return self._metrics_with_enforced_budget_below_baseline()

    @property
    def loosened_metrics(self) -> tuple[str, ...]:
        return self._metrics_with_enforced_budget_above_baseline()

    def _metrics_with_enforced_budget_below_baseline(self) -> tuple[str, ...]:
        return self._compare_metrics(enforced_below_baseline=True)

    def _metrics_with_enforced_budget_above_baseline(self) -> tuple[str, ...]:
        return self._compare_metrics(enforced_below_baseline=False)

    def _compare_metrics(
        self, *, enforced_below_baseline: bool
    ) -> tuple[str, ...]:
        metrics: list[str] = []
        for field_name in _STREAM_PERFORMANCE_BUDGET_FIELDS:
            baseline_value = getattr(self.baseline_budget, field_name)
            enforced_value = getattr(self.enforced_budget, field_name)
            if enforced_below_baseline:
                matches = enforced_value < baseline_value
            else:
                matches = enforced_value > baseline_value
            if matches:
                metrics.append(field_name)
        return tuple(metrics)


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
    reasoning_representation: StreamReasoningRepresentation | None = None
    segment_instance_ordinal: int | None = None
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
        _validate_reasoning_fields(
            self.kind,
            self.visibility,
            self.reasoning_representation,
            self.segment_instance_ordinal,
        )
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
        if self.reasoning_representation is not None:
            result["reasoning_representation"] = (
                self.reasoning_representation.value
            )
        if self.segment_instance_ordinal is not None:
            result["segment_instance_ordinal"] = self.segment_instance_ordinal
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
        if self.kind is StreamItemKind.REASONING_DELTA:
            assert (
                self.text_delta
            ), "reasoning deltas must carry non-empty text"
        elif self.kind in _TEXT_DELTA_KINDS:
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
    reasoning_representation: StreamReasoningRepresentation | None = None
    segment_instance_ordinal: int | None = None
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
        _validate_reasoning_fields(
            self.kind,
            self.visibility,
            self.reasoning_representation,
            self.segment_instance_ordinal,
        )
        self._validate_kind_payload()

    def _validate_kind_payload(self) -> None:
        if self.kind is StreamItemKind.REASONING_DELTA:
            assert (
                self.text_delta
            ), "reasoning deltas must carry non-empty text"
        elif self.kind in _TEXT_DELTA_KINDS:
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
            reasoning_representation=item.reasoning_representation,
            segment_instance_ordinal=item.segment_instance_ordinal,
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
        reasoning_representation=projection.reasoning_representation,
        segment_instance_ordinal=projection.segment_instance_ordinal,
        metadata=dict(projection.metadata),
        provider_family=projection.provider_family,
        provider_event_type=projection.provider_event_type,
    )


def project_stream_consumer_item(
    item: object,
    sequence: int,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    unsupported_message: str,
    accumulate: bool = False,
) -> StreamConsumerProjection:
    state = StreamProjectionState(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        accumulate=accumulate,
    )
    return state.project(
        item,
        sequence,
        unsupported_message=unsupported_message,
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
    last_sequence: int | None = None
    iterator = items.__aiter__()
    try:
        async for item in iterator:
            if accumulator is not None:
                accumulator.add(item)
                last_sequence = _validate_lossless_sequence_gap(
                    item, last_sequence
                )
            yield project_canonical_stream_item(item)
        if accumulator is not None:
            accumulator.validate_complete()
    finally:
        if iterator is not items:
            await _close_async_iterables(iterator, items)
        else:
            await _close_async_iterable(iterator)


def stream_iterator(source: object) -> AsyncIterator[Any]:
    assert isinstance(source, AsyncIterable)
    return source.__aiter__()


def stream_consumer_iterator(
    source: object,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    unsupported_message: str = "unsupported stream consumer item",
    close_source_on_generator_exit: bool = True,
) -> AsyncIterator[StreamConsumerProjection]:
    _assert_non_empty_string(unsupported_message, "unsupported_message")
    assert isinstance(close_source_on_generator_exit, bool)
    consumer_projections = getattr(source, "consumer_projections", None)
    if callable(consumer_projections):
        iterator = consumer_projections(
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
        )
        assert isinstance(iterator, AsyncIterable)
        return _validated_consumer_projection_iterator(iterator.__aiter__())
    assert isinstance(
        source, AsyncIterable
    ), "stream consumer source must be an async iterable"
    return _validated_stream_consumer_item_iterator(
        source.__aiter__(),
        source=source,
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        unsupported_message=unsupported_message,
        close_source_on_generator_exit=close_source_on_generator_exit,
    )


async def _validated_stream_consumer_item_iterator(
    iterator: AsyncIterator[Any],
    *,
    source: AsyncIterable[Any],
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    unsupported_message: str,
    close_source_on_generator_exit: bool,
) -> AsyncIterator[StreamConsumerProjection]:
    state = StreamProjectionState(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
    )
    sequence = 0
    close_source = False
    try:
        async for item in iterator:
            projection = state.project(
                item,
                sequence,
                unsupported_message=unsupported_message,
            )
            yield projection
            sequence += 1
        state.validate_complete()
    except (GeneratorExit, CancelledError):
        close_source = close_source_on_generator_exit
        raise
    except Exception:
        close_source = close_source_on_generator_exit
        raise
    finally:
        if iterator is not source or close_source:
            await _close_async_iterable(iterator)


async def _validated_consumer_projection_iterator(
    iterator: AsyncIterator[Any],
) -> AsyncIterator[StreamConsumerProjection]:
    accumulator = CanonicalStreamAccumulator()
    last_sequence: int | None = None
    try:
        async for item in iterator:
            if not isinstance(item, StreamConsumerProjection):
                raise StreamValidationError(
                    "consumer projection stream item must be "
                    "StreamConsumerProjection"
                )
            canonical_item = canonical_item_from_consumer_projection(item)
            accumulator.add(canonical_item)
            last_sequence = _validate_lossless_sequence_gap(
                canonical_item, last_sequence
            )
            yield item
        accumulator.validate_complete()
    finally:
        await _close_async_iterable(iterator)


def _validate_lossless_sequence_gap(
    item: CanonicalStreamItem,
    last_sequence: int | None,
) -> int:
    assert isinstance(item, CanonicalStreamItem)
    if last_sequence is not None and item.sequence != last_sequence + 1:
        raise StreamValidationError("lossless consumer stream sequence gap")
    return item.sequence


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
    correlation = _stream_observability_correlation(item)
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


def _stream_observability_correlation(
    item: CanonicalStreamItem,
) -> dict[str, object]:
    if item.kind is not StreamItemKind.REASONING_DELTA:
        return item.correlation.to_trace_dict()
    correlation: dict[str, object] = {}
    for field_name, string_value in (
        ("protocol_item_id", item.correlation.protocol_item_id),
        ("model_continuation_id", item.correlation.model_continuation_id),
    ):
        if string_value is not None:
            digest = sha256(string_value.encode("utf-8")).hexdigest()
            correlation[field_name] = f"sha256:{digest}"
    for field_name, index_value in (
        ("provider_output_index", item.correlation.provider_output_index),
        ("provider_summary_index", item.correlation.provider_summary_index),
    ):
        if index_value is not None:
            correlation[field_name] = index_value
    return correlation


def _stream_observability_summary(
    item: CanonicalStreamItem,
) -> dict[str, object]:
    summary: dict[str, object] = {}
    if item.text_delta is not None:
        summary["text_delta_length"] = len(item.text_delta)
    if item.reasoning_representation is not None:
        summary["reasoning_representation"] = (
            item.reasoning_representation.value
        )
    if item.segment_instance_ordinal is not None:
        summary["segment_instance_ordinal"] = item.segment_instance_ordinal
    if item.kind is not StreamItemKind.REASONING_DELTA:
        if isinstance(item.data, dict):
            data_keys, data_keys_truncated = observability_key_sample(
                item.data
            )
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
    if (
        item.provider_payload is not None
        and item.kind is not StreamItemKind.REASONING_DELTA
    ):
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
_MODEL_CONTINUATION_TERMINAL_KINDS = frozenset(
    {
        StreamItemKind.MODEL_CONTINUATION_COMPLETED,
        StreamItemKind.MODEL_CONTINUATION_ERROR,
        StreamItemKind.MODEL_CONTINUATION_CANCELLED,
    }
)
_MODEL_CONTINUATION_KINDS = frozenset(
    {
        StreamItemKind.MODEL_CONTINUATION_STARTED,
        *tuple(_MODEL_CONTINUATION_TERMINAL_KINDS),
    }
)
_USAGE_KINDS = frozenset(
    {
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    }
)
_STREAM_PERFORMANCE_BUDGET_FIELDS = (
    "time_to_first_item_ms",
    "cancellation_latency_ms",
    "close_latency_ms",
    "max_queue_depth",
    "max_memory_bytes",
    "per_item_overhead_us",
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


@dataclass(slots=True)
class _TextChannelBoundaryState:
    started: bool = False
    done: bool = False


@dataclass(slots=True)
class _ToolCallBoundaryState:
    started: bool = False
    ready: bool = False
    done: bool = False


@dataclass(slots=True)
class _ToolExecutionBoundaryState:
    started: bool = False
    terminal_kind: StreamItemKind | None = None


@dataclass(slots=True)
class _ModelContinuationBoundaryState:
    started: bool = False
    terminal_kind: StreamItemKind | None = None


def _assert_positive_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value > 0, f"{field_name} must be positive"


def _assert_non_negative_int(value: object, field_name: str) -> None:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must not be negative"


def _validate_reasoning_fields(
    kind: StreamItemKind,
    visibility: StreamVisibility,
    representation: StreamReasoningRepresentation | None,
    segment_instance_ordinal: int | None,
) -> None:
    if kind is StreamItemKind.REASONING_DELTA:
        assert isinstance(
            representation, StreamReasoningRepresentation
        ), "reasoning_representation must be a StreamReasoningRepresentation"
        assert visibility is StreamVisibility.PRIVATE
        _assert_non_negative_int(
            segment_instance_ordinal, "segment_instance_ordinal"
        )
        return
    assert (
        representation is None
    ), "reasoning_representation is only valid on reasoning deltas"
    assert (
        segment_instance_ordinal is None
    ), "segment_instance_ordinal is only valid on reasoning deltas"


def _assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def _assert_legacy_inventory_metadata(
    categories: tuple[StreamLegacyBoundaryCategory, ...],
    scope: StreamLegacyInventoryScope,
    classification: StreamLegacySurfaceClassification,
) -> None:
    assert isinstance(categories, tuple), "categories must be a tuple"
    assert categories, "categories must not be empty"
    assert len(set(categories)) == len(categories)
    for category in categories:
        assert isinstance(category, StreamLegacyBoundaryCategory)
    assert isinstance(scope, StreamLegacyInventoryScope)
    assert isinstance(classification, StreamLegacySurfaceClassification)
    if scope is StreamLegacyInventoryScope.PRODUCTION_RUNTIME:
        assert StreamLegacyBoundaryCategory.TEST_FIXTURE not in categories
        assert classification is StreamLegacySurfaceClassification.REMOVE_NOW
    elif scope is StreamLegacyInventoryScope.TEST_FIXTURE:
        assert categories == (StreamLegacyBoundaryCategory.TEST_FIXTURE,)
        assert (
            classification is not StreamLegacySurfaceClassification.REMOVE_NOW
        )
    else:
        assert categories == (StreamLegacyBoundaryCategory.HELPER_ONLY,)
        assert (
            classification is not StreamLegacySurfaceClassification.REMOVE_NOW
        )


def _assert_legacy_inventory_namespace(
    module: str,
    qualname: str,
    scope: StreamLegacyInventoryScope,
) -> None:
    _assert_non_empty_string(module, "module")
    _assert_non_empty_string(qualname, "qualname")
    assert isinstance(scope, StreamLegacyInventoryScope)
    if scope is StreamLegacyInventoryScope.PRODUCTION_RUNTIME:
        return
    if scope is StreamLegacyInventoryScope.TEST_FIXTURE:
        assert module.startswith("tests.")
        assert "legacy_fixture" in qualname or "legacy_rejection" in qualname
        return
    assert module.startswith("avalan.") or module.startswith("tests.")
    assert "legacy_helper" in qualname or "migration_helper" in qualname


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
] = ()


_LEGACY_STREAM_CLASSIFIER_INVENTORY: tuple[
    StreamLegacyClassifierInventoryEntry, ...
] = ()


_LEGACY_STREAM_RUNTIME_BOUNDARY_INVENTORY: tuple[
    StreamLegacyRuntimeBoundaryInventoryEntry, ...
] = ()


def legacy_stream_surface_inventory() -> (
    tuple[StreamLegacySurfaceInventoryEntry, ...]
):
    return _LEGACY_STREAM_SURFACE_INVENTORY


def legacy_stream_classifier_inventory() -> (
    tuple[StreamLegacyClassifierInventoryEntry, ...]
):
    return _LEGACY_STREAM_CLASSIFIER_INVENTORY


def legacy_stream_runtime_boundary_inventory() -> (
    tuple[StreamLegacyRuntimeBoundaryInventoryEntry, ...]
):
    return _LEGACY_STREAM_RUNTIME_BOUNDARY_INVENTORY


def classify_legacy_stream_surface(
    surface: StreamLegacySurface,
) -> StreamLegacySurfaceInventoryEntry:
    assert isinstance(surface, StreamLegacySurface)
    for entry in _LEGACY_STREAM_SURFACE_INVENTORY:
        if entry.surface is surface:
            return entry
    raise StreamValidationError("unknown legacy stream surface")


def classify_legacy_stream_classifier(
    module: str,
    qualname: str,
) -> StreamLegacyClassifierInventoryEntry:
    _assert_non_empty_string(module, "module")
    _assert_non_empty_string(qualname, "qualname")
    for entry in _LEGACY_STREAM_CLASSIFIER_INVENTORY:
        if entry.module == module and entry.qualname == qualname:
            return entry
    raise StreamValidationError("unknown legacy stream classifier")


def classify_legacy_stream_runtime_boundary(
    module: str,
    qualname: str,
) -> StreamLegacyRuntimeBoundaryInventoryEntry:
    _assert_non_empty_string(module, "module")
    _assert_non_empty_string(qualname, "qualname")
    for entry in _LEGACY_STREAM_RUNTIME_BOUNDARY_INVENTORY:
        if entry.module == module and entry.qualname == qualname:
            return entry
    raise StreamValidationError("unknown legacy stream runtime boundary")


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
    answer_boundary = _TextChannelBoundaryState()
    reasoning_boundary = _TextChannelBoundaryState()
    reasoning_segments = _CanonicalReasoningValidationState()
    usage_completed = False
    tool_call_states: dict[str, _ToolCallBoundaryState] = {}
    tool_execution_states: dict[str, _ToolExecutionBoundaryState] = {}
    model_continuation_states: dict[str, _ModelContinuationBoundaryState] = {}

    for index, item in enumerate(result):
        _validate_stream_start(item, index == 0)
        _validate_sequence_identity(item, session_id, run_id, turn_id)
        last_sequence = _validate_sequence_order(item, last_sequence)
        _validate_parent_sequence(item)
        _observe_canonical_reasoning_segment(reasoning_segments, item)

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

        if item.kind is StreamItemKind.USAGE_COMPLETED:
            if usage_completed:
                raise StreamValidationError("duplicate completed usage item")
            _validate_open_channel_boundaries_closed(
                answer_boundary,
                reasoning_boundary,
                tool_call_states,
                tool_execution_states,
                model_continuation_states,
            )
            usage_completed = True
        else:
            _validate_post_final_usage_item(item, usage_completed)

        _validate_answer_boundary(item, answer_boundary)
        _validate_reasoning_boundary(item, reasoning_boundary)
        _validate_tool_call_boundary(item, tool_call_states)
        _validate_tool_execution_boundary(item, tool_execution_states)

        if outcome is not None:
            if (
                outcome is StreamTerminalOutcome.COMPLETED
                and not usage_completed
                and item.usage is None
            ):
                raise StreamValidationError(
                    "completed stream missing final usage"
                )
            _validate_open_channel_boundaries_closed(
                answer_boundary,
                reasoning_boundary,
                tool_call_states,
                tool_execution_states,
                model_continuation_states,
            )
            terminal = outcome

        _validate_model_continuation_boundary(item, model_continuation_states)

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
    state: _TextChannelBoundaryState,
) -> None:
    _validate_text_channel_boundary(
        item,
        state,
        delta_kind=StreamItemKind.ANSWER_DELTA,
        done_kind=StreamItemKind.ANSWER_DONE,
        channel_name="answer",
    )


@dataclass(slots=True)
class _CanonicalReasoningValidationState:
    segments: StreamReasoningSegmentState = field(
        default_factory=StreamReasoningSegmentState
    )
    continuation_id: str | None = None
    initialized: bool = False


def _observe_canonical_reasoning_segment(
    state: _CanonicalReasoningValidationState,
    item: CanonicalStreamItem,
) -> None:
    if item.kind is not StreamItemKind.REASONING_DELTA:
        state.segments.complete_segment()
        return
    assert item.reasoning_representation is not None
    assert item.segment_instance_ordinal is not None
    continuation_id = item.correlation.model_continuation_id
    if not state.initialized:
        state.continuation_id = continuation_id
        state.initialized = True
    elif continuation_id is not None and (
        continuation_id != state.continuation_id
    ):
        state.segments = StreamReasoningSegmentState()
        state.continuation_id = continuation_id
    state.segments.observe_delta(
        item.reasoning_representation,
        item.correlation,
        item.segment_instance_ordinal,
        follows_completion=(
            item.metadata.get(REASONING_SEGMENT_BOUNDARY_METADATA_KEY)
            == "completed"
        ),
    )


def _validate_reasoning_boundary(
    item: CanonicalStreamItem,
    state: _TextChannelBoundaryState,
) -> None:
    _validate_text_channel_boundary(
        item,
        state,
        delta_kind=StreamItemKind.REASONING_DELTA,
        done_kind=StreamItemKind.REASONING_DONE,
        channel_name="reasoning",
    )


def _validate_text_channel_boundary(
    item: CanonicalStreamItem,
    state: _TextChannelBoundaryState,
    *,
    delta_kind: StreamItemKind,
    done_kind: StreamItemKind,
    channel_name: str,
) -> None:
    if item.kind is delta_kind:
        if state.done:
            raise StreamValidationError(
                f"{channel_name} item emitted after {channel_name} done"
            )
        state.started = True
        return
    if item.kind is done_kind:
        if not state.started:
            raise StreamValidationError(f"{channel_name} done before content")
        if state.done:
            raise StreamValidationError(f"duplicate {channel_name} done item")
        state.done = True


def _validate_tool_call_boundary(
    item: CanonicalStreamItem,
    states: dict[str, _ToolCallBoundaryState],
) -> None:
    if item.kind not in _TOOL_CALL_KINDS:
        return
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    state = states.setdefault(tool_call_id, _ToolCallBoundaryState())
    state.started = True
    if state.done:
        raise StreamValidationError(
            "tool-call item emitted after tool-call done"
        )
    if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        if state.ready:
            raise StreamValidationError(
                "tool-call argument emitted after ready"
            )
    elif item.kind is StreamItemKind.TOOL_CALL_READY:
        if state.ready:
            raise StreamValidationError("duplicate tool-call ready item")
        state.ready = True
    elif item.kind is StreamItemKind.TOOL_CALL_DONE:
        if not state.ready and not _is_marked_tool_call_terminal_close(item):
            raise StreamValidationError("tool-call done before ready")
        state.done = True


def _is_marked_tool_call_terminal_close(item: CanonicalStreamItem) -> bool:
    close_reason = item.metadata.get("tool_call.close_reason")
    return close_reason in {"cancelled", "error", "malformed"}


def _validate_tool_execution_boundary(
    item: CanonicalStreamItem,
    states: dict[str, _ToolExecutionBoundaryState],
) -> None:
    if item.channel is not StreamChannel.TOOL_EXECUTION:
        return
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    state = states.setdefault(tool_call_id, _ToolExecutionBoundaryState())
    if state.terminal_kind is not None:
        if item.kind in _TOOL_EXECUTION_TERMINAL_KINDS:
            raise StreamValidationError("duplicate tool execution terminal")
        raise StreamValidationError(
            "tool execution item emitted after terminal item"
        )
    if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
        if state.started:
            raise StreamValidationError("duplicate tool execution start")
        state.started = True
        return
    if not state.started:
        if item.kind in _TOOL_EXECUTION_TERMINAL_KINDS:
            raise StreamValidationError("tool execution terminal before start")
        raise StreamValidationError("tool execution item before start")
    if item.kind in _TOOL_EXECUTION_TERMINAL_KINDS:
        state.terminal_kind = item.kind


def _validate_model_continuation_boundary(
    item: CanonicalStreamItem,
    states: dict[str, _ModelContinuationBoundaryState],
) -> None:
    if item.kind not in _MODEL_CONTINUATION_KINDS:
        return
    continuation_id = item.correlation.model_continuation_id
    if continuation_id is None:
        raise StreamValidationError(
            "model continuation item missing model_continuation_id"
        )
    state = states.setdefault(
        continuation_id, _ModelContinuationBoundaryState()
    )
    if state.terminal_kind is not None:
        if item.kind in _MODEL_CONTINUATION_TERMINAL_KINDS:
            raise StreamValidationError(
                "duplicate model continuation terminal"
            )
        raise StreamValidationError(
            "model continuation item emitted after terminal item"
        )
    if item.kind is StreamItemKind.MODEL_CONTINUATION_STARTED:
        if state.started:
            raise StreamValidationError("duplicate model continuation start")
        state.started = True
        return
    if not state.started:
        raise StreamValidationError("model continuation terminal before start")
    state.terminal_kind = item.kind


def _validate_open_channel_boundaries_closed(
    answer_boundary: _TextChannelBoundaryState,
    reasoning_boundary: _TextChannelBoundaryState,
    tool_call_states: dict[str, _ToolCallBoundaryState],
    tool_execution_states: dict[str, _ToolExecutionBoundaryState],
    model_continuation_states: dict[str, _ModelContinuationBoundaryState],
) -> None:
    if answer_boundary.started and not answer_boundary.done:
        raise StreamValidationError("answer channel missing done")
    if reasoning_boundary.started and not reasoning_boundary.done:
        raise StreamValidationError("reasoning channel missing done")
    for tool_call_id, state in tool_call_states.items():
        if not state.started or state.done:
            continue
        if not state.ready:
            raise StreamValidationError(
                f"tool call {tool_call_id} missing ready"
            )
        raise StreamValidationError(f"tool call {tool_call_id} missing done")
    for tool_call_id, execution_state in tool_execution_states.items():
        if execution_state.started and execution_state.terminal_kind is None:
            raise StreamValidationError(
                f"tool execution {tool_call_id} missing terminal"
            )
    for (
        continuation_id,
        continuation_state,
    ) in model_continuation_states.items():
        if (
            continuation_state.started
            and continuation_state.terminal_kind is None
        ):
            raise StreamValidationError(
                f"model continuation {continuation_id} missing terminal"
            )


@dataclass(slots=True)
class _RetainedReasoningSegment:
    representation: StreamReasoningRepresentation
    segment_instance_ordinal: int
    provider_item_id: str | None
    output_index: int | None
    summary_index: int | None
    continuation_id: str | None
    chunks: deque[str] = field(default_factory=deque)
    characters: int = 0
    utf8_bytes: int = 0
    separator_after: str = ""
    closed: bool = False
    leading_partial: bool = False
    _materialized_text: str | None = None
    materialization_count: int = 0

    @property
    def identity(self) -> tuple[object, ...]:
        return (
            self.representation,
            self.segment_instance_ordinal,
            self.provider_item_id,
            self.output_index,
            self.summary_index,
            self.continuation_id,
        )

    def append(self, text: str) -> tuple[int, int]:
        assert isinstance(text, str)
        assert text
        encoded_length = len(text.encode("utf-8"))
        self.chunks.append(text)
        self.characters += len(text)
        self.utf8_bytes += encoded_length
        self._materialized_text = None
        return len(text), encoded_length

    def materialize(self) -> str:
        if self._materialized_text is None:
            self.materialization_count += 1
            self._materialized_text = "".join(self.chunks)
            self.chunks.clear()
            if self._materialized_text:
                self.chunks.append(self._materialized_text)
        return self._materialized_text

    def trailing_line_feeds(self) -> int:
        count = 0
        for chunk in reversed(self.chunks):
            exhausted = True
            for character in reversed(chunk):
                if not character.isspace():
                    exhausted = False
                    break
                if character == "\n":
                    count += 1
            if not exhausted:
                break
        return count

    def trim_prefix(
        self,
        minimum_characters: int,
        minimum_utf8_bytes: int,
    ) -> tuple[int, int]:
        assert minimum_characters >= 0
        assert minimum_utf8_bytes >= 0
        removed_characters = 0
        removed_utf8_bytes = 0
        while self.chunks and (
            removed_characters < minimum_characters
            or removed_utf8_bytes < minimum_utf8_bytes
        ):
            chunk = self.chunks.popleft()
            position = 0
            while position < len(chunk) and (
                removed_characters < minimum_characters
                or removed_utf8_bytes < minimum_utf8_bytes
            ):
                character = chunk[position]
                removed_characters += 1
                removed_utf8_bytes += len(character.encode("utf-8"))
                position += 1
            if position < len(chunk):
                self.chunks.appendleft(chunk[position:])
        if removed_characters:
            self.characters -= removed_characters
            self.utf8_bytes -= removed_utf8_bytes
            self.leading_partial = True
            self._materialized_text = None
        return removed_characters, removed_utf8_bytes


class StreamReasoningSegmentAccumulator:
    """Retain one bounded structured reasoning tail."""

    def __init__(
        self,
        *,
        segment_limit: int,
        character_limit: int,
        utf8_byte_limit: int,
    ) -> None:
        for field_name, value in (
            ("segment_limit", segment_limit),
            ("character_limit", character_limit),
            ("utf8_byte_limit", utf8_byte_limit),
        ):
            _assert_non_negative_int(value, field_name)
        self._segment_limit = segment_limit
        self._character_limit = character_limit
        self._utf8_byte_limit = utf8_byte_limit
        self._segments: deque[_RetainedReasoningSegment] = deque()
        self._active: _RetainedReasoningSegment | None = None
        self._characters = 0
        self._utf8_bytes = 0
        self._dropped_segments = 0
        self._dropped_characters = 0
        self._dropped_utf8_bytes = 0
        self._terminal_outcome: StreamTerminalOutcome | None = None

    @property
    def segments(self) -> tuple[StreamReasoningSegment, ...]:
        status, completed = self._public_completion()
        return tuple(
            StreamReasoningSegment(
                representation=segment.representation,
                segment_instance_ordinal=(segment.segment_instance_ordinal),
                text=segment.materialize(),
                completed=completed,
                status=status,
                terminal_outcome=self._terminal_outcome,
                provider_item_id=segment.provider_item_id,
                output_index=segment.output_index,
                summary_index=segment.summary_index,
                continuation_id=segment.continuation_id,
            )
            for segment in self._segments
            if segment.characters
        )

    @property
    def text(self) -> str:
        parts: list[str] = []
        for segment in self._segments:
            if not segment.characters:
                continue
            parts.append(segment.materialize())
            if segment.separator_after:
                parts.append(segment.separator_after)
        return "".join(parts)

    @property
    def truncation(self) -> StreamReasoningTruncation:
        first = next(
            (segment for segment in self._segments if segment.characters),
            None,
        )
        return StreamReasoningTruncation(
            truncated=bool(
                self._dropped_segments
                or self._dropped_characters
                or self._dropped_utf8_bytes
            ),
            dropped_segments=self._dropped_segments,
            dropped_characters=self._dropped_characters,
            dropped_utf8_bytes=self._dropped_utf8_bytes,
            leading_segment_partial=(
                first.leading_partial if first is not None else False
            ),
        )

    @property
    def character_count(self) -> int:
        return self._characters

    @property
    def utf8_byte_count(self) -> int:
        return self._utf8_bytes

    @property
    def materialization_count(self) -> int:
        """Return how many retained segment strings were joined."""
        return sum(segment.materialization_count for segment in self._segments)

    def observe(
        self,
        item: CanonicalStreamItem | StreamConsumerProjection,
    ) -> None:
        """Observe one canonical item or consumer projection."""
        assert isinstance(
            item, (CanonicalStreamItem, StreamConsumerProjection)
        )
        if item.kind is StreamItemKind.REASONING_DELTA:
            self._add_delta(item)
            return
        if item.kind is StreamItemKind.STREAM_CLOSED:
            return
        self._close_active()
        outcome = stream_terminal_outcome_for_kind(item.kind)
        if outcome is not None:
            self._terminal_outcome = outcome

    def _add_delta(
        self,
        item: CanonicalStreamItem | StreamConsumerProjection,
    ) -> None:
        assert item.text_delta is not None
        assert item.reasoning_representation is not None
        assert item.segment_instance_ordinal is not None
        identity = (
            item.reasoning_representation,
            item.segment_instance_ordinal,
            item.correlation.protocol_item_id,
            item.correlation.provider_output_index,
            item.correlation.provider_summary_index,
            item.correlation.model_continuation_id,
        )
        active = self._active
        if active is None or active.closed or active.identity != identity:
            self._close_active()
            active = _RetainedReasoningSegment(
                representation=item.reasoning_representation,
                segment_instance_ordinal=item.segment_instance_ordinal,
                provider_item_id=item.correlation.protocol_item_id,
                output_index=item.correlation.provider_output_index,
                summary_index=item.correlation.provider_summary_index,
                continuation_id=item.correlation.model_continuation_id,
            )
            self._append_separator_before(item.text_delta)
            self._segments.append(active)
            self._active = active
        characters, utf8_bytes = active.append(item.text_delta)
        self._characters += characters
        self._utf8_bytes += utf8_bytes
        self._enforce_limits()

    def _append_separator_before(
        self,
        text: str,
    ) -> None:
        previous = next(
            (
                candidate
                for candidate in reversed(self._segments)
                if candidate.characters
            ),
            None,
        )
        if previous is None:
            return
        leading = 0
        for character in text:
            if not character.isspace():
                break
            if character == "\n":
                leading += 1
        missing = max(0, 2 - previous.trailing_line_feeds() - leading)
        if not missing:
            return
        separator = "\n" * missing
        previous.separator_after = separator
        self._characters += missing
        self._utf8_bytes += missing

    def _close_active(self) -> None:
        active = self._active
        if active is None:
            return
        active.closed = True
        self._active = None
        if not active.characters:
            self._remove_empty_last_segment(active)
        self._enforce_limits()

    def _remove_empty_last_segment(
        self,
        segment: _RetainedReasoningSegment,
    ) -> None:
        assert not segment.characters
        assert self._segments and self._segments[-1] is segment
        self._segments.pop()
        self._dropped_segments += 1
        if self._segments:
            previous = self._segments[-1]
            separator = previous.separator_after
            if separator:
                previous.separator_after = ""
                self._characters -= len(separator)
                self._utf8_bytes -= len(separator.encode("utf-8"))
                self._record_dropped_text(separator)

    def _enforce_limits(self) -> None:
        while len(self._segments) > self._segment_limit:
            oldest = self._segments[0]
            if not oldest.closed:
                self._trim_all_active_text(oldest)
                break
            self._drop_oldest_segment()

        while self._over_text_limit() and self._segments:
            oldest = self._segments[0]
            if oldest.closed and len(self._segments) > 1:
                self._drop_oldest_segment()
                continue
            required_characters = max(
                0, self._characters - self._character_limit
            )
            required_utf8_bytes = max(
                0, self._utf8_bytes - self._utf8_byte_limit
            )
            removed_characters, removed_utf8_bytes = oldest.trim_prefix(
                required_characters,
                required_utf8_bytes,
            )
            if not removed_characters:
                break
            self._characters -= removed_characters
            self._utf8_bytes -= removed_utf8_bytes
            self._dropped_characters += removed_characters
            self._dropped_utf8_bytes += removed_utf8_bytes
            if not oldest.characters and oldest.closed:
                self._drop_empty_oldest_segment()

    def _trim_all_active_text(
        self,
        segment: _RetainedReasoningSegment,
    ) -> None:
        assert not segment.closed
        removed_characters, removed_utf8_bytes = segment.trim_prefix(
            segment.characters,
            segment.utf8_bytes,
        )
        self._characters -= removed_characters
        self._utf8_bytes -= removed_utf8_bytes
        self._dropped_characters += removed_characters
        self._dropped_utf8_bytes += removed_utf8_bytes

    def _drop_oldest_segment(self) -> None:
        segment = self._segments.popleft()
        assert segment.closed
        text_characters = segment.characters
        text_utf8_bytes = segment.utf8_bytes
        separator = segment.separator_after
        separator_characters = len(separator)
        separator_utf8_bytes = len(separator.encode("utf-8"))
        self._characters -= text_characters + separator_characters
        self._utf8_bytes -= text_utf8_bytes + separator_utf8_bytes
        self._dropped_segments += 1
        self._dropped_characters += text_characters + separator_characters
        self._dropped_utf8_bytes += text_utf8_bytes + separator_utf8_bytes

    def _drop_empty_oldest_segment(self) -> None:
        segment = self._segments[0]
        assert segment.closed
        assert not segment.characters
        self._drop_oldest_segment()

    def _record_dropped_text(self, text: str) -> None:
        self._dropped_characters += len(text)
        self._dropped_utf8_bytes += len(text.encode("utf-8"))

    def _over_text_limit(self) -> bool:
        return (
            self._characters > self._character_limit
            or self._utf8_bytes > self._utf8_byte_limit
        )

    def _public_completion(
        self,
    ) -> tuple[StreamReasoningSegmentStatus, bool]:
        if self._terminal_outcome is StreamTerminalOutcome.COMPLETED:
            return StreamReasoningSegmentStatus.COMPLETED, True
        if self._terminal_outcome in (
            StreamTerminalOutcome.ERRORED,
            StreamTerminalOutcome.CANCELLED,
        ):
            return StreamReasoningSegmentStatus.INCOMPLETE, False
        return StreamReasoningSegmentStatus.IN_PROGRESS, False


class CanonicalStreamAccumulator:
    def __init__(
        self,
        *,
        retention_policy: StreamRetentionPolicy | None = None,
    ) -> None:
        if retention_policy is None:
            retention_policy = StreamRetentionPolicy()
        assert isinstance(retention_policy, StreamRetentionPolicy)
        self._retention_policy = retention_policy
        self._items: list[CanonicalStreamItem] = []
        self._answer_text: list[str] = []
        self._reasoning = StreamReasoningSegmentAccumulator(
            segment_limit=retention_policy.reasoning_segment_limit,
            character_limit=retention_policy.reasoning_character_limit,
            utf8_byte_limit=retention_policy.reasoning_text_byte_limit,
        )
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
        self._terminal_item: CanonicalStreamItem | None = None
        self._closed = False
        self._answer_boundary = _TextChannelBoundaryState()
        self._reasoning_boundary = _TextChannelBoundaryState()
        self._reasoning_segments = _CanonicalReasoningValidationState()
        self._usage_completed = False
        self._tool_call_states: dict[str, _ToolCallBoundaryState] = {}
        self._tool_execution_states: dict[str, _ToolExecutionBoundaryState] = (
            {}
        )
        self._model_continuation_states: dict[
            str, _ModelContinuationBoundaryState
        ] = {}

    @property
    def retention_policy(self) -> StreamRetentionPolicy:
        return self._retention_policy

    @property
    def items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._items)

    @property
    def answer_text(self) -> str:
        return "".join(self._answer_text)

    @property
    def reasoning_text(self) -> str:
        return self._reasoning.text

    @property
    def reasoning_segments(self) -> tuple[StreamReasoningSegment, ...]:
        return self._reasoning.segments

    @property
    def reasoning_truncation(self) -> StreamReasoningTruncation:
        return self._reasoning.truncation

    @property
    def retained_reasoning_characters(self) -> int:
        return self._reasoning.character_count

    @property
    def retained_reasoning_utf8_bytes(self) -> int:
        return self._reasoning.utf8_byte_count

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
    def terminal_item(self) -> CanonicalStreamItem | None:
        return self._terminal_item

    @property
    def closed(self) -> bool:
        return self._closed

    def add(self, item: CanonicalStreamItem) -> None:
        assert isinstance(item, CanonicalStreamItem)
        self._validate_next(item)
        self._append_retained(
            self._items,
            item,
            self._retention_policy.accumulator_item_limit,
        )
        self._accumulate(item)

    def add_many(
        self,
        items: Iterable[CanonicalStreamItem],
    ) -> "CanonicalStreamAccumulator":
        for item in items:
            self.add(item)
        return self

    def validate_complete(self) -> tuple[CanonicalStreamItem, ...]:
        if self._session_id is None:
            raise StreamValidationError(
                "stream must contain at least one item"
            )
        if self._terminal_outcome is None:
            raise StreamValidationError("stream missing terminal outcome")
        _validate_open_channel_boundaries_closed(
            self._answer_boundary,
            self._reasoning_boundary,
            self._tool_call_states,
            self._tool_execution_states,
            self._model_continuation_states,
        )
        return self.items

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

        self._validate_usage(item)
        _observe_canonical_reasoning_segment(self._reasoning_segments, item)
        _validate_answer_boundary(item, self._answer_boundary)
        _validate_reasoning_boundary(item, self._reasoning_boundary)
        _validate_tool_call_boundary(item, self._tool_call_states)
        _validate_tool_execution_boundary(item, self._tool_execution_states)

        if outcome is not None:
            if (
                outcome is StreamTerminalOutcome.COMPLETED
                and not self._usage_completed
                and item.usage is None
            ):
                raise StreamValidationError(
                    "completed stream missing final usage"
                )
            _validate_open_channel_boundaries_closed(
                self._answer_boundary,
                self._reasoning_boundary,
                self._tool_call_states,
                self._tool_execution_states,
                self._model_continuation_states,
            )
            self._terminal_outcome = outcome
            self._terminal_item = item
        _validate_model_continuation_boundary(
            item, self._model_continuation_states
        )
        self._last_sequence = last_sequence

    def _validate_usage(self, item: CanonicalStreamItem) -> None:
        if item.kind is StreamItemKind.USAGE_COMPLETED:
            if self._usage_completed:
                raise StreamValidationError("duplicate completed usage item")
            _validate_open_channel_boundaries_closed(
                self._answer_boundary,
                self._reasoning_boundary,
                self._tool_call_states,
                self._tool_execution_states,
                self._model_continuation_states,
            )
            self._usage_completed = True
        else:
            _validate_post_final_usage_item(item, self._usage_completed)

    def _accumulate(self, item: CanonicalStreamItem) -> None:
        self._reasoning.observe(item)
        if item.kind is StreamItemKind.ANSWER_DELTA:
            assert item.text_delta is not None
            self._answer_text.append(item.text_delta)
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
            self._append_retained(
                self._diagnostics,
                item,
                self._retention_policy.replay_history_item_limit,
            )
        if item.channel is StreamChannel.FLOW:
            self._append_retained(
                self._flow_items,
                item,
                self._retention_policy.flow_history_item_limit,
            )
        if item.channel is StreamChannel.USAGE:
            self._append_retained(
                self._usage_items,
                item,
                self._retention_policy.metrics_history_item_limit,
            )
            self._final_usage = item.usage
        if (
            item.kind is StreamItemKind.STREAM_COMPLETED
            and item.usage is not None
        ):
            self._final_usage = item.usage
        if item.channel is StreamChannel.CONTROL:
            self._append_retained(
                self._control_items,
                item,
                self._retention_policy.replay_history_item_limit,
            )
        if item.kind is StreamItemKind.STREAM_CLOSED:
            self._closed = True

    def _append_retained(
        self,
        items: list[CanonicalStreamItem],
        item: CanonicalStreamItem,
        limit: int,
    ) -> None:
        assert limit >= 0
        if limit == 0:
            return
        items.append(item)
        overflow = len(items) - limit
        if overflow > 0:
            del items[:overflow]


@dataclass(slots=True)
class StreamProjectionState:
    stream_session_id: str
    run_id: str
    turn_id: str
    accumulator: CanonicalStreamAccumulator = field(
        default_factory=CanonicalStreamAccumulator
    )
    has_canonical_items: bool = False
    accumulate: bool = True

    def __post_init__(self) -> None:
        for field_name, value in (
            ("stream_session_id", self.stream_session_id),
            ("run_id", self.run_id),
            ("turn_id", self.turn_id),
        ):
            _assert_non_empty_string(value, field_name)
        assert isinstance(self.accumulator, CanonicalStreamAccumulator)
        assert isinstance(self.has_canonical_items, bool)
        assert isinstance(self.accumulate, bool)

    def project(
        self,
        item: object,
        sequence: int,
        *,
        unsupported_message: str,
    ) -> StreamConsumerProjection:
        self._validate_projection_arguments(sequence, unsupported_message)
        return self._project_one(item, unsupported_message)

    def project_many(
        self,
        item: object,
        sequence: int,
        *,
        unsupported_message: str,
    ) -> tuple[StreamConsumerProjection, ...]:
        self._validate_projection_arguments(sequence, unsupported_message)
        return (self._project_one(item, unsupported_message),)

    def _validate_projection_arguments(
        self,
        sequence: int,
        unsupported_message: str,
    ) -> None:
        assert isinstance(sequence, int), "sequence must be an integer"
        assert sequence >= 0, "sequence must not be negative"
        _assert_non_empty_string(unsupported_message, "unsupported_message")

    def _project_one(
        self,
        item: object,
        unsupported_message: str,
    ) -> StreamConsumerProjection:
        if isinstance(item, CanonicalStreamItem):
            return self._project_canonical_item(item)
        if isinstance(item, StreamConsumerProjection):
            return self._project_consumer_projection(item)
        raise StreamValidationError(unsupported_message)

    def validate_complete(self) -> None:
        if self.has_canonical_items and self.accumulate:
            self.accumulator.validate_complete()

    def terminal_projection(self) -> StreamConsumerProjection | None:
        for item in reversed(self.accumulator.items):
            if item.is_stream_terminal:
                return project_canonical_stream_item(item)
        if self.accumulator.terminal_item is not None:
            return project_canonical_stream_item(
                self.accumulator.terminal_item
            )
        return None

    def _project_canonical_item(
        self,
        item: CanonicalStreamItem,
    ) -> StreamConsumerProjection:
        self.has_canonical_items = True
        if self.accumulate:
            self.accumulator.add(item)
        return project_canonical_stream_item(item)

    def _project_consumer_projection(
        self,
        projection: StreamConsumerProjection,
    ) -> StreamConsumerProjection:
        self.has_canonical_items = True
        if self.accumulate:
            self.accumulator.add(
                canonical_item_from_consumer_projection(projection)
            )
        return projection


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
    except StreamConsumerCancellation:
        await close_provider_stream()
        raise CancelledError() from None
    except StreamConsumerClosure:
        await close_provider_stream()
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


@dataclass(frozen=True, slots=True)
class _LocalTextSourceFragment:
    text: str
    metadata: dict[str, LooseJsonValue] = field(default_factory=dict)
    source_index: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.text, str)
        assert self.text
        assert isinstance(self.metadata, dict)
        if self.source_index is not None:
            _assert_non_negative_int(self.source_index, "source_index")


@dataclass(slots=True)
class _LocalTextStreamParser:
    _reasoning_start_tag: str = "<think>"
    _reasoning_end_tag: str = "</think>"
    _tool_start_tag: str = "<tool_call"
    _tool_end_tag: str = "</tool_call>"
    _reasoning_buffer: str = ""
    _reasoning_fragments: deque[_LocalTextSourceFragment] = field(
        default_factory=deque
    )
    _reasoning_active: bool = False
    _reasoning_done_pending: bool = False
    _reasoning_segments: StreamReasoningSegmentState = field(
        default_factory=StreamReasoningSegmentState
    )
    _tool_buffer: str = ""
    _tool_fragments: deque[_LocalTextSourceFragment] = field(
        default_factory=deque
    )
    _tool_state: str = "outside"
    _tool_call_id: str | None = None
    _tool_call_index: int = 0
    _tool_name: str | None = None
    _tool_argument_deltas: list[str] = field(default_factory=list)
    _next_source_index: int = 0

    def push(
        self,
        token: str,
        metadata: dict[str, LooseJsonValue] | None = None,
    ) -> tuple[StreamProviderEvent, ...]:
        assert isinstance(token, str)
        assert metadata is None or isinstance(metadata, dict)
        if not token:
            return ()
        source_index = None
        if metadata is not None:
            source_index = self._next_source_index
            self._next_source_index += 1
        fragment = _LocalTextSourceFragment(
            text=token,
            metadata=dict(metadata or {}),
            source_index=source_index,
        )
        return self._push_reasoning(fragment)

    def flush(self) -> tuple[StreamProviderEvent, ...]:
        events: list[StreamProviderEvent] = []
        if self._reasoning_buffer:
            fragments = self._take_reasoning_fragments(
                len(self._reasoning_buffer)
            )
            self._reasoning_buffer = ""
            if self._reasoning_active:
                self._append_reasoning_fragments(events, fragments)
            else:
                self._append_pending_reasoning_done(events)
                events.extend(self._push_tool_fragments(fragments))
        if self._reasoning_active:
            self._reasoning_active = False
            self._append_reasoning_done(events)
        self._append_pending_reasoning_done(events)
        events.extend(self._flush_tool())
        return tuple(events)

    def _push_reasoning(
        self, fragment: _LocalTextSourceFragment
    ) -> tuple[StreamProviderEvent, ...]:
        self._reasoning_buffer += fragment.text
        self._append_source_fragment(self._reasoning_fragments, fragment)
        events: list[StreamProviderEvent] = []
        while self._reasoning_buffer:
            if self._reasoning_active:
                end_index = self._reasoning_buffer.find(
                    self._reasoning_end_tag
                )
                if end_index != -1:
                    self._append_reasoning_fragments(
                        events,
                        self._take_reasoning_fragments(end_index),
                    )
                    self._discard_reasoning_fragments(
                        len(self._reasoning_end_tag)
                    )
                    self._reasoning_buffer = self._reasoning_buffer[
                        end_index + len(self._reasoning_end_tag) :
                    ]
                    self._reasoning_active = False
                    self._reasoning_segments.complete_segment()
                    self._reasoning_done_pending = True
                    continue
                flush_length = self._flushable_prefix_length(
                    self._reasoning_buffer,
                    self._reasoning_end_tag,
                )
                if not flush_length:
                    break
                self._append_reasoning_fragments(
                    events,
                    self._take_reasoning_fragments(flush_length),
                )
                self._reasoning_buffer = self._reasoning_buffer[flush_length:]
                continue

            start_index = self._reasoning_buffer.find(
                self._reasoning_start_tag
            )
            if start_index != -1:
                visible_prefix = self._reasoning_buffer[:start_index]
                visible_fragments = self._take_reasoning_fragments(start_index)
                continuing_reasoning = self._reasoning_done_pending
                if continuing_reasoning and (
                    not visible_prefix or visible_prefix.isspace()
                ):
                    self._append_reasoning_fragments(events, visible_fragments)
                else:
                    self._append_pending_reasoning_done(events)
                    events.extend(self._push_tool_fragments(visible_fragments))
                self._discard_reasoning_fragments(
                    len(self._reasoning_start_tag)
                )
                self._reasoning_buffer = self._reasoning_buffer[
                    start_index + len(self._reasoning_start_tag) :
                ]
                self._reasoning_active = True
                self._reasoning_done_pending = False
                continue
            if (
                self._reasoning_done_pending
                and self._awaiting_repeated_reasoning_start()
            ):
                break
            flush_length = self._flushable_prefix_length(
                self._reasoning_buffer,
                self._reasoning_start_tag,
            )
            if not flush_length:
                break
            self._append_pending_reasoning_done(events)
            events.extend(
                self._push_tool_fragments(
                    self._take_reasoning_fragments(flush_length)
                )
            )
            self._reasoning_buffer = self._reasoning_buffer[flush_length:]
        return tuple(events)

    def _push_tool_fragments(
        self, fragments: Iterable[_LocalTextSourceFragment]
    ) -> tuple[StreamProviderEvent, ...]:
        fragments = tuple(fragments)
        if not fragments:
            return ()
        self._tool_buffer += "".join(fragment.text for fragment in fragments)
        for fragment in fragments:
            self._append_source_fragment(self._tool_fragments, fragment)
        events: list[StreamProviderEvent] = []
        while self._tool_buffer:
            if self._tool_state == "outside":
                start_index = self._tool_start_index()
                if start_index is not None:
                    self._append_answer_fragments(
                        events,
                        self._take_tool_fragments(start_index),
                    )
                    self._tool_buffer = self._tool_buffer[start_index:]
                    self._tool_state = "opening"
                    self._tool_call_id = self._next_tool_call_id()
                    continue
                flush_length = self._tool_flushable_prefix_length()
                if not flush_length:
                    break
                self._append_answer_fragments(
                    events,
                    self._take_tool_fragments(flush_length),
                )
                self._tool_buffer = self._tool_buffer[flush_length:]
                continue

            if self._tool_state == "opening":
                tag_end = self._tool_buffer.find(">")
                if tag_end == -1:
                    break
                opening_tag = self._tool_buffer[: tag_end + 1]
                self._tool_name = self._tool_name_from_opening_tag(opening_tag)
                self._discard_tool_fragments(tag_end + 1)
                self._tool_buffer = self._tool_buffer[tag_end + 1 :]
                self._tool_state = "body"
                continue

            end_index = self._tool_buffer.find(self._tool_end_tag)
            if end_index != -1:
                self._append_tool_argument_fragments(
                    events,
                    self._take_tool_fragments(end_index),
                )
                self._discard_tool_fragments(len(self._tool_end_tag))
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
            self._append_tool_argument_fragments(
                events,
                self._take_tool_fragments(flush_length),
            )
            self._tool_buffer = self._tool_buffer[flush_length:]
        return tuple(events)

    def _flush_tool(self) -> tuple[StreamProviderEvent, ...]:
        events: list[StreamProviderEvent] = []
        fragments = self._take_tool_fragments(len(self._tool_buffer))
        if self._tool_state == "outside":
            self._append_answer_fragments(events, fragments)
            self._tool_buffer = ""
            return tuple(events)

        self._append_tool_argument_fragments(events, fragments)
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

    def _append_answer_fragments(
        self,
        events: list[StreamProviderEvent],
        fragments: Iterable[_LocalTextSourceFragment],
    ) -> None:
        for fragment in fragments:
            self._reasoning_segments.complete_segment()
            events.append(
                StreamProviderEvent(
                    kind=StreamItemKind.ANSWER_DELTA,
                    text_delta=fragment.text,
                    metadata=dict(fragment.metadata),
                )
            )

    def _append_reasoning_fragments(
        self,
        events: list[StreamProviderEvent],
        fragments: Iterable[_LocalTextSourceFragment],
    ) -> None:
        for fragment in fragments:
            events.append(
                self._reasoning_delta(fragment.text, fragment.metadata)
            )

    def _append_reasoning_done(
        self,
        events: list[StreamProviderEvent],
    ) -> None:
        assert isinstance(events, list)
        self._reasoning_segments.complete_segment()

    def _append_pending_reasoning_done(
        self,
        events: list[StreamProviderEvent],
    ) -> None:
        if not self._reasoning_done_pending:
            return
        self._reasoning_done_pending = False
        self._append_reasoning_done(events)

    def _reasoning_delta(
        self,
        text: str,
        metadata: dict[str, LooseJsonValue] | None = None,
    ) -> StreamProviderEvent:
        representation = StreamReasoningRepresentation.NATIVE_TEXT
        follows_boundary = (
            self._reasoning_segments.next_allocation_follows_boundary
        )
        event_metadata = dict(metadata or {})
        if follows_boundary:
            event_metadata[REASONING_SEGMENT_BOUNDARY_METADATA_KEY] = (
                "completed"
            )
        return StreamProviderEvent(
            kind=StreamItemKind.REASONING_DELTA,
            text_delta=text,
            visibility=StreamVisibility.PRIVATE,
            reasoning_representation=representation,
            segment_instance_ordinal=self._reasoning_segments.allocate(
                representation
            ),
            metadata=event_metadata,
        )

    def _append_tool_argument_fragments(
        self,
        events: list[StreamProviderEvent],
        fragments: Iterable[_LocalTextSourceFragment],
    ) -> None:
        for fragment in fragments:
            self._reasoning_segments.complete_segment()
            assert self._tool_call_id is not None
            self._tool_argument_deltas.append(fragment.text)
            events.append(
                StreamProviderEvent(
                    kind=StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                    text_delta=fragment.text,
                    correlation=StreamItemCorrelation(
                        tool_call_id=self._tool_call_id
                    ),
                    metadata=dict(fragment.metadata),
                )
            )

    def _take_reasoning_fragments(
        self, length: int
    ) -> tuple[_LocalTextSourceFragment, ...]:
        return self._take_source_fragments(self._reasoning_fragments, length)

    def _discard_reasoning_fragments(self, length: int) -> None:
        self._take_reasoning_fragments(length)

    def _take_tool_fragments(
        self, length: int
    ) -> tuple[_LocalTextSourceFragment, ...]:
        return self._take_source_fragments(self._tool_fragments, length)

    def _discard_tool_fragments(self, length: int) -> None:
        self._take_tool_fragments(length)

    @staticmethod
    def _take_source_fragments(
        fragments: deque[_LocalTextSourceFragment],
        length: int,
    ) -> tuple[_LocalTextSourceFragment, ...]:
        _assert_non_negative_int(length, "length")
        remaining = length
        result: list[_LocalTextSourceFragment] = []
        while remaining:
            assert fragments, "local text provenance must match buffered text"
            fragment = fragments.popleft()
            if len(fragment.text) <= remaining:
                result.append(fragment)
                remaining -= len(fragment.text)
                continue
            result.append(
                _LocalTextSourceFragment(
                    text=fragment.text[:remaining],
                    metadata=dict(fragment.metadata),
                    source_index=fragment.source_index,
                )
            )
            fragments.appendleft(
                _LocalTextSourceFragment(
                    text=fragment.text[remaining:],
                    metadata=dict(fragment.metadata),
                    source_index=fragment.source_index,
                )
            )
            remaining = 0
        return tuple(result)

    @staticmethod
    def _append_source_fragment(
        fragments: deque[_LocalTextSourceFragment],
        fragment: _LocalTextSourceFragment,
    ) -> None:
        if (
            fragments
            and fragments[-1].source_index is None
            and fragment.source_index is None
        ):
            previous = fragments.pop()
            fragments.append(
                _LocalTextSourceFragment(
                    text=previous.text + fragment.text,
                )
            )
            return
        fragments.append(fragment)

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

    def _awaiting_repeated_reasoning_start(self) -> bool:
        stripped = self._reasoning_buffer.lstrip()
        if not stripped:
            return True
        return self._reasoning_start_tag.startswith(stripped)

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


@dataclass(slots=True)
class LocalTextStreamEventParser:
    _parser: _LocalTextStreamParser = field(
        default_factory=_LocalTextStreamParser
    )

    def push(
        self,
        text: str,
        metadata: dict[str, LooseJsonValue] | None = None,
    ) -> tuple[StreamProviderEvent, ...]:
        assert isinstance(text, str)
        assert metadata is None or isinstance(metadata, dict)
        return self._parser.push(text, metadata)

    def flush(self) -> tuple[StreamProviderEvent, ...]:
        return self._parser.flush()


def stream_token_metadata(
    *,
    token_id: int | None = None,
    probability: float | None = None,
    step: int | None = None,
    probability_distribution: str | None = None,
    candidates: Iterable[tuple[str, int | None, float | None]] | None = None,
    provider_name: str | None = None,
) -> dict[str, LooseJsonValue]:
    if token_id is not None:
        _assert_non_negative_int(token_id, "token_id")
    if probability is not None:
        assert isinstance(probability, int | float)
        assert not isinstance(probability, bool)
    if step is not None:
        _assert_non_negative_int(step, "step")
    if probability_distribution is not None:
        _assert_non_empty_string(
            probability_distribution, "probability_distribution"
        )
    if provider_name is not None:
        _assert_non_empty_string(provider_name, "provider_name")

    metadata: dict[str, LooseJsonValue] = {}
    if token_id is not None:
        metadata["token_id"] = token_id
    if probability is not None:
        metadata["probability"] = float(probability)
    if step is not None:
        metadata["step"] = step
    if probability_distribution is not None:
        metadata["probability_distribution"] = probability_distribution
    if candidates is not None:
        candidate_metadata: list[dict[str, LooseJsonValue]] = []
        for token, candidate_id, candidate_probability in candidates:
            _assert_non_empty_string(token, "candidate token")
            if candidate_id is not None:
                _assert_non_negative_int(candidate_id, "candidate_id")
            if candidate_probability is not None:
                assert isinstance(candidate_probability, int | float)
                assert not isinstance(candidate_probability, bool)
            candidate_entry: dict[str, LooseJsonValue] = {"token": token}
            if candidate_id is not None:
                candidate_entry["token_id"] = candidate_id
            if candidate_probability is not None:
                candidate_entry["probability"] = float(candidate_probability)
            candidate_metadata.append(candidate_entry)
        metadata["tokens"] = cast(LooseJsonValue, candidate_metadata)
    if provider_name is not None:
        metadata["provider_name"] = provider_name
    return metadata


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
    started: bool = False
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
    _reasoning_segments: StreamReasoningSegmentState = field(
        default_factory=StreamReasoningSegmentState
    )
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
        if (
            event.kind is StreamItemKind.REASONING_DELTA
            and event.text_delta == ""
        ):
            return ()
        self._reasoning_segments.observe(event)
        if event.kind is StreamItemKind.REASONING_DONE:
            return ()
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

        self._validate_event_after_final_usage(event)
        self._track_tool_call_state(event)
        item = self._item(
            kind=event.kind,
            text_delta=event.text_delta,
            data=event.data,
            usage=event.usage,
            correlation=event.correlation,
            visibility=event.visibility,
            reasoning_representation=event.reasoning_representation,
            segment_instance_ordinal=event.segment_instance_ordinal,
            metadata=event.metadata,
            provider_payload=event.provider_payload,
            provider_event_type=event.provider_event_type,
        )
        self._track_channel_boundary(item)
        return (item,)

    def _validate_event_after_final_usage(
        self,
        event: StreamProviderEvent,
    ) -> None:
        if not self._usage_completed:
            return
        if event.kind is StreamItemKind.STREAM_DIAGNOSTIC:
            return
        raise StreamValidationError(
            "semantic stream item emitted after final usage"
        )

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
        if isinstance(exc, StreamProviderAdapterError):
            error = exc.error
            return self._terminal(
                StreamItemKind.STREAM_ERRORED,
                provider_event=StreamProviderEvent(
                    kind=StreamItemKind.STREAM_ERRORED,
                    data=(
                        exc.safe_data
                        if exc.safe_data is not None
                        else {
                            "error_type": error.__class__.__name__,
                            "message": str(error),
                        }
                    ),
                    provider_payload=exc.provider_payload,
                    provider_event_type=exc.provider_event_type,
                ),
            )
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
        incomplete_tool_call_error = self._incomplete_tool_call_error()
        if incomplete_tool_call_error is not None:
            raise StreamValidationError(incomplete_tool_call_error)
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
        for tool_call_id, state in self._tool_call_states.items():
            if not state.started or state.done:
                continue
            metadata: dict[str, LooseJsonValue] = {}
            if state.malformed:
                metadata["tool_call.close_reason"] = "malformed"
            elif not state.ready:
                metadata["tool_call.close_reason"] = "error"
            item = self._item(
                kind=StreamItemKind.TOOL_CALL_DONE,
                correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
                metadata=metadata,
            )
            state.done = True
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
        reasoning_representation: StreamReasoningRepresentation | None = None,
        segment_instance_ordinal: int | None = None,
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
            correlation=(
                _EMPTY_STREAM_ITEM_CORRELATION
                if correlation is None
                else correlation
            ),
            text_delta=text_delta,
            data=data,
            usage=usage,
            terminal_outcome=terminal_outcome,
            visibility=visibility,
            reasoning_representation=reasoning_representation,
            segment_instance_ordinal=segment_instance_ordinal,
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
            state.started = True
        elif event.kind is StreamItemKind.TOOL_CALL_READY:
            if state.ready:
                raise StreamValidationError("duplicate tool-call ready item")
            state.started = True
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


class TextGenerationStream(AsyncIterator[CanonicalStreamItem], ABC):
    _generator: AsyncIterator[CanonicalStreamItem] | None = None

    @abstractmethod
    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[CanonicalStreamItem]:
        raise NotImplementedError()

    @abstractmethod
    async def __anext__(self) -> CanonicalStreamItem:
        raise NotImplementedError()

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
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
        _assert_non_empty_string(stream_session_id, "stream_session_id")
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(turn_id, "turn_id")
        if provider_family is not None:
            provider_family_value(provider_family)
        if capabilities is not None:
            assert isinstance(capabilities, StreamProviderCapabilities)
        assert isinstance(close_after_terminal, bool)
        return self.__aiter__()


class TextGenerationSingleStream(TextGenerationStream):
    _content: str
    _item_index: int = 0
    _provider_family: str | None = None
    _usage: object | None = None

    def __init__(
        self,
        content: str,
        *,
        provider_family: ProviderFamily | str | None = None,
        usage: object | None = None,
    ) -> None:
        assert isinstance(content, str)
        self._content = content
        self._provider_family = provider_family_value(provider_family)
        self._usage = usage

    @property
    def content(self) -> str:
        return self._content

    @property
    def provider_family(self) -> str | None:
        return self._provider_family

    @property
    def usage(self) -> object | None:
        return self._usage

    @property
    def canonical_items(self) -> tuple[CanonicalStreamItem, ...]:
        return self._canonical_items(
            stream_session_id="single-stream",
            run_id="single-run",
            turn_id="single-turn",
            provider_family=self._provider_family,
            capabilities=None,
            close_after_terminal=True,
        )

    def _canonical_items(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
        provider_family: ProviderFamily | str | None,
        capabilities: StreamProviderCapabilities | None,
        close_after_terminal: bool,
    ) -> tuple[CanonicalStreamItem, ...]:
        text = self._content
        usage = self._usage if self._usage is not None else {}
        normalized_provider_family = provider_family_value(provider_family)
        metadata: dict[str, LooseJsonValue] = {}
        if capabilities is not None:
            metadata["capabilities"] = cast(
                LooseJsonValue, capabilities.to_metadata()
            )
        items = [
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=0,
                kind=StreamItemKind.STREAM_STARTED,
                channel=StreamChannel.CONTROL,
                metadata=metadata,
                provider_family=normalized_provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=1,
                kind=StreamItemKind.ANSWER_DELTA,
                channel=StreamChannel.ANSWER,
                text_delta=text,
                provider_family=normalized_provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=2,
                kind=StreamItemKind.ANSWER_DONE,
                channel=StreamChannel.ANSWER,
                provider_family=normalized_provider_family,
            ),
            CanonicalStreamItem(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                sequence=3,
                kind=StreamItemKind.STREAM_COMPLETED,
                channel=StreamChannel.CONTROL,
                usage=cast(Any, usage),
                terminal_outcome=StreamTerminalOutcome.COMPLETED,
                provider_family=normalized_provider_family,
            ),
        ]
        if close_after_terminal:
            items.append(
                CanonicalStreamItem(
                    stream_session_id=stream_session_id,
                    run_id=run_id,
                    turn_id=turn_id,
                    sequence=4,
                    kind=StreamItemKind.STREAM_CLOSED,
                    channel=StreamChannel.CONTROL,
                    provider_family=normalized_provider_family,
                )
            )
        return tuple(items)

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
        _assert_non_empty_string(stream_session_id, "stream_session_id")
        _assert_non_empty_string(run_id, "run_id")
        _assert_non_empty_string(turn_id, "turn_id")
        if capabilities is not None:
            assert isinstance(capabilities, StreamProviderCapabilities)
        assert isinstance(close_after_terminal, bool)
        return self._iterate_canonical_items(
            self._canonical_items(
                stream_session_id=stream_session_id,
                run_id=run_id,
                turn_id=turn_id,
                provider_family=provider_family or self._provider_family,
                capabilities=capabilities,
                close_after_terminal=close_after_terminal,
            )
        )

    @staticmethod
    async def _iterate_canonical_items(
        items: tuple[CanonicalStreamItem, ...],
    ) -> AsyncIterator[CanonicalStreamItem]:
        for item in items:
            yield item

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
    ) -> AsyncIterator[CanonicalStreamItem]:
        self._item_index = 0
        return self

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        self._item_index = 0
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        items = self.canonical_items
        if self._item_index >= len(items):
            raise StopAsyncIteration
        item = items[self._item_index]
        self._item_index += 1
        return item


class TextGenerationNonStreamResult(TextGenerationStream):
    """Expose a provider-neutral rich non-stream generation result."""

    def __init__(
        self,
        events: Iterable[StreamProviderEvent],
        *,
        answer_text: str,
        provider_family: ProviderFamily | str | None = None,
        usage: object | None = None,
    ) -> None:
        assert isinstance(answer_text, str)
        normalized_provider_family = provider_family_value(provider_family)
        normalized_events = tuple(events)
        assert normalized_events
        assert all(
            isinstance(event, StreamProviderEvent)
            for event in normalized_events
        )
        terminal_indices = tuple(
            index
            for index, event in enumerate(normalized_events)
            if is_stream_terminal_kind(event.kind)
        )
        assert terminal_indices == (len(normalized_events) - 1,)
        derived_answer = "".join(
            event.text_delta or ""
            for event in normalized_events
            if event.kind is StreamItemKind.ANSWER_DELTA
        )
        assert answer_text == derived_answer
        self._events = normalized_events
        self._answer_text = answer_text
        self._provider_family = normalized_provider_family
        self._usage = usage
        self._generator = None

    @property
    def events(self) -> tuple[StreamProviderEvent, ...]:
        return self._events

    @property
    def answer_text(self) -> str:
        return self._answer_text

    @property
    def content(self) -> str:
        return self._answer_text

    @property
    def provider_family(self) -> str | None:
        return self._provider_family

    @property
    def usage(self) -> object | None:
        return self._usage

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
        return normalize_provider_stream(
            self._iterate_provider_events(),
            stream_session_id=stream_session_id,
            run_id=run_id,
            turn_id=turn_id,
            provider_family=provider_family or self._provider_family,
            capabilities=capabilities,
            close_after_terminal=close_after_terminal,
        )

    async def _iterate_provider_events(
        self,
    ) -> AsyncIterator[StreamProviderEvent]:
        for event in self._events:
            yield event

    def __call__(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[CanonicalStreamItem]:
        self._generator = self._default_canonical_stream()
        return self

    def __aiter__(self) -> AsyncIterator[CanonicalStreamItem]:
        self._generator = self._default_canonical_stream()
        return self

    async def __anext__(self) -> CanonicalStreamItem:
        if self._generator is None:
            self._generator = self._default_canonical_stream()
        return await self._generator.__anext__()

    def _default_canonical_stream(self) -> AsyncIterator[CanonicalStreamItem]:
        return self.canonical_stream(
            stream_session_id="non-stream-result",
            run_id="non-stream-run",
            turn_id="non-stream-turn",
            provider_family=self._provider_family,
        )

    async def to_str(self) -> str:
        return self._answer_text
