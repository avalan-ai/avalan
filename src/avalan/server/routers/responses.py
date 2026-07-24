from ...agent.orchestrator import Orchestrator
from ...model.reasoning import ReasoningSummaryCapabilityError
from ...model.stream import (
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    StreamReasoningRepresentation,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
)
from ...server.entities import (
    SKILL_CONTENT_REDACTION,
    ModelVisibleServerProtocolTextRedactor,
    ResponsesRequest,
    ServerOutputRedactionChannel,
    ServerOutputRedactionSettings,
    coerce_server_output_redaction_settings,
    sanitize_model_visible_server_protocol_text,
    sanitize_server_protocol_text,
    sanitize_server_protocol_value,
    server_output_redaction_settings_from_state,
)
from ...utils import to_json
from .. import di_get_logger, di_get_orchestrator
from ..remote_container import validate_remote_container_profile_selection
from ..sse import sse_headers, sse_message
from . import orchestrate, resolve_model_id
from .streaming import (
    cleanup_stream_sources,
    protocol_stream_terminal_snapshot,
    stream_consumer_iterator,
    stream_terminal_succeeded,
)

from asyncio import CancelledError
from collections.abc import Mapping
from dataclasses import dataclass, field
from logging import Logger
from types import MappingProxyType
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

_MAX_COALESCED_DELTA_CHARS = 4096

_RESPONSE_SSE_ITEM_TYPES = {
    "reasoning",
    "reasoning_text",
    "function_call",
    "custom_tool_call_input",
    "output_text",
}
_RESPONSE_SSE_CONTENT_PART_TYPES = {
    "reasoning_text",
    "input_text",
    "output_text",
}
_RESPONSE_SSE_TOOL_ITEM_TYPES = {
    "function_call",
    "custom_tool_call_input",
}
_RESPONSE_SSE_OUTPUT_INDEX_FIELDS: Mapping[str, int] = MappingProxyType(
    {"output_index": 0}
)
_RESPONSE_SSE_CONTENT_INDEX_FIELDS: Mapping[str, int] = MappingProxyType(
    {
        "output_index": 0,
        "content_index": 0,
    }
)
_RESPONSE_SSE_UNSET = object()
_RESPONSES_TERMINAL_STATUSES = MappingProxyType(
    {
        StreamTerminalOutcome.COMPLETED: "completed",
        StreamTerminalOutcome.ERRORED: "failed",
        StreamTerminalOutcome.CANCELLED: "cancelled",
        StreamTerminalOutcome.INPUT_REQUIRED: "incomplete",
    }
)


def _response_sse_index_value(
    data: dict[str, Any],
    key: str,
) -> int | None:
    value = data.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


@dataclass(frozen=True, slots=True)
class _ResponsesSSEEvent:
    event: str
    data: dict[str, Any]
    correlation_key: str | None = None
    canonical_channel: StreamChannel | None = None
    representation: StreamReasoningRepresentation | None = None
    item_id: str | None = None
    segment_instance_ordinal: int | None = None
    summary_index: int | None = None
    continuation_id: str | None = None

    def message(self) -> str:
        return sse_message(to_json(self.data), event=self.event)

    def can_coalesce(self, other: "_ResponsesSSEEvent") -> bool:
        assert isinstance(other, _ResponsesSSEEvent)
        self_sequence = self.data.get("sequence_number")
        other_sequence = other.data.get("sequence_number")
        if (self_sequence is None) != (other_sequence is None):
            return False
        if self_sequence is not None and (
            not isinstance(self_sequence, int)
            or isinstance(self_sequence, bool)
            or not isinstance(other_sequence, int)
            or isinstance(other_sequence, bool)
            or other_sequence != self_sequence + 1
        ):
            return False
        if (
            self.canonical_channel is None
            or other.canonical_channel is None
            or self.canonical_channel is not other.canonical_channel
        ):
            return False
        if (
            _response_sse_index_value(self.data, "output_index") is None
            or _response_sse_index_value(other.data, "output_index") is None
        ):
            return False
        index_key = (
            "summary_index"
            if self.event == "response.reasoning_summary_text.delta"
            else "content_index"
        )
        if (
            _response_sse_index_value(self.data, index_key) is None
            or _response_sse_index_value(other.data, index_key) is None
        ):
            return False
        if self.event == "response.tool_execution.output" and (
            self.data.get("data") is not None
            or other.data.get("data") is not None
        ):
            return False
        return (
            self.event == other.event
            and self.correlation_key == other.correlation_key
            and self.representation is other.representation
            and self.item_id == other.item_id
            and self.segment_instance_ordinal == other.segment_instance_ordinal
            and self.summary_index == other.summary_index
            and self.continuation_id == other.continuation_id
            and self.data.get("type") == other.data.get("type")
            and self.data.get("output_index") == other.data.get("output_index")
            and self.data.get("content_index")
            == other.data.get("content_index")
            and isinstance(self.data.get("delta"), str)
            and isinstance(other.data.get("delta"), str)
            and self.event
            in {
                "response.output_text.delta",
                "response.reasoning_text.delta",
                "response.reasoning_summary_text.delta",
                "response.custom_tool_call_input.delta",
                "response.tool_execution.output",
            }
        )

    def coalesce(self, other: "_ResponsesSSEEvent") -> "_ResponsesSSEEvent":
        assert self.can_coalesce(other)
        data = dict(self.data)
        data["delta"] = self.data["delta"] + other.data["delta"]
        data["sequence_number"] = other.data.get(
            "sequence_number", self.data.get("sequence_number")
        )
        return _ResponsesSSEEvent(
            event=self.event,
            data=data,
            correlation_key=self.correlation_key,
            canonical_channel=self.canonical_channel,
            representation=self.representation,
            item_id=self.item_id,
            segment_instance_ordinal=self.segment_instance_ordinal,
            summary_index=self.summary_index,
            continuation_id=self.continuation_id,
        )

    def coalesced_delta_length(self, other: "_ResponsesSSEEvent") -> int:
        assert self.can_coalesce(other)
        return len(self.data["delta"]) + len(other.data["delta"])

    def with_sequence(self, sequence_number: int) -> "_ResponsesSSEEvent":
        assert isinstance(sequence_number, int)
        assert not isinstance(sequence_number, bool)
        assert sequence_number >= 0
        data = dict(self.data)
        data["sequence_number"] = sequence_number
        return _ResponsesSSEEvent(
            event=self.event,
            data=data,
            correlation_key=self.correlation_key,
            canonical_channel=self.canonical_channel,
            representation=self.representation,
            item_id=self.item_id,
            segment_instance_ordinal=self.segment_instance_ordinal,
            summary_index=self.summary_index,
            continuation_id=self.continuation_id,
        )


@dataclass(frozen=True, slots=True)
class _ResponsesSSEStreamEnvelope:
    response_id: str
    timestamp: int
    model_id: str

    def __post_init__(self) -> None:
        assert isinstance(self.response_id, str)
        assert isinstance(self.timestamp, int)
        assert not isinstance(self.timestamp, bool)
        assert isinstance(self.model_id, str)

    def created_event(self) -> _ResponsesSSEEvent:
        return _ResponsesSSEEvent(
            event="response.created",
            data={
                "type": "response.created",
                "response": {
                    "id": self.response_id,
                    "created_at": self.timestamp,
                    "model": self.model_id,
                    "type": "response",
                    "status": "in_progress",
                },
            },
        )


@dataclass(frozen=True, slots=True)
class _ResponsesSSEItemState:
    output_item_type: str
    content_part_type: str | None = None
    tool_call_id: str | None = None

    def __post_init__(self) -> None:
        assert self.output_item_type in _RESPONSE_SSE_ITEM_TYPES
        if self.content_part_type is not None:
            assert self.content_part_type in _RESPONSE_SSE_CONTENT_PART_TYPES
        if self.tool_call_id is not None:
            assert isinstance(self.tool_call_id, str)
            assert self.tool_call_id.strip()

    @property
    def is_tool_call(self) -> bool:
        return self.output_item_type in _RESPONSE_SSE_TOOL_ITEM_TYPES


@dataclass(slots=True)
class _ResponsesSSEProjectionAdapter:
    state: _ResponsesSSEItemState | None = None
    answer_redactor: ModelVisibleServerProtocolTextRedactor = field(
        default_factory=ModelVisibleServerProtocolTextRedactor
    )
    reasoning_redactor: ModelVisibleServerProtocolTextRedactor = field(
        default_factory=ModelVisibleServerProtocolTextRedactor
    )
    answer_pending_sequence: int | None = None
    reasoning_pending_sequence: int | None = None

    @property
    def active_tool_call_id(self) -> str | None:
        if self.state is None or not self.state.is_tool_call:
            return None
        return self.state.tool_call_id

    def switch(self, token: StreamConsumerProjection | None) -> list[str]:
        new_state = _response_projection_state(
            token,
            self.active_tool_call_id,
        )
        events = _switch_state(self.state, new_state)
        self.state = new_state
        return events

    def close(self) -> list[str]:
        events = _switch_state(self.state, None)
        self.state = None
        return events

    def flush_model_text_before_switch(
        self,
        token: StreamConsumerProjection,
    ) -> list[_ResponsesSSEEvent]:
        new_state = _response_projection_state(
            token,
            self.active_tool_call_id,
        )
        if self.state == new_state:
            return []
        if self.state is None:
            return []
        if self.state.output_item_type == "reasoning_text":
            return self._flush_reasoning_text(token.sequence)
        if self.state.output_item_type == "output_text":
            return self._flush_answer_text(token.sequence)
        return []

    def flush_model_text_events(self, seq: int) -> list[_ResponsesSSEEvent]:
        assert isinstance(seq, int) and not isinstance(seq, bool)
        ordered_events: list[tuple[int, list[_ResponsesSSEEvent]]] = []
        if self.reasoning_redactor.has_pending:
            ordered_events.append(
                (
                    (
                        seq
                        if self.reasoning_pending_sequence is None
                        else self.reasoning_pending_sequence
                    ),
                    self._flush_reasoning_text(seq),
                )
            )
        if self.answer_redactor.has_pending:
            ordered_events.append(
                (
                    (
                        seq
                        if self.answer_pending_sequence is None
                        else self.answer_pending_sequence
                    ),
                    self._flush_answer_text(seq),
                )
            )
        events: list[_ResponsesSSEEvent] = []
        for _sequence, sequence_events in sorted(
            ordered_events,
            key=lambda item: item[0],
        ):
            events.extend(sequence_events)
        return events

    def record_model_text_pending(
        self,
        token: StreamConsumerProjection,
        redactor: ModelVisibleServerProtocolTextRedactor | None,
    ) -> None:
        if redactor is None:
            return
        if token.kind is StreamItemKind.ANSWER_DELTA:
            if redactor.has_pending:
                if self.answer_pending_sequence is None:
                    self.answer_pending_sequence = token.sequence
            else:
                self.answer_pending_sequence = None
        elif token.kind is StreamItemKind.REASONING_DELTA:
            if redactor.has_pending:
                if self.reasoning_pending_sequence is None:
                    self.reasoning_pending_sequence = token.sequence
            else:
                self.reasoning_pending_sequence = None

    def model_text_redactor(
        self, token: StreamConsumerProjection
    ) -> ModelVisibleServerProtocolTextRedactor | None:
        if token.kind is StreamItemKind.ANSWER_DELTA:
            return self.answer_redactor
        if token.kind is StreamItemKind.REASONING_DELTA:
            return self.reasoning_redactor
        return None

    def _flush_answer_text(
        self,
        fallback_seq: int,
    ) -> list[_ResponsesSSEEvent]:
        events = _model_visible_flush_events(
            "response.output_text.delta",
            self.answer_redactor,
            (
                fallback_seq
                if self.answer_pending_sequence is None
                else self.answer_pending_sequence
            ),
        )
        self.answer_pending_sequence = None
        return events

    def _flush_reasoning_text(
        self,
        fallback_seq: int,
    ) -> list[_ResponsesSSEEvent]:
        events = _model_visible_flush_events(
            "response.reasoning_text.delta",
            self.reasoning_redactor,
            (
                fallback_seq
                if self.reasoning_pending_sequence is None
                else self.reasoning_pending_sequence
            ),
        )
        self.reasoning_pending_sequence = None
        return events


@dataclass(frozen=True, slots=True)
class _ResponsesReasoningIdentity:
    representation: StreamReasoningRepresentation
    segment_instance_ordinal: int
    provider_item_id: str | None
    provider_output_index: int | None
    provider_summary_index: int | None
    continuation_id: str | None

    @classmethod
    def from_projection(
        cls,
        projection: StreamConsumerProjection,
    ) -> "_ResponsesReasoningIdentity":
        assert projection.kind is StreamItemKind.REASONING_DELTA
        assert projection.reasoning_representation is not None
        assert projection.segment_instance_ordinal is not None
        correlation = projection.correlation
        return cls(
            representation=projection.reasoning_representation,
            segment_instance_ordinal=projection.segment_instance_ordinal,
            provider_item_id=correlation.protocol_item_id,
            provider_output_index=correlation.provider_output_index,
            provider_summary_index=correlation.provider_summary_index,
            continuation_id=correlation.model_continuation_id,
        )

    @property
    def item_key(self) -> tuple[object, ...]:
        if self.representation is StreamReasoningRepresentation.SUMMARY:
            provider_key = (
                self.provider_item_id,
                self.provider_output_index,
            )
            if provider_key != (None, None):
                return (
                    self.representation,
                    self.continuation_id,
                    *provider_key,
                )
        return (
            self.representation,
            self.continuation_id,
            self.provider_item_id,
            self.provider_output_index,
            self.segment_instance_ordinal,
        )

    @property
    def part_key(self) -> tuple[object, ...]:
        return (
            self.representation,
            self.segment_instance_ordinal,
            self.provider_item_id,
            self.provider_output_index,
            self.provider_summary_index,
            self.continuation_id,
        )

    @property
    def has_optional_identity(self) -> bool:
        return any(
            value is not None
            for value in (
                self.provider_item_id,
                self.provider_output_index,
                self.provider_summary_index,
                self.continuation_id,
            )
        )


class _ResponsesReasoningRetentionError(RuntimeError):
    """Report prospective Responses reasoning retention overflow."""


class _ResponsesSourceAfterTerminalError(RuntimeError):
    """Report a source failure after the protocol terminal was owned."""


class _ResponsesCleanupError(RuntimeError):
    """Report a content-free Responses cleanup failure."""


_RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE = (
    "Responses source failed after terminal outcome."
)
_RESPONSES_CLEANUP_ERROR_MESSAGE = "Responses stream cleanup failed."


@dataclass(slots=True)
class _ResponsesReasoningAdmission:
    segment_limit: int
    character_limit: int
    utf8_byte_limit: int
    segment_count: int = 0
    emitted_character_count: int = 0
    emitted_utf8_byte_count: int = 0
    pending_character_count: int = 0
    pending_utf8_byte_count: int = 0
    pending_identity: tuple[object, ...] | None = None
    marker_reserved: bool = False

    @classmethod
    def from_policy(
        cls,
        policy: StreamRetentionPolicy,
    ) -> "_ResponsesReasoningAdmission":
        assert isinstance(policy, StreamRetentionPolicy)
        return cls(
            segment_limit=policy.responses_reasoning_item_segment_limit,
            character_limit=(policy.responses_reasoning_item_character_limit),
            utf8_byte_limit=(policy.responses_reasoning_item_text_byte_limit),
        )

    def admit_part(self) -> None:
        prospective = self.segment_count + 1
        if prospective > self.segment_limit:
            raise _ResponsesReasoningRetentionError(
                "Responses reasoning segment limit exceeded"
            )
        self.segment_count = prospective

    def admit_push(
        self,
        identity: tuple[object, ...],
        value: str,
        redactor: ModelVisibleServerProtocolTextRedactor,
    ) -> None:
        assert isinstance(identity, tuple)
        assert isinstance(value, str)
        assert isinstance(redactor, ModelVisibleServerProtocolTextRedactor)
        if self.pending_identity is not None:
            assert self.pending_identity == identity
        (
            _chunks,
            preview_pending_characters,
            preview_pending_bytes,
            preview_redacted,
        ) = redactor.preview_push(value)
        reserve = self.marker_reserved or bool(preview_pending_characters)
        reserve = reserve or preview_redacted
        marker_characters = len(SKILL_CONTENT_REDACTION) if reserve else 0
        marker_bytes = (
            len(SKILL_CONTENT_REDACTION.encode("utf-8")) if reserve else 0
        )
        prospective_characters = (
            self.emitted_character_count
            + self.pending_character_count
            + len(value)
            + marker_characters
        )
        prospective_bytes = (
            self.emitted_utf8_byte_count
            + self.pending_utf8_byte_count
            + len(value.encode("utf-8"))
            + marker_bytes
        )
        if prospective_characters > self.character_limit:
            raise _ResponsesReasoningRetentionError(
                "Responses reasoning character limit exceeded"
            )
        if prospective_bytes > self.utf8_byte_limit:
            raise _ResponsesReasoningRetentionError(
                "Responses reasoning byte limit exceeded"
            )
        if preview_pending_characters:
            assert preview_pending_bytes >= preview_pending_characters

    def commit_push(
        self,
        identity: tuple[object, ...],
        chunks: tuple[str, ...],
        redactor: ModelVisibleServerProtocolTextRedactor,
    ) -> None:
        assert isinstance(identity, tuple)
        assert isinstance(chunks, tuple)
        assert isinstance(redactor, ModelVisibleServerProtocolTextRedactor)
        self.emitted_character_count += sum(len(chunk) for chunk in chunks)
        self.emitted_utf8_byte_count += sum(
            len(chunk.encode("utf-8")) for chunk in chunks
        )
        self.pending_character_count = redactor.pending_character_count
        self.pending_utf8_byte_count = redactor.pending_utf8_byte_count
        self.pending_identity = (
            identity if self.pending_character_count else None
        )
        self.marker_reserved = bool(self.pending_character_count)
        self._assert_committed_within_limits()

    def admit_flush(
        self,
        identity: tuple[object, ...],
        redactor: ModelVisibleServerProtocolTextRedactor,
    ) -> None:
        assert isinstance(identity, tuple)
        assert isinstance(redactor, ModelVisibleServerProtocolTextRedactor)
        if self.pending_identity is not None:
            assert self.pending_identity == identity
        chunks, _redacted = redactor.preview_flush()
        prospective_characters = self.emitted_character_count + sum(
            len(chunk) for chunk in chunks
        )
        prospective_bytes = self.emitted_utf8_byte_count + sum(
            len(chunk.encode("utf-8")) for chunk in chunks
        )
        if prospective_characters > self.character_limit:
            raise _ResponsesReasoningRetentionError(
                "Responses reasoning character limit exceeded"
            )
        if prospective_bytes > self.utf8_byte_limit:
            raise _ResponsesReasoningRetentionError(
                "Responses reasoning byte limit exceeded"
            )

    def commit_flush(self, chunks: tuple[str, ...]) -> None:
        assert isinstance(chunks, tuple)
        self.emitted_character_count += sum(len(chunk) for chunk in chunks)
        self.emitted_utf8_byte_count += sum(
            len(chunk.encode("utf-8")) for chunk in chunks
        )
        self.pending_character_count = 0
        self.pending_utf8_byte_count = 0
        self.pending_identity = None
        self.marker_reserved = False
        self._assert_committed_within_limits()

    def _assert_committed_within_limits(self) -> None:
        assert (
            self.emitted_character_count + self.pending_character_count
            <= self.character_limit
        )
        assert (
            self.emitted_utf8_byte_count + self.pending_utf8_byte_count
            <= self.utf8_byte_limit
        )


@dataclass(slots=True)
class _ResponsesProjectedItem:
    kind: str
    output_index: int
    item_id: str
    continuation_id: str | None = None
    representation: StreamReasoningRepresentation | None = None
    reasoning_item_key: tuple[object, ...] | None = None
    reasoning_part_key: tuple[object, ...] | None = None
    segment_instance_ordinal: int | None = None
    summary_index: int | None = None
    provider_summary_index: int | None = None
    tool_name: str | None = None
    text: list[str] = field(default_factory=list)
    summary: list[dict[str, str]] = field(default_factory=list)
    redactor: ModelVisibleServerProtocolTextRedactor | None = None
    admission: _ResponsesReasoningAdmission | None = None
    quarantined: bool = False
    content_closed: bool = False


class _ResponsesSSEProjector:
    """Project canonical items into one aggregate Responses lifecycle."""

    def __init__(
        self,
        response_id: str,
        output_redaction_settings: ServerOutputRedactionSettings,
        retention_policy: StreamRetentionPolicy | None = None,
    ) -> None:
        assert isinstance(response_id, str) and response_id.strip()
        self.response_id = response_id
        self.output_redaction_settings = output_redaction_settings
        self.retention_policy = retention_policy or StreamRetentionPolicy()
        self.state: _ResponsesProjectedItem | None = None
        self._tool_states: dict[str, _ResponsesProjectedItem] = {}
        self._next_output_index = 0
        self.failure: dict[str, object] | None = None
        self.redaction_latched = False
        self._quarantine_next_reasoning = False
        self._last_reasoning_identity: _ResponsesReasoningIdentity | None = (
            None
        )

    def events_for(
        self,
        projection: StreamConsumerProjection,
    ) -> list[_ResponsesSSEEvent]:
        assert isinstance(projection, StreamConsumerProjection)
        if self.failure is not None:
            return []
        if projection.kind is StreamItemKind.REASONING_DELTA:
            return self._reasoning_events(projection)
        if projection.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            return self._tool_call_events(projection)
        if projection.kind is StreamItemKind.ANSWER_DELTA:
            return self._answer_events(projection)
        if projection.kind in {
            StreamItemKind.TOOL_CALL_READY,
            StreamItemKind.TOOL_CALL_DONE,
        }:
            tool_call_id = projection.tool_call_id
            assert tool_call_id is not None
            return self._close_tool(tool_call_id, status="completed")
        if projection.is_stream_terminal:
            status = (
                "completed"
                if projection.terminal_outcome
                is StreamTerminalOutcome.COMPLETED
                else "incomplete"
            )
            events = self.close(
                status=status,
                identity_lost=True,
                include_tools=True,
            )
            if (
                projection.kind is StreamItemKind.STREAM_COMPLETED
                and projection.usage is not None
            ):
                events.append(
                    _ResponsesSSEEvent(
                        event="response.usage.completed",
                        data={
                            "type": "response.usage.completed",
                            "usage": projection.usage,
                        },
                        canonical_channel=projection.channel,
                    )
                )
            return events
        if projection.kind in {
            StreamItemKind.STREAM_STARTED,
            StreamItemKind.STREAM_CLOSED,
        }:
            return []
        if projection.kind is StreamItemKind.REASONING_DONE:
            return self._complete_reasoning()
        if projection.kind is StreamItemKind.ANSWER_DONE:
            return self._complete_answer()
        events = self.close(
            status="completed",
            identity_lost=True,
            include_tools=False,
        )
        events.extend(
            self._unsequenced_events(
                _canonical_item_to_sse_events(
                    projection,
                    0,
                    output_redaction_settings=self.output_redaction_settings,
                )
            )
        )
        return events

    def close(
        self,
        *,
        status: str = "completed",
        identity_lost: bool = False,
        include_tools: bool = True,
    ) -> list[_ResponsesSSEEvent]:
        assert status in {"completed", "incomplete"}
        assert isinstance(identity_lost, bool)
        assert isinstance(include_tools, bool)
        state = self.state
        events: list[_ResponsesSSEEvent] = []
        if state is not None and state.kind == "reasoning_summary":
            events.extend(
                self._close_summary_part(
                    state,
                    identity_lost=identity_lost,
                )
            )
        elif state is not None and state.kind == "reasoning_text":
            if not state.content_closed:
                events.extend(
                    self._close_native_reasoning(
                        state,
                        identity_lost=identity_lost,
                    )
                )
        elif state is not None and state.kind == "output_text":
            if not state.content_closed:
                events.extend(
                    self._flush_text(state, "response.output_text.delta")
                )
                events.append(
                    self._item_event(
                        state,
                        "response.output_text.done",
                        {
                            "item_id": state.item_id,
                            "content_index": 0,
                            "text": "".join(state.text),
                        },
                    )
                )
                events.append(self._content_part_done_event(state))
                state.content_closed = True
        if state is not None:
            events.append(self._output_item_done_event(state, status))
            self.state = None
        if include_tools:
            for tool_state in sorted(
                self._tool_states.values(),
                key=lambda candidate: candidate.output_index,
            ):
                events.extend(self._close_tool_state(tool_state, status))
            self._tool_states.clear()
        return events

    def _close_tool(
        self,
        tool_call_id: str,
        *,
        status: str,
    ) -> list[_ResponsesSSEEvent]:
        assert isinstance(tool_call_id, str) and tool_call_id.strip()
        assert status in {"completed", "incomplete"}
        state = self._tool_states.pop(tool_call_id, None)
        if state is None:
            return []
        return self._close_tool_state(state, status)

    def _close_tool_state(
        self,
        state: _ResponsesProjectedItem,
        status: str,
    ) -> list[_ResponsesSSEEvent]:
        assert state.kind in {"function_call", "custom_tool_call_input"}
        events: list[_ResponsesSSEEvent] = []
        if state.kind == "function_call":
            events.append(
                self._item_event(
                    state,
                    "response.function_call_arguments.done",
                    {"id": state.item_id},
                )
            )
        else:
            events.append(
                self._item_event(
                    state,
                    "response.custom_tool_call_input.done",
                    {"id": state.item_id, "content_index": 0},
                )
            )
            events.append(self._content_part_done_event(state))
        events.append(self._output_item_done_event(state, status))
        return events

    def _complete_reasoning(self) -> list[_ResponsesSSEEvent]:
        state = self.state
        if state is None:
            return []
        if state.kind == "reasoning_summary":
            return self._close_summary_part(state, identity_lost=False)
        if state.kind == "reasoning_text" and not state.content_closed:
            events = self._close_native_reasoning(
                state,
                identity_lost=False,
            )
            state.content_closed = True
            return events
        return []

    def _complete_answer(self) -> list[_ResponsesSSEEvent]:
        state = self.state
        if state is None or state.kind != "output_text":
            return []
        if state.content_closed:
            return []
        events = self._flush_text(state, "response.output_text.delta")
        events.append(
            self._item_event(
                state,
                "response.output_text.done",
                {
                    "item_id": state.item_id,
                    "content_index": 0,
                    "text": "".join(state.text),
                },
            )
        )
        events.append(self._content_part_done_event(state))
        state.content_closed = True
        return events

    def _reasoning_events(
        self,
        projection: StreamConsumerProjection,
    ) -> list[_ResponsesSSEEvent]:
        identity = _ResponsesReasoningIdentity.from_projection(projection)
        previous_identity = self._last_reasoning_identity
        identity_lost = bool(
            previous_identity is not None
            and previous_identity.has_optional_identity
            and not identity.has_optional_identity
        )
        self._last_reasoning_identity = identity
        events: list[_ResponsesSSEEvent] = []
        state = self.state
        expected_kind = (
            "reasoning_summary"
            if identity.representation is StreamReasoningRepresentation.SUMMARY
            else "reasoning_text"
        )
        if (
            state is None
            or state.kind != expected_kind
            or state.reasoning_item_key != identity.item_key
        ):
            events.extend(self.close(status="completed", include_tools=False))
            state = self._open_reasoning_item(identity)
            self.state = state
            if state.kind == "reasoning_text":
                assert state.admission is not None
                try:
                    state.admission.admit_part()
                except _ResponsesReasoningRetentionError:
                    return self._retention_failure_events(events)
            events.append(self._output_item_added_event(state))
            if state.kind == "reasoning_text":
                events.append(self._content_part_added_event(state))
        assert state is not None
        if state.kind == "reasoning_summary":
            if state.reasoning_part_key != identity.part_key:
                events.extend(
                    self._close_summary_part(state, identity_lost=False)
                )
                assert state.admission is not None
                try:
                    state.admission.admit_part()
                except _ResponsesReasoningRetentionError:
                    return self._retention_failure_events(events)
                state.reasoning_part_key = identity.part_key
                state.segment_instance_ordinal = (
                    identity.segment_instance_ordinal
                )
                state.provider_summary_index = identity.provider_summary_index
                state.summary_index = len(state.summary)
                state.text = []
                state.redactor = self._reasoning_redactor()
                state.quarantined = self._consume_reasoning_quarantine(
                    identity
                )
                events.append(self._summary_part_added_event(state))
            if identity_lost:
                state.quarantined = True
                self._quarantine_next_reasoning = True
            try:
                events.extend(
                    self._append_model_text(
                        state,
                        projection.text_delta or "",
                        "response.reasoning_summary_text.delta",
                    )
                )
            except _ResponsesReasoningRetentionError:
                return self._retention_failure_events(events)
            return events
        if identity_lost:
            state.quarantined = True
            self._quarantine_next_reasoning = True
        try:
            events.extend(
                self._append_model_text(
                    state,
                    projection.text_delta or "",
                    "response.reasoning_text.delta",
                )
            )
        except _ResponsesReasoningRetentionError:
            return self._retention_failure_events(events)
        return events

    def _tool_call_events(
        self,
        projection: StreamConsumerProjection,
    ) -> list[_ResponsesSSEEvent]:
        item_id = projection.tool_call_id
        assert item_id is not None
        function_call = _projection_function_call_delta(projection)
        kind = (
            "function_call"
            if function_call is not None
            else "custom_tool_call_input"
        )
        events: list[_ResponsesSSEEvent] = []
        state = self._tool_states.get(item_id)
        if state is not None and state.kind != kind:
            raise StreamValidationError(
                "Responses tool call changed protocol item type"
            )
        if state is None:
            events.extend(
                self.close(
                    status="completed",
                    identity_lost=True,
                    include_tools=False,
                )
            )
            state = self._open_item(
                kind,
                item_id=item_id,
                continuation_id=(projection.correlation.model_continuation_id),
            )
            state.tool_name = _response_tool_name(projection)
            self._tool_states[item_id] = state
            events.append(self._output_item_added_event(state))
            if kind == "custom_tool_call_input":
                events.append(self._content_part_added_event(state))
        assert state is not None
        projected = _canonical_item_to_sse_events(
            projection,
            0,
            item_id,
            output_redaction_settings=self.output_redaction_settings,
        )
        if kind == "function_call":
            state.text.append(
                sanitize_server_protocol_text(
                    projection.text_delta or "",
                    output_redaction_settings=self.output_redaction_settings,
                    protocol="openai",
                )
            )
        else:
            state.text.append(
                _sanitize_response_tool_text_delta(
                    projection.text_delta or "",
                    tool_name=state.tool_name,
                    output_redaction_settings=self.output_redaction_settings,
                )
            )
        for event in self._unsequenced_events(projected):
            data = dict(event.data)
            data["output_index"] = state.output_index
            if "content_index" in data:
                data["content_index"] = 0
            events.append(
                _ResponsesSSEEvent(
                    event=event.event,
                    data=data,
                    correlation_key=item_id,
                    canonical_channel=event.canonical_channel,
                    item_id=item_id,
                    continuation_id=state.continuation_id,
                )
            )
        return events

    def _answer_events(
        self,
        projection: StreamConsumerProjection,
    ) -> list[_ResponsesSSEEvent]:
        events: list[_ResponsesSSEEvent] = []
        state = self.state
        if state is None or state.kind != "output_text":
            events.extend(
                self.close(
                    status="completed",
                    identity_lost=True,
                    include_tools=False,
                )
            )
            state = self._open_item(
                "output_text",
                item_id=f"msg_{self.response_id}_{self._next_output_index}",
                continuation_id=(projection.correlation.model_continuation_id),
            )
            state.redactor = ModelVisibleServerProtocolTextRedactor(
                self.output_redaction_settings,
                protocol="openai",
                channel="answer",
            )
            self.state = state
            events.append(self._output_item_added_event(state))
            events.append(self._content_part_added_event(state))
        assert state is not None
        events.extend(
            self._append_model_text(
                state,
                projection.text_delta or "",
                "response.output_text.delta",
            )
        )
        return events

    def _open_reasoning_item(
        self,
        identity: _ResponsesReasoningIdentity,
    ) -> _ResponsesProjectedItem:
        output_index = self._allocate_output_index()
        state = _ResponsesProjectedItem(
            kind=(
                "reasoning_summary"
                if identity.representation
                is StreamReasoningRepresentation.SUMMARY
                else "reasoning_text"
            ),
            output_index=output_index,
            item_id=f"rs_{self.response_id}_{output_index}",
            continuation_id=identity.continuation_id,
            representation=identity.representation,
            reasoning_item_key=identity.item_key,
            reasoning_part_key=(
                None
                if identity.representation
                is StreamReasoningRepresentation.SUMMARY
                else identity.part_key
            ),
            segment_instance_ordinal=identity.segment_instance_ordinal,
            provider_summary_index=identity.provider_summary_index,
            redactor=self._reasoning_redactor(),
            admission=_ResponsesReasoningAdmission.from_policy(
                self.retention_policy
            ),
        )
        if state.kind == "reasoning_text":
            state.quarantined = self._consume_reasoning_quarantine(identity)
        return state

    def _open_item(
        self,
        kind: str,
        *,
        item_id: str,
        continuation_id: str | None,
    ) -> _ResponsesProjectedItem:
        return _ResponsesProjectedItem(
            kind=kind,
            output_index=self._allocate_output_index(),
            item_id=item_id,
            continuation_id=continuation_id,
        )

    def _allocate_output_index(self) -> int:
        output_index = self._next_output_index
        self._next_output_index += 1
        return output_index

    def _consume_reasoning_quarantine(
        self,
        identity: _ResponsesReasoningIdentity,
    ) -> bool:
        assert isinstance(identity, _ResponsesReasoningIdentity)
        if not identity.has_optional_identity:
            return False
        quarantined = self._quarantine_next_reasoning
        self._quarantine_next_reasoning = False
        return quarantined

    def _retention_failure_events(
        self,
        prefix: list[_ResponsesSSEEvent],
    ) -> list[_ResponsesSSEEvent]:
        self.failure = {
            "error": {
                "type": "server_error",
                "code": "reasoning_summary_retention_exceeded",
                "message": (
                    "Reasoning summary exceeded the configured retention "
                    "limit."
                ),
            }
        }
        prefix.extend(self.close(status="incomplete"))
        return prefix

    def _reasoning_redactor(self) -> ModelVisibleServerProtocolTextRedactor:
        return ModelVisibleServerProtocolTextRedactor(
            self.output_redaction_settings,
            protocol="openai",
            channel="reasoning",
        )

    def _append_model_text(
        self,
        state: _ResponsesProjectedItem,
        value: str,
        event: str,
    ) -> list[_ResponsesSSEEvent]:
        if self.redaction_latched or state.quarantined:
            return []
        redactor = state.redactor
        identity = state.reasoning_part_key or state.reasoning_item_key
        admission = state.admission
        if redactor is not None and admission is not None:
            assert identity is not None
            admission.admit_push(identity, value, redactor)
        chunks = (
            redactor.push(value)
            if redactor is not None
            else _model_visible_stream_deltas(
                value,
                None,
                output_redaction_settings=self.output_redaction_settings,
                channel=(
                    "answer" if state.kind == "output_text" else "reasoning"
                ),
            )
        )
        if redactor is not None and admission is not None:
            assert identity is not None
            admission.commit_push(identity, chunks, redactor)
        if SKILL_CONTENT_REDACTION in chunks:
            self.redaction_latched = True
        events: list[_ResponsesSSEEvent] = []
        for chunk in chunks:
            state.text.append(chunk)
            events.append(self._text_delta_event(state, event, chunk))
        return events

    def _flush_text(
        self,
        state: _ResponsesProjectedItem,
        event: str,
    ) -> list[_ResponsesSSEEvent]:
        redactor = state.redactor
        assert redactor is not None
        if self.redaction_latched or state.quarantined:
            redactor.flush()
            return []
        identity = state.reasoning_part_key or state.reasoning_item_key
        admission = state.admission
        force_marker = redactor.pending_requires_skill_marker
        if admission is not None and not force_marker:
            assert identity is not None
            admission.admit_flush(identity, redactor)
        chunks: tuple[str, ...]
        if force_marker:
            redactor.flush()
            chunks = (SKILL_CONTENT_REDACTION,)
        else:
            chunks = redactor.flush()
        if admission is not None:
            admission.commit_flush(chunks)
        if SKILL_CONTENT_REDACTION in chunks:
            self.redaction_latched = True
        events: list[_ResponsesSSEEvent] = []
        for chunk in chunks:
            state.text.append(chunk)
            events.append(self._text_delta_event(state, event, chunk))
        return events

    def _close_summary_part(
        self,
        state: _ResponsesProjectedItem,
        *,
        identity_lost: bool,
    ) -> list[_ResponsesSSEEvent]:
        assert isinstance(identity_lost, bool)
        if state.reasoning_part_key is None or state.summary_index is None:
            return []
        redactor = state.redactor
        had_pending = bool(redactor is not None and redactor.has_pending)
        was_latched = self.redaction_latched
        events = self._flush_text(
            state,
            "response.reasoning_summary_text.delta",
        )
        text = "".join(state.text)
        common = {
            "item_id": state.item_id,
            "summary_index": state.summary_index,
        }
        events.append(
            self._item_event(
                state,
                "response.reasoning_summary_text.done",
                {**common, "text": text},
            )
        )
        part = {"type": "summary_text", "text": text}
        events.append(
            self._item_event(
                state,
                "response.reasoning_summary_part.done",
                {**common, "part": part},
            )
        )
        state.summary.append(part)
        state.reasoning_part_key = None
        state.summary_index = None
        state.text = []
        state.redactor = None
        state.quarantined = False
        if (
            identity_lost
            and had_pending
            and not was_latched
            and not self.redaction_latched
        ):
            self._quarantine_next_reasoning = True
        return events

    def _close_native_reasoning(
        self,
        state: _ResponsesProjectedItem,
        *,
        identity_lost: bool,
    ) -> list[_ResponsesSSEEvent]:
        assert isinstance(identity_lost, bool)
        redactor = state.redactor
        had_pending = bool(redactor is not None and redactor.has_pending)
        was_latched = self.redaction_latched
        events = self._flush_text(state, "response.reasoning_text.delta")
        events.append(
            self._item_event(
                state,
                "response.reasoning_text.done",
                {
                    "item_id": state.item_id,
                    "content_index": 0,
                    "text": "".join(state.text),
                },
            )
        )
        events.append(self._content_part_done_event(state))
        state.quarantined = False
        if (
            identity_lost
            and had_pending
            and not was_latched
            and not self.redaction_latched
        ):
            self._quarantine_next_reasoning = True
        return events

    def _text_delta_event(
        self,
        state: _ResponsesProjectedItem,
        event: str,
        delta: str,
    ) -> _ResponsesSSEEvent:
        data: dict[str, Any] = {
            "type": event,
            "item_id": state.item_id,
            "output_index": state.output_index,
            "delta": delta,
        }
        if state.kind == "reasoning_summary":
            data["summary_index"] = state.summary_index
        else:
            data["content_index"] = 0
        return _ResponsesSSEEvent(
            event=event,
            data=data,
            correlation_key=state.item_id,
            canonical_channel=(
                StreamChannel.ANSWER
                if state.kind == "output_text"
                else StreamChannel.REASONING
            ),
            representation=state.representation,
            item_id=state.item_id,
            segment_instance_ordinal=state.segment_instance_ordinal,
            summary_index=state.summary_index,
            continuation_id=state.continuation_id,
        )

    def _output_item_added_event(
        self,
        state: _ResponsesProjectedItem,
    ) -> _ResponsesSSEEvent:
        item: dict[str, Any] = {"id": state.item_id}
        if state.kind == "reasoning_summary":
            item.update(
                {
                    "type": "reasoning",
                    "status": "in_progress",
                    "summary": [],
                }
            )
        elif state.kind == "reasoning_text":
            item.update({"type": "reasoning_text", "status": "in_progress"})
        elif state.kind == "output_text":
            item.update(
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                }
            )
        else:
            item["type"] = state.kind
            if state.tool_name is not None:
                item["name"] = state.tool_name
        return self._item_event(
            state,
            "response.output_item.added",
            {"item": item},
        )

    def _output_item_done_event(
        self,
        state: _ResponsesProjectedItem,
        status: str,
    ) -> _ResponsesSSEEvent:
        item: dict[str, Any] = {
            "id": state.item_id,
            "status": status,
        }
        if state.kind == "reasoning_summary":
            item.update({"type": "reasoning", "summary": list(state.summary)})
        elif state.kind == "reasoning_text":
            item.update(
                {
                    "type": "reasoning_text",
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": "".join(state.text),
                        }
                    ],
                }
            )
        elif state.kind == "output_text":
            item.update(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "".join(state.text),
                        }
                    ],
                }
            )
        else:
            item["type"] = state.kind
            if state.kind == "function_call":
                item["call_id"] = state.item_id
                item["name"] = state.tool_name or ""
                item["arguments"] = "".join(state.text)
            else:
                item["input"] = "".join(state.text)
        return self._item_event(
            state,
            "response.output_item.done",
            {"item": item},
        )

    def _summary_part_added_event(
        self,
        state: _ResponsesProjectedItem,
    ) -> _ResponsesSSEEvent:
        assert state.summary_index is not None
        return self._item_event(
            state,
            "response.reasoning_summary_part.added",
            {
                "item_id": state.item_id,
                "summary_index": state.summary_index,
                "part": {"type": "summary_text", "text": ""},
            },
        )

    def _content_part_added_event(
        self,
        state: _ResponsesProjectedItem,
    ) -> _ResponsesSSEEvent:
        part_type = {
            "reasoning_text": "reasoning_text",
            "custom_tool_call_input": "input_text",
            "output_text": "output_text",
        }[state.kind]
        part: dict[str, Any] = {"type": part_type}
        if state.kind == "custom_tool_call_input":
            part["id"] = state.item_id
        return self._item_event(
            state,
            "response.content_part.added",
            {
                "item_id": state.item_id,
                "content_index": 0,
                "part": part,
            },
        )

    def _content_part_done_event(
        self,
        state: _ResponsesProjectedItem,
    ) -> _ResponsesSSEEvent:
        part_type = {
            "reasoning_text": "reasoning_text",
            "custom_tool_call_input": "input_text",
            "output_text": "output_text",
        }[state.kind]
        part: dict[str, Any] = {
            "type": part_type,
            "text": "".join(state.text),
        }
        if state.kind == "custom_tool_call_input":
            part["id"] = state.item_id
        return self._item_event(
            state,
            "response.content_part.done",
            {
                "item_id": state.item_id,
                "content_index": 0,
                "part": part,
            },
        )

    def _item_event(
        self,
        state: _ResponsesProjectedItem,
        event: str,
        extra: dict[str, Any],
    ) -> _ResponsesSSEEvent:
        data = {
            "type": event,
            "output_index": state.output_index,
            **extra,
        }
        return _ResponsesSSEEvent(
            event=event,
            data=data,
            correlation_key=state.item_id,
            representation=state.representation,
            item_id=state.item_id,
            segment_instance_ordinal=state.segment_instance_ordinal,
            summary_index=state.summary_index,
            continuation_id=state.continuation_id,
        )

    @staticmethod
    def _unsequenced_events(
        events: list[_ResponsesSSEEvent],
    ) -> list[_ResponsesSSEEvent]:
        result: list[_ResponsesSSEEvent] = []
        for event in events:
            data = dict(event.data)
            data.pop("sequence_number", None)
            result.append(
                _ResponsesSSEEvent(
                    event=event.event,
                    data=data,
                    correlation_key=event.correlation_key,
                    canonical_channel=event.canonical_channel,
                    representation=event.representation,
                    item_id=event.item_id,
                    segment_instance_ordinal=(event.segment_instance_ordinal),
                    summary_index=event.summary_index,
                    continuation_id=event.continuation_id,
                )
            )
        return result


router = APIRouter(tags=["responses"])


def _server_output_redaction_settings(
    request: Request,
) -> ServerOutputRedactionSettings:
    return server_output_redaction_settings_from_state(request.app.state)


@router.post(
    "/responses",
    response_model=None,
    dependencies=[Depends(validate_remote_container_profile_selection)],
)
async def create_response(
    request: ResponsesRequest,
    logger: Logger = Depends(di_get_logger),
    orchestrator: Orchestrator = Depends(di_get_orchestrator),
    output_redaction_settings: ServerOutputRedactionSettings = Depends(
        _server_output_redaction_settings
    ),
) -> dict[str, Any] | JSONResponse | StreamingResponse:
    assert orchestrator and isinstance(orchestrator, Orchestrator)
    assert logger and isinstance(logger, Logger)
    assert request and request.messages
    model_id = resolve_model_id(orchestrator, request.model)

    try:
        response, response_id, timestamp = await orchestrate(
            request, logger, orchestrator
        )
    except ReasoningSummaryCapabilityError as error:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "reasoning_summary_unsupported",
                "message": str(error),
                "provider": error.provider,
                "requested_mode": error.requested_mode.value,
            },
        ) from error
    output_redaction_settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )

    if request.stream:

        async def generate() -> AsyncIterator[str]:
            projector = _ResponsesSSEProjector(
                str(response_id),
                output_redaction_settings,
            )
            stream_envelope = _ResponsesSSEStreamEnvelope(
                response_id=str(response_id),
                timestamp=timestamp,
                model_id=model_id,
            )
            pending_event: _ResponsesSSEEvent | None = None
            terminal_projection: StreamConsumerProjection | None = None
            next_event_sequence = 0
            iterator = stream_consumer_iterator(
                response,
                stream_session_id="responses-sse-stream",
                run_id=str(response_id),
                turn_id="responses-sse-turn",
                unsupported_message=(
                    "unsupported stream item for Responses SSE projection"
                ),
                close_source_on_generator_exit=False,
            )
            cancelled = False
            terminal_owned = False
            source_error: Exception | None = None

            def event_message(event: _ResponsesSSEEvent) -> str:
                nonlocal next_event_sequence
                numbered = event.with_sequence(next_event_sequence)
                next_event_sequence += 1
                return numbered.message()

            def enqueue_event(event: _ResponsesSSEEvent) -> list[str]:
                nonlocal pending_event
                if pending_event is None:
                    pending_event = event
                    return []
                if (
                    pending_event.can_coalesce(event)
                    and pending_event.coalesced_delta_length(event)
                    <= _MAX_COALESCED_DELTA_CHARS
                ):
                    pending_event = pending_event.coalesce(event)
                    return []
                messages = [event_message(pending_event)]
                pending_event = event
                return messages

            def flush_event() -> list[str]:
                nonlocal pending_event
                if pending_event is None:
                    return []
                messages = [event_message(pending_event)]
                pending_event = None
                return messages

            try:
                yield event_message(stream_envelope.created_event())

                while True:
                    try:
                        token = await anext(iterator)
                    except StopAsyncIteration:
                        break
                    except Exception as error:
                        source_error = error
                        break

                    if token.is_stream_terminal:
                        terminal_projection = token

                    for ev in projector.events_for(token):
                        for message in enqueue_event(ev):
                            yield message
                    if projector.failure is not None:
                        break

                if projector.failure is not None:
                    cancelled = True
                    for message in enqueue_event(
                        _ResponsesSSEEvent(
                            event="response.failed",
                            data={
                                "type": "response.failed",
                                **projector.failure,
                            },
                        )
                    ):
                        yield message
                    terminal_owned = True
                    for event in flush_event():
                        yield event
                    return

                if terminal_projection is None:
                    for ev in projector.close(
                        status="incomplete",
                        identity_lost=True,
                    ):
                        for message in enqueue_event(ev):
                            yield message
                    for event in flush_event():
                        yield event
                    if source_error is not None:
                        raise source_error
                    raise StreamValidationError(
                        "stream missing terminal outcome"
                    )

                for ev in _terminal_response_events(
                    terminal_projection,
                    output_redaction_settings=output_redaction_settings,
                ):
                    data = dict(ev.data)
                    data.pop("sequence_number", None)
                    for message in enqueue_event(
                        _ResponsesSSEEvent(
                            event=ev.event,
                            data=data,
                            correlation_key=ev.correlation_key,
                            canonical_channel=ev.canonical_channel,
                        )
                    ):
                        yield message
                terminal_owned = True
                for event in flush_event():
                    yield event

                if source_error is not None:
                    logger.error(
                        _RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE,
                    )
                    raise _ResponsesSourceAfterTerminalError(
                        _RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE
                    ) from None

                if source_error is None and stream_terminal_succeeded(
                    terminal_projection
                ):
                    await orchestrator.sync_messages(response)
            except CancelledError:
                cancelled = True
                raise
            finally:
                cleanup_failed = False
                try:
                    await cleanup_stream_sources(
                        response,
                        iterator,
                        cancelled=cancelled,
                    )
                except Exception:
                    cleanup_failed = True
                if cleanup_failed:
                    logger.error(_RESPONSES_CLEANUP_ERROR_MESSAGE)
                    if (
                        terminal_owned
                        and source_error is None
                        and projector.failure is None
                    ):
                        raise _ResponsesCleanupError(
                            _RESPONSES_CLEANUP_ERROR_MESSAGE
                        ) from None

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers=sse_headers(),
        )

    projector = _ResponsesSSEProjector(
        str(response_id),
        output_redaction_settings,
    )
    indexed_output: dict[int, dict[str, Any]] = {}
    terminal_projection: StreamConsumerProjection | None = None
    source_error: Exception | None = None
    iterator = stream_consumer_iterator(
        response,
        stream_session_id="responses-non-stream",
        run_id=str(response_id),
        turn_id="responses-non-stream-turn",
        unsupported_message=(
            "unsupported stream item for Responses non-stream projection"
        ),
        close_source_on_generator_exit=False,
    )
    cleanup_failed = False
    try:
        while True:
            try:
                projection = await anext(iterator)
            except StopAsyncIteration:
                break
            except Exception as error:
                source_error = error
                break
            if projection.is_stream_terminal:
                terminal_projection = projection
            for event in projector.events_for(projection):
                if event.event != "response.output_item.done":
                    continue
                item = event.data.get("item")
                output_index = _response_sse_index_value(
                    event.data,
                    "output_index",
                )
                if isinstance(item, dict) and output_index is not None:
                    if output_index in indexed_output:
                        raise StreamValidationError(
                            "duplicate Responses outward output index"
                        )
                    indexed_output[output_index] = dict(item)
            if projector.failure is not None:
                break
    finally:
        try:
            await cleanup_stream_sources(
                response,
                iterator,
                cancelled=projector.failure is not None,
            )
        except Exception:
            cleanup_failed = True

    if projector.failure is not None:
        if cleanup_failed:
            logger.error(_RESPONSES_CLEANUP_ERROR_MESSAGE)
        return JSONResponse(
            status_code=500,
            content=projector.failure,
        )
    if cleanup_failed:
        logger.error(_RESPONSES_CLEANUP_ERROR_MESSAGE)
        if source_error is None:
            raise _ResponsesCleanupError(
                _RESPONSES_CLEANUP_ERROR_MESSAGE
            ) from None
    if terminal_projection is None:
        assert source_error is not None
        raise source_error
    if source_error is not None:
        logger.error(_RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE)
        raise _ResponsesSourceAfterTerminalError(
            _RESPONSES_SOURCE_AFTER_TERMINAL_MESSAGE
        ) from None
    terminal_snapshot = protocol_stream_terminal_snapshot(terminal_projection)
    output_indices = sorted(indexed_output)
    if output_indices != list(range(len(output_indices))):
        raise StreamValidationError(
            "non-contiguous Responses outward output indices"
        )
    output = [indexed_output[index] for index in output_indices]
    status = _responses_terminal_status(terminal_snapshot.outcome)
    body = {
        "id": str(response_id),
        "created": timestamp,
        "model": model_id,
        "type": "response",
        "status": status,
        "output": output,
        "usage": {
            "input_text_tokens": response.input_token_count,
            "output_text_tokens": response.output_token_count,
            "total_tokens": (
                response.input_token_count + response.output_token_count
            ),
        },
    }
    if status != "completed":
        terminal_events = _terminal_response_events(
            terminal_projection,
            output_redaction_settings=output_redaction_settings,
        )
        if terminal_events:
            terminal_error = terminal_events[0].data.get("error")
            if terminal_error is not None:
                body["error"] = terminal_error
    if status == "completed" and source_error is None:
        await orchestrator.sync_messages(response)
    return body


def _terminal_response_events(
    terminal: StreamConsumerProjection | StreamTerminalOutcome | None,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> list[_ResponsesSSEEvent]:
    terminal_snapshot = protocol_stream_terminal_snapshot(terminal)
    terminal_outcome = terminal_snapshot.outcome
    if (
        terminal_outcome is None
        or terminal_outcome is StreamTerminalOutcome.COMPLETED
    ):
        data: dict[str, Any] = {"type": "response.completed"}
        if terminal_snapshot.sequence is not None:
            data["sequence_number"] = terminal_snapshot.sequence
        return [
            _ResponsesSSEEvent(
                event="response.completed",
                data=data,
            )
        ]
    if terminal_outcome is StreamTerminalOutcome.CANCELLED:
        data = {"type": "response.cancelled"}
        if terminal_snapshot.sequence is not None:
            data["sequence_number"] = terminal_snapshot.sequence
        return [
            _ResponsesSSEEvent(
                event="response.cancelled",
                data=data,
            )
        ]
    if terminal_outcome is StreamTerminalOutcome.INPUT_REQUIRED:
        data = {"type": "response.incomplete"}
        if terminal_snapshot.sequence is not None:
            data["sequence_number"] = terminal_snapshot.sequence
        return [
            _ResponsesSSEEvent(
                event="response.incomplete",
                data=data,
            )
        ]
    assert terminal_outcome is StreamTerminalOutcome.ERRORED
    data = {"type": "response.failed"}
    if terminal_snapshot.sequence is not None:
        data["sequence_number"] = terminal_snapshot.sequence
        if terminal_snapshot.data is not None:
            sanitized_error = sanitize_server_protocol_value(
                terminal_snapshot.data,
                output_redaction_settings=output_redaction_settings,
                protocol="openai",
            )
            if isinstance(sanitized_error, dict) and isinstance(
                sanitized_error.get("error"), dict
            ):
                data["error"] = sanitized_error["error"]
            else:
                data["error"] = sanitized_error
    return [
        _ResponsesSSEEvent(
            event="response.failed",
            data=data,
        )
    ]


def _responses_terminal_status(
    outcome: StreamTerminalOutcome | None,
) -> str:
    assert outcome is None or isinstance(outcome, StreamTerminalOutcome)
    return _RESPONSES_TERMINAL_STATUSES[
        outcome or StreamTerminalOutcome.COMPLETED
    ]


def _token_to_sse(token: StreamConsumerProjection, seq: int) -> list[str]:
    assert isinstance(token, StreamConsumerProjection)
    return [event.message() for event in _token_to_sse_events(token, seq)]


def _token_to_sse_events(
    token: StreamConsumerProjection,
    seq: int,
    active_tool_call_id: str | None = None,
    model_text_redactor: ModelVisibleServerProtocolTextRedactor | None = None,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> list[_ResponsesSSEEvent]:
    assert isinstance(token, StreamConsumerProjection)
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    return _canonical_item_to_sse_events(
        token,
        seq,
        active_tool_call_id,
        model_text_redactor,
        output_redaction_settings=output_redaction_settings,
    )


def _stream_tool_call_protocol_id(
    item: StreamConsumerProjection,
) -> str | None:
    if item.kind is not StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        return None
    return item.tool_call_id


def _model_visible_stream_deltas(
    value: str,
    redactor: ModelVisibleServerProtocolTextRedactor | None,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
    channel: ServerOutputRedactionChannel = "answer",
) -> tuple[str, ...]:
    if redactor is not None:
        return redactor.push(value)
    sanitized = sanitize_model_visible_server_protocol_text(
        value,
        output_redaction_settings=output_redaction_settings,
        protocol="openai",
        channel=channel,
    )
    return (sanitized,) if sanitized else ()


def _model_visible_flush_events(
    event_name: str,
    redactor: ModelVisibleServerProtocolTextRedactor,
    seq: int,
) -> list[_ResponsesSSEEvent]:
    assert isinstance(event_name, str) and event_name.strip()
    assert isinstance(redactor, ModelVisibleServerProtocolTextRedactor)
    assert isinstance(seq, int) and not isinstance(seq, bool)
    return [
        _ResponsesSSEEvent(
            event=event_name,
            data=_response_sse_delta_data(event_name, delta, seq),
        )
        for delta in redactor.flush()
    ]


def _canonical_item_to_sse(
    item: StreamConsumerProjection, seq: int
) -> list[str]:
    return [
        event.message() for event in _canonical_item_to_sse_events(item, seq)
    ]


def _canonical_item_to_sse_events(
    item: StreamConsumerProjection,
    seq: int,
    active_tool_call_id: str | None = None,
    model_text_redactor: ModelVisibleServerProtocolTextRedactor | None = None,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> list[_ResponsesSSEEvent]:
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    if item.kind is StreamItemKind.REASONING_DELTA:
        return [
            _ResponsesSSEEvent(
                event="response.reasoning_text.delta",
                data=_response_sse_delta_data(
                    "response.reasoning_text.delta",
                    delta,
                    seq,
                ),
                canonical_channel=item.channel,
            )
            for delta in _model_visible_stream_deltas(
                item.text_delta or "",
                model_text_redactor,
                output_redaction_settings=output_redaction_settings,
                channel="reasoning",
            )
        ]
    if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        function_call = _projection_function_call_delta(item)
        if function_call is not None:
            function_call = _sanitize_response_sse_payload(
                function_call,
                tool_name=_response_tool_name(item),
                output_redaction_settings=output_redaction_settings,
            )
            return [
                _ResponsesSSEEvent(
                    event="response.function_call_arguments.delta",
                    data=_response_sse_delta_data(
                        "response.function_call_arguments.delta",
                        to_json(function_call),
                        seq,
                        id_value=function_call["id"],
                    ),
                    correlation_key=str(function_call["id"]),
                    canonical_channel=item.channel,
                )
            ]
        protocol_id = (
            _stream_tool_call_protocol_id(item) or active_tool_call_id
        )
        data = _response_sse_delta_data(
            "response.custom_tool_call_input.delta",
            sanitize_server_protocol_text(
                item.text_delta or "",
                output_redaction_settings=output_redaction_settings,
                protocol="openai",
            ),
            seq,
        )
        if protocol_id is not None:
            data["id"] = protocol_id
        return [
            _ResponsesSSEEvent(
                event="response.custom_tool_call_input.delta",
                data=data,
                correlation_key=protocol_id,
                canonical_channel=item.channel,
            )
        ]
    if item.kind is StreamItemKind.ANSWER_DELTA:
        return [
            _ResponsesSSEEvent(
                event="response.output_text.delta",
                data=_response_sse_delta_data(
                    "response.output_text.delta",
                    delta,
                    seq,
                ),
                canonical_channel=item.channel,
            )
            for delta in _model_visible_stream_deltas(
                item.text_delta or "",
                model_text_redactor,
                output_redaction_settings=output_redaction_settings,
                channel="answer",
            )
        ]
    if item.kind in (
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    ):
        event = (
            "response.usage.completed"
            if item.kind is StreamItemKind.USAGE_COMPLETED
            else "response.usage.delta"
        )
        return [
            _ResponsesSSEEvent(
                event=event,
                data={
                    "type": event,
                    "usage": item.usage,
                    "sequence_number": seq,
                },
                canonical_channel=item.channel,
            )
        ]
    if item.kind is StreamItemKind.STREAM_COMPLETED and item.usage is not None:
        return [
            _ResponsesSSEEvent(
                event="response.usage.completed",
                data={
                    "type": "response.usage.completed",
                    "usage": item.usage,
                    "sequence_number": seq,
                },
                canonical_channel=item.channel,
            )
        ]
    if item.kind in {
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
        StreamItemKind.TOOL_EXECUTION_CANCELLED,
    }:
        return [
            _tool_execution_sse_event(
                item,
                seq,
                output_redaction_settings=output_redaction_settings,
            )
        ]
    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
        return [
            _ResponsesSSEEvent(
                event="response.diagnostic",
                data={
                    "type": "response.diagnostic",
                    "delta": (
                        sanitize_server_protocol_text(
                            item.text_delta,
                            output_redaction_settings=(
                                output_redaction_settings
                            ),
                            protocol="openai",
                        )
                        if item.text_delta is not None
                        else None
                    ),
                    "data": _sanitize_response_sse_payload(
                        item.data,
                        tool_name=_response_tool_name(item),
                        output_redaction_settings=output_redaction_settings,
                    ),
                    "sequence_number": seq,
                },
                canonical_channel=item.channel,
            )
        ]
    return []


def _tool_execution_sse_event(
    item: StreamConsumerProjection,
    seq: int,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> _ResponsesSSEEvent:
    event_names = {
        StreamItemKind.TOOL_EXECUTION_STARTED: (
            "response.tool_execution.started"
        ),
        StreamItemKind.TOOL_EXECUTION_OUTPUT: "response.tool_execution.output",
        StreamItemKind.TOOL_EXECUTION_PROGRESS: (
            "response.tool_execution.progress"
        ),
        StreamItemKind.TOOL_EXECUTION_COMPLETED: (
            "response.tool_execution.completed"
        ),
        StreamItemKind.TOOL_EXECUTION_ERROR: "response.tool_execution.error",
        StreamItemKind.TOOL_EXECUTION_CANCELLED: (
            "response.tool_execution.cancelled"
        ),
    }
    event = event_names[item.kind]
    tool_name = _response_tool_name(item)
    data: dict[str, Any] = {
        "type": event,
        "id": item.tool_call_id,
        "sequence_number": seq,
    }
    if item.text_delta is not None:
        data["delta"] = _sanitize_response_tool_text_delta(
            item.text_delta,
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
        )
    if item.data is not None:
        data["data"] = _sanitize_response_sse_payload(
            item.data,
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
        )
    return _ResponsesSSEEvent(
        event=event,
        data=data,
        correlation_key=item.tool_call_id,
        canonical_channel=item.channel,
    )


def _projection_function_call_delta(
    item: StreamConsumerProjection,
) -> dict[str, Any] | None:
    if not isinstance(item.data, dict):
        return None
    name = item.data.get("name")
    if not isinstance(name, str):
        return None
    return {
        "id": item.tool_call_id,
        "name": name,
        "arguments": item.data.get("arguments"),
    }


def _response_tool_name(item: StreamConsumerProjection) -> str | None:
    if isinstance(item.data, dict):
        name = item.data.get("name")
        if isinstance(name, str) and name:
            return name
    metadata_name = item.metadata.get("tool_name")
    return metadata_name if isinstance(metadata_name, str) else None


def _sanitize_response_tool_text_delta(
    value: str,
    *,
    tool_name: str | None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    if (
        isinstance(tool_name, str)
        and tool_name.startswith("skills.")
        and settings.should_redact(
            "skills_tool_content",
            protocol="openai",
        )
    ):
        return to_json(
            _sanitize_response_sse_payload(
                {"content": value},
                tool_name=tool_name,
                output_redaction_settings=settings,
            )
        )
    return sanitize_server_protocol_text(
        value,
        output_redaction_settings=settings,
        protocol="openai",
    )


def _sanitize_response_sse_payload(
    value: object,
    *,
    tool_name: str | None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> dict[str, Any]:
    sanitized = sanitize_server_protocol_value(
        value,
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
        protocol="openai",
    )
    return sanitized if isinstance(sanitized, dict) else {"value": sanitized}


def _response_sse_delta_data(
    event: str,
    delta: str,
    seq: int,
    *,
    id_value: object = _RESPONSE_SSE_UNSET,
) -> dict[str, Any]:
    assert isinstance(event, str)
    assert event.strip()
    assert isinstance(delta, str)
    assert isinstance(seq, int)
    assert not isinstance(seq, bool)
    data: dict[str, Any] = {"type": event, "delta": delta}
    if id_value is not _RESPONSE_SSE_UNSET:
        data["id"] = id_value
    data.update(_RESPONSE_SSE_CONTENT_INDEX_FIELDS)
    data["sequence_number"] = seq
    return data


def _response_sse_indexed_data(
    event: str,
    *,
    content_index: bool = False,
) -> dict[str, Any]:
    assert isinstance(event, str)
    assert event.strip()
    data: dict[str, Any] = {"type": event}
    data.update(
        _RESPONSE_SSE_CONTENT_INDEX_FIELDS
        if content_index
        else _RESPONSE_SSE_OUTPUT_INDEX_FIELDS
    )
    return data


def _switch_state(
    state: _ResponsesSSEItemState | None,
    new_state: _ResponsesSSEItemState | None,
) -> list[str]:
    events: list[str] = []
    if state != new_state:
        if state is not None and state.output_item_type == "reasoning_text":
            events.append(_reasoning_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())
        elif state is not None and state.output_item_type == "function_call":
            events.append(_function_call_arguments_done(state.tool_call_id))
            events.append(_output_item_done(state.tool_call_id))
        elif (
            state is not None
            and state.output_item_type == "custom_tool_call_input"
        ):
            events.append(_custom_tool_call_input_done(state.tool_call_id))
            events.append(_content_part_done(state.tool_call_id))
            events.append(_output_item_done(state.tool_call_id))
        elif state is not None and state.output_item_type == "output_text":
            events.append(_output_text_done())
            events.append(_content_part_done())
            events.append(_output_item_done())

        if new_state is not None:
            events.append(_output_item_added(new_state))
            if new_state.content_part_type is not None:
                events.append(
                    _content_part_added(
                        new_state.content_part_type,
                        new_state.tool_call_id,
                    )
                )

    return events


def _response_projection_state(
    token: StreamConsumerProjection | None,
    active_tool_call_id: str | None = None,
) -> _ResponsesSSEItemState | None:
    if active_tool_call_id is not None:
        assert isinstance(active_tool_call_id, str)
        assert active_tool_call_id.strip()
    if isinstance(token, StreamConsumerProjection):
        if token.kind is StreamItemKind.REASONING_DELTA:
            return _ResponsesSSEItemState(
                output_item_type="reasoning_text",
                content_part_type="reasoning_text",
            )
        elif token.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            tool_call_id = (
                _stream_tool_call_protocol_id(token) or active_tool_call_id
            )
            if _projection_function_call_delta(token) is not None:
                return _ResponsesSSEItemState(
                    output_item_type="function_call",
                    tool_call_id=tool_call_id,
                )
            return _ResponsesSSEItemState(
                output_item_type="custom_tool_call_input",
                content_part_type="input_text",
                tool_call_id=tool_call_id,
            )
        elif token.kind is StreamItemKind.ANSWER_DELTA:
            return _ResponsesSSEItemState(
                output_item_type="output_text",
                content_part_type="output_text",
            )
        return None
    elif token is None:
        return None
    raise AssertionError("unsupported response stream item")


def _output_item_added(state: _ResponsesSSEItemState) -> str:
    item = {"type": state.output_item_type}
    if state.tool_call_id is not None:
        item["id"] = state.tool_call_id
    data = _response_sse_indexed_data("response.output_item.added")
    data["item"] = item
    return sse_message(
        to_json(data),
        event="response.output_item.added",
    )


def _output_item_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data("response.output_item.done")
    if id is not None:
        data["item"] = {"id": id}
    return sse_message(to_json(data), event="response.output_item.done")


def _function_call_arguments_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data("response.function_call_arguments.done")
    if id is not None:
        data["id"] = id
    return sse_message(
        to_json(data),
        event="response.function_call_arguments.done",
    )


def _reasoning_text_done() -> str:
    return sse_message(
        to_json(
            _response_sse_indexed_data(
                "response.reasoning_text.done",
                content_index=True,
            )
        ),
        event="response.reasoning_text.done",
    )


def _custom_tool_call_input_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data(
        "response.custom_tool_call_input.done",
        content_index=True,
    )
    if id is not None:
        data["id"] = id
    return sse_message(
        to_json(data),
        event="response.custom_tool_call_input.done",
    )


def _output_text_done() -> str:
    return sse_message(
        to_json(
            _response_sse_indexed_data(
                "response.output_text.done",
                content_index=True,
            )
        ),
        event="response.output_text.done",
    )


def _content_part_added(part_type: str, id: str | None = None) -> str:
    part = {"type": part_type}
    if id is not None:
        part["id"] = id
    data = _response_sse_indexed_data(
        "response.content_part.added",
        content_index=True,
    )
    data["part"] = part
    return sse_message(
        to_json(data),
        event="response.content_part.added",
    )


def _content_part_done(id: str | None = None) -> str:
    data = _response_sse_indexed_data(
        "response.content_part.done",
        content_index=True,
    )
    if id is not None:
        data["part"] = {"id": id}
    return sse_message(to_json(data), event="response.content_part.done")
