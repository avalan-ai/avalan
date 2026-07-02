from ..observability import observability_key_sample
from ..types import (
    LooseJsonValue,
    assert_non_negative_int,
    assert_non_negative_number,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast


class EventType(StrEnum):
    CALL_PREPARE_BEFORE = "call_prepare_before"
    CALL_PREPARE_AFTER = "call_prepare_after"
    END = "end"
    ENGINE_AGENT_CALL_BEFORE = "engine_agent_call_before"
    ENGINE_AGENT_CALL_AFTER = "engine_agent_call_after"
    ENGINE_RUN_BEFORE = "engine_run_before"
    ENGINE_RUN_AFTER = "engine_run_after"
    INPUT_TOKEN_COUNT_BEFORE = "input_token_count_before"
    INPUT_TOKEN_COUNT_AFTER = "input_token_count_after"
    MEMORY_APPEND_BEFORE = "memory_append_before"
    MEMORY_APPEND_AFTER = "memory_append_after"
    MEMORY_PERMANENT_MESSAGE_ADD = "memory_permanent_message_add"
    MEMORY_PERMANENT_MESSAGE_ADDED = "memory_permanent_message_added"
    MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUE = (
        "memory_permanent_message_session_continue"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUED = (
        "memory_permanent_message_session_continued"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_START = (
        "memory_permanent_message_session_start"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_STARTED = (
        "memory_permanent_message_session_started"
    )
    MODEL_EXECUTE_BEFORE = "model_execute_before"
    MODEL_EXECUTE_AFTER = "model_execute_after"
    MODEL_MANAGER_CALL_BEFORE = "model_manager_call_before"
    MODEL_MANAGER_CALL_AFTER = "model_manager_call_after"
    FLOW_CANCELLED = "flow_cancelled"
    FLOW_COMPLETED = "flow_completed"
    FLOW_CONDITION_EVALUATED = "flow_condition_evaluated"
    FLOW_CONTAINER_EVENT = "flow_container_event"
    FLOW_EDGE_ELIGIBLE = "flow_edge_eligible"
    FLOW_EDGE_ROUTED = "flow_edge_routed"
    FLOW_JOIN_READY = "flow_join_ready"
    FLOW_MANAGER_CALL_BEFORE = "flow_manager_call_before"
    FLOW_MANAGER_CALL_AFTER = "flow_manager_call_after"
    FLOW_NODE_CANCELLED = "flow_node_cancelled"
    FLOW_NODE_COMPLETED = "flow_node_completed"
    FLOW_NODE_FAILED = "flow_node_failed"
    FLOW_NODE_PAUSED = "flow_node_paused"
    FLOW_NODE_RESUMED = "flow_node_resumed"
    FLOW_NODE_RETRYING = "flow_node_retrying"
    FLOW_NODE_SKIPPED = "flow_node_skipped"
    FLOW_NODE_STARTED = "flow_node_started"
    FLOW_OUTPUT_SELECTED = "flow_output_selected"
    FLOW_STARTED = "flow_started"
    FLOW_VALIDATION = "flow_validation"
    SKILL_CHECK_DIAGNOSTICS_PRODUCED = "skill_check_diagnostics_produced"
    SKILL_DISABLED = "skill_disabled"
    SKILL_DUPLICATE = "skill_duplicate"
    SKILL_MALFORMED = "skill_malformed"
    SKILL_MATCH_AMBIGUOUS = "skill_match_ambiguous"
    SKILL_MATCH_CANDIDATES_RETURNED = "skill_match_candidates_returned"
    SKILL_MATCH_EMPTY = "skill_match_empty"
    SKILL_MATCH_QUERY_EVALUATED = "skill_match_query_evaluated"
    SKILL_READ_ALLOWED = "skill_read_allowed"
    SKILL_READ_BLOCKED = "skill_read_blocked"
    SKILL_READ_DELETED = "skill_read_deleted"
    SKILL_READ_DENIED = "skill_read_denied"
    SKILL_READ_STALE = "skill_read_stale"
    SKILL_READ_TRUNCATED = "skill_read_truncated"
    SKILL_REGISTERED = "skill_registered"
    SKILL_REGISTRY_BUILD_COMPLETED = "skill_registry_build_completed"
    SKILL_REGISTRY_BUILD_FAILED = "skill_registry_build_failed"
    SKILL_REGISTRY_BUILD_STARTED = "skill_registry_build_started"
    SKILL_SHADOWED = "skill_shadowed"
    SKILL_SOURCE_ACCEPTED = "skill_source_accepted"
    SKILL_SOURCE_POLICY_DENIED = "skill_source_policy_denied"
    SKILL_SOURCE_SKIPPED = "skill_source_skipped"
    SKILL_SOURCE_UNAVAILABLE = "skill_source_unavailable"
    START = "start"
    STREAM_END = "stream_end"
    TOKEN_GENERATED = "token_generated"
    TOOL_DETECT = "tool_detect"
    TOOL_DIAGNOSTIC = "tool_diagnostic"
    TOOL_EXECUTE = "tool_execute"
    TOOL_MODEL_RUN = "tool_model_run"
    TOOL_MODEL_RESPONSE = "tool_model_response"
    TOOL_PROCESS = "tool_process"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"


TOOL_TYPES = {et for et in EventType if et.value.startswith("tool_")}


class EventPayloadKind(StrEnum):
    CANONICAL_STREAM = "canonical_stream"
    TEMPORARY_LEGACY = "temporary_legacy"


@dataclass(frozen=True, kw_only=True, slots=True)
class EventObservabilityPayload:
    kind: EventPayloadKind
    data: Mapping[str, LooseJsonValue]
    owner: str | None = None
    removal_condition: str | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.kind, EventPayloadKind)
        assert isinstance(self.data, Mapping)
        assert self.data
        for key in self.data:
            assert isinstance(key, str)
            assert key.strip()
        if self.kind is EventPayloadKind.TEMPORARY_LEGACY:
            assert isinstance(self.owner, str)
            assert self.owner.strip()
            assert isinstance(self.removal_condition, str)
            assert self.removal_condition.strip()
        else:
            assert self.owner is None
            assert self.removal_condition is None

    @classmethod
    def canonical_stream(
        cls, data: Mapping[str, LooseJsonValue]
    ) -> "EventObservabilityPayload":
        return cls(kind=EventPayloadKind.CANONICAL_STREAM, data=data)

    @classmethod
    def temporary_legacy(
        cls,
        data: Mapping[str, LooseJsonValue],
        *,
        owner: str,
        removal_condition: str,
    ) -> "EventObservabilityPayload":
        return cls(
            kind=EventPayloadKind.TEMPORARY_LEGACY,
            data=data,
            owner=owner,
            removal_condition=removal_condition,
        )

    def to_dict(self) -> dict[str, LooseJsonValue]:
        payload: dict[str, LooseJsonValue] = {
            "kind": self.kind.value,
            "data": dict(self.data),
        }
        if self.owner is not None:
            payload["owner"] = self.owner
        if self.removal_condition is not None:
            payload["removal_condition"] = self.removal_condition
        return payload


@dataclass(frozen=True, kw_only=True, slots=True)
class Event:
    type: EventType
    payload: Any | None = None
    observability_payload: EventObservabilityPayload | None = None
    started: float | None = None
    finished: float | None = None
    elapsed: float | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.type, EventType | str)
        if self.observability_payload is not None:
            assert isinstance(
                self.observability_payload, EventObservabilityPayload
            )
        for field_name, value in (
            ("started", self.started),
            ("finished", self.finished),
            ("elapsed", self.elapsed),
        ):
            if value is not None:
                assert_non_negative_number(value, field_name)

    @classmethod
    def from_observability_payload(
        cls,
        *,
        type: EventType,
        observability_payload: EventObservabilityPayload,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> "Event":
        return cls(
            type=type,
            payload=dict(observability_payload.data),
            observability_payload=observability_payload,
            started=started,
            finished=finished,
            elapsed=elapsed,
        )

    @property
    def observability(self) -> EventObservabilityPayload:
        if self.observability_payload is not None:
            return self.observability_payload
        return EventObservabilityPayload.temporary_legacy(
            self._legacy_observability_data(),
            owner="event-listener-facade",
            removal_condition=(
                "Remove after CLI, server, and protocol listeners consume "
                "canonical stream projections."
            ),
        )

    def for_history(self) -> "Event":
        if self.payload is None:
            return self
        observability_payload = self.observability
        if (
            self.observability_payload is observability_payload
            and isinstance(self.payload, Mapping)
            and dict(self.payload) == dict(observability_payload.data)
        ):
            return self
        return Event(
            type=self.type,
            observability_payload=observability_payload,
            started=self.started,
            finished=self.finished,
            elapsed=self.elapsed,
        )

    def _legacy_observability_data(self) -> dict[str, LooseJsonValue]:
        event_type = (
            self.type.value if isinstance(self.type, EventType) else self.type
        )
        data: dict[str, LooseJsonValue] = {"event_type": event_type}
        if self.started is not None:
            data["started"] = float(self.started)
        if self.finished is not None:
            data["finished"] = float(self.finished)
        if self.elapsed is not None:
            data["elapsed"] = float(self.elapsed)
        if self.payload is not None:
            data["payload_summary"] = cast(
                dict[str, object], self._payload_summary()
            )
        return data

    def _payload_summary(self) -> dict[str, LooseJsonValue]:
        payload = self.payload
        summary: dict[str, LooseJsonValue] = {
            "type": type(payload).__name__,
        }
        if isinstance(payload, Mapping):
            keys, keys_truncated = observability_key_sample(payload)
            summary["keys"] = cast(list[object], keys)
            summary["size"] = len(payload)
            if keys_truncated:
                summary["keys_truncated"] = True
        elif isinstance(payload, list | tuple | set | frozenset):
            summary["size"] = len(payload)
        return summary


@dataclass(frozen=True, kw_only=True, slots=True)
class EventStatsSnapshot:
    triggers: Mapping[EventType, int]
    total_triggers: int
    published: int
    delivered: int
    dropped: int
    coalesced: int
    failed: int
    queue_depth: int
    max_queue_depth: int
    listener_lag: float
    critical_wait_time: float


@dataclass(slots=True)
class EventStats:
    triggers: dict[EventType, int] = field(default_factory=dict)
    total_triggers: int = 0
    published: int = 0
    delivered: int = 0
    dropped: int = 0
    coalesced: int = 0
    failed: int = 0
    queue_depth: int = 0
    max_queue_depth: int = 0
    listener_lag: float = 0.0
    critical_wait_time: float = 0.0

    def record_trigger(self, event_type: EventType) -> None:
        assert isinstance(event_type, EventType)
        self.total_triggers += 1
        self.triggers[event_type] = self.triggers.get(event_type, 0) + 1

    def record_published(
        self, event_type: EventType, *, queue_depth: int = 0
    ) -> None:
        assert_non_negative_int(queue_depth, "queue_depth")
        self.record_trigger(event_type)
        self.published += 1
        self.record_queue_depth(queue_depth)

    def record_delivered(self, *, queue_depth: int = 0) -> None:
        assert_non_negative_int(queue_depth, "queue_depth")
        self.delivered += 1
        self.record_queue_depth(queue_depth)

    def record_dropped(
        self, count: int = 1, *, queue_depth: int | None = None
    ) -> None:
        assert_non_negative_int(count, "count")
        if queue_depth is not None:
            assert_non_negative_int(queue_depth, "queue_depth")
        self.dropped += count
        if queue_depth is not None:
            self.record_queue_depth(queue_depth)

    def record_queue_depth(self, queue_depth: int) -> None:
        assert_non_negative_int(queue_depth, "queue_depth")
        self.queue_depth = queue_depth
        self.max_queue_depth = max(self.max_queue_depth, queue_depth)

    def record_coalesced(self, count: int = 1) -> None:
        assert_non_negative_int(count, "count")
        self.coalesced += count

    def record_failed(self, count: int = 1) -> None:
        assert_non_negative_int(count, "count")
        self.failed += count

    def record_listener_lag(self, seconds: float) -> None:
        assert_non_negative_number(seconds, "seconds")
        self.listener_lag = max(self.listener_lag, float(seconds))

    def record_critical_wait_time(self, seconds: float) -> None:
        assert_non_negative_number(seconds, "seconds")
        self.critical_wait_time = max(self.critical_wait_time, float(seconds))

    def snapshot(self) -> EventStatsSnapshot:
        return EventStatsSnapshot(
            triggers=MappingProxyType(dict(self.triggers)),
            total_triggers=self.total_triggers,
            published=self.published,
            delivered=self.delivered,
            dropped=self.dropped,
            coalesced=self.coalesced,
            failed=self.failed,
            queue_depth=self.queue_depth,
            max_queue_depth=self.max_queue_depth,
            listener_lag=self.listener_lag,
            critical_wait_time=self.critical_wait_time,
        )
