from ..event import (
    Event,
    EventObservabilityPayload,
    EventPayloadKind,
    EventType,
    validate_observer_id,
)
from ..interaction import (
    AnswerProvenance,
    InputErrorCode,
    RequestState,
    ResolutionStatus,
)
from ..types import (
    JsonValue,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .privacy import REDACTED_MARKER, PrivacySafeValue, PrivacySanitizer
from .usage import (
    USAGE_COUNTER_NAMES,
    TaskUsageMetadata,
    UsageSource,
    UsageTotals,
    freeze_usage_metadata,
)

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

TaskEventValue: TypeAlias = JsonValue

_UNKNOWN_EVENT_TYPE = "unknown"
_EVENT_SANITIZATION_FAILED = "event_sanitization_failed"
_TOKEN_EVENT_TYPES = frozenset(
    {
        "input_token_count_after",
        "input_token_count_before",
        "token_generated",
    }
)
_ENGINE_EVENT_TYPES = frozenset(
    {
        "call_prepare_after",
        "call_prepare_before",
        "end",
        "flow_manager_call_after",
        "flow_manager_call_before",
        "start",
        "stream_end",
    }
)
_USAGE_OBSERVED_EVENT_TYPE = "usage_observed"
_USAGE_EVENT_TYPES = frozenset({_USAGE_OBSERVED_EVENT_TYPE})
_RAW_EVENT_SHAPE_ATTRIBUTES = frozenset(
    {
        "type",
        "payload",
        "observability_payload",
        "started",
        "finished",
        "elapsed",
    }
)
_REASONING_OBSERVABILITY_SUMMARY_FIELDS = frozenset(
    {
        "reasoning_representation",
        "segment_instance_ordinal",
        "text_delta_length",
    }
)
_REASONING_OBSERVABILITY_CORRELATION_STRING_FIELDS = frozenset(
    {
        "model_continuation_id",
        "protocol_item_id",
    }
)
_REASONING_OBSERVABILITY_CORRELATION_INDEX_FIELDS = frozenset(
    {
        "provider_output_index",
        "provider_summary_index",
    }
)
_STREAM_TERMINAL_OUTCOMES = frozenset(
    {
        "cancelled",
        "completed",
        "errored",
        "input_required",
    }
)
_INTERACTION_OBSERVABILITY_BOOLEAN_FIELDS = frozenset(
    {
        "duplicate",
        "stale",
    }
)
_INTERACTION_REQUEST_STATES = frozenset(item.value for item in RequestState)
_INTERACTION_RESOLUTION_STATUSES = frozenset(
    item.value for item in ResolutionStatus
)
_INTERACTION_PROVENANCE_CATEGORIES = frozenset(
    item.value for item in AnswerProvenance
)
_INTERACTION_VALIDATION_CODES = frozenset(
    item.value for item in InputErrorCode
)
_TELEMETRY_NAME_CHARACTERS = frozenset("._:-")
_TELEMETRY_NAME_LIMIT = 256
_CORRELATION_SURROGATE_PREFIX = "sha256:"
_CORRELATION_SURROGATE_HEX_LENGTH = 64
_LOWERCASE_HEX_CHARACTERS = frozenset("0123456789abcdef")


class TaskEventCategory(StrEnum):
    TOKEN = "token"
    TOOL = "tool"
    MODEL = "model"
    ENGINE = "engine"
    MEMORY = "memory"
    USAGE = "usage"
    INTERACTION = "interaction"
    UNKNOWN = "unknown"


class TaskInteractionEventType(StrEnum):
    INPUT_REQUIRED = "task_input_required"
    INPUT_RESUMED = "task_input_resumed"
    INPUT_CANCELLED = "task_input_cancelled"
    INPUT_EXPIRED = "task_input_expired"
    INPUT_SUPERSEDED = "task_input_superseded"


def task_interaction_event_payload(
    *,
    event_type: TaskInteractionEventType,
    request_id: str,
    continuation_id: str,
    segment_id: str,
    next_segment_id: str | None = None,
) -> TaskEventValue:
    assert isinstance(event_type, TaskInteractionEventType)
    _assert_non_empty_string(request_id, "request_id")
    _assert_non_empty_string(continuation_id, "continuation_id")
    _assert_non_empty_string(segment_id, "segment_id")
    if next_segment_id is not None:
        _assert_non_empty_string(next_segment_id, "next_segment_id")
        assert event_type in {
            TaskInteractionEventType.INPUT_RESUMED,
            TaskInteractionEventType.INPUT_SUPERSEDED,
        }
    value: dict[str, object] = {
        "request_id": request_id,
        "continuation_id": continuation_id,
        "segment_id": segment_id,
    }
    if next_segment_id is not None:
        value["next_segment_id"] = next_segment_id
    return freeze_task_event_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedTaskEventDraft:
    event_type: str
    category: TaskEventCategory
    payload: TaskEventValue = None

    def __post_init__(self) -> None:
        _assert_event_type(self.event_type)
        assert isinstance(self.category, TaskEventCategory)
        object.__setattr__(
            self,
            "payload",
            freeze_task_event_value(self.payload),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedTaskEvent:
    event_id: str
    run_id: str
    sequence: int
    event_type: str
    category: TaskEventCategory
    created_at: datetime
    payload: TaskEventValue = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.event_id, "event_id")
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.sequence, int)
        assert not isinstance(self.sequence, bool)
        assert self.sequence > 0
        _assert_event_type(self.event_type)
        assert isinstance(self.category, TaskEventCategory)
        assert isinstance(self.created_at, datetime)
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        object.__setattr__(
            self,
            "payload",
            freeze_task_event_value(self.payload),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SanitizedTaskUsageEvent:
    run_id: str
    source: UsageSource
    totals: UsageTotals
    attempt_id: str | None = None
    segment_id: str | None = None
    metadata: Mapping[str, object] | None = None
    event_type: str = _USAGE_OBSERVED_EVENT_TYPE
    category: TaskEventCategory = TaskEventCategory.USAGE
    payload: TaskEventValue = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        if self.segment_id is not None:
            _assert_non_empty_string(self.segment_id, "segment_id")
            assert (
                self.attempt_id is not None
            ), "segment usage requires an attempt"
        assert isinstance(self.source, UsageSource)
        assert isinstance(self.totals, UsageTotals)
        assert self.totals.has_observations
        assert self.event_type == _USAGE_OBSERVED_EVENT_TYPE
        assert self.category == TaskEventCategory.USAGE
        metadata = freeze_usage_metadata(self.metadata)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(
            self,
            "payload",
            freeze_task_event_value(
                _usage_event_payload(self.source, self.totals, metadata)
            ),
        )


class TaskEventStore(Protocol):
    async def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        category: TaskEventCategory,
        payload: TaskEventValue,
        attempt_id: str | None = None,
    ) -> SanitizedTaskEvent: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class RawTaskEventListener:
    store: TaskEventStore
    run_id: str
    sanitizer: PrivacySanitizer
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.sanitizer, PrivacySanitizer)
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")

    async def __call__(self, event: Event) -> None:
        draft = sanitize_raw_task_event_closed(event, self.sanitizer)
        await self.store.append_event(
            self.run_id,
            attempt_id=self.attempt_id,
            category=draft.category,
            event_type=draft.event_type,
            payload=draft.payload,
        )


def sanitize_raw_task_event(
    event: object,
    sanitizer: PrivacySanitizer,
) -> SanitizedTaskEventDraft:
    assert isinstance(sanitizer, PrivacySanitizer)
    event_type = _raw_event_type(event)
    payload = sanitizer.sanitize_event(
        event_type,
        _raw_event_payload(event),
    )
    assert isinstance(payload, dict)
    _restore_reasoning_observability(event, payload)
    _restore_interaction_observability(event, payload)
    return SanitizedTaskEventDraft(
        event_type=event_type,
        category=task_event_category(event_type),
        payload=cast(TaskEventValue, payload),
    )


def _restore_reasoning_observability(
    event: object,
    payload: dict[str, PrivacySafeValue],
) -> None:
    observability_payload = getattr(event, "observability_payload", None)
    if (
        not isinstance(observability_payload, EventObservabilityPayload)
        or observability_payload.kind is not EventPayloadKind.CANONICAL_STREAM
    ):
        return
    canonical_stream = payload.get("canonical_stream")
    if not isinstance(canonical_stream, dict):
        return
    source = observability_payload.data
    if source.get("kind") != "reasoning.delta":
        return
    provider_family = source.get("provider_family")
    if _is_safe_provider_family(provider_family):
        canonical_stream["provider_family"] = cast(str, provider_family)
    provider_event_type = source.get("provider_event_type")
    if _is_safe_provider_event_type(provider_event_type):
        canonical_stream["provider_event_type"] = cast(
            str,
            provider_event_type,
        )
    terminal_outcome = source.get("terminal_outcome")
    if (
        isinstance(terminal_outcome, str)
        and terminal_outcome in _STREAM_TERMINAL_OUTCOMES
    ):
        canonical_stream["terminal_outcome"] = terminal_outcome
    correlation = _reasoning_observability_correlation(
        source.get("correlation")
    )
    if correlation:
        canonical_stream["correlation"] = correlation
    source_summary = source.get("summary")
    summary = _reasoning_observability_summary(source_summary)
    if summary:
        canonical_stream["summary"] = summary
    usage = _reasoning_observability_usage(source.get("usage"))
    if usage:
        canonical_stream["usage"] = usage


def _restore_interaction_observability(
    event: object,
    payload: dict[str, PrivacySafeValue],
) -> None:
    observability_payload = getattr(event, "observability_payload", None)
    if not isinstance(observability_payload, EventObservabilityPayload):
        return
    if observability_payload.kind not in {
        EventPayloadKind.CANONICAL_STREAM,
        EventPayloadKind.INTERACTION_LIFECYCLE,
    }:
        return
    event_type = _raw_event_type(event)
    if event_type != EventType.INTERACTION_LIFECYCLE.value:
        return
    payload.pop("canonical_stream", None)
    payload.pop("interaction_lifecycle", None)
    interaction = _interaction_observability_fields(observability_payload.data)
    if interaction:
        payload["interaction_lifecycle"] = interaction


def _interaction_observability_fields(
    source: Mapping[str, object],
) -> dict[str, PrivacySafeValue]:
    interaction: dict[str, PrivacySafeValue] = {}
    for field_name in (
        "agent_id",
        "branch_id",
        "request_id",
        "run_id",
        "turn_id",
    ):
        value = source.get(field_name)
        try:
            interaction[field_name] = validate_observer_id(value, field_name)
        except AssertionError:
            return {}
    task_id = source.get("task_id")
    if task_id is not None:
        try:
            interaction["task_id"] = validate_observer_id(task_id, "task_id")
        except AssertionError:
            return {}
    state = source.get("state")
    if not isinstance(state, str) or state not in _INTERACTION_REQUEST_STATES:
        return {}
    interaction["state"] = state
    resolution = source.get("resolution_category")
    if (
        isinstance(resolution, str)
        and resolution in _INTERACTION_RESOLUTION_STATUSES
    ):
        interaction["resolution_category"] = resolution
    provenance = source.get("provenance_category")
    if (
        isinstance(provenance, str)
        and provenance in _INTERACTION_PROVENANCE_CATEGORIES
    ):
        interaction["provenance_category"] = provenance
    surface = source.get("surface")
    if _is_safe_provider_event_type(surface):
        interaction["surface"] = cast(str, surface)
    wait_duration_ms = source.get("wait_duration_ms")
    if _is_non_negative_int(wait_duration_ms):
        interaction["wait_duration_ms"] = cast(int, wait_duration_ms)
    validation_code = source.get("validation_code")
    if (
        isinstance(validation_code, str)
        and validation_code in _INTERACTION_VALIDATION_CODES
    ):
        interaction["validation_code"] = validation_code
    for field_name in _INTERACTION_OBSERVABILITY_BOOLEAN_FIELDS:
        value = source.get(field_name)
        if isinstance(value, bool):
            interaction[field_name] = value
    return interaction


def _reasoning_observability_summary(
    value: object,
) -> dict[str, PrivacySafeValue]:
    if not isinstance(value, Mapping):
        return {}
    summary: dict[str, PrivacySafeValue] = {}
    for field_name in _REASONING_OBSERVABILITY_SUMMARY_FIELDS:
        field_value = value.get(field_name)
        if _is_safe_reasoning_observability_summary_field(
            field_name,
            field_value,
        ):
            summary[field_name] = cast(PrivacySafeValue, field_value)
    return summary


def _reasoning_observability_correlation(
    value: object,
) -> dict[str, PrivacySafeValue]:
    if not isinstance(value, Mapping):
        return {}
    correlation: dict[str, PrivacySafeValue] = {}
    for field_name in _REASONING_OBSERVABILITY_CORRELATION_STRING_FIELDS:
        field_value = value.get(field_name)
        if _is_safe_correlation_value(field_value):
            correlation[field_name] = cast(str, field_value)
    for field_name in _REASONING_OBSERVABILITY_CORRELATION_INDEX_FIELDS:
        field_value = value.get(field_name)
        if _is_non_negative_int(field_value):
            correlation[field_name] = cast(int, field_value)
    return correlation


def _reasoning_observability_usage(
    value: object,
) -> dict[str, PrivacySafeValue]:
    if not isinstance(value, Mapping):
        return {}
    usage: dict[str, PrivacySafeValue] = {}
    for field_name in USAGE_COUNTER_NAMES:
        field_value = value.get(field_name)
        if _is_non_negative_int(field_value):
            usage[field_name] = cast(int, field_value)
    return usage


def _is_safe_reasoning_observability_summary_field(
    field_name: str,
    value: object,
) -> bool:
    if field_name == "reasoning_representation":
        return isinstance(value, str) and value in {"native_text", "summary"}
    return _is_non_negative_int(value)


def _is_safe_provider_family(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _TELEMETRY_NAME_LIMIT
        and value == value.strip()
        and value.isascii()
        and any(character.islower() for character in value)
        and all(
            character.islower()
            or character.isdigit()
            or character in _TELEMETRY_NAME_CHARACTERS
            for character in value
        )
    )


def _is_safe_provider_event_type(value: object) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _TELEMETRY_NAME_LIMIT
        and value == value.strip()
        and value.isascii()
        and any(character.islower() for character in value)
        and all(
            character.isalnum() or character in _TELEMETRY_NAME_CHARACTERS
            for character in value
        )
    )


def _is_safe_correlation_value(value: object) -> bool:
    if not isinstance(value, str):
        return False
    if not value.startswith(_CORRELATION_SURROGATE_PREFIX):
        return False
    digest = value.removeprefix(_CORRELATION_SURROGATE_PREFIX)
    return len(digest) == _CORRELATION_SURROGATE_HEX_LENGTH and all(
        character in _LOWERCASE_HEX_CHARACTERS for character in digest
    )


def _is_non_negative_int(value: object) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
    )


def sanitize_raw_task_event_closed(
    event: object,
    sanitizer: PrivacySanitizer,
) -> SanitizedTaskEventDraft:
    assert isinstance(sanitizer, PrivacySanitizer)
    try:
        return sanitize_raw_task_event(event, sanitizer)
    except Exception:
        return SanitizedTaskEventDraft(
            event_type=_EVENT_SANITIZATION_FAILED,
            category=TaskEventCategory.UNKNOWN,
            payload={
                "event_type": _EVENT_SANITIZATION_FAILED,
                "privacy": REDACTED_MARKER,
            },
        )


def task_event_category(event_type: str) -> TaskEventCategory:
    _assert_event_type(event_type)
    if event_type in _TOKEN_EVENT_TYPES:
        return TaskEventCategory.TOKEN
    if event_type.startswith("tool_"):
        return TaskEventCategory.TOOL
    if event_type.startswith("skill_"):
        return TaskEventCategory.TOOL
    if event_type.startswith("model_"):
        return TaskEventCategory.MODEL
    if event_type.startswith("memory_"):
        return TaskEventCategory.MEMORY
    if event_type.startswith("interaction_"):
        return TaskEventCategory.INTERACTION
    if event_type.startswith("container_"):
        return TaskEventCategory.UNKNOWN
    if event_type in _USAGE_EVENT_TYPES:
        return TaskEventCategory.USAGE
    if (
        event_type in _ENGINE_EVENT_TYPES
        or event_type.startswith("engine_")
        or event_type.startswith("flow_")
    ):
        return TaskEventCategory.ENGINE
    return TaskEventCategory.UNKNOWN


def freeze_task_event_value(value: object) -> TaskEventValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        assert isfinite(value), "event floats must be finite"
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, TaskEventValue] = {}
        for key, item in value.items():
            assert isinstance(key, str), "event payload keys must be strings"
            assert key.strip(), "event payload keys must not be empty"
            frozen[key] = freeze_task_event_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(freeze_task_event_value(item) for item in value)
    raise AssertionError("event payload must be privacy-safe")


def _raw_event_type(event: object) -> str:
    if not _has_raw_event_shape(event):
        return _UNKNOWN_EVENT_TYPE
    raw_type = getattr(event, "type", None)
    if isinstance(raw_type, EventType):
        return raw_type.value
    if isinstance(raw_type, str) and _is_safe_event_type(raw_type):
        return raw_type
    return _UNKNOWN_EVENT_TYPE


def _raw_event_payload(event: object) -> Mapping[str, object]:
    if not _has_raw_event_shape(event):
        return MappingProxyType({})
    payload: dict[str, object] = {}
    raw_payload = getattr(event, "payload", None)
    if isinstance(raw_payload, Mapping):
        payload.update(raw_payload)
    observability_payload = getattr(event, "observability_payload", None)
    if (
        isinstance(observability_payload, EventObservabilityPayload)
        and observability_payload.kind is EventPayloadKind.CANONICAL_STREAM
    ):
        payload["canonical_stream"] = dict(observability_payload.data)
    elif (
        isinstance(observability_payload, EventObservabilityPayload)
        and observability_payload.kind
        is EventPayloadKind.INTERACTION_LIFECYCLE
    ):
        payload["interaction_lifecycle"] = dict(observability_payload.data)
    started = getattr(event, "started", None)
    finished = getattr(event, "finished", None)
    elapsed = getattr(event, "elapsed", None)
    if _finite_number(started):
        payload["started_at"] = started
    if _finite_number(finished):
        payload["finished_at"] = finished
    if _finite_number(elapsed):
        elapsed = cast(float, elapsed)
        payload["duration_ms"] = elapsed * 1000
    return MappingProxyType(payload)


def _has_raw_event_shape(event: object) -> bool:
    return all(
        hasattr(event, attribute) for attribute in _RAW_EVENT_SHAPE_ATTRIBUTES
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _assert_event_type(value: str) -> None:
    assert isinstance(value, str), "event_type must be a string"
    assert _is_safe_event_type(value), "event_type must be safe"


def _is_safe_event_type(value: str) -> bool:
    if not value or len(value) > 64:
        return False
    if not value[0].isalpha():
        return False
    return all(character.isalnum() or character == "_" for character in value)


def _usage_event_payload(
    source: UsageSource,
    totals: UsageTotals,
    metadata: TaskUsageMetadata,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_type": _USAGE_OBSERVED_EVENT_TYPE,
        "source": source.value,
    }
    provider_family = metadata.get("provider_family")
    if isinstance(provider_family, str):
        payload["provider_family"] = provider_family
    flow_node = metadata.get("flow_node")
    if isinstance(flow_node, str):
        payload["flow_node"] = flow_node
    for counter_name in USAGE_COUNTER_NAMES:
        value = getattr(totals, counter_name)
        if value is not None:
            payload[counter_name] = value
    return payload
