from ..event import (
    Event,
    EventObservabilityPayload,
    EventPayloadKind,
    EventType,
)
from ..types import (
    JsonValue,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from .privacy import REDACTED_MARKER, PrivacySanitizer
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


class TaskEventCategory(StrEnum):
    TOKEN = "token"
    TOOL = "tool"
    MODEL = "model"
    ENGINE = "engine"
    MEMORY = "memory"
    USAGE = "usage"
    UNKNOWN = "unknown"


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
    metadata: Mapping[str, object] | None = None
    event_type: str = _USAGE_OBSERVED_EVENT_TYPE
    category: TaskEventCategory = TaskEventCategory.USAGE
    payload: TaskEventValue = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
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
    return SanitizedTaskEventDraft(
        event_type=event_type,
        category=task_event_category(event_type),
        payload=cast(TaskEventValue, payload),
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
