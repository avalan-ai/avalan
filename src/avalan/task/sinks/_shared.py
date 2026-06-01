from ..event import TaskEventCategory
from ..usage import UsageTotals

_OTHER_EVENT_TYPE = "other"
_UNKNOWN_EVENT_TYPE = "unknown"
KNOWN_EVENT_TYPES = (
    "call_prepare_after",
    "call_prepare_before",
    "end",
    "engine_error",
    "engine_start",
    "engine_stop",
    "event_sanitization_failed",
    "input_token_count_after",
    "input_token_count_before",
    "memory_lookup",
    "memory_store",
    "model_complete",
    "model_end",
    "model_error",
    "model_start",
    "start",
    "stream_end",
    "token_generated",
    "tool_call",
    "tool_error",
    "tool_result",
    "unknown",
)


def event_type_label(
    event_type: str,
    category: TaskEventCategory,
    known_event_types: tuple[str, ...],
) -> str:
    if event_type in known_event_types:
        return event_type
    if category == TaskEventCategory.UNKNOWN:
        return _UNKNOWN_EVENT_TYPE
    return _OTHER_EVENT_TYPE


def usage_counter_values(
    totals: UsageTotals,
) -> tuple[tuple[str, int], ...]:
    values: list[tuple[str, int]] = []
    for counter_name in (
        "input_tokens",
        "cached_input_tokens",
        "cache_creation_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "total_tokens",
    ):
        value = getattr(totals, counter_name)
        if value is not None:
            values.append((counter_name, value))
    return tuple(values)


def assert_label_value(value: str | None, field_name: str) -> None:
    assert_non_empty_string(value, field_name)
    assert value is not None
    assert len(value) <= 64, f"{field_name} must be no longer than 64 chars"
    assert all(
        character.isalnum() or character == "_" for character in value
    ), f"{field_name} must be a safe label value"


def assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
