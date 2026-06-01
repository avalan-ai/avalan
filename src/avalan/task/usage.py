from ..types import (
    JsonValue,
)
from ..types import (
    assert_counter as _assert_counter,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

TaskUsageValue: TypeAlias = JsonValue
TaskUsageMetadata: TypeAlias = Mapping[str, TaskUsageValue]


class UsageSource(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


def _empty_metadata() -> TaskUsageMetadata:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageTotals:
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None

    def __post_init__(self) -> None:
        _assert_counter(self.input_tokens, "input_tokens")
        _assert_counter(self.cached_input_tokens, "cached_input_tokens")
        _assert_counter(
            self.cache_creation_input_tokens,
            "cache_creation_input_tokens",
        )
        _assert_counter(self.output_tokens, "output_tokens")
        _assert_counter(self.reasoning_tokens, "reasoning_tokens")
        _assert_counter(self.total_tokens, "total_tokens")

    @property
    def has_observations(self) -> bool:
        return any(
            value is not None
            for value in (
                self.input_tokens,
                self.cached_input_tokens,
                self.cache_creation_input_tokens,
                self.output_tokens,
                self.reasoning_tokens,
                self.total_tokens,
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageObservation:
    source: UsageSource
    totals: UsageTotals

    def __post_init__(self) -> None:
        assert isinstance(self.source, UsageSource)
        assert isinstance(self.totals, UsageTotals)


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageRecord:
    usage_id: str
    run_id: str
    attempt_id: str | None
    sequence: int
    source: UsageSource
    totals: UsageTotals
    created_at: datetime
    metadata: TaskUsageMetadata = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.usage_id, "usage_id")
        _assert_non_empty_string(self.run_id, "run_id")
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        assert isinstance(self.sequence, int), "sequence must be an integer"
        assert not isinstance(
            self.sequence, bool
        ), "sequence must be an integer"
        assert self.sequence > 0, "sequence must be positive"
        assert isinstance(self.source, UsageSource)
        assert isinstance(self.totals, UsageTotals)
        assert isinstance(self.created_at, datetime)
        object.__setattr__(
            self,
            "metadata",
            freeze_usage_metadata(self.metadata),
        )


class UsageResponse(Protocol):
    input_token_count: int | None
    output_token_count: int | None


class UsageCallbackResponse(Protocol):
    def add_done_callback(
        self,
        callback: Callable[[], Awaitable[None] | None],
    ) -> None: ...


class TaskUsageStore(Protocol):
    async def append_usage(
        self,
        run_id: str,
        *,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UsageRecord: ...


def freeze_usage_metadata(
    value: Mapping[str, object] | None,
) -> TaskUsageMetadata:
    if value is None:
        return _empty_metadata()
    assert isinstance(value, Mapping), "metadata must be a mapping"
    return cast(TaskUsageMetadata, freeze_usage_value(value))


def freeze_usage_value(value: object) -> TaskUsageValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        assert isfinite(value), "usage floats must be finite"
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, TaskUsageValue] = {}
        for key, item in value.items():
            assert isinstance(key, str), "usage metadata keys must be strings"
            assert key.strip(), "usage metadata keys must not be empty"
            frozen[key] = freeze_usage_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(freeze_usage_value(item) for item in value)
    raise AssertionError("usage metadata must be privacy-safe")


def usage_observation_from_response(
    response: object,
) -> UsageObservation | None:
    usage = _usage_container(response)
    if usage is not None:
        totals = _usage_totals_from_value(usage, _PROVIDER_COUNTER_PATHS)
        if totals is not None:
            return UsageObservation(source=UsageSource.EXACT, totals=totals)

    totals = _usage_totals_from_value(response, _RESPONSE_COUNTER_PATHS)
    if totals is None:
        return None
    return UsageObservation(
        source=_usage_source_from_response(response),
        totals=totals,
    )


def usage_totals_from_response(response: object) -> UsageTotals | None:
    observation = usage_observation_from_response(response)
    return observation.totals if observation is not None else None


def attach_response_usage_recorder(
    response: object,
    *,
    store: TaskUsageStore,
    run_id: str,
    attempt_id: str | None = None,
    source: UsageSource | None = None,
    metadata: Mapping[str, object] | None = None,
) -> bool:
    add_done_callback = getattr(response, "add_done_callback", None)
    if not callable(add_done_callback):
        return False
    recorded = False

    async def record_usage() -> None:
        nonlocal recorded
        if recorded:
            return
        recorded = True
        observation = usage_observation_from_response(response)
        if observation is None:
            return
        await store.append_usage(
            run_id,
            attempt_id=attempt_id,
            source=source or observation.source,
            totals=observation.totals,
            metadata=metadata,
        )

    cast(UsageCallbackResponse, response).add_done_callback(record_usage)
    return True


def aggregate_usage_totals(records: tuple[UsageRecord, ...]) -> UsageTotals:
    return UsageTotals(
        input_tokens=_sum_counter(records, "input_tokens"),
        cached_input_tokens=_sum_counter(records, "cached_input_tokens"),
        cache_creation_input_tokens=_sum_counter(
            records,
            "cache_creation_input_tokens",
        ),
        output_tokens=_sum_counter(records, "output_tokens"),
        reasoning_tokens=_sum_counter(records, "reasoning_tokens"),
        total_tokens=_sum_counter(records, "total_tokens"),
    )


def _sum_counter(
    records: tuple[UsageRecord, ...],
    field_name: str,
) -> int | None:
    total = 0
    observed = False
    for record in records:
        value = _counter_value(record.totals, field_name)
        if value is None:
            continue
        observed = True
        total += value
    if not observed:
        return None
    return total


def _counter_value(totals: UsageTotals, field_name: str) -> int | None:
    match field_name:
        case "input_tokens":
            return totals.input_tokens
        case "cached_input_tokens":
            return totals.cached_input_tokens
        case "cache_creation_input_tokens":
            return totals.cache_creation_input_tokens
        case "output_tokens":
            return totals.output_tokens
        case "reasoning_tokens":
            return totals.reasoning_tokens
        case "total_tokens":
            return totals.total_tokens
        case _:
            raise AssertionError("unknown usage counter")


CounterPath = tuple[str, ...]
CounterPathMap = Mapping[str, tuple[CounterPath, ...]]

_PROVIDER_COUNTER_PATHS: CounterPathMap = {
    "input_tokens": (
        ("input_tokens",),
        ("input_token_count",),
        ("inputTokens",),
        ("prompt_tokens",),
        ("prompt_token_count",),
    ),
    "cached_input_tokens": (
        ("cached_input_tokens",),
        ("cached_input_token_count",),
        ("cache_read_input_tokens",),
        ("cacheReadInputTokens",),
        ("cached_content_token_count",),
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    ),
    "cache_creation_input_tokens": (
        ("cache_creation_input_tokens",),
        ("cache_creation_input_token_count",),
        ("cacheCreationInputTokens",),
    ),
    "output_tokens": (
        ("output_tokens",),
        ("output_token_count",),
        ("outputTokens",),
        ("completion_tokens",),
        ("candidates_token_count",),
    ),
    "reasoning_tokens": (
        ("reasoning_tokens",),
        ("reasoning_token_count",),
        ("reasoningTokens",),
        ("thoughts_token_count",),
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ),
    "total_tokens": (
        ("total_tokens",),
        ("total_token_count",),
        ("totalTokens",),
    ),
}

_RESPONSE_COUNTER_PATHS: CounterPathMap = {
    "input_tokens": (("input_token_count",),),
    "cached_input_tokens": (("cached_input_token_count",),),
    "cache_creation_input_tokens": (("cache_creation_input_token_count",),),
    "output_tokens": (("output_token_count",),),
    "reasoning_tokens": (("reasoning_token_count",),),
    "total_tokens": (("total_token_count",),),
}


def _usage_container(response: object) -> object | None:
    for attribute in ("usage", "usage_metadata"):
        value = _value_at_path(response, (attribute,))
        if value is not None:
            return value
    return None


def _usage_totals_from_value(
    value: object, counter_paths: CounterPathMap
) -> UsageTotals | None:
    counters = {
        name: _counter_from_paths(value, paths)
        for name, paths in counter_paths.items()
    }
    if not any(counter is not None for counter in counters.values()):
        return None
    return UsageTotals(
        input_tokens=counters["input_tokens"],
        cached_input_tokens=counters["cached_input_tokens"],
        cache_creation_input_tokens=counters["cache_creation_input_tokens"],
        output_tokens=counters["output_tokens"],
        reasoning_tokens=counters["reasoning_tokens"],
        total_tokens=counters["total_tokens"],
    )


def _counter_from_paths(
    value: object, paths: tuple[CounterPath, ...]
) -> int | None:
    for path in paths:
        counter = _counter_from_value(_value_at_path(value, path))
        if counter is not None:
            return counter
    return None


def _counter_from_value(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _usage_source_from_response(response: object) -> UsageSource:
    source = _value_at_path(response, ("usage_source",))
    if isinstance(source, UsageSource):
        return source
    if isinstance(source, str):
        try:
            return UsageSource(source)
        except ValueError:
            return UsageSource.ESTIMATED
    return UsageSource.ESTIMATED


def _value_at_path(value: object, path: CounterPath) -> object | None:
    current = value
    for item in path:
        if isinstance(current, Mapping):
            current = current.get(item)
        else:
            current = getattr(current, item, None)
        if current is None:
            return None
    return current
