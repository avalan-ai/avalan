from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from math import isfinite
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast

TaskUsageValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["TaskUsageValue", ...]
    | Mapping[str, "TaskUsageValue"]
)
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


def usage_totals_from_response(response: object) -> UsageTotals | None:
    counters = (
        _counter_from_attribute(response, "input_token_count"),
        _counter_from_attribute(response, "cached_input_token_count"),
        _counter_from_attribute(response, "cache_creation_input_token_count"),
        _counter_from_attribute(response, "output_token_count"),
        _counter_from_attribute(response, "reasoning_token_count"),
        _counter_from_attribute(response, "total_token_count"),
    )
    if not any(value is not None for value in counters):
        return None
    total_tokens = counters[5]
    if (
        total_tokens is None
        and counters[0] is not None
        and counters[3] is not None
    ):
        total_tokens = counters[0] + counters[3]
    return UsageTotals(
        input_tokens=counters[0],
        cached_input_tokens=counters[1],
        cache_creation_input_tokens=counters[2],
        output_tokens=counters[3],
        reasoning_tokens=counters[4],
        total_tokens=total_tokens,
    )


def attach_response_usage_recorder(
    response: object,
    *,
    store: TaskUsageStore,
    run_id: str,
    attempt_id: str | None = None,
    source: UsageSource = UsageSource.ESTIMATED,
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
        totals = usage_totals_from_response(response)
        if totals is None:
            return
        await store.append_usage(
            run_id,
            attempt_id=attempt_id,
            source=source,
            totals=totals,
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


def _counter_from_attribute(response: object, name: str) -> int | None:
    value = getattr(response, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _assert_counter(value: int | None, field_name: str) -> None:
    if value is None:
        return
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    assert value >= 0, f"{field_name} must be non-negative"


def _assert_non_empty_string(value: str | None, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"
