from ..types import (
    JsonValue,
)
from ..types import (
    assert_counter as _assert_counter,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from hashlib import sha256
from itertools import count
from math import isfinite
from re import fullmatch
from types import MappingProxyType
from typing import Protocol, TypeAlias, cast
from weakref import WeakKeyDictionary

TaskUsageValue: TypeAlias = JsonValue
TaskUsageMetadata: TypeAlias = Mapping[str, TaskUsageValue]


class UsageSource(StrEnum):
    EXACT = "exact"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class UsageProviderFamily(StrEnum):
    ANTHROPIC = "anthropic"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    GOOGLE = "google"
    HUGGING_FACE = "hugging_face"
    LOCAL = "local"
    OLLAMA = "ollama"
    OPENAI = "openai"
    OPENAI_COMPATIBLE = "openai_compatible"
    OTHER = "other"


class UsageCounterPresence(StrEnum):
    MISSING = "missing"
    REPORTED_ZERO = "reported_zero"
    REPORTED_POSITIVE = "reported_positive"


USAGE_COUNTER_NAMES = (
    "input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)
USAGE_METADATA_KEYS = (
    "provider_family",
    "cache_creation_ephemeral_5m_input_tokens",
    "cache_creation_ephemeral_1h_input_tokens",
    "cache_read_ephemeral_5m_input_tokens",
    "cache_read_ephemeral_1h_input_tokens",
)
_LOCAL_USAGE_CALL_KEY_ATTRIBUTE = "_avalan_usage_call_key"
_LOCAL_USAGE_CALL_KEY_COUNTER = count(1)
_LOCAL_USAGE_CALL_KEYS: WeakKeyDictionary[object, str] = WeakKeyDictionary()
_LOCAL_UNTRACKED_USAGE_CALL_KEYS: dict[int, list[tuple[object, str]]] = {}


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
    metadata: TaskUsageMetadata = field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        assert isinstance(self.source, UsageSource)
        assert isinstance(self.totals, UsageTotals)
        object.__setattr__(
            self,
            "metadata",
            freeze_usage_metadata(self.metadata),
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


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageObservationEntry:
    response: object
    sequence: int
    observation: UsageObservation

    def __post_init__(self) -> None:
        assert isinstance(self.sequence, int), "sequence must be an integer"
        assert not isinstance(
            self.sequence, bool
        ), "sequence must be an integer"
        assert self.sequence > 0, "sequence must be positive"
        assert isinstance(self.observation, UsageObservation)


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
        usage_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UsageRecord: ...

    async def list_usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        source: UsageSource | None = None,
    ) -> tuple[UsageRecord, ...]: ...


def freeze_usage_metadata(
    value: Mapping[str, object] | None,
) -> TaskUsageMetadata:
    if value is None:
        return _empty_metadata()
    assert isinstance(value, Mapping), "metadata must be a mapping"
    metadata: dict[str, TaskUsageValue] = {}
    provider_family = _provider_family_value(value.get("provider_family"))
    if provider_family is not None:
        metadata["provider_family"] = provider_family.value
    for key in USAGE_METADATA_KEYS:
        if key == "provider_family":
            continue
        counter = _counter_from_value(value.get(key))
        if counter is not None:
            metadata[key] = counter
    if not metadata:
        return _empty_metadata()
    return MappingProxyType(metadata)


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
    child_observations = _child_usage_observations(response)
    if child_observations:
        return _aggregate_usage_observations(child_observations)

    usage = _usage_container(response)
    metadata = _usage_metadata_from_response(response, usage)
    if isinstance(usage, list | tuple):
        observations = _usage_observations_from_usage_items(
            usage,
            response=response,
        )
        if observations:
            return _aggregate_usage_observations(observations)

    if usage is not None:
        totals = _usage_totals_from_value(usage, _PROVIDER_COUNTER_PATHS)
        if totals is not None:
            return UsageObservation(
                source=UsageSource.EXACT,
                totals=totals,
                metadata=metadata,
            )

    totals = _usage_totals_from_value(response, _RESPONSE_COUNTER_PATHS)
    if totals is None:
        return None
    return UsageObservation(
        source=UsageSource.ESTIMATED,
        totals=totals,
        metadata=metadata,
    )


def usage_counter_presence(
    totals: UsageTotals,
) -> Mapping[str, UsageCounterPresence]:
    assert isinstance(totals, UsageTotals)
    return MappingProxyType(
        {
            name: _usage_counter_presence(_counter_value(totals, name))
            for name in USAGE_COUNTER_NAMES
        }
    )


def usage_smoke_summary(
    *,
    task_variant: str,
    success: bool,
    schema_valid: bool,
    expected_output_match: bool,
    totals: UsageTotals,
    required_counters: Iterable[str] = (),
) -> Mapping[str, TaskUsageValue]:
    _assert_smoke_task_variant(task_variant)
    assert isinstance(success, bool), "success must be a boolean"
    assert isinstance(schema_valid, bool), "schema_valid must be a boolean"
    assert isinstance(
        expected_output_match, bool
    ), "expected_output_match must be a boolean"
    assert isinstance(totals, UsageTotals)
    required_counter_names = _usage_required_counter_names(required_counters)
    presence = usage_counter_presence(totals)
    return MappingProxyType(
        {
            "task_variant": task_variant,
            "success": success,
            "schema_valid": schema_valid,
            "expected_output_match": expected_output_match,
            "required_usage_present": all(
                presence[name] != UsageCounterPresence.MISSING
                for name in required_counter_names
            ),
            "usage_field_presence": MappingProxyType(
                {name: presence[name].value for name in USAGE_COUNTER_NAMES}
            ),
            "usage_totals": _usage_totals_snapshot(totals),
        }
    )


def usage_observations_from_response(
    response: object,
) -> tuple[UsageObservation, ...]:
    return tuple(
        entry.observation
        for entry in usage_observation_entries_from_response(response)
    )


def usage_observation_entries_from_response(
    response: object,
) -> tuple[UsageObservationEntry, ...]:
    child_entries = _child_usage_observation_entries(response)
    if child_entries:
        return child_entries
    usage = _usage_container(response)
    if isinstance(usage, list | tuple):
        entries = _usage_observation_entries_from_usage_items(
            usage,
            response=response,
        )
        if entries:
            return entries

    observation = usage_observation_from_response(response)
    if observation is None:
        return ()
    return (
        UsageObservationEntry(
            response=response,
            sequence=1,
            observation=observation,
        ),
    )


def usage_totals_from_response(response: object) -> UsageTotals | None:
    observation = usage_observation_from_response(response)
    return observation.totals if observation is not None else None


def _usage_counter_presence(value: int | None) -> UsageCounterPresence:
    if value is None:
        return UsageCounterPresence.MISSING
    if value == 0:
        return UsageCounterPresence.REPORTED_ZERO
    return UsageCounterPresence.REPORTED_POSITIVE


def _usage_totals_snapshot(totals: UsageTotals) -> TaskUsageMetadata:
    return MappingProxyType(
        {name: _counter_value(totals, name) for name in USAGE_COUNTER_NAMES}
    )


def _assert_smoke_task_variant(value: str) -> None:
    _assert_non_empty_string(value, "task_variant")
    assert fullmatch(
        r"[a-z][a-z0-9_.-]{0,63}", value
    ), "task_variant must be a safe label"


def _usage_required_counter_names(
    values: Iterable[str],
) -> tuple[str, ...]:
    names: list[str] = []
    for value in values:
        assert isinstance(value, str), "required counters must be strings"
        assert value in USAGE_COUNTER_NAMES, "unknown required usage counter"
        if value not in names:
            names.append(value)
    return tuple(names)


def attach_response_usage_recorder(
    response: object,
    *,
    store: TaskUsageStore,
    run_id: str,
    attempt_id: str | None = None,
    usage_id: str | None = None,
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
        entries = usage_observation_entries_from_response(response)
        for entry in entries:
            await store.append_usage(
                run_id,
                attempt_id=attempt_id,
                usage_id=_usage_record_id(
                    entry.response,
                    run_id=run_id,
                    attempt_id=attempt_id,
                    sequence=entry.sequence,
                    usage_id=usage_id,
                    observation_count=len(entries),
                ),
                source=source or entry.observation.source,
                totals=entry.observation.totals,
                metadata=(
                    metadata
                    if metadata is not None
                    else entry.observation.metadata
                ),
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


def stable_usage_id(
    *,
    run_id: str,
    attempt_id: str | None,
    call_key: str,
) -> str:
    _assert_non_empty_string(run_id, "run_id")
    if attempt_id is not None:
        _assert_non_empty_string(attempt_id, "attempt_id")
    _assert_non_empty_string(call_key, "call_key")
    digest = sha256()
    for value in (run_id, attempt_id or "", call_key):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return f"usage-{digest.hexdigest()}"


def stable_usage_id_for_response(
    response: object,
    *,
    run_id: str,
    attempt_id: str | None,
    sequence: int,
) -> str:
    assert isinstance(sequence, int), "sequence must be an integer"
    assert not isinstance(sequence, bool), "sequence must be an integer"
    assert sequence > 0, "sequence must be positive"
    return stable_usage_id(
        run_id=run_id,
        attempt_id=attempt_id,
        call_key=f"{_usage_call_key(response)}:{sequence}",
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


def _usage_record_id(
    response: object,
    *,
    run_id: str,
    attempt_id: str | None,
    sequence: int,
    usage_id: str | None,
    observation_count: int,
) -> str:
    if usage_id is None:
        return stable_usage_id_for_response(
            response,
            run_id=run_id,
            attempt_id=attempt_id,
            sequence=sequence,
        )
    if observation_count == 1:
        return usage_id
    return stable_usage_id(
        run_id=run_id,
        attempt_id=attempt_id,
        call_key=f"{usage_id}:{sequence}",
    )


def _usage_call_key(response: object) -> str:
    explicit = _explicit_usage_call_key(response)
    if explicit is not None:
        return _hashed_call_key("explicit", explicit)
    return _local_usage_call_key(response)


def _explicit_usage_call_key(response: object) -> str | None:
    for attribute in (
        "usage_call_key",
        "usage_identity",
        "usage_key",
        "usage_id",
    ):
        value = getattr(response, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _local_usage_call_key(response: object) -> str:
    value = getattr(response, _LOCAL_USAGE_CALL_KEY_ATTRIBUTE, None)
    if isinstance(value, str) and value.strip():
        return value

    key = _weak_usage_call_key(response)
    if key is not None:
        return key

    key = f"local:{next(_LOCAL_USAGE_CALL_KEY_COUNTER)}"
    try:
        setattr(response, _LOCAL_USAGE_CALL_KEY_ATTRIBUTE, key)
    except (AttributeError, TypeError):
        return _local_usage_call_key_for_untracked_object(response)
    return key


def _weak_usage_call_key(response: object) -> str | None:
    try:
        return _LOCAL_USAGE_CALL_KEYS.get(response)
    except TypeError:
        return None


def _local_usage_call_key_for_untracked_object(response: object) -> str:
    response_id = id(response)
    bucket = _LOCAL_UNTRACKED_USAGE_CALL_KEYS.setdefault(response_id, [])
    for tracked_response, key in bucket:
        if tracked_response is response:
            return key
    key = f"local:untracked:{next(_LOCAL_USAGE_CALL_KEY_COUNTER)}"
    try:
        _LOCAL_USAGE_CALL_KEYS[response] = key
    except TypeError:
        bucket.append((response, key))
        return key
    return key


def _hashed_call_key(prefix: str, value: str) -> str:
    digest = sha256(value.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


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


def _provider_family_value(value: object) -> UsageProviderFamily | None:
    if isinstance(value, UsageProviderFamily):
        return value
    if isinstance(value, str):
        try:
            return UsageProviderFamily(value)
        except ValueError:
            return None
    return None


def _child_usage_observations(
    response: object,
    *,
    seen: set[int] | None = None,
) -> tuple[UsageObservation, ...]:
    return tuple(
        entry.observation
        for entry in _child_usage_observation_entries(response, seen=seen)
    )


def _child_usage_observation_entries(
    response: object,
    *,
    seen: set[int] | None = None,
) -> tuple[UsageObservationEntry, ...]:
    if seen is None:
        seen = set()
    response_id = id(response)
    if response_id in seen:
        return ()
    seen.add(response_id)
    usage_responses = getattr(response, "usage_responses", None)
    if usage_responses is None:
        return ()
    if callable(usage_responses):
        usage_responses = usage_responses()
    if not isinstance(usage_responses, list | tuple):
        return ()

    entries: list[UsageObservationEntry] = []
    for usage_response in usage_responses:
        if usage_response is response:
            continue
        usage_response_id = id(usage_response)
        if usage_response_id in seen:
            continue
        usage = _usage_container(usage_response)
        if usage is not None:
            seen.add(usage_response_id)
            if isinstance(usage, list | tuple):
                entries.extend(
                    _usage_observation_entries_from_usage_items(
                        usage,
                        response=usage_response,
                    )
                )
            else:
                totals = _usage_totals_from_value(
                    usage,
                    _PROVIDER_COUNTER_PATHS,
                )
                if totals is not None:
                    entries.append(
                        UsageObservationEntry(
                            response=usage_response,
                            sequence=1,
                            observation=UsageObservation(
                                source=UsageSource.EXACT,
                                totals=totals,
                                metadata=_usage_metadata_from_response(
                                    usage_response,
                                    usage,
                                ),
                            ),
                        )
                    )
            continue

        child_entries = _child_usage_observation_entries(
            usage_response,
            seen=seen,
        )
        if child_entries:
            entries.extend(child_entries)
            continue

        totals = _usage_totals_from_value(
            usage_response,
            _RESPONSE_COUNTER_PATHS,
        )
        if totals is not None:
            entries.append(
                UsageObservationEntry(
                    response=usage_response,
                    sequence=1,
                    observation=UsageObservation(
                        source=UsageSource.ESTIMATED,
                        totals=totals,
                        metadata=_usage_metadata_from_response(
                            usage_response,
                            None,
                        ),
                    ),
                )
            )
    return tuple(entries)


def _usage_observations_from_usage_items(
    usage_items: list[object] | tuple[object, ...],
    *,
    response: object,
) -> tuple[UsageObservation, ...]:
    return tuple(
        entry.observation
        for entry in _usage_observation_entries_from_usage_items(
            usage_items,
            response=response,
        )
    )


def _usage_observation_entries_from_usage_items(
    usage_items: list[object] | tuple[object, ...],
    *,
    response: object,
) -> tuple[UsageObservationEntry, ...]:
    observations: list[UsageObservation] = []
    for usage in usage_items:
        totals = _usage_totals_from_value(usage, _PROVIDER_COUNTER_PATHS)
        if totals is None:
            continue
        observations.append(
            UsageObservation(
                source=UsageSource.EXACT,
                totals=totals,
                metadata=_usage_metadata_from_response(response, usage),
            )
        )
    return tuple(
        UsageObservationEntry(
            response=response,
            sequence=sequence,
            observation=observation,
        )
        for sequence, observation in enumerate(observations, start=1)
    )


def _aggregate_usage_observations(
    observations: tuple[UsageObservation, ...],
) -> UsageObservation:
    assert observations, "observations must not be empty"
    metadata = _shared_usage_metadata(observations)
    return UsageObservation(
        source=_aggregate_usage_source(observations),
        totals=UsageTotals(
            input_tokens=_sum_observation_counter(
                observations,
                "input_tokens",
            ),
            cached_input_tokens=_sum_observation_counter(
                observations,
                "cached_input_tokens",
            ),
            cache_creation_input_tokens=_sum_observation_counter(
                observations,
                "cache_creation_input_tokens",
            ),
            output_tokens=_sum_observation_counter(
                observations,
                "output_tokens",
            ),
            reasoning_tokens=_sum_observation_counter(
                observations,
                "reasoning_tokens",
            ),
            total_tokens=_sum_observation_counter(
                observations,
                "total_tokens",
            ),
        ),
        metadata=metadata,
    )


def _aggregate_usage_source(
    observations: tuple[UsageObservation, ...],
) -> UsageSource:
    sources = {observation.source for observation in observations}
    if len(sources) == 1:
        return observations[0].source
    return UsageSource.ESTIMATED


def _shared_usage_metadata(
    observations: tuple[UsageObservation, ...],
) -> TaskUsageMetadata:
    first = observations[0].metadata
    if all(observation.metadata == first for observation in observations):
        return first
    return _empty_metadata()


def _sum_observation_counter(
    observations: tuple[UsageObservation, ...],
    field_name: str,
) -> int | None:
    total = 0
    observed = False
    for observation in observations:
        value = _counter_value(observation.totals, field_name)
        if value is None:
            continue
        observed = True
        total += value
    if not observed:
        return None
    return total


CounterPath = tuple[str, ...]
CounterPathMap = Mapping[str, tuple[CounterPath, ...]]

_PROVIDER_COUNTER_PATHS: CounterPathMap = {
    "input_tokens": (
        ("input_tokens",),
        ("input_token_count",),
        ("inputTokens",),
        ("promptTokenCount",),
        ("prompt_tokens",),
        ("prompt_token_count",),
    ),
    "cached_input_tokens": (
        ("cached_input_tokens",),
        ("cached_input_token_count",),
        ("cache_read_input_tokens",),
        ("cacheReadInputTokens",),
        ("cached_content_token_count",),
        ("cachedContentTokenCount",),
        ("input_tokens_details", "cached_tokens"),
        ("prompt_tokens_details", "cached_tokens"),
    ),
    "cache_creation_input_tokens": (
        ("cache_creation_input_tokens",),
        ("cache_creation_input_token_count",),
        ("cacheCreationInputTokens",),
        ("cache_write_input_tokens",),
        ("cacheWriteInputTokens",),
    ),
    "output_tokens": (
        ("output_tokens",),
        ("output_token_count",),
        ("outputTokens",),
        ("candidatesTokenCount",),
        ("completion_tokens",),
        ("candidates_token_count",),
    ),
    "reasoning_tokens": (
        ("reasoning_tokens",),
        ("reasoning_token_count",),
        ("reasoningTokens",),
        ("thoughtsTokenCount",),
        ("thoughts_token_count",),
        ("output_tokens_details", "thinking_tokens"),
        ("output_tokens_details", "reasoning_tokens"),
        ("completion_tokens_details", "reasoning_tokens"),
    ),
    "total_tokens": (
        ("total_tokens",),
        ("total_token_count",),
        ("totalTokens",),
        ("totalTokenCount",),
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

_USAGE_METADATA_COUNTER_PATHS: CounterPathMap = {
    "cache_creation_ephemeral_5m_input_tokens": (
        ("cache_creation_ephemeral_5m_input_tokens",),
        ("cacheCreationEphemeral5mInputTokens",),
        ("cache_creation", "ephemeral_5m_input_tokens"),
        ("cacheCreation", "ephemeral5mInputTokens"),
        ("cacheCreation", "ephemeral_5m_input_tokens"),
        ("cache_details", "cache_creation", "ephemeral_5m_input_tokens"),
        ("cache_details", "cache_write", "ephemeral_5m_input_tokens"),
        ("cacheDetails", "cacheCreation", "ephemeral5mInputTokens"),
        ("cacheDetails", "cacheWrite", "ephemeral5mInputTokens"),
    ),
    "cache_creation_ephemeral_1h_input_tokens": (
        ("cache_creation_ephemeral_1h_input_tokens",),
        ("cacheCreationEphemeral1hInputTokens",),
        ("cache_creation", "ephemeral_1h_input_tokens"),
        ("cacheCreation", "ephemeral1hInputTokens"),
        ("cacheCreation", "ephemeral_1h_input_tokens"),
        ("cache_details", "cache_creation", "ephemeral_1h_input_tokens"),
        ("cache_details", "cache_write", "ephemeral_1h_input_tokens"),
        ("cacheDetails", "cacheCreation", "ephemeral1hInputTokens"),
        ("cacheDetails", "cacheWrite", "ephemeral1hInputTokens"),
    ),
    "cache_read_ephemeral_5m_input_tokens": (
        ("cache_read_ephemeral_5m_input_tokens",),
        ("cacheReadEphemeral5mInputTokens",),
        ("cache_read", "ephemeral_5m_input_tokens"),
        ("cacheRead", "ephemeral5mInputTokens"),
        ("cacheRead", "ephemeral_5m_input_tokens"),
        ("cache_details", "cache_read", "ephemeral_5m_input_tokens"),
        ("cacheDetails", "cacheRead", "ephemeral5mInputTokens"),
    ),
    "cache_read_ephemeral_1h_input_tokens": (
        ("cache_read_ephemeral_1h_input_tokens",),
        ("cacheReadEphemeral1hInputTokens",),
        ("cache_read", "ephemeral_1h_input_tokens"),
        ("cacheRead", "ephemeral1hInputTokens"),
        ("cacheRead", "ephemeral_1h_input_tokens"),
        ("cache_details", "cache_read", "ephemeral_1h_input_tokens"),
        ("cacheDetails", "cacheRead", "ephemeral1hInputTokens"),
    ),
}


def _usage_container(response: object) -> object | None:
    for attribute in ("usage", "usage_metadata", "usageMetadata"):
        value = _value_at_path(response, (attribute,))
        if value is not None:
            return value
    return None


def _usage_metadata_from_response(
    response: object,
    usage: object | None,
) -> TaskUsageMetadata:
    metadata: dict[str, object] = {}
    for value in (response, usage):
        if value is None:
            continue
        provider_family = _provider_family_value(
            _value_at_path(value, ("provider_family",))
        )
        if provider_family is not None and "provider_family" not in metadata:
            metadata["provider_family"] = provider_family
        for key, counter in _usage_metadata_counters_from_value(value).items():
            metadata.setdefault(key, counter)
    return freeze_usage_metadata(metadata)


def _usage_metadata_counters_from_value(value: object) -> dict[str, int]:
    return {
        name: counter
        for name, paths in _USAGE_METADATA_COUNTER_PATHS.items()
        if (counter := _counter_from_paths(value, paths)) is not None
    }


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
