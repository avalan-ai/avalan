"""Define immutable trigger records and deterministic identities."""

from .definition import (
    AtPolicy,
    AtTrigger,
    CronTrigger,
    IntervalTrigger,
    RecurringPolicy,
    TriggerSpec,
    integer,
    timestamp,
    utc_text,
)
from .error import TriggerError, TriggerErrorCode
from .schedule import ScheduleProvenance

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from json import dumps
from re import fullmatch
from types import MappingProxyType
from uuid import UUID, uuid5

OCCURRENCE_NAMESPACE = UUID("f279760b-a940-5ea5-a11d-7c96cd4fd690")
MAX_REVISION = 9223372036854775807


def opaque(value: object, path: str) -> str:
    """Require a nonempty opaque identifier without reflecting its value."""
    if not isinstance(value, str) or not value.strip():
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, path)
    return value


def digest(value: object, path: str) -> str:
    """Validate a canonical SHA-256 digest."""
    if not isinstance(value, str) or not fullmatch(r"[0-9a-f]{64}", value):
        raise TriggerError(TriggerErrorCode.INVALID_CONFIG, path)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class OwnerScopeId:
    """Represent authority supplied by a trusted host, never configuration."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        opaque(self.value, "owner_scope_id")


def occurrence_id(
    owner: OwnerScopeId, trigger_id: str, revision: int, scheduled_at: datetime
) -> str:
    """Derive one slot identity independently of optional task HMAC keys."""
    assert isinstance(owner, OwnerScopeId)
    opaque(trigger_id, "trigger_id")
    integer(revision, 1, MAX_REVISION, "revision")
    name = dumps(
        [owner.value, trigger_id, revision, utc_text(scheduled_at)],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return str(uuid5(OCCURRENCE_NAMESPACE, name))


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerSealedInput:
    """Store an opaque host-encrypted revision input, including bindings."""

    ciphertext: bytes = field(repr=False)
    key_id: str = field(repr=False)
    algorithm: str
    metadata: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.ciphertext, bytes) or not self.ciphertext:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "input.ciphertext"
            )
        opaque(self.key_id, "input.key_id")
        opaque(self.algorithm, "input.algorithm")
        if not isinstance(self.metadata, Mapping) or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "input.metadata"
            )
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerDefinition:
    """Retain one immutable registered revision and its resolved schedule."""

    owner_scope_id: OwnerScopeId
    trigger_id: str
    revision: int
    name: str
    task_definition_id: str
    execution_deployment_id: str
    semantic_hash: str
    schedule: TriggerSpec
    input: TriggerSealedInput = field(repr=False)
    policy: AtPolicy
    registered_at: datetime
    schedule_provenance: ScheduleProvenance
    resolved_anchor: datetime | None = None
    schema_version: int = 1
    schedule_semantics_version: int = 1

    def __post_init__(self) -> None:
        assert isinstance(self.owner_scope_id, OwnerScopeId)
        for name in (
            "trigger_id",
            "task_definition_id",
            "execution_deployment_id",
        ):
            opaque(getattr(self, name), name)
        integer(self.revision, 1, MAX_REVISION, "revision")
        integer(
            self.schema_version,
            1,
            1,
            "schema_version",
            TriggerErrorCode.UNSUPPORTED_VERSION,
        )
        integer(
            self.schedule_semantics_version,
            1,
            1,
            "schedule_semantics_version",
            TriggerErrorCode.UNSUPPORTED_VERSION,
        )
        if not isinstance(self.name, str) or not fullmatch(
            r"[a-z][a-z0-9-]{0,62}", self.name
        ):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "name")
        digest(self.semantic_hash, "semantic_hash")
        if not isinstance(
            self.schedule, CronTrigger | IntervalTrigger | AtTrigger
        ):
            raise TriggerError(TriggerErrorCode.INVALID_SCHEDULE, "schedule")
        expected_policy = (
            AtPolicy
            if isinstance(self.schedule, AtTrigger)
            else RecurringPolicy
        )
        if type(self.policy) is not expected_policy:
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "policy")
        assert isinstance(self.input, TriggerSealedInput)
        assert isinstance(self.schedule_provenance, ScheduleProvenance)
        if (
            type(self.schedule_provenance.semantics_version) is not int
            or self.schedule_provenance.semantics_version
            != self.schedule_semantics_version
        ):
            raise TriggerError(
                TriggerErrorCode.UNSUPPORTED_VERSION, "schedule_provenance"
            )
        opaque(
            self.schedule_provenance.parser_version,
            "schedule_provenance.parser_version",
        )
        opaque(
            self.schedule_provenance.timezone_data,
            "schedule_provenance.timezone_data",
        )
        object.__setattr__(
            self, "registered_at", timestamp(self.registered_at)
        )
        if isinstance(self.schedule, IntervalTrigger):
            if self.resolved_anchor is None:
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "resolved_anchor"
                )
            object.__setattr__(
                self, "resolved_anchor", timestamp(self.resolved_anchor)
            )
            if (
                self.schedule.start_at is not None
                and self.schedule.start_at != self.resolved_anchor
            ):
                raise TriggerError(
                    TriggerErrorCode.INVALID_SCHEDULE, "resolved_anchor"
                )
        elif self.resolved_anchor is not None:
            raise TriggerError(
                TriggerErrorCode.INVALID_SCHEDULE, "resolved_anchor"
            )


class TriggerStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    EXHAUSTED = "exhausted"
    ERROR = "error"


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerState:
    owner_scope_id: OwnerScopeId
    trigger_id: str
    revision: int
    generation: int
    status: TriggerStatus
    next_at: datetime | None
    last_processed_at: datetime
    retry_after: datetime | None = None
    failure_count: int = 0
    last_error_code: TriggerErrorCode | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.owner_scope_id, OwnerScopeId)
        opaque(self.trigger_id, "trigger_id")
        integer(self.revision, 1, MAX_REVISION, "revision")
        integer(self.generation, 1, MAX_REVISION, "generation")
        integer(self.failure_count, 0, 20, "failure_count")
        assert isinstance(self.status, TriggerStatus)
        object.__setattr__(
            self,
            "last_processed_at",
            timestamp(self.last_processed_at, "last_processed_at"),
        )
        for name in ("next_at", "retry_after"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, timestamp(value, name))
        if (self.status is TriggerStatus.EXHAUSTED) != (self.next_at is None):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "state.next_at"
            )
        if self.last_error_code is not None:
            assert isinstance(self.last_error_code, TriggerErrorCode)
        if self.status is TriggerStatus.ERROR and self.last_error_code is None:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "state.last_error_code"
            )
        if self.retry_after is not None and (
            self.failure_count == 0 or self.status is TriggerStatus.EXHAUSTED
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "state.retry_after"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerSnapshot:
    definition: TriggerDefinition
    state: TriggerState

    def __post_init__(self) -> None:
        assert isinstance(self.definition, TriggerDefinition)
        assert isinstance(self.state, TriggerState)
        if (
            self.definition.owner_scope_id != self.state.owner_scope_id
            or self.definition.trigger_id != self.state.trigger_id
            or self.definition.revision != self.state.revision
        ):
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "snapshot")


class OccurrenceDisposition(StrEnum):
    ADMITTED = "admitted"
    EXPIRED = "expired"
    SKIPPED_MISFIRE = "skipped_misfire"
    SKIPPED_OVERLAP = "skipped_overlap"
    COALESCED = "coalesced"
    SUPERSEDED = "superseded"


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerOccurrence:
    owner_scope_id: OwnerScopeId
    trigger_id: str
    revision: int
    occurrence_id: str
    scheduled_at: datetime
    decided_at: datetime
    disposition: OccurrenceDisposition
    run_id: str | None = None

    def __post_init__(self) -> None:
        expected = occurrence_id(
            self.owner_scope_id,
            self.trigger_id,
            self.revision,
            self.scheduled_at,
        )
        if self.occurrence_id != expected:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "occurrence_id"
            )
        assert isinstance(self.disposition, OccurrenceDisposition)
        object.__setattr__(self, "scheduled_at", timestamp(self.scheduled_at))
        object.__setattr__(self, "decided_at", timestamp(self.decided_at))
        if self.scheduled_at > self.decided_at:
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "occurrence.decided_at"
            )
        if (self.disposition is OccurrenceDisposition.ADMITTED) != (
            self.run_id is not None
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "occurrence.run_id"
            )
        if self.run_id is not None:
            opaque(self.run_id, "run_id")

    @property
    def dispatched_at(self) -> datetime | None:
        return (
            self.decided_at
            if self.disposition is OccurrenceDisposition.ADMITTED
            else None
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerCoverageSpan:
    owner_scope_id: OwnerScopeId
    trigger_id: str
    revision: int
    span_id: str
    first_at: datetime
    until_at: datetime
    decided_at: datetime
    disposition: OccurrenceDisposition
    exact_count: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.owner_scope_id, OwnerScopeId)
        opaque(self.trigger_id, "trigger_id")
        opaque(self.span_id, "span_id")
        integer(self.revision, 1, MAX_REVISION, "revision")
        for name in ("first_at", "until_at", "decided_at"):
            object.__setattr__(
                self, name, timestamp(getattr(self, name), name)
            )
        if self.first_at >= self.until_at or self.first_at > self.decided_at:
            raise TriggerError(TriggerErrorCode.INVALID_CONFIG, "span.range")
        if (
            not isinstance(self.disposition, OccurrenceDisposition)
            or self.disposition is OccurrenceDisposition.ADMITTED
        ):
            raise TriggerError(
                TriggerErrorCode.INVALID_CONFIG, "span.disposition"
            )
        if self.exact_count is not None:
            integer(self.exact_count, 1, MAX_REVISION, "span.exact_count")


class TriggerEventKind(StrEnum):
    CREATED = "created"
    REPLACED = "replaced"
    PAUSED = "paused"
    RESUMED = "resumed"
    DECIDED = "decided"
    FAILED = "failed"
    EXAMINED = "examined"


@dataclass(frozen=True, slots=True, kw_only=True)
class TriggerEvent:
    owner_scope_id: OwnerScopeId
    trigger_id: str
    revision: int
    generation: int
    event_id: str
    kind: TriggerEventKind
    recorded_at: datetime
    occurrence_ids: tuple[str, ...] = ()
    span_ids: tuple[str, ...] = ()
    error_code: TriggerErrorCode | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.owner_scope_id, OwnerScopeId)
        opaque(self.trigger_id, "trigger_id")
        opaque(self.event_id, "event_id")
        integer(self.revision, 1, MAX_REVISION, "revision")
        integer(self.generation, 1, MAX_REVISION, "generation")
        assert isinstance(self.kind, TriggerEventKind)
        object.__setattr__(self, "recorded_at", timestamp(self.recorded_at))
        for name in ("occurrence_ids", "span_ids"):
            values = getattr(self, name)
            assert isinstance(values, tuple)
            integer(len(values), 0, 1000, name)
            for value in values:
                opaque(value, name)
            if len(values) != len(set(values)):
                raise TriggerError(TriggerErrorCode.INVALID_CONFIG, name)
        if self.error_code is not None:
            assert isinstance(self.error_code, TriggerErrorCode)
