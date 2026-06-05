from ..types import (
    JsonValue,
)
from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_positive_int as _assert_positive_int,
)
from .definition import TaskDefinition
from .event import SanitizedTaskEvent, TaskEventCategory, TaskEventValue
from .state import (
    TaskAttemptState,
    TaskRunState,
    is_terminal_attempt_state,
    is_terminal_run_state,
    is_valid_attempt_transition,
    is_valid_run_transition,
)
from .usage import UsageRecord, UsageSource, UsageTotals

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

if TYPE_CHECKING:
    from .artifact import (
        TaskArtifactProvenance,
        TaskArtifactPurpose,
        TaskArtifactRecord,
        TaskArtifactRef,
        TaskArtifactRetention,
        TaskArtifactState,
    )
    from .idempotency import (
        TaskIdempotencyIdentity,
        TaskIdempotencyReservation,
        TaskIdempotencyReservationResult,
    )

TaskSnapshotValue: TypeAlias = JsonValue
TaskSnapshotMetadata: TypeAlias = Mapping[str, TaskSnapshotValue]


class TaskStoreError(RuntimeError):
    pass


class TaskStoreConflictError(TaskStoreError):
    pass


class TaskStoreNotFoundError(TaskStoreError):
    pass


def empty_snapshot_metadata() -> TaskSnapshotMetadata:
    return MappingProxyType({})


def freeze_snapshot_metadata(
    value: Mapping[str, object] | None,
) -> TaskSnapshotMetadata:
    if value is None:
        return empty_snapshot_metadata()
    assert isinstance(value, Mapping), "metadata must be a mapping"
    return cast(TaskSnapshotMetadata, _freeze_snapshot_value(value))


def freeze_snapshot_value(value: object) -> TaskSnapshotValue:
    return _freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskDefinitionRecord:
    definition_id: str
    definition: TaskDefinition
    spec_hash: str
    created_at: datetime
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.definition_id, "definition_id")
        assert isinstance(self.definition, TaskDefinition)
        _assert_non_empty_string(self.spec_hash, "spec_hash")
        _assert_datetime(self.created_at, "created_at")
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutionPayload:
    input_value: TaskSnapshotValue
    file_values: tuple[TaskSnapshotValue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_value",
            freeze_snapshot_value(self.input_value),
        )
        assert isinstance(
            self.file_values,
            tuple,
        ), "file_values must be a tuple"
        object.__setattr__(
            self,
            "file_values",
            tuple(freeze_snapshot_value(value) for value in self.file_values),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutionRequest:
    definition_id: str
    input_summary: TaskSnapshotValue = None
    input_payload: TaskExecutionPayload | None = None
    file_summaries: tuple[TaskSnapshotValue, ...] = ()
    idempotency_key: str | None = None
    queue: str | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.definition_id, "definition_id")
        object.__setattr__(
            self,
            "input_summary",
            freeze_snapshot_value(self.input_summary),
        )
        if self.input_payload is not None:
            assert isinstance(self.input_payload, TaskExecutionPayload)
        assert isinstance(
            self.file_summaries, tuple
        ), "file_summaries must be a tuple"
        object.__setattr__(
            self,
            "file_summaries",
            tuple(
                freeze_snapshot_value(summary)
                for summary in self.file_summaries
            ),
        )
        if self.idempotency_key is not None:
            _assert_non_empty_string(
                self.idempotency_key,
                "idempotency_key",
            )
        if self.queue is not None:
            _assert_non_empty_string(self.queue, "queue")
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutionResult:
    output_summary: TaskSnapshotValue = None
    error: TaskSnapshotValue = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_summary",
            freeze_snapshot_value(self.output_summary),
        )
        object.__setattr__(
            self,
            "error",
            freeze_snapshot_value(self.error),
        )
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClaim:
    worker_id: str
    claim_token: str
    claimed_at: datetime
    lease_expires_at: datetime
    heartbeat_at: datetime | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.worker_id, "worker_id")
        _assert_non_empty_string(self.claim_token, "claim_token")
        _assert_datetime(self.claimed_at, "claimed_at")
        _assert_datetime(self.lease_expires_at, "lease_expires_at")
        assert (
            self.lease_expires_at > self.claimed_at
        ), "lease_expires_at must be after claimed_at"
        if self.heartbeat_at is not None:
            _assert_datetime(self.heartbeat_at, "heartbeat_at")
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutionContext:
    run_id: str
    attempt_id: str
    attempt_number: int
    claim: TaskClaim | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_positive_int(self.attempt_number, "attempt_number")
        if self.claim is not None:
            assert isinstance(self.claim, TaskClaim)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRun:
    run_id: str
    definition_id: str
    state: TaskRunState
    request: TaskExecutionRequest
    created_at: datetime
    updated_at: datetime
    claim: TaskClaim | None = None
    last_attempt_id: str | None = None
    result: TaskExecutionResult | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.run_id, "run_id")
        _assert_non_empty_string(self.definition_id, "definition_id")
        assert isinstance(self.state, TaskRunState)
        assert isinstance(self.request, TaskExecutionRequest)
        _assert_datetime(self.created_at, "created_at")
        _assert_datetime(self.updated_at, "updated_at")
        assert self.updated_at >= self.created_at
        if self.claim is not None:
            assert isinstance(self.claim, TaskClaim)
        if self.last_attempt_id is not None:
            _assert_non_empty_string(self.last_attempt_id, "last_attempt_id")
        if self.result is not None:
            assert isinstance(self.result, TaskExecutionResult)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskAttempt:
    attempt_id: str
    run_id: str
    attempt_number: int
    state: TaskAttemptState
    context: TaskExecutionContext
    created_at: datetime
    updated_at: datetime
    result: TaskExecutionResult | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_non_empty_string(self.run_id, "run_id")
        _assert_positive_int(self.attempt_number, "attempt_number")
        assert isinstance(self.state, TaskAttemptState)
        assert isinstance(self.context, TaskExecutionContext)
        assert self.context.run_id == self.run_id
        assert self.context.attempt_id == self.attempt_id
        assert self.context.attempt_number == self.attempt_number
        _assert_datetime(self.created_at, "created_at")
        _assert_datetime(self.updated_at, "updated_at")
        assert self.updated_at >= self.created_at
        if self.result is not None:
            assert isinstance(self.result, TaskExecutionResult)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskTransition:
    transition_id: str
    run_id: str
    from_state: TaskRunState
    to_state: TaskRunState
    reason: str
    created_at: datetime
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.transition_id, "transition_id")
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.from_state, TaskRunState)
        assert isinstance(self.to_state, TaskRunState)
        _assert_non_empty_string(self.reason, "reason")
        _assert_datetime(self.created_at, "created_at")
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskAttemptTransition:
    transition_id: str
    attempt_id: str
    run_id: str
    from_state: TaskAttemptState
    to_state: TaskAttemptState
    reason: str
    created_at: datetime
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.transition_id, "transition_id")
        _assert_non_empty_string(self.attempt_id, "attempt_id")
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.from_state, TaskAttemptState)
        assert isinstance(self.to_state, TaskAttemptState)
        _assert_non_empty_string(self.reason, "reason")
        _assert_datetime(self.created_at, "created_at")
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )


class TaskStore(Protocol):
    async def register_definition(
        self,
        definition: TaskDefinition,
        *,
        definition_hash: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskDefinitionRecord: ...

    async def get_definition(
        self,
        definition_id: str,
    ) -> TaskDefinitionRecord: ...

    async def create_run(
        self,
        request: TaskExecutionRequest,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun: ...

    async def get_run(self, run_id: str) -> TaskRun: ...

    async def transition_run(
        self,
        run_id: str,
        *,
        from_states: Collection[TaskRunState],
        to_state: TaskRunState,
        reason: str,
        result: TaskExecutionResult | None = None,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun: ...

    async def assign_claim(
        self,
        run_id: str,
        *,
        from_states: Collection[TaskRunState],
        worker_id: str,
        lease_expires_at: datetime,
        reason: str,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun: ...

    async def create_attempt(
        self,
        run_id: str,
        *,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttempt: ...

    async def get_attempt(self, attempt_id: str) -> TaskAttempt: ...

    async def list_attempts(self, run_id: str) -> tuple[TaskAttempt, ...]: ...

    async def transition_attempt(
        self,
        attempt_id: str,
        *,
        from_states: Collection[TaskAttemptState],
        to_state: TaskAttemptState,
        reason: str,
        result: TaskExecutionResult | None = None,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttempt: ...

    async def list_run_transitions(
        self,
        run_id: str,
    ) -> tuple[TaskTransition, ...]: ...

    async def list_attempt_transitions(
        self,
        attempt_id: str,
    ) -> tuple[TaskAttemptTransition, ...]: ...

    async def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        category: TaskEventCategory,
        payload: TaskEventValue,
        attempt_id: str | None = None,
    ) -> SanitizedTaskEvent: ...

    async def list_events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]: ...

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
    ) -> tuple[UsageRecord, ...]: ...

    async def usage_totals(self, run_id: str) -> UsageTotals: ...

    async def append_artifact(
        self,
        run_id: str,
        *,
        ref: "TaskArtifactRef",
        purpose: "TaskArtifactPurpose",
        state: "TaskArtifactState | None" = None,
        attempt_id: str | None = None,
        provenance: "TaskArtifactProvenance | None" = None,
        retention: "TaskArtifactRetention | None" = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskArtifactRecord": ...

    async def get_artifact(
        self,
        artifact_id: str,
    ) -> "TaskArtifactRecord": ...

    async def list_artifacts(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        purpose: "TaskArtifactPurpose | None" = None,
        state: "TaskArtifactState | None" = None,
    ) -> tuple["TaskArtifactRecord", ...]: ...

    async def list_retention_artifacts(
        self,
        *,
        expired_at: datetime,
        purpose: "TaskArtifactPurpose | None" = None,
        limit: int = 100,
    ) -> tuple["TaskArtifactRecord", ...]: ...

    async def transition_artifact(
        self,
        artifact_id: str,
        *,
        from_states: Collection["TaskArtifactState"],
        to_state: "TaskArtifactState",
        reason: str,
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskArtifactRecord": ...

    async def reserve_idempotency_key(
        self,
        identity: "TaskIdempotencyIdentity",
        *,
        run_id: str,
        expires_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> "TaskIdempotencyReservationResult": ...

    async def lookup_idempotency_key(
        self,
        identity: "TaskIdempotencyIdentity",
    ) -> "TaskIdempotencyReservation | None": ...


def validate_run_transition_request(
    *,
    current_state: TaskRunState,
    from_states: Collection[TaskRunState],
    to_state: TaskRunState,
) -> None:
    assert isinstance(current_state, TaskRunState)
    _assert_state_collection(from_states, TaskRunState, "from_states")
    assert isinstance(to_state, TaskRunState)
    if current_state not in from_states:
        raise TaskStoreConflictError("task run state did not match")
    if not is_valid_run_transition(current_state, to_state):
        raise TaskStoreConflictError("task run transition is not valid")


def allows_cancel_request_without_claim_token(
    run: TaskRun,
    to_state: TaskRunState,
    claim_token: str | None,
) -> bool:
    return (
        run.claim is not None
        and to_state == TaskRunState.CANCEL_REQUESTED
        and claim_token is None
    )


def validate_attempt_transition_request(
    *,
    current_state: TaskAttemptState,
    from_states: Collection[TaskAttemptState],
    to_state: TaskAttemptState,
) -> None:
    assert isinstance(current_state, TaskAttemptState)
    _assert_state_collection(from_states, TaskAttemptState, "from_states")
    assert isinstance(to_state, TaskAttemptState)
    if current_state not in from_states:
        raise TaskStoreConflictError("task attempt state did not match")
    if not is_valid_attempt_transition(current_state, to_state):
        raise TaskStoreConflictError("task attempt transition is not valid")


def ensure_run_is_mutable(state: TaskRunState) -> None:
    if is_terminal_run_state(state):
        raise TaskStoreConflictError("terminal task run cannot be changed")


def ensure_attempt_is_mutable(state: TaskAttemptState) -> None:
    if is_terminal_attempt_state(state):
        raise TaskStoreConflictError("terminal task attempt cannot be changed")


def _freeze_snapshot_value(value: object) -> TaskSnapshotValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        assert isfinite(value), "snapshot floats must be finite"
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, TaskSnapshotValue] = {}
        for key, item in value.items():
            assert isinstance(key, str), "snapshot keys must be strings"
            assert key.strip(), "snapshot keys must not be empty"
            frozen[key] = _freeze_snapshot_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze_snapshot_value(item) for item in value)
    raise AssertionError("snapshot value must be privacy-safe")


def _assert_datetime(value: datetime, field_name: str) -> None:
    assert isinstance(value, datetime), f"{field_name} must be a datetime"


def _assert_state_collection(
    values: Collection[object],
    state_type: type[TaskAttemptState] | type[TaskRunState],
    field_name: str,
) -> None:
    assert isinstance(values, Collection), f"{field_name} must be a collection"
    assert values, f"{field_name} must not be empty"
    for value in values:
        assert isinstance(
            value, state_type
        ), f"{field_name} must contain {state_type.__name__} values"
