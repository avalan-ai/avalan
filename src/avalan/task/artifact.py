from ..types import (
    assert_non_empty_string as _assert_non_empty_string,
)
from ..types import (
    assert_optional_non_negative_int as _assert_non_negative_int,
)
from ..types import (
    assert_optional_positive_int as _assert_positive_int,
)
from .store import (
    TaskSnapshotMetadata,
    TaskSnapshotValue,
    empty_snapshot_metadata,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)

from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import BinaryIO, Protocol


class TaskArtifactPurpose(StrEnum):
    INPUT = "input"
    OUTPUT = "output"
    CONVERTED = "converted"
    INTERMEDIATE = "intermediate"


class TaskArtifactState(StrEnum):
    READY = "ready"
    DELETED = "deleted"
    LOST = "lost"


TASK_ARTIFACT_TERMINAL_STATES = frozenset(
    {
        TaskArtifactState.DELETED,
        TaskArtifactState.LOST,
    }
)
VALID_TASK_ARTIFACT_TRANSITIONS = {
    TaskArtifactState.READY: frozenset(
        {
            TaskArtifactState.DELETED,
            TaskArtifactState.LOST,
        }
    ),
    TaskArtifactState.DELETED: frozenset(),
    TaskArtifactState.LOST: frozenset(),
}


class ArtifactStoreError(RuntimeError):
    pass


class ArtifactStoreConflictError(ArtifactStoreError):
    pass


class ArtifactStoreNotFoundError(ArtifactStoreError):
    pass


class ArtifactStorePolicyError(ArtifactStoreError):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactRetention:
    expires_at: datetime | None = None
    delete_after_days: int | None = None
    retain_metadata: bool = True
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        if self.expires_at is not None:
            _assert_datetime(self.expires_at, "expires_at")
        _assert_positive_int(self.delete_after_days, "delete_after_days")
        assert isinstance(self.retain_metadata, bool)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def summary(self) -> TaskSnapshotValue:
        value: dict[str, object] = {
            "retain_metadata": self.retain_metadata,
        }
        if self.expires_at is not None:
            value["expires_at"] = self.expires_at.isoformat()
        if self.delete_after_days is not None:
            value["delete_after_days"] = self.delete_after_days
        if self.metadata:
            value["metadata"] = self.metadata
        return freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactProvenance:
    source_artifact_id: str | None = None
    source_run_id: str | None = None
    source_attempt_id: str | None = None
    operation: str | None = None
    converter: str | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        for field_name in (
            "source_artifact_id",
            "source_run_id",
            "source_attempt_id",
            "operation",
            "converter",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _assert_non_empty_string(value, field_name)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def summary(self) -> TaskSnapshotValue:
        value: dict[str, object] = {}
        for field_name in (
            "source_artifact_id",
            "source_run_id",
            "source_attempt_id",
            "operation",
            "converter",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None:
                value[field_name] = field_value
        if self.metadata:
            value["metadata"] = self.metadata
        return freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactRef:
    artifact_id: str
    store: str
    storage_key: str
    media_type: str | None = None
    size_bytes: int | None = None
    sha256: str | None = None
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.artifact_id, "artifact_id")
        _assert_non_empty_string(self.store, "store")
        _assert_non_empty_string(self.storage_key, "storage_key")
        if self.media_type is not None:
            _assert_non_empty_string(self.media_type, "media_type")
        _assert_non_negative_int(self.size_bytes, "size_bytes")
        if self.sha256 is not None:
            _assert_sha256(self.sha256)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def summary(self, *, include_metadata: bool = True) -> TaskSnapshotValue:
        value: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "store": self.store,
        }
        if self.media_type is not None:
            value["media_type"] = self.media_type
        if self.size_bytes is not None:
            value["size_bytes"] = self.size_bytes
        if self.sha256 is not None:
            value["sha256"] = self.sha256
        if include_metadata and self.metadata:
            value["metadata"] = self.metadata
        return freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactStat:
    ref: TaskArtifactRef
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        assert isinstance(self.ref, TaskArtifactRef)
        _assert_non_negative_int(self.size_bytes, "size_bytes")
        assert self.size_bytes is not None
        _assert_sha256(self.sha256)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskArtifactRecord:
    artifact_id: str
    run_id: str
    purpose: TaskArtifactPurpose
    state: TaskArtifactState
    ref: TaskArtifactRef
    created_at: datetime
    updated_at: datetime
    attempt_id: str | None = None
    provenance: TaskArtifactProvenance = field(
        default_factory=TaskArtifactProvenance
    )
    retention: TaskArtifactRetention = field(
        default_factory=TaskArtifactRetention
    )
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        _assert_non_empty_string(self.artifact_id, "artifact_id")
        _assert_non_empty_string(self.run_id, "run_id")
        assert isinstance(self.purpose, TaskArtifactPurpose)
        assert isinstance(self.state, TaskArtifactState)
        assert isinstance(self.ref, TaskArtifactRef)
        assert self.ref.artifact_id == self.artifact_id
        _assert_datetime(self.created_at, "created_at")
        _assert_datetime(self.updated_at, "updated_at")
        assert self.updated_at >= self.created_at
        if self.attempt_id is not None:
            _assert_non_empty_string(self.attempt_id, "attempt_id")
        assert isinstance(self.provenance, TaskArtifactProvenance)
        assert isinstance(self.retention, TaskArtifactRetention)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def summary(self) -> TaskSnapshotValue:
        value: dict[str, object] = {
            "artifact_id": self.artifact_id,
            "purpose": self.purpose.value,
            "state": self.state.value,
            "ref": self.ref.summary(include_metadata=False),
        }
        if self.attempt_id is not None:
            value["attempt_id"] = self.attempt_id
        provenance = self.provenance.summary()
        if provenance:
            value["provenance"] = provenance
        retention = self.retention.summary()
        if retention:
            value["retention"] = retention
        if self.metadata:
            value["metadata"] = self.metadata
        return freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskOutputArtifact:
    ref: TaskArtifactRef
    state: TaskArtifactState = TaskArtifactState.READY
    provenance: TaskArtifactProvenance = field(
        default_factory=lambda: TaskArtifactProvenance(operation="output")
    )
    retention: TaskArtifactRetention = field(
        default_factory=TaskArtifactRetention
    )
    metadata: TaskSnapshotMetadata = field(
        default_factory=empty_snapshot_metadata
    )

    def __post_init__(self) -> None:
        assert isinstance(self.ref, TaskArtifactRef)
        assert isinstance(self.state, TaskArtifactState)
        assert isinstance(self.provenance, TaskArtifactProvenance)
        assert isinstance(self.retention, TaskArtifactRetention)
        object.__setattr__(
            self,
            "metadata",
            freeze_snapshot_metadata(self.metadata),
        )

    def summary(self) -> TaskSnapshotValue:
        return freeze_snapshot_value(
            {
                "state": self.state.value,
                "ref": self.ref.summary(include_metadata=False),
            }
        )


class ArtifactStore(Protocol):
    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef: ...

    async def open(self, ref: TaskArtifactRef) -> BinaryIO: ...

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat: ...

    async def delete(self, ref: TaskArtifactRef) -> None: ...


def is_terminal_artifact_state(state: TaskArtifactState) -> bool:
    assert isinstance(state, TaskArtifactState)
    return state in TASK_ARTIFACT_TERMINAL_STATES


def is_valid_artifact_transition(
    from_state: TaskArtifactState,
    to_state: TaskArtifactState,
) -> bool:
    assert isinstance(from_state, TaskArtifactState)
    assert isinstance(to_state, TaskArtifactState)
    return to_state in VALID_TASK_ARTIFACT_TRANSITIONS[from_state]


def validate_artifact_transition(
    from_state: TaskArtifactState,
    to_state: TaskArtifactState,
) -> None:
    if not is_valid_artifact_transition(from_state, to_state):
        raise ArtifactStoreConflictError(
            "task artifact transition is not valid"
        )


def task_output_artifact_from_value(
    value: object,
) -> TaskOutputArtifact | None:
    if isinstance(value, TaskArtifactRef):
        return TaskOutputArtifact(ref=value)
    if isinstance(value, TaskArtifactRecord):
        return TaskOutputArtifact(
            ref=value.ref,
            state=value.state,
            provenance=value.provenance,
            retention=value.retention,
            metadata=value.metadata,
        )
    return None


def artifact_retention_expired(
    record: TaskArtifactRecord,
    expired_at: datetime,
) -> bool:
    assert isinstance(record, TaskArtifactRecord)
    _assert_datetime(expired_at, "expired_at")
    retention = record.retention
    if retention.expires_at is not None:
        return retention.expires_at <= expired_at
    if retention.delete_after_days is not None:
        deadline = record.created_at + timedelta(
            days=retention.delete_after_days
        )
        return deadline <= expired_at
    return False


def assert_artifact_state_collection(
    values: Collection[object],
    field_name: str,
) -> None:
    assert isinstance(values, Collection), f"{field_name} must be a collection"
    assert values, f"{field_name} must not be empty"
    for value in values:
        assert isinstance(
            value, TaskArtifactState
        ), f"{field_name} must contain TaskArtifactState values"


def _assert_datetime(value: datetime, field_name: str) -> None:
    assert isinstance(value, datetime), f"{field_name} must be a datetime"


def _assert_sha256(value: str) -> None:
    _assert_non_empty_string(value, "sha256")
    assert len(value) == 64, "sha256 must be a SHA-256 hex digest"
    assert all(
        character in "0123456789abcdef" for character in value
    ), "sha256 must be a lowercase SHA-256 hex digest"
