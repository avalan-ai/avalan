"""Track physical bytes separately from retained artifact references."""

from .artifact import (
    ArtifactStore,
    ArtifactStoreConflictError,
    ArtifactStoreNotFoundError,
    TaskArtifactRef,
)

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import UUID, uuid4, uuid5

_OBJECT_NAMESPACE = UUID("3efc9327-10ac-52dc-a94e-d922fcb202b9")


def physical_artifact_identity(ref: TaskArtifactRef) -> str:
    """Identify a physical backend key independently of optional metadata."""
    assert isinstance(ref, TaskArtifactRef)
    name = f"{len(ref.store)}:{ref.store}{ref.storage_key}"
    return str(uuid5(_OBJECT_NAMESPACE, name))


class ArtifactObjectState(StrEnum):
    LIVE = "live"
    DELETING = "deleting"
    DELETED = "deleted"


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactObject:
    """Identify immutable physical content without borrowing a run ID."""

    object_id: str
    ref: TaskArtifactRef = field(repr=False)
    state: ArtifactObjectState = ArtifactObjectState.LIVE
    cleanup_token: str | None = None
    was_staged: bool = field(default=False, compare=False)
    staged_at: datetime = field(
        default_factory=lambda: datetime.now(UTC), compare=False
    )

    def __post_init__(self) -> None:
        assert str(UUID(self.object_id)) == self.object_id
        assert isinstance(self.ref, TaskArtifactRef)
        assert self.object_id == physical_artifact_identity(self.ref)
        assert self.ref.size_bytes is not None and self.ref.sha256 is not None
        assert type(self.was_staged) is bool
        assert isinstance(self.state, ArtifactObjectState)
        assert (
            isinstance(self.staged_at, datetime)
            and self.staged_at.utcoffset() is not None
        )
        assert (self.state == ArtifactObjectState.LIVE) == (
            self.cleanup_token is None
        )
        if self.cleanup_token is not None:
            assert str(UUID(self.cleanup_token)) == self.cleanup_token

    @classmethod
    def from_ref(cls, ref: TaskArtifactRef) -> "ArtifactObject":
        """Derive physical identity from an unambiguous backend/key pair."""
        return cls(object_id=physical_artifact_identity(ref), ref=ref)

    def require_same_bytes(self, other: "ArtifactObject") -> None:
        if (
            self.object_id != other.object_id
            or self.ref.store != other.ref.store
            or self.ref.storage_key != other.ref.storage_key
            or self.ref.sha256 != other.ref.sha256
            or self.ref.size_bytes != other.ref.size_bytes
        ):
            raise ArtifactStoreConflictError("physical artifact changed")

    def new_reference(self) -> TaskArtifactRef:
        """Allocate a per-run record ID while retaining physical identity."""
        return replace(self.ref, artifact_id=str(uuid4()))


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRevisionOwner:
    """Represent a host-owned immutable resource revision."""

    owner_scope: str = field(repr=False)
    resource_id: str
    revision: int

    def __post_init__(self) -> None:
        assert self.owner_scope.strip() and self.resource_id.strip()
        assert type(self.revision) is int and self.revision > 0


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactStagingOwner:
    """Retain bytes until the stable recovery identity is settled."""

    staging_id: str
    owner_scope: str = field(repr=False)
    recovery_id: str

    def __post_init__(self) -> None:
        assert str(UUID(self.staging_id)) == self.staging_id
        assert self.owner_scope.strip() and self.recovery_id.strip()


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactDeletion:
    """Carry a committed deletion fence that rejects new acquisitions."""

    object: ArtifactObject

    def __post_init__(self) -> None:
        assert self.object.state == ArtifactObjectState.DELETING


class ArtifactDeletionStore(Protocol):
    async def finish_delete(self, deletion: ArtifactDeletion) -> None: ...


async def delete_owned_bytes(
    deletion: ArtifactDeletion,
    stores: Mapping[str, ArtifactStore],
    ownership: ArtifactDeletionStore,
) -> None:
    """Delete outside the database lock and retain the fence on failure.

    Retrying an uncertain backend acknowledgment with the same fence is
    safe: acquisition stays blocked until a new physical key is allocated.
    """
    store = stores.get(deletion.object.ref.store)
    if store is None:
        raise ArtifactStoreNotFoundError("artifact backend is not configured")
    try:
        await store.delete(deletion.object.ref)
    except ArtifactStoreNotFoundError:
        # A prior deletion may have succeeded before its acknowledgment.
        pass
    await ownership.finish_delete(deletion)
