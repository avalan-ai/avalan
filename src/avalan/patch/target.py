"""Inspect a trusted local workspace through an incapable patch target.

This module deliberately exposes only bounded, no-follow inspection.  It does
not register a tool and its commit entry point always returns a typed incapable
outcome.  The synchronous POSIX syscalls are confined to ``_inspect_many``;
the Avalan-facing boundary is asynchronous and has no write primitive.
"""

from asyncio import CancelledError, create_subprocess_exec, sleep
from base64 import b64decode, b64encode
from ctypes import (
    CDLL,
    POINTER,
    Structure,
    byref,
    c_char,
    c_int,
    c_int32,
    c_uint32,
    c_uint64,
)
from dataclasses import dataclass, field, replace
from enum import Enum
from hashlib import sha256
from hmac import compare_digest, digest
from importlib.util import find_spec
from json import dumps, loads
from os import (
    O_CLOEXEC,
    O_DIRECTORY,
    O_NOFOLLOW,
    O_NONBLOCK,
    O_RDONLY,
    close,
    environ,
    fstat,
    open,
)
from os import read as read_fd
from os import stat as stat_at
from pathlib import Path
from secrets import token_bytes
from stat import S_ISDIR, S_ISLNK, S_ISREG
from subprocess import PIPE
from sys import executable, platform, stdin, stdout
from typing import Never, NewType, Protocol, TypedDict, final
from unicodedata import normalize

from cryptography import __file__ as cryptography_file
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)

from avalan.patch.domain import (
    ByteSize,
    Capability,
    ContextKind,
    FileMode,
    LogicalPath,
    MetadataProfile,
    PatchApprovalId,
    PatchContextId,
    PatchDomainId,
    PatchLimits,
    PatchProtocolId,
    PatchTargetId,
    PatchValidationError,
    PatchWorkspaceId,
    SourceBytes,
)
from avalan.patch.planner import (
    PlannerFile,
    PlannerParentMount,
    PlannerWorkspace,
)


class TargetErrorCode(str, Enum):
    """Name non-disclosing target inspection outcomes."""

    PATH_DENIED = "patch.path_denied"
    CAPABILITY_UNAVAILABLE = "patch.capability_unavailable"
    WORKER_UNAVAILABLE = "patch.worker_unavailable"
    LINK_DENIED = "patch.link_denied"
    SPECIAL_FILE_DENIED = "patch.special_file_denied"
    HARDLINK_DENIED = "patch.hardlink_denied"
    ALIAS_DENIED = "patch.alias_denied"
    MOUNT_DENIED = "patch.mount_denied"
    LIMIT_EXCEEDED = "patch.limit_exceeded"
    WITNESS_STALE = "patch.stale"
    ISOLATION_DENIED = "patch.isolation_denied"
    METADATA_DENIED = "patch.metadata_denied"


WorkerAuthoritySignature = NewType("WorkerAuthoritySignature", str)


class TargetInspectionError(PatchValidationError):
    """Report a stable target error without retaining a backing path."""

    def __init__(self, code: TargetErrorCode) -> None:
        """Initialize one closed target rejection."""
        super().__init__(code.value)
        self.code = code


class TargetPrimitive(str, Enum):
    """Name the finite target properties needed before advertisement."""

    ROOTED_CONTAINMENT = "rooted_containment"
    NOFOLLOW_INSPECTION = "nofollow_inspection"
    REGULAR_FILE_IDENTITY = "regular_file_identity"
    METADATA_PRESERVATION = "metadata_preservation"
    BOUNDED_READ = "bounded_read"
    BOUNDED_WRITE = "bounded_write"
    REPLACE_PUBLICATION = "replace_publication"
    NOREPLACE_CREATE_MOVE = "noreplace_create_move"
    DIRECTORY_ENTRY_DELETE = "directory_entry_delete"
    SAME_FILESYSTEM_MOVE = "same_filesystem_move"
    STAGING = "staging"
    STRUCTURAL_VERIFICATION = "structural_verification"
    COORDINATION = "coordination"
    PERSISTENCE = "persistence"
    CANCELLATION_SETTLEMENT = "cancellation_settlement"
    JOURNAL_DELIVERY = "journal_delivery"
    APPROVAL = "approval"
    DURABLE_FENCING = "durable_fencing"


class TargetIncapableReason(str, Enum):
    """Name why a target cannot advertise an effectful patch operation."""

    COMMIT_DEFERRED = "commit_deferred"
    MISSING_PRIMITIVE = "missing_primitive"
    MISSING_METADATA = "missing_metadata"
    MISSING_COORDINATION = "missing_coordination"
    MISSING_APPROVAL = "missing_approval"
    MISSING_FENCING = "missing_fencing"


class AliasMode(str, Enum):
    """Name the target-declared logical-path identity projection."""

    CASE_SENSITIVE = "case_sensitive"
    CASE_INSENSITIVE = "case_insensitive"


class LocalPlatformProfile(str, Enum):
    """Name the finite local platform profiles this target can inspect."""

    POSIX = "posix"
    UNSUPPORTED = "unsupported"


class MetadataClassification(str, Enum):
    """Classify security-relevant POSIX mode facts before planning."""

    ORDINARY = "ordinary"
    EXECUTABLE = "executable"
    PRIVILEGED = "privileged"


class ProbeState(str, Enum):
    """Name the read-only outcome of one future mutation primitive probe."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ForeignWriterGuarantee(str, Enum):
    """State the honest Phase 4 guarantee for foreign workspace writers."""

    REVALIDATE_BEFORE_COMMIT = "revalidate_before_commit"


class _WorkerRootPayload(TypedDict):
    """Encode one root witness across the authenticated worker boundary."""

    identity: list[int]
    mount_id: str
    filesystem_id: str


class _WorkerRequestPayload(TypedDict):
    """Encode one complete authenticated inspection worker request."""

    operation: str
    root: str
    cwd: str | None
    maximum: int
    aggregate_maximum: int
    authority_signature: WorkerAuthoritySignature
    paths: list[str]
    expected_root: _WorkerRootPayload | None


class _WorkerSnapshotParentPayload(TypedDict):
    """Encode one parent witness attached to a worker snapshot."""

    path: str | None
    identity: list[int]
    mount_id: str


class _WorkerSnapshotMetadataPayload(TypedDict):
    """Encode stable metadata facts for one worker snapshot."""

    mode: int
    has_bom: bool
    representation: str


class _WorkerSnapshotPayload(TypedDict, total=False):
    """Encode one present or absent snapshot on the worker channel."""

    path: str
    present: bool
    bytes: str
    metadata: _WorkerSnapshotMetadataPayload
    identity: list[int]
    link_count: int
    parent: _WorkerSnapshotParentPayload
    classification: str


class _WorkerResponsePayload(TypedDict, total=False):
    """Encode one worker result without exposing its filesystem handles."""

    error: str
    identity: list[object]
    mount_id: str
    filesystem_id: str
    snapshots: list[object]


@dataclass(frozen=True, slots=True)
class WorkerIsolationPolicy:
    """Describe a trusted worker channel without its backing handle."""

    channel_id: str = "local-inspection-channel-v1"
    worker_instance_id: str = "local-inspection-worker-v1"
    safe_cwd_label: str = "target-private"
    inherited_descriptor_count: int = 0
    credential_count: int = 0
    network_enabled: bool = False
    workspace_imports_enabled: bool = False

    def __post_init__(self) -> None:
        """Reject a worker profile that could inherit ambient authority."""
        if (
            not self.channel_id
            or not self.worker_instance_id
            or not self.safe_cwd_label
            or self.inherited_descriptor_count != 0
            or self.credential_count != 0
            or self.network_enabled
            or self.workspace_imports_enabled
        ):
            raise TargetInspectionError(TargetErrorCode.ISOLATION_DENIED)


@dataclass(frozen=True, slots=True, eq=False)
class _WorkerAuthorization:
    """Keep the worker-channel bearer unforgeable outside trusted runtime."""

    token: bytes = field(default_factory=lambda: token_bytes(32))

    def __reduce__(self) -> Never:
        """Refuse serialization of the private worker-channel bearer."""
        raise TypeError("private worker authorization is not serializable")


_RUNTIME_TARGET_AUTHORITY_DOMAIN = (
    b"avalan.patch.runtime-target-authority.v1\0"
)
_RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES = b64decode(
    "1QBaXsznZDDCjbfP0rrLeqyYK7IupAVbE5mSfXO5PXY="
)


@final
@dataclass(frozen=True, slots=True, eq=False)
class _RuntimeTargetAuthority:
    """Carry a deployment-signed capability bound to one local root.

    The deployment authority keeps the Ed25519 signing key outside workspace
    Python.  This module has only its public verifier, so importing it cannot
    mint a capability or retarget a captured capability to another root.
    """

    _signature: bytes

    def __post_init__(self) -> None:
        """Reject malformed capability material before it reaches a root."""
        if type(self._signature) is not bytes or len(self._signature) != 64:
            raise TargetInspectionError(TargetErrorCode.PATH_DENIED)

    def __copy__(self) -> Never:
        """Refuse copying a runtime-owned authority capability."""
        raise TypeError("runtime target authority is not copyable")

    def __deepcopy__(self, memo: object) -> Never:
        """Refuse deep copying a runtime-owned authority capability."""
        del memo
        raise TypeError("runtime target authority is not copyable")

    def __reduce_ex__(self, protocol: int) -> Never:
        """Refuse serialization of a runtime-owned authority capability."""
        del protocol
        raise TypeError("runtime target authority is not serializable")

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Reject subclasses that could alter authority validation."""
        del cls, kwargs
        raise TypeError("runtime target authority cannot be subclassed")


def _runtime_target_authority_message(root: Path) -> bytes:
    """Return the domain-separated payload signed for one configured root."""
    return _RUNTIME_TARGET_AUTHORITY_DOMAIN + str(root).encode("utf-8")


def _authority_allows_root(authority: object, root: Path) -> bool:
    """Verify an exact external capability before accepting a local root."""
    if (
        type(authority) is not _RuntimeTargetAuthority
        or not root.is_absolute()
    ):
        return False
    try:
        Ed25519PublicKey.from_public_bytes(
            _RUNTIME_TARGET_AUTHORITY_VERIFIER_BYTES
        ).verify(
            authority._signature,
            _runtime_target_authority_message(root),
        )
    except (AttributeError, InvalidSignature, TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True, slots=True)
class PrimitiveProbe:
    """Record a read-only capability probe for one future write primitive."""

    primitive: TargetPrimitive
    state: ProbeState

    def __post_init__(self) -> None:
        """Reject probes for primitives that are already effectful today."""
        if self.primitive in _INSPECTION_PRIMITIVES:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class RootWitness:
    """Bind a resolved scope to one opened root identity and mount."""

    identity: "FileIdentity"
    mount_id: str
    filesystem_id: str = ""


@dataclass(frozen=True, slots=True)
class _MountTopology:
    """Store opaque mount-table facts for one retained descriptor."""

    mount_id: str
    filesystem_id: str


class _DarwinStatFs(Structure):
    """Represent Darwin's platform-owned statfs mount-table observation."""

    _fields_ = (
        ("f_bsize", c_uint32),
        ("f_iosize", c_int32),
        ("f_blocks", c_uint64),
        ("f_bfree", c_uint64),
        ("f_bavail", c_uint64),
        ("f_files", c_uint64),
        ("f_ffree", c_uint64),
        ("f_fsid", c_int32 * 2),
        ("f_owner", c_uint32),
        ("f_type", c_uint32),
        ("f_flags", c_uint32),
        ("f_fssubtype", c_uint32),
        ("f_fstypename", c_char * 16),
        ("f_mntonname", c_char * 1024),
        ("f_mntfromname", c_char * 1024),
        ("f_reserved", c_uint32 * 8),
    )


@dataclass(frozen=True, slots=True)
class EphemeralWorkerWitness:
    """Separate a replaceable worker instance from plan-bound identities."""

    channel_id: str
    worker_instance_id: str
    fence_id: str

    def __post_init__(self) -> None:
        """Reject empty opaque worker-channel witness fields."""
        if (
            not self.channel_id
            or not self.worker_instance_id
            or not self.fence_id
        ):
            raise TargetInspectionError(TargetErrorCode.ISOLATION_DENIED)


_INSPECTION_PRIMITIVES = frozenset(
    (
        TargetPrimitive.ROOTED_CONTAINMENT,
        TargetPrimitive.NOFOLLOW_INSPECTION,
        TargetPrimitive.REGULAR_FILE_IDENTITY,
        TargetPrimitive.BOUNDED_READ,
    )
)
_FUTURE_MUTATION_PRIMITIVES = frozenset(
    (
        TargetPrimitive.BOUNDED_WRITE,
        TargetPrimitive.REPLACE_PUBLICATION,
        TargetPrimitive.NOREPLACE_CREATE_MOVE,
        TargetPrimitive.DIRECTORY_ENTRY_DELETE,
        TargetPrimitive.SAME_FILESYSTEM_MOVE,
        TargetPrimitive.STAGING,
        TargetPrimitive.STRUCTURAL_VERIFICATION,
    )
)


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    """Store stable target, filesystem, mount, and policy witnesses."""

    context_id: PatchContextId
    workspace_id: PatchWorkspaceId
    domain_id: PatchDomainId
    target_id: PatchTargetId
    protocol_id: PatchProtocolId
    filesystem_id: str
    mount_id: str
    policy_revision: str
    persistent_lease_id: str
    approval_channel_id: PatchApprovalId

    def __post_init__(self) -> None:
        """Reject blank immutable trusted witness fields."""
        if any(
            not item
            for item in (
                self.filesystem_id,
                self.mount_id,
                self.policy_revision,
                self.persistent_lease_id,
                self.approval_channel_id,
            )
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class ScopeSelection:
    """Carry only a requested trusted context kind, never a path or backend."""

    context_kind: ContextKind


@dataclass(frozen=True, slots=True, repr=False)
class TrustedLocalRoot:
    """Keep a runtime-owned absolute root out of context-visible values."""

    _path: Path
    _runtime_authority: _RuntimeTargetAuthority = field(kw_only=True)

    def __post_init__(self) -> None:
        """Require a configured absolute root without resolving user input."""
        if (
            not _authority_allows_root(
                self._runtime_authority,
                self._path,
            )
            or not self._path.is_absolute()
        ):
            raise TargetInspectionError(TargetErrorCode.PATH_DENIED)


@dataclass(frozen=True, slots=True)
class LocalTargetProfile:
    """Store immutable local inspection policy selected by trusted runtime."""

    identity: TargetIdentity
    root: TrustedLocalRoot
    cwd: LogicalPath | None
    limits: PatchLimits
    max_snapshot_bytes: ByteSize
    _runtime_authority: _RuntimeTargetAuthority = field(kw_only=True)
    alias_mode: AliasMode = AliasMode.CASE_SENSITIVE
    unicode_normalization: str = "NFC"
    hidden_paths_allowed: bool = False
    platform: LocalPlatformProfile = LocalPlatformProfile.POSIX
    worker_policy: WorkerIsolationPolicy = field(
        default_factory=WorkerIsolationPolicy
    )
    _worker_authorization: _WorkerAuthorization = field(
        init=False,
        repr=False,
        compare=False,
        default_factory=_WorkerAuthorization,
    )

    def __post_init__(self) -> None:
        """Reject unsupported target configuration before worker creation."""
        if (
            not _authority_allows_root(
                self._runtime_authority,
                self.root._path,
            )
            or self.root._runtime_authority is not self._runtime_authority
            or self.unicode_normalization not in {"NFC", "NFD"}
            or self.max_snapshot_bytes.value == 0
            or not isinstance(self.platform, LocalPlatformProfile)
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class _WorkerInspectionProfile:
    """Carry the sole filesystem facts accepted by the isolated worker."""

    root_path: Path
    cwd: LogicalPath | None
    max_snapshot_bytes: int
    max_aggregate_snapshot_bytes: int


@dataclass(frozen=True, slots=True)
class ResolvedMutationScope:
    """Store a complete trusted scope without a backing-path projection."""

    context_kind: ContextKind
    identity: TargetIdentity
    cwd: LogicalPath | None
    limits: PatchLimits
    capabilities: frozenset[Capability]
    primitives: frozenset[TargetPrimitive]
    root_witness: RootWitness | None = None
    worker: EphemeralWorkerWitness | None = None
    _worker_authorization: _WorkerAuthorization | None = None

    def __post_init__(self) -> None:
        """Require immutable inspection authority and primitive witnesses."""
        if (
            type(self.capabilities) is not frozenset
            or type(self.primitives) is not frozenset
            or Capability.READ_FOR_MUTATION not in self.capabilities
            or Capability.OBSERVE_MUTATION_PRECONDITIONS
            not in self.capabilities
            or (
                self.root_witness is not None
                and not isinstance(self.root_witness, RootWitness)
            )
            or self.worker is not None
            and not isinstance(self.worker, EphemeralWorkerWitness)
            or self._worker_authorization is not None
            and not isinstance(
                self._worker_authorization, _WorkerAuthorization
            )
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


class ScopeResolver(Protocol):
    """Resolve a trusted immutable scope asynchronously."""

    async def resolve(
        self, selection: ScopeSelection
    ) -> ResolvedMutationScope:
        """Return only the runtime-configured scope for a matching context."""


@dataclass(frozen=True, slots=True)
class LocalScopeResolver:
    """Bind a local target profile without model-controlled fields."""

    profile: LocalTargetProfile

    def __post_init__(self) -> None:
        """Reject profiles not minted by the configured runtime factory."""
        if not _authority_allows_root(
            self.profile._runtime_authority,
            self.profile.root._path,
        ):
            raise TargetInspectionError(TargetErrorCode.ISOLATION_DENIED)

    async def resolve(
        self, selection: ScopeSelection
    ) -> ResolvedMutationScope:
        """Return a local scope or fail before filesystem inspection."""
        await sleep(0)
        if selection.context_kind is not ContextKind.LOCAL:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        witness = await _worker_root_witness(self.profile)
        if (
            witness.filesystem_id != self.profile.identity.filesystem_id
            or witness.mount_id != self.profile.identity.mount_id
        ):
            raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED)
        primitives = (
            _INSPECTION_PRIMITIVES
            if self.profile.platform is LocalPlatformProfile.POSIX
            else frozenset()
        )
        worker = EphemeralWorkerWitness(
            self.profile.worker_policy.channel_id,
            self.profile.worker_policy.worker_instance_id,
            _fence_id(self.profile.identity, witness),
        )
        return ResolvedMutationScope(
            ContextKind.LOCAL,
            self.profile.identity,
            self.profile.cwd,
            self.profile.limits,
            frozenset(
                (
                    Capability.READ_FOR_MUTATION,
                    Capability.OBSERVE_MUTATION_PRECONDITIONS,
                )
            ),
            primitives,
            witness,
            worker,
            self.profile._worker_authorization,
        )

    async def rebind_ephemeral(
        self, scope: ResolvedMutationScope
    ) -> ResolvedMutationScope:
        """Replace only a worker instance while preserving plan-bound facts."""
        await sleep(0)
        if (
            scope.identity != self.profile.identity
            or scope.root_witness is None
            or scope._worker_authorization
            is not self.profile._worker_authorization
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        worker = EphemeralWorkerWitness(
            self.profile.worker_policy.channel_id,
            self.profile.worker_policy.worker_instance_id + "-rebound",
            _fence_id(self.profile.identity, scope.root_witness),
        )
        return replace(scope, worker=worker)


@dataclass(frozen=True, slots=True)
class TargetHandshake:
    """Bind advertised target facts to immutable scope identities."""

    identity: TargetIdentity
    primitives: frozenset[TargetPrimitive]
    incapable_reasons: tuple[TargetIncapableReason, ...]
    probes: tuple[PrimitiveProbe, ...] = ()
    platform: LocalPlatformProfile = LocalPlatformProfile.UNSUPPORTED
    foreign_writer_guarantee: ForeignWriterGuarantee = (
        ForeignWriterGuarantee.REVALIDATE_BEFORE_COMMIT
    )
    worker: EphemeralWorkerWitness | None = None

    def __post_init__(self) -> None:
        """Keep a capability handshake deterministic and immutable."""
        if (
            type(self.primitives) is not frozenset
            or type(self.incapable_reasons) is not tuple
            or len(set(self.incapable_reasons)) != len(self.incapable_reasons)
            or type(self.probes) is not tuple
            or len({item.primitive for item in self.probes})
            != len(self.probes)
            or not isinstance(self.platform, LocalPlatformProfile)
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)

    def supports_inspection(self) -> bool:
        """Return whether all inspection primitives are present."""
        return _INSPECTION_PRIMITIVES.issubset(self.primitives)

    def advertised_operations(self) -> frozenset[Capability]:
        """Return no effects while commit remains intentionally absent."""
        return frozenset()


@dataclass(frozen=True, slots=True)
class FileIdentity:
    """Store a target-native regular-file identity without a backing path."""

    device: int
    inode: int

    def __post_init__(self) -> None:
        """Reject invalid POSIX identity observations."""
        if self.device < 0 or self.inode < 0:
            raise TargetInspectionError(TargetErrorCode.SPECIAL_FILE_DENIED)

    def opaque(self) -> str:
        """Return a stable opaque planner projection of this identity."""
        return sha256(f"{self.device}:{self.inode}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ParentWitness:
    """Record retained parent and mount identity for later revalidation."""

    path: LogicalPath | None
    identity: FileIdentity
    mount_id: str


@dataclass(frozen=True, slots=True)
class TargetSnapshot:
    """Store one bounded target-native source observation."""

    path: LogicalPath
    present: bool
    bytes_value: SourceBytes | None
    metadata: MetadataProfile | None
    identity: FileIdentity | None
    link_count: int
    parent: ParentWitness
    security_metadata: MetadataClassification = MetadataClassification.ORDINARY

    def __post_init__(self) -> None:
        """Keep absent and present snapshot facts structurally disjoint."""
        if self.present:
            if (
                self.bytes_value is None
                or self.metadata is None
                or self.identity is None
                or self.link_count != 1
            ):
                raise TargetInspectionError(
                    TargetErrorCode.SPECIAL_FILE_DENIED
                )
        elif (
            self.bytes_value is not None
            or self.metadata is not None
            or self.identity is not None
            or self.link_count != 0
        ):
            raise TargetInspectionError(TargetErrorCode.SPECIAL_FILE_DENIED)
        if self.security_metadata is MetadataClassification.PRIVILEGED:
            raise TargetInspectionError(TargetErrorCode.METADATA_DENIED)


@dataclass(frozen=True, slots=True)
class InspectionRequest:
    """Request immutable logical snapshots from a selected target scope."""

    scope: ResolvedMutationScope
    paths: tuple[LogicalPath, ...]

    def __post_init__(self) -> None:
        """Reject empty, duplicated, or cross-scope inspection requests."""
        if (
            not self.paths
            or len(set(self.paths)) != len(self.paths)
            or self.scope.context_kind is not ContextKind.LOCAL
        ):
            raise TargetInspectionError(TargetErrorCode.PATH_DENIED)


@dataclass(frozen=True, slots=True)
class InspectionBatch:
    """Return one read-once immutable batch and a planner projection."""

    snapshots: tuple[TargetSnapshot, ...]

    def __post_init__(self) -> None:
        """Reject duplicate logical paths or regular-file identities."""
        paths = tuple(item.path for item in self.snapshots)
        identities = tuple(
            item.identity
            for item in self.snapshots
            if item.identity is not None
        )
        if len(paths) != len(set(paths)) or len(identities) != len(
            set(identities)
        ):
            raise TargetInspectionError(TargetErrorCode.ALIAS_DENIED)

    def planner_workspace(self) -> PlannerWorkspace:
        """Project observed target facts without another filesystem read."""
        files = tuple(
            PlannerFile(
                item.path,
                item.bytes_value,
                item.metadata,
                item.parent.path,
                item.parent.mount_id,
                item.identity.opaque(),
            )
            for item in self.snapshots
            if item.present
            and item.bytes_value is not None
            and item.metadata is not None
            and item.identity is not None
        )
        parents = frozenset(
            item.parent.path
            for item in self.snapshots
            if item.parent.path is not None
        )
        mounts = {
            item.parent.path: item.parent.mount_id for item in self.snapshots
        }
        parent_mounts = tuple(
            PlannerParentMount(path, mount_id)
            for path, mount_id in sorted(
                mounts.items(),
                key=lambda item: "" if item[0] is None else item[0].value,
            )
        )
        return PlannerWorkspace(files, parents, parent_mounts)


@dataclass(frozen=True, slots=True)
class CommitUnavailable:
    """Return the only Phase 4 commit outcome without target mutation."""

    code: TargetErrorCode = TargetErrorCode.CAPABILITY_UNAVAILABLE


class MutationTarget(Protocol):
    """Expose typed async inspection and incapable commit operations only."""

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return identity-bound target capability evidence."""

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Return one bounded immutable snapshot batch."""

    async def commit(self, request: InspectionRequest) -> CommitUnavailable:
        """Return a typed unavailable result without a namespace write."""


@dataclass(frozen=True, slots=True)
class LocalInspectionTarget:
    """Provide rooted local inspection through an effect-free target facade."""

    profile: LocalTargetProfile

    def __post_init__(self) -> None:
        """Reject target profiles not minted by the runtime factory."""
        if not _authority_allows_root(
            self.profile._runtime_authority,
            self.profile.root._path,
        ):
            raise TargetInspectionError(TargetErrorCode.ISOLATION_DENIED)

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return local inspection facts without advertisement."""
        await sleep(0)
        self._require_scope(scope)
        return TargetHandshake(
            self.profile.identity,
            scope.primitives,
            (
                TargetIncapableReason.COMMIT_DEFERRED,
                TargetIncapableReason.MISSING_PRIMITIVE,
                TargetIncapableReason.MISSING_METADATA,
                TargetIncapableReason.MISSING_COORDINATION,
                TargetIncapableReason.MISSING_APPROVAL,
                TargetIncapableReason.MISSING_FENCING,
            ),
            _future_primitive_probes(),
            self.profile.platform,
            ForeignWriterGuarantee.REVALIDATE_BEFORE_COMMIT,
            scope.worker,
        )

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Inspect paths once through rooted no-follow descriptor traversal."""
        await sleep(0)
        self._require_scope(request.scope)
        if (
            self.profile.platform is not LocalPlatformProfile.POSIX
            or request.scope.root_witness is None
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        _validate_aliases(request.paths, self.profile)
        snapshots = await _worker_inspect(
            self.profile,
            request.paths,
            request.scope.root_witness,
        )
        return InspectionBatch(snapshots)

    async def commit(self, request: InspectionRequest) -> CommitUnavailable:
        """Reject commit before opening any path or staging namespace."""
        self._require_scope(request.scope)
        return CommitUnavailable()

    def _require_scope(self, scope: ResolvedMutationScope) -> None:
        """Reject replacement of any plan-bound local target witness."""
        if (
            scope.identity != self.profile.identity
            or scope.worker is None
            or scope.worker.channel_id != self.profile.worker_policy.channel_id
            or scope._worker_authorization
            is not self.profile._worker_authorization
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


_WORKER_TOKEN_ENV = "AVALAN_PATCH_LOCAL_WORKER_TOKEN"
_SEATBELT_EXECUTABLE = "/usr/bin/sandbox-exec"
_SEATBELT_SYSTEM_READ_DATA_PATHS = (
    "/",
    "/System/Library/dyld",
    "/System/Volumes/Preboot/Cryptexes/OS/System/Library/dyld",
    "/private/var/db/dyld",
    "/usr/lib/dyld",
)
_SEATBELT_SYSTEM_MACH_LOOKUP_SERVICES = (
    "com.apple.logd",
    "com.apple.system.notification_center",
    "com.apple.system.opendirectoryd.libinfo",
)
_WORKER_BOOTSTRAP = """import sys
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

source_root = sys.argv[1]
cryptography_root = sys.argv[2]
cffi_backend_path = sys.argv[3]
ffi_spec = spec_from_file_location("_cffi_backend", cffi_backend_path)
if ffi_spec is None or ffi_spec.loader is None:
    raise RuntimeError("cryptography ffi runtime is unavailable")
ffi_backend = module_from_spec(ffi_spec)
sys.modules["_cffi_backend"] = ffi_backend
ffi_spec.loader.exec_module(ffi_backend)
cryptography_spec = spec_from_file_location(
    "cryptography",
    cryptography_root + "/__init__.py",
    submodule_search_locations=[cryptography_root],
)
if cryptography_spec is None or cryptography_spec.loader is None:
    raise RuntimeError("cryptography runtime is unavailable")
cryptography = module_from_spec(cryptography_spec)
sys.modules["cryptography"] = cryptography
cryptography_spec.loader.exec_module(cryptography)
avalan = ModuleType("avalan")
avalan.__path__ = [source_root + "/avalan"]
sys.modules["avalan"] = avalan
patch = ModuleType("avalan.patch")
patch.__path__ = [source_root + "/avalan/patch"]
sys.modules["avalan.patch"] = patch
planner = ModuleType("avalan.patch.planner")
class _PlannerUnavailable:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("planner projection is unavailable in worker")
planner.PlannerFile = _PlannerUnavailable
planner.PlannerParentMount = _PlannerUnavailable
planner.PlannerWorkspace = _PlannerUnavailable
sys.modules["avalan.patch.planner"] = planner
from avalan.patch.target import _worker_main
raise SystemExit(_worker_main())
"""


async def _worker_root_witness(profile: LocalTargetProfile) -> RootWitness:
    """Ask the isolated worker to observe the configured root identity."""
    response = await _worker_request(profile, "witness", (), None)
    identity = response.get("identity")
    mount_id = response.get("mount_id")
    if (
        not isinstance(identity, list)
        or len(identity) != 2
        or not all(type(item) is int for item in identity)
        or not isinstance(mount_id, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    filesystem_id = response.get("filesystem_id")
    if not isinstance(filesystem_id, str):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    device, inode = identity
    assert type(device) is int and type(inode) is int
    return RootWitness(FileIdentity(device, inode), mount_id, filesystem_id)


async def _worker_inspect(
    profile: LocalTargetProfile,
    paths: tuple[LogicalPath, ...],
    expected_root: RootWitness,
) -> tuple[TargetSnapshot, ...]:
    """Request bounded snapshots from the authenticated isolated worker."""
    response = await _worker_request(profile, "inspect", paths, expected_root)
    snapshots = response.get("snapshots")
    if not isinstance(snapshots, list):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return tuple(_snapshot_from_worker(item) for item in snapshots)


async def _worker_request(
    profile: LocalTargetProfile,
    operation: str,
    paths: tuple[LogicalPath, ...],
    expected_root: RootWitness | None,
) -> _WorkerResponsePayload:
    """Run one sealed no-network worker request and settle cancellation."""
    if (
        not _authority_allows_root(
            profile._runtime_authority,
            profile.root._path,
        )
        or not await _seatbelt_worker_available()
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    expected_root_payload: _WorkerRootPayload | None
    if expected_root is None:
        expected_root_payload = None
    else:
        expected_root_payload = {
            "identity": [
                expected_root.identity.device,
                expected_root.identity.inode,
            ],
            "mount_id": expected_root.mount_id,
            "filesystem_id": expected_root.filesystem_id,
        }
    payload: _WorkerRequestPayload = {
        "operation": operation,
        "root": str(profile.root._path),
        "cwd": profile.cwd.value if profile.cwd is not None else None,
        "maximum": profile.max_snapshot_bytes.value,
        "aggregate_maximum": profile.limits.snapshot_bytes.value,
        "authority_signature": WorkerAuthoritySignature(
            b64encode(profile._runtime_authority._signature).decode()
        ),
        "paths": [path.value for path in paths],
        "expected_root": expected_root_payload,
    }
    raw_payload = dumps(payload, separators=(",", ":")).encode()
    token = profile._worker_authorization.token
    message = dumps(
        {
            "payload": payload,
            "mac": digest(token, raw_payload, "sha256").hex(),
        },
        separators=(",", ":"),
    ).encode()
    worker_argv = (
        executable,
        "-I",
        "-S",
        "-c",
        _WORKER_BOOTSTRAP,
        str(Path(__file__).resolve().parents[2]),
        str(Path(cryptography_file).resolve().parent),
        str(_cffi_backend_runtime_path()),
    )
    try:
        process = await create_subprocess_exec(
            _SEATBELT_EXECUTABLE,
            "-p",
            _worker_seatbelt_profile(profile, worker_argv, token.hex()),
            "--",
            *worker_argv,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd="/",
            env={_WORKER_TOKEN_ENV: token.hex()},
            close_fds=True,
        )
    except OSError as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    try:
        response_bytes, error_bytes = await process.communicate(message)
    except CancelledError:
        process.terminate()
        await process.wait()
        raise
    if process.returncode != 0:
        del error_bytes
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    try:
        envelope = loads(response_bytes)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    if not isinstance(envelope, dict):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    response = _worker_response_payload(envelope.get("payload"))
    response_mac = envelope.get("mac")
    if not isinstance(response_mac, str):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    raw_response = dumps(response, separators=(",", ":")).encode()
    if not compare_digest(
        response_mac, digest(token, raw_response, "sha256").hex()
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    error = response.get("error")
    if error is not None:
        try:
            code = TargetErrorCode(error)
        except ValueError as exc:
            raise TargetInspectionError(
                TargetErrorCode.WORKER_UNAVAILABLE
            ) from exc
        raise TargetInspectionError(code)
    return response


async def _seatbelt_worker_available() -> bool:
    """Return whether Avalan can enforce this worker's network denial."""
    return Path(_SEATBELT_EXECUTABLE).is_file()


def _cffi_backend_runtime_path() -> Path:
    """Return the exact native cryptography runtime path or fail closed."""
    specification = find_spec("_cffi_backend")
    if specification is None or specification.origin is None:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return Path(specification.origin).resolve()


def _worker_seatbelt_profile(
    profile: LocalTargetProfile,
    worker_argv: tuple[str, ...],
    token: str,
) -> str:
    """Generate a narrow no-network Seatbelt policy for one worker."""
    interpreter = Path(executable).resolve()
    target_source_root = Path(__file__).resolve().parent
    if cryptography_file is None:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    cryptography_root = Path(cryptography_file).resolve().parent
    del token, worker_argv
    read_paths = (
        *_SEATBELT_SYSTEM_READ_DATA_PATHS,
        str(Path(executable).parent),
        str(interpreter.parent),
        str(interpreter.parent.parent),
        str(profile.root._path),
        str(target_source_root),
        str(cryptography_root),
        str(_cffi_backend_runtime_path()),
    )
    lines = [
        "(version 1)",
        "(deny default)",
        "(allow process*)",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
    ]
    for path in read_paths:
        lines.append(_seatbelt_read_data(path))
    for service in _SEATBELT_SYSTEM_MACH_LOOKUP_SERVICES:
        lines.append(
            "(allow mach-lookup (global-name "
            + _seatbelt_string(service)
            + "))"
        )
    lines.extend(("(deny network*)", "(deny process-fork)"))
    return "\n".join(lines) + "\n"


def _seatbelt_read_data(path: str) -> str:
    """Grant read data for one trusted worker path without write authority."""
    if path == "/":
        return '(allow file-read-data (literal "/"))'
    return "(allow file-read* (subpath " + _seatbelt_string(path) + "))"


def _seatbelt_string(value: str) -> str:
    """Encode one trusted literal for a generated Seatbelt expression."""
    if not value or "\x00" in value or "\n" in value or "\r" in value:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _snapshot_from_worker(value: object) -> TargetSnapshot:
    """Decode one authenticated worker snapshot without a backing path."""
    if not isinstance(value, dict):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    path = value.get("path")
    present = value.get("present")
    parent = value.get("parent")
    if not isinstance(path, str) or type(present) is not bool:
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    parent_witness = _parent_from_worker(parent)
    if not present:
        return TargetSnapshot(
            LogicalPath(path), False, None, None, None, 0, parent_witness
        )
    raw_bytes = value.get("bytes")
    metadata = value.get("metadata")
    identity = value.get("identity")
    link_count = value.get("link_count")
    classification = value.get("classification")
    if (
        not isinstance(raw_bytes, str)
        or not isinstance(metadata, dict)
        or not isinstance(identity, list)
        or len(identity) != 2
        or not all(type(item) is int for item in identity)
        or type(link_count) is not int
        or not isinstance(classification, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    mode = metadata.get("mode")
    has_bom = metadata.get("has_bom")
    representation = metadata.get("representation")
    if (
        type(mode) is not int
        or type(has_bom) is not bool
        or not isinstance(representation, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    try:
        bytes_value = b64decode(raw_bytes, validate=True)
        security_metadata = MetadataClassification(classification)
    except (ValueError, UnicodeError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    return TargetSnapshot(
        LogicalPath(path),
        True,
        SourceBytes(bytes_value),
        MetadataProfile(FileMode(mode), has_bom, representation),
        FileIdentity(identity[0], identity[1]),
        link_count,
        parent_witness,
        security_metadata,
    )


def _parent_from_worker(value: object) -> ParentWitness:
    """Decode one worker parent identity without a host-path field."""
    if not isinstance(value, dict):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    path = value.get("path")
    identity = value.get("identity")
    mount_id = value.get("mount_id")
    if (
        path is not None
        and not isinstance(path, str)
        or not isinstance(identity, list)
        or len(identity) != 2
        or not all(type(item) is int for item in identity)
        or not isinstance(mount_id, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return ParentWitness(
        LogicalPath(path) if path is not None else None,
        FileIdentity(identity[0], identity[1]),
        mount_id,
    )


def _future_primitive_probes() -> tuple[PrimitiveProbe, ...]:
    """Return read-only Phase 4 evidence for deferred write primitives."""
    return tuple(
        PrimitiveProbe(primitive, ProbeState.UNAVAILABLE)
        for primitive in sorted(
            _FUTURE_MUTATION_PRIMITIVES, key=lambda item: item.value
        )
    )


def _capture_root_witness(root: TrustedLocalRoot) -> RootWitness:
    """Open a configured root and retain only opaque identity facts."""
    descriptor = _open_directory(root._path)
    try:
        status = fstat(descriptor)
        return RootWitness(
            FileIdentity(status.st_dev, status.st_ino),
            _root_mount_id(descriptor, status),
            _filesystem_id(descriptor),
        )
    finally:
        close(descriptor)


def _fence_id(identity: TargetIdentity, witness: RootWitness) -> str:
    """Derive an opaque non-commit fence from plan-bound target facts."""
    return sha256(
        ":".join(
            (
                identity.persistent_lease_id,
                identity.target_id.value,
                witness.identity.opaque(),
                witness.mount_id,
            )
        ).encode()
    ).hexdigest()


def _inspection_barrier(stage: str) -> None:
    """Provide a deterministic test-only point between no-follow operations."""
    del stage


def _validate_aliases(
    paths: tuple[LogicalPath, ...], profile: LocalTargetProfile
) -> None:
    """Reject target-native case and Unicode aliases before inspection."""
    aliases = tuple(_alias_key(item, profile) for item in paths)
    if (
        len(paths) > profile.limits.path_count.value
        or len(paths) > profile.limits.file_count.value
        or len(aliases) != len(set(aliases))
    ):
        raise TargetInspectionError(TargetErrorCode.ALIAS_DENIED)
    for path in paths:
        if len(path.value.encode()) > profile.limits.path_length.value:
            raise TargetInspectionError(TargetErrorCode.LIMIT_EXCEEDED)
        _validate_path(path, profile)


def _alias_key(path: LogicalPath, profile: LocalTargetProfile) -> str:
    """Return the configured target-native logical identity projection."""
    value = normalize(
        "NFC" if profile.unicode_normalization == "NFC" else "NFD",
        path.value,
    )
    return (
        value.casefold()
        if profile.alias_mode is AliasMode.CASE_INSENSITIVE
        else value
    )


def _validate_path(path: LogicalPath, profile: LocalTargetProfile) -> None:
    """Apply lexical denial before directory inspection."""
    components = path.value.split("/")
    forbidden_bidi = {
        "\u202a",
        "\u202b",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
    for component in components:
        if (
            component != component.strip()
            or component == ".git"
            or (component.startswith(".") and not profile.hidden_paths_allowed)
            or any(
                ord(char) < 32 or char in forbidden_bidi for char in component
            )
            or component.startswith(("~", "$"))
            or ":" in component
        ):
            raise TargetInspectionError(TargetErrorCode.PATH_DENIED)


def _inspect_many(
    profile: LocalTargetProfile | _WorkerInspectionProfile,
    paths: tuple[LogicalPath, ...],
    expected_root: RootWitness,
) -> tuple[TargetSnapshot, ...]:
    """Perform confined read-only POSIX inspection below one trusted root."""
    root_path = (
        profile.root._path
        if isinstance(profile, LocalTargetProfile)
        else profile.root_path
    )
    try:
        root_fd = _open_directory(root_path)
    except TargetInspectionError as exc:
        if exc.code is TargetErrorCode.WORKER_UNAVAILABLE:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
        raise
    try:
        root_status = fstat(root_fd)
        current_root = RootWitness(
            FileIdentity(root_status.st_dev, root_status.st_ino),
            _root_mount_id(root_fd, root_status),
            _filesystem_id(root_fd),
        )
        if current_root != expected_root:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        cwd_fd, cwd_identity = _open_cwd(
            root_fd,
            profile.cwd,
            current_root.filesystem_id,
            current_root.mount_id,
        )
        try:
            root_stat = fstat(cwd_fd)
            if root_stat.st_dev != expected_root.identity.device:
                raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED)
            mount_id = current_root.mount_id
            snapshots: list[TargetSnapshot] = []
            aggregate = 0
            limit = (
                profile.limits.snapshot_bytes.value
                if isinstance(profile, LocalTargetProfile)
                else profile.max_aggregate_snapshot_bytes
            )
            for path in paths:
                remaining = limit - aggregate
                snapshot = _inspect_path(
                    cwd_fd,
                    cwd_identity,
                    mount_id,
                    current_root.filesystem_id,
                    current_root.mount_id,
                    path,
                    profile,
                    remaining,
                )
                if snapshot.bytes_value is not None:
                    aggregate += len(snapshot.bytes_value._value)
                snapshots.append(snapshot)
            return tuple(snapshots)
        finally:
            close(cwd_fd)
    finally:
        close(root_fd)


def _open_directory(path: Path) -> int:
    """Open one trusted root without following link components."""
    try:
        descriptor = open(
            path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC
        )
    except OSError as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    try:
        status = fstat(descriptor)
        if not S_ISDIR(status.st_mode):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        _inspection_barrier("root")
        opened = fstat(descriptor)
        if (
            opened.st_dev != status.st_dev
            or opened.st_ino != status.st_ino
            or not S_ISDIR(opened.st_mode)
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    except BaseException:
        close(descriptor)
        raise
    return descriptor


def _open_cwd(
    root_fd: int,
    cwd: LogicalPath | None,
    expected_filesystem_id: str | None = None,
    expected_mount_id: str | None = None,
) -> tuple[int, FileIdentity]:
    """Open the trusted effective cwd through retained no-follow handles."""
    descriptor = root_fd
    owned = False
    try:
        if expected_filesystem_id is None:
            expected_filesystem_id = _filesystem_id(root_fd)
        if expected_mount_id is None:
            expected_mount_id = _root_mount_id(root_fd, fstat(root_fd))
        if cwd is not None:
            root_status = fstat(root_fd)
            for component in cwd.value.split("/"):
                next_descriptor = _open_child_directory(
                    descriptor,
                    component,
                    root_status.st_dev,
                    expected_filesystem_id,
                    expected_mount_id,
                )
                if owned:
                    close(descriptor)
                descriptor = next_descriptor
                owned = True
        status = fstat(descriptor)
        identity = FileIdentity(status.st_dev, status.st_ino)
        if not owned:
            duplicate = open(
                ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC, dir_fd=root_fd
            )
            return duplicate, identity
        return descriptor, identity
    except BaseException:
        if owned:
            close(descriptor)
        raise


def _open_child_directory(
    parent_fd: int,
    component: str,
    expected_device: int | None = None,
    expected_filesystem_id: str | None = None,
    expected_mount_id: str | None = None,
) -> int:
    """Open one existing directory component without link traversal."""
    try:
        status = stat_at(component, dir_fd=parent_fd, follow_symlinks=False)
        if S_ISLNK(status.st_mode):
            raise TargetInspectionError(TargetErrorCode.LINK_DENIED)
        if not S_ISDIR(status.st_mode):
            raise TargetInspectionError(TargetErrorCode.SPECIAL_FILE_DENIED)
        _inspection_barrier("component")
        descriptor = open(
            component,
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC,
            dir_fd=parent_fd,
        )
        try:
            opened = fstat(descriptor)
            if (
                not S_ISDIR(opened.st_mode)
                or opened.st_dev != status.st_dev
                or opened.st_ino != status.st_ino
                or (
                    expected_device is not None
                    and opened.st_dev != expected_device
                )
                or (
                    expected_filesystem_id is not None
                    and _filesystem_id(descriptor) != expected_filesystem_id
                )
                or (
                    expected_mount_id is not None
                    and _root_mount_id(descriptor, opened) != expected_mount_id
                )
            ):
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            return descriptor
        except BaseException:
            close(descriptor)
            raise
    except TargetInspectionError:
        raise
    except OSError as exc:
        raise TargetInspectionError(TargetErrorCode.PATH_DENIED) from exc


def _inspect_path(
    cwd_fd: int,
    cwd_identity: FileIdentity,
    mount_id: str,
    filesystem_id: str,
    expected_mount_id: str,
    path: LogicalPath,
    profile: LocalTargetProfile | _WorkerInspectionProfile,
    remaining_snapshot_bytes: int | None = None,
) -> TargetSnapshot:
    """Inspect one path from retained parents without mutation."""
    parts = path.value.split("/")
    parent_fd = cwd_fd
    owned = False
    parent_path: LogicalPath | None = None
    try:
        current_cwd = fstat(cwd_fd)
        if (
            current_cwd.st_dev != cwd_identity.device
            or current_cwd.st_ino != cwd_identity.inode
            or _filesystem_id(cwd_fd) != filesystem_id
            or _root_mount_id(cwd_fd, current_cwd) != expected_mount_id
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        for index, component in enumerate(parts[:-1]):
            parent_fd = _advance_parent(
                parent_fd,
                owned,
                component,
                current_cwd.st_dev,
                filesystem_id,
                expected_mount_id,
            )
            owned = True
            parent_path = LogicalPath("/".join(parts[: index + 1]))
        parent_status = fstat(parent_fd)
        parent = ParentWitness(
            parent_path,
            FileIdentity(parent_status.st_dev, parent_status.st_ino),
            mount_id,
        )
        return _snapshot_leaf(
            parent_fd,
            path,
            parts[-1],
            parent,
            profile,
            expected_mount_id,
            remaining_snapshot_bytes,
        )
    finally:
        if owned:
            close(parent_fd)


def _advance_parent(
    parent_fd: int,
    owned: bool,
    component: str,
    expected_device: int,
    expected_filesystem_id: str,
    expected_mount_id: str,
) -> int:
    """Advance to one child directory and release only owned prior handles."""
    descriptor = _open_child_directory(
        parent_fd,
        component,
        expected_device,
        expected_filesystem_id,
        expected_mount_id,
    )
    if owned:
        close(parent_fd)
    return descriptor


def _snapshot_leaf(
    parent_fd: int,
    path: LogicalPath,
    leaf: str,
    parent: ParentWitness,
    profile: LocalTargetProfile | _WorkerInspectionProfile,
    expected_mount_id: str | None = None,
    remaining_snapshot_bytes: int | None = None,
) -> TargetSnapshot:
    """Return absent or supported-regular-file facts from one parent handle."""
    try:
        status = stat_at(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return TargetSnapshot(path, False, None, None, None, 0, parent)
    except OSError as exc:
        raise TargetInspectionError(TargetErrorCode.PATH_DENIED) from exc
    if S_ISLNK(status.st_mode):
        raise TargetInspectionError(TargetErrorCode.LINK_DENIED)
    if not S_ISREG(status.st_mode):
        raise TargetInspectionError(TargetErrorCode.SPECIAL_FILE_DENIED)
    if status.st_nlink != 1:
        raise TargetInspectionError(TargetErrorCode.HARDLINK_DENIED)
    if status.st_dev != parent.identity.device:
        raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED)
    _inspection_barrier("leaf")
    descriptor = _open_regular(parent_fd, leaf)
    try:
        opened = fstat(descriptor)
        if not S_ISREG(opened.st_mode):
            raise TargetInspectionError(TargetErrorCode.SPECIAL_FILE_DENIED)
        if (
            opened.st_nlink != 1
            or opened.st_dev != status.st_dev
            or opened.st_ino != status.st_ino
            or (
                expected_mount_id is not None
                and _root_mount_id(descriptor, opened) != expected_mount_id
            )
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        maximum = (
            profile.max_snapshot_bytes.value
            if isinstance(profile, LocalTargetProfile)
            else profile.max_snapshot_bytes
        )
        if (
            remaining_snapshot_bytes is not None
            and opened.st_size > remaining_snapshot_bytes
        ):
            raise TargetInspectionError(TargetErrorCode.LIMIT_EXCEEDED)
        value = _read_bounded(
            descriptor,
            min(
                maximum,
                (
                    remaining_snapshot_bytes
                    if remaining_snapshot_bytes is not None
                    else maximum
                ),
            ),
        )
    finally:
        close(descriptor)
    has_bom, representation = _snapshot_representation(value)
    security_metadata = _classify_metadata(opened.st_mode)
    if security_metadata is MetadataClassification.PRIVILEGED:
        raise TargetInspectionError(TargetErrorCode.METADATA_DENIED)
    metadata = MetadataProfile(
        FileMode(opened.st_mode & 0o777),
        has_bom,
        representation,
    )
    return TargetSnapshot(
        path,
        True,
        SourceBytes(value),
        metadata,
        FileIdentity(opened.st_dev, opened.st_ino),
        opened.st_nlink,
        parent,
        security_metadata,
    )


def _snapshot_representation(value: bytes) -> tuple[bool, str]:
    """Validate one supported UTF-8 source representation for a snapshot."""
    bom = b"\xef\xbb\xbf"
    has_bom = value.startswith(bom)
    raw = value[len(bom) :] if has_bom else value
    if raw.startswith(bom):
        raise TargetInspectionError(TargetErrorCode.METADATA_DENIED)
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise TargetInspectionError(TargetErrorCode.METADATA_DENIED) from exc
    if "\x00" in text or "\r" in text.replace("\r\n", ""):
        raise TargetInspectionError(TargetErrorCode.METADATA_DENIED)
    has_lf = "\n" in text.replace("\r\n", "")
    has_crlf = "\r\n" in text
    if has_lf and has_crlf:
        raise TargetInspectionError(TargetErrorCode.METADATA_DENIED)
    return has_bom, "crlf" if has_crlf else "lf"


def _open_regular(parent_fd: int, leaf: str) -> int:
    """Open one leaf without following a symbolic link."""
    try:
        return open(
            leaf,
            O_RDONLY | O_NONBLOCK | O_NOFOLLOW | O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise TargetInspectionError(TargetErrorCode.LINK_DENIED) from exc


def _read_bounded(descriptor: int, maximum: int) -> bytes:
    """Read exactly one prebounded regular-file size or fail closed."""
    status = fstat(descriptor)
    if status.st_size > maximum:
        raise TargetInspectionError(TargetErrorCode.LIMIT_EXCEEDED)
    chunks: list[bytes] = []
    remaining = status.st_size
    while remaining:
        chunk = read_fd(descriptor, min(65_536, remaining))
        if not chunk:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        chunks.append(chunk)
        remaining -= len(chunk)
    value = b"".join(chunks)
    after = fstat(descriptor)
    if (
        after.st_dev != status.st_dev
        or after.st_ino != status.st_ino
        or after.st_nlink != status.st_nlink
        or after.st_size != status.st_size
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return value


def _classify_metadata(mode: int) -> MetadataClassification:
    """Classify executable and privileged POSIX mode facts before planning."""
    if mode & 0o6000:
        return MetadataClassification.PRIVILEGED
    if mode & 0o111:
        return MetadataClassification.EXECUTABLE
    return MetadataClassification.ORDINARY


def _worker_main() -> int:
    """Serve an authenticated request in an isolated worker process."""
    token_value = environ.get(_WORKER_TOKEN_ENV)
    if token_value is None:
        return 2
    try:
        token = bytes.fromhex(token_value)
        envelope = loads(stdin.buffer.read())
        if not isinstance(envelope, dict):
            return 2
        payload = envelope.get("payload")
        message_mac = envelope.get("mac")
        if not isinstance(payload, dict) or not isinstance(message_mac, str):
            return 2
        raw_payload = dumps(payload, separators=(",", ":")).encode()
        if not compare_digest(
            message_mac, digest(token, raw_payload, "sha256").hex()
        ):
            return 2
        response = _worker_response(payload)
    except (TargetInspectionError, TypeError, ValueError, UnicodeError):
        return 2
    raw_response = dumps(response, separators=(",", ":")).encode()
    stdout.buffer.write(
        dumps(
            {
                "payload": response,
                "mac": digest(token, raw_response, "sha256").hex(),
            },
            separators=(",", ":"),
        ).encode()
    )
    return 0


def _worker_response_payload(value: object) -> _WorkerResponsePayload:
    """Decode one authenticated worker response without free-form maps."""
    if not isinstance(value, dict):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    error = value.get("error")
    if error is not None:
        if not isinstance(error, str):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return {"error": error}
    response: _WorkerResponsePayload = {}
    identity = value.get("identity")
    if isinstance(identity, list):
        decoded_identity: list[object] = []
        decoded_identity.extend(identity)
        response["identity"] = decoded_identity
    mount_id = value.get("mount_id")
    if isinstance(mount_id, str):
        response["mount_id"] = mount_id
    filesystem_id = value.get("filesystem_id")
    if isinstance(filesystem_id, str):
        response["filesystem_id"] = filesystem_id
    snapshots = value.get("snapshots")
    if isinstance(snapshots, list):
        decoded_snapshots: list[object] = []
        decoded_snapshots.extend(snapshots)
        response["snapshots"] = decoded_snapshots
    return response


def _worker_root_payload(value: object) -> _WorkerRootPayload:
    """Decode one sealed root witness before worker inspection begins."""
    if not isinstance(value, dict):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    identity = value.get("identity")
    mount_id = value.get("mount_id")
    filesystem_id = value.get("filesystem_id")
    if (
        not isinstance(identity, list)
        or len(identity) != 2
        or not all(type(item) is int for item in identity)
        or not isinstance(mount_id, str)
        or not isinstance(filesystem_id, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return {
        "identity": [identity[0], identity[1]],
        "mount_id": mount_id,
        "filesystem_id": filesystem_id,
    }


def _worker_response(payload: object) -> _WorkerResponsePayload:
    """Run one worker operation and project only typed snapshot facts."""
    try:
        profile = _worker_profile(payload)
        assert isinstance(payload, dict)
        operation = payload.get("operation")
        if operation == "witness":
            witness = _worker_capture_root(profile)
            return {
                "identity": [witness.identity.device, witness.identity.inode],
                "mount_id": witness.mount_id,
                "filesystem_id": witness.filesystem_id,
            }
        if operation != "inspect":
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        raw_paths = payload.get("paths")
        if not isinstance(raw_paths, list):
            raise TargetInspectionError(TargetErrorCode.PATH_DENIED)
        paths: list[str] = []
        for path in raw_paths:
            if not isinstance(path, str):
                raise TargetInspectionError(TargetErrorCode.PATH_DENIED)
            paths.append(path)
        expected_root = _worker_root(payload.get("expected_root"))
        snapshots = _inspect_many(
            profile,
            tuple(LogicalPath(path) for path in paths),
            expected_root,
        )
        encoded_snapshots: list[object] = []
        encoded_snapshots.extend(
            _snapshot_to_worker(item) for item in snapshots
        )
        return {"snapshots": encoded_snapshots}
    except TargetInspectionError as exc:
        return {"error": exc.code.value}


def _worker_profile(payload: object) -> _WorkerInspectionProfile:
    """Decode the minimal trusted worker configuration from its parent."""
    if not isinstance(payload, dict):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    root = payload.get("root")
    cwd = payload.get("cwd")
    maximum = payload.get("maximum")
    aggregate_maximum = payload.get("aggregate_maximum")
    encoded_authority = payload.get("authority_signature")
    if (
        not isinstance(root, str)
        or not Path(root).is_absolute()
        or cwd is not None
        and not isinstance(cwd, str)
        or type(maximum) is not int
        or maximum <= 0
        or type(aggregate_maximum) is not int
        or aggregate_maximum <= 0
        or not isinstance(encoded_authority, str)
    ):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    try:
        authority = _RuntimeTargetAuthority(
            b64decode(encoded_authority, validate=True)
        )
    except (TargetInspectionError, ValueError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.WORKER_UNAVAILABLE
        ) from exc
    root_path = Path(root)
    if not _authority_allows_root(authority, root_path):
        raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
    return _WorkerInspectionProfile(
        root_path,
        LogicalPath(cwd) if cwd is not None else None,
        maximum,
        aggregate_maximum,
    )


def _worker_root(value: object) -> RootWitness:
    """Decode the parent-sealed root witness for one worker operation."""
    payload = _worker_root_payload(value)
    identity = payload["identity"]
    return RootWitness(
        FileIdentity(identity[0], identity[1]),
        payload["mount_id"],
        payload["filesystem_id"],
    )


def _worker_capture_root(profile: _WorkerInspectionProfile) -> RootWitness:
    """Observe one worker-owned root descriptor without returning a path."""
    descriptor = _open_directory(profile.root_path)
    try:
        status = fstat(descriptor)
        return RootWitness(
            FileIdentity(status.st_dev, status.st_ino),
            _root_mount_id(descriptor, status),
            _filesystem_id(descriptor),
        )
    finally:
        close(descriptor)


def _snapshot_to_worker(snapshot: TargetSnapshot) -> _WorkerSnapshotPayload:
    """Encode one target snapshot onto the authenticated private channel."""
    parent: _WorkerSnapshotParentPayload = {
        "path": (
            snapshot.parent.path.value
            if snapshot.parent.path is not None
            else None
        ),
        "identity": [
            snapshot.parent.identity.device,
            snapshot.parent.identity.inode,
        ],
        "mount_id": snapshot.parent.mount_id,
    }
    if not snapshot.present:
        return {
            "path": snapshot.path.value,
            "present": False,
            "parent": parent,
        }
    assert (
        snapshot.bytes_value is not None
        and snapshot.metadata is not None
        and snapshot.identity is not None
    )
    return {
        "path": snapshot.path.value,
        "present": True,
        "bytes": b64encode(snapshot.bytes_value._value).decode(),
        "metadata": _WorkerSnapshotMetadataPayload(
            mode=snapshot.metadata.mode.value,
            has_bom=snapshot.metadata.has_utf8_bom,
            representation=snapshot.metadata.newline,
        ),
        "identity": [snapshot.identity.device, snapshot.identity.inode],
        "link_count": snapshot.link_count,
        "parent": parent,
        "classification": snapshot.security_metadata.value,
    }


def _filesystem_id(descriptor: int) -> str:
    """Return a target-native filesystem identity or fail closed."""
    return _mount_topology(descriptor).filesystem_id


def _mount_topology(descriptor: int) -> _MountTopology:
    """Read Darwin mount-table topology for one retained descriptor."""
    if platform != "darwin":
        raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED)
    try:
        libc = CDLL(None)
        fstatfs = libc.fstatfs
        fstatfs.argtypes = (c_int, POINTER(_DarwinStatFs))
        fstatfs.restype = c_int
        observation = _DarwinStatFs()
        if fstatfs(descriptor, byref(observation)) != 0:
            raise OSError("fstatfs failed")
    except (AttributeError, OSError) as exc:
        raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED) from exc
    filesystem_id = sha256(
        f"{observation.f_fsid[0]}:{observation.f_fsid[1]}".encode()
    ).hexdigest()
    topology = ":".join(
        (
            str(observation.f_fsid[0]),
            str(observation.f_fsid[1]),
            str(observation.f_type),
            str(observation.f_fssubtype),
            bytes(observation.f_fstypename).rstrip(b"\0").decode(),
            bytes(observation.f_mntonname).rstrip(b"\0").decode(),
            bytes(observation.f_mntfromname).rstrip(b"\0").decode(),
        )
    )
    return _MountTopology(sha256(topology.encode()).hexdigest(), filesystem_id)


def _root_mount_id(descriptor: int, status: object) -> str:
    """Bind a root witness to a platform-owned mount-table topology."""
    if not hasattr(status, "st_dev"):
        raise TargetInspectionError(TargetErrorCode.MOUNT_DENIED)
    return _mount_topology(descriptor).mount_id


if __name__ == "__main__":
    raise SystemExit(_worker_main())
