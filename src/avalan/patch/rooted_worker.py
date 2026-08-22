"""Apply rooted mutation primitives without local target authority."""

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from errno import ENOSYS, EOPNOTSUPP, EXDEV
from hashlib import sha256
from os import (
    O_CLOEXEC,
    O_CREAT,
    O_DIRECTORY,
    O_EXCL,
    O_NOFOLLOW,
    O_RDONLY,
    O_RDWR,
    O_WRONLY,
    close,
    fchmod,
    fstat,
    fsync,
    link,
    open,
    readlink,
    replace,
    unlink,
)
from os import read as read_fd
from os import stat as stat_at
from os import write as write_fd
from pathlib import Path
from secrets import token_bytes
from stat import S_ISLNK, S_ISREG
from sys import platform as sys_platform
from typing import Callable

from cffi import FFI

from avalan.patch.coordinator import (
    ArtifactJournal,
    JournalStep,
    SealedCommitCommand,
    SettlementJournal,
    WorkerReport,
    WorkerState,
)
from avalan.patch.domain import (
    ArtifactState,
    Capability,
    CommitStepState,
    FileMode,
    LogicalPath,
    MetadataProfile,
    PatchLineageId,
    PatchPlanId,
    PatchStepId,
    PostconditionState,
    SourceBytes,
)
from avalan.patch.planner import PlannedFile, PlannedLineage
from avalan.patch.target import (
    FileIdentity as FileIdentity,
)
from avalan.patch.target import (
    RootWitness as RootWitness,
)
from avalan.patch.target import (
    TargetErrorCode as TargetErrorCode,
)
from avalan.patch.target import (
    TargetInspectionError as TargetInspectionError,
)
from avalan.patch.target import (
    TargetSnapshot,
    _capture_protected_metadata,
    _filesystem_id,
    _inspect_many,
    _namespace_mount_binding,
    _open_child_directory,
    _open_cwd,
    _open_directory,
    _probe_metadata_round_trip,
    _ProtectedMetadata,
    _restore_protected_metadata,
    _root_mount_id,
    _snapshot_representation,
    _snapshot_to_worker,
    _WorkerInspectionProfile,
)

_F_GETPATH = 50
_PATH_MAX = 1_024
_ROOT_DESCRIPTOR: ContextVar[int | None] = ContextVar(
    "rooted_worker_root_descriptor", default=None
)
_PARENT_IDENTITIES: ContextVar[dict[LogicalPath | None, FileIdentity]] = (
    ContextVar("rooted_worker_parent_identities", default={})
)
_COMMIT_CONTEXT: ContextVar["_CommitContext | None"] = ContextVar(
    "rooted_worker_commit_context", default=None
)
_ROOTED_BARRIER: ContextVar[Callable[[str], None] | None] = ContextVar(
    "rooted_worker_barrier", default=None
)
_CFFI = FFI()
_CFFI.cdef("int fcntl(int, int, ...);")
_LIBC = _CFFI.dlopen(None)


def _barrier(stage: str) -> None:
    """Invoke the selected fixed worker barrier when one is configured."""
    callback = _ROOTED_BARRIER.get()
    if callback is not None:
        callback(stage)


@dataclass(frozen=True, slots=True)
class RootedMutationProfile:
    """Carry only rooted filesystem facts needed by mutation primitives."""

    root_path: Path
    cwd: LogicalPath | None
    creation_mode: FileMode


@dataclass(frozen=True, slots=True)
class RootedInspectionProfile:
    """Carry only rooted filesystem facts needed by inspection primitives."""

    root_path: Path
    cwd: LogicalPath | None
    max_snapshot_bytes: int
    max_aggregate_snapshot_bytes: int


def capture_rooted_root(path: Path) -> RootWitness:
    """Capture the immutable identity of one selected worker root."""
    root, _mount_binding = capture_rooted_root_binding(path)
    return root


def capture_rooted_root_binding(path: Path) -> tuple[RootWitness, str]:
    """Capture one worker root and its opaque local mount binding."""
    descriptor = _open_directory(path)
    try:
        status = fstat(descriptor)
        return (
            RootWitness(
                FileIdentity(status.st_dev, status.st_ino),
                _root_mount_id(descriptor, status),
                _filesystem_id(descriptor),
            ),
            _namespace_mount_binding(descriptor),
        )
    finally:
        close(descriptor)


def probe_rooted_metadata(
    workspace: Path,
    root: RootWitness,
    mount_binding: str | None = None,
) -> str:
    """Probe metadata through the authenticated selected workspace.

    Retain the workspace descriptor for every probe operation.
    """
    root_descriptor = _open_directory(workspace)
    workspace_descriptor: int | None = None
    name = ".avalan-patch-metadata-" + sha256(token_bytes(32)).hexdigest()[:24]
    descriptor: int | None = None
    created = False
    try:
        _validate_rooted_witness(root_descriptor, root)
        _validate_rooted_mount_binding(root_descriptor, mount_binding)
        workspace_descriptor = open(
            ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC, dir_fd=root_descriptor
        )
        _validate_rooted_witness(workspace_descriptor, root)
        _validate_rooted_mount_binding(workspace_descriptor, mount_binding)
        descriptor = open(
            name,
            O_CREAT | O_EXCL | O_NOFOLLOW | O_RDWR | O_CLOEXEC,
            0o600,
            dir_fd=workspace_descriptor,
        )
        created = True
        if write_fd(descriptor, b"metadata-probe\n") != len(
            b"metadata-probe\n"
        ):
            raise OSError("metadata probe write stalled")
        fchmod(descriptor, 0o600)
        fsync(descriptor)
        _probe_metadata_round_trip(descriptor)
        metadata = _capture_protected_metadata(descriptor)
        fsync(descriptor)
        return sha256(
            b"rooted-metadata-probe-v1:" + metadata.digest().value.encode()
        ).hexdigest()
    finally:
        try:
            if descriptor is not None:
                close(descriptor)
        finally:
            try:
                if created:
                    assert workspace_descriptor is not None
                    try:
                        unlink(name, dir_fd=workspace_descriptor)
                        fsync(workspace_descriptor)
                    except FileNotFoundError:
                        raise OSError(
                            "metadata probe artifact disappeared"
                        ) from None
            finally:
                try:
                    if workspace_descriptor is not None:
                        _validate_rooted_witness(workspace_descriptor, root)
                        _validate_rooted_mount_binding(
                            workspace_descriptor, mount_binding
                        )
                finally:
                    try:
                        if workspace_descriptor is not None:
                            close(workspace_descriptor)
                    finally:
                        close(root_descriptor)


def _validate_rooted_witness(descriptor: int, expected: RootWitness) -> None:
    """Require a retained descriptor to remain the authenticated workspace."""
    status = fstat(descriptor)
    observed = RootWitness(
        FileIdentity(status.st_dev, status.st_ino),
        _root_mount_id(descriptor, status),
        _filesystem_id(descriptor),
    )
    if observed != expected:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


def _validate_rooted_mount_binding(
    descriptor: int, expected: str | None
) -> None:
    """Require one descriptor to retain its child-local mount binding."""
    if (
        expected is not None
        and _namespace_mount_binding(descriptor) != expected
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


def validate_rooted_root_binding(
    path: Path, root: RootWitness, mount_binding: str
) -> None:
    """Revalidate the root witness and its process-local mount binding."""
    descriptor = _open_directory(path)
    try:
        _validate_rooted_witness(descriptor, root)
        _validate_rooted_mount_binding(descriptor, mount_binding)
    finally:
        close(descriptor)


def inspect_rooted(
    profile: RootedInspectionProfile,
    paths: tuple[LogicalPath, ...],
    expected_root: RootWitness,
    mount_binding: str | None = None,
) -> tuple[TargetSnapshot, ...]:
    """Inspect exact paths through neutral rooted worker primitives."""
    descriptor = _open_directory(profile.root_path)
    try:
        _validate_rooted_witness(descriptor, expected_root)
        _validate_rooted_mount_binding(descriptor, mount_binding)
    finally:
        close(descriptor)
    worker_profile = _WorkerInspectionProfile(
        profile.root_path,
        profile.cwd,
        profile.max_snapshot_bytes,
        profile.max_aggregate_snapshot_bytes,
    )
    try:
        return _inspect_many(worker_profile, paths, expected_root)
    finally:
        descriptor = _open_directory(profile.root_path)
        try:
            _validate_rooted_witness(descriptor, expected_root)
            _validate_rooted_mount_binding(descriptor, mount_binding)
        finally:
            close(descriptor)


def rooted_snapshot_payload(snapshot: TargetSnapshot) -> Mapping[str, object]:
    """Encode one rooted snapshot for the authenticated worker channel."""
    return _snapshot_to_worker(snapshot)


class _ArtifactUncertainError(OSError):
    """Mark a target-private staging artifact whose cleanup is unobservable."""


@dataclass(frozen=True, slots=True)
class RootedMutationCommand:
    """Carry the closed target-owned subset of one sealed patch plan.

    Sandbox workers decode this value from the canonical authenticated wire
    schema.  It deliberately has no policy, approval, or host service handles;
    those facts remain bound by the separately authenticated command envelope.
    """

    plan_id: PatchPlanId
    lineages: tuple[PlannedLineage, ...]
    effects: frozenset[Capability]

    def __post_init__(self) -> None:
        """Require a complete nonempty immutable target transaction."""
        if (
            type(self.plan_id) is not PatchPlanId
            or type(self.lineages) is not tuple
            or not self.lineages
            or any(type(item) is not PlannedLineage for item in self.lineages)
            or type(self.effects) is not frozenset
            or any(type(item) is not Capability for item in self.effects)
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class _CommitContext:
    """Retain the descriptors and identities needed at each native effect."""

    root_fd: int
    cwd_fd: int
    cwd_identity: FileIdentity
    root: RootWitness
    root_path: Path
    mount_binding: str | None = None


def _steps(
    command: SealedCommitCommand | RootedMutationCommand,
) -> tuple[tuple[PatchStepId, PatchLineageId], ...]:
    """Derive exact sealed commit IDs without a mutable target decision."""
    return tuple(
        (
            PatchStepId(
                "step_"
                + sha256(
                    (
                        _command_plan_id(command).value
                        + ":"
                        + lineage.lineage_id.value
                        + ":"
                        + str(index)
                        + ":"
                        + operation
                    ).encode()
                ).hexdigest()[:32]
            ),
            lineage.lineage_id,
        )
        for lineage in _command_lineages(command)
        for index, operation in enumerate(lineage.step_graph, start=1)
    )


def _artifacts(
    command: SealedCommitCommand | RootedMutationCommand,
) -> tuple[str, ...]:
    """Return target-owned artifact journals in immutable lineage order."""
    return tuple(
        "artifact:" + item.lineage_id.value
        for item in _command_lineages(command)
    )


def _command_plan_id(
    command: SealedCommitCommand | RootedMutationCommand,
) -> PatchPlanId:
    """Return the exact plan identifier for either trusted command shape."""
    if isinstance(command, RootedMutationCommand):
        return command.plan_id
    return command.plan.plan_id


def _command_lineages(
    command: SealedCommitCommand | RootedMutationCommand,
) -> tuple[PlannedLineage, ...]:
    """Return the immutable target lineages for one trusted command."""
    if isinstance(command, RootedMutationCommand):
        return command.lineages
    return command.plan.candidate.lineages


def _command_effects(
    command: SealedCommitCommand | RootedMutationCommand,
) -> frozenset[Capability]:
    """Return the policy-authorized effects bound to the command."""
    if isinstance(command, RootedMutationCommand):
        return command.effects
    return command.plan.binding.final.effects


def _failed_report(
    command: SealedCommitCommand | RootedMutationCommand,
    state: CommitStepState,
) -> WorkerReport:
    """Return no-effect or uncertain truth without inventing a commit."""
    return _report(
        _steps(command),
        [state for _ in _steps(command)],
        _artifacts(command),
        [ArtifactState.ABSENT for _ in _artifacts(command)],
        PostconditionState.UNKNOWN,
    )


def _commit_rooted(
    command: SealedCommitCommand | RootedMutationCommand,
    profile: RootedMutationProfile,
    witness: RootWitness,
    fence_check: Callable[[], None] | None = None,
    barrier: Callable[[str], None] | None = None,
    mount_binding: str | None = None,
) -> WorkerReport:
    """Apply one transaction with a scoped fixed-boundary callback."""
    token = _ROOTED_BARRIER.set(barrier)
    try:
        return _commit_rooted_impl(
            command, profile, witness, fence_check, mount_binding
        )
    finally:
        _ROOTED_BARRIER.reset(token)


def _commit_rooted_impl(
    command: SealedCommitCommand | RootedMutationCommand,
    profile: RootedMutationProfile,
    witness: RootWitness,
    fence_check: Callable[[], None] | None = None,
    mount_binding: str | None = None,
) -> WorkerReport:
    """Use retained root and parent descriptors for every write primitive."""
    steps = _steps(command)
    states = [CommitStepState.NOT_COMMITTED for _ in steps]
    artifacts = _artifacts(command)
    artifact_states = [ArtifactState.ABSENT for _ in artifacts]
    root_fd = _open_directory(profile.root_path)
    _barrier("target.open_handle")
    root_token = _ROOT_DESCRIPTOR.set(root_fd)
    parent_token = _PARENT_IDENTITIES.set(
        {
            path: FileIdentity(identity[0], identity[1])
            for lineage in _command_lineages(command)
            for path, identity in lineage.parent_identities
        }
    )
    try:
        status = fstat(root_fd)
        _validate_rooted_mount_binding(root_fd, mount_binding)
        current = RootWitness(
            FileIdentity(status.st_dev, status.st_ino),
            _root_mount_id(root_fd, status),
            _filesystem_id(root_fd),
        )
        if current != witness:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        cwd_fd, cwd_identity = _open_cwd(
            root_fd, profile.cwd, current.filesystem_id, current.mount_id
        )
        context_token = _COMMIT_CONTEXT.set(
            _CommitContext(
                root_fd,
                cwd_fd,
                cwd_identity,
                current,
                profile.root_path,
                mount_binding,
            )
        )
        try:
            offset = 0
            for index, lineage in enumerate(_command_lineages(command)):
                if any(
                    path is not None and "/" in path.value
                    for path in (
                        lineage.source_path,
                        lineage.destination_path,
                    )
                ):
                    raise TargetInspectionError(
                        TargetErrorCode.CAPABILITY_UNAVAILABLE
                    )
                count = len(lineage.step_graph)
                indices = tuple(range(offset, offset + count))
                offset += count
                try:
                    _commit_lineage(
                        cwd_fd,
                        cwd_identity,
                        current,
                        profile,
                        _command_effects(command),
                        lineage,
                        indices,
                        states,
                        index,
                        artifact_states,
                        fence_check,
                    )
                except TargetInspectionError:
                    return _report(
                        steps,
                        states,
                        artifacts,
                        artifact_states,
                        PostconditionState.UNKNOWN,
                    )
                except OSError:
                    unknown = next(
                        (
                            item
                            for item in indices
                            if states[item] is CommitStepState.NOT_COMMITTED
                        ),
                        indices[-1],
                    )
                    states[unknown] = CommitStepState.UNKNOWN
                    return _report(
                        steps,
                        states,
                        artifacts,
                        artifact_states,
                        PostconditionState.UNKNOWN,
                    )
            if offset != len(steps):
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                )
            try:
                _barrier("verification.before")
                postcondition = _verify(command, cwd_fd, cwd_identity, current)
            except OSError:
                postcondition = PostconditionState.UNKNOWN
            return _report(
                steps, states, artifacts, artifact_states, postcondition
            )
        finally:
            _COMMIT_CONTEXT.reset(context_token)
            close(cwd_fd)
    finally:
        try:
            _validate_rooted_mount_binding(root_fd, mount_binding)
        finally:
            try:
                _PARENT_IDENTITIES.reset(parent_token)
                _ROOT_DESCRIPTOR.reset(root_token)
            finally:
                close(root_fd)


def _report(
    steps: tuple[tuple[PatchStepId, PatchLineageId], ...],
    states: list[CommitStepState],
    artifacts: tuple[str, ...],
    artifact_states: list[ArtifactState],
    postcondition: PostconditionState,
) -> WorkerReport:
    """Freeze exact step and artifact truth for coordinator settlement."""
    return WorkerReport(
        WorkerState.SETTLED,
        SettlementJournal(
            tuple(
                JournalStep(identifier, lineage, state)
                for (identifier, lineage), state in zip(
                    steps, states, strict=True
                )
            ),
            tuple(
                ArtifactJournal(identifier, state)
                for identifier, state in zip(
                    artifacts, artifact_states, strict=True
                )
            ),
            postcondition,
        ),
    )


def _parent(
    cwd_fd: int,
    cwd_identity: FileIdentity,
    root: RootWitness,
    path: LogicalPath,
) -> tuple[int, str]:
    """Open a no-follow parent from the retained cwd descriptor."""
    status = fstat(cwd_fd)
    if (
        status.st_dev != cwd_identity.device
        or status.st_ino != cwd_identity.inode
        or _filesystem_id(cwd_fd) != root.filesystem_id
        or _root_mount_id(cwd_fd, status) != root.mount_id
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    descriptor = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC, dir_fd=cwd_fd)
    try:
        for part in path.value.split("/")[:-1]:
            next_descriptor = _open_child_directory(
                descriptor,
                part,
                status.st_dev,
                root.filesystem_id,
                root.mount_id,
            )
            close(descriptor)
            descriptor = next_descriptor
        expected = _PARENT_IDENTITIES.get().get(
            LogicalPath("/".join(path.value.split("/")[:-1]))
            if "/" in path.value
            else None
        )
        current = fstat(descriptor)
        if expected is not None and (
            current.st_dev != expected.device
            or current.st_ino != expected.inode
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        root_descriptor = _ROOT_DESCRIPTOR.get()
        if root_descriptor is None or not _is_contained(
            root_descriptor, descriptor
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        _barrier("parent_opened")
        if not _is_contained(
            root_descriptor, descriptor
        ) or not _rebind_parent(cwd_fd, cwd_identity, root, path, descriptor):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        return descriptor, path.value.rsplit("/", 1)[-1]
    except BaseException:
        close(descriptor)
        raise


def _expected(
    parent_fd: int,
    leaf: str,
    expected: PlannedFile,
) -> tuple[int, _ProtectedMetadata]:
    """Prove the exact regular-file before or postcondition at one barrier."""
    if (
        not expected.present
        or expected.bytes_value is None
        or expected.metadata is None
        or expected.digest is None
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    try:
        status = stat_at(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError as exc:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
    if (
        S_ISLNK(status.st_mode)
        or not S_ISREG(status.st_mode)
        or status.st_nlink != 1
        or status.st_size != expected.size.value
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    descriptor = open(
        leaf, O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd
    )
    try:
        _barrier("target.open_handle")
        opened = fstat(descriptor)
        value = _read_exact(descriptor, expected.size.value)
        bom, newline = _snapshot_representation(value)
        metadata = MetadataProfile(
            FileMode(opened.st_mode & 0o777), bom, newline
        )
        protected_metadata = _capture_protected_metadata(descriptor)
        if (
            opened.st_nlink != 1
            or expected.identity is not None
            and (
                opened.st_dev != expected.identity[0]
                or opened.st_ino != expected.identity[1]
            )
            or value != expected.bytes_value._value
            or metadata != expected.metadata
            or SourceBytes(value).digest() != expected.digest
            or expected.protected_metadata is not None
            and protected_metadata.digest() != expected.protected_metadata
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    except BaseException:
        _barrier("target.close_handle")
        close(descriptor)
        raise
    _barrier("target.close_handle")
    return descriptor, protected_metadata


def _is_contained(root_fd: int, descriptor: int) -> bool:
    """Return whether a live descriptor remains beneath the retained root."""
    root_path = _descriptor_path(root_fd)
    descriptor_path = _descriptor_path(descriptor)
    return descriptor_path == root_path or root_path in descriptor_path.parents


def _descriptor_path(descriptor: int) -> Path:
    """Return the kernel path for one retained platform descriptor."""
    if sys_platform == "linux":
        try:
            value = readlink(f"/proc/self/fd/{descriptor}")
        except OSError as error:
            raise TargetInspectionError(
                TargetErrorCode.WITNESS_STALE
            ) from error
        if not value.startswith("/") or value.endswith(" (deleted)"):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        return Path(value)
    if sys_platform != "darwin":
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    buffer = _CFFI.new("char[]", _PATH_MAX)
    if _LIBC.fcntl(descriptor, _F_GETPATH, buffer) != 0:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    value = _CFFI.string(buffer).decode("utf-8", "strict")
    if not value:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return Path(value)


def _rebind_parent(
    cwd_fd: int,
    cwd_identity: FileIdentity,
    root: RootWitness,
    path: LogicalPath,
    descriptor: int,
) -> bool:
    """Prove the opened parent is still reachable through the retained root."""
    reopened = open(".", O_RDONLY | O_DIRECTORY | O_CLOEXEC, dir_fd=cwd_fd)
    try:
        for part in path.value.split("/")[:-1]:
            next_descriptor = _open_child_directory(
                reopened,
                part,
                cwd_identity.device,
                root.filesystem_id,
                root.mount_id,
            )
            close(reopened)
            reopened = next_descriptor
        observed = fstat(descriptor)
        current = fstat(reopened)
        return (
            observed.st_dev == current.st_dev
            and observed.st_ino == current.st_ino
        )
    except TargetInspectionError:
        return False
    finally:
        close(reopened)


def _namespace_effect(
    parent_fd: int,
    path: LogicalPath,
    effect: Callable[[], None],
    *,
    entries: tuple[tuple[int, LogicalPath, str, int], ...] = (),
    fence_check: Callable[[], None] | None = None,
) -> None:
    """Revalidate rooted handles immediately before one namespace syscall."""
    _before_namespace_effect(
        parent_fd,
        path,
        entries=entries,
        fence_check=fence_check,
    )
    effect()


def _before_namespace_effect(
    parent_fd: int,
    path: LogicalPath,
    *,
    entries: tuple[tuple[int, LogicalPath, str, int], ...] = (),
    fence_check: Callable[[], None] | None = None,
) -> None:
    """Revalidate root and fence immediately before a namespace syscall."""
    _barrier("target.namespace_before_final_check")
    _validate_namespace_context(
        parent_fd,
        path,
        entries,
    )
    _barrier("target.namespace_after_final_check")
    _barrier("target.namespace_before_effect")
    _validate_namespace_context(
        parent_fd,
        path,
        entries,
    )
    _require_effect_fence(fence_check)


def _validate_namespace_context(
    parent_fd: int,
    path: LogicalPath,
    entries: tuple[tuple[int, LogicalPath, str, int], ...],
) -> None:
    """Prove all rooted parent and source identities remain live and sealed."""
    context = _COMMIT_CONTEXT.get()
    if context is None:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    _validate_context_root_binding(context)
    root_status = fstat(context.root_fd)
    cwd_status = fstat(context.cwd_fd)
    try:
        configured_root = stat_at(context.root_path, follow_symlinks=False)
    except OSError as exc:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
    if (
        root_status.st_dev != context.root.identity.device
        or root_status.st_ino != context.root.identity.inode
        or configured_root.st_dev != root_status.st_dev
        or configured_root.st_ino != root_status.st_ino
        or _filesystem_id(context.root_fd) != context.root.filesystem_id
        or _root_mount_id(context.root_fd, root_status)
        != context.root.mount_id
        or cwd_status.st_dev != context.cwd_identity.device
        or cwd_status.st_ino != context.cwd_identity.inode
        or not _is_contained(context.root_fd, context.cwd_fd)
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    _validate_parent_context(parent_fd, path, context)
    for (
        source_parent_fd,
        source_path,
        source_leaf,
        source_descriptor,
    ) in entries:
        _validate_parent_context(source_parent_fd, source_path, context)
        source_status = fstat(source_descriptor)
        try:
            entry_status = stat_at(
                source_leaf,
                dir_fd=source_parent_fd,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
        if (
            not S_ISREG(source_status.st_mode)
            or source_status.st_dev != entry_status.st_dev
            or source_status.st_ino != entry_status.st_ino
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


def _validate_context_root_binding(context: _CommitContext) -> None:
    """Reopen and validate the configured root at a final effect barrier."""
    descriptor = _open_directory(context.root_path)
    try:
        _validate_rooted_witness(descriptor, context.root)
        _validate_rooted_mount_binding(descriptor, context.mount_binding)
    finally:
        close(descriptor)


def _validate_parent_context(
    parent_fd: int, path: LogicalPath, context: _CommitContext
) -> None:
    """Prove an opened parent is the planned reachable directory now."""
    parent_path = (
        LogicalPath("/".join(path.value.split("/")[:-1]))
        if "/" in path.value
        else None
    )
    expected = _PARENT_IDENTITIES.get().get(parent_path)
    status = fstat(parent_fd)
    if (
        expected is None
        or status.st_dev != expected.device
        or status.st_ino != expected.inode
        or not _is_contained(context.root_fd, parent_fd)
        or not _rebind_parent(
            context.cwd_fd,
            context.cwd_identity,
            context.root,
            path,
            parent_fd,
        )
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


def _read_exact(descriptor: int, maximum: int) -> bytes:
    """Read a planned bounded byte value and reject length-changing races."""
    parts: list[bytes] = []
    remaining = maximum + 1
    while remaining:
        value = read_fd(descriptor, remaining)
        if not value:
            break
        parts.append(value)
        remaining -= len(value)
    result = b"".join(parts)
    if len(result) != maximum:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return result


def _absent(parent_fd: int, leaf: str) -> None:
    """Prove immediate absence before a link-based no-replace publication."""
    try:
        stat_at(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)


def _stage(
    parent_fd: int,
    value: bytes,
    mode: int,
    protected_metadata: _ProtectedMetadata | None = None,
    *,
    path: LogicalPath,
    fence_check: Callable[[], None] | None = None,
) -> str:
    """Create complete collision-safe directory-local private staging bytes."""
    _barrier("target.stage_artifact")
    _barrier("artifact.stage")
    for _ in range(32):
        name = ".avalan-patch-" + sha256(token_bytes(32)).hexdigest()[:32]
        try:
            _before_namespace_effect(
                parent_fd,
                path,
                fence_check=fence_check,
            )
            descriptor = open(
                name,
                O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY | O_CLOEXEC,
                mode,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        try:
            offset = 0
            while offset < len(value):
                _barrier("artifact.stage_write_before")
                count = write_fd(descriptor, value[offset:])
                if count <= 0:
                    raise OSError("staging write stalled")
                offset += count
            fchmod(descriptor, mode)
            if protected_metadata is not None:
                _restore_protected_metadata(descriptor, protected_metadata)
            fsync(descriptor)
        except BaseException:
            try:
                _barrier("artifact.stage_cleanup_before")
                _namespace_effect(
                    parent_fd,
                    path,
                    lambda: unlink(name, dir_fd=parent_fd),
                    entries=((parent_fd, path, name, descriptor),),
                    fence_check=fence_check,
                )
            except BaseException as cleanup_error:
                close(descriptor)
                raise _ArtifactUncertainError from cleanup_error
            close(descriptor)
            raise
        close(descriptor)
        return name
    raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


def _publish_new(
    parent_fd: int,
    path: LogicalPath,
    leaf: str,
    value: bytes,
    mode: int,
    artifact_states: list[ArtifactState],
    artifact_index: int,
    fence_check: Callable[[], None] | None = None,
    protected_metadata: _ProtectedMetadata | None = None,
) -> None:
    """Publish a complete file through atomic no-replace linking."""
    try:
        stage = _stage(
            parent_fd,
            value,
            mode,
            protected_metadata,
            path=path,
            fence_check=fence_check,
        )
    except _ArtifactUncertainError:
        artifact_states[artifact_index] = ArtifactState.UNKNOWN
        raise
    descriptor = open(
        stage, O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd
    )
    try:
        _barrier("publication.before_link")
        _namespace_effect(
            parent_fd,
            path,
            lambda: link(
                stage,
                leaf,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            ),
            entries=((parent_fd, path, stage, descriptor),),
            fence_check=fence_check,
        )
    except FileExistsError as exc:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
    except OSError as exc:
        if exc.errno in {ENOSYS, EOPNOTSUPP, EXDEV}:
            raise TargetInspectionError(
                TargetErrorCode.CAPABILITY_UNAVAILABLE
            ) from exc
        raise
    finally:
        try:
            _barrier("artifact.cleanup_before")
            _namespace_effect(
                parent_fd,
                path,
                lambda: unlink(stage, dir_fd=parent_fd),
                entries=((parent_fd, path, stage, descriptor),),
                fence_check=fence_check,
            )
        except BaseException as cleanup_error:
            artifact_states[artifact_index] = ArtifactState.LEAKED
            if not isinstance(cleanup_error, OSError):
                raise
        else:
            artifact_states[artifact_index] = ArtifactState.CLEANED
        close(descriptor)


def _publish_update(
    parent_fd: int,
    path: LogicalPath,
    leaf: str,
    value: bytes,
    mode: int,
    protected_metadata: _ProtectedMetadata,
    artifact_states: list[ArtifactState],
    artifact_index: int,
    expected_descriptor: int,
    fence_check: Callable[[], None] | None = None,
) -> None:
    """Atomically replace one staged regular file without truncation."""
    try:
        stage = _stage(
            parent_fd,
            value,
            mode,
            protected_metadata,
            path=path,
            fence_check=fence_check,
        )
    except _ArtifactUncertainError:
        artifact_states[artifact_index] = ArtifactState.UNKNOWN
        raise
    descriptor = open(
        stage, O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd
    )
    try:
        _namespace_effect(
            parent_fd,
            path,
            lambda: replace(
                stage, leaf, src_dir_fd=parent_fd, dst_dir_fd=parent_fd
            ),
            entries=(
                (parent_fd, path, stage, descriptor),
                (parent_fd, path, leaf, expected_descriptor),
            ),
            fence_check=fence_check,
        )
    except BaseException:
        try:
            _namespace_effect(
                parent_fd,
                path,
                lambda: unlink(stage, dir_fd=parent_fd),
                entries=((parent_fd, path, stage, descriptor),),
                fence_check=fence_check,
            )
        except BaseException as cleanup_error:
            artifact_states[artifact_index] = ArtifactState.LEAKED
            if not isinstance(cleanup_error, OSError):
                raise
        raise
    finally:
        close(descriptor)
    artifact_states[artifact_index] = ArtifactState.CLEANED


def _require_effect_fence(check: Callable[[], None] | None) -> None:
    """Require a current owner immediately before a requested effect."""
    if check is not None:
        check()


def _commit_lineage(
    cwd_fd: int,
    cwd_identity: FileIdentity,
    root: RootWitness,
    profile: RootedMutationProfile,
    authorized_effects: frozenset[Capability],
    lineage: PlannedLineage,
    indices: tuple[int, ...],
    states: list[CommitStepState],
    artifact_index: int,
    artifact_states: list[ArtifactState],
    fence_check: Callable[[], None] | None,
) -> None:
    """Execute one declared terminal lineage in sealed deterministic order."""
    initial = lineage.initial
    final = lineage.final
    source = lineage.source_path
    destination = lineage.destination_path
    if source is None and destination is not None and final.present:
        assert final.bytes_value is not None
        parent, leaf = _parent(cwd_fd, cwd_identity, root, destination)
        try:
            _absent(parent, leaf)
            _barrier("requested_effect.step_before")
            _require_effect_fence(fence_check)
            artifact_states[artifact_index] = ArtifactState.STAGED
            _publish_new(
                parent,
                destination,
                leaf,
                final.bytes_value._value,
                profile.creation_mode.value,
                artifact_states,
                artifact_index,
                fence_check,
            )
            states[indices[0]] = CommitStepState.COMMITTED
        finally:
            close(parent)
        return
    if source is not None and destination is None and initial.present:
        parent, leaf = _parent(cwd_fd, cwd_identity, root, source)
        try:
            descriptor, _ = _expected(parent, leaf, initial)
            try:
                _require_effect_fence(fence_check)
                _namespace_effect(
                    parent,
                    source,
                    lambda: unlink(leaf, dir_fd=parent),
                    entries=((parent, source, leaf, descriptor),),
                    fence_check=fence_check,
                )
            finally:
                close(descriptor)
            states[indices[0]] = CommitStepState.COMMITTED
        finally:
            close(parent)
        return
    if (
        source is not None
        and destination is not None
        and source == destination
    ):
        assert final.bytes_value is not None and final.metadata is not None
        parent, leaf = _parent(cwd_fd, cwd_identity, root, source)
        try:
            descriptor, protected_metadata = _expected(parent, leaf, initial)
            try:
                mode = fstat(descriptor).st_mode
                if initial.protected_metadata is None:
                    raise TargetInspectionError(
                        TargetErrorCode.METADATA_DENIED
                    )
                if mode & 0o7000 or (
                    mode & 0o111
                    and Capability.UPDATE_EXECUTABLE not in authorized_effects
                ):
                    raise TargetInspectionError(
                        TargetErrorCode.METADATA_DENIED
                    )
                _barrier("requested_effect.step_before")
                _require_effect_fence(fence_check)
                artifact_states[artifact_index] = ArtifactState.STAGED
                _publish_update(
                    parent,
                    source,
                    leaf,
                    final.bytes_value._value,
                    final.metadata.mode.value,
                    protected_metadata,
                    artifact_states,
                    artifact_index,
                    descriptor,
                    fence_check,
                )
            finally:
                close(descriptor)
            states[indices[0]] = CommitStepState.COMMITTED
        finally:
            close(parent)
        return
    if (
        source is None
        or destination is None
        or not initial.present
        or not final.present
    ):
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    assert final.bytes_value is not None and final.metadata is not None
    source_parent, source_leaf = _parent(cwd_fd, cwd_identity, root, source)
    try:
        destination_parent, destination_leaf = _parent(
            cwd_fd, cwd_identity, root, destination
        )
        try:
            descriptor, protected_metadata = _expected(
                source_parent, source_leaf, initial
            )
            try:
                mode = fstat(descriptor).st_mode
                if initial.protected_metadata is None:
                    raise TargetInspectionError(
                        TargetErrorCode.METADATA_DENIED
                    )
                if mode & 0o7000 or (
                    mode & 0o111
                    and Capability.UPDATE_EXECUTABLE not in authorized_effects
                ):
                    raise TargetInspectionError(
                        TargetErrorCode.METADATA_DENIED
                    )
                _absent(destination_parent, destination_leaf)
                artifact_states[artifact_index] = ArtifactState.STAGED
                if (
                    initial.bytes_value is not None
                    and initial.bytes_value._value == final.bytes_value._value
                ):
                    try:
                        _barrier("publication.before_link")
                        _require_effect_fence(fence_check)
                        _namespace_effect(
                            destination_parent,
                            destination,
                            lambda: link(
                                source_leaf,
                                destination_leaf,
                                src_dir_fd=source_parent,
                                dst_dir_fd=destination_parent,
                                follow_symlinks=False,
                            ),
                            entries=(
                                (
                                    source_parent,
                                    source,
                                    source_leaf,
                                    descriptor,
                                ),
                            ),
                            fence_check=fence_check,
                        )
                    except OSError as exc:
                        if exc.errno in {ENOSYS, EOPNOTSUPP, EXDEV}:
                            raise TargetInspectionError(
                                TargetErrorCode.CAPABILITY_UNAVAILABLE
                            ) from exc
                        raise
                    artifact_states[artifact_index] = ArtifactState.ABSENT
                else:
                    _require_effect_fence(fence_check)
                    _publish_new(
                        destination_parent,
                        destination,
                        destination_leaf,
                        final.bytes_value._value,
                        final.metadata.mode.value,
                        artifact_states,
                        artifact_index,
                        fence_check,
                        protected_metadata,
                    )
                states[indices[0]] = CommitStepState.COMMITTED
                _barrier("move.source_remove_before")
                _require_effect_fence(fence_check)
                _namespace_effect(
                    source_parent,
                    source,
                    lambda: unlink(source_leaf, dir_fd=source_parent),
                    entries=(
                        (source_parent, source, source_leaf, descriptor),
                    ),
                    fence_check=fence_check,
                )
            finally:
                close(descriptor)
            states[indices[1]] = CommitStepState.COMMITTED
        finally:
            close(destination_parent)
    finally:
        close(source_parent)


def _verify(
    command: SealedCommitCommand | RootedMutationCommand,
    cwd_fd: int,
    cwd_identity: FileIdentity,
    root: RootWitness,
) -> PostconditionState:
    """Verify the final requested entries without executing workspace code."""
    for lineage in _command_lineages(command):
        final = lineage.final
        path = (
            lineage.destination_path if final.present else lineage.source_path
        )
        if path is None:
            return PostconditionState.UNKNOWN
        parent, leaf = _parent(cwd_fd, cwd_identity, root, path)
        try:
            if not final.present:
                try:
                    stat_at(leaf, dir_fd=parent, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                return PostconditionState.SUPERSEDED
            try:
                descriptor, _ = _expected(parent, leaf, final)
            except TargetInspectionError:
                return PostconditionState.SUPERSEDED
            close(descriptor)
        finally:
            close(parent)
    return PostconditionState.ESTABLISHED
