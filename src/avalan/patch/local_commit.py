"""Run test-profile local commits through rooted directory handles only.

This internal module is deliberately not imported by ``avalan.patch`` or the
inspection-worker bootstrap.  A caller reaches it only after a trusted local
scope completed the full capability handshake and the coordinator supplied a
sealed owner/fence command.
"""

from asyncio import (
    CancelledError,
    Task,
    create_subprocess_exec,
    create_task,
    shield,
    sleep,
    to_thread,
    wait_for,
)
from base64 import b64decode, b64encode
from contextvars import ContextVar
from dataclasses import dataclass, field
from errno import ENOSYS, EOPNOTSUPP, EXDEV
from hashlib import sha256
from hmac import compare_digest, digest
from json import dumps, loads
from os import (
    O_CLOEXEC,
    O_CREAT,
    O_DIRECTORY,
    O_EXCL,
    O_NOFOLLOW,
    O_RDONLY,
    O_WRONLY,
    close,
    environ,
    fchmod,
    fstat,
    fsync,
    getuid,
    link,
    open,
    replace,
    unlink,
)
from os import read as read_fd
from os import stat as stat_at
from os import write as write_fd
from pathlib import Path
from pickle import dumps as pickle_dumps
from pickle import loads as pickle_loads
from secrets import token_bytes
from stat import S_ISLNK, S_ISREG
from subprocess import PIPE
from sys import executable, stdin, stdout
from time import monotonic
from time import sleep as blocking_sleep
from typing import Callable, NewType, TypedDict

from cffi import FFI

from avalan.patch.coordinator import (
    ArtifactJournal,
    CoordinatorError,
    JournalStep,
    RootedLocalCommitWorker,
    SealedCommitCommand,
    SettlementJournal,
    WorkerReport,
    WorkerState,
    _rooted_local_worker,
)
from avalan.patch.domain import (
    ArtifactState,
    Capability,
    CommitStepState,
    FileMode,
    LogicalPath,
    MetadataProfile,
    PatchLineageId,
    PatchRequestId,
    PatchStepId,
    PostconditionState,
    SourceBytes,
)
from avalan.patch.planner import PlannedFile, PlannedLineage
from avalan.patch.target import (
    _FUTURE_MUTATION_PRIMITIVES,
    _SEATBELT_EXECUTABLE,
    _WORKER_TOKEN_ENV,
    FileIdentity,
    LocalPlatformProfile,
    LocalTargetProfile,
    ResolvedMutationScope,
    RootWitness,
    TargetErrorCode,
    TargetHandshake,
    TargetInspectionError,
    _capture_protected_metadata,
    _filesystem_id,
    _open_child_directory,
    _open_cwd,
    _open_directory,
    _ProtectedMetadata,
    _restore_protected_metadata,
    _root_mount_id,
    _seatbelt_string,
    _snapshot_representation,
    _worker_seatbelt_profile,
)

_F_GETPATH = 50
_PATH_MAX = 1_024
_ROOT_DESCRIPTOR: ContextVar[int | None] = ContextVar(
    "local_commit_root_descriptor", default=None
)
_PARENT_IDENTITIES: ContextVar[dict[LogicalPath | None, FileIdentity]] = (
    ContextVar("local_commit_parent_identities", default={})
)
_COMMIT_CONTEXT: ContextVar["_CommitContext | None"] = ContextVar(
    "local_commit_context", default=None
)
_CFFI = FFI()
_CFFI.cdef("int fcntl(int, int, ...);")
_LIBC = _CFFI.dlopen(None)
_SEATBELT_BARRIER_ENV = "AVALAN_PATCH_COMMIT_BARRIER"
_SEATBELT_RELEASE_ENV = "AVALAN_PATCH_COMMIT_RELEASE"
_SEATBELT_BARRIER_TIMEOUT_SECONDS = 2.0
_SEATBELT_RUNTIME_READ_PATHS = (Path("/opt/homebrew/opt/openssl@3/lib"),)
_SEATBELT_BOUNDARIES = frozenset(
    (
        "artifact.cleanup_before",
        "artifact.stage",
        "artifact.stage_cleanup_before",
        "artifact.stage_write_before",
        "move.source_remove_before",
        "publication.before_link",
        "parent_opened",
        "requested_effect.step_before",
        "target.close_handle",
        "target.namespace_after_final_check",
        "target.namespace_before_final_check",
        "target.namespace_before_effect",
        "target.open_handle",
        "target.stage_artifact",
        "verification.before",
    )
)
_SEATBELT_WORKER_SESSION: tuple[str, str, bytes] | None = None
_SEATBELT_WORKER_SEQUENCE = 0
SeatbeltPlanBinding = NewType("SeatbeltPlanBinding", str)
SeatbeltRequestBinding = NewType("SeatbeltRequestBinding", str)


class _SeatbeltCommitPayload(TypedDict):
    """Encode one authenticated commit worker request."""

    command: str
    cwd: str | None
    fence: int
    namespace: str
    plan_id: SeatbeltPlanBinding
    request_id: SeatbeltRequestBinding
    root: str
    version: int
    witness: "_SeatbeltWitnessPayload"


class _SeatbeltWitnessPayload(TypedDict):
    """Encode the immutable root witness passed to the worker."""

    device: int
    filesystem_id: str
    inode: int
    mount_id: str


class _SeatbeltCommitResponse(TypedDict):
    """Encode one authenticated commit worker response."""

    artifacts: list[dict[str, str]]
    postcondition: str
    state: str
    steps: list[dict[str, str]]


class _ArtifactUncertainError(OSError):
    """Mark a target-private staging artifact whose cleanup is unobservable."""


@dataclass(frozen=True, slots=True)
class _CommitContext:
    """Retain the descriptors and identities needed at each native effect."""

    root_fd: int
    cwd_fd: int
    cwd_identity: FileIdentity
    root: RootWitness
    root_path: Path


@dataclass(frozen=True, slots=True)
class _ParentHandle:
    """Bind one opened parent descriptor to its sealed logical parent path."""

    descriptor: int
    leaf: str
    path: LogicalPath | None


@dataclass(frozen=True, slots=True)
class _Channel:
    """Bind an internal rooted commit worker to one resolved scope."""

    target: "LocalCommitTarget"
    scope: ResolvedMutationScope
    _settlements: dict[PatchRequestId, Task[WorkerReport]] = field(
        default_factory=dict, compare=False
    )

    async def commit_local(self, command: SealedCommitCommand) -> WorkerReport:
        """Run once and retain its task across caller cancellation."""
        request_id = command.lease.request_id
        task = self._settlements.get(request_id)
        if task is None:
            task = create_task(self.target._commit(self.scope, command))
            self._settlements[request_id] = task
        return await shield(task)

    async def reconcile_local(
        self, request_id: PatchRequestId
    ) -> WorkerReport:
        """Await the owned task once without reissuing local effects."""
        task = self._settlements.get(request_id)
        if task is None:
            return WorkerReport(WorkerState.LIVE, None)
        try:
            return await wait_for(
                shield(task), _SEATBELT_BARRIER_TIMEOUT_SECONDS
            )
        except TimeoutError:
            return WorkerReport(WorkerState.LIVE, None)


@dataclass(frozen=True, slots=True)
class LocalCommitTarget:
    """Offer local mutation authority only in an explicit test profile."""

    profile: LocalTargetProfile

    def __post_init__(self) -> None:
        """Fail closed unless the runtime selected the test profile."""
        if (
            not self.profile.mutation_test_profile
            or self.profile.platform is not LocalPlatformProfile.DARWIN
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return the active local receipt after scope verification."""
        self._require_scope(scope)
        _commit_barrier("target.negotiate_capabilities")
        if not _FUTURE_MUTATION_PRIMITIVES.issubset(scope.primitives):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        return TargetHandshake(
            self.profile.identity,
            scope.primitives,
            (),
            scope.probes,
            self.profile.platform,
            worker=scope.worker,
        )

    async def worker(
        self, scope: ResolvedMutationScope
    ) -> RootedLocalCommitWorker:
        """Mint the coordinator worker after a complete handshake."""
        handshake = await self.handshake(scope)
        required = {
            Capability.CREATE,
            Capability.UPDATE,
            Capability.DELETE,
            Capability.MOVE,
        }
        if not required.issubset(handshake.advertised_operations()):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        return _rooted_local_worker(_Channel(self, scope))

    def _require_scope(self, scope: ResolvedMutationScope) -> None:
        """Reject a changed target, worker witness, or unauthenticated root."""
        if (
            scope.identity != self.profile.identity
            or scope.root_witness is None
            or scope.worker is None
            or scope._worker_authorization
            is not self.profile._worker_authorization
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)

    async def _commit(
        self, scope: ResolvedMutationScope, command: SealedCommitCommand
    ) -> WorkerReport:
        """Apply the exact sealed graph in the isolated Seatbelt worker."""
        try:
            self._require_scope(scope)
            if command.plan.binding.target != self.profile.identity:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            assert scope.root_witness is not None
            return await _commit_in_seatbelt(
                command, self.profile, scope.root_witness
            )
        except TargetInspectionError:
            return _failed_report(command, CommitStepState.NOT_COMMITTED)
        except OSError:
            return _failed_report(command, CommitStepState.UNKNOWN)


def _steps(
    command: SealedCommitCommand,
) -> tuple[tuple[PatchStepId, PatchLineageId], ...]:
    """Derive exact sealed commit IDs without a mutable target decision."""
    return tuple(
        (
            PatchStepId(
                "step_"
                + sha256(
                    (
                        command.plan.plan_id.value
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
        for lineage in command.plan.candidate.lineages
        for index, operation in enumerate(lineage.step_graph, start=1)
    )


def _artifacts(command: SealedCommitCommand) -> tuple[str, ...]:
    """Return target-owned artifact journals in immutable lineage order."""
    return tuple(
        "artifact:" + item.lineage_id.value
        for item in command.plan.candidate.lineages
    )


def _failed_report(
    command: SealedCommitCommand, state: CommitStepState
) -> WorkerReport:
    """Return no-effect or uncertain truth without inventing a commit."""
    return _report(
        _steps(command),
        [state for _ in _steps(command)],
        _artifacts(command),
        [ArtifactState.ABSENT for _ in _artifacts(command)],
        PostconditionState.UNKNOWN,
    )


@dataclass(frozen=True, slots=True)
class _SeatbeltRoot:
    """Keep the internal worker root path out of serialized patch state."""

    _path: Path


@dataclass(frozen=True, slots=True)
class _SeatbeltCommitProfile:
    """Carry only filesystem facts needed by the mutation subprocess."""

    root: _SeatbeltRoot
    cwd: LogicalPath | None
    creation_mode: FileMode


async def _commit_in_seatbelt(
    command: SealedCommitCommand,
    profile: LocalTargetProfile,
    witness: RootWitness,
) -> WorkerReport:
    """Execute one authenticated command in a Darwin write sandbox."""
    namespace = _commit_namespace(profile, witness)
    token = token_bytes(32)
    marker = namespace / (
        ".avalan-patch-barrier-" + sha256(token).hexdigest()[:32]
    )
    release = namespace / (
        ".avalan-patch-release-" + sha256(token + b"release").hexdigest()[:32]
    )
    payload: _SeatbeltCommitPayload = {
        "command": b64encode(pickle_dumps(command)).decode("ascii"),
        "cwd": profile.cwd.value if profile.cwd is not None else None,
        "fence": command.lease.fence,
        "namespace": str(namespace),
        "plan_id": SeatbeltPlanBinding(command.plan.plan_id.value),
        "request_id": SeatbeltRequestBinding(command.lease.request_id.value),
        "root": str(profile.root._path),
        "version": 1,
        "witness": {
            "device": witness.identity.device,
            "filesystem_id": witness.filesystem_id,
            "inode": witness.identity.inode,
            "mount_id": witness.mount_id,
        },
    }
    raw_payload = dumps(payload, separators=(",", ":")).encode()
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
        "-c",
        (
            "import sys\nsys.path.append(sys.argv[1])\n"
            "from avalan.patch.local_commit import _seatbelt_worker_main\n"
            "raise SystemExit(_seatbelt_worker_main())"
        ),
        str(Path(__file__).resolve().parents[2]),
    )
    try:
        process = await create_subprocess_exec(
            _SEATBELT_EXECUTABLE,
            "-p",
            _commit_seatbelt_profile(profile, namespace, token.hex()),
            "--",
            *worker_argv,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd="/",
            env={
                _SEATBELT_BARRIER_ENV: str(marker),
                _SEATBELT_RELEASE_ENV: str(release),
                _WORKER_TOKEN_ENV: token.hex(),
            },
            close_fds=True,
        )
    except OSError as exc:
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from exc
    relay = create_task(_relay_seatbelt_barriers(marker, release, token))
    relay_error: BaseException | None = None
    try:
        response_bytes, error_bytes = await process.communicate(message)
    except CancelledError:
        raise
    finally:
        relay.cancel()
        try:
            await relay
        except CancelledError:
            pass
        except BaseException as exc:
            relay_error = exc
        for path in (marker, release):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    if relay_error is not None:
        raise relay_error
    if process.returncode != 0:
        del error_bytes
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return _decode_seatbelt_response(command, token, response_bytes)


async def _relay_seatbelt_barriers(
    marker: Path, release: Path, token: bytes
) -> None:
    """Forward authenticated fixed worker boundaries to local test hooks."""
    last_sequence = 0
    last_value: str | None = None
    while True:
        value = await to_thread(_read_barrier_message, marker, token)
        if value is None:
            await sleep(0.001)
            continue
        sequence, stage = _barrier_stage(value)
        if sequence < last_sequence or (
            sequence == last_sequence and value != last_value
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        if sequence == last_sequence:
            await sleep(0.001)
            continue
        if sequence != last_sequence + 1:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        try:
            await to_thread(_commit_barrier, stage)
        except _ArtifactUncertainError:
            await to_thread(
                _write_barrier_message,
                release,
                "failure:artifact_unknown:0:" + value,
                token,
            )
            last_sequence = sequence
            last_value = value
            continue
        except TargetInspectionError as exc:
            await to_thread(
                _write_barrier_message,
                release,
                "failure:target:" + exc.code.value + ":" + value,
                token,
            )
            last_sequence = sequence
            last_value = value
            continue
        except OSError as exc:
            error_number = exc.errno if exc.errno is not None else 0
            await to_thread(
                _write_barrier_message,
                release,
                "failure:os:" + str(error_number) + ":" + value,
                token,
            )
            last_sequence = sequence
            last_value = value
            continue
        await to_thread(_write_barrier_message, release, value, token)
        last_sequence = sequence
        last_value = value


def _barrier_stage(value: str) -> tuple[int, str]:
    """Decode one positive fixed barrier sequence and its exact stage."""
    sequence, separator, stage = value.partition(":")
    if (
        not sequence.isdecimal()
        or not separator
        or stage not in _SEATBELT_BOUNDARIES
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    number = int(sequence)
    if number <= 0:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return number, stage


def _barrier_token() -> bytes:
    """Read one exact authenticated child barrier token from environment."""
    token_value = environ.get(_WORKER_TOKEN_ENV)
    if token_value is None:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    try:
        token = bytes.fromhex(token_value)
    except ValueError as exc:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
    if len(token) != 32:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return token


def _read_barrier_message(path: Path, token: bytes) -> str | None:
    """Read one signed atomic barrier message or reject malformed state."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        envelope = loads(raw)
        if not isinstance(envelope, dict) or set(envelope) != {"mac", "value"}:
            raise ValueError
        value = envelope["value"]
        message_mac = envelope["mac"]
        if (
            not isinstance(value, str)
            or not isinstance(message_mac, str)
            or not compare_digest(
                message_mac, digest(token, value.encode(), "sha256").hex()
            )
        ):
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE) from exc
    return value


def _write_barrier_message(path: Path, value: str, token: bytes) -> None:
    """Publish one signed marker atomically in the private worker namespace."""
    payload = dumps(
        {
            "mac": digest(token, value.encode(), "sha256").hex(),
            "value": value,
        },
        separators=(",", ":"),
    ).encode()
    temporary = path.with_name(path.name + ".next")
    temporary_created = False
    try:
        descriptor = open(
            temporary,
            O_CREAT | O_EXCL | O_NOFOLLOW | O_WRONLY | O_CLOEXEC,
            0o600,
        )
        temporary_created = True
        try:
            offset = 0
            while offset < len(payload):
                count = write_fd(descriptor, payload[offset:])
                if count <= 0:
                    raise OSError("barrier marker write stalled")
                offset += count
            fsync(descriptor)
        finally:
            close(descriptor)
        replace(temporary, path)
        parent = open(path.parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        try:
            fsync(parent)
        finally:
            close(parent)
    except BaseException:
        if temporary_created:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        raise


def _commit_namespace(
    profile: LocalTargetProfile, witness: RootWitness
) -> Path:
    """Require one configured private namespace on the rooted filesystem."""
    namespace = profile.commit_namespace
    if namespace is None:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    descriptor = _open_directory(namespace)
    root_descriptor = _open_directory(profile.root._path)
    try:
        status = fstat(descriptor)
        root_status = fstat(root_descriptor)
        if (
            status.st_dev != witness.identity.device
            or _filesystem_id(descriptor) != witness.filesystem_id
            or _root_mount_id(descriptor, status) != witness.mount_id
            or status.st_mode & 0o077
            or status.st_uid != getuid()
            or root_status.st_dev != witness.identity.device
            or root_status.st_ino != witness.identity.inode
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    finally:
        close(root_descriptor)
        close(descriptor)
    return namespace


def _commit_seatbelt_profile(
    profile: LocalTargetProfile, namespace: Path, token: str
) -> str:
    """Grant mutation only to the rooted workspace and private namespace."""
    base = _worker_seatbelt_profile(profile, (), token)
    runtime_paths = frozenset(
        candidate
        for configured in _SEATBELT_RUNTIME_READ_PATHS
        for candidate in (configured, configured.resolve())
        if candidate.is_dir()
    )
    runtime_reads = tuple(
        "(allow file-read* (subpath " + _seatbelt_string(str(path)) + "))"
        for path in sorted(runtime_paths)
    )
    return base + "\n".join(
        (
            "(allow file-read* (subpath "
            + _seatbelt_string(str(Path(executable).parent.parent))
            + "))",
            "(allow file-read* (subpath "
            + _seatbelt_string(str(Path(__file__).resolve().parents[2]))
            + "))",
            *runtime_reads,
            "(allow file-read* (subpath "
            + _seatbelt_string(str(namespace))
            + "))",
            "(allow file-write* (subpath "
            + _seatbelt_string(str(profile.root._path))
            + "))",
            "(allow file-write* (subpath "
            + _seatbelt_string(str(namespace))
            + "))",
            "",
        )
    )


def _decode_seatbelt_response(
    command: SealedCommitCommand, token: bytes, value: bytes
) -> WorkerReport:
    """Authenticate and decode one bounded child journal response."""
    try:
        envelope = loads(value)
        if not isinstance(envelope, dict):
            raise ValueError
        payload = envelope.get("payload")
        response_mac = envelope.get("mac")
        if not isinstance(payload, dict) or not isinstance(response_mac, str):
            raise ValueError
        raw_payload = dumps(payload, separators=(",", ":")).encode()
        if not compare_digest(
            response_mac, digest(token, raw_payload, "sha256").hex()
        ):
            raise ValueError
        response = _seatbelt_response(payload)
        if response["state"] != WorkerState.SETTLED.value:
            raise ValueError
        steps = tuple(
            JournalStep(
                PatchStepId(item["id"]),
                PatchLineageId(item["lineage"]),
                CommitStepState(item["state"]),
            )
            for item in response["steps"]
        )
        artifacts = tuple(
            ArtifactJournal(item["id"], ArtifactState(item["state"]))
            for item in response["artifacts"]
        )
        journal = SettlementJournal(
            steps, artifacts, PostconditionState(response["postcondition"])
        )
    except (KeyError, TypeError, ValueError, CoordinatorError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from exc
    if tuple(item.identifier for item in journal.steps) != tuple(
        identifier for identifier, _ in _steps(command)
    ) or tuple(item.identifier for item in journal.artifacts) != _artifacts(
        command
    ):
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return WorkerReport(WorkerState.SETTLED, journal)


def _seatbelt_response(value: object) -> _SeatbeltCommitResponse:
    """Validate the finite JSON shape exposed by the child worker."""
    if not isinstance(value, dict):
        raise ValueError
    state = value.get("state")
    postcondition = value.get("postcondition")
    steps = value.get("steps")
    artifacts = value.get("artifacts")
    if (
        not isinstance(state, str)
        or not isinstance(postcondition, str)
        or not isinstance(steps, list)
        or not isinstance(artifacts, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {"id", "lineage", "state"}
            or any(not isinstance(field, str) for field in item.values())
            for item in steps
        )
        or any(
            not isinstance(item, dict)
            or set(item) != {"id", "state"}
            or any(not isinstance(field, str) for field in item.values())
            for item in artifacts
        )
    ):
        raise ValueError
    return {
        "artifacts": artifacts,
        "postcondition": postcondition,
        "state": state,
        "steps": steps,
    }


def _seatbelt_worker_main() -> int:
    """Run one HMAC-bound local mutation in the Seatbelt child process."""
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
        response = _seatbelt_worker_response(_seatbelt_payload(payload))
    except (OSError, TargetInspectionError, TypeError, ValueError):
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


def _seatbelt_payload(value: object) -> _SeatbeltCommitPayload:
    """Verify every command binding before deserializing its plan graph."""
    if not isinstance(value, dict) or set(value) != {
        "command",
        "cwd",
        "fence",
        "namespace",
        "plan_id",
        "request_id",
        "root",
        "version",
        "witness",
    }:
        raise ValueError
    command = value["command"]
    cwd = value["cwd"]
    fence = value["fence"]
    namespace = value["namespace"]
    plan_id = value["plan_id"]
    request_id = value["request_id"]
    root = value["root"]
    version = value["version"]
    witness = value["witness"]
    if (
        not isinstance(command, str)
        or cwd is not None
        and not isinstance(cwd, str)
        or type(fence) is not int
        or not isinstance(namespace, str)
        or not isinstance(plan_id, str)
        or not isinstance(request_id, str)
        or not isinstance(root, str)
        or version != 1
        or not isinstance(witness, dict)
        or set(witness) != {"device", "filesystem_id", "inode", "mount_id"}
        or type(witness["device"]) is not int
        or not isinstance(witness["filesystem_id"], str)
        or type(witness["inode"]) is not int
        or not isinstance(witness["mount_id"], str)
    ):
        raise ValueError
    seatbelt_witness: _SeatbeltWitnessPayload = {
        "device": witness["device"],
        "filesystem_id": witness["filesystem_id"],
        "inode": witness["inode"],
        "mount_id": witness["mount_id"],
    }
    return {
        "command": command,
        "cwd": cwd,
        "fence": fence,
        "namespace": namespace,
        "plan_id": SeatbeltPlanBinding(plan_id),
        "request_id": SeatbeltRequestBinding(request_id),
        "root": root,
        "version": version,
        "witness": seatbelt_witness,
    }


def _seatbelt_worker_response(
    payload: _SeatbeltCommitPayload,
) -> _SeatbeltCommitResponse:
    """Decode a sealed command and return only its typed settlement journal."""
    try:
        command = pickle_loads(b64decode(payload["command"], validate=True))
    except (TypeError, ValueError) as exc:
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from exc
    if (
        type(command) is not SealedCommitCommand
        or command.lease.request_id.value != payload["request_id"]
        or command.lease.fence != payload["fence"]
        or command.plan.plan_id.value != payload["plan_id"]
    ):
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    witness = payload["witness"]
    assert type(witness["device"]) is int
    assert type(witness["inode"]) is int
    assert isinstance(witness["mount_id"], str)
    assert isinstance(witness["filesystem_id"], str)
    local_profile = _SeatbeltCommitProfile(
        _SeatbeltRoot(Path(payload["root"])),
        LogicalPath(payload["cwd"]) if payload["cwd"] is not None else None,
        FileMode(0o644),
    )
    report = _commit_rooted(
        command,
        local_profile,
        RootWitness(
            FileIdentity(witness["device"], witness["inode"]),
            witness["mount_id"],
            witness["filesystem_id"],
        ),
    )
    assert report.journal is not None
    return {
        "artifacts": [
            {"id": item.identifier, "state": item.state.value}
            for item in report.journal.artifacts
        ],
        "postcondition": report.journal.postcondition.value,
        "state": report.state.value,
        "steps": [
            {
                "id": item.identifier.value,
                "lineage": item.lineage.value,
                "state": item.state.value,
            }
            for item in report.journal.steps
        ],
    }


def _commit_rooted(
    command: SealedCommitCommand,
    profile: LocalTargetProfile | _SeatbeltCommitProfile,
    witness: RootWitness,
) -> WorkerReport:
    """Use retained root and parent descriptors for every write primitive."""
    steps = _steps(command)
    states = [CommitStepState.NOT_COMMITTED for _ in steps]
    artifacts = _artifacts(command)
    artifact_states = [ArtifactState.ABSENT for _ in artifacts]
    root_fd = _open_directory(profile.root._path)
    _commit_barrier("target.open_handle")
    root_token = _ROOT_DESCRIPTOR.set(root_fd)
    parent_token = _PARENT_IDENTITIES.set(
        {
            path: FileIdentity(identity[0], identity[1])
            for lineage in command.plan.candidate.lineages
            for path, identity in lineage.parent_identities
        }
    )
    try:
        status = fstat(root_fd)
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
                profile.root._path,
            )
        )
        try:
            offset = 0
            for index, lineage in enumerate(command.plan.candidate.lineages):
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
                        command.plan.binding.final.effects,
                        lineage,
                        indices,
                        states,
                        index,
                        artifact_states,
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
                _commit_barrier("verification.before")
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
        _PARENT_IDENTITIES.reset(parent_token)
        _ROOT_DESCRIPTOR.reset(root_token)
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
        _commit_barrier("parent_opened")
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
        _commit_barrier("target.open_handle")
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
        _commit_barrier("target.close_handle")
        close(descriptor)
        raise
    _commit_barrier("target.close_handle")
    return descriptor, protected_metadata


def _is_contained(root_fd: int, descriptor: int) -> bool:
    """Return whether a live descriptor remains beneath the retained root."""
    root_path = _descriptor_path(root_fd)
    descriptor_path = _descriptor_path(descriptor)
    return descriptor_path == root_path or root_path in descriptor_path.parents


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
) -> None:
    """Revalidate rooted handles immediately before one namespace syscall."""
    _commit_barrier("target.namespace_before_final_check")
    _validate_namespace_context(
        parent_fd,
        path,
        entries,
    )
    _commit_barrier("target.namespace_after_final_check")
    _validate_namespace_context(
        parent_fd,
        path,
        entries,
    )
    _commit_barrier("target.namespace_before_effect")
    effect()


def _validate_namespace_context(
    parent_fd: int,
    path: LogicalPath,
    entries: tuple[tuple[int, LogicalPath, str, int], ...],
) -> None:
    """Prove all rooted parent and source identities remain live and sealed."""
    context = _COMMIT_CONTEXT.get()
    if context is None:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
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


def _descriptor_path(descriptor: int) -> Path:
    """Return the kernel's current path for one retained Darwin descriptor."""
    buffer = _CFFI.new("char[]", _PATH_MAX)
    if _LIBC.fcntl(descriptor, _F_GETPATH, buffer) != 0:
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    value = _CFFI.string(buffer).decode("utf-8", "strict")
    if not value:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    return Path(value)


def _commit_barrier(stage: str) -> None:
    """Relay a fixed child boundary to its authenticated host coordinator."""
    global _SEATBELT_WORKER_SEQUENCE
    global _SEATBELT_WORKER_SESSION
    marker_value = environ.get(_SEATBELT_BARRIER_ENV)
    release_value = environ.get(_SEATBELT_RELEASE_ENV)
    if marker_value is None or release_value is None:
        return
    if stage not in _SEATBELT_BOUNDARIES:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    marker = Path(marker_value)
    release = Path(release_value)
    token = _barrier_token()
    session = (marker_value, release_value, token)
    if session != _SEATBELT_WORKER_SESSION:
        _SEATBELT_WORKER_SESSION = session
        _SEATBELT_WORKER_SEQUENCE = 0
    previous = _read_barrier_message(marker, token)
    if _SEATBELT_WORKER_SEQUENCE == 0:
        if previous is not None:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    elif previous is None:
        raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    else:
        sequence, _ = _barrier_stage(previous)
        if sequence != _SEATBELT_WORKER_SEQUENCE:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
    value = str(_SEATBELT_WORKER_SEQUENCE + 1) + ":" + stage
    _write_barrier_message(marker, value, token)
    _SEATBELT_WORKER_SEQUENCE += 1
    deadline = monotonic() + _SEATBELT_BARRIER_TIMEOUT_SECONDS
    while monotonic() < deadline:
        response = _read_barrier_message(release, token)
        if response == value:
            release.unlink()
            return
        if response is not None:
            parts = response.split(":", 3)
            if len(parts) != 4:
                continue
            failure, kind, detail, failed_value = parts
            if failure == "failure" and failed_value == value:
                release.unlink()
                if kind == "target":
                    try:
                        code = TargetErrorCode(detail)
                    except ValueError as exc:
                        raise TargetInspectionError(
                            TargetErrorCode.WITNESS_STALE
                        ) from exc
                    raise TargetInspectionError(code)
                if kind == "artifact_unknown" and detail == "0":
                    raise _ArtifactUncertainError(
                        "authenticated host test barrier failed"
                    )
                if kind == "os" and detail.isdecimal():
                    if detail == "0":
                        raise OSError("authenticated host test barrier failed")
                    raise OSError(
                        int(detail), "authenticated host test barrier failed"
                    )
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        blocking_sleep(0.001)
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
) -> str:
    """Create complete collision-safe directory-local private staging bytes."""
    _commit_barrier("target.stage_artifact")
    _commit_barrier("artifact.stage")
    for _ in range(32):
        name = ".avalan-patch-" + sha256(token_bytes(32)).hexdigest()[:32]
        try:
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
                _commit_barrier("artifact.stage_write_before")
                count = write_fd(descriptor, value[offset:])
                if count <= 0:
                    raise OSError("staging write stalled")
                offset += count
            fchmod(descriptor, mode)
            if protected_metadata is not None:
                _restore_protected_metadata(descriptor, protected_metadata)
            fsync(descriptor)
        except BaseException:
            close(descriptor)
            try:
                _commit_barrier("artifact.stage_cleanup_before")
                unlink(name, dir_fd=parent_fd)
            except OSError as cleanup_error:
                raise _ArtifactUncertainError from cleanup_error
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
    protected_metadata: _ProtectedMetadata | None = None,
) -> None:
    """Publish a complete file through atomic no-replace linking."""
    try:
        stage = _stage(parent_fd, value, mode, protected_metadata)
    except _ArtifactUncertainError:
        artifact_states[artifact_index] = ArtifactState.UNKNOWN
        raise
    descriptor = open(
        stage, O_RDONLY | O_NOFOLLOW | O_CLOEXEC, dir_fd=parent_fd
    )
    try:
        _commit_barrier("publication.before_link")
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
            _commit_barrier("artifact.cleanup_before")
            _namespace_effect(
                parent_fd,
                path,
                lambda: unlink(stage, dir_fd=parent_fd),
                entries=((parent_fd, path, stage, descriptor),),
            )
        except OSError:
            artifact_states[artifact_index] = ArtifactState.LEAKED
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
) -> None:
    """Atomically replace one staged regular file without truncation."""
    try:
        stage = _stage(parent_fd, value, mode, protected_metadata)
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
        )
    except BaseException:
        try:
            _namespace_effect(
                parent_fd,
                path,
                lambda: unlink(stage, dir_fd=parent_fd),
                entries=((parent_fd, path, stage, descriptor),),
            )
        except OSError:
            artifact_states[artifact_index] = ArtifactState.LEAKED
        raise
    finally:
        close(descriptor)
    artifact_states[artifact_index] = ArtifactState.CLEANED


def _commit_lineage(
    cwd_fd: int,
    cwd_identity: FileIdentity,
    root: RootWitness,
    profile: LocalTargetProfile | _SeatbeltCommitProfile,
    authorized_effects: frozenset[Capability],
    lineage: PlannedLineage,
    indices: tuple[int, ...],
    states: list[CommitStepState],
    artifact_index: int,
    artifact_states: list[ArtifactState],
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
            _commit_barrier("requested_effect.step_before")
            artifact_states[artifact_index] = ArtifactState.STAGED
            _publish_new(
                parent,
                destination,
                leaf,
                final.bytes_value._value,
                profile.creation_mode.value,
                artifact_states,
                artifact_index,
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
                _namespace_effect(
                    parent,
                    source,
                    lambda: unlink(leaf, dir_fd=parent),
                    entries=((parent, source, leaf, descriptor),),
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
                _commit_barrier("requested_effect.step_before")
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
                        _commit_barrier("publication.before_link")
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
                        )
                    except OSError as exc:
                        if exc.errno in {ENOSYS, EOPNOTSUPP, EXDEV}:
                            raise TargetInspectionError(
                                TargetErrorCode.CAPABILITY_UNAVAILABLE
                            ) from exc
                        raise
                    artifact_states[artifact_index] = ArtifactState.ABSENT
                else:
                    _publish_new(
                        destination_parent,
                        destination,
                        destination_leaf,
                        final.bytes_value._value,
                        final.metadata.mode.value,
                        artifact_states,
                        artifact_index,
                        protected_metadata,
                    )
                states[indices[0]] = CommitStepState.COMMITTED
                _commit_barrier("move.source_remove_before")
                _namespace_effect(
                    source_parent,
                    source,
                    lambda: unlink(source_leaf, dir_fd=source_parent),
                    entries=(
                        (source_parent, source, source_leaf, descriptor),
                    ),
                )
            finally:
                close(descriptor)
            states[indices[1]] = CommitStepState.COMMITTED
        finally:
            close(destination_parent)
    finally:
        close(source_parent)


def _verify(
    command: SealedCommitCommand,
    cwd_fd: int,
    cwd_identity: FileIdentity,
    root: RootWitness,
) -> PostconditionState:
    """Verify the final requested entries without executing workspace code."""
    for lineage in command.plan.candidate.lineages:
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
