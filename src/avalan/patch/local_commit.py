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
from dataclasses import dataclass, field
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
    fstat,
    fsync,
    getuid,
    open,
    replace,
)
from os import write as write_fd
from pathlib import Path
from pickle import dumps as pickle_dumps
from pickle import loads as pickle_loads
from secrets import token_bytes
from subprocess import PIPE
from sys import executable, stdin, stdout
from time import monotonic
from time import sleep as blocking_sleep
from typing import Callable, NewType, TypedDict

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
    PatchLineageId,
    PatchRequestId,
    PatchStepId,
    PostconditionState,
)
from avalan.patch.rooted_worker import (
    RootedMutationCommand,
    RootedMutationProfile,
    _artifacts,
    _ArtifactUncertainError,
    _failed_report,
    _steps,
)
from avalan.patch.rooted_worker import (
    _commit_rooted as _commit_rooted_neutral,
)
from avalan.patch.target import (
    _FUTURE_MUTATION_PRIMITIVES,
    _WORKER_TOKEN_ENV,
    FileIdentity,
    LocalTargetProfile,
    ResolvedMutationScope,
    RootWitness,
    TargetErrorCode,
    TargetHandshake,
    TargetInspectionError,
    _filesystem_id,
    _HostRootBinding,
    _is_local_mutation_test_platform,
    _open_directory,
    _root_mount_id,
    _seatbelt_string,
    _validate_host_root_binding,
    _worker_sandbox_command,
    _worker_seatbelt_profile,
)


def _commit_rooted(
    command: SealedCommitCommand | RootedMutationCommand,
    profile: LocalTargetProfile | RootedMutationProfile,
    witness: RootWitness,
    fence_check: Callable[[], None] | None = None,
) -> WorkerReport:
    """Adapt the local test profile to neutral rooted primitives."""
    mutation = (
        profile
        if isinstance(profile, RootedMutationProfile)
        else RootedMutationProfile(
            profile.root._path,
            profile.cwd,
            profile.creation_mode,
        )
    )
    return _commit_rooted_neutral(
        command,
        mutation,
        witness,
        fence_check,
        _commit_barrier,
    )


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
        if not _is_local_mutation_test_platform(self.profile):
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
            or scope._host_root_binding is None
            or scope._worker_authorization
            is not self.profile._worker_authorization
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        _validate_host_root_binding(
            self.profile.root, scope._host_root_binding
        )

    async def _commit(
        self, scope: ResolvedMutationScope, command: SealedCommitCommand
    ) -> WorkerReport:
        """Apply the exact sealed graph in the isolated Seatbelt worker."""
        try:
            self._require_scope(scope)
            if command.plan.binding.target != self.profile.identity:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            assert scope.root_witness is not None
            assert scope._host_root_binding is not None
            return await _commit_in_seatbelt(
                command,
                self.profile,
                scope.root_witness,
                scope._host_root_binding,
            )
        except TargetInspectionError:
            return _failed_report(command, CommitStepState.NOT_COMMITTED)
        except OSError:
            return _failed_report(command, CommitStepState.UNKNOWN)


async def _commit_in_seatbelt(
    command: SealedCommitCommand,
    profile: LocalTargetProfile,
    witness: RootWitness,
    host_root_binding: _HostRootBinding,
) -> WorkerReport:
    """Execute one authenticated command in the selected native sandbox."""
    _validate_host_root_binding(profile.root, host_root_binding)
    namespace = _commit_namespace(profile, witness)

    def validate_host_root() -> None:
        """Require the dispatch root to stay bound before a child effect."""
        _validate_host_root_binding(profile.root, host_root_binding)

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
        process_command, environment = _worker_sandbox_command(
            profile,
            worker_argv,
            {
                _SEATBELT_BARRIER_ENV: str(marker),
                _SEATBELT_RELEASE_ENV: str(release),
                _WORKER_TOKEN_ENV: token.hex(),
            },
            (profile.root._path, namespace),
            _commit_seatbelt_profile(profile, namespace, token.hex()),
        )
        process = await create_subprocess_exec(
            *process_command,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE,
            cwd="/",
            env=environment,
            close_fds=True,
        )
    except OSError as exc:
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from exc
    relay = create_task(
        _relay_seatbelt_barriers(marker, release, token, validate_host_root)
    )
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
    marker: Path,
    release: Path,
    token: bytes,
    root_binding_check: Callable[[], None] | None = None,
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
            if root_binding_check is not None:
                await to_thread(root_binding_check)
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
    local_profile = RootedMutationProfile(
        Path(payload["root"]),
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
