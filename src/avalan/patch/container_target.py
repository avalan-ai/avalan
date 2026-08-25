"""Bind patch mutation to one sealed persistent Docker volume service.

Ordinary container profiles receive the workspace volume read-only.  This
module is the separate trusted patch authority: it mounts that same volume
read-write only in a no-network, read-only-rootfs worker container and sends
the worker authenticated sealed patch transactions over stdio.
"""

from asyncio import Lock, create_subprocess_exec, wait_for
from asyncio.subprocess import DEVNULL, PIPE, Process
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from hmac import digest
from json import JSONDecodeError, dumps, loads
from pathlib import Path
from re import fullmatch
from secrets import token_bytes

from avalan.patch.coordinator import (
    RootedCommandAuthorityValidator,
    RootedSandboxCommitWorker,
    SealedCommitCommand,
    WorkerReport,
    _rooted_sandbox_endpoint,
    _sandbox_worker_for_endpoint,
)
from avalan.patch.domain import (
    ByteSize,
    Capability,
    ContextKind,
    LogicalPath,
    PatchLimits,
    PatchProtocolId,
)
from avalan.patch.durable_store import DurablePatchStoreBinding
from avalan.patch.policy import TrustedPatchPolicy
from avalan.patch.sandbox_commit import (
    _MAX_MESSAGE_BYTES,
    _MESSAGE_VERSION,
    _PROCESS_IO_SECONDS,
    _PROCESS_REAP_SECONDS,
    SandboxChannelId,
    SandboxContextLifetimeId,
    SandboxExecutionPlanFingerprint,
    SandboxPatchSdkService,
    SandboxPatchServiceConfiguration,
    SandboxProfileReceipt,
    SandboxSessionId,
    SandboxWorkerImplementationId,
    SandboxWorkerProtocolVersion,
    _identity_payload,
    _ImplementationBundle,
    _report_from_payload,
    _response_payload,
    _root_from_payload,
    _root_payload,
    _RuntimeAttestation,
    _SandboxEndpoint,
    _sealed_command_payload,
)
from avalan.patch.target import (
    _FUTURE_MUTATION_PRIMITIVES,
    EphemeralWorkerWitness,
    InspectionBatch,
    InspectionRequest,
    LocalPlatformProfile,
    PrimitiveProbe,
    ProbeState,
    ResolvedMutationScope,
    RootWitness,
    ScopeSelection,
    TargetErrorCode,
    TargetHandshake,
    TargetIdentity,
    TargetInspectionError,
    TargetPrimitive,
    TargetSnapshot,
    _snapshot_from_worker,
)
from avalan.patch.toolset import (
    PatchApprovalBinding,
    PatchCoordinatorBinding,
    PatchPersistenceBinding,
    PatchRuntimeBinding,
)

_CONTAINER_BACKEND = "container"
_CONTAINER_WORKSPACE = "/workspace"
_CONTAINER_PRIVATE = "/private"
_CONTAINER_SEED_OWNERSHIP_BOOTSTRAP = """from os import chown, walk
from pathlib import Path
root = Path("/workspace")
chown(root, 0, 0)
for base, directories, files in walk(root, followlinks=False):
    for name in (*directories, *files):
        chown(Path(base) / name, 0, 0, follow_symlinks=False)
"""
_CONTAINER_PROTOCOL = SandboxWorkerProtocolVersion(
    "container-patch-runtime-v1"
)
_CONTAINER_IMAGE_PATTERN = (
    r"(?:python:3\.11-slim-bookworm@sha256:|sha256:)[a-f0-9]{64}"
)
_VOLUME_RESOURCE_LABEL = "avalan.patch.resource"
_VOLUME_OWNER_LABEL = "avalan.patch.owner"
_VOLUME_GUARD_PREFIX = "avalan_patch_lease_"
_VOLUME_GUARD_SLEEP_SECONDS = "86400"


@dataclass(slots=True, repr=False)
class _OwnedVolumeClaim:
    """Track one authenticated persistent volume known to this authority."""

    resource_digest: str
    owner_receipt: str
    active_attachments: int = 0


_OWNED_VOLUMES: dict[str, _OwnedVolumeClaim] = {}
_OWNED_VOLUMES_LOCK = Lock()
_DOCKER_COMMAND_TIMEOUT_SECONDS = 10.0
_DOCKER_BUILD_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True, slots=True, repr=False)
class ContainerPersistentLeaseAuthority:
    """Carry host-only authority to authenticate one persistent lease."""

    _key: bytes = field(repr=False)

    def __post_init__(self) -> None:
        """Require one opaque fixed-width host-issued authority value."""
        if type(self._key) is not bytes or len(self._key) != 32:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)

    @classmethod
    def from_bytes(cls, value: bytes) -> "ContainerPersistentLeaseAuthority":
        """Wrap one exact authority value issued by the trusted host."""
        return cls(value)

    def _owner_receipt(self, resource_digest: str) -> str:
        """Authenticate the full persistent resource identity privately."""
        return digest(self._key, resource_digest.encode(), "sha256").hex()


def container_protocol_id(
    version: SandboxWorkerProtocolVersion = _CONTAINER_PROTOCOL,
) -> PatchProtocolId:
    """Return the immutable protocol identity for the container worker."""
    if not isinstance(version, str):
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    return PatchProtocolId(
        "protocol_" + sha256(version.encode()).hexdigest()[:16]
    )


@dataclass(frozen=True, slots=True)
class ContainerPatchImage:
    """Name the one pinned test-profile worker image."""

    reference: str

    def __post_init__(self) -> None:
        """Reject mutable, caller-selected, or non-Python worker images."""
        if fullmatch(_CONTAINER_IMAGE_PATTERN, self.reference) is None:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True, repr=False)
class ContainerPatchRuntimeContext:
    """Carry host-issued identity and finite limits for one lease."""

    identity: TargetIdentity
    limits: PatchLimits
    max_snapshot_bytes: ByteSize
    cwd: LogicalPath | None
    channel_id: SandboxChannelId
    context_lifetime_id: SandboxContextLifetimeId
    implementation_id: SandboxWorkerImplementationId

    def __post_init__(self) -> None:
        """Require immutable identities not supplied by a tool request."""
        if (
            type(self.identity) is not TargetIdentity
            or type(self.limits) is not PatchLimits
            or type(self.max_snapshot_bytes) is not ByteSize
            or self.cwd is not None
            and type(self.cwd) is not LogicalPath
            or not self.channel_id
            or not self.context_lifetime_id
            or not self.implementation_id
            or self.identity.protocol_id != container_protocol_id()
            or self.identity.implementation_id != self.implementation_id
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


@dataclass(frozen=True, slots=True, repr=False)
class ContainerPatchRuntimeSettings:
    """Describe the selected, read-only ordinary container context."""

    image: ContainerPatchImage
    context: ContainerPatchRuntimeContext
    seed_root: Path
    execution_plan_fingerprint: SandboxExecutionPlanFingerprint
    persistent_lease_authority: ContainerPersistentLeaseAuthority
    test_profile: bool = False
    root_subdirectory: LogicalPath | None = None

    def __post_init__(self) -> None:
        """Require the explicit Linux test profile and a trusted seed tree."""
        if (
            type(self.image) is not ContainerPatchImage
            or type(self.context) is not ContainerPatchRuntimeContext
            or not self.seed_root.is_absolute()
            or not self.seed_root.is_dir()
            or not self.execution_plan_fingerprint
            or type(self.persistent_lease_authority)
            is not ContainerPersistentLeaseAuthority
            or not self.test_profile
            or self.root_subdirectory is not None
            and type(self.root_subdirectory) is not LogicalPath
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)

    def create_runtime(self) -> "ContainerPatchRuntime":
        """Create the one service owner for this persistent lease."""
        return ContainerPatchRuntime(self)


@dataclass(frozen=True, slots=True)
class ContainerRuntimeReceipt:
    """Bind a live Docker child and its root witness to one lease."""

    session_id: SandboxSessionId
    profile_receipt: SandboxProfileReceipt
    root: RootWitness
    worker: EphemeralWorkerWitness
    primitive_receipts: Mapping[TargetPrimitive, str]
    runtime_command_digest: str
    backend_policy_digest: str
    child_process_identity: str
    canary_receipt: str

    def __post_init__(self) -> None:
        """Require every finite runtime receipt before advertisement."""
        primitives = _FUTURE_MUTATION_PRIMITIVES | _container_primitives()
        if (
            not self.session_id
            or not self.profile_receipt
            or type(self.root) is not RootWitness
            or type(self.worker) is not EphemeralWorkerWitness
            or set(self.primitive_receipts) != primitives
            or any(not value for value in self.primitive_receipts.values())
            or any(
                not value
                for value in (
                    self.runtime_command_digest,
                    self.backend_policy_digest,
                    self.child_process_identity,
                    self.canary_receipt,
                )
            )
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)


def _container_primitives() -> frozenset[TargetPrimitive]:
    """Return the service-only primitives proven by the Docker profile."""
    return frozenset(
        (
            TargetPrimitive.PERSISTENCE,
            TargetPrimitive.CANCELLATION_SETTLEMENT,
            TargetPrimitive.JOURNAL_DELIVERY,
            TargetPrimitive.APPROVAL,
            TargetPrimitive.DURABLE_FENCING,
        )
    )


def _docker_name(prefix: str, value: str) -> str:
    """Return one non-content-derived Docker object name."""
    return prefix + sha256(value.encode()).hexdigest()[:24]


def _container_root_path(settings: ContainerPatchRuntimeSettings) -> str:
    """Return the sealed logical root inside the persistent Docker volume."""
    subdirectory = settings.root_subdirectory
    if subdirectory is None:
        return _CONTAINER_WORKSPACE
    return _CONTAINER_WORKSPACE + "/" + subdirectory.value


def _persistent_resource_digest(
    settings: ContainerPatchRuntimeSettings,
) -> str:
    """Seal the backing resource identity used by one persistent lease."""
    identity = settings.context.identity
    payload = "\x00".join(
        (
            identity.domain_id.value,
            identity.workspace_id.value,
            identity.protocol_id.value,
            identity.filesystem_id,
            identity.mount_id,
            identity.persistent_lease_id,
            identity.implementation_id,
            settings.execution_plan_fingerprint,
            (
                ""
                if settings.root_subdirectory is None
                else settings.root_subdirectory.value
            ),
        )
    )
    return sha256(payload.encode()).hexdigest()


def _volume_owner_receipt(
    authority: ContainerPersistentLeaseAuthority, resource_digest: str
) -> str:
    """Authenticate a resource digest without exporting service authority."""
    return authority._owner_receipt(resource_digest)


def _volume_labels(
    resource_digest: str, owner_receipt: str
) -> tuple[str, str]:
    """Return the two fixed labels required for an owned volume."""
    return (
        _VOLUME_RESOURCE_LABEL + "=" + resource_digest,
        _VOLUME_OWNER_LABEL + "=" + owner_receipt,
    )


def _owned_volume_matches(
    inspected: str, resource_digest: str, owner_receipt: str
) -> bool:
    """Return whether Docker inspection proves the exact volume ownership."""
    try:
        rows = loads(inspected)
    except (JSONDecodeError, TypeError):
        return False
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        return False
    labels = rows[0].get("Labels")
    if labels is None:
        configuration = rows[0].get("Config")
        labels = (
            None
            if type(configuration) is not dict
            else configuration.get("Labels")
        )
    return (
        type(labels) is dict
        and labels.get(_VOLUME_RESOURCE_LABEL) == resource_digest
        and labels.get(_VOLUME_OWNER_LABEL) == owner_receipt
    )


def _owned_volume_guard_is_stopped(
    inspected: str, resource_digest: str, owner_receipt: str
) -> bool:
    """Return whether one exact authenticated durable guard is stopped."""
    if not _owned_volume_matches(inspected, resource_digest, owner_receipt):
        return False
    try:
        rows = loads(inspected)
    except (JSONDecodeError, TypeError):
        return False
    if type(rows) is not list or len(rows) != 1 or type(rows[0]) is not dict:
        return False
    state = rows[0].get("State")
    return type(state) is dict and state.get("Running") is False


async def _volume_has_no_live_attachment(volume: str) -> bool:
    """Return whether Docker proves no live container uses one volume."""
    attached = await _docker_output(
        ("docker", "ps", "--quiet", "--filter", "volume=" + volume),
        False,
    )
    return attached is not None and not attached.strip()


def _docker_env() -> dict[str, str]:
    """Return the minimal environment used to invoke the local Docker CLI."""
    return {"PATH": "/usr/local/bin:/usr/bin:/bin"}


def _make_container_bundle_readable(root: Path) -> None:
    """Expose only immutable bundle bytes to the capability-dropped child."""
    if not root.is_dir() or root.is_symlink():
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_symlink():
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        if path.is_dir():
            path.chmod(0o555)
        elif path.is_file() and path.stat().st_nlink == 1:
            path.chmod(0o444)
        else:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    root.chmod(0o555)


@dataclass(slots=True, repr=False)
class _ContainerRuntimeProcess:
    """Own a persistent no-network Docker worker and authenticated channel."""

    settings: ContainerPatchRuntimeSettings
    _process: Process | None = field(default=None, init=False)
    _container_id: str | None = field(default=None, init=False)
    _volume_name: str | None = field(default=None, init=False)
    _volume_resource_digest: str | None = field(default=None, init=False)
    _volume_owner_receipt: str | None = field(default=None, init=False)
    _volume_attached: bool = field(default=False, init=False)
    _volume_owned: bool = field(default=False, init=False)
    _volume_guard_name: str | None = field(default=None, init=False)
    _token: bytes | None = field(default=None, init=False)
    _receipt: SandboxProfileReceipt | None = field(default=None, init=False)
    _root: RootWitness | None = field(default=None, init=False)
    _session_id: SandboxSessionId | None = field(default=None, init=False)
    _implementation_digest_value: str | None = field(default=None, init=False)
    _bundle: _ImplementationBundle | None = field(default=None, init=False)
    _attestation: _RuntimeAttestation | None = field(default=None, init=False)
    _sequence: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _reap_lock: Lock = field(default_factory=Lock, init=False)
    _closed: bool = field(default=False, init=False)

    @property
    def volume_name(self) -> str | None:
        """Return the opaque persistent volume name without a host path."""
        return self._volume_name

    async def _has_local_volume_claim(
        self, volume: str, resource_digest: str, owner_receipt: str
    ) -> bool:
        """Return whether this host already owns the exact volume claim."""
        async with _OWNED_VOLUMES_LOCK:
            claim = _OWNED_VOLUMES.get(volume)
            if claim is None:
                return False
            if (
                claim.resource_digest != resource_digest
                or claim.owner_receipt != owner_receipt
            ):
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            return True

    async def _claim_volume(
        self,
        volume: str,
        resource_digest: str,
        owner_receipt: str,
        *,
        created: bool,
        guard_name: str | None = None,
    ) -> None:
        """Reserve one authenticated volume before any worker can attach."""
        async with _OWNED_VOLUMES_LOCK:
            claim = _OWNED_VOLUMES.get(volume)
            if created:
                if claim is not None:
                    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            elif claim is not None:
                if (
                    claim.resource_digest != resource_digest
                    or claim.owner_receipt != owner_receipt
                ):
                    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
                claim.active_attachments += 1
                self._volume_name = volume
                self._volume_resource_digest = resource_digest
                self._volume_owner_receipt = owner_receipt
                self._volume_attached = True
                self._volume_owned = True
                return
            if not await _volume_has_no_live_attachment(volume):
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            if guard_name is None:
                guard_name = await self._acquire_volume_guard(
                    volume, resource_digest, owner_receipt
                )
            elif self._volume_guard_name != guard_name:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            claim = _OwnedVolumeClaim(resource_digest, owner_receipt)
            _OWNED_VOLUMES[volume] = claim
            self._volume_guard_name = guard_name
            claim.active_attachments += 1
        self._volume_name = volume
        self._volume_resource_digest = resource_digest
        self._volume_owner_receipt = owner_receipt
        self._volume_attached = True
        self._volume_owned = True

    async def _acquire_volume_guard(
        self, volume: str, resource_digest: str, owner_receipt: str
    ) -> str:
        """Acquire a Docker-named host-only guard for one recovered lease."""
        guard_name = _docker_name(
            _VOLUME_GUARD_PREFIX,
            self.settings.context.identity.persistent_lease_id,
        )
        labels = _volume_labels(resource_digest, owner_receipt)
        command = (
            "docker",
            "run",
            "--detach",
            "--name",
            guard_name,
            "--label",
            labels[0],
            "--label",
            labels[1],
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "4",
            "--memory",
            "16m",
            self.settings.image.reference,
            "python3",
            "-I",
            "-c",
            "from time import sleep;sleep("
            + _VOLUME_GUARD_SLEEP_SECONDS
            + ")",
        )
        created = await _docker_output(command, False)
        if created is not None and created.strip():
            return guard_name
        inspected = await _docker_output(
            ("docker", "inspect", guard_name), False
        )
        if (
            inspected is None
            or not _owned_volume_guard_is_stopped(
                inspected, resource_digest, owner_receipt
            )
            or not await _volume_has_no_live_attachment(volume)
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        removed = await _docker_output(
            ("docker", "rm", "--force", guard_name), False
        )
        if removed is None:
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        created = await _docker_output(command, False)
        if created is None or not created.strip():
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        return guard_name

    async def _release_volume_attachment(self) -> None:
        """Release only this worker's in-process authenticated attachment."""
        volume = self._volume_name
        resource_digest = self._volume_resource_digest
        owner_receipt = self._volume_owner_receipt
        if (
            not self._volume_attached
            or volume is None
            or resource_digest is None
            or owner_receipt is None
        ):
            provisional_guard_name = self._volume_guard_name
            self._volume_guard_name = None
            self._volume_name = None
            self._volume_resource_digest = None
            self._volume_owner_receipt = None
            self._volume_owned = False
            if (
                provisional_guard_name is not None
                and resource_digest is not None
                and owner_receipt is not None
            ):
                await self._release_volume_guard(
                    provisional_guard_name, resource_digest, owner_receipt
                )
            return
        guard_name: str | None = None
        async with _OWNED_VOLUMES_LOCK:
            claim = _OWNED_VOLUMES.get(volume)
            if (
                claim is None
                or claim.resource_digest != resource_digest
                or claim.owner_receipt != owner_receipt
                or claim.active_attachments <= 0
            ):
                self._volume_attached = False
                self._volume_owned = False
                return
            claim.active_attachments -= 1
            if claim.active_attachments == 0:
                _OWNED_VOLUMES.pop(volume, None)
                guard_name = self._volume_guard_name or _docker_name(
                    _VOLUME_GUARD_PREFIX,
                    self.settings.context.identity.persistent_lease_id,
                )
        self._volume_attached = False
        if guard_name is not None:
            await self._release_volume_guard(
                guard_name, resource_digest, owner_receipt
            )
            self._volume_guard_name = None

    async def _release_volume_guard(
        self, guard_name: str, resource_digest: str, owner_receipt: str
    ) -> None:
        """Remove only a final authenticated host-side lease guard."""
        inspected = await _docker_output(
            ("docker", "inspect", guard_name), False
        )
        if inspected is None or not _owned_volume_matches(
            inspected, resource_digest, owner_receipt
        ):
            return
        await _docker_output(("docker", "rm", "--force", guard_name), False)

    async def _cleanup_new_volume(
        self, volume: str, resource_digest: str, owner_receipt: str
    ) -> None:
        """Remove a failed-start volume only while holding its guard."""
        if not self._volume_owned:
            return
        try:
            guard_name = await self._acquire_volume_guard(
                volume, resource_digest, owner_receipt
            )
        except TargetInspectionError:
            await self._clear_failed_volume_state(
                volume, resource_digest, owner_receipt
            )
            return
        try:
            if not await _volume_has_no_live_attachment(volume):
                return
            inspected = await _docker_output(
                ("docker", "volume", "inspect", volume), False
            )
            if inspected is not None and _owned_volume_matches(
                inspected, resource_digest, owner_receipt
            ):
                await _docker_output(("docker", "volume", "rm", volume), False)
        finally:
            await self._release_volume_guard(
                guard_name, resource_digest, owner_receipt
            )
            await self._clear_failed_volume_state(
                volume, resource_digest, owner_receipt
            )

    async def _clear_failed_volume_state(
        self, volume: str, resource_digest: str, owner_receipt: str
    ) -> None:
        """Forget only this failed starter's local volume attachment."""
        async with _OWNED_VOLUMES_LOCK:
            claim = _OWNED_VOLUMES.get(volume)
            if (
                claim is not None
                and claim.resource_digest == resource_digest
                and claim.owner_receipt == owner_receipt
                and claim.active_attachments == 0
            ):
                _OWNED_VOLUMES.pop(volume, None)
        self._volume_name = None
        self._volume_resource_digest = None
        self._volume_owner_receipt = None
        self._volume_attached = False
        self._volume_owned = False
        self._volume_guard_name = None

    async def _dispose_owned_volume(self) -> None:
        """Delete only an idle volume while holding its authenticated guard."""
        volume = self._volume_name
        resource_digest = self._volume_resource_digest
        owner_receipt = self._volume_owner_receipt
        if (
            not self._volume_owned
            or volume is None
            or resource_digest is None
            or owner_receipt is None
        ):
            return
        async with _OWNED_VOLUMES_LOCK:
            claim = _OWNED_VOLUMES.get(volume)
            if claim is not None:
                return
        guard_name = await self._acquire_volume_guard(
            volume, resource_digest, owner_receipt
        )
        try:
            if not await _volume_has_no_live_attachment(volume):
                return
            inspected = await _docker_output(
                ("docker", "volume", "inspect", volume), False
            )
            if inspected is None or not _owned_volume_matches(
                inspected, resource_digest, owner_receipt
            ):
                return
            removed = await _docker_output(
                ("docker", "volume", "rm", volume), False
            )
            if removed is None:
                return
            self._volume_name = None
            self._volume_resource_digest = None
            self._volume_owner_receipt = None
            self._volume_owned = False
            self._volume_guard_name = None
        finally:
            await self._release_volume_guard(
                guard_name, resource_digest, owner_receipt
            )

    async def start(
        self,
    ) -> tuple[RootWitness, ContainerRuntimeReceipt]:
        """Start and attest the only trusted writable container service."""
        async with self._lock:
            if self._closed:
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            if self._process is not None:
                assert self._root is not None
                return self._root, await self._runtime_receipt_locked()
            token = token_bytes(32)
            context = self.settings.context
            session_id = SandboxSessionId(
                "session_"
                + sha256(
                    token
                    + context.context_lifetime_id.encode()
                    + context.identity.persistent_lease_id.encode()
                ).hexdigest()[:32]
            )
            volume_name = _docker_name(
                "avalan_patch_", context.identity.persistent_lease_id
            )
            resource_digest = _persistent_resource_digest(self.settings)
            owner_receipt = _volume_owner_receipt(
                self.settings.persistent_lease_authority, resource_digest
            )
            container_name = _docker_name("avalan_patch_worker_", session_id)
            bundle = _ImplementationBundle.create(
                self.settings.seed_root, include_dependencies=False
            )
            _make_container_bundle_readable(bundle.root)
            root_path = _container_root_path(self.settings)
            profile_receipt = SandboxProfileReceipt(
                sha256(
                    "\x00".join(
                        (
                            self.settings.image.reference,
                            self.settings.execution_plan_fingerprint,
                            context.identity.context_id.value,
                            context.identity.workspace_id.value,
                            context.identity.domain_id.value,
                            context.identity.filesystem_id,
                            context.identity.mount_id,
                            context.identity.persistent_lease_id,
                            context.implementation_id,
                            "" if context.cwd is None else context.cwd.value,
                            context.channel_id,
                            context.context_lifetime_id,
                            root_path,
                            _CONTAINER_PRIVATE,
                        )
                    ).encode()
                ).hexdigest()
            )
            config = {
                "root": root_path,
                "namespace": _CONTAINER_PRIVATE,
                "cwd": None if context.cwd is None else context.cwd.value,
                "maximum": context.max_snapshot_bytes.value,
                "aggregate_maximum": context.limits.snapshot_bytes.value,
                "token": token.hex(),
                "receipt": profile_receipt,
                "identity": _identity_payload(context.identity),
                "channel_id": context.channel_id,
                "implementation_id": context.implementation_id,
                "implementation_digest": bundle.digest,
                "source_digest": bundle.source_digest,
                "implementation_root": "/implementation",
                "read_canary": "/host-canary",
                "session_id": session_id,
                "execution_plan": self.settings.execution_plan_fingerprint,
                "backend": _CONTAINER_BACKEND,
                "workspace_view": root_path,
                "private_view": _CONTAINER_PRIVATE,
                "context_lifetime": context.context_lifetime_id,
                "protocol": context.identity.protocol_id.value,
                "persistent_lease": context.identity.persistent_lease_id,
                "filesystem": context.identity.filesystem_id,
                "mount": context.identity.mount_id,
            }
            encoded = b64encode(
                dumps(config, separators=(",", ":")).encode()
            ).decode()
            create = (
                "docker",
                "create",
                "--interactive",
                "--name",
                container_name,
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "64",
                "--memory",
                "256m",
                "--tmpfs",
                "/private:rw,nosuid,nodev,noexec,size=16m",
                "--mount",
                "type=volume,src="
                + volume_name
                + ",dst=/workspace,volume-nocopy",
                "--mount",
                "type=bind,src="
                + str(bundle.root)
                + ",dst=/implementation,readonly",
                "--env",
                "AVALAN_SANDBOX_PATCH_SESSION=" + encoded,
                self.settings.image.reference,
                "python3",
                "-I",
                "-c",
                _worker_bootstrap(),
                "/implementation",
            )
            seed_ownership = (
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "CHOWN",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                "32",
                "--memory",
                "64m",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,noexec,size=1m",
                "--mount",
                "type=volume,src=" + volume_name + ",dst=/workspace",
                self.settings.image.reference,
                "python3",
                "-I",
                "-c",
                _CONTAINER_SEED_OWNERSHIP_BOOTSTRAP,
            )
            created = False
            try:
                self._volume_name = volume_name
                self._volume_resource_digest = resource_digest
                self._volume_owner_receipt = owner_receipt
                local_claim = await self._has_local_volume_claim(
                    volume_name, resource_digest, owner_receipt
                )
                guard_name: str | None = None
                if not local_claim:
                    guard_name = await self._acquire_volume_guard(
                        volume_name, resource_digest, owner_receipt
                    )
                    self._volume_guard_name = guard_name
                inspected = await _docker_output(
                    ("docker", "volume", "inspect", volume_name), False
                )
                if inspected is None:
                    await _docker_output(
                        (
                            "docker",
                            "volume",
                            "create",
                            "--label",
                            _volume_labels(resource_digest, owner_receipt)[0],
                            "--label",
                            _volume_labels(resource_digest, owner_receipt)[1],
                            volume_name,
                        )
                    )
                    created = True
                    inspected = await _docker_output(
                        (
                            "docker",
                            "volume",
                            "inspect",
                            volume_name,
                        )
                    )
                    assert inspected is not None
                if not _owned_volume_matches(
                    inspected, resource_digest, owner_receipt
                ):
                    raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
                await self._claim_volume(
                    volume_name,
                    resource_digest,
                    owner_receipt,
                    created=created,
                    guard_name=guard_name,
                )
                container_id = await _docker_output(create)
                assert container_id is not None
                if created:
                    await _docker_output(
                        (
                            "docker",
                            "cp",
                            str(self.settings.seed_root) + "/.",
                            container_id.strip() + ":/workspace",
                        )
                    )
                    await _docker_output(seed_ownership)
                process = await create_subprocess_exec(
                    "docker",
                    "start",
                    "--attach",
                    "--interactive",
                    container_id.strip(),
                    stdin=PIPE,
                    stdout=PIPE,
                    stderr=DEVNULL,
                    env=_docker_env(),
                    close_fds=True,
                )
            except BaseException as error:
                await _docker_output(
                    ("docker", "rm", "--force", container_name), False
                )
                bundle.close()
                await self._release_volume_attachment()
                if created:
                    await self._cleanup_new_volume(
                        volume_name, resource_digest, owner_receipt
                    )
                if isinstance(error, TargetInspectionError):
                    raise
                raise TargetInspectionError(
                    TargetErrorCode.CAPABILITY_UNAVAILABLE
                ) from error
            self._process = process
            self._container_id = container_id.strip()
            self._token = token
            self._receipt = profile_receipt
            self._session_id = session_id
            self._implementation_digest_value = bundle.digest
            self._bundle = bundle
            self._sequence = 0
            try:
                canary = await self._request_locked("canary", {})
                if (
                    set(canary)
                    != {"pid", "outside_read_denied", "metadata_probe"}
                    or canary["outside_read_denied"] is not True
                    or not isinstance(canary["pid"], int)
                    or not isinstance(canary["metadata_probe"], str)
                ):
                    raise TargetInspectionError(
                        TargetErrorCode.CAPABILITY_UNAVAILABLE
                    )
                command_digest = sha256(
                    dumps(
                        (create, seed_ownership), separators=(",", ":")
                    ).encode()
                ).hexdigest()
                policy_digest = sha256(
                    dumps(
                        (create[2:25], seed_ownership[2:23]),
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                self._attestation = _RuntimeAttestation(
                    command_digest,
                    policy_digest,
                    sha256(
                        (session_id + "\x00" + str(canary["pid"])).encode()
                    ).hexdigest(),
                    sha256(
                        dumps(
                            canary, separators=(",", ":"), sort_keys=True
                        ).encode()
                    ).hexdigest(),
                )
                self._root = _root_from_payload(
                    (await self._request_locked("witness", {})).get("root")
                )
                return self._root, await self._runtime_receipt_locked()
            except BaseException:
                await self._reap()
                await self._release_volume_attachment()
                if created:
                    await self._cleanup_new_volume(
                        volume_name, resource_digest, owner_receipt
                    )
                raise

    async def inspect(
        self, paths: tuple[LogicalPath, ...], root: RootWitness
    ) -> tuple[TargetSnapshot, ...]:
        """Inspect only through the container root witness."""
        response = await self._request(
            "inspect",
            {
                "paths": [path.value for path in paths],
                "root": _root_payload(root),
            },
        )
        snapshots = response.get("snapshots")
        if not isinstance(snapshots, list):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return tuple(_snapshot_from_worker(item) for item in snapshots)

    async def commit(
        self,
        command: SealedCommitCommand,
        validator: RootedCommandAuthorityValidator,
    ) -> WorkerReport:
        """Commit one sealed transaction through the narrow worker only."""
        root = self._root
        receipt = self._receipt
        session_id = self._session_id
        implementation = self._implementation_digest_value
        if (
            root is None
            or receipt is None
            or session_id is None
            or implementation is None
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        context = self.settings.context
        payload = _sealed_command_payload(
            command,
            context.identity,
            context.cwd,
            root,
            {
                "backend": _CONTAINER_BACKEND,
                "execution_plan": self.settings.execution_plan_fingerprint,
                "workspace_view": _container_root_path(self.settings),
                "private_view": _CONTAINER_PRIVATE,
                "channel": context.channel_id,
                "protocol": context.identity.protocol_id.value,
                "implementation": context.implementation_id,
                "implementation_digest": implementation,
                "receipt": receipt,
                "session": session_id,
                "context_lifetime": context.context_lifetime_id,
                "persistent_lease": context.identity.persistent_lease_id,
                "filesystem": context.identity.filesystem_id,
                "mount": context.identity.mount_id,
            },
        )
        response = await self._request(
            "commit", payload, command=command, validator=validator
        )
        return _report_from_payload(command, response)

    async def close(self) -> None:
        """Fence the worker while retaining only its persistent workspace."""
        self._closed = True
        await self._reap()
        await self._release_volume_attachment()

    async def dispose(self) -> None:
        """Remove the owned persistent volume after its lease is retired."""
        await self.close()
        await self._dispose_owned_volume()

    async def _runtime_receipt_locked(self) -> ContainerRuntimeReceipt:
        """Return the current complete receipt while holding the lock."""
        root = self._root
        receipt = self._receipt
        token = self._token
        session = self._session_id
        attestation = self._attestation
        if (
            root is None
            or receipt is None
            or token is None
            or session is None
            or attestation is None
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        context = self.settings.context
        worker = EphemeralWorkerWitness(
            context.channel_id,
            context.implementation_id
            + "-"
            + sha256(
                (receipt + attestation.child_process_identity).encode()
            ).hexdigest()[:16],
            "fence-"
            + sha256(
                (receipt + context.identity.persistent_lease_id).encode()
            ).hexdigest()[:32],
        )
        primitives = _FUTURE_MUTATION_PRIMITIVES | _container_primitives()
        receipts = {
            primitive: (
                sha256(
                    (
                        receipt
                        + "\x00"
                        + primitive.value
                        + "\x00"
                        + attestation.canary_receipt
                    ).encode()
                ).hexdigest()
            )
            for primitive in primitives
        }
        return ContainerRuntimeReceipt(
            session,
            receipt,
            root,
            worker,
            receipts,
            attestation.runtime_command_digest,
            attestation.backend_policy_digest,
            attestation.child_process_identity,
            attestation.canary_receipt,
        )

    async def _request(
        self,
        kind: str,
        body: Mapping[str, object],
        *,
        command: SealedCommitCommand | None = None,
        validator: RootedCommandAuthorityValidator | None = None,
    ) -> Mapping[str, object]:
        """Serialize one authenticated request over the sole worker channel."""
        async with self._lock:
            return await self._request_locked(
                kind, body, command=command, validator=validator
            )

    async def _request_locked(
        self,
        kind: str,
        body: Mapping[str, object],
        *,
        command: SealedCommitCommand | None = None,
        validator: RootedCommandAuthorityValidator | None = None,
    ) -> Mapping[str, object]:
        """Exchange one replay-resistant message with the child container."""
        process = self._process
        token = self._token
        receipt = self._receipt
        if (
            process is None
            or token is None
            or receipt is None
            or process.returncode is not None
        ):
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        if process.stdin is None or process.stdout is None:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        self._sequence += 1
        context = self.settings.context
        payload = {
            "version": _MESSAGE_VERSION,
            "sequence": self._sequence,
            "kind": kind,
            "receipt": receipt,
            "identity": _identity_payload(context.identity),
            "channel_id": context.channel_id,
            "implementation_id": context.implementation_id,
            "body": body,
        }
        raw = dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        line = (
            dumps(
                {
                    "payload": payload,
                    "mac": digest(token, raw, "sha256").hex(),
                },
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        if len(line) > _MAX_MESSAGE_BYTES:
            raise TargetInspectionError(TargetErrorCode.LIMIT_EXCEEDED)
        process.stdin.write(line)
        try:
            await wait_for(process.stdin.drain(), timeout=_PROCESS_IO_SECONDS)
        except TimeoutError as error:
            await self._reap()
            raise TargetInspectionError(
                TargetErrorCode.WORKER_UNAVAILABLE
            ) from error
        while True:
            try:
                response_line = await wait_for(
                    process.stdout.readline(), timeout=_PROCESS_IO_SECONDS
                )
            except TimeoutError as error:
                await self._reap()
                raise TargetInspectionError(
                    TargetErrorCode.WORKER_UNAVAILABLE
                ) from error
            if not response_line or len(response_line) > _MAX_MESSAGE_BYTES:
                await self._reap()
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            response = _response_payload(response_line, token, payload)
            if set(response) != {"control", "effect"}:
                return response
            effect = response["effect"]
            if (
                response["control"] != "fence"
                or type(effect) is not int
                or effect <= 0
                or command is None
                or validator is None
            ):
                await self._reap()
                raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
            allowed = await validator.is_rooted_command_current(command)
            permit_payload = {
                **payload,
                "kind": "fence_permit",
                "body": {"effect": effect, "allowed": allowed},
            }
            permit_raw = dumps(
                permit_payload, separators=(",", ":"), sort_keys=True
            ).encode()
            process.stdin.write(
                dumps(
                    {
                        "payload": permit_payload,
                        "mac": digest(token, permit_raw, "sha256").hex(),
                    },
                    separators=(",", ":"),
                ).encode()
                + b"\n"
            )
            try:
                await wait_for(
                    process.stdin.drain(), timeout=_PROCESS_IO_SECONDS
                )
            except TimeoutError as error:
                await self._reap()
                raise TargetInspectionError(
                    TargetErrorCode.WORKER_UNAVAILABLE
                ) from error

    async def _reap(self) -> None:
        """Terminate the worker and erase every session bearer."""
        async with self._reap_lock:
            process = self._process
            container = self._container_id
            self._process = None
            self._container_id = None
            self._token = None
            self._receipt = None
            self._root = None
            self._session_id = None
            self._implementation_digest_value = None
            self._attestation = None
            self._sequence = 0
            bundle = self._bundle
            self._bundle = None
            if process is not None:
                try:
                    if process.returncode is None:
                        process.terminate()
                    await wait_for(
                        process.wait(), timeout=_PROCESS_REAP_SECONDS
                    )
                except TimeoutError:
                    if process.returncode is None:
                        process.kill()
                    await wait_for(
                        process.wait(), timeout=_PROCESS_REAP_SECONDS
                    )
            if container is not None:
                await _docker_output(
                    ("docker", "rm", "--force", container), False
                )
            if bundle is not None:
                bundle.close()


def _worker_bootstrap() -> str:
    """Return immutable worker startup code with no workspace import path."""
    return "".join(
        (
            "import sys,types;root=sys.argv[1];sys.path.insert(0,root);",
            "package=types.ModuleType('avalan');",
            "package.__path__=[root+'/avalan'];",
            "sys.modules['avalan']=package;",
            "patch=types.ModuleType('avalan.patch');",
            "patch.__path__=[root+'/avalan/patch'];",
            "sys.modules['avalan.patch']=patch;",
            "model=types.ModuleType('avalan.model');",
            "model.__path__=[root+'/avalan/model'];",
            "sys.modules['avalan.model']=model;",
            "from avalan.patch.sandbox_worker import main;",
            "raise SystemExit(main())",
        )
    )


async def _docker_output(
    command: tuple[str, ...],
    required: bool = True,
    timeout: float = _DOCKER_COMMAND_TIMEOUT_SECONDS,
) -> str | None:
    """Run one fixed Docker CLI command without forwarding host secrets."""
    if (
        type(timeout) is not float
        or timeout <= 0.0
        or timeout > _DOCKER_BUILD_TIMEOUT_SECONDS
    ):
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    process: Process | None = None
    try:
        process = await create_subprocess_exec(
            *command,
            stdin=DEVNULL,
            stdout=PIPE,
            stderr=DEVNULL,
            env=_docker_env(),
            close_fds=True,
        )
        output, _ = await wait_for(process.communicate(), timeout=timeout)
    except TimeoutError as timeout_error:
        failure: BaseException = timeout_error
        if process is not None:
            try:
                if process.returncode is None:
                    process.terminate()
                await wait_for(
                    process.wait(), timeout=_PROCESS_REAP_SECONDS
                )
            except TimeoutError:
                if process.returncode is None:
                    process.kill()
                try:
                    await wait_for(
                        process.wait(), timeout=_PROCESS_REAP_SECONDS
                    )
                except TimeoutError as reap_error:
                    failure = reap_error
        if not required:
            return None
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from failure
    except OSError as error:
        if not required:
            return None
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from error
    if process.returncode != 0:
        if not required:
            return None
        raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
    try:
        return output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise TargetInspectionError(
            TargetErrorCode.CAPABILITY_UNAVAILABLE
        ) from error


@dataclass(slots=True, repr=False)
class ContainerPatchRuntime:
    """Own the narrow Docker mutation service for one persistent lease."""

    settings: ContainerPatchRuntimeSettings
    _profile_guard: ContainerPatchRuntimeSettings = field(
        init=False, repr=False
    )
    _process: _ContainerRuntimeProcess = field(init=False, repr=False)
    _scope: ResolvedMutationScope | None = field(
        default=None, init=False, repr=False
    )
    _receipt: ContainerRuntimeReceipt | None = field(
        default=None, init=False, repr=False
    )
    _receipt_guard: ContainerRuntimeReceipt | None = field(
        default=None, init=False, repr=False
    )
    _endpoint: object | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    @property
    def profile(self) -> ContainerPatchRuntimeContext:
        """Return the immutable host-issued context profile."""
        return self.settings.context

    def __post_init__(self) -> None:
        """Freeze the exact selected service profile before activation."""
        self._profile_guard = replace(self.settings)
        self._process = _ContainerRuntimeProcess(self.settings)

    async def resolve(
        self, selection: ScopeSelection
    ) -> ResolvedMutationScope:
        """Start and bind only the selected container context."""
        if selection.context_kind is not ContextKind.CONTAINER:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        async with self._lock:
            if self.settings != self._profile_guard:
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            if self._scope is not None:
                return self._scope
            root, receipt = await self._process.start()
            scope = ResolvedMutationScope(
                ContextKind.CONTAINER,
                self.settings.context.identity,
                self.settings.context.cwd,
                self.settings.context.limits,
                frozenset(
                    (
                        Capability.READ_FOR_MUTATION,
                        Capability.OBSERVE_MUTATION_PRECONDITIONS,
                    )
                ),
                frozenset(
                    (
                        TargetPrimitive.ROOTED_CONTAINMENT,
                        TargetPrimitive.NOFOLLOW_INSPECTION,
                        TargetPrimitive.REGULAR_FILE_IDENTITY,
                        TargetPrimitive.BOUNDED_READ,
                    )
                ),
                root,
                receipt.worker,
                None,
                (),
            )
            self._scope = scope
            self._receipt = receipt
            self._receipt_guard = replace(receipt)
            return scope

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Advertise only primitive receipts proven inside the service."""
        receipt = await self._require_scope(scope)
        primitives = (
            scope.primitives
            | _FUTURE_MUTATION_PRIMITIVES
            | _container_primitives()
        )
        probes = tuple(
            PrimitiveProbe(
                item, ProbeState.AVAILABLE, receipt.primitive_receipts[item]
            )
            for item in sorted(
                _FUTURE_MUTATION_PRIMITIVES | _container_primitives(),
                key=lambda item: item.value,
            )
        )
        return TargetHandshake(
            self.settings.context.identity,
            primitives,
            (),
            probes,
            LocalPlatformProfile.LINUX,
            worker=scope.worker,
        )

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Read the selected volume through the authenticated worker only."""
        await self._require_scope(request.scope)
        assert request.scope.root_witness is not None
        return InspectionBatch(
            await self._process.inspect(
                request.paths, request.scope.root_witness
            )
        )

    async def worker(
        self, scope: ResolvedMutationScope
    ) -> RootedSandboxCommitWorker:
        """Mint a coordinator-only worker after a complete handshake."""
        await self.handshake(scope)
        endpoint = self._endpoint
        if type(endpoint) is not _SandboxEndpoint:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return _sandbox_worker_for_endpoint(_rooted_sandbox_endpoint(endpoint))

    def _bind_sandbox_endpoint(
        self, scope: ResolvedMutationScope
    ) -> _SandboxEndpoint:
        """Return the private worker channel after loader issuance."""
        if (
            scope is not self._scope
            or self._receipt is None
            or self._receipt != self._receipt_guard
            or self.settings != self._profile_guard
        ):
            raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
        endpoint = self._endpoint
        if endpoint is None:
            endpoint = _SandboxEndpoint(self, scope)
            self._endpoint = endpoint
        if type(endpoint) is not _SandboxEndpoint:
            raise TargetInspectionError(TargetErrorCode.WORKER_UNAVAILABLE)
        return endpoint

    async def close(self) -> None:
        """Fence live work without deleting the persistent volume lease."""
        async with self._lock:
            await self._process.close()
            self._scope = None
            self._receipt = None
            self._receipt_guard = None
            self._endpoint = None

    async def dispose(self) -> None:
        """Retire the exact owned volume after the logical lease is closed."""
        async with self._lock:
            await self._process.dispose()
            self._scope = None
            self._receipt = None
            self._receipt_guard = None
            self._endpoint = None

    async def _require_scope(
        self, scope: ResolvedMutationScope
    ) -> ContainerRuntimeReceipt:
        """Reject stale, copied, retargeted, or closed scope handles."""
        async with self._lock:
            receipt = self._receipt
            if (
                self.settings != self._profile_guard
                or scope is not self._scope
                or receipt is None
                or receipt != self._receipt_guard
                or scope.worker is not receipt.worker
                or scope.root_witness != receipt.root
            ):
                raise TargetInspectionError(TargetErrorCode.WITNESS_STALE)
            return receipt


@dataclass(frozen=True, slots=True)
class ContainerInspectionTarget:
    """Expose inspection only through the selected container runtime."""

    runtime: ContainerPatchRuntime

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return the runtime-owned target handshake."""
        return await self.runtime.handshake(scope)

    async def inspect(self, request: InspectionRequest) -> InspectionBatch:
        """Inspect only through the sealed service worker."""
        return await self.runtime.inspect(request)


@dataclass(frozen=True, slots=True)
class ContainerPatchTarget:
    """Expose only an opaque container commit worker capability."""

    runtime: ContainerPatchRuntime

    async def handshake(self, scope: ResolvedMutationScope) -> TargetHandshake:
        """Return the selected container mutation handshake."""
        return await self.runtime.handshake(scope)

    async def worker(
        self, scope: ResolvedMutationScope
    ) -> RootedSandboxCommitWorker:
        """Return the coordinator-owned opaque container worker."""
        return await self.runtime.worker(scope)


@dataclass(frozen=True, slots=True, repr=False)
class ContainerPatchRuntimeBinder:
    """Bind public patch tools only to the selected Docker test profile."""

    runtime: ContainerPatchRuntime
    configuration: SandboxPatchServiceConfiguration
    policy: TrustedPatchPolicy
    approval: PatchApprovalBinding
    coordinator: PatchCoordinatorBinding
    persistence: PatchPersistenceBinding

    def __post_init__(self) -> None:
        """Require one shared durable store and exact production services."""
        if (
            type(self.runtime) is not ContainerPatchRuntime
            or type(self.configuration) is not SandboxPatchServiceConfiguration
            or type(self.policy) is not TrustedPatchPolicy
            or type(self.approval) is not PatchApprovalBinding
            or type(self.coordinator) is not PatchCoordinatorBinding
            or type(self.persistence) is not PatchPersistenceBinding
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        store = self.coordinator.durable_store
        if store is None or self.persistence.durable_store is not store:
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)

    @classmethod
    def from_shared_store(
        cls,
        settings: ContainerPatchRuntimeSettings,
        configuration: SandboxPatchServiceConfiguration,
        policy: TrustedPatchPolicy,
        approval: PatchApprovalBinding,
        binding: DurablePatchStoreBinding,
    ) -> "ContainerPatchRuntimeBinder":
        """Bind the container lease to the shared durable coordinator store."""
        if (
            type(settings) is not ContainerPatchRuntimeSettings
            or type(binding) is not DurablePatchStoreBinding
        ):
            raise TargetInspectionError(TargetErrorCode.CAPABILITY_UNAVAILABLE)
        return cls(
            settings.create_runtime(),
            configuration,
            policy,
            approval,
            PatchCoordinatorBinding(True, binding.store),
            PatchPersistenceBinding(True, binding.store),
        )

    async def bind(self) -> PatchRuntimeBinding:
        """Start the selected Docker service and bind its full handshake."""
        scope = await self.runtime.resolve(
            ScopeSelection(ContextKind.CONTAINER)
        )
        target = ContainerPatchTarget(self.runtime)
        inspection = ContainerInspectionTarget(self.runtime)
        handshake = await target.handshake(scope)
        store = self.coordinator.durable_store
        assert store is not None
        service = SandboxPatchSdkService(
            self.runtime,
            scope,
            handshake,
            inspection,
            store,
            self.policy,
            self.configuration,
        )
        return PatchRuntimeBinding(
            scope,
            handshake,
            self.policy,
            self.approval,
            self.coordinator,
            self.persistence,
            service,
            (service,),
        )
