from .backend import (
    ContainerAsyncBackend,
    ContainerBackendContainer,
    ContainerBackendDiagnostic,
    ContainerBackendDiagnosticCode,
    ContainerBackendError,
    ContainerBackendImageResolution,
    ContainerBackendInspection,
    ContainerBackendOperation,
    ContainerBackendOperationResult,
    ContainerBackendProbeResult,
    ContainerBackendStats,
    ContainerBackendStream,
    ContainerBackendStreamChunk,
    ContainerBackendWaitResult,
    container_pool_key,
)
from .conformance import (
    ContainerBackend,
)
from .output import (
    ContainerOutputContract,
    ContainerOutputDecisionType,
    ContainerOutputDiagnostic,
    ContainerOutputDiagnosticCode,
    ContainerOutputValidationResult,
)
from .settings import (
    ContainerBackendCapabilities,
    ContainerBackendSupportLevel,
    ContainerBuildPolicy,
    ContainerDeviceClass,
    ContainerMountAccess,
    ContainerMountDeclaration,
    ContainerMountType,
    ContainerNetworkMode,
    ContainerPullPolicy,
    ContainerRunPlan,
)

from asyncio import (
    CancelledError,
    StreamReader,
    create_subprocess_exec,
    create_task,
    gather,
)
from asyncio.subprocess import PIPE
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from json import loads as json_loads
from pathlib import Path
from shutil import which
from types import MappingProxyType
from typing import Protocol, cast, final
from uuid import uuid4

_APPLE_BACKEND = ContainerBackend.APPLE_CONTAINER
_APPLE_PLATFORM = "linux/arm64"
_MAX_CAPTURE_BYTES = 1048576
_MOUNTABLE_TYPES = (
    ContainerMountType.WORKSPACE,
    ContainerMountType.OUTPUT,
)
_SHARED_MOUNT_PREFIXES = ("/Users/", "/Volumes/", "/private/", "/tmp/")
_BACKEND_ERROR_MARKERS = (
    "api server",
    "connection invalid",
    "container service",
    "container-apiserver",
    "invalid state",
    "no such container",
    "not found",
    "xpc connection",
)
_START_BACKEND_ERROR_MARKERS = (
    "api server",
    "connection invalid",
    "container service",
    "container-apiserver",
    "invalid state",
    "xpc connection",
)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class AppleContainerCommandResult:
    args: Sequence[str]
    return_code: int
    stdout: bytes = b""
    stderr: bytes = b""

    def __post_init__(self) -> None:
        assert not isinstance(self.args, str | bytes)
        for arg in self.args:
            assert isinstance(arg, str) and arg
        assert isinstance(self.return_code, int)
        assert isinstance(self.stdout, bytes)
        assert isinstance(self.stderr, bytes)
        object.__setattr__(self, "args", tuple(self.args))

    @property
    def ok(self) -> bool:
        return self.return_code == 0


class AppleContainerCommandRunner(Protocol):
    def available(self) -> bool:
        raise NotImplementedError  # pragma: no cover

    async def run(
        self,
        args: Sequence[str],
    ) -> AppleContainerCommandResult:
        raise NotImplementedError  # pragma: no cover


@final
class AppleContainerSubprocessRunner:
    def __init__(self, executable: str = "container") -> None:
        assert isinstance(executable, str) and executable
        self._executable = executable

    def available(self) -> bool:
        return which(self._executable) is not None

    async def run(
        self,
        args: Sequence[str],
    ) -> AppleContainerCommandResult:
        assert not isinstance(args, str | bytes)
        resolved_args = tuple(args)
        process = await create_subprocess_exec(
            self._executable,
            *resolved_args,
            stdout=PIPE,
            stderr=PIPE,
        )
        assert isinstance(process.stdout, StreamReader)
        assert isinstance(process.stderr, StreamReader)
        stdout_task = create_task(
            _read_limited(process.stdout, _MAX_CAPTURE_BYTES)
        )
        stderr_task = create_task(
            _read_limited(process.stderr, _MAX_CAPTURE_BYTES)
        )
        try:
            return_code = await process.wait()
            stdout, stderr = await gather(stdout_task, stderr_task)
        except CancelledError:
            process.kill()
            await process.wait()
            stdout_task.cancel()
            stderr_task.cancel()
            await gather(stdout_task, stderr_task, return_exceptions=True)
            raise
        return AppleContainerCommandResult(
            args=resolved_args,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )


@dataclass(slots=True)
class _AppleContainerState:
    plan: ContainerRunPlan
    image_reference: str | None = None
    start_result: AppleContainerCommandResult | None = None
    removed: bool = False


async def _read_limited(reader: StreamReader, limit: int) -> bytes:
    assert isinstance(reader, StreamReader)
    assert limit > 0
    chunks: list[bytes] = []
    remaining = limit
    while True:
        chunk = await reader.read(65536)
        if not chunk:
            return b"".join(chunks)
        if remaining > 0:
            chunks.append(chunk[:remaining])
            remaining = max(0, remaining - len(chunk))


@final
class AppleContainerBackend(ContainerAsyncBackend):
    def __init__(
        self,
        runner: AppleContainerCommandRunner | None = None,
    ) -> None:
        self._runner = runner or AppleContainerSubprocessRunner()
        self._containers: dict[str, _AppleContainerState] = {}

    async def probe(self) -> ContainerBackendProbeResult:
        if not self._runner.available():
            return ContainerBackendProbeResult(
                backend=_APPLE_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        "Apple container CLI was not found on PATH",
                        retryable=True,
                    ),
                ),
            )
        try:
            result = await self._runner.run(
                (
                    "system",
                    "status",
                    "--format",
                    "json",
                )
            )
        except OSError:
            return ContainerBackendProbeResult(
                backend=_APPLE_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        "Apple container CLI failed to start",
                        retryable=True,
                    ),
                ),
            )
        if not result.ok:
            return ContainerBackendProbeResult(
                backend=_APPLE_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        _command_failure_message(
                            "Apple container service is unavailable",
                            result,
                        ),
                        retryable=True,
                    ),
                ),
            )
        return ContainerBackendProbeResult(
            backend=_APPLE_BACKEND,
            available=True,
            capabilities=apple_container_capabilities(),
        )

    async def resolve_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendImageResolution:
        diagnostics = _plan_diagnostics(
            plan,
            operation=ContainerBackendOperation.IMAGE_RESOLUTION,
        )
        assert plan.image.digest is not None
        digest = plan.image.digest
        if not diagnostics:
            resolution = await self._verified_local_image(
                plan,
                operation=ContainerBackendOperation.IMAGE_RESOLUTION,
                allow_missing=(
                    plan.image.pull_policy is not ContainerPullPolicy.NEVER
                ),
            )
            if resolution is not None:
                _, digest, image_diagnostics = resolution
                diagnostics = image_diagnostics
        return ContainerBackendImageResolution(
            reference=plan.image.reference,
            digest=digest,
            platform=plan.image.platform,
            diagnostics=diagnostics,
        )

    async def pull_image(
        self,
        plan: ContainerRunPlan,
        image: ContainerBackendImageResolution,
    ) -> ContainerBackendOperationResult:
        assert isinstance(image, ContainerBackendImageResolution)
        if plan.image.pull_policy is ContainerPullPolicy.NEVER:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.PULL_DENIED,
                    ContainerBackendOperation.IMAGE_PULL,
                    "image pull is disabled by policy",
                )
            )
        result = await self._run_checked(
            (
                "image",
                "pull",
                "--platform",
                _APPLE_PLATFORM,
                "--progress",
                "plain",
                plan.image.reference,
            ),
            operation=ContainerBackendOperation.IMAGE_PULL,
            failure_code=ContainerBackendDiagnosticCode.PULL_FAILED,
            failure_message="Apple container image pull failed",
            retryable=True,
        )
        resolution = await self._verified_local_image(
            plan,
            operation=ContainerBackendOperation.IMAGE_PULL,
            allow_missing=False,
        )
        if resolution is not None:
            _, _, diagnostics = resolution
            if diagnostics:
                return ContainerBackendOperationResult(
                    operation=ContainerBackendOperation.IMAGE_PULL,
                    diagnostics=diagnostics,
                    metadata={"return_code": str(result.return_code)},
                )
        return ContainerBackendOperationResult(
            operation=ContainerBackendOperation.IMAGE_PULL,
            metadata={"return_code": str(result.return_code)},
        )

    async def build_image(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendOperationResult:
        raise ContainerBackendError(
            _diagnostic(
                ContainerBackendDiagnosticCode.BUILD_DENIED,
                ContainerBackendOperation.IMAGE_BUILD,
                "Apple container backend does not build shell images",
            )
        )

    async def create(
        self,
        plan: ContainerRunPlan,
    ) -> ContainerBackendContainer:
        diagnostics = _plan_diagnostics(
            plan,
            operation=ContainerBackendOperation.CREATE,
        )
        if diagnostics:
            raise ContainerBackendError(diagnostics[0])
        container_id = _container_name(plan)
        resolution = await self._verified_local_image(
            plan,
            operation=ContainerBackendOperation.CREATE,
            allow_missing=False,
        )
        assert resolution is not None
        local_reference, _, image_diagnostics = resolution
        if image_diagnostics:
            raise ContainerBackendError(image_diagnostics[0])
        assert plan.image.digest is not None
        create_reference = await self._create_verified_image_tag(
            source_reference=local_reference,
            target_reference=_verified_image_reference(container_id),
            expected_digest=plan.image.digest,
            operation=ContainerBackendOperation.CREATE,
        )
        try:
            await self._run_checked(
                _create_args(plan, container_id, create_reference),
                operation=ContainerBackendOperation.CREATE,
                failure_code=ContainerBackendDiagnosticCode.CREATE_FAILED,
                failure_message="Apple container create failed",
                retryable=True,
            )
        except ContainerBackendError:
            await self._delete_image_reference(create_reference)
            raise
        self._containers[container_id] = _AppleContainerState(
            plan=plan,
            image_reference=create_reference,
        )
        return ContainerBackendContainer(
            container_id=container_id,
            backend=_APPLE_BACKEND,
            plan_fingerprint=container_pool_key(plan),
        )

    async def start(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        state = self._state(container, ContainerBackendOperation.START)
        try:
            result = await self._runner.run(
                (
                    "start",
                    "--attach",
                    container.container_id,
                )
            )
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    ContainerBackendOperation.START,
                    "Apple container CLI failed to start",
                    retryable=True,
                )
            ) from error
        if not result.ok and _looks_like_start_backend_failure(result):
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.START_FAILED,
                    ContainerBackendOperation.START,
                    _command_failure_message(
                        "Apple container start failed",
                        result,
                    ),
                    retryable=True,
                )
            )
        state.start_result = result
        return ContainerBackendOperationResult(
            operation=ContainerBackendOperation.START,
            metadata={"exit_code": str(result.return_code)},
        )

    async def attach(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        self._state(container, ContainerBackendOperation.ATTACH)
        return ContainerBackendOperationResult(
            operation=ContainerBackendOperation.ATTACH,
        )

    async def stream(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStreamChunk, ...]:
        state = self._state(container, ContainerBackendOperation.STREAM)
        result = _required_start_result(
            state,
            ContainerBackendOperation.STREAM,
        )
        chunks: list[ContainerBackendStreamChunk] = []
        if result.stdout:
            chunks.append(
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.STDOUT,
                    content=result.stdout,
                    sequence=0,
                )
            )
        if result.stderr:
            chunks.append(
                ContainerBackendStreamChunk(
                    stream=ContainerBackendStream.STDERR,
                    content=result.stderr,
                    sequence=len(chunks),
                )
            )
        return tuple(chunks)

    async def wait(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendWaitResult:
        state = self._state(container, ContainerBackendOperation.WAIT)
        result = _required_start_result(
            state,
            ContainerBackendOperation.WAIT,
        )
        return ContainerBackendWaitResult(exit_code=result.return_code)

    async def inspect(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendInspection:
        state = self._state(container, ContainerBackendOperation.INSPECT)
        result = await self._run_checked(
            ("inspect", container.container_id),
            operation=ContainerBackendOperation.INSPECT,
            failure_code=ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
            failure_message="Apple container inspect failed",
            retryable=True,
        )
        inspected = _first_json_mapping(result.stdout)
        status = (
            _string_field(
                inspected,
                ("status",),
                ("state",),
                ("State", "Status"),
            )
            or "unknown"
        )
        exit_code = _int_field(
            inspected,
            ("exitCode",),
            ("exit_code",),
            ("State", "ExitCode"),
        )
        if exit_code is None and state.start_result is not None:
            exit_code = state.start_result.return_code
        return ContainerBackendInspection(
            container_id=container.container_id,
            status=status,
            exit_code=exit_code,
            metadata={"backend": _APPLE_BACKEND.value},
        )

    async def stats(
        self,
        container: ContainerBackendContainer,
    ) -> tuple[ContainerBackendStats, ...]:
        self._state(container, ContainerBackendOperation.STATS)
        try:
            result = await self._runner.run(
                (
                    "stats",
                    "--format",
                    "json",
                    "--no-stream",
                    container.container_id,
                )
            )
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    ContainerBackendOperation.STATS,
                    "Apple container CLI failed to start",
                    retryable=True,
                )
            ) from error
        if not result.ok:
            if _looks_like_backend_failure(result):
                raise ContainerBackendError(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.STATS,
                        _command_failure_message(
                            "Apple container stats failed",
                            result,
                        ),
                        retryable=True,
                    )
                )
            return ()
        return _stats_from_json(result.stdout)

    async def stop(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._cleanup_command(
            container,
            ContainerBackendOperation.STOP,
            ("stop", "--time", "1", container.container_id),
        )

    async def kill(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        return await self._cleanup_command(
            container,
            ContainerBackendOperation.KILL,
            ("kill", container.container_id),
        )

    async def remove(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        result = await self._cleanup_command(
            container,
            ContainerBackendOperation.REMOVE,
            ("rm", "--force", container.container_id),
        )
        if result.ok:
            state = self._containers.get(container.container_id)
            if state is not None:
                if not state.removed and state.image_reference is not None:
                    await self._delete_image_reference(state.image_reference)
                state.removed = True
        return result

    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        self._state(container, ContainerBackendOperation.COPY_OUTPUTS)
        assert isinstance(contract, ContainerOutputContract)
        return ContainerOutputValidationResult(
            decision=ContainerOutputDecisionType.REJECT,
            contract=contract,
            diagnostics=(
                ContainerOutputDiagnostic(
                    code=ContainerOutputDiagnosticCode.CONTRACT_DISABLED,
                    path=".",
                    message=(
                        "Apple container backend does not support output copy"
                    ),
                ),
            ),
        )

    async def cleanup(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        result = await self.remove(container)
        return ContainerBackendOperationResult(
            operation=ContainerBackendOperation.CLEANUP,
            diagnostics=result.diagnostics,
            metadata=result.metadata,
        )

    async def _cleanup_command(
        self,
        container: ContainerBackendContainer,
        operation: ContainerBackendOperation,
        args: Sequence[str],
    ) -> ContainerBackendOperationResult:
        state = self._state(container, operation)
        if operation is ContainerBackendOperation.REMOVE and state.removed:
            return ContainerBackendOperationResult(operation=operation)
        try:
            result = await self._runner.run(args)
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    operation,
                    "Apple container CLI failed to start",
                    retryable=True,
                )
            ) from error
        diagnostics: tuple[ContainerBackendDiagnostic, ...] = ()
        if not result.ok:
            diagnostics = (
                _diagnostic(
                    ContainerBackendDiagnosticCode.CLEANUP_FAILED,
                    operation,
                    _command_failure_message(
                        "Apple container cleanup command failed",
                        result,
                    ),
                    retryable=True,
                ),
            )
        return ContainerBackendOperationResult(
            operation=operation,
            diagnostics=diagnostics,
            metadata={"return_code": str(result.return_code)},
        )

    async def _run_checked(
        self,
        args: Sequence[str],
        *,
        operation: ContainerBackendOperation,
        failure_code: ContainerBackendDiagnosticCode,
        failure_message: str,
        retryable: bool = False,
    ) -> AppleContainerCommandResult:
        try:
            result = await self._runner.run(args)
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    operation,
                    "Apple container CLI failed to start",
                    retryable=True,
                )
            ) from error
        if not result.ok:
            raise ContainerBackendError(
                _diagnostic(
                    failure_code,
                    operation,
                    _command_failure_message(failure_message, result),
                    retryable=retryable,
                )
            )
        return result

    def _state(
        self,
        container: ContainerBackendContainer,
        operation: ContainerBackendOperation,
    ) -> _AppleContainerState:
        assert isinstance(container, ContainerBackendContainer)
        if container.backend is not _APPLE_BACKEND:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
                    operation,
                    "container belongs to a different backend",
                )
            )
        state = self._containers.get(container.container_id)
        if state is None:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    operation,
                    "Apple container state is unavailable",
                    retryable=True,
                )
            )
        return state

    async def _verified_local_image(
        self,
        plan: ContainerRunPlan,
        *,
        operation: ContainerBackendOperation,
        allow_missing: bool,
    ) -> (
        tuple[
            str,
            str,
            tuple[ContainerBackendDiagnostic, ...],
        ]
        | None
    ):
        assert plan.image.digest is not None
        last_result: AppleContainerCommandResult | None = None
        for reference in _local_image_references(plan.image.reference):
            try:
                result = await self._runner.run(
                    (
                        "image",
                        "inspect",
                        reference,
                    )
                )
            except OSError:
                return (
                    reference,
                    plan.image.digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                            operation,
                            "Apple container CLI failed to start",
                            retryable=True,
                        ),
                    ),
                )
            if not result.ok:
                last_result = result
                continue
            digest = _image_digest_from_json(result.stdout)
            if digest is None:
                return (
                    reference,
                    plan.image.digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.IMAGE_DENIED,
                            operation,
                            "Apple container image digest is unavailable",
                        ),
                    ),
                )
            if digest != plan.image.digest:
                return (
                    reference,
                    digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.IMAGE_DENIED,
                            operation,
                            "local Apple container image digest does not"
                            " match approved plan",
                        ),
                    ),
                )
            return reference, digest, ()
        if allow_missing:
            return None
        result = last_result or AppleContainerCommandResult(
            args=(),
            return_code=1,
        )
        return (
            plan.image.reference,
            plan.image.digest,
            (
                _diagnostic(
                    ContainerBackendDiagnosticCode.IMAGE_DENIED,
                    operation,
                    _command_failure_message(
                        "Apple container image is unavailable locally",
                        result,
                    ),
                ),
            ),
        )

    async def _create_verified_image_tag(
        self,
        *,
        source_reference: str,
        target_reference: str,
        expected_digest: str,
        operation: ContainerBackendOperation,
    ) -> str:
        await self._run_checked(
            ("image", "tag", source_reference, target_reference),
            operation=operation,
            failure_code=ContainerBackendDiagnosticCode.IMAGE_DENIED,
            failure_message="Apple container image tag failed",
        )
        digest, diagnostics = await self._inspect_image_reference(
            target_reference,
            expected_digest=expected_digest,
            operation=operation,
        )
        if diagnostics:
            raise ContainerBackendError(diagnostics[0])
        assert digest == expected_digest
        return target_reference

    async def _inspect_image_reference(
        self,
        reference: str,
        *,
        expected_digest: str,
        operation: ContainerBackendOperation,
    ) -> tuple[str | None, tuple[ContainerBackendDiagnostic, ...]]:
        try:
            result = await self._runner.run(("image", "inspect", reference))
        except OSError:
            return (
                None,
                (
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        operation,
                        "Apple container CLI failed to start",
                        retryable=True,
                    ),
                ),
            )
        if not result.ok:
            return (
                None,
                (
                    _diagnostic(
                        ContainerBackendDiagnosticCode.IMAGE_DENIED,
                        operation,
                        _command_failure_message(
                            "Apple container image is unavailable locally",
                            result,
                        ),
                    ),
                ),
            )
        digest = _image_digest_from_json(result.stdout)
        if digest is None:
            return (
                None,
                (
                    _diagnostic(
                        ContainerBackendDiagnosticCode.IMAGE_DENIED,
                        operation,
                        "Apple container image digest is unavailable",
                    ),
                ),
            )
        if digest != expected_digest:
            return (
                digest,
                (
                    _diagnostic(
                        ContainerBackendDiagnosticCode.IMAGE_DENIED,
                        operation,
                        "local Apple container image digest does not"
                        " match approved plan",
                    ),
                ),
            )
        return digest, ()

    async def _delete_image_reference(self, reference: str) -> None:
        try:
            await self._runner.run(("image", "rm", reference))
        except OSError:
            return None
        return None


def apple_container_capabilities() -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=_APPLE_BACKEND,
        host_os="darwin",
        guest_os="linux",
        architecture="arm64",
        runtime_name="Apple container CLI",
        support_level=ContainerBackendSupportLevel.OPT_IN,
        platform_emulation=False,
        rootless=False,
        user_namespace=False,
        build=False,
        pull=True,
        network_modes=(ContainerNetworkMode.NONE,),
        mount_types=_MOUNTABLE_TYPES,
        resource_limits=True,
        device_classes=(ContainerDeviceClass.CPU,),
        per_container_vm_isolation=True,
        vm_backed=True,
        streaming_attach=False,
        stats=False,
        lifecycle_normalization=True,
        shared_mount_prefixes=_SHARED_MOUNT_PREFIXES,
    )


def _create_args(
    plan: ContainerRunPlan,
    container_id: str,
    image_reference: str,
) -> tuple[str, ...]:
    args: list[str] = [
        "create",
        "--name",
        container_id,
        "--platform",
        _APPLE_PLATFORM,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--network",
        "none",
        "--no-dns",
        "--user",
        "1000:1000",
        "--workdir",
        plan.command.cwd,
    ]
    if plan.resources.cpu_count is not None:
        args.extend(("--cpus", str(plan.resources.cpu_count)))
    if plan.resources.memory_bytes is not None:
        args.extend(
            (
                "--memory",
                _memory_argument(plan.resources.memory_bytes),
            )
        )
    if plan.resources.pids is not None:
        args.extend(("--ulimit", f"nproc={plan.resources.pids}"))
    for mount in plan.mounts:
        args.extend(("--mount", _mount_argument(mount)))
    args.append(image_reference)
    args.extend(plan.command.argv)
    return tuple(args)


def _local_image_references(reference: str) -> tuple[str, ...]:
    references = [reference]
    local_reference = _local_image_reference(reference)
    if local_reference not in references:
        references.append(local_reference)
    return tuple(references)


def _local_image_reference(reference: str) -> str:
    image_name = reference.split("@", 1)[0]
    if _has_image_tag(image_name):
        return image_name
    return f"{image_name}:latest"


def _verified_image_reference(container_id: str) -> str:
    return f"avalan-verified:{container_id}"


def _has_image_tag(reference: str) -> bool:
    last_segment = reference.rsplit("/", 1)[-1]
    return ":" in last_segment


def _mount_argument(mount: ContainerMountDeclaration) -> str:
    assert isinstance(mount, ContainerMountDeclaration)
    assert mount.source is not None
    access = cast(ContainerMountAccess, mount.access)
    parts = [
        "type=bind",
        f"source={_host_path(mount.source)}",
        f"target={mount.target}",
    ]
    if access is ContainerMountAccess.READ:
        parts.append("readonly")
    return ",".join(parts)


def _host_path(source: str) -> str:
    return Path(source).expanduser().resolve().as_posix()


def _memory_argument(memory_bytes: int) -> str:
    assert memory_bytes > 0
    mib = max(1, (memory_bytes + 1048575) // 1048576)
    return f"{mib}M"


def _plan_diagnostics(
    plan: ContainerRunPlan,
    *,
    operation: ContainerBackendOperation,
) -> tuple[ContainerBackendDiagnostic, ...]:
    assert isinstance(plan, ContainerRunPlan)
    diagnostics: list[ContainerBackendDiagnostic] = []
    if plan.backend is not _APPLE_BACKEND:
        diagnostics.append(
            _capability_mismatch(operation, "plan backend is not apple")
        )
    if plan.image.platform != _APPLE_PLATFORM:
        diagnostics.append(
            _capability_mismatch(
                operation,
                "Apple container backend supports linux/arm64 images only",
            )
        )
    if plan.image.build_policy is not ContainerBuildPolicy.DISABLED:
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.BUILD_DENIED,
                operation,
                "Apple container backend does not build shell images",
            )
        )
    network = cast(ContainerNetworkMode, plan.network.mode)
    if network is not ContainerNetworkMode.NONE:
        diagnostics.append(
            _capability_mismatch(
                operation,
                f"network mode {network.value} is not supported",
            )
        )
    if plan.network.egress_allowlist:
        diagnostics.append(
            _capability_mismatch(
                operation,
                "network egress allowlists are not supported",
            )
        )
    if plan.environment_names:
        diagnostics.append(
            _capability_mismatch(
                operation,
                "environment inheritance is not supported",
            )
        )
    if plan.secret_names:
        diagnostics.append(
            _capability_mismatch(
                operation, "secret injection is not supported"
            )
        )
    for mount in plan.mounts:
        diagnostics.extend(_mount_diagnostics(mount, operation))
    return tuple(diagnostics)


def _mount_diagnostics(
    mount: ContainerMountDeclaration,
    operation: ContainerBackendOperation,
) -> tuple[ContainerBackendDiagnostic, ...]:
    mount_type = cast(ContainerMountType, mount.mount_type)
    access = cast(ContainerMountAccess, mount.access)
    diagnostics: list[ContainerBackendDiagnostic] = []
    if mount_type not in _MOUNTABLE_TYPES:
        diagnostics.append(
            _capability_mismatch(
                operation,
                f"mount type {mount_type.value} is not supported",
            )
        )
    if mount_type is ContainerMountType.WORKSPACE:
        if access is not ContainerMountAccess.READ:
            diagnostics.append(
                _capability_mismatch(
                    operation,
                    "workspace mount must be read-only",
                )
            )
    if mount.source is None:
        diagnostics.append(
            _capability_mismatch(operation, "mount source is required")
        )
    elif not _shared_mount_source(mount.source):
        diagnostics.append(
            _capability_mismatch(
                operation,
                "mount source is outside Apple shared mount prefixes",
            )
        )
    return tuple(diagnostics)


def _shared_mount_source(source: str) -> bool:
    path = _host_path(source)
    return any(
        path == prefix.rstrip("/") or path.startswith(prefix)
        for prefix in _SHARED_MOUNT_PREFIXES
    )


def _container_name(plan: ContainerRunPlan) -> str:
    return f"avalan-{container_pool_key(plan)[:12]}-{uuid4().hex[:8]}"


def _required_start_result(
    state: _AppleContainerState,
    operation: ContainerBackendOperation,
) -> AppleContainerCommandResult:
    if state.start_result is None:
        raise ContainerBackendError(
            _diagnostic(
                ContainerBackendDiagnosticCode.WAIT_FAILED,
                operation,
                "Apple container has not been started",
                retryable=True,
            )
        )
    return state.start_result


def _diagnostic(
    code: ContainerBackendDiagnosticCode,
    operation: ContainerBackendOperation,
    message: str,
    *,
    retryable: bool = False,
) -> ContainerBackendDiagnostic:
    return ContainerBackendDiagnostic(
        code=code,
        operation=operation,
        backend=_APPLE_BACKEND,
        message=message,
        retryable=retryable,
    )


def _capability_mismatch(
    operation: ContainerBackendOperation,
    message: str,
) -> ContainerBackendDiagnostic:
    return _diagnostic(
        ContainerBackendDiagnosticCode.CAPABILITY_MISMATCH,
        operation,
        message,
    )


def _command_failure_message(
    prefix: str,
    result: AppleContainerCommandResult,
) -> str:
    detail = (
        (result.stderr or result.stdout)
        .decode(
            "utf-8",
            errors="replace",
        )
        .strip()
    )
    if not detail:
        return f"{prefix}: exit code {result.return_code}"
    return f"{prefix}: {detail}"


def _looks_like_backend_failure(
    result: AppleContainerCommandResult,
) -> bool:
    return _contains_marker(result, _BACKEND_ERROR_MARKERS)


def _looks_like_start_backend_failure(
    result: AppleContainerCommandResult,
) -> bool:
    return _contains_marker(result, _START_BACKEND_ERROR_MARKERS)


def _contains_marker(
    result: AppleContainerCommandResult,
    markers: Sequence[str],
) -> bool:
    output = (
        (result.stderr + b"\n" + result.stdout)
        .decode(
            "utf-8",
            errors="replace",
        )
        .lower()
    )
    return any(marker in output for marker in markers)


def _first_json_mapping(raw: bytes) -> Mapping[str, object]:
    try:
        value = json_loads(raw.decode("utf-8"))
    except (JSONDecodeError, UnicodeDecodeError):
        return MappingProxyType({})
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, Mapping):
            return cast(Mapping[str, object], first)
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return MappingProxyType({})


def _stats_from_json(raw: bytes) -> tuple[ContainerBackendStats, ...]:
    try:
        value = json_loads(raw.decode("utf-8"))
    except (JSONDecodeError, UnicodeDecodeError):
        return ()
    samples = value if isinstance(value, list) else (value,)
    stats: list[ContainerBackendStats] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        mapping = cast(Mapping[str, object], sample)
        stats.append(
            ContainerBackendStats(
                cpu_nanos=_int_field(
                    mapping,
                    ("cpuNanos",),
                    ("cpu_nanos",),
                    ("cpu", "nanos"),
                )
                or 0,
                memory_bytes=_int_field(
                    mapping,
                    ("memoryBytes",),
                    ("memory_bytes",),
                    ("memory", "bytes"),
                )
                or 0,
                pids=_int_field(mapping, ("pids",), ("pidCount",)) or 0,
            )
        )
    return tuple(stats)


def _image_digest_from_json(raw: bytes) -> str | None:
    inspected = _first_json_mapping(raw)
    digest = _string_field(
        inspected,
        ("configuration", "descriptor", "digest"),
        ("descriptor", "digest"),
        ("digest",),
        ("Digest",),
        ("id",),
        ("Id",),
    )
    if digest is None:
        return None
    if digest.startswith("sha256:"):
        return digest
    if len(digest) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in digest
    ):
        return f"sha256:{digest.lower()}"
    return digest


def _string_field(
    mapping: Mapping[str, object],
    *paths: tuple[str, ...],
) -> str | None:
    for path in paths:
        value = _path_value(mapping, path)
        if isinstance(value, str) and value:
            return value
    return None


def _int_field(
    mapping: Mapping[str, object],
    *paths: tuple[str, ...],
) -> int | None:
    for path in paths:
        value = _path_value(mapping, path)
        if isinstance(value, int) and value >= 0:
            return value
    return None


def _path_value(
    mapping: Mapping[str, object],
    path: tuple[str, ...],
) -> object:
    value: object = mapping
    for part in path:
        if not isinstance(value, Mapping):
            return None
        value = cast(Mapping[str, object], value).get(part)
    return value
