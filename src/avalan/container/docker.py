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
    ContainerOutputValidationResult,
    validate_copied_outputs,
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
    Task,
    create_subprocess_exec,
    create_task,
    gather,
    shield,
    wait_for,
)
from asyncio.subprocess import PIPE
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError
from json import loads as json_loads
from pathlib import Path
from platform import machine, system
from shutil import which
from tempfile import TemporaryDirectory
from typing import Protocol, cast, final
from uuid import uuid4

_DOCKER_BACKEND = ContainerBackend.DOCKER
_DOCKER_CLEANUP_TIMEOUT_SECONDS = 15.0
_DOCKER_COMMAND_TIMEOUT_SECONDS = 30.0
_DOCKER_EXECUTION_TIMEOUT_SECONDS = 3600.0
_DOCKER_KILL_GRACE_SECONDS = 2.0
_DOCKER_PROBE_TIMEOUT_SECONDS = 10.0
_DOCKER_PULL_TIMEOUT_SECONDS = 300.0
_MAX_CAPTURE_BYTES = 1048576
_OUTPUT_COPY_SOURCE = "/outputs"
_MOUNTABLE_TYPES = (
    ContainerMountType.INPUT,
    ContainerMountType.WORKSPACE,
    ContainerMountType.SCRATCH,
    ContainerMountType.OUTPUT,
    ContainerMountType.CACHE,
)
_NETWORK_MODES = (
    ContainerNetworkMode.NONE,
    ContainerNetworkMode.LOOPBACK,
    ContainerNetworkMode.FULL,
)
_BACKEND_ERROR_MARKERS = (
    "cannot connect to the docker daemon",
    "connection refused",
    "daemon is not running",
    "docker daemon",
    "socket",
)
_ORPHAN_MARKERS = (
    "orphan",
    "permission denied",
    "removal already in progress",
)


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class DockerCommandResult:
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


class DockerCommandRunner(Protocol):
    def available(self) -> bool:
        raise NotImplementedError  # pragma: no cover

    async def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> DockerCommandResult:
        raise NotImplementedError  # pragma: no cover


@final
class DockerSubprocessRunner:
    def __init__(
        self,
        executable: str = "docker",
        *,
        default_timeout_seconds: float | None = (
            _DOCKER_COMMAND_TIMEOUT_SECONDS
        ),
    ) -> None:
        assert isinstance(executable, str) and executable
        if default_timeout_seconds is not None:
            assert default_timeout_seconds > 0
        self._executable = executable
        self._default_timeout_seconds = default_timeout_seconds

    def available(self) -> bool:
        return which(self._executable) is not None

    async def run(
        self,
        args: Sequence[str],
        *,
        timeout_seconds: float | None = None,
    ) -> DockerCommandResult:
        assert not isinstance(args, str | bytes)
        resolved_args = tuple(args)
        timeout = (
            self._default_timeout_seconds
            if timeout_seconds is None
            else timeout_seconds
        )
        if timeout is not None:
            assert timeout > 0
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
            return_code = await wait_for(process.wait(), timeout=timeout)
            stdout, stderr = await gather(stdout_task, stderr_task)
        except TimeoutError as error:
            await _kill_process(process, stdout_task, stderr_task)
            raise TimeoutError(
                f"Docker CLI command timed out after {_timeout_label(timeout)}"
            ) from error
        except CancelledError:
            await _kill_process(process, stdout_task, stderr_task)
            raise
        return DockerCommandResult(
            args=resolved_args,
            return_code=return_code,
            stdout=stdout,
            stderr=stderr,
        )


async def _kill_process(
    process: object,
    stdout_task: Task[bytes],
    stderr_task: Task[bytes],
) -> None:
    kill = getattr(process, "kill", None)
    if callable(kill):
        try:
            kill()
        except ProcessLookupError:
            pass
    wait = getattr(process, "wait", None)
    if callable(wait):
        try:
            await wait_for(wait(), timeout=_DOCKER_KILL_GRACE_SECONDS)
        except TimeoutError:
            pass
    stdout_task.cancel()
    stderr_task.cancel()
    await gather(stdout_task, stderr_task, return_exceptions=True)


def _timeout_label(timeout_seconds: float | None) -> str:
    if timeout_seconds is None:
        return "the configured deadline"
    return f"{timeout_seconds:g} seconds"


@dataclass(slots=True)
class _DockerContainerState:
    plan: ContainerRunPlan
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
class DockerContainerBackend(ContainerAsyncBackend):
    def __init__(
        self,
        runner: DockerCommandRunner | None = None,
    ) -> None:
        self._runner = runner or DockerSubprocessRunner()
        self._containers: dict[str, _DockerContainerState] = {}

    async def probe(self) -> ContainerBackendProbeResult:
        if not self._runner.available():
            return ContainerBackendProbeResult(
                backend=_DOCKER_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        "Docker CLI was not found on PATH",
                        retryable=True,
                    ),
                ),
            )
        version = await self._run_probe_command(
            (
                "version",
                "--format",
                "{{json .}}",
            )
        )
        if isinstance(version, ContainerBackendProbeResult):
            return version
        info = await self._run_probe_command(
            (
                "info",
                "--format",
                "{{json .}}",
            )
        )
        if isinstance(info, ContainerBackendProbeResult):
            return info
        info_payload = _first_json_mapping(info.stdout)
        if not info_payload:
            return ContainerBackendProbeResult(
                backend=_DOCKER_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        "Docker daemon info is unavailable",
                        retryable=True,
                    ),
                ),
            )
        return ContainerBackendProbeResult(
            backend=_DOCKER_BACKEND,
            available=True,
            capabilities=docker_container_capabilities(
                rootless=_docker_info_rootless(info_payload),
                remote_engine=_docker_info_remote(info_payload),
            ),
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
                "pull",
                "--platform",
                plan.image.platform,
                plan.image.reference,
            ),
            operation=ContainerBackendOperation.IMAGE_PULL,
            failure_code=ContainerBackendDiagnosticCode.PULL_FAILED,
            failure_message="Docker image pull failed",
            retryable=True,
            timeout_seconds=_DOCKER_PULL_TIMEOUT_SECONDS,
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
                "Docker backend does not build shell images",
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
        resolution = await self._verified_local_image(
            plan,
            operation=ContainerBackendOperation.CREATE,
            allow_missing=False,
        )
        assert resolution is not None
        local_reference, _, image_diagnostics = resolution
        if image_diagnostics:
            raise ContainerBackendError(image_diagnostics[0])
        container_id = _container_name(plan)
        try:
            await self._run_checked(
                _create_args(plan, container_id, local_reference),
                operation=ContainerBackendOperation.CREATE,
                failure_code=ContainerBackendDiagnosticCode.CREATE_FAILED,
                failure_message="Docker container create failed",
                retryable=True,
            )
        except CancelledError:
            cleanup_diagnostic = await self._cleanup_after_failed_create(
                container_id
            )
            if cleanup_diagnostic is not None:
                raise ContainerBackendError(cleanup_diagnostic)
            raise
        except ContainerBackendError as error:
            cleanup_diagnostic = await self._cleanup_after_failed_create(
                container_id
            )
            if cleanup_diagnostic is not None:
                raise ContainerBackendError(cleanup_diagnostic) from error
            raise
        self._containers[container_id] = _DockerContainerState(plan=plan)
        return ContainerBackendContainer(
            container_id=container_id,
            backend=_DOCKER_BACKEND,
            plan_fingerprint=container_pool_key(plan),
        )

    async def start(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendOperationResult:
        self._state(container, ContainerBackendOperation.START)
        result = await self._run_checked(
            ("start", container.container_id),
            operation=ContainerBackendOperation.START,
            failure_code=ContainerBackendDiagnosticCode.START_FAILED,
            failure_message="Docker container start failed",
            retryable=True,
        )
        return ContainerBackendOperationResult(
            operation=ContainerBackendOperation.START,
            metadata={"return_code": str(result.return_code)},
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
        self._state(container, ContainerBackendOperation.STREAM)
        try:
            result = await self._runner.run(
                (
                    "logs",
                    "--follow",
                    container.container_id,
                ),
                timeout_seconds=_DOCKER_EXECUTION_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    ContainerBackendOperation.STREAM,
                    "Docker container log stream timed out",
                    retryable=True,
                )
            ) from error
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    ContainerBackendOperation.STREAM,
                    "Docker CLI failed to start",
                    retryable=True,
                )
            ) from error
        if not result.ok:
            if _looks_like_backend_failure(result):
                raise ContainerBackendError(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.STREAM,
                        _command_failure_message(
                            "Docker container log stream failed",
                            result,
                        ),
                        retryable=True,
                    )
                )
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.ATTACH_FAILED,
                    ContainerBackendOperation.STREAM,
                    _command_failure_message(
                        "Docker container log stream failed",
                        result,
                    ),
                    retryable=True,
                )
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
        self._state(container, ContainerBackendOperation.WAIT)
        result = await self._run_checked(
            ("wait", container.container_id),
            operation=ContainerBackendOperation.WAIT,
            failure_code=ContainerBackendDiagnosticCode.WAIT_FAILED,
            failure_message="Docker container wait failed",
            retryable=True,
            timeout_seconds=_DOCKER_EXECUTION_TIMEOUT_SECONDS,
        )
        exit_code = _wait_exit_code(result.stdout)
        if exit_code is None:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.WAIT_FAILED,
                    ContainerBackendOperation.WAIT,
                    "Docker container wait did not return an exit code",
                    retryable=True,
                )
            )
        return ContainerBackendWaitResult(exit_code=exit_code)

    async def inspect(
        self,
        container: ContainerBackendContainer,
    ) -> ContainerBackendInspection:
        self._state(container, ContainerBackendOperation.INSPECT)
        result = await self._run_checked(
            ("inspect", "--format", "{{json .}}", container.container_id),
            operation=ContainerBackendOperation.INSPECT,
            failure_code=ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
            failure_message="Docker container inspect failed",
            retryable=True,
        )
        inspected = _first_json_mapping(result.stdout)
        status = (
            _string_field(inspected, ("State", "Status"), ("status",))
            or "unknown"
        )
        exit_code = _int_field(
            inspected,
            ("State", "ExitCode"),
            ("exit_code",),
        )
        return ContainerBackendInspection(
            container_id=container.container_id,
            status=status,
            exit_code=exit_code,
            metadata={"backend": _DOCKER_BACKEND.value},
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
                    "--no-stream",
                    "--format",
                    "{{json .}}",
                    container.container_id,
                ),
                timeout_seconds=_DOCKER_COMMAND_TIMEOUT_SECONDS,
            )
        except TimeoutError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    ContainerBackendOperation.STATS,
                    "Docker container stats timed out",
                    retryable=True,
                )
            ) from error
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    ContainerBackendOperation.STATS,
                    "Docker CLI failed to start",
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
                            "Docker container stats failed",
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
            ("rm", "--force", "--volumes", container.container_id),
        )
        if result.ok:
            self._state(
                container,
                ContainerBackendOperation.REMOVE,
            ).removed = True
        return result

    async def copy_outputs(
        self,
        container: ContainerBackendContainer,
        contract: ContainerOutputContract,
    ) -> ContainerOutputValidationResult:
        self._state(container, ContainerBackendOperation.COPY_OUTPUTS)
        assert isinstance(contract, ContainerOutputContract)
        with TemporaryDirectory(prefix="avalan-docker-output-") as root:
            try:
                result = await self._runner.run(
                    (
                        "cp",
                        f"{container.container_id}:{_OUTPUT_COPY_SOURCE}/.",
                        root,
                    ),
                    timeout_seconds=_DOCKER_COMMAND_TIMEOUT_SECONDS,
                )
            except TimeoutError as error:
                raise ContainerBackendError(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.TIMEOUT,
                        ContainerBackendOperation.COPY_OUTPUTS,
                        "Docker output copy timed out",
                        retryable=True,
                    )
                ) from error
            except OSError as error:
                raise ContainerBackendError(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.COPY_OUTPUTS,
                        "Docker CLI failed to start",
                        retryable=True,
                    )
                ) from error
            if not result.ok:
                if _looks_like_backend_failure(result):
                    raise ContainerBackendError(
                        _diagnostic(
                            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                            ContainerBackendOperation.COPY_OUTPUTS,
                            _command_failure_message(
                                "Docker output copy failed",
                                result,
                            ),
                            retryable=True,
                        )
                    )
                raise ContainerBackendError(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.COPY_FAILED,
                        ContainerBackendOperation.COPY_OUTPUTS,
                        _command_failure_message(
                            "Docker output copy failed",
                            result,
                        ),
                        retryable=True,
                    )
                )
            return validate_copied_outputs(root, contract)

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

    async def _run_probe_command(
        self,
        args: Sequence[str],
    ) -> DockerCommandResult | ContainerBackendProbeResult:
        try:
            result = await self._runner.run(
                args,
                timeout_seconds=_DOCKER_PROBE_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return ContainerBackendProbeResult(
                backend=_DOCKER_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        "Docker CLI probe timed out",
                        retryable=True,
                    ),
                ),
            )
        except OSError:
            return ContainerBackendProbeResult(
                backend=_DOCKER_BACKEND,
                available=False,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        ContainerBackendOperation.PROBE,
                        "Docker CLI failed to start",
                        retryable=True,
                    ),
                ),
            )
        if result.ok:
            return result
        return ContainerBackendProbeResult(
            backend=_DOCKER_BACKEND,
            available=False,
            diagnostics=(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    ContainerBackendOperation.PROBE,
                    _command_failure_message(
                        "Docker daemon is unavailable",
                        result,
                    ),
                    retryable=True,
                ),
            ),
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
            result = await self._runner.run(
                args,
                timeout_seconds=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return ContainerBackendOperationResult(
                operation=operation,
                diagnostics=(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.TIMEOUT,
                        operation,
                        "Docker cleanup command timed out",
                        retryable=True,
                    ),
                ),
            )
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    operation,
                    "Docker CLI failed to start",
                    retryable=True,
                )
            ) from error
        diagnostics: tuple[ContainerBackendDiagnostic, ...] = ()
        if not result.ok:
            failure_code = (
                ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE
                if _looks_like_backend_failure(result)
                else _cleanup_failure_code(result, operation)
            )
            diagnostics = (
                _diagnostic(
                    failure_code,
                    operation,
                    _command_failure_message(
                        "Docker cleanup command failed",
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

    async def _cleanup_after_failed_create(
        self, container_id: str
    ) -> ContainerBackendDiagnostic | None:
        cleanup = create_task(self._remove_untracked_container(container_id))
        try:
            return await shield(cleanup)
        except CancelledError:
            return _diagnostic(
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
                ContainerBackendOperation.CLEANUP,
                "Docker failed-create cleanup was interrupted",
                retryable=True,
            )

    async def _remove_untracked_container(
        self, container_id: str
    ) -> ContainerBackendDiagnostic | None:
        try:
            result = await self._runner.run(
                ("rm", "--force", "--volumes", container_id),
                timeout_seconds=_DOCKER_CLEANUP_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            return _diagnostic(
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
                ContainerBackendOperation.CLEANUP,
                "Docker failed-create cleanup timed out",
                retryable=True,
            )
        except OSError:
            return _diagnostic(
                ContainerBackendDiagnosticCode.CLEANUP_FAILED,
                ContainerBackendOperation.CLEANUP,
                "Docker failed-create cleanup could not start",
                retryable=True,
            )
        if result.ok:
            return None
        return _diagnostic(
            _cleanup_failure_code(result, ContainerBackendOperation.CLEANUP),
            ContainerBackendOperation.CLEANUP,
            _command_failure_message(
                "Docker failed-create cleanup command failed",
                result,
            ),
            retryable=True,
        )

    async def _run_checked(
        self,
        args: Sequence[str],
        *,
        operation: ContainerBackendOperation,
        failure_code: ContainerBackendDiagnosticCode,
        failure_message: str,
        retryable: bool = False,
        timeout_seconds: float = _DOCKER_COMMAND_TIMEOUT_SECONDS,
    ) -> DockerCommandResult:
        try:
            result = await self._runner.run(
                args,
                timeout_seconds=timeout_seconds,
            )
        except TimeoutError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.TIMEOUT,
                    operation,
                    f"{failure_message}: timed out after "
                    f"{_timeout_label(timeout_seconds)}",
                    retryable=True,
                )
            ) from error
        except OSError as error:
            raise ContainerBackendError(
                _diagnostic(
                    ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                    operation,
                    "Docker CLI failed to start",
                    retryable=True,
                )
            ) from error
        if not result.ok:
            if _looks_like_backend_failure(result):
                raise ContainerBackendError(
                    _diagnostic(
                        ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                        operation,
                        _command_failure_message(failure_message, result),
                        retryable=True,
                    )
                )
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
    ) -> _DockerContainerState:
        assert isinstance(container, ContainerBackendContainer)
        if container.backend is not _DOCKER_BACKEND:
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
                    "Docker container state is unavailable",
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
        last_result: DockerCommandResult | None = None
        for reference in _local_image_references(plan.image.reference):
            try:
                result = await self._runner.run(
                    (
                        "image",
                        "inspect",
                        "--format",
                        "{{json .}}",
                        reference,
                    ),
                    timeout_seconds=_DOCKER_COMMAND_TIMEOUT_SECONDS,
                )
            except TimeoutError:
                return (
                    reference,
                    plan.image.digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.TIMEOUT,
                            operation,
                            "Docker image inspect timed out",
                            retryable=True,
                        ),
                    ),
                )
            except OSError:
                return (
                    reference,
                    plan.image.digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                            operation,
                            "Docker CLI failed to start",
                            retryable=True,
                        ),
                    ),
                )
            if not result.ok:
                if _looks_like_backend_failure(result):
                    return (
                        reference,
                        plan.image.digest,
                        (
                            _diagnostic(
                                ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE,
                                operation,
                                _command_failure_message(
                                    "Docker image inspect failed",
                                    result,
                                ),
                                retryable=True,
                            ),
                        ),
                    )
                last_result = result
                continue
            inspected = _first_json_mapping(result.stdout)
            digest = _image_digest(inspected, expected=plan.image.digest)
            if digest is None:
                return (
                    reference,
                    plan.image.digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.IMAGE_DENIED,
                            operation,
                            "Docker image digest is unavailable",
                        ),
                    ),
                )
            platform_diagnostic = _image_platform_diagnostic(
                inspected,
                plan.image.platform,
                operation,
            )
            if platform_diagnostic is not None:
                return (reference, digest, (platform_diagnostic,))
            if digest != plan.image.digest:
                return (
                    reference,
                    digest,
                    (
                        _diagnostic(
                            ContainerBackendDiagnosticCode.IMAGE_DENIED,
                            operation,
                            "local Docker image digest does not match"
                            " approved plan",
                        ),
                    ),
                )
            return reference, digest, ()
        if allow_missing:
            return None
        result = last_result or DockerCommandResult(args=(), return_code=1)
        return (
            plan.image.reference,
            plan.image.digest,
            (
                _diagnostic(
                    ContainerBackendDiagnosticCode.IMAGE_DENIED,
                    operation,
                    _command_failure_message(
                        "Docker image is unavailable locally",
                        result,
                    ),
                ),
            ),
        )


def docker_container_capabilities(
    *,
    rootless: bool,
    remote_engine: bool = False,
) -> ContainerBackendCapabilities:
    return ContainerBackendCapabilities(
        backend=_DOCKER_BACKEND,
        host_os=_host_os(),
        guest_os="linux",
        architecture=_host_architecture(),
        runtime_name="Docker Engine",
        support_level=ContainerBackendSupportLevel.SUPPORTED,
        platform_emulation=True,
        rootless=rootless,
        user_namespace=True,
        build=False,
        pull=True,
        network_modes=_NETWORK_MODES,
        mount_types=_MOUNTABLE_TYPES,
        resource_limits=True,
        device_classes=(ContainerDeviceClass.CPU,),
        remote_engine=remote_engine,
        streaming_attach=True,
        stats=True,
        lifecycle_normalization=True,
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
        plan.image.platform,
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--network",
        _network_argument(cast(ContainerNetworkMode, plan.network.mode)),
        "--user",
        "1000:1000",
        "--workdir",
        plan.command.cwd,
    ]
    if plan.resources.cpu_count is not None:
        args.extend(("--cpus", str(plan.resources.cpu_count)))
    if plan.resources.memory_bytes is not None:
        args.extend(("--memory", str(plan.resources.memory_bytes)))
    if plan.resources.pids is not None:
        args.extend(("--pids-limit", str(plan.resources.pids)))
    output_mount_present = False
    for mount in plan.mounts:
        output_mount_present = (
            output_mount_present or mount.target == _OUTPUT_COPY_SOURCE
        )
        args.extend(_mount_args(mount))
    if not output_mount_present:
        args.extend(("--tmpfs", f"{_OUTPUT_COPY_SOURCE}:rw,nosuid,nodev"))
    args.append(image_reference)
    args.extend(plan.command.argv)
    return tuple(args)


def _mount_args(mount: ContainerMountDeclaration) -> tuple[str, ...]:
    assert isinstance(mount, ContainerMountDeclaration)
    mount_type = cast(ContainerMountType, mount.mount_type)
    if mount.source is None and mount_type in {
        ContainerMountType.SCRATCH,
        ContainerMountType.OUTPUT,
    }:
        return ("--tmpfs", f"{mount.target}:rw,nosuid,nodev")
    assert mount.source is not None
    access = cast(ContainerMountAccess, mount.access)
    parts = [
        "type=bind",
        f"source={_host_path(mount.source)}",
        f"target={mount.target}",
    ]
    if access is ContainerMountAccess.READ:
        parts.append("readonly")
    return ("--mount", ",".join(parts))


def _network_argument(mode: ContainerNetworkMode) -> str:
    if mode is ContainerNetworkMode.FULL:
        return "bridge"
    return "none"


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


def _has_image_tag(reference: str) -> bool:
    last_segment = reference.rsplit("/", 1)[-1]
    return ":" in last_segment


def _host_path(source: str) -> str:
    return Path(source).expanduser().resolve().as_posix()


def _container_name(plan: ContainerRunPlan) -> str:
    return f"avalan-{container_pool_key(plan)[:12]}-{uuid4().hex[:8]}"


def _plan_diagnostics(
    plan: ContainerRunPlan,
    *,
    operation: ContainerBackendOperation,
) -> tuple[ContainerBackendDiagnostic, ...]:
    assert isinstance(plan, ContainerRunPlan)
    diagnostics: list[ContainerBackendDiagnostic] = []
    if plan.backend is not _DOCKER_BACKEND:
        diagnostics.append(
            _capability_mismatch(operation, "plan backend is not docker")
        )
    if plan.image.build_policy is not ContainerBuildPolicy.DISABLED:
        diagnostics.append(
            _diagnostic(
                ContainerBackendDiagnosticCode.BUILD_DENIED,
                operation,
                "Docker backend does not build shell images",
            )
        )
    network = cast(ContainerNetworkMode, plan.network.mode)
    if network not in _NETWORK_MODES:
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
                operation,
                "secret injection is not supported",
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
    if mount.source is None and mount_type not in {
        ContainerMountType.SCRATCH,
        ContainerMountType.OUTPUT,
    }:
        diagnostics.append(
            _capability_mismatch(operation, "mount source is required")
        )
    return tuple(diagnostics)


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
        backend=_DOCKER_BACKEND,
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
    result: DockerCommandResult,
) -> str:
    detail = (
        (result.stderr or result.stdout)
        .decode("utf-8", errors="replace")
        .strip()
    )
    if not detail:
        return f"{prefix}: exit code {result.return_code}"
    return f"{prefix}: {detail}"


def _looks_like_backend_failure(result: DockerCommandResult) -> bool:
    output = (
        (result.stderr + b"\n" + result.stdout)
        .decode("utf-8", errors="replace")
        .lower()
    )
    return any(marker in output for marker in _BACKEND_ERROR_MARKERS)


def _cleanup_failure_code(
    result: DockerCommandResult,
    operation: ContainerBackendOperation,
) -> ContainerBackendDiagnosticCode:
    if operation not in {
        ContainerBackendOperation.REMOVE,
        ContainerBackendOperation.CLEANUP,
    }:
        return ContainerBackendDiagnosticCode.CLEANUP_FAILED
    output = (
        (result.stderr + b"\n" + result.stdout)
        .decode("utf-8", errors="replace")
        .lower()
    )
    if any(marker in output for marker in _ORPHAN_MARKERS):
        return ContainerBackendDiagnosticCode.ORPHAN_QUARANTINED
    return ContainerBackendDiagnosticCode.CLEANUP_FAILED


def _first_json_mapping(raw: bytes) -> Mapping[str, object]:
    try:
        value = json_loads(raw.decode("utf-8"))
    except (JSONDecodeError, UnicodeDecodeError):
        return {}
    if isinstance(value, list) and value:
        first = value[0]
        if isinstance(first, Mapping):
            return cast(Mapping[str, object], first)
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    return {}


def _image_digest(
    inspected: Mapping[str, object],
    *,
    expected: str,
) -> str | None:
    repo_digests = _repo_digests(inspected)
    for repo_digest in repo_digests:
        if repo_digest == expected:
            return repo_digest
    if repo_digests:
        return repo_digests[0]
    return None


def _repo_digests(inspected: Mapping[str, object]) -> tuple[str, ...]:
    value = inspected.get("RepoDigests", inspected.get("repo_digests", ()))
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        return ()
    digests: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        digest = item.rsplit("@", 1)[-1]
        if digest.startswith("sha256:"):
            digests.append(digest)
    return tuple(digests)


def _image_platform_diagnostic(
    inspected: Mapping[str, object],
    expected_platform: str,
    operation: ContainerBackendOperation,
) -> ContainerBackendDiagnostic | None:
    expected_os, expected_architecture = expected_platform.split("/", 1)
    image_os = _string_field(inspected, ("Os",), ("os",))
    image_architecture = _string_field(
        inspected,
        ("Architecture",),
        ("architecture",),
    )
    if image_os is None or image_architecture is None:
        return _capability_mismatch(
            operation,
            "image platform metadata is unavailable",
        )
    if image_os is not None and image_os.lower() != expected_os.lower():
        return _capability_mismatch(
            operation,
            f"image OS {image_os} does not match {expected_os}",
        )
    if (
        image_architecture is not None
        and image_architecture.lower() != expected_architecture.lower()
    ):
        return _capability_mismatch(
            operation,
            "image architecture "
            f"{image_architecture} does not match {expected_architecture}",
        )
    return None


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
                )
                or 0,
                memory_bytes=(
                    _int_field(
                        mapping,
                        ("memoryBytes",),
                        ("memory_bytes",),
                    )
                    or _memory_usage_bytes(
                        _string_field(mapping, ("MemUsage",))
                    )
                    or 0
                ),
                pids=(
                    _int_field(mapping, ("pids",), ("PIDs",), ("Pids",)) or 0
                ),
            )
        )
    return tuple(stats)


def _memory_usage_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    used = value.split("/", 1)[0].strip()
    return _quantity_bytes(used)


def _quantity_bytes(value: str) -> int | None:
    if not value:
        return None
    number = ""
    suffix = ""
    for character in value:
        if character.isdigit() or character == ".":
            number += character
        else:
            suffix += character
    if not number:
        return None
    multiplier = {
        "b": 1,
        "kb": 1000,
        "kib": 1024,
        "mb": 1000 * 1000,
        "mib": 1024 * 1024,
        "gb": 1000 * 1000 * 1000,
        "gib": 1024 * 1024 * 1024,
    }.get(suffix.strip().lower() or "b")
    if multiplier is None:
        return None
    return int(float(number) * multiplier)


def _wait_exit_code(raw: bytes) -> int | None:
    text = raw.decode("utf-8", errors="replace").strip().splitlines()
    if not text:
        return None
    try:
        return int(text[0])
    except ValueError:
        return None


def _docker_info_rootless(info: Mapping[str, object]) -> bool:
    security_options = info.get("SecurityOptions", ())
    if isinstance(security_options, Sequence) and not isinstance(
        security_options,
        str | bytes,
    ):
        return any(
            isinstance(option, str) and "rootless" in option.lower()
            for option in security_options
        )
    return False


def _docker_info_remote(info: Mapping[str, object]) -> bool:
    name = _string_field(info, ("OperatingSystem",), ("Name",))
    return name is not None and "docker desktop" in name.lower()


def _host_os() -> str:
    return system().lower() or "unknown"


def _host_architecture() -> str:
    architecture = machine().lower()
    if architecture in {"x86_64", "amd64"}:
        return "amd64"
    if architecture in {"aarch64", "arm64"}:
        return "arm64"
    return architecture or "unknown"


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
        if isinstance(value, str) and value.isdecimal():
            return int(value)
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
