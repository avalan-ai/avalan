from ...entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from .entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    SHELL_STATUS_ERROR_CODES,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    GeneratedOutputPlan,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
)
from .filesystem import (
    file_digest_and_base64,
    inspect_path,
    list_directory,
    private_temp_directory,
    probe_image_dimensions,
    remove_file,
    remove_tree,
    resolve_policy_path,
)
from .settings import ShellToolSettings

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Semaphore,
    create_subprocess_exec,
    create_task,
    gather,
    wait,
    wait_for,
)
from asyncio.streams import StreamReader
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Awaitable, Callable
from os import kill as os_kill
from os import name as os_name
from pathlib import Path
from signal import SIGKILL, SIGTERM
from stat import S_IMODE
from time import perf_counter
from typing import Any, Protocol

_PROCESS_CLEANUP_GRACE_SECONDS = 0.2


class CommandExecutor(Protocol):
    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        raise NotImplementedError


class LocalCommandExecutor:
    def __init__(self, settings: ShellToolSettings | None = None) -> None:
        settings = ShellToolSettings() if settings is None else settings
        assert isinstance(
            settings,
            ShellToolSettings,
        ), "settings must be shell tool settings"
        self._settings = settings
        self._process_semaphore = Semaphore(
            settings.max_concurrent_processes,
        )
        self._heavy_semaphore = Semaphore(
            settings.max_concurrent_heavy_processes,
        )

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        if not isinstance(spec, ExecutionSpec):
            raise NotImplementedError(
                "local shell execution is not implemented"
            )
        if spec.executable is None:
            return ExecutionResult(
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                argv=_result_argv(spec),
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                status=ShellExecutionStatus.COMMAND_UNAVAILABLE,
                exit_code=None,
                stdout="",
                stderr="",
                stdout_media_type=spec.stdout_media_type,
                output_kind=spec.output_kind,
                stdout_bytes=0,
                stderr_bytes=0,
                stdout_truncated=False,
                stderr_truncated=False,
                timed_out=False,
                cancelled=False,
                duration_ms=0,
                error_code=ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
                error_message="command is unavailable",
                metadata=spec.metadata,
            )
        if spec.resource_class == "heavy":
            async with self._heavy_semaphore:
                async with self._process_semaphore:
                    return await self._execute_spawn(spec, stream=stream)
        async with self._process_semaphore:
            return await self._execute_spawn(spec, stream=stream)

    async def _execute_spawn(
        self,
        spec: ExecutionSpec,
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    ) -> ExecutionResult:
        start_time = perf_counter()
        if spec.output_plan is None:
            return await self._execute_prepared_spawn(
                spec,
                start_time=start_time,
                runtime_output_prefix=None,
                generated_output_replacements=(),
                stream=stream,
            )
        try:
            async with private_temp_directory() as output_directory:
                await _enforce_runtime_output_directory(
                    output_directory,
                    cwd=spec.cwd,
                    workspace_root=self._settings.workspace_root,
                )
                runtime_output_prefix = (
                    output_directory / spec.output_plan.prefix_name
                )
                generated_output_replacements = (
                    await _generated_output_path_replacements(
                        spec.output_plan,
                        runtime_output_prefix,
                    )
                )
                return await self._execute_prepared_spawn(
                    spec,
                    start_time=start_time,
                    runtime_output_prefix=runtime_output_prefix,
                    generated_output_replacements=(
                        generated_output_replacements
                    ),
                    stream=stream,
                )
        except (OSError, _GeneratedOutputError):
            return _result(
                spec,
                status=ShellExecutionStatus.TOOL_ERROR,
                exit_code=None,
                stdout_bytes=b"",
                stderr_bytes=b"",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_ms=_duration_ms(start_time),
                error_code=(
                    ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED
                ),
                error_message="generated output preparation failed",
            )

    async def _execute_prepared_spawn(
        self,
        spec: ExecutionSpec,
        *,
        start_time: float,
        runtime_output_prefix: Path | None,
        generated_output_replacements: tuple[tuple[str, str], ...],
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        try:
            try:
                process = await create_subprocess_exec(
                    *_spawn_argv(
                        spec,
                        runtime_output_prefix=runtime_output_prefix,
                    ),
                    cwd=spec.cwd,
                    env=spec.env,
                    stdin=PIPE if spec.stdin is not None else DEVNULL,
                    stdout=PIPE,
                    stderr=PIPE,
                    start_new_session=os_name == "posix",
                )
            except (OSError, ValueError):
                return _result(
                    spec,
                    status=ShellExecutionStatus.SPAWN_FAILED,
                    exit_code=None,
                    stdout_bytes=b"",
                    stderr_bytes=b"",
                    stdout_truncated=False,
                    stderr_truncated=False,
                    duration_ms=_duration_ms(start_time),
                    error_message="process spawn failed",
                    generated_output_replacements=(
                        generated_output_replacements
                    ),
                )

            stdout_reader = None
            stderr_reader = None
            process_waiter = None
            stdout_capture = _StreamCapture()
            stderr_capture = _StreamCapture()
            try:
                await _emit_stream_event(
                    stream,
                    kind=ToolExecutionStreamKind.LOG,
                    content="process started",
                    metadata={"command": spec.command},
                )
                await _emit_stream_event(
                    stream,
                    kind=ToolExecutionStreamKind.PROGRESS,
                    content="started",
                    progress=0.0,
                    metadata={"command": spec.command},
                )
                stdout_reader = create_task(
                    _collect_stream(
                        process.stdout,
                        spec.max_stdout_bytes,
                        self._settings.stream_read_chunk_bytes,
                        stdout_capture,
                        stream_event=stream,
                        stream_kind=ToolExecutionStreamKind.STDOUT,
                    )
                )
                stderr_reader = create_task(
                    _collect_stream(
                        process.stderr,
                        spec.max_stderr_bytes,
                        self._settings.stream_read_chunk_bytes,
                        stderr_capture,
                        stream_event=stream,
                        stream_kind=ToolExecutionStreamKind.STDERR,
                    )
                )
                process_waiter = create_task(
                    _write_stdin_and_wait(process, spec.stdin)
                )
                reader_failed = await wait_for(
                    _wait_for_process_or_reader_failure(
                        process_waiter,
                        stdout_reader,
                        stderr_reader,
                    ),
                    timeout=spec.timeout_seconds,
                )
                if reader_failed:
                    await _kill_process_group(process)
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
            except TimeoutError:
                await _terminate_process_group(process)
                if process_waiter is not None:
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
                stdout, stderr = await _reader_results(
                    stdout_reader,
                    stderr_reader,
                    stdout_capture=stdout_capture,
                    stderr_capture=stderr_capture,
                )
                stdout_bytes, stdout_truncated = stdout
                stderr_bytes, stderr_truncated = stderr
                await _emit_stream_event(
                    stream,
                    kind=ToolExecutionStreamKind.PROGRESS,
                    content=ShellExecutionStatus.TIMEOUT.value,
                    progress=1.0,
                    metadata={
                        "exit_code": getattr(process, "returncode", None),
                        "status": ShellExecutionStatus.TIMEOUT.value,
                        "timed_out": True,
                    },
                )
                return _result(
                    spec,
                    status=ShellExecutionStatus.TIMEOUT,
                    exit_code=getattr(process, "returncode", None),
                    stdout_bytes=stdout_bytes,
                    stderr_bytes=stderr_bytes,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    duration_ms=_duration_ms(start_time),
                    timed_out=True,
                    error_message="command timed out",
                    generated_output_replacements=(
                        generated_output_replacements
                    ),
                )
            except CancelledError:
                await _kill_process_group(process)
                if process_waiter is not None:
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
                await _cancel_reader_tasks(stdout_reader, stderr_reader)
                raise
            except Exception:
                await _kill_process_group(process)
                if process_waiter is not None:
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
                await _cancel_reader_tasks(stdout_reader, stderr_reader)
                raise

            stdout, stderr = await _reader_results(
                stdout_reader,
                stderr_reader,
                cleanup_timeout=None,
                stdout_capture=stdout_capture,
                stderr_capture=stderr_capture,
            )
            stdout_bytes, stdout_truncated = stdout
            stderr_bytes, stderr_truncated = stderr
            exit_code = (
                process.returncode if process.returncode is not None else 0
            )
            status = _status_for_exit_code(exit_code, spec)
            error_code = None
            error_message = (
                None
                if status is ShellExecutionStatus.COMPLETED
                else f"command exited with status {exit_code}"
            )
            if _reader_task_failed(stdout_reader) or _reader_task_failed(
                stderr_reader,
            ):
                status = ShellExecutionStatus.TOOL_ERROR
                error_code = ShellExecutionErrorCode.TOOL_ERROR
                error_message = "stream collection failed"
            generated_files: tuple[GeneratedFile, ...] = ()
            if spec.output_plan is not None:
                try:
                    generated_files = await _collect_generated_files(
                        spec.output_plan,
                        runtime_output_prefix,
                        self._settings.stream_read_chunk_bytes,
                    )
                except (OSError, _GeneratedOutputError):
                    status = ShellExecutionStatus.TOO_LARGE
                    error_code = (
                        ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED
                    )
                    error_message = "generated output capture failed"
            await _emit_stream_event(
                stream,
                kind=ToolExecutionStreamKind.PROGRESS,
                content=status.value,
                progress=1.0,
                metadata={
                    "exit_code": exit_code,
                    "status": status.value,
                },
            )
            return _result(
                spec,
                status=status,
                exit_code=exit_code,
                stdout_bytes=stdout_bytes,
                stderr_bytes=stderr_bytes,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
                duration_ms=_duration_ms(start_time),
                generated_files=generated_files,
                error_code=error_code,
                error_message=error_message,
                generated_output_replacements=generated_output_replacements,
            )
        except _GeneratedOutputError:
            return _result(
                spec,
                status=ShellExecutionStatus.TOOL_ERROR,
                exit_code=None,
                stdout_bytes=b"",
                stderr_bytes=b"",
                stdout_truncated=False,
                stderr_truncated=False,
                duration_ms=_duration_ms(start_time),
                error_code=(
                    ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED
                ),
                error_message="generated output preparation failed",
            )


async def _terminate_process_group(process: object) -> None:
    _signal_process_group(process, SIGTERM, "terminate")
    if await _wait_for_process_exit(process):
        return
    _signal_process_group(process, SIGKILL, "kill")
    await _wait_for_process_exit(process)


async def _kill_process_group(process: object) -> None:
    _signal_process_group(process, SIGKILL, "kill")
    await _wait_for_process_exit(process)


async def _write_stdin_and_wait(
    process: object,
    stdin: bytes | None,
) -> None:
    await _write_stdin(process, stdin)
    wait_method = getattr(process, "wait")
    await wait_method()


def _signal_process_group(
    process: object,
    signal_number: int,
    fallback_method_name: str,
) -> None:
    pid = getattr(process, "pid", None)
    if os_name == "posix" and isinstance(pid, int):
        try:
            os_kill(-pid, signal_number)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    fallback_method = getattr(process, fallback_method_name, None)
    if fallback_method is None:
        return
    try:
        fallback_method()
    except (ProcessLookupError, OSError):
        pass


async def _wait_for_process_exit(process: object) -> bool:
    wait = getattr(process, "wait", None)
    if wait is None:
        return True
    try:
        await wait_for(wait(), timeout=_PROCESS_CLEANUP_GRACE_SECONDS)
    except TimeoutError:
        return False
    except (ProcessLookupError, OSError):
        return True
    return True


async def _reader_results(
    stdout_reader: Any,
    stderr_reader: Any,
    *,
    cleanup_timeout: float | None = _PROCESS_CLEANUP_GRACE_SECONDS,
    stdout_capture: "_StreamCapture | None" = None,
    stderr_capture: "_StreamCapture | None" = None,
) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
    tasks = {
        task for task in (stdout_reader, stderr_reader) if task is not None
    }
    if tasks:
        if cleanup_timeout is None:
            _, pending = await wait(tasks)
        else:
            _, pending = await wait(tasks, timeout=cleanup_timeout)
        if pending:
            await _cancel_reader_tasks(*pending)
    return (
        _reader_result(stdout_reader, stdout_capture),
        _reader_result(stderr_reader, stderr_capture),
    )


async def _wait_for_process_or_reader_failure(
    process_waiter: Any,
    stdout_reader: Any,
    stderr_reader: Any,
) -> bool:
    tasks = {
        task
        for task in (process_waiter, stdout_reader, stderr_reader)
        if task is not None
    }
    while True:
        done, tasks = await wait(tasks, return_when=FIRST_COMPLETED)
        for task in done:
            if task is process_waiter:
                await process_waiter
                return False
            if _reader_task_failed(task):
                return True


def _reader_result(
    reader_task: Any,
    capture: "_StreamCapture | None" = None,
) -> tuple[bytes, bool]:
    fallback = capture.snapshot() if capture is not None else (b"", False)
    if reader_task is None or not reader_task.done():
        return fallback
    if reader_task.cancelled():
        return fallback
    try:
        result = reader_task.result()
    except Exception:
        return fallback
    if (
        not isinstance(result, tuple)
        or len(result) != 2
        or not isinstance(result[0], bytes)
        or not isinstance(result[1], bool)
    ):
        return fallback
    return result[0], result[1]


def _reader_task_failed(reader_task: Any) -> bool:
    if reader_task is None or not reader_task.done():
        return False
    if reader_task.cancelled():
        return False
    try:
        reader_task.result()
    except Exception:
        return True
    return False


async def _cancel_reader_tasks(*reader_tasks: Any) -> None:
    tasks = [task for task in reader_tasks if task is not None]
    for task in tasks:
        task.cancel()
    if tasks:
        await gather(*tasks, return_exceptions=True)


def _spawn_argv(
    spec: ExecutionSpec,
    *,
    runtime_output_prefix: Path | None = None,
) -> tuple[str, ...]:
    assert spec.executable is not None, "executable must be resolved"
    if not spec.argv:
        return (spec.executable,)
    if spec.output_plan is None:
        assert (
            runtime_output_prefix is None
        ), "runtime output prefix requires an output plan"
        return (spec.executable, *spec.argv[1:])
    if runtime_output_prefix is None:
        raise _GeneratedOutputError
    argv_tail = spec.argv[1:]
    if argv_tail.count(GENERATED_OUTPUT_PREFIX_PLACEHOLDER) != 1:
        raise _GeneratedOutputError
    return (
        spec.executable,
        *(
            (
                str(runtime_output_prefix)
                if argument == GENERATED_OUTPUT_PREFIX_PLACEHOLDER
                else argument
            )
            for argument in argv_tail
        ),
    )


def _status_for_exit_code(
    exit_code: int,
    spec: ExecutionSpec,
) -> ShellExecutionStatus:
    if exit_code == 0:
        return ShellExecutionStatus.COMPLETED
    exit_code_statuses = spec.metadata.get("exit_code_statuses", {})
    if (
        isinstance(exit_code_statuses, dict)
        and exit_code in exit_code_statuses
    ):
        return ShellExecutionStatus(exit_code_statuses[exit_code])
    return ShellExecutionStatus.NONZERO_EXIT


async def _write_stdin(process: object, stdin: bytes | None) -> None:
    if stdin is None:
        return
    writer = getattr(process, "stdin", None)
    if writer is None:
        return
    if not hasattr(writer, "write"):
        setattr(process, "stdin", stdin)
        return
    try:
        writer.write(stdin)
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        await _close_stdin_writer(writer)


async def _close_stdin_writer(writer: object) -> None:
    close = getattr(writer, "close", None)
    if close is None:
        return
    try:
        close()
    except (BrokenPipeError, ConnectionResetError):
        return
    wait_closed = getattr(writer, "wait_closed", None)
    if wait_closed is None:
        return
    try:
        await wait_for(
            wait_closed(),
            timeout=_PROCESS_CLEANUP_GRACE_SECONDS,
        )
    except (BrokenPipeError, ConnectionResetError, TimeoutError):
        pass


async def _collect_stream(
    stream: StreamReader | None,
    byte_cap: int,
    chunk_size: int,
    capture: "_StreamCapture | None" = None,
    *,
    stream_event: (
        Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
    ) = None,
    stream_kind: ToolExecutionStreamKind | None = None,
) -> tuple[bytes, bool]:
    assert byte_cap >= 0, "byte_cap must be non-negative"
    assert chunk_size > 0, "chunk_size must be positive"
    assert (
        stream_event is None or stream_kind is not None
    ), "stream kind is required when callback is provided"
    capture = _StreamCapture() if capture is None else capture
    if stream is None:
        return capture.snapshot()

    while True:
        chunk = await stream.read(chunk_size)
        if not chunk:
            break
        captured = capture.append(chunk, byte_cap)
        if stream_event is not None and captured:
            assert stream_kind is not None
            await stream_event(
                ToolExecutionStreamEvent(
                    kind=stream_kind,
                    content=captured.decode(errors="replace"),
                    metadata={
                        "bytes": len(captured),
                        "truncated": len(captured) < len(chunk),
                    },
                )
            )
    return capture.snapshot()


async def _emit_stream_event(
    stream_event: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    *,
    kind: ToolExecutionStreamKind,
    content: str | None = None,
    progress: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    if stream_event is None:
        return
    await stream_event(
        ToolExecutionStreamEvent(
            kind=kind,
            content=content,
            progress=progress,
            metadata=metadata or {},
        )
    )


class _StreamCapture:
    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._captured_bytes = 0
        self._truncated = False

    def append(self, chunk: bytes, byte_cap: int) -> bytes:
        remaining = byte_cap - self._captured_bytes
        if remaining <= 0:
            self._truncated = True
            return b""
        if len(chunk) > remaining:
            captured = chunk[:remaining]
            self._chunks.append(captured)
            self._captured_bytes += remaining
            self._truncated = True
            return captured
        self._chunks.append(chunk)
        self._captured_bytes += len(chunk)
        return chunk

    def snapshot(self) -> tuple[bytes, bool]:
        return b"".join(self._chunks), self._truncated


class _GeneratedOutputError(Exception):
    pass


async def _enforce_runtime_output_directory(
    output_directory: Path,
    *,
    cwd: str,
    workspace_root: str,
) -> None:
    directory_metadata = await inspect_path(output_directory)
    if (
        directory_metadata.is_symlink
        or not directory_metadata.is_directory
        or S_IMODE(directory_metadata.mode) & 0o077
    ):
        raise _GeneratedOutputError
    resolved_directory = directory_metadata.resolved_path
    resolved_cwd = await resolve_policy_path(cwd)
    resolved_workspace_root = await resolve_policy_path(workspace_root)
    if _is_relative_to(resolved_directory, resolved_cwd) or _is_relative_to(
        resolved_directory,
        resolved_workspace_root,
    ):
        raise _GeneratedOutputError


async def _generated_output_path_replacements(
    plan: GeneratedOutputPlan,
    runtime_output_prefix: Path,
) -> tuple[tuple[str, str], ...]:
    prefix_path = Path(runtime_output_prefix)
    output_directory = prefix_path.parent
    replacements = {
        str(prefix_path): plan.display_prefix,
        str(output_directory): "[generated_output_directory]",
    }
    try:
        replacements[str(await resolve_policy_path(prefix_path))] = (
            plan.display_prefix
        )
    except OSError:
        pass
    try:
        replacements[str(await resolve_policy_path(output_directory))] = (
            "[generated_output_directory]"
        )
    except OSError:
        pass
    return tuple(
        sorted(
            (
                (source, replacement)
                for source, replacement in replacements.items()
                if source
            ),
            key=lambda item: len(item[0]),
            reverse=True,
        )
    )


async def _cleanup_output_directory(output_directory: Path) -> None:
    try:
        await remove_tree(output_directory)
    except OSError:
        try:
            metadata = await inspect_path(output_directory)
        except OSError:
            return
        if not metadata.is_file and not metadata.is_symlink:
            return
        try:
            await remove_file(output_directory)
        except OSError:
            pass


async def _collect_generated_files(
    plan: GeneratedOutputPlan,
    runtime_output_prefix: Path | None,
    chunk_size: int,
) -> tuple[GeneratedFile, ...]:
    if runtime_output_prefix is None:
        raise _GeneratedOutputError
    prefix_path = Path(runtime_output_prefix)
    if prefix_path.name != plan.prefix_name:
        raise _GeneratedOutputError
    output_directory = prefix_path.parent
    directory_metadata = await inspect_path(output_directory)
    if (
        directory_metadata.is_symlink
        or not directory_metadata.is_directory
        or S_IMODE(directory_metadata.mode) & 0o077
    ):
        raise _GeneratedOutputError
    resolved_directory = directory_metadata.resolved_path
    entries = await list_directory(output_directory)
    candidates = tuple(
        sorted(
            (
                path
                for path in entries
                if _matches_generated_output_prefix(path, prefix_path)
            ),
            key=lambda path: path.name,
        )
    )
    generated_files: list[GeneratedFile] = []
    total_bytes = 0
    for path in candidates:
        suffix = path.suffix
        if suffix not in plan.allowed_suffixes:
            raise _GeneratedOutputError
        if len(generated_files) >= plan.max_files:
            raise _GeneratedOutputError
        metadata = await inspect_path(path)
        if (
            metadata.is_symlink
            or not metadata.is_file
            or metadata.hardlink_count > 1
            or not _is_relative_to(metadata.resolved_path, resolved_directory)
        ):
            raise _GeneratedOutputError
        if metadata.size > plan.max_file_bytes:
            raise _GeneratedOutputError
        total_bytes += metadata.size
        if total_bytes > plan.max_total_bytes:
            raise _GeneratedOutputError
        media_type = plan.suffix_media_types[suffix]
        width, height = await _generated_file_dimensions(
            path,
            media_type,
            plan,
        )
        sha256, content_base64 = await file_digest_and_base64(
            path,
            chunk_size=chunk_size,
            max_inline_bytes=plan.max_inline_bytes,
        )
        generated_files.append(
            GeneratedFile(
                display_path=_display_generated_path(path, prefix_path, plan),
                media_type=media_type,
                suffix=suffix,
                bytes=metadata.size,
                sha256=sha256,
                page=_generated_page_number(path, prefix_path),
                width=width,
                height=height,
                content_base64=content_base64,
            )
        )
    return tuple(generated_files)


async def _generated_file_dimensions(
    path: Path,
    media_type: str,
    plan: GeneratedOutputPlan,
) -> tuple[int | None, int | None]:
    if not media_type.startswith("image/"):
        return None, None
    dimensions = await probe_image_dimensions(path)
    if dimensions is None:
        return None, None
    width, height = dimensions
    if (
        plan.max_raster_long_edge_pixels is not None
        and max(width, height) > plan.max_raster_long_edge_pixels
    ):
        raise _GeneratedOutputError
    if (
        plan.max_raster_pixels is not None
        and width * height > plan.max_raster_pixels
    ):
        raise _GeneratedOutputError
    return width, height


def _display_generated_path(
    path: Path,
    prefix_path: Path,
    plan: GeneratedOutputPlan,
) -> str:
    suffix = _generated_display_suffix(path, prefix_path)
    return f"{plan.display_prefix}{suffix}"


def _generated_display_suffix(path: Path, prefix_path: Path) -> str:
    suffix = path.name[len(prefix_path.name) :]
    if suffix and suffix == path.suffix:
        return suffix
    stem_suffix = path.stem[len(prefix_path.name) :]
    if (
        len(stem_suffix) > 1
        and stem_suffix[0] in ("-", "_")
        and stem_suffix[1:].isdecimal()
    ):
        return suffix
    raise _GeneratedOutputError


def _matches_generated_output_prefix(path: Path, prefix_path: Path) -> bool:
    prefix = prefix_path.name
    name = path.name
    if name == prefix:
        return True
    if not name.startswith(prefix):
        return False
    remainder = name[len(prefix) :]
    return bool(remainder) and remainder[0] in (".", "-", "_")


def _generated_page_number(path: Path, prefix_path: Path) -> int | None:
    suffix = path.stem[len(prefix_path.name) :].lstrip("-_")
    if not suffix.isdecimal():
        return None
    return int(suffix)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus,
    exit_code: int | None,
    stdout_bytes: bytes,
    stderr_bytes: bytes,
    stdout_truncated: bool,
    stderr_truncated: bool,
    duration_ms: int,
    error_message: str | None,
    generated_files: tuple[GeneratedFile, ...] = (),
    error_code: ShellExecutionErrorCode | None = None,
    timed_out: bool = False,
    generated_output_replacements: tuple[tuple[str, str], ...] = (),
) -> ExecutionResult:
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    if generated_output_replacements:
        stdout = _scrub_generated_output_paths(
            stdout,
            generated_output_replacements,
        )
        stderr = _scrub_generated_output_paths(
            stderr,
            generated_output_replacements,
        )
    return ExecutionResult(
        backend=spec.backend,
        tool_name=spec.tool_name,
        command=spec.command,
        argv=_result_argv(spec),
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_media_type=spec.stdout_media_type,
        output_kind=spec.output_kind,
        generated_files=generated_files,
        stdout_bytes=len(stdout_bytes),
        stderr_bytes=len(stderr_bytes),
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        cancelled=False,
        duration_ms=duration_ms,
        error_code=(
            error_code
            if error_code is not None
            else SHELL_STATUS_ERROR_CODES[status]
        ),
        error_message=error_message,
        metadata=spec.metadata,
    )


def _scrub_generated_output_paths(
    value: str,
    replacements: tuple[tuple[str, str], ...],
) -> str:
    for source, replacement in replacements:
        if source:
            value = value.replace(source, replacement)
    return value


def _result_argv(spec: ExecutionSpec) -> tuple[str, ...]:
    if spec.output_plan is None:
        return spec.argv
    return spec.display_argv


def _duration_ms(start_time: float) -> int:
    return max(0, round((perf_counter() - start_time) * 1000))
