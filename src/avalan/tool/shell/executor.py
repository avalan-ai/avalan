from ...entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from .entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
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
    make_directory,
    private_temp_directory,
    probe_image_dimensions,
    remove_file,
    remove_tree,
    resolve_policy_path,
)
from .filesystem import (
    read_bytes as _read_bytes,
)
from .filesystem import (
    write_bytes as _write_bytes,
)
from .process import (
    ShellProcessLimiter,
    ShellProcessRuntime,
    _cancel_reader_tasks,
    _collect_stream,
    _emit_stream_event,
    _GeneratedOutputError,
    _kill_process_group,
    _reader_results,
    _reader_task_failed,
    _StreamCapture,
    _terminate_process_group,
    _wait_for_process_or_reader_failure,
    _write_stdin_and_wait,
)
from .settings import ShellToolSettings

from asyncio import (
    CancelledError,
    create_task,
    gather,
    wait_for,
)
from collections.abc import Awaitable, Callable
from pathlib import Path
from stat import S_IMODE
from time import perf_counter
from typing import Protocol
from uuid import uuid4


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
    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        *,
        process_limiter: ShellProcessLimiter | None = None,
        process_runtime: ShellProcessRuntime | None = None,
    ) -> None:
        settings = ShellToolSettings() if settings is None else settings
        assert isinstance(
            settings,
            ShellToolSettings,
        ), "settings must be shell tool settings"
        assert not (
            process_limiter is not None and process_runtime is not None
        ), "process_limiter and process_runtime cannot both be set"
        if process_runtime is None:
            process_runtime = ShellProcessRuntime(
                settings,
                process_limiter=process_limiter,
            )
        assert isinstance(
            process_runtime,
            ShellProcessRuntime,
        ), "process_runtime must be a shell process runtime"
        self._settings = settings
        self._process_runtime = process_runtime

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
        if spec.backend != "local":
            return ExecutionResult(
                backend=spec.backend,
                tool_name=spec.tool_name,
                command=spec.command,
                argv=_result_argv(spec),
                display_argv=spec.display_argv,
                cwd=spec.cwd,
                display_cwd=spec.display_cwd,
                status=ShellExecutionStatus.POLICY_DENIED,
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
                error_code=ShellExecutionErrorCode.POLICY_DENIED,
                error_message=(
                    "local execution cannot run isolated shell specs"
                ),
                metadata=spec.metadata,
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
        async with self._process_runtime.limit(spec.resource_class):
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
                process = await self._process_runtime.spawn(
                    spec,
                    runtime_output_prefix=runtime_output_prefix,
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
                        settings=self._settings,
                    )
                except (OSError, _GeneratedOutputError) as error:
                    status = ShellExecutionStatus.TOO_LARGE
                    error_code = (
                        ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED
                    )
                    generated_error = (
                        str(error)
                        if isinstance(error, _GeneratedOutputError)
                        else ""
                    )
                    error_message = (
                        generated_error or "generated output capture failed"
                    )
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
    *,
    settings: ShellToolSettings,
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
        display_path = _display_generated_path(path, prefix_path, plan)
        generated_metadata: dict[str, object] = {}
        if content_base64 is None:
            generated_metadata[
                GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY
            ] = await _materialize_generated_output_file(
                path,
                display_path,
                settings,
            )
        generated_files.append(
            GeneratedFile(
                display_path=display_path,
                media_type=media_type,
                suffix=suffix,
                bytes=metadata.size,
                sha256=sha256,
                page=_generated_page_number(path, prefix_path),
                width=width,
                height=height,
                content_base64=content_base64,
                metadata=generated_metadata,
            )
        )
    return tuple(generated_files)


async def _materialize_generated_output_file(
    path: Path,
    display_path: str,
    settings: ShellToolSettings,
) -> str:
    workspace_root = Path(settings.workspace_root).resolve()
    materialized_root = workspace_root / settings.materialized_input_files_dir
    await _make_directory_tree(materialized_root, stop_at=workspace_root)
    target_dir = materialized_root / uuid4().hex
    await make_directory(target_dir)
    target_path = target_dir / _safe_materialized_filename(
        Path(display_path).name
    )
    await _write_bytes(target_path, await _read_bytes(path))
    return str(target_path.resolve())


async def _make_directory_tree(path: Path, *, stop_at: Path) -> None:
    if path == stop_at:
        return
    try:
        await make_directory(path)
    except FileNotFoundError:
        if path.parent == path or not _is_relative_to(path.parent, stop_at):
            raise
        await _make_directory_tree(path.parent, stop_at=stop_at)
        try:
            await make_directory(path)
        except FileExistsError:
            pass
    except FileExistsError:
        pass


def _safe_materialized_filename(filename: str) -> str:
    safe = filename.lstrip(".")
    return safe or "generated"


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
        raise _GeneratedOutputError(
            f"generated image dimensions {width}x{height} exceed "
            f"maximum long edge {plan.max_raster_long_edge_pixels}"
        )
    if (
        plan.max_raster_pixels is not None
        and width * height > plan.max_raster_pixels
    ):
        raise _GeneratedOutputError(
            f"generated image dimensions {width}x{height} exceed "
            f"maximum pixels {plan.max_raster_pixels}"
        )
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
