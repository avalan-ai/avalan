from ...entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from .entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ExecutionSpec,
    ShellResourceClass,
)
from .settings import ShellToolSettings

from asyncio import (
    FIRST_COMPLETED,
    Lock,
    Semaphore,
    create_subprocess_exec,
    gather,
    wait,
    wait_for,
)
from asyncio.streams import StreamReader
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from os import kill as os_kill
from os import name as os_name
from pathlib import Path
from signal import SIGKILL, SIGTERM
from typing import Any, Literal, final

_PROCESS_CLEANUP_GRACE_SECONDS = 0.2
ShellProcessStdinMode = Literal["spec", "pipe", "devnull"]


@final
class ShellProcessLimiter:
    def __init__(self, settings: ShellToolSettings) -> None:
        assert isinstance(
            settings,
            ShellToolSettings,
        ), "settings must be shell tool settings"
        self._max_concurrent_processes = settings.max_concurrent_processes
        self._max_concurrent_heavy_processes = (
            settings.max_concurrent_heavy_processes
        )
        self._process_semaphore = Semaphore(
            settings.max_concurrent_processes,
        )
        self._heavy_semaphore = Semaphore(
            settings.max_concurrent_heavy_processes,
        )

    @property
    def max_concurrent_processes(self) -> int:
        return self._max_concurrent_processes

    @property
    def max_concurrent_heavy_processes(self) -> int:
        return self._max_concurrent_heavy_processes

    @asynccontextmanager
    async def limit(
        self,
        resource_class: ShellResourceClass,
    ) -> AsyncIterator[None]:
        assert resource_class in (
            "standard",
            "heavy",
        ), "resource_class must be a shell resource class"
        if resource_class == "heavy":
            async with self._heavy_semaphore:
                async with self._process_semaphore:
                    yield
            return
        async with self._process_semaphore:
            yield


@final
class ShellProcessRuntime:
    def __init__(
        self,
        settings: ShellToolSettings,
        *,
        process_limiter: ShellProcessLimiter | None = None,
    ) -> None:
        assert isinstance(
            settings,
            ShellToolSettings,
        ), "settings must be shell tool settings"
        if process_limiter is None:
            process_limiter = ShellProcessLimiter(settings)
        assert isinstance(
            process_limiter,
            ShellProcessLimiter,
        ), "process_limiter must be a shell process limiter"
        self._process_limiter = process_limiter
        self._multi_process_limit_lock = Lock()

    @property
    def process_limiter(self) -> ShellProcessLimiter:
        return self._process_limiter

    @asynccontextmanager
    async def limit(
        self,
        resource_class: ShellResourceClass,
    ) -> AsyncIterator[None]:
        async with self._process_limiter.limit(resource_class):
            yield

    @asynccontextmanager
    async def limit_many(
        self,
        resource_classes: tuple[ShellResourceClass, ...],
    ) -> AsyncIterator[None]:
        assert isinstance(
            resource_classes,
            tuple,
        ), "resource_classes must be a tuple"
        assert resource_classes, "resource_classes must not be empty"
        assert (
            len(resource_classes)
            <= self._process_limiter.max_concurrent_processes
        ), "resource_classes must not exceed process capacity"
        assert (
            sum(1 for value in resource_classes if value == "heavy")
            <= self._process_limiter.max_concurrent_heavy_processes
        ), "resource_classes must not exceed heavy process capacity"
        # Byte pipelines need all children alive before stdout pumping starts.
        # Acquire the whole batch under one runtime lock so competing pipelines
        # cannot each hold a partial set of permits and wait forever.
        async with self._multi_process_limit_lock:
            async with AsyncExitStack() as stack:
                for resource_class in resource_classes:
                    await stack.enter_async_context(
                        self.limit(resource_class),
                    )
                yield

    async def spawn(
        self,
        spec: ExecutionSpec,
        *,
        runtime_output_prefix: Path | None = None,
        stdin_mode: ShellProcessStdinMode = "spec",
        stdin_fd: int | None = None,
        stdout_fd: int | None = None,
    ) -> Any:
        return await _spawn_process(
            spec,
            runtime_output_prefix=runtime_output_prefix,
            stdin_mode=stdin_mode,
            stdin_fd=stdin_fd,
            stdout_fd=stdout_fd,
        )


async def _spawn_process(
    spec: ExecutionSpec,
    *,
    runtime_output_prefix: Path | None = None,
    stdin_mode: ShellProcessStdinMode = "spec",
    stdin_fd: int | None = None,
    stdout_fd: int | None = None,
) -> Any:
    return await create_subprocess_exec(
        *_spawn_argv(
            spec,
            runtime_output_prefix=runtime_output_prefix,
        ),
        cwd=spec.cwd,
        env=spec.env,
        stdin=_stdin_target(spec, stdin_mode, stdin_fd=stdin_fd),
        stdout=_stdout_target(stdout_fd),
        stderr=PIPE,
        start_new_session=os_name == "posix",
    )


def _stdin_target(
    spec: ExecutionSpec,
    stdin_mode: ShellProcessStdinMode,
    *,
    stdin_fd: int | None = None,
) -> int:
    assert stdin_mode in (
        "spec",
        "pipe",
        "devnull",
    ), "stdin_mode must be spec, pipe, or devnull"
    if stdin_fd is not None:
        _assert_stdio_fd(stdin_fd, "stdin_fd")
        return stdin_fd
    if stdin_mode == "pipe":
        return PIPE
    if stdin_mode == "devnull":
        return DEVNULL
    return PIPE if spec.stdin is not None else DEVNULL


def _stdout_target(stdout_fd: int | None) -> int:
    if stdout_fd is None:
        return PIPE
    _assert_stdio_fd(stdout_fd, "stdout_fd")
    return stdout_fd


def _assert_stdio_fd(value: object, name: str) -> None:
    assert isinstance(value, int), f"{name} must be an integer file descriptor"
    assert not isinstance(value, bool), f"{name} must be an integer"
    assert value >= 0, f"{name} must be non-negative"


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
        # Process group teardown can race with child exit or permission
        # checks; fall back to the process method below.
        with suppress(ProcessLookupError, PermissionError, OSError):
            os_kill(-pid, signal_number)
            return
    fallback_method = getattr(process, fallback_method_name, None)
    if fallback_method is None:
        return
    # The fallback signal is best-effort during cleanup; the process may
    # already be gone by the time it runs.
    with suppress(ProcessLookupError, OSError):
        fallback_method()


async def _wait_for_process_exit(process: object) -> bool:
    wait_method = getattr(process, "wait", None)
    if wait_method is None:
        return True
    try:
        await wait_for(
            wait_method(),
            timeout=_PROCESS_CLEANUP_GRACE_SECONDS,
        )
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
                process_waiter.result()
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
        # The child may exit before consuming stdin; cleanup still runs below.
        return
    finally:
        await _close_stdin_writer(writer)


async def _close_stdin_writer(writer: object) -> None:
    close = getattr(writer, "close", None)
    if close is None:
        return
    try:
        close()
    except (BrokenPipeError, ConnectionResetError):
        # Closing stdin is best-effort when the child has already exited.
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
        # wait_closed can lose a teardown race or exceed the cleanup grace
        # period after the close request has already been issued.
        return


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
