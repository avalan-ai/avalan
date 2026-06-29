from ...entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from .entities import (
    SHELL_STATUS_ERROR_CODES,
    ExecutionResult,
    ExecutionSpec,
    ShellCompositionResult,
    ShellCompositionSpec,
    ShellExecutionErrorCode,
    ShellExecutionModeValue,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellExecutionStepSpec,
)
from .executor import (
    CommandExecutor,
    LocalCommandExecutor,
    _duration_ms,
    _result,
    _result_argv,
    _status_for_exit_code,
)
from .process import (
    ShellProcessLimiter,
    ShellProcessRuntime,
    _cancel_reader_tasks,
    _close_stdin_writer,
    _collect_stream,
    _emit_stream_event,
    _kill_process_group,
    _reader_results,
    _reader_task_failed,
    _StreamCapture,
    _terminate_process_group,
    _wait_for_process_or_reader_failure,
    _write_stdin_and_wait,
)
from .settings import ShellPipelineTransport, ShellToolSettings

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Task,
    create_task,
    gather,
    wait,
)
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from os import close as os_close
from os import pipe as os_pipe
from time import perf_counter
from typing import Any, Protocol, cast, final

_BENIGN_STATUSES = {
    ShellExecutionStatus.COMPLETED,
    ShellExecutionStatus.NO_MATCHES,
}
_PIPELINE_CLEANUP_DRAIN_SECONDS = 0.2


class CompositionExecutor(Protocol):
    async def execute_composition(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ShellCompositionResult:
        raise NotImplementedError


class LocalCompositionExecutor:
    def __init__(
        self,
        settings: ShellToolSettings | None = None,
        *,
        command_executor: CommandExecutor | None = None,
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
        self._command_executor = command_executor or LocalCommandExecutor(
            settings=settings,
            process_runtime=process_runtime,
        )

    async def execute_composition(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ShellCompositionResult:
        if not isinstance(spec, ShellCompositionSpec):
            raise NotImplementedError(
                "local shell composition execution is not implemented"
            )
        if spec.mode == "pipeline":
            return await self._execute_pipeline(spec, stream=stream)
        if spec.mode == "serial":
            return await self._execute_serial(spec, stream=stream)
        if spec.mode == "parallel":
            return await self._execute_parallel(spec, stream=stream)
        raise NotImplementedError("unsupported shell composition mode")

    async def _execute_pipeline(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    ) -> ShellCompositionResult:
        start_time = perf_counter()
        blocked = _pipeline_preflight_result(
            spec,
            max_concurrent_processes=(
                self._process_runtime.process_limiter.max_concurrent_processes
            ),
            max_concurrent_heavy_processes=(
                self._process_runtime.process_limiter.max_concurrent_heavy_processes
            ),
            start_time=start_time,
        )
        if blocked is not None:
            return blocked
        state = _PipelineState(
            spec,
            stream=stream,
            chunk_size=self._settings.stream_read_chunk_bytes,
            transport=self._settings.pipeline_transport,
        )
        worker = create_task(self._run_pipeline(spec, state, stream=stream))
        try:
            done, _ = await wait({worker}, timeout=spec.timeout_seconds)
            if worker in done:
                return await worker
            state.mark_error(
                ShellExecutionStatus.TIMEOUT,
                "composition timed out",
            )
            await state.close_stdin_writers()
            await state.terminate_processes()
            await state.drain_tasks()
            worker.cancel()
            await gather(worker, return_exceptions=True)
            result = state.result(
                status=ShellExecutionStatus.TIMEOUT,
                duration_ms=_duration_ms(start_time),
                timed_out=True,
                error_message="composition timed out",
            )
            await _emit_step_completion_progress(stream, result.steps)
            return result
        except CancelledError:
            await state.close_stdin_writers()
            await state.kill_processes()
            await state.cancel_tasks()
            worker.cancel()
            await gather(worker, return_exceptions=True)
            raise

    async def _run_pipeline(
        self,
        spec: ShellCompositionSpec,
        state: "_PipelineState",
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    ) -> ShellCompositionResult:
        start_time = perf_counter()
        try:
            resource_classes = tuple(
                stage.step.spec.resource_class for stage in state.stages
            )
            async with self._process_runtime.limit_many(resource_classes):
                try:
                    state.open_native_pipes()
                except OSError:
                    state.mark_error(
                        ShellExecutionStatus.SPAWN_FAILED,
                        "process spawn failed",
                    )
                    await state.close_stdin_writers()
                    return state.result(
                        status=ShellExecutionStatus.SPAWN_FAILED,
                        duration_ms=_duration_ms(start_time),
                        error_message="process spawn failed",
                    )
                for stage in state.stages:
                    if not await self._spawn_pipeline_stage(stage, state):
                        await state.close_stdin_writers()
                        await state.terminate_processes()
                        await state.drain_tasks()
                        return state.result(
                            status=ShellExecutionStatus.SPAWN_FAILED,
                            duration_ms=_duration_ms(start_time),
                            error_message="process spawn failed",
                        )
                    await _emit_stream_event(
                        stream,
                        kind=ToolExecutionStreamKind.LOG,
                        content="process started",
                        metadata=_stage_stream_metadata(stage),
                    )
                    await _emit_stream_event(
                        stream,
                        kind=ToolExecutionStreamKind.PROGRESS,
                        content="started",
                        progress=0.0,
                        metadata=_stage_stream_metadata(stage),
                    )
                    state.start_stderr_reader(stage)
                state.start_stdout_routes()
                pipeline_error = await state.wait_for_tasks()
                if pipeline_error is not None:
                    status, error_message = pipeline_error
                    result = state.result(
                        status=status,
                        duration_ms=_duration_ms(start_time),
                        error_message=error_message,
                    )
                    await _emit_step_completion_progress(
                        stream,
                        result.steps,
                    )
                    return result
                result = state.result(duration_ms=_duration_ms(start_time))
                await _emit_step_completion_progress(stream, result.steps)
                return result
        except Exception:
            await state.close_stdin_writers()
            await state.kill_processes()
            await state.cancel_tasks()
            raise

    async def _spawn_pipeline_stage(
        self,
        stage: "_PipelineStage",
        state: "_PipelineState",
    ) -> bool:
        stage.start_time = perf_counter()
        try:
            stage.process = await self._process_runtime.spawn(
                stage.step.spec,
                stdin_mode="devnull" if stage.index == 0 else "pipe",
                stdin_fd=state.stdin_fd_for_stage(stage),
                stdout_fd=state.stdout_fd_for_stage(stage),
            )
            state.stage_spawned(stage)
        except (OSError, ValueError):
            stage.mark_error(
                ShellExecutionStatus.SPAWN_FAILED,
                "process spawn failed",
            )
            return False
        return True

    async def _execute_serial(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    ) -> ShellCompositionResult:
        start_time = perf_counter()
        results: list[ShellExecutionStepResult] = []
        route_bytes: dict[str, bytes] = {}
        referenced_step_ids = _referenced_step_ids(spec)
        worker = create_task(
            self._run_serial(
                spec,
                results,
                route_bytes,
                referenced_step_ids,
                stream=stream,
            )
        )
        try:
            done, _ = await wait({worker}, timeout=spec.timeout_seconds)
            if worker in done:
                try:
                    await worker
                except _IntermediateTooLarge:
                    return _composition_result(
                        spec,
                        tuple(results),
                        status=ShellExecutionStatus.TOO_LARGE,
                        duration_ms=_duration_ms(start_time),
                        error_message="routed stdout exceeded cap",
                    )
                return _composition_result(
                    spec,
                    tuple(results),
                    duration_ms=_duration_ms(start_time),
                )
            worker.cancel()
            await gather(worker, return_exceptions=True)
            return _composition_result(
                spec,
                _ordered_serial_results(
                    spec,
                    results,
                    timeout_missing=True,
                ),
                status=ShellExecutionStatus.TIMEOUT,
                duration_ms=_duration_ms(start_time),
                timed_out=True,
                error_message="composition timed out",
            )
        except CancelledError:
            worker.cancel()
            await gather(worker, return_exceptions=True)
            raise

    async def _run_serial(
        self,
        spec: ShellCompositionSpec,
        results: list[ShellExecutionStepResult],
        route_bytes: dict[str, bytes],
        referenced_step_ids: set[str],
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    ) -> None:
        by_id: dict[str, ShellExecutionStepResult] = {}
        final_index = len(spec.steps) - 1
        for index, step in enumerate(spec.steps):
            stdout_visible = index == final_index
            if step.stdin_from is None:
                result = await self._command_executor.execute(
                    step.spec,
                    stream=_stage_stream_callback(
                        stream,
                        step,
                        index,
                        stdout_visible=stdout_visible,
                    ),
                )
            else:
                producer = by_id.get(step.stdin_from.step_id)
                stdin = route_bytes.get(step.stdin_from.step_id)
                if producer is None or stdin is None:
                    results.append(
                        _blocked_step_result(
                            step,
                            status=ShellExecutionStatus.TOOL_ERROR,
                            error_message="routed stdin source is unavailable",
                        )
                    )
                    return
                if producer.status not in _BENIGN_STATUSES:
                    results.append(
                        _blocked_step_result(
                            step,
                            status=ShellExecutionStatus.TOOL_ERROR,
                            error_message="routed stdin source failed",
                        )
                    )
                    return
                result = await self._execute_step_with_stdin(
                    step,
                    stdin,
                    stream=_stage_stream_callback(
                        stream,
                        step,
                        index,
                        stdout_visible=stdout_visible,
                    ),
                )
            step_result = _step_result_from_execution(
                step,
                result,
                stdout_visible=stdout_visible,
            )
            results.append(step_result)
            by_id[step.id] = step_result
            if step.id in referenced_step_ids:
                if result.stdout_truncated:
                    raise _IntermediateTooLarge
                stdout = result.stdout.encode()
                if len(stdout) > spec.max_intermediate_bytes:
                    raise _IntermediateTooLarge
                route_bytes[step.id] = stdout

    async def _execute_parallel(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    ) -> ShellCompositionResult:
        start_time = perf_counter()
        results: dict[str, ShellExecutionStepResult] = {}
        tasks: dict[Task[ExecutionResult], ShellExecutionStepSpec] = {}
        for index, step in enumerate(spec.steps):
            tasks[
                create_task(
                    self._command_executor.execute(
                        step.spec,
                        stream=_stage_stream_callback(
                            stream,
                            step,
                            index,
                            stdout_visible=False,
                        ),
                    )
                )
            ] = step
        try:
            done, pending = await wait(
                set(tasks),
                timeout=spec.timeout_seconds,
            )
            try:
                for task in done:
                    step = tasks[task]
                    results[step.id] = _step_result_from_execution(
                        step,
                        await task,
                    )
            except Exception:
                for task in pending:
                    task.cancel()
                await gather(*pending, return_exceptions=True)
                raise
            if pending:
                for task in pending:
                    task.cancel()
                await gather(*pending, return_exceptions=True)
                ordered_results = _ordered_parallel_results(
                    spec,
                    results,
                    timeout_missing=True,
                )
                return _composition_result(
                    spec,
                    ordered_results,
                    status=ShellExecutionStatus.TIMEOUT,
                    duration_ms=_duration_ms(start_time),
                    timed_out=True,
                    error_message="composition timed out",
                )
            ordered_results = _ordered_parallel_results(spec, results)
            return _composition_result(
                spec,
                ordered_results,
                duration_ms=_duration_ms(start_time),
            )
        except CancelledError:
            for task in tasks:
                task.cancel()
            await gather(*tasks, return_exceptions=True)
            raise

    async def _execute_step_with_stdin(
        self,
        step: ShellExecutionStepSpec,
        stdin: bytes,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        spec = step.spec
        blocked = _blocked_execution_result(spec)
        if blocked is not None:
            return blocked
        start_time = perf_counter()
        async with self._process_runtime.limit(spec.resource_class):
            process = None
            stdout_reader = None
            stderr_reader = None
            process_waiter = None
            try:
                try:
                    process = await self._process_runtime.spawn(
                        spec,
                        stdin_mode="pipe",
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
                    )
                stdout_capture = _StreamCapture()
                stderr_capture = _StreamCapture()
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
                    _write_stdin_and_wait(process, stdin)
                )
                reader_failed = await _wait_for_process_or_reader_failure(
                    process_waiter,
                    stdout_reader,
                    stderr_reader,
                )
                if reader_failed:
                    await _kill_process_group(process)
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
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
                return _result(
                    spec,
                    status=status,
                    exit_code=exit_code,
                    stdout_bytes=stdout_bytes,
                    stderr_bytes=stderr_bytes,
                    stdout_truncated=stdout_truncated,
                    stderr_truncated=stderr_truncated,
                    duration_ms=_duration_ms(start_time),
                    error_code=error_code,
                    error_message=error_message,
                )
            except CancelledError:
                if process is not None:
                    await _kill_process_group(process)
                if process_waiter is not None:
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
                await _cancel_reader_tasks(stdout_reader, stderr_reader)
                raise
            except Exception:
                if process is not None:
                    await _kill_process_group(process)
                if process_waiter is not None:
                    process_waiter.cancel()
                    await gather(process_waiter, return_exceptions=True)
                await _cancel_reader_tasks(stdout_reader, stderr_reader)
                raise


@final
class BackendBoundaryCompositionExecutor:
    """Delegate isolated serial/parallel and deny byte pipelines.

    Sandbox and container byte pipelines require a future trusted structured
    runner. They must not be lowered to shell text or shell evaluation.
    """

    def __init__(
        self,
        *,
        backend: ShellExecutionModeValue,
        command_executor: CommandExecutor,
        settings: ShellToolSettings | None = None,
    ) -> None:
        assert backend in {
            "sandbox",
            "container",
        }, "backend must be sandbox or container"
        assert hasattr(
            command_executor,
            "execute",
        ), "command_executor must execute commands"
        self._backend = backend
        self._delegate = LocalCompositionExecutor(
            settings=settings,
            command_executor=command_executor,
        )

    async def execute_composition(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ShellCompositionResult:
        if not isinstance(spec, ShellCompositionSpec):
            raise NotImplementedError(
                "isolated shell composition execution is not implemented"
            )
        start_time = perf_counter()
        backend_error = _backend_boundary_error(spec, self._backend)
        if backend_error is not None:
            return _composition_not_started_result(
                spec,
                status=ShellExecutionStatus.POLICY_DENIED,
                start_time=start_time,
                error_message=backend_error,
            )
        if spec.mode == "pipeline" or _has_stdin_routing(spec):
            return _composition_not_started_result(
                spec,
                status=ShellExecutionStatus.POLICY_DENIED,
                start_time=start_time,
                error_message=_isolated_byte_pipeline_error(self._backend),
            )
        if spec.mode == "serial":
            return await self._delegate.execute_composition(
                spec,
                stream=stream,
            )
        if spec.mode == "parallel":
            return await self._delegate.execute_composition(
                spec,
                stream=stream,
            )
        raise NotImplementedError("unsupported shell composition mode")


@final
@dataclass(slots=True)
class _PipelineStage:
    step: ShellExecutionStepSpec
    index: int
    stdout_visible: bool
    pipeline_transport: ShellPipelineTransport
    stdout_capture_available: bool
    process: object | None = None
    start_time: float = 0.0
    stdout_capture: _StreamCapture = field(default_factory=_StreamCapture)
    stderr_capture: _StreamCapture = field(default_factory=_StreamCapture)
    status_override: ShellExecutionStatus | None = None
    error_message: str | None = None

    def mark_error(
        self,
        status: ShellExecutionStatus,
        error_message: str,
    ) -> None:
        self.status_override = status
        self.error_message = error_message


@final
class _PipelineState:
    def __init__(
        self,
        spec: ShellCompositionSpec,
        *,
        stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
        chunk_size: int,
        transport: ShellPipelineTransport,
    ) -> None:
        self._spec = spec
        self._stream = stream
        self._chunk_size = chunk_size
        self._transport = transport
        final_index = len(spec.steps) - 1
        self.stages = [
            _PipelineStage(
                step=step,
                index=index,
                stdout_visible=index == final_index,
                pipeline_transport=transport,
                stdout_capture_available=(
                    transport != "native" or index == final_index
                ),
            )
            for index, step in enumerate(spec.steps)
        ]
        self._tasks: set[Task[Any]] = set()
        self._task_roles: dict[Task[Any], str] = {}
        self._error_status: ShellExecutionStatus | None = None
        self._error_message: str | None = None
        self._native_pipes: _NativePipelinePipes | None = None

    def open_native_pipes(self) -> None:
        if self._transport != "native":
            return
        self._native_pipes = _NativePipelinePipes(len(self.stages))

    def stdin_fd_for_stage(self, stage: _PipelineStage) -> int | None:
        if self._native_pipes is None:
            return None
        return self._native_pipes.stdin_fd(stage.index)

    def stdout_fd_for_stage(self, stage: _PipelineStage) -> int | None:
        if self._native_pipes is None:
            return None
        return self._native_pipes.stdout_fd(stage.index)

    def stage_spawned(self, stage: _PipelineStage) -> None:
        if self._native_pipes is None:
            return
        self._native_pipes.stage_spawned(stage.index)

    def mark_error(
        self,
        status: ShellExecutionStatus,
        error_message: str,
    ) -> None:
        if self._error_status is None:
            self._error_status = status
            self._error_message = error_message
        for stage in self.stages:
            if stage.process is not None and stage.status_override is None:
                stage.mark_error(status, error_message)

    def start_stderr_reader(self, stage: _PipelineStage) -> None:
        assert stage.process is not None, "stage process must be spawned"
        process = cast(Any, stage.process)
        task = create_task(
            _collect_stream(
                process.stderr,
                stage.step.spec.max_stderr_bytes,
                self._chunk_size,
                stage.stderr_capture,
                stream_event=_stage_stream_callback(
                    self._stream,
                    stage.step,
                    stage.index,
                ),
                stream_kind=ToolExecutionStreamKind.STDERR,
            )
        )
        self._add_task(task, "reader")

    def start_stdout_routes(self) -> None:
        for index, stage in enumerate(self.stages):
            assert stage.process is not None, "stage process must be spawned"
            process = cast(Any, stage.process)
            if index == len(self.stages) - 1:
                stdout_task = create_task(
                    _collect_stream(
                        process.stdout,
                        self._spec.max_stdout_bytes,
                        self._chunk_size,
                        stage.stdout_capture,
                        stream_event=_stage_stream_callback(
                            self._stream,
                            stage.step,
                            stage.index,
                        ),
                        stream_kind=ToolExecutionStreamKind.STDOUT,
                    )
                )
                self._add_task(stdout_task, "reader")
            else:
                if self._transport == "buffered":
                    pump_task = create_task(
                        _pump_intermediate_stdout(
                            stage,
                            self.stages[index + 1],
                            max_bytes=self._spec.max_intermediate_bytes,
                            chunk_size=self._chunk_size,
                        )
                    )
                    self._add_task(pump_task, "pump")
            wait_task = create_task(process.wait())
            self._add_task(wait_task, "wait")

    async def wait_for_tasks(
        self,
    ) -> tuple[ShellExecutionStatus, str] | None:
        pending = set(self._tasks)
        while pending:
            done, pending = await wait(
                pending,
                return_when=FIRST_COMPLETED,
            )
            for task in done:
                try:
                    task.result()
                except _IntermediateTooLarge:
                    self.mark_error(
                        ShellExecutionStatus.TOO_LARGE,
                        "intermediate stdout exceeded cap",
                    )
                    await self.close_stdin_writers()
                    await self.terminate_processes()
                    await self._cancel_pending(pending)
                    return (
                        ShellExecutionStatus.TOO_LARGE,
                        "intermediate stdout exceeded cap",
                    )
                except Exception:
                    if self._task_roles.get(task) == "wait":
                        raise
                    self.mark_error(
                        ShellExecutionStatus.TOOL_ERROR,
                        "stream collection failed",
                    )
                    await self.kill_processes()
                    await self._cancel_pending(pending)
                    return (
                        ShellExecutionStatus.TOOL_ERROR,
                        "stream collection failed",
                    )
        return None

    async def close_stdin_writers(self) -> None:
        self.close_native_fds()
        for process in self._processes():
            await _close_stdin_writer(getattr(process, "stdin", None))

    def close_native_fds(self) -> None:
        if self._native_pipes is None:
            return
        self._native_pipes.close_all()

    async def terminate_processes(self) -> None:
        await gather(
            *(
                _terminate_process_group(process)
                for process in self._processes()
            ),
            return_exceptions=True,
        )

    async def kill_processes(self) -> None:
        await gather(
            *(_kill_process_group(process) for process in self._processes()),
            return_exceptions=True,
        )

    async def drain_tasks(self) -> None:
        if not self._tasks:
            return
        _, pending = await wait(
            self._tasks,
            timeout=_PIPELINE_CLEANUP_DRAIN_SECONDS,
        )
        await self._cancel_pending(pending)

    async def cancel_tasks(self) -> None:
        await self._cancel_pending(self._tasks)

    def result(
        self,
        *,
        status: ShellExecutionStatus | None = None,
        duration_ms: int,
        timed_out: bool = False,
        error_message: str | None = None,
    ) -> ShellCompositionResult:
        step_results = tuple(
            _step_result_from_pipeline_stage(stage) for stage in self.stages
        )
        final_stage = self.stages[-1]
        stdout_bytes, stdout_truncated = final_stage.stdout_capture.snapshot()
        result_status = (
            status if status is not None else _aggregate_status(step_results)
        )
        result_error_message = error_message or _aggregate_error_message(
            result_status,
            step_results,
        )
        stderr, stderr_bytes, stderr_truncated = _aggregate_stderr(
            self._spec,
            step_results,
        )
        return ShellCompositionResult(
            mode=self._spec.mode,
            status=result_status,
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr,
            steps=step_results,
            stdout_bytes=len(stdout_bytes),
            stderr_bytes=stderr_bytes,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            timed_out=timed_out,
            cancelled=False,
            duration_ms=duration_ms,
            error_code=SHELL_STATUS_ERROR_CODES[result_status],
            error_message=result_error_message,
            metadata={
                "mode": self._spec.mode,
                "pipeline_transport": self._transport,
            },
        )

    def _add_task(self, task: Task[Any], role: str) -> None:
        self._tasks.add(task)
        self._task_roles[task] = role

    async def _cancel_pending(self, pending: set[Task[Any]]) -> None:
        await _cancel_reader_tasks(*pending)

    def _processes(self) -> tuple[object, ...]:
        return tuple(
            stage.process for stage in self.stages if stage.process is not None
        )


class _IntermediateTooLarge(Exception):
    pass


@final
class _NativePipelinePipes:
    def __init__(self, stage_count: int) -> None:
        assert stage_count > 0, "stage_count must be positive"
        self._pipes: list[list[int | None]] = []
        try:
            for _ in range(stage_count - 1):
                read_fd, write_fd = os_pipe()
                self._pipes.append([read_fd, write_fd])
        except OSError:
            self.close_all()
            raise

    def stdin_fd(self, stage_index: int) -> int | None:
        assert stage_index >= 0, "stage_index must be non-negative"
        if stage_index == 0:
            return None
        read_fd = self._pipes[stage_index - 1][0]
        assert read_fd is not None, "native pipeline stdin fd is closed"
        return read_fd

    def stdout_fd(self, stage_index: int) -> int | None:
        assert stage_index >= 0, "stage_index must be non-negative"
        if stage_index >= len(self._pipes):
            return None
        write_fd = self._pipes[stage_index][1]
        assert write_fd is not None, "native pipeline stdout fd is closed"
        return write_fd

    def stage_spawned(self, stage_index: int) -> None:
        assert stage_index >= 0, "stage_index must be non-negative"
        if stage_index > 0:
            self._close_pipe_end(stage_index - 1, 0)
        if stage_index < len(self._pipes):
            self._close_pipe_end(stage_index, 1)

    def close_all(self) -> None:
        for pipe_index in range(len(self._pipes)):
            self._close_pipe_end(pipe_index, 0)
            self._close_pipe_end(pipe_index, 1)

    def _close_pipe_end(self, pipe_index: int, end_index: int) -> None:
        fd = self._pipes[pipe_index][end_index]
        if fd is None:
            return
        self._pipes[pipe_index][end_index] = None
        with suppress(OSError):
            os_close(fd)


async def _pump_intermediate_stdout(
    source: _PipelineStage,
    destination: _PipelineStage,
    *,
    max_bytes: int,
    chunk_size: int,
) -> None:
    assert source.process is not None, "source process must be spawned"
    assert (
        destination.process is not None
    ), "destination process must be spawned"
    stdout = getattr(source.process, "stdout", None)
    writer = getattr(destination.process, "stdin", None)
    write_open = True
    try:
        if stdout is None:
            return
        while True:
            chunk = await stdout.read(chunk_size)
            if not chunk:
                return
            captured = source.stdout_capture.append(chunk, max_bytes)
            if captured and write_open:
                write_open = await _write_pipe_chunk(writer, captured)
            if len(captured) < len(chunk):
                source.mark_error(
                    ShellExecutionStatus.TOO_LARGE,
                    "intermediate stdout exceeded cap",
                )
                raise _IntermediateTooLarge
    finally:
        await _close_stdin_writer(writer)


async def _write_pipe_chunk(writer: object, chunk: bytes) -> bool:
    if writer is None:
        return False
    write = getattr(writer, "write", None)
    if write is None:
        return False
    try:
        write(chunk)
        drain = getattr(writer, "drain", None)
        if drain is not None:
            await drain()
    except (BrokenPipeError, ConnectionResetError):
        return False
    return True


def _pipeline_preflight_result(
    spec: ShellCompositionSpec,
    *,
    max_concurrent_processes: int,
    max_concurrent_heavy_processes: int,
    start_time: float,
) -> ShellCompositionResult | None:
    capacity_error = _pipeline_capacity_error(
        spec,
        max_concurrent_processes=max_concurrent_processes,
        max_concurrent_heavy_processes=max_concurrent_heavy_processes,
    )
    if capacity_error is not None:
        return _pipeline_not_started_result(
            spec,
            status=ShellExecutionStatus.POLICY_DENIED,
            start_time=start_time,
            error_message=capacity_error,
        )
    for index, step in enumerate(spec.steps):
        blocked = _blocked_execution_result(step.spec)
        if blocked is not None:
            steps: list[ShellExecutionStepResult] = []
            final_index = len(spec.steps) - 1
            for result_index, result_step in enumerate(spec.steps):
                if result_index == index:
                    steps.append(
                        _step_result_from_execution(
                            result_step,
                            blocked,
                            stdout_visible=result_index == final_index,
                        )
                    )
                    continue
                steps.append(
                    _blocked_step_result(
                        result_step,
                        status=ShellExecutionStatus.TOOL_ERROR,
                        error_message=(
                            "pipeline was not started because step "
                            f"'{step.id}' failed preflight"
                        ),
                        stdout_visible=result_index == final_index,
                    )
                )
            return _composition_result(
                spec,
                tuple(steps),
                status=blocked.status,
                duration_ms=_duration_ms(start_time),
                error_message=blocked.error_message,
            )
    return None


def _pipeline_capacity_error(
    spec: ShellCompositionSpec,
    *,
    max_concurrent_processes: int,
    max_concurrent_heavy_processes: int,
) -> str | None:
    process_count = len(spec.steps)
    if process_count > max_concurrent_processes:
        return (
            "pipeline requires "
            f"{process_count} concurrent process slots but local limit is "
            f"{max_concurrent_processes}"
        )
    heavy_count = sum(
        1 for step in spec.steps if step.spec.resource_class == "heavy"
    )
    if heavy_count > max_concurrent_heavy_processes:
        return (
            "pipeline requires "
            f"{heavy_count} concurrent heavy process slots but local heavy "
            f"limit is {max_concurrent_heavy_processes}"
        )
    return None


def _pipeline_not_started_result(
    spec: ShellCompositionSpec,
    *,
    status: ShellExecutionStatus,
    start_time: float,
    error_message: str,
) -> ShellCompositionResult:
    return _composition_not_started_result(
        spec,
        status=status,
        start_time=start_time,
        error_message=error_message,
    )


def _composition_not_started_result(
    spec: ShellCompositionSpec,
    *,
    status: ShellExecutionStatus,
    start_time: float,
    error_message: str,
) -> ShellCompositionResult:
    final_index = len(spec.steps) - 1
    return _composition_result(
        spec,
        tuple(
            _blocked_step_result(
                step,
                status=status,
                error_message=error_message,
                stdout_visible=index == final_index,
            )
            for index, step in enumerate(spec.steps)
        ),
        status=status,
        duration_ms=_duration_ms(start_time),
        error_message=error_message,
    )


def _backend_boundary_error(
    spec: ShellCompositionSpec,
    backend: ShellExecutionModeValue,
) -> str | None:
    for step in spec.steps:
        if step.spec.backend != backend:
            return (
                f"{backend} composition cannot run "
                f"{step.spec.backend} shell specs"
            )
    return None


def _has_stdin_routing(spec: ShellCompositionSpec) -> bool:
    return any(step.stdin_from is not None for step in spec.steps)


def _isolated_byte_pipeline_error(
    backend: ShellExecutionModeValue,
) -> str:
    return (
        f"{backend} byte pipelines require a trusted structured runner "
        "and cannot be lowered to shell evaluation"
    )


def _blocked_execution_result(spec: ExecutionSpec) -> ExecutionResult | None:
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
            error_message="local execution cannot run isolated shell specs",
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
    return None


def _step_result_from_pipeline_stage(
    stage: _PipelineStage,
) -> ShellExecutionStepResult:
    spec = stage.step.spec
    stdout_bytes, stdout_truncated = stage.stdout_capture.snapshot()
    stderr_bytes, stderr_truncated = stage.stderr_capture.snapshot()
    exit_code = _stage_exit_code(stage)
    status = stage.status_override or (
        _status_for_exit_code(exit_code, spec)
        if exit_code is not None
        else ShellExecutionStatus.SPAWN_FAILED
    )
    error_message = stage.error_message
    if error_message is None and status is not ShellExecutionStatus.COMPLETED:
        error_message = (
            "process spawn failed"
            if exit_code is None
            else f"command exited with status {exit_code}"
        )
    return ShellExecutionStepResult(
        id=stage.step.id,
        command=spec.command,
        status=status,
        exit_code=exit_code,
        stdout=(
            stdout_bytes.decode("utf-8", errors="replace")
            if stage.stdout_visible
            else ""
        ),
        stderr=stderr_bytes.decode("utf-8", errors="replace"),
        stdout_bytes=len(stdout_bytes),
        stderr_bytes=len(stderr_bytes),
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        duration_ms=_stage_duration_ms(stage),
        error_code=SHELL_STATUS_ERROR_CODES[status],
        error_message=error_message,
        metadata=_step_metadata(
            stage.step,
            stdout_visible=stage.stdout_visible,
            pipeline_transport=stage.pipeline_transport,
            stdout_capture_available=stage.stdout_capture_available,
        ),
    )


def _stage_exit_code(stage: _PipelineStage) -> int | None:
    if stage.process is None:
        return None
    returncode = getattr(stage.process, "returncode", None)
    if returncode is None:
        return None
    assert isinstance(returncode, int), "returncode must be an integer"
    return returncode


def _stage_duration_ms(stage: _PipelineStage) -> int:
    if stage.start_time <= 0:
        return 0
    return _duration_ms(stage.start_time)


def _step_result_from_execution(
    step: ShellExecutionStepSpec,
    result: ExecutionResult,
    *,
    stdout_visible: bool = True,
) -> ShellExecutionStepResult:
    return ShellExecutionStepResult(
        id=step.id,
        command=result.command,
        status=result.status,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        stdout_bytes=result.stdout_bytes,
        stderr_bytes=result.stderr_bytes,
        stdout_truncated=result.stdout_truncated,
        stderr_truncated=result.stderr_truncated,
        duration_ms=result.duration_ms,
        error_code=result.error_code,
        error_message=result.error_message,
        metadata=_step_metadata(
            step,
            result.metadata,
            stdout_visible=stdout_visible,
        ),
    )


def _blocked_step_result(
    step: ShellExecutionStepSpec,
    *,
    status: ShellExecutionStatus,
    error_message: str,
    stdout_visible: bool = True,
) -> ShellExecutionStepResult:
    return ShellExecutionStepResult(
        id=step.id,
        command=step.spec.command,
        status=status,
        exit_code=None,
        stdout="",
        stderr="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_ms=0,
        error_code=SHELL_STATUS_ERROR_CODES[status],
        error_message=error_message,
        metadata=_step_metadata(step, stdout_visible=stdout_visible),
    )


def _ordered_serial_results(
    spec: ShellCompositionSpec,
    results: list[ShellExecutionStepResult],
    *,
    timeout_missing: bool = False,
) -> tuple[ShellExecutionStepResult, ...]:
    by_id = {result.id: result for result in results}
    ordered: list[ShellExecutionStepResult] = []
    for step in spec.steps:
        result = by_id.get(step.id)
        if result is not None:
            ordered.append(result)
        elif timeout_missing:
            ordered.append(
                _blocked_step_result(
                    step,
                    status=ShellExecutionStatus.TIMEOUT,
                    error_message="composition timed out",
                )
            )
    return tuple(ordered)


def _ordered_parallel_results(
    spec: ShellCompositionSpec,
    results: dict[str, ShellExecutionStepResult],
    *,
    timeout_missing: bool = False,
) -> tuple[ShellExecutionStepResult, ...]:
    ordered: list[ShellExecutionStepResult] = []
    for step in spec.steps:
        result = results.get(step.id)
        if result is not None:
            ordered.append(result)
        elif timeout_missing:
            ordered.append(
                _blocked_step_result(
                    step,
                    status=ShellExecutionStatus.TIMEOUT,
                    error_message="composition timed out",
                )
            )
    return tuple(ordered)


def _composition_result(
    spec: ShellCompositionSpec,
    steps: tuple[ShellExecutionStepResult, ...],
    *,
    status: ShellExecutionStatus | None = None,
    duration_ms: int,
    timed_out: bool = False,
    error_message: str | None = None,
) -> ShellCompositionResult:
    result_status = status if status is not None else _aggregate_status(steps)
    stdout, stdout_bytes, stdout_truncated = _aggregate_stdout(spec, steps)
    stderr, stderr_bytes, stderr_truncated = _aggregate_stderr(spec, steps)
    result_error_message = error_message or _aggregate_error_message(
        result_status,
        steps,
    )
    return ShellCompositionResult(
        mode=spec.mode,
        status=result_status,
        stdout=stdout,
        stderr=stderr,
        steps=steps,
        stdout_bytes=stdout_bytes,
        stderr_bytes=stderr_bytes,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        timed_out=timed_out,
        cancelled=False,
        duration_ms=duration_ms,
        error_code=SHELL_STATUS_ERROR_CODES[result_status],
        error_message=result_error_message,
        metadata={"mode": spec.mode},
    )


def _aggregate_status(
    steps: tuple[ShellExecutionStepResult, ...],
) -> ShellExecutionStatus:
    for step in steps:
        if step.status not in _BENIGN_STATUSES:
            return step.status
    if len(steps) == 1 and steps[0].status is ShellExecutionStatus.NO_MATCHES:
        return ShellExecutionStatus.NO_MATCHES
    return ShellExecutionStatus.COMPLETED


def _aggregate_error_message(
    status: ShellExecutionStatus,
    steps: tuple[ShellExecutionStepResult, ...],
) -> str | None:
    if status is ShellExecutionStatus.COMPLETED:
        return None
    for step in steps:
        if step.status is status and step.error_message is not None:
            return step.error_message
    return status.value


def _aggregate_stdout(
    spec: ShellCompositionSpec,
    steps: tuple[ShellExecutionStepResult, ...],
) -> tuple[str, int, bool]:
    if spec.mode == "serial":
        return _cap_text(steps[-1].stdout, spec.max_stdout_bytes)
    if spec.mode == "parallel":
        return _cap_text(
            "".join(_labeled_stdout(step) for step in steps if step.stdout),
            spec.max_stdout_bytes,
        )
    return _cap_text(steps[-1].stdout, spec.max_stdout_bytes)


def _labeled_stdout(step: ShellExecutionStepResult) -> str:
    return f"[{step.id}:{step.command}]\n{step.stdout}"


def _aggregate_stderr(
    spec: ShellCompositionSpec,
    steps: tuple[ShellExecutionStepResult, ...],
) -> tuple[str, int, bool]:
    return _cap_text(
        "".join(_labeled_stderr(step) for step in steps if step.stderr),
        spec.max_stderr_bytes,
    )


def _labeled_stderr(step: ShellExecutionStepResult) -> str:
    return f"[{step.id}:{step.command}]\n{step.stderr}"


def _cap_text(value: str, byte_cap: int) -> tuple[str, int, bool]:
    value_bytes = value.encode()
    if len(value_bytes) <= byte_cap:
        return value, len(value_bytes), False
    return (
        value_bytes[:byte_cap].decode("utf-8", errors="replace"),
        (byte_cap),
        True,
    )


def _referenced_step_ids(spec: ShellCompositionSpec) -> set[str]:
    return {
        step.stdin_from.step_id
        for step in spec.steps
        if step.stdin_from is not None
    }


def _step_metadata(
    step: ShellExecutionStepSpec,
    metadata: dict[str, object] | None = None,
    *,
    stdout_visible: bool = True,
    pipeline_transport: ShellPipelineTransport | None = None,
    stdout_capture_available: bool | None = None,
) -> dict[str, object]:
    result_metadata = dict(
        step.spec.metadata if metadata is None else metadata
    )
    result_metadata.update(
        {
            "argv": step.spec.argv,
            "backend": step.spec.backend,
            "display_argv": step.spec.display_argv,
            "display_cwd": step.spec.display_cwd,
            "stdout_media_type": step.spec.stdout_media_type,
            "stdout_visible": stdout_visible,
        }
    )
    if step.stdin_from is not None:
        result_metadata["stdin_from"] = {
            "step_id": step.stdin_from.step_id,
            "stream": step.stdin_from.stream,
        }
    if pipeline_transport is not None:
        result_metadata["pipeline_transport"] = pipeline_transport
    if stdout_capture_available is not None:
        result_metadata["stdout_capture_available"] = stdout_capture_available
    return result_metadata


def _stage_stream_callback(
    stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    step: ShellExecutionStepSpec,
    stage_index: int,
    *,
    stdout_visible: bool = True,
) -> Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None:
    if stream is None:
        return None

    async def forward(event: ToolExecutionStreamEvent) -> None:
        if event.kind is ToolExecutionStreamKind.STDOUT and not stdout_visible:
            return
        metadata = dict(event.metadata)
        metadata.update(
            {
                "command": step.spec.command,
                "stage_id": step.id,
                "stage_index": stage_index,
                "step_id": step.id,
            }
        )
        await stream(
            ToolExecutionStreamEvent(
                kind=event.kind,
                content=event.content,
                progress=event.progress,
                metadata=metadata,
            )
        )

    return forward


def _stage_stream_metadata(stage: _PipelineStage) -> dict[str, object]:
    return {
        "command": stage.step.spec.command,
        "stage_id": stage.step.id,
        "stage_index": stage.index,
        "step_id": stage.step.id,
    }


async def _emit_step_completion_progress(
    stream: Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None,
    steps: tuple[ShellExecutionStepResult, ...],
) -> None:
    if stream is None:
        return
    for index, step in enumerate(steps):
        await _emit_stream_event(
            stream,
            kind=ToolExecutionStreamKind.PROGRESS,
            content=step.status.value,
            progress=1.0,
            metadata={
                "command": step.command,
                "stage_id": step.id,
                "stage_index": index,
                "status": step.status.value,
                "step_id": step.id,
            },
        )
