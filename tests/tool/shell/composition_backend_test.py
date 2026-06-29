from asyncio import CancelledError, Event
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.tool.shell import (
    BackendBoundaryCompositionExecutor,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    ShellCompositionSpec,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellExecutionStepSpec,
    ShellOutputKind,
    ShellStreamRef,
)


class BackendBoundaryCompositionExecutorTest(IsolatedAsyncioTestCase):
    async def test_non_spec_remains_unimplemented(self) -> None:
        executor = BackendBoundaryCompositionExecutor(
            backend="sandbox",
            command_executor=_UnexpectedCommandExecutor(),
        )

        with self.assertRaises(NotImplementedError):
            await executor.execute_composition(object())  # type: ignore[arg-type]

    async def test_unknown_mode_remains_unimplemented(self) -> None:
        spec = _composition(("one",), mode="serial", backend="sandbox")
        object.__setattr__(spec, "mode", "batch")
        executor = BackendBoundaryCompositionExecutor(
            backend="sandbox",
            command_executor=_UnexpectedCommandExecutor(),
        )

        with self.assertRaises(NotImplementedError):
            await executor.execute_composition(spec)

    async def test_sandbox_pipeline_fails_closed_before_execution(
        self,
    ) -> None:
        spec = _composition(("read", "count"), backend="sandbox")
        executor = BackendBoundaryCompositionExecutor(
            backend="sandbox",
            command_executor=_UnexpectedCommandExecutor(),
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=AssertionError("isolated pipeline must not spawn"),
        ):
            result = await executor.execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.POLICY_DENIED,
        )
        self.assertIn("sandbox byte pipelines", result.error_message or "")
        self.assertIn("trusted structured runner", result.error_message or "")
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.POLICY_DENIED,
                ShellExecutionStatus.POLICY_DENIED,
            ),
        )

    async def test_container_pipeline_fails_closed_before_execution(
        self,
    ) -> None:
        spec = _composition(("read", "count"), backend="container")
        executor = BackendBoundaryCompositionExecutor(
            backend="container",
            command_executor=_UnexpectedCommandExecutor(),
        )

        result = await executor.execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertIn("container byte pipelines", result.error_message or "")
        self.assertIn("shell evaluation", result.error_message or "")

    async def test_isolated_serial_routing_fails_before_execution(
        self,
    ) -> None:
        spec = _composition(
            ("read", "count"),
            mode="serial",
            backend="sandbox",
            routed=True,
        )
        executor = BackendBoundaryCompositionExecutor(
            backend="sandbox",
            command_executor=_UnexpectedCommandExecutor(),
        )

        result = await executor.execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.POLICY_DENIED,
                ShellExecutionStatus.POLICY_DENIED,
            ),
        )
        self.assertIn("trusted structured runner", result.error_message or "")

    async def test_backend_mismatch_fails_before_execution(self) -> None:
        spec = _composition(("local",), mode="serial", backend="local")
        executor = BackendBoundaryCompositionExecutor(
            backend="sandbox",
            command_executor=_UnexpectedCommandExecutor(),
        )

        result = await executor.execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertIn("cannot run local", result.error_message or "")

    async def test_sandbox_serial_delegates_independent_specs_and_streams(
        self,
    ) -> None:
        spec = _composition(
            ("first", "second"),
            mode="serial",
            backend="sandbox",
        )
        command_executor = _RecordingCommandExecutor(
            {
                "first": _execution_result(
                    spec.steps[0].spec,
                    stdout="first stdout\n",
                ),
                "second": _execution_result(
                    spec.steps[1].spec,
                    stdout="second stdout\n",
                ),
            },
            emit_streams=True,
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=AssertionError("serial delegation must not spawn"),
        ):
            result = await BackendBoundaryCompositionExecutor(
                backend="sandbox",
                command_executor=command_executor,
            ).execute_composition(spec, stream=record)

        self.assertEqual(command_executor.calls, ("first", "second"))
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "second stdout\n")
        self.assertEqual(
            [
                event.content
                for event in events
                if event.kind is ToolExecutionStreamKind.STDOUT
            ],
            ["second stdout\n"],
        )
        self.assertEqual(
            [
                event.content
                for event in events
                if event.kind is ToolExecutionStreamKind.STDERR
            ],
            ["first stderr\n", "second stderr\n"],
        )

    async def test_container_parallel_delegates_and_aggregates_negative(
        self,
    ) -> None:
        spec = _composition(
            ("first", "second"),
            mode="parallel",
            backend="container",
        )
        command_executor = _RecordingCommandExecutor(
            {
                "first": _execution_result(
                    spec.steps[0].spec,
                    stdout="first\n",
                ),
                "second": _execution_result(
                    spec.steps[1].spec,
                    status=ShellExecutionStatus.NONZERO_EXIT,
                    stdout="second\n",
                    error_message="command exited with status 2",
                ),
            }
        )

        result = await BackendBoundaryCompositionExecutor(
            backend="container",
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(command_executor.calls, ("first", "second"))
        self.assertEqual(result.status, ShellExecutionStatus.NONZERO_EXIT)
        self.assertEqual(
            result.stdout,
            "[first:first]\nfirst\n[second:second]\nsecond\n",
        )

    async def test_sandbox_serial_timeout_cancels_delegate(self) -> None:
        spec = _composition(
            ("first", "second"),
            mode="serial",
            backend="sandbox",
            timeout_seconds=0.01,
        )
        command_executor = _BlockingCommandExecutor()

        result = await BackendBoundaryCompositionExecutor(
            backend="sandbox",
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertTrue(command_executor.cancelled.is_set())


class _UnexpectedCommandExecutor:
    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        raise AssertionError("command executor must not be called")


class _RecordingCommandExecutor:
    def __init__(
        self,
        results: dict[str, ExecutionResult],
        *,
        emit_streams: bool = False,
    ) -> None:
        self._results = results
        self._emit_streams = emit_streams
        self._calls: list[str] = []

    @property
    def calls(self) -> tuple[str, ...]:
        return tuple(self._calls)

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self._calls.append(spec.command)
        result = self._results[spec.command]
        if self._emit_streams and stream is not None:
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content=result.stdout,
                )
            )
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDERR,
                    content=f"{spec.command} stderr\n",
                )
            )
        return result


class _BlockingCommandExecutor:
    def __init__(self) -> None:
        self.started = Event()
        self.cancelled = Event()

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.started.set()
        try:
            await Event().wait()
            raise AssertionError("blocking command should not complete")
        except CancelledError:
            self.cancelled.set()
            raise


def _composition(
    step_ids: tuple[str, ...],
    *,
    mode: Literal["pipeline", "serial", "parallel"] = "pipeline",
    backend: Literal["local", "sandbox", "container"],
    routed: bool = False,
    timeout_seconds: float = 2.0,
) -> ShellCompositionSpec:
    return ShellCompositionSpec(
        mode=mode,
        steps=tuple(
            _step(
                step_id,
                backend=backend,
                stdin_from=(
                    step_ids[index - 1] if routed and index > 0 else None
                ),
            )
            for index, step_id in enumerate(step_ids)
        ),
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
        max_intermediate_bytes=1024,
    )


def _step(
    step_id: str,
    *,
    backend: Literal["local", "sandbox", "container"],
    stdin_from: str | None = None,
) -> ShellExecutionStepSpec:
    return ShellExecutionStepSpec(
        id=step_id,
        spec=_direct_spec(step_id, backend=backend),
        stdin_from=(
            None
            if stdin_from is None
            else ShellStreamRef(step_id=stdin_from, stream="stdout")
        ),
    )


def _direct_spec(
    command: str,
    *,
    backend: Literal["local", "sandbox", "container"],
) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend=backend,
        tool_name=f"shell.{command}",
        command=command,
        executable="/trusted/bin/tool",
        argv=(command,),
        display_argv=(command,),
        cwd=str(Path.cwd().resolve()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=2.0,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )


def _execution_result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
    stdout: str = "",
    stderr: str = "",
    error_message: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        backend=spec.backend,
        tool_name=spec.tool_name,
        command=spec.command,
        argv=spec.argv,
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=0 if status is ShellExecutionStatus.COMPLETED else 2,
        stdout=stdout,
        stderr=stderr,
        stdout_media_type=spec.stdout_media_type,
        output_kind=spec.output_kind,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=1,
        error_code=ShellExecutionErrorCode(status.value),
        error_message=error_message,
        metadata=spec.metadata,
    )


if __name__ == "__main__":
    main()
