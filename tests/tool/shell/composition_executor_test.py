from asyncio import (
    CancelledError,
    Event,
    create_task,
    sleep,
    wait_for,
)
from asyncio.subprocess import DEVNULL, PIPE
from collections.abc import Awaitable, Callable
from pathlib import Path
from sys import executable as python_executable
from typing import Literal
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.tool.shell.composition_executor import (
    CompositionExecutor,
    LocalCompositionExecutor,
    _aggregate_error_message,
    _write_pipe_chunk,
)
from avalan.tool.shell.entities import (
    ExecutionResult,
    ExecutionSpec,
    ShellCompositionSpec,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellExecutionStepResult,
    ShellExecutionStepSpec,
    ShellOutputKind,
    ShellResourceClass,
    ShellStreamRef,
)
from avalan.tool.shell.policy import ExecutionPolicy
from avalan.tool.shell.settings import ShellToolSettings


async def _expect_cancelled(awaitable: Awaitable[object]) -> None:
    try:
        await awaitable
    except CancelledError:
        return
    raise AssertionError("Expected awaitable to be cancelled")


class LocalCompositionExecutorTest(IsolatedAsyncioTestCase):
    async def test_protocol_stub_is_inert(self) -> None:
        class InertCompositionExecutor(CompositionExecutor):
            pass

        executor = InertCompositionExecutor()

        with self.assertRaises(NotImplementedError):
            await executor.execute_composition(_composition(("read",)))

    async def test_non_spec_remains_unimplemented(self) -> None:
        executor = LocalCompositionExecutor()

        with self.assertRaises(NotImplementedError):
            await executor.execute_composition(object())  # type: ignore[arg-type]

    async def test_unknown_mode_remains_unimplemented(self) -> None:
        spec = _composition(("read",))
        object.__setattr__(spec, "mode", "batch")

        with self.assertRaises(NotImplementedError):
            await LocalCompositionExecutor().execute_composition(spec)

    async def test_pipeline_executes_cat_sed_wc_shape(self) -> None:
        spec = _composition(
            ("read", "select", "count"),
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\n"
                        "stdout.write('one\\ntwo\\nthree\\n')\n",
                    ),
                ),
                "select": _step(
                    "select",
                    "sed",
                    _python_spec(
                        "sed",
                        "from sys import stdin, stdout\n"
                        "stdout.write(''.join("
                        "line for line in stdin if 'two' in line))\n",
                    ),
                    stdin_from="read",
                ),
                "count": _step(
                    "count",
                    "wc",
                    _python_spec(
                        "wc",
                        "from sys import stdin, stdout\n"
                        "stdout.write(f'{stdin.read().count(chr(10))}\\n')\n",
                    ),
                    stdin_from="select",
                ),
            },
            max_intermediate_bytes=64,
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "1\n")
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.COMPLETED,
                ShellExecutionStatus.COMPLETED,
                ShellExecutionStatus.COMPLETED,
            ),
        )
        self.assertEqual(result.steps[0].stdout, "")
        self.assertEqual(
            result.steps[0].stdout_bytes, len(b"one\ntwo\nthree\n")
        )
        self.assertFalse(result.steps[0].metadata["stdout_visible"])
        self.assertTrue(result.steps[-1].metadata["stdout_visible"])

    async def test_pipeline_rg_no_matches_can_still_complete(self) -> None:
        spec = _composition(
            ("search", "count"),
            steps={
                "search": _step(
                    "search",
                    "rg",
                    _python_spec(
                        "rg",
                        "from sys import exit\nexit(1)\n",
                        metadata={"exit_code_statuses": {1: "no_matches"}},
                    ),
                ),
                "count": _step(
                    "count",
                    "wc",
                    _python_spec(
                        "wc",
                        "from sys import stdin, stdout\n"
                        "stdout.write(f'{stdin.read().count(chr(10))}\\n')\n",
                    ),
                    stdin_from="search",
                ),
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.steps[0].status, ShellExecutionStatus.NO_MATCHES
        )
        self.assertEqual(result.stdout, "0\n")

    async def test_pipeline_routes_json_to_jq_shape(self) -> None:
        spec = _composition(
            ("read", "filter"),
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\n"
                        'stdout.write(\'{"name": "avalan"}\\n\')\n',
                        stdout_media_type="application/json",
                        output_kind=ShellOutputKind.JSON,
                    ),
                ),
                "filter": _step(
                    "filter",
                    "jq",
                    _python_spec(
                        "jq",
                        "from json import loads\n"
                        "from sys import stdin, stdout\n"
                        "stdout.write(loads(stdin.read())['name'] + '\\n')\n",
                    ),
                    stdin_from="read",
                ),
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "avalan\n")
        self.assertEqual(result.steps[0].stdout, "")

    async def test_serial_routes_declared_stdout(self) -> None:
        spec = _composition(
            ("read", "filter"),
            mode="serial",
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\n"
                        "stdout.write('{\"count\": 3}\\n')\n",
                        stdout_media_type="application/json",
                        output_kind=ShellOutputKind.JSON,
                    ),
                ),
                "filter": _step(
                    "filter",
                    "jq",
                    _python_spec(
                        "jq",
                        "from json import loads\n"
                        "from sys import stdin, stdout\n"
                        "value = loads(stdin.read())['count']\n"
                        "stdout.write(str(value) + '\\n')\n",
                    ),
                    stdin_from="read",
                ),
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "3\n")
        self.assertEqual(result.steps[0].stdout, '{"count": 3}\n')
        self.assertEqual(
            result.steps[1].metadata["stdin_from"]["step_id"], "read"
        )

    async def test_serial_independent_steps_delegate_to_command_executor(
        self,
    ) -> None:
        command_executor = _RecordingCommandExecutor(
            {
                "first": _execution_result(
                    _python_spec("first", "print('unused')"),
                    stdout="one\n",
                ),
                "second": _execution_result(
                    _python_spec("second", "print('unused')"),
                    stdout="two\n",
                ),
            }
        )
        spec = _composition(
            ("first", "second"),
            mode="serial",
            steps={
                "first": _step(
                    "first",
                    "first",
                    _python_spec("first", "print('unused')"),
                ),
                "second": _step(
                    "second",
                    "second",
                    _python_spec("second", "print('unused')"),
                ),
            },
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(command_executor.calls, ("first", "second"))
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "two\n")

    async def test_parallel_orders_results_deterministically(self) -> None:
        spec = _composition(
            ("slow", "fast"),
            mode="parallel",
            steps={
                "slow": _step(
                    "slow",
                    "slow",
                    _python_spec("slow", "print('unused')"),
                ),
                "fast": _step(
                    "fast",
                    "fast",
                    _python_spec("fast", "print('unused')"),
                ),
            },
        )
        command_executor = _RecordingCommandExecutor(
            {
                "slow": _execution_result(
                    spec.steps[0].spec,
                    stdout="slow\n",
                ),
                "fast": _execution_result(
                    spec.steps[1].spec,
                    stdout="fast\n",
                ),
            },
            delays={"slow": 0.02, "fast": 0.0},
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(
            tuple(step.id for step in result.steps),
            ("slow", "fast"),
        )
        self.assertEqual(
            result.stdout,
            "[slow:slow]\nslow\n[fast:fast]\nfast\n",
        )

    async def test_parallel_uses_shared_process_limits(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=2)
        release = Event()
        tracker = _ProcessTracker(target_active=2)
        processes = [
            _BlockingProcess(release=release, tracker=tracker)
            for _ in range(3)
        ]
        spec = _composition(
            ("one", "two", "three"),
            mode="parallel",
            steps={
                name: _step(name, "rg", _direct_spec("rg"))
                for name in ("one", "two", "three")
            },
            timeout_seconds=2.0,
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence(processes),
        ):
            running = create_task(
                LocalCompositionExecutor(
                    settings=settings
                ).execute_composition(spec)
            )
            try:
                await wait_for(tracker.target_reached.wait(), timeout=1)
                await sleep(0)

                self.assertEqual(tracker.maximum_active, 2)
                self.assertEqual(processes[2].spawn_args, ())
            finally:
                release.set()
            result = await wait_for(running, timeout=1)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(tracker.maximum_active, 2)
        self.assertNotEqual(processes[2].spawn_args, ())

    async def test_pipeline_nonzero_middle_stage_fails_aggregate(
        self,
    ) -> None:
        spec = _composition(
            ("read", "filter", "count"),
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\nstdout.write('one\\n')\n",
                    ),
                ),
                "filter": _step(
                    "filter",
                    "sed",
                    _python_spec(
                        "sed",
                        "from sys import stdin, stdout, exit\n"
                        "stdout.write(stdin.read())\n"
                        "exit(2)\n",
                    ),
                    stdin_from="read",
                ),
                "count": _step(
                    "count",
                    "wc",
                    _python_spec(
                        "wc",
                        "from sys import stdin, stdout\n"
                        "stdout.write(f'{stdin.read().count(chr(10))}\\n')\n",
                    ),
                    stdin_from="filter",
                ),
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.NONZERO_EXIT)
        self.assertEqual(result.steps[1].exit_code, 2)
        self.assertEqual(
            result.steps[1].status, ShellExecutionStatus.NONZERO_EXIT
        )
        self.assertEqual(result.stdout, "1\n")

    async def test_pipeline_command_unavailable_fails_closed(self) -> None:
        spec = _composition(
            ("missing",),
            steps={
                "missing": _step(
                    "missing",
                    "rg",
                    _direct_spec("rg", executable=None),
                )
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(
            result.status, ShellExecutionStatus.COMMAND_UNAVAILABLE
        )
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
        )
        self.assertEqual(len(result.steps), 1)

    async def test_pipeline_later_command_unavailable_keeps_step_shape(
        self,
    ) -> None:
        spec = _composition(
            ("read", "missing", "count"),
            steps={
                "read": _step("read", "cat", _direct_spec("cat")),
                "missing": _step(
                    "missing",
                    "sed",
                    _direct_spec("sed", executable=None),
                    stdin_from="read",
                ),
                "count": _step(
                    "count",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="missing",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=AssertionError("preflight failure must not spawn"),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(
            result.status, ShellExecutionStatus.COMMAND_UNAVAILABLE
        )
        self.assertEqual(
            tuple(step.id for step in result.steps),
            ("read", "missing", "count"),
        )
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.TOOL_ERROR,
                ShellExecutionStatus.COMMAND_UNAVAILABLE,
                ShellExecutionStatus.TOOL_ERROR,
            ),
        )
        self.assertIn("missing", result.steps[0].error_message or "")

    async def test_pipeline_invalid_backend_is_policy_denied(self) -> None:
        spec = _composition(
            ("isolated",),
            steps={
                "isolated": _step(
                    "isolated",
                    "rg",
                    _direct_spec("rg", backend="sandbox"),
                )
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertIn("isolated", result.error_message or "")

    async def test_pipeline_later_invalid_backend_keeps_step_shape(
        self,
    ) -> None:
        spec = _composition(
            ("read", "count"),
            steps={
                "read": _step("read", "cat", _direct_spec("cat")),
                "count": _step(
                    "count",
                    "wc",
                    _direct_spec("wc", backend="sandbox"),
                    stdin_from="read",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=AssertionError("preflight failure must not spawn"),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            tuple(step.id for step in result.steps),
            ("read", "count"),
        )
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.TOOL_ERROR,
                ShellExecutionStatus.POLICY_DENIED,
            ),
        )
        self.assertIn("count", result.steps[0].error_message or "")

    async def test_pipeline_over_process_capacity_fails_before_spawn(
        self,
    ) -> None:
        settings = ShellToolSettings(max_concurrent_processes=2)
        spec = _composition(
            ("one", "two", "three"),
            steps={
                "one": _step("one", "cat", _direct_spec("cat")),
                "two": _step(
                    "two",
                    "sed",
                    _direct_spec("sed"),
                    stdin_from="one",
                ),
                "three": _step(
                    "three",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="two",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=AssertionError("oversized pipeline must not spawn"),
        ):
            result = await LocalCompositionExecutor(
                settings=settings,
            ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertIn("3 concurrent process slots", result.error_message or "")
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.POLICY_DENIED,
                ShellExecutionStatus.POLICY_DENIED,
                ShellExecutionStatus.POLICY_DENIED,
            ),
        )

    async def test_pipeline_over_heavy_capacity_fails_before_spawn(
        self,
    ) -> None:
        settings = ShellToolSettings(
            max_concurrent_processes=3,
            max_concurrent_heavy_processes=1,
        )
        spec = _composition(
            ("render", "ocr"),
            steps={
                "render": _step(
                    "render",
                    "pdftoppm",
                    _direct_spec("pdftoppm", resource_class="heavy"),
                ),
                "ocr": _step(
                    "ocr",
                    "tesseract",
                    _direct_spec("tesseract", resource_class="heavy"),
                    stdin_from="render",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=AssertionError(
                "oversized heavy pipeline must not spawn"
            ),
        ):
            result = await LocalCompositionExecutor(
                settings=settings,
            ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertIn("2 concurrent heavy", result.error_message or "")
        self.assertEqual(len(result.steps), 2)

    async def test_pipeline_spawn_failure_cleans_started_children(
        self,
    ) -> None:
        first = _TerminableProcess(stdout=b"", stderr=b"warn")
        spec = _composition(
            ("first", "second"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "second": _step(
                    "second",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, OSError("failed")]),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.SPAWN_FAILED)
        self.assertEqual(first.terminate_count, 1)
        self.assertEqual(
            result.steps[1].status, ShellExecutionStatus.SPAWN_FAILED
        )

    async def test_pipeline_unspawned_stage_has_zero_duration(self) -> None:
        first = _TerminableProcess(stdout=b"", stderr=b"")
        spec = _composition(
            ("first", "second", "third"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "second": _step(
                    "second",
                    "sed",
                    _direct_spec("sed"),
                    stdin_from="first",
                ),
                "third": _step(
                    "third",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="second",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, OSError("failed")]),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.SPAWN_FAILED)
        self.assertEqual(result.steps[2].duration_ms, 0)

    async def test_pipeline_first_spawn_failure_handles_empty_task_set(
        self,
    ) -> None:
        spec = _composition(
            ("first", "second"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "second": _step(
                    "second",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([OSError("failed")]),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.SPAWN_FAILED)
        self.assertEqual(result.steps[1].duration_ms, 0)

    async def test_pipeline_timeout_returns_partial_final_stdout(self) -> None:
        first = _TerminableProcess(stdout=b"input", stderr=b"")
        final = _TerminableProcess(stdout=b"partial", stderr=b"warning")
        spec = _composition(
            ("first", "final"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "final": _step(
                    "final",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
            timeout_seconds=0.01,
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, final]),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.stdout, "partial")
        self.assertIn("warning", result.stderr)
        self.assertEqual(first.terminate_count, 1)
        self.assertEqual(final.terminate_count, 1)
        self.assertTrue(final.stdin.closed)

    async def test_pipeline_cancellation_kills_every_child(self) -> None:
        first = _TerminableProcess(stdout=b"input", stderr=b"")
        final = _TerminableProcess(stdout=b"", stderr=b"")
        spec = _composition(
            ("first", "final"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "final": _step(
                    "final",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
            timeout_seconds=10.0,
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, final]),
        ):
            running = create_task(
                LocalCompositionExecutor().execute_composition(spec)
            )
            await wait_for(final.wait_started.wait(), timeout=1)
            running.cancel()
            await _expect_cancelled(running)

        self.assertEqual(first.kill_count, 1)
        self.assertEqual(final.kill_count, 1)

    async def test_pipeline_start_log_failure_cleans_children(self) -> None:
        process = _TerminableProcess(stdout=b"", stderr=b"")
        spec = _composition(
            ("first",),
            steps={"first": _step("first", "cat", _direct_spec("cat"))},
        )

        async def fail(_: ToolExecutionStreamEvent) -> None:
            raise RuntimeError("log failed")

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([process]),
        ):
            with self.assertRaisesRegex(RuntimeError, "log failed"):
                await LocalCompositionExecutor().execute_composition(
                    spec,
                    stream=fail,
                )

        self.assertEqual(process.kill_count, 1)

    async def test_pipeline_wait_failure_kills_children_and_propagates(
        self,
    ) -> None:
        process = _FailingWaitProcess(
            release=Event(),
            stdout=b"",
            stderr=b"",
        )
        spec = _composition(
            ("first",),
            steps={"first": _step("first", "cat", _direct_spec("cat"))},
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([process]),
        ):
            with self.assertRaisesRegex(OSError, "wait failed"):
                await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(process.kill_count, 1)

    async def test_pipeline_stream_callback_failure_returns_tool_error(
        self,
    ) -> None:
        first = _TerminableProcess(stdout=b"input", stderr=b"")
        final = _TerminableProcess(stdout=b"visible", stderr=b"")
        spec = _composition(
            ("first", "final"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "final": _step(
                    "final",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
        )

        async def fail_on_stdout(event: ToolExecutionStreamEvent) -> None:
            if event.kind is ToolExecutionStreamKind.STDOUT:
                raise RuntimeError("stream failed")

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, final]),
        ):
            result = await wait_for(
                LocalCompositionExecutor().execute_composition(
                    spec,
                    stream=fail_on_stdout,
                ),
                timeout=1,
            )

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(result.error_message, "stream collection failed")
        self.assertEqual(first.kill_count, 1)
        self.assertEqual(final.kill_count, 1)

    async def test_pipeline_intermediate_overflow_is_bounded(self) -> None:
        first = _TerminableProcess(stdout=b"abcdef", stderr=b"")
        final = _TerminableProcess(stdout=b"", stderr=b"")
        spec = _composition(
            ("first", "final"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "final": _step(
                    "final",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
            max_intermediate_bytes=3,
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, final]),
        ):
            result = await LocalCompositionExecutor(
                settings=ShellToolSettings(stream_read_chunk_bytes=2)
            ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(result.steps[0].stdout_bytes, 3)
        self.assertTrue(result.steps[0].stdout_truncated)
        self.assertEqual(final.stdin.data, b"abc")
        self.assertLessEqual(result.steps[0].stdout_bytes, 3)

    async def test_pipeline_missing_source_stdout_still_closes_downstream(
        self,
    ) -> None:
        first = _BlockingProcess(release=Event(), stdout=b"", stderr=b"")
        first.stdout = None
        first._release.set()
        final = _BlockingProcess(release=Event(), stdout=b"done", stderr=b"")
        final._release.set()
        spec = _composition(
            ("first", "final"),
            steps={
                "first": _step("first", "cat", _direct_spec("cat")),
                "final": _step(
                    "final",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="first",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([first, final]),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertTrue(final.stdin.closed)

    async def test_pipeline_process_without_returncode_maps_spawn_failed(
        self,
    ) -> None:
        process = _BlockingProcess(release=Event(), stdout=b"", stderr=b"")
        process.returncode = None
        process._release.set()
        spec = _composition(
            ("first",),
            steps={"first": _step("first", "cat", _direct_spec("cat"))},
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([process]),
        ):
            result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.SPAWN_FAILED)
        self.assertIsNone(result.steps[0].exit_code)

    async def test_pipeline_final_stdout_truncates_at_composition_cap(
        self,
    ) -> None:
        spec = _composition(
            ("read", "final"),
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\nstdout.write('input')\n",
                    ),
                ),
                "final": _step(
                    "final",
                    "wc",
                    _python_spec(
                        "wc",
                        "from sys import stdout\nstdout.write('abcdef')\n",
                    ),
                    stdin_from="read",
                ),
            },
            max_stdout_bytes=3,
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.stdout, "abc")
        self.assertEqual(result.stdout_bytes, 3)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.steps[-1].stdout_truncated)

    async def test_pipeline_stderr_is_stage_labeled_and_truncated(
        self,
    ) -> None:
        spec = _composition(
            ("warn",),
            steps={
                "warn": _step(
                    "warn",
                    "rg",
                    _python_spec(
                        "rg",
                        "from sys import stderr\nstderr.write('warning\\n')\n",
                        max_stderr_bytes=4,
                    ),
                )
            },
            max_stderr_bytes=20,
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.steps[0].stderr, "warn")
        self.assertTrue(result.steps[0].stderr_truncated)
        self.assertIn("[warn:rg]", result.stderr)

    async def test_pipeline_streams_final_stdout_and_stage_stderr_only(
        self,
    ) -> None:
        spec = _composition(
            ("read", "count"),
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout, stderr\n"
                        "stdout.write('INTERMEDIATE_STDOUT')\n"
                        "stderr.write('read warning\\n')\n",
                    ),
                ),
                "count": _step(
                    "count",
                    "wc",
                    _python_spec(
                        "wc",
                        "from sys import stdin, stdout, stderr\n"
                        "stdin.read()\n"
                        "stdout.write('FINAL_STDOUT')\n"
                        "stderr.write('count warning\\n')\n",
                    ),
                    stdin_from="read",
                ),
            },
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await LocalCompositionExecutor().execute_composition(
            spec,
            stream=record,
        )

        self.assertEqual(result.stdout, "FINAL_STDOUT")
        stdout_events = [
            event
            for event in events
            if event.kind is ToolExecutionStreamKind.STDOUT
        ]
        stderr_events = [
            event
            for event in events
            if event.kind is ToolExecutionStreamKind.STDERR
        ]
        progress_events = [
            event
            for event in events
            if event.kind is ToolExecutionStreamKind.PROGRESS
        ]

        self.assertEqual(
            [event.content for event in stdout_events], ["FINAL_STDOUT"]
        )
        self.assertNotIn(
            "INTERMEDIATE_STDOUT",
            "".join(event.content or "" for event in events),
        )
        self.assertEqual(
            sorted(
                [
                    (
                        event.content,
                        event.metadata["stage_id"],
                        event.metadata["stage_index"],
                    )
                    for event in stderr_events
                ],
                key=lambda item: item[2],
            ),
            [
                (
                    "read warning\n",
                    "read",
                    0,
                ),
                ("count warning\n", "count", 1),
            ],
        )
        self.assertTrue(
            any(
                event.content == "completed"
                and event.metadata.get("stage_id") == "count"
                for event in progress_events
            )
        )

    async def test_serial_stream_suppresses_non_final_stdout(self) -> None:
        spec = _composition(
            ("first", "second"),
            mode="serial",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
            },
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await LocalCompositionExecutor(
            command_executor=_StreamingCommandExecutor(),
        ).execute_composition(spec, stream=record)

        self.assertEqual(result.stdout, "second stdout\n")
        self.assertEqual(
            [
                event.content
                for event in events
                if event.kind is ToolExecutionStreamKind.STDOUT
            ],
            ["second stdout\n"],
        )
        self.assertNotIn(
            "first stdout\n",
            "".join(event.content or "" for event in events),
        )
        self.assertEqual(
            [
                (
                    event.content,
                    event.metadata["stage_id"],
                    event.metadata["stage_index"],
                )
                for event in events
                if event.kind is ToolExecutionStreamKind.STDERR
            ],
            [
                ("first stderr\n", "first", 0),
                ("second stderr\n", "second", 1),
            ],
        )

    async def test_parallel_stream_suppresses_all_stdout(self) -> None:
        spec = _composition(
            ("first", "second"),
            mode="parallel",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
            },
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        result = await LocalCompositionExecutor(
            command_executor=_StreamingCommandExecutor(),
        ).execute_composition(spec, stream=record)

        self.assertEqual(
            result.stdout,
            "[first:first]\nfirst stdout\n[second:second]\nsecond stdout\n",
        )
        self.assertEqual(
            [
                event.content
                for event in events
                if event.kind is ToolExecutionStreamKind.STDOUT
            ],
            [],
        )
        self.assertNotIn(
            "first stdout\n",
            "".join(event.content or "" for event in events),
        )
        self.assertNotIn(
            "second stdout\n",
            "".join(event.content or "" for event in events),
        )
        self.assertEqual(
            sorted(
                [
                    (
                        event.content,
                        event.metadata["stage_id"],
                        event.metadata["stage_index"],
                    )
                    for event in events
                    if event.kind is ToolExecutionStreamKind.STDERR
                ],
                key=lambda item: item[2],
            ),
            [
                ("first stderr\n", "first", 0),
                ("second stderr\n", "second", 1),
            ],
        )

    async def test_serial_routing_overflow_returns_too_large(self) -> None:
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\nstdout.write('abcdef')\n",
                    ),
                ),
                "count": _step(
                    "count",
                    "wc",
                    _python_spec(
                        "wc",
                        "from sys import stdin, stdout\n"
                        "stdout.write(stdin.read())\n",
                    ),
                    stdin_from="read",
                ),
            },
            max_intermediate_bytes=3,
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(tuple(step.id for step in result.steps), ("read",))
        self.assertEqual(result.error_message, "routed stdout exceeded cap")

    async def test_serial_routing_truncated_stdout_returns_too_large(
        self,
    ) -> None:
        read_spec = _direct_spec("cat")
        count_spec = _direct_spec("wc")
        command_executor = _RecordingCommandExecutor(
            {
                "cat": _execution_result(
                    read_spec,
                    stdout="abc",
                    stdout_truncated=True,
                ),
            }
        )
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step("count", "wc", count_spec, stdin_from="read"),
            },
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(result.steps[0].stdout, "abc")

    async def test_serial_routing_missing_source_returns_tool_error(
        self,
    ) -> None:
        read_spec = _direct_spec("cat")
        command_executor = _RecordingCommandExecutor(
            {"cat": _execution_result(read_spec, stdout="abc")}
        )
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step("count", "wc", _direct_spec("wc")),
            },
        )
        object.__setattr__(
            spec.steps[1],
            "stdin_from",
            ShellStreamRef(step_id="missing", stream="stdout"),
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.steps[-1].error_message,
            "routed stdin source is unavailable",
        )

    async def test_serial_routing_failed_source_returns_tool_error(
        self,
    ) -> None:
        read_spec = _direct_spec("cat")
        count_spec = _direct_spec("wc")
        command_executor = _RecordingCommandExecutor(
            {
                "cat": _execution_result(
                    read_spec,
                    stdout="abc",
                    status=ShellExecutionStatus.NONZERO_EXIT,
                ),
            }
        )
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step("count", "wc", count_spec, stdin_from="read"),
            },
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.NONZERO_EXIT)
        self.assertEqual(
            result.steps[-1].error_message,
            "routed stdin source failed",
        )

    async def test_serial_routed_spawn_failure_returns_spawn_failed(
        self,
    ) -> None:
        read_spec = _direct_spec("cat")
        command_executor = _RecordingCommandExecutor(
            {"cat": _execution_result(read_spec, stdout="abc")}
        )
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step(
                    "count",
                    "wc",
                    _direct_spec("wc"),
                    stdin_from="read",
                ),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            side_effect=OSError("failed"),
        ):
            result = await LocalCompositionExecutor(
                command_executor=command_executor,
            ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.SPAWN_FAILED)
        self.assertEqual(
            result.steps[-1].status,
            ShellExecutionStatus.SPAWN_FAILED,
        )

    async def test_serial_routed_blocked_step_returns_policy_denied(
        self,
    ) -> None:
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step(
                    "read",
                    "cat",
                    _python_spec(
                        "cat",
                        "from sys import stdout\nstdout.write('abc')\n",
                    ),
                ),
                "count": _step(
                    "count",
                    "wc",
                    _direct_spec("wc", backend="sandbox"),
                    stdin_from="read",
                ),
            },
        )

        result = await LocalCompositionExecutor().execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            result.steps[-1].status,
            ShellExecutionStatus.POLICY_DENIED,
        )

    async def test_serial_routed_stream_failure_returns_tool_error(
        self,
    ) -> None:
        read_spec = _direct_spec("cat")
        count_spec = _direct_spec("wc")
        command_executor = _RecordingCommandExecutor(
            {"cat": _execution_result(read_spec, stdout="abc")}
        )
        process = _BlockingProcess(
            release=Event(),
            stdout=b"visible",
            stderr=b"",
        )
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step("count", "wc", count_spec, stdin_from="read"),
            },
        )

        async def fail_on_stdout(event: ToolExecutionStreamEvent) -> None:
            if event.kind is ToolExecutionStreamKind.STDOUT:
                raise RuntimeError("stream failed")

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([process]),
        ):
            result = await LocalCompositionExecutor(
                command_executor=command_executor,
            ).execute_composition(spec, stream=fail_on_stdout)

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.steps[-1].error_message,
            "stream collection failed",
        )
        self.assertEqual(process.kill_count, 1)

    async def test_serial_routed_cancellation_kills_child(self) -> None:
        read_spec = _direct_spec("cat")
        count_spec = _direct_spec("wc")
        command_executor = _RecordingCommandExecutor(
            {"cat": _execution_result(read_spec, stdout="abc")}
        )
        process = _TerminableProcess(stdout=b"", stderr=b"")
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step("count", "wc", count_spec, stdin_from="read"),
            },
            timeout_seconds=10.0,
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([process]),
        ):
            running = create_task(
                LocalCompositionExecutor(
                    command_executor=command_executor,
                ).execute_composition(spec)
            )
            await wait_for(process.wait_started.wait(), timeout=1)
            running.cancel()
            await _expect_cancelled(running)

        self.assertEqual(process.kill_count, 1)

    async def test_serial_routed_wait_failure_kills_child_and_propagates(
        self,
    ) -> None:
        read_spec = _direct_spec("cat")
        count_spec = _direct_spec("wc")
        command_executor = _RecordingCommandExecutor(
            {"cat": _execution_result(read_spec, stdout="abc")}
        )
        process = _FailingWaitProcess(
            release=Event(),
            stdout=b"",
            stderr=b"",
        )
        spec = _composition(
            ("read", "count"),
            mode="serial",
            steps={
                "read": _step("read", "cat", read_spec),
                "count": _step("count", "wc", count_spec, stdin_from="read"),
            },
        )

        with patch(
            "avalan.tool.shell.process.create_subprocess_exec",
            new=_fake_process_sequence([process]),
        ):
            with self.assertRaisesRegex(OSError, "wait failed"):
                await LocalCompositionExecutor(
                    command_executor=command_executor,
                ).execute_composition(spec)

        self.assertEqual(process.kill_count, 1)

    async def test_serial_timeout_returns_timeout_steps(self) -> None:
        command_executor = _BlockingCommandExecutor()
        spec = _composition(
            ("first", "second"),
            mode="serial",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
            },
            timeout_seconds=0.01,
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(
            tuple(step.status for step in result.steps),
            (
                ShellExecutionStatus.TIMEOUT,
                ShellExecutionStatus.TIMEOUT,
            ),
        )
        self.assertTrue(command_executor.cancelled.is_set())

    async def test_serial_timeout_after_partial_progress_keeps_step_shape(
        self,
    ) -> None:
        command_executor = _BlockingCommandExecutor(
            completed={"first": "done\n"},
        )
        spec = _composition(
            ("first", "second", "third"),
            mode="serial",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
                "third": _step("third", "third", _direct_spec("third")),
            },
            timeout_seconds=0.01,
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(
            tuple(step.id for step in result.steps),
            ("first", "second", "third"),
        )
        self.assertEqual(
            result.steps[0].status, ShellExecutionStatus.COMPLETED
        )
        self.assertEqual(result.steps[0].stdout, "done\n")
        self.assertEqual(
            tuple(step.status for step in result.steps[1:]),
            (
                ShellExecutionStatus.TIMEOUT,
                ShellExecutionStatus.TIMEOUT,
            ),
        )
        self.assertTrue(command_executor.cancelled.is_set())

    async def test_serial_cancellation_propagates(self) -> None:
        command_executor = _BlockingCommandExecutor()
        spec = _composition(
            ("first",),
            mode="serial",
            steps={"first": _step("first", "first", _direct_spec("first"))},
            timeout_seconds=10.0,
        )
        running = create_task(
            LocalCompositionExecutor(
                command_executor=command_executor,
            ).execute_composition(spec)
        )
        await wait_for(command_executor.started.wait(), timeout=1)
        running.cancel()

        await _expect_cancelled(running)

        self.assertTrue(command_executor.cancelled.is_set())

    async def test_parallel_timeout_orders_missing_steps(self) -> None:
        command_executor = _BlockingCommandExecutor(
            completed={"first": "done\n"},
        )
        spec = _composition(
            ("first", "second"),
            mode="parallel",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
            },
            timeout_seconds=0.01,
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(
            tuple(step.id for step in result.steps),
            ("first", "second"),
        )
        self.assertEqual(result.steps[1].status, ShellExecutionStatus.TIMEOUT)

    async def test_parallel_task_failure_cancels_pending_and_propagates(
        self,
    ) -> None:
        command_executor = _FailingCommandExecutor()
        spec = _composition(
            ("fail", "pending"),
            mode="parallel",
            steps={
                "fail": _step("fail", "fail", _direct_spec("fail")),
                "pending": _step(
                    "pending",
                    "pending",
                    _direct_spec("pending"),
                ),
            },
            timeout_seconds=0.01,
        )

        with self.assertRaisesRegex(RuntimeError, "command failed"):
            await LocalCompositionExecutor(
                command_executor=command_executor,
            ).execute_composition(spec)

        self.assertTrue(command_executor.pending_cancelled.is_set())

    async def test_parallel_cancellation_propagates(self) -> None:
        command_executor = _BlockingCommandExecutor()
        spec = _composition(
            ("first", "second"),
            mode="parallel",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
            },
            timeout_seconds=10.0,
        )
        running = create_task(
            LocalCompositionExecutor(
                command_executor=command_executor,
            ).execute_composition(spec)
        )
        await wait_for(command_executor.started.wait(), timeout=1)
        running.cancel()

        await _expect_cancelled(running)

        self.assertTrue(command_executor.cancelled.is_set())

    async def test_single_no_matches_status_is_preserved(self) -> None:
        spec = _composition(
            ("search",),
            mode="serial",
            steps={"search": _step("search", "rg", _direct_spec("rg"))},
        )
        command_executor = _RecordingCommandExecutor(
            {
                "rg": _execution_result(
                    spec.steps[0].spec,
                    status=ShellExecutionStatus.NO_MATCHES,
                )
            }
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.status, ShellExecutionStatus.NO_MATCHES)
        self.assertEqual(result.error_message, "no_matches")

    async def test_parallel_stdout_is_capped(self) -> None:
        spec = _composition(
            ("first", "second"),
            mode="parallel",
            steps={
                "first": _step("first", "first", _direct_spec("first")),
                "second": _step("second", "second", _direct_spec("second")),
            },
            max_stdout_bytes=10,
        )
        command_executor = _RecordingCommandExecutor(
            {
                "first": _execution_result(spec.steps[0].spec, stdout="abc\n"),
                "second": _execution_result(
                    spec.steps[1].spec,
                    stdout="def\n",
                ),
            }
        )

        result = await LocalCompositionExecutor(
            command_executor=command_executor,
        ).execute_composition(spec)

        self.assertEqual(result.stdout_bytes, 10)
        self.assertTrue(result.stdout_truncated)

    async def test_pipe_chunk_edge_cases_return_false(self) -> None:
        self.assertFalse(await _write_pipe_chunk(None, b"abc"))
        self.assertFalse(await _write_pipe_chunk(object(), b"abc"))
        self.assertFalse(await _write_pipe_chunk(_BrokenPipeStdin(), b"abc"))
        self.assertFalse(await _write_pipe_chunk(_ResetPipeStdin(), b"abc"))

    def test_aggregate_error_message_falls_back_to_status(self) -> None:
        self.assertEqual(
            _aggregate_error_message(
                ShellExecutionStatus.TIMEOUT,
                (
                    _blocked_step_result_for_test(
                        "step",
                        ShellExecutionStatus.COMPLETED,
                    ),
                ),
            ),
            "timeout",
        )


class _RecordingCommandExecutor:
    def __init__(
        self,
        results: dict[str, ExecutionResult],
        *,
        delays: dict[str, float] | None = None,
    ) -> None:
        self._results = results
        self._delays = {} if delays is None else delays
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
        delay = self._delays.get(spec.command, 0.0)
        if delay:
            await sleep(delay)
        return self._results[spec.command]


class _StreamingCommandExecutor:
    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        stdout = f"{spec.command} stdout\n"
        stderr = f"{spec.command} stderr\n"
        if stream is not None:
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDOUT,
                    content=stdout,
                )
            )
            await stream(
                ToolExecutionStreamEvent(
                    kind=ToolExecutionStreamKind.STDERR,
                    content=stderr,
                )
            )
        return _execution_result(spec, stdout=stdout, stderr=stderr)


class _BlockingCommandExecutor:
    def __init__(
        self,
        *,
        completed: dict[str, str] | None = None,
    ) -> None:
        self.started = Event()
        self.cancelled = Event()
        self._completed = {} if completed is None else completed

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.started.set()
        if spec.command in self._completed:
            return _execution_result(
                spec,
                stdout=self._completed[spec.command],
            )
        try:
            await sleep(10)
        except CancelledError:
            self.cancelled.set()
            raise
        return _execution_result(spec)


class _FailingCommandExecutor:
    def __init__(self) -> None:
        self.pending_cancelled = Event()

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        if spec.command == "fail":
            raise RuntimeError("command failed")
        try:
            await sleep(10)
        except CancelledError:
            self.pending_cancelled.set()
            raise
        return _execution_result(spec)


class _FakeStream:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data
        self._offset = 0
        self.read_calls = 0
        self.read_sizes: list[int] = []
        self.drained_bytes = 0

    async def read(self, size: int) -> bytes:
        self.read_calls += 1
        self.read_sizes.append(size)
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        self.drained_bytes += len(chunk)
        return chunk


class _FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        pass


class _BrokenPipeStdin(_FakeStdin):
    def write(self, data: bytes) -> None:
        raise BrokenPipeError


class _ResetPipeStdin(_FakeStdin):
    async def drain(self) -> None:
        raise ConnectionResetError


class _ProcessTracker:
    def __init__(self, *, target_active: int = 1) -> None:
        self.active = 0
        self.maximum_active = 0
        self.target_reached = Event()
        self._target_active = target_active

    def enter(self) -> None:
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if self.active >= self._target_active:
            self.target_reached.set()

    def exit(self) -> None:
        self.active -= 1


class _BlockingProcess:
    returncode = 0

    def __init__(
        self,
        *,
        release: Event,
        tracker: _ProcessTracker | None = None,
        stdout: bytes = b"ok",
        stderr: bytes = b"",
    ) -> None:
        self._release = release
        self._tracker = tracker
        self.stdout: object = _FakeStream(stdout)
        self.stderr: object = _FakeStream(stderr)
        self.stdin: _FakeStdin = _FakeStdin()
        self.kill_count = 0
        self.terminate_count = 0
        self.wait_started = Event()
        self.spawn_args: tuple[object, ...] = ()
        self.spawn_kwargs: dict[str, object] = {}

    async def wait(self) -> None:
        self.wait_started.set()
        if self._tracker is not None:
            self._tracker.enter()
        try:
            await self._release.wait()
        finally:
            if self._tracker is not None:
                self._tracker.exit()

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15
        self._release.set()

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9
        self._release.set()


class _TerminableProcess(_BlockingProcess):
    def __init__(
        self,
        *,
        stdout: bytes,
        stderr: bytes,
        terminate_releases: bool = True,
    ) -> None:
        super().__init__(release=Event(), stdout=stdout, stderr=stderr)
        self.returncode = None
        self._terminate_releases = terminate_releases

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15
        if self._terminate_releases:
            self._release.set()


class _FailingWaitProcess(_BlockingProcess):
    async def wait(self) -> None:
        self.wait_started.set()
        raise OSError("wait failed")


def _composition(
    step_ids: tuple[str, ...],
    *,
    mode: Literal["pipeline", "serial", "parallel"] = "pipeline",
    steps: dict[str, ShellExecutionStepSpec] | None = None,
    timeout_seconds: float = 2.0,
    max_stdout_bytes: int = 1024,
    max_stderr_bytes: int = 1024,
    max_intermediate_bytes: int = 1024,
) -> ShellCompositionSpec:
    step_map = (
        {
            step_id: _step(step_id, "rg", _direct_spec("rg"))
            for step_id in step_ids
        }
        if steps is None
        else steps
    )
    return ShellCompositionSpec(
        mode=mode,
        steps=tuple(step_map[step_id] for step_id in step_ids),
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        max_intermediate_bytes=max_intermediate_bytes,
    )


def _step(
    step_id: str,
    command: str,
    spec: ExecutionSpec,
    *,
    stdin_from: str | None = None,
) -> ShellExecutionStepSpec:
    return ShellExecutionStepSpec(
        id=step_id,
        spec=spec,
        stdin_from=(
            None
            if stdin_from is None
            else ShellStreamRef(step_id=stdin_from, stream="stdout")
        ),
    )


def _python_spec(
    command: str,
    script: str,
    *,
    metadata: dict[str, object] | None = None,
    stdout_media_type: str = "text/plain",
    output_kind: ShellOutputKind = ShellOutputKind.TEXT,
    max_stdout_bytes: int = 1024,
    max_stderr_bytes: int = 1024,
) -> ExecutionSpec:
    return _direct_spec(
        command,
        executable=python_executable,
        argv=(python_executable, "-u", "-c", script),
        metadata=metadata,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _direct_spec(
    command: str,
    *,
    backend: Literal["local", "sandbox", "container"] = "local",
    executable: str | None = "/trusted/bin/tool",
    argv: tuple[str, ...] | None = None,
    resource_class: ShellResourceClass = "standard",
    metadata: dict[str, object] | None = None,
    stdout_media_type: str = "text/plain",
    output_kind: ShellOutputKind = ShellOutputKind.TEXT,
    max_stdout_bytes: int = 1024,
    max_stderr_bytes: int = 1024,
) -> ExecutionSpec:
    spec_argv = (command,) if argv is None else argv
    return ExecutionPolicy().create_execution_spec(
        backend=backend,
        tool_name=f"shell.{command}",
        command=command,
        executable=executable,
        argv=spec_argv,
        display_argv=spec_argv,
        cwd=str(Path.cwd().resolve()),
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        resource_class=resource_class,
        output_plan=None,
        timeout_seconds=1.0,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
        metadata=metadata,
    )


def _execution_result(
    spec: ExecutionSpec,
    *,
    stdout: str = "",
    stderr: str = "",
    status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
    stdout_truncated: bool = False,
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
        exit_code=0 if status is ShellExecutionStatus.COMPLETED else 1,
        stdout=stdout,
        stderr=stderr,
        stdout_media_type=spec.stdout_media_type,
        output_kind=spec.output_kind,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=len(stderr.encode()),
        stdout_truncated=stdout_truncated,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=1,
        error_code=ShellExecutionErrorCode(status.value),
        error_message=(
            None if status is ShellExecutionStatus.COMPLETED else status.value
        ),
        metadata=spec.metadata,
    )


def _blocked_step_result_for_test(
    step_id: str,
    status: ShellExecutionStatus,
) -> ShellExecutionStepResult:
    return ShellExecutionStepResult(
        id=step_id,
        command="command",
        status=status,
        exit_code=0,
        stdout="",
        stderr="",
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        duration_ms=0,
        error_code=ShellExecutionErrorCode(status.value),
    )


def _fake_process_sequence(
    items: list[_BlockingProcess | Exception],
) -> object:
    calls = {"index": 0}

    async def fake_create_subprocess_exec(
        *args: object,
        **kwargs: object,
    ) -> _BlockingProcess:
        index = calls["index"]
        calls["index"] += 1
        item = items[index]
        if isinstance(item, Exception):
            raise item
        item.spawn_args = args
        item.spawn_kwargs = kwargs
        if kwargs.get("stdin") == DEVNULL:
            item.stdin = _FakeStdin()
            item.stdin.closed = True
        else:
            self_stdin = item.stdin
            self_stdin.closed = kwargs.get("stdin") is not PIPE
        return item

    return fake_create_subprocess_exec


if __name__ == "__main__":
    main()
