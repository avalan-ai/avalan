from asyncio import CancelledError, Event, create_task, gather, sleep, wait_for
from asyncio.subprocess import DEVNULL, PIPE
from base64 import b64encode
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from hashlib import sha256
from pathlib import Path
from stat import S_IMODE
from sys import executable as python_executable
from tempfile import TemporaryDirectory
from typing import Any, Literal
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.entities import (
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from avalan.tool.shell.entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    ExecutionResult,
    ExecutionSpec,
    GeneratedOutputPlan,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellResourceClass,
)
from avalan.tool.shell.executor import (
    CommandExecutor,
    LocalCommandExecutor,
    _cleanup_output_directory,
    _collect_generated_files,
    _collect_stream,
    _generated_output_path_replacements,
    _GeneratedOutputError,
    _matches_generated_output_prefix,
    _reader_results,
    _reader_task_failed,
    _signal_process_group,
    _spawn_argv,
    _wait_for_process_exit,
)
from avalan.tool.shell.filesystem import resolve_policy_path
from avalan.tool.shell.policy import ExecutionPolicy
from avalan.tool.shell.settings import ShellToolSettings


class LocalCommandExecutorTest(IsolatedAsyncioTestCase):
    async def test_protocol_stub_is_inert(self) -> None:
        class InertCommandExecutor(CommandExecutor):
            pass

        executor = InertCommandExecutor()
        spec = await ExecutionPolicy().normalize(_request())

        with self.assertRaises(NotImplementedError):
            await executor.execute(spec)

    async def test_missing_executable_returns_command_unavailable(
        self,
    ) -> None:
        spec = await ExecutionPolicy().normalize(
            _request(metadata={"source": "test"})
        )
        executor = LocalCommandExecutor()

        result = await executor.execute(spec)

        self.assertIsInstance(result, ExecutionResult)
        self.assertEqual(result.backend, spec.backend)
        self.assertEqual(result.tool_name, "shell.rg")
        self.assertEqual(result.command, "rg")
        self.assertEqual(result.argv, spec.argv)
        self.assertEqual(result.display_argv, spec.display_argv)
        self.assertEqual(Path(result.cwd), Path.cwd().resolve())
        self.assertEqual(result.display_cwd, ".")
        self.assertEqual(
            result.status,
            ShellExecutionStatus.COMMAND_UNAVAILABLE,
        )
        self.assertIsNone(result.exit_code)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout_media_type, "text/plain")
        self.assertEqual(result.output_kind, spec.output_kind)
        self.assertEqual(result.stdout_bytes, 0)
        self.assertEqual(result.stderr_bytes, 0)
        self.assertFalse(result.stdout_truncated)
        self.assertFalse(result.stderr_truncated)
        self.assertFalse(result.timed_out)
        self.assertFalse(result.cancelled)
        self.assertEqual(result.duration_ms, 0)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
        )
        self.assertEqual(result.error_message, "command is unavailable")
        self.assertEqual(
            result.metadata,
            {
                "source": "test",
                "local_host_approval": "required",
                "exit_code_statuses": {1: "no_matches"},
            },
        )

    async def test_non_spec_remains_unimplemented(
        self,
    ) -> None:
        executor = LocalCommandExecutor()

        with self.assertRaises(NotImplementedError):
            await executor.execute(object())  # type: ignore[arg-type]

    async def test_local_executor_refuses_isolated_specs(self) -> None:
        executor = LocalCommandExecutor()

        for backend in ("sandbox", "container"):
            with self.subTest(backend=backend):
                result = await executor.execute(
                    _direct_spec(
                        backend=backend,
                        executable=python_executable,
                        argv=(python_executable, "-c", "print('unsafe')"),
                    )
                )

                self.assertEqual(result.backend, backend)
                self.assertEqual(
                    result.status, ShellExecutionStatus.POLICY_DENIED
                )
                self.assertEqual(
                    result.error_code,
                    ShellExecutionErrorCode.POLICY_DENIED,
                )
                self.assertIn("isolated", result.error_message or "")

    def test_executor_source_uses_only_exec_subprocess_api(self) -> None:
        source = Path("src/avalan/tool/shell/executor.py").read_text()

        self.assertNotIn("create_subprocess_shell", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("subprocess.run", source)
        self.assertNotIn(".communicate(", source)

    async def test_real_subprocess_executes_with_devnull_stdin_and_env(
        self,
    ) -> None:
        script = (
            "from os import environ, getcwd\n"
            "from sys import stderr, stdin, stdout\n"
            "stdout.write(f'cwd={getcwd()}\\n')\n"
            'stdout.write(f\'env={environ.get("AVALAN_E2E_VALUE", "")}\\n\')\n'
            "stdout.write(f'stdin={stdin.read()}\\n')\n"
            "stderr.write('warn\\n')\n"
        )
        with TemporaryDirectory() as temporary_directory:
            spec = _direct_spec(
                executable=python_executable,
                argv=(python_executable, "-c", script),
                display_argv=("python", "-c", "<inline>"),
                cwd=temporary_directory,
                env={"AVALAN_E2E_VALUE": "visible", "LC_ALL": "C"},
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
            )

            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.exit_code, 0)
        self.assertIn(
            f"cwd={Path(temporary_directory).resolve()}",
            result.stdout,
        )
        self.assertIn("env=visible", result.stdout)
        self.assertIn("stdin=\n", result.stdout)
        self.assertEqual(result.stderr, "warn\n")
        self.assertEqual(result.stdout_bytes, len(result.stdout.encode()))
        self.assertEqual(result.stderr_bytes, len(result.stderr.encode()))

    async def test_execute_emits_stdout_and_stderr_stream_events(
        self,
    ) -> None:
        process = _FakeProcess(stdout=b"abcdef", stderr=b"warn")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stdout_bytes=3,
            max_stderr_bytes=10,
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(
                spec,
                stream=record,
            )

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "abc")
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(result.stderr, "warn")
        self.assertFalse(result.stderr_truncated)
        self.assertEqual(
            [
                (event.kind, event.content, event.progress)
                for event in events[:2]
            ],
            [
                (ToolExecutionStreamKind.LOG, "process started", None),
                (ToolExecutionStreamKind.PROGRESS, "started", 0.0),
            ],
        )
        self.assertEqual(
            (events[-1].kind, events[-1].content, events[-1].progress),
            (ToolExecutionStreamKind.PROGRESS, "completed", 1.0),
        )
        output_events = [
            event
            for event in events
            if event.kind
            in {
                ToolExecutionStreamKind.STDOUT,
                ToolExecutionStreamKind.STDERR,
            }
        ]
        self.assertCountEqual(
            [(event.kind, event.content) for event in output_events],
            [
                (ToolExecutionStreamKind.STDOUT, "abc"),
                (ToolExecutionStreamKind.STDERR, "warn"),
            ],
        )
        metadata_by_kind = {
            event.kind: event.metadata for event in output_events
        }
        self.assertEqual(
            metadata_by_kind[ToolExecutionStreamKind.STDOUT],
            {"bytes": 3, "truncated": True},
        )
        self.assertEqual(
            metadata_by_kind[ToolExecutionStreamKind.STDERR],
            {"bytes": 4, "truncated": False},
        )

    async def test_stream_callback_failure_kills_process_and_propagates(
        self,
    ) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"warning")
        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        async def fail(_: ToolExecutionStreamEvent) -> None:
            raise RuntimeError("stream failed")

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            with self.assertRaisesRegex(RuntimeError, "stream failed"):
                await LocalCommandExecutor().execute(spec, stream=fail)

        self.assertEqual(process.kill_count, 1)
        self.assertEqual(process.terminate_count, 0)

    async def test_output_stream_failure_kills_running_process(self) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"")
        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)
            if event.kind is ToolExecutionStreamKind.STDOUT:
                raise RuntimeError("stream failed")

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await wait_for(
                LocalCommandExecutor().execute(spec, stream=record),
                timeout=1,
            )

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(result.error_message, "stream collection failed")
        self.assertEqual(result.stdout, "partial")
        self.assertEqual(process.kill_count, 1)
        self.assertEqual(process.terminate_count, 0)
        self.assertEqual(
            (events[-1].kind, events[-1].content, events[-1].progress),
            (ToolExecutionStreamKind.PROGRESS, "tool_error", 1.0),
        )

    async def test_process_wait_failure_kills_process_and_propagates(
        self,
    ) -> None:
        process = _FailingWaitExecutionProcess(stdout=b"", stderr=b"")
        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            with self.assertRaisesRegex(OSError, "wait failed"):
                await LocalCommandExecutor().execute(spec)

        self.assertEqual(process.kill_count, 1)
        self.assertEqual(process.terminate_count, 0)

    async def test_collect_stream_requires_matching_callback_and_kind(
        self,
    ) -> None:
        async def record(_: ToolExecutionStreamEvent) -> None:
            pass

        with self.assertRaises(AssertionError):
            await _collect_stream(
                _FakeStream(b"abc"),
                3,
                3,
                stream_event=record,
            )

    async def test_real_subprocess_timeout_returns_partial_output(
        self,
    ) -> None:
        script = (
            "from sys import stdout\n"
            "from time import sleep\n"
            "stdout.write('started\\n')\n"
            "stdout.flush()\n"
            "sleep(10)\n"
        )
        spec = _direct_spec(
            executable=python_executable,
            argv=(python_executable, "-u", "-c", script),
            display_argv=("python", "-c", "<inline>"),
            timeout_seconds=0.5,
            max_stdout_bytes=1024,
            max_stderr_bytes=1024,
        )

        result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.error_code, ShellExecutionErrorCode.TIMEOUT)
        self.assertTrue(result.timed_out)
        self.assertEqual(result.stdout, "started\n")
        self.assertEqual(result.stdout_bytes, len(b"started\n"))

    async def test_resolved_spec_spawns_trusted_executable(self) -> None:
        spec = await ExecutionPolicy(
            resolver=_ResolvedResolver("/trusted/bin/rg"),
        ).normalize(_request(metadata={"source": "test"}))
        process = _FakeProcess(stdout=b"match\n", stderr=b"")
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _FakeProcess:
            if kwargs.get("stdin") == DEVNULL:
                process.stdin = None
            calls.append((args, kwargs))
            return process

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            result = await LocalCommandExecutor().execute(spec)

        expected_args = ("/trusted/bin/rg", *spec.argv[1:])
        self.assertEqual(
            calls,
            [
                (
                    expected_args,
                    {
                        "cwd": str(Path.cwd().resolve()),
                        "env": spec.env,
                        "stdin": DEVNULL,
                        "stdout": PIPE,
                        "stderr": PIPE,
                        "start_new_session": True,
                    },
                )
            ],
        )
        self.assertEqual(process.stdin, None)
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.error_code, ShellExecutionErrorCode.COMPLETED)
        self.assertIsNone(result.error_message)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "match\n")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout_bytes, 6)
        self.assertEqual(result.stderr_bytes, 0)
        self.assertGreaterEqual(result.duration_ms, 0)
        self.assertEqual(
            result.metadata,
            {
                "source": "test",
                "local_host_approval": "required",
                "exit_code_statuses": {1: "no_matches"},
            },
        )

    async def test_spawn_uses_pipe_for_trusted_stdin(self) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"input", stderr=b"")
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _FakeProcess:
            calls.append((args, kwargs))
            return process

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(calls[0][0], ("/trusted/bin/cat",))
        self.assertEqual(calls[0][1]["stdin"], PIPE)
        self.assertIsInstance(process.stdin, _FakeStdin)
        self.assertEqual(process.stdin.data, b"input")
        self.assertTrue(process.stdin.closed)
        self.assertEqual(result.stdout, "input")

    async def test_trusted_stdin_starts_readers_before_drain_completes(
        self,
    ) -> None:
        release_stdin = Event()
        process = _FakeProcess(stdout=b"output", stderr=b"")
        stdin = _BlockingStdin(release_stdin)
        process.stdin = stdin
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            running = create_task(LocalCommandExecutor().execute(spec))
            try:
                await wait_for(stdin.drain_started.wait(), timeout=1)
                await sleep(0)

                self.assertEqual(stdin.data, b"input")
                self.assertGreater(process.stdout.read_calls, 0)
            finally:
                release_stdin.set()

            result = await wait_for(running, timeout=1)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "output")
        self.assertTrue(process.stdin.closed)

    async def test_trusted_stdin_tolerates_missing_pipe(self) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = None

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)

    async def test_trusted_stdin_tolerates_non_writer_pipe(self) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = object()

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(process.stdin, b"input")

    async def test_trusted_stdin_tolerates_broken_pipe(self) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = _BrokenPipeStdin()

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertTrue(process.stdin.closed)

    async def test_trusted_stdin_tolerates_close_connection_reset(
        self,
    ) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = _CloseResetStdin()

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertTrue(process.stdin.closed)

    async def test_trusted_stdin_tolerates_wait_closed_broken_pipe(
        self,
    ) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = _WaitClosedBrokenPipeStdin()

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertTrue(process.stdin.closed)

    async def test_trusted_stdin_tolerates_writer_without_wait_closed(
        self,
    ) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = _NoWaitClosedStdin()

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertTrue(process.stdin.closed)

    async def test_trusted_stdin_tolerates_writer_without_close(
        self,
    ) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
        )
        process = _FakeProcess(stdout=b"", stderr=b"")
        process.stdin = _NoCloseStdin()

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(process.stdin.data, b"input")

    async def test_spawn_omits_model_command_when_argv_is_empty(
        self,
    ) -> None:
        spec = _direct_spec(executable="/trusted/bin/tool", argv=())
        process = _FakeProcess(stdout=b"", stderr=b"")
        calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _FakeProcess:
            calls.append((args, kwargs))
            return process

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(calls[0][0], ("/trusted/bin/tool",))
        self.assertEqual(result.argv, ())
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)

    async def test_exit_code_metadata_can_map_no_matches(self) -> None:
        spec = await ExecutionPolicy(
            resolver=_ResolvedResolver("/trusted/bin/rg"),
        ).normalize(_request())
        process = _FakeProcess(returncode=1, stdout=b"", stderr=b"")

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.NO_MATCHES)
        self.assertEqual(result.error_code, ShellExecutionErrorCode.NO_MATCHES)
        self.assertEqual(result.error_message, "command exited with status 1")

    async def test_nonzero_exit_defaults_to_nonzero_status(self) -> None:
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stderr_bytes=11,
        )
        process = _FakeProcess(
            returncode=2,
            stdout=b"partial",
            stderr=b"bad utf8: \xff",
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.NONZERO_EXIT)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.NONZERO_EXIT,
        )
        self.assertEqual(result.exit_code, 2)
        self.assertEqual(result.stdout, "partial")
        self.assertEqual(result.stderr, "bad utf8: \ufffd")
        self.assertEqual(result.stderr_bytes, 11)

    async def test_stdout_and_stderr_are_read_concurrently(self) -> None:
        stdout_release = Event()
        stderr_release = Event()
        process = _FakeProcess(
            stdout=_ControlledStream(
                chunks=(b"out",),
                before_first_read=stdout_release.wait,
            ),
            stderr=_ControlledStream(
                chunks=(b"err",),
                before_first_read=stderr_release.wait,
            ),
        )
        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _FakeProcess:
            return process

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            task = create_task(LocalCommandExecutor().execute(spec))
            await wait_for(process.wait_started.wait(), timeout=1)
            await sleep(0)

            self.assertEqual(process.stdout.read_calls, 1)
            self.assertEqual(process.stderr.read_calls, 1)
            stdout_release.set()
            stderr_release.set()
            result = await wait_for(task, timeout=1)

        self.assertEqual(result.stdout, "out")
        self.assertEqual(result.stderr, "err")

    async def test_stream_reads_use_configured_chunk_size(self) -> None:
        settings = ShellToolSettings(stream_read_chunk_bytes=3)
        process = _FakeProcess(stdout=b"abcdef", stderr=b"ghij")
        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor(settings).execute(spec)

        self.assertEqual(result.stdout, "abcdef")
        self.assertEqual(result.stderr, "ghij")
        self.assertTrue(all(size == 3 for size in process.stdout.read_sizes))
        self.assertTrue(all(size == 3 for size in process.stderr.read_sizes))

    async def test_stream_caps_are_independent_and_drain_to_eof(self) -> None:
        settings = ShellToolSettings(stream_read_chunk_bytes=4)
        process = _FakeProcess(stdout=b"abcdefghijkl", stderr=b"uvwxyz")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stdout_bytes=5,
            max_stderr_bytes=2,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor(settings).execute(spec)

        self.assertEqual(result.stdout, "abcde")
        self.assertEqual(result.stderr, "uv")
        self.assertEqual(result.stdout_bytes, 5)
        self.assertEqual(result.stderr_bytes, 2)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        self.assertEqual(process.stdout.drained_bytes, 12)
        self.assertEqual(process.stderr.drained_bytes, 6)

    async def test_multibyte_utf8_split_at_cap_uses_replacement(self) -> None:
        process = _FakeProcess(stdout="é".encode(), stderr=b"")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stdout_bytes=1,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.stdout, "\ufffd")
        self.assertEqual(result.stdout_bytes, 1)
        self.assertTrue(result.stdout_truncated)

    async def test_large_stream_retains_only_cap_while_draining(self) -> None:
        settings = ShellToolSettings(stream_read_chunk_bytes=7)
        process = _FakeProcess(stdout=b"x" * 100, stderr=b"")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stdout_bytes=13,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor(settings).execute(spec)

        self.assertEqual(result.stdout_bytes, 13)
        self.assertEqual(len(result.stdout.encode()), 13)
        self.assertTrue(result.stdout_truncated)
        self.assertEqual(process.stdout.drained_bytes, 100)
        self.assertLessEqual(
            result.stdout_bytes,
            spec.max_stdout_bytes + settings.stream_read_chunk_bytes,
        )

    async def test_stream_read_failure_returns_tool_error(self) -> None:
        process = _FakeProcess(
            stdout=_FailingStream(OSError("/private/tmp/secret")),
            stderr=b"warning",
        )
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stderr_bytes=20,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(result.error_code, ShellExecutionErrorCode.TOOL_ERROR)
        self.assertEqual(result.error_message, "stream collection failed")
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "warning")
        self.assertEqual(result.stdout_bytes, 0)
        self.assertEqual(result.stderr_bytes, 7)
        self.assertNotIn("/private/tmp/secret", result.error_message)

    async def test_stream_read_failure_preserves_completed_peer_stream(
        self,
    ) -> None:
        process = _FakeProcess(
            stdout=b"partial output",
            stderr=_FailingStream(RuntimeError("stderr failed")),
        )
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stdout_bytes=20,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(result.error_message, "stream collection failed")
        self.assertEqual(result.stdout, "partial output")
        self.assertEqual(result.stderr, "")
        self.assertEqual(result.stdout_bytes, 14)
        self.assertEqual(result.stderr_bytes, 0)

    async def test_spawn_failures_return_spawn_failed(self) -> None:
        for error in (
            FileNotFoundError("missing"),
            PermissionError("denied"),
            OSError("failed"),
            ValueError("/private/tmp/secret"),
        ):
            with self.subTest(error=type(error).__name__):
                spec = _direct_spec(
                    executable="/trusted/bin/tool",
                    argv=("tool", "--version"),
                )

                async def fake_create_subprocess_exec(
                    *args: object,
                    **kwargs: object,
                ) -> _FakeProcess:
                    raise error

                with patch(
                    "avalan.tool.shell.executor.create_subprocess_exec",
                    new=fake_create_subprocess_exec,
                ):
                    result = await LocalCommandExecutor().execute(spec)

                self.assertEqual(
                    result.status,
                    ShellExecutionStatus.SPAWN_FAILED,
                )
                self.assertEqual(
                    result.error_code,
                    ShellExecutionErrorCode.SPAWN_FAILED,
                )
                self.assertEqual(result.error_message, "process spawn failed")
                self.assertIsNone(result.exit_code)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "")
                self.assertEqual(result.stdout_bytes, 0)
                self.assertEqual(result.stderr_bytes, 0)
                self.assertFalse(result.timed_out)
                self.assertNotIn("/private/tmp/secret", result.error_message)

    async def test_standard_processes_are_limited_by_settings(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=2)
        executor = LocalCommandExecutor(settings=settings)
        release = Event()
        tracker = _ProcessTracker(target_active=2)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _BlockingProcess:
            nonlocal spawn_count
            spawn_count += 1
            return _BlockingProcess(release=release, tracker=tracker)

        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            tasks = [create_task(executor.execute(spec)) for _ in range(4)]
            try:
                await wait_for(tracker.target_reached.wait(), timeout=1)
                await sleep(0)

                self.assertEqual(spawn_count, 2)
                self.assertEqual(tracker.maximum_active, 2)
            finally:
                release.set()

            results = await wait_for(gather(*tasks), timeout=1)

        self.assertEqual(spawn_count, 4)
        self.assertEqual(tracker.maximum_active, 2)
        self.assertTrue(
            all(
                result.status is ShellExecutionStatus.COMPLETED
                for result in results
            )
        )

    async def test_heavy_processes_are_limited_independently(self) -> None:
        settings = ShellToolSettings(
            max_concurrent_processes=4,
            max_concurrent_heavy_processes=1,
        )
        executor = LocalCommandExecutor(settings=settings)
        release = Event()
        tracker = _ProcessTracker(target_active=1)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _BlockingProcess:
            nonlocal spawn_count
            spawn_count += 1
            return _BlockingProcess(release=release, tracker=tracker)

        spec = _direct_spec(
            executable="/trusted/bin/pdftotext",
            argv=("pdftotext", "document.pdf", "-"),
            resource_class="heavy",
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            tasks = [create_task(executor.execute(spec)) for _ in range(3)]
            try:
                await wait_for(tracker.target_reached.wait(), timeout=1)
                await sleep(0)
                await sleep(0)

                self.assertEqual(spawn_count, 1)
                self.assertEqual(tracker.maximum_active, 1)
            finally:
                release.set()

            results = await wait_for(gather(*tasks), timeout=1)

        self.assertEqual(spawn_count, 3)
        self.assertEqual(tracker.maximum_active, 1)
        self.assertTrue(
            all(
                result.status is ShellExecutionStatus.COMPLETED
                for result in results
            )
        )

    async def test_queued_heavy_processes_do_not_starve_standard_work(
        self,
    ) -> None:
        settings = ShellToolSettings(
            max_concurrent_processes=2,
            max_concurrent_heavy_processes=1,
        )
        executor = LocalCommandExecutor(settings=settings)
        heavy_release = Event()
        standard_release = Event()
        heavy_tracker = _ProcessTracker(target_active=1)
        standard_started = Event()
        spawn_order: list[str] = []

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _BlockingProcess:
            executable = args[0]
            if executable == "/trusted/bin/heavy":
                spawn_order.append("heavy")
                return _BlockingProcess(
                    release=heavy_release,
                    tracker=heavy_tracker,
                )
            spawn_order.append("standard")
            standard_started.set()
            return _BlockingProcess(release=standard_release)

        heavy_spec = _direct_spec(
            executable="/trusted/bin/heavy",
            argv=("pdftotext", "document.pdf", "-"),
            resource_class="heavy",
        )
        standard_spec = _direct_spec(
            executable="/trusted/bin/standard",
            argv=("rg", "needle"),
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            first_heavy = create_task(executor.execute(heavy_spec))
            try:
                await wait_for(heavy_tracker.target_reached.wait(), timeout=1)
                queued_heavy = [
                    create_task(executor.execute(heavy_spec)) for _ in range(2)
                ]
                await sleep(0)
                standard = create_task(executor.execute(standard_spec))

                await wait_for(standard_started.wait(), timeout=1)
                self.assertEqual(spawn_order, ["heavy", "standard"])
            finally:
                standard_release.set()
                heavy_release.set()

            await wait_for(
                gather(first_heavy, standard, *queued_heavy),
                timeout=1,
            )

    async def test_semaphore_released_after_spawn_failure(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=1)
        executor = LocalCommandExecutor(settings=settings)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _FakeProcess:
            nonlocal spawn_count
            spawn_count += 1
            if spawn_count == 1:
                raise OSError("failed")
            return _FakeProcess(stdout=b"ok", stderr=b"")

        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            failed = await executor.execute(spec)
            completed = await wait_for(executor.execute(spec), timeout=1)

        self.assertEqual(failed.status, ShellExecutionStatus.SPAWN_FAILED)
        self.assertEqual(completed.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(spawn_count, 2)

    async def test_semaphore_released_after_normal_completion(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=1)
        executor = LocalCommandExecutor(settings=settings)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _FakeProcess:
            nonlocal spawn_count
            spawn_count += 1
            return _FakeProcess(stdout=b"ok", stderr=b"")

        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            first = await executor.execute(spec)
            second = await wait_for(executor.execute(spec), timeout=1)

        self.assertEqual(first.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(second.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(spawn_count, 2)

    async def test_semaphore_released_after_timeout_exception(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=1)
        executor = LocalCommandExecutor(settings=settings)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> object:
            nonlocal spawn_count
            spawn_count += 1
            if spawn_count == 1:
                return _TimeoutProcess()
            return _FakeProcess(stdout=b"ok", stderr=b"")

        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            failed = await executor.execute(spec)
            completed = await wait_for(executor.execute(spec), timeout=1)

        self.assertEqual(failed.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(completed.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(spawn_count, 2)

    async def test_semaphore_released_after_active_cancellation(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=1)
        executor = LocalCommandExecutor(settings=settings)
        release = Event()
        tracker = _ProcessTracker(target_active=1)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> object:
            nonlocal spawn_count
            spawn_count += 1
            if spawn_count == 1:
                return _BlockingProcess(release=release, tracker=tracker)
            return _FakeProcess(stdout=b"ok", stderr=b"")

        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            running = create_task(executor.execute(spec))
            await wait_for(tracker.target_reached.wait(), timeout=1)
            running.cancel()
            with self.assertRaises(CancelledError):
                await running

            completed = await wait_for(executor.execute(spec), timeout=1)

        self.assertEqual(completed.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(spawn_count, 2)

    async def test_semaphore_released_after_queued_cancellation(self) -> None:
        settings = ShellToolSettings(max_concurrent_processes=1)
        executor = LocalCommandExecutor(settings=settings)
        release = Event()
        tracker = _ProcessTracker(target_active=1)
        spawn_count = 0

        async def fake_create_subprocess_exec(
            *args: object,
            **kwargs: object,
        ) -> _BlockingProcess | _FakeProcess:
            nonlocal spawn_count
            spawn_count += 1
            if spawn_count == 1:
                return _BlockingProcess(release=release, tracker=tracker)
            return _FakeProcess(stdout=b"ok", stderr=b"")

        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=fake_create_subprocess_exec,
        ):
            running = create_task(executor.execute(spec))
            try:
                await wait_for(tracker.target_reached.wait(), timeout=1)
                queued = create_task(executor.execute(spec))
                await sleep(0)

                self.assertEqual(spawn_count, 1)
                queued.cancel()
                with self.assertRaises(CancelledError):
                    await queued
            finally:
                release.set()

            completed_running = await wait_for(running, timeout=1)
            completed_after_cancel = await wait_for(
                executor.execute(spec),
                timeout=1,
            )

        self.assertEqual(
            completed_running.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertEqual(
            completed_after_cancel.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertEqual(spawn_count, 2)

    async def test_forged_execution_spec_is_rejected_before_execution(
        self,
    ) -> None:
        with self.assertRaises(AssertionError):
            ExecutionSpec(
                _policy_owned=object(),
                backend="local",
                tool_name="shell.rg",
                command="rg",
                executable=None,
                argv=("rg",),
                display_argv=("rg",),
                cwd="/workspace",
                display_cwd=".",
                env={},
                stdin=None,
                stdout_media_type="text/plain",
                output_kind=ShellOutputKind.TEXT,
                resource_class="standard",
                output_plan=None,
                timeout_seconds=1.0,
                max_stdout_bytes=1,
                max_stderr_bytes=1,
            )

    async def test_timeout_returns_partial_output_and_cleans_process(
        self,
    ) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"warning")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            timeout_seconds=0.001,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.error_code, ShellExecutionErrorCode.TIMEOUT)
        self.assertEqual(result.error_message, "command timed out")
        self.assertTrue(result.timed_out)
        self.assertFalse(result.cancelled)
        self.assertEqual(result.exit_code, -15)
        self.assertEqual(result.stdout, "partial")
        self.assertEqual(result.stderr, "warning")
        self.assertEqual(result.stdout_bytes, 7)
        self.assertEqual(result.stderr_bytes, 7)
        self.assertEqual(process.terminate_count, 1)
        self.assertEqual(process.kill_count, 0)

    async def test_timeout_emits_terminal_stream_progress(self) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"warning")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            timeout_seconds=0.001,
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(
                spec,
                stream=record,
            )

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(
            (events[-1].kind, events[-1].content, events[-1].progress),
            (ToolExecutionStreamKind.PROGRESS, "timeout", 1.0),
        )
        self.assertEqual(
            events[-1].metadata,
            {"exit_code": -15, "status": "timeout", "timed_out": True},
        )
        self.assertEqual(
            [
                event.content
                for event in events
                if event.kind is ToolExecutionStreamKind.STDOUT
            ],
            ["partial"],
        )
        self.assertEqual(
            [
                event.content
                for event in events
                if event.kind is ToolExecutionStreamKind.STDERR
            ],
            ["warning"],
        )

    async def test_timeout_drains_large_stdout_and_stderr(
        self,
    ) -> None:
        settings = ShellToolSettings(stream_read_chunk_bytes=4)
        process = _TerminableProcess(stdout=b"x" * 64, stderr=b"y" * 64)
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            timeout_seconds=0.001,
            max_stdout_bytes=5,
            max_stderr_bytes=7,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor(settings).execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.stdout, "xxxxx")
        self.assertEqual(result.stderr, "yyyyyyy")
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        self.assertEqual(process.stdout.drained_bytes, 64)
        self.assertEqual(process.stderr.drained_bytes, 64)

    async def test_timeout_preserves_partial_output_when_reader_hangs(
        self,
    ) -> None:
        stdout_ready = Event()
        process = _ReaderPendingTimeoutProcess(stdout_ready)
        process.stdout = _HangingAfterChunksStream(
            chunks=(b"partial output",),
            exhausted=stdout_ready,
        )
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            timeout_seconds=0.01,
            max_stdout_bytes=20,
        )

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.stdout, "partial output")
        self.assertEqual(result.stdout_bytes, 14)
        self.assertFalse(result.stdout_truncated)
        self.assertEqual(process.stdout.drained_bytes, 14)
        self.assertGreaterEqual(process.stdout.read_calls, 2)

    async def test_normal_completion_waits_for_delayed_stream_eof(
        self,
    ) -> None:
        release_stdout = Event()
        process = _FakeProcess(
            stdout=_ControlledStream(
                chunks=(b"late output",),
                before_first_read=release_stdout.wait,
            ),
            stderr=b"",
        )
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            max_stdout_bytes=20,
        )

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                running = create_task(LocalCommandExecutor().execute(spec))
                try:
                    await wait_for(process.wait_started.wait(), timeout=1)
                    await sleep(0.01)

                    self.assertFalse(running.done())
                finally:
                    release_stdout.set()

                result = await wait_for(running, timeout=1)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "late output")
        self.assertEqual(result.stdout_bytes, 11)
        self.assertFalse(result.stdout_truncated)

    async def test_timeout_uses_kill_after_terminate_grace_expires(
        self,
    ) -> None:
        process = _TerminableProcess(
            stdout=b"partial",
            stderr=b"",
            terminate_releases=False,
        )
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            timeout_seconds=0.001,
        )

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.exit_code, -9)
        self.assertEqual(process.terminate_count, 1)
        self.assertEqual(process.kill_count, 1)

    async def test_timeout_tolerates_cleanup_errors_without_leaking_them(
        self,
    ) -> None:
        process = _CleanupErrorProcess(stdout=b"partial", stderr=b"")
        spec = _direct_spec(
            executable="/trusted/bin/tool",
            argv=("tool",),
            timeout_seconds=0.001,
        )

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.error_message, "command timed out")
        self.assertNotIn("/private/tmp/secret", result.error_message)

    async def test_timeout_while_writing_stdin_returns_partial_output(
        self,
    ) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"")
        stdin = _BlockingStdin(Event())
        process.stdin = stdin
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
            timeout_seconds=0.001,
        )

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await wait_for(
                LocalCommandExecutor().execute(spec),
                timeout=1,
            )

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.stdout, "partial")
        self.assertTrue(result.timed_out)
        self.assertTrue(stdin.closed)
        self.assertEqual(process.terminate_count, 1)

    async def test_timeout_while_writing_stdin_bounds_close_wait(
        self,
    ) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"")
        stdin = _HangingWaitClosedStdin()
        process.stdin = stdin
        spec = _direct_spec(
            executable="/trusted/bin/cat",
            argv=("cat",),
            stdin=b"input",
            timeout_seconds=0.001,
        )

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await wait_for(
                    LocalCommandExecutor().execute(spec),
                    timeout=1,
                )

        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
        self.assertEqual(result.stdout, "partial")
        self.assertTrue(stdin.closed)
        self.assertTrue(stdin.wait_closed_started.is_set())
        self.assertEqual(process.terminate_count, 1)

    async def test_cancellation_kills_process_and_propagates(
        self,
    ) -> None:
        process = _TerminableProcess(stdout=b"partial", stderr=b"")
        spec = _direct_spec(executable="/trusted/bin/tool", argv=("tool",))

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            running = create_task(LocalCommandExecutor().execute(spec))
            await wait_for(process.wait_started.wait(), timeout=1)
            running.cancel()

            with self.assertRaises(CancelledError):
                await running

        self.assertEqual(process.kill_count, 1)

    async def test_media_tool_timeouts_honor_normalized_caps(
        self,
    ) -> None:
        for command, timeout_seconds in (
            ("pdftotext", 0.001),
            ("tesseract", 0.002),
        ):
            with self.subTest(command=command):
                process = _TerminableProcess(stdout=b"partial", stderr=b"")
                spec = _direct_spec(
                    executable=f"/trusted/bin/{command}",
                    argv=(command,),
                    command=command,
                    tool_name=f"shell.{command}",
                    resource_class="heavy",
                    timeout_seconds=timeout_seconds,
                )

                with patch(
                    "avalan.tool.shell.executor.create_subprocess_exec",
                    new=_fake_process_factory(process),
                ):
                    result = await LocalCommandExecutor().execute(spec)

                self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)
                self.assertTrue(result.timed_out)
                self.assertEqual(process.terminate_count, 1)

    async def test_generated_output_capture_returns_bounded_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root, max_inline_bytes=100)
            png = _png_bytes(width=2, height=3)
            mode_seen: list[int] = []
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                output_directory = _spawned_output_prefix(process).parent
                mode_seen.append(S_IMODE(output_directory.stat().st_mode))
                (
                    output_directory / f"{plan.prefix_name}-0001.png"
                ).write_bytes(png)

            process = _GeneratedOutputProcess(write_outputs)
            spec = _direct_spec(
                executable="/trusted/bin/pdftoppm",
                argv=(
                    "pdftoppm",
                    "input.pdf",
                    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
                ),
                display_argv=("pdftoppm", "input.pdf", plan.display_prefix),
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.GENERATED_FILES,
                output_plan=plan,
                resource_class="heavy",
            )

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)

            runtime_prefix = _spawned_output_prefix(process)
            output_directory = runtime_prefix.parent

        self.assertEqual(mode_seen, [0o700])
        self.assertFalse(output_directory.exists())
        self.assertEqual(runtime_prefix.name, plan.prefix_name)
        self.assertTrue(output_directory.name.startswith("avalan-shell-"))
        self.assertTrue(runtime_prefix.is_absolute())
        self.assertNotIn(
            GENERATED_OUTPUT_PREFIX_PLACEHOLDER, process.spawn_args
        )
        self.assertIn(str(runtime_prefix), process.spawn_args)
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.argv, spec.display_argv)
        self.assertEqual(result.display_argv, spec.display_argv)
        self.assertNotIn(str(output_directory), " ".join(result.argv))
        self.assertEqual(len(result.generated_files), 1)
        generated = result.generated_files[0]
        self.assertEqual(generated.display_path, "GENERATED_PREFIX-0001.png")
        self.assertEqual(generated.media_type, "image/png")
        self.assertEqual(generated.suffix, ".png")
        self.assertEqual(generated.bytes, len(png))
        self.assertEqual(generated.sha256, sha256(png).hexdigest())
        self.assertEqual(generated.page, 1)
        self.assertEqual(generated.width, 2)
        self.assertEqual(generated.height, 3)
        self.assertEqual(
            generated.content_base64,
            b64encode(png).decode("ascii"),
        )

    async def test_generated_output_scrubs_private_paths_from_streams(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            raw_stdout_seen: list[bytes] = []
            raw_stderr_seen: list[bytes] = []
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                output_directory = _spawned_output_prefix(process).parent
                (
                    output_directory / f"{plan.prefix_name}-0001.png"
                ).write_bytes(_png_bytes())

            def after_spawn() -> None:
                runtime_prefix = _spawned_output_prefix(process)
                output_directory = runtime_prefix.parent
                raw_stdout = (
                    f"created {runtime_prefix}-0001.png in {output_directory}"
                ).encode()
                raw_stderr = f"prefix {runtime_prefix}".encode()
                raw_stdout_seen.append(raw_stdout)
                raw_stderr_seen.append(raw_stderr)
                process.stdout = _FakeStream(raw_stdout)
                process.stderr = _FakeStream(raw_stderr)

            process = _GeneratedOutputProcess(write_outputs)
            spec = _direct_spec(
                executable="/trusted/bin/pdftoppm",
                argv=(
                    "pdftoppm",
                    "input.pdf",
                    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
                ),
                display_argv=("pdftoppm", "input.pdf", plan.display_prefix),
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.GENERATED_FILES,
                output_plan=plan,
                resource_class="heavy",
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
            )

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process, after_spawn=after_spawn),
            ):
                result = await LocalCommandExecutor().execute(spec)
            runtime_prefix = _spawned_output_prefix(process)
            output_directory = runtime_prefix.parent

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout_bytes, len(raw_stdout_seen[0]))
        self.assertEqual(result.stderr_bytes, len(raw_stderr_seen[0]))
        self.assertNotIn(str(runtime_prefix), result.stdout)
        self.assertNotIn(str(runtime_prefix), result.stderr)
        self.assertNotIn(str(output_directory), result.stdout)
        self.assertNotIn(str(output_directory), result.stderr)
        self.assertIn("GENERATED_PREFIX-0001.png", result.stdout)
        self.assertIn("[generated_output_directory]", result.stdout)
        self.assertIn("GENERATED_PREFIX", result.stderr)

    async def test_generated_output_scrubs_resolved_private_stream_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            raw_stdout_seen: list[bytes] = []
            raw_stderr_seen: list[bytes] = []
            output_directory = root / "avalan-shell-runtime"
            runtime_prefix = output_directory / plan.prefix_name
            resolved_directory = Path("/resolved") / output_directory.name
            resolved_prefix = resolved_directory / runtime_prefix.name
            process = _GeneratedOutputProcess(lambda: None)

            @asynccontextmanager
            async def fake_private_temp_directory() -> AsyncIterator[Path]:
                output_directory.mkdir(mode=0o700)
                try:
                    yield output_directory
                finally:
                    await _cleanup_output_directory(output_directory)

            async def fake_resolve_policy_path(path: str | Path) -> Path:
                source_path = Path(path)
                if source_path == output_directory:
                    return resolved_directory
                if source_path == runtime_prefix:
                    return resolved_prefix
                return await resolve_policy_path(source_path)

            def after_spawn() -> None:
                raw_stdout = (
                    f"created {resolved_prefix}-0001.png in "
                    f"{resolved_directory}"
                ).encode()
                raw_stderr = f"prefix {resolved_prefix}".encode()
                raw_stdout_seen.append(raw_stdout)
                raw_stderr_seen.append(raw_stderr)
                process.stdout = _FakeStream(raw_stdout)
                process.stderr = _FakeStream(raw_stderr)

            spec = _direct_spec(
                executable="/trusted/bin/pdftoppm",
                argv=(
                    "pdftoppm",
                    "input.pdf",
                    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
                ),
                display_argv=("pdftoppm", "input.pdf", plan.display_prefix),
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.GENERATED_FILES,
                output_plan=plan,
                resource_class="heavy",
                max_stdout_bytes=1024,
                max_stderr_bytes=1024,
            )

            with (
                patch(
                    "avalan.tool.shell.executor.create_subprocess_exec",
                    new=_fake_process_factory(
                        process, after_spawn=after_spawn
                    ),
                ),
                patch(
                    "avalan.tool.shell.executor.resolve_policy_path",
                    new=fake_resolve_policy_path,
                ),
                patch(
                    "avalan.tool.shell.executor.private_temp_directory",
                    new=fake_private_temp_directory,
                ),
            ):
                result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertNotIn(str(resolved_prefix), result.stdout)
        self.assertNotIn(str(resolved_prefix), result.stderr)
        self.assertNotIn(str(resolved_directory), result.stdout)
        self.assertNotIn(str(resolved_directory), result.stderr)
        self.assertIn("GENERATED_PREFIX-0001.png", result.stdout)
        self.assertIn("[generated_output_directory]", result.stdout)
        self.assertIn("GENERATED_PREFIX", result.stderr)

    async def test_generated_output_replacements_tolerate_resolve_errors(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            runtime_prefix = _private_output_prefix(
                root,
                prefix_name=plan.prefix_name,
            )
            output_directory = runtime_prefix.parent

            with patch(
                "avalan.tool.shell.executor.resolve_policy_path",
                side_effect=OSError("/private/tmp/secret"),
            ):
                replacements = await _generated_output_path_replacements(
                    plan,
                    runtime_prefix,
                )

        self.assertIn(
            (str(runtime_prefix), plan.display_prefix),
            replacements,
        )
        self.assertIn(
            (str(output_directory), "[generated_output_directory]"),
            replacements,
        )
        self.assertNotIn("/private/tmp/secret", str(replacements))

    async def test_generated_output_failure_scrubs_private_stream_paths(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            raw_stdout_seen: list[bytes] = []
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                output_directory = _spawned_output_prefix(process).parent
                (output_directory / f"{plan.prefix_name}-1.txt").write_text(
                    "not allowed",
                    encoding="utf-8",
                )

            def after_spawn() -> None:
                runtime_prefix = _spawned_output_prefix(process)
                raw_stdout = f"created {runtime_prefix}-1.txt".encode()
                raw_stdout_seen.append(raw_stdout)
                process.stdout = _FakeStream(raw_stdout)

            process = _GeneratedOutputProcess(write_outputs)
            spec = _direct_spec(
                executable="/trusted/bin/pdftoppm",
                argv=(
                    "pdftoppm",
                    "input.pdf",
                    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
                ),
                display_argv=("pdftoppm", "input.pdf", plan.display_prefix),
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.GENERATED_FILES,
                output_plan=plan,
                resource_class="heavy",
                max_stdout_bytes=1024,
            )

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process, after_spawn=after_spawn),
            ):
                result = await LocalCommandExecutor().execute(spec)
            runtime_prefix = _spawned_output_prefix(process)
            output_directory = runtime_prefix.parent

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
        self.assertEqual(result.stdout_bytes, len(raw_stdout_seen[0]))
        self.assertNotIn(str(runtime_prefix), result.stdout)
        self.assertNotIn(str(output_directory), result.stdout)
        self.assertIn("GENERATED_PREFIX-1.txt", result.stdout)

    async def test_generated_output_inline_cap_omits_content(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root, max_inline_bytes=1)
            png = _png_bytes(width=1, height=1)
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                (
                    _spawned_output_prefix(process).parent
                    / f"{plan.prefix_name}-1.png"
                ).write_bytes(png)

            process = _GeneratedOutputProcess(write_outputs)
            spec = _direct_spec(
                executable="/trusted/bin/pdftoppm",
                argv=("pdftoppm", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
                display_argv=("pdftoppm", plan.display_prefix),
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                stdout_media_type="application/json",
                output_kind=ShellOutputKind.GENERATED_FILES,
                output_plan=plan,
            )

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)

        self.assertIsNone(result.generated_files[0].content_base64)

    async def test_generated_output_rejects_unsafe_suffix(self) -> None:
        result, output_directory = await _run_generated_output_case(
            lambda plan: (Path(plan.prefix).parent / "page-1.txt").write_text(
                "not allowed", encoding="utf-8"
            )
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
        self.assertEqual(
            result.error_message,
            "generated output capture failed",
        )
        self.assertFalse(output_directory.exists())
        self.assertNotIn(str(output_directory), result.error_message)

    async def test_generated_output_rejects_file_count_cap(self) -> None:
        def write_outputs(plan: _GeneratedOutputRuntimePlan) -> None:
            output_directory = Path(plan.prefix).parent
            (output_directory / "page-1.png").write_bytes(_png_bytes())
            (output_directory / "page-2.png").write_bytes(_png_bytes())

        result, _ = await _run_generated_output_case(
            write_outputs,
            max_files=1,
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)

    async def test_generated_output_rejects_single_file_cap(self) -> None:
        result, _ = await _run_generated_output_case(
            lambda plan: (Path(plan.prefix).parent / "page-1.png").write_bytes(
                _png_bytes()
            ),
            max_file_bytes=3,
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)

    async def test_generated_output_rejects_total_file_cap(self) -> None:
        def write_outputs(plan: _GeneratedOutputRuntimePlan) -> None:
            output_directory = Path(plan.prefix).parent
            (output_directory / "page-1.png").write_bytes(_png_bytes())
            (output_directory / "page-2.png").write_bytes(_png_bytes())

        result, _ = await _run_generated_output_case(
            write_outputs,
            max_total_bytes=len(_png_bytes()) + 1,
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)

    async def test_generated_output_ignores_prefix_lookalike_names(
        self,
    ) -> None:
        def write_outputs(plan: _GeneratedOutputRuntimePlan) -> None:
            output_directory = Path(plan.prefix).parent
            (output_directory / "page-1.png").write_bytes(_png_bytes())
            (output_directory / "pageevil.png").write_bytes(b"x" * 128)

        result, _ = await _run_generated_output_case(
            write_outputs,
            max_file_bytes=64,
        )

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            tuple(file.display_path for file in result.generated_files),
            ("GENERATED_PREFIX-1.png",),
        )

    async def test_generated_output_accepts_underscore_page_number(
        self,
    ) -> None:
        result, _ = await _run_generated_output_case(
            lambda plan: (
                Path(plan.prefix).parent / "page_0002.png"
            ).write_bytes(_png_bytes())
        )

        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(len(result.generated_files), 1)
        self.assertEqual(
            result.generated_files[0].display_path,
            "GENERATED_PREFIX_0002.png",
        )
        self.assertEqual(result.generated_files[0].page, 2)

    async def test_generated_output_rejects_arbitrary_prefix_suffix(
        self,
    ) -> None:
        result, output_directory = await _run_generated_output_case(
            lambda plan: (
                Path(plan.prefix).parent / "page-secret.png"
            ).write_bytes(_png_bytes())
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
        self.assertFalse(output_directory.exists())
        self.assertEqual(result.generated_files, ())

    def test_generated_output_prefix_matching_requires_boundary(self) -> None:
        prefix_path = Path("/tmp/avalan-shell-abc/page")

        self.assertTrue(
            _matches_generated_output_prefix(
                Path("/tmp/avalan-shell-abc/page"),
                prefix_path,
            )
        )
        self.assertTrue(
            _matches_generated_output_prefix(
                Path("/tmp/avalan-shell-abc/page-1.png"),
                prefix_path,
            )
        )
        self.assertTrue(
            _matches_generated_output_prefix(
                Path("/tmp/avalan-shell-abc/page_1.png"),
                prefix_path,
            )
        )
        self.assertTrue(
            _matches_generated_output_prefix(
                Path("/tmp/avalan-shell-abc/page.png"),
                prefix_path,
            )
        )
        self.assertFalse(
            _matches_generated_output_prefix(
                Path("/tmp/avalan-shell-abc/pageevil.png"),
                prefix_path,
            )
        )
        self.assertFalse(
            _matches_generated_output_prefix(
                Path("/tmp/avalan-shell-abc/other.png"),
                prefix_path,
            )
        )

    def test_generated_output_spawn_requires_runtime_prefix(self) -> None:
        spec = _generated_spec(_output_plan(Path("/tmp")))

        with self.assertRaises(_GeneratedOutputError):
            _spawn_argv(spec, runtime_output_prefix=None)

    async def test_generated_output_prepared_spawn_reports_runtime_error(
        self,
    ) -> None:
        spec = _generated_spec(_output_plan(Path("/tmp")))

        result = await LocalCommandExecutor()._execute_prepared_spawn(
            spec,
            start_time=0.0,
            runtime_output_prefix=None,
            generated_output_replacements=(),
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )

    def test_generated_output_spawn_rejects_missing_placeholder(self) -> None:
        spec = _generated_spec(_output_plan(Path("/tmp")))
        object.__setattr__(spec, "argv", ("pdftoppm", "input.pdf"))

        with self.assertRaises(_GeneratedOutputError):
            _spawn_argv(spec, runtime_output_prefix=Path("/tmp/page"))

    async def test_generated_output_collection_requires_runtime_prefix(
        self,
    ) -> None:
        plan = _output_plan(Path("/tmp"))

        with self.assertRaises(_GeneratedOutputError):
            await _collect_generated_files(plan, None, chunk_size=64)

    async def test_generated_output_collection_rejects_prefix_name_mismatch(
        self,
    ) -> None:
        plan = _output_plan(Path("/tmp"))

        with self.assertRaises(_GeneratedOutputError):
            await _collect_generated_files(
                plan,
                Path("/tmp/avalan-shell-runtime/other"),
                chunk_size=64,
            )

    async def test_generated_output_rejects_raster_dimension_cap(self) -> None:
        result, _ = await _run_generated_output_case(
            lambda plan: (Path(plan.prefix).parent / "page-1.png").write_bytes(
                _png_bytes(width=3, height=2)
            ),
            max_raster_long_edge_pixels=2,
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)

    async def test_generated_output_rejects_raster_pixel_cap(self) -> None:
        result, _ = await _run_generated_output_case(
            lambda plan: (Path(plan.prefix).parent / "page-1.png").write_bytes(
                _png_bytes(width=3, height=2)
            ),
            max_raster_long_edge_pixels=3,
            max_raster_pixels=5,
        )

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)

    async def test_generated_output_accepts_missing_dimensions_and_page(
        self,
    ) -> None:
        result, _ = await _run_generated_output_case(
            lambda plan: (Path(plan.prefix).parent / "page.png").write_bytes(
                b"not a png"
            )
        )

        generated = result.generated_files[0]
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertIsNone(generated.page)
        self.assertIsNone(generated.width)
        self.assertIsNone(generated.height)

    async def test_generated_output_accepts_non_image_media_type(self) -> None:
        with TemporaryDirectory():
            plan = GeneratedOutputPlan(
                prefix_name="part",
                display_prefix="GENERATED_PREFIX",
                allowed_suffixes=(".txt",),
                suffix_media_types={".txt": "text/plain"},
                max_files=1,
                max_file_bytes=100,
                max_total_bytes=100,
                max_inline_bytes=100,
            )
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                (
                    _spawned_output_prefix(process).parent / "part.txt"
                ).write_text(
                    "value",
                    encoding="utf-8",
                )

            process = _GeneratedOutputProcess(write_outputs)
            spec = _direct_spec(
                executable="/trusted/bin/tool",
                argv=("tool", GENERATED_OUTPUT_PREFIX_PLACEHOLDER),
                display_argv=("tool", plan.display_prefix),
                output_kind=ShellOutputKind.GENERATED_FILES,
                output_plan=plan,
            )

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)

        generated = result.generated_files[0]
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(generated.media_type, "text/plain")
        self.assertIsNone(generated.width)
        self.assertIsNone(generated.height)

    async def test_generated_output_rejects_symlink_and_hardlink(
        self,
    ) -> None:
        metadata = {
            "path": Path("page-1.png"),
            "resolved_path": Path("/tmp/page-1.png"),
            "mode": 0,
            "size": len(_png_bytes()),
            "is_file": True,
            "is_directory": False,
            "is_symlink": False,
            "is_special_file": False,
        }
        for patched_metadata in (
            {**metadata, "is_file": False, "is_symlink": True},
            {**metadata, "hardlink_count": 2},
        ):
            with self.subTest(patched_metadata=patched_metadata):

                async def fake_inspect_path(path: str | Path) -> object:
                    source = Path(path)
                    if source.name.startswith("avalan-shell-"):
                        return type(
                            "Metadata",
                            (),
                            {
                                "path": source,
                                "resolved_path": source,
                                "mode": 0o700,
                                "is_directory": True,
                                "is_symlink": False,
                            },
                        )()
                    return type(
                        "Metadata",
                        (),
                        {
                            **patched_metadata,
                            "path": source,
                            "resolved_path": source,
                        },
                    )()

                with patch(
                    "avalan.tool.shell.executor.inspect_path",
                    new=fake_inspect_path,
                ):
                    result, _ = await _run_generated_output_case(
                        lambda plan: (
                            Path(plan.prefix).parent / "page-1.png"
                        ).write_bytes(_png_bytes())
                    )

                self.assertEqual(
                    result.status,
                    ShellExecutionStatus.TOO_LARGE,
                )

    async def test_generated_output_rejects_replaced_output_directory_symlink(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            outside_directory = root / "outside"
            outside_directory.mkdir()
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                output_directory = _spawned_output_prefix(process).parent
                output_directory.rmdir()
                output_directory.symlink_to(
                    outside_directory,
                    target_is_directory=True,
                )

            process = _GeneratedOutputProcess(write_outputs)
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(
                    _generated_spec(plan)
                )

            output_directory = _spawned_output_prefix(process).parent
            self.assertFalse(output_directory.exists())
            self.assertFalse(output_directory.is_symlink())

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )

    async def test_generated_output_rejects_permissive_output_directory(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            process: _GeneratedOutputProcess

            def write_outputs() -> None:
                output_directory = _spawned_output_prefix(process).parent
                output_directory.chmod(0o755)
                (output_directory / "page-1.png").write_bytes(_png_bytes())

            process = _GeneratedOutputProcess(write_outputs)
            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(
                    _generated_spec(plan)
                )

            output_directory = _spawned_output_prefix(process).parent
            self.assertFalse(output_directory.exists())

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )

    async def test_generated_output_cleanup_runs_on_spawn_failure(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            directory_seen: list[bool] = []
            output_directory_seen: list[Path] = []
            spec = _generated_spec(plan)

            async def fake_create_subprocess_exec(
                *args: object,
                **kwargs: object,
            ) -> _FakeProcess:
                runtime_prefix = args[-1]
                assert isinstance(runtime_prefix, str)
                output_directory = Path(runtime_prefix).parent
                directory_seen.append(output_directory.is_dir())
                output_directory_seen.append(output_directory)
                raise OSError("failed")

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=fake_create_subprocess_exec,
            ):
                result = await LocalCommandExecutor().execute(spec)
            output_directory = output_directory_seen[0]

        self.assertEqual(directory_seen, [True])
        self.assertFalse(output_directory.exists())
        self.assertEqual(result.status, ShellExecutionStatus.SPAWN_FAILED)

    async def test_generated_output_private_temp_failure_returns_tool_error(
        self,
    ) -> None:
        plan = _output_plan(Path("/tmp"))

        @asynccontextmanager
        async def fake_private_temp_directory() -> AsyncIterator[Path]:
            raise OSError("failed")
            yield Path("/tmp/unreachable")

        with (
            patch(
                "avalan.tool.shell.executor.private_temp_directory",
                new=fake_private_temp_directory,
            ),
            patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_spawn_should_not_run,
            ),
        ):
            result = await LocalCommandExecutor().execute(
                _generated_spec(plan)
            )

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.error_message,
            "generated output preparation failed",
        )

    async def test_generated_output_rejects_workspace_runtime_directory(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            workspace = root / "workspace"
            output_directory = workspace / "avalan-shell-runtime"
            plan = _output_plan(root)
            spec = _generated_spec(plan, cwd=str(workspace.resolve()))

            @asynccontextmanager
            async def fake_private_temp_directory() -> AsyncIterator[Path]:
                output_directory.mkdir(parents=True, mode=0o700)
                try:
                    yield output_directory
                finally:
                    await _cleanup_output_directory(output_directory)

            with patch(
                "avalan.tool.shell.executor.private_temp_directory",
                new=fake_private_temp_directory,
            ):
                result = await LocalCommandExecutor().execute(spec)

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )
        self.assertFalse(output_directory.exists())

    async def test_generated_output_rejects_permissive_runtime_directory(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "avalan-shell-runtime"
            plan = _output_plan(root)

            @asynccontextmanager
            async def fake_private_temp_directory() -> AsyncIterator[Path]:
                output_directory.mkdir(mode=0o755)
                try:
                    yield output_directory
                finally:
                    await _cleanup_output_directory(output_directory)

            with (
                patch(
                    "avalan.tool.shell.executor.private_temp_directory",
                    new=fake_private_temp_directory,
                ),
                patch(
                    "avalan.tool.shell.executor.create_subprocess_exec",
                    new=_spawn_should_not_run,
                ),
            ):
                result = await LocalCommandExecutor().execute(
                    _generated_spec(plan)
                )

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
        )

    async def test_generated_output_cleanup_runs_on_timeout(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            process = _TerminableProcess(stdout=b"", stderr=b"")
            spec = _generated_spec(plan, timeout_seconds=0.001)

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                result = await LocalCommandExecutor().execute(spec)
            output_directory = _spawned_output_prefix(process).parent

        self.assertFalse(output_directory.exists())
        self.assertEqual(result.status, ShellExecutionStatus.TIMEOUT)

    async def test_generated_output_cleanup_runs_on_cancellation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            plan = _output_plan(root)
            process = _TerminableProcess(stdout=b"", stderr=b"")
            spec = _generated_spec(plan)

            with patch(
                "avalan.tool.shell.executor.create_subprocess_exec",
                new=_fake_process_factory(process),
            ):
                running = create_task(LocalCommandExecutor().execute(spec))
                await wait_for(process.wait_started.wait(), timeout=1)
                running.cancel()
                with self.assertRaises(CancelledError):
                    await running
                output_directory = _spawned_output_prefix(process).parent

        self.assertFalse(output_directory.exists())
        self.assertEqual(process.kill_count, 1)

    async def test_generated_output_rejects_workspace_root_runtime_directory(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            output_directory = root / "avalan-shell-runtime"
            plan = _output_plan(root)
            settings = ShellToolSettings(workspace_root=str(root))

            @asynccontextmanager
            async def fake_private_temp_directory() -> AsyncIterator[Path]:
                output_directory.mkdir(mode=0o700)
                try:
                    yield output_directory
                finally:
                    await _cleanup_output_directory(output_directory)

            with (
                patch(
                    "avalan.tool.shell.executor.private_temp_directory",
                    new=fake_private_temp_directory,
                ),
                patch(
                    "avalan.tool.shell.executor.create_subprocess_exec",
                    new=_spawn_should_not_run,
                ),
            ):
                result = await LocalCommandExecutor(settings).execute(
                    _generated_spec(plan)
                )

        self.assertEqual(result.status, ShellExecutionStatus.TOOL_ERROR)
        self.assertFalse(output_directory.exists())

    async def test_generated_output_capture_filesystem_error_is_formatted(
        self,
    ) -> None:
        with patch(
            "avalan.tool.shell.executor.list_directory",
            side_effect=OSError("/private/tmp/secret"),
        ):
            result, _ = await _run_generated_output_case(lambda plan: None)

        self.assertEqual(result.status, ShellExecutionStatus.TOO_LARGE)
        self.assertEqual(
            result.error_message,
            "generated output capture failed",
        )

    async def test_generated_output_cleanup_ignores_remove_errors(
        self,
    ) -> None:
        with patch(
            "avalan.tool.shell.executor.remove_tree",
            side_effect=OSError("/private/tmp/secret"),
        ):
            await _cleanup_output_directory(Path("/private/tmp/secret"))

    async def test_generated_output_cleanup_ignores_directory_replace_errors(
        self,
    ) -> None:
        metadata = type(
            "Metadata",
            (),
            {
                "is_file": False,
                "is_symlink": False,
            },
        )()

        with (
            patch(
                "avalan.tool.shell.executor.remove_tree",
                side_effect=OSError("/private/tmp/secret"),
            ),
            patch(
                "avalan.tool.shell.executor.inspect_path",
                return_value=metadata,
            ),
        ):
            await _cleanup_output_directory(Path("/private/tmp/secret"))

    async def test_generated_output_cleanup_ignores_unlink_errors(
        self,
    ) -> None:
        metadata = type(
            "Metadata",
            (),
            {
                "is_file": True,
                "is_symlink": False,
            },
        )()

        with (
            patch(
                "avalan.tool.shell.executor.remove_tree",
                side_effect=OSError("/private/tmp/secret"),
            ),
            patch(
                "avalan.tool.shell.executor.inspect_path",
                return_value=metadata,
            ),
            patch(
                "avalan.tool.shell.executor.remove_file",
                side_effect=OSError("/private/tmp/secret"),
            ),
        ):
            await _cleanup_output_directory(Path("/private/tmp/secret"))

    def test_posix_process_group_signal_uses_negative_pid(self) -> None:
        process = _SignalProcess(pid=123)
        signals: list[tuple[int, int]] = []

        def fake_kill(pid: int, signal_number: int) -> None:
            signals.append((pid, signal_number))

        with patch("avalan.tool.shell.executor.os_kill", new=fake_kill):
            _signal_process_group(process, 15, "terminate")

        self.assertEqual(signals, [(-123, 15)])
        self.assertEqual(process.terminate_count, 0)

    def test_posix_process_group_signal_falls_back_on_error(self) -> None:
        process = _SignalProcess(pid=123)

        def fake_kill(pid: int, signal_number: int) -> None:
            raise PermissionError("denied")

        with patch("avalan.tool.shell.executor.os_kill", new=fake_kill):
            _signal_process_group(process, 15, "terminate")

        self.assertEqual(process.terminate_count, 1)

    def test_process_group_signal_tolerates_missing_fallback(self) -> None:
        _signal_process_group(object(), 15, "terminate")

    async def test_wait_for_process_exit_handles_missing_wait(self) -> None:
        self.assertTrue(await _wait_for_process_exit(object()))

    async def test_wait_for_process_exit_tolerates_cleanup_error(self) -> None:
        self.assertTrue(await _wait_for_process_exit(_WaitErrorProcess()))

    async def test_reader_results_cancel_tasks_after_cleanup_timeout(
        self,
    ) -> None:
        stdout_reader = create_task(sleep(10))
        stderr_reader = create_task(sleep(10))

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            stdout, stderr = await _reader_results(
                stdout_reader,
                stderr_reader,
            )

        self.assertEqual(stdout, (b"", False))
        self.assertEqual(stderr, (b"", False))
        self.assertTrue(stdout_reader.cancelled())
        self.assertTrue(stderr_reader.cancelled())

    async def test_reader_results_accept_missing_reader_tasks(self) -> None:
        stdout, stderr = await _reader_results(None, None)

        self.assertEqual(stdout, (b"", False))
        self.assertEqual(stderr, (b"", False))

    async def test_reader_results_preserve_finished_stream_when_peer_hangs(
        self,
    ) -> None:
        async def completed_reader() -> tuple[bytes, bool]:
            return b"partial", True

        stdout_reader = create_task(completed_reader())
        stderr_reader = create_task(sleep(10))

        with patch(
            "avalan.tool.shell.executor._PROCESS_CLEANUP_GRACE_SECONDS",
            0.001,
        ):
            stdout, stderr = await _reader_results(
                stdout_reader,
                stderr_reader,
            )

        self.assertEqual(stdout, (b"partial", True))
        self.assertEqual(stderr, (b"", False))
        self.assertFalse(stdout_reader.cancelled())
        self.assertTrue(stderr_reader.cancelled())

    async def test_reader_results_tolerate_reader_failure_and_bad_shape(
        self,
    ) -> None:
        async def failed_reader() -> tuple[bytes, bool]:
            raise OSError("read failed")

        async def invalid_reader() -> object:
            return "invalid"

        stdout_reader = create_task(failed_reader())
        stderr_reader = create_task(invalid_reader())

        stdout, stderr = await _reader_results(stdout_reader, stderr_reader)

        self.assertEqual(stdout, (b"", False))
        self.assertEqual(stderr, (b"", False))

    async def test_reader_task_failed_ignores_missing_pending_and_cancelled(
        self,
    ) -> None:
        pending = create_task(sleep(10))
        cancelled = create_task(sleep(10))
        cancelled.cancel()
        try:
            with self.assertRaises(CancelledError):
                await cancelled

            self.assertFalse(_reader_task_failed(None))
            self.assertFalse(_reader_task_failed(pending))
            self.assertFalse(_reader_task_failed(cancelled))
        finally:
            pending.cancel()
            with self.assertRaises(CancelledError):
                await pending


class _FakeProcess:
    def __init__(
        self,
        *,
        returncode: int | None = 0,
        stdout: object = b"",
        stderr: object = b"",
    ) -> None:
        self.returncode = returncode
        self.stdout = (
            stdout if isinstance(stdout, _FakeStream) else _FakeStream(stdout)
        )
        self.stderr = (
            stderr if isinstance(stderr, _FakeStream) else _FakeStream(stderr)
        )
        self.stdin = _FakeStdin()
        self.wait_started = Event()
        self.spawn_args: tuple[object, ...] = ()
        self.spawn_kwargs: dict[str, object] = {}

    async def wait(self) -> None:
        self.wait_started.set()


class _TimeoutProcess:
    returncode = None
    stdout = None
    stderr = None
    stdin = None
    terminate_count = 0
    kill_count = 0
    wait_started = Event()

    async def wait(self) -> None:
        self.wait_started.set()
        await sleep(10)

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9


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
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)
        self.stdin = _FakeStdin()
        self.kill_count = 0
        self.terminate_count = 0

    async def wait(self) -> None:
        if self._tracker is not None:
            self._tracker.enter()
        try:
            await self._release.wait()
        finally:
            if self._tracker is not None:
                self._tracker.exit()

    def terminate(self) -> None:
        self.terminate_count += 1
        self._release.set()
        self.returncode = -15

    def kill(self) -> None:
        self.kill_count += 1
        self._release.set()
        self.returncode = -9


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
        self.wait_started = Event()
        self._terminate_releases = terminate_releases

    async def wait(self) -> None:
        self.wait_started.set()
        await self._release.wait()

    def terminate(self) -> None:
        self.terminate_count += 1
        self.returncode = -15
        if self._terminate_releases:
            self._release.set()


class _ReaderPendingTimeoutProcess(_TerminableProcess):
    def __init__(self, stdout_ready: Event) -> None:
        super().__init__(stdout=b"", stderr=b"")
        self._stdout_ready = stdout_ready

    async def wait(self) -> None:
        self.wait_started.set()
        await self._stdout_ready.wait()
        await self._release.wait()


class _FailingWaitExecutionProcess(_TerminableProcess):
    async def wait(self) -> None:
        self.wait_started.set()
        raise OSError("wait failed")


class _CleanupErrorProcess(_TerminableProcess):
    def terminate(self) -> None:
        self.terminate_count += 1
        raise ProcessLookupError("/private/tmp/secret")

    def kill(self) -> None:
        self.kill_count += 1
        raise OSError("/private/tmp/secret")


class _SignalProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.terminate_count = 0

    def terminate(self) -> None:
        self.terminate_count += 1


class _WaitErrorProcess:
    async def wait(self) -> None:
        raise OSError("cleanup failed")


class _GeneratedOutputProcess(_FakeProcess):
    def __init__(self, write_outputs: Callable[[], None]) -> None:
        super().__init__(stdout=b"", stderr=b"")
        self._write_outputs = write_outputs

    async def wait(self) -> None:
        self.wait_started.set()
        self._write_outputs()


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


class _CloseResetStdin(_FakeStdin):
    def close(self) -> None:
        self.closed = True
        raise ConnectionResetError


class _WaitClosedBrokenPipeStdin(_FakeStdin):
    async def wait_closed(self) -> None:
        raise BrokenPipeError


class _NoWaitClosedStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


class _NoCloseStdin:
    def __init__(self) -> None:
        self.data = b""

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        pass


class _BlockingStdin(_FakeStdin):
    def __init__(self, release: Event) -> None:
        super().__init__()
        self._release = release
        self.drain_started = Event()

    async def drain(self) -> None:
        self.drain_started.set()
        await self._release.wait()


class _HangingWaitClosedStdin(_BlockingStdin):
    def __init__(self) -> None:
        super().__init__(Event())
        self.wait_closed_started = Event()

    async def wait_closed(self) -> None:
        self.wait_closed_started.set()
        await sleep(10)


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


class _FailingStream(_FakeStream):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self._error = error

    async def read(self, size: int) -> bytes:
        self.read_calls += 1
        self.read_sizes.append(size)
        raise self._error


class _ControlledStream(_FakeStream):
    def __init__(
        self,
        *,
        chunks: tuple[bytes, ...],
        before_first_read: object,
    ) -> None:
        super().__init__()
        self._chunks = list(chunks)
        self._before_first_read = before_first_read

    async def read(self, size: int) -> bytes:
        self.read_calls += 1
        self.read_sizes.append(size)
        if self.read_calls == 1:
            await self._before_first_read()
        if not self._chunks:
            return b""
        chunk = self._chunks.pop(0)
        self.drained_bytes += len(chunk)
        return chunk


class _HangingAfterChunksStream(_FakeStream):
    def __init__(self, *, chunks: tuple[bytes, ...], exhausted: Event) -> None:
        super().__init__()
        self._chunks = list(chunks)
        self._exhausted = exhausted

    async def read(self, size: int) -> bytes:
        self.read_calls += 1
        self.read_sizes.append(size)
        if self._chunks:
            chunk = self._chunks.pop(0)
            self.drained_bytes += len(chunk)
            if not self._chunks:
                self._exhausted.set()
            return chunk
        await sleep(10)
        return b""


class _ResolvedResolver:
    def __init__(self, executable: str) -> None:
        self._executable = executable

    async def resolve(self, command: object) -> str:
        return self._executable


def _request(
    *,
    metadata: dict[str, object] | None = None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.rg",
        command="rg",
        options={"pattern": "needle"},
        paths=(),
        cwd=None,
        metadata={} if metadata is None else metadata,
    )


def _direct_spec(
    *,
    backend: Literal["local", "sandbox", "container"] = "local",
    executable: str,
    argv: tuple[str, ...],
    display_argv: tuple[str, ...] | None = None,
    command: str = "rg",
    tool_name: str = "shell.rg",
    stdin: bytes | None = None,
    resource_class: ShellResourceClass = "standard",
    timeout_seconds: float = 1.0,
    max_stdout_bytes: int = 10,
    max_stderr_bytes: int = 10,
    cwd: str | None = None,
    stdout_media_type: str = "text/plain",
    output_kind: ShellOutputKind = ShellOutputKind.TEXT,
    output_plan: GeneratedOutputPlan | None = None,
    env: dict[str, str] | None = None,
) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend=backend,
        tool_name=tool_name,
        command=command,
        executable=executable,
        argv=argv,
        display_argv=argv if display_argv is None else display_argv,
        cwd=str(Path.cwd().resolve()) if cwd is None else cwd,
        display_cwd=".",
        env={"LC_ALL": "C"} if env is None else env,
        stdin=stdin,
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        resource_class=resource_class,
        output_plan=output_plan,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _generated_spec(
    plan: GeneratedOutputPlan,
    *,
    cwd: str | None = None,
    timeout_seconds: float = 1.0,
) -> ExecutionSpec:
    return _direct_spec(
        executable="/trusted/bin/pdftoppm",
        argv=(
            "pdftoppm",
            "input.pdf",
            GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
        ),
        display_argv=("pdftoppm", "input.pdf", plan.display_prefix),
        command="pdftoppm",
        tool_name="shell.pdftoppm",
        cwd=cwd,
        stdout_media_type="application/json",
        output_kind=ShellOutputKind.GENERATED_FILES,
        output_plan=plan,
        resource_class="heavy",
        timeout_seconds=timeout_seconds,
    )


def _output_plan(
    root: Path,
    *,
    directory_name: str | None = None,
    prefix_name: str = "page",
    max_files: int = 8,
    max_file_bytes: int = 1024,
    max_total_bytes: int = 4096,
    max_inline_bytes: int = 1024,
    max_raster_long_edge_pixels: int | None = 128,
    max_raster_pixels: int | None = 16384,
) -> GeneratedOutputPlan:
    return GeneratedOutputPlan(
        prefix_name=prefix_name,
        display_prefix="GENERATED_PREFIX",
        allowed_suffixes=(".png",),
        suffix_media_types={".png": "image/png"},
        max_files=max_files,
        max_file_bytes=max_file_bytes,
        max_total_bytes=max_total_bytes,
        max_inline_bytes=max_inline_bytes,
        max_raster_long_edge_pixels=max_raster_long_edge_pixels,
        max_raster_pixels=max_raster_pixels,
    )


class _GeneratedOutputRuntimePlan:
    def __init__(
        self,
        plan: GeneratedOutputPlan,
        runtime_output_prefix: Path,
    ) -> None:
        self.prefix = str(runtime_output_prefix)
        self.prefix_name = plan.prefix_name
        self.display_prefix = plan.display_prefix


def _private_output_prefix(
    root: Path,
    *,
    directory_name: str | None = None,
    prefix_name: str = "page",
) -> Path:
    private_name = directory_name or (
        f"avalan-shell-{sha256(str(root).encode()).hexdigest()[:32]}"
    )
    return root.parent / private_name / prefix_name


async def _run_generated_output_case(
    write_outputs: Callable[[_GeneratedOutputRuntimePlan], object],
    **plan_kwargs: Any,
) -> tuple[ExecutionResult, Path]:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        plan = _output_plan(root, **plan_kwargs)

        def write_plan_outputs() -> None:
            write_outputs(
                _GeneratedOutputRuntimePlan(
                    plan,
                    _spawned_output_prefix(process),
                )
            )

        spec = _generated_spec(plan)
        process = _GeneratedOutputProcess(write_plan_outputs)

        with patch(
            "avalan.tool.shell.executor.create_subprocess_exec",
            new=_fake_process_factory(process),
        ):
            result = await LocalCommandExecutor().execute(spec)
        output_directory = _spawned_output_prefix(process).parent

    return result, output_directory


def _spawned_output_prefix(process: _FakeProcess) -> Path:
    runtime_prefix = process.spawn_args[-1]
    assert isinstance(runtime_prefix, str), "runtime prefix must be a string"
    return Path(runtime_prefix)


def _png_bytes(*, width: int = 1, height: int = 1) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def _fake_process_factory(
    process: _FakeProcess,
    *,
    after_spawn: Callable[[], None] | None = None,
) -> object:
    async def fake_create_subprocess_exec(
        *args: object,
        **kwargs: object,
    ) -> _FakeProcess:
        process.spawn_args = args
        process.spawn_kwargs = kwargs
        if after_spawn is not None:
            after_spawn()
        return process

    return fake_create_subprocess_exec


async def _spawn_should_not_run(
    *args: object,
    **kwargs: object,
) -> _FakeProcess:
    raise AssertionError("spawn should not run")


if __name__ == "__main__":
    main()
