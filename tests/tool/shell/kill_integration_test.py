from asyncio import create_subprocess_exec, wait_for
from asyncio.subprocess import PIPE
from pathlib import Path
from shutil import which
from sys import executable as python_executable
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import ToolCallContext
from avalan.tool.shell import (
    ExecutionPolicy,
    LocalCommandExecutor,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSettings,
)
from avalan.tool.shell.kill import REDACTED_KILL_STDERR
from avalan.tool.shell.tools import KillTool


class KillLocalIntegrationTest(IsolatedAsyncioTestCase):
    async def test_signals_owned_ready_child_and_observes_exit(self) -> None:
        kill_executable = which("kill")
        if kill_executable is None:
            self.skipTest("kill is unavailable")
        process = await create_subprocess_exec(
            python_executable,
            "-c",
            "import sys,time; print('ready', flush=True); time.sleep(60)",
            stdout=PIPE,
        )
        try:
            assert process.stdout is not None
            self.assertEqual(
                await wait_for(process.stdout.readline(), 2), b"ready\n"
            )
            with TemporaryDirectory() as temporary_directory:
                output = await _tool(
                    Path(temporary_directory),
                    Path(kill_executable),
                )(
                    pid=process.pid,
                    context=ToolCallContext(),
                )

            assert isinstance(output, ShellFormattedResult)
            self.assertIs(
                output.execution_result.status,
                ShellExecutionStatus.COMPLETED,
            )
            self.assertLess(await wait_for(process.wait(), 2), 0)
        finally:
            if process.returncode is None:
                process.terminate()
                await process.wait()

    async def test_nonlocal_execution_never_launches_binary(self) -> None:
        for execution_mode in ("sandbox", "container"):
            with self.subTest(execution_mode=execution_mode):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_kill(root)
                    settings = ShellToolSettings(
                        execution_mode=execution_mode,
                        workspace_root=str(root),
                        allow_process_tools=True,
                        allow_process_control=True,
                        executable_paths={"kill": str(executable)},
                    )
                    output = await KillTool(
                        settings=settings,
                        policy=ExecutionPolicy(settings=settings),
                        executor=LocalCommandExecutor(settings=settings),
                    )(
                        pid=42,
                        context=ToolCallContext(),
                    )

                    self.assertFalse(marker.exists())
                    assert isinstance(output, ShellFormattedResult)
                    self.assertIs(
                        output.execution_result.status,
                        ShellExecutionStatus.POLICY_DENIED,
                    )

    async def test_executes_typed_argv_and_discards_process_output(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_kill(root)
            tool = _tool(root, executable)

            output = await tool(
                pid=42,
                signal="INT",
                context=ToolCallContext(),
            )

            self.assertIsInstance(output, ShellFormattedResult)
            assert isinstance(output, ShellFormattedResult)
            result = output.execution_result
            self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
            self.assertEqual(result.argv, ("kill", "-s", "INT", "--", "42"))
            self.assertEqual(result.stdout, "")
            self.assertEqual(result.stderr, REDACTED_KILL_STDERR)
            self.assertEqual(result.metadata, {})
            self.assertEqual(
                marker.read_text(encoding="utf-8"), "-s\nINT\n--\n42\n"
            )

    async def test_nonzero_exit_is_generic_and_redacted(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_kill(root)
            output = await _tool(root, executable)(
                pid=99,
                context=ToolCallContext(),
            )

        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIs(result.status, ShellExecutionStatus.NONZERO_EXIT)
        self.assertEqual(result.error_message, "kill exited non-zero")
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, REDACTED_KILL_STDERR)


def _tool(root: Path, executable: Path) -> KillTool:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_process_tools=True,
        allow_process_control=True,
        executable_paths={"kill": str(executable)},
    )
    return KillTool(
        settings=settings,
        policy=ExecutionPolicy(settings=settings),
        executor=LocalCommandExecutor(settings=settings),
    )


def _fake_kill(root: Path) -> tuple[Path, Path]:
    executable = root / "kill"
    marker = Path(f"{executable}.args")
    executable.write_text(
        '#!/bin/sh\nprintf \'%s\\n\' "$@" > "$0.args"\n'
        "printf 'private stdout\\n'\n"
        "printf 'private process details\\n' >&2\n"
        '[ "$4" != 99 ]\n',
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


if __name__ == "__main__":
    main()
