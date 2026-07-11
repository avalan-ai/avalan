from os import getpid, getppid
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.tool.shell import (
    ExecutableResolver,
    ExecutionPolicy,
    PathOperand,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionRequest,
    ShellExecutionErrorCode,
    ShellPolicyDenied,
    ShellToolSettings,
)
from avalan.tool.shell.kill import REDACTED_KILL_STDERR, redacted_stderr

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class KillPolicyTest(IsolatedAsyncioTestCase):
    async def test_normalizes_exact_signal_argv(self) -> None:
        for signal in ("TERM", "INT", "KILL"):
            with self.subTest(signal=signal):
                resolver = _RecordingResolver("/trusted/bin/kill")
                spec = await _policy(resolver=resolver).normalize(
                    _request(42, signal)
                )

                expected = ("kill", "-s", signal, "--", "42")
                self.assertEqual(spec.argv, expected)
                self.assertEqual(spec.display_argv, expected)
                self.assertEqual(resolver.commands, ["kill"])

    async def test_rejects_invalid_pids_and_signals_before_resolution(
        self,
    ) -> None:
        invalid = (
            (None, "TERM"),
            (True, "TERM"),
            (0, "TERM"),
            (1, "TERM"),
            (-1, "TERM"),
            (getpid(), "TERM"),
            (getppid(), "TERM"),
            (2**31, "TERM"),
            (42, ""),
            (42, "term"),
            (42, "9"),
            (42, "HUP"),
            (42, None),
        )
        for pid, signal in invalid:
            with self.subTest(pid=pid, signal=signal):
                resolver = _RecordingResolver("/trusted/bin/kill")
                with self.assertRaises(ShellPolicyDenied):
                    await _policy(resolver=resolver).normalize(
                        _request(pid, signal)
                    )
                self.assertEqual(resolver.commands, [])

    async def test_nonlocal_backends_fail_closed_before_resolution(
        self,
    ) -> None:
        for execution_mode in ("sandbox", "container"):
            with self.subTest(execution_mode=execution_mode):
                resolver = _RecordingResolver("/trusted/bin/kill")
                settings = ShellToolSettings(
                    execution_mode=execution_mode,
                    workspace_root=str(FIXTURE_ROOT),
                    allow_process_tools=True,
                    allow_process_control=True,
                )
                with self.assertRaises(ShellPolicyDenied) as raised:
                    await ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ).normalize(_request(42, "TERM"))
                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )
                self.assertEqual(resolver.commands, [])

    async def test_requires_both_process_opt_ins_before_resolution(
        self,
    ) -> None:
        for allow_process_tools, allow_process_control in (
            (False, False),
            (True, False),
            (False, True),
        ):
            with self.subTest(
                allow_process_tools=allow_process_tools,
                allow_process_control=allow_process_control,
            ):
                resolver = _RecordingResolver("/trusted/bin/kill")
                settings = ShellToolSettings(
                    workspace_root=str(FIXTURE_ROOT),
                    allow_process_tools=allow_process_tools,
                    allow_process_control=allow_process_control,
                )
                with self.assertRaises(ShellPolicyDenied):
                    await ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ).normalize(_request(42, "TERM"))
                self.assertEqual(resolver.commands, [])

    async def test_rejects_unknown_options_paths_and_compositions(
        self,
    ) -> None:
        with self.assertRaises(ShellPolicyDenied):
            await _policy().normalize(
                ShellCommandRequest(
                    tool_name="shell.kill",
                    command="kill",
                    options={"pid": 42, "signal": "TERM", "force": True},
                    paths=(),
                    cwd=None,
                )
            )
        with self.assertRaises(ShellPolicyDenied):
            await _policy().normalize(
                ShellCommandRequest(
                    tool_name="shell.kill",
                    command="kill",
                    options={"pid": 42, "signal": "TERM"},
                    paths=(
                        PathOperand(
                            name="input",
                            path="filesystem/visible.txt",
                            kind="text_file",
                            access="read",
                        ),
                    ),
                    cwd=None,
                )
            )
        for mode in ("serial", "parallel", "pipeline"):
            with self.subTest(mode=mode):
                with self.assertRaises(ShellPolicyDenied) as raised:
                    await ExecutionPolicy(
                        settings=ShellToolSettings(
                            workspace_root=str(FIXTURE_ROOT),
                            allow_pipelines=True,
                            allow_process_tools=True,
                            allow_process_control=True,
                        )
                    ).normalize_composition(
                        ShellCompositionRequest(
                            mode=mode,
                            steps=(
                                ShellCommandStepRequest(
                                    id="process",
                                    command="kill",
                                    options={"pid": 42, "signal": "TERM"},
                                ),
                            ),
                        )
                    )
                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )


class KillOutputTest(TestCase):
    def test_redacts_diagnostics(self) -> None:
        self.assertEqual(redacted_stderr(""), "")
        self.assertEqual(
            redacted_stderr("private process details"),
            REDACTED_KILL_STDERR,
        )
        with self.assertRaises(AssertionError):
            redacted_stderr(object())  # type: ignore[arg-type]


def _request(pid: object, signal: object) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.kill",
        command="kill",
        options={"pid": pid, "signal": signal},
        paths=(),
        cwd=None,
    )


def _policy(resolver: ExecutableResolver | None = None) -> ExecutionPolicy:
    settings = ShellToolSettings(
        workspace_root=str(FIXTURE_ROOT),
        allow_process_tools=True,
        allow_process_control=True,
    )
    return ExecutionPolicy(settings=settings, resolver=resolver)


class _RecordingResolver:
    def __init__(self, executable: str) -> None:
        self.executable = executable
        self.commands: list[str] = []

    async def resolve(self, command: object) -> str | None:
        logical_id = getattr(command, "logical_id")
        self.commands.append(logical_id)
        return self.executable


if __name__ == "__main__":
    main()
