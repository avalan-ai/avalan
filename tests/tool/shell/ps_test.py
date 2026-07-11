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
from avalan.tool.shell.ps import (
    PS_MAX_PID,
    REDACTED_PS_STDERR,
    process_rows_stdout,
    redacted_stderr,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class PsPolicyTest(IsolatedAsyncioTestCase):
    async def test_normalizes_exact_fixed_format_argv(self) -> None:
        resolver = _RecordingResolver("/trusted/bin/ps")
        spec = await _policy(resolver=resolver).normalize(_request((42,)))

        expected = (
            "ps",
            "-p",
            "42",
            "-o",
            "pid=",
            "-o",
            "ppid=",
            "-o",
            "state=",
            "-o",
            "etime=",
            "-o",
            "comm=",
        )
        self.assertEqual(spec.argv, expected)
        self.assertEqual(spec.display_argv, expected)
        self.assertEqual(
            spec.metadata["exit_code_statuses"],
            {1: "no_matches"},
        )
        self.assertEqual(resolver.commands, ["ps"])

    async def test_rejects_invalid_pid_sets_before_resolution(self) -> None:
        invalid = (
            None,
            (),
            "42",
            (0,),
            (-1,),
            (True,),
            (PS_MAX_PID + 1,),
            (42, 42),
            (42, 43),
        )
        for pids in invalid:
            with self.subTest(pids=pids):
                resolver = _RecordingResolver("/trusted/bin/ps")
                with self.assertRaises(ShellPolicyDenied):
                    await _policy(resolver=resolver).normalize(
                        ShellCommandRequest(
                            tool_name="shell.ps",
                            command="ps",
                            options={"pids": pids},
                            paths=(),
                            cwd=None,
                        )
                    )
                self.assertEqual(resolver.commands, [])

    async def test_rejects_options_paths_gate_and_composition(self) -> None:
        with self.assertRaises(ShellPolicyDenied):
            await _policy().normalize(
                ShellCommandRequest(
                    tool_name="shell.ps",
                    command="ps",
                    options={"pids": (1,), "all": True},
                    paths=(),
                    cwd=None,
                )
            )
        with self.assertRaises(ShellPolicyDenied):
            await _policy().normalize(
                ShellCommandRequest(
                    tool_name="shell.ps",
                    command="ps",
                    options={"pids": (1,)},
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
                settings = ShellToolSettings(
                    workspace_root=str(FIXTURE_ROOT),
                    allow_pipelines=True,
                    allow_process_tools=True,
                )
                with self.assertRaises(ShellPolicyDenied) as raised:
                    await ExecutionPolicy(
                        settings=settings
                    ).normalize_composition(
                        ShellCompositionRequest(
                            mode=mode,
                            steps=(
                                ShellCommandStepRequest(
                                    id="process",
                                    command="ps",
                                    options={"pids": (1,)},
                                ),
                            ),
                        )
                    )
                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )

    async def test_process_gate_denies_before_resolution(self) -> None:
        resolver = _RecordingResolver("/trusted/bin/ps")
        settings = ShellToolSettings(workspace_root=str(FIXTURE_ROOT))
        with self.assertRaises(ShellPolicyDenied):
            policy = ExecutionPolicy(settings=settings, resolver=resolver)
            await policy.normalize(_request((1,)))
        self.assertEqual(resolver.commands, [])


class PsOutputTest(TestCase):
    def test_returns_canonical_rows_and_discards_truncated_tail(self) -> None:
        raw = " 42 1 S 01:02 /usr/bin/worker\n"
        self.assertEqual(
            process_rows_stdout(raw, requested_pids=(42,)),
            "42 1 S 01:02 /usr/bin/worker\n",
        )
        self.assertEqual(
            process_rows_stdout(
                "42 1 S 01:02 worker\n43 1 R 00:0",
                requested_pids=(42,),
                stdout_truncated=True,
            ),
            "42 1 S 01:02 worker\n",
        )
        self.assertEqual(redacted_stderr(""), "")
        self.assertEqual(
            redacted_stderr("private details"),
            REDACTED_PS_STDERR,
        )

    def test_rejects_unrequested_duplicate_and_forged_rows(self) -> None:
        malformed_values = (
            "PID PPID STATE ELAPSED COMMAND\n",
            "42 1 S 99:99 invalid-elapsed\n",
            "42 1 S 00:01 bad\x1bcommand\n",
            "2147483648 1 S 00:01 too-large\n",
            "42 1 S 00:01\n42 1 R 00:02 forged\n",
        )
        for value in malformed_values:
            with self.subTest(value=value):
                self.assertEqual(
                    process_rows_stdout(value, requested_pids=(42,)),
                    "",
                )
        self.assertEqual(
            process_rows_stdout(
                "42 1 S 00:01 worker\n99 1 S 00:01 forged\n",
                requested_pids=(42,),
            ),
            "",
        )

    def test_rejects_every_invalid_canonical_field_shape(self) -> None:
        invalid_values = (
            "\n",
            "42 1 1 00:01 worker\n",
            f"42 1 S 00:01 {'x' * 4097}\n",
            "42 1 S invalid worker\n",
            "42 1 S 24:00:00 worker\n",
        )
        for value in invalid_values:
            with self.subTest(value=value[:80]):
                self.assertEqual(
                    process_rows_stdout(value, requested_pids=(42,)),
                    "",
                )
        self.assertEqual(
            process_rows_stdout(
                "42 1 S 00:01 worker\n42 1 R 00:02 forged\n",
                requested_pids=(42,),
            ),
            "",
        )
        self.assertEqual(
            process_rows_stdout(
                "42 1 S 00:01 worker\n43 1 R 00:02 worker\n",
                requested_pids=(42, 43),
            ),
            "",
        )


def _request(pids: object) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.ps",
        command="ps",
        options={"pids": pids},
        paths=(),
        cwd=None,
    )


def _policy(resolver: ExecutableResolver | None = None) -> ExecutionPolicy:
    settings = ShellToolSettings(
        workspace_root=str(FIXTURE_ROOT),
        allow_process_tools=True,
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
