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
        self.assertEqual(spec.metadata["_ps_view"], "summary")
        self.assertEqual(resolver.commands, ["ps"])

    async def test_normalizes_exact_fixed_resource_argv(self) -> None:
        resolver = _RecordingResolver("/trusted/bin/ps")
        spec = await _policy(resolver=resolver).normalize(
            _request((42,), view="resources")
        )

        expected = (
            "ps",
            "-p",
            "42",
            "-o",
            "pid=",
            "-o",
            "pcpu=",
            "-o",
            "pmem=",
            "-o",
            "rss=",
            "-o",
            "vsz=",
            "-o",
            "time=",
            "-o",
            "nice=",
        )
        self.assertEqual(spec.argv, expected)
        self.assertEqual(spec.display_argv, expected)
        self.assertEqual(spec.metadata["_ps_view"], "resources")
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

    async def test_rejects_invalid_views_before_resolution(self) -> None:
        for view in (None, True, "", "resource", "SUMMARY", 1):
            with self.subTest(view=view):
                resolver = _RecordingResolver("/trusted/bin/ps")
                with self.assertRaises(ShellPolicyDenied) as raised:
                    await _policy(resolver=resolver).normalize(
                        _request((42,), view=view)
                    )
                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.INVALID_OPTION,
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
            f"{'9' * 5000} 1 S 00:01 worker\n",
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

    def test_returns_canonical_resource_rows_for_portable_time_forms(
        self,
    ) -> None:
        values = (
            "42 0.0 1.5 1024 4096 0:00.02 0\n",
            "42 100.1 100.0 0 9223372036854775807 01:02:03 -20\n",
            "42 250.0 2.0 2048 8192 12-03:04:05 19\n",
            "42 0.1 0.2 3 4 123:45.67 20\n",
            "42 1000000000.0 101.0 1 2 23:00:00 1",
        )
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    process_rows_stdout(
                        f"  {value}",
                        requested_pids=(42,),
                        view="resources",
                    ),
                    value,
                )

    def test_rejects_unsafe_resource_numbers_and_times(self) -> None:
        invalid_fields = (
            ("NaN", "1.0", "1", "2", "0:00.00", "0"),
            ("inf", "1.0", "1", "2", "0:00.00", "0"),
            ("1e2", "1.0", "1", "2", "0:00.00", "0"),
            ("1,5", "1.0", "1", "2", "0:00.00", "0"),
            ("-1.0", "1.0", "1", "2", "0:00.00", "0"),
            ("1", "1.0", "1", "2", "0:00.00", "0"),
            ("1.00", "1.0", "1", "2", "0:00.00", "0"),
            ("10000000000.0", "1.0", "1", "2", "0:00.00", "0"),
            ("000000000.0", "1.0", "1", "2", "0:00.00", "0"),
            ("1.0", "100.001", "1", "2", "0:00.00", "0"),
            ("1.0", "101", "1", "2", "0:00.00", "0"),
            ("1.0", "-1.0", "1", "2", "0:00.00", "0"),
            ("1.0", "000000000.0", "1", "2", "0:00.00", "0"),
            ("1.0", "1.0", "+1", "2", "0:00.00", "0"),
            ("1.0", "1.0", "01", "2", "0:00.00", "0"),
            (
                "1.0",
                "1.0",
                "9223372036854775808",
                "2",
                "0:00.00",
                "0",
            ),
            ("1.0", "1.0", "1", "-2", "0:00.00", "0"),
            ("1.0", "1.0", "1", "02", "0:00.00", "0"),
            (
                "1.0",
                "1.0",
                "1",
                "9223372036854775808",
                "0:00.00",
                "0",
            ),
            ("1.0", "1.0", "1", "2", "1-00:00", "0"),
            ("1.0", "1.0", "1", "2", "00:00.00", "0"),
            ("1.0", "1.0", "1", "2", "0:00.0", "0"),
            ("1.0", "1.0", "1", "2", "0:00.000", "0"),
            ("1.0", "1.0", "1", "2", "0:60.00", "0"),
            ("1.0", "1.0", "1", "2", "00:00:60", "0"),
            ("1.0", "1.0", "1", "2", "1-24:00:00", "0"),
            ("1.0", "1.0", "1", "2", "12345678901-00:00:00", "0"),
            ("1.0", "1.0", "1", "2", "2147483648:00:00", "0"),
            ("1.0", "1.0", "1", "2", f"{'1' * 41}:00.00", "0"),
            ("1.0", "1.0", "1", "2", "0:00.00", "-21"),
            ("1.0", "1.0", "1", "2", "0:00.00", "21"),
            ("1.0", "1.0", "1", "2", "0:00.00", "00"),
            ("1.0", "1.0", "1", "2", "0:00.00", "-0"),
            ("1.0", "1.0", "1", "2", "0:00.00", "+1"),
            ("1.0", "1.0", "1", "2", "0:00.00", "-"),
            ("1.0", "1.0", "1", "2", "0:00.00", "1000"),
        )
        for fields in invalid_fields:
            with self.subTest(fields=fields):
                cpu, memory, rss, virtual_size, cpu_time, nice = fields
                value = (
                    f"42 {cpu} {memory} {rss} {virtual_size} "
                    f"{cpu_time} {nice}\n"
                )
                self.assertEqual(
                    process_rows_stdout(
                        value,
                        requested_pids=(42,),
                        view="resources",
                    ),
                    "",
                )

    def test_rejects_malformed_forged_and_truncated_resource_rows(
        self,
    ) -> None:
        invalid_values = (
            "PID CPU MEM RSS VSZ TIME NICE\n",
            "42 1.0 2.0 3 4 0:00.00\n",
            "42 1.0 2.0 3 4 0:00.00 0 extra\n",
            "42 1.0 2.0 3 4 0:00.00 x\n",
            "42 1.0 2.0 3 4 0:00.00 0\u00a0\n",
            "42 1.0 2.0 3 4 0:00.00 0\x1b\n",
            f"{'9' * 5000} 1.0 2.0 3 4 0:00.00 0\n",
            "43 1.0 2.0 3 4 0:00.00 0\n",
            "42 1.0 2.0 3 4 0:00.00 0\n42 1.0 2.0 3 4 0:01.00 0\n",
            "42 1.0 2.0 3 4 0:00.00 0\n43 1.0 2.0 3 4 0:00.00 0\n",
        )
        for value in invalid_values:
            with self.subTest(value=value[:80]):
                self.assertEqual(
                    process_rows_stdout(
                        value,
                        requested_pids=(42,),
                        view="resources",
                    ),
                    "",
                )
        self.assertEqual(
            process_rows_stdout(
                "42 1.0 2.0 3 4 0:00.00 0\nsecret",
                requested_pids=(42,),
                view="resources",
                stdout_truncated=True,
            ),
            "42 1.0 2.0 3 4 0:00.00 0\n",
        )
        with self.assertRaises(AssertionError):
            process_rows_stdout(
                "",
                requested_pids=(42,),
                view="invalid",  # type: ignore[arg-type]
            )
        self.assertEqual(
            process_rows_stdout(
                "42 1 S 00:01 worker\n43 1 R 00:02 worker\n",
                requested_pids=(42, 43),
            ),
            "",
        )


def _request(
    pids: object,
    *,
    view: object = "summary",
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.ps",
        command="ps",
        options={"pids": pids, "view": view},
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
