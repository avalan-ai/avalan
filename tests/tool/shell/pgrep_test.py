from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
from avalan.tool.shell import (
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellCommandStepRequest,
    ShellCompositionRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSettings,
)
from avalan.tool.shell.pgrep import (
    REDACTED_PGREP_PATTERN,
    REDACTED_PGREP_STDERR,
    pid_only_stdout,
    redacted_stderr,
)
from avalan.tool.shell.tools import PgrepTool

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
PRIVATE_PATTERN = "private-worker-pattern"


class PgrepPolicyTest(IsolatedAsyncioTestCase):
    async def test_normalizes_default_and_structured_flag_argv(self) -> None:
        cases: tuple[tuple[dict[str, object], tuple[str, ...]], ...] = (
            (
                {"pattern": PRIVATE_PATTERN},
                ("pgrep", "--", PRIVATE_PATTERN),
            ),
            (
                {
                    "pattern": PRIVATE_PATTERN,
                    "full": True,
                    "exact": True,
                    "ignore_case": True,
                    "newest": True,
                    "parent_pid": 42,
                },
                (
                    "pgrep",
                    "-f",
                    "-x",
                    "-i",
                    "-n",
                    "-P",
                    "42",
                    "--",
                    PRIVATE_PATTERN,
                ),
            ),
            (
                {"pattern": "-worker", "oldest": True},
                ("pgrep", "-o", "--", "-worker"),
            ),
        )
        for options, expected_argv in cases:
            with self.subTest(options=options):
                resolver = _RecordingResolver("/trusted/bin/pgrep")
                spec = await _policy(resolver=resolver).normalize(
                    _request(options)
                )

                self.assertEqual(spec.argv, expected_argv)
                self.assertEqual(
                    spec.display_argv,
                    (*expected_argv[:-1], REDACTED_PGREP_PATTERN),
                )
                self.assertEqual(spec.executable, "/trusted/bin/pgrep")
                self.assertEqual(
                    spec.metadata["exit_code_statuses"],
                    {1: "no_matches"},
                )
                self.assertEqual(resolver.commands, ["pgrep"])

    async def test_accepts_numeric_redaction_and_unicode_patterns(
        self,
    ) -> None:
        for pattern in ("1234", "0012", "[redacted]", "café-worker"):
            with self.subTest(pattern=pattern):
                spec = await _policy().normalize(
                    _request({"pattern": pattern})
                )

                self.assertEqual(spec.argv[-1], pattern)
                self.assertEqual(
                    spec.display_argv[-1],
                    REDACTED_PGREP_PATTERN,
                )

    async def test_rejects_invalid_patterns_before_resolution(self) -> None:
        invalid_patterns: tuple[object, ...] = (
            None,
            "",
            "   ",
            7,
            "worker\x00name",
            "worker\nname",
            "worker\ud800name",
        )
        for pattern in invalid_patterns:
            with self.subTest(pattern=pattern):
                resolver = _RecordingResolver("/trusted/bin/pgrep")

                with self.assertRaises(ShellPolicyDenied):
                    await _policy(resolver=resolver).normalize(
                        _request({"pattern": pattern})
                    )

                self.assertEqual(resolver.commands, [])

        resolver = _RecordingResolver("/trusted/bin/pgrep")
        settings = ShellToolSettings(
            workspace_root=str(FIXTURE_ROOT),
            allow_process_tools=True,
            max_filter_pattern_bytes=4,
        )
        with self.assertRaises(ShellPolicyDenied) as raised:
            await ExecutionPolicy(
                settings=settings,
                resolver=resolver,
            ).normalize(_request({"pattern": "café"}))

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
        )
        self.assertEqual(resolver.commands, [])

    async def test_rejects_invalid_options_and_paths(self) -> None:
        invalid_options: tuple[dict[str, object], ...] = (
            {"pattern": "worker", "unknown": True},
            {"pattern": "worker", "full": 1},
            {"pattern": "worker", "newest": True, "oldest": True},
            {"pattern": "worker", "parent_pid": 0},
            {"pattern": "worker", "parent_pid": 2**31},
            {"pattern": "worker", "parent_pid": "42"},
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ShellPolicyDenied):
                    await _policy().normalize(_request(options))

        resolver = _RecordingResolver("/trusted/bin/pgrep")
        with self.assertRaises(ShellPolicyDenied):
            await _policy(resolver=resolver).normalize(
                ShellCommandRequest(
                    tool_name="shell.pgrep",
                    command="pgrep",
                    options={"pattern": "worker"},
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
        self.assertEqual(resolver.commands, [])

    async def test_process_gate_denies_before_resolution(self) -> None:
        resolver = _RecordingResolver("/trusted/bin/pgrep")
        settings = ShellToolSettings(workspace_root=str(FIXTURE_ROOT))

        with self.assertRaises(ShellPolicyDenied) as raised:
            await ExecutionPolicy(
                settings=settings,
                resolver=resolver,
            ).normalize(_request({"pattern": PRIVATE_PATTERN}))

        self.assertIs(
            raised.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )
        self.assertEqual(resolver.commands, [])

    async def test_pgrep_is_denied_in_every_composition_mode(self) -> None:
        for mode in ("serial", "parallel", "pipeline"):
            with self.subTest(mode=mode):
                resolver = _RecordingResolver("/trusted/bin/pgrep")
                settings = ShellToolSettings(
                    workspace_root=str(FIXTURE_ROOT),
                    allow_pipelines=True,
                    allow_process_tools=True,
                )
                request = ShellCompositionRequest(
                    mode=mode,
                    steps=(
                        ShellCommandStepRequest(
                            id="processes",
                            command="pgrep",
                            options={"pattern": PRIVATE_PATTERN},
                        ),
                    ),
                )

                with self.assertRaises(ShellPolicyDenied) as raised:
                    await ExecutionPolicy(
                        settings=settings,
                        resolver=resolver,
                    ).normalize_composition(request)

                self.assertIs(
                    raised.exception.error_code,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )
                self.assertEqual(resolver.commands, [])


class PgrepOutputTest(TestCase):
    def test_filters_pid_output_and_redacts_diagnostics(self) -> None:
        value = (
            "4242\n"
            "worker process name\n"
            f"{PRIVATE_PATTERN}\n"
            "0012\n"
            "2147483648\n"
            "99999999999\n"
            "14242\n"
        )

        self.assertEqual(pid_only_stdout(value), "4242\n14242\n")
        self.assertEqual(
            pid_only_stdout("4242\n142", stdout_truncated=True),
            "4242\n",
        )
        self.assertEqual(redacted_stderr(""), "")
        self.assertEqual(
            redacted_stderr(f"diagnostic {PRIVATE_PATTERN}"),
            REDACTED_PGREP_STDERR,
        )


class PgrepToolTest(IsolatedAsyncioTestCase):
    async def test_schema_is_structured_bounded_and_non_streaming(
        self,
    ) -> None:
        settings = ShellToolSettings(
            workspace_root=str(FIXTURE_ROOT),
            allow_process_tools=True,
        )
        tool = PgrepTool(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=_RecordingResolver("/trusted/bin/pgrep"),
            ),
            executor=_RecordingExecutor(PRIVATE_PATTERN),
        )

        schema = tool.json_schema()["function"]["parameters"]
        properties = schema["properties"]

        self.assertEqual(schema["required"], ["pattern"])
        self.assertEqual(properties["pattern"]["minLength"], 1)
        self.assertEqual(properties["parent_pid"]["minimum"], 1)
        self.assertEqual(properties["parent_pid"]["maximum"], 2**31 - 1)
        self.assertEqual(
            set(properties),
            {
                "pattern",
                "full",
                "exact",
                "ignore_case",
                "newest",
                "oldest",
                "parent_pid",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertFalse(tool.supports_streaming)

    async def test_model_boundary_sanitizes_typed_executor_result(
        self,
    ) -> None:
        settings = ShellToolSettings(
            workspace_root=str(FIXTURE_ROOT),
            allow_process_tools=True,
        )
        executor = _RecordingExecutor(PRIVATE_PATTERN)
        tool = PgrepTool(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=_RecordingResolver("/trusted/bin/pgrep"),
            ),
            executor=executor,
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        output = await tool(
            PRIVATE_PATTERN,
            context=ToolCallContext(stream_event=record),
        )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertIsNotNone(executor.spec)
        assert executor.spec is not None
        self.assertEqual(executor.spec.argv[-1], PRIVATE_PATTERN)
        self.assertEqual(result.stdout, "4242\n")
        self.assertEqual(result.stderr, REDACTED_PGREP_STDERR)
        self.assertEqual(
            result.argv,
            ("pgrep", "--", REDACTED_PGREP_PATTERN),
        )
        self.assertEqual(result.display_argv, result.argv)
        self.assertEqual(result.stdout_bytes, 5)
        self.assertEqual(
            result.stderr_bytes,
            len(REDACTED_PGREP_STDERR.encode("utf-8")),
        )
        self.assertEqual(events, [])
        self.assertNotIn(PRIVATE_PATTERN, output)
        self.assertNotIn("worker process name", output)

    async def test_model_boundary_preserves_absent_error_message(self) -> None:
        settings = ShellToolSettings(
            workspace_root=str(FIXTURE_ROOT),
            allow_process_tools=True,
        )
        tool = PgrepTool(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=_RecordingResolver("/trusted/bin/pgrep"),
            ),
            executor=_RecordingExecutor(
                PRIVATE_PATTERN,
                include_error_message=False,
            ),
        )

        output = await tool(
            PRIVATE_PATTERN,
            context=ToolCallContext(),
        )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        self.assertIsNone(output.execution_result.error_message)


def _policy(
    *,
    resolver: "_RecordingResolver | None" = None,
) -> ExecutionPolicy:
    settings = ShellToolSettings(
        workspace_root=str(FIXTURE_ROOT),
        allow_process_tools=True,
    )
    return ExecutionPolicy(
        settings=settings,
        resolver=resolver or _RecordingResolver("/trusted/bin/pgrep"),
    )


def _request(options: dict[str, object]) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.pgrep",
        command="pgrep",
        options=options,
        paths=(),
        cwd=None,
    )


class _RecordingResolver:
    def __init__(self, executable: str | None) -> None:
        self.executable = executable
        self.commands: list[str] = []

    async def resolve(
        self,
        command: ShellCommandDefinition,
    ) -> str | None:
        self.commands.append(command.logical_id)
        return self.executable


class _RecordingExecutor:
    def __init__(
        self,
        pattern: str,
        *,
        include_error_message: bool = True,
    ) -> None:
        self.pattern = pattern
        self.include_error_message = include_error_message
        self.spec: ExecutionSpec | None = None

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ExecutionResult:
        self.spec = spec
        stdout = f"4242\nworker process name\n{self.pattern}\n"
        stderr = f"diagnostic {self.pattern}"
        return ExecutionResult(
            backend="custom",
            tool_name="custom.pgrep",
            command="pgrep",
            argv=spec.argv,
            display_argv=spec.argv,
            cwd=spec.cwd,
            display_cwd=spec.display_cwd,
            status=ShellExecutionStatus.NONZERO_EXIT,
            exit_code=2,
            stdout=stdout,
            stderr=stderr,
            stdout_media_type=spec.stdout_media_type,
            output_kind=ShellOutputKind.TEXT,
            stdout_bytes=len(stdout.encode("utf-8")),
            stderr_bytes=len(stderr.encode("utf-8")),
            duration_ms=1,
            error_code=ShellExecutionErrorCode.NONZERO_EXIT,
            error_message=(
                f"private error {self.pattern}"
                if self.include_error_message
                else None
            ),
            metadata={"private": self.pattern},
        )


if __name__ == "__main__":
    main()
