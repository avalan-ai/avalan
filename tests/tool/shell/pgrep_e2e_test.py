from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolExecutionStreamEvent,
    ToolManagerSettings,
)
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    ExecutionPolicy,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSet,
    ShellToolSettings,
)
from avalan.tool.shell.pgrep import (
    REDACTED_PGREP_PATTERN,
    REDACTED_PGREP_STDERR,
)

PRIVATE_PATTERN = "private-e2e-worker"


class PgrepToolManagerE2ETest(IsolatedAsyncioTestCase):
    async def test_selected_tool_runs_with_safe_model_facing_result(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_pgrep(root)
            manager = _manager(root, executable, allow_process_tools=True)
            events: list[ToolExecutionStreamEvent] = []

            async def record(event: ToolExecutionStreamEvent) -> None:
                events.append(event)

            outcome = await manager.execute_call(
                ToolCall(
                    id="pgrep-success",
                    name="shell.pgrep",
                    arguments={
                        "pattern": PRIVATE_PATTERN,
                        "full": True,
                        "parent_pid": 42,
                    },
                ),
                context=ToolCallContext(stream_event=record),
            )
            launched = marker.read_text(encoding="utf-8")

            self.assertEqual(
                tuple(descriptor.name for descriptor in manager.list_tools()),
                ("shell.pgrep",),
            )
            descriptor = manager.describe_tool("shell.pgrep")
            assert descriptor is not None
            assert descriptor.parameter_schema is not None
            self.assertEqual(
                descriptor.parameter_schema["required"],
                ["pattern"],
            )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "4242\n14242\n")
        self.assertEqual(result.stderr, REDACTED_PGREP_STDERR)
        self.assertEqual(
            result.argv,
            (
                "pgrep",
                "-f",
                "-P",
                "42",
                "--",
                REDACTED_PGREP_PATTERN,
            ),
        )
        self.assertIn(PRIVATE_PATTERN, launched)
        self.assertNotIn(PRIVATE_PATTERN, outcome.result)
        self.assertNotIn("worker process name", outcome.result)
        self.assertEqual(events, [])

    async def test_default_process_gate_denies_before_launch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_pgrep(root)
            manager = _manager(root, executable, allow_process_tools=False)

            outcome = await manager.execute_call(
                ToolCall(
                    id="pgrep-denied",
                    name="shell.pgrep",
                    arguments={"pattern": PRIVATE_PATTERN},
                ),
                context=ToolCallContext(),
            )

            self.assertFalse(marker.exists())

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")
        self.assertNotIn(PRIVATE_PATTERN, outcome.result)


def _manager(
    root: Path,
    executable: Path,
    *,
    allow_process_tools: bool,
) -> ToolManager:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_process_tools=allow_process_tools,
        executable_paths={"pgrep": str(executable)},
    )
    return ToolManager.create_instance(
        available_toolsets=[
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
            )
        ],
        enable_tools=["shell.pgrep"],
        settings=ToolManagerSettings(),
    )


def _fake_pgrep(root: Path) -> tuple[Path, Path]:
    executable = root / "pgrep"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        "printf '4242\\nworker process name\\n14242\\n'\n"
        "printf 'private diagnostic %s\\n' \"$*\" >&2\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


if __name__ == "__main__":
    main()
