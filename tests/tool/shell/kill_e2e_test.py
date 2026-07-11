from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallResult,
    ToolManagerSettings,
    ToolValue,
)
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSet,
    ShellToolSettings,
)


class KillToolManagerE2ETest(IsolatedAsyncioTestCase):
    async def test_selected_tool_has_bounded_schema_and_executes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_kill(root)
            manager = _manager(root, executable, allow_process_control=True)
            descriptor = manager.describe_tool("shell.kill")
            assert descriptor is not None
            assert descriptor.parameter_schema is not None
            schema = descriptor.parameter_schema
            self.assertEqual(schema["required"], ["pid"])
            self.assertEqual(schema["properties"]["pid"]["minimum"], 2)
            self.assertEqual(schema["properties"]["pid"]["maximum"], 2**31 - 1)
            self.assertEqual(
                schema["properties"]["signal"]["enum"],
                ["TERM", "INT", "KILL"],
            )

            outcome = await manager.execute_call(
                ToolCall(
                    id="kill-success",
                    name="shell.kill",
                    arguments={"pid": 42},
                ),
                context=ToolCallContext(),
            )

            self.assertTrue(marker.exists())

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        self.assertIs(
            outcome.result.execution_result.status,
            ShellExecutionStatus.COMPLETED,
        )

    async def test_existing_process_opt_in_alone_never_launches(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_kill(root)
            manager = _manager(root, executable, allow_process_control=False)

            outcome = await manager.execute_call(
                ToolCall(
                    id="kill-denied",
                    name="shell.kill",
                    arguments={"pid": 42},
                ),
                context=ToolCallContext(),
            )

            self.assertFalse(marker.exists())

        assert isinstance(outcome, ToolCallResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(result.error_message, "kill was denied by policy")

    async def test_invalid_schema_arguments_never_launch(self) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {},
            {"pid": "42"},
            {"pid": 42, "signal": "HUP"},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_kill(root)
                    manager = _manager(
                        root,
                        executable,
                        allow_process_control=True,
                    )
                    outcome = await manager.execute_call(
                        ToolCall(
                            id="kill-invalid",
                            name="shell.kill",
                            arguments=arguments,
                        ),
                        context=ToolCallContext(),
                    )
                    self.assertFalse(marker.exists())
                    self.assertIsInstance(outcome, ToolCallDiagnostic)
                    assert isinstance(outcome, ToolCallDiagnostic)
                    self.assertIs(
                        outcome.code,
                        ToolCallDiagnosticCode.ARGUMENT_VALIDATION_FAILED,
                    )


def _manager(
    root: Path,
    executable: Path,
    *,
    allow_process_control: bool,
) -> ToolManager:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_process_tools=True,
        allow_process_control=allow_process_control,
        executable_paths={"kill": str(executable)},
    )
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet(settings=settings)],
        enable_tools=["shell.kill"],
        settings=ToolManagerSettings(),
    )


def _fake_kill(root: Path) -> tuple[Path, Path]:
    executable = root / "kill"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        '#!/bin/sh\ntouch "$0.launched"\n',
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


if __name__ == "__main__":
    main()
