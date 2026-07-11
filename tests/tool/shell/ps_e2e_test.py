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


class PsToolManagerE2ETest(IsolatedAsyncioTestCase):
    async def test_selected_tool_has_bounded_schema_and_safe_result(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, _ = _fake_ps(root)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_process_tools=True,
                executable_paths={"ps": str(executable)},
            )
            manager = ToolManager.create_instance(
                available_toolsets=[ShellToolSet(settings=settings)],
                enable_tools=["shell.ps"],
                settings=ToolManagerSettings(),
            )
            descriptor = manager.describe_tool("shell.ps")
            assert descriptor is not None
            assert descriptor.parameter_schema is not None
            schema = descriptor.parameter_schema
            self.assertEqual(schema["required"], ["pids"])
            pids_schema = schema["properties"]["pids"]
            self.assertEqual(pids_schema["minItems"], 1)
            self.assertEqual(pids_schema["maxItems"], 1)
            self.assertIs(pids_schema["uniqueItems"], True)
            self.assertEqual(pids_schema["items"]["minimum"], 1)
            self.assertEqual(pids_schema["items"]["maximum"], 2**31 - 1)
            view_schema = schema["properties"]["view"]
            self.assertEqual(view_schema["enum"], ["summary", "resources"])
            self.assertEqual(view_schema["default"], "summary")

            outcome = await manager.execute_call(
                ToolCall(
                    id="ps-success",
                    name="shell.ps",
                    arguments={"pids": [1]},
                ),
                context=ToolCallContext(),
            )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        self.assertEqual(
            outcome.result.execution_result.stdout,
            "1 0 S 00:01 init\n",
        )

    async def test_resource_view_executes_through_public_manager(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            manager = _manager(root, executable, allow_process_tools=True)

            outcome = await manager.execute_call(
                ToolCall(
                    id="ps-resources",
                    name="shell.ps",
                    arguments={"pids": [1], "view": "resources"},
                ),
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertEqual(result.stdout, "1 0.0 0.1 512 2048 0:00.01 0\n")
        self.assertEqual(result.metadata, {})
        self.assertIn("-o pcpu=", launched)

    async def test_default_gate_denies_without_launch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_ps(root)
            manager = _manager(root, executable, allow_process_tools=False)

            outcome = await manager.execute_call(
                ToolCall(
                    id="ps-denied",
                    name="shell.ps",
                    arguments={"pids": [1]},
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
        self.assertEqual(result.error_message, "ps was denied by policy")

    async def test_schema_invalid_pid_arguments_never_launch(self) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {},
            {"pids": []},
            {"pids": [True]},
            {"pids": [1, 2]},
            {"pids": [1], "view": "private"},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_ps(root)
                    manager = _manager(
                        root,
                        executable,
                        allow_process_tools=True,
                    )

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="ps-invalid",
                            name="shell.ps",
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

    async def test_policy_invalid_pid_arguments_never_launch(self) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {"pids": [0]},
            {"pids": [2**31]},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_ps(root)
                    manager = _manager(
                        root,
                        executable,
                        allow_process_tools=True,
                    )

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="ps-invalid",
                            name="shell.ps",
                            arguments=arguments,
                        ),
                        context=ToolCallContext(),
                    )

                    self.assertFalse(marker.exists())
                    self.assertIsInstance(outcome, ToolCallResult)
                    assert isinstance(outcome, ToolCallResult)
                    self.assertIsInstance(
                        outcome.result,
                        ShellFormattedResult,
                    )
                    assert isinstance(outcome.result, ShellFormattedResult)
                    result = outcome.result.execution_result
                    self.assertIs(
                        result.status,
                        ShellExecutionStatus.POLICY_DENIED,
                    )
                    self.assertIsNotNone(result.error_code)
                    assert result.error_code is not None
                    self.assertEqual(result.error_code.value, "invalid_option")


def _manager(
    root: Path,
    executable: Path,
    *,
    allow_process_tools: bool,
) -> ToolManager:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_process_tools=allow_process_tools,
        executable_paths={"ps": str(executable)},
    )
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet(settings=settings)],
        enable_tools=["shell.ps"],
        settings=ToolManagerSettings(),
    )


def _fake_ps(root: Path) -> tuple[Path, Path]:
    executable = root / "ps"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        "  *pcpu=*) printf '1 0.0 0.1 512 2048 0:00.01 0\\n' ;;\n"
        "  *) printf '1 0 S 00:01 init\\n' ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


if __name__ == "__main__":
    main()
