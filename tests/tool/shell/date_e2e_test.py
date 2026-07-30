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


class DateToolManagerE2ETest(IsolatedAsyncioTestCase):
    async def test_selected_tool_exposes_bounded_schema_and_executes(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_date(root)
            manager = _manager(root, executable)
            descriptor = manager.describe_tool("shell.date")
            assert descriptor is not None
            assert descriptor.parameter_schema is not None
            schema = descriptor.parameter_schema
            custom_format_schema = schema["properties"]["custom_format"]

            outcome = await manager.execute_call(
                ToolCall(
                    id="date-success",
                    name="shell.date",
                    arguments={"utc": True, "format": "iso8601"},
                ),
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertEqual(
            tuple(tool.name for tool in manager.list_tools()),
            ("shell.date",),
        )
        self.assertEqual(schema.get("required", []), [])
        self.assertEqual(
            schema["properties"]["format"]["enum"],
            ["default", "date", "time", "iso8601", "unix"],
        )
        self.assertEqual(
            schema["properties"]["format"]["default"],
            "default",
        )
        self.assertIs(schema["properties"]["utc"]["default"], False)
        self.assertEqual(
            custom_format_schema["type"],
            ["string", "null"],
        )
        self.assertIsNone(custom_format_schema["default"])
        self.assertEqual(custom_format_schema["minLength"], 1)
        self.assertEqual(custom_format_schema["maxLength"], 128)
        self.assertIn("pattern", custom_format_schema)
        self.assertEqual(
            schema["not"]["required"],
            ["format", "custom_format"],
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, "2026-07-30T07:05:06+0000\n")
        self.assertEqual(launched, "-u +%Y-%m-%dT%H:%M:%S%z\n")

    async def test_valid_custom_format_executes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_date(root)
            manager = _manager(root, executable)

            outcome = await manager.execute_call(
                ToolCall(
                    id="date-custom-success",
                    name="shell.date",
                    arguments={
                        "utc": True,
                        "custom_format": "e2e=%Y-%m-%d %% %H:%M:%S %z",
                    },
                ),
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(
            result.stdout,
            "e2e=2026-07-30 % 07:05:06 +0000\n",
        )
        self.assertEqual(
            launched,
            "-u +e2e=%Y-%m-%d %% %H:%M:%S %z\n",
        )

    async def test_schema_rejects_unknown_format_types_and_options(
        self,
    ) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {"format": "rfc3339"},
            {"format": 1},
            {"utc": "true"},
            {"path": "visible.txt"},
            {"unknown": True},
            {"custom_format": 1},
            {"custom_format": ""},
            {"custom_format": "x" * 129},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_date(root)
                    manager = _manager(root, executable)

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="date-invalid",
                            name="shell.date",
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

    async def test_policy_rejects_custom_format_semantics_via_manager(
        self,
    ) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {"custom_format": "%Y\n"},
            {"custom_format": "\t"},
            {"custom_format": "\x00"},
            {"custom_format": "café"},
            {"custom_format": "%"},
            {"custom_format": "%a"},
            {"custom_format": "%-d"},
            {"custom_format": "%3Y"},
            {"custom_format": "%Ec"},
            {"custom_format": "%:z"},
            {"format": "date", "custom_format": "%Y"},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_date(root)
                    manager = _manager(root, executable)

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="date-invalid",
                            name="shell.date",
                            arguments=arguments,
                        ),
                        context=ToolCallContext(),
                    )

                    self.assertFalse(marker.exists())

                self.assertIsInstance(outcome, ToolCallResult)
                assert isinstance(outcome, ToolCallResult)
                self.assertIsInstance(outcome.result, ShellFormattedResult)
                assert isinstance(outcome.result, ShellFormattedResult)
                self.assertIs(
                    outcome.result.execution_result.status,
                    ShellExecutionStatus.POLICY_DENIED,
                )


def _manager(root: Path, executable: Path) -> ToolManager:
    settings = ShellToolSettings(
        workspace_root=str(root),
        executable_paths={"date": str(executable)},
    )
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet(settings=settings)],
        enable_tools=["shell.date"],
        settings=ToolManagerSettings(),
    )


def _fake_date(root: Path) -> tuple[Path, Path]:
    executable = root / "date"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        '  "-u +%Y-%m-%dT%H:%M:%S%z") '
        'printf "2026-07-30T07:05:06+0000\\n" ;;\n'
        '  "-u +e2e=%Y-%m-%d %% %H:%M:%S %z") '
        'printf "e2e=2026-07-30 %% 07:05:06 +0000\\n" ;;\n'
        '  *) printf "unexpected argv\\n" >&2; exit 64 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


if __name__ == "__main__":
    main()
