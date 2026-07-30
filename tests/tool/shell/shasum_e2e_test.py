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

_SHA256 = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


class ShasumToolManagerE2ETest(IsolatedAsyncioTestCase):
    async def test_selected_tool_exposes_bounded_schema_and_executes(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "visible.txt").write_text(
                "hello world",
                encoding="utf-8",
            )
            executable, marker = _fake_shasum(root)
            manager = _manager(root, executable)
            descriptor = manager.describe_tool("shell.shasum")
            assert descriptor is not None
            assert descriptor.parameter_schema is not None
            schema = descriptor.parameter_schema

            outcome = await manager.execute_call(
                ToolCall(
                    id="shasum-success",
                    name="shell.shasum",
                    arguments={
                        "paths": ["visible.txt"],
                        "algorithm": "256",
                    },
                ),
                context=ToolCallContext(),
            )
            launched = marker.read_text(encoding="utf-8")

        self.assertEqual(
            tuple(tool.name for tool in manager.list_tools()),
            ("shell.shasum",),
        )
        self.assertEqual(schema["required"], ["paths"])
        paths_schema = schema["properties"]["paths"]
        self.assertEqual(paths_schema["minItems"], 1)
        self.assertEqual(paths_schema["maxItems"], 128)
        self.assertEqual(
            schema["properties"]["algorithm"]["enum"],
            ["1", "224", "256", "384", "512", "512224", "512256"],
        )
        self.assertEqual(
            schema["properties"]["algorithm"]["default"],
            "1",
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.stdout, f"{_SHA256}  visible.txt\n")
        self.assertEqual(launched, "-a 256 -- visible.txt\n")

    async def test_schema_rejects_raw_modes_invalid_algorithms_and_bounds(
        self,
    ) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {},
            {"paths": []},
            {"paths": ["visible.txt"], "algorithm": "sha256"},
            {"paths": ["visible.txt"], "algorithm": 256},
            {"paths": ["visible.txt"], "check": True},
            {"paths": ["visible.txt"], "argv": ["-c"]},
            {"paths": ["visible.txt"] * 129},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    (root / "visible.txt").write_text(
                        "hello world",
                        encoding="utf-8",
                    )
                    executable, marker = _fake_shasum(root)
                    manager = _manager(root, executable)

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="shasum-invalid",
                            name="shell.shasum",
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

    async def test_policy_rejects_directory_without_launching(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "directory").mkdir()
            executable, marker = _fake_shasum(root)
            manager = _manager(root, executable)

            outcome = await manager.execute_call(
                ToolCall(
                    id="shasum-directory",
                    name="shell.shasum",
                    arguments={"paths": ["directory"], "algorithm": "256"},
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
        executable_paths={"shasum": str(executable)},
    )
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet(settings=settings)],
        enable_tools=["shell.shasum"],
        settings=ToolManagerSettings(),
    )


def _fake_shasum(root: Path) -> tuple[Path, Path]:
    executable = root / "shasum"
    marker = Path(f"{executable}.launched")
    executable.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" > "$0.launched"\n'
        'case "$*" in\n'
        '  "-a 256 -- visible.txt") '
        f'printf "{_SHA256}  visible.txt\\n" ;;\n'
        '  *) printf "unexpected argv\\n" >&2; exit 64 ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, marker


if __name__ == "__main__":
    main()
