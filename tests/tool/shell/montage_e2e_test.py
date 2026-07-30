from .image_fixtures import VALID_JPEG_BYTES

from json import loads
from pathlib import Path
from sys import executable as sys_executable
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
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSet,
    ShellToolSettings,
)


class MontageToolManagerE2ETest(IsolatedAsyncioTestCase):
    async def test_selected_tool_executes_sample_semantics(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_montage(root)
            paths = tuple(f"pages/page-{index:02d}" for index in range(1, 7))
            (root / "pages").mkdir()
            for path in paths:
                _write_jpeg(root / path)
            manager = _manager(root, executable, allow_media_tools=True)
            descriptor = manager.describe_tool("shell.montage")
            assert descriptor is not None
            assert descriptor.parameter_schema is not None
            schema = descriptor.parameter_schema

            outcome = await manager.execute_call(
                ToolCall(
                    id="montage-success",
                    name="shell.montage",
                    arguments={
                        "paths": list(paths),
                        "thumbnail": "425x550",
                        "tile": "3x2",
                        "geometry": "+8+8",
                        "output_format": "jpg",
                        "output_filename": "contact-01-06.jpg",
                    },
                ),
                context=ToolCallContext(),
            )
            launched = loads(marker.read_text(encoding="utf-8"))

        self.assertEqual(
            tuple(tool.name for tool in manager.list_tools()),
            ("shell.montage",),
        )
        self.assertEqual(schema["required"], ["paths"])
        self.assertIs(schema["additionalProperties"], False)
        self.assertEqual(
            schema["properties"]["output_format"]["enum"],
            ["jpg", "jpeg", "png"],
        )
        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertIsInstance(outcome.result, ShellFormattedResult)
        assert isinstance(outcome.result, ShellFormattedResult)
        result = outcome.result.execution_result
        self.assertIs(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(len(result.generated_files), 1)
        generated = result.generated_files[0]
        self.assertEqual(generated.display_path, "contact-01-06.jpg")
        self.assertEqual(generated.media_type, "image/jpeg")
        self.assertEqual((generated.width, generated.height), (16, 16))
        expected_inputs = [f"./{path}[0]" for path in paths]
        first_input_index = launched.index(expected_inputs[0])
        self.assertEqual(
            launched[first_input_index : first_input_index + len(paths)],
            expected_inputs,
        )
        self.assertIn(
            ["+set", "label", "+set", "caption"],
            [
                launched[index : index + 4]
                for index in range(len(launched) - 3)
            ],
        )
        list_limit_index = launched.index("list-length")
        self.assertEqual(launched[list_limit_index + 1], "6")
        self.assertEqual(launched[-8:-6], ["-thumbnail", "425x550"])
        self.assertEqual(launched[-6:-4], ["-tile", "3x2"])
        self.assertEqual(launched[-4:-2], ["-geometry", "+8+8"])
        self.assertEqual(launched[-2], "-strip")
        self.assertTrue(launched[-1].endswith("/montage.jpg"))
        self.assertNotIn("-jpg", launched)

    async def test_default_media_gate_denies_without_launch(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            executable, marker = _fake_montage(root)
            _write_jpeg(root / "one.jpg")
            _write_jpeg(root / "two.jpg")
            manager = _manager(root, executable, allow_media_tools=False)

            outcome = await manager.execute_call(
                ToolCall(
                    id="montage-media-denied",
                    name="shell.montage",
                    arguments={"paths": ["one.jpg", "two.jpg"]},
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
        self.assertIs(
            result.error_code, ShellExecutionErrorCode.DENIED_COMMAND
        )

    async def test_schema_invalid_arguments_never_launch(self) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {},
            {"paths": "one.jpg"},
            {"paths": ["one.jpg", "two.jpg"], "output_format": "gif"},
            {"paths": ["one.jpg", "two.jpg"], "private": True},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_montage(root)
                    manager = _manager(
                        root,
                        executable,
                        allow_media_tools=True,
                    )

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="montage-schema-invalid",
                            name="shell.montage",
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

    async def test_policy_invalid_options_never_launch(self) -> None:
        invalid_arguments: tuple[dict[str, ToolValue], ...] = (
            {"paths": ["one.jpg"]},
            {
                "paths": ["one.jpg", "two.jpg"],
                "thumbnail": "20x20;touch",
            },
            {
                "paths": ["one.jpg", "two.jpg"],
                "output_filename": "../escape.jpg",
            },
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with TemporaryDirectory() as temporary_directory:
                    root = Path(temporary_directory)
                    executable, marker = _fake_montage(root)
                    _write_jpeg(root / "one.jpg")
                    _write_jpeg(root / "two.jpg")
                    manager = _manager(
                        root,
                        executable,
                        allow_media_tools=True,
                    )

                    outcome = await manager.execute_call(
                        ToolCall(
                            id="montage-policy-invalid",
                            name="shell.montage",
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


def _manager(
    root: Path,
    executable: Path,
    *,
    allow_media_tools: bool,
) -> ToolManager:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_media_tools=allow_media_tools,
        executable_paths={"montage": str(executable)},
    )
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet(settings=settings)],
        enable_tools=["shell.montage"],
        settings=ToolManagerSettings(),
    )


def _fake_montage(root: Path) -> tuple[Path, Path]:
    executable = root / "montage"
    marker = Path(f"{executable}.launched")
    script = (
        f"#!{sys_executable}\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        "pathlib.Path(f'{sys.argv[0]}.launched').write_text(\n"
        "    json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        "output = pathlib.Path(sys.argv[-1])\n"
        f"output.write_bytes({VALID_JPEG_BYTES!r})\n"
    )
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    return executable, marker


def _write_jpeg(path: Path) -> None:
    path.write_bytes(VALID_JPEG_BYTES)


if __name__ == "__main__":
    main()
