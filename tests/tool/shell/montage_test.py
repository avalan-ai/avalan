from .image_fixtures import (
    VALID_JPEG_BYTES,
    multi_frame_ppm_bytes,
)

from json import loads
from pathlib import Path
from sys import executable as sys_executable
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolCallContext
from avalan.tool import Tool
from avalan.tool.shell import (
    ExecutionPolicy,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellOutputKind,
    ShellPolicyDenied,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
    unavailable_executable_lookup,
)
from avalan.tool.shell.commands.base import NormalizedPath
from avalan.tool.shell.commands.montage import (
    MONTAGE_INPUT_DIMENSIONS_METADATA_KEY,
    _input_dimensions,
    _montage_paths,
    _safe_imagemagick_path,
)
from avalan.tool.shell.entities import GENERATED_OUTPUT_PREFIX_PLACEHOLDER
from avalan.tool.shell.policy import (
    _annotate_montage_metadata,
    _enforce_montage_inputs,
)


class MontagePolicyTest(IsolatedAsyncioTestCase):
    async def test_sample_shape_normalizes_to_safe_jpeg_argv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pages = root / "pages"
            pages.mkdir()
            paths = tuple(f"pages/page-{index:02d}" for index in range(1, 7))
            for path in paths:
                _write_jpeg(root / path)
            settings = _settings(root)
            spec = await ExecutionPolicy(
                settings=settings,
                resolver=_ResolvedMontage(),
            ).normalize(
                _request(
                    paths,
                    thumbnail="425x550",
                    tile="3x2",
                    geometry="+8+8",
                    output_format="jpg",
                    output_filename="contact-01-06.jpg",
                )
            )

        self.assertEqual(spec.command, "montage")
        self.assertEqual(spec.executable, "/trusted/bin/montage")
        self.assertEqual(
            spec.argv,
            (
                "montage",
                "-define",
                "registry:filename:literal=true",
                "-limit",
                "memory",
                "268435456",
                "-limit",
                "map",
                "536870912",
                "-limit",
                "disk",
                "1073741824",
                "-limit",
                "thread",
                "2",
                "-limit",
                "list-length",
                "6",
                "./pages/page-01[0]",
                "./pages/page-02[0]",
                "./pages/page-03[0]",
                "./pages/page-04[0]",
                "./pages/page-05[0]",
                "./pages/page-06[0]",
                "+set",
                "label",
                "+set",
                "caption",
                "-thumbnail",
                "425x550",
                "-tile",
                "3x2",
                "-geometry",
                "+8+8",
                "-strip",
                GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
            ),
        )
        self.assertNotIn("-jpg", spec.argv)
        self.assertEqual(spec.display_argv[-1], "contact-01-06.jpg")
        self.assertEqual(spec.output_kind, ShellOutputKind.GENERATED_FILES)
        assert spec.output_plan is not None
        self.assertEqual(spec.output_plan.output_path_suffix, ".jpg")
        self.assertEqual(
            spec.output_plan.runtime_path("/outputs/montage"),
            "/outputs/montage.jpg",
        )
        self.assertEqual(
            spec.output_plan.suffix_media_types,
            {".jpg": "image/jpeg"},
        )
        self.assertEqual(spec.metadata["montage_input_count"], 6)
        self.assertEqual(
            spec.metadata["montage_projected_dimensions"],
            (1323, 1132),
        )
        self.assertEqual(spec.timeout_seconds, 30.0)

    async def test_defaults_derive_tile_and_support_png_quality(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = ("one.jpg", "two.jpg", "three.jpg")
            for path in paths:
                _write_jpeg(root / path)
            spec = await ExecutionPolicy(
                settings=_settings(root),
                resolver=_ResolvedMontage(),
            ).normalize(
                _request(
                    paths,
                    output_format="png",
                    output_filename="sheet.png",
                    quality=90,
                )
            )

        self.assertNotIn("-thumbnail", spec.argv)
        self.assertIn("2x2", spec.argv)
        self.assertEqual(spec.display_argv[-1], "sheet.png")
        self.assertEqual(spec.argv[-3:-1], ("-quality", "90"))
        assert spec.output_plan is not None
        self.assertEqual(spec.output_plan.output_path_suffix, ".png")
        self.assertEqual(
            spec.output_plan.suffix_media_types,
            {".png": "image/png"},
        )

    async def test_jpeg_alias_accepts_jpg_artifact_suffix(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jpeg(root / "one.jpg")
            _write_jpeg(root / "two.jpg")
            spec = await ExecutionPolicy(
                settings=_settings(root),
                resolver=_ResolvedMontage(),
            ).normalize(
                _request(
                    ("one.jpg", "two.jpg"),
                    thumbnail="10x10",
                    tile="2x1",
                    output_format="jpeg",
                    output_filename="sheet.jpg",
                )
            )

        assert spec.output_plan is not None
        self.assertEqual(spec.output_plan.output_path_suffix, ".jpg")
        self.assertEqual(spec.display_argv[-1], "sheet.jpg")

    async def test_trusted_operator_font_is_passed_as_one_argument(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jpeg(root / "one.jpg")
            _write_jpeg(root / "two.jpg")
            spec = await ExecutionPolicy(
                settings=_settings(
                    root,
                    montage_font="/System/Library/Fonts/Font With Spaces.ttc",
                ),
                resolver=_ResolvedMontage(),
            ).normalize(
                _request(
                    ("one.jpg", "two.jpg"),
                    thumbnail="10x10",
                    tile="2x1",
                )
            )

        font_option_index = spec.argv.index("-font")
        self.assertEqual(
            spec.argv[font_option_index : font_option_index + 2],
            (
                "-font",
                "/System/Library/Fonts/Font With Spaces.ttc",
            ),
        )
        self.assertEqual(
            spec.display_argv[font_option_index : font_option_index + 2],
            ("-font", "[configured_font]"),
        )
        self.assertEqual(
            spec.argv[font_option_index + 2 : font_option_index + 4],
            ("./one.jpg[0]", "./two.jpg[0]"),
        )

    async def test_literal_image_names_are_not_shell_or_coder_expanded(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = (
                "@list.jpg",
                "https:remote.jpg",
                "name*.jpg",
                "page-{1,2}.jpg",
            )
            for path in paths:
                _write_jpeg(root / path)
            spec = await ExecutionPolicy(
                settings=_settings(root),
                resolver=_ResolvedMontage(),
            ).normalize(_request(paths, thumbnail="10x10", tile="2x2"))

        input_index = spec.argv.index("./@list.jpg[0]")
        self.assertEqual(
            spec.argv[input_index : input_index + 4],
            (
                "./@list.jpg[0]",
                "./https:remote.jpg[0]",
                "./name*.jpg[0]",
                "./page-{1,2}.jpg[0]",
            ),
        )
        list_limit_index = spec.argv.index("list-length")
        self.assertEqual(spec.argv[list_limit_index + 1], "4")

    async def test_multiframe_inputs_select_exactly_first_scene(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = multi_frame_ppm_bytes(
                first_rgb=(255, 0, 0),
                second_rgb=(0, 0, 255),
            )
            (root / "first.pnm").write_bytes(frames)
            (root / "second.pnm").write_bytes(frames)

            spec = await ExecutionPolicy(
                settings=_settings(root),
                resolver=_ResolvedMontage(),
            ).normalize(
                _request(
                    ("first.pnm", "second.pnm"),
                    tile="2x1",
                    output_format="png",
                )
            )

        self.assertIn("./first.pnm[0]", spec.argv)
        self.assertIn("./second.pnm[0]", spec.argv)
        self.assertNotIn("./first.pnm", spec.argv)
        self.assertNotIn("./second.pnm", spec.argv)
        list_limit_index = spec.argv.index("list-length")
        self.assertEqual(spec.argv[list_limit_index + 1], "2")

    async def test_media_opt_in_and_command_availability_fail_closed(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jpeg(root / "one.jpg")
            _write_jpeg(root / "two.jpg")
            request = _request(
                ("one.jpg", "two.jpg"),
                thumbnail="10x10",
                tile="2x1",
            )
            with self.assertRaises(ShellPolicyDenied) as denied:
                await ExecutionPolicy(
                    settings=ShellToolSettings(workspace_root=str(root)),
                    resolver=_ResolvedMontage(),
                ).normalize(request)
            unavailable = await ExecutionPolicy(
                settings=_settings(root),
                resolver=TrustedExecutableResolver(
                    lookup=unavailable_executable_lookup,
                ),
            ).normalize(request)

        self.assertEqual(
            denied.exception.error_code,
            ShellExecutionErrorCode.DENIED_COMMAND,
        )
        self.assertIsNone(unavailable.executable)

    async def test_invalid_options_and_shapes_are_denied(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = ("one.jpg", "two.jpg")
            for path in paths:
                _write_jpeg(root / path)
            cases = (
                (
                    "one-input",
                    ("one.jpg",),
                    {"thumbnail": "10x10", "tile": "1x1"},
                ),
                ("unknown-option", paths, {"unsafe": True}),
                ("thumbnail-type", paths, {"thumbnail": 10}),
                ("thumbnail-shape", paths, {"thumbnail": "10x10;touch"}),
                ("thumbnail-zero", paths, {"thumbnail": "0x10"}),
                ("thumbnail-cap", paths, {"thumbnail": "2049x1"}),
                ("tile-shape", paths, {"tile": "2x"}),
                ("tile-capacity", paths, {"tile": "1x1"}),
                ("geometry-type", paths, {"geometry": 8}),
                ("geometry-shape", paths, {"geometry": "+8+8;touch"}),
                ("geometry-cap", paths, {"geometry": "+1025+0"}),
                ("quality-type", paths, {"quality": True}),
                ("quality-low", paths, {"quality": 0}),
                ("quality-high", paths, {"quality": 101}),
                ("format", paths, {"output_format": "gif"}),
                ("filename-type", paths, {"output_filename": 1}),
                (
                    "filename-directory",
                    paths,
                    {"output_filename": "tmp/sheet.jpg"},
                ),
                (
                    "filename-hidden",
                    paths,
                    {"output_filename": ".sheet.jpg"},
                ),
                (
                    "filename-format",
                    paths,
                    {
                        "output_format": "png",
                        "output_filename": "sheet.jpg",
                    },
                ),
                (
                    "projected-output",
                    paths,
                    {"thumbnail": "1025x1024", "tile": "2x1"},
                ),
            )
            for name, case_paths, options in cases:
                with self.subTest(name=name):
                    with self.assertRaises(ShellPolicyDenied):
                        await ExecutionPolicy(
                            settings=_settings(root),
                            resolver=_ResolvedMontage(),
                        ).normalize(_request(case_paths, **options))

    async def test_input_count_byte_and_pixel_limits_are_denied(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jpeg(root / "one.jpg")
            _write_jpeg(root / "two.jpg")
            inputs = ("one.jpg", "two.jpg")
            cases = (
                (
                    "count",
                    {"max_montage_inputs": 1},
                    tuple("one.jpg" for _ in range(2)),
                ),
                (
                    "bytes",
                    {"max_montage_input_bytes": 1},
                    inputs,
                ),
                (
                    "long-edge",
                    {"max_montage_input_long_edge_pixels": 15},
                    inputs,
                ),
                (
                    "per-file-pixels",
                    {"max_montage_input_pixels_per_file": 255},
                    inputs,
                ),
                (
                    "total-pixels",
                    {"max_montage_input_pixels": 511},
                    inputs,
                ),
                (
                    "output-pixels",
                    {"max_raster_pixels": 199},
                    inputs,
                ),
            )
            for name, overrides, case_paths in cases:
                with self.subTest(name=name):
                    with self.assertRaises(ShellPolicyDenied) as denied:
                        await ExecutionPolicy(
                            settings=_settings(root, **overrides),
                            resolver=_ResolvedMontage(),
                        ).normalize(
                            _request(
                                case_paths,
                                thumbnail="10x10",
                                tile="2x1",
                            )
                        )
                    self.assertIn(
                        denied.exception.error_code,
                        {
                            ShellExecutionErrorCode.TOO_LARGE,
                            ShellExecutionErrorCode.INVALID_OPTION,
                            (
                                ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED
                            ),
                        },
                    )

    async def test_unsupported_image_and_shell_like_path_are_denied(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jpeg(root / "one.jpg")
            (root / "bad.jpg").write_bytes(b"not an image")
            with self.assertRaises(ShellPolicyDenied) as signature_denied:
                await ExecutionPolicy(
                    settings=_settings(root),
                    resolver=_ResolvedMontage(),
                ).normalize(
                    _request(
                        ("one.jpg", "bad.jpg"),
                        thumbnail="10x10",
                        tile="2x1",
                    )
                )
            with self.assertRaises(ShellPolicyDenied):
                await ExecutionPolicy(
                    settings=_settings(root),
                    resolver=_ResolvedMontage(),
                ).normalize(
                    _request(
                        ("one.jpg", "$(touch-owned).jpg"),
                        thumbnail="10x10",
                        tile="2x1",
                    )
                )

        self.assertEqual(
            signature_denied.exception.error_code,
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
        )

    async def test_montage_metadata_helpers_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            invalid_path = root / "invalid.jpg"
            invalid_path.write_bytes(b"not an image")
            normalized = NormalizedPath(
                operand=PathOperand(
                    name="path",
                    path="invalid.jpg",
                    kind="image_file",
                    access="read",
                ),
                path=invalid_path,
                display_path="invalid.jpg",
                metadata=None,
            )
            metadata: dict[str, object] = {}

            await _annotate_montage_metadata(
                _request(("invalid.jpg", "invalid.jpg")),
                (normalized,),
                metadata,
            )
            with self.assertRaises(ShellPolicyDenied) as denied:
                await _enforce_montage_inputs(
                    (normalized,),
                    settings=_settings(root),
                )

        self.assertEqual(
            metadata[MONTAGE_INPUT_DIMENSIONS_METADATA_KEY],
            (),
        )
        self.assertEqual(
            denied.exception.error_code,
            ShellExecutionErrorCode.UNSUPPORTED_MEDIA_SIGNATURE,
        )


class MontageHostExecutionTest(IsolatedAsyncioTestCase):
    async def test_local_tool_returns_named_generated_jpeg(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            marker = root / "argv.json"
            executable = _fake_montage(root, marker)
            configured_font = "/private/fonts/Operator Font.ttf"
            paths = ("one.jpg", "two.jpg")
            for path in paths:
                _write_jpeg(root / path)
            toolset = ShellToolSet(
                settings=_settings(
                    root,
                    executable_paths={"montage": str(executable)},
                    montage_font=configured_font,
                )
            )
            tool = _tool(toolset)
            output = await tool(
                paths,
                thumbnail="20x20",
                tile="2x1",
                geometry="+2+3",
                output_format="jpg",
                output_filename="contact-01-02.jpg",
                quality=85,
                context=ToolCallContext(),
            )
            result = output.execution_result
            recorded = loads(marker.read_text(encoding="utf-8"))

        self.assertIsInstance(output, ShellFormattedResult)
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(len(result.generated_files), 1)
        generated = result.generated_files[0]
        self.assertEqual(generated.display_path, "contact-01-02.jpg")
        self.assertEqual(generated.media_type, "image/jpeg")
        self.assertEqual((generated.width, generated.height), (16, 16))
        self.assertIsNotNone(generated.content_base64)
        self.assertEqual(recorded[-1][-12:], "/montage.jpg")
        self.assertNotIn(GENERATED_OUTPUT_PREFIX_PLACEHOLDER, recorded)
        self.assertIn("registry:filename:literal=true", recorded)
        self.assertNotIn("-jpg", recorded)
        self.assertIn(configured_font, recorded)
        self.assertNotIn(configured_font, result.display_argv)
        self.assertNotIn(configured_font, str(output))
        self.assertIn("[configured_font]", result.display_argv)

    async def test_missing_montage_returns_command_unavailable(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_jpeg(root / "one.jpg")
            _write_jpeg(root / "two.jpg")
            settings = _settings(root)
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(
                    settings=settings,
                    resolver=TrustedExecutableResolver(
                        lookup=unavailable_executable_lookup,
                    ),
                ),
            )
            output = await _tool(toolset)(
                ("one.jpg", "two.jpg"),
                thumbnail="10x10",
                tile="2x1",
                context=ToolCallContext(),
            )

        self.assertEqual(
            output.execution_result.status,
            ShellExecutionStatus.COMMAND_UNAVAILABLE,
        )
        self.assertEqual(output.execution_result.generated_files, ())
        self.assertIn("command is unavailable", output)


class MontageSchemaTest(TestCase):
    def test_schema_exposes_structured_montage_arguments(self) -> None:
        schemas = ShellToolSet().json_schemas() or ()
        schema = next(
            item
            for item in schemas
            if item["function"]["name"] == "shell.montage"
        )
        parameters = schema["function"]["parameters"]

        self.assertEqual(parameters["required"], ["paths"])
        self.assertEqual(
            parameters["properties"]["output_format"]["enum"],
            ["jpg", "jpeg", "png"],
        )
        self.assertEqual(
            parameters["properties"]["paths"]["items"],
            {"type": "string"},
        )


class MontageCommandHelperTest(TestCase):
    def test_path_kind_and_metadata_checks_fail_closed(self) -> None:
        wrong_kind = NormalizedPath(
            operand=PathOperand(
                name="path",
                path="doc.pdf",
                kind="pdf_file",
                access="read",
            ),
            path=Path("doc.pdf"),
            display_path="doc.pdf",
            metadata=None,
        )
        with self.assertRaises(ShellPolicyDenied):
            _montage_paths((wrong_kind, wrong_kind), max_inputs=2)

        invalid_metadata = (
            {"_montage_input_dimensions": ((10, True), (10, 10))},
            {"_montage_input_dimensions": ((10, 0), (10, 10))},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ShellPolicyDenied):
                    _input_dimensions(metadata, input_count=2)

    def test_imagemagick_path_keeps_explicit_relative_and_absolute_paths(
        self,
    ) -> None:
        self.assertEqual(_safe_imagemagick_path("./one.jpg"), "./one.jpg")
        self.assertEqual(_safe_imagemagick_path("/one.jpg"), "/one.jpg")
        self.assertEqual(_safe_imagemagick_path("one.jpg"), "./one.jpg")


class _ResolvedMontage:
    async def resolve(self, command: object) -> str:
        return "/trusted/bin/montage"


def _settings(root: Path, **overrides: object) -> ShellToolSettings:
    values: dict[str, object] = {
        "workspace_root": str(root),
        "allow_media_tools": True,
    }
    values.update(overrides)
    return ShellToolSettings(**values)


def _request(paths: tuple[str, ...], **options: object) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name="shell.montage",
        command="montage",
        options=dict(options),
        paths=tuple(
            PathOperand(
                name=f"path_{index}",
                path=path,
                kind="image_file",
                access="read",
            )
            for index, path in enumerate(paths)
        ),
        cwd=None,
    )


def _tool(toolset: ShellToolSet) -> Tool:
    tool = next(
        item
        for item in toolset.tools
        if getattr(item, "__name__", "") == "montage"
    )
    assert isinstance(tool, Tool)
    return tool


def _write_jpeg(path: Path) -> None:
    path.write_bytes(VALID_JPEG_BYTES)


def _fake_montage(root: Path, marker: Path) -> Path:
    executable = root / "montage"
    script = (
        f"#!{sys_executable}\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n"
        f"marker = pathlib.Path({str(marker)!r})\n"
        "marker.write_text(json.dumps(sys.argv[1:]), encoding='utf-8')\n"
        "output = pathlib.Path(sys.argv[-1])\n"
        f"output.write_bytes({VALID_JPEG_BYTES!r})\n"
    )
    executable.write_text(script, encoding="utf-8")
    executable.chmod(0o700)
    return executable


if __name__ == "__main__":
    main()
