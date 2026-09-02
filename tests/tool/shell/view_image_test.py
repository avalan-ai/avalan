"""Cover policy-bound model-visible filesystem image results."""

from .image_fixtures import (
    VALID_JPEG_BYTES,
    VALID_PROGRESSIVE_JPEG_BYTES,
    valid_png_bytes,
)

from hashlib import sha256
from pathlib import Path
from struct import pack
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch
from zlib import compress, crc32

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallError,
    ToolCallResult,
    ToolManagerSettings,
    ToolResultImage,
    ToolResultImageDetail,
    ToolResultText,
)
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    ExecutionPolicy,
    GeneratedFile,
    ShellPathMetadata,
    ShellPolicyDenied,
    ShellToolSet,
    ShellToolSettings,
    ViewImageTool,
)
from avalan.tool.shell import filesystem as shell_filesystem
from avalan.tool.shell.entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
    ExecutionResult,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellOutputKind,
)
from avalan.tool.shell.filesystem import (
    inspect_path,
    validate_supported_image_bytes,
)
from avalan.tool.shell.tool_images import (
    _generated_image_block,
    _generated_image_bytes,
    _is_relative_to,
    _validate_dimensions,
    montage_tool_result,
)


class ViewImageToolTest(IsolatedAsyncioTestCase):
    """Verify genuine image attachments retain shell filesystem boundaries."""

    async def test_allowed_png_is_attached_with_exact_bytes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            expected = valid_png_bytes(width=7, height=5, label="approved")
            (root / "approved.jpg").write_bytes(expected)
            manager = _manager(root)

            outcome = await manager.execute_call(
                ToolCall(
                    id="view-image",
                    name="shell.view_image",
                    arguments={"path": "approved.jpg", "detail": "high"},
                ),
                context=ToolCallContext(),
            )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        self.assertEqual(len(outcome.content), 2)
        self.assertIsInstance(outcome.content[0], ToolResultText)
        self.assertIsInstance(outcome.content[1], ToolResultImage)
        image = outcome.content[1]
        assert isinstance(image, ToolResultImage)
        self.assertEqual(image.data, expected)
        self.assertEqual(image.media_type, "image/png")
        self.assertEqual((image.width, image.height), (7, 5))
        self.assertEqual(image.detail.value, "high")
        self.assertIn("sha256:", str(outcome.result))

    async def test_invalid_paths_and_images_are_rejected(self) -> None:
        cases = (
            "directory",
            "invalid.jpg",
            "link.jpg",
            "hardlink.jpg",
            "../outside.jpg",
        )
        with (
            TemporaryDirectory() as temporary_directory,
            TemporaryDirectory() as outside_directory,
        ):
            root = Path(temporary_directory)
            outside = Path(outside_directory) / "view-image-outside.jpg"
            outside.write_bytes(VALID_JPEG_BYTES)
            try:
                (root / "directory").mkdir()
                (root / "invalid.jpg").write_bytes(b"not an image")
                (root / "target.jpg").write_bytes(VALID_JPEG_BYTES)
                (root / "link.jpg").symlink_to(root / "target.jpg")
                (root / "hardlink.jpg").hardlink_to(root / "target.jpg")
                manager = _manager(root)
                for path in cases:
                    with self.subTest(path=path):
                        outcome = await manager.execute_call(
                            ToolCall(
                                id=f"view-{path}",
                                name="shell.view_image",
                                arguments={"path": path},
                            ),
                            context=ToolCallContext(),
                        )
                        self.assertIsInstance(outcome, ToolCallError)
            finally:
                outside.unlink(missing_ok=True)

    async def test_progressive_jpeg_is_attached_with_exact_bytes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "progressive.jpg").write_bytes(
                VALID_PROGRESSIVE_JPEG_BYTES
            )
            manager = _manager(root)

            outcome = await manager.execute_call(
                ToolCall(
                    id="view-progressive",
                    name="shell.view_image",
                    arguments={"path": "progressive.jpg"},
                ),
                context=ToolCallContext(),
            )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        image = outcome.content[1]
        assert isinstance(image, ToolResultImage)
        self.assertEqual(image.data, VALID_PROGRESSIVE_JPEG_BYTES)
        self.assertEqual((image.width, image.height), (17, 11))

    async def test_oversized_and_malformed_complete_images_are_rejected(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "large.png").write_bytes(
                valid_png_bytes(width=3, height=3)
            )
            (root / "truncated.png").write_bytes(
                valid_png_bytes(width=1, height=1)[:-12]
            )
            (root / "invalid-deflate.png").write_bytes(
                _png_with_invalid_compressed_pixels()
            )
            (root / "invalid-filter.png").write_bytes(
                _png_with_invalid_filter()
            )
            (root / "invalid-entropy.jpg").write_bytes(
                _jpeg_with_invalid_entropy_marker()
            )
            (root / "byte-limit.jpg").write_bytes(VALID_JPEG_BYTES)
            limited = _manager(
                root,
                max_raster_long_edge_pixels=2,
                max_ocr_pixels=4,
            )
            malformed = _manager(root)
            byte_limited = _manager(root, max_ocr_input_bytes=10)
            for path, manager in (
                ("large.png", limited),
                ("truncated.png", malformed),
                ("invalid-deflate.png", malformed),
                ("invalid-filter.png", malformed),
                ("invalid-entropy.jpg", malformed),
                ("byte-limit.jpg", byte_limited),
            ):
                with self.subTest(path=path):
                    outcome = await manager.execute_call(
                        ToolCall(
                            id=f"view-{path}",
                            name="shell.view_image",
                            arguments={"path": path},
                        ),
                        context=ToolCallContext(),
                    )
                    self.assertIsInstance(outcome, ToolCallError)

    def test_jpeg_fails_explicitly_without_optional_decoder(self) -> None:
        with (
            patch(
                "avalan.tool.shell.filesystem.import_module",
                side_effect=ImportError,
            ),
            self.assertRaisesRegex(ValueError, "Pillow dependency"),
        ):
            validate_supported_image_bytes(VALID_JPEG_BYTES)

    def test_png_validator_rejects_malformed_chunk_sequences(self) -> None:
        """Reject each security-relevant malformed PNG stream shape."""
        valid_header = _png_header()
        malformed = (
            b"\x89PNG\r\n\x1a\n",
            _png_stream(_png_chunk(b"IHDR", valid_header), b"\x00" * 11),
            _png_stream(_png_chunk(b"1HDR", valid_header)),
            _png_stream(_png_chunk(b"IHdR", valid_header)),
            _png_stream(
                _png_chunk(b"IHDR", valid_header),
                pack(">I", 99) + b"IDAT" + b"\x00" * 4,
            ),
            _png_stream(_png_chunk(b"IHDR", valid_header)[:-1] + b"\x00"),
            _png_stream(_png_chunk(b"IDAT", b"")),
            _png_stream(_png_chunk(b"IHDR", _png_header(width=0))),
            _png_stream(_png_chunk(b"IHDR", _png_header(bit_depth=7))),
            _png_stream(
                _png_chunk(b"IHDR", valid_header),
                _png_chunk(b"IHDR", valid_header),
            ),
            _png_stream(
                _png_chunk(b"IHDR", valid_header),
                _png_chunk(b"ABCD", b""),
            ),
            _png_stream(
                _png_chunk(b"IHDR", _png_header(color_type=0)),
                _png_chunk(b"PLTE", b"\x00\x00\x00"),
            ),
            _png_stream(
                _png_chunk(
                    b"IHDR",
                    _png_header(bit_depth=1, color_type=3),
                ),
                _png_chunk(b"PLTE", b"\x00\x00\x00"),
                _png_chunk(b"IEND", b""),
            ),
            _png_stream(
                _png_chunk(b"IHDR", valid_header),
                _png_chunk(b"IDAT", b""),
                _png_chunk(b"abCd", b""),
                _png_chunk(b"IDAT", b""),
                _png_chunk(b"IEND", b""),
            ),
            _png_stream(
                _png_chunk(b"IHDR", valid_header),
                _png_chunk(b"IEND", b"unexpected"),
            ),
            _png_stream(
                _png_chunk(b"IHDR", valid_header),
                _png_chunk(b"IDAT", compress(b"\x00\x00\x00\x00")[:-1]),
                _png_chunk(b"IEND", b""),
            ),
        )

        for data in malformed:
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(
                    ValueError,
                    "malformed PNG",
                ),
            ):
                validate_supported_image_bytes(data)

        with (
            patch(
                "avalan.tool.shell.filesystem._png_scanline_bytes",
                side_effect=((4,), (3,)),
            ),
            self.assertRaisesRegex(ValueError, "malformed PNG"),
        ):
            shell_filesystem._validate_png(
                valid_png_bytes(width=1, height=1),
                max_long_edge_pixels=None,
                max_pixels=None,
            )

    def test_png_validator_accepts_adam7_and_enforces_pixel_limit(
        self,
    ) -> None:
        """Validate interlaced scanlines while retaining configured limits."""
        interlaced = _png_stream(
            _png_chunk(b"IHDR", _png_header(interlace=1)),
            _png_chunk(b"IDAT", compress(b"\x00\x00\x00\x00")),
            _png_chunk(b"IEND", b""),
        )

        self.assertEqual(
            validate_supported_image_bytes(interlaced),
            ("image/png", 1, 1),
        )
        with self.assertRaisesRegex(ValueError, "configured limits"):
            validate_supported_image_bytes(
                valid_png_bytes(width=2, height=2),
                max_pixels=3,
            )

    def test_jpeg_validator_rejects_malformed_markers_and_scans(self) -> None:
        """Reject incomplete, ambiguous, and invalid JPEG marker framing."""
        jpeg = shell_filesystem.JPEG_SIGNATURE
        valid_frame = _jpeg_frame(component_count=1)
        two_component_frame = _jpeg_frame(component_count=2)
        malformed = (
            jpeg + b"x",
            jpeg + b"\xff",
            jpeg + b"\xff\xd9x",
            jpeg + b"\xff\x01",
            jpeg + b"\xff\xe0",
            jpeg + b"\xff\xe0\x00\x01",
            jpeg + b"\xff\xc0\x00\x08\x08\x00\x01\x00\x01\x00",
            jpeg + b"\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x00\x00",
            jpeg + _jpeg_scan(component_count=1),
            jpeg + valid_frame + _jpeg_scan(component_count=0),
            jpeg
            + two_component_frame
            + _jpeg_scan(
                component_count=2,
                components=(1, 1),
            ),
            jpeg + _jpeg_frame(component_count=1, width=0),
            jpeg + valid_frame,
        )

        for data in malformed:
            with (
                self.subTest(data=data),
                self.assertRaisesRegex(
                    ValueError,
                    "malformed JPEG",
                ),
            ):
                shell_filesystem._validate_jpeg(data)

        for data, offset in (
            (b"entropy", 0),
            (b"\xff", 0),
            (b"\xff\xff", 0),
            (b"\xff\x00", 0),
        ):
            with (
                self.subTest(scan=data),
                self.assertRaisesRegex(
                    ValueError,
                    "malformed JPEG",
                ),
            ):
                shell_filesystem._jpeg_scan_marker_offset(data, offset)
        self.assertEqual(
            shell_filesystem._jpeg_scan_marker_offset(
                b"\xff\x00\xff\xd0\xff\xd9",
                0,
            ),
            4,
        )

    def test_jpeg_pixel_decoder_rejects_a_format_mismatch(self) -> None:
        """Treat decoder format drift as malformed model-visible image data."""

        class FakeImage:
            format = "PNG"
            size = (16, 16)

            def __enter__(self) -> "FakeImage":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def load(self) -> None:
                return None

        class FakeImageModule:
            @staticmethod
            def open(_: object) -> FakeImage:
                return FakeImage()

        with (
            patch(
                "avalan.tool.shell.filesystem.import_module",
                return_value=FakeImageModule,
            ),
            self.assertRaisesRegex(ValueError, "malformed JPEG"),
        ):
            shell_filesystem._validate_jpeg_pixels(
                VALID_JPEG_BYTES,
                (16, 16),
            )

    async def test_nonlocal_targets_reject_without_host_fallback(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "image.jpg").write_bytes(VALID_JPEG_BYTES)
            for execution_mode in ("sandbox", "container"):
                with self.subTest(execution_mode=execution_mode):
                    policy = ExecutionPolicy(
                        ShellToolSettings(
                            workspace_root=str(root),
                            allow_media_tools=True,
                            execution_mode=execution_mode,
                        )
                    )
                    with self.assertRaises(ShellPolicyDenied) as raised:
                        await policy.resolve_view_image_path("image.jpg")
                    self.assertIn(
                        "host fallback is disabled",
                        str(raised.exception),
                    )

    async def test_view_image_policy_rejects_invalid_arguments_and_gate(
        self,
    ) -> None:
        """Validate caller input before resolving a filesystem image path."""
        policy = ExecutionPolicy(ShellToolSettings())

        with self.assertRaisesRegex(TypeError, "path must be a string"):
            await policy.resolve_view_image_path(1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "cwd must be a string"):
            await policy.resolve_view_image_path(
                "image.png",
                cwd=1,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ShellPolicyDenied, "media tools"):
            await policy.resolve_view_image_path("image.png")

    async def test_path_swap_cannot_read_an_out_of_scope_symlink(self) -> None:
        with (
            TemporaryDirectory() as temporary_directory,
            TemporaryDirectory() as outside_directory,
        ):
            root = Path(temporary_directory)
            outside = Path(outside_directory) / "outside.jpg"
            outside.write_bytes(VALID_JPEG_BYTES)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
            )
            policy = _ReplacingPolicy(settings, outside)
            tool = ViewImageTool(settings=settings, policy=policy)
            (root / "image.jpg").write_bytes(VALID_JPEG_BYTES)

            with self.assertRaisesRegex(
                ValueError,
                "changed while being read",
            ):
                await tool(
                    "image.jpg",
                    context=ToolCallContext(),
                )

    async def test_generated_image_reopen_rejects_a_path_swap(self) -> None:
        with (
            TemporaryDirectory() as temporary_directory,
            TemporaryDirectory() as outside_directory,
        ):
            root = Path(temporary_directory)
            settings = ShellToolSettings(workspace_root=str(root))
            generated_root = root / settings.materialized_input_files_dir
            generated_root.mkdir(parents=True)
            generated_path = generated_root / "montage.jpg"
            generated_path.write_bytes(VALID_JPEG_BYTES)
            outside = Path(outside_directory) / "private.jpg"
            outside.write_bytes(b"private host file")
            generated = GeneratedFile(
                display_path="montage.jpg",
                media_type="image/jpeg",
                suffix=".jpg",
                bytes=len(VALID_JPEG_BYTES),
                sha256=sha256(VALID_JPEG_BYTES).hexdigest(),
                width=16,
                height=16,
                metadata={
                    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY: str(
                        generated_path
                    )
                },
            )

            async def inspect_then_replace(
                path: str | Path,
            ) -> ShellPathMetadata:
                metadata = await inspect_path(path)
                Path(path).unlink()
                Path(path).symlink_to(outside)
                return metadata

            with (
                patch(
                    "avalan.tool.shell.tool_images.inspect_path",
                    side_effect=inspect_then_replace,
                ),
                self.assertRaisesRegex(
                    ValueError,
                    "changed while being read",
                ),
            ):
                await _generated_image_bytes(
                    generated,
                    backend="local",
                    settings=settings,
                )

    async def test_montage_skips_nonvisible_generated_artifacts(self) -> None:
        """Do not claim a model-visible montage without attachable pixels."""
        settings = ShellToolSettings()
        formatted = _formatted_montage(
            (
                GeneratedFile(
                    display_path="notes.txt",
                    media_type="text/plain",
                    suffix=".txt",
                    bytes=4,
                ),
                GeneratedFile(
                    display_path="partial.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=4,
                    truncated=True,
                ),
            )
        )

        outcome = await montage_tool_result(formatted, settings=settings)

        self.assertEqual(outcome.result, formatted)
        self.assertEqual(outcome.content, ())

    async def test_generated_image_delivery_rejects_untrusted_artifacts(
        self,
    ) -> None:
        """Retain target and artifact boundaries before attaching pixels."""
        pixels = valid_png_bytes(width=1, height=1)
        settings = ShellToolSettings()
        unavailable = GeneratedFile(
            display_path="image.png",
            media_type="image/png",
            suffix=".png",
            bytes=len(pixels),
            width=1,
            height=1,
        )

        unavailable_block = await _generated_image_block(
            unavailable,
            backend="sandbox",
            settings=settings,
            detail=ToolResultImageDetail.AUTO,
        )
        self.assertFalse(unavailable_block.available)
        self.assertIn(
            "did not return image bytes",
            unavailable_block.unavailable_reason or "",
        )

        invalid_base64 = GeneratedFile(
            display_path="image.png",
            media_type="image/png",
            suffix=".png",
            bytes=len(pixels),
            content_base64="%",
        )
        with self.assertRaisesRegex(ValueError, "invalid base64"):
            await _generated_image_bytes(
                invalid_base64,
                backend="local",
                settings=settings,
            )
        self.assertIsNone(
            await _generated_image_bytes(
                unavailable,
                backend="container",
                settings=settings,
            )
        )
        self.assertIsNone(
            await _generated_image_bytes(
                unavailable,
                backend="local",
                settings=settings,
            )
        )

        with (
            TemporaryDirectory() as temporary_directory,
            TemporaryDirectory() as outside_directory,
        ):
            root = Path(temporary_directory)
            local_settings = ShellToolSettings(workspace_root=str(root))
            materialized_root = (
                root / local_settings.materialized_input_files_dir
            )
            materialized_root.mkdir(parents=True)
            directory = materialized_root / "directory"
            directory.mkdir()
            outside = Path(outside_directory) / "outside-image.png"
            outside.write_bytes(pixels)
            outside_artifact = GeneratedFile(
                display_path="outside.png",
                media_type="image/png",
                suffix=".png",
                bytes=len(pixels),
                metadata={
                    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY: str(outside)
                },
            )
            directory_artifact = GeneratedFile(
                display_path="directory.png",
                media_type="image/png",
                suffix=".png",
                bytes=0,
                metadata={
                    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY: str(
                        directory
                    )
                },
            )
            self.assertIsNone(
                await _generated_image_bytes(
                    outside_artifact,
                    backend="local",
                    settings=local_settings,
                )
            )
            self.assertIsNone(
                await _generated_image_bytes(
                    directory_artifact,
                    backend="local",
                    settings=local_settings,
                )
            )

    async def test_generated_image_metadata_must_match_verified_pixels(
        self,
    ) -> None:
        """Reject generated-image labels that disagree with validated bytes."""
        pixels = valid_png_bytes(width=1, height=1)
        settings = ShellToolSettings()
        mismatched_type = GeneratedFile(
            display_path="image.jpg",
            media_type="image/jpeg",
            suffix=".jpg",
            bytes=len(pixels),
            width=1,
            height=1,
            transient_content=pixels,
        )
        mismatched_dimensions = GeneratedFile(
            display_path="image.png",
            media_type="image/png",
            suffix=".png",
            bytes=len(pixels),
            width=2,
            height=1,
            transient_content=pixels,
        )

        with self.assertRaisesRegex(ValueError, "media type"):
            await _generated_image_block(
                mismatched_type,
                backend="local",
                settings=settings,
                detail=ToolResultImageDetail.AUTO,
            )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            await _generated_image_block(
                mismatched_dimensions,
                backend="local",
                settings=settings,
                detail=ToolResultImageDetail.AUTO,
            )
        with self.assertRaisesRegex(ValueError, "configured limits"):
            _validate_dimensions(
                3,
                3,
                ShellToolSettings(
                    max_raster_long_edge_pixels=2,
                    max_ocr_pixels=4,
                ),
            )
        self.assertFalse(_is_relative_to(Path("/outside"), Path("/workspace")))


def _manager(
    root: Path,
    *,
    max_raster_long_edge_pixels: int = 2048,
    max_ocr_input_bytes: int = 26214400,
    max_ocr_pixels: int = 20000000,
) -> ToolManager:
    settings = ShellToolSettings(
        workspace_root=str(root),
        allow_media_tools=True,
        max_raster_long_edge_pixels=max_raster_long_edge_pixels,
        max_ocr_input_bytes=max_ocr_input_bytes,
        max_ocr_pixels=max_ocr_pixels,
    )
    return ToolManager.create_instance(
        available_toolsets=[ShellToolSet(settings=settings)],
        enable_tools=["shell.view_image"],
        settings=ToolManagerSettings(),
    )


def _formatted_montage(
    generated_files: tuple[GeneratedFile, ...],
) -> ShellFormattedResult:
    return ShellFormattedResult(
        "montage metadata",
        ExecutionResult(
            backend="local",
            tool_name="shell.montage",
            command="montage",
            argv=("montage",),
            display_argv=("montage",),
            cwd="/workspace",
            display_cwd=".",
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="",
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.GENERATED_FILES,
            generated_files=generated_files,
        ),
    )


def _png_with_invalid_compressed_pixels() -> bytes:
    data = bytearray(valid_png_bytes(width=1, height=1))
    chunk_type_index = data.index(b"IDAT")
    chunk_length = int.from_bytes(
        data[chunk_type_index - 4 : chunk_type_index],
        "big",
    )
    content_start = chunk_type_index + 4
    content_end = content_start + chunk_length
    data[content_start:content_end] = b"x" * chunk_length
    checksum = crc32(data[chunk_type_index:content_end]) & 0xFFFFFFFF
    data[content_end : content_end + 4] = pack(">I", checksum)
    return bytes(data)


def _png_header(
    *,
    width: int = 1,
    height: int = 1,
    bit_depth: int = 8,
    color_type: int = 2,
    interlace: int = 0,
) -> bytes:
    return pack(
        ">IIBBBBB",
        width,
        height,
        bit_depth,
        color_type,
        0,
        0,
        interlace,
    )


def _png_stream(*chunks: bytes) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def _png_with_invalid_filter() -> bytes:
    header = pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + b"".join(
        (
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", compress(b"\x05\x00\x00\x00")),
            _png_chunk(b"IEND", b""),
        )
    )


def _jpeg_with_invalid_entropy_marker() -> bytes:
    scan_marker = VALID_JPEG_BYTES.index(b"\xff\xda")
    segment_length = int.from_bytes(
        VALID_JPEG_BYTES[scan_marker + 2 : scan_marker + 4],
        "big",
    )
    entropy_offset = scan_marker + 2 + segment_length
    return VALID_JPEG_BYTES[:entropy_offset] + b"\xff\xc0\x00" + b"\xff\xd9"


def _png_chunk(chunk_type: bytes, content: bytes) -> bytes:
    checksum = crc32(chunk_type + content) & 0xFFFFFFFF
    return (
        pack(">I", len(content)) + chunk_type + content + pack(">I", checksum)
    )


def _jpeg_frame(
    *,
    component_count: int,
    width: int = 1,
    height: int = 1,
) -> bytes:
    components = b"".join(
        bytes((identifier, 0x11, 0))
        for identifier in range(1, component_count + 1)
    )
    data = b"\x08" + pack(">HHB", height, width, component_count) + components
    return b"\xff\xc0" + pack(">H", len(data) + 2) + data


def _jpeg_scan(
    *,
    component_count: int,
    components: tuple[int, ...] | None = None,
) -> bytes:
    identifiers = components or tuple(range(1, component_count + 1))
    data = (
        bytes((component_count,))
        + b"".join(bytes((identifier, 0)) for identifier in identifiers)
        + b"\x00\x3f\x00"
    )
    return b"\xff\xda" + pack(">H", len(data) + 2) + data


class _ReplacingPolicy(ExecutionPolicy):
    def __init__(self, settings: ShellToolSettings, outside: Path) -> None:
        super().__init__(settings)
        self._outside = outside

    async def resolve_view_image_path(
        self,
        path: str,
        *,
        cwd: str | None = None,
    ) -> tuple[Path, str, ShellPathMetadata]:
        (
            resolved_path,
            display_path,
            metadata,
        ) = await super().resolve_view_image_path(path, cwd=cwd)
        resolved_path.unlink()
        resolved_path.symlink_to(self._outside)
        return resolved_path, display_path, metadata


if __name__ == "__main__":
    main()
