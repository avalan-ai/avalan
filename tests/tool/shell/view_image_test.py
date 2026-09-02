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
from avalan.tool.shell.entities import (
    GENERATED_FILE_MATERIALIZED_PATH_METADATA_KEY,
)
from avalan.tool.shell.filesystem import (
    inspect_path,
    validate_supported_image_bytes,
)
from avalan.tool.shell.tool_images import _generated_image_bytes


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
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            outside = root.parent / "view-image-outside.jpg"
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
        resolved_path, display_path, metadata = (
            await super().resolve_view_image_path(path, cwd=cwd)
        )
        resolved_path.unlink()
        resolved_path.symlink_to(self._outside)
        return resolved_path, display_path, metadata


if __name__ == "__main__":
    main()
