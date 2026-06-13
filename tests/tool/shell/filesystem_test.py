from asyncio import run as asyncio_run
from os import stat_result
from pathlib import Path
from stat import S_IFIFO
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main
from unittest.mock import AsyncMock, patch

from avalan.tool.shell import filesystem as shell_filesystem
from avalan.tool.shell.filesystem import (
    DEFAULT_SIGNATURE_BYTES,
    PNG_SIGNATURE,
    ShellPathMetadata,
    ensure_file_size_at_most,
    file_digest_and_base64,
    file_size,
    inspect_path,
    list_directory,
    make_directory,
    private_temp_directory,
    probe_image_dimensions,
    read_image_signature,
    read_pdf_signature,
    read_signature,
    remove_file,
    remove_tree,
    resolve_policy_path,
    signature_is_binary,
    sniff_binary,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"


class ShellFilesystemTest(IsolatedAsyncioTestCase):
    async def test_inspect_path_reports_regular_files_and_directories(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            file_path = root / "visible.txt"
            file_path.write_text("hello", encoding="utf-8")

            file_metadata = await inspect_path(file_path)
            directory_metadata = await inspect_path(root)

        self.assertEqual(file_metadata.path.name, "visible.txt")
        self.assertEqual(file_metadata.size, 5)
        self.assertGreaterEqual(file_metadata.hardlink_count, 1)
        self.assertTrue(file_metadata.is_file)
        self.assertFalse(file_metadata.is_directory)
        self.assertFalse(file_metadata.is_symlink)
        self.assertFalse(file_metadata.is_special_file)
        self.assertTrue(directory_metadata.is_directory)
        self.assertFalse(directory_metadata.is_special_file)

    async def test_inspect_path_reports_symlinks_without_following(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            file_path = root / "visible.txt"
            file_path.write_text("hello", encoding="utf-8")
            symlink_path = root / "linked.txt"
            symlink_path.symlink_to(file_path)

            metadata = await inspect_path(symlink_path)

        self.assertTrue(metadata.is_symlink)
        self.assertFalse(metadata.is_file)
        self.assertFalse(metadata.is_special_file)

    async def test_inspect_path_preserves_missing_and_permission_errors(
        self,
    ) -> None:
        with self.assertRaises(FileNotFoundError):
            await inspect_path(FIXTURE_ROOT / "filesystem" / "missing.txt")

        with patch(
            "avalan.tool.shell.filesystem._stat_path",
            new=AsyncMock(side_effect=PermissionError("denied")),
        ):
            with self.assertRaises(PermissionError):
                await inspect_path("locked.txt")

    async def test_inspect_path_reports_mocked_special_files(self) -> None:
        fake_stat = stat_result(
            (
                S_IFIFO | 0o600,
                0,
                0,
                0,
                0,
                0,
                123,
                0,
                0,
                0,
            )
        )

        with (
            patch(
                "avalan.tool.shell.filesystem._stat_path",
                new=AsyncMock(return_value=fake_stat),
            ),
            patch(
                "avalan.tool.shell.filesystem._resolve_path",
                new=AsyncMock(return_value=Path("/workspace/fifo")),
            ),
        ):
            metadata = await inspect_path("fifo")

        self.assertEqual(metadata.size, 123)
        self.assertTrue(metadata.is_special_file)
        self.assertFalse(metadata.is_file)
        self.assertFalse(metadata.is_directory)
        self.assertFalse(metadata.is_symlink)

    async def test_resolve_policy_path_does_not_expand_user_syntax(
        self,
    ) -> None:
        resolved = await resolve_policy_path("~")

        self.assertEqual(resolved.name, "~")

    async def test_read_signature_uses_default_and_explicit_caps(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "large.bin"
            path.write_bytes(b"a" * (DEFAULT_SIGNATURE_BYTES + 10))

            default_signature = await read_signature(path)
            explicit_signature = await read_signature(path, max_bytes=3)

        self.assertEqual(len(default_signature), DEFAULT_SIGNATURE_BYTES)
        self.assertEqual(explicit_signature, b"aaa")

    async def test_sniff_binary_uses_bounded_signature(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            text_path = Path(temporary_directory) / "large.txt"
            binary_path = Path(temporary_directory) / "binary.bin"
            text_path.write_bytes(b"a" * (DEFAULT_SIGNATURE_BYTES + 10))
            binary_path.write_bytes(b"a" * DEFAULT_SIGNATURE_BYTES + b"\x00")

            with patch(
                "avalan.tool.shell.filesystem._read_bytes_prefix",
                wraps=shell_filesystem._read_bytes_prefix,
            ) as read_prefix:
                is_text_binary = await sniff_binary(text_path)
                is_binary_binary = await sniff_binary(binary_path)

        self.assertFalse(is_text_binary)
        self.assertFalse(is_binary_binary)
        self.assertEqual(read_prefix.call_count, 2)
        self.assertEqual(
            read_prefix.call_args_list[0].args,
            (text_path, DEFAULT_SIGNATURE_BYTES),
        )

    def test_signature_binary_detection_rejects_nul_and_invalid_utf8(
        self,
    ) -> None:
        self.assertFalse(signature_is_binary(b"text\n"))
        self.assertTrue(signature_is_binary(b"text\x00more"))
        self.assertTrue(signature_is_binary(b"text\xff"))
        with self.assertRaises(AssertionError):
            signature_is_binary("text")  # type: ignore[arg-type]

    async def test_file_size_helpers_return_and_bound_sizes(self) -> None:
        path = FIXTURE_ROOT / "filesystem" / "visible.txt"

        size = await file_size(path)
        accepted = await ensure_file_size_at_most(path, max_bytes=size)

        self.assertEqual(accepted, size)
        with self.assertRaises(AssertionError):
            await ensure_file_size_at_most(path, max_bytes=size - 1)

    async def test_pdf_and_image_signature_helpers_read_bounded_prefixes(
        self,
    ) -> None:
        pdf_signature = await read_pdf_signature(
            FIXTURE_ROOT / "media" / "small.pdf"
        )
        image_signature = await read_image_signature(
            FIXTURE_ROOT / "ocr" / "small.pgm"
        )

        self.assertEqual(pdf_signature, b"%PDF-")
        self.assertTrue(image_signature.startswith(b"P2\n3 3\n"))

    async def test_probe_image_dimensions_reads_pnm_and_png_headers(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            png_path = Path(temporary_directory) / "image.png"
            png_path.write_bytes(_png_header(width=7, height=11))

            pnm_dimensions = await probe_image_dimensions(
                FIXTURE_ROOT / "ocr" / "small.pgm"
            )
            png_dimensions = await probe_image_dimensions(png_path)

        self.assertEqual(pnm_dimensions, (3, 3))
        self.assertEqual(png_dimensions, (7, 11))
        self.assertIsNone(
            await probe_image_dimensions(
                FIXTURE_ROOT / "ocr" / "unsupported-signature.dat"
            )
        )

    async def test_private_temp_directory_cleans_up_after_exit(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            async with private_temp_directory(directory=root) as temp_path:
                child_path = temp_path / "value.txt"
                child_path.write_text("value", encoding="utf-8")
                self.assertTrue(child_path.is_file())

            self.assertFalse(temp_path.exists())

    async def test_private_temp_directory_cleans_up_after_error(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            temp_path: Path | None = None
            with self.assertRaises(RuntimeError):
                async with private_temp_directory(
                    prefix="shell-test-",
                    directory=root,
                ) as created_path:
                    temp_path = created_path
                    raise RuntimeError("stop")

            self.assertIsNotNone(temp_path)
            self.assertFalse(temp_path.exists())

    async def test_private_temp_directory_ignores_missing_cleanup_path(
        self,
    ) -> None:
        with (
            patch(
                "avalan.tool.shell.filesystem._remove_tree",
                side_effect=OSError("remove failed"),
            ),
            patch(
                "avalan.tool.shell.filesystem.inspect_path",
                side_effect=OSError("missing"),
            ),
        ):
            async with private_temp_directory():
                pass

    async def test_private_temp_directory_ignores_directory_cleanup_error(
        self,
    ) -> None:
        metadata = ShellPathMetadata(
            path=Path("/tmp/generated"),
            resolved_path=Path("/tmp/generated"),
            mode=0o700,
            size=0,
            is_file=False,
            is_directory=True,
            is_symlink=False,
            is_special_file=False,
        )

        with (
            patch(
                "avalan.tool.shell.filesystem._remove_tree",
                side_effect=OSError("remove failed"),
            ),
            patch(
                "avalan.tool.shell.filesystem.inspect_path",
                return_value=metadata,
            ),
        ):
            async with private_temp_directory():
                pass

    async def test_private_temp_directory_preserves_error_when_inspect_fails(
        self,
    ) -> None:
        with (
            patch(
                "avalan.tool.shell.filesystem._remove_tree",
                side_effect=OSError("remove failed"),
            ),
            patch(
                "avalan.tool.shell.filesystem.inspect_path",
                side_effect=OSError("missing"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                async with private_temp_directory():
                    raise RuntimeError("body failed")

    async def test_private_temp_directory_preserves_error_for_directory(
        self,
    ) -> None:
        metadata = ShellPathMetadata(
            path=Path("/tmp/generated"),
            resolved_path=Path("/tmp/generated"),
            mode=0o700,
            size=0,
            is_file=False,
            is_directory=True,
            is_symlink=False,
            is_special_file=False,
        )

        with (
            patch(
                "avalan.tool.shell.filesystem._remove_tree",
                side_effect=OSError("remove failed"),
            ),
            patch(
                "avalan.tool.shell.filesystem.inspect_path",
                return_value=metadata,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "body failed"):
                async with private_temp_directory():
                    raise RuntimeError("body failed")

    async def test_private_temp_directory_ignores_unlink_cleanup_error(
        self,
    ) -> None:
        metadata = ShellPathMetadata(
            path=Path("/tmp/generated"),
            resolved_path=Path("/tmp/generated"),
            mode=0o600,
            size=1,
            is_file=True,
            is_directory=False,
            is_symlink=False,
            is_special_file=False,
        )

        with (
            patch(
                "avalan.tool.shell.filesystem._remove_tree",
                side_effect=OSError("remove failed"),
            ),
            patch(
                "avalan.tool.shell.filesystem.inspect_path",
                return_value=metadata,
            ),
            patch(
                "avalan.tool.shell.filesystem._remove_file",
                side_effect=OSError("unlink failed"),
            ),
        ):
            async with private_temp_directory():
                pass

    async def test_generated_output_filesystem_helpers(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directory = await make_directory(root / "generated")
            file_path = directory / "page-1.png"
            file_path.write_bytes(b"abc")
            transient_path = directory / "transient.txt"
            transient_path.write_text("remove", encoding="utf-8")

            entries = await list_directory(directory)
            digest, inline = await file_digest_and_base64(
                file_path,
                chunk_size=2,
                max_inline_bytes=3,
            )
            await remove_file(transient_path)
            await remove_tree(directory)

            self.assertEqual(set(entries), {file_path, transient_path})
            self.assertEqual(
                digest,
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(inline, "YWJj")
            self.assertFalse(transient_path.exists())
            self.assertFalse(directory.exists())

    async def test_rejects_invalid_public_inputs(self) -> None:
        with self.assertRaises(AssertionError):
            await resolve_policy_path(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await inspect_path(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await read_signature(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await read_signature("value.bin", max_bytes=True)  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await ensure_file_size_at_most("value.bin", max_bytes=-1)
        with self.assertRaises(AssertionError):
            await file_digest_and_base64(
                object(),  # type: ignore[arg-type]
                chunk_size=1,
                max_inline_bytes=1,
            )
        with self.assertRaises(AssertionError):
            await list_directory(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await make_directory(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await remove_tree(object())  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            await remove_file(object())  # type: ignore[arg-type]


class ShellFilesystemPrivateProbeTest(TestCase):
    def test_metadata_rejects_invalid_fields(self) -> None:
        valid = {
            "path": Path("input.txt"),
            "resolved_path": Path("/workspace/input.txt"),
            "mode": 0,
            "size": 0,
            "is_file": True,
            "is_directory": False,
            "is_symlink": False,
            "is_special_file": False,
        }
        invalid_kwargs = (
            {"path": "input.txt"},
            {"resolved_path": "/workspace/input.txt"},
            {"mode": -1},
            {"size": -1},
            {"hardlink_count": -1},
            {"is_file": 1},
            {"is_directory": 0},
            {"is_symlink": 1},
            {"is_special_file": 0},
        )
        for kwargs in invalid_kwargs:
            with self.subTest(kwargs=kwargs):
                values = dict(valid)
                values.update(kwargs)
                with self.assertRaises(AssertionError):
                    ShellPathMetadata(**values)  # type: ignore[arg-type]

    def test_png_probe_rejects_short_zero_or_non_png_headers(self) -> None:
        self.assertIsNone(shell_filesystem._probe_png_dimensions(b""))
        self.assertIsNone(
            shell_filesystem._probe_png_dimensions(PNG_SIGNATURE)
        )
        self.assertIsNone(
            shell_filesystem._probe_png_dimensions(
                _png_header(width=0, height=1)
            )
        )
        with self.assertRaises(AssertionError):
            shell_filesystem._probe_png_dimensions("png")  # type: ignore[arg-type]

    def test_pnm_probe_rejects_incomplete_invalid_or_zero_dimensions(
        self,
    ) -> None:
        self.assertIsNone(shell_filesystem._probe_pnm_dimensions(b""))
        self.assertIsNone(shell_filesystem._probe_pnm_dimensions(b"P2\n3\n"))
        self.assertIsNone(
            shell_filesystem._probe_pnm_dimensions(b"P2\nabc 3\n255\n")
        )
        self.assertIsNone(
            shell_filesystem._probe_pnm_dimensions(b"P2\n0 3\n255\n")
        )
        self.assertEqual(
            shell_filesystem._probe_pnm_dimensions(
                b"P6\n# comment\n4 5\n255\n"
            ),
            (4, 5),
        )
        with self.assertRaises(AssertionError):
            shell_filesystem._probe_pnm_dimensions("pnm")  # type: ignore[arg-type]

    def test_async_context_manager_rejects_invalid_prefix(self) -> None:
        with self.assertRaises(AssertionError):
            asyncio_run(_enter_private_temp_directory_with_empty_prefix())


async def _enter_private_temp_directory_with_empty_prefix() -> None:
    async with private_temp_directory(prefix=""):
        pass


def _png_header(*, width: int, height: int) -> bytes:
    return (
        PNG_SIGNATURE
        + b"\x00\x00\x00\r"
        + b"IHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


if __name__ == "__main__":
    main()
