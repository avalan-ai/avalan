from asyncio import run as asyncio_run
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from avalan.filesystem import (
    assert_text_encoding,
    make_private_directory,
    read_bytes,
    read_bytes_prefix,
    read_text,
    remove_tree,
    resolve_path,
    stat_path,
    write_bytes,
    write_text,
)


class FilesystemTestCase(TestCase):
    def test_read_text_defaults_to_utf8_and_accepts_encoding(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.txt"
            path.write_text("hello", encoding="utf-8")
            latin_path = Path(temporary_directory) / "latin.txt"
            latin_path.write_bytes("Café".encode("latin-1"))

            default = asyncio_run(read_text(path))
            latin = asyncio_run(read_text(latin_path, encoding="latin-1"))

        self.assertEqual(default, "hello")
        self.assertEqual(latin, "Café")

    def test_read_bytes_returns_file_contents(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.bin"
            path.write_bytes(b"abc")

            content = asyncio_run(read_bytes(path))

        self.assertEqual(content, b"abc")

    def test_read_bytes_prefix_limits_file_contents(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.bin"
            path.write_bytes(b"abcdef")

            content = asyncio_run(read_bytes_prefix(path, 3))
            empty = asyncio_run(read_bytes_prefix(path, 0))

        self.assertEqual(content, b"abc")
        self.assertEqual(empty, b"")

    def test_stat_path_supports_symlink_policy(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "value.txt"
            path.write_text("hello", encoding="utf-8")
            symlink = root / "link.txt"
            symlink.symlink_to(path)

            followed = asyncio_run(stat_path(symlink))
            not_followed = asyncio_run(
                stat_path(symlink, follow_symlinks=False)
            )

        self.assertNotEqual(followed.st_mode, not_followed.st_mode)

    def test_resolve_path_does_not_expand_user_syntax(self) -> None:
        resolved = asyncio_run(resolve_path("~"))

        self.assertEqual(resolved.name, "~")

    def test_private_directory_helpers_create_and_remove_tree(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)

            path = asyncio_run(
                make_private_directory(
                    prefix="avalan-test-",
                    directory=root,
                )
            )
            child = path / "child.txt"
            child.write_text("value", encoding="utf-8")

            self.assertTrue(path.is_dir())
            self.assertTrue(child.is_file())
            asyncio_run(remove_tree(path))
            self.assertFalse(path.exists())

    def test_write_text_uses_default_and_explicit_encoding(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.txt"
            latin_path = Path(temporary_directory) / "latin.txt"

            count = asyncio_run(write_text(path, "hello"))
            latin_count = asyncio_run(
                write_text(latin_path, "Café", encoding="latin-1")
            )

            default = path.read_text(encoding="utf-8")
            latin = latin_path.read_bytes()

        self.assertEqual(count, 5)
        self.assertEqual(latin_count, 4)
        self.assertEqual(default, "hello")
        self.assertEqual(latin, "Café".encode("latin-1"))

    def test_write_bytes_persists_file_contents(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "value.bin"

            count = asyncio_run(write_bytes(path, b"abc"))

            content = path.read_bytes()

        self.assertEqual(count, 3)
        self.assertEqual(content, b"abc")

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(AssertionError):
            assert_text_encoding("")
        with self.assertRaises(AssertionError):
            asyncio_run(read_text(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(read_text("value.txt", encoding=""))
        with self.assertRaises(AssertionError):
            asyncio_run(read_bytes(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(read_bytes_prefix(object(), 1))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(read_bytes_prefix("value.bin", True))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(read_bytes_prefix("value.bin", -1))
        with self.assertRaises(AssertionError):
            asyncio_run(stat_path(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(stat_path("value.txt", follow_symlinks=1))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(resolve_path(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(resolve_path("value.txt", strict=1))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(make_private_directory(prefix=""))
        with self.assertRaises(AssertionError):
            asyncio_run(
                make_private_directory(
                    prefix="tmp-",
                    directory=object(),  # type: ignore[arg-type]
                )
            )
        with self.assertRaises(AssertionError):
            asyncio_run(remove_tree(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(write_text(object(), "value"))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(write_text("value.txt", b"value"))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(write_text("value.txt", "value", encoding=""))
        with self.assertRaises(AssertionError):
            asyncio_run(write_bytes(object(), b"value"))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(write_bytes("value.bin", "value"))  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
