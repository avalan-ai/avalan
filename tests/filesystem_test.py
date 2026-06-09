from asyncio import run as asyncio_run
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main

from avalan.filesystem import (
    assert_text_encoding,
    read_bytes,
    read_text,
    run_awaitable,
)


async def _value() -> str:
    return "ok"


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

    def test_rejects_invalid_arguments(self) -> None:
        with self.assertRaises(AssertionError):
            assert_text_encoding("")
        with self.assertRaises(AssertionError):
            asyncio_run(read_text(object()))  # type: ignore[arg-type]
        with self.assertRaises(AssertionError):
            asyncio_run(read_text("value.txt", encoding=""))
        with self.assertRaises(AssertionError):
            asyncio_run(read_bytes(object()))  # type: ignore[arg-type]

    def test_run_awaitable_works_inside_and_outside_running_loop(self) -> None:
        async def exercise() -> str:
            return run_awaitable(_value())

        self.assertEqual(run_awaitable(_value()), "ok")
        self.assertEqual(asyncio_run(exercise()), "ok")


if __name__ == "__main__":
    main()
