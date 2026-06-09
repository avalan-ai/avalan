from asyncio import to_thread
from pathlib import Path

DEFAULT_TEXT_ENCODING = "utf-8"


def assert_text_encoding(encoding: str) -> None:
    assert (
        isinstance(encoding, str) and encoding.strip()
    ), "encoding must be a non-empty string"


async def read_text(
    path: str | Path,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> str:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert_text_encoding(encoding)
    return await to_thread(Path(path).read_text, encoding=encoding)


async def read_bytes(path: str | Path) -> bytes:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await to_thread(Path(path).read_bytes)


async def write_text(
    path: str | Path,
    data: str,
    *,
    encoding: str = DEFAULT_TEXT_ENCODING,
) -> int:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(data, str), "data must be a string"
    assert_text_encoding(encoding)
    return await to_thread(Path(path).write_text, data, encoding=encoding)


async def write_bytes(path: str | Path, data: bytes) -> int:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(data, bytes), "data must be bytes"
    return await to_thread(Path(path).write_bytes, data)
