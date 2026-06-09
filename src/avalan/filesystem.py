from asyncio import get_running_loop, to_thread
from asyncio import run as asyncio_run
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_TEXT_ENCODING = "utf-8"
T = TypeVar("T")


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


def _run_coroutine(coroutine: Coroutine[Any, Any, T]) -> T:
    return asyncio_run(coroutine)


def run_awaitable(coroutine: Coroutine[Any, Any, T]) -> T:
    try:
        get_running_loop()
    except RuntimeError:
        return _run_coroutine(coroutine)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(_run_coroutine, coroutine).result()
