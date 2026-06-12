from .types import (
    assert_absolute_path_sequence as _assert_absolute_path_sequence,
)
from .types import assert_non_empty_string as _assert_non_empty_string

from asyncio import to_thread
from base64 import b64encode
from collections.abc import Sequence
from hashlib import sha256
from os import pathsep, stat_result
from pathlib import Path
from shutil import rmtree, which
from tempfile import mkdtemp

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


async def read_bytes_prefix(path: str | Path, max_bytes: int) -> bytes:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(max_bytes, int), "max_bytes must be an integer"
    assert not isinstance(max_bytes, bool), "max_bytes must be an integer"
    assert max_bytes >= 0, "max_bytes must not be negative"
    return await to_thread(_read_bytes_prefix, Path(path), max_bytes)


async def file_digest_and_base64(
    path: str | Path,
    *,
    chunk_size: int,
    max_inline_bytes: int,
) -> tuple[str, str | None]:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(chunk_size, int), "chunk_size must be an integer"
    assert not isinstance(chunk_size, bool), "chunk_size must be an integer"
    assert chunk_size > 0, "chunk_size must be positive"
    assert isinstance(
        max_inline_bytes,
        int,
    ), "max_inline_bytes must be an integer"
    assert not isinstance(
        max_inline_bytes,
        bool,
    ), "max_inline_bytes must be an integer"
    assert max_inline_bytes >= 0, "max_inline_bytes must not be negative"
    return await to_thread(
        _file_digest_and_base64,
        Path(path),
        chunk_size,
        max_inline_bytes,
    )


async def stat_path(
    path: str | Path,
    *,
    follow_symlinks: bool = True,
) -> stat_result:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(follow_symlinks, bool), "follow_symlinks must be boolean"
    return await to_thread(
        Path(path).stat,
        follow_symlinks=follow_symlinks,
    )


async def resolve_path(
    path: str | Path,
    *,
    strict: bool = False,
) -> Path:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(strict, bool), "strict must be boolean"
    return await to_thread(Path(path).resolve, strict=strict)


async def list_directory(path: str | Path) -> tuple[Path, ...]:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await to_thread(_list_directory, Path(path))


async def make_directory(path: str | Path, *, mode: int = 0o700) -> Path:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(mode, int), "mode must be an integer"
    assert not isinstance(mode, bool), "mode must be an integer"
    assert mode >= 0, "mode must not be negative"
    return await to_thread(_make_directory, Path(path), mode)


async def make_private_directory(
    *,
    prefix: str,
    directory: str | Path | None = None,
) -> Path:
    assert isinstance(prefix, str), "prefix must be a string"
    assert prefix.strip(), "prefix must not be empty"
    if directory is not None:
        assert isinstance(
            directory,
            str | Path,
        ), "directory must be a string or path"
    return Path(await to_thread(mkdtemp, prefix=prefix, dir=directory))


async def remove_tree(path: str | Path) -> None:
    assert isinstance(path, str | Path), "path must be a string or path"
    await to_thread(rmtree, path)


async def remove_file(path: str | Path) -> None:
    assert isinstance(path, str | Path), "path must be a string or path"
    await to_thread(Path(path).unlink)


async def which_executable(
    executable_name: str,
    search_paths: Sequence[str],
) -> str | None:
    _assert_non_empty_string(executable_name, "executable_name")
    _assert_absolute_path_sequence(search_paths, "search_paths")
    return await to_thread(
        which,
        executable_name,
        path=pathsep.join(search_paths),
    )


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


def _read_bytes_prefix(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as input_file:
        return input_file.read(max_bytes)


def _file_digest_and_base64(
    path: Path,
    chunk_size: int,
    max_inline_bytes: int,
) -> tuple[str, str | None]:
    digest = sha256()
    inline_chunks: list[bytes] | None = [] if max_inline_bytes > 0 else None
    total_bytes = 0
    with path.open("rb") as input_file:
        while True:
            chunk = input_file.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
            digest.update(chunk)
            if inline_chunks is not None:
                if total_bytes <= max_inline_bytes:
                    inline_chunks.append(chunk)
                else:
                    inline_chunks = None
    if inline_chunks is None or total_bytes == 0:
        return digest.hexdigest(), None
    return digest.hexdigest(), b64encode(b"".join(inline_chunks)).decode(
        "ascii"
    )


def _list_directory(path: Path) -> tuple[Path, ...]:
    return tuple(path.iterdir())


def _make_directory(path: Path, mode: int) -> Path:
    path.mkdir(mode=mode)
    return path
