from ...filesystem import file_digest_and_base64 as _file_digest_and_base64
from ...filesystem import list_directory as _list_directory
from ...filesystem import make_directory as _make_directory
from ...filesystem import make_private_directory as _make_private_directory
from ...filesystem import read_bytes_prefix as _read_bytes_prefix
from ...filesystem import remove_file as _remove_file
from ...filesystem import remove_tree as _remove_tree
from ...filesystem import resolve_path as _resolve_path
from ...filesystem import stat_path as _stat_path
from ...filesystem import write_bytes as _write_bytes
from ...types import assert_non_negative_int as _assert_non_negative_int

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import final

DEFAULT_SIGNATURE_BYTES = 8192
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@final
@dataclass(frozen=True, kw_only=True, slots=True)
class ShellPathMetadata:
    path: Path
    resolved_path: Path
    mode: int
    size: int
    is_file: bool
    is_directory: bool
    is_symlink: bool
    is_special_file: bool
    hardlink_count: int = 1

    def __post_init__(self) -> None:
        assert isinstance(self.path, Path), "path must be a path"
        assert isinstance(
            self.resolved_path,
            Path,
        ), "resolved_path must be a path"
        _assert_non_negative_int(self.mode, "mode")
        _assert_non_negative_int(self.size, "size")
        _assert_non_negative_int(self.hardlink_count, "hardlink_count")
        for field_name in (
            "is_file",
            "is_directory",
            "is_symlink",
            "is_special_file",
        ):
            assert isinstance(
                getattr(self, field_name),
                bool,
            ), f"{field_name} must be boolean"


async def resolve_policy_path(path: str | Path) -> Path:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await _resolve_path(path, strict=False)


async def inspect_path(path: str | Path) -> ShellPathMetadata:
    assert isinstance(path, str | Path), "path must be a string or path"
    source_path = Path(path)
    stat_result = await _stat_path(source_path, follow_symlinks=False)
    mode = stat_result.st_mode
    is_file = S_ISREG(mode)
    is_directory = S_ISDIR(mode)
    is_symlink = S_ISLNK(mode)
    return ShellPathMetadata(
        path=source_path,
        resolved_path=await _resolve_path(source_path, strict=False),
        mode=mode,
        size=stat_result.st_size,
        hardlink_count=stat_result.st_nlink,
        is_file=is_file,
        is_directory=is_directory,
        is_symlink=is_symlink,
        is_special_file=not (is_file or is_directory or is_symlink),
    )


async def read_signature(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_SIGNATURE_BYTES,
) -> bytes:
    assert isinstance(path, str | Path), "path must be a string or path"
    _assert_non_negative_int(max_bytes, "max_bytes")
    return await _read_bytes_prefix(path, max_bytes)


def signature_is_binary(signature: bytes) -> bool:
    assert isinstance(signature, bytes), "signature must be bytes"
    if b"\x00" in signature:
        return True
    try:
        signature.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


async def sniff_binary(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_SIGNATURE_BYTES,
) -> bool:
    signature = await read_signature(path, max_bytes=max_bytes)
    return signature_is_binary(signature)


async def file_size(path: str | Path) -> int:
    return (await inspect_path(path)).size


async def file_digest_and_base64(
    path: str | Path,
    *,
    chunk_size: int,
    max_inline_bytes: int,
) -> tuple[str, str | None]:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await _file_digest_and_base64(
        path,
        chunk_size=chunk_size,
        max_inline_bytes=max_inline_bytes,
    )


async def list_directory(path: str | Path) -> tuple[Path, ...]:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await _list_directory(path)


async def make_directory(path: str | Path, *, mode: int = 0o700) -> Path:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await _make_directory(path, mode=mode)


async def remove_tree(path: str | Path) -> None:
    assert isinstance(path, str | Path), "path must be a string or path"
    await _remove_tree(path)


async def remove_file(path: str | Path) -> None:
    assert isinstance(path, str | Path), "path must be a string or path"
    await _remove_file(path)


async def write_bytes(path: str | Path, data: bytes) -> int:
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(data, bytes), "data must be bytes"
    return await _write_bytes(path, data)


async def ensure_file_size_at_most(
    path: str | Path,
    *,
    max_bytes: int,
) -> int:
    _assert_non_negative_int(max_bytes, "max_bytes")
    size = await file_size(path)
    assert size <= max_bytes, "file exceeds maximum size"
    return size


async def read_pdf_signature(path: str | Path) -> bytes:
    return await read_signature(path, max_bytes=5)


async def read_image_signature(path: str | Path) -> bytes:
    return await read_signature(path)


async def probe_image_dimensions(path: str | Path) -> tuple[int, int] | None:
    signature = await read_image_signature(path)
    png_dimensions = _probe_png_dimensions(signature)
    if png_dimensions is not None:
        return png_dimensions
    return _probe_pnm_dimensions(signature)


@asynccontextmanager
async def private_temp_directory(
    *,
    prefix: str = "avalan-shell-",
    directory: str | Path | None = None,
) -> AsyncIterator[Path]:
    temp_path = await _make_private_directory(
        prefix=prefix,
        directory=directory,
    )
    try:
        yield temp_path
    finally:
        try:
            await _remove_tree(temp_path)
        except OSError:
            try:
                metadata = await inspect_path(temp_path)
            except OSError:
                pass
            else:
                if metadata.is_file or metadata.is_symlink:
                    try:
                        await _remove_file(temp_path)
                    except OSError:
                        pass


def _probe_png_dimensions(signature: bytes) -> tuple[int, int] | None:
    assert isinstance(signature, bytes), "signature must be bytes"
    if len(signature) < 24 or not signature.startswith(PNG_SIGNATURE):
        return None
    width = int.from_bytes(signature[16:20], "big")
    height = int.from_bytes(signature[20:24], "big")
    if width <= 0 or height <= 0:
        return None
    return width, height


def _probe_pnm_dimensions(signature: bytes) -> tuple[int, int] | None:
    assert isinstance(signature, bytes), "signature must be bytes"
    if not signature.startswith((b"P2", b"P5", b"P6")):
        return None
    tokens: list[bytes] = []
    for line in signature.splitlines()[1:]:
        content = line.split(b"#", 1)[0]
        tokens.extend(content.split())
        if len(tokens) >= 2:
            break
    if len(tokens) < 2:
        return None
    try:
        width = int(tokens[0])
        height = int(tokens[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
