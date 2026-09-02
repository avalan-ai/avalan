from ...filesystem import file_digest_and_base64 as _file_digest_and_base64
from ...filesystem import list_directory as _list_directory
from ...filesystem import make_directory as _make_directory
from ...filesystem import make_private_directory as _make_private_directory
from ...filesystem import read_bytes as _read_bytes
from ...filesystem import read_bytes_prefix as _read_bytes_prefix
from ...filesystem import read_validated_bytes as _read_validated_bytes
from ...filesystem import remove_file as _remove_file
from ...filesystem import remove_tree as _remove_tree
from ...filesystem import resolve_path as _resolve_path
from ...filesystem import stat_path as _stat_path
from ...filesystem import write_bytes as _write_bytes
from ...types import assert_non_negative_int as _assert_non_negative_int

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from re import compile as compile_pattern
from stat import S_ISDIR, S_ISLNK, S_ISREG
from typing import final
from zlib import crc32, decompressobj
from zlib import error as ZlibError

DEFAULT_SIGNATURE_BYTES = 8192
IMAGE_DIMENSION_SCAN_BYTES = 1048576
PDF_PAGE_BOX_SCAN_BYTES = 1048576
PDF_PAGE_BOX_PATTERN = compile_pattern(
    rb"/(?:CropBox|MediaBox)\s*\[\s*"
    rb"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    rb"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    rb"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s+"
    rb"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]"
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8"
_JPEG_START_OF_FRAME_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


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
    device: int | None = None
    inode: int | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.path, Path), "path must be a path"
        assert isinstance(
            self.resolved_path,
            Path,
        ), "resolved_path must be a path"
        _assert_non_negative_int(self.mode, "mode")
        _assert_non_negative_int(self.size, "size")
        _assert_non_negative_int(self.hardlink_count, "hardlink_count")
        for field_name in ("device", "inode"):
            value = getattr(self, field_name)
            if value is not None:
                _assert_non_negative_int(value, field_name)
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
        device=stat_result.st_dev,
        inode=stat_result.st_ino,
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


async def read_bytes(path: str | Path) -> bytes:
    assert isinstance(path, str | Path), "path must be a string or path"
    return await _read_bytes(path)


async def read_validated_bytes(
    path: str | Path,
    metadata: ShellPathMetadata,
    *,
    max_bytes: int,
) -> bytes:
    """Read one unchanged regular file without following a final symlink."""
    assert isinstance(path, str | Path), "path must be a string or path"
    assert isinstance(
        metadata,
        ShellPathMetadata,
    ), "metadata must be path data"
    _assert_non_negative_int(max_bytes, "max_bytes")
    if metadata.device is None or metadata.inode is None:
        raise ValueError("validated file identity is unavailable")
    try:
        return await _read_validated_bytes(
            path,
            device=metadata.device,
            inode=metadata.inode,
            mode=metadata.mode,
            size=metadata.size,
            hardlink_count=metadata.hardlink_count,
            max_bytes=max_bytes,
        )
    except OSError:
        raise ValueError("file changed while being read") from None


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


async def probe_pdf_page_boxes(
    path: str | Path,
    *,
    max_bytes: int = PDF_PAGE_BOX_SCAN_BYTES,
) -> tuple[tuple[float, float], ...]:
    data = await read_signature(path, max_bytes=max_bytes)
    return _probe_pdf_page_boxes(data)


async def read_image_signature(path: str | Path) -> bytes:
    return await read_signature(path, max_bytes=IMAGE_DIMENSION_SCAN_BYTES)


async def probe_image_dimensions(path: str | Path) -> tuple[int, int] | None:
    signature = await read_image_signature(path)
    return probe_image_dimensions_from_bytes(signature)


def probe_image_dimensions_from_bytes(
    signature: bytes,
) -> tuple[int, int] | None:
    """Return bounded raster dimensions from encoded image bytes."""
    assert isinstance(signature, bytes), "signature must be bytes"
    png_dimensions = _probe_png_dimensions(signature)
    if png_dimensions is not None:
        return png_dimensions
    jpeg_dimensions = _probe_jpeg_dimensions(signature)
    if jpeg_dimensions is not None:
        return jpeg_dimensions
    return _probe_pnm_dimensions(signature)


def validate_supported_image_bytes(
    data: bytes,
    *,
    max_long_edge_pixels: int | None = None,
    max_pixels: int | None = None,
) -> tuple[str, int, int]:
    """Validate one complete PNG or JPEG image and return its metadata."""
    assert isinstance(data, bytes), "image data must be bytes"
    if data.startswith(PNG_SIGNATURE):
        dimensions = _validate_png(
            data,
            max_long_edge_pixels=max_long_edge_pixels,
            max_pixels=max_pixels,
        )
        return "image/png", *dimensions
    if data.startswith(JPEG_SIGNATURE):
        dimensions = _validate_jpeg(data)
        _validate_image_dimensions(
            *dimensions,
            max_long_edge_pixels=max_long_edge_pixels,
            max_pixels=max_pixels,
        )
        _validate_jpeg_pixels(data, dimensions)
        return "image/jpeg", *dimensions
    raise ValueError("unsupported image signature")


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


def _validate_png(
    data: bytes,
    *,
    max_long_edge_pixels: int | None,
    max_pixels: int | None,
) -> tuple[int, int]:
    """Return dimensions only when the complete PNG chunk stream is valid."""
    if len(data) < 8 + 12 or not data.startswith(PNG_SIGNATURE):
        raise ValueError("malformed PNG image")
    offset = len(PNG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    bit_depth: int | None = None
    color_type: int | None = None
    interlace: int | None = None
    compressed_chunks: list[bytes] = []
    saw_idat = False
    saw_iend = False
    saw_non_idat_after_idat = False
    saw_plte = False
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("malformed PNG image")
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        if len(chunk_type) != 4 or any(
            value not in range(ord("A"), ord("Z") + 1)
            and value not in range(ord("a"), ord("z") + 1)
            for value in chunk_type
        ):
            raise ValueError("malformed PNG image")
        if chunk_type[2] not in range(ord("A"), ord("Z") + 1):
            raise ValueError("malformed PNG image")
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("malformed PNG image")
        chunk_data = data[offset + 8 : offset + 8 + length]
        checksum = int.from_bytes(data[offset + 8 + length : end], "big")
        if crc32(chunk_type + chunk_data) & 0xFFFFFFFF != checksum:
            raise ValueError("malformed PNG image")
        if dimensions is None:
            if chunk_type != b"IHDR" or length != 13:
                raise ValueError("malformed PNG image")
            width = int.from_bytes(chunk_data[:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            if width <= 0 or height <= 0:
                raise ValueError("malformed PNG image")
            dimensions = width, height
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            if (
                (color_type, bit_depth)
                not in {
                    (0, 1),
                    (0, 2),
                    (0, 4),
                    (0, 8),
                    (0, 16),
                    (2, 8),
                    (2, 16),
                    (3, 1),
                    (3, 2),
                    (3, 4),
                    (3, 8),
                    (4, 8),
                    (4, 16),
                    (6, 8),
                    (6, 16),
                }
                or chunk_data[10] != 0
                or chunk_data[11] != 0
                or chunk_data[12] not in {0, 1}
            ):
                raise ValueError("malformed PNG image")
            interlace = chunk_data[12]
            _validate_image_dimensions(
                width,
                height,
                max_long_edge_pixels=max_long_edge_pixels,
                max_pixels=max_pixels,
            )
        elif chunk_type == b"IHDR":
            raise ValueError("malformed PNG image")
        if chunk_type not in {
            b"IHDR",
            b"PLTE",
            b"IDAT",
            b"IEND",
        } and chunk_type[0] not in range(ord("a"), ord("z") + 1):
            raise ValueError("malformed PNG image")
        if chunk_type == b"PLTE":
            assert color_type is not None and bit_depth is not None
            if (
                saw_plte
                or saw_idat
                or color_type in {0, 4}
                or length == 0
                or length % 3 != 0
                or length > 768
                or (color_type == 3 and length // 3 > 2**bit_depth)
            ):
                raise ValueError("malformed PNG image")
            saw_plte = True
        if chunk_type == b"IDAT":
            if saw_non_idat_after_idat:
                raise ValueError("malformed PNG image")
            saw_idat = True
            compressed_chunks.append(chunk_data)
        elif saw_idat and chunk_type != b"IEND":
            saw_non_idat_after_idat = True
        if chunk_type == b"IEND":
            if length != 0 or end != len(data):
                raise ValueError("malformed PNG image")
            saw_iend = True
            break
        offset = end
    if (
        dimensions is None
        or not saw_idat
        or not saw_iend
        or (color_type == 3 and not saw_plte)
    ):
        raise ValueError("malformed PNG image")
    assert bit_depth is not None and color_type is not None
    assert interlace is not None
    expected_bytes = _png_decoded_bytes(
        *dimensions,
        bit_depth=bit_depth,
        color_type=color_type,
        interlace=interlace,
    )
    decompressor = decompressobj()
    try:
        decoded = decompressor.decompress(
            b"".join(compressed_chunks),
            expected_bytes + 1,
        )
    except ZlibError:
        raise ValueError("malformed PNG image") from None
    if (
        len(decoded) != expected_bytes
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise ValueError("malformed PNG image")
    decoded_offset = 0
    for scanline_bytes in _png_scanline_bytes(
        *dimensions,
        bit_depth=bit_depth,
        color_type=color_type,
        interlace=interlace,
    ):
        if decoded[decoded_offset] > 4:
            raise ValueError("malformed PNG image")
        decoded_offset += scanline_bytes
    if decoded_offset != expected_bytes:
        raise ValueError("malformed PNG image")
    return dimensions


def _png_decoded_bytes(
    width: int,
    height: int,
    *,
    bit_depth: int,
    color_type: int,
    interlace: int,
) -> int:
    return sum(
        _png_scanline_bytes(
            width,
            height,
            bit_depth=bit_depth,
            color_type=color_type,
            interlace=interlace,
        )
    )


def _png_scanline_bytes(
    width: int,
    height: int,
    *,
    bit_depth: int,
    color_type: int,
    interlace: int,
) -> tuple[int, ...]:
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]

    def pass_scanlines(
        x_start: int,
        y_start: int,
        x_step: int,
        y_step: int,
    ) -> tuple[int, ...]:
        if width <= x_start or height <= y_start:
            return ()
        pass_width = (width - x_start + x_step - 1) // x_step
        pass_height = (height - y_start + y_step - 1) // y_step
        row_bytes = (pass_width * channels * bit_depth + 7) // 8
        return (row_bytes + 1,) * pass_height

    if interlace == 0:
        return pass_scanlines(0, 0, 1, 1)
    return tuple(
        scanline_bytes
        for adam7_pass in (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        )
        for scanline_bytes in pass_scanlines(*adam7_pass)
    )


def _validate_image_dimensions(
    width: int,
    height: int,
    *,
    max_long_edge_pixels: int | None,
    max_pixels: int | None,
) -> None:
    if max_long_edge_pixels is not None:
        _assert_non_negative_int(
            max_long_edge_pixels,
            "max_long_edge_pixels",
        )
        if max(width, height) > max_long_edge_pixels:
            raise ValueError("image dimensions exceed the configured limits")
    if max_pixels is not None:
        _assert_non_negative_int(max_pixels, "max_pixels")
        if width * height > max_pixels:
            raise ValueError("image dimensions exceed the configured limits")


def _probe_jpeg_dimensions(signature: bytes) -> tuple[int, int] | None:
    assert isinstance(signature, bytes), "signature must be bytes"
    if not signature.startswith(JPEG_SIGNATURE):
        return None
    offset = len(JPEG_SIGNATURE)
    while offset < len(signature):
        if signature[offset] != 0xFF:
            return None
        while offset < len(signature) and signature[offset] == 0xFF:
            offset += 1
        if offset >= len(signature):
            return None
        marker = signature[offset]
        offset += 1
        if marker == 0x01:
            continue
        if marker in {0x00, 0xD8} or 0xD0 <= marker <= 0xD7:
            continue
        if marker in {0xD9, 0xDA}:
            return None
        if offset + 2 > len(signature):
            return None
        segment_length = int.from_bytes(signature[offset : offset + 2], "big")
        if segment_length < 2:
            return None
        segment_end = offset + segment_length
        if segment_end > len(signature):
            return None
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            if segment_length < 8 or offset + 7 > len(signature):
                return None
            height = int.from_bytes(
                signature[offset + 3 : offset + 5],
                "big",
            )
            width = int.from_bytes(
                signature[offset + 5 : offset + 7],
                "big",
            )
            if width <= 0 or height <= 0:
                return None
            return width, height
        offset = segment_end
    return None


def _validate_jpeg(data: bytes) -> tuple[int, int]:
    """Return JPEG dimensions when marker and scan framing are valid."""
    offset = len(JPEG_SIGNATURE)
    dimensions: tuple[int, int] | None = None
    frame_components: frozenset[int] = frozenset()
    while offset < len(data):
        if data[offset] != 0xFF:
            raise ValueError("malformed JPEG image")
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise ValueError("malformed JPEG image")
        marker = data[offset]
        offset += 1
        if marker == 0xD9:
            if dimensions is None or offset != len(data):
                raise ValueError("malformed JPEG image")
            return dimensions
        if marker in {0x00, 0x01, 0xD8} or 0xD0 <= marker <= 0xD7:
            raise ValueError("malformed JPEG image")
        if offset + 2 > len(data):
            raise ValueError("malformed JPEG image")
        segment_length = int.from_bytes(data[offset : offset + 2], "big")
        if segment_length < 2:
            raise ValueError("malformed JPEG image")
        segment_end = offset + segment_length
        if segment_end > len(data):
            raise ValueError("malformed JPEG image")
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            component_count = data[offset + 7]
            if (
                dimensions is not None
                or component_count == 0
                or component_count > 4
                or segment_length != 8 + 3 * component_count
            ):
                raise ValueError("malformed JPEG image")
            height = int.from_bytes(data[offset + 3 : offset + 5], "big")
            width = int.from_bytes(data[offset + 5 : offset + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("malformed JPEG image")
            dimensions = width, height
            components: set[int] = set()
            component_offset = offset + 8
            for _ in range(component_count):
                identifier = data[component_offset]
                sampling = data[component_offset + 1]
                horizontal = sampling >> 4
                vertical = sampling & 0x0F
                if (
                    identifier in components
                    or horizontal == 0
                    or horizontal > 4
                    or vertical == 0
                    or vertical > 4
                    or data[component_offset + 2] > 3
                ):
                    raise ValueError("malformed JPEG image")
                components.add(identifier)
                component_offset += 3
            frame_components = frozenset(components)
        if marker == 0xDA:
            if dimensions is None:
                raise ValueError("malformed JPEG image")
            scan_component_count = data[offset + 2]
            if (
                scan_component_count == 0
                or scan_component_count > len(frame_components)
                or segment_length != 6 + 2 * scan_component_count
            ):
                raise ValueError("malformed JPEG image")
            scan_components = {
                data[offset + 3 + 2 * index]
                for index in range(scan_component_count)
            }
            if len(
                scan_components
            ) != scan_component_count or not scan_components.issubset(
                frame_components
            ):
                raise ValueError("malformed JPEG image")
            offset = _jpeg_scan_marker_offset(data, segment_end)
            continue
        offset = segment_end
    raise ValueError("malformed JPEG image")


def _jpeg_scan_marker_offset(data: bytes, offset: int) -> int:
    while offset < len(data):
        marker_offset = data.find(b"\xff", offset)
        if marker_offset < 0:
            raise ValueError("malformed JPEG image")
        offset = marker_offset + 1
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            raise ValueError("malformed JPEG image")
        marker = data[offset]
        if marker == 0x00 or 0xD0 <= marker <= 0xD7:
            offset += 1
            continue
        return marker_offset
    raise ValueError("malformed JPEG image")


def _validate_jpeg_pixels(
    data: bytes,
    dimensions: tuple[int, int],
) -> None:
    """Fully decode bounded JPEG pixels through the declared vendor extra."""
    try:
        image_module = import_module("PIL.Image")
    except ImportError:
        raise ValueError(
            "JPEG pixel validation requires the optional Pillow dependency"
        ) from None
    try:
        with image_module.open(BytesIO(data)) as image:
            if image.format != "JPEG" or image.size != dimensions:
                raise ValueError("malformed JPEG image")
            image.load()
    except (OSError, SyntaxError, ValueError):
        raise ValueError("malformed JPEG image") from None


def _probe_pdf_page_boxes(data: bytes) -> tuple[tuple[float, float], ...]:
    assert isinstance(data, bytes), "data must be bytes"
    boxes: list[tuple[float, float]] = []
    for match in PDF_PAGE_BOX_PATTERN.finditer(data):
        left, bottom, right, top = (float(value) for value in match.groups())
        width = abs(right - left)
        height = abs(top - bottom)
        if width > 0 and height > 0:
            boxes.append((width, height))
    return tuple(boxes)


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
