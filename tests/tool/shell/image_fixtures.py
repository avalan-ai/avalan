"""Provide compact, valid raster fixtures for shell tests."""

from base64 import b64decode
from binascii import crc32
from struct import pack
from zlib import compress

VALID_JPEG_BYTES = b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgG"
    "BgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/wAALCAAQ"
    "ABABAREA/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAA"
    "AAAA/9oACAEBAAA/AKpgP//Z"
)

VALID_PROGRESSIVE_JPEG_BYTES = b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
    "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
    "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
    "MjIyMjIyMjIyMjIyMjL/wgARCAALABEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAA"
    "AAAAAAAAAAb/xAAVAQEBAAAAAAAAAAAAAAAAAAAAAv/aAAwDAQACEAMQAAABjRcA"
    "f//EABQQAQAAAAAAAAAAAAAAAAAAACD/2gAIAQEAAQUCX//EABQRAQAAAAAAAAAA"
    "AAAAAAAAABD/2gAIAQMBAT8BP//EABQRAQAAAAAAAAAAAAAAAAAAABD/2gAIAQIB"
    "AT8BP//EABQQAQAAAAAAAAAAAAAAAAAAACD/2gAIAQEABj8CX//EABQQAQAAAAAA"
    "AAAAAAAAAAAAACD/2gAIAQEAAT8hX//aAAwDAQACAAMAAAAQBB//xAAUEQEAAAAA"
    "AAAAAAAAAAAAAAAQ/9oACAEDAQE/ED//xAAUEQEAAAAAAAAAAAAAAAAAAAAQ/9oA"
    "CAECAQE/ED//xAAUEAEAAAAAAAAAAAAAAAAAAAAg/9oACAEBAAE/EF//2Q=="
)


def valid_png_bytes(
    *,
    width: int,
    height: int,
    label: str | None = None,
    caption: str | None = None,
) -> bytes:
    """Return a valid, compact RGB PNG with optional text properties."""
    assert width > 0, "width must be positive"
    assert height > 0, "height must be positive"
    chunks = [
        _png_chunk(b"IHDR", pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    ]
    if label is not None:
        chunks.append(
            _png_chunk(b"tEXt", b"label\x00" + label.encode("latin-1"))
        )
    if caption is not None:
        chunks.append(
            _png_chunk(b"tEXt", b"caption\x00" + caption.encode("latin-1"))
        )
    row = b"\x00" + b"\x00\x00\x00" * width
    chunks.append(_png_chunk(b"IDAT", compress(row * height)))
    chunks.append(_png_chunk(b"IEND", b""))
    return b"\x89PNG\r\n\x1a\n" + b"".join(chunks)


def multi_frame_ppm_bytes(
    *,
    first_rgb: tuple[int, int, int],
    second_rgb: tuple[int, int, int],
) -> bytes:
    """Return a valid two-frame one-pixel binary PPM stream."""
    return _ppm_frame(first_rgb) + _ppm_frame(second_rgb)


def _png_chunk(chunk_type: bytes, content: bytes) -> bytes:
    checksum = crc32(chunk_type + content) & 0xFFFFFFFF
    return (
        pack(">I", len(content)) + chunk_type + content + pack(">I", checksum)
    )


def _ppm_frame(rgb: tuple[int, int, int]) -> bytes:
    assert len(rgb) == 3, "rgb must contain three channels"
    assert all(0 <= channel <= 255 for channel in rgb)
    return b"P6\n1 1\n255\n" + bytes(rgb)
