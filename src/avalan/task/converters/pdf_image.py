from ...types import assert_non_empty_string as _assert_non_empty_string
from ..feature_gate import TaskFeature
from ..store import TaskSnapshotMetadata, freeze_snapshot_metadata
from . import (
    TaskFileConversionDependencyError,
    TaskFileConversionError,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionResult,
    TaskFileConverterCapability,
)

from collections.abc import Mapping
from importlib import import_module
from io import BytesIO
from types import MappingProxyType
from typing import Any, cast

_PDF_IMAGE_MAX_INPUT_BYTES = 100 * 1024 * 1024
_PDF_IMAGE_MAX_OUTPUT_BYTES = 50 * 1024 * 1024
_PDF_IMAGE_MAX_PAGES = 256
_PDF_IMAGE_MIN_DPI = 36
_PDF_IMAGE_MAX_DPI = 300
_PDF_IMAGE_MIN_QUALITY = 1
_PDF_IMAGE_MAX_QUALITY = 100
_PDF_IMAGE_MAX_PIXELS = 40_000_000
_PDF_IMAGE_ESTIMATED_MEMORY_BYTES = 512 * 1024 * 1024
_PDF_IMAGE_TIMEOUT_SECONDS = 120
_PDF_IMAGE_OPTIONS_SCHEMA = MappingProxyType(
    {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "format": {"enum": ("png", "jpeg")},
            "dpi": {
                "type": "integer",
                "minimum": _PDF_IMAGE_MIN_DPI,
                "maximum": _PDF_IMAGE_MAX_DPI,
            },
            "pages": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "start": {"type": "integer", "minimum": 1},
                    "end": {"type": "integer", "minimum": 1},
                },
            },
            "quality": {
                "type": "integer",
                "minimum": _PDF_IMAGE_MIN_QUALITY,
                "maximum": _PDF_IMAGE_MAX_QUALITY,
            },
        },
    }
)


def pdf_image_converter_capability() -> TaskFileConverterCapability:
    return TaskFileConverterCapability(
        source_mime_types=("application/pdf",),
        output_mime_types=("image/png", "image/jpeg"),
        supports_streaming=False,
        max_input_bytes=_PDF_IMAGE_MAX_INPUT_BYTES,
        max_output_bytes=_PDF_IMAGE_MAX_OUTPUT_BYTES,
        max_pages=_PDF_IMAGE_MAX_PAGES,
        min_dpi=_PDF_IMAGE_MIN_DPI,
        max_dpi=_PDF_IMAGE_MAX_DPI,
        min_quality=_PDF_IMAGE_MIN_QUALITY,
        max_quality=_PDF_IMAGE_MAX_QUALITY,
        max_pixels=_PDF_IMAGE_MAX_PIXELS,
        estimated_memory_bytes=_PDF_IMAGE_ESTIMATED_MEMORY_BYTES,
        timeout_seconds=_PDF_IMAGE_TIMEOUT_SECONDS,
        options_schema=_PDF_IMAGE_OPTIONS_SCHEMA,
        dependency_gates=(TaskFeature.PDF_IMAGE_CONVERSION,),
    )


class PdfImageFileConverter:
    name = "pdf_image"
    version = "1"

    def __init__(self) -> None:
        self._capability = pdf_image_converter_capability()

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    def validate_options(self, options: Mapping[str, object]) -> None:
        assert isinstance(options, Mapping)
        _validate_options(options, self._capability)

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        if source_media_type is not None:
            _assert_non_empty_string(source_media_type, "source_media_type")
            _validate_source_media_type(source_media_type, self._capability)
        safe_options = options or {}
        _validate_options(safe_options, self._capability)
        pages = await self.convert_pages(
            content,
            source_media_type=source_media_type,
            options=_single_page_options(safe_options),
        )
        assert pages.pages
        page = pages.pages[0]
        return TaskFileConversionResult(
            content=page.content,
            media_type=page.media_type,
            metadata=page.metadata,
        )

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        assert isinstance(content, bytes)
        if source_media_type is not None:
            _assert_non_empty_string(source_media_type, "source_media_type")
            _validate_source_media_type(source_media_type, self._capability)
        safe_options = options or {}
        _validate_options(safe_options, self._capability)
        try:
            return _convert_pdf_pages(
                content,
                options=safe_options,
                capability=self._capability,
            )
        except TaskFileConversionError:
            raise
        except ImportError as error:
            raise TaskFileConversionDependencyError(
                TaskFeature.PDF_IMAGE_CONVERSION
            ) from error
        except Exception as error:
            raise TaskFileConversionError(
                "PDF image conversion failed"
            ) from error


def _validate_options(
    options: Mapping[str, object],
    capability: TaskFileConverterCapability,
) -> None:
    allowed_keys = {"dpi", "format", "pages", "quality"}
    for key in options:
        if key not in allowed_keys:
            raise TaskFileConversionError("PDF image option is not supported")

    output_format = _output_format(options.get("format"))
    _validate_dpi(options.get("dpi"), capability)
    _validate_pages(options.get("pages"), capability)
    _validate_quality(options.get("quality"), output_format, capability)


def _output_format(value: object) -> str:
    if value is None:
        return "png"
    if isinstance(value, str) and value in {"png", "jpeg"}:
        return value
    raise TaskFileConversionError("PDF image output format is not supported")


def _validate_source_media_type(
    value: str,
    capability: TaskFileConverterCapability,
) -> None:
    if value.lower() not in capability.source_mime_types:
        raise TaskFileConversionError(
            "PDF image source media type is not supported"
        )


def _validate_dpi(
    value: object,
    capability: TaskFileConverterCapability,
) -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool):
        raise TaskFileConversionError("PDF image DPI is invalid")
    if capability.min_dpi is not None and value < capability.min_dpi:
        raise TaskFileConversionError("PDF image DPI is outside bounds")
    if capability.max_dpi is not None and value > capability.max_dpi:
        raise TaskFileConversionError("PDF image DPI is outside bounds")


def _validate_pages(
    value: object,
    capability: TaskFileConverterCapability,
) -> None:
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise TaskFileConversionError("PDF image page range is invalid")
    allowed_keys = {"end", "start"}
    for key in value:
        if key not in allowed_keys:
            raise TaskFileConversionError("PDF image page option is invalid")
    start = _page_bound(value.get("start"), default=1)
    end_value = value.get("end")
    end = _page_bound(end_value, default=start)
    if start > end:
        raise TaskFileConversionError("PDF image page range is invalid")
    page_count = end - start + 1
    if capability.max_pages is not None and page_count > capability.max_pages:
        raise TaskFileConversionError("PDF image page range exceeds limit")


def _page_bound(value: object, *, default: int) -> int:
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise TaskFileConversionError("PDF image page range is invalid")
    return value


def _validate_quality(
    value: object,
    output_format: str,
    capability: TaskFileConverterCapability,
) -> None:
    if value is None:
        return
    if output_format != "jpeg":
        raise TaskFileConversionError("PDF image quality requires JPEG output")
    if not isinstance(value, int) or isinstance(value, bool):
        raise TaskFileConversionError("PDF image quality is invalid")
    if capability.min_quality is not None and value < capability.min_quality:
        raise TaskFileConversionError("PDF image quality is outside bounds")
    if capability.max_quality is not None and value > capability.max_quality:
        raise TaskFileConversionError("PDF image quality is outside bounds")


def _convert_pdf_pages(
    content: bytes,
    *,
    options: Mapping[str, object],
    capability: TaskFileConverterCapability,
) -> TaskFileConversionPageCollection:
    pdfium = import_module("pypdfium2")
    document = pdfium.PdfDocument(content)
    try:
        page_count = len(document)
        if page_count < 1:
            raise TaskFileConversionError("PDF image source has no pages")
        start, end = _selected_pages(options.get("pages"), page_count)
        selected_count = end - start + 1
        if (
            capability.max_pages is not None
            and selected_count > capability.max_pages
        ):
            raise TaskFileConversionError("PDF image page range exceeds limit")
        dpi = _selected_dpi(options.get("dpi"))
        output_format = _output_format(options.get("format"))
        quality = _selected_quality(options.get("quality"))
        pages: list[TaskFileConversionPageResult] = []
        for page_number in range(start, end + 1):
            pages.append(
                _render_pdf_page(
                    document,
                    page_number=page_number,
                    page_count=page_count,
                    dpi=dpi,
                    output_format=output_format,
                    quality=quality,
                    capability=capability,
                )
            )
        return TaskFileConversionPageCollection(
            pages=tuple(pages),
            metadata={
                "backend": "pypdfium2",
                "format": output_format,
                "dpi": dpi,
            },
        )
    finally:
        close = getattr(document, "close", None)
        if callable(close):
            close()


def _selected_pages(value: object, page_count: int) -> tuple[int, int]:
    assert isinstance(page_count, int)
    assert page_count > 0
    if value is None:
        return 1, page_count
    assert isinstance(value, Mapping)
    start = _page_bound(value.get("start"), default=1)
    end_value = value.get("end")
    end = (
        page_count
        if end_value is None
        else _page_bound(end_value, default=start)
    )
    if start > end:
        raise TaskFileConversionError("PDF image page range is invalid")
    if start > page_count or end > page_count:
        raise TaskFileConversionError("PDF image page range exceeds document")
    return start, end


def _single_page_options(
    options: Mapping[str, object],
) -> Mapping[str, object]:
    single_options = dict(options)
    pages = options.get("pages")
    if isinstance(pages, Mapping):
        start = pages.get("start", 1)
    else:
        start = 1
    single_options["pages"] = {"start": start, "end": start}
    return single_options


def _selected_dpi(value: object) -> int:
    if value is None:
        return 144
    assert isinstance(value, int)
    return value


def _selected_quality(value: object) -> int:
    if value is None:
        return 85
    assert isinstance(value, int)
    return value


def _render_pdf_page(
    document: Any,
    *,
    page_number: int,
    page_count: int,
    dpi: int,
    output_format: str,
    quality: int,
    capability: TaskFileConverterCapability,
) -> TaskFileConversionPageResult:
    page = document[page_number - 1]
    try:
        image = page.render(scale=dpi / 72).to_pil()
        try:
            width_pixels, height_pixels = _image_dimensions(image)
            pixels = width_pixels * height_pixels
            if (
                capability.max_pixels is not None
                and pixels > capability.max_pixels
            ):
                raise TaskFileConversionError(
                    "PDF image output page exceeds the pixel limit"
                )
            content = _encode_image(
                image,
                output_format=output_format,
                quality=quality,
            )
        finally:
            close_image = getattr(image, "close", None)
            if callable(close_image):
                close_image()
    finally:
        close_page = getattr(page, "close", None)
        if callable(close_page):
            close_page()
    if (
        capability.max_output_bytes is not None
        and len(content) > capability.max_output_bytes
    ):
        raise TaskFileConversionError(
            "PDF image output exceeds the byte limit"
        )
    media_type = f"image/{output_format}"
    return TaskFileConversionPageResult(
        page_index=page_number,
        page_count=page_count,
        content=content,
        media_type=media_type,
        width_pixels=width_pixels,
        height_pixels=height_pixels,
        metadata={
            "format": output_format,
            "dpi": dpi,
            "page_number": page_number,
        },
    )


def _encode_image(
    image: Any,
    *,
    output_format: str,
    quality: int,
) -> bytes:
    buffer = BytesIO()
    if output_format == "jpeg":
        image.save(buffer, format="JPEG", quality=quality)
    else:
        image.save(buffer, format="PNG")
    content = buffer.getvalue()
    if not content:
        raise TaskFileConversionError("PDF image output is empty")
    return content


def _image_dimensions(image: Any) -> tuple[int, int]:
    size = getattr(image, "size", None)
    if (
        isinstance(size, tuple)
        and len(size) == 2
        and isinstance(size[0], int)
        and isinstance(size[1], int)
    ):
        dimensions = cast(tuple[int, int], size)
        if dimensions[0] > 0 and dimensions[1] > 0:
            return dimensions
    width = getattr(image, "width", None)
    height = getattr(image, "height", None)
    if (
        isinstance(width, int)
        and isinstance(height, int)
        and width > 0
        and height > 0
    ):
        return width, height
    raise TaskFileConversionError("PDF image dimensions are unavailable")


def safe_pdf_image_capability_metadata(
    capability: TaskFileConverterCapability,
) -> TaskSnapshotMetadata:
    return freeze_snapshot_metadata(
        {
            "source_mime_types": capability.source_mime_types,
            "output_mime_types": capability.output_mime_types,
            "max_input_bytes": capability.max_input_bytes,
            "max_output_bytes": capability.max_output_bytes,
            "max_pages": capability.max_pages,
            "min_dpi": capability.min_dpi,
            "max_dpi": capability.max_dpi,
            "min_quality": capability.min_quality,
            "max_quality": capability.max_quality,
            "max_pixels": capability.max_pixels,
            "estimated_memory_bytes": capability.estimated_memory_bytes,
            "timeout_seconds": capability.timeout_seconds,
            "options_schema": capability.options_schema,
        }
    )
