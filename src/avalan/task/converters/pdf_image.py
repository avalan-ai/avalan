from ...types import assert_non_empty_string as _assert_non_empty_string
from ..feature_gate import TaskFeature
from ..store import TaskSnapshotMetadata, freeze_snapshot_metadata
from . import (
    TaskFileConversionError,
    TaskFileConversionResult,
    TaskFileConverterCapability,
)

from collections.abc import Mapping
from types import MappingProxyType

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
        assert isinstance(content, bytes)
        if source_media_type is not None:
            _assert_non_empty_string(source_media_type, "source_media_type")
            _validate_source_media_type(source_media_type, self._capability)
        _validate_options(options or {}, self._capability)
        raise TaskFileConversionError("PDF image conversion is unavailable")


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
    if value in {"png", "jpeg"}:
        return str(value)
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
