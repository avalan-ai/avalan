from ...types import assert_non_empty_string as _assert_non_empty_string
from ..feature_gate import TaskFeature
from . import (
    TaskFileConversionDependencyError,
    TaskFileConversionError,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    markdown_converter_capability,
)

from collections.abc import Mapping
from importlib import import_module
from typing import Any, cast


class MarkdownFileConverter:
    name = "markdown"
    version = "1"
    media_type = "text/markdown"

    @property
    def capability(self) -> TaskFileConverterCapability:
        return markdown_converter_capability()

    def validate_options(
        self,
        options: Mapping[str, object] | None = None,
    ) -> None:
        _MarkdownConversionOptions.from_mapping(options)

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        assert isinstance(content, bytes), "content must be bytes"
        if source_media_type is not None:
            _assert_non_empty_string(source_media_type, "source_media_type")
        conversion_options = _MarkdownConversionOptions.from_mapping(options)
        try:
            text = content.decode(
                conversion_options.encoding,
                errors=conversion_options.errors,
            )
        except UnicodeError as error:
            raise TaskFileConversionError(
                "file content could not be decoded"
            ) from error
        markdown = self._markdown_from_text(
            text,
            source_media_type=source_media_type,
            heading_style=conversion_options.html_heading_style,
        )
        encoded = markdown.encode("utf-8")
        return TaskFileConversionResult(
            content=encoded,
            media_type=self.media_type,
            metadata={
                "encoding": "utf-8",
                "source_encoding": conversion_options.encoding,
                "characters": len(markdown),
                "source_media_type": source_media_type,
            },
        )

    def _markdown_from_text(
        self,
        text: str,
        *,
        source_media_type: str | None,
        heading_style: str,
    ) -> str:
        if source_media_type is None:
            return text
        lowered = source_media_type.lower()
        if lowered in {"text/markdown", "text/x-markdown", "text/plain"}:
            return text
        if lowered in {"text/html", "application/xhtml+xml"}:
            return str(_markdownify_html(text, heading_style=heading_style))
        raise TaskFileConversionError(
            "file media type is not supported by the markdown converter"
        )


class _MarkdownConversionOptions:
    def __init__(
        self,
        *,
        encoding: str,
        errors: str,
        html_heading_style: str,
    ) -> None:
        self.encoding = encoding
        self.errors = errors
        self.html_heading_style = html_heading_style

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> "_MarkdownConversionOptions":
        options = dict(value or {})
        unknown = set(options) - {"encoding", "errors", "html_heading_style"}
        if unknown:
            raise TaskFileConversionError(
                "markdown converter option is not supported"
            )
        encoding = _string_option(options, "encoding", default="utf-8")
        errors = _string_option(options, "errors", default="strict")
        html_heading_style = _string_option(
            options,
            "html_heading_style",
            default="ATX",
        )
        if errors not in {"strict", "replace", "ignore"}:
            raise TaskFileConversionError(
                "markdown converter error handling is not supported"
            )
        if html_heading_style not in {"ATX", "ATX_CLOSED", "SETEXT"}:
            raise TaskFileConversionError(
                "markdown converter heading style is not supported"
            )
        return cls(
            encoding=encoding,
            errors=errors,
            html_heading_style=html_heading_style,
        )


def _string_option(
    options: Mapping[str, object],
    key: str,
    *,
    default: str,
) -> str:
    value = options.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise TaskFileConversionError("markdown converter option is invalid")
    return value


def _markdownify_html(text: str, *, heading_style: str) -> str:
    try:
        markdownify = cast(Any, import_module("markdownify").markdownify)
    except ModuleNotFoundError as error:
        raise TaskFileConversionDependencyError(
            TaskFeature.DOCUMENT_CONVERSION
        ) from error
    return cast(str, markdownify(text, heading_style=heading_style))
