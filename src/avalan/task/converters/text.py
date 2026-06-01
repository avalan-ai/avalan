from ...types import assert_non_empty_string as _assert_non_empty_string
from . import TaskFileConversionError, TaskFileConversionResult

from collections.abc import Mapping


class TextFileConverter:
    name = "text"
    version = "1"
    media_type = "text/plain"

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
        conversion_options = _TextConversionOptions.from_mapping(options)
        if not _is_text_media_type(source_media_type):
            raise TaskFileConversionError(
                "file media type is not supported by the text converter"
            )
        try:
            text = content.decode(
                conversion_options.encoding,
                errors=conversion_options.errors,
            )
        except UnicodeError as error:
            raise TaskFileConversionError(
                "file content could not be decoded"
            ) from error
        if conversion_options.newline == "lf":
            text = text.replace("\r\n", "\n").replace("\r", "\n")
        encoded = text.encode("utf-8")
        return TaskFileConversionResult(
            content=encoded,
            media_type=self.media_type,
            metadata={
                "encoding": "utf-8",
                "source_encoding": conversion_options.encoding,
                "characters": len(text),
                "source_media_type": source_media_type,
            },
        )


class _TextConversionOptions:
    def __init__(
        self,
        *,
        encoding: str,
        errors: str,
        newline: str,
    ) -> None:
        self.encoding = encoding
        self.errors = errors
        self.newline = newline

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
    ) -> "_TextConversionOptions":
        options = dict(value or {})
        unknown = set(options) - {"encoding", "errors", "newline"}
        if unknown:
            raise TaskFileConversionError(
                "text converter option is not supported"
            )
        encoding = _string_option(options, "encoding", default="utf-8")
        errors = _string_option(options, "errors", default="strict")
        newline = _string_option(options, "newline", default="preserve")
        if errors not in {"strict", "replace", "ignore"}:
            raise TaskFileConversionError(
                "text converter error handling is not supported"
            )
        if newline not in {"preserve", "lf"}:
            raise TaskFileConversionError(
                "text converter newline mode is not supported"
            )
        return cls(encoding=encoding, errors=errors, newline=newline)


def _is_text_media_type(source_media_type: str | None) -> bool:
    if source_media_type is None:
        return True
    lowered = source_media_type.lower()
    return (
        lowered.startswith("text/")
        or lowered in {"application/json", "application/xml"}
        or lowered.endswith("+json")
        or lowered.endswith("+xml")
    )


def _string_option(
    options: Mapping[str, object],
    key: str,
    *,
    default: str,
) -> str:
    value = options.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise TaskFileConversionError("text converter option is invalid")
    return value
