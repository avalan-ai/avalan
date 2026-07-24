"""Validate primitive values used by canonical interactions."""

from .error import InputErrorCode, InputValidationError

from datetime import UTC, datetime
from math import isfinite
from re import compile as compile_pattern
from unicodedata import bidirectional, category, normalize

MAX_STATE_REVISION = 9_007_199_254_740_991
MAX_REQUEST_CHARACTERS = 32_768
MAX_REQUEST_UTF8_BYTES = 131_072
MAX_OPAQUE_ID_CHARACTERS = 128
MAX_OPAQUE_ID_UTF8_BYTES = 512
QUESTION_ID_PATTERN = compile_pattern(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")

_BIDI_CONTROL_CLASSES = frozenset(
    {
        "LRE",
        "RLE",
        "LRO",
        "RLO",
        "PDF",
        "LRI",
        "RLI",
        "FSI",
        "PDI",
    }
)


def normalize_text(value: object, path: str) -> str:
    """Return NFC text after rejecting invalid Unicode scalar values."""
    if not isinstance(value, str):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a string",
        )
    normalized = normalize("NFC", value)
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value contains an invalid Unicode scalar",
        ) from exc
    return normalized


def validate_opaque_id(
    value: object,
    path: str,
    *,
    maximum_characters: int = MAX_OPAQUE_ID_CHARACTERS,
    maximum_bytes: int = MAX_OPAQUE_ID_UTF8_BYTES,
) -> str:
    """Return a non-empty opaque identifier without control characters."""
    normalized = normalize_text(value, path)
    if not normalized:
        raise InputValidationError(
            InputErrorCode.EMPTY,
            path,
            "identifier must not be empty",
        )
    _validate_length(
        normalized,
        path,
        1,
        maximum_characters,
        maximum_bytes,
    )
    _reject_identifier_controls(normalized, path)
    return normalized


def validate_question_id(value: object, path: str = "question_id") -> str:
    """Return a canonical stable question identifier."""
    normalized = normalize_text(value, path)
    if QUESTION_ID_PATTERN.fullmatch(normalized) is None:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "question identifier has an invalid format",
        )
    return normalized


def validate_choice_value(value: object, path: str = "choice.value") -> str:
    """Return a bounded stable choice value."""
    normalized = normalize_text(value, path)
    _validate_length(normalized, path, 1, 128, 512)
    _reject_identifier_controls(normalized, path)
    return normalized


def validate_presentation_text(
    value: object,
    path: str,
    *,
    minimum: int,
    maximum: int,
    maximum_bytes: int,
) -> str:
    """Return bounded single-line presentation text."""
    normalized = normalize_text(value, path)
    _validate_length(
        normalized,
        path,
        minimum,
        maximum,
        maximum_bytes,
    )
    if not normalized.strip():
        raise InputValidationError(
            InputErrorCode.EMPTY,
            path,
            "value must not contain only whitespace",
        )
    _reject_newlines(normalized, path)
    return normalized


def validate_single_line_text(value: object, path: str) -> str:
    """Return a bounded single-line answer."""
    normalized = normalize_text(value, path)
    _validate_length(normalized, path, 0, 4_096, 16_384)
    _reject_newlines(normalized, path)
    return normalized


def validate_multiline_text(value: object, path: str) -> str:
    """Return a bounded multiline answer with LF newlines."""
    normalized = normalize_text(value, path)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    _validate_length(normalized, path, 0, 65_536, 262_144)
    return normalized


def validate_other_text(value: object, path: str = "other.text") -> str:
    """Return a bounded non-empty free-form selection alternative."""
    normalized = normalize_text(value, path)
    _validate_length(normalized, path, 1, 4_096, 16_384)
    if not normalized.strip():
        raise InputValidationError(
            InputErrorCode.EMPTY,
            path,
            "free-form alternative must not contain only whitespace",
        )
    _reject_newlines(normalized, path)
    return normalized


def validate_aware_datetime(value: object, path: str) -> datetime:
    """Return a timezone-aware timestamp normalized to UTC."""
    if not isinstance(value, datetime):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a datetime",
        )
    if value.tzinfo is None or value.utcoffset() is None:
        raise InputValidationError(
            InputErrorCode.NAIVE_TIMESTAMP,
            path,
            "timestamp must include a timezone",
        )
    return value.astimezone(UTC)


def validate_bool(value: object, path: str) -> bool:
    """Return a strict boolean value."""
    if not isinstance(value, bool):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a boolean",
        )
    return value


def validate_int(
    value: object,
    path: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    """Return a strict integer within inclusive bounds."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be an integer",
        )
    if value < minimum or value > maximum:
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            path,
            "integer is outside its permitted range",
        )
    return value


def validate_state_revision(value: object, path: str) -> int:
    """Return a JSON-safe unsigned state revision."""
    return validate_int(
        value,
        path,
        minimum=0,
        maximum=MAX_STATE_REVISION,
    )


def validate_total_request_content(values: tuple[str, ...]) -> None:
    """Reject semantic request content above the aggregate bound."""
    characters = sum(len(value) for value in values)
    utf8_bytes = sum(len(value.encode("utf-8")) for value in values)
    if (
        characters > MAX_REQUEST_CHARACTERS
        or utf8_bytes > MAX_REQUEST_UTF8_BYTES
    ):
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            "request",
            "request content exceeds its aggregate bound",
        )


def validate_finite_number(value: object, path: str) -> int | float:
    """Return a finite JSON number."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            path,
            "value must be a number",
        )
    if not isfinite(value):
        raise InputValidationError(
            InputErrorCode.NON_JSON_VALUE,
            path,
            "number must be finite",
        )
    return value


def _validate_length(
    value: str,
    path: str,
    minimum: int,
    maximum: int,
    maximum_bytes: int,
) -> None:
    if len(value) < minimum or len(value) > maximum:
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            path,
            "value is outside its character bound",
        )
    if len(value.encode("utf-8")) > maximum_bytes:
        raise InputValidationError(
            InputErrorCode.OUT_OF_BOUNDS,
            path,
            "value exceeds its byte bound",
        )


def _reject_newlines(value: str, path: str) -> None:
    if "\r" in value or "\n" in value:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "value must not contain a newline",
        )


def _reject_identifier_controls(value: str, path: str) -> None:
    if any(
        category(character) in {"Cc", "Cf", "Cs"}
        or bidirectional(character) in _BIDI_CONTROL_CLASSES
        for character in value
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            path,
            "identifier or stable value contains a control character",
        )
