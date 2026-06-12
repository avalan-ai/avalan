from collections.abc import Mapping, Sequence
from datetime import datetime
from os import PathLike
from os.path import isabs
from re import compile as compile_pattern
from typing import TypeAlias

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = (
    JsonScalar | tuple["JsonValue", ...] | Mapping[str, "JsonValue"]
)
MutableJsonValue: TypeAlias = (
    JsonScalar | list["MutableJsonValue"] | dict[str, "MutableJsonValue"]
)
LooseJsonValue: TypeAlias = JsonScalar | list[object] | dict[str, object]
JsonObject: TypeAlias = dict[str, MutableJsonValue]

_ENV_NAME_PATTERN = compile_pattern(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MEDIA_TYPE_PATTERN = compile_pattern(
    r"^[A-Za-z0-9][A-Za-z0-9.+-]*/[A-Za-z0-9][A-Za-z0-9.+-]*$"
)
_SAFE_PATH_NAME_PATTERN = compile_pattern(r"^[A-Za-z0-9_-]+$")
_SHA256_HEX_PATTERN = compile_pattern(r"^[0-9a-f]{64}$")


def assert_bool(value: object, field_name: str) -> None:
    assert isinstance(value, bool), f"{field_name} must be a boolean"


def assert_non_empty_string(value: object, field_name: str) -> None:
    assert isinstance(value, str), f"{field_name} must be a string"
    assert value.strip(), f"{field_name} must not be empty"


def assert_int(value: object, field_name: str) -> None:
    _int_value(value, field_name)


def assert_positive_int(value: object, field_name: str) -> None:
    assert _int_value(value, field_name) > 0, f"{field_name} must be positive"


def assert_optional_positive_int(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert_positive_int(value, field_name)


def assert_positive_number(value: object, field_name: str) -> None:
    assert (
        _number_value(value, field_name) > 0
    ), f"{field_name} must be positive"


def assert_optional_positive_number(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert_positive_number(value, field_name)


def assert_non_negative_int(value: object, field_name: str) -> None:
    assert (
        _int_value(value, field_name) >= 0
    ), f"{field_name} must not be negative"


def assert_optional_non_negative_int(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert_non_negative_int(value, field_name)


def assert_string_tuple(value: object, field_name: str) -> None:
    assert isinstance(value, tuple), f"{field_name} must be a tuple"
    for item in value:
        assert_non_empty_string(item, field_name)


def assert_string_sequence(value: object, field_name: str) -> None:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    for item in value:
        assert isinstance(item, str), f"{field_name} must contain strings"


def assert_counter(value: object | None, field_name: str) -> None:
    assert_optional_non_negative_int(value, field_name)


def assert_non_negative_number(value: object, field_name: str) -> None:
    assert (
        _number_value(value, field_name) >= 0
    ), f"{field_name} must not be negative"


def assert_optional_non_negative_number(
    value: object | None,
    field_name: str,
) -> None:
    if value is None:
        return
    assert_non_negative_number(value, field_name)


def assert_non_empty_string_sequence(
    value: object,
    field_name: str,
) -> None:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    assert value, f"{field_name} must not be empty"
    for item in value:
        assert_non_empty_string(item, field_name)


def assert_string_mapping(value: object, field_name: str) -> None:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    for key, item in value.items():
        assert_non_empty_string(key, f"{field_name} key")
        assert_non_empty_string(item, f"{field_name}.{key}")


def assert_known_string(
    value: object,
    field_name: str,
    known_values: Sequence[str],
) -> None:
    assert_non_empty_string(value, field_name)
    assert known_values, "known_values must not be empty"
    for known_value in known_values:
        assert_non_empty_string(known_value, "known_values")
    assert value in known_values, f"{field_name} contains unsupported value"


def assert_known_string_sequence(
    value: object,
    field_name: str,
    known_values: Sequence[str],
) -> None:
    assert_non_empty_string_sequence(value, field_name)
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    assert known_values, "known_values must not be empty"
    for known_value in known_values:
        assert_non_empty_string(known_value, "known_values")
    for item in value:
        assert item in known_values, f"{field_name} contains unsupported value"


def assert_optional_bounded_number(
    value: object | None,
    field_name: str,
    *,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> None:
    if value is None:
        return
    numeric_value = _number_value(value, field_name)
    if min_value is not None:
        _assert_bound_number(min_value, f"{field_name} minimum")
        if min_inclusive:
            assert (
                numeric_value >= min_value
            ), f"{field_name} must be at least {min_value}"
        else:
            assert (
                numeric_value > min_value
            ), f"{field_name} must be greater than {min_value}"
    if max_value is not None:
        _assert_bound_number(max_value, f"{field_name} maximum")
        if max_inclusive:
            assert (
                numeric_value <= max_value
            ), f"{field_name} must be at most {max_value}"
        else:
            assert (
                numeric_value < max_value
            ), f"{field_name} must be less than {max_value}"


def assert_env_name(value: object, field_name: str) -> None:
    assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert _ENV_NAME_PATTERN.match(value), f"{field_name} must be an env name"


def assert_safe_suffix(value: object, field_name: str) -> None:
    assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert "\x00" not in value, f"{field_name} must not contain NUL"
    assert "/" not in value, f"{field_name} must not contain path separators"
    assert "\\" not in value, f"{field_name} must not contain path separators"
    assert value not in (".", ".."), f"{field_name} must be a file suffix"


def assert_safe_path_name(value: object, field_name: str) -> None:
    assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert (
        _SAFE_PATH_NAME_PATTERN.match(value) is not None
    ), f"{field_name} must be a safe path name"


def assert_safe_suffix_sequence(value: object, field_name: str) -> None:
    assert_non_empty_string_sequence(value, field_name)
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    for item in value:
        assert_safe_suffix(item, field_name)


def assert_media_type(value: object, field_name: str) -> None:
    assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert (
        _MEDIA_TYPE_PATTERN.match(value) is not None
    ), f"{field_name} must be a media type"


def assert_suffix_media_type_mapping(value: object, field_name: str) -> None:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    for suffix, media_type in value.items():
        assert_safe_suffix(suffix, f"{field_name} key")
        assert_media_type(media_type, f"{field_name}.{suffix}")


def assert_sha256_hex(value: object, field_name: str = "sha256") -> None:
    assert_non_empty_string(value, field_name)
    assert isinstance(value, str), f"{field_name} must be a string"
    assert (
        _SHA256_HEX_PATTERN.match(value) is not None
    ), f"{field_name} must be a lowercase SHA-256 hex digest"


def assert_absolute_path(value: object, field_name: str) -> None:
    assert isinstance(
        value, str | PathLike
    ), f"{field_name} must be a string or path"
    path = str(value)
    assert "\x00" not in path, f"{field_name} must not contain NUL"
    assert isabs(path), f"{field_name} must be absolute"


def assert_absolute_path_sequence(value: object, field_name: str) -> None:
    assert isinstance(value, Sequence), f"{field_name} must be a sequence"
    assert not isinstance(
        value, str | bytes
    ), f"{field_name} must be a sequence"
    for item in value:
        assert_absolute_path(item, field_name)


def assert_absolute_path_mapping(value: object, field_name: str) -> None:
    assert isinstance(value, Mapping), f"{field_name} must be a mapping"
    for key, item in value.items():
        assert_non_empty_string(key, f"{field_name} key")
        assert_absolute_path(item, f"{field_name}.{key}")


def coerce_datetime(value: object, field_name: str = "datetime") -> datetime:
    if isinstance(value, datetime):
        return value
    assert isinstance(value, str), f"{field_name} must be a datetime string"
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AssertionError(
            f"{field_name} must be an ISO datetime string"
        ) from error


def _int_value(value: object, field_name: str) -> int:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    return value


def _number_value(value: object, field_name: str) -> int | float:
    assert isinstance(value, int | float), f"{field_name} must be numeric"
    assert not isinstance(value, bool), f"{field_name} must be numeric"
    return value


def _assert_bound_number(value: object, field_name: str) -> None:
    _number_value(value, field_name)
