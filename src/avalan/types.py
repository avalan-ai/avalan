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
