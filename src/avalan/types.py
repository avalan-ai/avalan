from collections.abc import Mapping
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


def _int_value(value: object, field_name: str) -> int:
    assert isinstance(value, int), f"{field_name} must be an integer"
    assert not isinstance(value, bool), f"{field_name} must be an integer"
    return value


def _number_value(value: object, field_name: str) -> int | float:
    assert isinstance(value, int | float), f"{field_name} must be numeric"
    assert not isinstance(value, bool), f"{field_name} must be numeric"
    return value
