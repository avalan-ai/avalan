from ....types import assert_int_sequence as _assert_int_sequence
from ....types import assert_string_sequence as _assert_string_sequence
from ..entities import (
    PathOperand,
    ShellCommandRequest,
)

from collections.abc import Mapping, Sequence
from typing import Literal


def _line_reader_request(
    *,
    command: Literal["head", "tail"],
    path: str,
    lines: int,
    cwd: str | None,
    timeout_seconds: float | None,
    max_stdout_bytes: int | None,
    max_stderr_bytes: int | None,
    byte_count: int | None = None,
    start_line: int | None = None,
    start_byte: int | None = None,
) -> ShellCommandRequest:
    options: dict[str, object] = {"lines": lines}
    if command == "head":
        options["byte_count"] = byte_count
    else:
        options["start_line"] = start_line
        options["byte_count"] = byte_count
        options["start_byte"] = start_byte
    return ShellCommandRequest(
        tool_name=f"shell.{command}",
        command=command,
        options=options,
        paths=_path_operands((path,), kind="text_file"),
        cwd=_optional_cwd(cwd),
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _optional_cwd(cwd: str | None) -> str | None:
    return None if cwd == "" else cwd


def _path_operands(
    paths: Sequence[str],
    *,
    kind: Literal[
        "any",
        "file",
        "text_file",
        "json_file",
        "pdf_file",
        "image_file",
    ],
) -> tuple[PathOperand, ...]:
    normalized_paths = _string_tuple(paths, "paths")
    return tuple(
        PathOperand(
            name=f"path_{index}",
            path=path,
            kind=kind,
            access="read",
        )
        for index, path in enumerate(normalized_paths)
    )


def _string_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    _assert_string_sequence(value, name)
    return tuple(value)


def _copied_json_schema_properties(
    properties: Mapping[str, object],
) -> dict[str, object]:
    return {
        name: (
            dict(schema_property)
            if isinstance(schema_property, Mapping)
            else schema_property
        )
        for name, schema_property in properties.items()
    }


def _optional_int_tuple(
    value: Sequence[int] | None,
    name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    _assert_int_sequence(value, name)
    return tuple(value)


def _optional_string_tuple(
    value: Sequence[str] | None,
    name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _string_tuple(value, name)
