from ..entities import ShellExecutionErrorCode
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _contains_unsafe_control,
    _literal_option,
    _optional_bounded_int_option,
    _relative_argv_path,
    _validate_filter_paths,
    _validate_known_options,
    policy_denied,
)

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import ShellToolSettings

_FIELD_SEPARATORS = {
    "whitespace": None,
    "tab": "\\t",
    "comma": ",",
    "pipe": "|",
}


def _awk_fields(
    value: object,
    settings: "ShellToolSettings",
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "fields must be a sequence",
        )
    if not value or len(value) > settings.max_awk_fields:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "fields count is out of range",
        )
    fields: list[int] = []
    for field in value:
        if not isinstance(field, int) or isinstance(field, bool):
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "field indexes must be integers",
            )
        if field < 1 or field > settings.max_awk_fields:
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "field index is out of range",
            )
        fields.append(field)
    return tuple(fields)


def _separator_option(
    options: Mapping[str, object],
    name: str,
    *,
    default: str,
    max_bytes: int,
) -> str:
    value = options.get(name, default)
    if not isinstance(value, str) or value == "":
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be a non-empty string",
        )
    if len(value.encode("utf-8")) > max_bytes or _contains_unsafe_control(
        value
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} is unsafe",
        )
    return value


def _optional_pattern_option(
    options: Mapping[str, object],
    settings: "ShellToolSettings",
) -> str | None:
    value = options.get("pattern")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pattern must be a non-empty string",
        )
    if len(
        value.encode("utf-8")
    ) > settings.max_filter_pattern_bytes or _contains_unsafe_control(value):
        raise policy_denied(
            ShellExecutionErrorCode.UNSAFE_FILTER,
            "pattern is unsafe",
        )
    return value


def _awk_program(
    *,
    fields: tuple[int, ...] | None,
    has_pattern: bool,
    start_line: int | None,
    end_line: int | None,
    settings: "ShellToolSettings",
) -> str:
    predicates: list[str] = []
    if start_line is not None:
        predicates.append(f"NR >= {start_line}")
    if end_line is not None:
        predicates.append(f"NR <= {end_line}")
    if has_pattern:
        predicates.append("$0 ~ pat")
    condition = " && ".join(predicates)
    projection = (
        "$0" if fields is None else ", ".join(f"${field}" for field in fields)
    )
    program = (
        f"{{ print {projection} }}"
        if not condition
        else (f"{condition} {{ print {projection} }}")
    )
    if len(program.encode("utf-8")) > settings.max_filter_program_bytes:
        raise policy_denied(
            ShellExecutionErrorCode.UNSAFE_FILTER,
            "awk program is too large",
        )
    selector_count = len(predicates) + (0 if fields is None else len(fields))
    if selector_count > settings.max_filter_selectors:
        raise policy_denied(
            ShellExecutionErrorCode.UNSAFE_FILTER,
            "awk selector count is too large",
        )
    return program


def _awk_path_argument(cwd: Path, path: Path) -> str:
    argument = _relative_argv_path(cwd, path)
    if argument.startswith("-") or "=" in argument:
        return f"./{argument}"
    return argument


def _awk_display_path_argument(display_path: str) -> str:
    if display_path.startswith("-") or "=" in display_path:
        return f"./{display_path}"
    return display_path


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "fields",
            "field_separator",
            "output_separator",
            "pattern",
            "start_line",
            "end_line",
        },
        command="awk",
    )
    _validate_filter_paths(
        context.paths, command="awk", allowed_kinds=("text_file",)
    )
    fields = _awk_fields(request.options.get("fields"), settings)
    field_separator = _literal_option(
        request.options,
        "field_separator",
        default="whitespace",
        allowed=tuple(_FIELD_SEPARATORS),
    )
    output_separator = _separator_option(
        request.options,
        "output_separator",
        default=" ",
        max_bytes=settings.max_awk_separator_bytes,
    )
    pattern = _optional_pattern_option(request.options, settings)
    start_line = _optional_bounded_int_option(
        request.options,
        "start_line",
        min_value=1,
        max_value=2**31 - 1,
    )
    end_line = _optional_bounded_int_option(
        request.options,
        "end_line",
        min_value=1,
        max_value=2**31 - 1,
    )
    if (
        start_line is not None
        and end_line is not None
        and start_line > end_line
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "start_line must not exceed end_line",
        )
    program = _awk_program(
        fields=fields,
        has_pattern=pattern is not None,
        start_line=start_line,
        end_line=end_line,
        settings=settings,
    )
    argv_parts = [context.executable_name]
    separator = _FIELD_SEPARATORS[field_separator]
    if separator is not None:
        argv_parts.extend(("-F", separator))
    argv_parts.extend(("-v", f"OFS={output_separator}"))
    if pattern is not None:
        argv_parts.extend(("-v", f"pat={pattern}"))
    argv_parts.append(program)
    path_arguments = tuple(
        _awk_path_argument(context.workspace.cwd, path.path)
        for path in context.paths
    )
    display_path_arguments = tuple(
        _awk_display_path_argument(path.display_path) for path in context.paths
    )
    argv_parts.extend(path_arguments)
    display_parts = list(argv_parts[: -len(path_arguments)])
    display_parts.extend(display_path_arguments)
    return tuple(argv_parts), tuple(display_parts), None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="awk",
    executable_name="awk",
    dependency_group=ShellDependencyGroup.TEXT_FILTERS,
    container_package_hints=("gawk", "mawk"),
    argv_builder=build_argv,
    supports_double_dash=False,
)
