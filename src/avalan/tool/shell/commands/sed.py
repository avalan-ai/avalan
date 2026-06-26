from ..entities import ShellExecutionErrorCode
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _contains_unsafe_control,
    _option_safe_display_path_argument,
    _option_safe_path_argument,
    _optional_bounded_int_option,
    _validate_filter_paths,
    _validate_known_options,
    policy_denied,
)

from collections.abc import Mapping, Sequence
from re import compile as compile_pattern
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import ShellToolSettings

_LINE_RANGE_PATTERN = compile_pattern(r"^[1-9][0-9]*(,[1-9][0-9]*)?$")
_SED_LITERAL_PATTERN_ESCAPES = frozenset(("\\", "/", ".", "*", "[", "^", "$"))


def _sed_selectors(
    options: Mapping[str, object],
    settings: "ShellToolSettings",
) -> tuple[str, ...]:
    line_ranges = _optional_string_sequence(
        options.get("line_ranges"), "line_ranges"
    )
    patterns = _optional_string_sequence(options.get("patterns"), "patterns")
    start_line = _optional_bounded_int_option(
        options,
        "start_line",
        min_value=1,
        max_value=2**31 - 1,
    )
    end_line = _optional_bounded_int_option(
        options,
        "end_line",
        min_value=1,
        max_value=2**31 - 1,
    )
    has_line_window = start_line is not None or end_line is not None
    if (
        start_line is not None
        and end_line is not None
        and start_line > end_line
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "start_line must not exceed end_line",
        )
    if not line_ranges and not patterns and not has_line_window:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "sed requires at least one selector",
        )
    selector_count = len(line_ranges) + len(patterns) + int(has_line_window)
    if selector_count > settings.max_filter_selectors:
        raise policy_denied(
            ShellExecutionErrorCode.UNSAFE_FILTER,
            "sed selector count is too large "
            f"({selector_count} > {settings.max_filter_selectors})",
        )
    selectors: list[str] = []
    total_selector_bytes = 0
    for line_range in line_ranges:
        if not _LINE_RANGE_PATTERN.match(line_range):
            raise policy_denied(
                ShellExecutionErrorCode.UNSAFE_FILTER,
                "sed line range is unsafe",
            )
        if "," in line_range:
            first, last = (int(part) for part in line_range.split(",", 1))
            if first > last:
                raise policy_denied(
                    ShellExecutionErrorCode.INVALID_OPTION,
                    "sed line range is inverted",
                )
        selector = f"{line_range}p"
        total_selector_bytes = _add_selector_bytes(
            total_selector_bytes,
            selector,
            settings,
        )
        selectors.append(selector)
    if has_line_window:
        selector = _sed_line_window_selector(start_line, end_line)
        total_selector_bytes = _add_selector_bytes(
            total_selector_bytes,
            selector,
            settings,
        )
        selectors.append(selector)
    for pattern in patterns:
        if len(
            pattern.encode("utf-8")
        ) > settings.max_filter_pattern_bytes or _contains_unsafe_sed_pattern(
            pattern
        ):
            raise policy_denied(
                ShellExecutionErrorCode.UNSAFE_FILTER,
                "sed pattern is unsafe",
            )
        selector = f"/{_escape_sed_pattern(pattern)}/p"
        total_selector_bytes = _add_selector_bytes(
            total_selector_bytes,
            selector,
            settings,
        )
        selectors.append(selector)
    return tuple(selectors)


def _sed_line_window_selector(
    start_line: int | None,
    end_line: int | None,
) -> str:
    if start_line is None:
        assert end_line is not None
        return f"1,{end_line}p"
    if end_line is None:
        return f"{start_line},$p"
    return f"{start_line},{end_line}p"


def _optional_string_sequence(value: object, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            f"{name} must be a sequence",
        )
    values: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise policy_denied(
                ShellExecutionErrorCode.INVALID_OPTION,
                f"{name} entries must be non-empty strings",
            )
        values.append(item)
    return tuple(values)


def _add_selector_bytes(
    total_bytes: int,
    selector: str,
    settings: "ShellToolSettings",
) -> int:
    selector_bytes = len(selector.encode("utf-8"))
    total = total_bytes + selector_bytes
    if (
        selector_bytes > settings.max_filter_pattern_bytes
        or total > settings.max_filter_program_bytes
    ):
        raise policy_denied(
            ShellExecutionErrorCode.UNSAFE_FILTER,
            "sed selector is too large",
        )
    return total


def _escape_sed_pattern(pattern: str) -> str:
    return "".join(
        (
            f"\\{character}"
            if character in _SED_LITERAL_PATTERN_ESCAPES
            else character
        )
        for character in pattern
    )


def _contains_unsafe_sed_pattern(value: str) -> bool:
    return ";" in value or _contains_unsafe_control(value)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={
            "line_ranges",
            "patterns",
            "start_line",
            "end_line",
        },
        command="sed",
    )
    _validate_filter_paths(
        context.paths, command="sed", allowed_kinds=("text_file",)
    )
    selectors = _sed_selectors(request.options, context.settings)
    path_arguments = tuple(
        _option_safe_path_argument(context.workspace.cwd, path.path)
        for path in context.paths
    )
    display_path_arguments = tuple(
        _option_safe_display_path_argument(path.display_path)
        for path in context.paths
    )
    argv_parts = [context.executable_name, "-n"]
    for selector in selectors:
        argv_parts.extend(("-e", selector))
    argv_parts.extend(path_arguments)
    display_parts = [context.executable_name, "-n"]
    for selector in selectors:
        display_parts.extend(("-e", selector))
    display_parts.extend(display_path_arguments)
    return tuple(argv_parts), tuple(display_parts), None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="sed",
    executable_name="sed",
    dependency_group=ShellDependencyGroup.TEXT_FILTERS,
    container_package_hints=("sed",),
    argv_builder=build_argv,
    supports_double_dash=False,
)
