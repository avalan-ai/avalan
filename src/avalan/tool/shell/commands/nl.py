from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bounded_int_option,
    _literal_option,
    _option_safe_path_argument,
    _single_path,
    _validate_known_options,
)

_BODY_NUMBERING_STYLES = {
    "all": "a",
    "nonempty": "t",
    "none": "n",
}
_NUMBER_FORMATS = {
    "left": "ln",
    "right": "rn",
    "right_zero": "rz",
}
_NUMBER_SEPARATORS = {
    "colon_space": ": ",
    "space": " ",
    "tab": "\t",
    "two_spaces": "  ",
}
_SECTION_DELIMITER = "\x01\x02"
RESERVED_SECTION_DELIMITER_SEQUENCES = tuple(
    (_SECTION_DELIMITER * repeat).encode("utf-8") for repeat in (1, 2, 3)
)
_MAX_LINE_NUMBER_VALUE = 2**31 - 1
_MAX_NUMBER_WIDTH = 100


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={
            "body_numbering",
            "line_increment",
            "number_format",
            "number_separator",
            "number_width",
            "starting_line_number",
        },
        command="nl",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("file", "text_file"),
        command="nl",
    )
    body_numbering = _literal_option(
        request.options,
        "body_numbering",
        default="all",
        allowed=tuple(_BODY_NUMBERING_STYLES),
    )
    number_format = _literal_option(
        request.options,
        "number_format",
        default="right",
        allowed=tuple(_NUMBER_FORMATS),
    )
    number_separator = _literal_option(
        request.options,
        "number_separator",
        default="tab",
        allowed=tuple(_NUMBER_SEPARATORS),
    )
    starting_line_number = _bounded_int_option(
        request.options,
        "starting_line_number",
        default=1,
        min_value=1,
        max_value=_MAX_LINE_NUMBER_VALUE,
    )
    line_increment = _bounded_int_option(
        request.options,
        "line_increment",
        default=1,
        min_value=1,
        max_value=_MAX_LINE_NUMBER_VALUE,
    )
    number_width = _bounded_int_option(
        request.options,
        "number_width",
        default=6,
        min_value=1,
        max_value=_MAX_NUMBER_WIDTH,
    )
    path_argument = _option_safe_path_argument(
        context.workspace.cwd,
        path.path,
    )
    argv = (
        context.executable_name,
        "-b",
        _BODY_NUMBERING_STYLES[body_numbering],
        "-n",
        _NUMBER_FORMATS[number_format],
        f"-s{_NUMBER_SEPARATORS[number_separator]}",
        "-v",
        str(starting_line_number),
        "-i",
        str(line_increment),
        "-w",
        str(number_width),
        "-d",
        _SECTION_DELIMITER,
        "-p",
        "--",
        path_argument,
    )
    return argv, argv, None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="nl",
    executable_name="nl",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("coreutils",),
    argv_builder=build_argv,
)
