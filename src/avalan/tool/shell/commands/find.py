from ..entities import ShellExecutionErrorCode
from .base import (
    NormalizedPath,
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bounded_int_option,
    _contains_traversal,
    _contains_unsafe_control,
    _literal_option,
    _option_safe_display_path_argument,
    _option_safe_path_argument,
    _validate_known_options,
    _validate_path_kind,
    is_denied_display_path,
    path_matches_sensitive_denylist,
    policy_denied,
)

_MAX_FIND_DEPTH = 10
_MAX_FIND_NAME_BYTES = 255
_SENSITIVE_PRUNE_NAMES = (
    ".aws",
    ".config",
    ".docker",
    ".env*",
    ".git",
    ".git-credentials",
    ".gcloud",
    ".kube",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".ssh",
    "*_rsa",
    "*.key",
    "*.pem",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
)
_FIND_EXPRESSION_TOKENS = frozenset(("!", "(", ")"))
_FIND_GLOB_META_CHARACTERS = frozenset("*?[")


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"entry_type", "max_depth", "name"},
        command="find",
    )
    for path in context.paths:
        _validate_path_kind(
            path,
            allowed_kinds=("file", "directory", "any"),
            command="find",
        )
    max_depth = _bounded_int_option(
        request.options,
        "max_depth",
        default=3,
        min_value=0,
        max_value=_MAX_FIND_DEPTH,
    )
    entry_type = _literal_option(
        request.options,
        "entry_type",
        default="any",
        allowed=("any", "file", "directory"),
    )
    name = _find_name_option(request.options.get("name"), context)
    root_arguments = _find_root_arguments(context)
    display_root_arguments = _find_display_root_arguments(context.paths)
    argv_parts = [
        context.executable_name,
        *root_arguments,
        "-maxdepth",
        str(max_depth),
        "(",
        *_find_prune_expression(context.settings.allow_hidden),
        ")",
        "-prune",
        "-o",
        *_find_type_predicate(entry_type),
    ]
    display_parts = [
        context.executable_name,
        *display_root_arguments,
        "-maxdepth",
        str(max_depth),
        *_find_type_predicate(entry_type),
    ]
    if name is not None:
        argv_parts.extend(("-name", name))
        display_parts.extend(("-name", name))
    argv_parts.append("-print")
    display_parts.append("-print")
    return tuple(argv_parts), tuple(display_parts), None


def filter_output(value: str) -> str:
    assert isinstance(value, str), "value must be a string"
    return "\n".join(_filtered_output_line(line) for line in value.split("\n"))


def _find_name_option(
    value: object,
    context: ShellCommandPolicyContext,
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "name must be a non-empty string",
        )
    if (
        "/" in value
        or _contains_traversal(value)
        or _contains_unsafe_control(value)
    ):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "name must be a safe basename",
        )
    if any(character in value for character in _FIND_GLOB_META_CHARACTERS):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "name must be exact",
        )
    if len(value.encode("utf-8")) > _MAX_FIND_NAME_BYTES:
        raise policy_denied(
            ShellExecutionErrorCode.ARGUMENT_TOO_LARGE,
            "name is too large",
        )
    if not context.settings.allow_hidden and value.startswith("."):
        raise policy_denied(
            ShellExecutionErrorCode.HIDDEN_PATH,
            "hidden names are disabled",
        )
    if path_matches_sensitive_denylist(value):
        raise policy_denied(
            ShellExecutionErrorCode.SENSITIVE_PATH,
            "name is denied",
        )
    return value


def _find_root_arguments(
    context: ShellCommandPolicyContext,
) -> tuple[str, ...]:
    if not context.paths:
        return (".",)
    return tuple(
        _find_safe_path_argument(
            _option_safe_path_argument(context.workspace.cwd, path.path)
        )
        for path in context.paths
    )


def _find_display_root_arguments(
    paths: tuple[NormalizedPath, ...],
) -> tuple[str, ...]:
    if not paths:
        return (".",)
    return tuple(
        _find_safe_path_argument(
            _option_safe_display_path_argument(path.display_path)
        )
        for path in paths
    )


def _find_safe_path_argument(path_argument: str) -> str:
    if path_argument in _FIND_EXPRESSION_TOKENS:
        return f"./{path_argument}"
    return path_argument


def _find_prune_expression(allow_hidden: bool) -> tuple[str, ...]:
    tests: list[tuple[str, ...]] = []
    if not allow_hidden:
        tests.append(("-name", ".*", "-a", "!", "-name", "."))
    tests.extend(_find_sensitive_prune_tests())
    expression: list[str] = []
    for test in tests:
        if expression:
            expression.append("-o")
        expression.extend(test)
    return tuple(expression)


def _find_sensitive_prune_tests() -> tuple[tuple[str, ...], ...]:
    return tuple(("-iname", name) for name in _SENSITIVE_PRUNE_NAMES)


def _find_type_predicate(entry_type: str) -> tuple[str, ...]:
    if entry_type == "file":
        return "-type", "f"
    if entry_type == "directory":
        return "-type", "d"
    return ()


def _filtered_output_line(line: str) -> str:
    if is_denied_display_path(line.removeprefix("./").rstrip("/")):
        return "[redacted_path]"
    return line


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="find",
    executable_name="find",
    dependency_group=ShellDependencyGroup.CORE,
    container_package_hints=("findutils",),
    argv_builder=build_argv,
    output_filter=filter_output,
    supports_double_dash=False,
)
