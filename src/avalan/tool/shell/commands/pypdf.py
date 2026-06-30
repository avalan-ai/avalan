from ..entities import (
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellOutputKind,
)
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _literal_option,
    _media_path_argument,
    _pdf_page_range,
    _single_path,
    _validate_known_options,
    policy_denied,
)
from .python_pdf import (
    add_python_pdf_unavailable_status,
    python_pdf_argv_prefix,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={"mode", "first_page", "last_page"},
        command="pypdf",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("pdf_file",),
        command="pypdf",
    )
    mode = _literal_option(
        request.options,
        "mode",
        default="metadata",
        allowed=("metadata", "text"),
    )
    add_python_pdf_unavailable_status(context)
    path_argument = _media_path_argument(context.workspace.cwd, path.path)
    argv_parts = python_pdf_argv_prefix(context, "pypdf")
    argv_parts.extend(("--mode", mode))
    if mode == "text":
        first_page, last_page = _pdf_page_range(
            request.options,
            max_pages=settings.max_pdf_text_pages,
        )
        context.metadata["page_range"] = {
            "first": first_page,
            "last": last_page,
        }
        argv_parts.extend(
            (
                "--first-page",
                str(first_page),
                "--last-page",
                str(last_page),
            )
        )
    elif "first_page" in request.options or "last_page" in request.options:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "pypdf metadata mode does not accept page options",
        )
    argv_parts.extend(("--", path_argument))
    return tuple(argv_parts), tuple(argv_parts), None


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    if request.options.get("mode", "metadata") == "metadata":
        return "application/json", ShellOutputKind.JSON
    return "text/plain", ShellOutputKind.TEXT


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="pypdf",
    executable_name="python3",
    dependency_group=ShellDependencyGroup.PYTHON_PDF,
    container_package_hints=("python3", "avalan", "pypdf"),
    argv_builder=build_argv,
    output_contract=output_contract,
    media_risk=True,
    supports_double_dash=False,
)
