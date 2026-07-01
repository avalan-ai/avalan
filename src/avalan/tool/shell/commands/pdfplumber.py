from ..entities import ShellCommandRequest, ShellOutputKind
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _literal_option,
    _media_path_argument,
    _pdf_page_range,
    _single_path,
    _validate_known_options,
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
        allowed_options={
            "mode",
            "first_page",
            "last_page",
            "layout",
        },
        command="pdfplumber",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("pdf_file",),
        command="pdfplumber",
    )
    mode = _literal_option(
        request.options,
        "mode",
        default="text",
        allowed=("text", "tables"),
    )
    layout = _bool_option(request.options, "layout", default=False)
    first_page, last_page = _pdf_page_range(
        request.options,
        max_pages=settings.max_pdf_text_pages,
    )
    add_python_pdf_unavailable_status(context)
    context.metadata["page_range"] = {"first": first_page, "last": last_page}
    path_argument = _media_path_argument(context.workspace.cwd, path.path)
    argv_parts = python_pdf_argv_prefix(context, "pdfplumber")
    argv_parts.extend(
        (
            "--mode",
            mode,
            "--first-page",
            str(first_page),
            "--last-page",
            str(last_page),
        )
    )
    if layout:
        argv_parts.append("--layout")
    argv_parts.extend(("--", path_argument))
    return tuple(argv_parts), tuple(argv_parts), None


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    if request.options.get("mode", "text") == "tables":
        return "application/json", ShellOutputKind.JSON
    return "text/plain", ShellOutputKind.TEXT


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="pdfplumber",
    executable_name="python3",
    dependency_group=ShellDependencyGroup.PYTHON_PDF,
    container_package_hints=("python3", "avalan", "pdfplumber"),
    argv_builder=build_argv,
    output_contract=output_contract,
    media_risk=True,
    supports_double_dash=False,
)
