from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _media_display_path_argument,
    _media_path_argument,
    _pdf_page_range,
    _single_path,
    _validate_known_options,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "first_page",
            "last_page",
            "layout",
            "no_page_breaks",
        },
        command="pdftotext",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("pdf_file",),
        command="pdftotext",
    )
    first_page, last_page = _pdf_page_range(
        request.options,
        max_pages=settings.max_pdf_text_pages,
    )
    layout = _bool_option(request.options, "layout", default=False)
    no_page_breaks = _bool_option(
        request.options,
        "no_page_breaks",
        default=False,
    )
    context.metadata["page_range"] = {"first": first_page, "last": last_page}
    path_argument = _media_path_argument(context.workspace.cwd, path.path)
    display_path_argument = _media_display_path_argument(path.display_path)
    argv_parts = [
        context.executable_name,
        "-f",
        str(first_page),
        "-l",
        str(last_page),
    ]
    if layout:
        argv_parts.append("-layout")
    if no_page_breaks:
        argv_parts.append("-nopgbrk")
    argv_parts.extend((path_argument, "-"))
    display_parts = list(argv_parts)
    display_parts[-2] = display_path_argument
    return tuple(argv_parts), tuple(display_parts), None


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="pdftotext",
    executable_name="pdftotext",
    dependency_group=ShellDependencyGroup.POPPLER,
    container_package_hints=("poppler-utils", "poppler"),
    argv_builder=build_argv,
    media_risk=True,
    supports_double_dash=False,
)
