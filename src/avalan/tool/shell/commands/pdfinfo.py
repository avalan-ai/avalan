from ..entities import ShellExecutionErrorCode
from .base import (
    ShellCommandDefinition,
    ShellCommandPolicyContext,
    ShellDependencyGroup,
)
from .helpers import (
    _bool_option,
    _media_display_path_argument,
    _media_path_argument,
    _optional_bounded_int_option,
    _single_path,
    _validate_known_options,
    policy_denied,
)

from collections.abc import Mapping


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], None]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={"boxes", "first_page", "iso_dates", "last_page"},
        command="pdfinfo",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("pdf_file",),
        command="pdfinfo",
    )
    boxes = _bool_option(request.options, "boxes", default=False)
    iso_dates = _bool_option(request.options, "iso_dates", default=False)
    page_range = _optional_pdf_page_range(
        request.options,
        max_pages=settings.max_pdf_text_pages,
        default_first_page=boxes,
    )
    path_argument = _media_path_argument(context.workspace.cwd, path.path)
    display_path_argument = _media_display_path_argument(path.display_path)
    argv_parts = [context.executable_name]
    if page_range is not None:
        first_page, last_page = page_range
        context.metadata["page_range"] = {
            "first": first_page,
            "last": last_page,
        }
        argv_parts.extend(("-f", str(first_page), "-l", str(last_page)))
    if boxes:
        argv_parts.append("-box")
    if iso_dates:
        argv_parts.append("-isodates")
    argv_parts.append(path_argument)
    display_parts = list(argv_parts)
    display_parts[-1] = display_path_argument
    return tuple(argv_parts), tuple(display_parts), None


def _optional_pdf_page_range(
    options: Mapping[str, object],
    *,
    max_pages: int,
    default_first_page: bool,
) -> tuple[int, int] | None:
    first_page = _optional_bounded_int_option(
        options,
        "first_page",
        min_value=1,
        max_value=2**31 - 1,
    )
    last_page = _optional_bounded_int_option(
        options,
        "last_page",
        min_value=1,
        max_value=2**31 - 1,
    )
    if first_page is None and last_page is None:
        if not default_first_page:
            return None
        first_page = 1
        last_page = 1
    elif first_page is None:
        first_page = 1
    elif last_page is None:
        last_page = first_page
    assert first_page is not None, "first_page must be normalized"
    assert last_page is not None, "last_page must be normalized"
    if first_page > last_page:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_PAGE_RANGE,
            "first_page must not exceed last_page",
        )
    page_count = last_page - first_page + 1
    if page_count > max_pages:
        raise policy_denied(
            ShellExecutionErrorCode.PDF_PAGE_CAP_EXCEEDED,
            "PDF page range is too large",
        )
    return first_page, last_page


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="pdfinfo",
    executable_name="pdfinfo",
    dependency_group=ShellDependencyGroup.POPPLER,
    container_package_hints=("poppler-utils", "poppler"),
    argv_builder=build_argv,
    media_risk=True,
    supports_double_dash=False,
)
