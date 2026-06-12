from ..entities import (
    GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
    GeneratedOutputPlan,
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
    _bool_option,
    _literal_option,
    _media_display_path_argument,
    _media_path_argument,
    _pdf_page_range,
    _single_path,
    _validate_known_options,
    policy_denied,
)

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..settings import ShellToolSettings


def _raster_dpi_option(
    options: Mapping[str, object],
    settings: "ShellToolSettings",
) -> int:
    value = options.get("dpi", min(150, settings.max_pdf_raster_dpi))
    if not isinstance(value, int) or isinstance(value, bool):
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "dpi must be an integer",
        )
    if value < 1:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "dpi is out of range",
        )
    if value > settings.max_pdf_raster_dpi:
        raise policy_denied(
            ShellExecutionErrorCode.RASTER_DPI_CAP_EXCEEDED,
            "raster DPI is too large",
        )
    return value


def _generated_output_plan(
    settings: "ShellToolSettings",
    *,
    suffix: str,
) -> GeneratedOutputPlan:
    suffix_value = f".{suffix}"
    return GeneratedOutputPlan(
        prefix_name="page",
        display_prefix="GENERATED_PREFIX",
        allowed_suffixes=(suffix_value,),
        suffix_media_types={suffix_value: f"image/{suffix}"},
        max_files=settings.max_output_files,
        max_file_bytes=settings.max_output_file_bytes,
        max_total_bytes=settings.max_total_output_file_bytes,
        max_inline_bytes=settings.max_inline_output_file_bytes,
        max_raster_long_edge_pixels=settings.max_raster_long_edge_pixels,
        max_raster_pixels=settings.max_raster_pixels,
    )


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], GeneratedOutputPlan]:
    request = context.request
    settings = context.settings
    _validate_known_options(
        request.options,
        allowed_options={
            "first_page",
            "last_page",
            "dpi",
            "grayscale",
            "format",
        },
        command="pdftoppm",
    )
    path = _single_path(
        context.paths,
        allowed_kinds=("pdf_file",),
        command="pdftoppm",
    )
    first_page, last_page = _pdf_page_range(
        request.options,
        max_pages=settings.max_pdf_raster_pages,
    )
    if last_page - first_page + 1 > settings.max_output_files:
        raise policy_denied(
            ShellExecutionErrorCode.GENERATED_OUTPUT_CAP_EXCEEDED,
            "generated output file count is too large",
        )
    dpi = _raster_dpi_option(request.options, settings)
    output_format = _literal_option(
        request.options,
        "format",
        default="png",
        allowed=tuple(settings.allowed_pdf_raster_formats),
    )
    grayscale = _bool_option(request.options, "grayscale", default=False)
    output_plan = _generated_output_plan(settings, suffix=output_format)
    context.metadata["page_range"] = {"first": first_page, "last": last_page}
    context.metadata["dpi"] = dpi
    context.metadata["generated_output_display_prefix"] = (
        output_plan.display_prefix
    )
    path_argument = _media_path_argument(context.workspace.cwd, path.path)
    display_path_argument = _media_display_path_argument(path.display_path)
    argv_parts = [
        context.executable_name,
        "-f",
        str(first_page),
        "-l",
        str(last_page),
        "-r",
        str(dpi),
    ]
    if grayscale:
        argv_parts.append("-gray")
    argv_parts.extend(
        (
            f"-{output_format}",
            path_argument,
            GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
        )
    )
    display_parts = list(argv_parts)
    display_parts[-2] = display_path_argument
    display_parts[-1] = output_plan.display_prefix
    return tuple(argv_parts), tuple(display_parts), output_plan


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    return "application/json", ShellOutputKind.GENERATED_FILES


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="pdftoppm",
    executable_name="pdftoppm",
    dependency_group=ShellDependencyGroup.POPPLER,
    container_package_hints=("poppler-utils", "poppler"),
    argv_builder=build_argv,
    output_contract=output_contract,
    media_risk=True,
    supports_double_dash=False,
)
