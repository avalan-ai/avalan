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
    _literal_option,
    _required_string_option,
    _validate_known_options,
    policy_denied,
)
from .python_pdf import (
    add_python_pdf_unavailable_status,
    python_pdf_argv_prefix,
    reportlab_output_plan,
)


def build_argv(
    context: ShellCommandPolicyContext,
) -> tuple[tuple[str, ...], tuple[str, ...], GeneratedOutputPlan]:
    request = context.request
    _validate_known_options(
        request.options,
        allowed_options={"text", "title", "page_size"},
        command="reportlab",
    )
    if context.paths:
        raise policy_denied(
            ShellExecutionErrorCode.INVALID_OPTION,
            "reportlab does not accept input paths",
        )
    text = _required_string_option(request.options, "text")
    title = _required_string_option(request.options, "title")
    page_size = _literal_option(
        request.options,
        "page_size",
        default="letter",
        allowed=("letter", "a4"),
    )
    output_plan = reportlab_output_plan(context)
    add_python_pdf_unavailable_status(context)
    context.metadata["generated_output_display_prefix"] = (
        output_plan.display_prefix
    )
    context.metadata["page_size"] = page_size
    argv_parts = python_pdf_argv_prefix(context, "reportlab")
    argv_parts.extend(
        (
            "--page-size",
            page_size,
            "--title",
            title,
            "--text",
            text,
            "--output",
            GENERATED_OUTPUT_PREFIX_PLACEHOLDER,
        )
    )
    display_parts = list(argv_parts)
    display_parts[-1] = f"{output_plan.display_prefix}.pdf"
    return tuple(argv_parts), tuple(display_parts), output_plan


def output_contract(
    request: ShellCommandRequest,
) -> tuple[str, ShellOutputKind]:
    return "application/json", ShellOutputKind.GENERATED_FILES


COMMAND_DEFINITION = ShellCommandDefinition(
    logical_id="reportlab",
    executable_name="python3",
    dependency_group=ShellDependencyGroup.PYTHON_PDF,
    container_package_hints=("python3", "avalan", "reportlab"),
    argv_builder=build_argv,
    output_contract=output_contract,
    media_risk=True,
    supports_double_dash=False,
)
