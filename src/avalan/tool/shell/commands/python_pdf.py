from ..entities import GeneratedOutputPlan
from .base import ShellCommandPolicyContext

PYTHON_PDF_RUNNER_MODULE = "avalan.tool.shell.python_pdf"
PYTHON_PDF_UNAVAILABLE_EXIT_CODE = 127


def python_pdf_argv_prefix(
    context: ShellCommandPolicyContext,
    command: str,
) -> list[str]:
    assert isinstance(
        context,
        ShellCommandPolicyContext,
    ), "context must be a shell command policy context"
    assert (
        isinstance(command, str) and command.strip()
    ), "command must be a non-empty string"
    return [
        context.executable_name,
        "-I",
        "-m",
        PYTHON_PDF_RUNNER_MODULE,
        command,
    ]


def add_python_pdf_unavailable_status(
    context: ShellCommandPolicyContext,
) -> None:
    assert isinstance(
        context,
        ShellCommandPolicyContext,
    ), "context must be a shell command policy context"
    context.metadata["exit_code_statuses"] = {
        PYTHON_PDF_UNAVAILABLE_EXIT_CODE: "command_unavailable",
    }


def reportlab_output_plan(
    context: ShellCommandPolicyContext,
) -> GeneratedOutputPlan:
    assert isinstance(
        context,
        ShellCommandPolicyContext,
    ), "context must be a shell command policy context"
    settings = context.settings
    return GeneratedOutputPlan(
        prefix_name="document",
        display_prefix="GENERATED_PREFIX",
        allowed_suffixes=(".pdf",),
        suffix_media_types={".pdf": "application/pdf"},
        max_files=1,
        max_file_bytes=settings.max_output_file_bytes,
        max_total_bytes=settings.max_total_output_file_bytes,
        max_inline_bytes=settings.max_inline_output_file_bytes,
    )
