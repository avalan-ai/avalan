from ...entities import ToolCallContext
from ...types import assert_int_sequence as _assert_int_sequence
from ...types import assert_string_sequence as _assert_string_sequence
from .. import Tool
from .entities import (
    ExecutionResult,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
)
from .executor import CommandExecutor
from .formatting import format_shell_result
from .policy import ExecutionPolicy
from .settings import ShellToolSettings

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

ShellResultFormatter = Callable[[ExecutionResult], str]


class _ShellCommandTool(Tool, ABC):
    _executor: CommandExecutor
    _formatter: ShellResultFormatter
    _policy: ExecutionPolicy
    _settings: ShellToolSettings
    supports_streaming = True

    def __init__(
        self,
        *,
        command: str,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None,
    ) -> None:
        super().__init__()
        self.__name__ = command
        self._settings = settings
        self._policy = policy
        self._executor = executor
        self._formatter = formatter or self._format_result

    async def _execute_request(
        self,
        request: ShellCommandRequest,
        *,
        context: ToolCallContext,
    ) -> str:
        try:
            spec = await self._policy.normalize(request)
        except ShellPolicyDenied as error:
            return self._formatter(_policy_denied_result(request, error))
        if context.stream_event is not None:
            result = await self._executor.execute(
                spec,
                stream=context.stream_event,
            )
        else:
            result = await self._executor.execute(spec)
        return self._formatter(result)

    def _format_result(self, result: ExecutionResult) -> str:
        return format_shell_result(result, settings=self._settings)

    @abstractmethod
    async def __call__(self, *args: object, **kwargs: object) -> str:
        raise NotImplementedError


class RgTool(_ShellCommandTool):
    """Search workspace text with ripgrep.

    Args:
        pattern: Search pattern to pass as a literal tool argument.
        paths: Workspace-relative files or directories to search.
        cwd: Workspace-relative working directory for the command.
        case: Case-sensitivity mode.
        fixed_strings: Treat the pattern as a fixed string.
        context_lines: Number of context lines around each match.
        max_matches_per_file: Maximum matches to return per file.
        globs: Include or exclude glob patterns for ripgrep.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="rg",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        pattern: str,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        case: Literal["sensitive", "insensitive", "smart"] = "sensitive",
        fixed_strings: bool = False,
        context_lines: int = 0,
        max_matches_per_file: int | None = None,
        globs: Sequence[str] = (),
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.rg",
                command="rg",
                options={
                    "pattern": pattern,
                    "case": case,
                    "fixed_strings": fixed_strings,
                    "context_lines": context_lines,
                    "max_matches_per_file": max_matches_per_file,
                    "globs": _string_tuple(globs, "globs"),
                },
                paths=_path_operands(paths, kind="any"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class HeadTool(_ShellCommandTool):
    """Read the first lines of a workspace text file.

    Args:
        path: Workspace-relative file path to read.
        lines: Number of leading lines to return.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="head",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        lines: int = 80,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            _line_reader_request(
                command="head",
                path=path,
                lines=lines,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class TailTool(_ShellCommandTool):
    """Read the last lines of a workspace text file.

    Args:
        path: Workspace-relative file path to read.
        lines: Number of trailing lines to return.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="tail",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        lines: int = 80,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            _line_reader_request(
                command="tail",
                path=path,
                lines=lines,
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class LsTool(_ShellCommandTool):
    """List workspace directory entries or a single file path.

    Args:
        path: Optional workspace-relative file or directory path to list.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="ls",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        paths = () if path is None else (path,)
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.ls",
                command="ls",
                options={},
                paths=_path_operands(paths, kind="any"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class CatTool(_ShellCommandTool):
    """Read a workspace text file.

    Args:
        path: Workspace-relative text file path to read.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="cat",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.cat",
                command="cat",
                options={},
                paths=_path_operands((path,), kind="text_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class FileTool(_ShellCommandTool):
    """Identify workspace file types.

    Args:
        paths: Workspace-relative regular file paths to inspect.
        cwd: Workspace-relative working directory for the command.
        brief: Omit file names from command output.
        mime_type: Emit MIME type output.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="file",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        brief: bool = False,
        mime_type: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.file",
                command="file",
                options={"brief": brief, "mime_type": mime_type},
                paths=_path_operands(paths, kind="file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class FindTool(_ShellCommandTool):
    """Find workspace entries with constrained selectors.

    Args:
        paths: Workspace-relative file or directory roots to search.
        cwd: Workspace-relative working directory for the command.
        max_depth: Maximum traversal depth below each root.
        entry_type: Entry type to include.
        name: Optional exact basename to match.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="find",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        paths: Sequence[str] = (),
        cwd: str | None = None,
        max_depth: int = 3,
        entry_type: Literal["any", "file", "directory"] = "any",
        name: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.find",
                command="find",
                options={
                    "max_depth": max_depth,
                    "entry_type": entry_type,
                    "name": name,
                },
                paths=_path_operands(paths, kind="any"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class WcTool(_ShellCommandTool):
    """Count lines, words, or bytes in workspace text files.

    Args:
        paths: Workspace-relative text file paths to count.
        cwd: Workspace-relative working directory for the command.
        lines: Include line counts.
        words: Include word counts.
        count_bytes: Include byte counts.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="wc",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        cwd: str | None = None,
        lines: bool = True,
        words: bool = False,
        count_bytes: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.wc",
                command="wc",
                options={
                    "lines": lines,
                    "words": words,
                    "count_bytes": count_bytes,
                },
                paths=_path_operands(paths, kind="text_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class AwkTool(_ShellCommandTool):
    """Select fields and lines from workspace text files.

    Args:
        paths: Workspace-relative text file paths to read.
        fields: Optional one-based field indexes to print.
        field_separator: Input field separator mode.
        output_separator: Separator to use between printed fields.
        pattern: Optional literal pattern to match whole input lines.
        start_line: Optional first one-based line number to include.
        end_line: Optional last one-based line number to include.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="awk",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        fields: Sequence[int] | None = None,
        field_separator: Literal[
            "whitespace",
            "tab",
            "comma",
            "pipe",
        ] = "whitespace",
        output_separator: str = " ",
        pattern: str | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.awk",
                command="awk",
                options={
                    "fields": _optional_int_tuple(fields, "fields"),
                    "field_separator": field_separator,
                    "output_separator": output_separator,
                    "pattern": pattern,
                    "start_line": start_line,
                    "end_line": end_line,
                },
                paths=_path_operands(paths, kind="text_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class SedTool(_ShellCommandTool):
    """Select line ranges and patterns from workspace text files.

    Args:
        paths: Workspace-relative text file paths to read.
        line_ranges: One-based line ranges such as "1" or "1,10".
        patterns: Literal patterns to select from input lines.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="sed",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        paths: Sequence[str],
        line_ranges: Sequence[str] = (),
        patterns: Sequence[str] = (),
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.sed",
                command="sed",
                options={
                    "line_ranges": _string_tuple(
                        line_ranges,
                        "line_ranges",
                    ),
                    "patterns": _string_tuple(patterns, "patterns"),
                },
                paths=_path_operands(paths, kind="text_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class JqTool(_ShellCommandTool):
    """Transform workspace JSON files with a constrained jq filter.

    Args:
        filter: Constrained jq filter expression.
        paths: Workspace-relative JSON file paths to read.
        cwd: Workspace-relative working directory for the command.
        raw_output: Emit raw string output.
        compact: Emit compact JSON output.
        slurp: Read all inputs into an array.
        sort_keys: Sort JSON object keys in output.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="jq",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        filter: str,
        paths: Sequence[str],
        cwd: str | None = None,
        raw_output: bool = False,
        compact: bool = False,
        slurp: bool = False,
        sort_keys: bool = False,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.jq",
                command="jq",
                options={
                    "filter": filter,
                    "raw_output": raw_output,
                    "compact": compact,
                    "slurp": slurp,
                    "sort_keys": sort_keys,
                },
                paths=_path_operands(paths, kind="json_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PdfInfoTool(_ShellCommandTool):
    """Inspect metadata for a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to inspect.
        first_page: Optional first one-based page number for page details.
        last_page: Optional last one-based page number for page details.
        boxes: Include page bounding boxes.
        iso_dates: Emit dates in ISO-8601 format.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="pdfinfo",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        first_page: int | None = None,
        last_page: int | None = None,
        boxes: bool = False,
        iso_dates: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.pdfinfo",
                command="pdfinfo",
                options={
                    "first_page": first_page,
                    "last_page": last_page,
                    "boxes": boxes,
                    "iso_dates": iso_dates,
                },
                paths=_path_operands((path,), kind="pdf_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PdfToTextTool(_ShellCommandTool):
    """Extract text from a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to read.
        first_page: First one-based page number to extract.
        last_page: Optional last one-based page number to extract.
        layout: Preserve physical page layout where supported.
        no_page_breaks: Suppress page break markers in text output.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="pdftotext",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        layout: bool = False,
        no_page_breaks: bool = False,
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.pdftotext",
                command="pdftotext",
                options={
                    "first_page": first_page,
                    "last_page": last_page,
                    "layout": layout,
                    "no_page_breaks": no_page_breaks,
                },
                paths=_path_operands((path,), kind="pdf_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class PdfToPpmTool(_ShellCommandTool):
    """Rasterize pages from a workspace PDF file.

    Args:
        path: Workspace-relative PDF file path to rasterize.
        first_page: First one-based page number to rasterize.
        last_page: Optional last one-based page number to rasterize.
        dpi: Rasterization dots per inch.
        grayscale: Render grayscale output.
        format: Raster image format.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="pdftoppm",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        first_page: int = 1,
        last_page: int | None = None,
        dpi: int | None = None,
        grayscale: bool = False,
        format: Literal["png"] = "png",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.pdftoppm",
                command="pdftoppm",
                options={
                    "first_page": first_page,
                    "last_page": last_page,
                    "dpi": dpi,
                    "grayscale": grayscale,
                    "format": format,
                },
                paths=_path_operands((path,), kind="pdf_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


class TesseractTool(_ShellCommandTool):
    """Recognize text in a workspace image file.

    Args:
        path: Workspace-relative image file path to read.
        languages: OCR language identifiers to use.
        psm: Tesseract page segmentation mode.
        oem: Optional Tesseract OCR engine mode.
        dpi: Optional input image DPI hint.
        output_format: OCR output format.
        cwd: Workspace-relative working directory for the command.
        timeout_seconds: Optional execution timeout in seconds.
        max_stdout_bytes: Optional stdout byte cap.
        max_stderr_bytes: Optional stderr byte cap.

    Returns:
        Formatted shell execution result.
    """

    def __init__(
        self,
        *,
        settings: ShellToolSettings,
        policy: ExecutionPolicy,
        executor: CommandExecutor,
        formatter: ShellResultFormatter | None = None,
    ) -> None:
        super().__init__(
            command="tesseract",
            settings=settings,
            policy=policy,
            executor=executor,
            formatter=formatter,
        )

    async def __call__(
        self,
        path: str,
        languages: Sequence[str] | None = None,
        psm: int = 3,
        oem: int | None = None,
        dpi: int | None = None,
        output_format: Literal["txt"] = "txt",
        cwd: str | None = None,
        timeout_seconds: float | None = None,
        max_stdout_bytes: int | None = None,
        max_stderr_bytes: int | None = None,
        *,
        context: ToolCallContext,
    ) -> str:
        return await self._execute_request(
            ShellCommandRequest(
                tool_name="shell.tesseract",
                command="tesseract",
                options={
                    "languages": _optional_string_tuple(
                        languages,
                        "languages",
                    ),
                    "psm": psm,
                    "oem": oem,
                    "dpi": dpi,
                    "output_format": output_format,
                },
                paths=_path_operands((path,), kind="image_file"),
                cwd=cwd,
                timeout_seconds=timeout_seconds,
                max_stdout_bytes=max_stdout_bytes,
                max_stderr_bytes=max_stderr_bytes,
            ),
            context=context,
        )


def _line_reader_request(
    *,
    command: Literal["head", "tail"],
    path: str,
    lines: int,
    cwd: str | None,
    timeout_seconds: float | None,
    max_stdout_bytes: int | None,
    max_stderr_bytes: int | None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name=f"shell.{command}",
        command=command,
        options={"lines": lines},
        paths=_path_operands((path,), kind="text_file"),
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _path_operands(
    paths: Sequence[str],
    *,
    kind: Literal[
        "any",
        "file",
        "text_file",
        "json_file",
        "pdf_file",
        "image_file",
    ],
) -> tuple[PathOperand, ...]:
    normalized_paths = _string_tuple(paths, "paths")
    return tuple(
        PathOperand(
            name=f"path_{index}",
            path=path,
            kind=kind,
            access="read",
        )
        for index, path in enumerate(normalized_paths)
    )


def _string_tuple(value: Sequence[str], name: str) -> tuple[str, ...]:
    _assert_string_sequence(value, name)
    return tuple(value)


def _optional_int_tuple(
    value: Sequence[int] | None,
    name: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    _assert_int_sequence(value, name)
    return tuple(value)


def _optional_string_tuple(
    value: Sequence[str] | None,
    name: str,
) -> tuple[str, ...] | None:
    if value is None:
        return None
    return _string_tuple(value, name)


def _policy_denied_result(
    request: ShellCommandRequest,
    error: ShellPolicyDenied,
) -> ExecutionResult:
    return ExecutionResult(
        backend="local",
        tool_name=request.tool_name,
        command=request.command,
        argv=(request.command,),
        display_argv=(request.command,),
        cwd=".",
        display_cwd=".",
        status=ShellExecutionStatus.POLICY_DENIED,
        exit_code=None,
        stdout="",
        stderr="",
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        stdout_bytes=0,
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=0,
        error_code=error.error_code,
        error_message=str(error),
        metadata=request.metadata,
    )
