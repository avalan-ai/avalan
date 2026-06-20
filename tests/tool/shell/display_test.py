from copy import copy, deepcopy
from dataclasses import asdict
from json import dumps
from pathlib import Path
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallResult,
    ToolManagerSettings,
    ToolValue,
)
from avalan.tool.display import ToolDisplayProjection
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    SHELL_COMMAND_IDS,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
)
from avalan.tool.shell.entities import ShellFormattedResult

_CALL_ARGUMENTS: dict[str, dict[str, object]] = {
    "shell.rg": {
        "pattern": "visible",
        "paths": ["filesystem/visible.txt"],
    },
    "shell.head": {"path": "filesystem/visible.txt", "lines": 5},
    "shell.tail": {"path": "filesystem/visible.txt", "lines": 5},
    "shell.ls": {"path": "filesystem"},
    "shell.cat": {"path": "filesystem/visible.txt"},
    "shell.file": {"paths": ["filesystem/visible.txt"]},
    "shell.find": {"paths": ["filesystem"], "name": "visible.txt"},
    "shell.wc": {"paths": ["filesystem/visible.txt"], "words": True},
    "shell.awk": {
        "paths": ["filters/table.csv"],
        "fields": [1, 2],
        "field_separator": "comma",
    },
    "shell.sed": {
        "paths": ["filters/lines.txt"],
        "line_ranges": ["1,2"],
    },
    "shell.jq": {"filter": ".", "paths": ["json/valid.json"]},
    "shell.pdfinfo": {"path": "media/small.pdf"},
    "shell.pdftotext": {"path": "media/small.pdf", "last_page": 1},
    "shell.pdftoppm": {"path": "media/small.pdf", "last_page": 1},
    "shell.tesseract": {"path": "ocr/small.pgm", "languages": ["eng"]},
}
_EXPECTED_ACTIONS = {
    "shell.rg": "search",
    "shell.head": "read",
    "shell.tail": "read",
    "shell.ls": "list",
    "shell.cat": "read",
    "shell.file": "identify",
    "shell.find": "find",
    "shell.wc": "count",
    "shell.awk": "select",
    "shell.sed": "select",
    "shell.jq": "transform",
    "shell.pdfinfo": "inspect",
    "shell.pdftotext": "extract",
    "shell.pdftoppm": "rasterize",
    "shell.tesseract": "recognize",
}


class ShellDisplayProjectionCallTest(TestCase):
    def test_each_command_exposes_call_intent_projection(self) -> None:
        manager = _shell_manager(["shell"])

        for command_id in SHELL_COMMAND_IDS:
            name = f"shell.{command_id}"
            with self.subTest(name=name):
                call = ToolCall(
                    id=f"call-{command_id}",
                    name=name,
                    arguments=cast(
                        dict[str, ToolValue],
                        _CALL_ARGUMENTS[name],
                    ),
                )
                descriptor = manager.describe_tool_call(call)

                assert descriptor is not None
                self.assertIsNotNone(descriptor.display_projector)
                projection = descriptor.project_display(call)

                self.assertIsInstance(projection, ToolDisplayProjection)
                assert isinstance(projection, ToolDisplayProjection)
                self.assertEqual(projection.label, name)
                self.assertEqual(projection.action, _EXPECTED_ACTIONS[name])

    def test_rg_call_projection_describes_pattern_search(self) -> None:
        projection = _call_projection(
            "shell.rg",
            {
                "pattern": "needle",
                "paths": ["filesystem/visible.txt"],
                "case": "smart",
            },
        )

        self.assertEqual(projection.action, "search")
        self.assertEqual(projection.target, "needle")
        self.assertIn("Search", projection.summary or "")
        self.assertEqual(_detail_value(projection, "pattern"), "needle")
        self.assertEqual(_detail_value(projection, "case"), "smart")

    def test_cat_call_projection_describes_path_read(self) -> None:
        projection = _call_projection(
            "shell.cat",
            {"path": "filesystem/visible.txt"},
        )

        self.assertEqual(projection.action, "read")
        self.assertEqual(projection.target, "filesystem/visible.txt")
        self.assertIn("Read", projection.summary or "")
        self.assertEqual(
            _detail_value(projection, "paths"),
            "filesystem/visible.txt",
        )

    def test_sensitive_call_path_is_redacted(self) -> None:
        projection = _call_projection("shell.cat", {"path": "credentials"})
        payload = dumps(projection.to_payload(), sort_keys=True)

        self.assertTrue(projection.redacted)
        self.assertNotIn("credentials", payload)
        self.assertIn("[redacted]", payload)

    def test_unsafe_call_paths_are_redacted_before_policy(self) -> None:
        for path in (
            "/Users/mariano/private/report.txt",
            "~/private/report.txt",
            "../private/report.txt",
            "safe/../private/report.txt",
            "$HOME/private/report.txt",
            "C:\\private\\report.txt",
        ):
            with self.subTest(path=path):
                projection = _call_projection("shell.cat", {"path": path})
                payload = dumps(projection.to_payload(), sort_keys=True)

                self.assertTrue(projection.redacted)
                self.assertNotIn(path, payload)
                self.assertIn("[redacted]", payload)

    def test_unsafe_rg_globs_are_redacted_before_policy(self) -> None:
        for glob in (
            "/Users/mariano/private/**",
            "~/private/**",
            "../private/**",
            "$HOME/private/**",
            "!/Users/mariano/private/**",
        ):
            with self.subTest(glob=glob):
                projection = _call_projection(
                    "shell.rg",
                    {
                        "pattern": "needle",
                        "paths": ["filesystem"],
                        "globs": [glob],
                    },
                )
                payload = dumps(projection.to_payload(), sort_keys=True)

                self.assertNotIn(glob.lstrip("!"), payload)
                self.assertIn("[redacted]", payload)

    def test_unsafe_find_name_is_redacted_before_policy(self) -> None:
        for name in (
            "/Users/mariano/private",
            "$HOME/private",
            "../private",
            ".env",
        ):
            with self.subTest(name=name):
                projection = _call_projection(
                    "shell.find",
                    {"paths": ["filesystem"], "name": name},
                )
                payload = dumps(projection.to_payload(), sort_keys=True)

                self.assertTrue(projection.redacted)
                self.assertNotIn(name, payload)
                self.assertIn("[redacted]", payload)


class ShellDisplayProjectionTerminalTest(IsolatedAsyncioTestCase):
    def test_formatted_result_supports_copy_and_asdict(self) -> None:
        call = ToolCall(
            id="call-cat",
            name="shell.cat",
            arguments={"path": "filesystem/visible.txt"},
        )
        result = ExecutionResult(
            backend="local",
            tool_name="shell.cat",
            command="cat",
            argv=("cat", "filesystem/visible.txt"),
            display_argv=("cat", "filesystem/visible.txt"),
            cwd=".",
            display_cwd=".",
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="formatted",
            stderr="",
            stdout_media_type="text/plain",
            output_kind=ShellOutputKind.TEXT,
            error_code=ShellExecutionErrorCode.COMPLETED,
        )
        formatted = ShellFormattedResult("formatted", result)
        outcome = ToolCallResult(
            id="result-cat",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=formatted,
        )

        self.assertIs(copy(formatted), formatted)
        self.assertIs(deepcopy(formatted), formatted)
        self.assertEqual(asdict(outcome)["result"], "formatted")

    async def test_successful_execution_projection_uses_result_facts(
        self,
    ) -> None:
        executor = _StaticResultExecutor(
            stdout="RAW_STDOUT_SHOULD_NOT_APPEAR",
            stderr="RAW_STDERR_SHOULD_NOT_APPEAR",
            stdout_bytes=128,
            stderr_bytes=64,
            stdout_truncated=True,
            stderr_truncated=True,
            duration_ms=37,
        )
        manager = _shell_manager(["shell.cat"], executor=executor)
        outcome = await manager.execute_call(
            ToolCall(
                id="call-cat",
                name="shell.cat",
                arguments={"path": "filesystem/visible.txt"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        projection = _terminal_projection(manager, outcome)
        payload = dumps(projection.to_payload(), sort_keys=True)

        self.assertEqual(projection.status, "completed")
        self.assertEqual(_detail_value(projection, "exit code"), 0)
        self.assertEqual(_detail_value(projection, "duration ms"), 37)
        self.assertEqual(_detail_value(projection, "stdout bytes"), 128)
        self.assertEqual(_detail_value(projection, "stderr bytes"), 64)
        self.assertEqual(_detail_value(projection, "stdout truncated"), True)
        self.assertEqual(_detail_value(projection, "stderr truncated"), True)
        self.assertNotIn("RAW_STDOUT_SHOULD_NOT_APPEAR", payload)
        self.assertNotIn("RAW_STDERR_SHOULD_NOT_APPEAR", payload)

    async def test_rg_no_match_projection_is_clean(self) -> None:
        executor = _StaticResultExecutor(
            status=ShellExecutionStatus.NO_MATCHES,
            exit_code=1,
            error_code=ShellExecutionErrorCode.NO_MATCHES,
        )
        manager = _shell_manager(["shell.rg"], executor=executor)
        outcome = await manager.execute_call(
            ToolCall(
                id="call-rg",
                name="shell.rg",
                arguments={
                    "pattern": "missing",
                    "paths": ["filesystem/visible.txt"],
                },
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        projection = _terminal_projection(manager, outcome)

        self.assertEqual(projection.status, "no_matches")
        self.assertEqual(projection.severity, "info")
        self.assertIn("no matches", projection.summary or "")
        self.assertEqual(_detail_value(projection, "error code"), "no_matches")

    async def test_policy_denied_projection_does_not_expose_raw_path(
        self,
    ) -> None:
        manager = _shell_manager(
            ["shell.cat"],
            executor=_StaticResultExecutor(),
        )
        outcome = await manager.execute_call(
            ToolCall(
                id="call-denied",
                name="shell.cat",
                arguments={"path": "credentials"},
            ),
            context=ToolCallContext(),
        )

        self.assertIsInstance(outcome, ToolCallResult)
        assert isinstance(outcome, ToolCallResult)
        projection = _terminal_projection(manager, outcome)
        payload = dumps(projection.to_payload(), sort_keys=True)

        self.assertEqual(projection.status, "policy_denied")
        self.assertNotIn("credentials", payload)
        self.assertEqual(
            _detail_value(projection, "error code"),
            "sensitive_path",
        )

    def test_generated_output_projection_uses_display_paths(self) -> None:
        manager = _shell_manager(["shell.pdftoppm"])
        call = ToolCall(
            id="call-pdf",
            name="shell.pdftoppm",
            arguments={"path": "media/small.pdf", "last_page": 1},
        )
        result = ExecutionResult(
            backend="local",
            tool_name="shell.pdftoppm",
            command="pdftoppm",
            argv=("pdftoppm", "media/small.pdf", "GENERATED_PREFIX"),
            display_argv=(
                "pdftoppm",
                "media/small.pdf",
                "/private/tmp/avalan-shell-raw/page-1.png",
            ),
            cwd=".",
            display_cwd=".",
            status=ShellExecutionStatus.COMPLETED,
            exit_code=0,
            stdout="",
            stderr="",
            stdout_media_type="application/json",
            output_kind=ShellOutputKind.GENERATED_FILES,
            generated_files=(
                GeneratedFile(
                    display_path="GENERATED_PREFIX-1.png",
                    media_type="image/png",
                    suffix=".png",
                    bytes=42,
                    page=1,
                    width=10,
                    height=10,
                ),
            ),
            duration_ms=1,
            error_code=ShellExecutionErrorCode.COMPLETED,
        )
        outcome = ToolCallResult(
            id="result-pdf",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=ShellFormattedResult("formatted", result),
        )

        projection = _terminal_projection(manager, outcome)
        payload = dumps(projection.to_payload(), sort_keys=True)

        self.assertIn("GENERATED_PREFIX-1.png", payload)
        self.assertIn("image/png", payload)
        self.assertNotIn("/private/tmp", payload)
        self.assertNotIn("avalan-shell-raw", payload)
        self.assertEqual(
            _detail_value(projection, "generated output"),
            "GENERATED_PREFIX-1.png",
        )
        self.assertEqual(
            _detail_value(projection, "generated media type"),
            "image/png",
        )

    def test_terminal_projection_redacts_unsafe_display_argv(self) -> None:
        manager = _shell_manager(["shell.rg"])
        call = ToolCall(
            id="call-rg",
            name="shell.rg",
            arguments={"pattern": "needle", "paths": ["filesystem"]},
        )
        for argument in (
            "$HOME/private/**",
            "C:\\private\\**",
            "%USERPROFILE%\\private\\**",
        ):
            with self.subTest(argument=argument):
                result = ExecutionResult(
                    backend="local",
                    tool_name="shell.rg",
                    command="rg",
                    argv=("rg", "--glob", argument, "needle", "filesystem"),
                    display_argv=(
                        "rg",
                        "--glob",
                        argument,
                        "needle",
                        "filesystem",
                    ),
                    cwd=".",
                    display_cwd=".",
                    status=ShellExecutionStatus.COMPLETED,
                    exit_code=0,
                    stdout="",
                    stderr="",
                    stdout_media_type="text/plain",
                    output_kind=ShellOutputKind.TEXT,
                    error_code=ShellExecutionErrorCode.COMPLETED,
                )
                outcome = ToolCallResult(
                    id="result-rg",
                    call=call,
                    name=call.name,
                    arguments=call.arguments,
                    result=ShellFormattedResult("formatted", result),
                )

                projection = _terminal_projection(manager, outcome)
                payload = dumps(projection.to_payload(), sort_keys=True)

                self.assertTrue(projection.redacted)
                self.assertNotIn(argument, payload)
                self.assertIn("[redacted]", payload)

    def test_terminal_projection_keeps_non_path_display_argv(self) -> None:
        manager = _shell_manager(["shell.rg"])
        call = ToolCall(
            id="call-rg",
            name="shell.rg",
            arguments={"pattern": "needle", "paths": ["filesystem"]},
        )
        for argument in (
            "foo$",
            "\\d+",
            "$name",
            "$0",
            "s/$/x/",
        ):
            with self.subTest(argument=argument):
                result = ExecutionResult(
                    backend="local",
                    tool_name="shell.rg",
                    command="rg",
                    argv=("rg", argument, "filesystem"),
                    display_argv=("rg", argument, "filesystem"),
                    cwd=".",
                    display_cwd=".",
                    status=ShellExecutionStatus.COMPLETED,
                    exit_code=0,
                    stdout="",
                    stderr="",
                    stdout_media_type="text/plain",
                    output_kind=ShellOutputKind.TEXT,
                    error_code=ShellExecutionErrorCode.COMPLETED,
                )
                outcome = ToolCallResult(
                    id="result-rg",
                    call=call,
                    name=call.name,
                    arguments=call.arguments,
                    result=ShellFormattedResult("formatted", result),
                )

                projection = _terminal_projection(manager, outcome)
                payload = dumps(projection.to_payload(), sort_keys=True)

                self.assertFalse(projection.redacted)
                self.assertNotIn("[redacted]", payload)


class _StaticResultExecutor:
    def __init__(
        self,
        *,
        status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
        exit_code: int | None = 0,
        error_code: ShellExecutionErrorCode = (
            ShellExecutionErrorCode.COMPLETED
        ),
        stdout: str = "",
        stderr: str = "",
        stdout_bytes: int | None = None,
        stderr_bytes: int | None = None,
        stdout_truncated: bool = False,
        stderr_truncated: bool = False,
        duration_ms: int = 1,
    ) -> None:
        self.status = status
        self.exit_code = exit_code
        self.error_code = error_code
        self.stdout = stdout
        self.stderr = stderr
        self.stdout_bytes = stdout_bytes
        self.stderr_bytes = stderr_bytes
        self.stdout_truncated = stdout_truncated
        self.stderr_truncated = stderr_truncated
        self.duration_ms = duration_ms

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: object | None = None,
    ) -> ExecutionResult:
        return _execution_result(
            spec,
            status=self.status,
            exit_code=self.exit_code,
            error_code=self.error_code,
            stdout=self.stdout,
            stderr=self.stderr,
            stdout_bytes=self.stdout_bytes,
            stderr_bytes=self.stderr_bytes,
            stdout_truncated=self.stdout_truncated,
            stderr_truncated=self.stderr_truncated,
            duration_ms=self.duration_ms,
        )


def _shell_manager(
    enabled_tools: list[str],
    *,
    executor: _StaticResultExecutor | None = None,
) -> ToolManager:
    fixture_root = Path(__file__).parent / "fixtures"
    settings = ShellToolSettings(
        workspace_root=str(fixture_root),
        allow_media_tools=True,
    )
    resolver = TrustedExecutableResolver(
        executable_paths={
            command_id: "/bin/echo" for command_id in SHELL_COMMAND_IDS
        }
    )
    return ToolManager.create_instance(
        available_toolsets=[
            ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings, resolver=resolver),
                executor=executor,
            )
        ],
        enable_tools=enabled_tools,
        settings=ToolManagerSettings(),
    )


def _call_projection(
    name: str,
    arguments: dict[str, object],
) -> ToolDisplayProjection:
    manager = _shell_manager([name])
    call = ToolCall(
        id="call",
        name=name,
        arguments=cast(dict[str, ToolValue], arguments),
    )
    descriptor = manager.describe_tool_call(call)
    assert descriptor is not None
    projection = descriptor.project_display(call)
    assert isinstance(projection, ToolDisplayProjection)
    return projection


def _terminal_projection(
    manager: ToolManager,
    outcome: ToolCallResult,
) -> ToolDisplayProjection:
    descriptor = manager.describe_tool(outcome.call.name)
    assert descriptor is not None
    projection = descriptor.project_display(outcome.call, outcome)
    assert isinstance(projection, ToolDisplayProjection)
    return projection


def _execution_result(
    spec: ExecutionSpec,
    *,
    status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
    exit_code: int | None = 0,
    error_code: ShellExecutionErrorCode = ShellExecutionErrorCode.COMPLETED,
    stdout: str = "",
    stderr: str = "",
    stdout_bytes: int | None = None,
    stderr_bytes: int | None = None,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    duration_ms: int = 1,
    generated_files: tuple[GeneratedFile, ...] = (),
    output_kind: ShellOutputKind | None = None,
    stdout_media_type: str | None = None,
) -> ExecutionResult:
    return ExecutionResult(
        backend=spec.backend,
        tool_name=spec.tool_name,
        command=spec.command,
        argv=spec.argv,
        display_argv=spec.display_argv,
        cwd=spec.cwd,
        display_cwd=spec.display_cwd,
        status=status,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        stdout_media_type=stdout_media_type or spec.stdout_media_type,
        output_kind=output_kind or spec.output_kind,
        generated_files=generated_files,
        stdout_bytes=(
            len(stdout.encode()) if stdout_bytes is None else stdout_bytes
        ),
        stderr_bytes=(
            len(stderr.encode()) if stderr_bytes is None else stderr_bytes
        ),
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
        duration_ms=duration_ms,
        error_code=error_code,
        metadata=spec.metadata,
    )


def _detail_value(
    projection: ToolDisplayProjection,
    label: str,
) -> object:
    for detail in projection.details:
        if detail.label == label:
            return detail.value
    raise AssertionError(f"missing detail {label}")


if __name__ == "__main__":
    main()
