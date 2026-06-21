from copy import copy, deepcopy
from dataclasses import asdict
from json import dumps
from pathlib import Path
from pickle import dumps as pickle_dumps
from pickle import loads as pickle_loads
from typing import cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import (
    ToolCall,
    ToolCallContext,
    ToolCallError,
    ToolCallResult,
    ToolManagerSettings,
    ToolValue,
)
from avalan.tool.display import ToolDisplayProjection
from avalan.tool.manager import ToolManager
from avalan.tool.shell import (
    SHELL_COMMAND_DEFINITIONS,
    SHELL_COMMAND_IDS,
    ExecutionPolicy,
    ExecutionResult,
    ExecutionSpec,
    GeneratedFile,
    PathOperand,
    ShellCommandDefinition,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
)
from avalan.tool.shell.display import (
    project_shell_command_request,
    project_shell_execution_result,
)
from avalan.tool.shell.entities import ShellFormattedResult, ShellPathKind

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
    def test_invalid_call_arguments_do_not_project(self) -> None:
        manager = _shell_manager(["shell.cat"])
        call = ToolCall(
            id="call-cat",
            name="shell.cat",
            arguments=cast(dict[str, ToolValue], ["not", "a", "dict"]),
        )
        descriptor = manager.describe_tool_call(call)

        assert descriptor is not None
        self.assertIsNone(descriptor.project_display(call))

    def test_missing_required_call_arguments_do_not_project(self) -> None:
        manager = _shell_manager(["shell.rg"])
        call = ToolCall(id="call-rg", name="shell.rg", arguments={})
        descriptor = manager.describe_tool_call(call)

        assert descriptor is not None
        self.assertIsNone(descriptor.project_display(call))

    def test_none_call_arguments_use_tool_defaults(self) -> None:
        manager = _shell_manager(["shell.ls"])
        call = ToolCall(id="call-ls", name="shell.ls", arguments=None)
        descriptor = manager.describe_tool_call(call)

        assert descriptor is not None
        projection = descriptor.project_display(call)

        self.assertIsInstance(projection, ToolDisplayProjection)
        assert isinstance(projection, ToolDisplayProjection)
        self.assertEqual(projection.target, ".")
        self.assertEqual(projection.scope, "current directory")

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

    def test_default_find_call_projects_workspace_scope(self) -> None:
        projection = _call_projection("shell.find", {})

        self.assertEqual(projection.target, "workspace")
        self.assertEqual(projection.scope, "workspace")
        self.assertEqual(_detail_value(projection, "max depth"), 3)

    def test_default_ls_call_projects_current_directory_scope(self) -> None:
        projection = _call_projection("shell.ls", {})

        self.assertEqual(projection.target, ".")
        self.assertEqual(projection.scope, "current directory")

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

    def test_request_projection_includes_cwd_limits_and_metrics(self) -> None:
        projection = _call_projection(
            "shell.rg",
            {
                "pattern": "needle",
                "paths": ["filesystem"],
                "cwd": "filesystem",
                "globs": ["*.txt", "!*.bak"],
                "max_depth": 4,
                "timeout_seconds": 2.5,
                "max_stdout_bytes": 128,
                "max_stderr_bytes": 64,
            },
        )

        self.assertEqual(_detail_value(projection, "cwd"), "filesystem")
        self.assertEqual(_detail_value(projection, "globs"), "*.txt, !*.bak")
        self.assertEqual(
            _detail_value(projection, "timeout seconds"),
            2.5,
        )
        self.assertEqual(
            _detail_value(projection, "max stdout bytes"),
            128,
        )
        self.assertEqual(
            _detail_value(projection, "max stderr bytes"),
            64,
        )
        self.assertEqual(projection.metrics["timeout_seconds"], 2.5)
        self.assertEqual(projection.metrics["max_stdout_bytes"], 128)
        self.assertEqual(projection.metrics["max_stderr_bytes"], 64)
        self.assertEqual(projection.metrics["max_depth"], 4)

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

    def test_path_target_projection_truncates_long_path_lists(self) -> None:
        projection = _call_projection(
            "shell.file",
            {
                "paths": [
                    "one.txt",
                    "two.txt",
                    "three.txt",
                    "four.txt",
                ]
            },
        )

        self.assertEqual(projection.target, "one.txt, two.txt, three.txt, ...")
        self.assertEqual(projection.scope, "one.txt, two.txt, three.txt, ...")

    def test_find_name_without_value_does_not_add_name_detail(self) -> None:
        projection = _call_projection(
            "shell.find",
            {"paths": ["filesystem"], "name": None},
        )

        self.assertFalse(_has_detail(projection, "name"))

    def test_alternate_request_targets_are_projected(self) -> None:
        cases = (
            (
                "shell.sed",
                {
                    "paths": ["filters/lines.txt"],
                    "patterns": ["alpha", "beta"],
                },
                "alpha, beta",
            ),
            (
                "shell.sed",
                {"paths": ["filters/lines.txt"]},
                "filters/lines.txt",
            ),
            (
                "shell.awk",
                {
                    "paths": ["filters/table.csv"],
                    "pattern": "active",
                },
                "active",
            ),
            (
                "shell.awk",
                {
                    "paths": ["filters/table.csv"],
                    "fields": [2, 3],
                },
                "2, 3",
            ),
        )

        for name, arguments, target in cases:
            with self.subTest(name=name, target=target):
                projection = _call_projection(name, arguments)

                self.assertEqual(projection.target, target)

    def test_range_and_enabled_option_details_are_projected(self) -> None:
        awk_projection = _call_projection(
            "shell.awk",
            {"paths": ["filters/table.csv"], "start_line": 5},
        )
        sed_projection = _call_projection(
            "shell.sed",
            {"paths": ["filters/lines.txt"], "end_line": 8},
        )
        pdf_projection = _call_projection(
            "shell.pdfinfo",
            {"path": "media/small.pdf", "first_page": 2},
        )
        jq_projection = _call_projection(
            "shell.jq",
            {
                "filter": ".items[]",
                "paths": ["json/valid.json"],
                "raw_output": True,
                "compact": True,
            },
        )

        self.assertEqual(_detail_value(awk_projection, "start line"), 5)
        self.assertEqual(_detail_value(sed_projection, "line range"), "1-8")
        self.assertEqual(_detail_value(pdf_projection, "first page"), 2)
        self.assertEqual(
            _detail_value(jq_projection, "enabled options"),
            "raw output, compact",
        )

    def test_direct_request_projection_handles_edge_values(self) -> None:
        object_projection = project_shell_command_request(
            _request(
                "awk",
                options={"pattern": _DisplayValue()},
                paths=("filters/table.csv",),
                kind="text_file",
            )
        )
        head_projection = project_shell_command_request(
            _request(
                "head",
                options={},
                paths=("filesystem/visible.txt",),
                kind="text_file",
            )
        )
        glob_projection = project_shell_command_request(
            _request(
                "rg",
                options={"pattern": "needle", "globs": "*.py"},
                paths=("filesystem",),
            )
        )

        self.assertEqual(
            _detail_value(object_projection, "pattern"),
            "custom-display",
        )
        self.assertFalse(_has_detail(head_projection, "lines"))
        self.assertEqual(_detail_value(glob_projection, "globs"), "*.py")

    def test_request_projection_skips_unknown_output_contracts(self) -> None:
        projection = project_shell_command_request(
            _request("custom", tool_name="shell.custom")
        )

        self.assertEqual(projection.action, "run")
        self.assertEqual(projection.target, "custom")
        self.assertEqual(projection.summary, "Run a command.")
        self.assertFalse(_has_detail(projection, "output kind"))

    def test_request_projection_ignores_broken_output_contracts(self) -> None:
        previous = SHELL_COMMAND_DEFINITIONS.get("broken")
        SHELL_COMMAND_DEFINITIONS["broken"] = cast(
            ShellCommandDefinition,
            _BrokenCommandDefinition(),
        )
        try:
            projection = project_shell_command_request(
                _request("broken", tool_name="shell.broken")
            )
        finally:
            if previous is None:
                del SHELL_COMMAND_DEFINITIONS["broken"]
            else:
                SHELL_COMMAND_DEFINITIONS["broken"] = previous

        self.assertEqual(projection.target, "broken")
        self.assertFalse(_has_detail(projection, "output kind"))

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

    def test_formatted_result_supports_pickle_round_trip(self) -> None:
        result = _direct_execution_result()
        formatted = ShellFormattedResult("formatted", result)

        restored = pickle_loads(pickle_dumps(formatted))

        self.assertIsInstance(restored, ShellFormattedResult)
        self.assertEqual(restored, "formatted")
        self.assertEqual(restored.execution_result, result)

    def test_error_outcome_without_execution_result_does_not_project(
        self,
    ) -> None:
        manager = _shell_manager(["shell.cat"])
        call = ToolCall(
            id="call-cat",
            name="shell.cat",
            arguments={"path": "filesystem/visible.txt"},
        )
        error = ToolCallError(
            id="error-cat",
            call=call,
            name=call.name,
            arguments=call.arguments,
            error={"type": "ToolCallError"},
            message="failed",
        )
        descriptor = manager.describe_tool(call.name)

        assert descriptor is not None
        self.assertIsNone(descriptor.project_display(call, error))

    def test_result_carrier_outcome_projects_execution_result(self) -> None:
        manager = _shell_manager(["shell.cat"])
        call = ToolCall(
            id="call-cat",
            name="shell.cat",
            arguments={"path": "filesystem/visible.txt"},
        )
        result = _direct_execution_result(command="cat", tool_name="shell.cat")
        outcome = ToolCallResult(
            id="result-cat",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=cast(ToolValue, _ExecutionResultCarrier(result)),
        )

        projection = _terminal_projection(manager, outcome)

        self.assertEqual(projection.summary, "cat completed.")
        self.assertEqual(projection.status, "completed")

    def test_invalid_result_carrier_outcome_does_not_project(self) -> None:
        manager = _shell_manager(["shell.cat"])
        call = ToolCall(
            id="call-cat",
            name="shell.cat",
            arguments={"path": "filesystem/visible.txt"},
        )
        outcome = ToolCallResult(
            id="result-cat",
            call=call,
            name=call.name,
            arguments=call.arguments,
            result=cast(ToolValue, _InvalidExecutionResultCarrier()),
        )
        descriptor = manager.describe_tool(call.name)

        assert descriptor is not None
        self.assertIsNone(descriptor.project_display(call, outcome))

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

    def test_generated_output_projection_redacts_unsafe_display_path(
        self,
    ) -> None:
        projection = project_shell_execution_result(
            _direct_execution_result(
                command="pdftoppm",
                tool_name="shell.pdftoppm",
                display_argv=("pdftoppm", "media/small.pdf", "page"),
                output_kind=ShellOutputKind.GENERATED_FILES,
                stdout_media_type="application/json",
                generated_files=(
                    GeneratedFile(
                        display_path="/tmp/avalan-shell-output/page-1.png",
                        media_type="image/png",
                        suffix=".png",
                        bytes=42,
                        truncated=True,
                    ),
                ),
            )
        )
        payload = dumps(projection.to_payload(), sort_keys=True)

        self.assertTrue(projection.redacted)
        self.assertEqual(
            _detail_value(projection, "generated output"),
            "[redacted]",
        )
        assert projection.preview is not None
        self.assertTrue(projection.preview.redacted)
        self.assertTrue(projection.preview.truncated)
        self.assertNotIn("/tmp/avalan-shell-output", payload)

    def test_terminal_projection_falls_back_to_command_without_display_argv(
        self,
    ) -> None:
        projection = project_shell_execution_result(
            _direct_execution_result(display_argv=())
        )

        self.assertEqual(projection.target, "command")

    def test_terminal_status_summaries_cover_error_cases(self) -> None:
        cases = (
            (
                ShellExecutionStatus.NONZERO_EXIT,
                2,
                ShellExecutionErrorCode.NONZERO_EXIT,
                "command exited with status 2.",
            ),
            (
                ShellExecutionStatus.NONZERO_EXIT,
                None,
                ShellExecutionErrorCode.NONZERO_EXIT,
                "command exited with a non-zero status.",
            ),
            (
                ShellExecutionStatus.TIMEOUT,
                None,
                ShellExecutionErrorCode.TIMEOUT,
                "command timed out.",
            ),
            (
                ShellExecutionStatus.COMMAND_UNAVAILABLE,
                None,
                ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
                "command is unavailable.",
            ),
            (
                ShellExecutionStatus.CANCELLED,
                None,
                ShellExecutionErrorCode.CANCELLED,
                "command ended with cancelled.",
            ),
        )

        for status, exit_code, error_code, summary in cases:
            with self.subTest(status=status, exit_code=exit_code):
                projection = project_shell_execution_result(
                    _direct_execution_result(
                        status=status,
                        exit_code=exit_code,
                        error_code=error_code,
                    )
                )

                self.assertEqual(projection.summary, summary)
                self.assertEqual(projection.severity, "error")

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
            "!/Users/mariano/private/**",
            "--glob=/Users/mariano/private/**",
            "--glob=workspace,../private/**",
            "..",
            "safe/..",
            "safe/../private",
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
            "''",
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


def _request(
    command: str,
    *,
    tool_name: str | None = None,
    options: dict[str, object] | None = None,
    paths: tuple[str, ...] = (),
    kind: ShellPathKind = "any",
    cwd: str | None = None,
    timeout_seconds: float | None = None,
    max_stdout_bytes: int | None = None,
    max_stderr_bytes: int | None = None,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name=tool_name or f"shell.{command}",
        command=command,
        options=options or {},
        paths=tuple(
            PathOperand(
                name=f"path_{index}",
                path=path,
                kind=kind,
                access="read",
            )
            for index, path in enumerate(paths)
        ),
        cwd=cwd,
        timeout_seconds=timeout_seconds,
        max_stdout_bytes=max_stdout_bytes,
        max_stderr_bytes=max_stderr_bytes,
    )


def _terminal_projection(
    manager: ToolManager,
    outcome: ToolCallResult,
) -> ToolDisplayProjection:
    descriptor = manager.describe_tool(outcome.call.name)
    assert descriptor is not None
    projection = descriptor.project_display(outcome.call, outcome)
    assert isinstance(projection, ToolDisplayProjection)
    return projection


def _direct_execution_result(
    *,
    command: str = "command",
    tool_name: str = "shell.command",
    display_argv: tuple[str, ...] = ("command",),
    display_cwd: str = ".",
    status: ShellExecutionStatus = ShellExecutionStatus.COMPLETED,
    exit_code: int | None = 0,
    error_code: ShellExecutionErrorCode | None = (
        ShellExecutionErrorCode.COMPLETED
    ),
    generated_files: tuple[GeneratedFile, ...] = (),
    output_kind: ShellOutputKind = ShellOutputKind.TEXT,
    stdout_media_type: str = "text/plain",
) -> ExecutionResult:
    return ExecutionResult(
        backend="local",
        tool_name=tool_name,
        command=command,
        argv=(command,),
        display_argv=display_argv,
        cwd=".",
        display_cwd=display_cwd,
        status=status,
        exit_code=exit_code,
        stdout="",
        stderr="",
        stdout_media_type=stdout_media_type,
        output_kind=output_kind,
        generated_files=generated_files,
        stdout_bytes=0,
        stderr_bytes=0,
        duration_ms=1,
        error_code=error_code,
    )


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


def _has_detail(
    projection: ToolDisplayProjection,
    label: str,
) -> bool:
    return any(detail.label == label for detail in projection.details)


class _ExecutionResultCarrier:
    def __init__(self, execution_result: ExecutionResult) -> None:
        self.execution_result = execution_result


class _InvalidExecutionResultCarrier:
    execution_result = "not an execution result"


class _DisplayValue:
    def __str__(self) -> str:
        return "custom-display"


class _BrokenCommandDefinition:
    def output_contract(
        self,
        request: ShellCommandRequest,
    ) -> tuple[str, ShellOutputKind]:
        raise RuntimeError("broken output contract")


if __name__ == "__main__":
    main()
