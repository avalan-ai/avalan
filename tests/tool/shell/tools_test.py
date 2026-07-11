from importlib import import_module
from typing import Any, cast, get_type_hints
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.entities import ToolCallContext, ToolExecutionStreamEvent
from avalan.tool import ToolSet
from avalan.tool.shell.entities import (
    ExecutionResult,
    ExecutionSpec,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellOutputKind,
    ShellPolicyDenied,
)
from avalan.tool.shell.policy import ExecutionPolicy
from avalan.tool.shell.settings import ShellToolSettings
from avalan.tool.shell.tools import (
    AwkTool,
    CatTool,
    FileTool,
    FindTool,
    HeadTool,
    JqTool,
    LsTool,
    NlTool,
    PdfInfoTool,
    PdfPlumberTool,
    PdfToPpmTool,
    PdfToTextTool,
    PyPdfTool,
    ReportLabTool,
    RgTool,
    SedTool,
    TailTool,
    TesseractTool,
    WcTool,
    _ShellCommandTool,
)


class ShellToolPackageCompatibilityTest(TestCase):
    def test_facade_preserves_existing_imports(self) -> None:
        facade = import_module("avalan.tool.shell.tools")
        cases = (
            ("RgTool", "avalan.tool.shell.tools.rg"),
            ("PipelineTool", "avalan.tool.shell.tools.pipeline"),
            ("GitStatusTool", "avalan.tool.shell.tools.git_read"),
            ("_ShellCommandTool", "avalan.tool.shell.tools._base"),
            ("_ShellGitCommandTool", "avalan.tool.shell.tools.git_base"),
            ("_git_policy_denied_result", "avalan.tool.shell.tools.git_base"),
        )

        for name, module_name in cases:
            with self.subTest(name=name):
                self.assertIs(
                    getattr(facade, name),
                    getattr(import_module(module_name), name),
                )

    def test_leaf_tool_annotations_resolve_at_runtime(self) -> None:
        facade = import_module("avalan.tool.shell.tools")
        tool_classes = (
            value
            for name, value in vars(facade).items()
            if name.endswith("Tool") and isinstance(value, type)
        )

        for tool_class in tool_classes:
            with self.subTest(tool_class=tool_class.__name__):
                get_type_hints(tool_class.__init__)
                get_type_hints(tool_class.__call__)


class ShellToolWrapperTest(IsolatedAsyncioTestCase):
    async def test_rg_builds_request_and_formats_executor_result(self) -> None:
        spec = _spec("rg")
        result = _result("rg", stdout="match\n")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(result)
        formatter = _RecordingFormatter()
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "needle",
            paths=("src", "tests"),
            cwd="src",
            case="smart",
            fixed_strings=True,
            context_lines=2,
            before_context=4,
            after_context=5,
            max_matches_per_file=3,
            max_depth=6,
            max_filesize_bytes=4096,
            globs=("*.py", "!*.pyc"),
            timeout_seconds=1.5,
            max_stdout_bytes=100,
            max_stderr_bytes=50,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.rg:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(formatter.results, [result])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.rg")
        self.assertEqual(request.command, "rg")
        self.assertEqual(
            request.options,
            {
                "pattern": "needle",
                "case": "smart",
                "fixed_strings": True,
                "context_lines": 2,
                "before_context": 4,
                "after_context": 5,
                "max_matches_per_file": 3,
                "max_depth": 6,
                "max_filesize_bytes": 4096,
                "globs": ("*.py", "!*.pyc"),
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="src",
                    kind="any",
                    access="read",
                ),
                PathOperand(
                    name="path_1",
                    path="tests",
                    kind="any",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "src")
        self.assertEqual(request.timeout_seconds, 1.5)
        self.assertEqual(request.max_stdout_bytes, 100)
        self.assertEqual(request.max_stderr_bytes, 50)

    async def test_rg_files_mode_builds_request_without_pattern(
        self,
    ) -> None:
        spec = _spec("rg")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("rg", stdout="src/app.py\n"))
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        output = await cast(Any, tool)(
            paths=("src", "tests"),
            mode="files",
            max_depth=3,
            max_filesize_bytes=4096,
            globs=("*.py",),
            cwd=".",
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.rg:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.rg")
        self.assertEqual(request.command, "rg")
        self.assertEqual(request.options["mode"], "files")
        self.assertIsNone(request.options.get("pattern"))
        self.assertEqual(request.options["max_depth"], 3)
        self.assertEqual(request.options["max_filesize_bytes"], 4096)
        self.assertEqual(request.options["globs"], ("*.py",))
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="src",
                    kind="any",
                    access="read",
                ),
                PathOperand(
                    name="path_1",
                    path="tests",
                    kind="any",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, ".")

    async def test_rg_files_mode_forwards_search_options_for_policy_denial(
        self,
    ) -> None:
        spec = _spec("rg")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("rg", stdout=""))
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        await cast(Any, tool)(
            pattern="needle",
            case="smart",
            fixed_strings=True,
            context_lines=1,
            before_context=2,
            after_context=3,
            max_matches_per_file=4,
            mode="files",
            context=ToolCallContext(),
        )

        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(
            policy.requests[0].options,
            {
                "max_depth": None,
                "max_filesize_bytes": None,
                "globs": (),
                "mode": "files",
                "pattern": "needle",
                "case": "smart",
                "fixed_strings": True,
                "context_lines": 1,
                "before_context": 2,
                "after_context": 3,
                "max_matches_per_file": 4,
            },
        )

    async def test_shell_tool_forwards_stream_callback_to_executor(
        self,
    ) -> None:
        spec = _spec("rg")
        result = _result("rg", stdout="match\n")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(result)
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )
        events: list[ToolExecutionStreamEvent] = []

        async def record(event: ToolExecutionStreamEvent) -> None:
            events.append(event)

        output = await tool(
            "needle",
            context=ToolCallContext(stream_event=record),
        )

        self.assertEqual(output, "formatted:shell.rg:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(executor.streams, [record])

    async def test_head_and_tail_build_line_reader_requests(self) -> None:
        for tool_class, command in (
            (HeadTool, "head"),
            (TailTool, "tail"),
        ):
            with self.subTest(command=command):
                spec = _spec(command)
                result = _result(command, stdout="line\n")
                policy = _FakePolicy(spec)
                executor = _FakeExecutor(result)
                formatter = _RecordingFormatter()
                tool = tool_class(
                    settings=ShellToolSettings(),
                    policy=policy,  # type: ignore[arg-type]
                    executor=executor,
                    formatter=formatter,
                )
                kwargs = (
                    {"byte_count": 128}
                    if command == "head"
                    else {
                        "start_line": 5,
                        "byte_count": 256,
                        "start_byte": 32,
                    }
                )
                expected_options = (
                    {"lines": 25, "byte_count": 128}
                    if command == "head"
                    else {
                        "lines": 25,
                        "start_line": 5,
                        "byte_count": 256,
                        "start_byte": 32,
                    }
                )

                output = await tool(
                    "logs/app.txt",
                    lines=25,
                    **kwargs,
                    cwd="logs",
                    timeout_seconds=2.0,
                    max_stdout_bytes=200,
                    max_stderr_bytes=80,
                    context=ToolCallContext(),
                )

                self.assertEqual(
                    output, f"formatted:shell.{command}:completed"
                )
                self.assertEqual(executor.specs, [spec])
                self.assertEqual(len(policy.requests), 1)
                request = policy.requests[0]
                self.assertEqual(request.tool_name, f"shell.{command}")
                self.assertEqual(request.command, command)
                self.assertEqual(request.options, expected_options)
                self.assertEqual(
                    request.paths,
                    (
                        PathOperand(
                            name="path_0",
                            path="logs/app.txt",
                            kind="text_file",
                            access="read",
                        ),
                    ),
                )
                self.assertEqual(request.cwd, "logs")
                self.assertEqual(request.timeout_seconds, 2.0)
                self.assertEqual(request.max_stdout_bytes, 200)
                self.assertEqual(request.max_stderr_bytes, 80)

    async def test_ls_builds_optional_path_request(self) -> None:
        spec = _spec("ls")
        result = _result("ls", stdout="src/\n")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(result)
        formatter = _RecordingFormatter()
        tool = LsTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "src",
            cwd=".",
            timeout_seconds=1.25,
            max_stdout_bytes=300,
            max_stderr_bytes=90,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.ls:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.ls")
        self.assertEqual(request.command, "ls")
        self.assertEqual(request.options, {})
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="src",
                    kind="any",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, ".")
        self.assertEqual(request.timeout_seconds, 1.25)
        self.assertEqual(request.max_stdout_bytes, 300)
        self.assertEqual(request.max_stderr_bytes, 90)

    async def test_ls_builds_no_path_request(self) -> None:
        policy = _FakePolicy(_spec("ls"))
        executor = _FakeExecutor(_result("ls"))
        tool = LsTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        await tool(context=ToolCallContext())

        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(policy.requests[0].paths, ())
        self.assertEqual(len(executor.specs), 1)

    async def test_ls_builds_empty_path_request_as_no_path(self) -> None:
        policy = _FakePolicy(_spec("ls"))
        executor = _FakeExecutor(_result("ls"))
        tool = LsTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        await tool("", context=ToolCallContext())

        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(policy.requests[0].paths, ())
        self.assertEqual(len(executor.specs), 1)

    async def test_ls_builds_empty_cwd_request_as_default(self) -> None:
        policy = _FakePolicy(_spec("ls"))
        executor = _FakeExecutor(_result("ls"))
        tool = LsTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        await tool(cwd="", context=ToolCallContext())

        self.assertEqual(len(policy.requests), 1)
        self.assertIsNone(policy.requests[0].cwd)
        self.assertEqual(len(executor.specs), 1)

    async def test_cat_builds_text_file_request(self) -> None:
        spec = _spec("cat")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("cat", stdout="body\n"))
        formatter = _RecordingFormatter()
        tool = CatTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "README.md",
            cwd="docs",
            timeout_seconds=2.25,
            max_stdout_bytes=400,
            max_stderr_bytes=120,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.cat:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.cat")
        self.assertEqual(request.command, "cat")
        self.assertEqual(request.options, {})
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="README.md",
                    kind="text_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 2.25)
        self.assertEqual(request.max_stdout_bytes, 400)
        self.assertEqual(request.max_stderr_bytes, 120)

    async def test_nl_builds_text_file_request(self) -> None:
        spec = _spec("nl")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("nl", stdout="     1\tbody\n"))
        formatter = _RecordingFormatter()
        tool = NlTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "README.md",
            cwd="docs",
            body_numbering="all",
            number_format="right_zero",
            number_separator="colon_space",
            starting_line_number=10,
            line_increment=5,
            number_width=4,
            timeout_seconds=2.35,
            max_stdout_bytes=425,
            max_stderr_bytes=122,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.nl:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.nl")
        self.assertEqual(request.command, "nl")
        self.assertEqual(
            request.options,
            {
                "body_numbering": "all",
                "number_format": "right_zero",
                "number_separator": "colon_space",
                "starting_line_number": 10,
                "line_increment": 5,
                "number_width": 4,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="README.md",
                    kind="text_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 2.35)
        self.assertEqual(request.max_stdout_bytes, 425)
        self.assertEqual(request.max_stderr_bytes, 122)

    async def test_file_builds_regular_file_request(self) -> None:
        spec = _spec("file")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("file", stdout="text/plain\n"))
        formatter = _RecordingFormatter()
        tool = FileTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            ("README.md", "data.bin"),
            cwd="docs",
            brief=True,
            mime_type=True,
            timeout_seconds=2.5,
            max_stdout_bytes=450,
            max_stderr_bytes=125,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.file:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.file")
        self.assertEqual(request.command, "file")
        self.assertEqual(request.options, {"brief": True, "mime_type": True})
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="README.md",
                    kind="file",
                    access="read",
                ),
                PathOperand(
                    name="path_1",
                    path="data.bin",
                    kind="file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 2.5)
        self.assertEqual(request.max_stdout_bytes, 450)
        self.assertEqual(request.max_stderr_bytes, 125)

    async def test_find_builds_structured_search_request(self) -> None:
        spec = _spec("find")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("find", stdout="./README.md\n"))
        formatter = _RecordingFormatter()
        tool = FindTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            ("src", "tests"),
            cwd=".",
            min_depth=1,
            max_depth=2,
            entry_type="file",
            name="README.md",
            timeout_seconds=2.75,
            max_stdout_bytes=475,
            max_stderr_bytes=127,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.find:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.find")
        self.assertEqual(request.command, "find")
        self.assertEqual(
            request.options,
            {
                "min_depth": 1,
                "max_depth": 2,
                "entry_type": "file",
                "name": "README.md",
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="src",
                    kind="any",
                    access="read",
                ),
                PathOperand(
                    name="path_1",
                    path="tests",
                    kind="any",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, ".")
        self.assertEqual(request.timeout_seconds, 2.75)
        self.assertEqual(request.max_stdout_bytes, 475)
        self.assertEqual(request.max_stderr_bytes, 127)

    async def test_wc_builds_multi_path_request(self) -> None:
        spec = _spec("wc")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("wc", stdout="2 4 12 total\n"))
        formatter = _RecordingFormatter()
        tool = WcTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            ("one.txt", "two.txt"),
            cwd="texts",
            lines=True,
            words=True,
            count_bytes=True,
            timeout_seconds=3.0,
            max_stdout_bytes=500,
            max_stderr_bytes=140,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.wc:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.wc")
        self.assertEqual(request.command, "wc")
        self.assertEqual(
            request.options,
            {"lines": True, "words": True, "count_bytes": True},
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="one.txt",
                    kind="text_file",
                    access="read",
                ),
                PathOperand(
                    name="path_1",
                    path="two.txt",
                    kind="text_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "texts")
        self.assertEqual(request.timeout_seconds, 3.0)
        self.assertEqual(request.max_stdout_bytes, 500)
        self.assertEqual(request.max_stderr_bytes, 140)

    async def test_wc_preserves_false_count_flags_for_policy_default(
        self,
    ) -> None:
        policy = _FakePolicy(_spec("wc"))
        executor = _FakeExecutor(_result("wc"))
        tool = WcTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        await tool(
            ("one.txt",),
            lines=False,
            words=False,
            count_bytes=False,
            context=ToolCallContext(),
        )

        self.assertEqual(
            policy.requests[0].options,
            {"lines": False, "words": False, "count_bytes": False},
        )
        self.assertEqual(len(executor.specs), 1)

    async def test_file_and_find_preserve_default_options(self) -> None:
        cases = (
            (
                "file",
                _file_tool,
                {"paths": ("README.md",)},
                {"brief": False, "mime_type": False},
            ),
            (
                "find",
                _find_tool,
                {},
                {
                    "min_depth": None,
                    "max_depth": 3,
                    "entry_type": "any",
                    "name": None,
                },
            ),
        )
        for command, tool, arguments, options in cases:
            with self.subTest(command=command):
                policy = _FakePolicy(_spec(command))
                executor = _FakeExecutor(_result(command))

                await tool(
                    policy,
                    executor,
                    _RecordingFormatter(),
                )(**arguments, context=ToolCallContext())

                self.assertEqual(policy.requests[0].options, options)
                self.assertEqual(len(executor.specs), 1)

    async def test_awk_builds_structured_filter_request(self) -> None:
        spec = _spec("awk")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("awk", stdout="value\n"))
        formatter = _RecordingFormatter()
        tool = AwkTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            ("table.tsv",),
            fields=(1, 3),
            field_separator="tab",
            output_separator=",",
            pattern="active",
            start_line=2,
            end_line=20,
            cwd="data",
            timeout_seconds=4.0,
            max_stdout_bytes=600,
            max_stderr_bytes=160,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.awk:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.awk")
        self.assertEqual(request.command, "awk")
        self.assertEqual(
            request.options,
            {
                "fields": (1, 3),
                "field_separator": "tab",
                "output_separator": ",",
                "pattern": "active",
                "start_line": 2,
                "end_line": 20,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="table.tsv",
                    kind="text_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "data")
        self.assertEqual(request.timeout_seconds, 4.0)
        self.assertEqual(request.max_stdout_bytes, 600)
        self.assertEqual(request.max_stderr_bytes, 160)

    async def test_awk_preserves_default_filter_options(self) -> None:
        policy = _FakePolicy(_spec("awk"))
        executor = _FakeExecutor(_result("awk"))
        tool = AwkTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        await tool(("table.tsv",), context=ToolCallContext())

        self.assertEqual(
            policy.requests[0].options,
            {
                "fields": None,
                "field_separator": "whitespace",
                "output_separator": " ",
                "pattern": None,
                "start_line": None,
                "end_line": None,
            },
        )
        self.assertEqual(len(executor.specs), 1)

    async def test_sed_builds_structured_selector_request(self) -> None:
        spec = _spec("sed")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("sed", stdout="line\n"))
        formatter = _RecordingFormatter()
        tool = SedTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            ("one.txt", "two.txt"),
            line_ranges=("1,5", "8"),
            patterns=("error", "warning"),
            cwd="logs",
            timeout_seconds=4.5,
            max_stdout_bytes=700,
            max_stderr_bytes=170,
            start_line=2,
            end_line=50,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.sed:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.sed")
        self.assertEqual(request.command, "sed")
        self.assertEqual(
            request.options,
            {
                "line_ranges": ("1,5", "8"),
                "patterns": ("error", "warning"),
                "start_line": 2,
                "end_line": 50,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="one.txt",
                    kind="text_file",
                    access="read",
                ),
                PathOperand(
                    name="path_1",
                    path="two.txt",
                    kind="text_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "logs")
        self.assertEqual(request.timeout_seconds, 4.5)
        self.assertEqual(request.max_stdout_bytes, 700)
        self.assertEqual(request.max_stderr_bytes, 170)

    async def test_jq_builds_structured_json_filter_request(self) -> None:
        spec = _spec("jq", output_kind=ShellOutputKind.JSON)
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("jq", stdout='{"name":"one"}\n'))
        formatter = _RecordingFormatter()
        tool = JqTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            ".items[]",
            ("data.json",),
            cwd="fixtures",
            raw_output=True,
            compact=True,
            slurp=True,
            sort_keys=True,
            timeout_seconds=5.0,
            max_stdout_bytes=800,
            max_stderr_bytes=180,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.jq:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.jq")
        self.assertEqual(request.command, "jq")
        self.assertEqual(
            request.options,
            {
                "filter": ".items[]",
                "raw_output": True,
                "compact": True,
                "slurp": True,
                "sort_keys": True,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="data.json",
                    kind="json_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "fixtures")
        self.assertEqual(request.timeout_seconds, 5.0)
        self.assertEqual(request.max_stdout_bytes, 800)
        self.assertEqual(request.max_stderr_bytes, 180)

    async def test_pdfinfo_builds_structured_pdf_request(self) -> None:
        spec = _spec("pdfinfo")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("pdfinfo", stdout="Pages: 1\n"))
        formatter = _RecordingFormatter()
        tool = PdfInfoTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "docs/report.pdf",
            first_page=2,
            last_page=4,
            boxes=True,
            iso_dates=True,
            cwd="docs",
            timeout_seconds=5.5,
            max_stdout_bytes=850,
            max_stderr_bytes=185,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.pdfinfo:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.pdfinfo")
        self.assertEqual(request.command, "pdfinfo")
        self.assertEqual(
            request.options,
            {
                "first_page": 2,
                "last_page": 4,
                "boxes": True,
                "iso_dates": True,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="docs/report.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 5.5)
        self.assertEqual(request.max_stdout_bytes, 850)
        self.assertEqual(request.max_stderr_bytes, 185)

    async def test_pdftotext_builds_structured_pdf_request(self) -> None:
        spec = _spec("pdftotext")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("pdftotext", stdout="text\n"))
        formatter = _RecordingFormatter()
        tool = PdfToTextTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "docs/report.pdf",
            first_page=2,
            last_page=4,
            layout=True,
            no_page_breaks=True,
            cwd="docs",
            timeout_seconds=6.0,
            max_stdout_bytes=900,
            max_stderr_bytes=190,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.pdftotext:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.pdftotext")
        self.assertEqual(request.command, "pdftotext")
        self.assertEqual(
            request.options,
            {
                "first_page": 2,
                "last_page": 4,
                "layout": True,
                "no_page_breaks": True,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="docs/report.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 6.0)
        self.assertEqual(request.max_stdout_bytes, 900)
        self.assertEqual(request.max_stderr_bytes, 190)

    async def test_pdftoppm_builds_structured_raster_request(self) -> None:
        spec = _spec("pdftoppm", output_kind=ShellOutputKind.GENERATED_FILES)
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("pdftoppm"))
        formatter = _RecordingFormatter()
        tool = PdfToPpmTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "docs/report.pdf",
            first_page=3,
            last_page=5,
            dpi=200,
            grayscale=True,
            format="png",
            cwd="docs",
            timeout_seconds=7.0,
            max_stdout_bytes=1000,
            max_stderr_bytes=200,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.pdftoppm:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.pdftoppm")
        self.assertEqual(request.command, "pdftoppm")
        self.assertEqual(
            request.options,
            {
                "first_page": 3,
                "last_page": 5,
                "dpi": 200,
                "grayscale": True,
                "format": "png",
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="docs/report.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 7.0)
        self.assertEqual(request.max_stdout_bytes, 1000)
        self.assertEqual(request.max_stderr_bytes, 200)

    async def test_reportlab_builds_structured_pdf_request(self) -> None:
        spec = _spec(
            "reportlab",
            output_kind=ShellOutputKind.GENERATED_FILES,
        )
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("reportlab"))
        formatter = _RecordingFormatter()
        tool = ReportLabTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "Generated body",
            title="Generated Title",
            page_size="a4",
            cwd="docs",
            timeout_seconds=7.5,
            max_stdout_bytes=1050,
            max_stderr_bytes=205,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.reportlab:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.reportlab")
        self.assertEqual(request.command, "reportlab")
        self.assertEqual(
            request.options,
            {
                "text": "Generated body",
                "title": "Generated Title",
                "page_size": "a4",
            },
        )
        self.assertEqual(request.paths, ())
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 7.5)
        self.assertEqual(request.max_stdout_bytes, 1050)
        self.assertEqual(request.max_stderr_bytes, 205)

    async def test_pdfplumber_builds_structured_pdf_request(self) -> None:
        spec = _spec("pdfplumber")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("pdfplumber", stdout="text\n"))
        formatter = _RecordingFormatter()
        tool = PdfPlumberTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "docs/report.pdf",
            mode="tables",
            first_page=2,
            last_page=4,
            layout=True,
            cwd="docs",
            timeout_seconds=8.0,
            max_stdout_bytes=1100,
            max_stderr_bytes=210,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.pdfplumber:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.pdfplumber")
        self.assertEqual(request.command, "pdfplumber")
        self.assertEqual(
            request.options,
            {
                "mode": "tables",
                "first_page": 2,
                "last_page": 4,
                "layout": True,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="docs/report.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 8.0)
        self.assertEqual(request.max_stdout_bytes, 1100)
        self.assertEqual(request.max_stderr_bytes, 210)

    async def test_pypdf_builds_structured_pdf_request(self) -> None:
        spec = _spec("pypdf")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("pypdf", stdout="text\n"))
        formatter = _RecordingFormatter()
        tool = PyPdfTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "docs/report.pdf",
            mode="text",
            first_page=2,
            last_page=4,
            cwd="docs",
            timeout_seconds=8.5,
            max_stdout_bytes=1150,
            max_stderr_bytes=215,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.pypdf:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.pypdf")
        self.assertEqual(request.command, "pypdf")
        self.assertEqual(
            request.options,
            {
                "mode": "text",
                "first_page": 2,
                "last_page": 4,
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="docs/report.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "docs")
        self.assertEqual(request.timeout_seconds, 8.5)
        self.assertEqual(request.max_stdout_bytes, 1150)
        self.assertEqual(request.max_stderr_bytes, 215)

    async def test_pypdf_metadata_mode_preserves_page_options(
        self,
    ) -> None:
        spec = _spec("pypdf")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("pypdf"))
        formatter = _RecordingFormatter()
        tool = PyPdfTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        await tool(
            "docs/report.pdf",
            mode="metadata",
            first_page=2,
            context=ToolCallContext(),
        )
        await tool("docs/report.pdf", context=ToolCallContext())

        self.assertEqual(
            policy.requests[0].options,
            {"mode": "metadata", "first_page": 2, "last_page": None},
        )
        self.assertEqual(policy.requests[1].options, {"mode": "metadata"})

    async def test_tesseract_builds_structured_ocr_request(self) -> None:
        spec = _spec("tesseract")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("tesseract", stdout="text\n"))
        formatter = _RecordingFormatter()
        tool = TesseractTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "images/page.png",
            languages=("eng", "spa"),
            psm=6,
            oem=1,
            dpi=300,
            output_format="txt",
            cwd="images",
            timeout_seconds=8.0,
            max_stdout_bytes=1100,
            max_stderr_bytes=210,
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.tesseract:completed")
        self.assertEqual(executor.specs, [spec])
        self.assertEqual(len(policy.requests), 1)
        request = policy.requests[0]
        self.assertEqual(request.tool_name, "shell.tesseract")
        self.assertEqual(request.command, "tesseract")
        self.assertEqual(
            request.options,
            {
                "languages": ("eng", "spa"),
                "psm": 6,
                "oem": 1,
                "dpi": 300,
                "output_format": "txt",
            },
        )
        self.assertEqual(
            request.paths,
            (
                PathOperand(
                    name="path_0",
                    path="images/page.png",
                    kind="image_file",
                    access="read",
                ),
            ),
        )
        self.assertEqual(request.cwd, "images")
        self.assertEqual(request.timeout_seconds, 8.0)
        self.assertEqual(request.max_stdout_bytes, 1100)
        self.assertEqual(request.max_stderr_bytes, 210)

    async def test_media_wrappers_preserve_default_options(self) -> None:
        cases = (
            (
                "pdfinfo",
                _pdfinfo_tool,
                {"path": "report.pdf"},
                {
                    "first_page": None,
                    "last_page": None,
                    "boxes": False,
                    "iso_dates": False,
                },
            ),
            (
                "pdftotext",
                _pdftotext_tool,
                {"path": "report.pdf"},
                {
                    "first_page": 1,
                    "last_page": None,
                    "layout": False,
                    "no_page_breaks": False,
                },
            ),
            (
                "pdftoppm",
                _pdftoppm_tool,
                {"path": "report.pdf"},
                {
                    "first_page": 1,
                    "last_page": None,
                    "dpi": None,
                    "grayscale": False,
                    "format": "png",
                },
            ),
            (
                "reportlab",
                _reportlab_tool,
                {"text": "body"},
                {
                    "text": "body",
                    "title": "Document",
                    "page_size": "letter",
                },
            ),
            (
                "pdfplumber",
                _pdfplumber_tool,
                {"path": "report.pdf"},
                {
                    "mode": "text",
                    "first_page": 1,
                    "last_page": None,
                    "layout": False,
                },
            ),
            (
                "pypdf",
                _pypdf_tool,
                {"path": "report.pdf"},
                {"mode": "metadata"},
            ),
            (
                "tesseract",
                _tesseract_tool,
                {"path": "page.png"},
                {
                    "languages": None,
                    "psm": 3,
                    "oem": None,
                    "dpi": None,
                    "output_format": "txt",
                },
            ),
        )
        for command, tool, arguments, options in cases:
            with self.subTest(command=command):
                policy = _FakePolicy(_spec(command))
                executor = _FakeExecutor(_result(command))

                await tool(
                    policy,
                    executor,
                    _RecordingFormatter(),
                )(**arguments, context=ToolCallContext())

                self.assertEqual(policy.requests[0].options, options)
                self.assertEqual(len(executor.specs), 1)

    async def test_policy_denial_formats_without_executor(self) -> None:
        policy = _DenyingPolicy(
            ShellPolicyDenied(
                ShellExecutionErrorCode.INVALID_OPTION,
                "bad option",
            )
        )
        executor = _FakeExecutor(_result("rg"))
        formatter = _RecordingFormatter()
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool("needle", context=ToolCallContext())

        self.assertEqual(output, "formatted:shell.rg:policy_denied")
        self.assertEqual(executor.specs, [])
        self.assertEqual(len(formatter.results), 1)
        denied = formatter.results[0]
        self.assertEqual(denied.status, ShellExecutionStatus.POLICY_DENIED)
        self.assertEqual(
            denied.error_code,
            ShellExecutionErrorCode.INVALID_OPTION,
        )
        self.assertEqual(denied.error_message, "bad option")
        self.assertEqual(denied.command, "rg")
        self.assertEqual(denied.display_argv, ("rg",))

    async def test_policy_denial_uses_neutral_cwd(self) -> None:
        policy = _DenyingPolicy(
            ShellPolicyDenied(
                ShellExecutionErrorCode.INVALID_CWD,
                "bad cwd",
            )
        )
        executor = _FakeExecutor(_result("rg"))
        formatter = _RecordingFormatter()
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=formatter,
        )

        output = await tool(
            "needle",
            cwd="/Users/mariano/private",
            context=ToolCallContext(),
        )

        self.assertEqual(output, "formatted:shell.rg:policy_denied")
        self.assertEqual(executor.specs, [])
        self.assertEqual(len(formatter.results), 1)
        denied = formatter.results[0]
        self.assertEqual(denied.cwd, ".")
        self.assertEqual(denied.display_cwd, ".")
        self.assertNotIn("/Users/mariano/private", denied.cwd)
        self.assertNotIn("/Users/mariano/private", denied.display_cwd)

    async def test_core_reader_policy_denials_do_not_call_executor(
        self,
    ) -> None:
        for command, tool, arguments in (
            ("ls", _ls_tool, {}),
            ("cat", _cat_tool, {"path": "README.md"}),
            ("nl", _nl_tool, {"path": "README.md"}),
            ("file", _file_tool, {"paths": ("README.md",)}),
            ("find", _find_tool, {"paths": ("src",)}),
            ("wc", _wc_tool, {"paths": ("README.md",)}),
        ):
            with self.subTest(command=command):
                policy = _DenyingPolicy(
                    ShellPolicyDenied(
                        ShellExecutionErrorCode.DENIED_PATH,
                        "denied path",
                    )
                )
                executor = _FakeExecutor(_result(command))
                formatter = _RecordingFormatter()
                output = await tool(policy, executor, formatter)(
                    **arguments,
                    context=ToolCallContext(),
                )

                self.assertEqual(
                    output,
                    f"formatted:shell.{command}:policy_denied",
                )
                self.assertEqual(len(policy.requests), 1)
                self.assertEqual(executor.specs, [])
                self.assertEqual(len(formatter.results), 1)
                self.assertEqual(
                    formatter.results[0].error_code,
                    ShellExecutionErrorCode.DENIED_PATH,
                )

    async def test_filter_policy_denials_do_not_call_executor(self) -> None:
        for command, tool, arguments in (
            ("awk", _awk_tool, {"paths": ("table.tsv",)}),
            (
                "sed",
                _sed_tool,
                {"paths": ("logs.txt",), "line_ranges": ("1,2",)},
            ),
            ("jq", _jq_tool, {"filter": ".", "paths": ("data.json",)}),
        ):
            with self.subTest(command=command):
                policy = _DenyingPolicy(
                    ShellPolicyDenied(
                        ShellExecutionErrorCode.UNSAFE_FILTER,
                        "unsafe filter",
                    )
                )
                executor = _FakeExecutor(_result(command))
                formatter = _RecordingFormatter()
                output = await tool(policy, executor, formatter)(
                    **arguments,
                    context=ToolCallContext(),
                )

                self.assertEqual(
                    output,
                    f"formatted:shell.{command}:policy_denied",
                )
                self.assertEqual(len(policy.requests), 1)
                self.assertEqual(executor.specs, [])
                self.assertEqual(len(formatter.results), 1)
                self.assertEqual(
                    formatter.results[0].error_code,
                    ShellExecutionErrorCode.UNSAFE_FILTER,
                )

    async def test_media_policy_denials_do_not_call_executor(self) -> None:
        for command, tool, arguments in (
            ("pdfinfo", _pdfinfo_tool, {"path": "report.pdf"}),
            ("pdftotext", _pdftotext_tool, {"path": "report.pdf"}),
            ("pdftoppm", _pdftoppm_tool, {"path": "report.pdf"}),
            ("reportlab", _reportlab_tool, {"text": "body"}),
            ("pdfplumber", _pdfplumber_tool, {"path": "report.pdf"}),
            ("pypdf", _pypdf_tool, {"path": "report.pdf"}),
            ("tesseract", _tesseract_tool, {"path": "page.png"}),
        ):
            with self.subTest(command=command):
                policy = _DenyingPolicy(
                    ShellPolicyDenied(
                        ShellExecutionErrorCode.DENIED_COMMAND,
                        "media tools are disabled",
                    )
                )
                executor = _FakeExecutor(_result(command))
                formatter = _RecordingFormatter()
                output = await tool(policy, executor, formatter)(
                    **arguments,
                    context=ToolCallContext(),
                )

                self.assertEqual(
                    output,
                    f"formatted:shell.{command}:policy_denied",
                )
                self.assertEqual(len(policy.requests), 1)
                self.assertEqual(executor.specs, [])
                self.assertEqual(len(formatter.results), 1)
                self.assertEqual(
                    formatter.results[0].error_code,
                    ShellExecutionErrorCode.DENIED_COMMAND,
                )

    async def test_default_formatter_renders_shell_result(self) -> None:
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=_FakePolicy(_spec("rg")),  # type: ignore[arg-type]
            executor=_FakeExecutor(_result("rg", stdout="match\n")),
        )

        output = await tool("needle", context=ToolCallContext())

        self.assertIn("tool: shell.rg", output)
        self.assertIn("status: completed", output)
        self.assertIn("match", output)

    async def test_abstract_base_call_is_inert(self) -> None:
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=_FakePolicy(_spec("rg")),  # type: ignore[arg-type]
            executor=_FakeExecutor(_result("rg")),
            formatter=_RecordingFormatter(),
        )

        with self.assertRaises(NotImplementedError):
            await _ShellCommandTool.__call__(tool)

    async def test_invalid_structured_arguments_fail_before_policy(
        self,
    ) -> None:
        spec = _spec("rg")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("rg"))
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        with self.assertRaises(AssertionError):
            await tool(  # type: ignore[arg-type]
                "needle",
                paths="src",
                context=ToolCallContext(),
            )
        with self.assertRaises(AssertionError):
            await tool(  # type: ignore[list-item]
                "needle",
                paths=(1,),
                context=ToolCallContext(),
            )
        with self.assertRaises(AssertionError):
            await tool(  # type: ignore[arg-type]
                "needle",
                globs="*.py",
                context=ToolCallContext(),
            )
        with self.assertRaises(AssertionError):
            await tool(  # type: ignore[list-item]
                "needle",
                globs=(1,),
                context=ToolCallContext(),
            )
        self.assertEqual(policy.requests, [])
        self.assertEqual(executor.specs, [])

    async def test_invalid_core_reader_arguments_fail_before_policy(
        self,
    ) -> None:
        for command, tool, arguments in (
            ("cat", _cat_tool, {"path": ""}),
            ("nl", _nl_tool, {"path": ""}),
            ("file", _file_tool, {"paths": "README.md"}),
            ("find", _find_tool, {"paths": "src"}),
            ("wc", _wc_tool, {"paths": "README.md"}),
        ):
            with self.subTest(command=command):
                policy = _FakePolicy(_spec(command))
                executor = _FakeExecutor(_result(command))

                with self.assertRaises(AssertionError):
                    await tool(policy, executor, _RecordingFormatter())(
                        **arguments,
                        context=ToolCallContext(),
                    )
                self.assertEqual(policy.requests, [])
                self.assertEqual(executor.specs, [])

    async def test_empty_line_reader_path_fails_before_policy(self) -> None:
        policy = _FakePolicy(_spec("head"))
        executor = _FakeExecutor(_result("head"))
        tool = HeadTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )

        with self.assertRaises(AssertionError):
            await tool("", context=ToolCallContext())
        self.assertEqual(policy.requests, [])
        self.assertEqual(executor.specs, [])

    async def test_invalid_filter_arguments_fail_before_policy(self) -> None:
        invalid_cases = (
            (
                "awk-paths",
                _awk_tool,
                {"paths": "table.tsv"},
            ),
            (
                "awk-fields",
                _awk_tool,
                {"paths": ("table.tsv",), "fields": "1"},
            ),
            (
                "awk-field-item",
                _awk_tool,
                {"paths": ("table.tsv",), "fields": ("1",)},
            ),
            (
                "awk-field-bool",
                _awk_tool,
                {"paths": ("table.tsv",), "fields": (True,)},
            ),
            (
                "sed-line-ranges",
                _sed_tool,
                {"paths": ("logs.txt",), "line_ranges": "1,2"},
            ),
            (
                "sed-line-range-item",
                _sed_tool,
                {"paths": ("logs.txt",), "line_ranges": (1,)},
            ),
            (
                "sed-patterns",
                _sed_tool,
                {"paths": ("logs.txt",), "patterns": "error"},
            ),
            (
                "sed-pattern-item",
                _sed_tool,
                {"paths": ("logs.txt",), "patterns": (1,)},
            ),
            (
                "jq-paths",
                _jq_tool,
                {"filter": ".", "paths": "data.json"},
            ),
            (
                "jq-path-item",
                _jq_tool,
                {"filter": ".", "paths": (1,)},
            ),
        )
        for name, tool, arguments in invalid_cases:
            with self.subTest(name=name):
                policy = _FakePolicy(_spec(name.split("-", maxsplit=1)[0]))
                executor = _FakeExecutor(_result("rg"))

                with self.assertRaises(AssertionError):
                    await tool(policy, executor, _RecordingFormatter())(
                        **arguments,
                        context=ToolCallContext(),
                    )
                self.assertEqual(policy.requests, [])
                self.assertEqual(executor.specs, [])

    async def test_invalid_media_arguments_fail_before_policy(self) -> None:
        invalid_cases = (
            (
                "pdfinfo",
                _pdfinfo_tool,
                {"path": ""},
            ),
            (
                "pdftotext",
                _pdftotext_tool,
                {"path": ""},
            ),
            (
                "pdftoppm",
                _pdftoppm_tool,
                {"path": ""},
            ),
            (
                "pdfplumber",
                _pdfplumber_tool,
                {"path": ""},
            ),
            (
                "pypdf",
                _pypdf_tool,
                {"path": ""},
            ),
            (
                "tesseract-path",
                _tesseract_tool,
                {"path": ""},
            ),
            (
                "tesseract-languages",
                _tesseract_tool,
                {"path": "page.png", "languages": "eng"},
            ),
            (
                "tesseract-language-item",
                _tesseract_tool,
                {"path": "page.png", "languages": (1,)},
            ),
        )
        for name, tool, arguments in invalid_cases:
            with self.subTest(name=name):
                command = name.split("-", maxsplit=1)[0]
                policy = _FakePolicy(_spec(command))
                executor = _FakeExecutor(_result(command))

                with self.assertRaises(AssertionError):
                    await tool(policy, executor, _RecordingFormatter())(
                        **arguments,
                        context=ToolCallContext(),
                    )
                self.assertEqual(policy.requests, [])
                self.assertEqual(executor.specs, [])

    async def test_large_rg_request_construction_is_bounded(self) -> None:
        spec = _spec("rg")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("rg"))
        tool = RgTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )
        paths = tuple(f"path-{index}.txt" for index in range(128))
        globs = tuple(f"*.{index}" for index in range(32))

        await tool(
            "needle",
            paths=paths,
            globs=globs,
            context_lines=10,
            max_matches_per_file=100,
            context=ToolCallContext(),
        )

        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(len(policy.requests[0].paths), len(paths))
        self.assertEqual(policy.requests[0].options["globs"], globs)
        self.assertIsNone(policy.requests[0].options["before_context"])
        self.assertIsNone(policy.requests[0].options["after_context"])
        self.assertIsNone(policy.requests[0].options["max_depth"])
        self.assertIsNone(policy.requests[0].options["max_filesize_bytes"])
        self.assertEqual(len(executor.specs), 1)

    async def test_large_line_reader_request_construction_is_bounded(
        self,
    ) -> None:
        for tool_class, command in (
            (HeadTool, "head"),
            (TailTool, "tail"),
        ):
            with self.subTest(command=command):
                spec = _spec(command)
                policy = _FakePolicy(spec)
                executor = _FakeExecutor(_result(command))
                tool = tool_class(
                    settings=ShellToolSettings(),
                    policy=policy,  # type: ignore[arg-type]
                    executor=executor,
                    formatter=_RecordingFormatter(),
                )

                await tool(
                    "logs/large.txt",
                    lines=100_000,
                    context=ToolCallContext(),
                )

                self.assertEqual(len(policy.requests), 1)
                self.assertEqual(
                    policy.requests[0].options,
                    (
                        {"lines": 100_000, "byte_count": None}
                        if command == "head"
                        else {
                            "lines": 100_000,
                            "start_line": None,
                            "byte_count": None,
                            "start_byte": None,
                        }
                    ),
                )
                self.assertEqual(len(policy.requests[0].paths), 1)
                self.assertEqual(len(executor.specs), 1)

    async def test_large_wc_request_construction_is_bounded(self) -> None:
        spec = _spec("wc")
        policy = _FakePolicy(spec)
        executor = _FakeExecutor(_result("wc"))
        tool = WcTool(
            settings=ShellToolSettings(),
            policy=policy,  # type: ignore[arg-type]
            executor=executor,
            formatter=_RecordingFormatter(),
        )
        paths = tuple(f"path-{index}.txt" for index in range(512))

        await tool(paths, context=ToolCallContext())

        self.assertEqual(len(policy.requests), 1)
        self.assertEqual(len(policy.requests[0].paths), len(paths))
        self.assertEqual(policy.requests[0].paths[-1].name, "path_511")
        self.assertEqual(len(executor.specs), 1)

    async def test_large_filter_request_construction_is_bounded(self) -> None:
        awk_policy = _FakePolicy(_spec("awk"))
        awk_executor = _FakeExecutor(_result("awk"))
        awk_tool = AwkTool(
            settings=ShellToolSettings(),
            policy=awk_policy,  # type: ignore[arg-type]
            executor=awk_executor,
            formatter=_RecordingFormatter(),
        )
        sed_policy = _FakePolicy(_spec("sed"))
        sed_executor = _FakeExecutor(_result("sed"))
        sed_tool = SedTool(
            settings=ShellToolSettings(),
            policy=sed_policy,  # type: ignore[arg-type]
            executor=sed_executor,
            formatter=_RecordingFormatter(),
        )
        jq_policy = _FakePolicy(_spec("jq"))
        jq_executor = _FakeExecutor(_result("jq"))
        jq_tool = JqTool(
            settings=ShellToolSettings(),
            policy=jq_policy,  # type: ignore[arg-type]
            executor=jq_executor,
            formatter=_RecordingFormatter(),
        )
        paths = tuple(f"path-{index}.txt" for index in range(128))
        fields = tuple(range(1, 65))
        line_ranges = tuple(str(index) for index in range(1, 33))
        jq_filter = "." + ("[]" * 256)

        await awk_tool(
            paths,
            fields=fields,
            pattern="needle",
            context=ToolCallContext(),
        )
        await sed_tool(
            paths,
            line_ranges=line_ranges,
            context=ToolCallContext(),
        )
        await jq_tool(
            jq_filter,
            ("data.json",),
            context=ToolCallContext(),
        )

        self.assertEqual(len(awk_policy.requests[0].paths), len(paths))
        self.assertEqual(awk_policy.requests[0].options["fields"], fields)
        self.assertEqual(len(sed_policy.requests[0].paths), len(paths))
        self.assertEqual(
            sed_policy.requests[0].options["line_ranges"],
            line_ranges,
        )
        self.assertEqual(jq_policy.requests[0].options["filter"], jq_filter)
        self.assertEqual(len(awk_executor.specs), 1)
        self.assertEqual(len(sed_executor.specs), 1)
        self.assertEqual(len(jq_executor.specs), 1)

    async def test_large_media_request_construction_is_bounded(self) -> None:
        pdf_text_policy = _FakePolicy(_spec("pdftotext"))
        pdf_text_executor = _FakeExecutor(_result("pdftotext"))
        pdf_text_tool = PdfToTextTool(
            settings=ShellToolSettings(),
            policy=pdf_text_policy,  # type: ignore[arg-type]
            executor=pdf_text_executor,
            formatter=_RecordingFormatter(),
        )
        pdf_raster_policy = _FakePolicy(
            _spec("pdftoppm", output_kind=ShellOutputKind.GENERATED_FILES)
        )
        pdf_raster_executor = _FakeExecutor(_result("pdftoppm"))
        pdf_raster_tool = PdfToPpmTool(
            settings=ShellToolSettings(),
            policy=pdf_raster_policy,  # type: ignore[arg-type]
            executor=pdf_raster_executor,
            formatter=_RecordingFormatter(),
        )
        ocr_policy = _FakePolicy(_spec("tesseract"))
        ocr_executor = _FakeExecutor(_result("tesseract"))
        ocr_tool = TesseractTool(
            settings=ShellToolSettings(),
            policy=ocr_policy,  # type: ignore[arg-type]
            executor=ocr_executor,
            formatter=_RecordingFormatter(),
        )
        languages = tuple(f"lang{index}" for index in range(64))

        await pdf_text_tool(
            "report.pdf",
            first_page=1,
            last_page=10_000,
            context=ToolCallContext(),
        )
        await pdf_raster_tool(
            "report.pdf",
            first_page=1,
            last_page=10_000,
            dpi=10_000,
            context=ToolCallContext(),
        )
        await ocr_tool(
            "page.png",
            languages=languages,
            psm=13,
            oem=3,
            dpi=10_000,
            context=ToolCallContext(),
        )

        self.assertEqual(
            pdf_text_policy.requests[0].options["last_page"],
            10_000,
        )
        self.assertEqual(
            pdf_raster_policy.requests[0].options["dpi"],
            10_000,
        )
        self.assertEqual(
            ocr_policy.requests[0].options["languages"], languages
        )
        self.assertEqual(len(pdf_text_executor.specs), 1)
        self.assertEqual(len(pdf_raster_executor.specs), 1)
        self.assertEqual(len(ocr_executor.specs), 1)


class ShellToolSchemaTest(TestCase):
    def test_schemas_expose_structured_parameters_and_omit_context(
        self,
    ) -> None:
        toolset = ToolSet(
            namespace="shell",
            tools=[
                RgTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("rg")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("rg")),
                    formatter=_RecordingFormatter(),
                ),
                HeadTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("head")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("head")),
                    formatter=_RecordingFormatter(),
                ),
                TailTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("tail")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("tail")),
                    formatter=_RecordingFormatter(),
                ),
                LsTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("ls")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("ls")),
                    formatter=_RecordingFormatter(),
                ),
                CatTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("cat")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("cat")),
                    formatter=_RecordingFormatter(),
                ),
                NlTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("nl")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("nl")),
                    formatter=_RecordingFormatter(),
                ),
                FileTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("file")
                    ),
                    executor=_FakeExecutor(_result("file")),
                    formatter=_RecordingFormatter(),
                ),
                FindTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("find")
                    ),
                    executor=_FakeExecutor(_result("find")),
                    formatter=_RecordingFormatter(),
                ),
                WcTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("wc")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("wc")),
                    formatter=_RecordingFormatter(),
                ),
                AwkTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("awk")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("awk")),
                    formatter=_RecordingFormatter(),
                ),
                SedTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("sed")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("sed")),
                    formatter=_RecordingFormatter(),
                ),
                JqTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(_spec("jq")),  # type: ignore[arg-type]
                    executor=_FakeExecutor(_result("jq")),
                    formatter=_RecordingFormatter(),
                ),
                PdfInfoTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("pdfinfo")
                    ),
                    executor=_FakeExecutor(_result("pdfinfo")),
                    formatter=_RecordingFormatter(),
                ),
                PdfToTextTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("pdftotext")
                    ),
                    executor=_FakeExecutor(_result("pdftotext")),
                    formatter=_RecordingFormatter(),
                ),
                PdfToPpmTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec(
                            "pdftoppm",
                            output_kind=ShellOutputKind.GENERATED_FILES,
                        )
                    ),
                    executor=_FakeExecutor(_result("pdftoppm")),
                    formatter=_RecordingFormatter(),
                ),
                ReportLabTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec(
                            "reportlab",
                            output_kind=ShellOutputKind.GENERATED_FILES,
                        )
                    ),
                    executor=_FakeExecutor(_result("reportlab")),
                    formatter=_RecordingFormatter(),
                ),
                PdfPlumberTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("pdfplumber")
                    ),
                    executor=_FakeExecutor(_result("pdfplumber")),
                    formatter=_RecordingFormatter(),
                ),
                PyPdfTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("pypdf")
                    ),
                    executor=_FakeExecutor(_result("pypdf")),
                    formatter=_RecordingFormatter(),
                ),
                TesseractTool(
                    settings=ShellToolSettings(),
                    policy=_FakePolicy(  # type: ignore[arg-type]
                        _spec("tesseract")
                    ),
                    executor=_FakeExecutor(_result("tesseract")),
                    formatter=_RecordingFormatter(),
                ),
            ],
        )

        schemas = toolset.json_schemas()
        names = [schema["function"]["name"] for schema in schemas]
        parameters = {
            schema["function"]["name"]: schema["function"]["parameters"]
            for schema in schemas
        }

        self.assertEqual(
            names,
            [
                "shell.rg",
                "shell.head",
                "shell.tail",
                "shell.ls",
                "shell.cat",
                "shell.nl",
                "shell.file",
                "shell.find",
                "shell.wc",
                "shell.awk",
                "shell.sed",
                "shell.jq",
                "shell.pdfinfo",
                "shell.pdftotext",
                "shell.pdftoppm",
                "shell.reportlab",
                "shell.pdfplumber",
                "shell.pypdf",
                "shell.tesseract",
            ],
        )
        self.assertNotIn("context", parameters["shell.rg"]["properties"])
        self.assertNotIn("context", parameters["shell.ls"]["properties"])
        self.assertNotIn("context", parameters["shell.cat"]["properties"])
        self.assertNotIn("context", parameters["shell.nl"]["properties"])
        self.assertNotIn("context", parameters["shell.file"]["properties"])
        self.assertNotIn("context", parameters["shell.find"]["properties"])
        self.assertNotIn("context", parameters["shell.wc"]["properties"])
        self.assertNotIn("context", parameters["shell.awk"]["properties"])
        self.assertNotIn("context", parameters["shell.sed"]["properties"])
        self.assertNotIn("context", parameters["shell.jq"]["properties"])
        self.assertNotIn(
            "context",
            parameters["shell.pdfinfo"]["properties"],
        )
        self.assertNotIn(
            "context",
            parameters["shell.pdftotext"]["properties"],
        )
        self.assertNotIn(
            "context",
            parameters["shell.pdftoppm"]["properties"],
        )
        self.assertNotIn(
            "context",
            parameters["shell.reportlab"]["properties"],
        )
        self.assertNotIn(
            "context",
            parameters["shell.pdfplumber"]["properties"],
        )
        self.assertNotIn(
            "context",
            parameters["shell.pypdf"]["properties"],
        )
        self.assertNotIn(
            "context",
            parameters["shell.tesseract"]["properties"],
        )
        self.assertEqual(parameters["shell.rg"]["required"], [])
        self.assertEqual(
            set(parameters["shell.rg"]["properties"]),
            {
                "pattern",
                "mode",
                "paths",
                "cwd",
                "case",
                "fixed_strings",
                "context_lines",
                "before_context",
                "after_context",
                "max_matches_per_file",
                "max_depth",
                "max_filesize_bytes",
                "globs",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            parameters["shell.rg"]["properties"]["pattern"]["type"],
            ["string", "null"],
        )
        self.assertEqual(
            parameters["shell.rg"]["properties"]["mode"]["enum"],
            ["search", "files"],
        )
        self.assertEqual(
            parameters["shell.rg"]["properties"]["mode"]["default"],
            "search",
        )
        rg_any_of = parameters["shell.rg"]["anyOf"]
        self.assertEqual(len(rg_any_of), 2)
        search_schema = rg_any_of[0]
        files_schema = rg_any_of[1]
        search_properties = search_schema["properties"]
        files_properties = files_schema["properties"]
        self.assertEqual(search_schema["required"], ["pattern"])
        self.assertEqual(
            search_properties["pattern"]["type"],
            "string",
        )
        self.assertEqual(search_properties["pattern"]["minLength"], 1)
        self.assertEqual(search_properties["mode"]["enum"], ["search"])
        self.assertEqual(files_schema["required"], ["mode"])
        self.assertEqual(files_properties["mode"]["enum"], ["files"])
        self.assertNotIn("default", files_properties["mode"])
        self.assertNotIn("pattern", files_properties)
        self.assertNotIn("case", files_properties)
        self.assertNotIn("fixed_strings", files_properties)
        self.assertNotIn("context_lines", files_properties)
        self.assertNotIn("before_context", files_properties)
        self.assertNotIn("after_context", files_properties)
        self.assertNotIn("max_matches_per_file", files_properties)
        self.assertEqual(parameters["shell.head"]["required"], ["path"])
        self.assertEqual(parameters["shell.tail"]["required"], ["path"])
        self.assertEqual(
            set(parameters["shell.head"]["properties"]),
            {
                "path",
                "lines",
                "byte_count",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.tail"]["properties"]),
            {
                "path",
                "lines",
                "start_line",
                "byte_count",
                "start_byte",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(parameters["shell.ls"]["required"], [])
        self.assertEqual(parameters["shell.cat"]["required"], ["path"])
        self.assertEqual(parameters["shell.nl"]["required"], ["path"])
        self.assertEqual(parameters["shell.file"]["required"], ["paths"])
        self.assertEqual(parameters["shell.find"]["required"], [])
        self.assertEqual(parameters["shell.wc"]["required"], ["paths"])
        self.assertEqual(parameters["shell.awk"]["required"], ["paths"])
        self.assertEqual(parameters["shell.sed"]["required"], ["paths"])
        self.assertEqual(
            parameters["shell.jq"]["required"],
            ["filter", "paths"],
        )
        self.assertEqual(parameters["shell.pdfinfo"]["required"], ["path"])
        self.assertEqual(parameters["shell.pdftotext"]["required"], ["path"])
        self.assertEqual(parameters["shell.pdftoppm"]["required"], ["path"])
        self.assertEqual(parameters["shell.reportlab"]["required"], ["text"])
        self.assertEqual(parameters["shell.pdfplumber"]["required"], ["path"])
        self.assertEqual(parameters["shell.pypdf"]["required"], ["path"])
        self.assertEqual(parameters["shell.tesseract"]["required"], ["path"])
        self.assertEqual(
            set(parameters["shell.nl"]["properties"]),
            {
                "path",
                "cwd",
                "body_numbering",
                "number_format",
                "number_separator",
                "starting_line_number",
                "line_increment",
                "number_width",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            parameters["shell.nl"]["properties"]["body_numbering"]["enum"],
            ["all", "nonempty", "none"],
        )
        self.assertEqual(
            parameters["shell.nl"]["properties"]["number_format"]["enum"],
            ["left", "right", "right_zero"],
        )
        self.assertEqual(
            parameters["shell.nl"]["properties"]["number_separator"]["enum"],
            ["colon_space", "space", "tab", "two_spaces"],
        )
        self.assertEqual(
            set(parameters["shell.file"]["properties"]),
            {
                "paths",
                "cwd",
                "brief",
                "mime_type",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.find"]["properties"]),
            {
                "paths",
                "cwd",
                "min_depth",
                "max_depth",
                "entry_type",
                "name",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            parameters["shell.find"]["properties"]["entry_type"]["enum"],
            ["any", "file", "directory"],
        )
        self.assertEqual(
            parameters["shell.rg"]["properties"]["before_context"][
                "description"
            ],
            "Number of leading context lines before each match.",
        )
        self.assertEqual(
            parameters["shell.head"]["properties"]["byte_count"][
                "description"
            ],
            "Native byte count to read via head -c.",
        )
        self.assertEqual(
            parameters["shell.tail"]["properties"]["start_line"][
                "description"
            ],
            "One-based line number to start at via tail -n +N.",
        )
        self.assertEqual(
            parameters["shell.find"]["properties"]["min_depth"]["description"],
            "Minimum traversal depth below each root.",
        )
        self.assertEqual(
            set(parameters["shell.wc"]["properties"]),
            {
                "paths",
                "cwd",
                "lines",
                "words",
                "count_bytes",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.awk"]["properties"]),
            {
                "paths",
                "fields",
                "field_separator",
                "output_separator",
                "pattern",
                "start_line",
                "end_line",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            parameters["shell.awk"]["properties"]["field_separator"]["enum"],
            ["whitespace", "tab", "comma", "pipe"],
        )
        self.assertEqual(
            set(parameters["shell.sed"]["properties"]),
            {
                "paths",
                "line_ranges",
                "patterns",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
                "start_line",
                "end_line",
            },
        )
        self.assertEqual(
            set(parameters["shell.jq"]["properties"]),
            {
                "filter",
                "paths",
                "cwd",
                "raw_output",
                "compact",
                "slurp",
                "sort_keys",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertNotIn("script", parameters["shell.awk"]["properties"])
        self.assertNotIn("script", parameters["shell.sed"]["properties"])
        self.assertNotIn("command", parameters["shell.jq"]["properties"])
        self.assertNotIn("expression", parameters["shell.find"]["properties"])
        self.assertEqual(
            set(parameters["shell.pdfinfo"]["properties"]),
            {
                "path",
                "first_page",
                "last_page",
                "boxes",
                "iso_dates",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.pdftotext"]["properties"]),
            {
                "path",
                "first_page",
                "last_page",
                "layout",
                "no_page_breaks",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.pdftoppm"]["properties"]),
            {
                "path",
                "first_page",
                "last_page",
                "dpi",
                "grayscale",
                "format",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.reportlab"]["properties"]),
            {
                "text",
                "title",
                "page_size",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.pdfplumber"]["properties"]),
            {
                "path",
                "mode",
                "first_page",
                "last_page",
                "layout",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.pypdf"]["properties"]),
            {
                "path",
                "mode",
                "first_page",
                "last_page",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            set(parameters["shell.tesseract"]["properties"]),
            {
                "path",
                "languages",
                "psm",
                "oem",
                "dpi",
                "output_format",
                "cwd",
                "timeout_seconds",
                "max_stdout_bytes",
                "max_stderr_bytes",
            },
        )
        self.assertEqual(
            parameters["shell.pdftoppm"]["properties"]["format"]["enum"],
            ["png"],
        )
        self.assertEqual(
            parameters["shell.reportlab"]["properties"]["page_size"]["enum"],
            ["letter", "a4"],
        )
        self.assertEqual(
            parameters["shell.pdfplumber"]["properties"]["mode"]["enum"],
            ["text", "tables"],
        )
        self.assertEqual(
            parameters["shell.pypdf"]["properties"]["mode"]["enum"],
            ["metadata", "text"],
        )
        self.assertEqual(
            parameters["shell.tesseract"]["properties"]["output_format"][
                "enum"
            ],
            ["txt"],
        )
        for name in (
            "shell.pdfinfo",
            "shell.pdftotext",
            "shell.pdftoppm",
            "shell.reportlab",
            "shell.pdfplumber",
            "shell.pypdf",
            "shell.tesseract",
        ):
            self.assertNotIn("output_path", parameters[name]["properties"])
            self.assertNotIn("output_prefix", parameters[name]["properties"])
            self.assertNotIn("config_path", parameters[name]["properties"])
            self.assertNotIn("tessdata_path", parameters[name]["properties"])


def _ls_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> LsTool:
    return LsTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _cat_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> CatTool:
    return CatTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _nl_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> NlTool:
    return NlTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _file_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> FileTool:
    return FileTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _find_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> FindTool:
    return FindTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _wc_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> WcTool:
    return WcTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _awk_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> AwkTool:
    return AwkTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _sed_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> SedTool:
    return SedTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _jq_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> JqTool:
    return JqTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _pdfinfo_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> PdfInfoTool:
    return PdfInfoTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _pdftotext_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> PdfToTextTool:
    return PdfToTextTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _pdftoppm_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> PdfToPpmTool:
    return PdfToPpmTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _reportlab_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> ReportLabTool:
    return ReportLabTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _pdfplumber_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> PdfPlumberTool:
    return PdfPlumberTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _pypdf_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> PyPdfTool:
    return PyPdfTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


def _tesseract_tool(
    policy: object,
    executor: object,
    formatter: "_RecordingFormatter",
) -> TesseractTool:
    return TesseractTool(
        settings=ShellToolSettings(),
        policy=policy,  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        formatter=formatter,
    )


class _FakePolicy:
    def __init__(self, spec: ExecutionSpec) -> None:
        self.requests: list[ShellCommandRequest] = []
        self._spec = spec

    async def normalize(self, request: ShellCommandRequest) -> ExecutionSpec:
        self.requests.append(request)
        return self._spec


class _DenyingPolicy:
    def __init__(self, error: ShellPolicyDenied) -> None:
        self.requests: list[ShellCommandRequest] = []
        self._error = error

    async def normalize(self, request: ShellCommandRequest) -> ExecutionSpec:
        self.requests.append(request)
        raise self._error


class _FakeExecutor:
    def __init__(self, result: ExecutionResult) -> None:
        self.specs: list[ExecutionSpec] = []
        self.streams: list[object | None] = []
        self._result = result

    async def execute(
        self,
        spec: ExecutionSpec,
        *,
        stream: object | None = None,
    ) -> ExecutionResult:
        self.specs.append(spec)
        self.streams.append(stream)
        return self._result


class _RecordingFormatter:
    def __init__(self) -> None:
        self.results: list[ExecutionResult] = []

    def __call__(self, result: ExecutionResult) -> str:
        self.results.append(result)
        return f"formatted:{result.tool_name}:{result.status.value}"


def _spec(
    command: str,
    *,
    output_kind: ShellOutputKind = ShellOutputKind.TEXT,
) -> ExecutionSpec:
    return ExecutionPolicy().create_execution_spec(
        backend="local",
        tool_name=f"shell.{command}",
        command=command,
        executable="/usr/bin/true",
        argv=(command,),
        display_argv=(command,),
        cwd=".",
        display_cwd=".",
        env={"LC_ALL": "C"},
        stdin=None,
        stdout_media_type="text/plain",
        output_kind=output_kind,
        resource_class="standard",
        output_plan=None,
        timeout_seconds=1.0,
        max_stdout_bytes=1024,
        max_stderr_bytes=1024,
    )


def _result(command: str, stdout: str = "") -> ExecutionResult:
    return ExecutionResult(
        backend="local",
        tool_name=f"shell.{command}",
        command=command,
        argv=(command,),
        display_argv=(command,),
        cwd=".",
        display_cwd=".",
        status=ShellExecutionStatus.COMPLETED,
        exit_code=0,
        stdout=stdout,
        stderr="",
        stdout_media_type="text/plain",
        output_kind=ShellOutputKind.TEXT,
        stdout_bytes=len(stdout.encode()),
        stderr_bytes=0,
        stdout_truncated=False,
        stderr_truncated=False,
        timed_out=False,
        cancelled=False,
        duration_ms=1,
        error_code=ShellExecutionErrorCode.COMPLETED,
    )


if __name__ == "__main__":
    main()
