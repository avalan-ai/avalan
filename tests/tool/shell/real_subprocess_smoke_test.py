from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from errno import EACCES, EPERM
from os import environ, getpid
from pathlib import Path
from sys import executable as python_executable
from unittest import IsolatedAsyncioTestCase, main, skipUnless
from uuid import uuid4

from avalan.entities import ToolCallContext
from avalan.tool import Tool
from avalan.tool.shell import (
    SHELL_COMMAND_DEFINITIONS,
    ExecutionPolicy,
    LocalCommandExecutor,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSet,
    ShellToolSettings,
    TrustedExecutableResolver,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures"
SMOKE_ENV_NAME = "AVALAN_SHELL_REAL_SMOKE"
SMOKE_PATHS_ENV_NAME = "AVALAN_SHELL_REAL_SMOKE_PATHS"
TRUSTED_SEARCH_PATHS = (
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "/usr/bin",
    "/usr/sbin",
    "/bin",
    "/sbin",
)


@skipUnless(
    environ.get(SMOKE_ENV_NAME) == "1",
    f"Set {SMOKE_ENV_NAME}=1 to run real shell command smoke tests.",
)
class RealSubprocessSmokeTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self._search_paths = _trusted_search_paths()
        self._resolver = TrustedExecutableResolver(
            executable_search_paths=self._search_paths,
        )
        settings = ShellToolSettings(
            workspace_root=str(FIXTURE_ROOT),
            allow_media_tools=True,
            allow_process_tools=True,
            executable_search_paths=self._search_paths,
            max_inline_output_file_bytes=256,
        )
        self._policy = ExecutionPolicy(
            settings=settings,
            resolver=self._resolver,
        )
        self._executor = LocalCommandExecutor()
        self._toolset = ShellToolSet(
            settings=settings,
            policy=self._policy,
            executor=self._executor,
        )

    async def test_rg_smoke(self) -> None:
        await self._require_command("rg")
        output = await _call(
            _tool_by_name(self._toolset, "rg"),
            "needle",
            paths=("filesystem/visible.txt",),
        )

        _assert_completed(self, output, "rg")
        self.assertIn("filesystem/visible.txt", output)
        self.assertIn("needle", output)

    async def test_rg_native_context_depth_and_size_smoke(self) -> None:
        await self._require_command("rg")
        output = await _execute_request(
            self,
            self._policy,
            self._executor,
            ShellCommandRequest(
                tool_name="shell.rg",
                command="rg",
                options={
                    "pattern": "needle",
                    "before_context": 1,
                    "max_depth": 1,
                    "max_filesize_bytes": 1024,
                },
                paths=(
                    PathOperand(
                        name="input",
                        path="filesystem",
                        kind="directory",
                        access="read",
                    ),
                ),
                cwd=None,
            ),
        )

        self.assertIn("filesystem/visible.txt", output)
        self.assertIn("beta", output)
        self.assertIn("needle", output)
        self.assertNotIn("alpha", output)

    async def test_head_smoke(self) -> None:
        await self._require_command("head")
        output = await _call(
            _tool_by_name(self._toolset, "head"),
            "filesystem/visible.txt",
            lines=1,
        )

        _assert_completed(self, output, "head")
        self.assertIn("alpha", output)
        self.assertNotIn("beta", output)

    async def test_head_bytes_smoke(self) -> None:
        await self._require_command("head")
        output = await _execute_request(
            self,
            self._policy,
            self._executor,
            _shell_request(
                "head",
                {"byte_count": 5},
                path="filesystem/visible.txt",
            ),
        )

        self.assertEqual(output, "alpha")

    async def test_tail_smoke(self) -> None:
        await self._require_command("tail")
        output = await _call(
            _tool_by_name(self._toolset, "tail"),
            "filesystem/visible.txt",
            lines=1,
        )

        _assert_completed(self, output, "tail")
        self.assertIn("needle", output)

    async def test_tail_native_start_and_bytes_smoke(self) -> None:
        await self._require_command("tail")
        from_second_line = await _execute_request(
            self,
            self._policy,
            self._executor,
            _shell_request(
                "tail",
                {"start_line": 2, "lines": 80},
                path="filesystem/visible.txt",
            ),
        )
        last_bytes = await _execute_request(
            self,
            self._policy,
            self._executor,
            _shell_request(
                "tail",
                {"byte_count": 6, "lines": 80},
                path="filesystem/visible.txt",
            ),
        )
        from_seventh_byte = await _execute_request(
            self,
            self._policy,
            self._executor,
            _shell_request(
                "tail",
                {"start_byte": 7, "lines": 80},
                path="filesystem/visible.txt",
            ),
        )

        self.assertEqual(from_second_line, "beta\nneedle\n")
        self.assertEqual(last_bytes, "eedle\n")
        self.assertEqual(from_seventh_byte, "beta\nneedle\n")

    async def test_ls_smoke(self) -> None:
        await self._require_command("ls")
        output = await _call(
            _tool_by_name(self._toolset, "ls"),
            "filesystem",
        )

        _assert_completed(self, output, "ls")
        self.assertIn("visible.txt", output)
        self.assertNotIn(".hidden.txt", output)

    async def test_ls_empty_path_smoke(self) -> None:
        await self._require_command("ls")
        output = await _call(_tool_by_name(self._toolset, "ls"), "")

        _assert_completed(self, output, "ls")
        self.assertIn("filesystem", output)
        self.assertIn("filters", output)

    async def test_cat_smoke(self) -> None:
        await self._require_command("cat")
        output = await _call(
            _tool_by_name(self._toolset, "cat"),
            "filesystem/visible.txt",
        )

        _assert_completed(self, output, "cat")
        self.assertIn("alpha", output)
        self.assertIn("needle", output)

    async def test_nl_smoke(self) -> None:
        await self._require_command("nl")
        output = await _call(
            _tool_by_name(self._toolset, "nl"),
            "filesystem/visible.txt",
            number_format="right_zero",
            number_separator="colon_space",
            starting_line_number=10,
            line_increment=5,
            number_width=4,
        )

        _assert_completed(self, output, "nl")
        self.assertIn("0010: alpha", output)
        self.assertIn("0020: needle", output)

    async def test_pgrep_smoke_and_no_match(self) -> None:
        await self._require_command("pgrep")
        await _skip_if_process_table_unavailable(self, self._resolver)
        pattern = f"avalan-pgrep-smoke-{uuid4().hex}"
        child = await create_subprocess_exec(
            python_executable,
            "-c",
            "import time; time.sleep(30)",
            pattern,
        )
        try:
            output = await _call(
                _tool_by_name(self._toolset, "pgrep"),
                pattern,
                full=True,
                parent_pid=getpid(),
            )

            _assert_completed(self, output, "pgrep")
            self.assertIn(str(child.pid), output)
            self.assertNotIn(pattern, output)

            no_match_pattern = f"{pattern}-definitely-absent"
            no_match = await _call(
                _tool_by_name(self._toolset, "pgrep"),
                no_match_pattern,
                full=True,
                parent_pid=getpid(),
            )

            self.assertIn("status: no_matches", no_match)
            self.assertIn("exit_code: 1", no_match)
            self.assertNotIn(no_match_pattern, no_match)
        finally:
            child.terminate()
            await child.wait()

    async def test_ps_smoke_and_no_match(self) -> None:
        await self._require_command("ps")
        await _skip_if_ps_process_table_unavailable(self, self._resolver)

        output = await _call(
            _tool_by_name(self._toolset, "ps"),
            (getpid(),),
        )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        _assert_completed(self, output, "ps")
        rows = output.execution_result.stdout.splitlines()
        self.assertTrue(
            any(row.split(maxsplit=1)[0] == str(getpid()) for row in rows)
        )

        resources = await _call(
            _tool_by_name(self._toolset, "ps"),
            (getpid(),),
            view="resources",
        )

        self.assertIsInstance(resources, ShellFormattedResult)
        assert isinstance(resources, ShellFormattedResult)
        _assert_completed(self, resources, "ps")
        resource_rows = resources.execution_result.stdout.splitlines()
        self.assertEqual(len(resource_rows), 1)
        self.assertEqual(resource_rows[0].split()[0], str(getpid()))
        self.assertEqual(len(resource_rows[0].split()), 7)

        no_match = await _call(
            _tool_by_name(self._toolset, "ps"),
            (2**31 - 1,),
        )

        self.assertIn("status: no_matches", no_match)
        self.assertIn("exit_code: 1", no_match)

    async def test_lsof_smoke_and_no_match(self) -> None:
        await self._require_command("lsof")
        await _skip_if_lsof_process_table_unavailable(self, self._resolver)

        output = await _call(
            _tool_by_name(self._toolset, "lsof"),
            getpid(),
            limit=8,
        )

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        _assert_completed(self, output, "lsof")
        rows = output.execution_result.stdout.splitlines()
        self.assertTrue(rows)
        for row in rows:
            fields = row.split("\t")
            self.assertEqual(len(fields), 5)
            self.assertEqual(fields[0], str(getpid()))
            self.assertTrue(fields[1].isascii())
            self.assertTrue(fields[1].isdigit())

        no_match = await _call(
            _tool_by_name(self._toolset, "lsof"),
            2**31 - 1,
        )

        self.assertIn("status: no_matches", no_match)
        self.assertIn("exit_code: 1", no_match)

    async def test_file_smoke(self) -> None:
        await self._require_command("file")
        output = await _call(
            _tool_by_name(self._toolset, "file"),
            ("filesystem/visible.txt",),
            brief=True,
        )

        _assert_completed(self, output, "file")
        self.assertIn("text", output.lower())

    async def test_find_smoke(self) -> None:
        await self._require_command("find")
        output = await _call(
            _tool_by_name(self._toolset, "find"),
            ("filesystem",),
            entry_type="file",
            name="visible.txt",
            max_depth=1,
        )

        _assert_completed(self, output, "find")
        self.assertIn("filesystem/visible.txt", output)
        self.assertNotIn(".hidden.txt", output)

    async def test_find_min_depth_smoke(self) -> None:
        await self._require_command("find")
        output = await _execute_request(
            self,
            self._policy,
            self._executor,
            ShellCommandRequest(
                tool_name="shell.find",
                command="find",
                options={
                    "entry_type": "file",
                    "max_depth": 1,
                    "min_depth": 1,
                    "name": "visible.txt",
                },
                paths=(
                    PathOperand(
                        name="input",
                        path="filesystem",
                        kind="directory",
                        access="read",
                    ),
                ),
                cwd=None,
            ),
        )

        self.assertIn("filesystem/visible.txt", output)
        self.assertNotIn(".hidden.txt", output)

    async def test_find_expression_token_root_smoke(self) -> None:
        await self._require_command("find")
        for root, expected_bytes in (("!", 16), ("(", 16)):
            with self.subTest(root=root):
                output = await _call(
                    _tool_by_name(self._toolset, "find"),
                    (root,),
                    cwd="find_roots",
                    entry_type="file",
                    name="visible.txt",
                    max_depth=1,
                )

                _assert_completed(self, output, "find")
                self.assertIn(f"./{root}", output)
                self.assertIn("stdout_bytes: " + str(expected_bytes), output)

    async def test_wc_smoke(self) -> None:
        await self._require_command("wc")
        output = await _call(
            _tool_by_name(self._toolset, "wc"),
            ("filesystem/visible.txt",),
            lines=True,
            words=True,
            count_bytes=True,
        )

        _assert_completed(self, output, "wc")
        self.assertIn("3", output)
        self.assertIn("18", output)

    async def test_awk_smoke(self) -> None:
        await self._require_command("awk")
        output = await _call(
            _tool_by_name(self._toolset, "awk"),
            ("filters/table.csv",),
            fields=(1,),
            field_separator="comma",
            output_separator=",",
            start_line=2,
            end_line=2,
        )

        _assert_completed(self, output, "awk")
        self.assertIn("alpha", output)
        self.assertNotIn("beta", output)

    async def test_awk_default_output_separator_smoke(self) -> None:
        await self._require_command("awk")
        output = await _call(
            _tool_by_name(self._toolset, "awk"),
            ("filters/table.tsv",),
            fields=(1, 2),
            field_separator="tab",
            start_line=2,
            end_line=2,
        )

        _assert_completed(self, output, "awk")
        self.assertIn("alpha 1", output)
        self.assertNotIn("beta", output)

    async def test_sed_smoke(self) -> None:
        await self._require_command("sed")
        output = await _call(
            _tool_by_name(self._toolset, "sed"),
            ("filters/lines.txt",),
            line_ranges=("2,2",),
        )

        _assert_completed(self, output, "sed")
        self.assertIn("second line", output)
        self.assertNotIn("first line", output)

    async def test_sed_line_window_smoke(self) -> None:
        await self._require_command("sed")
        output = await _call(
            _tool_by_name(self._toolset, "sed"),
            ("filters/lines.txt",),
            start_line=2,
            end_line=2,
        )

        _assert_completed(self, output, "sed")
        self.assertIn("second line", output)
        self.assertNotIn("first line", output)
        self.assertNotIn("third line", output)

    async def test_jq_smoke(self) -> None:
        await self._require_command("jq")
        output = await _call(
            _tool_by_name(self._toolset, "jq"),
            ".items[0].name",
            ("json/valid.json",),
            raw_output=True,
        )

        _assert_completed(self, output, "jq")
        self.assertIn("alpha", output)

    async def test_pdfinfo_smoke(self) -> None:
        await self._require_command("pdfinfo")
        output = await _call(
            _tool_by_name(self._toolset, "pdfinfo"),
            "media/small.pdf",
        )

        _assert_completed(self, output, "pdfinfo")
        self.assertIn("stdout_bytes:", output)

    async def test_pdftotext_smoke(self) -> None:
        await self._require_command("pdftotext")
        output = await _call(
            _tool_by_name(self._toolset, "pdftotext"),
            "media/small.pdf",
            first_page=1,
            last_page=1,
        )

        _assert_completed(self, output, "pdftotext")
        self.assertIn("stdout_bytes:", output)

    async def test_pdftoppm_smoke(self) -> None:
        await self._require_command("pdftoppm")
        output = await _call(
            _tool_by_name(self._toolset, "pdftoppm"),
            "media/small.pdf",
            first_page=1,
            last_page=1,
            dpi=72,
        )

        _assert_completed(self, output, "pdftoppm")
        self.assertIn("generated_files:", output)
        self.assertIn("media_type: image/png", output)

    async def test_tesseract_smoke(self) -> None:
        await self._require_command("tesseract")
        output = await _call(
            _tool_by_name(self._toolset, "tesseract"),
            "ocr/small.pgm",
            languages=("eng",),
            psm=6,
        )

        _assert_completed(self, output, "tesseract")
        self.assertIn("stdout_bytes:", output)

    async def _require_command(self, command_id: str) -> None:
        command = SHELL_COMMAND_DEFINITIONS[command_id]
        executable = await self._resolver.resolve(command)
        if executable is None:
            self.skipTest(
                f"{command.dependency_group} command unavailable: "
                f"{command.executable_name}"
            )


async def _call(tool: Tool, *args: object, **kwargs: object) -> str:
    return await tool(*args, **kwargs, context=ToolCallContext())


async def _execute_request(
    test_case: IsolatedAsyncioTestCase,
    policy: ExecutionPolicy,
    executor: LocalCommandExecutor,
    request: ShellCommandRequest,
) -> str:
    spec = await policy.normalize(request)
    result = await executor.execute(spec)
    test_case.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
    test_case.assertEqual(result.exit_code, 0)
    return result.stdout


def _shell_request(
    command: str,
    options: dict[str, object],
    *,
    path: str,
) -> ShellCommandRequest:
    return ShellCommandRequest(
        tool_name=f"shell.{command}",
        command=command,
        options=options,
        paths=(
            PathOperand(
                name="input",
                path=path,
                kind="text_file",
                access="read",
            ),
        ),
        cwd=None,
    )


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> Tool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert isinstance(tool, Tool), "shell command must be a tool"
            return tool
    raise AssertionError(f"missing shell tool {command_id}")


def _assert_completed(
    test_case: IsolatedAsyncioTestCase,
    output: str,
    command_id: str,
) -> None:
    test_case.assertIn(f"tool: shell.{command_id}", output)
    test_case.assertIn(
        f"status: {ShellExecutionStatus.COMPLETED}",
        output,
    )
    test_case.assertIn("exit_code: 0", output)


async def _skip_if_process_table_unavailable(
    test_case: IsolatedAsyncioTestCase,
    resolver: TrustedExecutableResolver,
) -> None:
    executable = await resolver.resolve(SHELL_COMMAND_DEFINITIONS["pgrep"])
    assert (
        executable is not None
    ), "pgrep availability preflight requires pgrep"
    process = await create_subprocess_exec(
        executable,
        "-f",
        "-P",
        str(getpid()),
        "--",
        "avalan-pgrep-process-table-availability-probe",
        stdout=PIPE,
        stderr=PIPE,
    )
    _, stderr = await process.communicate()
    unavailable_markers = (
        "cannot get process list",
        "process table",
        "sysmond service not found",
    )
    if process.returncode == 3 and any(
        marker in stderr.decode("utf-8", errors="replace").lower()
        for marker in unavailable_markers
    ):
        test_case.skipTest("pgrep process table is unavailable")


async def _skip_if_ps_process_table_unavailable(
    test_case: IsolatedAsyncioTestCase,
    resolver: TrustedExecutableResolver,
) -> None:
    executable = await resolver.resolve(SHELL_COMMAND_DEFINITIONS["ps"])
    assert executable is not None, "ps availability preflight requires ps"
    try:
        process = await create_subprocess_exec(
            executable,
            "-p",
            str(getpid()),
            "-o",
            "pid=",
            "-o",
            "ppid=",
            "-o",
            "state=",
            "-o",
            "etime=",
            "-o",
            "comm=",
            stdout=PIPE,
            stderr=PIPE,
        )
    except PermissionError as error:
        if error.errno in {EACCES, EPERM}:
            test_case.skipTest("ps process table is unavailable")
        raise
    stdout, stderr = await process.communicate()
    diagnostic = stderr.decode("utf-8", errors="replace").lower()
    unavailable_markers = (
        "cannot get process list",
        "operation not permitted",
        "permission denied",
        "process table",
        "sysmond service not found",
    )
    if process.returncode != 0 and any(
        marker in diagnostic for marker in unavailable_markers
    ):
        test_case.skipTest("ps process table is unavailable")
    test_case.assertEqual(process.returncode, 0)
    test_case.assertIn(str(getpid()).encode("ascii"), stdout)


async def _skip_if_lsof_process_table_unavailable(
    test_case: IsolatedAsyncioTestCase,
    resolver: TrustedExecutableResolver,
) -> None:
    executable = await resolver.resolve(SHELL_COMMAND_DEFINITIONS["lsof"])
    assert executable is not None, "lsof availability preflight requires lsof"
    try:
        process = await create_subprocess_exec(
            executable,
            "-n",
            "-P",
            "-w",
            "-b",
            "-a",
            "-p",
            str(getpid()),
            "-F0pftaP",
            stdout=PIPE,
            stderr=PIPE,
        )
    except PermissionError as error:
        if error.errno in {EACCES, EPERM}:
            test_case.skipTest("lsof process table is unavailable")
        raise
    stdout, stderr = await process.communicate()
    diagnostic = stderr.decode("utf-8", errors="replace").lower()
    unavailable_markers = (
        "cannot get process list",
        "operation not permitted",
        "permission denied",
        "process table",
        "security level",
    )
    if process.returncode != 0 and any(
        marker in diagnostic for marker in unavailable_markers
    ):
        test_case.skipTest("lsof process table is unavailable")
    test_case.assertEqual(process.returncode, 0)
    test_case.assertIn(f"p{getpid()}".encode("ascii"), stdout)


def _trusted_search_paths() -> tuple[str, ...]:
    configured = environ.get(SMOKE_PATHS_ENV_NAME)
    if configured:
        return tuple(
            path
            for path in configured.split(":")
            if path and Path(path).is_absolute()
        )
    return TRUSTED_SEARCH_PATHS


if __name__ == "__main__":
    main()
