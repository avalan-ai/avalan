from os import environ
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, main, skipUnless

from avalan.entities import ToolCallContext
from avalan.tool import Tool
from avalan.tool.shell import (
    SHELL_COMMAND_DEFINITIONS,
    ExecutionPolicy,
    LocalCommandExecutor,
    ShellExecutionStatus,
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
    "/bin",
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
            executable_search_paths=self._search_paths,
            max_inline_output_file_bytes=256,
        )
        self._toolset = ShellToolSet(
            settings=settings,
            policy=ExecutionPolicy(
                settings=settings,
                resolver=self._resolver,
            ),
            executor=LocalCommandExecutor(),
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

    async def test_tail_smoke(self) -> None:
        await self._require_command("tail")
        output = await _call(
            _tool_by_name(self._toolset, "tail"),
            "filesystem/visible.txt",
            lines=1,
        )

        _assert_completed(self, output, "tail")
        self.assertIn("needle", output)

    async def test_ls_smoke(self) -> None:
        await self._require_command("ls")
        output = await _call(
            _tool_by_name(self._toolset, "ls"),
            "filesystem",
        )

        _assert_completed(self, output, "ls")
        self.assertIn("visible.txt", output)
        self.assertNotIn(".hidden.txt", output)

    async def test_cat_smoke(self) -> None:
        await self._require_command("cat")
        output = await _call(
            _tool_by_name(self._toolset, "cat"),
            "filesystem/visible.txt",
        )

        _assert_completed(self, output, "cat")
        self.assertIn("alpha", output)
        self.assertIn("needle", output)

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
