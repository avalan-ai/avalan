from base64 import b64decode
from importlib.util import find_spec
from json import loads
from pathlib import Path
from sys import executable
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, main, skipUnless

from avalan.entities import ToolCallContext
from avalan.tool import Tool
from avalan.tool.shell import (
    ExecutionPolicy,
    ShellExecutionErrorCode,
    ShellExecutionStatus,
    ShellFormattedResult,
    ShellToolSet,
    ShellToolSettings,
)


class PythonPdfShellToolFakeE2ETest(IsolatedAsyncioTestCase):
    async def test_runner_exit_127_maps_to_command_unavailable(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "valid.pdf").write_bytes(_minimal_pdf_bytes())
            fake_python = root / "fake-python"
            _write_fake_executable(
                fake_python,
                "from sys import stderr\n"
                "print(\n"
                "    'Python package is unavailable: pypdf',\n"
                "    file=stderr,\n"
                ")\n"
                "raise SystemExit(127)\n",
            )
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
                executable_paths={"pypdf": str(fake_python)},
            )
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
            ).with_enabled_tools(["shell.pypdf"])
            tool = _tool_by_name(toolset, "pypdf")

            output = await tool("valid.pdf", context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertEqual(
            result.status,
            ShellExecutionStatus.COMMAND_UNAVAILABLE,
        )
        self.assertEqual(
            result.error_code,
            ShellExecutionErrorCode.COMMAND_UNAVAILABLE,
        )
        self.assertEqual(result.exit_code, 127)
        self.assertIn("pypdf", result.stderr)

    async def test_reportlab_and_pdfplumber_run_through_local_shell_tool(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "valid.pdf").write_bytes(_minimal_pdf_bytes())
            fake_python = root / "fake-python-pdf"
            _write_fake_executable(
                fake_python,
                "from json import dumps\n"
                "from pathlib import Path\n"
                "from sys import argv, stderr\n"
                "command = argv[4] if len(argv) > 4 else ''\n"
                "if command == 'reportlab':\n"
                "    output = argv[argv.index('--output') + 1]\n"
                "    Path(output).with_suffix('.pdf').write_bytes(\n"
                "        b'%PDF-1.4\\n%%EOF\\n'\n"
                "    )\n"
                "    print(\n"
                "        dumps(\n"
                "            {\n"
                "                'backend': 'reportlab',\n"
                "                'output': output + '.pdf',\n"
                "            },\n"
                "            sort_keys=True,\n"
                "        )\n"
                "    )\n"
                "elif command == 'pdfplumber':\n"
                "    print('fake pdf text')\n"
                "else:\n"
                "    print('unsupported fake command', file=stderr)\n"
                "    raise SystemExit(1)\n",
            )
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
                executable_paths={
                    "reportlab": str(fake_python),
                    "pdfplumber": str(fake_python),
                },
            )
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
            ).with_enabled_tools(["shell.reportlab", "shell.pdfplumber"])
            reportlab = _tool_by_name(toolset, "reportlab")
            pdfplumber = _tool_by_name(toolset, "pdfplumber")

            reportlab_output = await reportlab(
                "Generated body",
                title="Generated title",
                context=ToolCallContext(),
            )
            pdfplumber_output = await pdfplumber(
                "valid.pdf",
                context=ToolCallContext(),
            )

        self.assertIsInstance(reportlab_output, ShellFormattedResult)
        assert isinstance(reportlab_output, ShellFormattedResult)
        reportlab_result = reportlab_output.execution_result
        self.assertEqual(
            reportlab_result.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertEqual(len(reportlab_result.generated_files), 1)
        generated = reportlab_result.generated_files[0]
        self.assertEqual(generated.display_path, "GENERATED_PREFIX.pdf")
        self.assertEqual(generated.media_type, "application/pdf")
        self.assertEqual(
            loads(reportlab_result.stdout),
            {"backend": "reportlab", "output": "GENERATED_PREFIX.pdf"},
        )

        self.assertIsInstance(pdfplumber_output, ShellFormattedResult)
        assert isinstance(pdfplumber_output, ShellFormattedResult)
        pdfplumber_result = pdfplumber_output.execution_result
        self.assertEqual(
            pdfplumber_result.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertEqual(pdfplumber_result.stdout_media_type, "text/plain")
        self.assertEqual(pdfplumber_result.stdout, "fake pdf text\n")


class PythonPdfShellToolE2ETest(IsolatedAsyncioTestCase):
    @skipUnless(find_spec("pypdf") is not None, "pypdf is unavailable")
    async def test_pypdf_metadata_runs_through_local_shell_tool(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "valid.pdf").write_bytes(_minimal_pdf_bytes())
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
                executable_paths={"pypdf": executable},
            )
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
            ).with_enabled_tools(["shell.pypdf"])
            tool = _tool_by_name(toolset, "pypdf")

            output = await tool("valid.pdf", context=ToolCallContext())

        self.assertIsInstance(output, ShellFormattedResult)
        assert isinstance(output, ShellFormattedResult)
        result = output.execution_result
        self.assertEqual(result.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(result.command, "pypdf")
        self.assertEqual(result.argv[:3], ("python3", "-I", "-m"))
        self.assertEqual(result.stdout_media_type, "application/json")
        payload = loads(result.stdout)
        self.assertEqual(payload["backend"], "pypdf")
        self.assertEqual(payload["page_count"], 1)

    @skipUnless(
        find_spec("reportlab") is not None
        and find_spec("pdfplumber") is not None,
        "reportlab or pdfplumber is unavailable",
    )
    async def test_reportlab_pdfplumber_round_trip_local_shell_tools(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
                executable_paths={
                    "reportlab": executable,
                    "pdfplumber": executable,
                },
            )
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
            ).with_enabled_tools(["shell.reportlab", "shell.pdfplumber"])
            reportlab = _tool_by_name(toolset, "reportlab")
            pdfplumber = _tool_by_name(toolset, "pdfplumber")

            generated = await reportlab(
                "Deterministic reportlab e2e text",
                title="ReportLab E2E",
                context=ToolCallContext(),
            )

            self.assertIsInstance(generated, ShellFormattedResult)
            assert isinstance(generated, ShellFormattedResult)
            reportlab_result = generated.execution_result
            self.assertEqual(
                reportlab_result.status,
                ShellExecutionStatus.COMPLETED,
            )
            self.assertEqual(len(reportlab_result.generated_files), 1)
            generated_file = reportlab_result.generated_files[0]
            self.assertEqual(generated_file.media_type, "application/pdf")
            assert generated_file.content_base64 is not None
            (root / "generated.pdf").write_bytes(
                b64decode(generated_file.content_base64)
            )

            extracted = await pdfplumber(
                "generated.pdf",
                first_page=1,
                last_page=1,
                context=ToolCallContext(),
            )

        self.assertIsInstance(extracted, ShellFormattedResult)
        assert isinstance(extracted, ShellFormattedResult)
        pdfplumber_result = extracted.execution_result
        self.assertEqual(
            pdfplumber_result.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertIn(
            "Deterministic reportlab e2e text",
            pdfplumber_result.stdout,
        )

    @skipUnless(
        find_spec("reportlab") is not None and find_spec("pypdf") is not None,
        "reportlab or pypdf is unavailable",
    )
    async def test_reportlab_pypdf_text_round_trip_local_shell_tools(
        self,
    ) -> None:
        expected_text = "Deterministic pypdf e2e text"
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
                executable_paths={
                    "reportlab": executable,
                    "pypdf": executable,
                },
            )
            toolset = ShellToolSet(
                settings=settings,
                policy=ExecutionPolicy(settings=settings),
            ).with_enabled_tools(["shell.reportlab", "shell.pypdf"])
            reportlab = _tool_by_name(toolset, "reportlab")
            pypdf = _tool_by_name(toolset, "pypdf")

            generated = await reportlab(
                expected_text,
                title="PyPDF Text E2E",
                context=ToolCallContext(),
            )

            self.assertIsInstance(generated, ShellFormattedResult)
            assert isinstance(generated, ShellFormattedResult)
            reportlab_result = generated.execution_result
            self.assertEqual(
                reportlab_result.status,
                ShellExecutionStatus.COMPLETED,
            )
            self.assertEqual(len(reportlab_result.generated_files), 1)
            generated_file = reportlab_result.generated_files[0]
            assert generated_file.content_base64 is not None
            (root / "generated.pdf").write_bytes(
                b64decode(generated_file.content_base64)
            )

            extracted = await pypdf(
                "generated.pdf",
                mode="text",
                first_page=1,
                last_page=1,
                context=ToolCallContext(),
            )

        self.assertIsInstance(extracted, ShellFormattedResult)
        assert isinstance(extracted, ShellFormattedResult)
        pypdf_result = extracted.execution_result
        self.assertEqual(
            pypdf_result.status,
            ShellExecutionStatus.COMPLETED,
        )
        self.assertNotEqual(pypdf_result.stdout.strip(), "")
        self.assertIn(expected_text, pypdf_result.stdout)


def _minimal_pdf_bytes() -> bytes:
    objects = (
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] "
            b"/Resources << >> >>\n"
            b"endobj\n"
        ),
    )
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(body))
        body.extend(item)
    xref_offset = len(body)
    body.extend(b"xref\n0 4\n")
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n")
    body.extend(f"{xref_offset}\n%%EOF\n".encode())
    return bytes(body)


def _write_fake_executable(path: Path, body: str) -> None:
    path.write_text(f"#!{executable}\n{body}", encoding="utf-8")
    path.chmod(0o700)


def _tool_by_name(toolset: ShellToolSet, command_id: str) -> Tool:
    for tool in toolset.tools:
        if getattr(tool, "__name__", "") == command_id:
            assert isinstance(tool, Tool), "shell command must be a tool"
            return tool
    raise AssertionError(f"missing shell tool {command_id}")


if __name__ == "__main__":
    main()
