from contextlib import redirect_stderr, redirect_stdout
from importlib.util import find_spec
from io import StringIO
from json import loads
from pathlib import Path
from runpy import run_module
from sys import executable as python_executable
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase, TestCase, main, skipIf
from unittest.mock import patch
from warnings import catch_warnings, simplefilter

from avalan.tool.shell.entities import (
    ExecutionResult,
    PathOperand,
    ShellCommandRequest,
    ShellExecutionStatus,
    ShellOutputKind,
)
from avalan.tool.shell.executor import LocalCommandExecutor
from avalan.tool.shell.policy import ExecutionPolicy
from avalan.tool.shell.python_pdf import (
    _draw_text_pdf,
    _reportlab_page_size,
    _required_int,
    _validate_page_range,
)
from avalan.tool.shell.python_pdf import (
    main as python_pdf_main,
)
from avalan.tool.shell.registry import ShellCommandDefinition
from avalan.tool.shell.settings import ShellToolSettings


class PythonPdfRunnerTest(TestCase):
    def test_main_reports_unsupported_command_from_parser_fallback(
        self,
    ) -> None:
        parser = _FakeParser(command="unsupported")

        with patch(
            "avalan.tool.shell.python_pdf._parser",
            return_value=parser,
        ):
            code, stdout, stderr = _run_main("unsupported")

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertEqual(parser.arguments, ("unsupported",))
        self.assertEqual(parser.errors, ("unsupported command",))
        self.assertIn("RuntimeError: unsupported command", stderr)

    def test_pypdf_main_outputs_metadata_and_text_with_stub(self) -> None:
        module = _FakePypdfModule(
            _FakePdfReader(
                pages=(
                    _FakePypdfPage("first page"),
                    _FakePypdfPage("second page"),
                ),
                metadata={"/Title": "Stub PDF", "/Empty": None},
            )
        )

        with patch(
            "avalan.tool.shell.python_pdf.import_module",
            return_value=module,
        ):
            metadata_code, metadata_stdout, metadata_stderr = _run_main(
                "pypdf",
                "--mode",
                "metadata",
                "input.pdf",
            )
            text_code, text_stdout, text_stderr = _run_main(
                "pypdf",
                "--mode",
                "text",
                "--first-page",
                "1",
                "--last-page",
                "2",
                "input.pdf",
            )

        self.assertEqual(metadata_code, 0)
        self.assertEqual(metadata_stderr, "")
        self.assertEqual(
            loads(metadata_stdout),
            {
                "backend": "pypdf",
                "encrypted": False,
                "metadata": {"Title": "Stub PDF"},
                "page_count": 2,
            },
        )
        self.assertEqual(text_code, 0)
        self.assertEqual(text_stderr, "")
        self.assertEqual(text_stdout, "first page\n\f\nsecond page\n")
        self.assertEqual(tuple(module.paths), ("input.pdf", "input.pdf"))
        self.assertEqual(tuple(module.strict_values), (False, False))

    def test_pypdf_main_reports_invalid_page_range(self) -> None:
        module = _FakePypdfModule(
            _FakePdfReader(pages=(_FakePypdfPage("only page"),))
        )

        with patch(
            "avalan.tool.shell.python_pdf.import_module",
            return_value=module,
        ):
            code, stdout, stderr = _run_main(
                "pypdf",
                "--mode",
                "text",
                "--first-page",
                "1",
                "--last-page",
                "2",
                "input.pdf",
            )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ValueError: page range exceeds PDF page count", stderr)

    def test_pypdf_main_rejects_encrypted_text(self) -> None:
        module = _FakePypdfModule(
            _FakePdfReader(
                pages=(_FakePypdfPage("secret"),),
                encrypted=True,
            )
        )

        with patch(
            "avalan.tool.shell.python_pdf.import_module",
            return_value=module,
        ):
            code, stdout, stderr = _run_main(
                "pypdf",
                "--mode",
                "text",
                "--first-page",
                "1",
                "--last-page",
                "1",
                "input.pdf",
            )

        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("ValueError: encrypted PDFs are unsupported", stderr)

    def test_pdfplumber_main_outputs_text_and_tables_with_stub(self) -> None:
        module = _FakePdfPlumberModule(
            (
                _FakePdfPlumberPage(1, "alpha"),
                _FakePdfPlumberPage(2, "beta"),
            )
        )

        with patch(
            "avalan.tool.shell.python_pdf.import_module",
            return_value=module,
        ):
            text_code, text_stdout, text_stderr = _run_main(
                "pdfplumber",
                "--mode",
                "text",
                "--first-page",
                "1",
                "--last-page",
                "2",
                "--layout",
                "input.pdf",
            )
            tables_code, tables_stdout, tables_stderr = _run_main(
                "pdfplumber",
                "--mode",
                "tables",
                "--first-page",
                "2",
                "--last-page",
                "2",
                "input.pdf",
            )

        self.assertEqual(text_code, 0)
        self.assertEqual(text_stderr, "")
        self.assertEqual(
            text_stdout,
            "alpha layout=True\n\f\nbeta layout=True\n",
        )
        self.assertEqual(tables_code, 0)
        self.assertEqual(tables_stderr, "")
        self.assertEqual(
            loads(tables_stdout),
            {
                "backend": "pdfplumber",
                "pages": [
                    {
                        "page": 2,
                        "tables": [[["beta", "cell"]]],
                    }
                ],
            },
        )
        self.assertEqual(tuple(module.paths), ("input.pdf", "input.pdf"))

    def test_reportlab_main_writes_pdf_output_with_stub(self) -> None:
        canvas_module = _FakeCanvasModule()
        modules = {
            "reportlab.pdfgen.canvas": canvas_module,
            "reportlab.lib.pagesizes": _FakePageSizesModule(),
        }

        with TemporaryDirectory() as temporary_directory:
            output_prefix = Path(temporary_directory) / "document"

            with patch(
                "avalan.tool.shell.python_pdf.import_module",
                side_effect=lambda name: modules[name],
            ):
                code, stdout, stderr = _run_main(
                    "reportlab",
                    "--page-size",
                    "a4",
                    "--title=--Stub Title",
                    "--text=- line one\nline two",
                    "--output",
                    str(output_prefix),
                )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(
            loads(stdout),
            {
                "backend": "reportlab",
                "output": f"{output_prefix}.pdf",
            },
        )
        self.assertEqual(
            canvas_module.instances[0].path,
            f"{output_prefix}.pdf",
        )
        self.assertEqual(canvas_module.instances[0].pagesize, (595.0, 842.0))
        self.assertEqual(canvas_module.instances[0].title, "--Stub Title")
        self.assertIn("- line one", canvas_module.instances[0].lines)
        self.assertIn("line two", canvas_module.instances[0].lines)
        self.assertTrue(canvas_module.instances[0].saved)

    def test_unavailable_library_returns_stable_exit_code(self) -> None:
        with patch(
            "avalan.tool.shell.python_pdf.import_module",
            side_effect=ImportError("missing"),
        ):
            code, stdout, stderr = _run_main(
                "pypdf",
                "--mode",
                "metadata",
                "input.pdf",
            )

        self.assertEqual(code, 127)
        self.assertEqual(stdout, "")
        self.assertIn("Python package is unavailable: pypdf", stderr)

    def test_reportlab_page_size_branches_are_validated(self) -> None:
        module = _FakePageSizesModule()

        self.assertEqual(
            _reportlab_page_size(module, "letter"),
            (612.0, 792.0),
        )
        with self.assertRaisesRegex(ValueError, "unsupported page size"):
            _reportlab_page_size(module, "legal")

    def test_reportlab_draw_text_starts_new_pages_when_needed(self) -> None:
        canvas = _FakeCanvas("out.pdf", pagesize=(100.0, 80.0))

        _draw_text_pdf(canvas, "one\ntwo", (100.0, 80.0))

        self.assertEqual(canvas.show_page_count, 2)
        self.assertEqual(canvas.lines, ["one", "two"])

    def test_runner_validation_helpers_reject_invalid_values(self) -> None:
        for value in (None, True, "1"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    "first_page is required",
                ):
                    _required_int(value, "first_page")

        for first_page, last_page in ((0, 1), (2, 1)):
            with self.subTest(first_page=first_page, last_page=last_page):
                with self.assertRaisesRegex(ValueError, "invalid page range"):
                    _validate_page_range(first_page, last_page, 2)

    def test_module_entrypoint_exits_with_main_status(self) -> None:
        stdout = StringIO()
        stderr = StringIO()

        with patch(
            "sys.argv",
            [
                "python_pdf",
                "pypdf",
                "--mode",
                "metadata",
                "missing.pdf",
            ],
        ):
            with patch("sys.stdout", stdout), patch("sys.stderr", stderr):
                with catch_warnings():
                    simplefilter("ignore", RuntimeWarning)
                    with self.assertRaises(SystemExit) as context:
                        run_module(
                            "avalan.tool.shell.python_pdf",
                            run_name="__main__",
                        )

        self.assertIn(context.exception.code, {1, 127})


class PythonPdfExecutorE2ETest(IsolatedAsyncioTestCase):
    @skipIf(find_spec("pypdf") is None, "pypdf is unavailable")
    async def test_pypdf_runs_through_policy_and_local_executor(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdf_path = root / "valid.pdf"
            pdf_path.write_bytes(_minimal_pdf_bytes("Executor PDF"))
            settings = ShellToolSettings(
                workspace_root=str(root),
                allow_media_tools=True,
            )
            policy = ExecutionPolicy(
                settings=settings,
                resolver=_PythonExecutableResolver(),
            )
            metadata = await _execute_pypdf(
                policy,
                settings,
                options={"mode": "metadata"},
            )
            text = await _execute_pypdf(
                policy,
                settings,
                options={
                    "mode": "text",
                    "first_page": 1,
                    "last_page": 1,
                },
            )

        self.assertEqual(metadata.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(metadata.stdout_media_type, "application/json")
        self.assertEqual(metadata.output_kind, ShellOutputKind.JSON)
        self.assertEqual(
            loads(metadata.stdout),
            {
                "backend": "pypdf",
                "encrypted": False,
                "metadata": {
                    "Producer": "avalan tests",
                    "Title": "Executor PDF",
                },
                "page_count": 1,
            },
        )
        self.assertEqual(text.status, ShellExecutionStatus.COMPLETED)
        self.assertEqual(text.stdout_media_type, "text/plain")
        self.assertEqual(text.output_kind, ShellOutputKind.TEXT)
        self.assertEqual(text.stdout, "\n")


async def _execute_pypdf(
    policy: ExecutionPolicy,
    settings: ShellToolSettings,
    *,
    options: dict[str, object],
) -> ExecutionResult:
    spec = await policy.normalize(
        ShellCommandRequest(
            tool_name="shell.pypdf",
            command="pypdf",
            options=options,
            paths=(
                PathOperand(
                    name="input",
                    path="valid.pdf",
                    kind="pdf_file",
                    access="read",
                ),
            ),
            cwd=None,
        )
    )
    return await LocalCommandExecutor(settings=settings).execute(spec)


def _run_main(*arguments: str) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        with patch("avalan.tool.shell.python_pdf.stderr", stderr):
            code = python_pdf_main(list(arguments))
    return code, stdout.getvalue(), stderr.getvalue()


def _minimal_pdf_bytes(title: str) -> bytes:
    objects = (
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            b"3 0 obj\n"
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] "
            b"/Resources << >> >>\n"
            b"endobj\n"
        ),
        (
            b"4 0 obj\n"
            + f"<< /Title ({title}) /Producer (avalan tests) >>\n".encode()
            + b"endobj\n"
        ),
    )
    body = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(body))
        body.extend(item)
    xref_offset = len(body)
    body.extend(b"xref\n0 5\n")
    body.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        body.extend(f"{offset:010d} 00000 n \n".encode())
    body.extend(b"trailer\n<< /Size 5 /Root 1 0 R /Info 4 0 R >>\nstartxref\n")
    body.extend(f"{xref_offset}\n%%EOF\n".encode())
    return bytes(body)


class _PythonExecutableResolver:
    async def resolve(self, command: ShellCommandDefinition) -> str:
        return python_executable


class _FakeParser:
    def __init__(self, *, command: str) -> None:
        self._command = command
        self.arguments: tuple[str, ...] = ()
        self._errors: list[str] = []

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    def parse_args(self, arguments: list[str]) -> "_FakeArguments":
        self.arguments = tuple(arguments)
        return _FakeArguments(self._command)

    def error(self, message: str) -> None:
        self._errors.append(message)
        raise RuntimeError(message)


class _FakeArguments:
    def __init__(self, command: str) -> None:
        self.command = command


class _FakePypdfModule:
    def __init__(self, reader: "_FakePdfReader") -> None:
        self._reader = reader
        self.paths: list[str] = []
        self.strict_values: list[bool] = []

    def PdfReader(self, path: str, *, strict: bool) -> "_FakePdfReader":
        self.paths.append(path)
        self.strict_values.append(strict)
        return self._reader


class _FakePdfReader:
    def __init__(
        self,
        *,
        pages: tuple["_FakePypdfPage", ...],
        metadata: dict[str, str | None] | None = None,
        encrypted: bool = False,
    ) -> None:
        self.pages = pages
        self.metadata = {} if metadata is None else metadata
        self.is_encrypted = encrypted


class _FakePypdfPage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfPlumberModule:
    def __init__(self, pages: tuple["_FakePdfPlumberPage", ...]) -> None:
        self._pages = pages
        self.paths: list[str] = []

    def open(self, path: str) -> "_FakePdf":
        self.paths.append(path)
        return _FakePdf(self._pages)


class _FakePdf:
    def __init__(self, pages: tuple["_FakePdfPlumberPage", ...]) -> None:
        self.pages = pages

    def __enter__(self) -> "_FakePdf":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None


class _FakePdfPlumberPage:
    def __init__(self, page_number: int, text: str) -> None:
        self.page_number = page_number
        self._text = text

    def extract_text(self, *, layout: bool) -> str:
        return f"{self._text} layout={layout}"

    def extract_tables(self) -> list[list[list[str]]]:
        return [[[self._text, "cell"]]]


class _FakeCanvasModule:
    def __init__(self) -> None:
        self.instances: list[_FakeCanvas] = []

    def Canvas(
        self,
        path: str,
        *,
        pagesize: tuple[float, float],
    ) -> "_FakeCanvas":
        canvas = _FakeCanvas(path, pagesize=pagesize)
        self.instances.append(canvas)
        return canvas


class _FakeCanvas:
    def __init__(self, path: str, *, pagesize: tuple[float, float]) -> None:
        self.path = path
        self.pagesize = pagesize
        self.title = ""
        self.lines: list[str] = []
        self.show_page_count = 0
        self.saved = False

    def setTitle(self, title: str) -> None:
        self.title = title

    def beginText(self, x_position: float, y_position: float) -> "_FakeText":
        return _FakeText(self.lines)

    def drawText(self, text: "_FakeText") -> None:
        return None

    def showPage(self) -> None:
        self.show_page_count += 1

    def save(self) -> None:
        self.saved = True


class _FakeText:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def setFont(self, name: str, size: int) -> None:
        return None

    def textLine(self, text: str) -> None:
        self._lines.append(text)


class _FakePageSizesModule:
    A4 = (595, 842)
    letter = (612, 792)


if __name__ == "__main__":
    main()
