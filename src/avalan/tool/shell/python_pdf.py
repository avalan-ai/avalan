from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from importlib import import_module
from json import dumps
from pathlib import Path
from sys import argv as sys_argv
from sys import stderr
from textwrap import wrap
from typing import Any, Protocol, cast

_UNAVAILABLE_EXIT_CODE = 127
_DEFAULT_TEXT_WRAP_WIDTH = 92
_PAGE_MARGIN_POINTS = 72
_LINE_SPACING_POINTS = 14
_REPORTLAB_FONT_NAME = "Helvetica"
_REPORTLAB_FONT_SIZE = 10


class _UnavailableLibrary(Exception):
    pass


class _ReportLabTextObject(Protocol):
    setFont: Callable[[str, int], None]
    textLine: Callable[[str], None]


class _ReportLabCanvas(Protocol):
    beginText: Callable[[float, float], _ReportLabTextObject]
    drawText: Callable[[_ReportLabTextObject], None]
    save: Callable[[], None]
    setTitle: Callable[[str], None]
    showPage: Callable[[], None]


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys_argv[1:] if argv is None else argv)
    parser = _parser()
    namespace = parser.parse_args(arguments)
    try:
        match namespace.command:
            case "pypdf":
                _run_pypdf(namespace)
            case "pdfplumber":
                _run_pdfplumber(namespace)
            case "reportlab":
                _run_reportlab(namespace)
            case _:
                parser.error("unsupported command")
    except _UnavailableLibrary as error:
        print(str(error), file=stderr)
        return _UNAVAILABLE_EXIT_CODE
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=stderr)
        return 1
    return 0


def _parser() -> ArgumentParser:
    parser = ArgumentParser(prog="avalan-shell-python-pdf")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pypdf = subparsers.add_parser("pypdf")
    pypdf.add_argument("--mode", choices=("metadata", "text"), required=True)
    pypdf.add_argument("--first-page", type=int)
    pypdf.add_argument("--last-page", type=int)
    pypdf.add_argument("path")

    pdfplumber = subparsers.add_parser("pdfplumber")
    pdfplumber.add_argument(
        "--mode",
        choices=("text", "tables"),
        required=True,
    )
    pdfplumber.add_argument("--first-page", type=int, required=True)
    pdfplumber.add_argument("--last-page", type=int, required=True)
    pdfplumber.add_argument("--layout", action="store_true")
    pdfplumber.add_argument("path")

    reportlab = subparsers.add_parser("reportlab")
    reportlab.add_argument(
        "--page-size",
        choices=("letter", "a4"),
        required=True,
    )
    reportlab.add_argument("--title", required=True)
    reportlab.add_argument("--text", required=True)
    reportlab.add_argument("--output", required=True)
    return parser


def _optional_module(module_name: str) -> Any:
    try:
        return import_module(module_name)
    except ImportError as error:
        raise _UnavailableLibrary(
            f"Python package is unavailable: {module_name}"
        ) from error


def _run_pypdf(arguments: Namespace) -> None:
    pypdf = _optional_module("pypdf")
    reader_class = getattr(pypdf, "PdfReader")
    reader = reader_class(arguments.path, strict=False)
    if arguments.mode == "metadata":
        print(dumps(_pypdf_metadata(reader), sort_keys=True))
        return
    first_page = _required_int(arguments.first_page, "first_page")
    last_page = _required_int(arguments.last_page, "last_page")
    print(_pypdf_text(reader, first_page=first_page, last_page=last_page))


def _pypdf_metadata(reader: object) -> dict[str, object]:
    encrypted = bool(getattr(reader, "is_encrypted", False))
    metadata = getattr(reader, "metadata", None)
    values: dict[str, str] = {}
    if metadata is not None:
        items = getattr(metadata, "items", None)
        if callable(items):
            values = {
                str(key).lstrip("/"): str(value)
                for key, value in items()
                if value is not None
            }
    page_count = None if encrypted else len(getattr(reader, "pages"))
    return {
        "backend": "pypdf",
        "encrypted": encrypted,
        "page_count": page_count,
        "metadata": values,
    }


def _pypdf_text(
    reader: object,
    *,
    first_page: int,
    last_page: int,
) -> str:
    if bool(getattr(reader, "is_encrypted", False)):
        raise ValueError("encrypted PDFs are unsupported")
    pages = getattr(reader, "pages")
    page_count = len(pages)
    _validate_page_range(first_page, last_page, page_count)
    page_text = []
    for page_number in range(first_page, last_page + 1):
        page = pages[page_number - 1]
        extract_text = getattr(page, "extract_text")
        text = extract_text() or ""
        page_text.append(text)
    return "\n\f\n".join(page_text)


def _run_pdfplumber(arguments: Namespace) -> None:
    pdfplumber = _optional_module("pdfplumber")
    with pdfplumber.open(arguments.path) as pdf:
        page_count = len(pdf.pages)
        first_page = _required_int(arguments.first_page, "first_page")
        last_page = _required_int(arguments.last_page, "last_page")
        _validate_page_range(first_page, last_page, page_count)
        pages = pdf.pages[first_page - 1 : last_page]
        if arguments.mode == "tables":
            print(dumps(_pdfplumber_tables(pages), sort_keys=True))
            return
        print(_pdfplumber_text(pages, layout=arguments.layout))


def _pdfplumber_text(pages: list[object], *, layout: bool) -> str:
    page_text = []
    for page in pages:
        extract_text = getattr(page, "extract_text")
        text = extract_text(layout=layout) or ""
        page_text.append(text)
    return "\n\f\n".join(page_text)


def _pdfplumber_tables(pages: list[object]) -> dict[str, object]:
    page_tables = []
    for page in pages:
        page_number = getattr(page, "page_number")
        extract_tables = getattr(page, "extract_tables")
        page_tables.append(
            {
                "page": page_number,
                "tables": extract_tables(),
            }
        )
    return {"backend": "pdfplumber", "pages": page_tables}


def _run_reportlab(arguments: Namespace) -> None:
    canvas_module = _optional_module("reportlab.pdfgen.canvas")
    pagesizes_module = _optional_module("reportlab.lib.pagesizes")
    page_size = _reportlab_page_size(pagesizes_module, arguments.page_size)
    output_path = Path(arguments.output).with_suffix(".pdf")
    canvas_class = getattr(canvas_module, "Canvas")
    canvas = cast(
        _ReportLabCanvas,
        canvas_class(str(output_path), pagesize=page_size),
    )
    canvas.setTitle(arguments.title)
    _draw_text_pdf(canvas, arguments.text, page_size)
    canvas.save()
    print(
        dumps(
            {
                "backend": "reportlab",
                "output": f"{arguments.output}.pdf",
            },
            sort_keys=True,
        )
    )


def _reportlab_page_size(module: object, name: str) -> tuple[float, float]:
    match name:
        case "a4":
            value = getattr(module, "A4")
        case "letter":
            value = getattr(module, "letter")
        case _:
            raise ValueError("unsupported page size")
    width, height = value
    return float(width), float(height)


def _draw_text_pdf(
    canvas: _ReportLabCanvas,
    text: str,
    page_size: tuple[float, float],
) -> None:
    height = page_size[1]
    y_position = height - _PAGE_MARGIN_POINTS
    text_object = _new_text_object(canvas, y_position)
    for line in _wrapped_text_lines(text):
        if y_position < _PAGE_MARGIN_POINTS:
            canvas.drawText(text_object)
            canvas.showPage()
            y_position = height - _PAGE_MARGIN_POINTS
            text_object = _new_text_object(canvas, y_position)
        text_object.textLine(line)
        y_position -= _LINE_SPACING_POINTS
    canvas.drawText(text_object)


def _new_text_object(
    canvas: _ReportLabCanvas,
    y_position: float,
) -> _ReportLabTextObject:
    text_object = canvas.beginText(_PAGE_MARGIN_POINTS, y_position)
    text_object.setFont(_REPORTLAB_FONT_NAME, _REPORTLAB_FONT_SIZE)
    return text_object


def _wrapped_text_lines(text: str) -> tuple[str, ...]:
    lines: list[str] = []
    for source_line in text.splitlines() or [""]:
        wrapped = wrap(
            source_line,
            width=_DEFAULT_TEXT_WRAP_WIDTH,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        lines.extend(wrapped or [""])
    return tuple(lines)


def _required_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} is required")
    return value


def _validate_page_range(
    first_page: int,
    last_page: int,
    page_count: int,
) -> None:
    if first_page < 1 or last_page < first_page:
        raise ValueError("invalid page range")
    if last_page > page_count:
        raise ValueError("page range exceeds PDF page count")


if __name__ == "__main__":
    raise SystemExit(main())
