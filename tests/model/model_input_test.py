from base64 import b64encode
from pathlib import Path
from tempfile import NamedTemporaryFile

from avalan.entities import (
    Message,
    MessageContentFile,
    MessageContentText,
    MessageRole,
)
from avalan.model import input_files


def test_input_files_with_text_and_file() -> None:
    with NamedTemporaryFile(suffix=".pdf") as tmp:
        tmp.write(b"%PDF-1.7")
        tmp.flush()

        result = input_files("Summarize", [tmp.name])

    assert isinstance(result, Message)
    assert result.role == MessageRole.USER
    assert isinstance(result.content, list)
    assert result.content[0] == MessageContentText(
        type="text", text="Summarize"
    )
    assert result.content[1] == MessageContentFile(
        type="file",
        file={
            "file_data": b64encode(b"%PDF-1.7").decode("ascii"),
            "filename": Path(tmp.name).name,
            "mime_type": "application/pdf",
        },
    )


def test_input_files_without_text() -> None:
    with NamedTemporaryFile(suffix=".md") as tmp:
        tmp.write(b"content")
        tmp.flush()

        result = input_files(None, [tmp.name])

    assert isinstance(result, Message)
    assert isinstance(result.content, list)
    assert result.content == [
        MessageContentFile(
            type="file",
            file={
                "file_data": b64encode(b"content").decode("ascii"),
                "filename": Path(tmp.name).name,
                "mime_type": "text/markdown",
            },
        )
    ]


def test_input_files_missing_file() -> None:
    missing = Path("tests/__missing_sdk_input__.pdf").resolve()
    assert not missing.exists()

    try:
        input_files("Summarize", [str(missing)])
    except AssertionError as exc:
        assert str(exc) == f"Input file not found: {missing}"
    else:
        raise AssertionError("Expected missing file assertion")


def test_input_files_without_paths() -> None:
    assert input_files("Hello", None) == "Hello"
    assert input_files(None, []) is None
