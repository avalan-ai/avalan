from ..entities import (
    Message,
    MessageContent,
    MessageContentFile,
    MessageContentText,
    MessageRole,
)

from base64 import b64encode
from mimetypes import guess_type
from pathlib import Path

_MIME_TYPES_BY_SUFFIX = {
    ".md": "text/markdown",
}


def input_files(
    input_text: str | None, file_paths: list[str] | None
) -> Message | str | None:
    """Build text-generation input from text and local file paths."""
    if not file_paths:
        return input_text

    content: list[MessageContent] = []
    if input_text:
        content.append(MessageContentText(type="text", text=input_text))

    for file_path in file_paths:
        path = Path(file_path)
        assert path.is_file(), f"Input file not found: {file_path}"
        mime_type = (
            _MIME_TYPES_BY_SUFFIX.get(path.suffix.lower())
            or guess_type(path.name)[0]
            or "application/octet-stream"
        )
        content.append(
            MessageContentFile(
                type="file",
                file={
                    "file_data": b64encode(path.read_bytes()).decode("ascii"),
                    "filename": path.name,
                    "mime_type": mime_type,
                },
            )
        )

    return Message(role=MessageRole.USER, content=content)
