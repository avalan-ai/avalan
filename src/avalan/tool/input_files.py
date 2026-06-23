from ..entities import Input, Message, MessageContentFile

from collections.abc import Iterator, Mapping


def input_file_string(file: Mapping[str, object], *keys: str) -> str | None:
    """Return the first non-empty string from a file payload."""
    for key in keys:
        value = file.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                return stripped
    return None


def iter_input_file_content(
    input_value: Input | None,
) -> Iterator[MessageContentFile]:
    """Yield file content parts from supported input message shapes."""
    if isinstance(input_value, Message):
        yield from message_file_content(input_value)
        return
    if not isinstance(input_value, list):
        return
    for item in input_value:
        if isinstance(item, Message):
            yield from message_file_content(item)


def message_file_content(message: Message) -> Iterator[MessageContentFile]:
    """Yield file content parts from a single message."""
    content = message.content
    if isinstance(content, MessageContentFile):
        yield content
        return
    if not isinstance(content, list):
        return
    for item in content:
        if isinstance(item, MessageContentFile):
            yield item
