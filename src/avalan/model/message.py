from typing import Literal, TypedDict

TemplateMessageRole = Literal[
    "assistant", "developer", "system", "tool", "user"
]


class TemplateMessageContent(TypedDict, total=False):
    type: Literal["file", "image_url", "text"]
    file: dict[str, object] | None
    text: str | None
    image_url: dict[str, str] | None


class TemplateMessage(TypedDict):
    role: TemplateMessageRole
    content: str | TemplateMessageContent | list[TemplateMessageContent]
