from typing import Literal, TypedDict

TemplateMessageRole = Literal["assistant", "system", "tool", "user"]


class TemplateMessageContent(TypedDict, total=False):
    type: Literal["image_url", "text"]
    text: str | None
    image_url: dict[str, str] | None


class TemplateMessage(TypedDict):
    role: TemplateMessageRole
    content: str | TemplateMessageContent | list[TemplateMessageContent]
