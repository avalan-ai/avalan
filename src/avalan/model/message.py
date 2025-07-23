from typing import Literal, TypedDict

TemplateMessageRole = Literal["assistant", "system", "tool", "user"]


class TemplateMessage(TypedDict):
    role: TemplateMessageRole
    content: str
