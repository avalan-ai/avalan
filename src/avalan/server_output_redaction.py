"""Define server output redaction settings."""

from typing import Literal

ServerOutputRedactionRule = Literal[
    "host_paths",
    "skill_body_echoes",
    "skill_source_paths",
    "skills_tool_content",
]
ServerOutputRedactionProtocol = Literal["openai", "mcp", "a2a"]
ServerOutputRedactionChannel = Literal["answer", "reasoning"]
SERVER_OUTPUT_REDACTION_RULES: tuple[ServerOutputRedactionRule, ...] = (
    "host_paths",
    "skill_body_echoes",
    "skill_source_paths",
    "skills_tool_content",
)
SERVER_OUTPUT_REDACTION_PROTOCOLS: tuple[
    ServerOutputRedactionProtocol, ...
] = ("openai", "mcp", "a2a")
SERVER_OUTPUT_REDACTION_CHANNELS: tuple[ServerOutputRedactionChannel, ...] = (
    "answer",
    "reasoning",
)
