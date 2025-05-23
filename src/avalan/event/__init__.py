from dataclasses import dataclass
from enum import StrEnum
from typing import Any

class EventType(StrEnum):
    START = "start"
    TOOL_PROCESS = "tool_process"
    TOOL_EXECUTE = "tool_execute"
    TOOL_RESULT = "tool_result"
    END = "end"
    STREAM_END = "stream_end"

@dataclass(frozen=True, kw_only=True)
class Event:
    type: EventType
    payload: dict[str, Any] | None = None
