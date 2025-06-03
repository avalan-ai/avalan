from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    START = "start"
    TOOL_DETECT = "tool_detect"
    TOOL_MODEL_RUN = "tool_model_run"
    TOOL_MODEL_RESPONSE = "tool_model_response"
    TOOL_PROCESS = "tool_process"
    TOOL_EXECUTE = "tool_execute"
    TOOL_RESULT = "tool_result"
    END = "end"
    STREAM_END = "stream_end"
    CALL_PREPARE_BEFORE = "call_prepare_before"
    CALL_PREPARE_AFTER = "call_prepare_after"
    MEMORY_APPEND_BEFORE = "memory_append_before"
    MEMORY_APPEND_AFTER = "memory_append_after"
    MODEL_EXECUTE_BEFORE = "model_execute_before"
    MODEL_EXECUTE_AFTER = "model_execute_after"
    INPUT_TOKEN_COUNT_BEFORE = "input_token_count_before"
    INPUT_TOKEN_COUNT_AFTER = "input_token_count_after"
    TOKEN_GENERATED = "token_generated"


@dataclass(frozen=True, kw_only=True)
class Event:
    type: EventType
    payload: dict[str, Any] | None = None
    started: float | None = None
    finished: float | None = None
    ellapsed: float | None = None


class EventStats:
    triggers: dict[EventType, int] = {}
    total_triggers: int = 0
