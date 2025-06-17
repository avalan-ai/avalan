from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    CALL_PREPARE_BEFORE = "call_prepare_before"
    CALL_PREPARE_AFTER = "call_prepare_after"
    END = "end"
    ENGINE_RUN_BEFORE = "engine_run_before"
    ENGINE_RUN_AFTER = "engine_run_after"
    INPUT_TOKEN_COUNT_BEFORE = "input_token_count_before"
    INPUT_TOKEN_COUNT_AFTER = "input_token_count_after"
    MEMORY_APPEND_BEFORE = "memory_append_before"
    MEMORY_APPEND_AFTER = "memory_append_after"
    MEMORY_PERMANENT_MESSAGE_ADD = "memory_permanent_message_add"
    MEMORY_PERMANENT_MESSAGE_ADDED = "memory_permanent_message_added"
    MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUE = (
        "memory_permanent_message_session_continue"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_CONTINUED = (
        "memory_permanent_message_session_continued"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_START = (
        "memory_permanent_message_session_start"
    )
    MEMORY_PERMANENT_MESSAGE_SESSION_STARTED = (
        "memory_permanent_message_session_started"
    )
    MODEL_EXECUTE_BEFORE = "model_execute_before"
    MODEL_EXECUTE_AFTER = "model_execute_after"
    START = "start"
    STREAM_END = "stream_end"
    TOKEN_GENERATED = "token_generated"
    TOOL_DETECT = "tool_detect"
    TOOL_EXECUTE = "tool_execute"
    TOOL_MODEL_RUN = "tool_model_run"
    TOOL_MODEL_RESPONSE = "tool_model_response"
    TOOL_PROCESS = "tool_process"
    TOOL_RESULT = "tool_result"


TOOL_TYPES = {et for et in EventType if et.value.startswith("tool_")}


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
