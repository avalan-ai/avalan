from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class EventType(StrEnum):
    CALL_PREPARE_BEFORE = "call_prepare_before"
    CALL_PREPARE_AFTER = "call_prepare_after"
    END = "end"
    ENGINE_AGENT_CALL_BEFORE = "engine_agent_call_before"
    ENGINE_AGENT_CALL_AFTER = "engine_agent_call_after"
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
    MODEL_MANAGER_CALL_BEFORE = "model_manager_call_before"
    MODEL_MANAGER_CALL_AFTER = "model_manager_call_after"
    FLOW_CANCELLED = "flow_cancelled"
    FLOW_COMPLETED = "flow_completed"
    FLOW_CONDITION_EVALUATED = "flow_condition_evaluated"
    FLOW_EDGE_ELIGIBLE = "flow_edge_eligible"
    FLOW_EDGE_ROUTED = "flow_edge_routed"
    FLOW_JOIN_READY = "flow_join_ready"
    FLOW_MANAGER_CALL_BEFORE = "flow_manager_call_before"
    FLOW_MANAGER_CALL_AFTER = "flow_manager_call_after"
    FLOW_NODE_CANCELLED = "flow_node_cancelled"
    FLOW_NODE_COMPLETED = "flow_node_completed"
    FLOW_NODE_FAILED = "flow_node_failed"
    FLOW_NODE_PAUSED = "flow_node_paused"
    FLOW_NODE_RESUMED = "flow_node_resumed"
    FLOW_NODE_RETRYING = "flow_node_retrying"
    FLOW_NODE_SKIPPED = "flow_node_skipped"
    FLOW_NODE_STARTED = "flow_node_started"
    FLOW_OUTPUT_SELECTED = "flow_output_selected"
    FLOW_STARTED = "flow_started"
    FLOW_VALIDATION = "flow_validation"
    START = "start"
    STREAM_END = "stream_end"
    TOKEN_GENERATED = "token_generated"
    TOOL_DETECT = "tool_detect"
    TOOL_DIAGNOSTIC = "tool_diagnostic"
    TOOL_EXECUTE = "tool_execute"
    TOOL_MODEL_RUN = "tool_model_run"
    TOOL_MODEL_RESPONSE = "tool_model_response"
    TOOL_PROCESS = "tool_process"
    TOOL_PROGRESS = "tool_progress"
    TOOL_RESULT = "tool_result"


TOOL_TYPES = {et for et in EventType if et.value.startswith("tool_")}


@dataclass(frozen=True, kw_only=True, slots=True)
class Event:
    type: EventType
    payload: dict[str, Any] | None = None
    started: float | None = None
    finished: float | None = None
    elapsed: float | None = None


class EventStats:
    triggers: dict[EventType, int] = {}
    total_triggers: int = 0
