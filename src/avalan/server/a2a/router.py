from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    ReasoningToken,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
)
from ...event import Event, EventType
from ...utils import to_json
from ..entities import ChatCompletionRequest, ChatMessage
from ..routers import orchestrate
from ..sse import sse_message
from .store import TaskStore

from asyncio import CancelledError
from collections.abc import AsyncGenerator, AsyncIterable
from datetime import datetime, timezone
from enum import Enum, auto
from json import JSONDecodeError
from logging import Logger
from re import compile
from time import time
from typing import Any, Final, Iterable
from uuid import uuid4

from a2a import types as a2a_types
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import ValidationError


def _di_get_logger(request: Request) -> Logger:
    from .. import di_get_logger as _impl

    return _impl(request)


async def _di_get_orchestrator(request: Request) -> Orchestrator:
    from .. import di_get_orchestrator as _impl

    return await _impl(request)


router = APIRouter(tags=["a2a"])
well_known_router = APIRouter()


_INPUT_TYPE_TO_MIME: dict[str, str] = {
    "text": "text/plain",
}


_OUTPUT_TYPE_TO_MIME: dict[str, str] = {
    "json": "application/json",
    "text": "text/markdown",
}


_DEFAULT_INPUT_MODE = "text/plain"
_DEFAULT_OUTPUT_MODE = "text/markdown"
_WORD_PATTERN = compile(r"[0-9A-Za-z]+")


_STREAM_RESPONSE_ID_UNSET: Final[object] = object()
_REASONING_ARTIFACT_ID: Final[str] = "reasoning"
_ANSWER_ARTIFACT_ID: Final[str] = "answer"
_REASONING_ARTIFACT_KIND: Final[str] = "reasoning"
_ANSWER_ARTIFACT_KIND: Final[str] = "answer"
_STATUS_TO_STATE: Final[dict[str, a2a_types.TaskState]] = {
    "accepted": a2a_types.TaskState.submitted,
    "in_progress": a2a_types.TaskState.working,
    "completed": a2a_types.TaskState.completed,
    "failed": a2a_types.TaskState.failed,
}
_FINAL_TASK_STATES: Final[set[a2a_types.TaskState]] = {
    a2a_types.TaskState.completed,
    a2a_types.TaskState.failed,
    a2a_types.TaskState.canceled,
    a2a_types.TaskState.rejected,
}
_ROLE_MAPPING: Final[dict[str, a2a_types.Role]] = {
    "agent": a2a_types.Role.agent,
    "assistant": a2a_types.Role.agent,
    "system": a2a_types.Role.agent,
    "user": a2a_types.Role.user,
}


def _timestamp_to_iso(value: float | None) -> str | None:
    if value is None:
        return None
    timestamp = datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
    return timestamp[:-6] + "Z" if timestamp.endswith("+00:00") else timestamp


def _status_to_state(status: str) -> a2a_types.TaskState:
    return _STATUS_TO_STATE.get(status, a2a_types.TaskState.unknown)


def _is_final_state(state: a2a_types.TaskState) -> bool:
    return state in _FINAL_TASK_STATES


def _role_from_payload(role: str | None) -> a2a_types.Role:
    if not role:
        return a2a_types.Role.agent
    return _ROLE_MAPPING.get(role.lower(), a2a_types.Role.agent)


def _message_parts_from_payload(content: Any) -> list[a2a_types.Part]:
    parts: list[a2a_types.Part] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text") or "")
                    parts.append(
                        a2a_types.Part(root=a2a_types.TextPart(text=text))
                    )
                else:
                    parts.append(
                        a2a_types.Part(
                            root=a2a_types.DataPart(
                                data={
                                    key: value for key, value in item.items()
                                }
                            )
                        )
                    )
            elif isinstance(item, str):
                parts.append(
                    a2a_types.Part(root=a2a_types.TextPart(text=item))
                )
    if not parts:
        parts.append(a2a_types.Part(root=a2a_types.TextPart(text="")))
    return parts


def _artifact_parts_from_payload(content: Any) -> list[a2a_types.Part]:
    parts: list[a2a_types.Part] = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text") or "")
                    parts.append(
                        a2a_types.Part(root=a2a_types.TextPart(text=text))
                    )
                else:
                    parts.append(
                        a2a_types.Part(
                            root=a2a_types.DataPart(
                                data={
                                    key: value for key, value in item.items()
                                }
                            )
                        )
                    )
            elif isinstance(item, str):
                parts.append(
                    a2a_types.Part(root=a2a_types.TextPart(text=item))
                )
            else:
                parts.append(
                    a2a_types.Part(
                        root=a2a_types.DataPart(data={"value": item})
                    )
                )
    if not parts:
        parts.append(a2a_types.Part(root=a2a_types.TextPart(text="")))
    return parts


def _task_metadata_from_overview(overview: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(overview.get("metadata") or {})
    metadata.pop("jsonrpc_id", None)
    model = overview.get("model")
    if model is not None:
        metadata.setdefault("model", model)
    instructions = overview.get("instructions")
    if instructions:
        metadata.setdefault("instructions", instructions)
    return metadata


class StreamState(Enum):
    """High level phases produced while translating orchestrator tokens."""

    REASONING = auto()
    TOOL = auto()
    ANSWER = auto()


class A2ATaskCreateRequest(ChatCompletionRequest):
    """Extends chat completion requests with A2A specific fields."""

    metadata: dict[str, Any] | None = None
    instructions: str | None = None

    def conversation(self) -> list[ChatMessage]:
        messages = [
            ChatMessage.model_validate(message) for message in self.messages
        ]
        if self.instructions:
            messages.insert(
                0,
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=self.instructions,
                ),
            )
        return messages


def di_get_task_store(request: Request) -> TaskStore:
    if not hasattr(request.app.state, "a2a_store"):
        request.app.state.a2a_store = TaskStore()
    store = request.app.state.a2a_store
    assert isinstance(store, TaskStore)
    return store


def _task_metadata(payload: A2ATaskCreateRequest) -> dict[str, Any]:
    metadata = dict(payload.metadata or {})
    if payload.temperature is not None:
        metadata.setdefault("temperature", payload.temperature)
    if payload.top_p is not None:
        metadata.setdefault("top_p", payload.top_p)
    if payload.max_tokens is not None:
        metadata.setdefault("max_tokens", payload.max_tokens)
    if payload.response_format:
        metadata.setdefault(
            "response_format",
            payload.response_format.model_dump(mode="json"),
        )
    return metadata


MODEL_FALLBACK: Final[str] = "default"
_JSONRPC_TEXT_PART_KINDS: Final[set[str]] = {"text", "input_text"}
_JSONRPC_MODEL_KEYS: Final[tuple[str, ...]] = (
    "model",
    "modelId",
    "defaultModel",
)
_JSONRPC_MODELS_KEYS: Final[tuple[str, ...]] = ("models", "modelIds")


def _default_model_id(orchestrator: Orchestrator) -> str:
    model_ids = getattr(orchestrator, "model_ids", None)
    if model_ids:
        candidates = sorted(
            str(model_id) for model_id in model_ids if model_id
        )
        if candidates:
            return candidates[0]
    return MODEL_FALLBACK


def _normalize_task_request(
    payload: dict[str, Any], orchestrator: Orchestrator
) -> dict[str, Any]:
    if "jsonrpc" not in payload:
        return payload
    return _normalize_jsonrpc_task_request(payload, orchestrator)


def _normalize_jsonrpc_task_request(
    payload: dict[str, Any], orchestrator: Orchestrator
) -> dict[str, Any]:
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("JSON-RPC params must be an object")

    configuration = params.get("configuration")
    if configuration is not None and not isinstance(configuration, dict):
        raise ValueError("JSON-RPC configuration must be an object")

    message = params.get("message")
    if not isinstance(message, dict):
        raise ValueError("JSON-RPC message must be an object")

    method = str(payload.get("method") or "")
    stream = method.endswith("/stream") or bool(params.get("stream"))

    normalized: dict[str, Any] = {
        "model": _select_jsonrpc_model(params, configuration, orchestrator),
        "messages": _collect_jsonrpc_messages(params, message),
        "stream": stream,
    }

    instructions = _extract_jsonrpc_instructions(params, configuration)
    if instructions:
        normalized["instructions"] = instructions

    metadata = _extract_jsonrpc_metadata(
        payload, params, configuration, message
    )
    if metadata:
        normalized["metadata"] = metadata

    return normalized


def _select_jsonrpc_model(
    params: dict[str, Any],
    configuration: dict[str, Any] | None,
    orchestrator: Orchestrator,
) -> str:
    for source in (params, configuration):
        if not isinstance(source, dict):
            continue
        for key in _JSONRPC_MODEL_KEYS:
            candidate = source.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
        for key in _JSONRPC_MODELS_KEYS:
            models = source.get(key)
            if isinstance(models, list):
                for candidate in models:
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate
    return _default_model_id(orchestrator)


def _collect_jsonrpc_messages(
    params: dict[str, Any], message: dict[str, Any]
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []

    def extend_from(candidate: Any) -> None:
        if not isinstance(candidate, list):
            return
        for item in candidate:
            if isinstance(item, dict):
                messages.append(_jsonrpc_message_to_chat(item))

    context = params.get("context")
    if isinstance(context, dict):
        extend_from(context.get("messages"))
        extend_from(context.get("conversation"))
        extend_from(context.get("history"))

    extend_from(params.get("messages"))
    extend_from(params.get("conversation"))

    messages.append(_jsonrpc_message_to_chat(message))
    return messages


def _jsonrpc_message_to_chat(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or MessageRole.USER.value)
    content = _jsonrpc_message_text(message)
    return {"role": role, "content": content}


def _jsonrpc_message_text(message: dict[str, Any]) -> str:
    parts = message.get("parts")
    if isinstance(parts, list) and parts:
        text_parts: list[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            kind = part.get("kind")
            if kind in _JSONRPC_TEXT_PART_KINDS:
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
            elif kind is not None:
                continue
        return "".join(text_parts)

    for key in ("text", "content"):
        value = message.get(key)
        if isinstance(value, str):
            return value
    return ""


def _extract_jsonrpc_instructions(
    params: dict[str, Any], configuration: dict[str, Any] | None
) -> str | None:
    candidates: list[Any] = [params.get("instructions")]

    context = params.get("context")
    if isinstance(context, dict):
        candidates.append(context.get("instructions"))

    if isinstance(configuration, dict):
        candidates.append(configuration.get("instructions"))

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _extract_jsonrpc_metadata(
    payload: dict[str, Any],
    params: dict[str, Any],
    configuration: dict[str, Any] | None,
    message: dict[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}

    jsonrpc_id = payload.get("id")
    if jsonrpc_id is not None:
        metadata["jsonrpc_id"] = jsonrpc_id

    if isinstance(configuration, dict) and configuration:
        metadata["configuration"] = configuration

    params_metadata = params.get("metadata")
    if isinstance(params_metadata, dict) and params_metadata:
        metadata["params_metadata"] = params_metadata

    message_metadata = message.get("metadata")
    if isinstance(message_metadata, dict) and message_metadata:
        metadata["message_metadata"] = message_metadata

    return metadata


class A2AResponseTranslator:
    """Convert orchestrator streaming output into A2A task artifacts."""

    def __init__(self, task_id: str, store: TaskStore) -> None:
        self._task_id = task_id
        self._store = store
        self._state: StreamState | None = None
        self._reasoning_artifact_id: str | None = None
        self._answer_artifact_id: str | None = None
        self._tool_artifact_id: str | None = None
        self._answer: list[str] = []

    @property
    def text(self) -> str:
        return "".join(self._answer)

    async def run_stream(
        self,
        response: AsyncIterable[Token | TokenDetail | Event | str],
    ) -> AsyncGenerator[dict[str, Any], None]:
        for event in await self._store.set_status(
            self._task_id, "in_progress"
        ):
            yield event
        async for item in response:
            for event in await self._process_item(item):
                yield event
        for event in await self._finish():
            yield event

    async def consume(
        self,
        response: AsyncIterable[Token | TokenDetail | Event | str],
    ) -> str:
        async for _ in self.run_stream(response):
            continue
        return self.text

    async def _process_item(
        self, item: Token | TokenDetail | Event | str
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        call_id = _call_identifier(item)
        state = _state_for_item(item)
        events.extend(await self._switch_state(state, call_id))

        if isinstance(item, ReasoningToken):
            events.extend(await self._ensure_reasoning_artifact())
            assert self._reasoning_artifact_id
            events.extend(
                await self._store.add_artifact_delta(
                    self._task_id,
                    self._reasoning_artifact_id,
                    {"type": "text", "text": item.token},
                )
            )
            return events

        if isinstance(item, Event):
            if item.type is EventType.TOOL_PROCESS:
                events.extend(await self._handle_tool_process(item))
                return events
            if item.type is EventType.TOOL_RESULT:
                events.extend(await self._handle_tool_result(item))
                return events
            return events

        if isinstance(item, ToolCallToken):
            events.extend(await self._handle_tool_token(item))
            return events

        text = _token_text(item)
        if text:
            events.extend(await self._ensure_answer_artifact())
            assert self._answer_artifact_id
            self._answer.append(text)
            events.extend(
                await self._store.add_artifact_delta(
                    self._task_id,
                    self._answer_artifact_id,
                    {"type": "text", "text": text},
                )
            )
        return events

    async def _finish(self) -> list[dict[str, Any]]:
        events = await self._switch_state(None, None)
        if self._reasoning_artifact_id:
            events.extend(
                await self._store.complete_artifact(
                    self._task_id, self._reasoning_artifact_id
                )
            )
            self._reasoning_artifact_id = None
        if self._answer_artifact_id:
            events.extend(
                await self._store.complete_artifact(
                    self._task_id, self._answer_artifact_id
                )
            )
            self._answer_artifact_id = None
        events.extend(await self._store.complete_task(self._task_id))
        return events

    async def _switch_state(
        self, state: StreamState | None, call_id: str | None
    ) -> list[dict[str, Any]]:
        changed = self._state is not state or (
            self._state is StreamState.TOOL
            and state is StreamState.TOOL
            and call_id is not None
            and call_id != self._tool_artifact_id
        )
        if not changed:
            return []

        events: list[dict[str, Any]] = []

        if self._state is StreamState.TOOL and self._tool_artifact_id:
            events.extend(
                await self._store.complete_artifact(
                    self._task_id, self._tool_artifact_id
                )
            )
            self._tool_artifact_id = None
        elif (
            self._state is StreamState.REASONING
            and state is None
            and self._reasoning_artifact_id
        ):
            events.extend(
                await self._store.complete_artifact(
                    self._task_id, self._reasoning_artifact_id
                )
            )
            self._reasoning_artifact_id = None
        elif (
            self._state is StreamState.ANSWER
            and state is None
            and self._answer_artifact_id
        ):
            events.extend(
                await self._store.complete_artifact(
                    self._task_id, self._answer_artifact_id
                )
            )
            self._answer_artifact_id = None

        self._state = state

        if state is StreamState.REASONING:
            events.extend(await self._ensure_reasoning_artifact())
        elif state is StreamState.TOOL:
            artifact = call_id or str(uuid4())
            (
                self._tool_artifact_id,
                created,
            ) = await self._store.ensure_artifact(
                self._task_id,
                artifact_id=artifact,
                name=None,
                kind="tool_call",
                role=str(MessageRole.ASSISTANT),
            )
            events.extend(created)
        elif state is StreamState.ANSWER:
            events.extend(await self._ensure_answer_artifact())

        return events

    async def _ensure_reasoning_artifact(self) -> list[dict[str, Any]]:
        if self._reasoning_artifact_id:
            return []
        artifact_id, created = await self._store.ensure_artifact(
            self._task_id,
            artifact_id=_REASONING_ARTIFACT_ID,
            name="Reasoning",
            kind=_REASONING_ARTIFACT_KIND,
            role=str(MessageRole.ASSISTANT),
            metadata={"channel": "reasoning"},
        )
        self._reasoning_artifact_id = artifact_id
        return created

    async def _ensure_answer_artifact(self) -> list[dict[str, Any]]:
        if self._answer_artifact_id:
            return []
        artifact_id, created = await self._store.ensure_artifact(
            self._task_id,
            artifact_id=_ANSWER_ARTIFACT_ID,
            name="Answer",
            kind=_ANSWER_ARTIFACT_KIND,
            role=str(MessageRole.ASSISTANT),
            metadata={"channel": "output"},
        )
        self._answer_artifact_id = artifact_id
        return created

    async def _handle_tool_process(self, event: Event) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        payload = event.payload or []
        if isinstance(payload, dict):
            calls: Iterable[ToolCall] = payload.get("calls", [])  # type: ignore[assignment]
        else:
            calls = payload  # type: ignore[assignment]
        for call in calls:
            if not isinstance(call, ToolCall):
                continue
            artifact_id = str(call.id)
            (
                self._tool_artifact_id,
                created,
            ) = await self._store.ensure_artifact(
                self._task_id,
                artifact_id=artifact_id,
                name=call.name,
                kind="tool_call",
                role=str(MessageRole.ASSISTANT),
                metadata={"arguments": call.arguments or {}},
            )
            events.extend(created)
            events.extend(
                await self._store.add_artifact_delta(
                    self._task_id,
                    self._tool_artifact_id,
                    {
                        "type": "arguments",
                        "arguments": call.arguments or {},
                    },
                )
            )
            events.extend(
                await self._store.add_status_event(
                    self._task_id,
                    status="in_progress",
                    metadata={
                        "phase": "tool_processing",
                        "tool_call_id": artifact_id,
                        "tool_name": call.name,
                    },
                )
            )
        return events

    async def _handle_tool_result(self, event: Event) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        payload = event.payload or {}
        result = payload.get("result") if isinstance(payload, dict) else None
        call = payload.get("call") if isinstance(payload, dict) else None
        artifact_id: str | None = None
        artifact_name: str | None = None
        content: Any = None
        metadata: dict[str, Any] = {}

        if isinstance(result, ToolCallResult):
            artifact_id = str(result.call.id)
            artifact_name = result.call.name
            content = {"type": "data", "data": result.result}
            metadata["status"] = "success"
        elif isinstance(result, ToolCallError):
            artifact_id = str(result.call.id)
            artifact_name = result.call.name
            content = {
                "type": "data",
                "data": {"error": result.message},
            }
            metadata["status"] = "error"
        elif isinstance(payload, ToolCall):
            artifact_id = str(payload.id)
            artifact_name = payload.name
            content = {"type": "data", "data": None}
        else:
            artifact_id = self._tool_artifact_id
            content = {"type": "data", "data": result}

        if call and isinstance(call, ToolCall):
            artifact_id = str(call.id)
            artifact_name = call.name

        if artifact_id is None:
            artifact_id = str(uuid4())

        self._tool_artifact_id, created = await self._store.ensure_artifact(
            self._task_id,
            artifact_id=artifact_id,
            name=artifact_name,
            kind="tool_call",
            role=str(MessageRole.ASSISTANT),
            metadata=metadata,
        )
        events.extend(created)
        events.extend(
            await self._store.add_artifact_delta(
                self._task_id,
                self._tool_artifact_id,
                content,
            )
        )
        events.extend(
            await self._store.complete_artifact(
                self._task_id, self._tool_artifact_id
            )
        )
        status_metadata: dict[str, Any] = {
            "phase": "tool_completed",
            "tool_call_id": artifact_id,
            "tool_name": artifact_name,
        }
        outcome = metadata.get("status")
        if outcome:
            status_metadata["tool_status"] = outcome
        events.extend(
            await self._store.add_status_event(
                self._task_id,
                status="completed",
                metadata=status_metadata,
            )
        )
        self._tool_artifact_id = None
        return events

    async def _handle_tool_token(
        self, token: ToolCallToken
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if token.call is not None:
            artifact_id = str(token.call.id)
            (
                self._tool_artifact_id,
                created,
            ) = await self._store.ensure_artifact(
                self._task_id,
                artifact_id=artifact_id,
                name=token.call.name,
                kind="tool_call",
                role=str(MessageRole.ASSISTANT),
                metadata={"arguments": token.call.arguments or {}},
            )
            events.extend(created)
            events.extend(
                await self._store.add_artifact_delta(
                    self._task_id,
                    self._tool_artifact_id,
                    {
                        "type": "arguments",
                        "arguments": token.call.arguments or {},
                    },
                )
            )
            events.extend(
                await self._store.add_status_event(
                    self._task_id,
                    status="in_progress",
                    metadata={
                        "phase": "tool_processing",
                        "tool_call_id": artifact_id,
                        "tool_name": token.call.name,
                    },
                )
            )
        else:
            if not self._tool_artifact_id:
                (
                    self._tool_artifact_id,
                    created,
                ) = await self._store.ensure_artifact(
                    self._task_id,
                    artifact_id=str(uuid4()),
                    name=None,
                    kind="tool_call",
                    role=str(MessageRole.ASSISTANT),
                )
                events.extend(created)
            events.extend(
                await self._store.add_artifact_delta(
                    self._task_id,
                    self._tool_artifact_id,
                    {"type": "text", "text": token.token},
                )
            )
        return events


class A2AStreamEventConverter:
    """Translate internal task events into A2A streaming responses."""

    def __init__(self, task_id: str, store: TaskStore) -> None:
        self._task_id = task_id
        self._store = store
        self._cached_response_id: str | int | None | object = (
            _STREAM_RESPONSE_ID_UNSET
        )
        self._artifact_progress: dict[str, int] = {}

    async def convert(self, event: dict[str, Any]) -> dict[str, Any]:
        try:
            converted = await self._convert(event)
        except Exception:  # pragma: no cover - defensive fallback
            return event
        if converted is None:
            return event
        return converted.model_dump(mode="json", by_alias=True)

    async def _convert(
        self, event: dict[str, Any]
    ) -> a2a_types.SendStreamingMessageSuccessResponse | None:
        name = event.get("event")
        if not isinstance(name, str):
            return None

        result: (
            a2a_types.Task
            | a2a_types.Message
            | a2a_types.TaskStatusUpdateEvent
            | a2a_types.TaskArtifactUpdateEvent
            | None
        )
        if name == "task.created":
            result = await self._task_result(event)
        elif name == "task.status.changed":
            payload = event.get("data")
            status_payload = payload if isinstance(payload, dict) else {}
            status_override = status_payload.get("status")
            extra_metadata: dict[str, Any] = {"event": name}
            metadata_payload = status_payload.get("metadata")
            if isinstance(metadata_payload, dict):
                for key, value in metadata_payload.items():
                    if value is not None:
                        extra_metadata[key] = value
            result = await self._status_update_result(
                event, status_override, extra_metadata
            )
        elif name == "task.failed":
            payload = event.get("data")
            error_payload = payload if isinstance(payload, dict) else {}
            error_message = error_payload.get("error")
            result = await self._status_update_result(
                event,
                "failed",
                {"event": name, "error": error_message},
            )
        elif name.startswith("message."):
            result = await self._message_result(event)
        elif name.startswith("artifact."):
            result = await self._artifact_result(event)
        elif name in {"task.stream.completed", "done"}:
            result = await self._status_update_result(
                event, None, {"event": name}
            )
        else:
            return None

        if result is None:
            return None

        return a2a_types.SendStreamingMessageSuccessResponse(
            id=await self._response_id(),
            result=result,
        )

    async def _response_id(self) -> str | int | None:
        if self._cached_response_id is _STREAM_RESPONSE_ID_UNSET:
            overview = await self._store.get_task_overview(self._task_id)
            metadata = overview.get("metadata") or {}
            self._cached_response_id = metadata.get("jsonrpc_id")
        return (
            None
            if self._cached_response_id is _STREAM_RESPONSE_ID_UNSET
            else self._cached_response_id
        )

    async def _task_result(self, event: dict[str, Any]) -> a2a_types.Task:
        overview = await self._store.get_task_overview(self._task_id)
        status = _status_to_state(str(overview.get("status") or ""))
        metadata = _task_metadata_from_overview(overview)
        task_status = a2a_types.TaskStatus(
            state=status,
            timestamp=_timestamp_to_iso(event.get("created_at")),
        )
        return a2a_types.Task(
            id=self._task_id,
            context_id=self._task_id,
            status=task_status,
            metadata=metadata or None,
        )

    async def _status_update_result(
        self,
        event: dict[str, Any],
        status_override: str | None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> a2a_types.TaskStatusUpdateEvent:
        overview = await self._store.get_task_overview(self._task_id)
        status_value = status_override or str(overview.get("status") or "")
        state = _status_to_state(status_value)
        metadata: dict[str, Any] = {"raw_status": status_value}
        error_text = overview.get("error")
        if error_text and state is a2a_types.TaskState.failed:
            metadata.setdefault("error", error_text)
        if extra_metadata:
            for key, value in extra_metadata.items():
                if value is not None:
                    metadata[key] = value
        task_status = a2a_types.TaskStatus(
            state=state,
            timestamp=_timestamp_to_iso(event.get("created_at")),
        )
        return a2a_types.TaskStatusUpdateEvent(
            task_id=self._task_id,
            context_id=self._task_id,
            status=task_status,
            final=_is_final_state(state),
            metadata=metadata or None,
        )

    async def _message_result(
        self, event: dict[str, Any]
    ) -> a2a_types.Message | None:
        payload = (
            event.get("data") if isinstance(event.get("data"), dict) else {}
        )
        message_data = (
            payload.get("message") if isinstance(payload, dict) else None
        )
        if not isinstance(message_data, dict):
            return None
        message_id_value = message_data.get("id") or message_data.get(
            "messageId"
        )
        if message_id_value is None:
            return None
        message_id = str(message_id_value)
        message_payload = await self._store.get_message_payload(
            self._task_id, message_id
        )
        metadata: dict[str, Any] = {}
        channel = message_payload.get("channel")
        if channel:
            metadata["channel"] = channel
        state = message_payload.get("state")
        if state:
            metadata["state"] = state
        event_name = event.get("event")
        if event_name:
            metadata["raw_event"] = event_name
        if event_name == "message.delta":
            metadata["delta"] = message_data.get("delta")
        if event_name == "message.completed":
            metadata["completed"] = True
        metadata = {
            key: value for key, value in metadata.items() if value is not None
        }
        return a2a_types.Message(
            message_id=message_id,
            task_id=self._task_id,
            context_id=self._task_id,
            role=_role_from_payload(message_payload.get("role")),
            parts=_message_parts_from_payload(message_payload.get("content")),
            metadata=metadata or None,
        )

    async def _artifact_result(
        self, event: dict[str, Any]
    ) -> a2a_types.TaskArtifactUpdateEvent | None:
        payload = (
            event.get("data") if isinstance(event.get("data"), dict) else {}
        )
        artifact_data = (
            payload.get("artifact") if isinstance(payload, dict) else None
        )
        if not isinstance(artifact_data, dict):
            return None
        artifact_id_value = artifact_data.get("id") or artifact_data.get(
            "artifactId"
        )
        if artifact_id_value is None:
            return None
        artifact_id = str(artifact_id_value)
        artifact_payload = await self._store.get_artifact(
            self._task_id, artifact_id
        )
        artifact_metadata = dict(artifact_payload.get("metadata") or {})
        role = artifact_payload.get("role")
        if role:
            artifact_metadata.setdefault("role", role)
        kind = artifact_payload.get("kind")
        if kind:
            artifact_metadata.setdefault("kind", kind)
        event_name = event.get("event")
        event_metadata: dict[str, Any] = {}
        if event_name:
            event_metadata["raw_event"] = event_name
        state = artifact_payload.get("state")
        if state:
            event_metadata["state"] = state
        append = event_name == "artifact.delta"
        last_chunk = event_name == "artifact.completed"
        payload_items: list[Any]
        previous_count = self._artifact_progress.get(artifact_id, 0)

        if event_name == "artifact.delta":
            delta_payload = artifact_data.get("payload")
            if isinstance(delta_payload, list):
                payload_items = list(delta_payload)
            elif delta_payload is None:
                payload_items = []
            else:
                payload_items = [delta_payload]
            self._artifact_progress[artifact_id] = previous_count + len(
                payload_items
            )
        else:
            content_payload = artifact_payload.get("content")
            if isinstance(content_payload, list):
                content_items = list(content_payload)
            elif content_payload is None:
                content_items = []
            else:
                content_items = [content_payload]

            if last_chunk:
                append = True
                if previous_count < len(content_items):
                    payload_items = content_items[previous_count:]
                elif content_items:
                    payload_items = content_items[-1:]
                else:
                    payload_items = []
                self._artifact_progress.pop(artifact_id, None)
            else:
                append = False
                payload_items = content_items
                self._artifact_progress[artifact_id] = len(payload_items)

        parts_payload: Any = payload_items
        artifact = a2a_types.Artifact(
            artifact_id=artifact_id,
            name=artifact_payload.get("name"),
            metadata=artifact_metadata or None,
            parts=_artifact_parts_from_payload(parts_payload),
        )
        event_metadata = {
            key: value
            for key, value in event_metadata.items()
            if value is not None
        }
        return a2a_types.TaskArtifactUpdateEvent(
            task_id=self._task_id,
            context_id=self._task_id,
            artifact=artifact,
            append=True if append else None,
            last_chunk=True if last_chunk else None,
            metadata=event_metadata or None,
        )


@router.post("/tasks")
async def create_task(
    request: Request,
    logger: Logger = Depends(_di_get_logger),
    orchestrator: Orchestrator = Depends(_di_get_orchestrator),
    store: TaskStore = Depends(di_get_task_store),
):
    try:
        raw_payload = await request.json()
    except (JSONDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="Invalid JSON body"
        ) from exc

    if not isinstance(raw_payload, dict):
        raise HTTPException(
            status_code=400, detail="Request body must be an object"
        )

    try:
        normalized_payload = _normalize_task_request(raw_payload, orchestrator)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        payload = A2ATaskCreateRequest.model_validate(normalized_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    if not payload.messages:
        raise HTTPException(
            status_code=400, detail="Provide at least one message"
        )

    conversation = payload.conversation()
    chat_request = ChatCompletionRequest(
        model=payload.model,
        messages=conversation,
        temperature=payload.temperature,
        top_p=payload.top_p,
        n=1,
        stream=payload.stream,
        stop=payload.stop,
        max_tokens=payload.max_tokens,
        response_format=payload.response_format,
    )

    response, task_uuid, _timestamp = await orchestrate(
        chat_request, logger, orchestrator
    )
    task_id = str(task_uuid)

    initial_events = await store.create_task(
        task_id,
        model=payload.model,
        instructions=payload.instructions,
        input_messages=[msg.model_dump(mode="json") for msg in conversation],
        metadata=_task_metadata(payload),
    )

    translator = A2AResponseTranslator(task_id, store)
    converter = A2AStreamEventConverter(task_id, store)

    async def stream() -> AsyncGenerator[str, None]:
        try:
            for event in initial_events:
                yield sse_message(
                    to_json(await converter.convert(event)),
                    event=event.get("event") or "message",
                )
            async for event in translator.run_stream(response):
                yield sse_message(
                    to_json(await converter.convert(event)),
                    event=event.get("event") or "message",
                )
        except CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception(
                "A2A streaming task %s failed", task_id, exc_info=exc
            )
            for event in await store.fail_task(task_id, str(exc)):
                yield sse_message(
                    to_json(await converter.convert(event)),
                    event=event.get("event") or "message",
                )
        finally:
            completion_event = {
                "event": "task.stream.completed",
                "task_id": task_id,
                "created_at": time(),
                "data": {},
            }
            yield sse_message(
                to_json(await converter.convert(completion_event)),
                event="task.stream.completed",
            )
            done_event = {
                "event": "done",
                "task_id": task_id,
                "created_at": time(),
                "data": {},
            }
            yield sse_message(
                to_json(await converter.convert(done_event)),
                event="done",
            )
            await orchestrator.sync_messages()

    if payload.stream:
        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        await translator.consume(response)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.exception("A2A task %s failed", task_id, exc_info=exc)
        await store.fail_task(task_id, str(exc))
        await orchestrator.sync_messages()
        raise HTTPException(
            status_code=500, detail="Task execution failed"
        ) from exc

    await orchestrator.sync_messages()
    task = await store.get_task(task_id)
    return _coerce("Task", task)


@router.get("/tasks/{task_id}")
async def get_task(
    task_id: str,
    store: TaskStore = Depends(di_get_task_store),
):
    task = await store.get_task(task_id)
    return _coerce("Task", task)


@router.get("/tasks/{task_id}/events")
async def list_task_events(
    task_id: str,
    after: int | None = Query(None),
    store: TaskStore = Depends(di_get_task_store),
):
    events = await store.get_events(task_id, after=after)
    return _coerce_list("TaskEvent", events)


@router.get("/tasks/{task_id}/artifacts/{artifact_id}")
async def get_artifact(
    task_id: str,
    artifact_id: str,
    store: TaskStore = Depends(di_get_task_store),
):
    artifact = await store.get_artifact(task_id, artifact_id)
    return _coerce("TaskArtifact", artifact)


@router.get("/agent")
async def agent_card(
    request: Request,
    orchestrator: Orchestrator = Depends(_di_get_orchestrator),
):
    interface_url = str(request.url_for("create_task"))
    card = _build_agent_card(
        orchestrator,
        getattr(request.app.state, "a2a_tool_name", "run"),
        getattr(request.app.state, "a2a_tool_description", None),
        interface_url,
    )
    return _coerce("AgentCard", card)


@well_known_router.get("/.well-known/a2a-agent.json")
async def well_known_agent_card(
    request: Request,
    orchestrator: Orchestrator = Depends(_di_get_orchestrator),
):
    interface_url = str(request.url_for("create_task"))
    card = _build_agent_card(
        orchestrator,
        getattr(request.app.state, "a2a_tool_name", "run"),
        getattr(request.app.state, "a2a_tool_description", None),
        interface_url,
    )
    return _coerce("AgentCard", card)


def _state_for_item(
    item: Token | TokenDetail | Event | str,
) -> StreamState | None:
    if isinstance(item, ReasoningToken):
        return StreamState.REASONING
    if isinstance(item, (ToolCallToken, Event)):
        if isinstance(item, Event) and item.type not in (
            EventType.TOOL_PROCESS,
            EventType.TOOL_RESULT,
        ):
            return None
        return StreamState.TOOL
    if isinstance(item, str):
        return StreamState.ANSWER
    return StreamState.ANSWER if isinstance(item, Token) else None


def _call_identifier(item: Token | TokenDetail | Event | str) -> str | None:
    if isinstance(item, ToolCallToken) and item.call is not None:
        return str(item.call.id)
    if isinstance(item, Event):
        if item.type is EventType.TOOL_PROCESS:
            payload = item.payload or []
            if isinstance(payload, dict):
                candidates = payload.get("calls", [])
            else:
                candidates = payload
            if candidates:
                call = candidates[0]
                if isinstance(call, ToolCall):
                    return str(call.id)
        if item.type is EventType.TOOL_RESULT and item.payload:
            result = item.payload.get("result")
            if isinstance(result, (ToolCallResult, ToolCallError)):
                return str(result.call.id)
            call = item.payload.get("call")
            if isinstance(call, ToolCall):
                return str(call.id)
    return None


def _token_text(item: Token | TokenDetail | Event | str) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Token):
        return item.token
    return ""


def _enum_value(value: Any) -> str | None:
    if value is None:
        return None
    candidate = getattr(value, "value", value)
    text = str(candidate)
    return text if text else None


def _append_unique(target: list[str], candidate: str | None) -> None:
    if not candidate:
        return
    text = candidate.strip()
    if not text or text in target:
        return
    target.append(text)


def _input_mode_for_spec(spec: Any) -> str:
    value = _enum_value(getattr(spec, "input_type", None))
    return _INPUT_TYPE_TO_MIME.get(value or "", _DEFAULT_INPUT_MODE)


def _output_mode_for_spec(spec: Any) -> str:
    value = _enum_value(getattr(spec, "output_type", None))
    return _OUTPUT_TYPE_TO_MIME.get(value or "", _DEFAULT_OUTPUT_MODE)


def _skill_tags(*values: str | None) -> list[str]:
    tags: list[str] = []
    for value in values:
        if not value:
            continue
        for match in _WORD_PATTERN.findall(value.lower()):
            if match not in tags:
                tags.append(match)
    if not tags:
        tags.append("general")
    return tags


def _skill_from_spec(
    index: int,
    spec: Any,
    tool_name: str | None,
    tool_description: str | None,
    orchestrator: Orchestrator,
) -> dict[str, Any]:
    goal = getattr(spec, "goal", None)
    goal_task = getattr(goal, "task", None) if goal else None
    goal_instructions = list(getattr(goal, "instructions", []) or [])

    description_parts: list[str] = []
    _append_unique(description_parts, tool_description)
    _append_unique(description_parts, goal_task)
    for item in goal_instructions:
        _append_unique(description_parts, item)
    if not description_parts:
        _append_unique(
            description_parts,
            orchestrator.name or "Avalan orchestrated agent",
        )

    name = next(
        (
            candidate
            for candidate in (
                goal_task,
                tool_name,
                orchestrator.name,
                "Avalan Agent",
            )
            if candidate and candidate.strip()
        ),
        "Avalan Agent",
    )

    skill: dict[str, Any] = {
        "id": f"skill-{index + 1}",
        "name": name.strip(),
        "description": " ".join(description_parts),
        "tags": _skill_tags(tool_name, goal_task, orchestrator.name),
    }

    examples = [
        item.strip() for item in goal_instructions if item and item.strip()
    ]
    if examples:
        skill["examples"] = examples

    input_mode = _input_mode_for_spec(spec)
    if input_mode:
        skill["input_modes"] = [input_mode]

    output_mode = _output_mode_for_spec(spec)
    if output_mode:
        skill["output_modes"] = [output_mode]

    return skill


def _default_skill(
    tool_name: str | None,
    tool_description: str | None,
    orchestrator: Orchestrator,
    input_modes: list[str],
    output_modes: list[str],
    examples: list[str],
) -> dict[str, Any]:
    name = next(
        (
            candidate
            for candidate in (
                tool_name,
                orchestrator.name,
                "Avalan Agent",
            )
            if candidate and candidate.strip()
        ),
        "Avalan Agent",
    )
    description_parts: list[str] = []
    _append_unique(description_parts, tool_description)
    _append_unique(description_parts, orchestrator.name)
    if not description_parts:
        _append_unique(description_parts, "Avalan orchestrated agent")

    skill: dict[str, Any] = {
        "id": "skill-1",
        "name": name.strip(),
        "description": " ".join(description_parts),
        "tags": _skill_tags(tool_name, orchestrator.name),
    }
    if examples:
        skill["examples"] = examples
    if input_modes:
        skill["input_modes"] = input_modes
    if output_modes:
        skill["output_modes"] = output_modes
    return skill


def _capability_extensions(
    instructions: list[str], model_ids: Iterable[str] | None
) -> list[dict[str, Any]]:
    extensions: list[dict[str, Any]] = []
    if instructions:
        extensions.append(
            {
                "uri": "https://avalan.ai/extensions/instructions",
                "description": "System and goal instructions for the agent.",
                "params": {"instructions": instructions},
                "required": False,
            }
        )
    if model_ids:
        extensions.append(
            {
                "uri": "https://avalan.ai/extensions/models",
                "description": "Models available to the orchestrated agent.",
                "params": {"models": sorted(model_ids)},
                "required": False,
            }
        )
    return extensions


def _build_agent_card(
    orchestrator: Orchestrator,
    tool_name: str | None,
    tool_description: str | None,
    interface_url: str,
) -> dict[str, Any]:
    instructions: list[str] = []
    skills: list[dict[str, Any]] = []
    default_input_modes: set[str] = set()
    default_output_modes: set[str] = set()

    operations = getattr(orchestrator, "operations", []) or []
    for index, operation in enumerate(operations):
        spec = getattr(operation, "specification", None)
        if spec is None:
            continue
        _append_unique(instructions, getattr(spec, "system_prompt", None))
        _append_unique(instructions, getattr(spec, "developer_prompt", None))
        goal = getattr(spec, "goal", None)
        if goal and getattr(goal, "instructions", None):
            for instruction in goal.instructions:
                _append_unique(instructions, instruction)

        default_input_modes.add(_input_mode_for_spec(spec))
        default_output_modes.add(_output_mode_for_spec(spec))
        skills.append(
            _skill_from_spec(
                index,
                spec,
                tool_name,
                tool_description,
                orchestrator,
            )
        )

    if not default_input_modes:
        default_input_modes.add(_DEFAULT_INPUT_MODE)
    if not default_output_modes:
        default_output_modes.add(_DEFAULT_OUTPUT_MODE)

    input_modes_list = sorted(default_input_modes)
    output_modes_list = sorted(default_output_modes)

    if not skills:
        skills.append(
            _default_skill(
                tool_name,
                tool_description,
                orchestrator,
                input_modes_list,
                output_modes_list,
                list(instructions),
            )
        )

    capabilities: dict[str, Any] = {
        "streaming": True,
        "state_transition_history": True,
    }
    extensions = _capability_extensions(
        list(instructions), getattr(orchestrator, "model_ids", None)
    )
    if extensions:
        capabilities["extensions"] = extensions

    return {
        "id": str(orchestrator.id),
        "name": orchestrator.name or "Avalan Agent",
        "version": "1.0",
        "description": orchestrator.name or "Avalan orchestrated agent",
        "url": interface_url,
        "capabilities": capabilities,
        "default_input_modes": input_modes_list,
        "default_output_modes": output_modes_list,
        "skills": skills,
    }


def _coerce(type_name: str, payload: dict[str, Any]) -> Any:
    cls = getattr(a2a_types, type_name, None)
    if cls is None:
        return payload
    try:
        if hasattr(cls, "model_validate"):
            filtered = _filter_payload(cls, payload)
            return cls.model_validate(filtered)
        return cls(**_filter_payload(cls, payload))
    except Exception:  # pragma: no cover - best effort compatibility
        return payload


def _coerce_list(type_name: str, payload: list[dict[str, Any]]) -> Any:
    cls = getattr(a2a_types, type_name, None)
    if cls is None:
        return payload
    try:
        convert = []
        for item in payload:
            if hasattr(cls, "model_validate"):
                convert.append(cls.model_validate(_filter_payload(cls, item)))
            else:
                convert.append(cls(**_filter_payload(cls, item)))
        return convert
    except Exception:  # pragma: no cover - compatibility fallback
        return payload


def _filter_payload(cls: Any, payload: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(cls, "model_fields", None)
    if not fields:
        return payload
    allowed = set(fields.keys())
    return {key: value for key, value in payload.items() if key in allowed}
