"""FastAPI router exposing Avalan via the A2A protocol."""

from asyncio import CancelledError
from collections.abc import AsyncGenerator, AsyncIterable
from enum import Enum, auto
from logging import Logger
from re import compile
from typing import Any, Iterable
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from a2a import types as a2a_types

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


class A2AResponseTranslator:
    """Convert orchestrator streaming output into A2A task artifacts."""

    def __init__(self, task_id: str, store: TaskStore) -> None:
        self._task_id = task_id
        self._store = store
        self._state: StreamState | None = None
        self._message_id: str | None = None
        self._artifact_id: str | None = None
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
            assert self._message_id
            events.extend(
                await self._store.add_message_delta(
                    self._task_id, self._message_id, item.token
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
            if not self._message_id:
                self._message_id, created = await self._store.ensure_message(
                    self._task_id,
                    role=str(MessageRole.ASSISTANT),
                    channel="output",
                )
                events.extend(created)
            self._answer.append(text)
            events.extend(
                await self._store.add_message_delta(
                    self._task_id, self._message_id, text
                )
            )
        return events

    async def _finish(self) -> list[dict[str, Any]]:
        events = await self._switch_state(None, None)
        events.extend(await self._store.complete_task(self._task_id))
        return events

    async def _switch_state(
        self, state: StreamState | None, call_id: str | None
    ) -> list[dict[str, Any]]:
        changed = self._state is not state or (
            self._state is StreamState.TOOL
            and state is StreamState.TOOL
            and call_id is not None
            and call_id != self._artifact_id
        )
        if not changed:
            return []

        events: list[dict[str, Any]] = []

        if self._state is StreamState.REASONING and self._message_id:
            events.extend(
                await self._store.complete_message(
                    self._task_id, self._message_id
                )
            )
            self._message_id = None
        elif self._state is StreamState.TOOL and self._artifact_id:
            events.extend(
                await self._store.complete_artifact(
                    self._task_id, self._artifact_id
                )
            )
            self._artifact_id = None
        elif self._state is StreamState.ANSWER and self._message_id:
            events.extend(
                await self._store.complete_message(
                    self._task_id, self._message_id
                )
            )
            self._message_id = None

        self._state = state

        if state is StreamState.REASONING:
            self._message_id, created = await self._store.ensure_message(
                self._task_id,
                role=str(MessageRole.ASSISTANT),
                channel="reasoning",
            )
            events.extend(created)
        elif state is StreamState.TOOL:
            artifact = call_id or str(uuid4())
            self._artifact_id, created = await self._store.ensure_artifact(
                self._task_id,
                artifact_id=artifact,
                name=None,
                kind="tool_call",
                role=str(MessageRole.ASSISTANT),
            )
            events.extend(created)
        elif state is StreamState.ANSWER:
            self._message_id, created = await self._store.ensure_message(
                self._task_id,
                role=str(MessageRole.ASSISTANT),
                channel="output",
            )
            events.extend(created)

        return events

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
            self._artifact_id, created = await self._store.ensure_artifact(
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
                    self._artifact_id,
                    {
                        "type": "arguments",
                        "arguments": call.arguments or {},
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
            content = {"type": "result", "content": result.result}
            metadata["status"] = "success"
        elif isinstance(result, ToolCallError):
            artifact_id = str(result.call.id)
            artifact_name = result.call.name
            content = {"type": "error", "error": result.message}
            metadata["status"] = "error"
        elif isinstance(payload, ToolCall):
            artifact_id = str(payload.id)
            artifact_name = payload.name
            content = {"type": "result", "content": None}
        else:
            artifact_id = self._artifact_id
            content = {"type": "result", "content": result}

        if call and isinstance(call, ToolCall):
            artifact_id = str(call.id)
            artifact_name = call.name

        if artifact_id is None:
            artifact_id = str(uuid4())

        self._artifact_id, created = await self._store.ensure_artifact(
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
                self._artifact_id,
                content,
            )
        )
        events.extend(
            await self._store.complete_artifact(
                self._task_id, self._artifact_id
            )
        )
        return events

    async def _handle_tool_token(
        self, token: ToolCallToken
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        if token.call is not None:
            artifact_id = str(token.call.id)
            self._artifact_id, created = await self._store.ensure_artifact(
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
                    self._artifact_id,
                    {
                        "type": "arguments",
                        "arguments": token.call.arguments or {},
                    },
                )
            )
        else:
            if not self._artifact_id:
                self._artifact_id, created = await self._store.ensure_artifact(
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
                    self._artifact_id,
                    {"type": "text", "text": token.token},
                )
            )
        return events


@router.post("/tasks")
async def create_task(
    payload: A2ATaskCreateRequest,
    logger: Logger = Depends(_di_get_logger),
    orchestrator: Orchestrator = Depends(_di_get_orchestrator),
    store: TaskStore = Depends(di_get_task_store),
):
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

    async def stream() -> AsyncGenerator[str, None]:
        try:
            for event in initial_events:
                yield sse_message(
                    to_json(event), event=event.get("event") or "message"
                )
            async for event in translator.run_stream(response):
                yield sse_message(
                    to_json(event), event=event.get("event") or "message"
                )
        except CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive path
            logger.exception(
                "A2A streaming task %s failed", task_id, exc_info=exc
            )
            for event in await store.fail_task(task_id, str(exc)):
                yield sse_message(
                    to_json(event), event=event.get("event") or "message"
                )
        finally:
            yield sse_message(
                to_json(
                    {
                        "event": "task.stream.completed",
                        "task_id": task_id,
                    }
                ),
                event="task.stream.completed",
            )
            yield sse_message("{}", event="done")
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
