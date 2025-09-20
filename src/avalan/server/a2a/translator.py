from dataclasses import dataclass
from enum import Enum
from time import time
from typing import Any

from ...entities import (
    ReasoningToken,
    ToolCallError,
    ToolCallResult,
    ToolCallToken,
    Token,
    TokenDetail,
)
from ...event import Event, EventType
from ...utils import to_json
from .schema import (
    TASK_STATUSES,
    dump_payload,
    task_status_value,
    validate_event,
    validate_task,
)
from .store import A2AEvent, A2ATask, A2ATaskStore


class StreamState(Enum):
    THINKING = "thinking"
    TOOL = "tool"
    ANSWERING = "answering"


@dataclass(slots=True)
class StreamArtifacts:
    reasoning_id: str
    answer_id: str
    tool_id: str | None = None


class A2ATranslator:
    def __init__(
        self,
        store: A2ATaskStore,
        task: A2ATask,
    ) -> None:
        self._store = store
        self._task = task
        self._state: StreamState | None = None
        self._artifacts = StreamArtifacts(
            reasoning_id=f"{task.id}:assistant:reasoning",
            answer_id=f"{task.id}:assistant:answer",
        )
        self._sequence = 0
        self._reasoning_buffer: list[str] = []
        self._answer_buffer: list[str] = []
        self._tool_inputs: dict[str, list[str]] = {}
        self._tool_outputs: dict[str, Any] = {}

    async def start(self) -> list[A2AEvent]:
        self._store.update_status(self._task.id, TASK_STATUSES.running)
        return [
            self._store.append_event(
                self._task.id,
                validate_event(
                    {
                        "type": "status.changed",
                        "task_id": self._task.id,
                        "status": task_status_value(TASK_STATUSES.running),
                        "timestamp": time(),
                    }
                ),
            )
        ]

    async def token(
        self,
        token: ReasoningToken | ToolCallToken | Token | TokenDetail | Event | str,
    ) -> list[A2AEvent]:
        events: list[A2AEvent] = []
        state = self._state
        new_state = self._detect_state(token)
        events.extend(self._transition(state, new_state))
        self._state = new_state

        if isinstance(token, ReasoningToken):
            self._reasoning_buffer.append(token.token)
            events.append(
                self._store.append_event(
                    self._task.id,
                    validate_event(
                        {
                            "type": "message.delta",
                            "task_id": self._task.id,
                            "message_id": self._artifacts.reasoning_id,
                            "delta": {
                                "type": "text",
                                "text": token.token,
                                "channel": "reasoning",
                            },
                            "sequence": self._sequence,
                        }
                    ),
                )
            )
        elif isinstance(token, Event):
            events.extend(self._handle_event(token))
        elif isinstance(token, ToolCallToken):
            events.extend(self._handle_tool_token(token))
        elif isinstance(token, (Token, TokenDetail)):
            text = token.token
            events.append(self._answer_delta(text))
        elif isinstance(token, str):
            events.append(self._answer_delta(token))

        self._sequence += 1
        return events

    async def finish(self, succeeded: bool = True) -> list[A2AEvent]:
        events: list[A2AEvent] = []
        events.extend(self._transition(self._state, None))
        status = TASK_STATUSES.completed if succeeded else TASK_STATUSES.failed
        self._store.update_status(self._task.id, status)

        summary = {
            "type": "status.changed",
            "task_id": self._task.id,
            "status": task_status_value(status),
            "timestamp": time(),
        }
        events.append(self._store.append_event(self._task.id, validate_event(summary)))

        messages = []
        if self._reasoning_buffer:
            messages.append(
                {
                    "id": self._artifacts.reasoning_id,
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "".join(self._reasoning_buffer),
                            "channel": "reasoning",
                        }
                    ],
                }
            )
        if self._answer_buffer:
            messages.append(
                {
                    "id": self._artifacts.answer_id,
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "".join(self._answer_buffer),
                            "channel": "output",
                        }
                    ],
                }
            )

        self._store.finalize_output(self._task.id, messages)

        task_payload = validate_task(
            {
                "id": self._task.id,
                "status": task_status_value(status),
                "created_at": self._task.created_at,
                "updated_at": time(),
                "input": self._task.input_messages,
                "output": messages,
                "artifacts": list(self._task.artifacts.values()),
                "metadata": self._task.metadata,
            }
        )
        events.append(
            self._store.append_event(
                self._task.id,
                validate_event(
                    {
                        "type": "task.completed" if succeeded else "task.failed",
                        "task": task_payload,
                    }
                ),
            )
        )
        return events

    def _answer_delta(self, text: str) -> A2AEvent:
        self._answer_buffer.append(text)
        payload = validate_event(
            {
                "type": "message.delta",
                "task_id": self._task.id,
                "message_id": self._artifacts.answer_id,
                "delta": {"type": "text", "text": text, "channel": "output"},
                "sequence": self._sequence,
            }
        )
        return self._store.append_event(self._task.id, payload)

    def _handle_event(self, event: Event) -> list[A2AEvent]:
        if event.type is EventType.TOOL_PROCESS:
            return self._handle_tool_process(event)
        if event.type is EventType.TOOL_RESULT:
            return self._handle_tool_result(event)
        if event.type in (
            EventType.TOOL_MODEL_RUN,
            EventType.TOOL_MODEL_RESPONSE,
        ):
            data = {
                "type": "artifact.delta",
                "task_id": self._task.id,
                "artifact": {
                    "id": f"{self._task.id}:tool:model",
                    "kind": "model",
                    "payload": to_json(event.payload),
                },
            }
            artifact = data["artifact"]
            self._store.upsert_artifact(self._task.id, artifact)
            return [self._store.append_event(self._task.id, validate_event(data))]
        return []

    def _handle_tool_process(self, event: Event) -> list[A2AEvent]:
        if not event.payload:
            return []
        events: list[A2AEvent] = []
        for call in event.payload:
            if not call:
                continue
            artifact = {
                "id": str(call.id),
                "kind": "tool_call",
                "name": call.name,
                "arguments": call.arguments,
                "status": "running",
            }
            self._tool_inputs[artifact["id"]] = []
            self._store.upsert_artifact(self._task.id, artifact)
            events.append(
                self._store.append_event(
                    self._task.id,
                    validate_event(
                        {
                            "type": "artifact.delta",
                            "task_id": self._task.id,
                            "artifact": artifact,
                        }
                    ),
                )
            )
        return events

    def _handle_tool_result(self, event: Event) -> list[A2AEvent]:
        if not event.payload or "result" not in event.payload:
            return []
        result = event.payload["result"]
        call = getattr(result, "call", None)
        artifact_id = str(call.id) if call else f"{self._task.id}:tool"
        payload: dict[str, Any] = {
            "id": artifact_id,
            "kind": "tool_result",
            "status": "succeeded",
        }
        if isinstance(result, ToolCallError):
            payload["status"] = "failed"
            payload["error"] = result.message
        elif isinstance(result, ToolCallResult):
            payload["output"] = result.result
        else:
            payload["output"] = result

        if artifact_id in self._tool_inputs:
            payload["input"] = "".join(self._tool_inputs[artifact_id])

        self._tool_outputs[artifact_id] = payload
        self._store.upsert_artifact(self._task.id, payload)
        return [
            self._store.append_event(
                self._task.id,
                validate_event(
                    {
                        "type": "artifact.delta",
                        "task_id": self._task.id,
                        "artifact": payload,
                    }
                ),
            )
        ]

    def _handle_tool_token(self, token: ToolCallToken) -> list[A2AEvent]:
        if token.call is None:
            if not self._artifacts.tool_id:
                self._artifacts.tool_id = f"{self._task.id}:tool-input"
            self._tool_inputs.setdefault(self._artifacts.tool_id, []).append(token.token)
            payload = validate_event(
                {
                    "type": "message.delta",
                    "task_id": self._task.id,
                    "message_id": self._artifacts.tool_id,
                    "delta": {
                        "type": "text",
                        "text": token.token,
                        "channel": "tool",
                    },
                    "sequence": self._sequence,
                }
            )
            return [self._store.append_event(self._task.id, payload)]

        call = token.call
        artifact = {
            "id": str(call.id),
            "kind": "tool_call",
            "name": call.name,
            "arguments": call.arguments,
            "status": "running",
        }
        self._tool_inputs[artifact["id"]] = [token.token]
        self._store.upsert_artifact(self._task.id, artifact)
        return [
            self._store.append_event(
                self._task.id,
                validate_event(
                    {
                        "type": "artifact.delta",
                        "task_id": self._task.id,
                        "artifact": artifact,
                    }
                ),
            )
        ]

    def _detect_state(
        self,
        token: ReasoningToken | ToolCallToken | Token | TokenDetail | Event | str,
    ) -> StreamState | None:
        if isinstance(token, ReasoningToken):
            return StreamState.THINKING
        if isinstance(token, (ToolCallToken, Event)):
            if isinstance(token, Event) and token.type not in (
                EventType.TOOL_PROCESS,
                EventType.TOOL_RESULT,
                EventType.TOOL_MODEL_RUN,
                EventType.TOOL_MODEL_RESPONSE,
            ):
                return self._state
            return StreamState.TOOL
        if token is None:
            return None
        return StreamState.ANSWERING

    def _transition(
        self,
        state: StreamState | None,
        new_state: StreamState | None,
    ) -> list[A2AEvent]:
        events: list[A2AEvent] = []
        if state is not None and state is not new_state:
            payload = validate_event(
                {
                    "type": "message.completed",
                    "task_id": self._task.id,
                    "message_id": self._message_id_for(state),
                    "timestamp": time(),
                }
            )
            events.append(self._store.append_event(self._task.id, payload))
        if new_state is not None and state is not new_state:
            payload = validate_event(
                {
                    "type": "message.started",
                    "task_id": self._task.id,
                    "message_id": self._message_id_for(new_state),
                    "timestamp": time(),
                }
            )
            events.append(self._store.append_event(self._task.id, payload))
        return events

    def _message_id_for(self, state: StreamState) -> str:
        if state is StreamState.THINKING:
            return self._artifacts.reasoning_id
        if state is StreamState.TOOL:
            if not self._artifacts.tool_id:
                self._artifacts.tool_id = f"{self._task.id}:tool"
            return self._artifacts.tool_id
        return self._artifacts.answer_id


def event_to_sse(event: A2AEvent) -> str:
    payload = dump_payload(event.payload)
    payload.setdefault("event_id", event.id)
    payload.setdefault("index", event.index)
    payload.setdefault("timestamp", event.timestamp)
    event_type = payload.get("type", "event")
    return f"event: {event_type}\n" + f"data: {to_json(payload)}\n\n"
