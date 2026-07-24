from ...agent.execution import AttachedInteractionRuntime
from ...agent.orchestrator import Orchestrator
from ...entities import (
    MessageRole,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallError,
    ToolCallResult,
    ToolDescriptor,
    ToolValue,
)
from ...interaction.entities import PrincipalScope, TaskId
from ...interaction.policy import InteractionActor
from ...model.stream import (
    CanonicalStreamItem,
    StreamConsumerProjection,
    StreamItemKind,
    StreamRetentionPolicy,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
)
from ...server.entities import (
    ChatCompletionRequest,
    ChatMessage,
    ContentFile,
    ContentImage,
    ContentText,
    MCPToolRequest,
    ModelVisibleServerProtocolTextRedactor,
    ServerOutputRedactionSettings,
    coerce_server_output_redaction_settings,
    sanitize_model_visible_server_protocol_text,
    sanitize_server_protocol_text,
    sanitize_server_protocol_value,
    server_output_redaction_settings_from_state,
)
from ...types import JsonObject, JsonScalar, MutableJsonValue
from ...utils import to_json
from ..container_policy import (
    RemoteContainerRequestError,
    remote_container_policy_from_state,
    validate_remote_container_arguments,
)
from ..interaction import (
    ServerInteractionHTTPError,
    ServerInteractionService,
)
from ..mcp_session import (
    MCPFormElicitationOutbound,
    MCPFormErrorCode,
    MCPFormSessionError,
    MCPFormSessionRegistry,
    MCPFormSessionView,
    MCPFormStatus,
    MCPFormStatusEvent,
    MCPFormStatusHook,
)
from ..mcp_tasks import (
    MCPTaskController,
    MCPTaskHandle,
    MCPTaskOutcome,
    MCPTaskProtocolError,
    MCPTaskRequest,
    with_related_task_metadata,
    without_related_task_metadata,
)
from ..sse import sse_bytes, sse_headers
from . import (
    MODEL_FALLBACK as DEFAULT_MODEL_FALLBACK,
)
from . import (
    orchestrate,
    resolve_model_id,
)
from .streaming import (
    ProtocolReasoningAdmission,
    ProtocolReasoningIdentity,
    ProtocolReasoningRedactedText,
    ProtocolReasoningRedactionState,
    ProtocolStreamAccumulator,
    ProtocolStreamSnapshot,
    cancellable_stream_iterator,
    canonical_flow_public_metadata,
    cleanup_stream_sources,
    protocol_stream_retention_settings,
    protocol_stream_usage_mappings,
    stream_consumer_iterator,
)

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Lock,
    Task,
    create_task,
    gather,
    wait,
)
from asyncio import Event as AsyncEvent
from collections import deque
from contextlib import suppress
from dataclasses import dataclass, field, replace
from json import JSONDecodeError, dumps, loads
from logging import Logger
from typing import (
    AsyncGenerator,
    AsyncIterator,
    Final,
    Iterator,
    Literal,
    Mapping,
    Protocol,
    TypeAlias,
    TypedDict,
    cast,
)
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from mcp.types import LATEST_PROTOCOL_VERSION

RS: Final[str] = "\x1e"
MODEL_FALLBACK: Final[str] = DEFAULT_MODEL_FALLBACK

JSONScalar: TypeAlias = JsonScalar
JSONValue: TypeAlias = MutableJsonValue
JSONObject: TypeAlias = JsonObject
MCPCancellationKey: TypeAlias = tuple[PrincipalScope, str, str | int]

Method = Literal[
    "initialize",
    "ping",
    "tasks/cancel",
    "tasks/get",
    "tasks/list",
    "tasks/result",
    "tools/list",
    "tools/call",
]
NotificationMethod = Literal[
    "notifications/cancelled",
    "notifications/initialized",
    "notifications/message",
]
AllowedMethod = Method | NotificationMethod

ResponseItem = CanonicalStreamItem | StreamConsumerProjection


@dataclass(frozen=True, slots=True)
class _MCPSessionContext:
    session_id: str
    owner: PrincipalScope
    actor: InteractionActor
    service: ServerInteractionService
    registry: MCPFormSessionRegistry
    negotiation: MCPFormSessionView

    @property
    def task_requestor(self) -> tuple[PrincipalScope, str] | None:
        if not _identifiable_principal(self.owner):
            return None
        return (self.owner, self.session_id)


def _default_model_id(orchestrator: Orchestrator) -> str:
    return resolve_model_id(orchestrator)


class JSONRPCRequest(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: str | int
    method: str
    params: JSONObject | None


class JSONRPCResult(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: str | int
    result: JSONObject


class JSONRPCError(TypedDict, total=False):
    jsonrpc: Literal["2.0"]
    id: str | int
    error: dict[str, JSONValue]


class SupportsAclose(Protocol):
    async def aclose(self) -> None: ...


@dataclass(slots=True)
class MCPResource:
    id: str
    uri: str
    http_uri: str
    mime_type: str
    text: str
    revision: int
    closed: bool = False


class MCPResourceStore:
    def __init__(
        self,
        resource_item_limit: int | None = None,
        resource_limit: int | None = None,
        resource_byte_limit: int | None = None,
    ) -> None:
        retention_settings = protocol_stream_retention_settings(
            StreamRetentionPolicy()
        )
        if resource_item_limit is None:
            resource_item_limit = retention_settings.resource_item_limit
        if resource_limit is None:
            resource_limit = retention_settings.resource_item_limit
        if resource_byte_limit is None:
            resource_byte_limit = retention_settings.resource_text_byte_limit
        assert isinstance(resource_item_limit, int)
        assert not isinstance(resource_item_limit, bool)
        assert resource_item_limit >= 0
        assert isinstance(resource_limit, int)
        assert not isinstance(resource_limit, bool)
        assert resource_limit > 0
        assert isinstance(resource_byte_limit, int)
        assert not isinstance(resource_byte_limit, bool)
        assert resource_byte_limit > 0
        self._resources: dict[str, MCPResource] = {}
        self._resource_chunks: dict[str, list[str]] = {}
        self._resource_order: list[str] = []
        self._counter = 0
        self._resource_item_limit = resource_item_limit
        self._resource_limit = resource_limit
        self._resource_byte_limit = resource_byte_limit
        self._lock = Lock()

    async def create(
        self,
        *,
        base_path: str,
        mime_type: str = "text/plain",
        initial_text: str = "",
    ) -> MCPResource:
        async with self._lock:
            self._counter += 1
            resource_id = f"{self._counter:08x}"
            uri = f"mcp://resources/{resource_id}"
            http_uri = f"{base_path}/resources/{resource_id}"
            resource = MCPResource(
                id=resource_id,
                uri=uri,
                http_uri=http_uri,
                mime_type=mime_type,
                text=self._retained_text([initial_text]),
                revision=1 if initial_text else 0,
            )
            self._resources[resource_id] = resource
            self._resource_chunks[resource_id] = (
                self._retained_chunks([initial_text]) if initial_text else []
            )
            self._resource_order.append(resource_id)
            self._enforce_resource_retention(
                protected_resource_ids={resource_id}
            )
            return replace(resource)

    async def append(self, resource_id: str, text: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            chunks = self._resource_chunks.setdefault(resource_id, [])
            chunks.append(text)
            retained_chunks = self._retained_chunks(chunks)
            self._resource_chunks[resource_id] = retained_chunks
            resource.text = self._retained_text(retained_chunks)
            resource.revision += 1
            self._resources[resource_id] = resource
            return replace(resource)

    async def close(self, resource_id: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            if not resource.closed:
                resource.closed = True
                resource.revision += 1
                self._resources[resource_id] = resource
            self._enforce_resource_retention(protected_resource_ids=set())
            return replace(resource)

    async def close_many(
        self, resource_ids: list[str]
    ) -> tuple[MCPResource, ...]:
        assert isinstance(resource_ids, list)
        for resource_id in resource_ids:
            assert isinstance(resource_id, str)
        protected_resource_ids = set(resource_ids)
        async with self._lock:
            closed_resources: list[MCPResource] = []
            for resource_id in resource_ids:
                resource = self._ensure(resource_id)
                if not resource.closed:
                    resource.closed = True
                    resource.revision += 1
                    self._resources[resource_id] = resource
                closed_resources.append(replace(resource))
            self._enforce_resource_retention(
                protected_resource_ids=protected_resource_ids
            )
            return tuple(closed_resources)

    async def prune_closed(self) -> None:
        async with self._lock:
            self._enforce_resource_retention(protected_resource_ids=set())

    async def get(self, resource_id: str) -> MCPResource:
        async with self._lock:
            resource = self._ensure(resource_id)
            return replace(resource)

    async def history(self, resource_id: str) -> tuple[str, ...]:
        async with self._lock:
            self._ensure(resource_id)
            return tuple(self._resource_chunks.get(resource_id, ()))

    def _ensure(self, resource_id: str) -> MCPResource:
        if resource_id not in self._resources:
            raise KeyError(resource_id)
        return self._resources[resource_id]

    def _retained_chunks(self, chunks: list[str]) -> list[str]:
        if self._resource_item_limit == 0:
            return []
        retained = list(chunks[-self._resource_item_limit :])
        bounded_reversed: list[str] = []
        remaining_bytes = self._resource_byte_limit

        for chunk in reversed(retained):
            chunk_size = len(chunk.encode("utf-8"))
            if chunk_size <= remaining_bytes:
                bounded_reversed.append(chunk)
                remaining_bytes -= chunk_size
                if remaining_bytes == 0:
                    break
                continue
            bounded = self._utf8_suffix(chunk, remaining_bytes)
            if bounded:
                bounded_reversed.append(bounded)
            break

        bounded_reversed.reverse()
        return bounded_reversed

    def _retained_text(self, chunks: list[str]) -> str:
        return "".join(self._retained_chunks(chunks))

    @staticmethod
    def _utf8_suffix(text: str, byte_limit: int) -> str:
        assert isinstance(byte_limit, int)
        assert not isinstance(byte_limit, bool)
        assert byte_limit > 0
        data = text.encode("utf-8")
        if len(data) <= byte_limit:
            return text

        start = len(data) - byte_limit
        while start < len(data) and (data[start] & 0b1100_0000) == 0b1000_0000:
            start += 1
        return data[start:].decode("utf-8")

    def _enforce_resource_retention(
        self, *, protected_resource_ids: set[str]
    ) -> None:
        assert isinstance(protected_resource_ids, set)
        while len(self._resources) > self._resource_limit:
            evicted = False
            for resource_id in list(self._resource_order):
                if resource_id in protected_resource_ids:
                    continue
                resource = self._resources.get(resource_id)
                if resource is None:
                    self._resource_order.remove(resource_id)
                    evicted = True
                    break
                if resource.closed:
                    self._resources.pop(resource_id, None)
                    self._resource_chunks.pop(resource_id, None)
                    self._resource_order.remove(resource_id)
                    evicted = True
                    break
            if not evicted:
                return


@dataclass(slots=True)
class _MCPReasoningSegment:
    identity: ProtocolReasoningIdentity
    chunks: deque[str] = field(default_factory=deque)
    characters: int = 0
    utf8_bytes: int = 0
    separator_after: str = ""
    leading_partial: bool = False
    completed: bool = False
    status: str = "in_progress"
    terminal_outcome: str | None = None

    def append(self, text: str) -> None:
        assert isinstance(text, str) and text
        self.chunks.append(text)
        self.characters += len(text)
        self.utf8_bytes += len(text.encode("utf-8"))

    def text(self) -> str:
        return "".join(self.chunks)

    def trailing_line_feeds(self) -> int:
        count = 0
        for chunk in reversed(self.chunks):
            exhausted = True
            for character in reversed(chunk):
                if not character.isspace():
                    exhausted = False
                    break
                if character == "\n":
                    count += 1
            if not exhausted:
                break
        return count

    def trim_prefix(
        self,
        minimum_characters: int,
        minimum_utf8_bytes: int,
    ) -> tuple[int, int]:
        assert minimum_characters >= 0
        assert minimum_utf8_bytes >= 0
        removed_characters = 0
        removed_utf8_bytes = 0
        while self.chunks and (
            removed_characters < minimum_characters
            or removed_utf8_bytes < minimum_utf8_bytes
        ):
            chunk = self.chunks.popleft()
            position = 0
            while position < len(chunk) and (
                removed_characters < minimum_characters
                or removed_utf8_bytes < minimum_utf8_bytes
            ):
                character = chunk[position]
                removed_characters += 1
                removed_utf8_bytes += len(character.encode("utf-8"))
                position += 1
            if position < len(chunk):
                self.chunks.appendleft(chunk[position:])
        if removed_characters:
            self.characters -= removed_characters
            self.utf8_bytes -= removed_utf8_bytes
            self.leading_partial = True
        return removed_characters, removed_utf8_bytes

    def close(self, outcome: StreamTerminalOutcome) -> None:
        assert isinstance(outcome, StreamTerminalOutcome)
        assert self.status == "in_progress"
        if outcome is StreamTerminalOutcome.COMPLETED:
            self.completed = True
            self.status = "completed"
            self.terminal_outcome = "completed"
            return
        self.completed = False
        self.status = "incomplete"
        terminal_outcomes = {
            StreamTerminalOutcome.ERRORED: "failed",
            StreamTerminalOutcome.CANCELLED: "cancelled",
            StreamTerminalOutcome.INPUT_REQUIRED: "input_required",
        }
        self.terminal_outcome = terminal_outcomes[outcome]


class _MCPReasoningOwner:
    def __init__(
        self,
        output_redaction_settings: ServerOutputRedactionSettings,
        *,
        retention_policy: StreamRetentionPolicy | None = None,
    ) -> None:
        assert isinstance(
            output_redaction_settings, ServerOutputRedactionSettings
        )
        assert retention_policy is None or isinstance(
            retention_policy, StreamRetentionPolicy
        )
        policy = retention_policy or StreamRetentionPolicy()
        self._segment_limit = policy.mcp_reasoning_segment_limit
        self._character_limit = policy.mcp_reasoning_character_limit
        self._utf8_byte_limit = policy.mcp_reasoning_text_byte_limit
        self._redaction = ProtocolReasoningRedactionState(
            output_redaction_settings,
            protocol="mcp",
        )
        self._segments: deque[_MCPReasoningSegment] = deque()
        self._active: _MCPReasoningSegment | None = None
        self._input_identity: ProtocolReasoningIdentity | None = None
        self._rejected_identity: ProtocolReasoningIdentity | None = None
        self._characters = 0
        self._utf8_bytes = 0
        self._dropped_segments = 0
        self._dropped_characters = 0
        self._dropped_utf8_bytes = 0
        self._terminal_outcome: StreamTerminalOutcome | None = None

    @property
    def redaction(self) -> ProtocolReasoningRedactionState:
        return self._redaction

    @property
    def segments(self) -> tuple[_MCPReasoningSegment, ...]:
        return tuple(
            segment for segment in self._segments if segment.characters
        )

    @property
    def truncated(self) -> bool:
        return bool(
            self._dropped_segments
            or self._dropped_characters
            or self._dropped_utf8_bytes
        )

    def push(
        self,
        item: CanonicalStreamItem,
    ) -> tuple[ProtocolReasoningRedactedText, ...]:
        assert isinstance(item, CanonicalStreamItem)
        assert item.kind is StreamItemKind.REASONING_DELTA
        assert item.text_delta is not None
        assert self._terminal_outcome is None
        identity = ProtocolReasoningIdentity.from_item(item)
        value = item.text_delta

        if self._rejected_identity == identity:
            self._record_rejected_text(value)
            return ()
        if self._rejected_identity is not None:
            self._rejected_identity = None

        new_identity = identity != self._input_identity
        admission = self._redaction.preview_push(identity, value)
        if admission.suppressed:
            if admission.marker_reserved:
                assert self._make_capacity(
                    admission,
                    separator_characters=0,
                    new_segment=False,
                )
            outputs = self._redaction.push(identity, value)
            self._append_outputs(outputs)
            if new_identity:
                self._close_active(StreamTerminalOutcome.COMPLETED)
            self._input_identity = identity
            return outputs

        if new_identity:
            self._close_active(StreamTerminalOutcome.COMPLETED)
        new_segment = self._active is None or self._active.identity != identity
        separator_characters = (
            self._separator_characters_before(value) if new_segment else 0
        )
        if not self._make_capacity(
            admission,
            separator_characters=separator_characters,
            new_segment=new_segment,
        ):
            outputs = self._resolve_before_rejection(identity)
            self._record_rejected_text(
                value,
                dropped_segment=not self._has_segment(identity),
            )
            if not self._redaction.redaction_latched:
                self._rejected_identity = identity
            self._input_identity = identity
            return outputs

        outputs = self._redaction.push(identity, value)
        self._append_outputs(outputs)
        self._input_identity = identity
        return outputs

    def complete(
        self,
        outcome: StreamTerminalOutcome = StreamTerminalOutcome.COMPLETED,
    ) -> tuple[
        tuple[ProtocolReasoningRedactedText, ...],
        _MCPReasoningSegment | None,
    ]:
        assert isinstance(outcome, StreamTerminalOutcome)
        if self._terminal_outcome is not None:
            return (), None
        identity = self._input_identity or self._redaction.identity
        outputs = self._redaction.complete(identity)
        self._append_outputs(outputs)
        closed = self._close_active(outcome)
        self._input_identity = None
        self._rejected_identity = None
        return outputs, closed

    def finish(
        self,
        outcome: StreamTerminalOutcome,
    ) -> tuple[
        tuple[ProtocolReasoningRedactedText, ...],
        _MCPReasoningSegment | None,
    ]:
        assert isinstance(outcome, StreamTerminalOutcome)
        if self._terminal_outcome is not None:
            return (), None
        outputs, closed = self.complete(outcome)
        self._terminal_outcome = outcome
        return outputs, closed

    def final_payload(self) -> dict[str, JSONValue]:
        assert self._terminal_outcome is StreamTerminalOutcome.COMPLETED
        segments = [
            self._segment_payload(segment) for segment in self.segments
        ]
        if not segments and not self.truncated:
            return {}
        return {
            "reasoning": self._flat_text(),
            "reasoningSegments": cast(JSONValue, segments),
            "reasoningTruncation": {
                "truncated": self.truncated,
                "dropped_segments": self._dropped_segments,
                "dropped_characters": self._dropped_characters,
                "dropped_utf8_bytes": self._dropped_utf8_bytes,
                "leading_segment_partial": (
                    self.segments[0].leading_partial
                    if self.segments
                    else False
                ),
            },
        }

    def _make_capacity(
        self,
        admission: ProtocolReasoningAdmission,
        *,
        separator_characters: int,
        new_segment: bool,
    ) -> bool:
        assert isinstance(admission, ProtocolReasoningAdmission)
        assert isinstance(separator_characters, int)
        assert not isinstance(separator_characters, bool)
        assert separator_characters >= 0
        required_characters = (
            admission.required_character_count + separator_characters
        )
        required_bytes = (
            admission.required_utf8_byte_count + separator_characters
        )
        if (
            required_characters > self._character_limit
            or required_bytes > self._utf8_byte_limit
            or new_segment
            and self._segment_limit == 0
        ):
            return False

        if new_segment and separator_characters:
            self._reserve_separator(separator_characters)
            required_characters = admission.required_character_count
            required_bytes = admission.required_utf8_byte_count

        if new_segment:
            while len(self._segments) >= self._segment_limit:
                assert self._drop_oldest_completed()

        while self._over_limit(required_characters, required_bytes):
            if self._drop_oldest_completed():
                continue
            assert self._trim_oldest(required_characters, required_bytes)
        return True

    def _append_outputs(
        self,
        outputs: tuple[ProtocolReasoningRedactedText, ...],
    ) -> None:
        for output in outputs:
            segment = self._segment_for(output.identity)
            segment.append(output.text)
            self._characters += len(output.text)
            self._utf8_bytes += len(output.text.encode("utf-8"))
            assert not self._over_limit(0, 0)

    def _segment_for(
        self,
        identity: ProtocolReasoningIdentity,
    ) -> _MCPReasoningSegment:
        active = self._active
        assert active is None or active.identity == identity
        if active is not None:
            return active
        assert len(self._segments) < self._segment_limit
        segment = _MCPReasoningSegment(identity=identity)
        self._segments.append(segment)
        self._active = segment
        return segment

    def _reserve_separator(self, characters: int) -> None:
        assert isinstance(characters, int) and not isinstance(characters, bool)
        assert characters > 0
        previous = next(
            (
                segment
                for segment in reversed(self._segments)
                if segment.characters
            ),
            None,
        )
        assert previous is not None
        assert not previous.separator_after
        previous.separator_after = "\n" * characters
        self._characters += characters
        self._utf8_bytes += characters

    def _separator_characters_before(self, text: str) -> int:
        previous = next(
            (
                segment
                for segment in reversed(self._segments)
                if segment.characters
            ),
            None,
        )
        return (
            self._missing_separator_characters(previous, text)
            if previous is not None
            else 0
        )

    @staticmethod
    def _missing_separator_characters(
        previous: _MCPReasoningSegment,
        text: str,
    ) -> int:
        leading = 0
        for character in text:
            if not character.isspace():
                break
            if character == "\n":
                leading += 1
        return max(0, 2 - previous.trailing_line_feeds() - leading)

    def _over_limit(
        self,
        reserved_characters: int,
        reserved_bytes: int,
    ) -> bool:
        return (
            self._characters + reserved_characters > self._character_limit
            or self._utf8_bytes + reserved_bytes > self._utf8_byte_limit
        )

    def _drop_oldest_completed(self) -> bool:
        if not self._segments or self._segments[0].status == "in_progress":
            return False
        segment = self._segments.popleft()
        assert self._active is not segment
        separator_characters = len(segment.separator_after)
        separator_bytes = len(segment.separator_after.encode("utf-8"))
        self._characters -= segment.characters + separator_characters
        self._utf8_bytes -= segment.utf8_bytes + separator_bytes
        self._dropped_segments += 1
        self._dropped_characters += segment.characters + separator_characters
        self._dropped_utf8_bytes += segment.utf8_bytes + separator_bytes
        return True

    def _trim_oldest(
        self,
        reserved_characters: int,
        reserved_bytes: int,
    ) -> bool:
        assert self._segments
        oldest = self._segments[0]
        required_characters = max(
            0,
            self._characters + reserved_characters - self._character_limit,
        )
        required_bytes = max(
            0,
            self._utf8_bytes + reserved_bytes - self._utf8_byte_limit,
        )
        removed_characters, removed_bytes = oldest.trim_prefix(
            required_characters,
            required_bytes,
        )
        assert removed_characters
        self._characters -= removed_characters
        self._utf8_bytes -= removed_bytes
        self._dropped_characters += removed_characters
        self._dropped_utf8_bytes += removed_bytes
        return True

    def _close_active(
        self,
        outcome: StreamTerminalOutcome,
    ) -> _MCPReasoningSegment | None:
        active = self._active
        if active is None:
            return None
        active.close(outcome)
        self._active = None
        return active

    def _resolve_before_rejection(
        self,
        identity: ProtocolReasoningIdentity,
    ) -> tuple[ProtocolReasoningRedactedText, ...]:
        outputs = self._redaction.complete(identity)
        self._append_outputs(outputs)
        self._close_active(StreamTerminalOutcome.COMPLETED)
        return outputs

    def _record_rejected_text(
        self,
        text: str,
        *,
        dropped_segment: bool = False,
    ) -> None:
        assert isinstance(text, str) and text
        assert isinstance(dropped_segment, bool)
        if dropped_segment:
            self._dropped_segments += 1
        self._dropped_characters += len(text)
        self._dropped_utf8_bytes += len(text.encode("utf-8"))

    def _has_segment(self, identity: ProtocolReasoningIdentity) -> bool:
        return any(segment.identity == identity for segment in self._segments)

    def _flat_text(self) -> str:
        parts: list[str] = []
        for segment in self.segments:
            parts.append(segment.text())
            if segment.separator_after:
                parts.append(segment.separator_after)
        return "".join(parts)

    @staticmethod
    def _segment_payload(
        segment: _MCPReasoningSegment,
    ) -> dict[str, JSONValue]:
        identity = segment.identity
        payload: dict[str, JSONValue] = {
            "representation": identity.representation.value,
            "segment_instance_ordinal": identity.segment_instance_ordinal,
            "text": segment.text(),
            "completed": segment.completed,
            "status": segment.status,
            "terminal_outcome": segment.terminal_outcome,
        }
        for field_name, value in (
            ("provider_item_id", identity.provider_item_id),
            ("output_index", identity.output_index),
            ("summary_index", identity.summary_index),
            ("continuation_id", identity.continuation_id),
        ):
            if value is not None:
                payload[field_name] = value
        return payload


@dataclass(slots=True)
class _MCPStreamProjectionState:
    accumulator: ProtocolStreamAccumulator
    tool_summaries: dict[str, dict[str, JSONValue]]
    resources: dict[str, MCPResource]
    resource_store: MCPResourceStore
    base_path: str
    output_redaction_settings: ServerOutputRedactionSettings = field(
        default_factory=ServerOutputRedactionSettings
    )
    answer_redactor: ModelVisibleServerProtocolTextRedactor = field(
        default_factory=ModelVisibleServerProtocolTextRedactor
    )
    reasoning: _MCPReasoningOwner | None = None

    def __post_init__(self) -> None:
        if self.reasoning is None:
            self.reasoning = _MCPReasoningOwner(self.output_redaction_settings)

    @property
    def reasoning_owner(self) -> _MCPReasoningOwner:
        assert isinstance(self.reasoning, _MCPReasoningOwner)
        return self.reasoning


def create_router() -> APIRouter:
    from .. import di_get_logger, di_get_orchestrator

    router = APIRouter(tags=["mcp"])

    @router.post("", response_model=None)
    @router.post("/", response_model=None)
    async def mcp_rpc(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> Response:
        assert isinstance(logger, Logger)
        assert isinstance(orchestrator, Orchestrator)

        message, messages = await _first_jsonrpc_message(request)
        method = message.get("method")
        if method is None:
            try:
                session = await _mcp_session_context(request, required=True)
                assert session is not None
                await session.registry.dispatch_response(
                    session.session_id,
                    session.owner,
                    message,
                )
            except (MCPFormSessionError, MCPTaskProtocolError) as exc:
                return _mcp_protocol_error_response(message, exc)
            return Response(status_code=202)
        if method not in {
            "initialize",
            "ping",
            "tasks/cancel",
            "tasks/get",
            "tasks/list",
            "tasks/result",
            "tools/list",
            "tools/call",
            "notifications/cancelled",
            "notifications/initialized",
        }:
            raise HTTPException(
                status_code=400, detail=f"Unsupported MCP method {method}"
            )
        if method == "initialize":
            return await _initialize_mcp_session(
                request, logger, orchestrator, message
            )
        try:
            session = await _mcp_session_context(request)
        except (MCPFormSessionError, MCPTaskProtocolError) as exc:
            return _mcp_protocol_error_response(message, exc)
        if method == "ping":
            return _handle_ping_message(logger, message)
        if method == "notifications/cancelled":
            if session is None:
                return _mcp_protocol_error_response(
                    message,
                    _mcp_session_unavailable(),
                )
            return _handle_cancelled_notification(
                request,
                logger,
                message,
                session,
            )
        if method.startswith("tasks/"):
            try:
                if session is None:
                    raise MCPTaskProtocolError(
                        code=-32001,
                        message="MCP task session is unavailable.",
                    )
                return await _handle_task_message(
                    request,
                    message,
                    session,
                )
            except (MCPFormSessionError, MCPTaskProtocolError) as exc:
                return _mcp_protocol_error_response(message, exc)
        if method == "tools/list":
            request.state.mcp_task_capable = (
                session is not None and session.task_requestor is not None
            )
            return _handle_list_tools_message(
                request, logger, orchestrator, message
            )
        if method == "tools/call":
            try:
                direct = _is_direct_skills_tool_call(orchestrator, message)
                task_request = _mcp_task_request(
                    request,
                    message,
                    session,
                    execution_task_support=(
                        "forbidden" if direct else "optional"
                    ),
                )
                if direct:
                    return await _handle_direct_skills_tool_call_message(
                        request, logger, orchestrator, message
                    )
                request_id, responses_request, progress_token = (
                    _parse_call_request(request, message, messages)
                )
                if task_request is not None:
                    assert session is not None
                    return await _start_tool_task_response(
                        request,
                        logger,
                        orchestrator,
                        request_id,
                        responses_request,
                        progress_token,
                        session,
                        task_request,
                    )
                return await _start_tool_streaming_response(
                    request,
                    logger,
                    orchestrator,
                    request_id,
                    responses_request,
                    progress_token,
                    session=session,
                )
            except (MCPFormSessionError, MCPTaskProtocolError) as exc:
                return _mcp_protocol_error_response(message, exc)
        assert method == "notifications/initialized"
        if session is not None:
            try:
                await session.registry.mark_initialized(
                    session.session_id,
                    session.owner,
                )
            except MCPFormSessionError as exc:
                return _mcp_protocol_error_response(message, exc)
        return _handle_initialized_notification(logger, message)

    @router.post("/initialize")
    async def mcp_initialize(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> JSONResponse:
        assert isinstance(logger, Logger)
        assert isinstance(orchestrator, Orchestrator)

        message, _ = await _expect_jsonrpc_message(request, {"initialize"})
        return await _initialize_mcp_session(
            request, logger, orchestrator, message
        )

    @router.post("/ping")
    async def mcp_ping(
        request: Request,
        logger: Logger = Depends(di_get_logger),
    ) -> JSONResponse:
        assert isinstance(logger, Logger)
        message, _ = await _expect_jsonrpc_message(request, {"ping"})
        return _handle_ping_message(logger, message)

    @router.post("/tools/list")
    async def mcp_list_tools(
        request: Request,
        logger: Logger = Depends(di_get_logger),
        orchestrator: Orchestrator = Depends(di_get_orchestrator),
    ) -> JSONResponse:
        assert isinstance(logger, Logger)
        assert isinstance(orchestrator, Orchestrator)

        message, _ = await _expect_jsonrpc_message(request, {"tools/list"})
        session = await _mcp_session_context(request)
        request.state.mcp_task_capable = (
            session is not None and session.task_requestor is not None
        )
        return _handle_list_tools_message(
            request, logger, orchestrator, message
        )

    @router.get("/resources/{resource_id}")
    async def mcp_get_resource(
        request: Request, resource_id: str
    ) -> PlainTextResponse:
        store = _get_resource_store(request)
        try:
            resource = await store.get(resource_id)
        except KeyError as exc:  # pragma: no cover - FastAPI handles
            raise HTTPException(
                status_code=404, detail="Resource not found"
            ) from exc
        return PlainTextResponse(resource.text, media_type=resource.mime_type)

    @router.post("/notifications/initialized")
    async def mcp_initialized_notification(
        request: Request,
        logger: Logger = Depends(di_get_logger),
    ) -> Response:
        assert isinstance(logger, Logger)
        message, _ = await _expect_jsonrpc_message(
            request, {"notifications/initialized"}
        )
        session = await _mcp_session_context(request)
        if session is not None:
            await session.registry.mark_initialized(
                session.session_id,
                session.owner,
            )
        return _handle_initialized_notification(logger, message)

    @router.delete("", response_model=None)
    @router.delete("/", response_model=None)
    async def mcp_close_session(request: Request) -> Response:
        session = await _mcp_session_context(request, required=True)
        assert session is not None
        _cancel_mcp_session_requests(request.app, session)
        await session.registry.close(session.session_id, session.owner)
        if session.task_requestor is not None:
            await _cancel_mcp_background_tasks(
                request.app,
                session.task_requestor,
            )
            await _get_task_controller(request).cleanup_requestor(
                session.task_requestor
            )
        return Response(status_code=204)

    return router


async def _consume_call_request(
    request: Request,
) -> tuple[str | int, MCPToolRequest, str | int]:
    call_message, messages = await _expect_jsonrpc_message(
        request, {"tools/call"}
    )
    return _parse_call_request(request, call_message, messages)


def _parse_call_request(
    request: Request,
    call_message: JSONObject,
    messages: AsyncIterator[JSONObject],
) -> tuple[str | int, MCPToolRequest, str | int]:
    method = call_message.get("method")
    if method != "tools/call":
        raise HTTPException(
            status_code=400, detail=f'Unsupported MCP method "{method}"'
        )

    params = call_message.get("params")
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    allowed_tool_name = cast(
        str, getattr(request.app.state, "mcp_tool_name", "run")
    )
    arguments = _extract_call_arguments(
        method, params, allowed_tool_name=allowed_tool_name
    )
    if not isinstance(arguments, dict):
        raise HTTPException(status_code=400, detail="Invalid tool arguments")

    try:
        container_request = validate_remote_container_arguments(
            arguments,
            policy=remote_container_policy_from_state(request.app.state),
        )
    except RemoteContainerRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    arguments = cast(dict[str, JSONValue], container_request.arguments)
    if container_request.profile is not None:
        request.state.mcp_container_profile = container_request.profile

    try:
        request_model = MCPToolRequest.model_validate(arguments)
    except Exception as exc:  # pragma: no cover - validation error path
        raise HTTPException(
            status_code=400, detail="Invalid MCP arguments"
        ) from exc

    progress_token = cast(str | int | None, params.get("progressToken"))
    if progress_token is None:
        meta = params.get("_meta")
        if isinstance(meta, dict):
            progress_token = cast(str | int | None, meta.get("progressToken"))
    if not progress_token:
        progress_token = str(uuid4())

    request.state._mcp_message_iter = messages
    return (
        cast(str | int, call_message.get("id", str(uuid4()))),
        request_model,
        progress_token,
    )


def _mcp_task_request(
    request: Request,
    message: Mapping[str, object],
    session: _MCPSessionContext | None,
    *,
    execution_task_support: Literal["forbidden", "optional"],
) -> MCPTaskRequest | None:
    params = message.get("params")
    if not isinstance(params, Mapping):
        raise MCPTaskProtocolError(
            code=-32602,
            message="Missing MCP params.",
        )
    if session is None or session.task_requestor is None:
        return None
    return _get_task_controller(request).task_request(
        params,
        request_type="tools/call",
        execution_task_support=execution_task_support,
        requestor=session.task_requestor,
    )


async def _expect_jsonrpc_message(
    request: Request, allowed_methods: set[AllowedMethod]
) -> tuple[JSONObject, AsyncIterator[JSONObject]]:
    message, messages = await _first_jsonrpc_message(request)
    method = cast(str | None, message.get("method"))
    if method not in allowed_methods:
        raise HTTPException(
            status_code=400, detail=f"Unsupported MCP method {method}"
        )

    return message, messages


async def _first_jsonrpc_message(
    request: Request,
) -> tuple[JSONObject, AsyncIterator[JSONObject]]:
    messages = _iter_jsonrpc_messages(request)
    try:
        message = await anext(messages)
    except (
        StopAsyncIteration
    ) as exc:  # pragma: no cover - defensive validation
        raise HTTPException(
            status_code=400, detail="Empty MCP request"
        ) from exc

    if not isinstance(message, dict):
        raise HTTPException(status_code=400, detail="Invalid MCP payload")

    return message, messages


def _server_info(request: Request) -> JSONObject:
    app = request.app
    name = getattr(app, "title", None) or "avalan"
    version = getattr(app, "version", None)
    if version is None:
        version = getattr(app.state, "version", None)
    if version is None:
        version = "0.0.0"
    return {"name": str(name), "version": str(version)}


def _server_capabilities(
    orchestrator: Orchestrator,
    *,
    task_controller: MCPTaskController | None = None,
    task_requestor: tuple[PrincipalScope, str] | None = None,
) -> dict[str, JSONValue]:
    capabilities: dict[str, JSONValue] = {
        "tools": {
            "list": True,
            "call": True,
            "listChanged": False,
        },
        "resources": {
            "subscribe": True,
            "listChanged": False,
        },
    }
    if task_controller is not None and task_requestor is not None:
        capabilities.update(
            task_controller.capability_dict(requestor=task_requestor)
        )
    return capabilities


async def _initialize_mcp_session(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    params_obj: JSONObject = params if isinstance(params, dict) else {}
    protocol_version = _negotiated_protocol_version(
        params_obj.get("protocolVersion")
    )
    capabilities = _server_capabilities(orchestrator)
    headers: dict[str, str] = {}
    actor_and_service = await _optional_mcp_actor(request)
    if (
        protocol_version == LATEST_PROTOCOL_VERSION
        and actor_and_service is not None
    ):
        actor, service = actor_and_service
        owner = actor.principal
        session_id = str(uuid4())
        registry = _get_form_session_registry(request)
        identifiable = _identifiable_principal(owner)
        try:
            await registry.initialize(
                session_id=session_id,
                owner=owner,
                protocol_version=protocol_version,
                capabilities=params_obj.get("capabilities", {}),
                can_route_and_resume=identifiable,
                preserves_newlines=bool(
                    getattr(
                        request.app.state,
                        "mcp_form_preserves_newlines",
                        True,
                    )
                ),
            )
        except MCPFormSessionError as exc:
            return _mcp_protocol_error_response(message, exc)
        headers["MCP-Session-Id"] = session_id
        if identifiable:
            capabilities = _server_capabilities(
                orchestrator,
                task_controller=_get_task_controller(request),
                task_requestor=(owner, session_id),
            )
        logger.debug(
            "Initialized MCP session",
            extra={
                "mcp_session_initialized": True,
                "mcp_form_available": False,
                "mcp_task_capable": identifiable,
            },
        )
    return _handle_initialize_message(
        request,
        logger,
        orchestrator,
        message,
        capabilities=capabilities,
        headers=headers,
    )


async def _optional_mcp_actor(
    request: Request,
) -> tuple[InteractionActor, ServerInteractionService] | None:
    service = getattr(request.app.state, "interaction_service", None)
    if not isinstance(service, ServerInteractionService):
        return None
    try:
        actor = await service.authenticate(request)
    except ServerInteractionHTTPError:
        return None
    return actor, service


async def _mcp_session_context(
    request: Request,
    *,
    required: bool = False,
) -> _MCPSessionContext | None:
    headers = getattr(request, "headers", {})
    session_id = headers.get("MCP-Session-Id")
    if session_id is None:
        session_id = headers.get("mcp-session-id")
    if not isinstance(session_id, str) or not session_id:
        if required:
            raise _mcp_session_unavailable()
        return None
    actor_and_service = await _optional_mcp_actor(request)
    if actor_and_service is None:
        raise _mcp_session_unavailable()
    actor, service = actor_and_service
    registry = _get_form_session_registry(request)
    negotiation = await registry.negotiation(session_id, actor.principal)
    protocol_header = headers.get("MCP-Protocol-Version")
    if protocol_header is None:
        protocol_header = headers.get("mcp-protocol-version")
    if protocol_header != negotiation.protocol_version:
        raise MCPFormSessionError(
            code=MCPFormErrorCode.INVALID_CAPABILITIES,
            rpc_code=-32602,
            safe_message="MCP protocol version does not match the session.",
        )
    return _MCPSessionContext(
        session_id=session_id,
        owner=actor.principal,
        actor=actor,
        service=service,
        registry=registry,
        negotiation=negotiation,
    )


def _identifiable_principal(principal: PrincipalScope) -> bool:
    return any(
        value is not None
        for value in (
            principal.user_id,
            principal.tenant_id,
            principal.participant_id,
            principal.session_id,
        )
    )


def _negotiated_protocol_version(value: object) -> str:
    if isinstance(value, str) and value in SUPPORTED_PROTOCOL_VERSIONS:
        return value
    return LATEST_PROTOCOL_VERSION


def _get_form_session_registry(request: Request) -> MCPFormSessionRegistry:
    registry = getattr(request.app.state, "mcp_form_session_registry", None)
    if not isinstance(registry, MCPFormSessionRegistry):
        registry = MCPFormSessionRegistry()
        request.app.state.mcp_form_session_registry = registry
    return registry


def _get_task_controller(request: Request) -> MCPTaskController:
    controller = getattr(request.app.state, "mcp_task_controller", None)
    if not isinstance(controller, MCPTaskController):
        controller = MCPTaskController()
        request.app.state.mcp_task_controller = controller
    return controller


def _mcp_session_unavailable() -> MCPFormSessionError:
    return MCPFormSessionError(
        code=MCPFormErrorCode.SESSION_NOT_FOUND,
        rpc_code=-32001,
        safe_message="MCP session is unavailable.",
    )


def _mcp_protocol_error_response(
    message: Mapping[str, object],
    error: MCPFormSessionError | MCPTaskProtocolError,
) -> JSONResponse:
    if isinstance(error, MCPFormSessionError):
        payload: JsonObject = {
            "code": error.rpc_code,
            "message": error.safe_message,
            "data": {"reason": error.code.value},
        }
    else:
        payload = error.as_error()
    response: JsonObject = {
        "jsonrpc": "2.0",
        "id": cast(JSONValue, message.get("id")),
        "error": payload,
    }
    return JSONResponse(response)


class StreamResponse(Protocol):
    input_token_count: int
    output_token_count: int
    _response_iterator: AsyncIterator[ResponseItem] | None

    async def to_str(self) -> str: ...
    async def sync_messages(self) -> None: ...
    def __aiter__(self) -> AsyncIterator[ResponseItem]: ...


def _build_chat_request(
    tool_request: MCPToolRequest, orchestrator: Orchestrator
) -> ChatCompletionRequest:
    model_id = resolve_model_id(orchestrator)
    content: str | list[ContentFile | ContentImage | ContentText]
    if tool_request.files:
        content = []
        if tool_request.input_string and tool_request.input_string.strip():
            content.append(
                ContentText(type="text", text=tool_request.input_string)
            )
        content.extend(
            ContentFile(type="file", file=file.as_content_file())
            for file in tool_request.files
        )
    else:
        content = tool_request.input_string or ""
    return ChatCompletionRequest(
        model=model_id,
        messages=[ChatMessage(role=MessageRole.USER, content=content)],
        stream=True,
    )


def _mcp_interaction_runtime(
    session: _MCPSessionContext | None,
    *,
    related_request_id: str | int,
    related_task_id: str | None = None,
    status_hook: MCPFormStatusHook | None = None,
) -> AttachedInteractionRuntime | None:
    if session is None or not session.negotiation.form_available:
        return None
    return AttachedInteractionRuntime(
        broker=session.service.configuration.broker,
        actor=session.actor,
        handler=session.registry.handler(
            session_id=session.session_id,
            owner=session.owner,
            related_request_id=related_request_id,
            related_task_id=related_task_id,
            status_hook=status_hook,
        ),
        task_id=(
            TaskId(related_task_id) if related_task_id is not None else None
        ),
        context_label="mcp",
    )


async def _start_tool_streaming_response(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    request_id: str | int,
    tool_request: MCPToolRequest,
    progress_token: str | int,
    *,
    session: _MCPSessionContext | None = None,
) -> Response:
    chat_request = _build_chat_request(tool_request, orchestrator)
    interaction_runtime = _mcp_interaction_runtime(
        session,
        related_request_id=request_id,
    )
    response, response_uuid, timestamp = await orchestrate(
        chat_request,
        logger,
        orchestrator,
        interaction_runtime=interaction_runtime,
    )
    response_typed = cast(StreamResponse, response)
    response_uuid_obj = (
        response_uuid
        if isinstance(response_uuid, UUID)
        else UUID(str(response_uuid))
    )

    cancel_event = AsyncEvent()
    cancellation_registered = session is not None and chat_request.stream
    if cancellation_registered:
        assert session is not None
        _register_mcp_cancellation(
            request,
            session,
            request_id,
            cancel_event,
        )
    message_iter = _iter_jsonrpc_messages(request)
    watcher = create_task(
        _watch_for_cancellation(
            message_iter,
            cancel_event,
            logger,
            request_id=request_id,
        )
    )

    resource_store = _get_resource_store(request)
    base_path = cast(
        str, getattr(request.app.state, "mcp_resource_base_path", "")
    )
    output_redaction_settings = server_output_redaction_settings_from_state(
        request.app.state
    )

    if not chat_request.stream:
        try:
            text = sanitize_model_visible_server_protocol_text(
                await response_typed.to_str(),
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
                channel="answer",
            )
        finally:
            watcher.cancel()
            with suppress(Exception):
                await watcher

        summary: dict[str, JSONValue] = {
            "id": str(response_uuid),
            "created": timestamp,
            "model": chat_request.model,
            "usage": {
                "input_text_tokens": getattr(
                    response_typed, "input_token_count", 0
                ),
                "output_text_tokens": getattr(
                    response_typed, "output_token_count", 0
                ),
                "total_tokens": (
                    getattr(response_typed, "input_token_count", 0)
                    + getattr(response_typed, "output_token_count", 0)
                ),
            },
        }
        result_message: JSONRPCResult = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}] if text else [],
                "structuredContent": summary,
            },
        }
        return JSONResponse(result_message)

    async def stream() -> AsyncGenerator[bytes, None]:
        try:
            response_stream = _stream_mcp_response(
                request_id=request_id,
                request_model=chat_request,
                response=response_typed,
                response_id=response_uuid_obj,
                timestamp=timestamp,
                progress_token=progress_token,
                orchestrator=orchestrator,
                logger=logger,
                resource_store=resource_store,
                base_path=base_path,
                cancel_event=cancel_event,
                output_redaction_settings=output_redaction_settings,
            )
            if interaction_runtime is not None:
                assert session is not None
                response_stream = _merge_mcp_session_outbound(
                    response_stream,
                    session,
                    related_request_id=request_id,
                )
            async for chunk in response_stream:
                yield sse_bytes(chunk)
        finally:
            watcher.cancel()
            with suppress(Exception):
                await watcher
            if cancellation_registered:
                assert session is not None
                _discard_mcp_cancellation(
                    request,
                    session,
                    request_id,
                    cancel_event,
                )

    return StreamingResponse(
        stream(), media_type="text/event-stream", headers=sse_headers()
    )


async def _start_tool_task_response(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    request_id: str | int,
    tool_request: MCPToolRequest,
    progress_token: str | int,
    session: _MCPSessionContext,
    task_request: MCPTaskRequest,
) -> JSONResponse:
    requestor = session.task_requestor
    assert requestor is not None
    controller = _get_task_controller(request)
    cancel_event = AsyncEvent()
    background: Task[None] | None = None

    async def cancel_operation() -> None:
        cancel_event.set()
        if background is not None and not background.done():
            background.cancel()
            await gather(background, return_exceptions=True)

    creation = await controller.create(
        task_request,
        requestor=requestor,
        cancellation_callback=cancel_operation,
    )
    task_id = creation.handle.task_id
    status_hook = _mcp_task_status_hook(creation.handle)
    interaction_runtime = _mcp_interaction_runtime(
        session,
        related_request_id=request_id,
        related_task_id=task_id,
        status_hook=status_hook,
    )
    chat_request = _build_chat_request(tool_request, orchestrator)
    background = create_task(
        _run_tool_task(
            request,
            logger,
            orchestrator,
            request_id=request_id,
            chat_request=chat_request,
            progress_token=progress_token,
            interaction_runtime=interaction_runtime,
            handle=creation.handle,
            cancel_event=cancel_event,
        ),
        name=f"mcp-task-{task_id}",
    )
    _track_mcp_background_task(request, background, requestor)

    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": creation.as_dict(),
    }
    return JSONResponse(payload)


async def _run_tool_task(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    *,
    request_id: str | int,
    chat_request: ChatCompletionRequest,
    progress_token: str | int,
    interaction_runtime: AttachedInteractionRuntime | None,
    handle: MCPTaskHandle,
    cancel_event: AsyncEvent,
) -> None:
    operation = create_task(
        _execute_tool_task(
            request,
            logger,
            orchestrator,
            request_id=request_id,
            chat_request=chat_request,
            progress_token=progress_token,
            interaction_runtime=interaction_runtime,
            handle=handle,
            cancel_event=cancel_event,
        )
    )
    cancelled = create_task(handle.wait_cancelled())
    try:
        done, _ = await wait(
            {operation, cancelled},
            return_when=FIRST_COMPLETED,
        )
        if cancelled in done and not operation.done():
            cancel_event.set()
            operation.cancel()
        await gather(operation, return_exceptions=True)
    finally:
        if not operation.done():
            operation.cancel()
        if not cancelled.done():
            cancelled.cancel()
        await gather(operation, cancelled, return_exceptions=True)


async def _execute_tool_task(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    *,
    request_id: str | int,
    chat_request: ChatCompletionRequest,
    progress_token: str | int,
    interaction_runtime: AttachedInteractionRuntime | None,
    handle: MCPTaskHandle,
    cancel_event: AsyncEvent,
) -> None:
    try:
        response, response_uuid, timestamp = await orchestrate(
            chat_request,
            logger,
            orchestrator,
            interaction_runtime=interaction_runtime,
        )
        response_uuid_obj = (
            response_uuid
            if isinstance(response_uuid, UUID)
            else UUID(str(response_uuid))
        )
        response_stream = _stream_mcp_response(
            request_id=request_id,
            request_model=chat_request,
            response=cast(StreamResponse, response),
            response_id=response_uuid_obj,
            timestamp=timestamp,
            progress_token=progress_token,
            orchestrator=orchestrator,
            logger=logger,
            resource_store=_get_resource_store(request),
            base_path=cast(
                str,
                getattr(request.app.state, "mcp_resource_base_path", ""),
            ),
            cancel_event=cancel_event,
            output_redaction_settings=(
                server_output_redaction_settings_from_state(request.app.state)
            ),
        )
        await _consume_tool_task(
            response_stream,
            request_id=request_id,
            handle=handle,
            cancel_event=cancel_event,
            logger=logger,
        )
    except CancelledError:
        cancel_event.set()
        raise
    except Exception:
        logger.exception("MCP task operation failed")
        await handle.fail(
            {
                "code": -32603,
                "message": "An internal server error occurred.",
            }
        )


def _mcp_task_status_hook(
    handle: MCPTaskHandle,
) -> MCPFormStatusHook:
    async def update(event: MCPFormStatusEvent) -> None:
        if event.status is MCPFormStatus.INPUT_REQUIRED:
            await handle.transition_input_required()
        else:
            try:
                await handle.transition_working()
            except MCPTaskProtocolError as exc:
                if exc.data != {
                    "policy": "avalan",
                    "reason": "state_mismatch",
                }:
                    raise

    return update


async def _consume_tool_task(
    response_stream: AsyncIterator[bytes],
    *,
    request_id: str | int,
    handle: MCPTaskHandle,
    cancel_event: AsyncEvent,
    logger: Logger,
) -> None:
    terminal: Mapping[str, object] | None = None
    try:
        async for chunk in response_stream:
            for raw_line in chunk.splitlines():
                if not raw_line:
                    continue
                message = loads(raw_line)
                if (
                    isinstance(message, Mapping)
                    and message.get("id") == request_id
                    and ("result" in message or "error" in message)
                ):
                    terminal = cast(Mapping[str, object], message)
        if terminal is None:
            await handle.fail(
                {
                    "code": -32603,
                    "message": "MCP task ended without an operation result.",
                }
            )
            return
        result = terminal.get("result")
        if isinstance(result, Mapping):
            await handle.complete(result)
            return
        error = terminal.get("error")
        if isinstance(error, Mapping):
            await handle.fail(error)
            return
        await handle.fail(
            {
                "code": -32603,
                "message": "MCP task returned an invalid operation result.",
            }
        )
    except CancelledError:
        cancel_event.set()
        raise
    except Exception:
        logger.exception("MCP task operation failed")
        await handle.fail(
            {
                "code": -32603,
                "message": "An internal server error occurred.",
            }
        )
    finally:
        close = getattr(response_stream, "aclose", None)
        if callable(close):
            await close()


def _track_mcp_background_task(
    request: Request,
    task: Task[None],
    requestor: tuple[PrincipalScope, str],
) -> None:
    registry = getattr(request.app.state, "mcp_background_tasks", None)
    if not isinstance(registry, dict):
        registry = {}
        request.app.state.mcp_background_tasks = registry
    tasks = registry.setdefault(requestor, set())
    tasks.add(task)

    def discard(completed: Task[None]) -> None:
        owned = registry.get(requestor)
        if not isinstance(owned, set):
            return
        owned.discard(completed)
        if not owned:
            registry.pop(requestor, None)

    task.add_done_callback(discard)


async def _cancel_mcp_background_tasks(
    app: object,
    requestor: tuple[PrincipalScope, str],
) -> None:
    state = getattr(app, "state", None)
    if state is None:
        return
    registry = getattr(state, "mcp_background_tasks", None)
    if not isinstance(registry, dict):
        return
    tasks = tuple(registry.pop(requestor, ()))
    for task in tasks:
        task.cancel()
    if tasks:
        await gather(*tasks, return_exceptions=True)


async def _handle_task_message(
    request: Request,
    message: JSONObject,
    session: _MCPSessionContext,
) -> Response:
    method = message.get("method")
    params_value = message.get("params")
    params = (
        without_related_task_metadata(params_value)
        if isinstance(params_value, Mapping)
        else {}
    )
    controller = _get_task_controller(request)
    requestor = session.task_requestor
    if requestor is None:
        raise MCPTaskProtocolError(
            code=-32001,
            message="MCP task session is unavailable.",
        )
    request_id = cast(str | int, message.get("id", str(uuid4())))
    if method == "tasks/list":
        cursor = params.get("cursor")
        if cursor is not None and not isinstance(cursor, str):
            raise MCPTaskProtocolError(
                code=-32602,
                message="Invalid task cursor.",
            )
        result = await controller.list(
            requestor=requestor,
            cursor=cursor,
        )
        return _mcp_jsonrpc_result(request_id, result)
    task_id = params.get("taskId")
    if not isinstance(task_id, str) or not task_id:
        raise MCPTaskProtocolError(
            code=-32602,
            message="Invalid task identifier.",
        )
    if method == "tasks/get":
        return _mcp_jsonrpc_result(
            request_id,
            await controller.get(task_id, requestor=requestor),
        )
    if method == "tasks/cancel":
        return _mcp_jsonrpc_result(
            request_id,
            await controller.cancel(task_id, requestor=requestor),
        )
    if method == "tasks/result":
        return _mcp_task_result_response(
            controller,
            task_id=task_id,
            request_id=request_id,
            requestor=requestor,
            session=session,
        )
    raise MCPTaskProtocolError(
        code=-32601,
        message="MCP task method is not supported.",
    )


def _mcp_task_result_response(
    controller: MCPTaskController,
    *,
    task_id: str,
    request_id: str | int,
    requestor: tuple[PrincipalScope, str],
    session: _MCPSessionContext,
) -> StreamingResponse:
    async def result_stream() -> AsyncIterator[bytes]:
        try:
            outcome = await controller.result(task_id, requestor=requestor)
        except MCPTaskProtocolError as exc:
            message: JSONObject = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": exc.as_error(),
            }
        else:
            message = _mcp_task_outcome_message(request_id, outcome)
        yield _encode_mcp_message(message)

    stream: AsyncIterator[bytes] = result_stream()
    if session.negotiation.form_available:
        stream = _merge_mcp_session_outbound(
            stream,
            session,
            related_task_id=task_id,
        )
    return StreamingResponse(
        (sse_bytes(chunk) async for chunk in stream),
        media_type="text/event-stream",
        headers=sse_headers(),
    )


def _mcp_task_outcome_message(
    request_id: str | int,
    outcome: MCPTaskOutcome,
) -> JSONObject:
    if outcome.result is not None:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": outcome.result,
        }
    assert outcome.error is not None
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": outcome.error,
    }


def _mcp_jsonrpc_result(
    request_id: str | int,
    result: Mapping[str, object],
) -> JSONResponse:
    payload: JSONObject = {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": cast(JSONObject, dict(result)),
    }
    return JSONResponse(payload)


def _handle_ping_message(
    logger: Logger,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    response_id = cast(str | int, message.get("id", str(uuid4())))
    payload: JSONRPCResult = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": {},
    }
    logger.debug(
        "Handled MCP ping request", extra={"response_id": response_id}
    )
    return JSONResponse(payload)


def _handle_initialize_message(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
    *,
    capabilities: Mapping[str, JSONValue] | None = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    params = message.get("params")
    params_obj: JSONObject = params if isinstance(params, dict) else {}
    protocol_version = _negotiated_protocol_version(
        params_obj.get("protocolVersion")
    )

    response_id = cast(str | int, message.get("id", str(uuid4())))
    payload: JSONRPCResult = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": {
            "protocolVersion": protocol_version,
            "capabilities": dict(
                capabilities or _server_capabilities(orchestrator)
            ),
            "serverInfo": _server_info(request),
        },
    }
    logger.debug(
        "Handled MCP initialize request",
        extra={"response_id": response_id},
    )
    return JSONResponse(payload, headers=dict(headers or {}))


def _handle_initialized_notification(
    logger: Logger,
    message: JSONObject,
) -> Response:
    if "id" in message:
        raise HTTPException(
            status_code=400, detail="MCP notifications cannot include an id"
        )

    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    logger.debug("Handled MCP initialized notification")
    return Response(status_code=204)


def _handle_list_tools_message(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    if params is not None and not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")

    tools = _collect_tool_descriptions(request, orchestrator)
    response_id = cast(str | int, message.get("id", str(uuid4())))
    result: dict[str, JSONValue] = {"tools": cast(JSONValue, tools)}
    next_cursor = getattr(request.app.state, "mcp_next_cursor", None)
    if next_cursor:
        result["nextCursor"] = next_cursor
    payload: JSONRPCResult = {
        "jsonrpc": "2.0",
        "id": response_id,
        "result": result,
    }
    logger.debug(
        "Handled MCP tools list request",
        extra={"response_id": response_id, "tool_count": len(tools)},
    )
    return JSONResponse(payload)


def _collect_tool_descriptions(
    request: Request,
    orchestrator: Orchestrator | None = None,
) -> list[dict[str, JSONValue]]:
    name = cast(str, getattr(request.app.state, "mcp_tool_name", "run"))
    description = cast(
        str,
        getattr(
            request.app.state,
            "mcp_tool_description",
            "Execute the Avalan orchestrator run endpoint.",
        ),
    )
    run_tool: dict[str, JSONValue] = {
        "name": name,
        "description": description,
        "inputSchema": MCPToolRequest.model_json_schema(),
    }
    if getattr(request.state, "mcp_task_capable", False) is True:
        run_tool["execution"] = {"taskSupport": "optional"}
    tools: list[dict[str, JSONValue]] = [run_tool]
    if orchestrator is None:
        return tools
    tool_manager = getattr(orchestrator, "tool", None)
    list_tools = getattr(tool_manager, "list_tools", None)
    if not callable(list_tools):
        return tools
    for descriptor in list_tools():
        tool_description = _skills_tool_description(descriptor)
        if tool_description is not None:
            tools.append(tool_description)
    return tools


def _skills_tool_description(
    descriptor: ToolDescriptor,
) -> dict[str, JSONValue] | None:
    assert isinstance(descriptor, ToolDescriptor)
    name = descriptor.name
    if not name.startswith("skills."):
        return None
    schema = descriptor.schema or descriptor.provider_safe_schema or {}
    function = schema.get("function") if isinstance(schema, dict) else None
    if not isinstance(function, dict):
        function = {}
    description = function.get("description")
    if not isinstance(description, str):
        description = ""
    input_schema = function.get("parameters")
    if not isinstance(input_schema, dict):
        input_schema = descriptor.parameter_schema
    if not isinstance(input_schema, dict):
        input_schema = {"type": "object", "properties": {}}
    return {
        "name": name,
        "description": description,
        "inputSchema": cast(JSONValue, input_schema),
    }


def _is_direct_skills_tool_call(
    orchestrator: Orchestrator,
    message: JSONObject,
) -> bool:
    params = message.get("params")
    if not isinstance(params, dict):
        return False
    name = params.get("name")
    if not isinstance(name, str) or not name.startswith("skills."):
        return False
    resolution = orchestrator.tool.resolve_tool_name(name)
    return (
        resolution.canonical_name is not None
        and resolution.canonical_name.startswith("skills.")
    )


async def _handle_direct_skills_tool_call_message(
    request: Request,
    logger: Logger,
    orchestrator: Orchestrator,
    message: JSONObject,
) -> JSONResponse:
    params = message.get("params")
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")
    name = params.get("name")
    if not isinstance(name, str) or not name.startswith("skills."):
        raise HTTPException(
            status_code=400, detail=f'Unsupported tool "{name}"'
        )
    resolution = orchestrator.tool.resolve_tool_name(name)
    canonical_name = resolution.canonical_name
    if canonical_name is None or not canonical_name.startswith("skills."):
        raise HTTPException(
            status_code=400, detail=f'Unsupported tool "{name}"'
        )
    raw_arguments = params.get("arguments")
    if raw_arguments is None:
        raw_arguments = {}
    if not isinstance(raw_arguments, dict):
        raise HTTPException(status_code=400, detail="Invalid tool arguments")
    try:
        container_request = validate_remote_container_arguments(
            raw_arguments,
            policy=remote_container_policy_from_state(request.app.state),
        )
    except RemoteContainerRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    arguments = container_request.arguments

    call = ToolCall(
        id=str(uuid4()),
        name=canonical_name,
        arguments=cast(dict[str, ToolValue], arguments),
    )
    context = _direct_tool_call_context(request, orchestrator, call)
    outcome = await orchestrator.tool.execute_call(call, context)
    response_id = cast(str | int, message.get("id", str(uuid4())))
    payload = _direct_tool_call_jsonrpc_result(
        request_id=response_id,
        tool_name=canonical_name,
        outcome=outcome,
        output_redaction_settings=(
            server_output_redaction_settings_from_state(request.app.state)
        ),
    )
    logger.debug(
        "Handled direct MCP skills tool call",
        extra={
            "response_id": response_id,
            "tool_name": canonical_name,
        },
    )
    return JSONResponse(payload)


def _direct_tool_call_context(
    request: Request,
    orchestrator: Orchestrator,
    call: ToolCall,
) -> ToolCallContext:
    context = getattr(request.app.state, "ctx", None)
    participant_id = getattr(context, "participant_id", None)
    agent_id = getattr(orchestrator, "_id", None)
    return ToolCallContext(
        agent_id=agent_id if isinstance(agent_id, UUID) else None,
        participant_id=(
            participant_id if isinstance(participant_id, UUID) else None
        ),
        calls=[call],
    )


def _direct_tool_call_jsonrpc_result(
    *,
    request_id: str | int,
    tool_name: str,
    outcome: ToolCallResult | ToolCallError | ToolCallDiagnostic,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONRPCResult:
    structured = _direct_tool_outcome_structured_content(
        outcome,
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
    )
    content_text = dumps(structured, separators=(",", ":"), sort_keys=True)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": content_text}],
            "structuredContent": structured,
        },
    }


def _direct_tool_outcome_structured_content(
    outcome: ToolCallResult | ToolCallError | ToolCallDiagnostic,
    *,
    tool_name: str,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONObject:
    if isinstance(outcome, ToolCallResult):
        return {
            "type": "tool.result",
            "toolCallId": str(outcome.call.id or outcome.id),
            "name": outcome.name,
            "result": cast(
                JSONValue,
                sanitize_server_protocol_value(
                    outcome.result,
                    tool_name=tool_name,
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                ),
            ),
        }
    if isinstance(outcome, ToolCallError):
        return {
            "type": "tool.error",
            "toolCallId": str(outcome.call.id or outcome.id),
            "name": outcome.name,
            "message": sanitize_server_protocol_text(
                outcome.message,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
            "error": cast(
                JSONValue,
                sanitize_server_protocol_value(
                    outcome.error,
                    tool_name=tool_name,
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                ),
            ),
        }
    return {
        "type": "tool.diagnostic",
        "toolCallId": str(outcome.call_id or outcome.id),
        "name": outcome.canonical_name or tool_name,
        "diagnostic": cast(
            JSONValue,
            sanitize_server_protocol_value(
                _tool_diagnostic_payload(
                    outcome,
                    output_redaction_settings=output_redaction_settings,
                ),
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        ),
    }


def _tool_diagnostic_payload(
    diagnostic: ToolCallDiagnostic,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> dict[str, JSONValue]:
    return {
        "id": str(diagnostic.id),
        "call_id": (
            str(diagnostic.call_id) if diagnostic.call_id is not None else None
        ),
        "requested_name": diagnostic.requested_name,
        "canonical_name": diagnostic.canonical_name,
        "status": diagnostic.status.value,
        "code": diagnostic.code.value,
        "stage": diagnostic.stage.value,
        "message": sanitize_server_protocol_text(
            diagnostic.message,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
        "retryable": diagnostic.retryable,
        "details": cast(
            JSONValue,
            sanitize_server_protocol_value(
                diagnostic.details,
                tool_name=diagnostic.canonical_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        ),
    }


def _extract_call_arguments(
    method: str, params: JSONObject, *, allowed_tool_name: str
) -> dict[str, JSONValue]:
    if method == "tools/call":
        name = params.get("name")
        if name is None or name != allowed_tool_name:
            raise HTTPException(
                status_code=400, detail=f'Unsupported tool "{name}"'
            )
        arguments = params.get("arguments")
        if not isinstance(arguments, dict):
            raise HTTPException(
                status_code=400, detail="Invalid tool arguments"
            )
        return arguments

    raise HTTPException(
        status_code=400, detail=f'Unsupported MCP method "{method}"'
    )


async def _watch_for_cancellation(
    messages: AsyncIterator[JSONObject],
    cancel_event: AsyncEvent,
    logger: Logger,
    *,
    request_id: str | int,
) -> None:
    async for message in messages:
        if not isinstance(message, dict):
            continue
        method = cast(str | None, message.get("method"))
        if method != "notifications/cancelled":
            continue
        params = message.get("params")
        if not isinstance(params, Mapping):
            continue
        cancelled_id = params.get("requestId")
        if not (
            (
                type(cancelled_id) is int
                or isinstance(cancelled_id, str)
                and cancelled_id
            )
            and type(cancelled_id) is type(request_id)
            and cancelled_id == request_id
        ):
            continue
        cancel_event.set()
        logger.debug("Received MCP cancellation notification")
        break


def _mcp_cancellations(
    app: object,
    *,
    create: bool = False,
) -> dict[MCPCancellationKey, AsyncEvent] | None:
    state = getattr(app, "state", None)
    if state is None:
        return None
    value = getattr(state, "mcp_stream_cancellations", None)
    if isinstance(value, dict):
        return cast(dict[MCPCancellationKey, AsyncEvent], value)
    if not create:
        return None
    cancellations: dict[MCPCancellationKey, AsyncEvent] = {}
    state.mcp_stream_cancellations = cancellations
    return cancellations


def _mcp_cancellation_key(
    session: _MCPSessionContext,
    request_id: str | int,
) -> MCPCancellationKey:
    return (session.owner, session.session_id, request_id)


def _register_mcp_cancellation(
    request: Request,
    session: _MCPSessionContext,
    request_id: str | int,
    cancel_event: AsyncEvent,
) -> None:
    cancellations = _mcp_cancellations(request.app, create=True)
    assert cancellations is not None
    cancellations[_mcp_cancellation_key(session, request_id)] = cancel_event


def _discard_mcp_cancellation(
    request: Request,
    session: _MCPSessionContext,
    request_id: str | int,
    cancel_event: AsyncEvent,
) -> None:
    cancellations = _mcp_cancellations(request.app)
    if cancellations is None:
        return
    key = _mcp_cancellation_key(session, request_id)
    if cancellations.get(key) is cancel_event:
        cancellations.pop(key)
    if not cancellations:
        delattr(request.app.state, "mcp_stream_cancellations")


def _cancel_mcp_session_requests(
    app: object,
    session: _MCPSessionContext,
) -> None:
    cancellations = _mcp_cancellations(app)
    if cancellations is None:
        return
    prefix = (session.owner, session.session_id)
    for key, event in tuple(cancellations.items()):
        if key[:2] == prefix:
            cancellations.pop(key)
            event.set()
    state = getattr(app, "state", None)
    if not cancellations and state is not None:
        delattr(state, "mcp_stream_cancellations")


def _handle_cancelled_notification(
    request: Request,
    logger: Logger,
    message: JSONObject,
    session: _MCPSessionContext,
) -> Response:
    if "id" in message:
        raise HTTPException(
            status_code=400,
            detail="MCP notifications cannot include an id",
        )
    params = message.get("params")
    if not isinstance(params, dict):
        raise HTTPException(status_code=400, detail="Missing MCP params")
    request_id = params.get("requestId")
    if not (
        type(request_id) is int or isinstance(request_id, str) and request_id
    ):
        raise HTTPException(
            status_code=400,
            detail="MCP cancellation requires a requestId",
        )
    cancellations = _mcp_cancellations(request.app)
    event = (
        cancellations.get(_mcp_cancellation_key(session, request_id))
        if cancellations is not None
        else None
    )
    if event is not None:
        event.set()
    logger.debug(
        "Handled MCP cancellation notification",
        extra={"request_id": request_id, "active": event is not None},
    )
    return Response(status_code=202)


async def _merge_mcp_session_outbound(
    source: AsyncIterator[bytes],
    session: _MCPSessionContext,
    *,
    related_request_id: str | int | None = None,
    related_task_id: str | None = None,
) -> AsyncIterator[bytes]:
    source_next = create_task(_next_mcp_chunk(source))
    outbound_next = create_task(
        session.registry.next_outbound(
            session.session_id,
            session.owner,
            timeout_seconds=30,
            related_request_id=related_request_id,
            related_task_id=related_task_id,
        )
    )
    try:
        while True:
            done, _ = await wait(
                {source_next, outbound_next},
                return_when=FIRST_COMPLETED,
            )
            if outbound_next in done:
                outbound = outbound_next.result()
                if outbound is not None:
                    yield _encode_mcp_message(_mcp_outbound_message(outbound))
                outbound_next = create_task(
                    session.registry.next_outbound(
                        session.session_id,
                        session.owner,
                        timeout_seconds=30,
                        related_request_id=related_request_id,
                        related_task_id=related_task_id,
                    )
                )
            if source_next in done:
                try:
                    chunk = source_next.result()
                except StopAsyncIteration:
                    return
                yield chunk
                source_next = create_task(_next_mcp_chunk(source))
    finally:
        pending = tuple(
            task for task in (source_next, outbound_next) if not task.done()
        )
        for task in pending:
            task.cancel()
        await gather(*pending, return_exceptions=True)
        close = getattr(source, "aclose", None)
        if callable(close):
            await close()


async def _next_mcp_chunk(source: AsyncIterator[bytes]) -> bytes:
    return await anext(source)


def _mcp_outbound_message(
    outbound: MCPFormElicitationOutbound,
) -> JSONObject:
    params: Mapping[str, object] = outbound.params
    if outbound.related_task_id is not None:
        params = with_related_task_metadata(
            params,
            outbound.related_task_id,
        )
    return {
        "jsonrpc": "2.0",
        "id": outbound.jsonrpc_id,
        "method": outbound.method,
        "params": cast(JSONObject, dict(params)),
    }


def _encode_mcp_message(message: Mapping[str, object]) -> bytes:
    return (dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


async def _cleanup_mcp_stream_sources(
    logger: Logger, *sources: object, cancelled: bool
) -> None:
    try:
        await cleanup_stream_sources(*sources, cancelled=cancelled)
    except BaseExceptionGroup as exc:
        logger.exception("MCP stream source cleanup failed", exc_info=exc)
    except (Exception, CancelledError) as exc:
        logger.exception("MCP stream source cleanup failed", exc_info=exc)


async def _stream_mcp_response(
    *,
    request_id: str | int,
    request_model: ChatCompletionRequest,
    response: StreamResponse,
    response_id: UUID,
    timestamp: int,
    progress_token: str | int,
    orchestrator: Orchestrator,
    logger: Logger,
    resource_store: MCPResourceStore,
    base_path: str,
    cancel_event: AsyncEvent,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> AsyncIterator[bytes]:
    output_redaction_settings = coerce_server_output_redaction_settings(
        output_redaction_settings
    )
    state = _MCPStreamProjectionState(
        accumulator=ProtocolStreamAccumulator(),
        tool_summaries={},
        resources={},
        resource_store=resource_store,
        base_path=base_path,
        output_redaction_settings=output_redaction_settings,
        answer_redactor=ModelVisibleServerProtocolTextRedactor(
            output_redaction_settings,
            protocol="mcp",
            channel="answer",
        ),
        reasoning=_MCPReasoningOwner(output_redaction_settings),
    )
    finished_normally = False
    response_iterator: AsyncIterator[StreamConsumerProjection] | None = None
    stream_error_message: JSONObject | None = None

    def emit(message: JSONObject) -> Iterator[bytes]:
        encoded = (
            dumps(cast(Mapping[str, object], message), separators=(",", ":"))
            + "\n"
        )
        yield encoded.encode("utf-8")

    try:
        response_iterator = stream_consumer_iterator(
            response,
            stream_session_id="mcp-stream",
            run_id=str(response_id),
            turn_id="mcp-turn",
            unsupported_message="unsupported MCP stream item",
            close_source_on_generator_exit=False,
        )
        async for item in cancellable_stream_iterator(
            response_iterator, cancel_event
        ):
            for notification in await _mcp_notifications(
                item, state, progress_token
            ):
                for payload in emit(notification):
                    yield payload

        finished_normally = not cancel_event.is_set()
    except GeneratorExit:
        cancel_event.set()
        state.reasoning_owner.finish(StreamTerminalOutcome.CANCELLED)
        await _cleanup_mcp_stream_sources(
            logger, response, response_iterator, cancelled=True
        )
        await _close_mcp_resource_notifications(
            resource_store, state.resources
        )
        await resource_store.prune_closed()
        await orchestrator.sync_messages(response)
        raise
    except CancelledError:
        cancel_event.set()
        finished_normally = False
    except Exception as exc:
        logger.exception("Error while streaming MCP response", exc_info=exc)
        cancel_event.set()
        finished_normally = False
        stream_error_message = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": "An internal server error occurred.",
            },
        }

    if cancel_event.is_set():
        await _cleanup_mcp_stream_sources(
            logger, response, response_iterator, cancelled=True
        )
        cancel_error_message: JSONObject = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": -32000, "message": "Request cancelled"},
        }
        error_message = stream_error_message or cancel_error_message
        reasoning_messages = _mcp_finish_reasoning_notifications(
            state,
            (
                StreamTerminalOutcome.ERRORED
                if stream_error_message is not None
                else StreamTerminalOutcome.CANCELLED
            ),
        )
        terminal_messages = await _collect_terminal_mcp_messages(
            resource_store, state.resources, error_message
        )
        terminal_messages = (*reasoning_messages, *terminal_messages)
        try:
            for message in terminal_messages:
                for payload in emit(message):
                    yield payload
        finally:
            await orchestrator.sync_messages(response)
        return

    if finished_normally:
        try:
            state.accumulator.validate_complete()
        except StreamValidationError as exc:
            logger.exception("Invalid MCP canonical stream", exc_info=exc)
            await _cleanup_mcp_stream_sources(
                logger, response, response_iterator, cancelled=False
            )
            validation_error_message: JSONObject = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": "An internal server error occurred.",
                },
            }
            reasoning_messages = _mcp_finish_reasoning_notifications(
                state,
                StreamTerminalOutcome.ERRORED,
            )
            terminal_messages = await _collect_terminal_mcp_messages(
                resource_store,
                state.resources,
                validation_error_message,
            )
            terminal_messages = (*reasoning_messages, *terminal_messages)
            try:
                for message in terminal_messages:
                    for payload in emit(message):
                        yield payload
            finally:
                await orchestrator.sync_messages(response)
            return

        snapshot = state.accumulator.snapshot()
        if snapshot.terminal_outcome is StreamTerminalOutcome.CANCELLED:
            await _cleanup_mcp_stream_sources(
                logger, response, response_iterator, cancelled=True
            )
            terminal_cancel_error_message: JSONObject = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": "Request cancelled"},
            }
            terminal_messages = await _collect_terminal_mcp_messages(
                resource_store,
                state.resources,
                terminal_cancel_error_message,
            )
            try:
                for message in terminal_messages:
                    for payload in emit(message):
                        yield payload
            finally:
                await orchestrator.sync_messages(response)
            return
        if snapshot.terminal_outcome is StreamTerminalOutcome.ERRORED:
            await _cleanup_mcp_stream_sources(
                logger, response, response_iterator, cancelled=False
            )
            error_message = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": _canonical_error_message(
                        snapshot,
                        output_redaction_settings=(
                            state.output_redaction_settings
                        ),
                    ),
                },
            }
            terminal_messages = await _collect_terminal_mcp_messages(
                resource_store, state.resources, error_message
            )
            try:
                for message in terminal_messages:
                    for payload in emit(message):
                        yield payload
            finally:
                await orchestrator.sync_messages(response)
            return

        answer_text = sanitize_model_visible_server_protocol_text(
            snapshot.answer_text,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
            channel="answer",
        )
        usage = snapshot.usage

        summary: dict[str, JSONValue] = {
            "id": str(response_id),
            "created": timestamp,
            "model": request_model.model,
            "usage": {
                "input_text_tokens": _usage_count(
                    usage,
                    "input_text_tokens",
                    response.input_token_count,
                    aliases=("input_tokens",),
                ),
                "output_text_tokens": _usage_count(
                    usage,
                    "output_text_tokens",
                    response.output_token_count,
                    aliases=("output_tokens",),
                ),
                "total_tokens": _usage_count(
                    usage,
                    "total_tokens",
                    (response.input_token_count + response.output_token_count),
                ),
            },
        }
        summary.update(state.reasoning_owner.final_payload())
        _merge_canonical_tool_call_arguments(
            state.tool_summaries,
            snapshot.tool_call_arguments,
            output_redaction_settings=state.output_redaction_settings,
        )
        if state.tool_summaries:
            summary["toolCalls"] = list(state.tool_summaries.values())

        result_message: JSONRPCResult = {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": (
                    [{"type": "text", "text": answer_text}]
                    if answer_text
                    else []
                ),
                "structuredContent": summary,
            },
        }
        await _cleanup_mcp_stream_sources(
            logger, response, response_iterator, cancelled=False
        )
        terminal_messages = await _collect_terminal_mcp_messages(
            resource_store, state.resources, cast(JSONObject, result_message)
        )
        try:
            for message in terminal_messages:
                for payload in emit(message):
                    yield payload
        finally:
            await orchestrator.sync_messages(response)


async def _mcp_notifications(
    item: StreamConsumerProjection,
    state: _MCPStreamProjectionState,
    progress_token: str | int,
) -> list[JSONObject]:
    return await _mcp_stream_item_notifications(item, state, progress_token)


async def _mcp_stream_item_notifications(
    item: StreamConsumerProjection,
    state: _MCPStreamProjectionState,
    progress_token: str | int,
) -> list[JSONObject]:
    return await _mcp_canonical_stream_item_notifications(
        canonical_item_from_consumer_projection(item),
        state,
        progress_token,
    )


async def _mcp_canonical_stream_item_notifications(
    item: CanonicalStreamItem,
    state: _MCPStreamProjectionState,
    progress_token: str | int,
) -> list[JSONObject]:
    notifications: list[JSONObject] = []

    if item.kind not in (
        StreamItemKind.REASONING_DELTA,
        StreamItemKind.REASONING_DONE,
    ):
        state.accumulator.add(item)
    if item.is_stream_terminal:
        assert item.terminal_outcome is not None
        reasoning_outputs, closed_segment = state.reasoning_owner.finish(
            item.terminal_outcome
        )
        assert closed_segment is None
        notifications.extend(
            _mcp_reasoning_output_notifications(reasoning_outputs)
        )
        notifications.extend(
            _mcp_model_text_flush_notifications(
                state,
                progress_token,
                item.sequence,
            )
        )
        if item.terminal_outcome is StreamTerminalOutcome.INPUT_REQUIRED:
            raise StreamValidationError(
                "MCP input-required projection is unavailable"
            )
    if item.kind is StreamItemKind.FLOW_EVENT:
        notifications.append(_canonical_flow_notification(item))
        return notifications
    if item.kind is StreamItemKind.TOOL_CALL_READY:
        _record_canonical_tool_call_ready(
            item,
            state.tool_summaries,
            output_redaction_settings=state.output_redaction_settings,
        )
        return notifications
    if item.kind is StreamItemKind.REASONING_DELTA:
        notifications.extend(
            _mcp_reasoning_output_notifications(
                state.reasoning_owner.push(item)
            )
        )
        return notifications
    if item.kind is StreamItemKind.REASONING_DONE:
        reasoning_outputs, _closed_segment = state.reasoning_owner.complete()
        notifications.extend(
            _mcp_reasoning_output_notifications(reasoning_outputs)
        )
        return notifications

    token_notification = _canonical_tool_notification(
        item,
        output_redaction_settings=state.output_redaction_settings,
    )
    if token_notification is not None:
        notifications.append(token_notification)
        return notifications
    tool_execution_notification = _canonical_tool_execution_notification(
        item,
        state.tool_summaries,
        output_redaction_settings=state.output_redaction_settings,
    )
    if tool_execution_notification is not None:
        notifications.append(tool_execution_notification)
        return notifications
    async for resource_notification in _canonical_tool_resource_notifications(
        item=item,
        tool_summaries=state.tool_summaries,
        resources=state.resources,
        resource_store=state.resource_store,
        base_path=state.base_path,
        output_redaction_settings=state.output_redaction_settings,
    ):
        notifications.append(resource_notification)
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        return notifications
    progress_notification = _canonical_progress_notification(
        item,
        progress_token,
        state.answer_redactor,
    )
    if progress_notification is not None:
        notifications.append(progress_notification)
        return notifications
    if item.kind is not StreamItemKind.ANSWER_DELTA:
        return notifications

    return notifications


def _mcp_reasoning_output_notifications(
    outputs: tuple[ProtocolReasoningRedactedText, ...],
) -> list[JSONObject]:
    notifications: list[JSONObject] = []
    for output in outputs:
        identity = output.identity
        data: dict[str, JSONValue] = {
            "type": "reasoning",
            "delta": output.text,
            "representation": identity.representation.value,
            "segment_instance_ordinal": identity.segment_instance_ordinal,
            "completed": False,
            "status": "in_progress",
            "terminal_outcome": None,
        }
        for field_name, value in (
            ("provider_item_id", identity.provider_item_id),
            ("output_index", identity.output_index),
            ("summary_index", identity.summary_index),
            ("continuation_id", identity.continuation_id),
        ):
            if value is not None:
                data[field_name] = value
        notifications.append(
            {
                "jsonrpc": "2.0",
                "method": "notifications/message",
                "params": {"level": "debug", "data": data},
            }
        )
    return notifications


def _mcp_reasoning_close_notification(
    segment: _MCPReasoningSegment,
) -> JSONObject:
    identity = segment.identity
    data: dict[str, JSONValue] = {
        "type": "reasoning",
        "delta": "",
        "representation": identity.representation.value,
        "segment_instance_ordinal": identity.segment_instance_ordinal,
        "completed": segment.completed,
        "status": segment.status,
        "terminal_outcome": segment.terminal_outcome,
    }
    for field_name, value in (
        ("provider_item_id", identity.provider_item_id),
        ("output_index", identity.output_index),
        ("summary_index", identity.summary_index),
        ("continuation_id", identity.continuation_id),
    ):
        if value is not None:
            data[field_name] = value
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": "debug", "data": data},
    }


def _mcp_finish_reasoning_notifications(
    state: _MCPStreamProjectionState,
    outcome: StreamTerminalOutcome,
) -> list[JSONObject]:
    outputs, closed_segment = state.reasoning_owner.finish(outcome)
    notifications = _mcp_reasoning_output_notifications(outputs)
    if (
        closed_segment is not None
        and outcome is not StreamTerminalOutcome.COMPLETED
    ):
        notifications.append(_mcp_reasoning_close_notification(closed_segment))
    return notifications


def _record_canonical_tool_call_ready(
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> None:
    assert item.kind is StreamItemKind.TOOL_CALL_READY
    tool_call_id = item.correlation.tool_call_id
    assert tool_call_id is not None
    data = item.data if isinstance(item.data, dict) else {}
    name = data.get("name")
    arguments = data.get("arguments")
    tool_name = name if isinstance(name, str) else None
    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {
            "id": tool_call_id,
            "name": None,
            "arguments": None,
        },
    )
    if isinstance(name, str) and name:
        tool_summary["name"] = name
    if arguments is not None:
        tool_summary["arguments"] = cast(
            JSONValue,
            sanitize_server_protocol_value(
                arguments,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )


def _canonical_progress_notification(
    item: CanonicalStreamItem,
    progress_token: str | int,
    redactor: ModelVisibleServerProtocolTextRedactor | None = None,
) -> JSONObject | None:
    if item.kind is StreamItemKind.ANSWER_DELTA:
        deltas = _model_visible_stream_deltas(
            item.text_delta or "",
            redactor,
        )
        if not deltas:
            return None
        message: dict[str, JSONValue] = {
            "type": "answer.delta",
            "delta": "".join(deltas),
        }
    elif item.kind is StreamItemKind.STREAM_COMPLETED:
        message = {"type": "answer.completed"}
    elif item.kind is StreamItemKind.STREAM_CANCELLED:
        message = {"type": "stream.cancelled"}
    elif item.kind is StreamItemKind.STREAM_ERRORED:
        message = {"type": "stream.errored"}
    else:
        return None
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {
            "progressToken": progress_token,
            "progress": item.sequence,
            "message": dumps(message, separators=(",", ":")),
        },
    }


def _mcp_model_text_flush_notifications(
    state: _MCPStreamProjectionState,
    progress_token: str | int,
    progress: int,
) -> list[JSONObject]:
    assert isinstance(state, _MCPStreamProjectionState)
    assert isinstance(progress, int) and not isinstance(progress, bool)
    notifications: list[JSONObject] = []
    for answer_delta in state.answer_redactor.flush():
        if answer_delta:
            message: dict[str, JSONValue] = {
                "type": "answer.delta",
                "delta": answer_delta,
            }
            notifications.append(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/progress",
                    "params": {
                        "progressToken": progress_token,
                        "progress": progress,
                        "message": dumps(message, separators=(",", ":")),
                    },
                }
            )
    return notifications


def _canonical_error_message(
    snapshot: ProtocolStreamSnapshot,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    terminal = snapshot.terminal_snapshot
    if terminal.outcome is StreamTerminalOutcome.ERRORED and isinstance(
        terminal.data, dict
    ):
        message = terminal.data.get("message")
        if isinstance(message, str) and message:
            return sanitize_server_protocol_text(
                message,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            )
    return "Stream errored."


def _canonical_reasoning_deltas(
    item: CanonicalStreamItem,
    redactor: ModelVisibleServerProtocolTextRedactor | None = None,
) -> tuple[str, ...] | None:
    if item.kind is StreamItemKind.REASONING_DELTA:
        return _model_visible_stream_deltas(
            item.text_delta or "",
            redactor,
        )
    if item.kind in (
        StreamItemKind.REASONING_DONE,
        StreamItemKind.STREAM_STARTED,
        StreamItemKind.STREAM_COMPLETED,
        StreamItemKind.STREAM_ERRORED,
        StreamItemKind.STREAM_CANCELLED,
        StreamItemKind.STREAM_INPUT_REQUIRED,
        StreamItemKind.STREAM_CLOSED,
        StreamItemKind.USAGE_UPDATE,
        StreamItemKind.USAGE_COMPLETED,
    ):
        return ()
    return None


def _canonical_reasoning_delta(item: CanonicalStreamItem) -> str | None:
    deltas = _canonical_reasoning_deltas(item)
    if deltas is None:
        return None
    return "".join(deltas)


def _model_visible_stream_deltas(
    value: str,
    redactor: ModelVisibleServerProtocolTextRedactor | None,
) -> tuple[str, ...]:
    if redactor is not None:
        return redactor.push(value)
    sanitized = sanitize_model_visible_server_protocol_text(value)
    return (sanitized,) if sanitized else ()


def _canonical_tool_notification(
    item: CanonicalStreamItem,
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONObject | None:
    if item.kind is not StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
        return None
    delta = sanitize_server_protocol_text(
        item.text_delta or "",
        output_redaction_settings=output_redaction_settings,
        protocol="mcp",
    )
    tool_call_id = item.correlation.tool_call_id
    data = item.data if isinstance(item.data, dict) else {}
    tool_name = _protocol_tool_name(item, data)
    name = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("name"),
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    arguments = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("arguments"),
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    has_call_metadata = "name" in data or "arguments" in data
    if not delta:
        if has_call_metadata and isinstance(tool_call_id, str):
            return _tool_call_notification(
                tool_call_id=tool_call_id,
                name=name,
                arguments=arguments,
            )
        return None
    message: dict[str, JSONValue] = {
        "type": "tool.input_delta",
        "delta": delta,
    }
    if tool_call_id is not None:
        message["toolCallId"] = tool_call_id
    if "name" in data:
        message["name"] = name
    if "arguments" in data:
        message["arguments"] = arguments
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "data": message,
        },
    }


def _canonical_flow_notification(item: CanonicalStreamItem) -> JSONObject:
    assert item.kind is StreamItemKind.FLOW_EVENT
    metadata = canonical_flow_public_metadata(item)
    message: dict[str, JSONValue] = {
        "type": "flow.event",
        "sequence": item.sequence,
        "metadata": cast(JSONValue, metadata),
    }
    event_type = metadata.get("event_type")
    if isinstance(event_type, str) and event_type:
        message["event"] = event_type
    flow_run_id = item.correlation.flow_run_id
    if flow_run_id is not None:
        message["flowRunId"] = flow_run_id
    node_id = item.correlation.node_id
    if node_id is not None:
        message["nodeId"] = node_id
    parent_sequence = item.correlation.parent_sequence
    if parent_sequence is not None:
        message["parentSequence"] = parent_sequence
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "data": message,
        },
    }


def _canonical_tool_execution_notification(
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> JSONObject | None:
    if item.kind not in (
        StreamItemKind.STREAM_DIAGNOSTIC,
        StreamItemKind.TOOL_EXECUTION_STARTED,
        StreamItemKind.TOOL_EXECUTION_COMPLETED,
        StreamItemKind.TOOL_EXECUTION_ERROR,
    ):
        return None
    data = item.data if isinstance(item.data, dict) else {}
    tool_call_id = item.correlation.tool_call_id
    if not isinstance(tool_call_id, str):
        return None
    tool_name = _protocol_tool_name(item, data)
    name = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("name"),
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    arguments = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get("arguments"),
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {"id": tool_call_id, "name": name, "arguments": arguments},
    )
    if isinstance(tool_name, str) and tool_name:
        tool_summary["name"] = name
    if arguments is not None:
        tool_summary["arguments"] = arguments

    if item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
        timings = cast(
            JSONValue,
            sanitize_server_protocol_value(
                data.get("timings"),
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )
        if isinstance(timings, dict):
            tool_summary["started"] = timings.get("started")
        return _tool_call_notification(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
        )
    if item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
        diagnostic = cast(
            JSONValue,
            sanitize_server_protocol_value(
                data.get("diagnostic"),
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )
        tool_summary["diagnostic"] = diagnostic
        return _tool_diagnostic_notification(
            tool_call_id=tool_call_id,
            name=name,
            arguments=arguments,
            diagnostic=diagnostic,
            timings=cast(
                JSONValue,
                sanitize_server_protocol_value(
                    data.get("timings"),
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                ),
            ),
        )

    payload_key = (
        "error"
        if item.kind is StreamItemKind.TOOL_EXECUTION_ERROR
        else "result"
    )
    payload_value = cast(
        JSONValue,
        sanitize_server_protocol_value(
            data.get(payload_key),
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
            protocol="mcp",
        ),
    )
    tool_summary[payload_key] = payload_value
    return _tool_result_notification(
        tool_call_id=tool_call_id,
        name=name,
        arguments=arguments,
        result=payload_value if payload_key == "result" else None,
        error=payload_value if payload_key == "error" else None,
        timings=cast(
            JSONValue,
            sanitize_server_protocol_value(
                data.get("timings"),
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        ),
    )


def _tool_call_notification(
    *,
    tool_call_id: str,
    name: JSONValue,
    arguments: JSONValue,
) -> JSONObject:
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "data": {
                "type": "tool.call",
                "toolCallId": tool_call_id,
                "name": name,
                "arguments": arguments,
            },
        },
    }


def _tool_diagnostic_notification(
    *,
    tool_call_id: str,
    name: JSONValue,
    arguments: JSONValue,
    diagnostic: JSONValue,
    timings: JSONValue,
) -> JSONObject:
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "warning",
            "data": {
                "type": "tool.diagnostic",
                "toolCallId": tool_call_id,
                "name": name,
                "arguments": arguments,
                "diagnostic": diagnostic,
                "timings": timings if isinstance(timings, dict) else {},
            },
        },
    }


def _tool_result_notification(
    *,
    tool_call_id: str,
    name: JSONValue,
    arguments: JSONValue,
    result: JSONValue,
    error: JSONValue,
    timings: JSONValue,
) -> JSONObject:
    message: dict[str, JSONValue] = {
        "type": "tool.result",
        "toolCallId": tool_call_id,
        "name": name,
        "arguments": arguments,
        "timings": timings if isinstance(timings, dict) else {},
    }
    if error is not None:
        message["error"] = error
    elif result is not None:
        message["resultDelta"] = result
    return {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {
            "level": "info",
            "data": message,
        },
    }


async def _canonical_tool_resource_notifications(
    *,
    item: CanonicalStreamItem,
    tool_summaries: dict[str, dict[str, JSONValue]],
    resources: dict[str, MCPResource],
    resource_store: MCPResourceStore,
    base_path: str,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> AsyncIterator[JSONObject]:
    if item.kind not in (
        StreamItemKind.TOOL_EXECUTION_OUTPUT,
        StreamItemKind.TOOL_EXECUTION_PROGRESS,
    ):
        return

    tool_call_id = item.correlation.tool_call_id
    if tool_call_id is None:
        return
    data = item.data
    if not isinstance(data, dict):
        return
    category = data.get("category")
    tool_name = _protocol_tool_name(item, data)
    content = _canonical_tool_resource_content(
        item,
        tool_name=tool_name,
        output_redaction_settings=output_redaction_settings,
    )
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT and category not in {
        "stdout",
        "stderr",
        "log",
        "logs",
    }:
        return
    if (
        item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS
        and category != "progress"
    ):
        return
    if not content:
        return

    name = "logs" if category == "log" else str(category)
    resource_key = f"{tool_call_id}:{name}"
    resource = resources.get(resource_key)
    if resource is None:
        stored_resource = await resource_store.create(
            base_path=base_path, initial_text=content
        )
        resource = replace(stored_resource, text=content)
    else:
        stored_resource = await resource_store.append(resource.id, content)
        # The shared store is lossy retained history; this per-request copy
        # stays lossless while the active MCP response is being emitted.
        resource = replace(stored_resource, text=resource.text + content)
    resources[resource_key] = resource

    tool_summary = tool_summaries.setdefault(
        tool_call_id,
        {
            "id": tool_call_id,
            "name": tool_name,
            "arguments": None,
        },
    )
    if tool_name and not tool_summary.get("name"):
        tool_summary["name"] = tool_name
    existing_resources = tool_summary.setdefault("resources", [])
    if isinstance(existing_resources, list):
        _append_tool_summary_resource(
            existing_resources, uri=resource.uri, name=name
        )

    yield _resource_notification(resource)


def _append_tool_summary_resource(
    resources: list[JSONValue],
    *,
    uri: str,
    name: str,
) -> None:
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        if resource.get("uri") == uri and resource.get("name") == name:
            return
    resources.append({"uri": uri, "name": name})


def _canonical_tool_resource_content(
    item: CanonicalStreamItem,
    *,
    tool_name: str | None = None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    if item.kind is StreamItemKind.TOOL_EXECUTION_OUTPUT:
        data = item.data if isinstance(item.data, dict) else {}
        content = data.get("content", item.text_delta)
        if not isinstance(content, str):
            return ""
        return _mcp_protocol_resource_text(
            content,
            tool_name=tool_name,
            output_redaction_settings=output_redaction_settings,
        )
    if item.kind is StreamItemKind.TOOL_EXECUTION_PROGRESS:
        data = item.data if isinstance(item.data, dict) else {}
        content = data.get("content")
        if isinstance(content, str):
            return _mcp_protocol_resource_text(
                content,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
            )
        progress = data.get("progress")
        return (
            to_json(
                sanitize_server_protocol_value(
                    {"progress": progress},
                    tool_name=tool_name,
                    output_redaction_settings=output_redaction_settings,
                    protocol="mcp",
                )
            )
            if progress is not None
            else ""
        )
    return ""


def _mcp_protocol_resource_text(
    value: str,
    *,
    tool_name: str | None,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> str:
    if isinstance(tool_name, str) and tool_name.startswith("skills."):
        return to_json(
            sanitize_server_protocol_value(
                {"content": value},
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            )
        )
    return sanitize_server_protocol_text(
        value,
        output_redaction_settings=output_redaction_settings,
        protocol="mcp",
    )


def _usage_count(
    usage: object | None,
    key: str,
    fallback: int,
    *,
    aliases: tuple[str, ...] = (),
) -> int:
    for usage_mapping in protocol_stream_usage_mappings(usage):
        for usage_key in (key, *aliases):
            value = usage_mapping.get(usage_key)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
    return fallback


def _metadata_string(
    metadata: Mapping[str, object],
    key: str,
) -> str | None:
    value = metadata.get(key)
    return value if isinstance(value, str) else None


def _protocol_tool_name(
    item: CanonicalStreamItem,
    data: Mapping[str, object],
) -> str | None:
    name = data.get("name")
    if isinstance(name, str) and name:
        return name
    return _metadata_string(item.metadata, "tool_name")


def _merge_canonical_tool_call_arguments(
    tool_summaries: dict[str, dict[str, JSONValue]],
    tool_call_arguments: Mapping[str, str],
    *,
    output_redaction_settings: ServerOutputRedactionSettings | None = None,
) -> None:
    for tool_call_id, arguments in tool_call_arguments.items():
        tool_summary = tool_summaries.setdefault(
            tool_call_id,
            {
                "id": tool_call_id,
                "name": None,
                "arguments": arguments,
            },
        )
        raw_tool_name = tool_summary.get("name")
        tool_name = raw_tool_name if isinstance(raw_tool_name, str) else None
        tool_summary["arguments"] = cast(
            JSONValue,
            sanitize_server_protocol_value(
                arguments,
                tool_name=tool_name,
                output_redaction_settings=output_redaction_settings,
                protocol="mcp",
            ),
        )


async def _close_response_iterator(response: StreamResponse) -> None:
    iterator = getattr(response, "_response_iterator", None)
    if iterator and hasattr(iterator, "aclose"):
        try:
            await cast(SupportsAclose, iterator).aclose()
        except Exception:  # pragma: no cover - best effort cleanup
            pass


def _resource_notification(resource: MCPResource) -> JSONObject:
    resource_payload: dict[str, JSONValue] = {
        "uri": resource.uri,
        "mimeType": resource.mime_type,
        "revision": resource.revision,
        "httpUri": resource.http_uri,
    }
    if resource.closed:
        resource_payload["closed"] = True
    else:
        resource_payload["delta"] = {"set": {"text": resource.text}}
    params: dict[str, JSONValue] = {
        "uri": resource.uri,
        "resources": [resource_payload],
    }
    return {
        "jsonrpc": "2.0",
        "method": "notifications/resources/updated",
        "params": params,
    }


async def _close_mcp_resource_notifications(
    resource_store: MCPResourceStore,
    resources: Mapping[str, MCPResource],
) -> tuple[JSONObject, ...]:
    notifications: list[JSONObject] = []
    resource_ids = [resource.id for resource in resources.values()]
    for closed in await resource_store.close_many(resource_ids):
        notifications.append(_resource_notification(closed))
    return tuple(notifications)


async def _collect_terminal_mcp_messages(
    resource_store: MCPResourceStore,
    resources: Mapping[str, MCPResource],
    terminal_message: JSONObject,
) -> tuple[JSONObject, ...]:
    messages = list(
        await _close_mcp_resource_notifications(resource_store, resources)
    )
    await resource_store.prune_closed()
    messages.append(terminal_message)
    return tuple(messages)


async def _terminal_mcp_messages(
    resource_store: MCPResourceStore,
    resources: Mapping[str, MCPResource],
    terminal_message: JSONObject,
) -> AsyncIterator[JSONObject]:
    for message in await _collect_terminal_mcp_messages(
        resource_store, resources, terminal_message
    ):
        yield message


def _get_resource_store(request: Request) -> MCPResourceStore:
    store = getattr(request.app.state, "mcp_resource_store", None)
    if store is None:
        store = MCPResourceStore()
        request.app.state.mcp_resource_store = store
    assert isinstance(store, MCPResourceStore)
    return store


async def close_mcp_state(app: object) -> None:
    """Close MCP sessions, operations, and task state owned by one app."""
    state = getattr(app, "state", None)
    if state is None:
        return
    registry = getattr(state, "mcp_form_session_registry", None)
    if isinstance(registry, MCPFormSessionRegistry):
        await registry.close_all()
        delattr(state, "mcp_form_session_registry")
    cancellations = _mcp_cancellations(app)
    if cancellations is not None:
        for event in tuple(cancellations.values()):
            event.set()
        cancellations.clear()
        delattr(state, "mcp_stream_cancellations")
    background = getattr(state, "mcp_background_tasks", {})
    tasks = (
        tuple(task for owned in background.values() for task in owned)
        if isinstance(background, dict)
        else ()
    )
    if hasattr(state, "mcp_background_tasks"):
        delattr(state, "mcp_background_tasks")
    for task in tasks:
        if isinstance(task, Task) and not task.done():
            task.cancel()
    if tasks:
        await gather(*tasks, return_exceptions=True)
    controller = getattr(state, "mcp_task_controller", None)
    if isinstance(controller, MCPTaskController):
        await controller.close()
        delattr(state, "mcp_task_controller")


async def _iter_jsonrpc_messages(
    request: Request,
) -> AsyncGenerator[JSONObject, None]:
    if hasattr(request.state, "_mcp_message_iter"):
        iterator = cast(
            AsyncIterator[JSONObject], request.state._mcp_message_iter
        )
        delattr(request.state, "_mcp_message_iter")
        async for message in iterator:
            yield message
        return

    buffer = ""
    async for chunk in request.stream():
        if not chunk:
            continue
        buffer += chunk.decode("utf-8")
        while RS in buffer:
            segment, buffer = buffer.split(RS, 1)
            segment = segment.strip()
            if not segment:
                continue
            try:
                obj = loads(segment)
            except JSONDecodeError as exc:
                raise HTTPException(
                    status_code=400, detail="Invalid MCP payload"
                ) from exc
            if not isinstance(obj, dict):
                raise HTTPException(
                    status_code=400, detail="Invalid MCP payload"
                )
            yield cast(JSONObject, obj)
    if buffer.strip():
        try:
            obj2 = loads(buffer)
        except JSONDecodeError as exc:
            raise HTTPException(
                status_code=400, detail="Invalid MCP payload"
            ) from exc
        if not isinstance(obj2, dict):
            raise HTTPException(status_code=400, detail="Invalid MCP payload")
        yield cast(JSONObject, obj2)
