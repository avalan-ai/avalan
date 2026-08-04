"""Expose authenticated structured-input control over HTTP."""

from ..agent.execution import AttachedInteractionRuntime
from ..agent.orchestrator import MemorySynchronizableResponse, Orchestrator
from ..interaction.broker import InteractionBroker, InteractionBrokerResult
from ..interaction.codec import decode_input_answer, encode_input_question
from ..interaction.entities import (
    AnsweredResolution,
    AnswerProvenance,
    DeclinedResolution,
    InputAnswer,
    InputRequest,
    InputRequestId,
    InteractionStoreRevision,
    PrincipalScope,
    RequestState,
    ResolutionIdempotencyKey,
    StateRevision,
)
from ..interaction.error import (
    InputContractError,
    InputErrorCode,
    InputValidationError,
)
from ..interaction.handler import (
    InputHandlerContext,
    InputHandlerOutcome,
)
from ..interaction.policy import (
    InteractionActor,
    InteractionAuthorizationDecision,
    InteractionAuthorizer,
    InteractionDisclosure,
    InteractionOperation,
    InteractionPolicy,
    InteractionRequestAuthorizationTarget,
)
from ..interaction.security import enforce_task_input_request_policy
from ..interaction.store import (
    CancelInteractionApplied,
    CancelInteractionCommand,
    CancelInteractionRejected,
    InteractionCorrelation,
    InteractionRecord,
    InteractionStoreReplayed,
    InteractionTerminalMetadata,
    ResolveInteractionApplied,
    ResolveInteractionCommand,
    ResolveInteractionRejected,
    ScopedInteractionLookup,
    WaitForInteractionChangeCommand,
)
from ..model.stream import (
    StreamConsumerProjection,
    StreamItemKind,
    StreamTerminalOutcome,
)
from ..types import LooseJsonValue
from ..utils import to_json
from .entities import (
    ModelVisibleServerProtocolTextRedactor,
    ServerOutputRedactionSettings,
)
from .routers.streaming import (
    cleanup_stream_sources,
    stream_terminal_succeeded,
)
from .sse import sse_headers, sse_message

from asyncio import (
    CancelledError,
    Event,
    Lock,
    Task,
    create_task,
    gather,
    get_running_loop,
    shield,
)
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from inspect import iscoroutinefunction
from typing import Any, Protocol, cast, final
from urllib.parse import quote
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

TASK_INPUT_EXTENSION = "https://avalan.ai/extensions/task-input/v1"
TASK_INPUT_EXTENSION_HEADER = "Avalan-Extensions"
_CACHE_CONTROL_HEADERS = {
    "Cache-Control": "private, no-store",
    TASK_INPUT_EXTENSION_HEADER: TASK_INPUT_EXTENSION,
}
_MAX_JSON_SAFE_INTEGER = 9_007_199_254_740_991
_TERMINAL_ENTRY_RETENTION_LIMIT = 128


class ServerInteractionHandling(StrEnum):
    """Identify negotiated transport handling for task input."""

    ATTACHED = "attached"
    DETACHED = "detached"
    UNAVAILABLE = "unavailable"


class ServerInteractionSurface(StrEnum):
    """Identify one server protocol projection."""

    CHAT = "chat"
    RESPONSES = "responses"
    SERVER = "server"


class InteractionPrincipalResolver(Protocol):
    """Resolve one authenticated interaction actor from an HTTP request."""

    async def __call__(self, request: Request) -> InteractionActor | None:
        """Return the authenticated actor or no principal."""
        ...


class ServerResponsesProjection(Protocol):
    """Preserve Responses projection state across transport segments."""

    def stream_messages(
        self,
        projection: StreamConsumerProjection,
    ) -> tuple[str, ...]:
        """Return ordered Responses messages for one canonical projection."""
        ...

    def finish_stream(
        self,
        terminal: StreamConsumerProjection,
    ) -> tuple[str, ...]:
        """Return ordered terminal Responses messages."""
        ...

    def observe_json(self, projection: StreamConsumerProjection) -> None:
        """Accumulate one canonical projection for a JSON response."""
        ...

    def json_body(
        self,
        terminal: StreamConsumerProjection,
        response: object,
    ) -> dict[str, Any]:
        """Return one semantically complete Responses JSON body."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ServerInteractionConfiguration:
    """Configure the authenticated HTTP interaction boundary."""

    broker: InteractionBroker = field(repr=False)
    principal_resolver: InteractionPrincipalResolver = field(repr=False)
    authorizer: InteractionAuthorizer = field(repr=False)
    policy: InteractionPolicy = field(
        default_factory=InteractionPolicy,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.policy, InteractionPolicy):
            raise TypeError("policy must be an interaction policy")
        broker_policy = getattr(self.broker, "policy", self.policy)
        if isinstance(broker_policy, InteractionPolicy) and (
            broker_policy != self.policy
        ):
            raise TypeError(
                "broker and server interaction policies must match"
            )
        _require_async_method(self.broker, "inspect", "broker")
        _require_async_method(self.broker, "resolve", "broker")
        _require_async_method(self.broker, "cancel", "broker")
        _require_async_method(self.broker, "wait", "broker")
        _require_async_callable(
            self.principal_resolver,
            "principal_resolver",
        )
        _require_async_method(self.authorizer, "authorize", "authorizer")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ServerInteractionLifecycleEvent:
    """Carry one content-safe run-scoped lifecycle observation."""

    sequence: int
    type: str
    request_id: str
    state: str | None = None
    surface: str | None = None
    validation_code: str | None = None
    idempotent: bool | None = None

    def __post_init__(self) -> None:
        if (
            type(self.sequence) is not int
            or self.sequence < 0
            or self.sequence > _MAX_JSON_SAFE_INTEGER
        ):
            raise ValueError("lifecycle sequence must be a safe integer")
        for field_name in ("type", "request_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        for field_name in ("state", "surface", "validation_code"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be non-empty or None")
        if self.idempotent is not None and type(self.idempotent) is not bool:
            raise ValueError("idempotent must be a boolean or None")

    def to_dict(self) -> dict[str, LooseJsonValue]:
        """Return the approved content-safe wire projection."""
        result: dict[str, LooseJsonValue] = {
            "sequence": self.sequence,
            "type": self.type,
            "request_id": self.request_id,
        }
        for field_name in (
            "state",
            "surface",
            "validation_code",
            "idempotent",
        ):
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        return result


@final
@dataclass(slots=True, kw_only=True)
class ServerDetachedSegment:
    """Retain one transport segment without cancelling its logical run."""

    iterator: AsyncIterator[StreamConsumerProjection]
    response: object
    orchestrator: Orchestrator
    protocol: ServerInteractionSurface
    response_id: str
    created: int
    model_id: str
    output_redaction_settings: ServerOutputRedactionSettings
    responses_projection: ServerResponsesProjection | None = None
    choice_count: int = 1
    pending_next: Task[StreamConsumerProjection] | None = field(
        default=None,
        repr=False,
    )
    resume_claimed: bool = field(default=False, init=False, repr=False)
    resume_exhausted: bool = field(default=False, init=False, repr=False)
    lock: Lock = field(default_factory=Lock, repr=False)
    completed_json: dict[str, Any] | None = None
    closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if not hasattr(self.iterator, "__anext__"):
            raise TypeError("iterator must be asynchronous")
        if not isinstance(self.orchestrator, Orchestrator):
            raise TypeError("orchestrator must be an Orchestrator")
        if not isinstance(self.protocol, ServerInteractionSurface):
            raise TypeError("protocol must be a server interaction surface")
        for field_name in ("response_id", "model_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be non-empty")
        if type(self.created) is not int:
            raise TypeError("created must be an integer")
        if type(self.choice_count) is not int or self.choice_count < 1:
            raise ValueError("choice_count must be positive")
        if not isinstance(
            self.output_redaction_settings,
            ServerOutputRedactionSettings,
        ):
            raise TypeError(
                "output_redaction_settings must be server settings"
            )
        if (
            self.responses_projection is not None
            and self.protocol is not ServerInteractionSurface.RESPONSES
        ):
            raise TypeError(
                "Responses projection requires the Responses protocol"
            )

    async def next_projection(self) -> StreamConsumerProjection:
        """Read one item without propagating transport cancellation."""
        async with self.lock:
            if self.closed:
                raise StopAsyncIteration
            task = self.pending_next
            if task is None:
                task = create_task(_read_projection(self.iterator))
                self.pending_next = task
        try:
            return await shield(task)
        finally:
            if task.done():
                async with self.lock:
                    if self.pending_next is task:
                        self.pending_next = None

    async def aclose(self, *, cancelled: bool) -> None:
        """Cancel, join, and close every retained stream source."""
        async with self.lock:
            if self.closed:
                return
            self.closed = True
            task = self.pending_next
            self.pending_next = None
        if task is not None:
            if not task.done():
                task.cancel()
            await gather(task, return_exceptions=True)
        await cleanup_stream_sources(
            self.response,
            self.iterator,
            cancelled=cancelled,
        )

    def claim_resume(self) -> None:
        """Claim sole ownership of the next detached transport segment."""
        if self.resume_claimed or self.resume_exhausted:
            raise _ServerHTTPError.conflict()
        self.resume_claimed = True

    def release_resume(self, *, exhausted: bool) -> None:
        """Release detached transport ownership after one consumer."""
        if not self.resume_claimed:
            raise RuntimeError("detached transport is not claimed")
        self.resume_claimed = False
        self.resume_exhausted = self.resume_exhausted or exhausted


@final
@dataclass(slots=True, kw_only=True)
class _InteractionEntry:
    """Bind one request to its authenticated run and retained segment."""

    run: "ServerInteractionRun"
    correlation: InteractionCorrelation
    request: InputRequest
    lifecycle: list[ServerInteractionLifecycleEvent] = field(
        default_factory=list
    )
    lock: Lock = field(default_factory=Lock, repr=False)
    cancel_idempotency_key: str | None = None
    segment: ServerDetachedSegment | None = None
    last_record: InteractionRecord | None = None

    @property
    def principal(self) -> PrincipalScope:
        """Return the complete trusted scope for this request."""
        return self.request.origin.principal

    def append(
        self,
        event_type: str,
        *,
        state: str | None = None,
        surface: str | None = None,
        validation_code: str | None = None,
        idempotent: bool | None = None,
    ) -> None:
        """Append one content-safe lifecycle event."""
        self.lifecycle.append(
            ServerInteractionLifecycleEvent(
                sequence=len(self.lifecycle),
                type=event_type,
                request_id=str(self.request.request_id),
                state=state,
                surface=surface,
                validation_code=validation_code,
                idempotent=idempotent,
            )
        )


@final
class _ServerAttachedInputHandler:
    """Keep the broker request pending for authenticated HTTP resolution."""

    def __init__(self, run: "ServerInteractionRun") -> None:
        self._run = run

    async def __call__(
        self,
        context: InputHandlerContext,
    ) -> InputHandlerOutcome:
        entry = await self._run.register(context.request)
        entry.append(
            "input_request.delivered",
            state=RequestState.PENDING.value,
            surface=self._run.surface.value,
        )
        try:
            await Event().wait()
        except CancelledError:
            entry.append("input_request.wait_ended")
            raise
        raise AssertionError("an interaction wait cannot finish without state")


@final
class ServerInteractionRun:
    """Own one invocation's runtime and transport state."""

    def __init__(
        self,
        *,
        service: "ServerInteractionService",
        actor: InteractionActor,
        handling: ServerInteractionHandling,
        surface: ServerInteractionSurface,
    ) -> None:
        self._service = service
        self.actor = actor
        self.handling = handling
        self.surface = surface
        self.runtime = AttachedInteractionRuntime(
            broker=service.configuration.broker,
            actor=actor,
            handler=_ServerAttachedInputHandler(self),
            policy=service.configuration.policy,
        )
        self._entries: dict[InputRequestId, _InteractionEntry] = {}
        self._latest_request_id: InputRequestId | None = None
        self._provisional_segment: ServerDetachedSegment | None = None
        self._lock = Lock()

    async def register(self, request: InputRequest) -> _InteractionEntry:
        """Register one broker-created request under its complete scope."""
        enforce_task_input_request_policy(
            request,
            "server_interaction.request",
        )
        correlation = InteractionCorrelation.from_request(request)
        async with self._lock:
            entry = self._entries.get(request.request_id)
            if entry is None:
                entry = _InteractionEntry(
                    run=self,
                    correlation=correlation,
                    request=request,
                )
                entry.append(
                    "input_request.created",
                    state=RequestState.PENDING.value,
                )
                self._entries[request.request_id] = entry
                self._latest_request_id = request.request_id
                provisional_segment = self._provisional_segment
                if provisional_segment is not None:
                    entry.segment = provisional_segment
                    entry.append(
                        "execution.suspended",
                        state=RequestState.PENDING.value,
                        surface=self.surface.value,
                    )
                await self._service._register(
                    entry,
                    provisional_segment=provisional_segment,
                )
                self._provisional_segment = None
            elif entry.correlation != correlation:
                raise RuntimeError("interaction correlation changed")
        return entry

    async def entry_for_projection(
        self,
        projection: StreamConsumerProjection,
    ) -> _InteractionEntry | None:
        """Return this run's exact entry for a canonical projection."""
        request_id = projection.correlation.request_id
        if request_id is None:
            return None
        async with self._lock:
            return self._entries.get(InputRequestId(request_id))

    async def install_segment(
        self,
        segment: ServerDetachedSegment,
    ) -> _InteractionEntry:
        """Retain iterator ownership for a later authenticated reconnect."""
        async with self._lock:
            request_id = self._latest_request_id
            if request_id is None:
                if (
                    self._provisional_segment is not None
                    and self._provisional_segment is not segment
                ):
                    raise RuntimeError("detached segment already exists")
                await self._service._retain_provisional_segment(segment)
                self._provisional_segment = segment
                raise RuntimeError("input has not been published")
            entry = self._entries[request_id]
            if entry.segment is not None and entry.segment is not segment:
                raise RuntimeError("detached segment already exists")
            entry.segment = segment
            entry.append(
                "execution.suspended",
                state=RequestState.PENDING.value,
                surface=self.surface.value,
            )
            return entry

    async def extension_events(
        self,
        projection: StreamConsumerProjection,
    ) -> tuple[dict[str, LooseJsonValue], ...]:
        """Project content-safe frozen extension lifecycle events."""
        entry = await self.entry_for_projection(projection)
        if entry is None:
            return ()
        request = entry.request
        request_id = str(request.request_id)
        request_href = input_request_href(request_id)
        if projection.kind is StreamItemKind.INTERACTION_CREATED:
            return (
                {
                    "type": "input_request.created",
                    "request_id": request_id,
                    "run_id": str(request.origin.run_id),
                    "turn_id": str(request.origin.turn_id),
                    "required": request.required,
                    "request_href": request_href,
                },
            )
        if projection.kind is StreamItemKind.INTERACTION_PENDING:
            return (
                {
                    "type": "input_request.presented",
                    "request_id": request_id,
                    "surface": self.surface.value,
                },
            )
        terminal_states = {
            StreamItemKind.INTERACTION_ANSWERED: "answered",
            StreamItemKind.INTERACTION_DECLINED: "declined",
            StreamItemKind.INTERACTION_CANCELLED: "cancelled",
            StreamItemKind.INTERACTION_TIMED_OUT: "timed_out",
            StreamItemKind.INTERACTION_UNAVAILABLE: "unavailable",
            StreamItemKind.INTERACTION_EXPIRED: "expired",
            StreamItemKind.INTERACTION_SUPERSEDED: "superseded",
        }
        resolution = terminal_states.get(projection.kind)
        if resolution is None:
            return ()
        latest = await self._service._record_for_entry(
            entry,
            InteractionOperation.INSPECT,
        )
        provenance = (
            latest.request.resolution.provenance.value
            if latest.request.resolution is not None
            else AnswerProvenance.EXTERNAL_CONTROLLER.value
        )
        return (
            {
                "type": "input_request.resolved",
                "request_id": request_id,
                "resolution": resolution,
                "provenance": provenance,
            },
        )

    async def input_required_event(
        self,
    ) -> dict[str, LooseJsonValue]:
        """Return the frozen segment-level detached event."""
        async with self._lock:
            request_id = self._latest_request_id
            if request_id is None:
                raise RuntimeError("input is not pending")
            request = self._entries[request_id].request
        return {
            "type": "response.input_required",
            "request_id": str(request.request_id),
            "continuation_id": str(request.continuation_id),
            "detached_resumption_available": True,
            "request_href": input_request_href(str(request.request_id)),
        }

    async def input_required_envelope(self) -> dict[str, LooseJsonValue]:
        """Return the frozen non-streaming detached envelope."""
        event = await self.input_required_event()
        return {
            "status": "input_required",
            "request_id": event["request_id"],
            "continuation_id": event["continuation_id"],
            "detached_resumption_available": True,
        }


@final
class ServerInteractionService:
    """Coordinate authenticated run-scoped HTTP interaction access."""

    def __init__(self, configuration: ServerInteractionConfiguration) -> None:
        if type(configuration) is not ServerInteractionConfiguration:
            raise TypeError(
                "configuration must be a server interaction configuration"
            )
        self.configuration = configuration
        self._entries: dict[
            tuple[PrincipalScope, InputRequestId],
            _InteractionEntry,
        ] = {}
        self._terminal_entries: dict[
            tuple[PrincipalScope, InputRequestId],
            _InteractionEntry,
        ] = {}
        self._provisional_segments: dict[int, ServerDetachedSegment] = {}
        self._lock = Lock()
        self._closed = False

    async def authenticate(self, request: Request) -> InteractionActor:
        """Resolve an authenticated actor without reading interaction state."""
        try:
            actor = await self.configuration.principal_resolver(request)
        except Exception:
            actor = None
        if not isinstance(actor, InteractionActor):
            raise _ServerHTTPError.authentication_required()
        return actor

    async def start_run(
        self,
        request: Request,
        *,
        handling: ServerInteractionHandling,
        surface: ServerInteractionSurface,
    ) -> ServerInteractionRun:
        """Create one authenticated run-scoped interaction runtime."""
        actor = await self.authenticate(request)
        return ServerInteractionRun(
            service=self,
            actor=actor,
            handling=handling,
            surface=surface,
        )

    async def _register(
        self,
        entry: _InteractionEntry,
        *,
        provisional_segment: ServerDetachedSegment | None = None,
    ) -> None:
        key = (entry.principal, entry.request.request_id)
        async with self._lock:
            if self._closed:
                raise RuntimeError("interaction service is closed")
            previous = self._entries.get(key)
            if previous is not None and previous is not entry:
                raise RuntimeError("opaque request identity collision")
            self._entries[key] = entry
            if provisional_segment is not None:
                self._provisional_segments.pop(id(provisional_segment), None)

    async def _retain_provisional_segment(
        self,
        segment: ServerDetachedSegment,
    ) -> None:
        """Retain pre-publication transport under service ownership."""
        async with self._lock:
            if self._closed:
                closed = True
            else:
                self._provisional_segments[id(segment)] = segment
                closed = False
        if closed:
            await segment.aclose(cancelled=True)
            raise RuntimeError("interaction service is closed")

    async def aclose(self) -> None:
        """Evict all requests after cancelling and joining retained reads."""
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            entries = tuple(self._entries.values())
            provisional_segments = tuple(self._provisional_segments.values())
            self._entries.clear()
            self._terminal_entries.clear()
            self._provisional_segments.clear()
        await gather(
            *(self._close_entry_segment(entry) for entry in entries),
            *(
                segment.aclose(cancelled=True)
                for segment in provisional_segments
            ),
            return_exceptions=True,
        )

    async def _close_entry_segment(self, entry: _InteractionEntry) -> None:
        """Detach, cancel, and join one entry's retained segment."""
        async with entry.lock:
            segment = entry.segment
            entry.segment = None
        if segment is not None:
            await segment.aclose(cancelled=True)

    async def _evict_entry(self, entry: _InteractionEntry) -> None:
        """Evict one entry after closing its retained segment."""
        key = (entry.principal, entry.request.request_id)
        async with self._lock:
            if self._entries.get(key) is entry:
                del self._entries[key]
            if self._terminal_entries.get(key) is entry:
                del self._terminal_entries[key]
        await self._close_entry_segment(entry)

    async def _retain_terminal_entry(
        self,
        entry: _InteractionEntry,
    ) -> None:
        """Retain one recent terminal entry within the replay bound."""
        key = (entry.principal, entry.request.request_id)
        evicted: list[_InteractionEntry] = []
        retention_limit = _TERMINAL_ENTRY_RETENTION_LIMIT
        async with self._lock:
            if self._entries.get(key) is not entry:
                return
            self._terminal_entries.pop(key, None)
            self._terminal_entries[key] = entry
            while len(self._terminal_entries) > retention_limit:
                oldest_key = next(iter(self._terminal_entries))
                oldest_entry = self._terminal_entries.pop(oldest_key)
                if self._entries.get(oldest_key) is oldest_entry:
                    del self._entries[oldest_key]
                    evicted.append(oldest_entry)
        if evicted:
            await gather(
                *(self._close_entry_segment(item) for item in evicted)
            )

    async def _accept_record(
        self,
        entry: _InteractionEntry,
        projection: InteractionRecord,
    ) -> InteractionRecord:
        """Accept one correlated record and close terminal transport state."""
        if projection.correlation != entry.correlation:
            raise _ServerHTTPError.not_found()
        entry.request = projection.request
        entry.last_record = projection
        if projection.request.state not in {
            RequestState.PENDING,
            RequestState.ANSWERED,
            RequestState.DECLINED,
            RequestState.TIMED_OUT,
        }:
            await self._close_entry_segment(entry)
        if projection.request.state is not RequestState.PENDING:
            await self._retain_terminal_entry(entry)
        return projection

    async def entry(
        self,
        actor: InteractionActor,
        request_id: str,
        operation: InteractionOperation,
    ) -> _InteractionEntry:
        """Return one scope-filtered and operation-authorized entry."""
        if (
            operation is InteractionOperation.RESOLVE
            and not self.configuration.policy.resolve_existing
        ):
            raise _ServerHTTPError.unavailable()
        if not isinstance(request_id, str) or not request_id:
            raise _ServerHTTPError.not_found()
        opaque_id = InputRequestId(request_id)
        async with self._lock:
            entry = self._entries.get((actor.principal, opaque_id))
        if entry is None:
            raise _ServerHTTPError.not_found()
        target = InteractionRequestAuthorizationTarget(
            request_id=entry.request.request_id,
            origin=entry.request.origin,
        )
        try:
            decision = await self.configuration.authorizer.authorize(
                actor,
                operation,
                target,
            )
        except Exception:
            raise _ServerHTTPError.forbidden() from None
        _validate_server_authorization(
            decision,
            actor,
            operation,
            target,
        )
        if not decision.allowed:
            raise _ServerHTTPError.forbidden()
        return entry

    async def _record_for_entry(
        self,
        entry: _InteractionEntry,
        operation: InteractionOperation,
    ) -> InteractionRecord:
        if operation not in {
            InteractionOperation.INSPECT,
            InteractionOperation.WAIT,
            InteractionOperation.RESOLVE,
            InteractionOperation.CANCEL_REQUEST,
        }:
            raise ValueError("unsupported server interaction operation")
        actor = entry.run.actor
        try:
            projection = await self.configuration.broker.inspect(
                ScopedInteractionLookup(
                    actor=actor,
                    correlation=entry.correlation,
                )
            )
        except InputContractError as error:
            raise _http_error_for_code(error.code) from None
        except Exception:
            raise _ServerHTTPError.unavailable() from None
        if isinstance(projection, InteractionTerminalMetadata):
            await self._evict_entry(entry)
            raise _ServerHTTPError.not_found()
        if not isinstance(projection, InteractionRecord):
            raise _ServerHTTPError.not_found()
        return await self._accept_record(entry, projection)

    async def inspect(
        self,
        actor: InteractionActor,
        request_id: str,
    ) -> tuple[_InteractionEntry, InteractionRecord]:
        """Inspect one request after exact scope authorization."""
        entry = await self.entry(
            actor,
            request_id,
            InteractionOperation.INSPECT,
        )
        return entry, await self._record_for_entry(
            entry,
            InteractionOperation.INSPECT,
        )

    async def poll(
        self,
        actor: InteractionActor,
        request_id: str,
        after_store_revision: int | None,
    ) -> tuple[_InteractionEntry, InteractionRecord]:
        """Wait for one coherent newer authorized interaction projection."""
        entry = await self.entry(
            actor,
            request_id,
            InteractionOperation.WAIT,
        )
        record = await self._record_for_entry(
            entry,
            InteractionOperation.WAIT,
        )
        if (
            record.request.state is not RequestState.PENDING
            or after_store_revision is None
            or (int(record.store_revision) > after_store_revision)
        ):
            return entry, record
        try:
            projection = await self.configuration.broker.wait(
                WaitForInteractionChangeCommand(
                    actor=actor,
                    correlation=entry.correlation,
                    after_store_revision=InteractionStoreRevision(
                        after_store_revision
                    ),
                )
            )
        except InputContractError as error:
            raise _http_error_for_code(error.code) from None
        except Exception:
            raise _ServerHTTPError.unavailable() from None
        if isinstance(projection, InteractionTerminalMetadata):
            await self._evict_entry(entry)
            raise _ServerHTTPError.not_found()
        if not isinstance(projection, InteractionRecord):
            raise _ServerHTTPError.not_found()
        return entry, await self._accept_record(entry, projection)

    async def resolve(
        self,
        actor: InteractionActor,
        request_id: str,
        payload: object,
    ) -> tuple[_InteractionEntry, InteractionRecord, bool]:
        """Resolve one exact request through broker-owned atomic semantics."""
        entry = await self.entry(
            actor,
            request_id,
            InteractionOperation.RESOLVE,
        )
        record = await self._record_for_entry(
            entry,
            InteractionOperation.RESOLVE,
        )
        if record.request.state is RequestState.EXPIRED:
            raise _ServerHTTPError.expired()
        try:
            command = _resolution_command(actor, record, payload)
        except InputContractError as error:
            entry.append(
                "input_request.validation_rejected",
                state=record.request.state.value,
                validation_code=error.code.value,
            )
            raise _ServerHTTPError.validation() from None
        try:
            result = await self.configuration.broker.resolve(command)
        except InputContractError as error:
            raise _http_error_for_code(error.code) from None
        except Exception:
            raise _ServerHTTPError.unavailable() from None
        store_result = _store_result(result)
        if isinstance(store_result, ResolveInteractionRejected):
            entry.append(
                "input_request.validation_rejected",
                state=record.request.state.value,
                validation_code=store_result.error.code.value,
            )
            raise _http_error_for_code(store_result.error.code)
        if isinstance(store_result, ResolveInteractionApplied):
            resolved = store_result.record
            idempotent = False
        elif isinstance(store_result, InteractionStoreReplayed):
            resolved = store_result.record
            idempotent = True
        else:
            raise _ServerHTTPError.unavailable()
        if resolved.correlation != entry.correlation:
            raise _ServerHTTPError.not_found()
        entry.request = resolved.request
        entry.last_record = resolved
        entry.append(
            "input_request.resolution_accepted",
            state=resolved.request.state.value,
            surface=ServerInteractionSurface.SERVER.value,
            idempotent=idempotent,
        )
        await self._retain_terminal_entry(entry)
        return entry, resolved, idempotent

    async def cancel(
        self,
        actor: InteractionActor,
        request_id: str,
        payload: object,
    ) -> tuple[_InteractionEntry, InteractionRecord, bool]:
        """Cancel one exact request without conflating transport loss."""
        entry = await self.entry(
            actor,
            request_id,
            InteractionOperation.CANCEL_REQUEST,
        )
        record = await self._record_for_entry(
            entry,
            InteractionOperation.CANCEL_REQUEST,
        )
        try:
            values = _validated_mutation_binding(payload, record)
            if "status" in values or "answers" in values:
                raise InputValidationError(
                    InputErrorCode.INVALID_FORMAT,
                    "binding",
                    "cancel payload contains resolution fields",
                )
            key = _required_string(
                values,
                "idempotency_key",
                maximum=256,
            )
        except InputContractError:
            entry.append(
                "input_request.validation_rejected",
                state=record.request.state.value,
                validation_code=InputErrorCode.INVALID_FORMAT.value,
            )
            raise _ServerHTTPError.validation() from None
        async with entry.lock:
            if entry.cancel_idempotency_key == key:
                return entry, record, True
            if entry.cancel_idempotency_key is not None:
                raise _ServerHTTPError.conflict()
            try:
                result = await self.configuration.broker.cancel(
                    CancelInteractionCommand(
                        actor=actor,
                        correlation=entry.correlation,
                        provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
                        expected_state_revision=StateRevision(
                            cast(int, values["expected_state_revision"])
                        ),
                    )
                )
            except InputContractError as error:
                raise _http_error_for_code(error.code) from None
            except Exception:
                raise _ServerHTTPError.unavailable() from None
            store_result = _store_result(result)
            if isinstance(store_result, CancelInteractionRejected):
                raise _http_error_for_code(store_result.error.code)
            if not isinstance(store_result, CancelInteractionApplied):
                raise _ServerHTTPError.unavailable()
            if store_result.record.correlation != entry.correlation:
                raise _ServerHTTPError.not_found()
            entry.cancel_idempotency_key = key
            entry.request = store_result.record.request
            entry.last_record = store_result.record
            entry.append(
                "input_request.cancelled",
                state=RequestState.CANCELLED.value,
                surface=ServerInteractionSurface.SERVER.value,
            )
        await self._close_entry_segment(entry)
        await self._retain_terminal_entry(entry)
        return entry, store_result.record, False


@dataclass(frozen=True, slots=True)
class _ServerHTTPError(Exception):
    """Carry one frozen content-safe HTTP error."""

    status_code: int
    code: str
    message: str
    headers: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def authentication_required(cls) -> "_ServerHTTPError":
        return cls(
            401,
            "input.authentication_required",
            "Authentication is required.",
            {"WWW-Authenticate": "Bearer"},
        )

    @classmethod
    def forbidden(cls) -> "_ServerHTTPError":
        return cls(403, "input.forbidden", "Operation is not permitted.")

    @classmethod
    def not_found(cls) -> "_ServerHTTPError":
        return cls(404, "input.not_found", "Input request not found.")

    @classmethod
    def validation(cls) -> "_ServerHTTPError":
        return cls(422, "input.validation", "Input submission is invalid.")

    @classmethod
    def stale_revision(cls) -> "_ServerHTTPError":
        return cls(
            409,
            "input.stale_revision",
            "Input request revision is stale.",
        )

    @classmethod
    def conflict(cls) -> "_ServerHTTPError":
        return cls(
            409,
            "input.already_resolved",
            "Input request is already resolved.",
        )

    @classmethod
    def expired(cls) -> "_ServerHTTPError":
        return cls(410, "input.expired", "Input request has expired.")

    @classmethod
    def unavailable(cls) -> "_ServerHTTPError":
        return cls(
            503,
            "input.unavailable",
            "Structured input is unavailable.",
        )


ServerInteractionHTTPError = _ServerHTTPError


def configure_server_interactions(
    app: Any,
    configuration: ServerInteractionConfiguration | None,
) -> None:
    """Install or remove the configured interaction service on an app."""
    previous = getattr(app.state, "interaction_service", None)
    if (
        isinstance(previous, ServerInteractionService)
        and previous.configuration is configuration
    ):
        return
    if isinstance(previous, ServerInteractionService):
        task = get_running_loop().create_task(
            previous.aclose(),
            name="server-interaction-reconfigure-close",
        )
        close_tasks = getattr(
            app.state,
            "interaction_service_close_tasks",
            [],
        )
        close_tasks.append(task)
        app.state.interaction_service_close_tasks = close_tasks
    if configuration is None:
        if hasattr(app.state, "interaction_service"):
            delattr(app.state, "interaction_service")
        return
    app.state.interaction_service = ServerInteractionService(configuration)


async def close_server_interactions(app: Any) -> None:
    """Cancel and join configured and replaced interaction services."""
    service = getattr(app.state, "interaction_service", None)
    if hasattr(app.state, "interaction_service"):
        delattr(app.state, "interaction_service")
    tasks = tuple(getattr(app.state, "interaction_service_close_tasks", ()))
    if hasattr(app.state, "interaction_service_close_tasks"):
        delattr(app.state, "interaction_service_close_tasks")
    if isinstance(service, ServerInteractionService):
        await service.aclose()
    if tasks:
        await gather(*tasks, return_exceptions=True)


def server_interaction_service(request: Request) -> ServerInteractionService:
    """Return configured service or fail closed without request lookup."""
    service = getattr(request.app.state, "interaction_service", None)
    if not isinstance(service, ServerInteractionService):
        raise _ServerHTTPError.unavailable()
    return service


async def prepare_openai_interaction_run(
    request: Request,
    extension: object,
    *,
    surface: ServerInteractionSurface,
) -> ServerInteractionRun | None:
    """Validate extension negotiation and create a run-scoped runtime."""
    header_present = _extension_header_present(request)
    if extension is None and not header_present:
        return None
    if extension is None or not header_present:
        raise _ServerHTTPError.unavailable()
    values = _mapping(extension)
    if set(values) != {"version", "handling"}:
        raise _ServerHTTPError.validation()
    if values.get("version") != "1":
        raise _ServerHTTPError.unavailable()
    try:
        raw_handling = values["handling"]
        if not isinstance(raw_handling, str):
            raise ValueError
        handling = ServerInteractionHandling(raw_handling)
    except (TypeError, ValueError):
        raise _ServerHTTPError.validation() from None
    if handling is ServerInteractionHandling.UNAVAILABLE:
        return None
    service = server_interaction_service(request)
    if not service.configuration.policy.advertise:
        raise _ServerHTTPError.unavailable()
    return await service.start_run(
        request,
        handling=handling,
        surface=surface,
    )


def task_input_extension_from_request(request: object) -> object:
    """Return the request-body task-input extension when present."""
    extensions = getattr(request, "extensions", None)
    if extensions is None:
        return None
    if isinstance(extensions, Mapping):
        task_input = extensions.get("task_input")
    elif hasattr(extensions, "task_input"):
        task_input = getattr(extensions, "task_input")
    else:
        raise _ServerHTTPError.validation()
    dump = getattr(task_input, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    return task_input


def interaction_response_headers(
    run: ServerInteractionRun | None,
) -> dict[str, str]:
    """Return frozen extension negotiation response headers."""
    return (
        {TASK_INPUT_EXTENSION_HEADER: TASK_INPUT_EXTENSION}
        if run is not None
        else {}
    )


def input_request_href(request_id: str) -> str:
    """Return one path containing only an encoded opaque request ID."""
    return f"/v1/input/requests/{quote(request_id, safe='')}"


def extension_sse_message(data: Mapping[str, LooseJsonValue]) -> str:
    """Encode one frozen extension event separately from answer text."""
    event_type = data.get("type")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("extension event requires a type")
    return sse_message(to_json(dict(data)), event=event_type)


def detached_segment(
    *,
    iterator: AsyncIterator[StreamConsumerProjection],
    response: object,
    orchestrator: Orchestrator,
    protocol: ServerInteractionSurface,
    response_id: str,
    created: int,
    model_id: str,
    output_redaction_settings: ServerOutputRedactionSettings,
    responses_projection: ServerResponsesProjection | None = None,
    choice_count: int = 1,
) -> ServerDetachedSegment:
    """Build one retained segment from a live protocol response."""
    return ServerDetachedSegment(
        iterator=iterator,
        response=response,
        orchestrator=orchestrator,
        protocol=protocol,
        response_id=response_id,
        created=created,
        model_id=model_id,
        output_redaction_settings=output_redaction_settings,
        responses_projection=responses_projection,
        choice_count=choice_count,
    )


router = APIRouter(prefix="/v1/input/requests", tags=["input"])


@router.get("/{request_id}")
async def inspect_input_request(
    request_id: str,
    request: Request,
) -> Response:
    """Return one authenticated semantic request inspection."""
    try:
        service = server_interaction_service(request)
        actor = await service.authenticate(request)
        entry, record = await service.inspect(actor, request_id)
        return _inspection_response(entry, record)
    except _ServerHTTPError as error:
        return _error_response(error)


@router.get("/{request_id}/poll")
async def poll_input_request(
    request_id: str,
    request: Request,
    after_store_revision: int | None = None,
    transport: str | None = None,
) -> Response:
    """Poll state or resume one retained detached transport segment."""
    try:
        service = server_interaction_service(request)
        actor = await service.authenticate(request)
        entry, record = await service.poll(
            actor,
            request_id,
            _validate_after_store_revision(after_store_revision),
        )
        segment = entry.segment
        if (
            segment is not None
            and record.request.state
            in {
                RequestState.ANSWERED,
                RequestState.DECLINED,
                RequestState.TIMED_OUT,
            }
            and (
                transport == "stream"
                or "text/event-stream"
                in request.headers.get("accept", "").lower()
            )
        ):
            segment.claim_resume()
            return StreamingResponse(
                _resume_segment(entry, record, segment),
                media_type="text/event-stream",
                headers={**sse_headers(), **_CACHE_CONTROL_HEADERS},
            )
        if (
            segment is not None
            and record.request.state
            in {
                RequestState.ANSWERED,
                RequestState.DECLINED,
                RequestState.TIMED_OUT,
            }
            and transport == "json"
        ):
            if segment.completed_json is not None:
                return JSONResponse(
                    segment.completed_json,
                    headers=_CACHE_CONTROL_HEADERS,
                )
            segment.claim_resume()
            return await _resume_segment_json(entry, record, segment)
        return _inspection_response(entry, record)
    except _ServerHTTPError as error:
        return _error_response(error)


@router.post("/{request_id}/resolve")
async def resolve_input_request(
    request_id: str,
    request: Request,
) -> Response:
    """Resolve one authenticated request atomically."""
    try:
        service = server_interaction_service(request)
        actor = await service.authenticate(request)
        payload = await _request_json(request)
        _entry, record, idempotent = await service.resolve(
            actor,
            request_id,
            payload,
        )
        return JSONResponse(
            {
                "kind": "resolution_accepted",
                "interaction_state": record.request.state.value,
                "idempotent": idempotent,
                "channel": "json",
            },
            headers={
                **_CACHE_CONTROL_HEADERS,
                "Avalan-Input-State-Revision": str(
                    int(record.request.state_revision)
                ),
                "Avalan-Input-Store-Revision": str(int(record.store_revision)),
            },
        )
    except _ServerHTTPError as error:
        return _error_response(error)


@router.post("/{request_id}/cancel")
async def cancel_input_request(
    request_id: str,
    request: Request,
) -> Response:
    """Cancel one authenticated request explicitly."""
    try:
        service = server_interaction_service(request)
        actor = await service.authenticate(request)
        payload = await _request_json(request)
        _entry, record, _idempotent = await service.cancel(
            actor,
            request_id,
            payload,
        )
        return JSONResponse(
            {
                "interaction_state": "cancelled",
                "accepted": True,
                "channel": "json",
            },
            headers={
                **_CACHE_CONTROL_HEADERS,
                "Avalan-Input-State-Revision": str(
                    int(record.request.state_revision)
                ),
                "Avalan-Input-Store-Revision": str(int(record.store_revision)),
            },
        )
    except _ServerHTTPError as error:
        return _error_response(error)


def _inspection_response(
    entry: _InteractionEntry,
    record: InteractionRecord,
) -> JSONResponse:
    body = _inspection_body(record)
    return JSONResponse(
        body,
        headers={
            **_CACHE_CONTROL_HEADERS,
            "Avalan-Input-Store-Revision": str(int(record.store_revision)),
            "Avalan-Input-Lifecycle-Count": str(len(entry.lifecycle)),
        },
    )


def _inspection_body(record: InteractionRecord) -> dict[str, Any]:
    request = record.request
    origin = request.origin
    questions = [
        dict(encode_input_question(question)) for question in request.questions
    ]
    return {
        "request_id": str(request.request_id),
        "continuation_id": str(request.continuation_id),
        "state_revision": int(request.state_revision),
        "run_id": str(origin.run_id),
        "turn_id": str(origin.turn_id),
        "agent_id": str(origin.agent_id),
        "branch_id": str(origin.branch_id),
        "task_id": str(origin.task_id) if origin.task_id is not None else None,
        "model_call_id": str(origin.model_call_id),
        "required": request.required,
        "reason": request.reason,
        "created_at": request.created_at.isoformat(),
        "state": request.state.value,
        "questions": questions,
    }


async def _resume_segment(
    entry: _InteractionEntry,
    record: InteractionRecord,
    segment: ServerDetachedSegment,
) -> AsyncIterator[str]:
    previous_stream_session_id = str(record.request.origin.stream_session_id)
    stream_session_id = str(uuid4())
    terminal: StreamConsumerProjection | None = None
    completed = False
    source_ended = False
    try:
        entry.append(
            "execution.resumed",
            state=record.request.state.value,
            surface=segment.protocol.value,
        )
        yield extension_sse_message(
            {
                "type": "execution.resumed",
                "request_id": str(record.request.request_id),
                "run_id": str(record.request.origin.run_id),
                "turn_id": str(record.request.origin.turn_id),
                "previous_stream_session_id": previous_stream_session_id,
                "stream_session_id": stream_session_id,
            }
        )
        redactor = ModelVisibleServerProtocolTextRedactor(
            segment.output_redaction_settings,
            protocol="openai",
            channel="answer",
        )
        while True:
            try:
                projection = await segment.next_projection()
            except StopAsyncIteration:
                source_ended = True
                break
            if projection.is_stream_terminal:
                terminal = projection
            for event in await entry.run.extension_events(projection):
                yield extension_sse_message(event)
            if segment.responses_projection is not None:
                for message in segment.responses_projection.stream_messages(
                    projection
                ):
                    yield message
            elif projection.kind is StreamItemKind.ANSWER_DELTA:
                for text in redactor.push(projection.text_delta or ""):
                    yield _resume_text_message(segment, text, projection)
        if terminal is None:
            raise RuntimeError("resumed stream has no terminal outcome")
        if segment.responses_projection is not None:
            for message in segment.responses_projection.finish_stream(
                terminal
            ):
                yield message
        else:
            for text in redactor.flush():
                yield _resume_text_message(segment, text, terminal)
            yield _resume_terminal_message(segment, terminal)
        completed = stream_terminal_succeeded(terminal)
        if completed:
            await segment.orchestrator.sync_messages(
                cast(MemorySynchronizableResponse, segment.response)
            )
    except CancelledError:
        entry.append(
            "transport.disconnected",
            state=entry.request.state.value,
            surface=segment.protocol.value,
        )
        raise
    finally:
        try:
            if completed:
                await segment.aclose(cancelled=False)
        finally:
            try:
                segment.release_resume(
                    exhausted=source_ended or terminal is not None,
                )
            finally:
                if terminal is not None:
                    await entry.run._service._retain_terminal_entry(entry)


async def _resume_segment_json(
    entry: _InteractionEntry,
    record: InteractionRecord,
    segment: ServerDetachedSegment,
) -> JSONResponse:
    text_parts: list[str] = []
    terminal: StreamConsumerProjection | None = None
    source_ended = False
    try:
        entry.append(
            "execution.resumed",
            state=record.request.state.value,
            surface=segment.protocol.value,
        )
        redactor = ModelVisibleServerProtocolTextRedactor(
            segment.output_redaction_settings,
            protocol="openai",
            channel="answer",
        )
        while True:
            try:
                projection = await segment.next_projection()
            except StopAsyncIteration:
                source_ended = True
                break
            if projection.kind is StreamItemKind.ANSWER_DELTA:
                text_parts.extend(redactor.push(projection.text_delta or ""))
            if segment.responses_projection is not None:
                segment.responses_projection.observe_json(projection)
            if projection.is_stream_terminal:
                terminal = projection
        text_parts.extend(redactor.flush())
        if terminal is None or not stream_terminal_succeeded(terminal):
            raise _ServerHTTPError.unavailable()
        text = "".join(text_parts)
        body: dict[str, Any]
        if segment.responses_projection is not None:
            body = segment.responses_projection.json_body(
                terminal,
                segment.response,
            )
        elif segment.protocol is ServerInteractionSurface.CHAT:
            body = {
                "id": segment.response_id,
                "object": "chat.completion",
                "created": segment.created,
                "model": segment.model_id,
                "choices": [
                    {
                        "index": index,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": "stop",
                    }
                    for index in range(segment.choice_count)
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            body = {
                "id": segment.response_id,
                "created": segment.created,
                "model": segment.model_id,
                "type": "response",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                ],
            }
        await segment.orchestrator.sync_messages(
            cast(MemorySynchronizableResponse, segment.response)
        )
        await segment.aclose(cancelled=False)
        async with segment.lock:
            segment.completed_json = body
        return JSONResponse(body, headers=_CACHE_CONTROL_HEADERS)
    finally:
        try:
            segment.release_resume(
                exhausted=source_ended or terminal is not None,
            )
        finally:
            if terminal is not None:
                await entry.run._service._retain_terminal_entry(entry)


def _resume_text_message(
    segment: ServerDetachedSegment,
    text: str,
    projection: StreamConsumerProjection | None,
) -> str:
    if segment.protocol is ServerInteractionSurface.RESPONSES:
        data: dict[str, Any] = {
            "type": "response.output_text.delta",
            "delta": text,
        }
        if projection is not None:
            data["sequence_number"] = projection.sequence
        return sse_message(
            to_json(data),
            event="response.output_text.delta",
        )
    return sse_message(
        to_json(
            {
                "id": segment.response_id,
                "object": "chat.completion.chunk",
                "created": segment.created,
                "model": segment.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
        )
    )


def _resume_terminal_message(
    segment: ServerDetachedSegment,
    terminal: StreamConsumerProjection,
) -> str:
    if segment.protocol is ServerInteractionSurface.RESPONSES:
        outcome = terminal.terminal_outcome
        event = (
            "response.completed"
            if outcome is StreamTerminalOutcome.COMPLETED
            else (
                "response.cancelled"
                if outcome is StreamTerminalOutcome.CANCELLED
                else "response.failed"
            )
        )
        return sse_message(to_json({"type": event}), event=event)
    if terminal.terminal_outcome is not StreamTerminalOutcome.COMPLETED:
        event = (
            "chat.completion.cancelled"
            if terminal.terminal_outcome is StreamTerminalOutcome.CANCELLED
            else "chat.completion.failed"
        )
        return sse_message(
            to_json(
                {
                    "id": segment.response_id,
                    "object": "chat.completion.chunk",
                    "created": segment.created,
                    "model": segment.model_id,
                    "type": event,
                    "choices": [],
                }
            ),
            event=event,
        )
    return sse_message(
        to_json(
            {
                "id": segment.response_id,
                "object": "chat.completion.chunk",
                "created": segment.created,
                "model": segment.model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
    ) + sse_message("[DONE]")


def _resolution_command(
    actor: InteractionActor,
    record: InteractionRecord,
    payload: object,
) -> ResolveInteractionCommand:
    values = _validated_mutation_binding(payload, record)
    status = values.get("status")
    now = datetime.now(timezone.utc)
    if status == "answered":
        raw_answers = values.get("answers")
        if not isinstance(raw_answers, list):
            raise InputValidationError(
                InputErrorCode.INVALID_TYPE,
                "answers",
                "answers must be a list",
            )
        answers: tuple[InputAnswer, ...] = tuple(
            decode_input_answer(answer) for answer in raw_answers
        )
        if any(
            answer.provenance is not AnswerProvenance.HUMAN
            for answer in answers
        ):
            raise InputValidationError(
                InputErrorCode.FORBIDDEN,
                "answers.provenance",
                "HTTP answers require human provenance",
            )
        resolution: AnsweredResolution | DeclinedResolution = (
            AnsweredResolution(
                request_id=record.request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=now,
                answers=answers,
            )
        )
    elif status == "declined":
        if "answers" in values:
            raise InputValidationError(
                InputErrorCode.INVALID_FORMAT,
                "answers",
                "declines cannot carry answers",
            )
        resolution = DeclinedResolution(
            request_id=record.request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=now,
        )
    else:
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "status",
            "status must be answered or declined",
        )
    return ResolveInteractionCommand(
        actor=actor,
        correlation=record.correlation,
        expected_state_revision=StateRevision(
            cast(int, values["expected_state_revision"])
        ),
        idempotency_key=ResolutionIdempotencyKey(
            _required_string(values, "idempotency_key", maximum=256)
        ),
        proposed_resolution=resolution,
    )


def _validated_mutation_binding(
    payload: object,
    record: InteractionRecord,
) -> dict[str, object]:
    values = _mapping(payload)
    required = {
        "continuation_id",
        "run_id",
        "turn_id",
        "task_id",
        "agent_id",
        "branch_id",
        "model_call_id",
        "expected_state_revision",
        "idempotency_key",
    }
    if not required.issubset(values):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "binding",
            "complete interaction binding is required",
        )
    allowed = required | {"status", "answers"}
    if not set(values).issubset(allowed):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            "binding",
            "unknown mutation fields are forbidden",
        )
    correlation = record.correlation
    expected: dict[str, object] = {
        "continuation_id": str(correlation.continuation_id),
        "run_id": str(correlation.run_id),
        "turn_id": str(correlation.turn_id),
        "task_id": (
            str(correlation.task_id)
            if correlation.task_id is not None
            else None
        ),
        "agent_id": str(correlation.agent_id),
        "branch_id": str(correlation.branch_id),
        "model_call_id": str(correlation.model_call_id),
    }
    if any(values.get(name) != value for name, value in expected.items()):
        raise InputValidationError(
            InputErrorCode.CORRELATION_MISMATCH,
            "binding",
            "interaction binding does not match",
        )
    revision = values.get("expected_state_revision")
    if (
        type(revision) is not int
        or revision < 0
        or revision > _MAX_JSON_SAFE_INTEGER
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "expected_state_revision",
            "state revision must be a safe non-negative integer",
        )
    _required_string(values, "idempotency_key", maximum=256)
    return values


def _validate_server_authorization(
    decision: object,
    actor: InteractionActor,
    operation: InteractionOperation,
    target: InteractionRequestAuthorizationTarget,
) -> None:
    if not isinstance(decision, InteractionAuthorizationDecision):
        raise _ServerHTTPError.forbidden()
    if (
        decision.actor != actor
        or decision.operation is not operation
        or decision.target != target
    ):
        raise _ServerHTTPError.forbidden()
    if (
        decision.allowed
        and operation
        in {InteractionOperation.INSPECT, InteractionOperation.WAIT}
        and decision.disclosure is not InteractionDisclosure.FULL
    ):
        raise _ServerHTTPError.forbidden()


def _store_result(result: object) -> object:
    if not isinstance(result, InteractionBrokerResult):
        raise _ServerHTTPError.unavailable()
    return result.store_result


def _http_error_for_code(code: InputErrorCode) -> _ServerHTTPError:
    if code is InputErrorCode.STALE_REVISION:
        return _ServerHTTPError.stale_revision()
    if code is InputErrorCode.EXPIRED:
        return _ServerHTTPError.expired()
    if code in {
        InputErrorCode.ALREADY_RESOLVED,
        InputErrorCode.IDEMPOTENCY_CONFLICT,
        InputErrorCode.IDEMPOTENCY_LEDGER_FULL,
        InputErrorCode.SUPERSEDED,
    }:
        return _ServerHTTPError.conflict()
    if code is InputErrorCode.NOT_FOUND:
        return _ServerHTTPError.not_found()
    if code is InputErrorCode.FORBIDDEN:
        return _ServerHTTPError.forbidden()
    if code is InputErrorCode.UNAVAILABLE:
        return _ServerHTTPError.unavailable()
    return _ServerHTTPError.validation()


def _error_response(error: _ServerHTTPError) -> JSONResponse:
    headers = {**_CACHE_CONTROL_HEADERS, **dict(error.headers)}
    return JSONResponse(
        {"code": error.code, "message": error.message},
        status_code=error.status_code,
        headers=headers,
    )


async def _request_json(request: Request) -> object:
    try:
        return await request.json()
    except Exception:
        raise _ServerHTTPError.validation() from None


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "payload",
            "payload must be an object",
        )
    if any(not isinstance(key, str) for key in value):
        raise InputValidationError(
            InputErrorCode.INVALID_TYPE,
            "payload",
            "object keys must be strings",
        )
    return {cast(str, key): item for key, item in value.items()}


def _required_string(
    values: Mapping[str, object],
    name: str,
    *,
    maximum: int,
) -> str:
    value = values.get(name)
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or len(value.encode("utf-8")) > maximum * 4
    ):
        raise InputValidationError(
            InputErrorCode.INVALID_FORMAT,
            name,
            "value must be a bounded non-empty string",
        )
    return value


def _validate_after_store_revision(value: int | None) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0 or value > _MAX_JSON_SAFE_INTEGER:
        raise _ServerHTTPError.validation()
    return value


def _extension_header_present(request: Request) -> bool:
    values = request.headers.get(TASK_INPUT_EXTENSION_HEADER, "")
    return TASK_INPUT_EXTENSION in {
        value.strip() for value in values.split(",") if value.strip()
    }


def _require_async_callable(value: object, path: str) -> None:
    if not callable(value):
        raise TypeError(f"{path} must be an async callable")
    callback = cast(Callable[..., object], value)
    call = getattr(value, "__call__", None)
    if not iscoroutinefunction(callback) and not (
        callable(call)
        and iscoroutinefunction(cast(Callable[..., object], call))
    ):
        raise TypeError(f"{path} must be an async callable")


def _require_async_method(value: object, name: str, path: str) -> None:
    method = getattr(value, name, None)
    if not callable(method) or not iscoroutinefunction(method):
        raise TypeError(f"{path}.{name} must be async")


async def _read_projection(
    iterator: AsyncIterator[StreamConsumerProjection],
) -> StreamConsumerProjection:
    """Read one projection through a concrete coroutine for task ownership."""
    return await anext(iterator)
