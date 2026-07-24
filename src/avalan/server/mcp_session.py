"""Bridge canonical task input to negotiated inbound MCP sessions."""

from ..interaction.entities import (
    AnsweredResolution,
    AnswerProvenance,
    ChoiceValue,
    ConfirmationAnswer,
    ConfirmationQuestion,
    DeclinedResolution,
    FreeFormOther,
    InputAnswer,
    InputQuestion,
    InputRequest,
    MultilineTextAnswer,
    MultilineTextQuestion,
    MultipleSelectionAnswer,
    MultipleSelectionQuestion,
    PrincipalScope,
    QuestionId,
    SelectedChoice,
    SingleSelectionAnswer,
    SingleSelectionQuestion,
    TextAnswer,
    TextQuestion,
    _validate_resolution_against_request,
)
from ..interaction.error import InputValidationError
from ..interaction.handler import (
    InputDisconnectReason,
    InputHandler,
    InputHandlerContext,
    InputHandlerDisconnected,
    InputHandlerOutcome,
    InputHandlerResolution,
)
from ..interaction.validation import validate_opaque_id
from ..types import JsonObject, MutableJsonValue

from asyncio import (
    CancelledError,
    Condition,
    Future,
    Lock,
    get_running_loop,
    shield,
    timeout,
)
from asyncio import (
    TimeoutError as AsyncTimeoutError,
)
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import blake2b
from hmac import compare_digest
from json import dumps
from math import isfinite
from re import compile as compile_pattern
from secrets import token_bytes
from typing import Literal, Protocol, TypeAlias, cast, final

MCP_PROTOCOL_VERSION: Literal["2025-11-25"] = "2025-11-25"
MCP_ELICITATION_CREATE_METHOD = "elicitation/create"
MCP_FORM_OTHER_PROPERTY_PREFIX = "__avalan_other__"
MCP_FORM_RESPONSE_MAX_BYTES = 1_048_576
MCP_RELATED_TASK_METADATA_KEY = "io.modelcontextprotocol/related-task"
MCP_REQUIRED_OTHER_METADATA_KEY = "ai.avalan/required-other"
MCP_INVALID_PARAMS = -32602
MCP_UNAVAILABLE = -32001
MCP_CONFLICT = -32009

MCPRequestId: TypeAlias = str | int
_JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"
_SENSITIVE_FORM_PATTERN = compile_pattern(
    r"(?:\bpasswords?\b|\bpasscodes?\b|\bsecrets?\b|\bcredentials?\b|"
    r"\bapi[\s._-]*(?:keys?|tokens?)\b|"
    r"\b(?:access|refresh|bearer|session|identity)[\s._-]*tokens?\b|"
    r"\bauth(?:entication|orization|orisation)?[\s._-]*tokens?\b|"
    r"\bprivate[\s._-]*keys?\b|\bpayments?\b|"
    r"\b(?:credit|payment)[\s._-]*cards?\b|"
    r"\bcard[\s._-]*(?:numbers?|security[\s._-]*codes?|"
    r"details?|credentials?|materials?)\b|"
    r"\bpayment[\s._-]*(?:details?|credentials?|materials?)\b|"
    r"\b(?:cvvs?|cvcs?|ibans?)\b|"
    r"\bbank[\s._-]*(?:accounts?|credentials?)\b|"
    r"\baccount[\s._-]*(?:numbers?|credentials?|details?|materials?)\b|"
    r"\brouting[\s._-]*(?:numbers?|codes?|details?|materials?)\b|"
    r"\b(?:mfa|2fa|otp|totp|hotp)s?\b|"
    r"\bpin[\s._-]*(?:codes?|numbers?|values?)\b|"
    r"\bone[\s._-]*time[\s._-]*(?:codes?|passwords?)\b|"
    r"\b(?:recovery|security|verification)[\s._-]*codes?\b|"
    r"\bauth(?:entication|orization|orisation)?\b|"
    r"\bauthentication[\s._-]*challenges?\b|\boauth\b)"
)
_SENSITIVE_ACRONYM_PATTERN = compile_pattern(r"\bPIN(?:s|S)?\b")
_PIN_WORD_PATTERN = compile_pattern(r"\bpins?\b")
_PIN_REQUEST_PATTERN = compile_pattern(
    r"\b(?:change|choose|confirm|create|disclose|enter|give|input|pick|"
    r"provide|re[\s._-]*enter|re[\s._-]*type|repeat|reset|reveal|select|"
    r"send|set|share|submit|supply|tell|type|update|validate|verify)\b"
    r"[^.!?\n]{0,120}\bpins?\b"
)
_PIN_POSSESSIVE_QUESTION_PATTERN = compile_pattern(
    r"\b(?:what|which)\s+(?:are|is|was|were)\b[^.!?\n]{0,120}"
    r"\b(?:my|our|their|your)\b[^.!?\n]{0,120}\bpins?\b"
)
_PIN_POSSESSIVE_PLEASE_PATTERN = compile_pattern(
    r"\b(?:my|our|their|your)\b[^.!?\n]{0,120}\bpins?\b\s*,?\s+please\b"
)
_PIN_AUTH_QUALIFIER = (
    r"(?:access|account|auth(?:entication|orization|orisation)?|"
    r"credentials?|identity|log[\s._-]*in|security|sign[\s._-]*in|"
    r"unlock|verification)"
)
_PIN_AUTH_CONTEXT_PATTERN = compile_pattern(
    rf"(?:\b{_PIN_AUTH_QUALIFIER}\b[^.!?\n]{{0,80}}\bpins?\b|"
    rf"\bpins?\b[^.!?\n]{{0,80}}\b{_PIN_AUTH_QUALIFIER}\b)"
)
_SECRET_VALUE_PATTERN = compile_pattern(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-(?:proj-)?[A-Za-z0-9_-]{16,}|"
    r"(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{12,}|"
    r"AIza[A-Za-z0-9_-]{20,}|"
    r"(?:AKIA|ASIA)[A-Z0-9]{16}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{10,}|"
    r"api[_-]?key[:=._-][A-Za-z0-9_./+=-]{12,}"
    r")(?![A-Za-z0-9])"
)
_CARD_NUMBER_CANDIDATE_PATTERN = compile_pattern(
    r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"
)


class MCPFormErrorCode(StrEnum):
    """Identify a content-safe MCP form adapter failure."""

    INVALID_CAPABILITIES = "mcp.form.invalid_capabilities"
    SESSION_CONFLICT = "mcp.form.session_conflict"
    SESSION_NOT_FOUND = "mcp.form.session_not_found"
    NOT_INITIALIZED = "mcp.form.not_initialized"
    CAPABILITY_UNAVAILABLE = "mcp.form.capability_unavailable"
    UNSAFE_REQUEST = "mcp.form.unsafe_request"
    MULTILINE_UNAVAILABLE = "mcp.form.multiline_unavailable"
    CAPACITY = "mcp.form.capacity"
    INVALID_RESPONSE = "mcp.form.invalid_response"
    OVERSIZED_RESPONSE = "mcp.form.oversized_response"
    AMBIGUOUS_RESPONSE = "mcp.form.ambiguous_response"
    STALE_RESPONSE = "mcp.form.stale_response"
    RESPONSE_NOT_PENDING = "mcp.form.response_not_pending"
    PEER_ERROR = "mcp.form.peer_error"
    STATUS_HOOK_FAILED = "mcp.form.status_hook_failed"
    WAIT_TIMED_OUT = "mcp.form.wait_timed_out"


class MCPFormSessionError(ValueError):
    """Expose only protocol and content-safe diagnostic codes."""

    def __init__(
        self,
        code: MCPFormErrorCode,
        rpc_code: int,
        safe_message: str,
    ) -> None:
        super().__init__(safe_message)
        self.code = code
        self.rpc_code = rpc_code
        self.safe_message = safe_message


@final
@dataclass(frozen=True, slots=True)
class MCPElicitationCapabilities:
    """Store normalized client elicitation modes."""

    form: bool
    url: bool
    legacy_form_only: bool = False


@final
@dataclass(frozen=True, slots=True)
class MCPFormSessionView:
    """Expose negotiated session state without its owner."""

    session_id: str
    protocol_version: Literal["2025-11-25"]
    elicitation: MCPElicitationCapabilities
    form_available: bool


class MCPFormStatus(StrEnum):
    """Identify content-safe reverse-elicitation lifecycle state."""

    INPUT_REQUIRED = "input_required"
    ANSWERED = "answered"
    DECLINED = "declined"
    CANCELLED = "cancelled"
    UNAVAILABLE = "unavailable"


@final
@dataclass(frozen=True, slots=True)
class MCPFormStatusEvent:
    """Carry content-free state to optional task integration."""

    session_id: str
    request_id: str
    status: MCPFormStatus
    related_task_id: str | None = None
    safe_code: MCPFormErrorCode | None = None


class MCPFormStatusHook(Protocol):
    """Observe content-free reverse-elicitation state."""

    async def __call__(self, event: MCPFormStatusEvent) -> None:
        """Handle one lifecycle event."""
        ...


@final
@dataclass(frozen=True, slots=True)
class MCPFormElicitationOutbound:
    """Carry one reverse request without transport encoding."""

    session_id: str
    jsonrpc_id: MCPRequestId
    related_request_id: MCPRequestId
    canonical_request_id: str
    params: JsonObject = field(repr=False)
    related_task_id: str | None = None
    method: Literal["elicitation/create"] = "elicitation/create"


@dataclass(slots=True)
class _Pending:
    outbound: MCPFormElicitationOutbound
    request: InputRequest = field(repr=False)
    future: Future[InputHandlerOutcome] = field(repr=False)
    published: bool = False


@dataclass(slots=True)
class _Session:
    session_id: str
    owner: PrincipalScope = field(repr=False)
    capabilities: MCPElicitationCapabilities
    can_route: bool
    preserves_newlines: bool
    outbound: deque[MCPFormElicitationOutbound] = field(repr=False)
    condition: Condition = field(repr=False)
    initialized: bool = False
    closed: bool = False
    sequence: int = 0
    pending: dict[MCPRequestId, _Pending] = field(
        default_factory=dict,
        repr=False,
    )
    stale: deque[MCPRequestId] = field(default_factory=deque, repr=False)
    replays: dict[MCPRequestId, bytes] = field(
        default_factory=dict,
        repr=False,
    )
    lock: Lock = field(default_factory=Lock, repr=False)

    @property
    def form_available(self) -> bool:
        return (
            self.initialized
            and not self.closed
            and self.can_route
            and self.capabilities.form
        )


def normalize_mcp_elicitation_capabilities(
    capabilities: object,
) -> MCPElicitationCapabilities:
    """Normalize modern modes and legacy empty elicitation as form-only."""
    if not isinstance(capabilities, Mapping):
        raise _error(
            MCPFormErrorCode.INVALID_CAPABILITIES,
            MCP_INVALID_PARAMS,
            "client capabilities must be an object",
        )
    elicitation = capabilities.get("elicitation")
    if elicitation is None:
        return MCPElicitationCapabilities(False, False)
    if not isinstance(elicitation, Mapping):
        raise _error(
            MCPFormErrorCode.INVALID_CAPABILITIES,
            MCP_INVALID_PARAMS,
            "elicitation capability must be an object",
        )
    if not elicitation:
        return MCPElicitationCapabilities(True, False, True)
    form = _mode(elicitation, "form")
    url = _mode(elicitation, "url")
    if not form and not url:
        raise _error(
            MCPFormErrorCode.INVALID_CAPABILITIES,
            MCP_INVALID_PARAMS,
            "elicitation capability must declare form or url",
        )
    return MCPElicitationCapabilities(form, url)


def mcp_form_other_property_name(question_id: QuestionId | str) -> str:
    """Return the collision-free property for a free-form alternative."""
    return f"{MCP_FORM_OTHER_PROPERTY_PREFIX}{question_id}"


def project_mcp_form_params(
    request: InputRequest,
    *,
    legacy_form_only: bool = False,
    preserves_newlines: bool = True,
) -> JsonObject:
    """Project canonical questions into the restricted MCP form grammar."""
    if type(request) is not InputRequest:
        raise TypeError("request must be an InputRequest")
    _reject_sensitive_request(request)
    if not preserves_newlines and any(
        type(question) is MultilineTextQuestion
        for question in request.questions
    ):
        raise _error(
            MCPFormErrorCode.MULTILINE_UNAVAILABLE,
            MCP_UNAVAILABLE,
            "the MCP client cannot preserve multiline answers",
        )
    properties: JsonObject = {}
    required: list[MutableJsonValue] = []
    required_other: list[MutableJsonValue] = []
    for question in request.questions:
        name = str(question.question_id)
        properties[name] = _question_schema(question)
        has_other = (
            isinstance(
                question,
                (SingleSelectionQuestion, MultipleSelectionQuestion),
            )
            and question.allow_other
        )
        if question.required and not has_other:
            required.append(name)
        elif question.required:
            required_other.append(name)
        if has_other:
            properties[mcp_form_other_property_name(name)] = {
                "type": "string",
                "title": "Other",
                "description": (
                    "Free-form alternative for "
                    f"{question.header or question.prompt}"
                ),
                "minLength": 1,
                "maxLength": 4_096,
            }
    schema: JsonObject = {
        "$schema": _JSON_SCHEMA_DIALECT,
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    params: JsonObject = {
        "message": request.reason,
        "requestedSchema": schema,
    }
    if not legacy_form_only:
        params["mode"] = "form"
    if required_other:
        params["_meta"] = {
            MCP_REQUIRED_OTHER_METADATA_KEY: required_other,
        }
    return params


@final
class MCPFormSessionRegistry:
    """Own bounded, principal-bound MCP form sessions and waiters."""

    def __init__(
        self,
        *,
        maximum_sessions: int = 128,
        maximum_pending_per_session: int = 8,
        response_wait_seconds: float = 60.0,
        stale_response_limit: int = 64,
    ) -> None:
        assert 0 < maximum_sessions <= 4_096
        assert 0 < maximum_pending_per_session <= 1_024
        assert (
            isinstance(response_wait_seconds, int | float)
            and not isinstance(response_wait_seconds, bool)
            and isfinite(response_wait_seconds)
            and 0 < response_wait_seconds <= 3_600
        )
        assert 0 < stale_response_limit <= 4_096
        self._maximum_sessions = maximum_sessions
        self._maximum_pending = maximum_pending_per_session
        self._wait_seconds = float(response_wait_seconds)
        self._stale_limit = stale_response_limit
        self._replay_key = token_bytes(32)
        self._sessions: dict[str, _Session] = {}
        self._lock = Lock()

    @property
    def session_count(self) -> int:
        """Return the retained session count."""
        return len(self._sessions)

    async def initialize(
        self,
        *,
        session_id: str,
        owner: PrincipalScope,
        protocol_version: object,
        capabilities: object,
        can_route_and_resume: bool,
        preserves_newlines: bool = True,
    ) -> MCPFormSessionView:
        """Create one owner-bound session pinned to MCP 2025-11-25."""
        session_id = validate_opaque_id(session_id, "mcp.session_id")
        if not isinstance(owner, PrincipalScope):
            raise TypeError("owner must be a PrincipalScope")
        if not isinstance(protocol_version, str) or not protocol_version:
            raise _invalid("protocolVersion must be a non-empty string")
        if type(can_route_and_resume) is not bool:
            raise TypeError("can_route_and_resume must be a boolean")
        if type(preserves_newlines) is not bool:
            raise TypeError("preserves_newlines must be a boolean")
        normalized = normalize_mcp_elicitation_capabilities(capabilities)
        state_lock = Lock()
        state = _Session(
            session_id=session_id,
            owner=owner,
            capabilities=normalized,
            can_route=can_route_and_resume,
            preserves_newlines=preserves_newlines,
            outbound=deque(),
            condition=Condition(state_lock),
            lock=state_lock,
        )
        async with self._lock:
            if session_id in self._sessions:
                raise _error(
                    MCPFormErrorCode.SESSION_CONFLICT,
                    MCP_CONFLICT,
                    "MCP session already exists",
                )
            if len(self._sessions) >= self._maximum_sessions:
                raise _unavailable_error("MCP session capacity is unavailable")
            self._sessions[session_id] = state
        return _view(state)

    async def mark_initialized(
        self,
        session_id: str,
        owner: PrincipalScope,
    ) -> MCPFormSessionView:
        """Activate reverse requests after notifications/initialized."""
        state = await self._owned(session_id, owner)
        async with state.lock:
            if state.closed:
                raise _session_not_found()
            state.initialized = True
            return _view(state)

    async def negotiation(
        self,
        session_id: str,
        owner: PrincipalScope,
    ) -> MCPFormSessionView:
        """Return content-safe negotiated state."""
        state = await self._owned(session_id, owner)
        async with state.lock:
            return _view(state)

    def handler(
        self,
        *,
        session_id: str,
        owner: PrincipalScope,
        related_request_id: MCPRequestId,
        related_task_id: str | None = None,
        status_hook: MCPFormStatusHook | None = None,
    ) -> InputHandler:
        """Return a handler associated with one originating client request."""
        session_id = validate_opaque_id(session_id, "mcp.session_id")
        if not isinstance(owner, PrincipalScope):
            raise TypeError("owner must be a PrincipalScope")
        _request_id(related_request_id)
        if related_task_id is not None:
            related_task_id = validate_opaque_id(
                related_task_id,
                "mcp.related_task_id",
            )

        async def handle(context: InputHandlerContext) -> InputHandlerOutcome:
            return await self._handle(
                session_id,
                owner,
                context,
                related_request_id,
                related_task_id,
                status_hook,
            )

        return handle

    async def next_outbound(
        self,
        session_id: str,
        owner: PrincipalScope,
        *,
        timeout_seconds: float = 30.0,
        related_request_id: MCPRequestId | None = None,
        related_task_id: str | None = None,
    ) -> MCPFormElicitationOutbound | None:
        """Return the next matching reverse request within a bounded wait."""
        assert (
            isinstance(timeout_seconds, int | float)
            and not isinstance(timeout_seconds, bool)
            and isfinite(timeout_seconds)
            and 0 < timeout_seconds <= 3_600
        )
        if related_request_id is not None:
            _request_id(related_request_id)
        if related_task_id is not None:
            related_task_id = validate_opaque_id(
                related_task_id,
                "mcp.related_task_id",
            )
        state = await self._owned(session_id, owner)
        try:
            async with timeout(timeout_seconds):
                async with state.condition:
                    while True:
                        if state.closed:
                            raise _session_not_found()
                        if not state.form_available:
                            return None
                        for index, item in enumerate(state.outbound):
                            if (
                                related_request_id is not None
                                and item.related_request_id
                                != related_request_id
                            ) or (
                                related_task_id is not None
                                and item.related_task_id != related_task_id
                            ):
                                continue
                            del state.outbound[index]
                            return item
                        else:
                            await state.condition.wait()
        except AsyncTimeoutError:
            return None

    async def dispatch_response(
        self,
        session_id: str,
        owner: PrincipalScope,
        response: object,
    ) -> None:
        """Deliver one exact JSON-RPC response to its bound waiter."""
        state = await self._owned(session_id, owner)
        try:
            response_id, result, peer_error = _response(response)
        except MCPFormSessionError as exc:
            await self._guard_malformed_response(state, response, exc)
            raise
        replay_fingerprint = _response_fingerprint(
            self._replay_key,
            state.session_id,
            response,
        )
        async with state.lock:
            pending = state.pending.get(response_id)
            if pending is None or not pending.published:
                retained = state.replays.get(response_id)
                if (
                    pending is None
                    and retained is not None
                    and compare_digest(
                        retained,
                        replay_fingerprint,
                    )
                ):
                    return
                code = (
                    MCPFormErrorCode.STALE_RESPONSE
                    if pending is None and response_id in state.stale
                    else MCPFormErrorCode.RESPONSE_NOT_PENDING
                )
                rpc_code = (
                    MCP_CONFLICT
                    if code is MCPFormErrorCode.STALE_RESPONSE
                    else MCP_INVALID_PARAMS
                )
                raise _error(code, rpc_code, "MCP response is not pending")
            if peer_error is not None:
                self._finish(state, response_id, _unavailable())
                if peer_error.get("code") == MCP_INVALID_PARAMS:
                    state.can_route = False
                    state.outbound.clear()
                    self._settle_all(
                        state,
                        InputDisconnectReason.HANDLER_UNAVAILABLE,
                    )
                    state.condition.notify_all()
                raise _error(
                    MCPFormErrorCode.PEER_ERROR,
                    MCP_UNAVAILABLE,
                    "MCP client rejected the form request",
                )
            assert result is not None
            _validate_related_task(
                result,
                pending.outbound.related_task_id,
            )
            outcome = _result_outcome(pending.request, result)
            self._finish(
                state,
                response_id,
                outcome,
                replay_fingerprint=replay_fingerprint,
            )

    async def withdraw_form(
        self,
        session_id: str,
        owner: PrincipalScope,
    ) -> None:
        """Withdraw form capability and settle all waiters as unavailable."""
        state = await self._owned(session_id, owner)
        async with state.condition:
            state.can_route = False
            state.outbound.clear()
            self._settle_all(state, InputDisconnectReason.HANDLER_UNAVAILABLE)
            state.condition.notify_all()

    async def close(
        self,
        session_id: str,
        owner: PrincipalScope,
    ) -> None:
        """Remove a session and settle its waiters as disconnected."""
        state = await self._owned(session_id, owner)
        async with self._lock:
            if self._sessions.get(state.session_id) is not state:
                raise _session_not_found()
            self._sessions.pop(state.session_id)
        await self._close_removed_states((state,))

    async def close_all(self) -> None:
        """Remove every session and settle its waiters as disconnected."""
        async with self._lock:
            states = tuple(self._sessions.values())
            self._sessions.clear()
        await self._close_removed_states(states)

    async def pending_count(
        self,
        session_id: str,
        owner: PrincipalScope,
    ) -> int:
        """Return the owner-bound live waiter count."""
        state = await self._owned(session_id, owner)
        async with state.condition:
            return len(state.pending)

    async def _owned(
        self,
        session_id: str,
        owner: PrincipalScope,
    ) -> _Session:
        session_id = validate_opaque_id(session_id, "mcp.session_id")
        if not isinstance(owner, PrincipalScope):
            raise TypeError("owner must be a PrincipalScope")
        async with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.owner != owner:
                raise _session_not_found()
            return state

    async def _handle(
        self,
        session_id: str,
        owner: PrincipalScope,
        context: InputHandlerContext,
        related_request_id: MCPRequestId,
        related_task_id: str | None,
        hook: MCPFormStatusHook | None,
    ) -> InputHandlerOutcome:
        try:
            state = await self._owned(session_id, owner)
            pending = await self._enqueue(
                state,
                context.request,
                related_request_id,
                related_task_id,
            )
        except MCPFormSessionError as exc:
            await _notify(
                hook,
                session_id,
                context.request,
                MCPFormStatus.UNAVAILABLE,
                related_task_id,
                exc.code,
            )
            return _unavailable()
        try:
            return await self._complete_pending(
                state,
                pending,
                hook,
                related_task_id,
            )
        except CancelledError:
            cleanup = get_running_loop().create_task(
                self._discard(state, pending.outbound.jsonrpc_id)
            )
            await _await_shielded_cleanup(cleanup)
            raise

    async def _complete_pending(
        self,
        state: _Session,
        pending: _Pending,
        hook: MCPFormStatusHook | None,
        related_task_id: str | None,
    ) -> InputHandlerOutcome:
        request = pending.request
        if not await _notify(
            hook,
            state.session_id,
            request,
            MCPFormStatus.INPUT_REQUIRED,
            related_task_id,
        ):
            await self._discard(state, pending.outbound.jsonrpc_id)
            return _unavailable()
        try:
            if await self._publish(state, pending):
                async with timeout(self._wait_seconds):
                    outcome = await pending.future
            else:
                outcome = await pending.future
        except AsyncTimeoutError:
            await self._discard(state, pending.outbound.jsonrpc_id)
            await _notify(
                hook,
                state.session_id,
                request,
                MCPFormStatus.UNAVAILABLE,
                related_task_id,
                MCPFormErrorCode.WAIT_TIMED_OUT,
            )
            return _unavailable()
        if not await _notify(
            hook,
            state.session_id,
            request,
            _outcome_status(outcome),
            related_task_id,
        ):
            return _unavailable()
        return outcome

    async def _enqueue(
        self,
        state: _Session,
        request: InputRequest,
        related_request_id: MCPRequestId,
        related_task_id: str | None,
    ) -> _Pending:
        async with state.lock:
            if request.origin.principal != state.owner:
                raise _session_not_found()
            if not state.initialized:
                raise _error(
                    MCPFormErrorCode.NOT_INITIALIZED,
                    MCP_UNAVAILABLE,
                    "MCP session is not initialized",
                )
            if not state.form_available:
                raise _unavailable_error("MCP form capability is unavailable")
            if len(state.pending) >= self._maximum_pending:
                raise _error(
                    MCPFormErrorCode.CAPACITY,
                    MCP_UNAVAILABLE,
                    "MCP form capacity is unavailable",
                )
            params = project_mcp_form_params(
                request,
                legacy_form_only=state.capabilities.legacy_form_only,
                preserves_newlines=state.preserves_newlines,
            )
            if related_task_id is not None:
                metadata = params.setdefault("_meta", {})
                assert isinstance(metadata, dict)
                metadata[MCP_RELATED_TASK_METADATA_KEY] = {
                    "taskId": related_task_id,
                }
            state.sequence += 1
            item = MCPFormElicitationOutbound(
                state.session_id,
                f"avalan-elicit-{state.sequence}",
                related_request_id,
                str(request.request_id),
                params,
                related_task_id,
            )
            pending = _Pending(
                item,
                request,
                get_running_loop().create_future(),
            )
            state.pending[item.jsonrpc_id] = pending
            return pending

    async def _publish(self, state: _Session, pending: _Pending) -> bool:
        async with state.condition:
            response_id = pending.outbound.jsonrpc_id
            if state.pending.get(response_id) is not pending:
                return False
            pending.published = True
            state.outbound.append(pending.outbound)
            state.condition.notify_all()
            return True

    async def _discard(
        self,
        state: _Session,
        response_id: MCPRequestId,
    ) -> None:
        async with state.condition:
            pending = state.pending.pop(response_id, None)
            if pending is not None:
                self._remove_outbound(state, pending)
                self._remember(state, response_id)
                pending.future.cancel()
                state.condition.notify_all()

    async def _guard_malformed_response(
        self,
        state: _Session,
        response: object,
        error: MCPFormSessionError,
    ) -> None:
        response_id = _possible_id(response)
        async with state.lock:
            if response_id is None and len(state.pending) > 1:
                raise _error(
                    MCPFormErrorCode.AMBIGUOUS_RESPONSE,
                    MCP_INVALID_PARAMS,
                    "MCP form response is ambiguous",
                ) from error
            return

    def _finish(
        self,
        state: _Session,
        response_id: MCPRequestId,
        outcome: InputHandlerOutcome,
        *,
        replay_fingerprint: bytes | None = None,
    ) -> None:
        pending = state.pending.pop(response_id)
        self._remove_outbound(state, pending)
        self._remember(state, response_id, replay_fingerprint)
        if not pending.future.done():
            pending.future.set_result(outcome)

    def _settle_all(
        self,
        state: _Session,
        reason: InputDisconnectReason,
    ) -> None:
        outcome = InputHandlerDisconnected(reason=reason)
        for response_id in tuple(state.pending):
            self._finish(state, response_id, outcome)

    async def _close_state(self, state: _Session) -> None:
        async with state.condition:
            state.closed = True
            state.outbound.clear()
            self._settle_all(
                state,
                InputDisconnectReason.CONTROL_CHANNEL_CLOSED,
            )
            state.replays.clear()
            state.condition.notify_all()

    async def _close_removed_states(
        self,
        states: tuple[_Session, ...],
    ) -> None:
        cleanup = get_running_loop().create_task(self._close_states(states))
        if await _await_shielded_cleanup(cleanup):
            raise CancelledError

    async def _close_states(self, states: tuple[_Session, ...]) -> None:
        for state in states:
            await self._close_state(state)

    def _remove_outbound(self, state: _Session, pending: _Pending) -> None:
        for index, item in enumerate(state.outbound):
            if item is pending.outbound:
                del state.outbound[index]
                return

    def _remember(
        self,
        state: _Session,
        response_id: MCPRequestId,
        replay_fingerprint: bytes | None = None,
    ) -> None:
        state.stale.append(response_id)
        if replay_fingerprint is not None:
            state.replays[response_id] = replay_fingerprint
        while len(state.stale) > self._stale_limit:
            evicted = state.stale.popleft()
            state.replays.pop(evicted, None)


def _question_schema(question: InputQuestion) -> JsonObject:
    schema = _presentation(question)
    if type(question) is ConfirmationQuestion:
        schema["type"] = "boolean"
        if question.default_value is not None:
            schema["default"] = question.default_value
    elif type(question) is TextQuestion:
        schema.update(
            {
                "type": "string",
                "minLength": max(
                    question.constraints.minimum_length,
                    int(question.required),
                ),
                "maxLength": question.constraints.maximum_length,
            }
        )
        if question.default_value is not None:
            schema["default"] = question.default_value
    elif type(question) is MultilineTextQuestion:
        schema.update(
            {
                "type": "string",
                "pattern": r"^(?:[^\r]|\r\n)*$",
                "minLength": max(
                    question.constraints.minimum_length,
                    int(question.required),
                ),
                "maxLength": question.constraints.maximum_length,
            }
        )
        if question.default_value is not None:
            schema["default"] = question.default_value
    elif type(question) is SingleSelectionQuestion:
        schema.update(
            {
                "type": "string",
                "enum": [str(choice.value) for choice in question.choices],
            }
        )
        _describe_choices(schema, question)
        if question.default_value is not None:
            schema["default"] = str(question.default_value)
    else:
        assert type(question) is MultipleSelectionQuestion
        schema.update(
            {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [str(choice.value) for choice in question.choices],
                },
                "uniqueItems": True,
                "minItems": (
                    0
                    if question.allow_other
                    else max(
                        question.constraints.minimum,
                        int(question.required),
                    )
                ),
                "maxItems": question.constraints.maximum,
            }
        )
        _describe_choices(schema, question)
        if question.default_value is not None:
            schema["default"] = [
                str(value) for value in question.default_value
            ]
    return schema


def _presentation(question: InputQuestion) -> JsonObject:
    result: JsonObject = {"title": question.header or question.prompt}
    description = []
    if question.header is not None:
        description.append(question.prompt)
    if question.help_text is not None:
        description.append(question.help_text)
    if description:
        result["description"] = " ".join(description)
    return result


def _describe_choices(
    schema: JsonObject,
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
) -> None:
    choices = []
    for choice in question.choices:
        text = f"{choice.value}: {choice.label}"
        if choice.description is not None:
            text = f"{text} ({choice.description})"
        choices.append(text)
    text = "Choices: " + "; ".join(choices)
    description = schema.get("description")
    schema["description"] = (
        text if description is None else f"{description} {text}"
    )


def _result_outcome(
    request: InputRequest,
    result: Mapping[str, object],
) -> InputHandlerOutcome:
    action = result.get("action")
    if action == "decline" or action == "cancel":
        if "content" in result:
            raise _invalid(f"{action} cannot carry form content")
        if action == "cancel":
            return InputHandlerDisconnected(
                reason=InputDisconnectReason.HANDLER_CANCELLED
            )
        return InputHandlerResolution(
            resolution=DeclinedResolution(
                request_id=request.request_id,
                provenance=AnswerProvenance.HUMAN,
                resolved_at=_resolved_at(request),
            )
        )
    if action != "accept":
        raise _invalid("elicitation action is invalid")
    content = result.get("content")
    if not isinstance(content, Mapping):
        raise _invalid("accepted form response requires content")
    expected = {str(question.question_id) for question in request.questions}
    expected.update(
        mcp_form_other_property_name(question.question_id)
        for question in request.questions
        if isinstance(
            question,
            (SingleSelectionQuestion, MultipleSelectionQuestion),
        )
        and question.allow_other
    )
    if any(not isinstance(key, str) or key not in expected for key in content):
        raise _invalid("form content contains an unknown property")
    try:
        answers = tuple(
            answer
            for question in request.questions
            if (answer := _answer(question, content)) is not None
        )
        resolution = AnsweredResolution(
            request_id=request.request_id,
            provenance=AnswerProvenance.HUMAN,
            resolved_at=_resolved_at(request),
            answers=answers,
        )
        _validate_resolution_against_request(request, resolution)
    except InputValidationError as exc:
        raise _invalid(exc.safe_message) from exc
    return InputHandlerResolution(resolution=resolution)


def _validate_related_task(
    result: Mapping[str, object],
    related_task_id: str | None,
) -> None:
    if related_task_id is None:
        return
    metadata = result.get("_meta")
    related = (
        metadata.get(MCP_RELATED_TASK_METADATA_KEY)
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(related, Mapping) or (
        related.get("taskId") != related_task_id
    ):
        raise _invalid("related task metadata is missing or mismatched")


def _answer(
    question: InputQuestion,
    content: Mapping[object, object],
) -> InputAnswer | None:
    name = str(question.question_id)
    if type(question) is ConfirmationQuestion:
        if name not in content:
            return _missing(question)
        value = content[name]
        if type(value) is not bool:
            raise _invalid("confirmation answer must be a boolean")
        return ConfirmationAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            value=value,
        )
    if type(question) is TextQuestion:
        if name not in content:
            return _missing(question)
        value = content[name]
        if not isinstance(value, str):
            raise _invalid("text answer must be a string")
        return TextAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            value=value,
        )
    if type(question) is MultilineTextQuestion:
        if name not in content:
            return _missing(question)
        value = content[name]
        if not isinstance(value, str):
            raise _invalid("text answer must be a string")
        return MultilineTextAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            value=value,
        )
    assert isinstance(
        question,
        (SingleSelectionQuestion, MultipleSelectionQuestion),
    )
    other_name = mcp_form_other_property_name(question.question_id)
    has_selected = name in content
    has_other = other_name in content
    if type(question) is SingleSelectionQuestion:
        if has_selected == has_other:
            if not has_selected:
                return _missing(question)
            raise _invalid("selection answer is ambiguous")
        selected = (
            _other(content[other_name])
            if has_other
            else _selected(question, content[name])
        )
        return SingleSelectionAnswer(
            question_id=question.question_id,
            provenance=AnswerProvenance.HUMAN,
            value=selected,
        )
    raw = content.get(name, [])
    if not isinstance(raw, list) or any(
        not isinstance(value, str) for value in raw
    ):
        raise _invalid("multiple selection must be an array of strings")
    selected_values = tuple(_selected(question, value) for value in raw)
    values = selected_values + (
        (_other(content[other_name]),) if has_other else ()
    )
    if not has_selected and not has_other:
        return None
    return MultipleSelectionAnswer(
        question_id=question.question_id,
        provenance=AnswerProvenance.HUMAN,
        values=values,
    )


def _selected(
    question: SingleSelectionQuestion | MultipleSelectionQuestion,
    value: object,
) -> SelectedChoice:
    if not isinstance(value, str):
        raise _invalid("selection answer must be a string")
    return SelectedChoice(value=ChoiceValue(value))


def _other(value: object) -> FreeFormOther:
    if not isinstance(value, str):
        raise _invalid("Other answer must be a string")
    return FreeFormOther(text=value)


def _missing(question: InputQuestion) -> InputAnswer | None:
    if question.required:
        raise _invalid("a required form answer is missing")
    return None


def _response(
    response: object,
) -> tuple[
    MCPRequestId,
    Mapping[str, object] | None,
    Mapping[str, object] | None,
]:
    _response_size(response)
    if not isinstance(response, Mapping) or response.get("jsonrpc") != "2.0":
        raise _invalid("JSON-RPC response is invalid")
    if "id" not in response:
        raise _invalid("JSON-RPC response id is missing")
    response_id = _request_id(response["id"])
    has_result = "result" in response
    has_error = "error" in response
    if has_result == has_error:
        raise _error(
            MCPFormErrorCode.AMBIGUOUS_RESPONSE,
            MCP_INVALID_PARAMS,
            "JSON-RPC response must contain one result or error",
        )
    value = response["result"] if has_result else response["error"]
    if not isinstance(value, Mapping):
        raise _invalid("JSON-RPC result or error must be an object")
    typed = cast(Mapping[str, object], value)
    if has_result:
        return response_id, typed, None
    if type(typed.get("code")) is not int or not isinstance(
        typed.get("message"),
        str,
    ):
        raise _invalid("JSON-RPC error is invalid")
    return response_id, None, typed


def _response_size(response: object) -> None:
    try:
        encoded = dumps(
            response,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _invalid("JSON-RPC response is not valid JSON") from exc
    if len(encoded) > MCP_FORM_RESPONSE_MAX_BYTES:
        raise _error(
            MCPFormErrorCode.OVERSIZED_RESPONSE,
            MCP_INVALID_PARAMS,
            "JSON-RPC response exceeds its byte limit",
        )


def _response_fingerprint(
    key: bytes,
    session_id: str,
    response: object,
) -> bytes:
    try:
        normalized = dumps(
            response,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _invalid("JSON-RPC response is not valid JSON") from exc
    encoded_session = session_id.encode()
    fingerprint = blake2b(key=key, digest_size=32)
    fingerprint.update(b"avalan-mcp-replay-v1")
    fingerprint.update(len(encoded_session).to_bytes(4, byteorder="big"))
    fingerprint.update(encoded_session)
    fingerprint.update(normalized)
    return fingerprint.digest()


def _reject_sensitive_request(request: InputRequest) -> None:
    values = [request.reason, request.context_label or ""]
    serialized_values: list[str] = []
    for question in request.questions:
        values.extend(
            (
                str(question.question_id),
                question.prompt,
                question.header or "",
                question.help_text or "",
            )
        )
        projected_question = cast(
            ConfirmationQuestion
            | TextQuestion
            | MultilineTextQuestion
            | SingleSelectionQuestion
            | MultipleSelectionQuestion,
            question,
        )
        default_value = projected_question.default_value
        if isinstance(default_value, str):
            values.append(default_value)
            serialized_values.append(default_value)
        elif isinstance(default_value, tuple):
            defaults = [str(value) for value in default_value]
            values.extend(defaults)
            serialized_values.extend(defaults)
        if isinstance(
            question,
            (SingleSelectionQuestion, MultipleSelectionQuestion),
        ):
            for choice in question.choices:
                serialized_value = str(choice.value)
                values.extend(
                    (
                        serialized_value,
                        choice.label,
                        choice.description or "",
                    )
                )
                serialized_values.append(serialized_value)
    if any(
        _SENSITIVE_FORM_PATTERN.search(value.casefold())
        or _SENSITIVE_ACRONYM_PATTERN.search(value)
        or _contains_sensitive_pin_context(value)
        for value in values
    ) or any(_looks_like_secret_value(value) for value in serialized_values):
        raise _error(
            MCPFormErrorCode.UNSAFE_REQUEST,
            MCP_INVALID_PARAMS,
            "sensitive or authentication input requires a separate flow",
        )


def _contains_sensitive_pin_context(value: str) -> bool:
    normalized = value.casefold()
    if _PIN_WORD_PATTERN.search(normalized) is None:
        return False
    return any(
        pattern.search(normalized)
        for pattern in (
            _PIN_REQUEST_PATTERN,
            _PIN_POSSESSIVE_QUESTION_PATTERN,
            _PIN_POSSESSIVE_PLEASE_PATTERN,
            _PIN_AUTH_CONTEXT_PATTERN,
        )
    )


def _looks_like_secret_value(value: str) -> bool:
    if _SECRET_VALUE_PATTERN.search(value):
        return True
    for candidate in _CARD_NUMBER_CANDIDATE_PATTERN.finditer(value):
        digits = "".join(
            character for character in candidate.group() if character.isdigit()
        )
        if _passes_luhn_check(digits):
            return True
    return False


def _passes_luhn_check(value: str) -> bool:
    total = 0
    parity = len(value) % 2
    for index, character in enumerate(value):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


async def _await_shielded_cleanup(cleanup: Future[None]) -> bool:
    cancelled = False
    while not cleanup.done():
        try:
            await shield(cleanup)
        except CancelledError:
            cancelled = True
    cleanup.result()
    return cancelled


async def _notify(
    hook: MCPFormStatusHook | None,
    session_id: str,
    request: InputRequest,
    status: MCPFormStatus,
    related_task_id: str | None,
    safe_code: MCPFormErrorCode | None = None,
) -> bool:
    if hook is None:
        return True
    try:
        await hook(
            MCPFormStatusEvent(
                session_id,
                str(request.request_id),
                status,
                related_task_id,
                safe_code,
            )
        )
    except Exception:
        return False
    return True


def _outcome_status(outcome: InputHandlerOutcome) -> MCPFormStatus:
    if isinstance(outcome, InputHandlerResolution):
        return (
            MCPFormStatus.ANSWERED
            if isinstance(outcome.resolution, AnsweredResolution)
            else MCPFormStatus.DECLINED
        )
    if (
        isinstance(outcome, InputHandlerDisconnected)
        and outcome.reason is InputDisconnectReason.HANDLER_CANCELLED
    ):
        return MCPFormStatus.CANCELLED
    return MCPFormStatus.UNAVAILABLE


def _mode(capabilities: Mapping[object, object], name: str) -> bool:
    if name not in capabilities or capabilities[name] is None:
        return False
    if not isinstance(capabilities[name], Mapping):
        raise _error(
            MCPFormErrorCode.INVALID_CAPABILITIES,
            MCP_INVALID_PARAMS,
            f"elicitation {name} capability must be an object",
        )
    return True


def _view(state: _Session) -> MCPFormSessionView:
    return MCPFormSessionView(
        state.session_id,
        MCP_PROTOCOL_VERSION,
        state.capabilities,
        state.form_available,
    )


def _possible_id(response: object) -> MCPRequestId | None:
    if not isinstance(response, Mapping):
        return None
    value = response.get("id")
    if type(value) is int or isinstance(value, str) and value:
        return value
    return None


def _request_id(value: object) -> MCPRequestId:
    if type(value) is int or isinstance(value, str) and value:
        return value
    raise _invalid("JSON-RPC id must be a non-empty string or integer")


def _resolved_at(request: InputRequest) -> datetime:
    return max(datetime.now(UTC), request.created_at)


def _unavailable() -> InputHandlerDisconnected:
    return InputHandlerDisconnected(
        reason=InputDisconnectReason.HANDLER_UNAVAILABLE
    )


def _invalid(message: str) -> MCPFormSessionError:
    return _error(
        MCPFormErrorCode.INVALID_RESPONSE,
        MCP_INVALID_PARAMS,
        message,
    )


def _unavailable_error(message: str) -> MCPFormSessionError:
    return _error(
        MCPFormErrorCode.CAPABILITY_UNAVAILABLE,
        MCP_UNAVAILABLE,
        message,
    )


def _session_not_found() -> MCPFormSessionError:
    return _error(
        MCPFormErrorCode.SESSION_NOT_FOUND,
        MCP_UNAVAILABLE,
        "MCP session is unavailable",
    )


def _error(
    code: MCPFormErrorCode,
    rpc_code: int,
    message: str,
) -> MCPFormSessionError:
    return MCPFormSessionError(code, rpc_code, message)
