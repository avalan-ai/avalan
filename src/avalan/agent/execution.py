"""Own invocation-scoped agent execution state."""

from ..entities import (
    EngineMessageIdempotencyKey,
    Input,
    Message,
    MessageRole,
    MessageToolCall,
    normalize_tool_arguments,
)
from ..interaction.broker import (
    InteractionBroker,
    InteractionBrokerRequest,
    InteractionBrokerResult,
    InteractionRequestResult,
)
from ..interaction.codec import encode_input_model_result
from ..interaction.continuation import (
    ContinuationDispatchId,
    derive_continuation_dispatch_id,
    derive_provider_idempotency_key,
)
from ..interaction.durable import DurableInteractionSuspension
from ..interaction.entities import (
    AgentId,
    BranchId,
    ContinuationId,
    ContinuationRevisionBinding,
    ContinuationSnapshot,
    ExecutionDefinitionRef,
    ExecutionOrigin,
    InputAnsweredResult,
    InputCancelledResult,
    InputDeclinedResult,
    InputModelResult,
    InputRequest,
    InputRequiredResult,
    InputTimedOutResult,
    InputUnavailableResult,
    ModelCallId,
    PrincipalScope,
    ProviderIdempotencyKey,
    RequestState,
    ResumeInputContinuation,
    RunId,
    StreamSessionId,
    TaskId,
    TerminateInputContinuation,
    TurnId,
)
from ..interaction.handler import _InputHandler
from ..interaction.policy import InteractionActor
from ..interaction.state import project_resolution_to_model
from ..interaction.store import (
    CreateInteractionApplied,
    CreateInteractionCommand,
    CreateInteractionRejected,
    InteractionBranchRegistration,
    InteractionBranchRegistrationApplied,
    InteractionBranchRegistrationRejected,
    InteractionBranchRegistrationReplayed,
    InteractionExecutionScope,
    RegisterInteractionBranchCommand,
    ScopeCancellationApplied,
    ScopeCancellationRejected,
    ScopeCancellationReplayed,
    TerminalizeInteractionScopeCommand,
)
from ..interaction.validation import MAX_STATE_REVISION, validate_opaque_id
from ..model.capability import (
    ContinuationSnapshotCodecRegistry,
    CorrelatedCapabilityResult,
    RegisteredContinuationSnapshotCodec,
    TaskInputCapabilityAdvertisement,
    TaskInputCapabilityCall,
)

from asyncio import Lock, Task, create_task, shield
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from inspect import iscoroutinefunction
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, TypeGuard, cast, final
from uuid import NAMESPACE_URL, uuid4, uuid5

MAXIMUM_EQUIVALENT_INPUT_REQUESTS = 3

ExecutionPromptInput: TypeAlias = (
    str | tuple[str, ...] | Message | tuple[Message, ...]
)
ModelResponse: TypeAlias = str

_EXECUTION_MEMORY_KEY_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "avalan.execution-memory.v1",
)

_INPUT_MODEL_RESULT_TYPES = (
    InputAnsweredResult,
    InputDeclinedResult,
    InputCancelledResult,
    InputTimedOutResult,
    InputUnavailableResult,
)

_UNCHANGED = object()


class AgentExecutionStatus(StrEnum):
    """Identify one invocation's control state."""

    RUNNING = "running"
    PREPARING_INPUT = "preparing_input"
    WAITING_FOR_INPUT = "waiting_for_input"
    RESUMING = "resuming"
    INPUT_REQUIRED = "input_required"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"


class ExecutionLedgerEntryKind(StrEnum):
    """Identify immutable execution-ledger records."""

    INPUT = "input"
    TRANSCRIPT = "transcript"
    MODEL_PROMPT = "model_prompt"
    MODEL_RESPONSE = "model_response"
    INTERACTION_RESERVED = "interaction_reserved"
    INTERACTION_PENDING = "interaction_pending"
    INTERACTION_ABANDONED = "interaction_abandoned"
    INTERACTION_RESULT = "interaction_result"
    INTERACTION_TERMINATED = "interaction_terminated"
    INPUT_REQUIRED = "input_required"
    MODEL_TURN = "model_turn"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERRORED = "errored"
    CLEANUP_CLAIMED = "cleanup_claimed"


class ExecutionStateError(RuntimeError):
    """Report an illegal invocation-state operation."""


class ExecutionRevisionError(ExecutionStateError):
    """Report a stale compare-and-swap mutation."""


class ExecutionCorrelationError(ExecutionStateError):
    """Report interaction data bound to another execution or request."""


class InteractionLoopLimitError(ExecutionStateError):
    """Report a bounded repeated structured-input loop."""


class ExecutionInputRequiredError(RuntimeError):
    """Expose segment suspension without disguising it as completion."""

    result: InputRequiredResult
    request: InputRequest | None
    durable: DurableInteractionSuspension | None

    def __init__(
        self,
        result: InputRequiredResult,
        *,
        request: InputRequest | None = None,
        durable: DurableInteractionSuspension | None = None,
    ) -> None:
        if type(result) is not InputRequiredResult:
            raise TypeError("result must be an input-required result")
        if request is not None and type(request) is not InputRequest:
            raise TypeError("request must be an input request or None")
        if durable is not None:
            if type(durable) is not DurableInteractionSuspension:
                raise TypeError(
                    "durable must be a durable interaction suspension"
                )
            durable_request = durable.command.request
            if (
                durable_request.request_id != result.request_id
                or durable_request.continuation_id != result.continuation_id
                or not result.detached_resumption_available
            ):
                raise ExecutionCorrelationError(
                    "durable suspension does not match input-required result"
                )
            if request is not None and request != durable_request:
                raise ExecutionCorrelationError(
                    "request does not match durable input suspension"
                )
            request = durable_request
        if request is not None and (
            request.request_id != result.request_id
            or request.continuation_id != result.continuation_id
        ):
            raise ExecutionCorrelationError(
                "request does not match input-required result"
            )
        self.result = result
        self.request = request
        self.durable = durable
        super().__init__("execution requires correlated input")


class ExecutionTerminatedError(RuntimeError):
    """Report an interaction outcome that terminates its logical run."""

    outcome: TerminateInputContinuation

    def __init__(self, outcome: TerminateInputContinuation) -> None:
        if type(outcome) is not TerminateInputContinuation:
            raise TypeError("outcome must terminate an input continuation")
        self.outcome = outcome
        super().__init__("input continuation terminated the execution")


@final
@dataclass(frozen=True, slots=True, kw_only=True, init=False)
class ModelPromptRecord:
    """Capture one immutable prompt dispatched to a model."""

    input: ExecutionPromptInput
    instructions: str | None
    system_prompt: str | None
    developer_prompt: str | None

    def __init__(
        self,
        *,
        input: Input,
        instructions: str | None,
        system_prompt: str | None,
        developer_prompt: str | None,
    ) -> None:
        object.__setattr__(self, "input", _freeze_prompt_input(input))
        for name, value in (
            ("instructions", instructions),
            ("system_prompt", system_prompt),
            ("developer_prompt", developer_prompt),
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError(f"{name} must be a string or None")
            object.__setattr__(self, name, value)


class ExecutionMemoryComponent(StrEnum):
    """Identify one memory-bearing execution-ledger component."""

    MESSAGE = "message"
    RESPONSE = "response"


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionMemoryEntry:
    """Carry one retry-stable execution-ledger memory append."""

    origin: ExecutionOrigin
    ledger_sequence: int
    component: ExecutionMemoryComponent
    component_index: int
    message: Message

    def __post_init__(self) -> None:
        if not isinstance(self.origin, ExecutionOrigin):
            raise TypeError("origin must be an execution origin")
        for name in ("ledger_sequence", "component_index"):
            value = getattr(self, name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
            ):
                raise TypeError(f"{name} must be a non-negative integer")
        if not isinstance(self.component, ExecutionMemoryComponent):
            raise TypeError("component must be an execution-memory component")
        if not isinstance(self.message, Message):
            raise TypeError("message must be a message")
        object.__setattr__(
            self,
            "message",
            _snapshot_messages((self.message,))[0],
        )

    @property
    def idempotency_key(self) -> EngineMessageIdempotencyKey:
        """Return the deterministic append key for this ledger component."""
        identity = "\x1f".join(
            (
                str(self.origin.run_id),
                str(self.origin.agent_id),
                str(self.origin.branch_id),
                str(self.ledger_sequence),
                self.component.value,
                str(self.component_index),
            )
        )
        return EngineMessageIdempotencyKey(
            value=uuid5(_EXECUTION_MEMORY_KEY_NAMESPACE, identity)
        )


class ExecutionMemorySink(Protocol):
    """Persist execution memory idempotently by its stable entry key."""

    async def append_execution_memory_entry(
        self,
        entry: ExecutionMemoryEntry,
    ) -> None:
        """Persist one entry exactly once for its idempotency key."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionLedgerEntry:
    """Record one immutable invocation transition and its typed payload."""

    sequence: int
    kind: ExecutionLedgerEntryKind
    origin: ExecutionOrigin
    messages: tuple[Message, ...] = ()
    prompt: ModelPromptRecord | None = None
    response: ModelResponse | None = None
    response_in_transcript: bool = False
    semantic_fingerprint: str | None = None
    task_input_call: TaskInputCapabilityCall | None = None
    interaction_assistant_message: Message | None = None
    request: InputRequest | None = None
    result: InputModelResult | None = None
    input_required: InputRequiredResult | None = None
    termination_outcome: TerminateInputContinuation | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sequence, int)
            or isinstance(self.sequence, bool)
            or self.sequence < 0
        ):
            raise TypeError("sequence must be a non-negative integer")
        if not isinstance(self.kind, ExecutionLedgerEntryKind):
            raise TypeError("kind must be an execution-ledger entry kind")
        if not isinstance(self.origin, ExecutionOrigin):
            raise TypeError("origin must be an execution origin")
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, Message) for message in self.messages
        ):
            raise TypeError("messages must be a tuple of messages")
        object.__setattr__(self, "messages", _snapshot_messages(self.messages))
        if self.prompt is not None and not isinstance(
            self.prompt, ModelPromptRecord
        ):
            raise TypeError("prompt must be a model-prompt record")
        if self.prompt is not None:
            object.__setattr__(self, "prompt", _snapshot_prompt(self.prompt))
        if self.response is not None and not isinstance(self.response, str):
            raise TypeError("response must be a model response")
        if not isinstance(self.response_in_transcript, bool):
            raise TypeError("response_in_transcript must be a boolean")
        if self.kind is ExecutionLedgerEntryKind.MODEL_RESPONSE:
            if bool(self.messages) is not self.response_in_transcript:
                raise ExecutionStateError(
                    "response transcript presence must be explicit"
                )
            if self.messages != (
                ()
                if self.response is None or not self.response_in_transcript
                else (
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=self.response,
                    ),
                )
            ):
                raise ExecutionCorrelationError(
                    "transcript response must exactly materialize "
                    "response text"
                )
        if self.semantic_fingerprint is not None:
            object.__setattr__(
                self,
                "semantic_fingerprint",
                _validate_fingerprint(self.semantic_fingerprint),
            )
        if self.task_input_call is not None:
            if type(self.task_input_call) is not TaskInputCapabilityCall:
                raise TypeError(
                    "task_input_call must be a task-input capability call"
                )
            object.__setattr__(
                self,
                "task_input_call",
                cast(
                    TaskInputCapabilityCall,
                    _snapshot_value(self.task_input_call),
                ),
            )
        if self.interaction_assistant_message is not None:
            if not isinstance(self.interaction_assistant_message, Message):
                raise TypeError(
                    "interaction_assistant_message must be a message"
                )
            object.__setattr__(
                self,
                "interaction_assistant_message",
                _snapshot_messages((self.interaction_assistant_message,))[0],
            )
        if (
            self.task_input_call is not None
            and self.interaction_assistant_message is not None
        ):
            _validate_originating_assistant_message(
                self.task_input_call,
                self.interaction_assistant_message,
            )
        if self.request is not None and type(self.request) is not InputRequest:
            raise TypeError("request must be an input request")
        if self.result is not None and not _is_input_model_result(self.result):
            raise TypeError("result must be an input model result")
        if (
            self.input_required is not None
            and type(self.input_required) is not InputRequiredResult
        ):
            raise TypeError("input_required must be an input-required result")
        if (
            self.termination_outcome is not None
            and type(self.termination_outcome)
            is not TerminateInputContinuation
        ):
            raise TypeError(
                "termination_outcome must terminate an input continuation"
            )
        populated = (
            bool(self.messages),
            self.prompt is not None,
            self.response is not None,
            self.response_in_transcript,
            self.semantic_fingerprint is not None,
            self.task_input_call is not None,
            self.interaction_assistant_message is not None,
            self.request is not None,
            self.result is not None,
            self.input_required is not None,
            self.termination_outcome is not None,
        )
        if not _entry_payload_is_legal(self.kind, populated):
            raise ExecutionStateError("ledger payload does not match its kind")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AgentExecutionSnapshot:
    """Expose one immutable compare-and-swap execution snapshot."""

    revision: int
    status: AgentExecutionStatus
    origin: ExecutionOrigin
    ledger: tuple[ExecutionLedgerEntry, ...]
    messages: tuple[Message, ...]
    pending_request: InputRequest | None
    active_interaction_fingerprint: str | None
    interaction_fingerprint_counts: tuple[tuple[str, int], ...]
    interaction_count: int
    memory_sync_cursor: int
    response_sync_cursor: int
    cleanup_started: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.revision, int)
            or isinstance(self.revision, bool)
            or not 0 <= self.revision <= MAX_STATE_REVISION
        ):
            raise TypeError("revision must be a bounded non-negative integer")
        if not isinstance(self.status, AgentExecutionStatus):
            raise TypeError("status must be an agent execution status")
        if not isinstance(self.origin, ExecutionOrigin):
            raise TypeError("origin must be an execution origin")
        if not isinstance(self.ledger, tuple) or not all(
            isinstance(entry, ExecutionLedgerEntry) for entry in self.ledger
        ):
            raise TypeError("ledger must contain execution-ledger entries")
        if tuple(entry.sequence for entry in self.ledger) != tuple(
            range(len(self.ledger))
        ):
            raise ExecutionStateError("ledger sequence is not contiguous")
        if not isinstance(self.messages, tuple) or not all(
            isinstance(message, Message) for message in self.messages
        ):
            raise TypeError("messages must be a tuple of messages")
        object.__setattr__(self, "messages", _snapshot_messages(self.messages))
        if (
            self.pending_request is not None
            and type(self.pending_request) is not InputRequest
        ):
            raise TypeError("pending_request must be an input request")
        if (
            self.pending_request is not None
            and self.pending_request.state is not RequestState.PENDING
            and not (
                self.status is AgentExecutionStatus.INPUT_REQUIRED
                and self.pending_request.state is RequestState.CREATED
            )
        ):
            raise ExecutionStateError(
                "stored interaction must be pending or durably staged"
            )
        if self.active_interaction_fingerprint is not None:
            _validate_fingerprint(self.active_interaction_fingerprint)
        if not isinstance(self.interaction_fingerprint_counts, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], int)
            or isinstance(item[1], bool)
            or item[1] < 1
            for item in self.interaction_fingerprint_counts
        ):
            raise TypeError("interaction counts must be positive pairs")
        if tuple(sorted(self.interaction_fingerprint_counts)) != (
            self.interaction_fingerprint_counts
        ):
            raise ExecutionStateError("interaction counts must be sorted")
        if (
            not isinstance(self.interaction_count, int)
            or isinstance(self.interaction_count, bool)
            or self.interaction_count < 0
        ):
            raise TypeError("interaction_count must be non-negative")
        if (
            sum(count for _, count in self.interaction_fingerprint_counts)
            != self.interaction_count
        ):
            raise ExecutionStateError("interaction counts do not add up")
        if (
            not isinstance(self.memory_sync_cursor, int)
            or isinstance(self.memory_sync_cursor, bool)
            or not 0 <= self.memory_sync_cursor <= len(self.messages)
        ):
            raise TypeError("memory_sync_cursor is outside the transcript")
        if (
            not isinstance(self.response_sync_cursor, int)
            or isinstance(self.response_sync_cursor, bool)
            or not 0 <= self.response_sync_cursor <= len(self.ledger)
        ):
            raise TypeError("response_sync_cursor is outside the ledger")
        if not isinstance(self.cleanup_started, bool):
            raise TypeError("cleanup_started must be a boolean")
        interaction_statuses = {
            AgentExecutionStatus.WAITING_FOR_INPUT,
            AgentExecutionStatus.INPUT_REQUIRED,
        }
        if self.status in interaction_statuses and (
            self.pending_request is None
            or self.active_interaction_fingerprint is None
        ):
            raise ExecutionStateError(
                "waiting execution must retain its pending interaction"
            )
        if self.status is AgentExecutionStatus.PREPARING_INPUT and (
            self.pending_request is not None
            or self.active_interaction_fingerprint is None
        ):
            raise ExecutionStateError(
                "preparing execution must retain only its reservation"
            )
        if self.status in {
            AgentExecutionStatus.RUNNING,
            AgentExecutionStatus.RESUMING,
        } and (
            self.pending_request is not None
            or self.active_interaction_fingerprint is not None
        ):
            raise ExecutionStateError(
                "active execution cannot retain a pending interaction"
            )
        _validate_ledger_origins(self.origin, self.ledger)
        _validate_terminal_ledger(
            self.status,
            self.ledger,
            self.cleanup_started,
        )
        replay = _replay_execution_ledger(self.ledger)
        _validate_snapshot_replay(self, replay)


def _validate_ledger_origins(
    active_origin: ExecutionOrigin,
    ledger: tuple[ExecutionLedgerEntry, ...],
) -> None:
    if not ledger:
        raise ExecutionStateError("execution ledger must not be empty")
    if ledger[0].kind is not ExecutionLedgerEntryKind.INPUT:
        raise ExecutionStateError("execution ledger must begin with input")
    initial = ledger[0].origin
    stable_fields = (
        "run_id",
        "task_id",
        "agent_id",
        "branch_id",
        "parent_branch_id",
        "definition",
        "principal",
    )
    for entry in ledger:
        if any(
            getattr(entry.origin, name) != getattr(initial, name)
            for name in stable_fields
        ):
            raise ExecutionCorrelationError(
                "ledger entry changed immutable execution identity"
            )
    for previous, entry in zip(ledger, ledger[1:], strict=False):
        if entry.kind is ExecutionLedgerEntryKind.MODEL_TURN:
            if (
                entry.origin.turn_id == previous.origin.turn_id
                or entry.origin.model_call_id == previous.origin.model_call_id
            ):
                raise ExecutionCorrelationError(
                    "model turn must mint new volatile identities"
                )
            continue
        if entry.origin != previous.origin:
            raise ExecutionCorrelationError(
                "only a model-turn entry may change volatile identity"
            )
    if ledger[-1].origin != active_origin:
        raise ExecutionCorrelationError(
            "active execution origin does not match the ledger tail"
        )


def _validate_terminal_ledger(
    status: AgentExecutionStatus,
    ledger: tuple[ExecutionLedgerEntry, ...],
    cleanup_started: bool,
) -> None:
    terminal_kinds = frozenset(
        {
            ExecutionLedgerEntryKind.COMPLETED,
            ExecutionLedgerEntryKind.CANCELLED,
            ExecutionLedgerEntryKind.ERRORED,
            ExecutionLedgerEntryKind.INTERACTION_TERMINATED,
        }
    )
    terminal_indexes = tuple(
        index
        for index, entry in enumerate(ledger)
        if entry.kind in terminal_kinds
    )
    cleanup_indexes = tuple(
        index
        for index, entry in enumerate(ledger)
        if entry.kind is ExecutionLedgerEntryKind.CLEANUP_CLAIMED
    )
    if status not in _TERMINAL_STATUSES:
        if terminal_indexes or cleanup_indexes or cleanup_started:
            raise ExecutionStateError(
                "active execution cannot contain terminal ledger entries"
            )
        return
    if len(terminal_indexes) != 1:
        raise ExecutionStateError(
            "terminal execution needs exactly one terminal ledger entry"
        )
    terminal_index = terminal_indexes[0]
    expected_kinds = {
        AgentExecutionStatus.COMPLETED: {ExecutionLedgerEntryKind.COMPLETED},
        AgentExecutionStatus.CANCELLED: {
            ExecutionLedgerEntryKind.CANCELLED,
            ExecutionLedgerEntryKind.INTERACTION_TERMINATED,
        },
        AgentExecutionStatus.ERRORED: {ExecutionLedgerEntryKind.ERRORED},
    }[status]
    if ledger[terminal_index].kind not in expected_kinds:
        raise ExecutionStateError(
            "terminal execution ledger does not match its status"
        )
    tail = ledger[terminal_index + 1 :]
    if any(
        entry.kind is not ExecutionLedgerEntryKind.CLEANUP_CLAIMED
        for entry in tail
    ):
        raise ExecutionStateError(
            "operational ledger entries cannot follow termination"
        )
    if (
        len(cleanup_indexes) > 1
        or (cleanup_indexes and cleanup_indexes[0] <= terminal_index)
        or cleanup_started != bool(cleanup_indexes)
    ):
        raise ExecutionStateError(
            "cleanup state does not match its unique ledger claim"
        )


@dataclass(frozen=True, slots=True)
class _ExecutionLedgerReplay:
    status: AgentExecutionStatus
    origin: ExecutionOrigin
    messages: tuple[Message, ...]
    pending_request: InputRequest | None
    active_interaction_fingerprint: str | None
    interaction_fingerprint_counts: tuple[tuple[str, int], ...]
    interaction_count: int
    cleanup_started: bool


def _replay_execution_ledger(
    ledger: tuple[ExecutionLedgerEntry, ...],
) -> _ExecutionLedgerReplay:
    first = ledger[0]
    status = AgentExecutionStatus.RUNNING
    origin = first.origin
    messages = first.messages
    pending_request: InputRequest | None = None
    active_fingerprint: str | None = None
    active_call: TaskInputCapabilityCall | None = None
    active_assistant_message: Message | None = None
    fingerprint_counts: dict[str, int] = {}
    interaction_count = 0
    cleanup_started = False

    for entry in ledger[1:]:
        kind = entry.kind
        if cleanup_started:
            raise ExecutionStateError(
                "execution ledger contains work after cleanup"
            )
        if status in _TERMINAL_STATUSES:
            if kind is not ExecutionLedgerEntryKind.CLEANUP_CLAIMED:
                raise ExecutionStateError(
                    "execution ledger contains work after termination"
                )
            cleanup_started = True
            continue

        match kind:
            case ExecutionLedgerEntryKind.INPUT:
                raise ExecutionStateError(
                    "execution ledger contains repeated input"
                )
            case ExecutionLedgerEntryKind.MODEL_PROMPT:
                _require_replay_status(status, AgentExecutionStatus.RUNNING)
            case ExecutionLedgerEntryKind.TRANSCRIPT:
                _require_replay_status(status, AgentExecutionStatus.RUNNING)
                messages = (*messages, *entry.messages)
            case ExecutionLedgerEntryKind.MODEL_RESPONSE:
                _require_replay_status(status, AgentExecutionStatus.RUNNING)
                messages = (*messages, *entry.messages)
            case ExecutionLedgerEntryKind.INTERACTION_RESERVED:
                _require_replay_status(status, AgentExecutionStatus.RUNNING)
                fingerprint = cast(str, entry.semantic_fingerprint)
                call = cast(TaskInputCapabilityCall, entry.task_input_call)
                assistant_message = cast(
                    Message,
                    entry.interaction_assistant_message,
                )
                next_count = fingerprint_counts.get(fingerprint, 0) + 1
                if next_count > MAXIMUM_EQUIVALENT_INPUT_REQUESTS:
                    raise ExecutionStateError(
                        "execution ledger exceeds the interaction loop limit"
                    )
                fingerprint_counts[fingerprint] = next_count
                interaction_count += 1
                active_fingerprint = fingerprint
                active_call = call
                active_assistant_message = assistant_message
                status = AgentExecutionStatus.PREPARING_INPUT
            case ExecutionLedgerEntryKind.INTERACTION_ABANDONED:
                _require_replay_status(
                    status,
                    AgentExecutionStatus.PREPARING_INPUT,
                )
                if entry.semantic_fingerprint != active_fingerprint:
                    raise ExecutionCorrelationError(
                        "abandoned interaction changed its reservation"
                    )
                active_fingerprint = None
                active_call = None
                active_assistant_message = None
                status = AgentExecutionStatus.RUNNING
            case ExecutionLedgerEntryKind.INTERACTION_PENDING:
                _require_replay_status(
                    status,
                    AgentExecutionStatus.PREPARING_INPUT,
                )
                request = cast(InputRequest, entry.request)
                call = cast(TaskInputCapabilityCall, active_call)
                if request.origin != origin:
                    raise ExecutionCorrelationError(
                        "pending interaction changed execution origin"
                    )
                _validate_request_matches_task_input_call(request, call)
                pending_request = request
                status = AgentExecutionStatus.WAITING_FOR_INPUT
            case ExecutionLedgerEntryKind.INPUT_REQUIRED:
                request = cast(InputRequest, entry.request)
                required = cast(InputRequiredResult, entry.input_required)
                if status is AgentExecutionStatus.PREPARING_INPUT:
                    if request.state is not RequestState.CREATED:
                        raise ExecutionStateError(
                            "durable input request was already persisted"
                        )
                    call = cast(TaskInputCapabilityCall, active_call)
                    if request.origin != origin:
                        raise ExecutionCorrelationError(
                            "durable input changed execution origin"
                        )
                    _validate_request_matches_task_input_call(request, call)
                    pending_request = request
                else:
                    _require_replay_status(
                        status,
                        AgentExecutionStatus.WAITING_FOR_INPUT,
                    )
                if request != pending_request or (
                    request.request_id != required.request_id
                    or request.continuation_id != required.continuation_id
                ):
                    raise ExecutionCorrelationError(
                        "input-required ledger entry is not correlated"
                    )
                status = AgentExecutionStatus.INPUT_REQUIRED
            case ExecutionLedgerEntryKind.INTERACTION_RESULT:
                if status is AgentExecutionStatus.INPUT_REQUIRED:
                    if (
                        pending_request is None
                        or pending_request.state is not RequestState.CREATED
                    ):
                        raise ExecutionStateError(
                            "durable result lacks its staged request"
                        )
                else:
                    _require_replay_status(
                        status,
                        AgentExecutionStatus.WAITING_FOR_INPUT,
                    )
                request = cast(InputRequest, entry.request)
                result = cast(InputModelResult, entry.result)
                call = cast(TaskInputCapabilityCall, active_call)
                assistant_message = cast(Message, active_assistant_message)
                assert pending_request is not None
                _validate_terminal_request(pending_request, request)
                outcome = project_resolution_to_model(
                    request,
                    containing_run_exists=True,
                )
                if (
                    type(outcome) is not ResumeInputContinuation
                    or outcome.result != result
                    or entry.task_input_call != call
                ):
                    raise ExecutionCorrelationError(
                        "interaction result changed its continuation"
                    )
                _validate_correlated_result_messages(
                    call,
                    assistant_message,
                    result,
                    entry.messages,
                )
                messages = (*messages, *entry.messages)
                pending_request = None
                active_fingerprint = None
                active_call = None
                active_assistant_message = None
                status = AgentExecutionStatus.RESUMING
            case ExecutionLedgerEntryKind.INTERACTION_TERMINATED:
                _require_replay_status(
                    status,
                    AgentExecutionStatus.WAITING_FOR_INPUT,
                )
                request = cast(InputRequest, entry.request)
                outcome = cast(
                    TerminateInputContinuation,
                    entry.termination_outcome,
                )
                assert pending_request is not None
                _validate_terminal_request(pending_request, request)
                projected = project_resolution_to_model(
                    request,
                    containing_run_exists=True,
                )
                if projected != outcome:
                    raise ExecutionCorrelationError(
                        "interaction termination changed its continuation"
                    )
                pending_request = None
                active_fingerprint = None
                active_call = None
                active_assistant_message = None
                status = AgentExecutionStatus.CANCELLED
            case ExecutionLedgerEntryKind.MODEL_TURN:
                if status not in {
                    AgentExecutionStatus.RUNNING,
                    AgentExecutionStatus.RESUMING,
                }:
                    raise ExecutionStateError(
                        "model turn follows an illegal execution state"
                    )
                origin = entry.origin
                status = AgentExecutionStatus.RUNNING
            case ExecutionLedgerEntryKind.COMPLETED:
                _require_replay_status(status, AgentExecutionStatus.RUNNING)
                status = AgentExecutionStatus.COMPLETED
            case ExecutionLedgerEntryKind.CANCELLED:
                pending_request = None
                active_fingerprint = None
                active_call = None
                active_assistant_message = None
                status = AgentExecutionStatus.CANCELLED
            case ExecutionLedgerEntryKind.ERRORED:
                pending_request = None
                active_fingerprint = None
                active_call = None
                active_assistant_message = None
                status = AgentExecutionStatus.ERRORED
            case ExecutionLedgerEntryKind.CLEANUP_CLAIMED:
                raise ExecutionStateError(
                    "cleanup cannot precede execution termination"
                )

    return _ExecutionLedgerReplay(
        status=status,
        origin=origin,
        messages=messages,
        pending_request=pending_request,
        active_interaction_fingerprint=active_fingerprint,
        interaction_fingerprint_counts=tuple(
            sorted(fingerprint_counts.items())
        ),
        interaction_count=interaction_count,
        cleanup_started=cleanup_started,
    )


def _validate_snapshot_replay(
    snapshot: AgentExecutionSnapshot,
    replay: _ExecutionLedgerReplay,
) -> None:
    exact_fields = (
        "status",
        "origin",
        "messages",
        "pending_request",
        "active_interaction_fingerprint",
        "interaction_fingerprint_counts",
        "interaction_count",
        "cleanup_started",
    )
    if any(
        getattr(snapshot, name) != getattr(replay, name)
        for name in exact_fields
    ):
        raise ExecutionStateError(
            "execution snapshot does not match its ledger replay"
        )
    ledger_mutations = len(snapshot.ledger) - 1
    initial_message_count = len(snapshot.ledger[0].messages)
    minimum_message_acks = max(
        0,
        snapshot.memory_sync_cursor - initial_message_count,
    )
    response_indexes = tuple(
        index
        for index, entry in enumerate(
            snapshot.ledger[: snapshot.response_sync_cursor]
        )
        if entry.response is not None and not entry.response_in_transcript
    )
    minimum_response_acks = len(response_indexes) + int(
        snapshot.response_sync_cursor > 0
        and (
            not response_indexes
            or response_indexes[-1] + 1 < snapshot.response_sync_cursor
        )
    )
    minimum_revision = (
        ledger_mutations + minimum_message_acks + minimum_response_acks
    )
    maximum_revision = (
        ledger_mutations
        + snapshot.memory_sync_cursor
        + snapshot.response_sync_cursor
    )
    if not minimum_revision <= snapshot.revision <= maximum_revision:
        raise ExecutionRevisionError(
            "execution revision does not match its ledger and sync cursors"
        )


def _require_replay_status(
    actual: AgentExecutionStatus,
    expected: AgentExecutionStatus,
) -> None:
    if actual is not expected:
        raise ExecutionStateError(
            f"illegal ledger transition from {actual.value}"
        )


class ExecutionIdFactory(Protocol):
    """Mint genuine invocation and provider-turn identities."""

    async def new_run_id(self) -> RunId:
        """Return a new logical run identifier."""
        ...

    async def new_turn_id(self) -> TurnId:
        """Return a new model-turn identifier."""
        ...

    async def new_task_id(self) -> TaskId:
        """Return a new invocation-task identifier."""
        ...

    async def new_model_call_id(self) -> ModelCallId:
        """Return a new provider-call identifier."""
        ...

    async def new_branch_id(self) -> BranchId:
        """Return a new execution-branch identifier."""
        ...

    async def new_stream_session_id(self) -> StreamSessionId:
        """Return a new transport-segment identifier."""
        ...


class BranchInteractionBroker(Protocol):
    """Expose only interaction operations scoped to one execution branch."""

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Admit one request owned by the bound execution branch."""
        ...

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Cancel only the bound execution branch."""
        ...


@final
class UuidExecutionIdFactory:
    """Mint opaque execution identities from random UUIDs."""

    async def new_run_id(self) -> RunId:
        """Return a new logical run identifier."""
        return RunId(str(uuid4()))

    async def new_turn_id(self) -> TurnId:
        """Return a new model-turn identifier."""
        return TurnId(str(uuid4()))

    async def new_task_id(self) -> TaskId:
        """Return a new invocation-task identifier."""
        return TaskId(str(uuid4()))

    async def new_model_call_id(self) -> ModelCallId:
        """Return a new provider-call identifier."""
        return ModelCallId(str(uuid4()))

    async def new_branch_id(self) -> BranchId:
        """Return a new execution-branch identifier."""
        return BranchId(str(uuid4()))

    async def new_stream_session_id(self) -> StreamSessionId:
        """Return a new transport-segment identifier."""
        return StreamSessionId(str(uuid4()))


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AttachedInteractionRuntime:
    """Bind an explicitly attached handler to one branch-scoped broker."""

    broker: InteractionBroker = field(repr=False)
    actor: InteractionActor
    handler: _InputHandler = field(repr=False)
    id_factory: ExecutionIdFactory = field(
        default_factory=UuidExecutionIdFactory,
        repr=False,
    )
    task_id: TaskId | None = None
    branch_id: BranchId | None = None
    parent_branch_id: BranchId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor, InteractionActor):
            raise TypeError("actor must be an interaction actor")
        _assert_async_methods(
            self.broker,
            ("request", "cancel_scope"),
            "attached_runtime.broker",
        )
        _assert_async_callable(self.handler, "attached_runtime.handler")
        _assert_execution_id_factory(self.id_factory, "attached_runtime")
        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                TaskId(validate_opaque_id(self.task_id, "task_id")),
            )
        for name in ("branch_id", "parent_branch_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    BranchId(validate_opaque_id(value, name)),
                )
        if (
            self.branch_id is not None
            and self.parent_branch_id is not None
            and self.branch_id == self.parent_branch_id
        ):
            raise ExecutionCorrelationError(
                "parent branch must differ from the active branch"
            )


class DurableInteractionStager(Protocol):
    """Build one portable suspension without persisting it."""

    async def __call__(
        self,
        request: InteractionBrokerRequest,
        *,
        execution: "AgentExecution",
        response: object,
        stream_sequence: int,
        staging: "DurableInteractionStagingContext",
    ) -> DurableInteractionSuspension:
        """Return one validated uncommitted durable suspension."""
        ...


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableInteractionStagingContext:
    """Bind one provider-owned replay snapshot to its reserved call."""

    task_input_call: TaskInputCapabilityCall
    continuation_id: ContinuationId
    dispatch_id: ContinuationDispatchId
    revision_binding: ContinuationRevisionBinding
    codec_registry: ContinuationSnapshotCodecRegistry = field(repr=False)
    codec: RegisteredContinuationSnapshotCodec
    provider_snapshot: ContinuationSnapshot = field(repr=False)
    provider_idempotency_key: ProviderIdempotencyKey
    provider_call_correlation_id: str

    def __post_init__(self) -> None:
        if type(self.task_input_call) is not TaskInputCapabilityCall:
            raise TypeError(
                "task_input_call must be a task-input capability call"
            )
        if (
            self.task_input_call.advertisement
            is not TaskInputCapabilityAdvertisement.DURABLE
        ):
            raise ExecutionCorrelationError(
                "durable staging requires a durable reserved call"
            )
        continuation_id = ContinuationId(
            validate_opaque_id(
                self.continuation_id,
                "continuation_id",
            )
        )
        object.__setattr__(self, "continuation_id", continuation_id)
        dispatch_id = ContinuationDispatchId(
            validate_opaque_id(
                self.dispatch_id,
                "dispatch_id",
            )
        )
        object.__setattr__(self, "dispatch_id", dispatch_id)
        if dispatch_id != derive_continuation_dispatch_id(continuation_id):
            raise ExecutionCorrelationError(
                "durable staging dispatch does not match the continuation"
            )
        if type(self.revision_binding) is not ContinuationRevisionBinding:
            raise TypeError(
                "revision_binding must be a continuation revision binding"
            )
        if type(self.codec_registry) is not ContinuationSnapshotCodecRegistry:
            raise TypeError("codec_registry must be a codec registry")
        if (
            type(self.codec) is not RegisteredContinuationSnapshotCodec
            or not self.codec_registry.is_registered(self.codec)
            or self.codec.revision_binding != self.revision_binding
        ):
            raise ExecutionCorrelationError(
                "durable staging codec is not registered for the revision"
            )
        if type(self.provider_snapshot) is not ContinuationSnapshot:
            raise TypeError(
                "provider_snapshot must be a continuation snapshot"
            )
        if not self.codec.accepts(self.provider_snapshot):
            raise ExecutionCorrelationError(
                "provider snapshot does not match the durable codec"
            )
        if (
            not isinstance(
                self.provider_idempotency_key,
                str,
            )
            or not self.provider_idempotency_key
        ):
            raise TypeError("provider_idempotency_key must be non-empty")
        if self.provider_idempotency_key != derive_provider_idempotency_key(
            continuation_id,
            dispatch_id,
        ):
            raise ExecutionCorrelationError(
                "provider idempotency key does not match durable dispatch"
            )
        correlation_id = validate_opaque_id(
            self.provider_call_correlation_id,
            "provider_call_correlation_id",
            maximum_characters=256,
            maximum_bytes=1_024,
        )
        object.__setattr__(
            self,
            "provider_call_correlation_id",
            correlation_id,
        )
        if correlation_id != str(self.task_input_call.call_id):
            raise ExecutionCorrelationError(
                "provider correlation does not match the reserved call"
            )
        snapshot = self.provider_snapshot
        if snapshot.provider_idempotency_key != self.provider_idempotency_key:
            raise ExecutionCorrelationError(
                "provider snapshot changed its idempotency key"
            )
        if (
            snapshot.payload.get("reserved_capability_call_id")
            != correlation_id
        ):
            raise ExecutionCorrelationError(
                "provider snapshot changed the reserved call"
            )
        encoded = self.codec_registry.export_snapshot(
            self.codec,
            snapshot,
        )
        restored = self.codec_registry.restore_snapshot(
            self.codec,
            encoded,
            self.revision_binding,
        )
        if restored != snapshot:
            raise ExecutionCorrelationError(
                "provider snapshot codec changed durable replay state"
            )


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class DurableInteractionRuntime:
    """Bind a trusted durable stager to one task execution."""

    actor: InteractionActor
    stager: DurableInteractionStager = field(repr=False)
    id_factory: ExecutionIdFactory = field(
        default_factory=UuidExecutionIdFactory,
        repr=False,
    )
    run_id: RunId | None = None
    task_id: TaskId | None = None
    branch_id: BranchId | None = None
    parent_branch_id: BranchId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.actor, InteractionActor):
            raise TypeError("actor must be an interaction actor")
        _assert_async_callable(self.stager, "durable_runtime.stager")
        _assert_execution_id_factory(self.id_factory, "durable_runtime")
        if self.run_id is not None:
            object.__setattr__(
                self,
                "run_id",
                RunId(validate_opaque_id(self.run_id, "run_id")),
            )
        if self.task_id is not None:
            object.__setattr__(
                self,
                "task_id",
                TaskId(validate_opaque_id(self.task_id, "task_id")),
            )
        for name in ("branch_id", "parent_branch_id"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(
                    self,
                    name,
                    BranchId(validate_opaque_id(value, name)),
                )
        if (
            self.branch_id is not None
            and self.parent_branch_id is not None
            and self.branch_id == self.parent_branch_id
        ):
            raise ExecutionCorrelationError(
                "parent branch must differ from the active branch"
            )


InteractionRuntime: TypeAlias = (
    AttachedInteractionRuntime | DurableInteractionRuntime
)


@final
class ExecutionBranchInteractionBroker:
    """Constrain broker admission and cancellation to one live branch."""

    def __init__(
        self,
        *,
        broker: InteractionBroker,
        actor: InteractionActor,
        current_origin: Callable[[], ExecutionOrigin],
    ) -> None:
        self._broker = broker
        self._actor = actor
        self._current_origin = current_origin
        self._registration_lock = Lock()
        self._registered_branches: set[tuple[RunId, BranchId]] = set()

    async def request(
        self,
        request: InteractionBrokerRequest,
    ) -> InteractionRequestResult:
        """Admit one request only when every branch binding matches."""
        if type(request) is not InteractionBrokerRequest:
            raise TypeError("request must be an interaction broker request")
        origin = self._current_origin()
        self._validate_actor(request.actor)
        if request.origin != origin:
            raise ExecutionCorrelationError(
                "interaction request does not match the active branch"
            )
        await self._ensure_branch_registered(origin)
        result = await self._broker.request(request)
        _validate_broker_request_result(request, result)
        return result

    async def cancel_scope(
        self,
        command: TerminalizeInteractionScopeCommand,
    ) -> InteractionBrokerResult:
        """Cancel only an exact scope rooted at the active branch."""
        if type(command) is not TerminalizeInteractionScopeCommand:
            raise TypeError("command must terminalize an interaction scope")
        origin = self._current_origin()
        self._validate_actor(command.actor)
        self._validate_scope(command.scope, origin)
        await self._ensure_branch_registered(origin)
        result = await self._broker.cancel_scope(command)
        if not isinstance(result, InteractionBrokerResult):
            raise ExecutionCorrelationError(
                "interaction cancellation returned invalid state"
            )
        store_result = result.store_result
        if isinstance(store_result, ScopeCancellationRejected):
            raise ExecutionCorrelationError(
                "interaction branch cancellation was rejected"
            )
        if not isinstance(
            store_result,
            (ScopeCancellationApplied, ScopeCancellationReplayed),
        ):
            raise ExecutionCorrelationError(
                "interaction cancellation returned unrelated state"
            )
        if store_result.command != command:
            raise ExecutionCorrelationError(
                "interaction cancellation returned mismatched state"
            )
        return result

    def _validate_actor(self, actor: InteractionActor) -> None:
        if actor != self._actor or actor.principal != self._actor.principal:
            raise ExecutionCorrelationError(
                "interaction actor does not match the active branch"
            )

    @staticmethod
    def _validate_scope(
        scope: InteractionExecutionScope,
        origin: ExecutionOrigin,
    ) -> None:
        if (
            scope.run_id != origin.run_id
            or scope.branch_id != origin.branch_id
            or scope.include_descendants
        ):
            raise ExecutionCorrelationError(
                "interaction scope does not match the active branch"
            )
        for scope_name, origin_name in (
            ("turn_id", "turn_id"),
            ("task_id", "task_id"),
            ("agent_id", "agent_id"),
        ):
            value = getattr(scope, scope_name)
            if value is not None and value != getattr(origin, origin_name):
                raise ExecutionCorrelationError(
                    "interaction scope does not match the active branch"
                )

    async def _ensure_branch_registered(
        self,
        origin: ExecutionOrigin,
    ) -> None:
        parent_branch_id = origin.parent_branch_id
        if parent_branch_id is None:
            return
        key = (origin.run_id, origin.branch_id)
        async with self._registration_lock:
            if key in self._registered_branches:
                return
            register_branch = getattr(self._broker, "register_branch", None)
            if register_branch is None or not callable(register_branch):
                raise ExecutionCorrelationError(
                    "interaction broker cannot register child branches"
                )
            registration = InteractionBranchRegistration(
                run_id=origin.run_id,
                branch_id=origin.branch_id,
                parent_branch_id=parent_branch_id,
                principal=origin.principal,
            )
            command = RegisterInteractionBranchCommand(
                actor=self._actor,
                registration=registration,
            )
            result = await register_branch(command)
            if not isinstance(result, InteractionBrokerResult):
                raise ExecutionCorrelationError(
                    "interaction branch registration returned invalid state"
                )
            store_result = result.store_result
            if isinstance(store_result, InteractionBranchRegistrationRejected):
                raise ExecutionCorrelationError(
                    "interaction branch registration was rejected"
                )
            if not isinstance(
                store_result,
                (
                    InteractionBranchRegistrationApplied,
                    InteractionBranchRegistrationReplayed,
                ),
            ):
                raise ExecutionCorrelationError(
                    "interaction branch registration returned unrelated state"
                )
            if (
                store_result.command != command
                or store_result.record.registration != registration
            ):
                raise ExecutionCorrelationError(
                    "interaction branch registration returned mismatched state"
                )
            self._registered_branches.add(key)


@final
class AgentExecution:
    """Own all mutable state for one logical agent invocation."""

    def __init__(
        self,
        *,
        origin: ExecutionOrigin,
        id_factory: ExecutionIdFactory,
        initial_messages: tuple[Message, ...],
        synced_message_prefix: int = 0,
        interaction_runtime: InteractionRuntime | None = None,
    ) -> None:
        if not isinstance(origin, ExecutionOrigin):
            raise TypeError("origin must be an execution origin")
        _validate_messages(initial_messages)
        initial_messages = _snapshot_messages(initial_messages)
        synced_message_prefix = _validate_synced_message_prefix(
            synced_message_prefix,
            len(initial_messages),
        )
        _assert_execution_id_factory(id_factory, "execution")
        if interaction_runtime is not None and not isinstance(
            interaction_runtime,
            AttachedInteractionRuntime | DurableInteractionRuntime,
        ):
            raise TypeError(
                "interaction_runtime must be an interaction runtime or None"
            )
        if interaction_runtime is not None:
            _validate_runtime_origin(
                interaction_runtime,
                origin,
                id_factory,
            )
        initial_entry = ExecutionLedgerEntry(
            sequence=0,
            kind=ExecutionLedgerEntryKind.INPUT,
            origin=origin,
            messages=initial_messages,
        )
        self._initial_origin = origin
        self._id_factory = id_factory
        self._interaction_runtime = interaction_runtime
        self._interaction_broker = (
            ExecutionBranchInteractionBroker(
                broker=interaction_runtime.broker,
                actor=interaction_runtime.actor,
                current_origin=lambda: self.origin,
            )
            if isinstance(interaction_runtime, AttachedInteractionRuntime)
            else None
        )
        self._lock = Lock()
        self._memory_sync_lock = Lock()
        self._provider_settlement_lock = Lock()
        self._provider_settlement_task: (
            Task[tuple[BaseException, ...]] | None
        ) = None
        self._state = AgentExecutionSnapshot(
            revision=0,
            status=AgentExecutionStatus.RUNNING,
            origin=origin,
            ledger=(initial_entry,),
            messages=initial_messages,
            pending_request=None,
            active_interaction_fingerprint=None,
            interaction_fingerprint_counts=(),
            interaction_count=0,
            memory_sync_cursor=synced_message_prefix,
            response_sync_cursor=0,
            cleanup_started=False,
        )

    @property
    def snapshot(self) -> AgentExecutionSnapshot:
        """Return the active immutable compare-and-swap snapshot."""
        return _snapshot_state(self._state)

    @property
    def revision(self) -> int:
        """Return the active compare-and-swap revision."""
        return self._state.revision

    @property
    def origin(self) -> ExecutionOrigin:
        """Return the active provider-turn origin."""
        return self._state.origin

    @property
    def initial_origin(self) -> ExecutionOrigin:
        """Return the immutable initial transport origin."""
        return self._initial_origin

    @property
    def definition(self) -> ExecutionDefinitionRef:
        """Return the immutable execution definition."""
        return self._initial_origin.definition

    @property
    def operation_id(self) -> str:
        """Return the stable operation identifier."""
        return self.definition.operation_id

    @property
    def operation_index(self) -> int:
        """Return the stable operation index."""
        return self.definition.operation_index

    @property
    def interaction_runtime(self) -> InteractionRuntime | None:
        """Return explicit attached handling or absence."""
        return self._interaction_runtime

    @property
    def interaction_broker(self) -> BranchInteractionBroker | None:
        """Return the branch-scoped broker facade or absence."""
        return self._interaction_broker

    @property
    def status(self) -> AgentExecutionStatus:
        """Return the current execution state."""
        return self._state.status

    @property
    def ledger(self) -> tuple[ExecutionLedgerEntry, ...]:
        """Return the immutable execution history."""
        return _snapshot_ledger(self._state.ledger)

    @property
    def messages(self) -> tuple[Message, ...]:
        """Return the authoritative transcript."""
        return _snapshot_messages(self._state.messages)

    @property
    def last_prompt(self) -> ModelPromptRecord | None:
        """Return the most recently dispatched prompt from the ledger."""
        prompt = next(
            (
                entry.prompt
                for entry in reversed(self._state.ledger)
                if entry.prompt is not None
            ),
            None,
        )
        return _snapshot_prompt(prompt) if prompt is not None else None

    @property
    def last_response(self) -> ModelResponse | None:
        """Return the most recently received response from the ledger."""
        return next(
            (
                entry.response
                for entry in reversed(self._state.ledger)
                if entry.response is not None
            ),
            None,
        )

    @property
    def interaction_count(self) -> int:
        """Return task-input cycles independently of domain-tool cycles."""
        return self._state.interaction_count

    @property
    def pending_request(self) -> InputRequest | None:
        """Return the sole unresolved request on this branch."""
        return self._state.pending_request

    async def record_prompt(
        self,
        prompt: ModelPromptRecord,
        *,
        expected_revision: int | None = None,
    ) -> int:
        """Record one exact model dispatch under compare-and-swap."""
        if not isinstance(prompt, ModelPromptRecord):
            raise TypeError("prompt must be a model-prompt record")
        async with self._lock:
            current = self._checked_state(expected_revision)
            _require_status(current, AgentExecutionStatus.RUNNING)
            return self._commit(
                current,
                kind=ExecutionLedgerEntryKind.MODEL_PROMPT,
                prompt=prompt,
            ).revision

    async def record_messages(
        self,
        messages: tuple[Message, ...],
        *,
        expected_revision: int | None = None,
    ) -> int:
        """Append typed ordinary model or tool messages to the transcript."""
        _validate_messages(messages)
        if not messages:
            raise ValueError("messages must not be empty")
        async with self._lock:
            current = self._checked_state(expected_revision)
            _require_status(current, AgentExecutionStatus.RUNNING)
            return self._commit(
                current,
                kind=ExecutionLedgerEntryKind.TRANSCRIPT,
                messages=messages,
                transcript=(*current.messages, *messages),
            ).revision

    async def record_response(
        self,
        response: ModelResponse,
        *,
        messages: tuple[Message, ...] = (),
        expected_revision: int | None = None,
    ) -> int:
        """Record one model response and any typed transcript messages."""
        response_in_transcript = _validate_response_messages(
            response,
            messages,
        )
        async with self._lock:
            current = self._checked_state(expected_revision)
            _require_status(current, AgentExecutionStatus.RUNNING)
            return self._commit(
                current,
                kind=ExecutionLedgerEntryKind.MODEL_RESPONSE,
                response=response,
                response_in_transcript=response_in_transcript,
                messages=messages,
                transcript=(*current.messages, *messages),
            ).revision

    async def complete_with_response(
        self,
        response: ModelResponse,
        *,
        messages: tuple[Message, ...] = (),
        expected_revision: int | None = None,
    ) -> bool:
        """Record one response and complete the invocation atomically."""
        response_in_transcript = _validate_response_messages(
            response,
            messages,
        )
        async with self._lock:
            current = self._checked_state(expected_revision)
            if current.status in _TERMINAL_STATUSES:
                return False
            _require_status(current, AgentExecutionStatus.RUNNING)
            try:
                response_state = self._commit(
                    current,
                    kind=ExecutionLedgerEntryKind.MODEL_RESPONSE,
                    response=response,
                    response_in_transcript=response_in_transcript,
                    messages=messages,
                    transcript=(*current.messages, *messages),
                )
                self._commit(
                    response_state,
                    kind=ExecutionLedgerEntryKind.COMPLETED,
                    status=AgentExecutionStatus.COMPLETED,
                    pending_request=None,
                    active_interaction_fingerprint=None,
                )
            except BaseException:
                self._state = current
                raise
            return True

    async def begin_interaction(
        self,
        semantic_fingerprint: str,
        task_input_call: TaskInputCapabilityCall,
        assistant_message: Message,
        *,
        expected_revision: int | None = None,
    ) -> int:
        """Reserve the branch's sole interaction and enforce its own limit."""
        fingerprint = _validate_fingerprint(semantic_fingerprint)
        if type(task_input_call) is not TaskInputCapabilityCall:
            raise TypeError(
                "task_input_call must be a task-input capability call"
            )
        if not isinstance(assistant_message, Message):
            raise TypeError("assistant_message must be a message")
        _validate_originating_assistant_message(
            task_input_call,
            assistant_message,
        )
        async with self._lock:
            current = self._checked_state(expected_revision)
            _require_status(current, AgentExecutionStatus.RUNNING)
            counts = dict(current.interaction_fingerprint_counts)
            count = counts.get(fingerprint, 0) + 1
            if count > MAXIMUM_EQUIVALENT_INPUT_REQUESTS:
                raise InteractionLoopLimitError(
                    "equivalent task-input request limit reached"
                )
            counts[fingerprint] = count
            return self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INTERACTION_RESERVED,
                status=AgentExecutionStatus.PREPARING_INPUT,
                semantic_fingerprint=fingerprint,
                task_input_call=task_input_call,
                interaction_assistant_message=assistant_message,
                active_interaction_fingerprint=fingerprint,
                interaction_fingerprint_counts=tuple(sorted(counts.items())),
                interaction_count=current.interaction_count + 1,
            ).revision

    async def abandon_interaction(
        self,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Release a reservation after admission fails without a request."""
        async with self._lock:
            current = self._checked_state(expected_revision)
            if (
                current.status is AgentExecutionStatus.RUNNING
                and current.active_interaction_fingerprint is None
            ):
                return False
            _require_status(current, AgentExecutionStatus.PREPARING_INPUT)
            fingerprint = cast(str, current.active_interaction_fingerprint)
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INTERACTION_ABANDONED,
                status=AgentExecutionStatus.RUNNING,
                semantic_fingerprint=fingerprint,
                active_interaction_fingerprint=None,
            )
            return True

    async def mark_interaction_pending(
        self,
        request: InputRequest,
        *,
        expected_revision: int | None = None,
    ) -> int:
        """Record authoritative broker identity before handler waiting."""
        if type(request) is not InputRequest:
            raise TypeError("request must be an input request")
        if request.state is not RequestState.PENDING:
            raise ExecutionCorrelationError(
                "interaction request must be authoritatively pending"
            )
        async with self._lock:
            current = self._checked_state(expected_revision)
            _require_status(current, AgentExecutionStatus.PREPARING_INPUT)
            if request.origin != current.origin:
                raise ExecutionCorrelationError(
                    "interaction origin does not match execution"
                )
            task_input_call, _ = _active_task_input(current.ledger)
            _validate_request_matches_task_input_call(
                request,
                task_input_call,
            )
            return self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INTERACTION_PENDING,
                status=AgentExecutionStatus.WAITING_FOR_INPUT,
                request=request,
                pending_request=request,
            ).revision

    async def record_interaction_result(
        self,
        request: InputRequest,
        result: InputModelResult,
        messages: tuple[Message, ...],
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Commit one correlated result and reject duplicate continuation."""
        if type(request) is not InputRequest:
            raise TypeError("request must be an input request")
        if not _is_input_model_result(result):
            raise TypeError("result must be an input model result")
        _validate_messages(messages)
        if not messages:
            raise ExecutionCorrelationError(
                "interaction result requires a correlated model message"
            )
        async with self._lock:
            current = self._checked_state(expected_revision)
            durable_staged = (
                current.status is AgentExecutionStatus.INPUT_REQUIRED
                and current.pending_request is not None
                and current.pending_request.state is RequestState.CREATED
            )
            if (
                current.status is not AgentExecutionStatus.WAITING_FOR_INPUT
                and not durable_staged
            ):
                replay = _find_result_replay(
                    current.ledger,
                    request,
                    result,
                    messages,
                )
                if replay:
                    return False
                _require_status(
                    current,
                    AgentExecutionStatus.WAITING_FOR_INPUT,
                )
            pending = cast(InputRequest, current.pending_request)
            _validate_terminal_request(pending, request)
            outcome = project_resolution_to_model(
                request,
                containing_run_exists=True,
            )
            if (
                type(outcome) is not ResumeInputContinuation
                or outcome.result != result
            ):
                raise ExecutionCorrelationError(
                    "model result does not match terminal request"
                )
            task_input_call, assistant_message = _active_task_input(
                current.ledger
            )
            _validate_correlated_result_messages(
                task_input_call,
                assistant_message,
                result,
                messages,
            )
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INTERACTION_RESULT,
                status=AgentExecutionStatus.RESUMING,
                messages=messages,
                task_input_call=task_input_call,
                request=request,
                result=result,
                transcript=(*current.messages, *messages),
                pending_request=None,
                active_interaction_fingerprint=None,
            )
            return True

    async def record_interaction_termination(
        self,
        request: InputRequest,
        outcome: TerminateInputContinuation,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Commit one exact terminal request and terminate its execution."""
        if type(request) is not InputRequest:
            raise TypeError("request must be an input request")
        if type(outcome) is not TerminateInputContinuation:
            raise TypeError("outcome must terminate an input continuation")
        async with self._lock:
            current = self._checked_state(expected_revision)
            if current.status is not AgentExecutionStatus.WAITING_FOR_INPUT:
                replay = _find_termination_replay(
                    current.ledger,
                    request,
                    outcome,
                )
                if replay:
                    return False
                _require_status(
                    current,
                    AgentExecutionStatus.WAITING_FOR_INPUT,
                )
            pending = cast(InputRequest, current.pending_request)
            _validate_terminal_request(pending, request)
            projected = project_resolution_to_model(
                request,
                containing_run_exists=True,
            )
            if projected != outcome:
                raise ExecutionCorrelationError(
                    "termination does not match terminal request"
                )
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INTERACTION_TERMINATED,
                status=AgentExecutionStatus.CANCELLED,
                request=request,
                termination_outcome=outcome,
                pending_request=None,
                active_interaction_fingerprint=None,
            )
            return True

    async def mark_input_required(
        self,
        result: InputRequiredResult,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """End only the current segment while retaining the logical run."""
        if type(result) is not InputRequiredResult:
            raise TypeError("result must be an input-required result")
        async with self._lock:
            current = self._checked_state(expected_revision)
            if current.status is AgentExecutionStatus.INPUT_REQUIRED:
                if _find_input_required_replay(current.ledger, result):
                    return False
            _require_status(
                current,
                AgentExecutionStatus.WAITING_FOR_INPUT,
            )
            request = current.pending_request
            if request is None or (
                request.request_id != result.request_id
                or request.continuation_id != result.continuation_id
            ):
                raise ExecutionCorrelationError(
                    "input-required result is not correlated"
                )
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INPUT_REQUIRED,
                status=AgentExecutionStatus.INPUT_REQUIRED,
                request=request,
                input_required=result,
                pending_request=request,
            )
            return True

    async def stage_durable_input_required(
        self,
        request: InputRequest,
        result: InputRequiredResult,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """End a segment around one uncommitted durable request."""
        if type(request) is not InputRequest:
            raise TypeError("request must be an input request")
        if request.state is not RequestState.CREATED:
            raise ExecutionCorrelationError(
                "deferred durable request must remain uncommitted"
            )
        if type(result) is not InputRequiredResult:
            raise TypeError("result must be an input-required result")
        if not result.detached_resumption_available:
            raise ExecutionCorrelationError(
                "durable input requires detached resumption"
            )
        async with self._lock:
            current = self._checked_state(expected_revision)
            if current.status is AgentExecutionStatus.INPUT_REQUIRED:
                if _find_input_required_replay(current.ledger, result):
                    return False
            _require_status(
                current,
                AgentExecutionStatus.PREPARING_INPUT,
            )
            if request.origin != current.origin:
                raise ExecutionCorrelationError(
                    "interaction origin does not match execution"
                )
            task_input_call, _ = _active_task_input(current.ledger)
            _validate_request_matches_task_input_call(
                request,
                task_input_call,
            )
            if (
                request.request_id != result.request_id
                or request.continuation_id != result.continuation_id
            ):
                raise ExecutionCorrelationError(
                    "input-required result is not correlated"
                )
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.INPUT_REQUIRED,
                status=AgentExecutionStatus.INPUT_REQUIRED,
                request=request,
                input_required=result,
                pending_request=request,
            )
            return True

    async def advance_model_turn(
        self,
        *,
        new_stream_session: bool = False,
        expected_revision: int | None = None,
    ) -> ExecutionOrigin:
        """Mint a new provider turn while preserving logical identity."""
        if not isinstance(new_stream_session, bool):
            raise TypeError("new_stream_session must be a boolean")
        async with self._lock:
            captured = self._checked_state(expected_revision)
            if captured.status not in {
                AgentExecutionStatus.RUNNING,
                AgentExecutionStatus.RESUMING,
            }:
                raise ExecutionStateError(
                    f"cannot advance from {captured.status.value}"
                )
            captured_revision = captured.revision
        turn_id = await self._id_factory.new_turn_id()
        model_call_id = await self._id_factory.new_model_call_id()
        stream_session_id = captured.origin.stream_session_id
        if new_stream_session:
            stream_session_id = await self._id_factory.new_stream_session_id()
        async with self._lock:
            current = self._state
            if current.revision != captured_revision:
                raise ExecutionRevisionError(
                    "execution changed while identities were minted"
                )
            previous = current.origin
            origin = ExecutionOrigin(
                run_id=previous.run_id,
                turn_id=turn_id,
                task_id=previous.task_id,
                agent_id=previous.agent_id,
                branch_id=previous.branch_id,
                parent_branch_id=previous.parent_branch_id,
                model_call_id=model_call_id,
                stream_session_id=stream_session_id,
                definition=previous.definition,
                principal=previous.principal,
            )
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.MODEL_TURN,
                status=AgentExecutionStatus.RUNNING,
                origin=origin,
            )
            return origin

    async def sync_memory(self, sink: ExecutionMemorySink) -> None:
        """Synchronize memory through one explicitly idempotent sink."""
        _assert_async_methods(
            sink,
            ("append_execution_memory_entry",),
            "execution.memory_sink",
        )
        async with self._memory_sync_lock:
            await self._sync_transcript_messages(sink)
            await self._sync_model_responses(sink)

    async def _sync_transcript_messages(
        self,
        sink: ExecutionMemorySink,
    ) -> None:
        while True:
            async with self._lock:
                current = self._state
                cursor = current.memory_sync_cursor
                if cursor == len(current.messages):
                    return
                entry = _memory_entry_for_message(current, cursor)
            await sink.append_execution_memory_entry(entry)
            async with self._lock:
                current = self._state
                if current.memory_sync_cursor != cursor:
                    raise ExecutionRevisionError(
                        "memory synchronization cursor changed"
                    )
                self._state = replace(
                    current,
                    revision=_next_revision(current),
                    memory_sync_cursor=cursor + 1,
                )

    async def _sync_model_responses(
        self,
        sink: ExecutionMemorySink,
    ) -> None:
        while True:
            async with self._lock:
                current = self._state
                cursor = current.response_sync_cursor
                response_index = next(
                    (
                        index
                        for index in range(cursor, len(current.ledger))
                        if current.ledger[index].response is not None
                        and not current.ledger[index].response_in_transcript
                    ),
                    None,
                )
                if response_index is None:
                    if cursor != len(current.ledger):
                        self._state = replace(
                            current,
                            revision=_next_revision(current),
                            response_sync_cursor=len(current.ledger),
                        )
                    return
                response = current.ledger[response_index].response
                assert response is not None
                ledger_entry = current.ledger[response_index]
                entry = ExecutionMemoryEntry(
                    origin=ledger_entry.origin,
                    ledger_sequence=ledger_entry.sequence,
                    component=ExecutionMemoryComponent.RESPONSE,
                    component_index=0,
                    message=Message(
                        role=MessageRole.ASSISTANT,
                        content=response,
                    ),
                )
            await sink.append_execution_memory_entry(entry)
            async with self._lock:
                current = self._state
                if current.response_sync_cursor != cursor:
                    raise ExecutionRevisionError(
                        "response synchronization cursor changed"
                    )
                self._state = replace(
                    current,
                    revision=_next_revision(current),
                    response_sync_cursor=response_index + 1,
                )

    async def complete(
        self,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Complete this invocation exactly once from a running state."""
        return await self._finish(
            AgentExecutionStatus.COMPLETED,
            ExecutionLedgerEntryKind.COMPLETED,
            expected_revision=expected_revision,
            running_only=True,
        )

    async def cancel(
        self,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Cancel this invocation exactly once."""
        return await self._finish(
            AgentExecutionStatus.CANCELLED,
            ExecutionLedgerEntryKind.CANCELLED,
            expected_revision=expected_revision,
            running_only=False,
        )

    async def fail(
        self,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Fail this invocation exactly once."""
        return await self._finish(
            AgentExecutionStatus.ERRORED,
            ExecutionLedgerEntryKind.ERRORED,
            expected_revision=expected_revision,
            running_only=False,
        )

    async def settle_provider_exit(
        self,
        *,
        cancelled: bool,
    ) -> tuple[BaseException, ...]:
        """Join one provider-exit settlement without masking failures."""
        if not isinstance(cancelled, bool):
            raise TypeError("cancelled must be a boolean")

        async with self._provider_settlement_lock:
            task = self._provider_settlement_task
            if task is None and self._state.status in _TERMINAL_STATUSES:
                return ()
            if task is None:
                task = create_task(
                    self._settle_provider_exit_once(cancelled=cancelled)
                )
                self._provider_settlement_task = task
                task.add_done_callback(self._provider_settlement_done)

        return await shield(task)

    async def _settle_provider_exit_once(
        self,
        *,
        cancelled: bool,
    ) -> tuple[BaseException, ...]:
        """Run the sole dynamic transition and terminal fallback."""
        secondary_failures: list[BaseException] = []
        try:
            if cancelled:
                await self.cancel()
            else:
                await self.fail()
        except BaseException as error:
            secondary_failures.append(error)

        async with self._lock:
            current = self._state
            if current.status not in _TERMINAL_STATUSES:
                try:
                    self._commit(
                        current,
                        kind=(
                            ExecutionLedgerEntryKind.CANCELLED
                            if cancelled
                            else ExecutionLedgerEntryKind.ERRORED
                        ),
                        status=(
                            AgentExecutionStatus.CANCELLED
                            if cancelled
                            else AgentExecutionStatus.ERRORED
                        ),
                        pending_request=None,
                        active_interaction_fingerprint=None,
                    )
                except BaseException as error:
                    secondary_failures.append(error)
        return tuple(secondary_failures)

    def _provider_settlement_done(
        self,
        task: Task[tuple[BaseException, ...]],
    ) -> None:
        """Release the completed coalescing task without awaiting I/O."""
        if self._provider_settlement_task is task:
            self._provider_settlement_task = None

    async def claim_cleanup(
        self,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Return whether this caller owns terminal invocation cleanup."""
        async with self._lock:
            current = self._checked_state(expected_revision)
            if current.cleanup_started:
                return False
            if current.status not in _TERMINAL_STATUSES:
                raise ExecutionStateError(
                    "cleanup can be claimed only after termination"
                )
            self._commit(
                current,
                kind=ExecutionLedgerEntryKind.CLEANUP_CLAIMED,
                cleanup_started=True,
            )
            return True

    async def _finish(
        self,
        status: AgentExecutionStatus,
        kind: ExecutionLedgerEntryKind,
        *,
        expected_revision: int | None,
        running_only: bool,
    ) -> bool:
        async with self._lock:
            current = self._checked_state(expected_revision)
            if current.status in _TERMINAL_STATUSES:
                return False
            if running_only:
                _require_status(current, AgentExecutionStatus.RUNNING)
            self._commit(
                current,
                kind=kind,
                status=status,
                pending_request=None,
                active_interaction_fingerprint=None,
            )
            return True

    def _checked_state(
        self,
        expected_revision: int | None,
    ) -> AgentExecutionSnapshot:
        current = self._state
        if expected_revision is None:
            return current
        if (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 0
        ):
            raise TypeError("expected_revision must be non-negative")
        if current.revision != expected_revision:
            raise ExecutionRevisionError(
                f"expected revision {expected_revision}, "
                f"found {current.revision}"
            )
        return current

    def _commit(
        self,
        current: AgentExecutionSnapshot,
        *,
        kind: ExecutionLedgerEntryKind,
        status: AgentExecutionStatus | None = None,
        origin: ExecutionOrigin | None = None,
        messages: tuple[Message, ...] = (),
        prompt: ModelPromptRecord | None = None,
        response: ModelResponse | None = None,
        response_in_transcript: bool = False,
        semantic_fingerprint: str | None = None,
        task_input_call: TaskInputCapabilityCall | None = None,
        interaction_assistant_message: Message | None = None,
        request: InputRequest | None = None,
        result: InputModelResult | None = None,
        input_required: InputRequiredResult | None = None,
        termination_outcome: TerminateInputContinuation | None = None,
        transcript: tuple[Message, ...] | None = None,
        pending_request: InputRequest | None | object = _UNCHANGED,
        active_interaction_fingerprint: str | None | object = _UNCHANGED,
        interaction_fingerprint_counts: (
            tuple[tuple[str, int], ...] | None
        ) = None,
        interaction_count: int | None = None,
        cleanup_started: bool | None = None,
    ) -> AgentExecutionSnapshot:
        active_origin = current.origin if origin is None else origin
        if (
            active_origin.run_id != self._initial_origin.run_id
            or active_origin.agent_id != self._initial_origin.agent_id
            or active_origin.branch_id != self._initial_origin.branch_id
            or active_origin.parent_branch_id
            != self._initial_origin.parent_branch_id
            or active_origin.task_id != self._initial_origin.task_id
            or active_origin.definition != self._initial_origin.definition
            or active_origin.principal != self._initial_origin.principal
        ):
            raise ExecutionCorrelationError(
                "execution identity changed across a state transition"
            )
        entry = ExecutionLedgerEntry(
            sequence=len(current.ledger),
            kind=kind,
            origin=active_origin,
            messages=messages,
            prompt=prompt,
            response=response,
            response_in_transcript=response_in_transcript,
            semantic_fingerprint=semantic_fingerprint,
            task_input_call=task_input_call,
            interaction_assistant_message=interaction_assistant_message,
            request=request,
            result=result,
            input_required=input_required,
            termination_outcome=termination_outcome,
        )
        next_pending = (
            current.pending_request
            if pending_request is _UNCHANGED
            else cast(InputRequest | None, pending_request)
        )
        next_fingerprint = (
            current.active_interaction_fingerprint
            if active_interaction_fingerprint is _UNCHANGED
            else cast(str | None, active_interaction_fingerprint)
        )
        next_state = replace(
            current,
            revision=_next_revision(current),
            status=current.status if status is None else status,
            origin=active_origin,
            ledger=(*current.ledger, entry),
            messages=current.messages if transcript is None else transcript,
            pending_request=next_pending,
            active_interaction_fingerprint=next_fingerprint,
            interaction_fingerprint_counts=(
                current.interaction_fingerprint_counts
                if interaction_fingerprint_counts is None
                else interaction_fingerprint_counts
            ),
            interaction_count=(
                current.interaction_count
                if interaction_count is None
                else interaction_count
            ),
            cleanup_started=(
                current.cleanup_started
                if cleanup_started is None
                else cleanup_started
            ),
        )
        self._state = next_state
        return next_state


_TERMINAL_STATUSES = frozenset(
    {
        AgentExecutionStatus.COMPLETED,
        AgentExecutionStatus.CANCELLED,
        AgentExecutionStatus.ERRORED,
    }
)


async def create_agent_execution(
    *,
    definition: ExecutionDefinitionRef,
    agent_id: AgentId,
    principal: PrincipalScope,
    initial_messages: tuple[Message, ...],
    synced_message_prefix: int = 0,
    id_factory: ExecutionIdFactory | None = None,
    interaction_runtime: InteractionRuntime | None = None,
) -> AgentExecution:
    """Create one invocation with genuinely minted execution identities."""
    if not isinstance(definition, ExecutionDefinitionRef):
        raise TypeError("definition must be an execution definition reference")
    if not isinstance(principal, PrincipalScope):
        raise TypeError("principal must be a principal scope")
    _validate_messages(initial_messages)
    synced_message_prefix = _validate_synced_message_prefix(
        synced_message_prefix,
        len(initial_messages),
    )
    if interaction_runtime is not None and not isinstance(
        interaction_runtime,
        AttachedInteractionRuntime | DurableInteractionRuntime,
    ):
        raise TypeError("interaction_runtime must be an interaction runtime")
    if interaction_runtime is not None:
        if interaction_runtime.actor.principal != principal:
            raise ExecutionCorrelationError(
                "interaction actor does not match execution principal"
            )
        if (
            id_factory is not None
            and id_factory is not interaction_runtime.id_factory
        ):
            raise ExecutionCorrelationError(
                "interaction runtime and execution need one identity factory"
            )
    factory = (
        id_factory
        or (
            interaction_runtime.id_factory
            if interaction_runtime is not None
            else None
        )
        or UuidExecutionIdFactory()
    )
    _assert_execution_id_factory(factory, "execution")
    branch_id = (
        interaction_runtime.branch_id
        if interaction_runtime is not None
        and interaction_runtime.branch_id is not None
        else await factory.new_branch_id()
    )
    origin = ExecutionOrigin(
        run_id=(
            interaction_runtime.run_id
            if isinstance(
                interaction_runtime,
                DurableInteractionRuntime,
            )
            and interaction_runtime.run_id is not None
            else await factory.new_run_id()
        ),
        turn_id=await factory.new_turn_id(),
        task_id=(
            interaction_runtime.task_id
            if interaction_runtime is not None
            and interaction_runtime.task_id is not None
            else await factory.new_task_id()
        ),
        agent_id=agent_id,
        branch_id=branch_id,
        parent_branch_id=(
            interaction_runtime.parent_branch_id
            if interaction_runtime is not None
            else None
        ),
        model_call_id=await factory.new_model_call_id(),
        stream_session_id=await factory.new_stream_session_id(),
        definition=definition,
        principal=principal,
    )
    return AgentExecution(
        origin=origin,
        id_factory=factory,
        initial_messages=initial_messages,
        synced_message_prefix=synced_message_prefix,
        interaction_runtime=interaction_runtime,
    )


def _freeze_prompt_input(value: Input) -> ExecutionPromptInput:
    if isinstance(value, str):
        return value
    if isinstance(value, Message):
        return _snapshot_messages((value,))[0]
    if not isinstance(value, list):
        raise TypeError("prompt input must be typed model input")
    if all(isinstance(item, str) for item in value):
        return tuple(cast(list[str], value))
    if all(isinstance(item, Message) for item in value):
        return _snapshot_messages(tuple(cast(list[Message], value)))
    raise TypeError("prompt input list must contain one supported type")


def _validate_messages(messages: object) -> tuple[Message, ...]:
    if not isinstance(messages, tuple) or not all(
        isinstance(message, Message) for message in messages
    ):
        raise TypeError("messages must be a tuple of messages")
    return messages


def _validate_response_messages(
    response: object,
    messages: tuple[Message, ...],
) -> bool:
    """Validate one immutable response and its exact transcript material."""
    if not isinstance(response, str):
        raise TypeError("response must be immutable text")
    _validate_messages(messages)
    if messages and messages != (
        Message(role=MessageRole.ASSISTANT, content=response),
    ):
        raise ExecutionCorrelationError(
            "transcript response must exactly materialize response text"
        )
    return bool(messages)


def _validate_synced_message_prefix(value: object, total: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("synced_message_prefix must be an integer")
    if not 0 <= value <= total:
        raise ValueError(
            "synced_message_prefix must be within the initial transcript"
        )
    return value


def _snapshot_messages(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    return tuple(
        cast(Message, _snapshot_value(message)) for message in messages
    )


def snapshot_execution_messages(
    messages: tuple[Message, ...],
) -> tuple[Message, ...]:
    """Return a detached recursive snapshot of execution messages."""
    _validate_messages(messages)
    return _snapshot_messages(messages)


def _memory_entry_for_message(
    snapshot: AgentExecutionSnapshot,
    cursor: int,
) -> ExecutionMemoryEntry:
    message_cursor = 0
    for ledger_entry in snapshot.ledger:
        for component_index, message in enumerate(ledger_entry.messages):
            if message_cursor == cursor:
                return ExecutionMemoryEntry(
                    origin=ledger_entry.origin,
                    ledger_sequence=ledger_entry.sequence,
                    component=ExecutionMemoryComponent.MESSAGE,
                    component_index=component_index,
                    message=message,
                )
            message_cursor += 1
    raise ExecutionStateError(
        "memory synchronization cursor has no ledger message"
    )


def _snapshot_value(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return {
            _snapshot_value(key): _snapshot_value(item)
            for key, item in value.items()
        }
    if isinstance(value, Mapping):
        return {
            _snapshot_value(key): _snapshot_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_snapshot_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_value(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            item.name: _snapshot_value(getattr(value, item.name))
            for item in fields(value)
            if item.init
        }
        return replace(value, **updates)
    return deepcopy(value)


def _snapshot_prompt(prompt: ModelPromptRecord) -> ModelPromptRecord:
    prompt_input = cast(
        Input,
        (
            list(prompt.input)
            if isinstance(prompt.input, tuple)
            else prompt.input
        ),
    )
    return ModelPromptRecord(
        input=prompt_input,
        instructions=prompt.instructions,
        system_prompt=prompt.system_prompt,
        developer_prompt=prompt.developer_prompt,
    )


def _snapshot_ledger(
    ledger: tuple[ExecutionLedgerEntry, ...],
) -> tuple[ExecutionLedgerEntry, ...]:
    return tuple(
        replace(
            entry,
            messages=_snapshot_messages(entry.messages),
            prompt=(
                _snapshot_prompt(entry.prompt)
                if entry.prompt is not None
                else None
            ),
        )
        for entry in ledger
    )


def _snapshot_state(state: AgentExecutionSnapshot) -> AgentExecutionSnapshot:
    return replace(
        state,
        ledger=_snapshot_ledger(state.ledger),
        messages=_snapshot_messages(state.messages),
    )


def _validate_fingerprint(value: object) -> str:
    return validate_opaque_id(
        value,
        "interaction.semantic_fingerprint",
        maximum_characters=128,
        maximum_bytes=512,
    )


def _is_input_model_result(value: object) -> TypeGuard[InputModelResult]:
    return type(value) in _INPUT_MODEL_RESULT_TYPES


def _entry_payload_is_legal(
    kind: ExecutionLedgerEntryKind,
    populated: tuple[
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
        bool,
    ],
) -> bool:
    (
        messages,
        prompt,
        response,
        response_in_transcript,
        fingerprint,
        task_input_call,
        interaction_assistant,
        request,
        result,
        required,
        termination,
    ) = populated
    match kind:
        case ExecutionLedgerEntryKind.INPUT:
            return not any(
                (
                    prompt,
                    response,
                    response_in_transcript,
                    fingerprint,
                    task_input_call,
                    interaction_assistant,
                    request,
                    result,
                    required,
                    termination,
                )
            )
        case ExecutionLedgerEntryKind.TRANSCRIPT:
            return messages and not any(
                (
                    prompt,
                    response,
                    response_in_transcript,
                    fingerprint,
                    task_input_call,
                    interaction_assistant,
                    request,
                    result,
                    required,
                    termination,
                )
            )
        case ExecutionLedgerEntryKind.MODEL_PROMPT:
            return prompt and not any(
                (
                    messages,
                    response,
                    response_in_transcript,
                    fingerprint,
                    task_input_call,
                    interaction_assistant,
                    request,
                    result,
                    required,
                    termination,
                )
            )
        case ExecutionLedgerEntryKind.MODEL_RESPONSE:
            return response and not any(
                (
                    prompt,
                    fingerprint,
                    task_input_call,
                    interaction_assistant,
                    request,
                    result,
                    required,
                    termination,
                )
            )
        case ExecutionLedgerEntryKind.INTERACTION_RESERVED:
            return (
                fingerprint
                and task_input_call
                and interaction_assistant
                and not any(
                    (
                        messages,
                        prompt,
                        response,
                        response_in_transcript,
                        request,
                        result,
                        required,
                        termination,
                    )
                )
            )
        case ExecutionLedgerEntryKind.INTERACTION_ABANDONED:
            return fingerprint and not any(
                (
                    messages,
                    prompt,
                    response,
                    response_in_transcript,
                    interaction_assistant,
                    task_input_call,
                    request,
                    result,
                    required,
                    termination,
                )
            )
        case ExecutionLedgerEntryKind.INTERACTION_PENDING:
            return request and not any(
                (
                    messages,
                    prompt,
                    response,
                    response_in_transcript,
                    fingerprint,
                    task_input_call,
                    interaction_assistant,
                    result,
                    required,
                    termination,
                )
            )
        case ExecutionLedgerEntryKind.INTERACTION_RESULT:
            return (
                request
                and result
                and messages
                and task_input_call
                and not any(
                    (
                        prompt,
                        response,
                        response_in_transcript,
                        fingerprint,
                        interaction_assistant,
                        required,
                        termination,
                    )
                )
            )
        case ExecutionLedgerEntryKind.INPUT_REQUIRED:
            return (
                request
                and required
                and not any(
                    (
                        messages,
                        prompt,
                        response,
                        response_in_transcript,
                        fingerprint,
                        task_input_call,
                        interaction_assistant,
                        result,
                        termination,
                    )
                )
            )
        case ExecutionLedgerEntryKind.INTERACTION_TERMINATED:
            return (
                request
                and termination
                and not any(
                    (
                        messages,
                        prompt,
                        response,
                        response_in_transcript,
                        fingerprint,
                        task_input_call,
                        interaction_assistant,
                        result,
                        required,
                    )
                )
            )
        case _:
            return not any(populated)


def _requests_share_identity(left: InputRequest, right: InputRequest) -> bool:
    return (
        left.request_id == right.request_id
        and left.continuation_id == right.continuation_id
        and left.origin == right.origin
    )


def _same_request_contract(left: InputRequest, right: InputRequest) -> bool:
    return (
        _requests_share_identity(left, right)
        and left.mode is right.mode
        and left.reason == right.reason
        and left.questions == right.questions
        and left.created_at == right.created_at
        and left.continuation_ttl_seconds == right.continuation_ttl_seconds
        and left.advisory_wait_seconds == right.advisory_wait_seconds
        and left.interaction_class is right.interaction_class
    )


def _validate_request_matches_task_input_call(
    request: InputRequest,
    call: TaskInputCapabilityCall,
) -> None:
    if (
        request.mode is not call.mode
        or request.reason != call.reason
        or request.questions != call.questions
    ):
        raise ExecutionCorrelationError(
            "interaction request changed its originating task-input contract"
        )


def _validate_broker_request_result(
    request: InteractionBrokerRequest,
    result: object,
) -> None:
    if type(result) is not InteractionRequestResult:
        raise ExecutionCorrelationError(
            "interaction request returned invalid state"
        )
    create_result = result.create_result
    if not isinstance(
        create_result,
        (CreateInteractionApplied, CreateInteractionRejected),
    ):
        raise ExecutionCorrelationError(
            "interaction request returned unrelated state"
        )
    _validate_broker_create_command(request, create_result.command)
    if isinstance(create_result, CreateInteractionRejected):
        if result.delivery is not None:
            raise ExecutionCorrelationError(
                "rejected interaction admission returned a delivery"
            )
        return

    created_request = create_result.command.request
    admitted_request = create_result.record.request
    if not _same_request_contract(created_request, admitted_request):
        raise ExecutionCorrelationError(
            "interaction admission changed its created request"
        )
    delivery = result.delivery
    if delivery is None:
        raise ExecutionCorrelationError(
            "interaction admission omitted its delivery"
        )
    if delivery.correlation != create_result.record.correlation:
        raise ExecutionCorrelationError(
            "interaction delivery changed admission correlation"
        )
    if not _same_request_contract(
        admitted_request,
        delivery.record.request,
    ):
        raise ExecutionCorrelationError(
            "interaction delivery changed admission identity"
        )


def _validate_broker_create_command(
    request: InteractionBrokerRequest,
    command: CreateInteractionCommand,
) -> None:
    admitted = command.request
    expected = replace(
        admitted,
        origin=request.origin,
        mode=request.mode,
        reason=request.reason,
        questions=request.questions,
        continuation_ttl_seconds=request.continuation_ttl_seconds,
        advisory_wait_seconds=request.advisory_wait_seconds,
    )
    if command.actor != request.actor or not _same_request_contract(
        admitted,
        expected,
    ):
        raise ExecutionCorrelationError(
            "interaction admission changed its requested contract"
        )


def _validate_terminal_request(
    pending: InputRequest,
    request: InputRequest,
) -> None:
    if not _same_request_contract(pending, request):
        raise ExecutionCorrelationError(
            "terminal request changed its immutable contract"
        )
    if (
        request.state in {RequestState.CREATED, RequestState.PENDING}
        or request.resolution is None
    ):
        raise ExecutionCorrelationError("interaction request is not terminal")


def _active_task_input(
    ledger: tuple[ExecutionLedgerEntry, ...],
) -> tuple[TaskInputCapabilityCall, Message]:
    for entry in reversed(ledger):
        if entry.kind is not ExecutionLedgerEntryKind.INTERACTION_RESERVED:
            continue
        call = entry.task_input_call
        assistant_message = entry.interaction_assistant_message
        if call is not None and assistant_message is not None:
            return call, assistant_message
    raise ExecutionStateError(
        "pending interaction has no originating task-input call message"
    )


def _validate_originating_assistant_message(
    call: TaskInputCapabilityCall,
    assistant_message: Message,
) -> None:
    expected_assistant = Message(
        role=MessageRole.ASSISTANT,
        content=assistant_message.content,
        tool_calls=[
            MessageToolCall(
                id=str(call.call_id),
                name=call.provider_name,
                arguments=normalize_tool_arguments(call.arguments),
            )
        ],
    )
    if assistant_message != expected_assistant:
        raise ExecutionCorrelationError(
            "interaction reservation needs its exact originating call message"
        )


def _validate_correlated_result_messages(
    call: TaskInputCapabilityCall,
    assistant_message: Message,
    result: InputModelResult,
    messages: tuple[Message, ...],
) -> None:
    correlated = CorrelatedCapabilityResult(
        call_id=call.call_id,
        canonical_name=call.canonical_name,
        provider_name=call.provider_name,
        payload=cast(Mapping[str, Any], encode_input_model_result(result)),
    )
    expected_result = correlated.tool_result_message(call)
    if messages != (assistant_message, expected_result):
        raise ExecutionCorrelationError(
            "interaction result needs its exact ordered assistant/tool pair"
        )


def _find_result_replay(
    ledger: tuple[ExecutionLedgerEntry, ...],
    request: InputRequest,
    result: InputModelResult,
    messages: tuple[Message, ...],
) -> bool:
    for entry in reversed(ledger):
        if entry.kind is not ExecutionLedgerEntryKind.INTERACTION_RESULT:
            continue
        assert entry.request is not None
        if not _requests_share_identity(entry.request, request):
            continue
        if (
            entry.request == request
            and entry.result == result
            and entry.messages == messages
        ):
            return True
        raise ExecutionCorrelationError(
            "interaction replay conflicts with the committed result"
        )
    return False


def _find_termination_replay(
    ledger: tuple[ExecutionLedgerEntry, ...],
    request: InputRequest,
    outcome: TerminateInputContinuation,
) -> bool:
    for entry in reversed(ledger):
        if entry.kind is not ExecutionLedgerEntryKind.INTERACTION_TERMINATED:
            continue
        assert entry.request is not None
        if not _requests_share_identity(entry.request, request):
            continue
        if entry.request == request and entry.termination_outcome == outcome:
            return True
        raise ExecutionCorrelationError(
            "interaction termination conflicts with the committed outcome"
        )
    return False


def _find_input_required_replay(
    ledger: tuple[ExecutionLedgerEntry, ...],
    result: InputRequiredResult,
) -> bool:
    return any(
        entry.kind is ExecutionLedgerEntryKind.INPUT_REQUIRED
        and entry.input_required == result
        for entry in ledger
    )


def _require_status(
    state: AgentExecutionSnapshot,
    expected: AgentExecutionStatus,
) -> None:
    if state.status is not expected:
        raise ExecutionStateError(
            f"expected {expected.value}, found {state.status.value}"
        )


def _next_revision(state: AgentExecutionSnapshot) -> int:
    if state.revision >= MAX_STATE_REVISION:
        raise ExecutionRevisionError("execution revision is exhausted")
    return state.revision + 1


def _validate_runtime_origin(
    runtime: InteractionRuntime,
    origin: ExecutionOrigin,
    id_factory: ExecutionIdFactory,
) -> None:
    if runtime.actor.principal != origin.principal:
        raise ExecutionCorrelationError(
            "interaction actor does not match execution principal"
        )
    if runtime.id_factory is not id_factory:
        raise ExecutionCorrelationError(
            "interaction runtime and execution need one identity factory"
        )
    if (
        isinstance(runtime, DurableInteractionRuntime)
        and runtime.run_id is not None
        and runtime.run_id != origin.run_id
    ):
        raise ExecutionCorrelationError(
            "durable run does not match execution origin"
        )
    if runtime.task_id is not None and runtime.task_id != origin.task_id:
        raise ExecutionCorrelationError(
            "interaction task does not match execution origin"
        )
    if runtime.branch_id is not None and runtime.branch_id != origin.branch_id:
        raise ExecutionCorrelationError(
            "interaction branch does not match execution origin"
        )
    if runtime.parent_branch_id != origin.parent_branch_id:
        raise ExecutionCorrelationError(
            "interaction parent branch does not match execution origin"
        )


def _assert_execution_id_factory(value: object, path: str) -> None:
    _assert_async_methods(
        value,
        (
            "new_run_id",
            "new_turn_id",
            "new_task_id",
            "new_model_call_id",
            "new_branch_id",
            "new_stream_session_id",
        ),
        f"{path}.id_factory",
    )


def _assert_async_methods(
    value: object,
    method_names: tuple[str, ...],
    path: str,
) -> None:
    for method_name in method_names:
        method = getattr(value, method_name, None)
        if not callable(method) or not iscoroutinefunction(
            cast(Callable[..., object], method)
        ):
            raise TypeError(f"{path}.{method_name} must be async")


def _assert_async_callable(value: object, path: str) -> None:
    if callable(value) and iscoroutinefunction(
        cast(Callable[..., object], value)
    ):
        return
    bound_call = getattr(value, "__call__", None)
    if not callable(bound_call) or not iscoroutinefunction(
        cast(Callable[..., object], bound_call)
    ):
        raise TypeError(f"{path} must be async")
