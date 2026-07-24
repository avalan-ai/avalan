from ....cli import CommandAbortException
from ....entities import (
    Input,
    Message,
    MessageRole,
    MessageToolCall,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
    ToolDescriptor,
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
    normalize_tool_arguments,
)
from ....event import Event, EventObservabilityPayload, EventType
from ....event.manager import EventManager
from ....interaction.broker import (
    InteractionBrokerRequest,
    InteractionRequestResult,
)
from ....interaction.continuation import (
    derive_continuation_dispatch_id,
    derive_provider_idempotency_key,
)
from ....interaction.durable import DurableInteractionSuspension
from ....interaction.entities import (
    RESERVED_INPUT_CAPABILITY_NAME,
    AnswerProvenance,
    ContinuationId,
    InputModelResult,
    InputRequest,
    InputRequiredResult,
    RequestState,
    ResumeInputContinuation,
    TerminateInputContinuation,
)
from ....interaction.handler import (
    InputHandlerContext,
    _InputHandlerOutcome,
)
from ....interaction.state import project_resolution_to_model
from ....interaction.store import (
    CreateInteractionApplied,
    InteractionExecutionScope,
    TerminalizeInteractionScopeCommand,
)
from ....model.call import ModelCallContext
from ....model.capability import (
    CapabilityBatchAccepted,
    CapabilityBatchRejected,
    CapabilityBatchRejectionCode,
    ModelCapabilityCatalog,
    ModelCapabilityValidationError,
    ProviderCapabilityCall,
    TaskInputCapabilityCall,
)
from ....model.response.parsers.tool import ToolCallResponseParser
from ....model.response.text import TextGenerationResponse
from ....model.stream import (
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID,
    LOCAL_STRUCTURED_OUTPUT_PROTOCOL_METADATA_KEY,
    NATIVE_STRUCTURED_OUTPUT_METADATA_KEY,
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamReasoningRepresentation,
    StreamReasoningSegmentState,
    StreamTerminalOutcome,
    StreamValidationError,
    StreamVisibility,
    canonical_item_from_consumer_projection,
    stream_channel_for_kind,
    stream_observability_payload,
)
from ....task.usage import (
    USAGE_COUNTER_NAMES,
    UsageObservation,
    usage_observation_from_response,
)
from ....tool.display import (
    ToolDisplayProjection,
    fallback_tool_call_display_projection,
    fallback_tool_outcome_display_projection,
    tool_display_projection_from_metadata,
    tool_display_projection_metadata,
)
from ....tool.manager import ToolManager
from ....tool_cycles import (
    DEFAULT_MAXIMUM_TOOL_CYCLES,
    UNLIMITED_TOOL_CYCLES,
    MaximumToolCycles,
    validate_maximum_tool_cycles,
)
from ....utils import tool_call_diagnostic_payload
from ... import AgentOperation
from ...engine import EngineAgent
from ...execution import (
    AgentExecution,
    AgentExecutionStatus,
    AttachedInteractionRuntime,
    DurableInteractionRuntime,
    DurableInteractionStagingContext,
    ExecutionInputRequiredError,
    ExecutionTerminatedError,
)
from ...orchestrator_response_contract import DurableOrchestratorResponse

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Task,
    create_task,
    ensure_future,
    gather,
    shield,
    sleep,
    wait,
)
from asyncio import (
    Event as AsyncioEvent,
)
from base64 import b64encode
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from hashlib import sha256
from inspect import Signature, isawaitable, signature
from json import JSONDecodeError, dumps, loads
from queue import Full, Queue
from time import perf_counter
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar, cast
from uuid import UUID, uuid4

_ToolConfirmationAction = Awaitable[str | bool | None] | str | bool | None
_T = TypeVar("_T")
_CLEANUP_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, kw_only=True, slots=True)
class _ToolExecutionOutcome:
    call: ToolCall
    context: ToolCallContext
    planned_index: int
    result: ToolCallOutcome | None
    history_recorded: bool = False


@dataclass(kw_only=True, slots=True)
class _CanonicalToolCallLifecycle:
    argument_deltas: list[str] = field(default_factory=list)
    ready_item: CanonicalStreamItem | None = None
    correlation: StreamItemCorrelation | None = None
    done: bool = False
    invalid: bool = False
    queued: bool = False
    incomplete_diagnostic_emitted: bool = False


_TOOL_CALL_LIFECYCLE_KINDS = frozenset(
    {
        StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
        StreamItemKind.TOOL_CALL_READY,
        StreamItemKind.TOOL_CALL_DONE,
    }
)
_INVALID_TOOL_CALL_ARGUMENTS = object()


class OrchestratorResponse(
    DurableOrchestratorResponse,
    AsyncIterator[CanonicalStreamItem],
):
    """Async iterator handling tool execution during streaming."""

    _LEGACY_STREAM_ERROR = (
        "unsupported legacy orchestrator response stream item"
    )
    DEFAULT_MAXIMUM_TOOL_CYCLES = DEFAULT_MAXIMUM_TOOL_CYCLES
    _MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES = 2
    _MAXIMUM_MODEL_TOOL_OUTPUT_CHARS = 12_000
    _MAXIMUM_STAGING_QUEUE_ITEMS = 4096
    _CANCELLATION_POLL_INTERVAL_SECONDS = 0.01

    _response: TextGenerationResponse
    _response_iterator: AsyncIterator[Any] | None
    _engine_agent: EngineAgent
    _operation: AgentOperation
    _engine_args: dict[str, Any]
    _event_manager: EventManager | None
    _tool_manager: ToolManager | None
    _capability_catalog: ModelCapabilityCatalog | None
    _calls: Queue[ToolCall]
    _tool_result_outcomes: Queue[_ToolExecutionOutcome]
    _input: Input
    _context: ModelCallContext
    _tool_context: ToolCallContext | None
    _call_history: list[ToolCall]
    _attempted_call_signatures: set[str]
    _tool_cycle_signatures: set[str]
    _tool_cycle_count: int
    _consecutive_non_executed_cycles: int
    _agent_id: UUID | None
    _participant_id: UUID | None
    _session_id: UUID | None
    _parser_queue: Queue[CanonicalStreamItem] | None
    _tool_parser: ToolCallResponseParser | None
    _cancellation_checker: Callable[[], Awaitable[None]] | None
    _canonical_items: list[CanonicalStreamItem]
    _canonical_yield_index: int
    _canonical_sequence: int
    _canonical_stream_terminal: StreamTerminalOutcome | None
    _canonical_stream_closed: bool
    _canonical_stream_session_id: str
    _canonical_run_id: str
    _canonical_turn_id: str
    _canonical_correlation: StreamItemCorrelation
    _canonical_answer_started: bool
    _canonical_answer_done: bool
    _canonical_reasoning_started: bool
    _canonical_reasoning_done: bool
    _canonical_reasoning_segments: StreamReasoningSegmentState
    _canonical_item_available: AsyncioEvent
    _canonical_tool_call_lifecycles: dict[str, _CanonicalToolCallLifecycle]
    _staged_tool_call_items: list[CanonicalStreamItem]
    _staged_tool_batch_invalid: bool
    _staged_tool_batch_rejection_code: CapabilityBatchRejectionCode | None
    _staged_tool_batch_present: bool
    _classified_tool_call_object_ids: set[int]
    _canonical_tool_call_provider_families: dict[str, str]
    _provider_tool_call_ids_by_canonical_id: dict[str, str]
    _text_parser_tool_call_ids: set[str]
    _canonical_tool_call_argument_delta_ids: set[str]
    _canonical_tool_call_ready_ids: set[str]
    _canonical_tool_call_done_ids: set[str]
    _canonical_tool_call_diagnostic_ids: set[str]
    _canonical_tool_execution_started_ids: set[str]
    _canonical_tool_execution_terminal_ids: set[str]
    _canonical_tool_call_reserved_ids: set[str]
    _response_tool_call_id_aliases: dict[str, str]
    _response_reserved_tool_call_ids: set[str]
    _canonical_tool_call_ids_by_object: dict[int, str]
    _canonical_tool_call_index: int
    _canonical_tool_call_diagnostic_index: int
    _active_model_continuation_id: str | None
    _response_drained: bool
    _response_answer_start_index: int
    _pending_tool_batch_task: Task[list[_ToolExecutionOutcome]] | None
    _maximum_tool_cycles: MaximumToolCycles
    _block_repeated_tool_calls: bool
    _final_response_text: str | None
    _task_input_call: TaskInputCapabilityCall | None
    _execution: AgentExecution | None
    _pending_interaction_task: Task[InteractionRequestResult] | None
    _pending_interaction_call: TaskInputCapabilityCall | None
    _pending_interaction_assistant_text: str
    _active_interaction_request: InputRequest | None
    _pending_interaction_published: bool
    _input_required_result: InputRequiredResult | None
    _execution_terminated: bool
    _execution_finalized: bool
    _execution_cleanup_task: Task[tuple[BaseException, ...]] | None
    _cancellation_cleanup_task: Task[None] | None
    _provider_cleanup_complete: bool
    _interaction_cleanup_complete: bool
    _interaction_cleanup_task: Task[None] | None

    def __init__(
        self,
        input: Input,
        response: TextGenerationResponse,
        engine_agent: EngineAgent,
        operation: AgentOperation,
        engine_args: dict[str, Any],
        context: ModelCallContext,
        event_manager: EventManager | None = None,
        tool: ToolManager | None = None,
        *,
        agent_id: UUID | None = None,
        participant_id: UUID | None = None,
        session_id: UUID | None = None,
        tool_confirm: (
            Callable[[ToolCall], _ToolConfirmationAction] | None
        ) = None,
        block_repeated_tool_calls: bool = False,
        enable_tool_parsing: bool = True,
        maximum_tool_cycles: MaximumToolCycles = DEFAULT_MAXIMUM_TOOL_CYCLES,
        initial_tool_cycle_count: int = 0,
        capability: ModelCapabilityCatalog | None = None,
    ) -> None:
        assert input and response and engine_agent and operation
        assert type(block_repeated_tool_calls) is bool
        maximum_tool_cycles = validate_maximum_tool_cycles(maximum_tool_cycles)
        assert (
            type(initial_tool_cycle_count) is int
            and initial_tool_cycle_count >= 0
        ), "initial_tool_cycle_count must be a non-negative integer"
        assert (
            maximum_tool_cycles == UNLIMITED_TOOL_CYCLES
            or initial_tool_cycle_count <= maximum_tool_cycles
        ), "initial tool cycles exceed the configured maximum"
        self._input = input
        self._response = response
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool_manager = None if tool and tool.is_empty else tool
        context_capability = context.capability
        if capability is None:
            capability = context_capability
        elif context_capability is None:
            context = replace(context, capability=capability)
        else:
            assert (
                context_capability is capability
            ), "response capability must match the model-call context"
        self._capability_catalog = capability
        assert (
            not enable_tool_parsing or capability is not None
        ), "tool parsing requires an explicit model capability catalog"
        self._context = context
        self._execution = context.execution
        self._finished = False
        self._step = 0
        self._tool_context = None
        self._call_history = []
        self._attempted_call_signatures = set()
        self._tool_cycle_signatures = set()
        self._tool_cycle_count = initial_tool_cycle_count
        self._consecutive_non_executed_cycles = 0
        self._agent_id = agent_id
        self._participant_id = participant_id
        self._session_id = session_id
        self._tool_confirm = tool_confirm
        self._tool_confirm_all = False
        self._calls = self._make_staging_queue()
        self._parser_queue = self._make_staging_queue()
        self._tool_result_outcomes = self._make_staging_queue()
        self._cancellation_checker = None
        self._model_responses = [response]
        self._tool_parser = (
            ToolCallResponseParser(capability, self._event_manager)
            if enable_tool_parsing and capability is not None
            else None
        )
        self._canonical_items = []
        self._canonical_yield_index = 0
        self._canonical_sequence = 0
        self._canonical_stream_terminal = None
        self._canonical_stream_closed = False
        origin = (
            self._execution.initial_origin
            if self._execution is not None
            else None
        )
        self._canonical_stream_session_id = str(
            origin.stream_session_id
            if origin is not None
            else session_id or uuid4()
        )
        self._canonical_run_id = str(
            origin.run_id if origin is not None else agent_id or uuid4()
        )
        self._canonical_turn_id = str(
            origin.turn_id if origin is not None else participant_id or uuid4()
        )
        self._canonical_correlation = (
            StreamItemCorrelation(
                task_id=(
                    str(origin.task_id)
                    if origin is not None and origin.task_id is not None
                    else None
                ),
                agent_id=origin.agent_id,
                branch_id=origin.branch_id,
                parent_branch_id=origin.parent_branch_id,
            )
            if origin is not None
            else StreamItemCorrelation(
                task_id=str(agent_id) if agent_id is not None else None
            )
        )
        self._canonical_answer_started = False
        self._canonical_answer_done = False
        self._canonical_reasoning_started = False
        self._canonical_reasoning_done = False
        self._canonical_reasoning_segments = StreamReasoningSegmentState()
        self._canonical_item_available = AsyncioEvent()
        self._canonical_tool_call_lifecycles = {}
        self._staged_tool_call_items = []
        self._staged_tool_batch_invalid = False
        self._staged_tool_batch_rejection_code = None
        self._staged_tool_batch_present = False
        self._classified_tool_call_object_ids = set()
        self._canonical_tool_call_provider_families = {}
        self._provider_tool_call_ids_by_canonical_id = {}
        self._text_parser_tool_call_ids = set()
        self._canonical_tool_call_argument_delta_ids = set()
        self._canonical_tool_call_ready_ids = set()
        self._canonical_tool_call_done_ids = set()
        self._canonical_tool_call_diagnostic_ids = set()
        self._canonical_tool_execution_started_ids = set()
        self._canonical_tool_execution_terminal_ids = set()
        self._canonical_tool_call_reserved_ids = set()
        self._response_tool_call_id_aliases = {}
        self._response_reserved_tool_call_ids = set()
        self._canonical_tool_call_ids_by_object = {}
        self._canonical_tool_call_index = 0
        self._canonical_tool_call_diagnostic_index = 0
        self._active_model_continuation_id = None
        self._response_drained = False
        self._response_answer_start_index = 0
        self._pending_tool_batch_task = None
        self._maximum_tool_cycles = maximum_tool_cycles
        self._block_repeated_tool_calls = block_repeated_tool_calls
        self._final_response_text = None
        self._task_input_call = None
        self._pending_interaction_task = None
        self._pending_interaction_call = None
        self._pending_interaction_assistant_text = ""
        self._active_interaction_request = None
        self._pending_interaction_published = False
        self._input_required_result = None
        self._execution_terminated = False
        self._execution_finalized = False
        self._execution_cleanup_task = None
        self._cancellation_cleanup_task = None
        self._provider_cleanup_complete = False
        self._interaction_cleanup_complete = False
        self._interaction_cleanup_task = None

    @property
    def input_token_count(self) -> int:
        return self._response.input_token_count

    @property
    def execution(self) -> AgentExecution | None:
        """Return the invocation state owned by this exact response."""
        return self._execution

    @property
    def continuation_generation_settings(self) -> dict[str, Any]:
        """Return provider-neutral settings for durable continuation."""
        settings = self._continuation_engine_args()
        settings["maximum_tool_cycles"] = self._maximum_tool_cycles
        settings["block_repeated_tool_calls"] = self._block_repeated_tool_calls
        return settings

    @property
    def continuation_tool_loop_count(self) -> int:
        """Return completed domain-tool cycles before suspension."""
        return self._tool_cycle_count

    @property
    def ownership_cleanup_complete(self) -> bool:
        """Return whether terminal cleanup permits ownership release."""
        execution = self._execution
        return (
            execution is None
            or execution.status
            not in {
                AgentExecutionStatus.CANCELLED,
                AgentExecutionStatus.ERRORED,
            }
            or (
                self._provider_cleanup_complete
                and (
                    self._execution_cleanup_task is None
                    or self._execution_cleanup_task.done()
                )
                and (
                    execution.interaction_runtime is None
                    or self._interaction_cleanup_complete
                )
            )
        )

    async def aclose(self) -> None:
        """Close this response and retry incomplete cancellation cleanup."""
        execution = self._execution
        if (
            execution is None
            or execution.status is AgentExecutionStatus.COMPLETED
        ):
            await self._response.aclose()
            return
        if execution.status is AgentExecutionStatus.INPUT_REQUIRED:
            if (
                isinstance(
                    execution.interaction_runtime,
                    AttachedInteractionRuntime,
                )
                and self._pending_interaction_task is not None
            ):
                await self._converge_stream_cancellation()
            else:
                await self._close_provider_response()
            return
        if execution.status is AgentExecutionStatus.ERRORED:
            await self._converge_stream_error_cleanup()
            return
        await self._converge_stream_cancellation()

    async def sync_messages(self) -> None:
        """Synchronize memory for this response's exact invocation once."""
        if not self.ownership_cleanup_complete:
            execution = self._execution
            if (
                execution is not None
                and execution.status is AgentExecutionStatus.ERRORED
            ):
                await self._converge_stream_error_cleanup()
            else:
                await self._converge_stream_cancellation()
        await self._engine_agent.sync_messages(self._execution)

    @property
    def output_token_count(self) -> int:
        return self._response.output_token_count

    @property
    def usage(self) -> object | None:
        usage_values = tuple(
            usage
            for response in self._model_responses
            if (usage := response.usage) is not None
        )
        if not usage_values:
            return None
        if len(usage_values) == 1:
            return usage_values[0]
        return usage_values

    @property
    def usage_responses(self) -> tuple[TextGenerationResponse, ...]:
        return tuple(self._model_responses)

    @property
    def canonical_items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._canonical_items)

    @property
    def task_input_call(self) -> TaskInputCapabilityCall | None:
        """Return the validated reserved call awaiting control-plane wiring."""
        return self._task_input_call

    async def consumer_projections(
        self,
        *,
        stream_session_id: str,
        run_id: str,
        turn_id: str,
    ) -> AsyncIterator[StreamConsumerProjection]:
        for value in (stream_session_id, run_id, turn_id):
            assert isinstance(value, str)
            assert value.strip()

        emitted = 0
        try:
            async for item in self:
                yield StreamConsumerProjection.from_item(item)
                emitted += 1
        except (CancelledError, Exception):
            while emitted < len(self._canonical_items):
                yield StreamConsumerProjection.from_item(
                    self._canonical_items[emitted]
                )
                emitted += 1
            raise

    def _queue_parser_output(
        self,
        item: object,
    ) -> None:
        if (
            self._capability_catalog is not None
            and getattr(item, "kind", None) is StreamItemKind.STREAM_DIAGNOSTIC
        ):
            self._mark_staged_tool_batch_invalid()
            return
        correlation = getattr(item, "correlation", None)
        if isinstance(correlation, StreamItemCorrelation):
            tool_call_id = correlation.tool_call_id
            if self._is_valid_tool_call_id(tool_call_id):
                assert tool_call_id is not None
                self._text_parser_tool_call_ids.add(
                    self._canonical_lifecycle_tool_call_id(tool_call_id)
                )
        if isinstance(item, StreamProviderEvent):
            self._append_canonical_provider_event_item(item)
            return
        if isinstance(item, CanonicalStreamItem):
            self._append_canonical_response_item(item)
            return
        if isinstance(item, StreamConsumerProjection):
            self._append_canonical_response_item(
                canonical_item_from_consumer_projection(item)
            )
            return
        raise StreamValidationError(self._LEGACY_STREAM_ERROR)

    def _append_canonical_provider_event_item(
        self,
        event: StreamProviderEvent,
    ) -> CanonicalStreamItem | None:
        assert isinstance(event, StreamProviderEvent)
        if (
            event.kind is StreamItemKind.REASONING_DELTA
            and event.text_delta == ""
        ):
            return None
        correlation = self._canonical_response_correlation(event.correlation)
        self._validate_canonical_reasoning_segment(
            event.kind,
            visibility=event.visibility,
            reasoning_representation=event.reasoning_representation,
            segment_instance_ordinal=event.segment_instance_ordinal,
            correlation=correlation,
            metadata=event.metadata,
        )
        if event.kind in _TOOL_CALL_LIFECYCLE_KINDS:
            return self._append_canonical_provider_tool_call_item(
                event,
                correlation=correlation,
            )
        if event.kind in (
            StreamItemKind.ANSWER_DONE,
            StreamItemKind.REASONING_DONE,
            StreamItemKind.USAGE_COMPLETED,
        ):
            return None
        if event.kind in (
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
            StreamItemKind.STREAM_CANCELLED,
        ):
            if event.kind is StreamItemKind.STREAM_COMPLETED:
                return None
            self._discard_untrusted_response_tool_call_batch()
            self._finish_canonical_stream(
                event.kind,
                data=event.data,
                usage=event.usage,
            )
            return None
        return self._append_canonical_item(
            event.kind,
            text_delta=event.text_delta,
            data=event.data,
            usage=event.usage,
            correlation=correlation,
            visibility=event.visibility,
            reasoning_representation=event.reasoning_representation,
            segment_instance_ordinal=event.segment_instance_ordinal,
            metadata=event.metadata,
            provider_event_type=event.provider_event_type,
        )

    def _append_canonical_provider_tool_call_item(
        self,
        event: StreamProviderEvent,
        *,
        correlation: StreamItemCorrelation,
    ) -> CanonicalStreamItem | None:
        assert event.kind in _TOOL_CALL_LIFECYCLE_KINDS
        tool_call_id = correlation.tool_call_id
        if not self._is_valid_tool_call_id(tool_call_id):
            if self._capability_catalog is not None:
                self._mark_staged_tool_batch_invalid(
                    CapabilityBatchRejectionCode.MISSING_CALL_ID
                )
                return None
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=None,
                code="orchestrator.tool_call.missing_id",
                message="Tool-call lifecycle item is missing tool_call_id.",
                details={"kind": event.kind.value},
                correlation=correlation,
            )
        assert tool_call_id is not None
        if (
            event.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            and event.text_delta is None
        ):
            tool_call_id = self._canonical_lifecycle_tool_call_id(tool_call_id)
            correlation = replace(
                correlation,
                tool_call_id=tool_call_id,
            )
            state = self._canonical_tool_call_lifecycle(tool_call_id)
            state.correlation = self._merged_tool_call_correlation(
                state.correlation,
                correlation,
                tool_call_id=tool_call_id,
            )
            state.invalid = True
            if self._capability_catalog is not None:
                self._mark_staged_tool_batch_invalid()
                return None
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code="orchestrator.tool_call.missing_argument_delta",
                message="Tool-call argument delta is missing text.",
                details={"kind": event.kind.value},
                correlation=correlation,
            )
        return self._append_canonical_tool_call_lifecycle_item(
            event.kind,
            text_delta=event.text_delta,
            data=event.data,
            correlation=correlation,
            visibility=event.visibility,
            metadata=event.metadata,
            provider_event_type=event.provider_event_type,
        )

    def _append_canonical_response_item(
        self, item: CanonicalStreamItem
    ) -> CanonicalStreamItem | None:
        assert isinstance(item, CanonicalStreamItem)
        correlation = self._canonical_response_correlation(item.correlation)
        self._validate_canonical_reasoning_segment(
            item.kind,
            visibility=item.visibility,
            reasoning_representation=item.reasoning_representation,
            segment_instance_ordinal=item.segment_instance_ordinal,
            correlation=correlation,
            metadata=item.metadata,
        )
        if item.kind in _TOOL_CALL_LIFECYCLE_KINDS:
            return self._append_canonical_tool_call_lifecycle_item(
                item.kind,
                text_delta=item.text_delta,
                data=item.data,
                usage=item.usage,
                correlation=correlation,
                visibility=item.visibility,
                metadata=item.metadata,
                provider_family=item.provider_family,
                provider_event_type=item.provider_event_type,
            )
        if item.kind in (
            StreamItemKind.STREAM_STARTED,
            StreamItemKind.STREAM_CLOSED,
            StreamItemKind.ANSWER_DONE,
            StreamItemKind.REASONING_DONE,
            StreamItemKind.USAGE_COMPLETED,
        ):
            return None
        if item.terminal_outcome is not None:
            if item.kind is StreamItemKind.STREAM_COMPLETED:
                return None
            self._discard_untrusted_response_tool_call_batch()
            self._finish_canonical_stream(
                item.kind,
                data=item.data,
                usage=item.usage,
                correlation=correlation,
            )
            return None
        assert self._canonical_stream_terminal is None
        assert not self._canonical_stream_closed
        canonical_item = replace(
            item,
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
            sequence=self._canonical_sequence,
            channel=stream_channel_for_kind(item.kind),
            correlation=correlation,
        )
        self._canonical_items.append(canonical_item)
        self._canonical_sequence += 1
        self._track_canonical_channel_boundary(canonical_item)
        self._track_canonical_lifecycle_item(canonical_item)
        self._notify_canonical_item_available()
        return canonical_item

    def _canonical_response_correlation(
        self,
        correlation: StreamItemCorrelation,
    ) -> StreamItemCorrelation:
        assert isinstance(correlation, StreamItemCorrelation)
        active_id = self._active_model_continuation_id
        incoming_id = correlation.model_continuation_id
        if active_id is not None:
            if incoming_id is not None and incoming_id != active_id:
                raise StreamValidationError(
                    "response item model continuation id conflicts with the "
                    "active continuation"
                )
            if incoming_id is None:
                correlation = replace(
                    correlation,
                    model_continuation_id=active_id,
                )
        if (
            correlation.task_id is None
            and self._canonical_correlation.task_id is not None
        ):
            correlation = replace(
                correlation,
                task_id=self._canonical_correlation.task_id,
            )
        return correlation

    def _validate_canonical_reasoning_segment(
        self,
        kind: StreamItemKind,
        *,
        visibility: StreamVisibility,
        reasoning_representation: StreamReasoningRepresentation | None,
        segment_instance_ordinal: int | None,
        correlation: StreamItemCorrelation,
        metadata: Mapping[str, Any],
    ) -> None:
        if kind is not StreamItemKind.REASONING_DELTA:
            self._canonical_reasoning_segments.complete_segment()
            return
        self._canonical_reasoning_segments.observe(
            StreamProviderEvent(
                kind=kind,
                text_delta="reasoning",
                correlation=correlation,
                visibility=visibility,
                reasoning_representation=reasoning_representation,
                segment_instance_ordinal=segment_instance_ordinal,
                metadata=dict(metadata),
            )
        )

    def _canonical_item_from_response_item(
        self,
        item: object,
    ) -> CanonicalStreamItem:
        if isinstance(item, StreamConsumerProjection):
            return canonical_item_from_consumer_projection(item)
        if not isinstance(item, CanonicalStreamItem):
            raise StreamValidationError(self._LEGACY_STREAM_ERROR)
        return item

    async def _process_canonical_text_tool_call_parser_stage(
        self,
        item: CanonicalStreamItem,
        *,
        after_parser_output: Callable[[int], None] | None = None,
    ) -> bool:
        if not self._tool_parser_canonicalizes_answer_delta(item):
            return False

        assert self._tool_parser is not None
        assert item.text_delta is not None
        token_item = self._canonical_token_event_item(item)
        await self._emit_token_generated_event(token_item)
        self._step += 1
        collection_index = len(self._canonical_items)
        for parser_item in await self._tool_parser.push(item.text_delta):
            self._queue_parser_output(parser_item)
            if after_parser_output is not None:
                after_parser_output(collection_index)
                collection_index = len(self._canonical_items)
        return True

    def _tool_parser_canonicalizes_answer_delta(
        self,
        item: CanonicalStreamItem,
    ) -> bool:
        if (
            item.kind is not StreamItemKind.ANSWER_DELTA
            or item.text_delta is None
            or self._tool_parser is None
            or self._is_locally_classified_answer(item)
        ):
            return False
        canonicalizes = getattr(
            self._tool_parser,
            "canonicalizes_answer_deltas",
            False,
        )
        return canonicalizes if isinstance(canonicalizes, bool) else False

    @staticmethod
    def _is_locally_classified_answer(
        item: CanonicalStreamItem,
    ) -> bool:
        return item.kind is StreamItemKind.ANSWER_DELTA and (
            item.metadata.get(NATIVE_STRUCTURED_OUTPUT_METADATA_KEY) is True
            or item.metadata.get(LOCAL_STRUCTURED_OUTPUT_PROTOCOL_METADATA_KEY)
            == LOCAL_STRUCTURED_OUTPUT_PROTOCOL_ID
        )

    def _canonical_token_event_item(
        self,
        item: CanonicalStreamItem,
    ) -> CanonicalStreamItem:
        return replace(
            item,
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
            sequence=self._canonical_sequence,
            channel=stream_channel_for_kind(item.kind),
        )

    async def _flush_canonical_text_tool_call_parser_stage(
        self,
        *,
        after_parser_output: Callable[[int], None] | None = None,
    ) -> None:
        if self._tool_parser is None:
            return
        collection_index = len(self._canonical_items)
        for item in await self._tool_parser.flush():
            self._queue_parser_output(item)
            if after_parser_output is not None:
                after_parser_output(collection_index)
                collection_index = len(self._canonical_items)

    async def _process_canonical_response_item(
        self,
        item: CanonicalStreamItem,
    ) -> None:
        if await self._process_canonical_text_tool_call_parser_stage(item):
            return

        canonical_item = self._append_canonical_response_item(item)
        if canonical_item is None:
            return
        if (
            canonical_item.kind is StreamItemKind.ANSWER_DELTA
            and canonical_item.text_delta is not None
        ):
            await self._emit_token_generated_event(canonical_item)
            self._step += 1
            if self._tool_parser and not (
                self._is_locally_classified_answer(canonical_item)
            ):
                for parser_item in await self._tool_parser.push(
                    canonical_item.text_delta
                ):
                    self._queue_parser_output(parser_item)
            return

    def _track_canonical_lifecycle_item(
        self,
        item: CanonicalStreamItem,
    ) -> None:
        tool_call_id = item.correlation.tool_call_id
        if tool_call_id is None:
            return
        if item.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            self._canonical_tool_call_argument_delta_ids.add(tool_call_id)
            assert item.text_delta is not None
            self._canonical_tool_call_lifecycle(
                tool_call_id
            ).argument_deltas.append(item.text_delta)
        elif item.kind is StreamItemKind.TOOL_CALL_READY:
            self._canonical_tool_call_ready_ids.add(tool_call_id)
            self._canonical_tool_call_lifecycle(tool_call_id).ready_item = item
        elif item.kind is StreamItemKind.TOOL_CALL_DONE:
            self._canonical_tool_call_done_ids.add(tool_call_id)
            self._canonical_tool_call_lifecycle(tool_call_id).done = True
        elif item.kind is StreamItemKind.STREAM_DIAGNOSTIC:
            self._canonical_tool_call_diagnostic_ids.add(tool_call_id)
        elif item.kind is StreamItemKind.TOOL_EXECUTION_STARTED:
            self._canonical_tool_execution_started_ids.add(tool_call_id)
        elif item.kind in (
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
        ):
            self._canonical_tool_execution_terminal_ids.add(tool_call_id)

    def _tool_call_ready_display_metadata(
        self,
        tool_call_id: str,
        data: Any | None,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if self._capability_catalog is not None:
            return metadata
        if tool_display_projection_from_metadata(metadata) is not None:
            return metadata
        if not isinstance(data, dict):
            return metadata
        name = data.get("name")
        if not isinstance(name, str):
            return metadata
        arguments = data.get("arguments")
        call = ToolCall(
            id=tool_call_id,
            name=name,
            arguments=(
                cast(dict[str, Any], arguments)
                if isinstance(arguments, dict)
                else None
            ),
        )
        projection = self._tool_call_display_projection(call)
        return {
            **({} if metadata is None else metadata),
            **cast(
                dict[str, Any],
                tool_display_projection_metadata(projection),
            ),
        }

    def _append_canonical_tool_call_lifecycle_item(
        self,
        kind: StreamItemKind,
        *,
        text_delta: str | None = None,
        data: Any | None = None,
        usage: Any | None = None,
        correlation: StreamItemCorrelation,
        visibility: StreamVisibility = StreamVisibility.PUBLIC,
        metadata: Mapping[str, Any] | None = None,
        provider_family: str | None = None,
        provider_event_type: str | None = None,
    ) -> CanonicalStreamItem | None:
        assert kind in _TOOL_CALL_LIFECYCLE_KINDS
        tool_call_id = correlation.tool_call_id
        if not self._is_valid_tool_call_id(tool_call_id):
            if self._capability_catalog is not None:
                self._mark_staged_tool_batch_invalid(
                    CapabilityBatchRejectionCode.MISSING_CALL_ID
                )
                return None
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=None,
                code="orchestrator.tool_call.missing_id",
                message="Tool-call lifecycle item is missing tool_call_id.",
                details={"kind": kind.value},
                correlation=correlation,
            )
        assert tool_call_id is not None
        tool_call_id = self._canonical_lifecycle_tool_call_id(tool_call_id)
        correlation = replace(correlation, tool_call_id=tool_call_id)
        if provider_family is not None:
            existing_family = self._canonical_tool_call_provider_families.get(
                tool_call_id
            )
            if (
                existing_family is not None
                and existing_family != provider_family
            ):
                if self._capability_catalog is not None:
                    self._mark_staged_tool_batch_invalid()
                    return None
                raise StreamValidationError(
                    "tool-call lifecycle provider family changed"
                )
            self._canonical_tool_call_provider_families[tool_call_id] = (
                provider_family
            )
        state = self._canonical_tool_call_lifecycle(tool_call_id)
        state.correlation = self._merged_tool_call_correlation(
            state.correlation,
            correlation,
            tool_call_id=tool_call_id,
        )
        diagnostic = self._canonical_tool_call_item_diagnostic(
            kind,
            text_delta=text_delta,
            data=data,
            tool_call_id=tool_call_id,
            state=state,
        )
        if diagnostic is not None or state.invalid:
            if not state.queued:
                state.invalid = True
                self._close_invalid_tool_call_lifecycle(
                    tool_call_id,
                    state=state,
                )
            return diagnostic

        item_metadata = None if metadata is None else dict(metadata)
        if (
            kind is StreamItemKind.TOOL_CALL_READY
            and self._capability_catalog is None
        ):
            item_metadata = self._tool_call_ready_display_metadata(
                tool_call_id,
                data,
                item_metadata,
            )

        item = (
            self._stage_canonical_tool_call_item(
                kind,
                text_delta=text_delta,
                data=data,
                usage=usage,
                correlation=correlation,
                visibility=visibility,
                metadata=item_metadata,
                provider_family=provider_family,
                provider_event_type=provider_event_type,
            )
            if self._capability_catalog is not None
            else self._append_canonical_item(
                kind,
                text_delta=text_delta,
                data=data,
                usage=usage,
                correlation=correlation,
                visibility=visibility,
                metadata=item_metadata,
                provider_family=provider_family,
                provider_event_type=provider_event_type,
            )
        )
        if item is None:
            return None
        if kind is StreamItemKind.TOOL_CALL_DONE:
            self._queue_completed_canonical_tool_call(tool_call_id, state)
        return item

    def _stage_canonical_tool_call_item(
        self,
        kind: StreamItemKind,
        *,
        text_delta: str | None,
        data: Any | None,
        usage: Any | None,
        correlation: StreamItemCorrelation,
        visibility: StreamVisibility,
        metadata: dict[str, Any] | None,
        provider_family: str | None,
        provider_event_type: str | None,
    ) -> CanonicalStreamItem:
        """Stage a model-authored tool-call lifecycle item privately."""
        assert kind in _TOOL_CALL_LIFECYCLE_KINDS
        item = CanonicalStreamItem(
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
            sequence=self._canonical_sequence,
            kind=kind,
            channel=stream_channel_for_kind(kind),
            correlation=correlation,
            text_delta=text_delta,
            data=cast(Any, data),
            usage=cast(Any, usage),
            visibility=visibility,
            metadata=metadata or {},
            provider_family=provider_family,
            provider_event_type=provider_event_type,
        )
        self._staged_tool_call_items.append(item)
        self._staged_tool_batch_present = True
        tool_call_id = correlation.tool_call_id
        assert tool_call_id is not None
        state = self._canonical_tool_call_lifecycle(tool_call_id)
        if kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            assert text_delta is not None
            state.argument_deltas.append(text_delta)
        elif kind is StreamItemKind.TOOL_CALL_READY:
            state.ready_item = item
        else:
            state.done = True
        return item

    def _mark_staged_tool_batch_invalid(
        self,
        code: CapabilityBatchRejectionCode = (
            CapabilityBatchRejectionCode.MALFORMED_CALL
        ),
    ) -> None:
        """Mark the current model tool-call batch invalid privately."""
        assert isinstance(code, CapabilityBatchRejectionCode)
        self._staged_tool_batch_present = True
        self._staged_tool_batch_invalid = True
        if self._staged_tool_batch_rejection_code is None:
            self._staged_tool_batch_rejection_code = code

    def _canonical_tool_call_item_diagnostic(
        self,
        kind: StreamItemKind,
        *,
        text_delta: str | None,
        data: Any | None,
        tool_call_id: str,
        state: _CanonicalToolCallLifecycle,
    ) -> CanonicalStreamItem | None:
        if state.invalid:
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code="orchestrator.tool_call.invalid_lifecycle",
                message="Tool-call lifecycle item followed an invalid item.",
                details={"kind": kind.value},
            )
        if kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA:
            if text_delta is None:
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.missing_argument_delta",
                    message="Tool-call argument delta is missing text.",
                    details={"kind": kind.value},
                )
            if state.done:
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.argument_after_done",
                    message="Tool-call argument delta arrived after done.",
                    details={"kind": kind.value},
                )
            if state.ready_item is not None:
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.argument_after_ready",
                    message="Tool-call argument delta arrived after ready.",
                    details={"kind": kind.value},
                )
            return None
        if kind is StreamItemKind.TOOL_CALL_READY:
            if state.done:
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.ready_after_done",
                    message="Tool-call ready arrived after done.",
                    details={"kind": kind.value},
                )
            if state.ready_item is not None:
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.duplicate_ready",
                    message="Duplicate tool-call ready item was ignored.",
                    details={"kind": kind.value},
                )
            ready_data = data if isinstance(data, dict) else None
            if ready_data is None:
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.missing_ready_data",
                    message="Tool-call ready item is missing data.",
                    details={"kind": kind.value},
                )
            name = ready_data.get("name")
            if not isinstance(name, str):
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.missing_name",
                    message="Tool-call ready item is missing a tool name.",
                    details={"kind": kind.value},
                )
            arguments = ready_data.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                return self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
                    message="Tool-call ready arguments must be an object.",
                    details={"kind": kind.value},
                )
            return None
        assert kind is StreamItemKind.TOOL_CALL_DONE
        if state.done:
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code="orchestrator.tool_call.duplicate_done",
                message="Duplicate tool-call done item was ignored.",
                details={"kind": kind.value},
            )
        if state.ready_item is None:
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code="orchestrator.tool_call.done_before_ready",
                message="Tool-call done arrived before ready.",
                details={"kind": kind.value},
            )
        return None

    def _queue_completed_canonical_tool_call(
        self,
        tool_call_id: str,
        state: _CanonicalToolCallLifecycle,
    ) -> None:
        assert tool_call_id
        if state.invalid or state.queued:
            return
        call = self._tool_call_from_canonical_lifecycle(
            tool_call_id,
            state,
        )
        if call is None:
            state.invalid = True
            return
        state.queued = True
        self._put_staging_item(self._calls, call, "tool call")

    def _tool_call_from_canonical_lifecycle(
        self,
        tool_call_id: str,
        state: _CanonicalToolCallLifecycle,
    ) -> ToolCall | None:
        assert tool_call_id
        ready_item = state.ready_item
        assert ready_item is not None
        if not isinstance(ready_item.data, dict):
            self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code="orchestrator.tool_call.missing_ready_data",
                message="Tool-call ready item is missing data.",
                details={"tool_call_id": tool_call_id},
            )
            return None
        data = ready_item.data
        name = data.get("name")
        if not isinstance(name, str):
            self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code="orchestrator.tool_call.missing_name",
                message="Tool-call ready item is missing name.",
                details={"tool_call_id": tool_call_id},
            )
            return None
        arguments = self._tool_call_arguments_from_lifecycle(
            tool_call_id,
            state,
            ready_data=data,
        )
        if arguments is _INVALID_TOOL_CALL_ARGUMENTS:
            return None
        return ToolCall(
            id=tool_call_id,
            name=name if isinstance(name, str) else "",
            arguments=cast(dict[str, Any] | None, arguments),
        )

    def _tool_call_arguments_from_lifecycle(
        self,
        tool_call_id: str,
        state: _CanonicalToolCallLifecycle,
        *,
        ready_data: dict[Any, Any],
    ) -> dict[str, Any] | None | object:
        assert tool_call_id
        if state.argument_deltas:
            argument_text = "".join(state.argument_deltas)
            try:
                arguments = loads(argument_text)
            except JSONDecodeError as exc:
                ready_arguments = ready_data.get("arguments")
                if isinstance(
                    ready_arguments, dict
                ) and not argument_text.lstrip().startswith(("{", "[")):
                    return cast(dict[str, Any], ready_arguments)
                self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
                    message="Tool-call arguments are malformed JSON.",
                    details={
                        "error": str(exc),
                        "arguments": argument_text,
                    },
                )
                return _INVALID_TOOL_CALL_ARGUMENTS
            if not isinstance(arguments, dict):
                self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
                    message="Tool-call arguments must decode to an object.",
                    details={"arguments": argument_text},
                )
                return _INVALID_TOOL_CALL_ARGUMENTS
            return cast(dict[str, Any], arguments)

        arguments = ready_data.get("arguments")
        if arguments is not None and not isinstance(arguments, dict):
            self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=tool_call_id,
                code=ToolCallDiagnosticCode.MALFORMED_ARGUMENTS.value,
                message="Tool-call ready arguments must be an object.",
                details={"tool_call_id": tool_call_id},
            )
            return _INVALID_TOOL_CALL_ARGUMENTS
        if "arguments" not in ready_data:
            return None
        return (
            cast(dict[str, Any], arguments)
            if isinstance(arguments, dict)
            else None
        )

    def _append_canonical_tool_call_lifecycle_diagnostic(
        self,
        *,
        tool_call_id: str | None,
        code: str,
        message: str,
        details: dict[str, Any],
        correlation: StreamItemCorrelation | None = None,
    ) -> CanonicalStreamItem | None:
        assert isinstance(code, str)
        assert code.strip()
        assert isinstance(message, str)
        assert message.strip()
        assert isinstance(details, dict)
        if self._capability_catalog is not None:
            self._mark_staged_tool_batch_invalid()
            if self._is_valid_tool_call_id(tool_call_id):
                assert tool_call_id is not None
                state = self._canonical_tool_call_lifecycles.get(tool_call_id)
                if state is not None:
                    state.invalid = True
            return None
        if self._is_valid_tool_call_id(tool_call_id):
            assert tool_call_id is not None
            correlation_id = tool_call_id
        else:
            correlation_id = self._next_tool_call_diagnostic_id()
        if correlation is None:
            state = self._canonical_tool_call_lifecycles.get(correlation_id)
            if state is not None:
                correlation = state.correlation
        resolved_correlation = (
            replace(correlation, tool_call_id=correlation_id)
            if correlation is not None
            else StreamItemCorrelation(tool_call_id=correlation_id)
        )
        return self._append_canonical_item(
            StreamItemKind.STREAM_DIAGNOSTIC,
            data={
                "code": code,
                "message": message,
                "stage": ToolCallDiagnosticStage.PARSE.value,
                "tool_call_id": correlation_id,
                "details": cast(Any, details),
            },
            correlation=resolved_correlation,
        )

    def _close_invalid_tool_call_lifecycle(
        self,
        tool_call_id: str,
        *,
        state: _CanonicalToolCallLifecycle,
    ) -> None:
        assert tool_call_id
        if state.done:
            return
        if self._capability_catalog is not None:
            self._mark_staged_tool_batch_invalid()
            state.done = True
            return
        if not state.argument_deltas and state.ready_item is None:
            return
        self._append_canonical_item(
            StreamItemKind.TOOL_CALL_DONE,
            correlation=(
                state.correlation
                or StreamItemCorrelation(tool_call_id=tool_call_id)
            ),
            metadata={"tool_call.close_reason": "malformed"},
        )

    def _finalize_incomplete_canonical_tool_calls(self) -> None:
        for tool_call_id, state in tuple(
            self._canonical_tool_call_lifecycles.items()
        ):
            if state.done or state.incomplete_diagnostic_emitted:
                continue
            state.invalid = True
            state.incomplete_diagnostic_emitted = True
            if self._capability_catalog is not None:
                self._mark_staged_tool_batch_invalid()
                state.done = True
                continue
            if state.ready_item is None:
                self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.missing_ready",
                    message="Tool-call lifecycle ended without ready.",
                    details={"tool_call_id": tool_call_id},
                )
            else:
                self._append_canonical_tool_call_lifecycle_diagnostic(
                    tool_call_id=tool_call_id,
                    code="orchestrator.tool_call.ready_without_done",
                    message="Tool-call ready was not followed by done.",
                    details={"tool_call_id": tool_call_id},
                )
            self._close_invalid_tool_call_lifecycle(tool_call_id, state=state)

    def _canonical_tool_call_lifecycle(
        self,
        tool_call_id: str,
    ) -> _CanonicalToolCallLifecycle:
        assert isinstance(tool_call_id, str)
        assert tool_call_id.strip()
        state = self._canonical_tool_call_lifecycles.get(tool_call_id)
        if state is None:
            state = _CanonicalToolCallLifecycle()
            self._canonical_tool_call_lifecycles[tool_call_id] = state
        return state

    @staticmethod
    def _merged_tool_call_correlation(
        existing: StreamItemCorrelation | None,
        incoming: StreamItemCorrelation,
        *,
        tool_call_id: str,
    ) -> StreamItemCorrelation:
        assert isinstance(incoming, StreamItemCorrelation)
        assert isinstance(tool_call_id, str)
        assert tool_call_id.strip()
        if existing is None:
            return replace(incoming, tool_call_id=tool_call_id)
        return StreamItemCorrelation(
            provider_request_id=(
                existing.provider_request_id or incoming.provider_request_id
            ),
            model_continuation_id=(
                existing.model_continuation_id
                or incoming.model_continuation_id
            ),
            tool_call_id=tool_call_id,
            flow_run_id=existing.flow_run_id or incoming.flow_run_id,
            node_id=existing.node_id or incoming.node_id,
            parent_sequence=(
                existing.parent_sequence
                if existing.parent_sequence is not None
                else incoming.parent_sequence
            ),
            protocol_item_id=(
                existing.protocol_item_id or incoming.protocol_item_id
            ),
            provider_output_index=(
                existing.provider_output_index
                if existing.provider_output_index is not None
                else incoming.provider_output_index
            ),
            provider_summary_index=(
                existing.provider_summary_index
                if existing.provider_summary_index is not None
                else incoming.provider_summary_index
            ),
            task_id=existing.task_id or incoming.task_id,
            artifact_id=existing.artifact_id or incoming.artifact_id,
        )

    def _begin_tool_call_lifecycle_response(self) -> None:
        self._response_answer_start_index = len(self._canonical_items)
        self._canonical_tool_call_lifecycles = {}
        self._staged_tool_call_items = []
        self._staged_tool_batch_invalid = False
        self._staged_tool_batch_rejection_code = None
        self._staged_tool_batch_present = False
        self._classified_tool_call_object_ids = set()
        self._canonical_tool_call_provider_families = {}
        self._provider_tool_call_ids_by_canonical_id = {}
        self._text_parser_tool_call_ids = set()
        self._response_tool_call_id_aliases = {}
        self._response_reserved_tool_call_ids = set()
        self._canonical_reasoning_segments = StreamReasoningSegmentState()

    def _canonical_lifecycle_tool_call_id(self, source_id: str) -> str:
        assert isinstance(source_id, str)
        assert source_id.strip()
        alias = self._response_tool_call_id_aliases.get(source_id)
        if alias is not None:
            return alias
        if not self._canonical_tool_call_id_used(source_id):
            self._reserve_canonical_tool_call_id(source_id)
            self._response_reserved_tool_call_ids.add(source_id)
            self._response_tool_call_id_aliases[source_id] = source_id
            self._provider_tool_call_ids_by_canonical_id[source_id] = source_id
            return source_id
        alias = self._next_generated_tool_call_id()
        self._response_reserved_tool_call_ids.add(alias)
        self._response_tool_call_id_aliases[source_id] = alias
        self._provider_tool_call_ids_by_canonical_id[alias] = source_id
        return alias

    def _reserve_canonical_tool_call_id(self, tool_call_id: str) -> None:
        assert isinstance(tool_call_id, str)
        assert tool_call_id.strip()
        self._canonical_tool_call_reserved_ids.add(tool_call_id)

    def _canonical_tool_call_id_used(self, tool_call_id: str) -> bool:
        assert isinstance(tool_call_id, str)
        assert tool_call_id.strip()
        return (
            tool_call_id in self._canonical_tool_call_reserved_ids
            or tool_call_id in self._canonical_tool_call_argument_delta_ids
            or tool_call_id in self._canonical_tool_call_ready_ids
            or tool_call_id in self._canonical_tool_call_done_ids
            or tool_call_id in self._canonical_tool_call_diagnostic_ids
            or tool_call_id in self._canonical_tool_execution_started_ids
            or tool_call_id in self._canonical_tool_execution_terminal_ids
        )

    def _next_generated_tool_call_id(self) -> str:
        while True:
            self._canonical_tool_call_index += 1
            tool_call_id = (
                f"orchestrator-tool-call-{self._canonical_tool_call_index}"
            )
            if not self._canonical_tool_call_id_used(tool_call_id):
                self._reserve_canonical_tool_call_id(tool_call_id)
                return tool_call_id

    @staticmethod
    def _is_valid_tool_call_id(tool_call_id: str | None) -> bool:
        return isinstance(tool_call_id, str) and bool(tool_call_id.strip())

    def _next_tool_call_diagnostic_id(self) -> str:
        while True:
            self._canonical_tool_call_diagnostic_index += 1
            tool_call_id = (
                "orchestrator-tool-call-diagnostic-"
                f"{self._canonical_tool_call_diagnostic_index}"
            )
            if not self._canonical_tool_call_id_used(tool_call_id):
                return tool_call_id

    @property
    def can_think(self) -> bool:
        return self._response.can_think

    @property
    def event_manager(self) -> EventManager | None:
        """Return the event manager owned by this response runtime."""
        return self._event_manager

    def set_cancellation_checker(
        self,
        checker: Callable[[], Awaitable[None]] | None,
    ) -> None:
        self._cancellation_checker = checker

    @property
    def is_thinking(self) -> bool:
        return self._response.is_thinking

    def set_thinking(self, thinking: bool) -> None:
        self._response.set_thinking(thinking)

    async def to_str(self) -> str:
        self._raise_if_terminal_failure()
        if self._final_response_text is not None:
            self._raise_if_completion_lost()
            return self._final_response_text
        if not self._canonical_items:
            self._append_canonical_item(StreamItemKind.STREAM_STARTED)
        try:
            output = await self._react(self._response)
            await self._finalize_execution(
                StreamItemKind.STREAM_COMPLETED,
                output=self._current_response_answer_text(),
            )
            self._finish_canonical_stream(
                self._execution_terminal_kind(StreamItemKind.STREAM_COMPLETED)
            )
            self._raise_if_completion_lost()
        except (ExecutionInputRequiredError, ExecutionTerminatedError):
            raise
        except CancelledError as exc:
            await self._settle_cancellation_failure(exc)
            raise
        except BaseException as exc:
            await self._settle_stream_failure(exc)
            raise
        self._final_response_text = output
        return output

    async def to_json(self) -> str:
        output = await self.to_str()
        return TextGenerationResponse.extract_json(output)

    async def to(self, entity_class: type) -> Any:
        json = await self.to_json()
        return entity_class(**loads(json))

    def __aiter__(self) -> "OrchestratorResponse":
        self._prepare_iteration(reset_yield_index=True)
        return self

    def _prepare_iteration(self, *, reset_yield_index: bool) -> None:
        assert isinstance(reset_yield_index, bool)
        if self._event_manager:
            self._response.add_done_callback(self._on_consumed)
        if not self._canonical_items:
            self._append_canonical_item(StreamItemKind.STREAM_STARTED)
        if not self._response_drained:
            self._response_iterator = aiter(self._response)
            self._begin_tool_call_lifecycle_response()
        self._calls = self._make_staging_queue()
        self._parser_queue = self._make_staging_queue()
        self._tool_context = self._new_tool_context(self._input)
        self._tool_result_outcomes = self._make_staging_queue()
        self._response_drained = False
        self._step = 0
        assert (
            self._pending_tool_batch_task is None
            or self._pending_tool_batch_task.done()
        )
        self._pending_tool_batch_task = None
        if reset_yield_index:
            self._canonical_yield_index = 0

    async def __anext__(self) -> CanonicalStreamItem:
        assert self._response_iterator

        try:
            while True:
                item = self._next_canonical_yield_item()
                if item is not None:
                    return item
                if self._canonical_stream_terminal is not None:
                    task = self._pending_tool_batch_task
                    if task is not None:
                        if not task.done():
                            await wait({task})
                        await self._consume_pending_tool_batch(task)
                    raise StopAsyncIteration
                try:
                    await self._next_item()
                except StopAsyncIteration:
                    item = self._next_canonical_yield_item()
                    if item is not None:
                        return item
                    raise
        except CancelledError as exc:
            await self._settle_cancellation_failure(exc)
            raise
        except StopAsyncIteration:
            raise
        except ExecutionInputRequiredError:
            raise
        except BaseException as exc:
            await self._settle_stream_failure(exc)
            raise

    def _next_canonical_yield_item(self) -> CanonicalStreamItem | None:
        if self._canonical_yield_index >= len(self._canonical_items):
            self._canonical_item_available.clear()
            return None
        item = self._canonical_items[self._canonical_yield_index]
        self._canonical_yield_index += 1
        if self._canonical_yield_index >= len(self._canonical_items):
            self._canonical_item_available.clear()
        return item

    async def _next_item(self) -> None:
        assert self._response_iterator

        await self._propagate_cancellation_to_pending_work()

        if self._pending_interaction_task is not None:
            await self._poll_pending_interaction()
            return None

        if self._parser_queue and not self._parser_queue.empty():
            self._append_canonical_response_item(self._parser_queue.get())
            return None

        if self._pending_tool_batch_task is not None:
            await self._await_pending_tool_batch()
            return None

        if self._response_drained and (
            not self._calls.empty() or self._staged_tool_batch_present
        ):
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            calls = self._drain_tool_call_batch()
            if self._task_input_call is not None:
                execution = self._execution
                if execution is None or execution.interaction_runtime is None:
                    return None
                call = self._task_input_call
                self._task_input_call = None
                await self._start_task_input(call)
                return None
            if not calls:
                return None
            self._pending_tool_batch_task = create_task(
                self._execute_tool_call_batch(
                    calls,
                    confirm=True,
                    abort_on_reject=True,
                    emit_ready=False,
                )
            )
            await self._await_pending_tool_batch()
            return None

        # Wait until all results are collected
        if self._calls.empty() and not self._tool_result_outcomes.empty():
            completed_outcomes: list[_ToolExecutionOutcome] = []
            while not self._tool_result_outcomes.empty():
                completed_outcomes.append(self._tool_result_outcomes.get())

            tool_messages = []
            tool_outcomes = []
            for outcome in completed_outcomes:
                if not outcome.history_recorded:
                    self._record_tool_outcome(outcome.result)
                self._tool_context = outcome.context
                tool_result = outcome.result
                if not isinstance(
                    tool_result,
                    (ToolCallResult, ToolCallError, ToolCallDiagnostic),
                ):
                    continue
                tool_outcomes.append(tool_result)
                provider_result = self._provider_facing_tool_outcome(
                    tool_result
                )
                tool_messages.extend(
                    self._tool_observation_messages(
                        provider_result,
                        call=self._provider_facing_tool_call(outcome.call),
                        json_output=True,
                    )
                )

            if self._execution is not None and tool_messages:
                await self._execution.record_messages(tuple(tool_messages))
            if not self._should_continue_tool_cycle(
                tool_messages,
                tool_outcomes,
            ):
                if self._event_manager and not self._finished:
                    self._finished = True
                    await self._event_manager.trigger(
                        Event(type=EventType.END)
                    )
                await self._finalize_execution(
                    StreamItemKind.STREAM_COMPLETED,
                    output=self._current_response_answer_text(),
                )
                self._finish_canonical_stream(
                    self._execution_terminal_kind(
                        StreamItemKind.STREAM_COMPLETED
                    )
                )
                raise StopAsyncIteration

            assert self._input and (
                (
                    isinstance(self._input, list)
                    and isinstance(self._input[0], Message)
                )
                or isinstance(self._input, Message)
            )

            messages = (
                list(cast(list[Message], self._input))
                if isinstance(self._input, list)
                else [self._input]
            )

            messages.extend(tool_messages)
            self._input = cast(Input, messages)
            self._tool_context = self._new_tool_context(self._input)

            model_context = await self._make_child_context(messages)
            model_origin = model_context.execution_origin
            continuation_id = str(
                model_origin.model_call_id
                if model_origin is not None
                else uuid4()
            )
            continuation_item = self._append_canonical_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                continuation_id,
            )
            try:
                await self._raise_if_cancelled(finish_stream=False)
                await self._trigger_canonical_observability_event(
                    EventType.TOOL_MODEL_RUN,
                    continuation_item,
                )
                inner_response = await self._await_with_session_cancellation(
                    self._engine_agent(model_context)
                )
            except CancelledError as exc:
                self._append_canonical_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                    continuation_id,
                )
                try:
                    await self._finalize_execution(
                        StreamItemKind.STREAM_CANCELLED
                    )
                    self._finish_canonical_stream(
                        self._execution_terminal_kind(
                            StreamItemKind.STREAM_CANCELLED
                        )
                    )
                except BaseException as cleanup_failure:
                    self._attach_cleanup_failures(exc, [cleanup_failure])
                raise
            except Exception as exc:
                self._append_canonical_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_ERROR,
                    continuation_id,
                    data={
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                await self._finalize_execution(StreamItemKind.STREAM_ERRORED)
                self._finish_canonical_stream(
                    StreamItemKind.STREAM_ERRORED,
                    data={
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                raise
            assert inner_response
            assert isinstance(inner_response, TextGenerationResponse)
            await self._install_continuation_response(
                inner_response,
                continuation_id,
                activate=True,
            )

            return None

        try:
            response_item = self._response_iterator.__anext__()
            item = await self._await_with_session_cancellation(response_item)
            canonical_item = self._canonical_item_from_response_item(item)
            await self._process_canonical_response_item(canonical_item)
            task_input = self._classify_completed_task_input_boundary(
                canonical_item
            )
            if task_input is not None:
                execution = self._execution
                durable_runtime = execution is not None and isinstance(
                    execution.interaction_runtime,
                    DurableInteractionRuntime,
                )
                if not durable_runtime:
                    await self._response.aclose()
                    self._response_drained = True
                self._finish_active_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_COMPLETED
                )
                self._task_input_call = None
                await self._start_task_input(task_input)
        except StopAsyncIteration:
            self._response_drained = True
            continuation_item = self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            await self._trigger_canonical_observability_event(
                EventType.TOOL_MODEL_RESPONSE,
                continuation_item,
            )
            try:
                await self._flush_canonical_text_tool_call_parser_stage()
            except Exception as exc:
                self._discard_untrusted_response_tool_call_batch()
                self._finish_canonical_stream(
                    StreamItemKind.STREAM_ERRORED,
                    data={
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
                raise
            self._finalize_incomplete_canonical_tool_calls()
            if (
                not self._calls.empty()
                or self._staged_tool_batch_present
                or not self._tool_result_outcomes.empty()
            ):
                return await self._next_item()
            if self._event_manager and not self._finished:
                self._finished = True
                await self._event_manager.trigger(Event(type=EventType.END))

            await self._finalize_execution(
                StreamItemKind.STREAM_COMPLETED,
                output=self._current_response_answer_text(),
            )
            self._finish_canonical_stream(
                self._execution_terminal_kind(StreamItemKind.STREAM_COMPLETED)
            )
            raise
        except CancelledError as exc:
            await self._settle_cancellation_failure(exc)
            raise
        except ExecutionInputRequiredError:
            raise
        except Exception as exc:
            await self._finalize_execution(StreamItemKind.STREAM_ERRORED)
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            self._discard_untrusted_response_tool_call_batch()
            self._finish_canonical_stream(
                StreamItemKind.STREAM_ERRORED,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            raise

        return None

    async def _watch_session_cancellation(self) -> None:
        assert self._cancellation_checker is not None
        while True:
            await self._raise_if_cancelled(finish_stream=False)
            await sleep(self._CANCELLATION_POLL_INTERVAL_SECONDS)

    async def _await_with_session_cancellation(
        self,
        awaitable: Awaitable[_T],
    ) -> _T:
        if self._cancellation_checker is None:
            return await awaitable

        operation_task = ensure_future(awaitable)
        cancellation_task = create_task(self._watch_session_cancellation())
        try:
            done, _ = await wait(
                {operation_task, cancellation_task},
                return_when=FIRST_COMPLETED,
            )
            if cancellation_task in done:
                await cancellation_task
            assert operation_task in done
            return operation_task.result()
        except (CancelledError, Exception):
            if not operation_task.done():
                operation_task.cancel()
                await gather(operation_task, return_exceptions=True)
            raise
        finally:
            if not cancellation_task.done():
                cancellation_task.cancel()
                await gather(cancellation_task, return_exceptions=True)

    async def _propagate_cancellation_to_pending_work(self) -> None:
        if self._cancellation_checker is None:
            return
        if (
            self._pending_tool_batch_task is None
            and self._active_model_continuation_id is None
        ):
            return

        try:
            await self._raise_if_cancelled(finish_stream=False)
        except CancelledError as exc:
            await self._settle_cancellation_failure(exc)
            raise

    async def _await_pending_tool_batch(self) -> None:
        task = self._pending_tool_batch_task
        assert task is not None

        if task.done():
            await self._consume_pending_tool_batch(task)
            return

        item_task = create_task(self._canonical_item_available.wait())
        cancellation_task = (
            create_task(self._watch_session_cancellation())
            if self._cancellation_checker is not None
            else None
        )
        try:
            wait_tasks: set[Any] = {task, item_task}
            if cancellation_task is not None:
                wait_tasks.add(cancellation_task)
            done, pending = await wait(
                wait_tasks,
                return_when=FIRST_COMPLETED,
            )
            if cancellation_task is not None and cancellation_task in done:
                await cancellation_task
        except CancelledError as exc:
            item_task.cancel()
            await gather(item_task, return_exceptions=True)
            if cancellation_task is not None:
                cancellation_task.cancel()
                await gather(cancellation_task, return_exceptions=True)
            try:
                await self._cancel_pending_tool_batch()
                self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(exc, [cleanup_failure])
            raise

        if item_task in pending:
            item_task.cancel()
            await gather(item_task, return_exceptions=True)
        if cancellation_task is not None and cancellation_task in pending:
            cancellation_task.cancel()
            await gather(cancellation_task, return_exceptions=True)

        if task not in done:
            return

        if self._canonical_yield_index < len(self._canonical_items):
            return
        await self._consume_pending_tool_batch(task)

    async def _cancel_pending_tool_batch(self) -> None:
        task = self._pending_tool_batch_task
        if task is None:
            return
        await self._cancel_task_with_deadline(
            task,
            "pending tool batch",
        )
        if task.done() and self._pending_tool_batch_task is task:
            self._pending_tool_batch_task = None

    @staticmethod
    async def _cancel_task_with_deadline(
        task: Task[Any],
        stage: str,
    ) -> None:
        """Cancel and observe one owned task within the cleanup deadline."""
        if not task.done():
            task.cancel()
            await sleep(0)
        if not task.done():
            await wait({task}, timeout=_CLEANUP_TIMEOUT_SECONDS)
        if not task.done():
            raise TimeoutError(
                f"{stage} cleanup exceeded "
                f"{_CLEANUP_TIMEOUT_SECONDS:g} seconds"
            )
        try:
            task.result()
        except BaseException:
            return

    @staticmethod
    async def _await_cleanup_task_with_deadline(
        task: Task[None],
        stage: str,
    ) -> None:
        """Await one retryable cleanup task within a fixed deadline."""
        await sleep(0)
        if not task.done():
            await wait({task}, timeout=_CLEANUP_TIMEOUT_SECONDS)
        if not task.done():
            task.cancel()
            await sleep(0)
            raise TimeoutError(
                f"{stage} cleanup exceeded "
                f"{_CLEANUP_TIMEOUT_SECONDS:g} seconds"
            )
        task.result()

    @staticmethod
    def _observe_cleanup_task(task: Task[Any]) -> None:
        """Observe an explicitly retained cleanup task when it finishes."""
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            return

    async def _settle_execution_with_deadline(
        self,
        *,
        cancelled: bool,
    ) -> tuple[BaseException, ...]:
        """Join retryable execution settlement within the cleanup budget."""
        execution = self._execution
        assert execution is not None
        task = self._execution_cleanup_task
        if task is None:
            task = create_task(
                execution.settle_provider_exit(cancelled=cancelled)
            )
            task.add_done_callback(self._observe_cleanup_task)
            self._execution_cleanup_task = task
        await sleep(0)
        if not task.done():
            await wait({task}, timeout=_CLEANUP_TIMEOUT_SECONDS)
        if not task.done():
            task.cancel()
            await sleep(0)
            if task.done() and self._execution_cleanup_task is task:
                self._execution_cleanup_task = None
            raise TimeoutError(
                "execution terminalization cleanup exceeded "
                f"{_CLEANUP_TIMEOUT_SECONDS:g} seconds"
            )
        if self._execution_cleanup_task is task:
            self._execution_cleanup_task = None
        return task.result()

    async def _cancel_active_model_continuation_response(self) -> None:
        if self._active_model_continuation_id is None:
            return
        await self._cancel_provider_response()

    def _provider_response_cleanup_is_complete(self) -> bool:
        """Return cleanup state without masking an existing stream failure."""
        return bool(getattr(self._response, "cleanup_complete", False))

    async def _cancel_provider_response(self) -> None:
        """Cancel and close the current provider response idempotently."""
        if self._provider_cleanup_complete:
            return
        try:
            await self._response.cancel()
        finally:
            try:
                await self._response.aclose()
            finally:
                self._provider_cleanup_complete = (
                    self._provider_response_cleanup_is_complete()
                )

    async def _close_provider_response(self) -> None:
        """Retry closing the active provider without cancelling it again."""
        if self._provider_cleanup_complete:
            return
        try:
            await self._response.aclose()
        finally:
            self._provider_cleanup_complete = (
                self._provider_response_cleanup_is_complete()
            )

    async def _converge_stream_cancellation(self) -> None:
        """Converge every cancellation exit on one terminal execution."""
        task = self._cancellation_cleanup_task
        if task is None:
            task = create_task(self._run_stream_cancellation_cleanup())
            self._cancellation_cleanup_task = task
        try:
            await shield(task)
        finally:
            if task.done() and self._cancellation_cleanup_task is task:
                self._cancellation_cleanup_task = None

    async def _settle_cancellation_failure(
        self,
        primary_failure: CancelledError,
    ) -> None:
        """Preserve cancellation and avoid duplicate same-boundary retries."""
        notes = getattr(primary_failure, "__notes__", ())
        if any(
            note.startswith("post-provider cleanup failure:") for note in notes
        ):
            self._provider_cleanup_complete = (
                self._provider_response_cleanup_is_complete()
            )
            return
        try:
            await self._converge_stream_cancellation()
        except BaseException as cleanup_failure:
            self._attach_cleanup_failures(
                primary_failure,
                [cleanup_failure],
            )

    async def _run_stream_cancellation_cleanup(self) -> None:
        """Run one coalesced provider and branch cancellation attempt."""
        cleanup_failures: list[BaseException] = []
        try:
            await self._cancel_pending_tool_batch()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            await self._cancel_provider_response()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            await self._cancel_pending_interaction()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            await self._finalize_execution(StreamItemKind.STREAM_CANCELLED)
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            self._discard_untrusted_response_tool_call_batch()
            self._finish_canonical_stream(
                self._execution_terminal_kind(StreamItemKind.STREAM_CANCELLED)
            )
        except BaseException as exc:
            cleanup_failures.append(exc)
        self._raise_cleanup_failures(cleanup_failures)

    async def _converge_stream_error_cleanup(self) -> None:
        """Join one retryable provider and branch cleanup after an error."""
        task = self._cancellation_cleanup_task
        if task is None:
            task = create_task(self._run_stream_error_cleanup())
            self._cancellation_cleanup_task = task
        try:
            await shield(task)
        finally:
            if task.done() and self._cancellation_cleanup_task is task:
                self._cancellation_cleanup_task = None

    async def _run_stream_error_cleanup(self) -> None:
        """Retry incomplete errored provider and branch cleanup once."""
        cleanup_failures: list[BaseException] = []
        try:
            await self._cancel_pending_tool_batch()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            await self._close_provider_response()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            await self._finalize_execution(StreamItemKind.STREAM_ERRORED)
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            await self._cancel_pending_interaction()
        except BaseException as exc:
            cleanup_failures.append(exc)
        try:
            self._discard_untrusted_response_tool_call_batch()
            self._finish_canonical_stream(
                self._execution_terminal_kind(StreamItemKind.STREAM_ERRORED)
            )
        except BaseException as exc:
            cleanup_failures.append(exc)
        self._raise_cleanup_failures(cleanup_failures)

    @staticmethod
    def _attach_cleanup_failures(
        primary_failure: BaseException,
        cleanup_failures: list[BaseException],
    ) -> None:
        """Attach cleanup failures without replacing the stream primary."""
        seen_failure_ids = {id(primary_failure)}
        for cleanup_failure in cleanup_failures:
            if id(cleanup_failure) in seen_failure_ids:
                continue
            seen_failure_ids.add(id(cleanup_failure))
            primary_failure.add_note(
                "post-provider cleanup failure: "
                f"{cleanup_failure.__class__.__name__}: "
                f"{cleanup_failure}"
            )

    @classmethod
    def _raise_cleanup_failures(
        cls,
        cleanup_failures: list[BaseException],
    ) -> None:
        """Raise the first cleanup failure with later failures as notes."""
        if not cleanup_failures:
            return
        primary_failure = cleanup_failures[0]
        cls._attach_cleanup_failures(
            primary_failure,
            cleanup_failures[1:],
        )
        raise primary_failure

    def _execution_terminal_kind(
        self,
        proposed_kind: StreamItemKind,
    ) -> StreamItemKind:
        """Return the canonical terminal matching actual execution state."""
        execution = self._execution
        if execution is None:
            return proposed_kind
        return {
            AgentExecutionStatus.COMPLETED: StreamItemKind.STREAM_COMPLETED,
            AgentExecutionStatus.CANCELLED: StreamItemKind.STREAM_CANCELLED,
            AgentExecutionStatus.ERRORED: StreamItemKind.STREAM_ERRORED,
        }.get(execution.status, proposed_kind)

    def _raise_if_completion_lost(self) -> None:
        """Reject a success projection when another terminal owner won."""
        self._raise_if_terminal_failure()
        execution = self._execution
        execution_status = execution.status if execution is not None else None
        canonical_terminal = self._canonical_stream_terminal
        if (
            execution_status in (None, AgentExecutionStatus.COMPLETED)
            and canonical_terminal is StreamTerminalOutcome.COMPLETED
        ):
            return
        raise RuntimeError("execution did not complete successfully")

    def _raise_if_terminal_failure(self) -> None:
        """Surface an already settled non-success terminal before reuse."""
        execution = self._execution
        execution_status = execution.status if execution is not None else None
        canonical_terminal = self._canonical_stream_terminal
        if (
            execution_status is AgentExecutionStatus.CANCELLED
            or canonical_terminal is StreamTerminalOutcome.CANCELLED
        ):
            raise CancelledError("execution was cancelled before completion")
        if (
            execution_status is AgentExecutionStatus.ERRORED
            or canonical_terminal is StreamTerminalOutcome.ERRORED
        ):
            raise RuntimeError("execution failed before completion")

    async def _install_continuation_response(
        self,
        response: TextGenerationResponse,
        continuation_id: str,
        *,
        activate: bool,
    ) -> None:
        """Install one continuation locally before acknowledging handoff."""
        try:
            self._model_responses.append(response)
            if activate:
                self._response = response
                self._response_drained = False
            self._set_active_model_continuation(continuation_id)
            if activate:
                self._prepare_iteration(reset_yield_index=False)
            self._engine_agent.acknowledge_provider_handoff(response)
        except BaseException as primary_failure:
            await self._settle_continuation_handoff_failure(
                response,
                primary_failure,
            )
            raise

    async def _settle_continuation_handoff_failure(
        self,
        response: TextGenerationResponse,
        primary_failure: BaseException,
    ) -> None:
        """Settle either owner after continuation installation fails."""
        execution = self._execution
        cancelled = isinstance(
            primary_failure,
            (CancelledError, KeyboardInterrupt, CommandAbortException),
        )

        async def capture(
            operation: Awaitable[Any],
        ) -> tuple[BaseException, ...]:
            try:
                result = await operation
            except BaseException as error:
                return (error,)
            if isinstance(result, tuple) and all(
                isinstance(item, BaseException) for item in result
            ):
                return cast(tuple[BaseException, ...], result)
            return ()

        cancel_task = create_task(capture(response.cancel()))
        settle_operation: Awaitable[Any] = (
            execution.settle_provider_exit(cancelled=cancelled)
            if execution is not None
            else sleep(0)
        )
        settle_task = create_task(capture(settle_operation))
        drain_task = create_task(
            capture(
                self._engine_agent.drain_pending_provider_cleanups(
                    execution,
                    abandon_unclaimed=True,
                )
            )
        )
        await sleep(0)
        close_task = create_task(capture(response.aclose()))
        results = await gather(
            cancel_task,
            close_task,
            settle_task,
            drain_task,
        )
        self._attach_cleanup_failures(
            primary_failure,
            [failure for result in results for failure in result],
        )

    async def _settle_stream_failure(
        self,
        primary_failure: BaseException,
    ) -> None:
        """Converge one provider exit while preserving its primary error."""
        execution = self._execution
        if (
            execution is not None
            and execution.status is AgentExecutionStatus.CANCELLED
        ):
            self._provider_cleanup_complete = (
                self._provider_response_cleanup_is_complete()
            )
            return
        if isinstance(primary_failure, KeyboardInterrupt):
            try:
                await self._converge_stream_cancellation()
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(
                    primary_failure,
                    [cleanup_failure],
                )
            return

        cleanup_failures: list[BaseException] = []
        try:
            await self._finalize_execution(StreamItemKind.STREAM_ERRORED)
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
        self._provider_cleanup_complete = (
            self._provider_response_cleanup_is_complete()
        )
        try:
            await self._cancel_pending_interaction()
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
        try:
            self._discard_untrusted_response_tool_call_batch()
            self._finish_canonical_stream(
                self._execution_terminal_kind(StreamItemKind.STREAM_ERRORED),
                data={
                    "error_type": primary_failure.__class__.__name__,
                    "message": str(primary_failure),
                },
            )
        except BaseException as cleanup_failure:
            cleanup_failures.append(cleanup_failure)
        self._attach_cleanup_failures(primary_failure, cleanup_failures)

    async def _consume_pending_tool_batch(
        self,
        task: Task[list[_ToolExecutionOutcome]],
    ) -> None:
        assert self._pending_tool_batch_task is task
        assert task.done()
        self._pending_tool_batch_task = None
        try:
            outcomes = task.result()
        except CancelledError as exc:
            try:
                await self._finalize_execution(StreamItemKind.STREAM_CANCELLED)
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(exc, [cleanup_failure])
            raise
        except CommandAbortException as exc:
            try:
                await self._finalize_execution(StreamItemKind.STREAM_CANCELLED)
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(exc, [cleanup_failure])
            if self._execution is None:
                raise
            return
        except Exception as exc:
            try:
                await self._finalize_execution(StreamItemKind.STREAM_ERRORED)
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(exc, [cleanup_failure])
            raise
        ordered = sorted(outcomes, key=lambda outcome: outcome.planned_index)
        for outcome in ordered:
            self._record_tool_outcome(outcome.result)
            self._put_staging_item(
                self._tool_result_outcomes,
                replace(outcome, history_recorded=True),
                "tool result outcome",
            )

    async def _react(
        self, response: TextGenerationResponse, output: str | None = None
    ) -> str:
        if self._event_manager:
            response.add_done_callback(self._on_consumed)

        structured_calls: list[ToolCall] = []
        if output is None:
            text, structured_calls = await self._response_text_and_calls(
                response
            )
        else:
            text = output

        if self._tool_context is None:
            self._tool_context = self._new_tool_context(self._input)

        current_response = response
        delta = text
        attached_answer_prefixes: list[str] = []
        while True:
            structured_batch = bool(structured_calls)
            calls = (
                structured_calls
                if structured_batch
                else (
                    self._capability_catalog.get_calls(delta)
                    if self._capability_catalog
                    else None
                )
            )
            if (
                not calls
                and not self._staged_tool_batch_present
                and self._task_input_call is None
            ):
                break
            classified_calls = (
                self._classify_complete_tool_call_batch(
                    list(calls or []),
                    text_originated=not structured_batch,
                )
                if self._task_input_call is None
                else []
            )
            if self._task_input_call is not None:
                execution = self._execution
                if execution is None or execution.interaction_runtime is None:
                    break
                call = self._task_input_call
                self._task_input_call = None
                if delta:
                    attached_answer_prefixes.append(delta)
                if isinstance(
                    execution.interaction_runtime,
                    DurableInteractionRuntime,
                ):
                    await self._start_task_input(call)
                    raise AssertionError(
                        "durable task input must suspend the execution"
                    )
                current_response = await self._run_attached_task_input(call)
                self._response = current_response
                (
                    new_text,
                    structured_calls,
                ) = await self._response_text_and_calls(current_response)
                delta = new_text
                continue
            if not classified_calls:
                break
            await self._trigger_derived_canonical_observability_event(
                EventType.TOOL_DETECT,
                StreamItemKind.STREAM_DIAGNOSTIC,
                summary={"stage": "tool_detection"},
            )

            results: list[ToolCallOutcome] = []
            pending_calls = classified_calls
            while pending_calls:
                batch, pending_calls = self._split_tool_call_batch(
                    pending_calls
                )
                outcomes = await self._await_with_session_cancellation(
                    self._execute_tool_call_batch(
                        batch,
                        confirm=True,
                        abort_on_reject=False,
                        emit_ready=True,
                    )
                )
                await self._raise_if_cancelled(finish_stream=False)
                for outcome in sorted(
                    outcomes, key=lambda item: item.planned_index
                ):
                    self._record_tool_outcome(outcome.result)
                    self._tool_context = outcome.context
                    if outcome.result is not None:
                        results.append(outcome.result)

            next_response = await self._react_process(delta, results)
            if next_response is None:
                break
            current_response = next_response
            self._response = current_response
            new_text, structured_calls = await self._response_text_and_calls(
                current_response
            )
            delta = new_text

        self._response = current_response
        return "".join((*attached_answer_prefixes, delta))

    async def _start_task_input(
        self,
        call: TaskInputCapabilityCall,
    ) -> None:
        """Start attached handling or stage a durable suspension."""
        execution = self._execution
        if execution is None or execution.interaction_runtime is None:
            raise RuntimeError(
                "task input requires an explicit interaction runtime"
            )
        assert self._pending_interaction_task is None
        fingerprint = sha256(
            repr((call.mode, call.reason, call.questions)).encode()
        ).hexdigest()
        assistant_text = self._current_response_answer_text()
        assistant_message = Message(
            role=MessageRole.ASSISTANT,
            content=assistant_text or None,
            tool_calls=[
                MessageToolCall(
                    id=str(call.call_id),
                    name=call.provider_name,
                    arguments=normalize_tool_arguments(call.arguments),
                )
            ],
        )
        await execution.begin_interaction(
            fingerprint,
            call,
            assistant_message,
        )
        try:
            self._pending_interaction_call = call
            self._pending_interaction_assistant_text = assistant_text
            self._pending_interaction_published = False
            runtime = execution.interaction_runtime
            if isinstance(runtime, DurableInteractionRuntime):
                try:
                    staging = self._durable_staging_context(call, execution)
                    request_spec = InteractionBrokerRequest(
                        actor=runtime.actor,
                        origin=execution.origin,
                        mode=call.mode,
                        reason=call.reason,
                        questions=call.questions,
                    )
                    durable = await runtime.stager(
                        request_spec,
                        execution=execution,
                        response=self,
                        stream_sequence=self._canonical_sequence,
                        staging=staging,
                    )
                    self._validate_durable_staging(
                        request_spec,
                        durable,
                        staging=staging,
                    )
                except BaseException as error:
                    try:
                        await self._response.aclose()
                    except BaseException as cleanup_failure:
                        self._attach_cleanup_failures(
                            error,
                            [cleanup_failure],
                        )
                    raise
                await self._response.aclose()
                self._response_drained = True
                request = durable.command.request
                required = InputRequiredResult(
                    request_id=request.request_id,
                    continuation_id=request.continuation_id,
                    detached_resumption_available=True,
                )
                await execution.stage_durable_input_required(
                    request,
                    required,
                )
                self._active_interaction_request = request
                self._input_required_result = required
                terminal_index = len(self._canonical_items)
                self._finish_canonical_stream(
                    StreamItemKind.STREAM_INPUT_REQUIRED,
                    correlation=self._interaction_correlation(request),
                )
                terminal_item = self._canonical_items[terminal_index]
                assert (
                    terminal_item.kind is StreamItemKind.STREAM_INPUT_REQUIRED
                )
                if self._event_manager is not None:
                    await self._event_manager.trigger_stream_item(
                        terminal_item
                    )
                self._pending_interaction_call = None
                self._pending_interaction_assistant_text = ""
                raise ExecutionInputRequiredError(
                    required,
                    request=request,
                    durable=durable,
                )
            assert isinstance(runtime, AttachedInteractionRuntime)

            async def attached_handler(
                context: InputHandlerContext,
            ) -> _InputHandlerOutcome:
                await self._publish_interaction_wait(context.request)
                return await runtime.handler(context)

            broker_request = InteractionBrokerRequest(
                actor=runtime.actor,
                origin=execution.origin,
                mode=call.mode,
                reason=call.reason,
                questions=call.questions,
                handler=attached_handler,
            )
            broker = execution.interaction_broker
            assert broker is not None
            self._pending_interaction_task = create_task(
                broker.request(broker_request),
                name=f"agent-input-{call.call_id}",
            )
        except ExecutionInputRequiredError:
            raise
        except Exception:
            self._pending_interaction_call = None
            self._pending_interaction_assistant_text = ""
            self._pending_interaction_published = False
            if execution.status is AgentExecutionStatus.PREPARING_INPUT:
                await execution.abandon_interaction()
            raise

    @staticmethod
    def _validate_durable_staging(
        request_spec: InteractionBrokerRequest,
        durable: DurableInteractionSuspension,
        *,
        staging: DurableInteractionStagingContext,
    ) -> None:
        """Reject staging output that changes the reserved request."""
        if type(durable) is not DurableInteractionSuspension:
            raise TypeError(
                "durable interaction stager returned an invalid suspension"
            )
        command = durable.command
        request = command.request
        if command.actor != request_spec.actor:
            raise RuntimeError("durable interaction actor changed in staging")
        expected = (
            request_spec.origin,
            request_spec.mode,
            request_spec.reason,
            request_spec.questions,
            request_spec.continuation_ttl_seconds,
            request_spec.advisory_wait_seconds,
        )
        actual = (
            request.origin,
            request.mode,
            request.reason,
            request.questions,
            request.continuation_ttl_seconds,
            request.advisory_wait_seconds,
        )
        if actual != expected or request.state is not RequestState.CREATED:
            raise RuntimeError(
                "durable interaction request changed in staging"
            )
        continuation = durable.continuation
        if (
            request.continuation_id != staging.continuation_id
            or continuation.provider_snapshot != staging.provider_snapshot
            or continuation.revision_binding != staging.revision_binding
            or continuation.provider_call_correlation_id
            != staging.provider_call_correlation_id
            or continuation.provider_call_id
            != staging.provider_snapshot.model_call_id
        ):
            raise RuntimeError(
                "durable continuation changed provider replay state"
            )

    def _durable_staging_context(
        self,
        call: TaskInputCapabilityCall,
        execution: AgentExecution,
    ) -> DurableInteractionStagingContext:
        """Export one provider-owned replay snapshot before source close."""
        capability = self._capability_catalog
        if capability is None:
            raise RuntimeError("durable staging requires a capability catalog")
        support = capability.support
        binding = capability.revision_binding
        registry = support.continuation_snapshot_codec_registry
        codec = support.continuation_snapshot_codec
        adapter = self._response.continuation_snapshot_adapter
        if (
            binding is None
            or registry is None
            or codec is None
            or adapter is None
        ):
            raise RuntimeError(
                "durable staging requires registered provider replay"
            )
        continuation_id = ContinuationId(f"continuation-{uuid4()}")
        dispatch_id = derive_continuation_dispatch_id(continuation_id)
        idempotency_key = derive_provider_idempotency_key(
            continuation_id,
            dispatch_id,
        )
        correlation_id = str(call.call_id)
        snapshot = adapter.export_continuation_snapshot(
            revision_binding=binding,
            model_call_id=execution.origin.model_call_id,
            provider_idempotency_key=idempotency_key,
            provider_call_correlation_id=correlation_id,
        )
        adapter.validate_continuation_snapshot_call(
            snapshot,
            expected_binding=binding,
            provider_call_correlation_id=correlation_id,
            expected_provider_name=call.provider_name,
            expected_arguments=call.arguments,
        )
        if snapshot.model_call_id != execution.origin.model_call_id:
            raise RuntimeError(
                "provider snapshot changed the execution model call"
            )
        return DurableInteractionStagingContext(
            task_input_call=call,
            continuation_id=continuation_id,
            dispatch_id=dispatch_id,
            revision_binding=binding,
            codec_registry=registry,
            codec=codec,
            provider_snapshot=snapshot,
            provider_idempotency_key=idempotency_key,
            provider_call_correlation_id=correlation_id,
        )

    async def _run_attached_task_input(
        self,
        call: TaskInputCapabilityCall,
    ) -> TextGenerationResponse:
        """Await one attached interaction for a materialized response."""
        execution = self._execution
        assert execution is not None
        assert isinstance(
            execution.interaction_runtime,
            AttachedInteractionRuntime,
        )
        try:
            await self._start_task_input(call)
            task = self._pending_interaction_task
            assert task is not None
            broker_wait: Awaitable[InteractionRequestResult] = (
                shield(task)
                if self._cancellation_checker is not None
                else task
            )
            result = await self._await_with_session_cancellation(broker_wait)
            response = await self._finish_task_input(
                result,
                raise_on_noncompletion=True,
            )
        except CancelledError as exc:
            try:
                await self._cancel_pending_interaction()
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(exc, [cleanup_failure])
            raise
        assert response is not None
        return response

    async def _poll_pending_interaction(self) -> None:
        """Expose lifecycle items while the attached handler remains open."""
        task = self._pending_interaction_task
        assert task is not None
        if not task.done():
            item_available = create_task(
                self._canonical_item_available.wait(),
                name="agent-input-stream-item",
            )
            cancellation = (
                create_task(
                    self._watch_session_cancellation(),
                    name="agent-input-cancellation",
                )
                if self._cancellation_checker is not None
                else None
            )
            waits: set[Task[Any]] = {task, item_available}
            if cancellation is not None:
                waits.add(cancellation)
            try:
                done, _ = await wait(waits, return_when=FIRST_COMPLETED)
            except CancelledError as exc:
                item_available.cancel()
                if cancellation is not None:
                    cancellation.cancel()
                await gather(
                    *(
                        item
                        for item in (item_available, cancellation)
                        if item is not None
                    ),
                    return_exceptions=True,
                )
                try:
                    await self._cancel_pending_interaction()
                except BaseException as cleanup_failure:
                    self._attach_cleanup_failures(exc, [cleanup_failure])
                raise
            if cancellation is not None and cancellation in done:
                item_available.cancel()
                await gather(item_available, return_exceptions=True)
                try:
                    cancellation.result()
                except CancelledError as exc:
                    primary_cancellation = exc
                else:
                    primary_cancellation = CancelledError()
                try:
                    await self._cancel_pending_interaction()
                except BaseException as cleanup_failure:
                    self._attach_cleanup_failures(
                        primary_cancellation,
                        [cleanup_failure],
                    )
                raise primary_cancellation
            item_available.cancel()
            if cancellation is not None:
                cancellation.cancel()
            await gather(
                *(
                    item
                    for item in (item_available, cancellation)
                    if item is not None
                ),
                return_exceptions=True,
            )
            if task not in done:
                return
        result = task.result()
        response = await self._finish_task_input(
            result,
            raise_on_noncompletion=False,
        )
        if response is None:
            return
        self._response = response
        self._response_iterator = aiter(response)
        self._response_drained = False
        self._begin_tool_call_lifecycle_response()

    async def _finish_task_input(
        self,
        broker_result: InteractionRequestResult,
        *,
        raise_on_noncompletion: bool,
    ) -> TextGenerationResponse | None:
        """Apply exactly one authoritative broker delivery."""
        task = self._pending_interaction_task
        call = self._pending_interaction_call
        assistant_text = self._pending_interaction_assistant_text
        self._pending_interaction_task = None
        self._pending_interaction_call = None
        self._pending_interaction_assistant_text = ""
        assert task is not None and call is not None
        if not isinstance(
            broker_result.create_result,
            CreateInteractionApplied,
        ):
            execution = self._execution
            assert execution is not None
            abandon = getattr(execution, "abandon_interaction", None)
            if callable(abandon):
                await abandon()
            self._pending_interaction_published = False
            raise RuntimeError("interaction admission was rejected")
        initial_request = broker_result.create_result.record.request
        await self._publish_interaction_wait(initial_request)
        delivery = broker_result.delivery
        assert delivery is not None
        request = delivery.record.request
        self._active_interaction_request = request
        if request.state is RequestState.PENDING:
            required = InputRequiredResult(
                request_id=request.request_id,
                continuation_id=request.continuation_id,
                detached_resumption_available=False,
            )
            execution = self._execution
            assert execution is not None
            await execution.mark_input_required(required)
            correlation = self._interaction_correlation(request)
            self._input_required_result = required
            terminal_index = len(self._canonical_items)
            self._finish_canonical_stream(
                StreamItemKind.STREAM_INPUT_REQUIRED,
                correlation=correlation,
            )
            terminal_item = self._canonical_items[terminal_index]
            assert terminal_item.kind is StreamItemKind.STREAM_INPUT_REQUIRED
            if self._event_manager is not None:
                await self._event_manager.trigger_stream_item(terminal_item)
            self._pending_interaction_published = False
            if raise_on_noncompletion:
                raise ExecutionInputRequiredError(
                    required,
                    request=request,
                )
            return None

        await self._append_interaction_terminal(request)
        outcome = project_resolution_to_model(
            request,
            containing_run_exists=True,
        )
        if isinstance(outcome, TerminateInputContinuation):
            execution = self._execution
            assert execution is not None
            await execution.record_interaction_termination(request, outcome)
            self._execution_terminated = True
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            self._pending_interaction_published = False
            if raise_on_noncompletion:
                raise ExecutionTerminatedError(outcome)
            return None
        assert isinstance(outcome, ResumeInputContinuation)
        try:
            return await self._resume_after_task_input(
                call,
                request,
                outcome.result,
                assistant_text=assistant_text,
            )
        finally:
            self._pending_interaction_published = False

    async def _publish_interaction_wait(self, request: InputRequest) -> None:
        """Publish authoritative identity before awaiting the real handler."""
        if self._pending_interaction_published:
            return
        execution = self._execution
        assert execution is not None
        self._active_interaction_request = request
        await execution.mark_interaction_pending(request)
        self._pending_interaction_published = True
        correlation = self._interaction_correlation(request)
        for kind in (
            StreamItemKind.INTERACTION_CREATED,
            StreamItemKind.INTERACTION_PENDING,
        ):
            item = self._append_canonical_item(kind, correlation=correlation)
            if self._event_manager is not None and item is not None:
                await self._event_manager.trigger_stream_item(item)

    async def _append_interaction_terminal(
        self,
        request: InputRequest,
    ) -> None:
        kinds = {
            RequestState.ANSWERED: StreamItemKind.INTERACTION_ANSWERED,
            RequestState.DECLINED: StreamItemKind.INTERACTION_DECLINED,
            RequestState.CANCELLED: StreamItemKind.INTERACTION_CANCELLED,
            RequestState.TIMED_OUT: StreamItemKind.INTERACTION_TIMED_OUT,
            RequestState.UNAVAILABLE: StreamItemKind.INTERACTION_UNAVAILABLE,
            RequestState.EXPIRED: StreamItemKind.INTERACTION_EXPIRED,
            RequestState.SUPERSEDED: StreamItemKind.INTERACTION_SUPERSEDED,
        }
        kind = kinds.get(request.state)
        if kind is None:
            raise RuntimeError("broker returned a nonterminal interaction")
        item = self._append_canonical_item(
            kind,
            correlation=self._interaction_correlation(request),
        )
        if self._event_manager is not None and item is not None:
            await self._event_manager.trigger_stream_item(item)

    async def _resume_after_task_input(
        self,
        call: TaskInputCapabilityCall,
        request: InputRequest,
        result: InputModelResult,
        *,
        assistant_text: str,
    ) -> TextGenerationResponse:
        """Append one correlated result and dispatch the next model turn."""
        capability = self._capability_catalog
        execution = self._execution
        assert capability is not None and execution is not None
        correlated = capability.project_result(call, result)
        arguments = normalize_tool_arguments(call.arguments)
        messages = (
            Message(
                role=MessageRole.ASSISTANT,
                content=assistant_text or None,
                tool_calls=[
                    MessageToolCall(
                        id=str(call.call_id),
                        name=call.provider_name,
                        arguments=arguments,
                    )
                ],
            ),
            correlated.tool_result_message(call),
        )
        committed = await execution.record_interaction_result(
            request,
            result,
            messages,
        )
        if not committed:
            raise RuntimeError("interaction continuation was already applied")
        self._input = cast(Input, list(execution.messages))
        self._tool_context = self._new_tool_context(self._input)
        context = await self._make_child_context(self._input)
        origin = context.execution_origin
        assert origin is not None
        continuation_id = str(origin.model_call_id)
        continuation_item = self._append_canonical_model_continuation(
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            continuation_id,
        )
        await self._trigger_canonical_observability_event(
            EventType.TOOL_MODEL_RUN,
            continuation_item,
        )
        try:
            response = await self._await_with_session_cancellation(
                self._engine_agent(context)
            )
        except CancelledError:
            self._append_canonical_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                continuation_id,
            )
            raise
        except Exception as exc:
            self._append_canonical_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                continuation_id,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            await self._finalize_execution(StreamItemKind.STREAM_ERRORED)
            self._finish_canonical_stream(
                StreamItemKind.STREAM_ERRORED,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            raise
        assert isinstance(response, TextGenerationResponse)
        await self._install_continuation_response(
            response,
            continuation_id,
            activate=False,
        )
        return response

    def _interaction_correlation(
        self,
        request: InputRequest,
    ) -> StreamItemCorrelation:
        origin = request.origin
        return StreamItemCorrelation(
            request_id=request.request_id,
            continuation_id=request.continuation_id,
            task_id=(
                str(origin.task_id) if origin.task_id is not None else None
            ),
            agent_id=origin.agent_id,
            branch_id=origin.branch_id,
            parent_branch_id=origin.parent_branch_id,
        )

    async def _cancel_pending_interaction(self) -> None:
        """Cancel broker wait and containing branch exactly once."""
        task = self._pending_interaction_task
        cleanup_failures: list[BaseException] = []
        execution = self._execution
        if execution is None or execution.interaction_runtime is None:
            if task is not None:
                try:
                    await self._cancel_task_with_deadline(
                        task,
                        "pending interaction",
                    )
                except BaseException as error:
                    cleanup_failures.append(error)
                if task.done() and self._pending_interaction_task is task:
                    self._pending_interaction_task = None
                    self._pending_interaction_call = None
                    self._pending_interaction_assistant_text = ""
                    self._pending_interaction_published = False
            self._raise_cleanup_failures(cleanup_failures)
            return
        active_request = (
            execution.pending_request or self._active_interaction_request
        )
        try:
            cleanup_failures.extend(
                await self._settle_execution_with_deadline(cancelled=True)
            )
        except BaseException as error:
            cleanup_failures.append(error)
        if active_request is not None:
            try:
                await self._append_interaction_cancellation_if_open(
                    active_request
                )
            except BaseException as error:
                cleanup_failures.append(error)
        try:
            self._finish_canonical_stream(
                self._execution_terminal_kind(StreamItemKind.STREAM_CANCELLED)
            )
        except BaseException as error:
            cleanup_failures.append(error)
        try:
            await self._finalize_interaction_cleanup(execution)
        except BaseException as error:
            cleanup_failures.append(error)
        if task is not None and self._interaction_cleanup_complete:
            try:
                await self._cancel_task_with_deadline(
                    task,
                    "pending interaction",
                )
            except BaseException as error:
                cleanup_failures.append(error)
            if task.done() and self._pending_interaction_task is task:
                self._pending_interaction_task = None
                self._pending_interaction_call = None
                self._pending_interaction_assistant_text = ""
                self._pending_interaction_published = False
        self._raise_cleanup_failures(cleanup_failures)

    async def _finalize_interaction_cleanup(
        self,
        execution: AgentExecution,
    ) -> None:
        """Run or join one retryable branch-cleanup attempt."""
        if self._interaction_cleanup_complete:
            return
        task = self._interaction_cleanup_task
        if task is None:
            task = create_task(self._run_interaction_cleanup(execution))
            self._interaction_cleanup_task = task
        try:
            await self._await_cleanup_task_with_deadline(
                task,
                "interaction branch",
            )
        finally:
            if task.done() and self._interaction_cleanup_task is task:
                self._interaction_cleanup_task = None

    async def _run_interaction_cleanup(
        self,
        execution: AgentExecution,
    ) -> None:
        """Terminalize one attached interaction scope and claim cleanup."""
        runtime = execution.interaction_runtime
        assert runtime is not None
        if isinstance(runtime, DurableInteractionRuntime):
            await execution.claim_cleanup()
            self._interaction_cleanup_complete = True
            return
        assert isinstance(runtime, AttachedInteractionRuntime)
        origin = execution.origin
        broker = execution.interaction_broker
        assert broker is not None
        await broker.cancel_scope(
            TerminalizeInteractionScopeCommand(
                actor=runtime.actor,
                scope=InteractionExecutionScope(
                    run_id=origin.run_id,
                    branch_id=origin.branch_id,
                ),
                provenance=AnswerProvenance.EXTERNAL_CONTROLLER,
            )
        )
        await execution.claim_cleanup()
        self._interaction_cleanup_complete = True

    async def _append_interaction_cancellation_if_open(
        self,
        request: InputRequest,
    ) -> None:
        """Close one published canonical interaction before stream cancel."""
        terminal_kinds = {
            StreamItemKind.INTERACTION_ANSWERED,
            StreamItemKind.INTERACTION_DECLINED,
            StreamItemKind.INTERACTION_CANCELLED,
            StreamItemKind.INTERACTION_TIMED_OUT,
            StreamItemKind.INTERACTION_UNAVAILABLE,
            StreamItemKind.INTERACTION_EXPIRED,
            StreamItemKind.INTERACTION_SUPERSEDED,
        }
        request_id = request.request_id
        if any(
            item.kind in terminal_kinds
            and item.correlation.request_id == request_id
            for item in self._canonical_items
        ):
            return
        created = any(
            item.kind is StreamItemKind.INTERACTION_CREATED
            and item.correlation.request_id == request_id
            for item in self._canonical_items
        )
        if not created:
            return
        pending = any(
            item.kind is StreamItemKind.INTERACTION_PENDING
            and item.correlation.request_id == request_id
            for item in self._canonical_items
        )
        if not pending:
            pending_item = self._append_canonical_item(
                StreamItemKind.INTERACTION_PENDING,
                correlation=self._interaction_correlation(request),
            )
            if self._event_manager is not None and pending_item is not None:
                await self._event_manager.trigger_stream_item(pending_item)
        item = self._append_canonical_item(
            StreamItemKind.INTERACTION_CANCELLED,
            correlation=self._interaction_correlation(request),
        )
        if self._event_manager is not None and item is not None:
            await self._event_manager.trigger_stream_item(item)

    async def _execute_tool_call_batch(
        self,
        calls: list[ToolCall],
        *,
        confirm: bool,
        abort_on_reject: bool,
        emit_ready: bool,
    ) -> list[_ToolExecutionOutcome]:
        assert calls
        calls = [self._tool_call_with_canonical_id(call) for call in calls]
        if len(calls) == 1:
            return [
                await self._execute_tool_call_with_lifecycle(
                    calls[0],
                    confirm=confirm,
                    abort_on_reject=abort_on_reject,
                    emit_ready=emit_ready,
                    planned_index=0,
                )
            ]

        confirmation_diagnostics = await self._confirm_parallel_tool_calls(
            calls,
            abort_on_reject=abort_on_reject,
            emit_ready=emit_ready,
        )
        if confirmation_diagnostics:
            return confirmation_diagnostics

        tasks: list[Task[_ToolExecutionOutcome]] = []
        task_calls: dict[Task[_ToolExecutionOutcome], ToolCall] = {}
        for index, call in enumerate(calls):
            task = create_task(
                self._execute_tool_call_with_lifecycle(
                    call,
                    confirm=False,
                    abort_on_reject=abort_on_reject,
                    emit_ready=emit_ready,
                    planned_index=index,
                    finish_stream_on_error=False,
                )
            )
            tasks.append(task)
            task_calls[task] = call
        pending = set(tasks)
        try:
            while pending:
                done, pending = await wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )
                failure: BaseException | None = None
                for task in done:
                    if task.cancelled():
                        failure = CancelledError()
                        break
                    exception = task.exception()
                    if exception is not None:
                        failure = exception
                        break
                if failure is None:
                    continue

                for task in pending:
                    self._append_canonical_pending_tool_cancellation(
                        task_calls[task],
                        emit_ready=emit_ready,
                    )
                    task.cancel()
                if pending:
                    await gather(*pending, return_exceptions=True)
                if isinstance(failure, CancelledError):
                    self._finish_canonical_stream(
                        StreamItemKind.STREAM_CANCELLED
                    )
                    raise failure
                self._finish_canonical_stream(
                    StreamItemKind.STREAM_ERRORED,
                    data={
                        "error_type": failure.__class__.__name__,
                        "message": str(failure),
                    },
                )
                raise failure
        except CancelledError:
            for task in tasks:
                self._append_canonical_pending_tool_cancellation(
                    task_calls[task],
                    emit_ready=emit_ready,
                )
                task.cancel()
            await gather(*tasks, return_exceptions=True)
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise

        return list(await gather(*tasks))

    def _append_canonical_pending_tool_cancellation(
        self,
        call: ToolCall,
        *,
        emit_ready: bool,
    ) -> None:
        if emit_ready:
            self._append_canonical_tool_call_ready(call)
        self._append_canonical_tool_execution_started(call)
        self._append_canonical_tool_execution_cancelled(
            call,
            stage=ToolCallDiagnosticStage.DISPATCH,
        )

    async def _execute_tool_call_with_lifecycle(
        self,
        call: ToolCall,
        *,
        confirm: bool,
        abort_on_reject: bool,
        emit_ready: bool,
        planned_index: int,
        finish_stream_on_error: bool = True,
    ) -> _ToolExecutionOutcome:
        if emit_ready:
            self._append_canonical_tool_call_ready(call)
        await self._raise_if_cancelled(
            call=call,
            finish_stream=finish_stream_on_error,
        )
        if confirm and self._tool_confirm and not self._tool_confirm_all:
            try:
                action = await self._tool_confirmation_action(call)
            except (CancelledError, Exception) as exc:
                self._append_canonical_tool_confirmation_failure(call, exc)
                if finish_stream_on_error:
                    self._finish_canonical_stream_for_exception(exc)
                raise
            if self._tool_confirmation_accepts_all(action):
                self._tool_confirm_all = True
            elif not self._tool_confirmation_accepts(action):
                self._append_canonical_tool_execution_started(call)
                diagnostic = self._tool_confirmation_rejected_diagnostic(call)
                self._append_canonical_tool_execution_terminal(
                    call,
                    diagnostic,
                )
                if abort_on_reject:
                    self._finish_canonical_stream(
                        StreamItemKind.STREAM_CANCELLED
                    )
                    raise CommandAbortException()
                return _ToolExecutionOutcome(
                    call=call,
                    context=self._tool_context
                    or self._new_tool_context(self._input),
                    planned_index=planned_index,
                    result=diagnostic,
                )

        start = perf_counter()
        started_item = self._append_canonical_tool_execution_started(call)
        await self._trigger_canonical_observability_event(
            EventType.TOOL_EXECUTE,
            started_item,
            started=start,
        )

        context = self._new_tool_context(
            self._tool_context.input if self._tool_context else None,
            stream_event=self._make_tool_stream_event_callback(call),
        )

        try:
            result = await self._execute_tool_call(
                call,
                context,
                confirm=False,
            )
        except CancelledError:
            self._append_canonical_tool_execution_cancelled(
                call,
                stage=ToolCallDiagnosticStage.DISPATCH,
            )
            if finish_stream_on_error:
                self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise
        except Exception as exc:
            self._append_canonical_tool_execution_error(call, exc)
            if finish_stream_on_error:
                self._finish_canonical_stream(
                    StreamItemKind.STREAM_ERRORED,
                    data={
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                )
            raise

        if self._is_cancellation_diagnostic(result):
            self._append_canonical_tool_execution_terminal(call, result)
            if finish_stream_on_error:
                self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise CancelledError()

        terminal_item = self._append_canonical_tool_execution_terminal(
            call, result
        )

        end = perf_counter()
        if self._event_manager:
            await self._trigger_tool_diagnostic_event(
                result=result,
                item=terminal_item,
                started=start,
                finished=end,
                elapsed=end - start,
            )
        await self._trigger_canonical_observability_event(
            EventType.TOOL_RESULT,
            terminal_item,
            started=start,
            finished=end,
            elapsed=end - start,
        )

        return _ToolExecutionOutcome(
            call=call,
            context=context,
            planned_index=planned_index,
            result=result,
        )

    async def _confirm_parallel_tool_calls(
        self,
        calls: list[ToolCall],
        *,
        abort_on_reject: bool,
        emit_ready: bool,
    ) -> list[_ToolExecutionOutcome]:
        if not self._tool_confirm or self._tool_confirm_all:
            return []
        if emit_ready:
            for call in calls:
                self._append_canonical_tool_call_ready(call)
        rejected_index: int | None = None
        for index, call in enumerate(calls):
            try:
                action = await self._tool_confirmation_action(call)
            except (CancelledError, Exception) as exc:
                self._append_canonical_parallel_confirmation_failure(
                    calls,
                    failed_index=index,
                    failure=exc,
                    emit_ready=emit_ready,
                )
                self._finish_canonical_stream_for_exception(exc)
                raise
            if self._tool_confirmation_accepts_all(action):
                self._tool_confirm_all = True
                return []
            if not self._tool_confirmation_accepts(action):
                rejected_index = index
                break
        if rejected_index is None:
            return []

        outcomes: list[_ToolExecutionOutcome] = []
        for index, call in enumerate(calls):
            if emit_ready:
                self._append_canonical_tool_call_ready(call)
            self._append_canonical_tool_execution_started(call)
            if index == rejected_index:
                diagnostic = self._tool_confirmation_rejected_diagnostic(call)
                self._append_canonical_tool_execution_terminal(
                    call,
                    diagnostic,
                )
            else:
                diagnostic = self._tool_confirmation_cancelled_diagnostic(call)
                self._append_canonical_tool_execution_terminal(
                    call,
                    diagnostic,
                )
            outcomes.append(
                _ToolExecutionOutcome(
                    call=call,
                    context=self._tool_context
                    or self._new_tool_context(self._input),
                    planned_index=index,
                    result=diagnostic,
                )
            )
        if abort_on_reject:
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise CommandAbortException()
        return outcomes

    async def _tool_confirmation_action(
        self, call: ToolCall
    ) -> str | bool | None:
        assert self._tool_confirm
        action = self._tool_confirm(call)
        if isawaitable(action):
            action = await action
        return action

    @staticmethod
    def _tool_confirmation_accepts(action: str | bool | None) -> bool:
        return action is True or action in {"a", "y"}

    @staticmethod
    def _tool_confirmation_accepts_all(action: str | bool | None) -> bool:
        return action == "a"

    def _append_canonical_parallel_confirmation_failure(
        self,
        calls: list[ToolCall],
        *,
        failed_index: int,
        failure: BaseException,
        emit_ready: bool,
    ) -> None:
        for index, call in enumerate(calls):
            if emit_ready:
                self._append_canonical_tool_call_ready(call)
            if index == failed_index:
                self._append_canonical_tool_confirmation_failure(
                    call,
                    failure,
                )
                continue
            self._append_canonical_tool_execution_started(call)
            self._append_canonical_tool_execution_terminal(
                call,
                self._tool_confirmation_cancelled_diagnostic(call),
            )

    def _append_canonical_tool_confirmation_failure(
        self,
        call: ToolCall,
        failure: BaseException,
    ) -> None:
        self._append_canonical_tool_execution_started(call)
        if isinstance(failure, CancelledError):
            self._append_canonical_tool_execution_terminal(
                call,
                self._tool_confirmation_cancelled_diagnostic(call),
            )
            return
        assert isinstance(failure, Exception)
        self._append_canonical_tool_execution_error(call, failure)

    def _finish_canonical_stream_for_exception(
        self, failure: BaseException
    ) -> None:
        if isinstance(failure, CancelledError):
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            return
        self._finish_canonical_stream(
            StreamItemKind.STREAM_ERRORED,
            data={
                "error_type": failure.__class__.__name__,
                "message": str(failure),
            },
        )

    def _classify_completed_task_input_boundary(
        self,
        item: CanonicalStreamItem,
    ) -> TaskInputCapabilityCall | None:
        """Classify a complete reserved call without another provider read."""
        execution = self._execution
        capability = self._capability_catalog
        if (
            item.kind is not StreamItemKind.TOOL_CALL_DONE
            or execution is None
            or execution.interaction_runtime is None
            or capability is None
        ):
            return None
        calls: list[ToolCall] = []
        while not self._calls.empty():
            calls.append(self._calls.get())
        current_id = item.correlation.tool_call_id
        current = next(
            (
                call
                for call in calls
                if call.id is not None and str(call.id) == current_id
            ),
            None,
        )
        provider_family = (
            self._canonical_tool_call_provider_families.get(current_id)
            if current_id is not None
            else None
        )
        try:
            canonical_name = (
                capability.canonical_name(
                    current.name,
                    provider_family=provider_family,
                )
                if current is not None
                else None
            )
        except ModelCapabilityValidationError:
            canonical_name = None
        if canonical_name != RESERVED_INPUT_CAPABILITY_NAME:
            for call in calls:
                self._put_staging_item(self._calls, call, "tool call")
            return None
        self._classify_complete_tool_call_batch(calls)
        return self._task_input_call

    def _drain_tool_call_batch(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        while not self._calls.empty():
            calls.append(self._calls.get())
        if not calls and not self._staged_tool_batch_present:
            return []
        already_classified = bool(calls) and all(
            id(call) in self._classified_tool_call_object_ids for call in calls
        )
        classified_calls = (
            calls
            if already_classified
            else self._classify_complete_tool_call_batch(calls)
        )
        if not classified_calls:
            return []
        calls = [
            call
            for call in classified_calls
            if self._should_execute_staged_tool_call(call)
        ]
        if not calls:
            return []
        batch, remaining = self._split_tool_call_batch(calls)
        for call in batch:
            self._classified_tool_call_object_ids.discard(id(call))
        for call in remaining:
            self._put_staging_item(self._calls, call, "tool call")
        return batch

    def _classify_complete_tool_call_batch(
        self,
        calls: list[ToolCall],
        *,
        text_originated: bool = False,
    ) -> list[ToolCall] | None:
        assert type(text_originated) is bool
        capability = self._capability_catalog
        if capability is None:
            return list(calls)

        if self._staged_tool_batch_invalid:
            rejection = CapabilityBatchRejected(
                code=(
                    self._staged_tool_batch_rejection_code
                    or CapabilityBatchRejectionCode.MALFORMED_CALL
                ),
                message="Batch contains an invalid capability call.",
            )
            self._discard_staged_tool_call_batch()
            self._append_capability_batch_rejection(
                rejection,
                call_count=len(calls),
            )
            return None

        if not calls:
            self._discard_staged_tool_call_batch()
            return []

        provider_families = {
            provider_family
            for call in calls
            if call.id is not None
            and (
                provider_family := (
                    self._canonical_tool_call_provider_families.get(
                        str(call.id)
                    )
                )
            )
            is not None
        }
        if len(provider_families) > 1:
            rejection = CapabilityBatchRejected(
                code=CapabilityBatchRejectionCode.MALFORMED_CALL,
                message="Batch contains conflicting provider families.",
            )
            self._discard_staged_tool_call_batch()
            self._append_capability_batch_rejection(
                rejection,
                call_count=len(calls),
            )
            return None
        provider_family = next(iter(provider_families), None)
        provider_calls: list[ProviderCapabilityCall] = []
        try:
            projection = capability.project(provider_family)
            for call in calls:
                provider_name = call.provider_name
                if provider_name is None:
                    try:
                        provider_name = projection.provider_name(call.name)
                    except ModelCapabilityValidationError:
                        provider_name = call.name
                parser_originated = text_originated or (
                    call.id is not None
                    and str(call.id) in self._text_parser_tool_call_ids
                )
                is_reserved = (
                    call.name == RESERVED_INPUT_CAPABILITY_NAME
                    or provider_name == RESERVED_INPUT_CAPABILITY_NAME
                )
                source_call_id = (
                    self._provider_tool_call_ids_by_canonical_id.get(
                        str(call.id)
                    )
                    if call.id is not None
                    else None
                )
                provider_call_id = source_call_id or call.id
                provider_calls.append(
                    ProviderCapabilityCall(
                        call_id=provider_call_id,
                        provider_name=provider_name,
                        arguments=(
                            "{"
                            if call.provider_arguments_malformed
                            else call.arguments
                        ),
                        structured=not (parser_originated and is_reserved),
                    )
                )
        except (AssertionError, ModelCapabilityValidationError):
            rejection = CapabilityBatchRejected(
                code=CapabilityBatchRejectionCode.MALFORMED_CALL,
                message="Batch contains an invalid capability call.",
            )
            self._discard_staged_tool_call_batch()
            self._append_capability_batch_rejection(
                rejection,
                call_count=len(calls),
            )
            return None

        classification = capability.classify_batch(
            provider_calls,
            provider_family=provider_family,
        )
        if isinstance(classification, CapabilityBatchRejected):
            self._discard_staged_tool_call_batch()
            self._append_capability_batch_rejection(
                classification,
                call_count=len(calls),
            )
            return None
        assert isinstance(classification, CapabilityBatchAccepted)
        if classification.task_input is not None:
            self._task_input_call = classification.task_input
            self._discard_staged_tool_call_batch()
            return []
        domain_calls = self._canonical_domain_calls(
            calls,
            list(classification.domain_calls),
        )
        if self._tool_manager is None:
            self._discard_staged_tool_call_batch()
            raise RuntimeError(
                "accepted domain capability calls require a tool registry"
            )
        self._publish_staged_domain_tool_calls(domain_calls)
        self._classified_tool_call_object_ids.update(
            id(call) for call in domain_calls
        )
        return domain_calls

    def _canonical_domain_calls(
        self,
        staged_calls: list[ToolCall],
        decoded_calls: list[ToolCall],
    ) -> list[ToolCall]:
        """Pair decoded domain calls with their response lifecycle IDs."""
        assert len(staged_calls) == len(decoded_calls)
        canonical_calls: list[ToolCall] = []
        for staged_call, decoded_call in zip(
            staged_calls, decoded_calls, strict=True
        ):
            if staged_call.id is None:
                canonical_calls.append(decoded_call)
                continue
            canonical_id = str(staged_call.id)
            if decoded_call.id is not None:
                self._provider_tool_call_ids_by_canonical_id.setdefault(
                    canonical_id,
                    str(decoded_call.id),
                )
            canonical_calls.append(replace(decoded_call, id=canonical_id))
        return canonical_calls

    def _publish_staged_domain_tool_calls(
        self,
        calls: list[ToolCall],
    ) -> None:
        """Publish staged lifecycle frames after domain batch acceptance."""
        accepted_ids = {str(call.id) for call in calls if call.id is not None}
        calls_by_id = {
            str(call.id): call for call in calls if call.id is not None
        }
        staged_items = [
            item
            for item in self._staged_tool_call_items
            if item.correlation.tool_call_id in accepted_ids
        ]
        correlations = {
            tool_call_id: state.correlation
            for tool_call_id, state in (
                self._canonical_tool_call_lifecycles.items()
            )
            if tool_call_id in accepted_ids
        }
        self._canonical_tool_call_lifecycles = {
            tool_call_id: _CanonicalToolCallLifecycle(
                correlation=correlation,
                queued=True,
            )
            for tool_call_id, correlation in correlations.items()
        }
        self._staged_tool_call_items = []
        self._staged_tool_batch_invalid = False
        self._staged_tool_batch_rejection_code = None
        self._staged_tool_batch_present = False
        for item in staged_items:
            metadata = dict(item.metadata)
            if item.kind is StreamItemKind.TOOL_CALL_READY:
                tool_call_id = item.correlation.tool_call_id
                assert tool_call_id is not None
                metadata.update(
                    self._tool_call_display_metadata(calls_by_id[tool_call_id])
                )
            self._append_canonical_item(
                item.kind,
                text_delta=item.text_delta,
                data=item.data,
                usage=item.usage,
                correlation=item.correlation,
                visibility=item.visibility,
                metadata=metadata,
                provider_family=item.provider_family,
                provider_event_type=item.provider_event_type,
            )

    def _discard_staged_tool_call_batch(self) -> None:
        """Discard private lifecycle state without exposing correlations."""
        for tool_call_id in self._response_reserved_tool_call_ids:
            if (
                tool_call_id
                not in self._canonical_tool_call_argument_delta_ids
                and tool_call_id not in self._canonical_tool_call_ready_ids
                and tool_call_id not in self._canonical_tool_call_done_ids
            ):
                self._canonical_tool_call_reserved_ids.discard(tool_call_id)
        self._canonical_tool_call_lifecycles = {}
        self._staged_tool_call_items = []
        self._staged_tool_batch_invalid = False
        self._staged_tool_batch_rejection_code = None
        self._staged_tool_batch_present = False
        self._canonical_tool_call_provider_families = {}
        self._provider_tool_call_ids_by_canonical_id = {}
        self._text_parser_tool_call_ids = set()
        self._response_tool_call_id_aliases = {}
        self._response_reserved_tool_call_ids = set()
        self._classified_tool_call_object_ids = set()

    def _discard_untrusted_response_tool_call_batch(self) -> None:
        """Discard all effects derived from an abnormal model response."""
        while not self._calls.empty():
            self._calls.get()
        self._task_input_call = None
        self._discard_staged_tool_call_batch()

    def _append_capability_batch_rejection(
        self,
        rejection: CapabilityBatchRejected,
        *,
        call_count: int,
    ) -> None:
        assert isinstance(rejection, CapabilityBatchRejected)
        assert isinstance(call_count, int) and call_count >= 0
        details: dict[str, Any] = {"call_count": call_count}
        self._append_canonical_guard_diagnostic(
            code=rejection.code.value,
            message=rejection.message,
            details=details,
        )

    def _should_execute_staged_tool_call(self, call: ToolCall) -> bool:
        if call.id is None:
            return True
        state = self._canonical_tool_call_lifecycles.get(str(call.id))
        return state is None or (state.done and not state.invalid)

    def _split_tool_call_batch(
        self, calls: list[ToolCall]
    ) -> tuple[list[ToolCall], list[ToolCall]]:
        assert calls
        if not self._can_parallelize_tool_calls():
            return [calls[0]], calls[1:]

        manager = cast(ToolManager, self._tool_manager)
        first = calls[0]
        if not manager.is_tool_call_parallel_safe(first):
            return [first], calls[1:]

        limit = manager.maximum_parallel_tool_calls
        batch: list[ToolCall] = []
        for index, call in enumerate(calls):
            if len(batch) >= limit or not manager.is_tool_call_parallel_safe(
                call
            ):
                return batch, calls[index:]
            batch.append(call)
        return batch, []

    def _can_parallelize_tool_calls(self) -> bool:
        return (
            type(self._tool_manager) is ToolManager
            and self._tool_manager.parallel_tool_calls
        )

    async def _response_text_and_calls(
        self,
        response: TextGenerationResponse,
    ) -> tuple[str, list[ToolCall]]:
        self._begin_tool_call_lifecycle_response()
        try:
            self._calls = self._make_staging_queue()
            text_parts: list[str] = []
            streamed_calls: list[ToolCall] = []

            def collect_parser_output(start_index: int) -> None:
                self._collect_response_text_and_calls(
                    start_index,
                    text_parts,
                    streamed_calls,
                )

            response_iterator = (
                aiter(response)
                if response.is_async_generator
                else response.canonical_stream(
                    stream_session_id=self._canonical_stream_session_id,
                    run_id=self._canonical_run_id,
                    turn_id=self._canonical_turn_id,
                )
            )
            while True:
                try:
                    response_item = response_iterator.__anext__()
                    item = await self._await_with_session_cancellation(
                        response_item
                    )
                except StopAsyncIteration:
                    break
                await self._raise_if_cancelled(finish_stream=False)
                canonical_item = self._canonical_item_from_response_item(item)
                collection_index = len(self._canonical_items)
                if await self._process_canonical_text_tool_call_parser_stage(
                    canonical_item,
                    after_parser_output=collect_parser_output,
                ):
                    continue

                appended_item = self._append_canonical_response_item(
                    canonical_item
                )
                if appended_item is None:
                    continue
                if (
                    self._classify_completed_task_input_boundary(
                        canonical_item
                    )
                    is not None
                ):
                    execution = self._execution
                    if execution is None or not isinstance(
                        execution.interaction_runtime,
                        DurableInteractionRuntime,
                    ):
                        await response.aclose()
                    break
                if (
                    appended_item.kind is StreamItemKind.ANSWER_DELTA
                    and appended_item.text_delta is not None
                ):
                    await self._emit_token_generated_event(appended_item)
                    self._step += 1
                    text_parts.append(appended_item.text_delta)
                self._collect_response_text_and_calls(
                    collection_index + 1, text_parts, streamed_calls
                )
            await self._flush_canonical_text_tool_call_parser_stage(
                after_parser_output=collect_parser_output,
            )
            self._finalize_incomplete_canonical_tool_calls()
            self._drain_reconstructed_tool_calls(streamed_calls)
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            return "".join(text_parts), streamed_calls
        except CancelledError as exc:
            try:
                await self._cancel_active_model_continuation_response()
            except BaseException as cleanup_failure:
                self._attach_cleanup_failures(exc, [cleanup_failure])
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_CANCELLED
            )
            raise
        except Exception as exc:
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            raise

    async def _non_stream_response_text_and_calls(
        self,
        text: str,
    ) -> tuple[str, list[ToolCall]]:
        if self._tool_parser is None:
            return text, []

        item = CanonicalStreamItem(
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
            sequence=self._canonical_sequence,
            kind=StreamItemKind.ANSWER_DELTA,
            channel=StreamChannel.ANSWER,
            text_delta=text,
        )
        if not self._tool_parser_canonicalizes_answer_delta(item):
            return text, []

        text_parts: list[str] = []
        calls: list[ToolCall] = []

        def collect_parser_output(start_index: int) -> None:
            self._collect_response_text_and_calls(
                start_index,
                text_parts,
                calls,
            )

        processed = await self._process_canonical_text_tool_call_parser_stage(
            item,
            after_parser_output=collect_parser_output,
        )
        assert processed
        await self._flush_canonical_text_tool_call_parser_stage(
            after_parser_output=collect_parser_output,
        )
        self._finalize_incomplete_canonical_tool_calls()
        self._drain_reconstructed_tool_calls(calls)
        return "".join(text_parts), calls

    def _collect_response_text_and_calls(
        self,
        start_index: int,
        text_parts: list[str],
        calls: list[ToolCall],
    ) -> None:
        for item in self._canonical_items[start_index:]:
            if (
                item.kind is StreamItemKind.ANSWER_DELTA
                and item.text_delta is not None
            ):
                text_parts.append(item.text_delta)

    def _drain_reconstructed_tool_calls(
        self,
        calls: list[ToolCall],
    ) -> None:
        while not self._calls.empty():
            calls.append(self._calls.get())

    async def _emit_token_generated_event(
        self,
        item: CanonicalStreamItem,
    ) -> None:
        if not self._should_emit_token_generated_event():
            return
        assert item.text_delta is not None
        token_str = item.text_delta
        token_id = item.metadata.get("token_id")
        if not isinstance(token_id, int) and self._should_enrich_token_ids():
            tokenizer = (
                self._engine_agent.engine.tokenizer
                if self._engine_agent.engine
                else None
            )
            if tokenizer:
                ids = tokenizer.encode(token_str, add_special_tokens=False)
                token_id = ids[0] if ids else None

        assert self._event_manager
        data = stream_observability_payload(item)
        summary = dict(cast(dict[str, Any], data.get("summary", {})))
        summary["model_id"] = self._engine_agent.engine.model_id
        summary["step"] = self._step
        if isinstance(token_id, int):
            summary["token_id"] = token_id
        data["summary"] = cast(Any, summary)
        payload = EventObservabilityPayload.canonical_stream(data)
        event = Event.from_observability_payload(
            type=EventType.TOKEN_GENERATED,
            observability_payload=payload,
        )
        await self._event_manager.trigger(event)

    def _should_emit_token_generated_event(self) -> bool:
        if not self._event_manager:
            return False
        should_emit = getattr(self._event_manager, "should_emit", None)
        if not callable(should_emit):
            return False
        result = should_emit(EventType.TOKEN_GENERATED)
        return result if isinstance(result, bool) else False

    def _should_enrich_token_ids(self) -> bool:
        if not self._event_manager:
            return False
        value = getattr(self._event_manager, "enrich_token_ids", False)
        return value if isinstance(value, bool) else False

    async def _execute_tool_call(
        self,
        call: ToolCall,
        context: ToolCallContext,
        *,
        confirm: bool,
    ) -> ToolCallOutcome | None:
        if self._tool_manager is None:
            return None
        if self._block_repeated_tool_calls:
            repeated_diagnostic = self._repeated_call_diagnostic(call)
            if repeated_diagnostic is not None:
                return repeated_diagnostic

        self._attempted_call_signatures.add(self._call_signature(call))
        if type(self._tool_manager) is ToolManager:
            confirmation = self._tool_confirm if confirm else None
            return await self._tool_manager.execute_call(
                call,
                context,
                confirm=confirmation,
            )
        return await self._tool_manager(call, context)

    async def _react_process(
        self, output: str, results: list[ToolCallOutcome]
    ) -> TextGenerationResponse | None:
        tool_messages: list[Message] = []
        for result in results:
            tool_messages.extend(
                self._tool_observation_messages(
                    self._provider_facing_tool_outcome(result),
                    json_output=False,
                )
            )

        if self._execution is not None and tool_messages:
            await self._execution.record_messages(tuple(tool_messages))
        if not self._should_continue_tool_cycle(tool_messages, results):
            return None

        assert self._input and (
            (
                isinstance(self._input, list)
                and isinstance(self._input[0], Message)
            )
            or isinstance(self._input, Message)
        )

        messages = (
            list(cast(list[Message], self._input))
            if isinstance(self._input, list)
            else [self._input]
        )
        messages.extend(tool_messages)

        self._input = cast(Input, messages)
        self._tool_context = self._new_tool_context(self._input)

        context = await self._make_child_context(messages)
        model_origin = context.execution_origin
        continuation_id = str(
            model_origin.model_call_id if model_origin is not None else uuid4()
        )
        continuation_item = self._append_canonical_model_continuation(
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            continuation_id,
        )
        try:
            await self._raise_if_cancelled(finish_stream=False)
            await self._trigger_canonical_observability_event(
                EventType.TOOL_MODEL_RUN,
                continuation_item,
            )
            response = await self._await_with_session_cancellation(
                self._engine_agent(context)
            )
        except CancelledError:
            self._append_canonical_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                continuation_id,
            )
            raise
        except Exception as exc:
            self._append_canonical_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                continuation_id,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            raise
        assert response
        assert isinstance(response, TextGenerationResponse)
        await self._install_continuation_response(
            response,
            continuation_id,
            activate=False,
        )
        return response

    async def _raise_if_cancelled(
        self,
        *,
        call: ToolCall | None = None,
        finish_stream: bool = True,
    ) -> None:
        assert isinstance(finish_stream, bool)
        if self._cancellation_checker is None:
            return

        try:
            await self._cancellation_checker()
        except CancelledError:
            if call is not None:
                self._append_canonical_tool_execution_started(call)
                self._append_canonical_tool_execution_cancelled(call)
            if finish_stream:
                self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise

    def _should_continue_tool_cycle(
        self,
        tool_messages: list[Message],
        outcomes: list[ToolCallOutcome],
    ) -> bool:
        if not tool_messages:
            self._append_canonical_guard_diagnostic(
                code="orchestrator.tool_cycle.empty_observation",
                message=(
                    "Tool cycle stopped without model-facing observations."
                ),
                details={
                    "attempted_call_signatures": (
                        self._attempted_call_signature_details()
                    ),
                    "outcome_count": len(outcomes),
                },
            )
            return False

        cycle_signature = self._tool_cycle_signature(tool_messages)
        if (
            self._block_repeated_tool_calls
            and cycle_signature in self._tool_cycle_signatures
        ):
            self._append_canonical_guard_diagnostic(
                code="orchestrator.tool_cycle.duplicate_observation",
                message="Tool cycle stopped after repeated observations.",
                details={
                    "attempted_call_signatures": (
                        self._attempted_call_signature_details()
                    ),
                    "cycle_count": self._tool_cycle_count,
                    "signature": cycle_signature,
                },
            )
            return False

        if (
            self._maximum_tool_cycles != UNLIMITED_TOOL_CYCLES
            and self._tool_cycle_count >= self._maximum_tool_cycles
        ):
            self._append_canonical_guard_diagnostic(
                code="orchestrator.tool_cycle.limit_exceeded",
                message="Tool cycle stopped after reaching the cycle limit.",
                details={
                    "attempted_call_signatures": (
                        self._attempted_call_signature_details()
                    ),
                    "cycle_count": self._tool_cycle_count,
                    "maximum_cycles": self._maximum_tool_cycles,
                },
            )
            return False

        non_executed = bool(outcomes) and all(
            isinstance(outcome, ToolCallDiagnostic) for outcome in outcomes
        )
        if non_executed:
            self._consecutive_non_executed_cycles += 1
        else:
            self._consecutive_non_executed_cycles = 0

        if (
            self._consecutive_non_executed_cycles
            > self._MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES
        ):
            self._append_canonical_guard_diagnostic(
                code="orchestrator.tool_cycle.non_executed_limit_exceeded",
                message=(
                    "Tool cycle stopped after repeated non-executed "
                    "tool attempts."
                ),
                details={
                    "attempted_call_signatures": (
                        self._attempted_call_signature_details()
                    ),
                    "consecutive_non_executed_cycles": (
                        self._consecutive_non_executed_cycles
                    ),
                    "maximum_consecutive_non_executed_cycles": (
                        self._MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES
                    ),
                },
            )
            return False

        self._tool_cycle_signatures.add(cycle_signature)
        self._tool_cycle_count += 1
        return True

    def _attempted_call_signature_details(self) -> list[str]:
        return sorted(self._attempted_call_signatures)

    def _record_tool_outcome(self, result: ToolCallOutcome | None) -> None:
        if isinstance(result, ToolCallResult):
            self._call_history.append(result)
        elif isinstance(result, ToolCallError):
            self._call_history.append(result.call)

    async def _trigger_tool_diagnostic_event(
        self,
        *,
        result: ToolCallOutcome | None,
        item: CanonicalStreamItem | None,
        started: float,
        finished: float,
        elapsed: float,
    ) -> None:
        if not isinstance(result, ToolCallDiagnostic):
            return
        await self._trigger_canonical_observability_event(
            EventType.TOOL_DIAGNOSTIC,
            item,
            started=started,
            finished=finished,
            elapsed=elapsed,
        )

    def _repeated_call_diagnostic(
        self, call: ToolCall
    ) -> ToolCallDiagnostic | None:
        signature = self._call_signature(call)
        if signature not in self._attempted_call_signatures:
            return None
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=self._diagnostic_tool_name(call),
            code=ToolCallDiagnosticCode.REPEATED_CALL,
            stage=ToolCallDiagnosticStage.GUARD,
            message="Tool call repeats a previous attempt.",
            details={
                "attempted_call_signature": signature,
                "attempted_call_signatures": cast(
                    Any,
                    sorted(self._attempted_call_signatures),
                ),
                "signature": signature,
            },
        )

    @classmethod
    def _tool_confirmation_rejected_diagnostic(
        cls,
        call: ToolCall,
    ) -> ToolCallDiagnostic:
        name = cls._diagnostic_tool_name(call)
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=name,
            canonical_name=name,
            code=ToolCallDiagnosticCode.USER_REJECTED,
            stage=ToolCallDiagnosticStage.CONFIRM,
            message="Tool call was rejected before execution.",
        )

    @classmethod
    def _tool_confirmation_cancelled_diagnostic(
        cls,
        call: ToolCall,
    ) -> ToolCallDiagnostic:
        name = cls._diagnostic_tool_name(call)
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=name,
            canonical_name=name,
            code=ToolCallDiagnosticCode.CANCELLED,
            stage=ToolCallDiagnosticStage.CONFIRM,
            message="Tool call was cancelled before execution.",
        )

    @classmethod
    def _tool_execution_cancelled_diagnostic(
        cls,
        call: ToolCall,
        *,
        stage: ToolCallDiagnosticStage,
    ) -> ToolCallDiagnostic:
        name = cls._diagnostic_tool_name(call)
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=name,
            canonical_name=name,
            code=ToolCallDiagnosticCode.CANCELLED,
            stage=stage,
            message="Tool call was cancelled.",
        )

    @staticmethod
    def _diagnostic_tool_name(call: ToolCall) -> str | None:
        return call.name if call.name.strip() else None

    @staticmethod
    def _call_signature(call: ToolCall) -> str:
        return dumps(
            {
                "arguments": call.arguments,
                "name": call.name,
            },
            default=str,
            sort_keys=True,
        )

    @classmethod
    def _tool_cycle_signature(cls, messages: list[Message]) -> str:
        payload = [
            cls._tool_cycle_message_payload(message) for message in messages
        ]
        return dumps(
            payload,
            default=str,
            sort_keys=True,
        )

    @classmethod
    def _tool_cycle_message_payload(cls, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "arguments": message.arguments,
            "content": message.content,
            "name": message.name,
            "role": message.role.value,
            "thinking": message.thinking,
            "tool_calls": (
                [asdict(tool_call) for tool_call in message.tool_calls]
                if message.tool_calls
                else None
            ),
        }
        if message.tool_call_result is not None:
            result = message.tool_call_result
            payload["tool_call_result"] = {
                "arguments": result.arguments,
                "call": cls._call_signature(result.call),
                "name": result.name,
                "result": cls._json_content(result.result),
            }
        if message.tool_call_error is not None:
            error = message.tool_call_error
            payload["tool_call_error"] = {
                "arguments": error.arguments,
                "call": cls._call_signature(error.call),
                "message": error.message,
                "name": error.name,
            }
        if message.tool_call_diagnostic is not None:
            diagnostic = message.tool_call_diagnostic
            payload["tool_call_diagnostic"] = {
                "call_id": diagnostic.call_id,
                "canonical_name": diagnostic.canonical_name,
                "code": diagnostic.code.value,
                "details": diagnostic.details,
                "message": diagnostic.message,
                "requested_name": diagnostic.requested_name,
                "retryable": diagnostic.retryable,
                "stage": diagnostic.stage.value,
                "status": diagnostic.status.value,
            }
        return payload

    @classmethod
    def _tool_observation_messages(
        cls,
        outcome: ToolCallOutcome,
        *,
        call: ToolCall | None = None,
        json_output: bool,
    ) -> list[Message]:
        if isinstance(outcome, ToolCallDiagnostic):
            return cls._diagnostic_messages(
                outcome,
                call=call,
                json_output=json_output,
            )

        model_outcome = cast(
            ToolCallResult | ToolCallError,
            cls._model_facing_outcome(outcome),
        )
        return [
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    MessageToolCall(
                        id=str(model_outcome.call.id),
                        name=model_outcome.name,
                        arguments=normalize_tool_arguments(
                            model_outcome.arguments or {}
                        ),
                    )
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                name=model_outcome.name,
                arguments=model_outcome.arguments,
                content=cls._outcome_content(
                    model_outcome,
                    json_output=json_output,
                ),
                tool_call_result=(
                    model_outcome
                    if isinstance(model_outcome, ToolCallResult)
                    else None
                ),
                tool_call_error=(
                    model_outcome
                    if isinstance(model_outcome, ToolCallError)
                    else None
                ),
            ),
        ]

    def _provider_facing_tool_call(self, call: ToolCall) -> ToolCall:
        """Restore the source provider ID for model continuation messages."""
        if call.id is None:
            return call
        canonical_id = str(call.id)
        provider_id = self._provider_tool_call_ids_by_canonical_id.get(
            canonical_id
        )
        if provider_id is None or provider_id == canonical_id:
            return call
        return replace(call, id=provider_id)

    def _provider_facing_tool_outcome(
        self,
        outcome: ToolCallOutcome,
    ) -> ToolCallOutcome:
        """Restore provider correlation while keeping public lifecycle IDs."""
        if isinstance(outcome, ToolCallResult | ToolCallError):
            provider_call = self._provider_facing_tool_call(outcome.call)
            if provider_call is outcome.call:
                return outcome
            return replace(
                outcome,
                call=provider_call,
            )
        if outcome.call_id is None:
            return outcome
        provider_id = self._provider_tool_call_ids_by_canonical_id.get(
            str(outcome.call_id)
        )
        if provider_id is None or provider_id == str(outcome.call_id):
            return outcome
        return replace(outcome, call_id=provider_id)

    @classmethod
    def _model_facing_outcome(
        cls, outcome: ToolCallOutcome
    ) -> ToolCallOutcome:
        if not isinstance(outcome, ToolCallResult):
            return outcome

        output = cls._model_tool_output_text(outcome.result)
        if len(output) <= cls._MAXIMUM_MODEL_TOOL_OUTPUT_CHARS:
            return outcome

        return replace(
            outcome,
            result=cls._truncated_model_tool_output(output),
        )

    @classmethod
    def _model_tool_output_text(cls, result: Any) -> str:
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        return cls._json_content(result)

    @classmethod
    def _truncated_model_tool_output(
        cls,
        output: str,
    ) -> dict[str, object]:
        limit = cls._MAXIMUM_MODEL_TOOL_OUTPUT_CHARS
        marker = (
            f"\n\n[truncated {len(output) - limit} characters "
            "from tool output]\n\n"
        )
        head_chars = limit // 2
        tail_chars = limit - head_chars
        excerpt = output[:head_chars] + marker + output[-tail_chars:]
        return {
            "truncated": True,
            "message": (
                "Tool output was truncated for model context. Re-run the "
                "tool with narrower arguments to inspect omitted content."
            ),
            "original_output_chars": len(output),
            "retained_output_chars": len(excerpt),
            "output_sha256": sha256(output.encode()).hexdigest(),
            "output": excerpt,
        }

    @classmethod
    def _diagnostic_messages(
        cls,
        diagnostic: ToolCallDiagnostic,
        *,
        call: ToolCall | None,
        json_output: bool,
    ) -> list[Message]:
        call_id = diagnostic.call_id or (call.id if call else None)
        name = (
            diagnostic.canonical_name
            or diagnostic.requested_name
            or (call.name if call else "tool")
        )
        arguments = call.arguments if call else None
        content = cls._outcome_content(
            diagnostic,
            json_output=json_output,
        )
        if call_id is None:
            return [
                Message(
                    role=MessageRole.ASSISTANT,
                    content=content,
                    tool_call_diagnostic=diagnostic,
                )
            ]

        return [
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    MessageToolCall(
                        id=str(call_id),
                        name=name,
                        arguments=normalize_tool_arguments(arguments or {}),
                    )
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                name=name,
                arguments=arguments,
                content=content,
                tool_call_diagnostic=diagnostic,
            ),
        ]

    @classmethod
    def _outcome_content(
        cls,
        outcome: ToolCallOutcome,
        *,
        json_output: bool,
    ) -> str:
        if isinstance(outcome, ToolCallDiagnostic):
            return cls._json_content(tool_call_diagnostic_payload(outcome))
        if isinstance(outcome, ToolCallError):
            return (
                cls._json_content(outcome.message)
                if json_output
                else outcome.message
            )

        result = outcome.result
        if not json_output and isinstance(result, str):
            return result
        if not json_output and result is None:
            return ""
        return cls._json_content(result)

    @staticmethod
    def _json_content(value: Any) -> str:
        return dumps(
            asdict(cast(Any, value)) if is_dataclass(value) else value,
            default=lambda o: (
                b64encode(o).decode()
                if isinstance(o, (bytes, bytearray, memoryview))
                else str(o)
            ),
        )

    async def _on_consumed(self) -> None:
        assert self._event_manager
        await self._event_manager.trigger(Event(type=EventType.STREAM_END))

    async def _make_child_context(
        self,
        messages: Input,
        *,
        advance_turn: bool = True,
    ) -> ModelCallContext:
        parent_context = self._context
        root_parent = (
            parent_context.root_parent or parent_context
            if parent_context
            else None
        )
        execution_origin = parent_context.execution_origin
        if self._execution is not None:
            if advance_turn:
                execution_origin = await self._execution.advance_model_turn()
            else:
                execution_origin = self._execution.origin
        context = ModelCallContext(
            specification=self._operation.specification,
            input=messages,
            capability=self._capability_catalog,
            engine_args=self._continuation_engine_args(),
            parent=parent_context,
            root_parent=root_parent,
            agent_id=(
                parent_context.agent_id if parent_context else self._agent_id
            ),
            participant_id=(
                parent_context.participant_id
                if parent_context
                else self._participant_id
            ),
            session_id=(
                parent_context.session_id
                if parent_context
                else self._session_id
            ),
            execution=self._execution,
            execution_origin=execution_origin,
            interaction_broker=parent_context.interaction_broker,
        )
        self._context = context
        return context

    def _new_tool_context(
        self,
        input: Input | None,
        *,
        stream_event: (
            Callable[[ToolExecutionStreamEvent], Awaitable[None]] | None
        ) = None,
    ) -> ToolCallContext:
        """Return one tool context bound to this exact execution branch."""
        return ToolCallContext(
            input=input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
            cancellation_checker=self._cancellation_checker,
            stream_event=stream_event,
            execution=self._execution,
            execution_origin=(
                self._execution.origin
                if self._execution is not None
                else self._context.execution_origin
            ),
            interaction_broker=self._context.interaction_broker,
        )

    def _continuation_engine_args(self) -> dict[str, Any]:
        engine_args = dict(self._engine_args)
        engine_args.pop("tool_choice", None)
        return engine_args

    def _append_canonical_item(
        self,
        kind: StreamItemKind,
        *,
        text_delta: str | None = None,
        data: Any | None = None,
        usage: Any | None = None,
        correlation: StreamItemCorrelation | None = None,
        terminal_outcome: StreamTerminalOutcome | None = None,
        visibility: StreamVisibility = StreamVisibility.PUBLIC,
        reasoning_representation: StreamReasoningRepresentation | None = None,
        segment_instance_ordinal: int | None = None,
        metadata: dict[str, Any] | None = None,
        provider_family: str | None = None,
        provider_event_type: str | None = None,
    ) -> CanonicalStreamItem | None:
        if (
            self._canonical_stream_closed
            or self._canonical_stream_terminal is not None
            and kind is not StreamItemKind.STREAM_CLOSED
        ):
            return None
        item = CanonicalStreamItem(
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
            sequence=self._canonical_sequence,
            kind=kind,
            channel=stream_channel_for_kind(kind),
            correlation=correlation or self._canonical_correlation,
            text_delta=text_delta,
            data=cast(Any, data),
            usage=cast(Any, usage),
            terminal_outcome=terminal_outcome,
            visibility=visibility,
            reasoning_representation=reasoning_representation,
            segment_instance_ordinal=segment_instance_ordinal,
            metadata=metadata or {},
            provider_family=provider_family,
            provider_event_type=provider_event_type,
        )
        self._canonical_items.append(item)
        self._canonical_sequence += 1
        self._track_canonical_channel_boundary(item)
        self._track_canonical_lifecycle_item(item)
        self._notify_canonical_item_available()
        return item

    def _notify_canonical_item_available(self) -> None:
        self._canonical_item_available.set()

    async def _trigger_canonical_observability_event(
        self,
        event_type: EventType,
        item: CanonicalStreamItem | None,
        *,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        assert isinstance(event_type, EventType)
        if self._event_manager is None or item is None:
            return
        payload = EventObservabilityPayload.canonical_stream(
            stream_observability_payload(item)
        )
        await self._event_manager.trigger(
            Event.from_observability_payload(
                type=event_type,
                observability_payload=payload,
                started=started,
                finished=finished,
                elapsed=elapsed,
            )
        )

    async def _trigger_derived_canonical_observability_event(
        self,
        event_type: EventType,
        kind: StreamItemKind,
        *,
        correlation: StreamItemCorrelation | None = None,
        summary: dict[str, Any] | None = None,
        started: float | None = None,
        finished: float | None = None,
        elapsed: float | None = None,
    ) -> None:
        assert isinstance(event_type, EventType)
        assert isinstance(kind, StreamItemKind)
        if self._event_manager is None:
            return
        data: dict[str, Any] = {
            "stream_session_id": self._canonical_stream_session_id,
            "run_id": self._canonical_run_id,
            "turn_id": self._canonical_turn_id,
            "sequence": self._canonical_sequence,
            "kind": kind.value,
            "channel": stream_channel_for_kind(kind).value,
            "visibility": "public",
            "derived": True,
        }
        trace = (correlation or self._canonical_correlation).to_trace_dict()
        if trace:
            data["correlation"] = trace
        if summary:
            data["summary"] = summary
        payload = EventObservabilityPayload.canonical_stream(cast(Any, data))
        await self._event_manager.trigger(
            Event.from_observability_payload(
                type=event_type,
                observability_payload=payload,
                started=started,
                finished=finished,
                elapsed=elapsed,
            )
        )

    def _canonical_answer_text(self) -> str:
        """Return the immutable public answer accumulated by this response."""
        return "".join(
            item.text_delta
            for item in self._canonical_items
            if item.kind is StreamItemKind.ANSWER_DELTA
            and item.text_delta is not None
        )

    def _current_response_answer_text(self) -> str:
        """Return answer text emitted by the active provider response."""
        return "".join(
            item.text_delta
            for item in self._canonical_items[
                self._response_answer_start_index :
            ]
            if item.kind is StreamItemKind.ANSWER_DELTA
            and item.text_delta is not None
        )

    async def _finalize_execution(
        self,
        kind: StreamItemKind,
        *,
        output: str | None = None,
    ) -> None:
        """Apply one idempotent execution terminal with immutable output."""
        execution = self._execution
        if execution is None or self._execution_finalized:
            return
        if execution.status in {
            AgentExecutionStatus.COMPLETED,
            AgentExecutionStatus.CANCELLED,
            AgentExecutionStatus.ERRORED,
        }:
            self._execution_finalized = True
            return
        if kind is StreamItemKind.STREAM_COMPLETED:
            if output:
                await execution.complete_with_response(
                    output,
                    messages=(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=output,
                        ),
                    ),
                )
            else:
                await execution.complete()
        elif kind is StreamItemKind.STREAM_CANCELLED:
            cleanup_failures = await self._settle_execution_with_deadline(
                cancelled=True,
            )
            self._execution_finalized = execution.status in {
                AgentExecutionStatus.COMPLETED,
                AgentExecutionStatus.CANCELLED,
                AgentExecutionStatus.ERRORED,
            }
            self._raise_cleanup_failures(list(cleanup_failures))
            return
        elif kind is StreamItemKind.STREAM_ERRORED:
            cleanup_failures = await self._settle_execution_with_deadline(
                cancelled=False,
            )
            self._execution_finalized = execution.status in {
                AgentExecutionStatus.COMPLETED,
                AgentExecutionStatus.CANCELLED,
                AgentExecutionStatus.ERRORED,
            }
            self._raise_cleanup_failures(list(cleanup_failures))
            return
        else:
            raise ValueError("unsupported execution terminal kind")
        self._execution_finalized = execution.status in {
            AgentExecutionStatus.COMPLETED,
            AgentExecutionStatus.CANCELLED,
            AgentExecutionStatus.ERRORED,
        }

    def _finish_canonical_stream(
        self,
        kind: StreamItemKind,
        *,
        data: Any | None = None,
        usage: Any | None = None,
        correlation: StreamItemCorrelation | None = None,
    ) -> None:
        outcomes = {
            StreamItemKind.STREAM_COMPLETED: StreamTerminalOutcome.COMPLETED,
            StreamItemKind.STREAM_ERRORED: StreamTerminalOutcome.ERRORED,
            StreamItemKind.STREAM_CANCELLED: StreamTerminalOutcome.CANCELLED,
            StreamItemKind.STREAM_INPUT_REQUIRED: (
                StreamTerminalOutcome.INPUT_REQUIRED
            ),
        }
        outcome = outcomes[kind]
        if self._canonical_stream_terminal is not None:
            return
        continuation_terminal = {
            StreamItemKind.STREAM_COMPLETED: (
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            ),
            StreamItemKind.STREAM_ERRORED: (
                StreamItemKind.MODEL_CONTINUATION_ERROR
            ),
            StreamItemKind.STREAM_CANCELLED: (
                StreamItemKind.MODEL_CONTINUATION_CANCELLED
            ),
        }.get(kind)
        if continuation_terminal is None:
            if self._active_model_continuation_id is not None:
                raise StreamValidationError(
                    "input-required stream cannot close an active model "
                    "continuation"
                )
        else:
            self._finish_active_model_continuation(
                continuation_terminal,
                data=data if kind is StreamItemKind.STREAM_ERRORED else None,
            )
        self._finalize_incomplete_canonical_tool_calls()
        self._append_open_canonical_channel_done_items()
        terminal_usage: Any | None = None
        if kind is StreamItemKind.STREAM_COMPLETED:
            terminal_usage = (
                cast(Any, usage)
                if usage is not None
                else self._canonical_usage()
            )
        self._append_canonical_item(
            kind,
            data=data,
            usage=terminal_usage,
            correlation=correlation,
            terminal_outcome=outcome,
        )
        self._canonical_stream_terminal = outcome
        if not self._canonical_stream_closed:
            self._append_canonical_item(StreamItemKind.STREAM_CLOSED)
            self._canonical_stream_closed = True

    def _append_open_canonical_channel_done_items(self) -> None:
        if self._canonical_answer_started and not self._canonical_answer_done:
            self._append_canonical_item(StreamItemKind.ANSWER_DONE)
        if (
            self._canonical_reasoning_started
            and not self._canonical_reasoning_done
        ):
            self._append_canonical_item(StreamItemKind.REASONING_DONE)

    def _track_canonical_channel_boundary(
        self,
        item: CanonicalStreamItem,
    ) -> None:
        if item.kind is StreamItemKind.ANSWER_DELTA:
            self._canonical_answer_started = True
        elif item.kind is StreamItemKind.ANSWER_DONE:
            self._canonical_answer_done = True
        elif item.kind is StreamItemKind.REASONING_DELTA:
            self._canonical_reasoning_started = True
        elif item.kind is StreamItemKind.REASONING_DONE:
            self._canonical_reasoning_done = True

    def _append_canonical_tool_call_ready(
        self, call: ToolCall
    ) -> CanonicalStreamItem | None:
        tool_call_id = self._canonical_tool_call_id(call)
        if tool_call_id in self._canonical_tool_call_ready_ids:
            return None
        self._canonical_tool_call_ready_ids.add(tool_call_id)
        correlation = StreamItemCorrelation(tool_call_id=tool_call_id)
        arguments = (
            self._json_content(call.arguments)
            if call.arguments is not None
            else ""
        )
        if (
            arguments
            and tool_call_id
            not in self._canonical_tool_call_argument_delta_ids
        ):
            self._append_canonical_item(
                StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
                text_delta=arguments,
                correlation=correlation,
            )
        ready_item = self._append_canonical_item(
            StreamItemKind.TOOL_CALL_READY,
            data={
                "name": call.name,
                "arguments": cast(Any, call.arguments),
            },
            correlation=correlation,
            metadata=self._tool_call_display_metadata(call),
        )
        self._append_canonical_item(
            StreamItemKind.TOOL_CALL_DONE,
            correlation=correlation,
        )
        return ready_item

    def _append_canonical_tool_execution_started(
        self, call: ToolCall
    ) -> CanonicalStreamItem | None:
        tool_call_id = self._canonical_tool_call_id(call)
        if (
            tool_call_id in self._canonical_tool_execution_started_ids
            or tool_call_id in self._canonical_tool_execution_terminal_ids
        ):
            return None
        self._canonical_tool_execution_started_ids.add(tool_call_id)
        return self._append_canonical_item(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            data={"name": call.name},
            correlation=self._canonical_tool_correlation(call),
            metadata=self._tool_call_display_metadata(call),
        )

    def _append_canonical_tool_execution_terminal(
        self,
        call: ToolCall,
        result: ToolCallOutcome | None,
    ) -> CanonicalStreamItem | None:
        if isinstance(result, ToolCallError):
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                call,
                {
                    "name": result.name,
                    "message": result.message,
                    "arguments": cast(Any, result.arguments),
                },
                result=result,
            )
        elif self._is_cancellation_diagnostic(result):
            assert isinstance(result, ToolCallDiagnostic)
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                call,
                self._canonical_tool_diagnostic_payload(call, result),
                result=result,
            )
        elif isinstance(result, ToolCallDiagnostic):
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                call,
                self._canonical_tool_diagnostic_payload(call, result),
                result=result,
            )
        elif isinstance(result, ToolCallResult):
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                call,
                {
                    "name": result.name,
                    "result": cast(Any, result.result),
                    "arguments": cast(Any, result.arguments),
                },
                result=result,
            )
        else:
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                call,
                None,
                result=self._empty_tool_call_result(call),
            )

    def _append_canonical_tool_execution_error(
        self,
        call: ToolCall,
        exc: Exception,
    ) -> None:
        self._append_canonical_tool_execution_result(
            StreamItemKind.TOOL_EXECUTION_ERROR,
            call,
            {
                "error_type": exc.__class__.__name__,
                "message": str(exc),
            },
            result=self._exception_tool_call_error(call, exc),
        )

    def _append_canonical_tool_execution_cancelled(
        self,
        call: ToolCall,
        *,
        stage: ToolCallDiagnosticStage = ToolCallDiagnosticStage.GUARD,
    ) -> None:
        self._append_canonical_tool_execution_terminal(
            call,
            self._tool_execution_cancelled_diagnostic(call, stage=stage),
        )

    def _append_canonical_tool_execution_result(
        self,
        kind: StreamItemKind,
        call: ToolCall,
        data: Any | None,
        *,
        result: ToolCallOutcome | None = None,
    ) -> CanonicalStreamItem | None:
        if kind in {
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
        }:
            tool_call_id = self._canonical_tool_call_id(call)
            if tool_call_id in self._canonical_tool_execution_terminal_ids:
                return None
            self._canonical_tool_execution_terminal_ids.add(tool_call_id)
        return self._append_canonical_item(
            kind,
            data=data,
            correlation=self._canonical_tool_correlation(call),
            metadata=self._tool_outcome_display_metadata(call, result),
        )

    def _tool_call_display_metadata(
        self,
        call: ToolCall,
    ) -> dict[str, Any]:
        projection = self._tool_call_display_projection(call)
        return cast(
            dict[str, Any], tool_display_projection_metadata(projection)
        )

    def _tool_outcome_display_metadata(
        self,
        call: ToolCall,
        result: ToolCallOutcome | None,
    ) -> dict[str, Any]:
        outcome = result or self._empty_tool_call_result(call)
        projection = self._tool_outcome_display_projection(call, outcome)
        return cast(
            dict[str, Any], tool_display_projection_metadata(projection)
        )

    def _tool_call_display_projection(
        self,
        call: ToolCall,
    ) -> ToolDisplayProjection:
        descriptor = self._tool_display_descriptor(call)
        if descriptor is not None:
            projection = self._project_tool_display(
                descriptor,
                call,
                call=call,
            )
            if isinstance(projection, ToolDisplayProjection):
                return projection
        return fallback_tool_call_display_projection(call)

    def _tool_outcome_display_projection(
        self,
        call: ToolCall,
        outcome: ToolCallOutcome,
    ) -> ToolDisplayProjection:
        descriptor = self._tool_display_descriptor(call)
        if descriptor is not None:
            projection = self._project_tool_display(
                descriptor,
                call,
                outcome,
                call=call,
                outcome=outcome,
            )
            if isinstance(projection, ToolDisplayProjection):
                return projection
        return fallback_tool_outcome_display_projection(outcome)

    @classmethod
    def _project_tool_display(
        cls,
        descriptor: ToolDescriptor,
        *args: Any,
        **kwargs: Any,
    ) -> ToolDisplayProjection | None:
        signature_value = cls._tool_display_projector_signature(descriptor)
        if signature_value is None:
            projection = descriptor.project_display(*args)
        elif cls._signature_accepts_positional(
            signature_value,
            args,
        ):
            projection = descriptor.project_display(*args)
        elif cls._signature_accepts_keywords(
            signature_value,
            kwargs,
        ):
            projection = descriptor.project_display(**kwargs)
        else:
            return None
        return (
            projection
            if isinstance(projection, ToolDisplayProjection)
            else None
        )

    @staticmethod
    def _tool_display_projector_signature(
        descriptor: ToolDescriptor,
    ) -> Signature | None:
        try:
            projector = descriptor.display_projector
        except AssertionError:
            return None
        if projector is None:
            return None
        try:
            return signature(projector)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _signature_accepts_positional(
        signature_value: Signature,
        args: tuple[Any, ...],
    ) -> bool:
        try:
            signature_value.bind(*args)
        except TypeError:
            return False
        return True

    @staticmethod
    def _signature_accepts_keywords(
        signature_value: Signature,
        kwargs: dict[str, Any],
    ) -> bool:
        try:
            signature_value.bind(**kwargs)
        except TypeError:
            return False
        return True

    def _tool_display_descriptor(
        self,
        call: ToolCall,
    ) -> ToolDescriptor | None:
        if self._tool_manager is None:
            return None
        describe_tool_call = getattr(
            self._tool_manager,
            "describe_tool_call",
            None,
        )
        if not callable(describe_tool_call):
            return None
        try:
            descriptor = describe_tool_call(call)
        except Exception:
            return None
        if isawaitable(descriptor):
            close = getattr(descriptor, "close", None)
            if callable(close):
                close()
            return None
        return descriptor if isinstance(descriptor, ToolDescriptor) else None

    @staticmethod
    def _empty_tool_call_result(call: ToolCall) -> ToolCallResult:
        return ToolCallResult(
            id=uuid4(),
            call=call,
            name=call.name,
            arguments=call.arguments,
            provider_name=call.provider_name,
            provider_name_encoded=call.provider_name_encoded,
            provider_arguments_malformed=call.provider_arguments_malformed,
            result=None,
        )

    @staticmethod
    def _exception_tool_call_error(
        call: ToolCall,
        exc: Exception,
    ) -> ToolCallError:
        return ToolCallError(
            id=uuid4(),
            call=call,
            name=call.name,
            arguments=call.arguments,
            provider_name=call.provider_name,
            provider_name_encoded=call.provider_name_encoded,
            provider_arguments_malformed=call.provider_arguments_malformed,
            error=exc,
            message=str(exc),
        )

    @staticmethod
    def _is_cancellation_diagnostic(
        result: ToolCallOutcome | None,
    ) -> bool:
        return (
            isinstance(result, ToolCallDiagnostic)
            and result.code is ToolCallDiagnosticCode.CANCELLED
        )

    @classmethod
    def _canonical_tool_diagnostic_payload(
        cls,
        call: ToolCall,
        diagnostic: ToolCallDiagnostic,
    ) -> dict[str, Any]:
        enriched = cls._tool_diagnostic_with_call_signature(
            call,
            diagnostic,
        )
        payload = tool_call_diagnostic_payload(enriched)
        if enriched.code is not ToolCallDiagnosticCode.REPEATED_CALL:
            return payload

        if "details" in payload:
            payload = dict(payload)
            del payload["details"]
        details: dict[str, Any] = {}
        for key in ("attempted_call_signature", "signature"):
            value = enriched.details.get(key)
            if isinstance(value, str):
                details[key] = value
        attempted_signatures = enriched.details.get(
            "attempted_call_signatures"
        )
        if isinstance(attempted_signatures, list) and all(
            isinstance(signature, str) for signature in attempted_signatures
        ):
            details["attempted_call_signatures"] = attempted_signatures
        if details:
            payload["details"] = details
        return payload

    @classmethod
    def _tool_diagnostic_with_call_signature(
        cls,
        call: ToolCall,
        diagnostic: ToolCallDiagnostic,
    ) -> ToolCallDiagnostic:
        if (
            diagnostic.code is not ToolCallDiagnosticCode.REPEATED_CALL
            or "signature" in diagnostic.details
        ):
            return diagnostic
        details: dict[str, Any] = dict(diagnostic.details)
        details["signature"] = cls._call_signature(call)
        return replace(
            diagnostic,
            details=cast(Any, details),
        )

    def _make_tool_stream_event_callback(
        self,
        call: ToolCall,
    ) -> Callable[[ToolExecutionStreamEvent], Awaitable[None]]:
        async def emit(event: ToolExecutionStreamEvent) -> None:
            assert isinstance(event, ToolExecutionStreamEvent)
            tool_call_id = self._canonical_tool_call_id(call)
            if tool_call_id in self._canonical_tool_execution_terminal_ids:
                return
            correlation = self._canonical_tool_correlation(call)
            metadata = dict(event.metadata)
            if event.kind in {
                ToolExecutionStreamKind.STDOUT,
                ToolExecutionStreamKind.STDERR,
                ToolExecutionStreamKind.LOG,
            }:
                item = self._append_canonical_item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    text_delta=event.content,
                    data={
                        "category": event.kind.value,
                        "content": event.content,
                        "metadata": cast(Any, metadata),
                    },
                    correlation=correlation,
                    metadata=self._tool_stream_display_metadata(metadata),
                )
                await self._trigger_canonical_observability_event(
                    EventType.TOOL_PROGRESS,
                    item,
                )
                return
            item = self._append_canonical_item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                data={
                    "category": event.kind.value,
                    "content": event.content,
                    "progress": event.progress,
                    "metadata": cast(Any, metadata),
                },
                correlation=correlation,
                metadata=self._tool_stream_display_metadata(metadata),
            )
            await self._trigger_canonical_observability_event(
                EventType.TOOL_PROGRESS,
                item,
            )

        return emit

    @staticmethod
    def _tool_stream_display_metadata(
        metadata: dict[str, Any],
    ) -> dict[str, Any] | None:
        projection = tool_display_projection_from_metadata(metadata)
        if projection is None:
            return None
        return cast(
            dict[str, Any],
            tool_display_projection_metadata(projection),
        )

    def _canonical_tool_correlation(
        self,
        call: ToolCall,
    ) -> StreamItemCorrelation:
        tool_call_id = self._canonical_tool_call_id(call)
        state = self._canonical_tool_call_lifecycles.get(tool_call_id)
        if state is not None and state.correlation is not None:
            return replace(state.correlation, tool_call_id=tool_call_id)
        return StreamItemCorrelation(tool_call_id=tool_call_id)

    def _append_canonical_model_continuation(
        self,
        kind: StreamItemKind,
        continuation_id: str,
        *,
        data: Any | None = None,
    ) -> CanonicalStreamItem | None:
        return self._append_canonical_item(
            kind,
            data=data,
            correlation=StreamItemCorrelation(
                model_continuation_id=continuation_id
            ),
        )

    def _set_active_model_continuation(self, continuation_id: str) -> None:
        assert self._active_model_continuation_id is None
        self._active_model_continuation_id = continuation_id

    def _finish_active_model_continuation(
        self,
        kind: StreamItemKind,
        *,
        data: Any | None = None,
    ) -> CanonicalStreamItem | None:
        continuation_id = self._active_model_continuation_id
        if continuation_id is None:
            return None
        self._active_model_continuation_id = None
        return self._append_canonical_model_continuation(
            kind,
            continuation_id,
            data=data,
        )

    def _append_canonical_guard_diagnostic(
        self,
        *,
        code: str,
        message: str,
        details: dict[str, Any],
    ) -> None:
        self._append_canonical_item(
            StreamItemKind.STREAM_DIAGNOSTIC,
            data={
                "code": code,
                "message": message,
                "stage": ToolCallDiagnosticStage.GUARD.value,
                "details": cast(Any, details),
            },
        )

    def _canonical_usage(self) -> dict[str, Any]:
        observation = usage_observation_from_response(self)
        if observation is None:
            return {}
        return self._usage_observation_payload(observation)

    @staticmethod
    def _usage_observation_payload(
        observation: UsageObservation,
    ) -> dict[str, Any]:
        return {
            "source": observation.source.value,
            "totals": {
                name: getattr(observation.totals, name)
                for name in USAGE_COUNTER_NAMES
            },
            "metadata": dict(observation.metadata),
        }

    def _canonical_tool_call_id(self, call: ToolCall) -> str:
        key = id(call)
        assigned_id = self._canonical_tool_call_ids_by_object.get(key)
        if assigned_id is not None:
            return assigned_id
        if call.id is not None:
            tool_call_id = str(call.id)
            self._reserve_canonical_tool_call_id(tool_call_id)
            return tool_call_id
        tool_call_id = self._next_generated_tool_call_id()
        self._canonical_tool_call_ids_by_object[key] = tool_call_id
        return tool_call_id

    def _tool_call_with_canonical_id(self, call: ToolCall) -> ToolCall:
        key = id(call)
        assigned_id = self._canonical_tool_call_ids_by_object.get(key)
        if assigned_id is not None:
            if call.id == assigned_id:
                return call
            return replace(call, id=assigned_id)

        if call.id is None:
            tool_call_id = self._next_generated_tool_call_id()
            self._canonical_tool_call_ids_by_object[key] = tool_call_id
            return replace(call, id=tool_call_id)

        source_id = str(call.id)
        if source_id in self._canonical_tool_call_lifecycles:
            self._reserve_canonical_tool_call_id(source_id)
            self._canonical_tool_call_ids_by_object[key] = source_id
            return call
        if not self._canonical_tool_call_id_used(source_id):
            self._reserve_canonical_tool_call_id(source_id)
            self._canonical_tool_call_ids_by_object[key] = source_id
            return call

        tool_call_id = self._next_generated_tool_call_id()
        self._canonical_tool_call_ids_by_object[key] = tool_call_id
        return replace(call, id=tool_call_id)

    def _append_canonical_tool_call_argument_delta(
        self,
        call: ToolCall,
        text_delta: str,
    ) -> None:
        self._canonical_tool_call_argument_delta_ids.add(
            self._canonical_tool_call_id(call)
        )
        self._append_canonical_item(
            StreamItemKind.TOOL_CALL_ARGUMENT_DELTA,
            text_delta=text_delta,
            correlation=StreamItemCorrelation(
                tool_call_id=self._canonical_tool_call_id(call)
            ),
        )

    @classmethod
    def _make_staging_queue(cls) -> Queue[Any]:
        return Queue(maxsize=cls._MAXIMUM_STAGING_QUEUE_ITEMS)

    @staticmethod
    def _put_staging_item(
        queue: Queue[Any],
        item: Any,
        name: str,
    ) -> None:
        try:
            queue.put_nowait(item)
        except Full as exc:
            raise RuntimeError(f"Orchestrator {name} queue is full.") from exc
