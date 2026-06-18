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
    ToolExecutionStreamEvent,
    ToolExecutionStreamKind,
)
from ....event import Event, EventObservabilityPayload, EventType
from ....event.manager import EventManager
from ....model.call import ModelCallContext
from ....model.response.parsers.tool import ToolCallResponseParser
from ....model.response.text import TextGenerationResponse
from ....model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemCorrelation,
    StreamItemKind,
    StreamProviderEvent,
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
    stream_channel_for_kind,
    stream_observability_payload,
)
from ....task.usage import (
    USAGE_COUNTER_NAMES,
    UsageObservation,
    usage_observation_from_response,
)
from ....tool.manager import ToolManager
from ....utils import tool_call_diagnostic_payload
from ... import AgentOperation
from ...engine import EngineAgent

from asyncio import (
    FIRST_COMPLETED,
    CancelledError,
    Task,
    create_task,
    ensure_future,
    gather,
    sleep,
    wait,
)
from asyncio import (
    Event as AsyncioEvent,
)
from base64 import b64encode
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from inspect import isawaitable
from json import JSONDecodeError, dumps, loads
from queue import Full, Queue
from time import perf_counter
from typing import Any, AsyncIterator, Awaitable, Callable, TypeVar, cast
from uuid import UUID, uuid4

_ToolConfirmationAction = Awaitable[str | bool | None] | str | bool | None
_T = TypeVar("_T")


@dataclass(frozen=True, kw_only=True, slots=True)
class _ToolExecutionOutcome:
    call: ToolCall
    context: ToolCallContext
    planned_index: int
    result: ToolCallOutcome | None


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


class OrchestratorResponse(AsyncIterator[CanonicalStreamItem]):
    """Async iterator handling tool execution during streaming."""

    _LEGACY_STREAM_ERROR = (
        "unsupported legacy orchestrator response stream item"
    )
    _MAXIMUM_TOOL_CYCLES = 8
    _MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES = 2
    _MAXIMUM_STAGING_QUEUE_ITEMS = 4096
    _CANCELLATION_POLL_INTERVAL_SECONDS = 0.01

    _response: TextGenerationResponse
    _response_iterator: AsyncIterator[Any] | None
    _engine_agent: EngineAgent
    _operation: AgentOperation
    _engine_args: dict[str, Any]
    _event_manager: EventManager | None
    _tool_manager: ToolManager | None
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
    _canonical_item_available: AsyncioEvent
    _canonical_tool_call_lifecycles: dict[str, _CanonicalToolCallLifecycle]
    _canonical_tool_call_argument_delta_ids: set[str]
    _canonical_tool_call_ready_ids: set[str]
    _canonical_tool_call_done_ids: set[str]
    _canonical_tool_call_diagnostic_ids: set[str]
    _canonical_tool_execution_started_ids: set[str]
    _canonical_tool_execution_terminal_ids: set[str]
    _canonical_tool_call_reserved_ids: set[str]
    _response_tool_call_id_aliases: dict[str, str]
    _canonical_tool_call_ids_by_object: dict[int, str]
    _canonical_tool_call_index: int
    _canonical_tool_call_diagnostic_index: int
    _active_model_continuation_id: str | None
    _response_drained: bool
    _pending_tool_batch_task: Task[list[_ToolExecutionOutcome]] | None

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
        enable_tool_parsing: bool = True,
    ) -> None:
        assert input and response and engine_agent and operation
        self._input = input
        self._response = response
        self._engine_agent = engine_agent
        self._operation = operation
        self._engine_args = engine_args
        self._event_manager = event_manager
        self._tool_manager = None if tool and tool.is_empty else tool
        self._context = context
        self._finished = False
        self._step = 0
        self._tool_context = None
        self._call_history = []
        self._attempted_call_signatures = set()
        self._tool_cycle_signatures = set()
        self._tool_cycle_count = 0
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
            ToolCallResponseParser(self._tool_manager, self._event_manager)
            if enable_tool_parsing and self._tool_manager
            else None
        )
        self._canonical_items = []
        self._canonical_yield_index = 0
        self._canonical_sequence = 0
        self._canonical_stream_terminal = None
        self._canonical_stream_closed = False
        self._canonical_stream_session_id = str(session_id or uuid4())
        self._canonical_run_id = str(agent_id or uuid4())
        self._canonical_turn_id = str(participant_id or uuid4())
        self._canonical_correlation = StreamItemCorrelation(
            task_id=str(agent_id) if agent_id is not None else None
        )
        self._canonical_answer_started = False
        self._canonical_answer_done = False
        self._canonical_reasoning_started = False
        self._canonical_reasoning_done = False
        self._canonical_item_available = AsyncioEvent()
        self._canonical_tool_call_lifecycles = {}
        self._canonical_tool_call_argument_delta_ids = set()
        self._canonical_tool_call_ready_ids = set()
        self._canonical_tool_call_done_ids = set()
        self._canonical_tool_call_diagnostic_ids = set()
        self._canonical_tool_execution_started_ids = set()
        self._canonical_tool_execution_terminal_ids = set()
        self._canonical_tool_call_reserved_ids = set()
        self._response_tool_call_id_aliases = {}
        self._canonical_tool_call_ids_by_object = {}
        self._canonical_tool_call_index = 0
        self._canonical_tool_call_diagnostic_index = 0
        self._active_model_continuation_id = None
        self._response_drained = False
        self._pending_tool_batch_task = None

    @property
    def input_token_count(self) -> int:
        return self._response.input_token_count

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
        if event.kind in _TOOL_CALL_LIFECYCLE_KINDS:
            return self._append_canonical_provider_tool_call_item(event)
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
            correlation=event.correlation,
        )

    def _append_canonical_provider_tool_call_item(
        self,
        event: StreamProviderEvent,
    ) -> CanonicalStreamItem | None:
        assert event.kind in _TOOL_CALL_LIFECYCLE_KINDS
        tool_call_id = event.correlation.tool_call_id
        if not self._is_valid_tool_call_id(tool_call_id):
            return self._append_canonical_tool_call_lifecycle_diagnostic(
                tool_call_id=None,
                code="orchestrator.tool_call.missing_id",
                message="Tool-call lifecycle item is missing tool_call_id.",
                details={"kind": event.kind.value},
                correlation=event.correlation,
            )
        assert tool_call_id is not None
        if (
            event.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
            and event.text_delta is None
        ):
            tool_call_id = self._canonical_lifecycle_tool_call_id(tool_call_id)
            correlation = replace(
                event.correlation,
                tool_call_id=tool_call_id,
            )
            state = self._canonical_tool_call_lifecycle(tool_call_id)
            state.correlation = self._merged_tool_call_correlation(
                state.correlation,
                correlation,
                tool_call_id=tool_call_id,
            )
            state.invalid = True
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
            correlation=event.correlation,
        )

    def _append_canonical_response_item(
        self, item: CanonicalStreamItem
    ) -> CanonicalStreamItem | None:
        assert isinstance(item, CanonicalStreamItem)
        if item.kind in _TOOL_CALL_LIFECYCLE_KINDS:
            return self._append_canonical_tool_call_lifecycle_item(
                item.kind,
                text_delta=item.text_delta,
                data=item.data,
                usage=item.usage,
                correlation=item.correlation,
                metadata=item.metadata,
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
            self._finish_canonical_stream(
                item.kind,
                data=item.data,
                usage=item.usage,
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
        )
        self._canonical_items.append(canonical_item)
        self._canonical_sequence += 1
        self._track_canonical_channel_boundary(canonical_item)
        self._track_canonical_lifecycle_item(canonical_item)
        self._notify_canonical_item_available()
        return canonical_item

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
        ):
            return False
        canonicalizes = getattr(
            self._tool_parser,
            "canonicalizes_answer_deltas",
            False,
        )
        return canonicalizes if isinstance(canonicalizes, bool) else False

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
            if self._tool_parser:
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

    def _append_canonical_tool_call_lifecycle_item(
        self,
        kind: StreamItemKind,
        *,
        text_delta: str | None = None,
        data: Any | None = None,
        usage: Any | None = None,
        correlation: StreamItemCorrelation,
        metadata: dict[str, Any] | None = None,
    ) -> CanonicalStreamItem | None:
        assert kind in _TOOL_CALL_LIFECYCLE_KINDS
        tool_call_id = correlation.tool_call_id
        if not self._is_valid_tool_call_id(tool_call_id):
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
        if diagnostic is not None:
            if not state.queued:
                state.invalid = True
                self._close_invalid_tool_call_lifecycle(
                    tool_call_id,
                    state=state,
                )
            return diagnostic

        item = self._append_canonical_item(
            kind,
            text_delta=text_delta,
            data=data,
            usage=usage,
            correlation=correlation,
            metadata=metadata,
        )
        if item is None:
            return None
        if kind is StreamItemKind.TOOL_CALL_DONE:
            self._queue_completed_canonical_tool_call(tool_call_id, state)
        return item

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
            task_id=existing.task_id or incoming.task_id,
            artifact_id=existing.artifact_id or incoming.artifact_id,
        )

    def _begin_tool_call_lifecycle_response(self) -> None:
        self._canonical_tool_call_lifecycles = {}
        self._response_tool_call_id_aliases = {}

    def _canonical_lifecycle_tool_call_id(self, source_id: str) -> str:
        assert isinstance(source_id, str)
        assert source_id.strip()
        alias = self._response_tool_call_id_aliases.get(source_id)
        if alias is not None:
            return alias
        if not self._canonical_tool_call_id_used(source_id):
            self._reserve_canonical_tool_call_id(source_id)
            self._response_tool_call_id_aliases[source_id] = source_id
            return source_id
        alias = self._next_generated_tool_call_id()
        self._response_tool_call_id_aliases[source_id] = alias
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
        if not self._canonical_items:
            self._append_canonical_item(StreamItemKind.STREAM_STARTED)
        try:
            output = await self._react(self._response)
        except CancelledError:
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise
        except Exception as exc:
            self._finish_canonical_stream(
                StreamItemKind.STREAM_ERRORED,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            raise
        self._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)
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
        self._tool_context = ToolCallContext(
            input=self._input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
            cancellation_checker=self._cancellation_checker,
        )
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

        while True:
            item = self._next_canonical_yield_item()
            if item is not None:
                return item
            try:
                await self._next_item()
            except StopAsyncIteration:
                item = self._next_canonical_yield_item()
                if item is not None:
                    return item
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

        if self._parser_queue and not self._parser_queue.empty():
            self._append_canonical_response_item(self._parser_queue.get())
            return None

        if self._pending_tool_batch_task is not None:
            await self._await_pending_tool_batch()
            return None

        if self._response_drained and not self._calls.empty():
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            calls = self._drain_tool_call_batch()
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
                self._record_tool_outcome(outcome.result)
                self._tool_context = outcome.context
                tool_result = outcome.result
                if not isinstance(
                    tool_result,
                    (ToolCallResult, ToolCallError, ToolCallDiagnostic),
                ):
                    continue
                tool_outcomes.append(tool_result)
                tool_messages.extend(
                    self._tool_observation_messages(
                        tool_result,
                        call=outcome.call,
                        json_output=True,
                    )
                )

            if not self._should_continue_tool_cycle(
                tool_messages,
                tool_outcomes,
            ):
                if self._event_manager and not self._finished:
                    self._finished = True
                    await self._event_manager.trigger(
                        Event(type=EventType.END)
                    )
                self._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)
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
            self._tool_context = ToolCallContext(
                input=self._input,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
                cancellation_checker=self._cancellation_checker,
            )

            model_context = self._make_child_context(messages)
            continuation_id = str(uuid4())
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
            except CancelledError:
                self._append_canonical_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_CANCELLED,
                    continuation_id,
                )
                self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
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
            self._model_responses.append(inner_response)
            self._response = inner_response
            self._response_drained = False
            self._set_active_model_continuation(continuation_id)
            self._prepare_iteration(reset_yield_index=False)

            return None

        try:
            response_item = self._response_iterator.__anext__()
            item = (
                await self._await_with_session_cancellation(response_item)
                if self._active_model_continuation_id is not None
                else await response_item
            )
            canonical_item = self._canonical_item_from_response_item(item)
            await self._process_canonical_response_item(canonical_item)
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
                or not self._tool_result_outcomes.empty()
            ):
                return await self._next_item()
            if self._event_manager and not self._finished:
                self._finished = True
                await self._event_manager.trigger(Event(type=EventType.END))

            self._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)
            raise
        except CancelledError:
            await self._cancel_active_model_continuation_response()
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_CANCELLED
            )
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise
        except Exception as exc:
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_ERROR,
                data={
                    "error_type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
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
        except CancelledError:
            await self._cancel_pending_tool_batch()
            await self._cancel_active_model_continuation_response()
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_CANCELLED
            )
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
            raise

    async def _await_pending_tool_batch(self) -> None:
        task = self._pending_tool_batch_task
        assert task is not None

        if task.done():
            self._consume_pending_tool_batch(task)
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
        except CancelledError:
            item_task.cancel()
            await gather(item_task, return_exceptions=True)
            if cancellation_task is not None:
                cancellation_task.cancel()
                await gather(cancellation_task, return_exceptions=True)
            await self._cancel_pending_tool_batch()
            self._finish_canonical_stream(StreamItemKind.STREAM_CANCELLED)
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
        self._consume_pending_tool_batch(task)

    async def _cancel_pending_tool_batch(self) -> None:
        task = self._pending_tool_batch_task
        if task is None:
            return
        if not task.done():
            task.cancel()
            await gather(task, return_exceptions=True)
        self._pending_tool_batch_task = None

    async def _cancel_active_model_continuation_response(self) -> None:
        if self._active_model_continuation_id is None:
            return
        await self._response.cancel()

    def _consume_pending_tool_batch(
        self,
        task: Task[list[_ToolExecutionOutcome]],
    ) -> None:
        assert self._pending_tool_batch_task is task
        assert task.done()
        self._pending_tool_batch_task = None
        outcomes = task.result()
        ordered = sorted(outcomes, key=lambda outcome: outcome.planned_index)
        for outcome in ordered:
            self._put_staging_item(
                self._tool_result_outcomes,
                outcome,
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
            self._tool_context = ToolCallContext(
                input=self._input,
                agent_id=self._agent_id,
                participant_id=self._participant_id,
                session_id=self._session_id,
                calls=list(self._call_history),
                cancellation_checker=self._cancellation_checker,
            )

        if not self._tool_manager:
            self._response = response
            return text

        current_response = response
        delta = text
        while True:
            await self._trigger_derived_canonical_observability_event(
                EventType.TOOL_DETECT,
                StreamItemKind.STREAM_DIAGNOSTIC,
                summary={"stage": "tool_detection"},
            )

            calls = (
                structured_calls
                if structured_calls
                else (
                    self._tool_manager.get_calls(delta)
                    if self._tool_manager
                    else None
                )
            )
            if not calls:
                break

            results: list[ToolCallOutcome] = []
            pending_calls = list(calls)
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
        return delta

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
                    context=self._tool_context or ToolCallContext(),
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

        context = ToolCallContext(
            input=self._tool_context.input if self._tool_context else None,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
            cancellation_checker=self._cancellation_checker,
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
                    context=self._tool_context or ToolCallContext(),
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

    def _drain_tool_call_batch(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        while not self._calls.empty():
            call = self._calls.get()
            if self._should_execute_staged_tool_call(call):
                calls.append(call)
        if not calls:
            return []
        batch, remaining = self._split_tool_call_batch(calls)
        for call in remaining:
            self._put_staging_item(self._calls, call, "tool call")
        return batch

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
            if not response.is_async_generator:
                text = (
                    await self._await_with_session_cancellation(
                        response.to_str()
                    )
                    if self._active_model_continuation_id is not None
                    else await response.to_str()
                )
                text, calls = await self._non_stream_response_text_and_calls(
                    text
                )
                self._finish_active_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_COMPLETED
                )
                return text, calls

            self._begin_tool_call_lifecycle_response()
            self._calls = self._make_staging_queue()
            text_parts: list[str] = []
            streamed_calls: list[ToolCall] = []

            def collect_parser_output(start_index: int) -> None:
                self._collect_response_text_and_calls(
                    start_index,
                    text_parts,
                    streamed_calls,
                )

            response_iterator = aiter(response)
            while True:
                try:
                    response_item = response_iterator.__anext__()
                    item = (
                        await self._await_with_session_cancellation(
                            response_item
                        )
                        if self._active_model_continuation_id is not None
                        else await response_item
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
        except CancelledError:
            await self._cancel_active_model_continuation_response()
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
            call = self._calls.get()
            if self._should_execute_staged_tool_call(call):
                calls.append(call)

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
                    result,
                    json_output=False,
                )
            )

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
        self._tool_context = ToolCallContext(
            input=self._input,
            agent_id=self._agent_id,
            participant_id=self._participant_id,
            session_id=self._session_id,
            calls=list(self._call_history),
            cancellation_checker=self._cancellation_checker,
        )

        context = self._make_child_context(messages)
        continuation_id = str(uuid4())
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
        self._model_responses.append(response)
        self._set_active_model_continuation(continuation_id)
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
        if cycle_signature in self._tool_cycle_signatures:
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

        if self._tool_cycle_count >= self._MAXIMUM_TOOL_CYCLES:
            self._append_canonical_guard_diagnostic(
                code="orchestrator.tool_cycle.limit_exceeded",
                message="Tool cycle stopped after reaching the cycle limit.",
                details={
                    "attempted_call_signatures": (
                        self._attempted_call_signature_details()
                    ),
                    "cycle_count": self._tool_cycle_count,
                    "maximum_cycles": self._MAXIMUM_TOOL_CYCLES,
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
        if isinstance(result, (ToolCallResult, ToolCallError)):
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

        return [
            Message(
                role=MessageRole.ASSISTANT,
                tool_calls=[
                    MessageToolCall(
                        id=str(outcome.call.id),
                        name=outcome.name,
                        arguments=cast(Any, outcome.arguments),
                    )
                ],
            ),
            Message(
                role=MessageRole.TOOL,
                name=outcome.name,
                arguments=outcome.arguments,
                content=cls._outcome_content(
                    outcome,
                    json_output=json_output,
                ),
                tool_call_result=(
                    outcome if isinstance(outcome, ToolCallResult) else None
                ),
                tool_call_error=(
                    outcome if isinstance(outcome, ToolCallError) else None
                ),
            ),
        ]

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
                        arguments=cast(Any, arguments),
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

    def _make_child_context(self, messages: Input) -> ModelCallContext:
        parent_context = self._context
        root_parent = (
            parent_context.root_parent or parent_context
            if parent_context
            else None
        )
        context = ModelCallContext(
            specification=self._operation.specification,
            input=messages,
            engine_args=dict(self._engine_args),
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
        )
        self._context = context
        return context

    def _append_canonical_item(
        self,
        kind: StreamItemKind,
        *,
        text_delta: str | None = None,
        data: Any | None = None,
        usage: Any | None = None,
        correlation: StreamItemCorrelation | None = None,
        terminal_outcome: StreamTerminalOutcome | None = None,
        metadata: dict[str, Any] | None = None,
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
            metadata=metadata or {},
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

    def _finish_canonical_stream(
        self,
        kind: StreamItemKind,
        *,
        data: Any | None = None,
        usage: Any | None = None,
    ) -> None:
        outcomes = {
            StreamItemKind.STREAM_COMPLETED: StreamTerminalOutcome.COMPLETED,
            StreamItemKind.STREAM_ERRORED: StreamTerminalOutcome.ERRORED,
            StreamItemKind.STREAM_CANCELLED: StreamTerminalOutcome.CANCELLED,
        }
        outcome = outcomes[kind]
        if self._canonical_stream_terminal is not None:
            return
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
            )
        elif self._is_cancellation_diagnostic(result):
            assert isinstance(result, ToolCallDiagnostic)
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                call,
                self._canonical_tool_diagnostic_payload(call, result),
            )
        elif isinstance(result, ToolCallDiagnostic):
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                call,
                self._canonical_tool_diagnostic_payload(call, result),
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
            )
        else:
            return self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                call,
                None,
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
            )
            await self._trigger_canonical_observability_event(
                EventType.TOOL_PROGRESS,
                item,
            )

        return emit

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
