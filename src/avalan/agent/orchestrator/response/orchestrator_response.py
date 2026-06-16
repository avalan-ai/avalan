from ....cli import CommandAbortException
from ....entities import (
    Input,
    Message,
    MessageRole,
    MessageToolCall,
    Token,
    TokenDetail,
    ToolCall,
    ToolCallContext,
    ToolCallDiagnostic,
    ToolCallDiagnosticCode,
    ToolCallDiagnosticStage,
    ToolCallError,
    ToolCallOutcome,
    ToolCallResult,
    ToolCallToken,
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
    StreamTerminalOutcome,
    StreamValidationError,
    canonical_item_from_consumer_projection,
    canonical_item_from_token,
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
    gather,
    wait,
)
from base64 import b64encode
from dataclasses import asdict, dataclass, is_dataclass, replace
from inspect import iscoroutine
from json import dumps, loads
from queue import Full, Queue
from time import perf_counter
from typing import Any, AsyncIterator, Awaitable, Callable, cast
from uuid import UUID, uuid4


@dataclass(frozen=True, kw_only=True, slots=True)
class LegacyToolEventShim:
    event_type: EventType
    canonical_kind: StreamItemKind
    canonical_channel: StreamChannel
    owner: str
    removal_condition: str

    def __post_init__(self) -> None:
        assert isinstance(self.event_type, EventType)
        assert self.event_type.value.startswith("tool_")
        assert isinstance(self.canonical_kind, StreamItemKind)
        assert isinstance(self.canonical_channel, StreamChannel)
        assert self.canonical_channel is stream_channel_for_kind(
            self.canonical_kind
        )
        assert isinstance(self.owner, str)
        assert self.owner.strip()
        assert isinstance(self.removal_condition, str)
        assert self.removal_condition.strip()


_LEGACY_TOOL_EVENT_SHIMS: tuple[LegacyToolEventShim, ...] = (
    LegacyToolEventShim(
        event_type=EventType.TOOL_DETECT,
        canonical_kind=StreamItemKind.STREAM_DIAGNOSTIC,
        canonical_channel=StreamChannel.CONTROL,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Tool detection listeners consume canonical stream diagnostics."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_PROCESS,
        canonical_kind=StreamItemKind.TOOL_CALL_READY,
        canonical_channel=StreamChannel.TOOL_CALL,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Tool-call listeners consume canonical tool-call lifecycle items."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_EXECUTE,
        canonical_kind=StreamItemKind.TOOL_EXECUTION_STARTED,
        canonical_channel=StreamChannel.TOOL_EXECUTION,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Tool execution listeners consume canonical execution items."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_RESULT,
        canonical_kind=StreamItemKind.TOOL_EXECUTION_COMPLETED,
        canonical_channel=StreamChannel.TOOL_EXECUTION,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Tool result listeners consume canonical terminal execution items."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_DIAGNOSTIC,
        canonical_kind=StreamItemKind.TOOL_EXECUTION_ERROR,
        canonical_channel=StreamChannel.TOOL_EXECUTION,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Tool diagnostic listeners consume canonical diagnostic items."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_MODEL_RUN,
        canonical_kind=StreamItemKind.MODEL_CONTINUATION_STARTED,
        canonical_channel=StreamChannel.CONTROL,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Continuation listeners consume canonical continuation items."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_MODEL_RESPONSE,
        canonical_kind=StreamItemKind.MODEL_CONTINUATION_COMPLETED,
        canonical_channel=StreamChannel.CONTROL,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Continuation listeners consume canonical continuation items."
        ),
    ),
    LegacyToolEventShim(
        event_type=EventType.TOOL_PROGRESS,
        canonical_kind=StreamItemKind.TOOL_EXECUTION_PROGRESS,
        canonical_channel=StreamChannel.TOOL_EXECUTION,
        owner="agent.orchestrator.response",
        removal_condition=(
            "Live tool listeners consume canonical execution progress items."
        ),
    ),
)


def legacy_tool_event_shim_inventory() -> tuple[LegacyToolEventShim, ...]:
    return _LEGACY_TOOL_EVENT_SHIMS


def classify_legacy_tool_event_shim(
    event_type: EventType,
) -> LegacyToolEventShim:
    assert isinstance(event_type, EventType)
    for shim in _LEGACY_TOOL_EVENT_SHIMS:
        if shim.event_type is event_type:
            return shim
    raise StreamValidationError("unknown legacy tool event shim")


@dataclass(frozen=True, kw_only=True, slots=True)
class _ToolExecutionOutcome:
    call: ToolCall
    context: ToolCallContext
    event: Event
    planned_index: int
    result: ToolCallOutcome | None


@dataclass(frozen=True, kw_only=True, slots=True)
class _OrchestratorResponseStreamItemProjection:
    canonical_item: CanonicalStreamItem | None
    event: Event | None
    legacy_token: Token | TokenDetail | str | None
    legacy_tool_call_token: ToolCallToken | None
    parser_text: str | None
    canonical_source: bool

    def __post_init__(self) -> None:
        assert (
            self.canonical_item is not None or self.event is not None
        ), "projection must carry a canonical item or event"
        assert not (
            self.canonical_item is not None and self.event is not None
        ), "projection must not carry both a canonical item and event"
        assert isinstance(self.canonical_source, bool)
        if self.canonical_item is not None:
            assert isinstance(self.canonical_item, CanonicalStreamItem)
        if self.event is not None:
            assert type(self.event) is Event
            assert self.legacy_token is None
            assert self.legacy_tool_call_token is None
            assert self.parser_text is None
            assert not self.canonical_source
        if self.legacy_tool_call_token is not None:
            assert type(self.legacy_tool_call_token) is ToolCallToken
            assert self.legacy_token is self.legacy_tool_call_token
            assert not self.canonical_source
        if self.parser_text is not None:
            assert isinstance(self.parser_text, str)
            assert self.legacy_token == self.parser_text
            assert not self.canonical_source
        if self.canonical_source:
            assert self.legacy_token is None
            assert self.legacy_tool_call_token is None
            assert self.parser_text is None


def _legacy_tool_event(
    event_type: EventType,
    *,
    payload: dict[str, Any] | None = None,
    started: float | None = None,
    finished: float | None = None,
    elapsed: float | None = None,
) -> Event:
    classify_legacy_tool_event_shim(event_type)
    return Event(
        type=event_type,
        payload=payload,
        started=started,
        finished=finished,
        elapsed=elapsed,
    )


class OrchestratorResponse(AsyncIterator[Token | TokenDetail | Event]):
    """Async iterator handling tool execution during streaming."""

    _MAXIMUM_TOOL_CYCLES = 8
    _MAXIMUM_CONSECUTIVE_NON_EXECUTED_CYCLES = 2
    _MAXIMUM_STAGING_QUEUE_ITEMS = 4096

    _response: TextGenerationResponse
    _response_iterator: AsyncIterator[Any] | None
    _engine_agent: EngineAgent
    _operation: AgentOperation
    _engine_args: dict[str, Any]
    _event_manager: EventManager | None
    _tool_manager: ToolManager | None
    _calls: Queue[ToolCall]
    _tool_call_events: Queue[Event]
    _tool_process_events: Queue[Event]
    _tool_result_emit_events: Queue[Event]
    _tool_result_events: Queue[Event]
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
    _parser_queue: Queue[Token | TokenDetail | Event] | None
    _tool_parser: ToolCallResponseParser | None
    _cancellation_checker: Callable[[], Awaitable[None]] | None
    _canonical_items: list[CanonicalStreamItem]
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
    _last_token_event_canonical_item: CanonicalStreamItem | None
    _canonical_tool_call_argument_delta_ids: set[str]
    _canonical_tool_call_ready_ids: set[str]
    _canonical_tool_execution_started_ids: set[str]
    _canonical_tool_execution_terminal_ids: set[str]
    _canonical_tool_call_ids_by_object: dict[int, str]
    _canonical_tool_call_index: int
    _pending_tool_call: ToolCall | None
    _pending_tool_call_anonymous: bool
    _pending_tool_call_argument_text: str
    _pending_tool_call_unanchored_deltas: list[str]
    _active_model_continuation_id: str | None
    _response_drained: bool

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
        tool_confirm: Callable[[ToolCall], str | None] | None = None,
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
        self._parser_queue = self._make_staging_queue()
        self._cancellation_checker = None
        self._model_responses = [response]
        self._tool_parser = (
            ToolCallResponseParser(self._tool_manager, self._event_manager)
            if enable_tool_parsing and self._tool_manager
            else None
        )
        self._canonical_items = []
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
        self._last_token_event_canonical_item = None
        self._canonical_tool_call_argument_delta_ids = set()
        self._canonical_tool_call_ready_ids = set()
        self._canonical_tool_execution_started_ids = set()
        self._canonical_tool_execution_terminal_ids = set()
        self._canonical_tool_call_ids_by_object = {}
        self._canonical_tool_call_index = 0
        self._pending_tool_call = None
        self._pending_tool_call_anonymous = False
        self._pending_tool_call_argument_text = ""
        self._pending_tool_call_unanchored_deltas = []
        self._active_model_continuation_id = None
        self._response_drained = False

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

        self.__aiter__()
        emitted = 0
        try:
            while True:
                while emitted < len(self._canonical_items):
                    yield StreamConsumerProjection.from_item(
                        self._canonical_items[emitted]
                    )
                    emitted += 1
                try:
                    item = await self.__anext__()
                except StopAsyncIteration:
                    break
                if emitted == len(self._canonical_items):
                    self._append_canonical_projection_item(item)
        except (CancelledError, Exception):
            while emitted < len(self._canonical_items):
                yield StreamConsumerProjection.from_item(
                    self._canonical_items[emitted]
                )
                emitted += 1
            raise

        while emitted < len(self._canonical_items):
            yield StreamConsumerProjection.from_item(
                self._canonical_items[emitted]
            )
            emitted += 1

    def _append_canonical_projection_item(
        self,
        item: CanonicalStreamItem | Token | TokenDetail | Event | str,
    ) -> None:
        projection = self._stream_item_projection(
            item,
            self._canonical_sequence,
        )
        if projection.event is not None:
            return
        if projection.legacy_tool_call_token is not None:
            return
        canonical_item = projection.canonical_item
        assert canonical_item is not None
        if projection.canonical_source:
            self._append_canonical_response_item(canonical_item)
            return
        assert self._canonical_stream_terminal is None
        assert not self._canonical_stream_closed
        matching_item = self._matching_token_event_canonical_item(
            canonical_item
        )
        if matching_item is not None:
            self._canonical_items.append(matching_item)
            self._track_canonical_channel_boundary(matching_item)
            return
        self._canonical_items.append(canonical_item)
        self._canonical_sequence += 1
        self._track_canonical_channel_boundary(canonical_item)

    def _stream_item_projection(
        self,
        item: object,
        sequence: int,
    ) -> _OrchestratorResponseStreamItemProjection:
        assert isinstance(sequence, int), "sequence must be an integer"
        assert sequence >= 0, "sequence must not be negative"
        if isinstance(item, Event):
            return _OrchestratorResponseStreamItemProjection(
                canonical_item=None,
                event=item,
                legacy_token=None,
                legacy_tool_call_token=None,
                parser_text=None,
                canonical_source=False,
            )
        if isinstance(item, CanonicalStreamItem):
            return _OrchestratorResponseStreamItemProjection(
                canonical_item=item,
                event=None,
                legacy_token=None,
                legacy_tool_call_token=None,
                parser_text=None,
                canonical_source=True,
            )
        if isinstance(item, StreamConsumerProjection):
            return _OrchestratorResponseStreamItemProjection(
                canonical_item=canonical_item_from_consumer_projection(item),
                event=None,
                legacy_token=None,
                legacy_tool_call_token=None,
                parser_text=None,
                canonical_source=True,
            )
        canonical_item = canonical_item_from_token(
            cast(Token | TokenDetail | str, item),
            sequence,
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
        )
        return _OrchestratorResponseStreamItemProjection(
            canonical_item=canonical_item,
            event=None,
            legacy_token=cast(Token | TokenDetail | str, item),
            legacy_tool_call_token=(
                item if isinstance(item, ToolCallToken) else None
            ),
            parser_text=item if isinstance(item, str) else None,
            canonical_source=False,
        )

    def _legacy_token_projection(
        self,
        item: object,
        sequence: int,
    ) -> _OrchestratorResponseStreamItemProjection:
        projection = self._stream_item_projection(item, sequence)
        assert projection.legacy_token is not None
        return projection

    def _stage_compatibility_event(self, event: Event) -> Event | None:
        if event.type == EventType.TOOL_PROCESS:
            self._put_staging_item(
                self._tool_process_events,
                event,
                "tool process event",
            )
            return None
        return event

    def _queue_parser_output(
        self,
        item: object,
    ) -> None:
        projection = self._stream_item_projection(
            item,
            self._canonical_sequence,
        )
        if projection.event is not None:
            staged = self._stage_compatibility_event(projection.event)
            if staged is None:
                return
            assert self._parser_queue
            self._put_staging_item(
                self._parser_queue,
                staged,
                "parser item",
            )
            return

        assert self._parser_queue
        self._put_staging_item(
            self._parser_queue,
            item,
            "parser item",
        )

    def _matching_token_event_canonical_item(
        self,
        item: CanonicalStreamItem | Token | TokenDetail | str,
    ) -> CanonicalStreamItem | None:
        pending = self._last_token_event_canonical_item
        self._last_token_event_canonical_item = None
        if pending is None:
            return None
        candidate = (
            item
            if isinstance(item, CanonicalStreamItem)
            else self._legacy_token_projection(
                item,
                pending.sequence,
            ).canonical_item
        )
        assert candidate is not None
        if (
            candidate.kind is pending.kind
            and candidate.channel is pending.channel
            and candidate.text_delta == pending.text_delta
            and candidate.metadata == pending.metadata
        ):
            return pending
        return None

    def _append_canonical_response_item(
        self, item: CanonicalStreamItem
    ) -> None:
        assert isinstance(item, CanonicalStreamItem)
        if item.kind in (
            StreamItemKind.STREAM_STARTED,
            StreamItemKind.STREAM_CLOSED,
        ):
            return
        if item.terminal_outcome is not None:
            self._finish_canonical_stream(
                item.kind,
                data=item.data,
                usage=item.usage,
            )
            return
        if item.kind is StreamItemKind.USAGE_COMPLETED:
            self._append_open_canonical_channel_done_items()
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

    def _has_canonical_final_usage(self) -> bool:
        return any(
            item.kind is StreamItemKind.USAGE_COMPLETED
            for item in self._canonical_items
        )

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
        if self._event_manager:
            self._response.add_done_callback(self._on_consumed)
        if not self._canonical_items:
            self._append_canonical_item(StreamItemKind.STREAM_STARTED)
        self._response_iterator = aiter(self._response)
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
        self._tool_call_events = self._make_staging_queue()
        self._tool_process_events = self._make_staging_queue()
        self._tool_result_emit_events = self._make_staging_queue()
        self._tool_result_events = self._make_staging_queue()
        self._response_drained = False
        self._pending_tool_call = None
        self._pending_tool_call_anonymous = False
        self._pending_tool_call_argument_text = ""
        self._pending_tool_call_unanchored_deltas = []
        self._step = 0
        return self

    async def __anext__(self) -> Token | TokenDetail | Event:
        assert self._response_iterator

        while True:
            item = await self._next_item()
            if item is not None:
                return item

    async def _next_item(self) -> Token | TokenDetail | Event | None:
        assert self._response_iterator

        if self._parser_queue and not self._parser_queue.empty():
            return self._parser_queue.get()

        if not self._tool_result_emit_events.empty():
            return self._tool_result_emit_events.get()

        if self._response_drained and not self._tool_process_events.empty():
            event = self._tool_process_events.get()
            assert event.type == EventType.TOOL_PROCESS
            self._put_staging_item(
                self._tool_call_events,
                event,
                "tool call event",
            )
            return event

        if self._response_drained and not self._tool_call_events.empty():
            event = self._tool_call_events.get()
            assert event.type == EventType.TOOL_PROCESS
            if self._event_manager:
                await self._event_manager.trigger(event)

            calls = cast(list[ToolCall], event.payload or [])
            if calls:
                for call in calls:
                    assert isinstance(call, ToolCall)
                    call = self._tool_call_with_canonical_id(call)
                    self._append_canonical_tool_call_ready(call)
                    self._put_staging_item(self._calls, call, "tool call")

        if self._response_drained and not self._calls.empty():
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            calls = self._drain_tool_call_batch()
            outcomes = await self._execute_tool_call_batch(
                calls,
                confirm=True,
                abort_on_reject=True,
                emit_ready=False,
            )
            ordered = sorted(
                outcomes, key=lambda outcome: outcome.planned_index
            )
            for outcome in ordered:
                self._record_tool_outcome(outcome.result)
                self._tool_context = outcome.context
                self._put_staging_item(
                    self._tool_result_events,
                    outcome.event,
                    "tool result event",
                )
            first, *remaining = ordered
            for outcome in remaining:
                self._put_staging_item(
                    self._tool_result_emit_events,
                    outcome.event,
                    "tool result emit event",
                )
            return first.event

        # Wait until all results are collected
        if (
            self._tool_call_events.empty()
            and self._calls.empty()
            and not self._tool_result_events.empty()
        ):
            result_events: list[Event] = []
            while not self._tool_result_events.empty():
                result_event = self._tool_result_events.get()
                result_events.append(result_event)

            tool_messages = []
            tool_outcomes = []
            for e in result_events:
                assert e.payload is not None and "result" in e.payload
                tool_result = e.payload["result"]
                event_call = e.payload.get("call")
                if not isinstance(
                    tool_result,
                    (ToolCallResult, ToolCallError, ToolCallDiagnostic),
                ):
                    continue
                tool_outcomes.append(tool_result)
                tool_messages.extend(
                    self._tool_observation_messages(
                        tool_result,
                        call=(
                            event_call
                            if isinstance(event_call, ToolCall)
                            else None
                        ),
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
            self._append_canonical_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_STARTED,
                continuation_id,
            )
            try:
                await self._raise_if_cancelled(finish_stream=False)
                event_tool_model_run = _legacy_tool_event(
                    EventType.TOOL_MODEL_RUN,
                    payload={
                        "model_id": self._engine_agent.engine.model_id,
                        "messages": messages,
                        "engine_args": self._engine_args,
                    },
                )
                if self._event_manager:
                    await self._event_manager.trigger(event_tool_model_run)
                inner_response = await self._engine_agent(model_context)
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
            self._set_active_model_continuation(continuation_id)
            self.__aiter__()

            event_tool_model_response = _legacy_tool_event(
                EventType.TOOL_MODEL_RESPONSE,
                payload={
                    "response": inner_response,
                    "model_id": self._engine_agent.engine.model_id,
                    "messages": messages,
                    "engine_args": self._engine_args,
                },
            )
            if self._event_manager:
                await self._event_manager.trigger(event_tool_model_response)

            return event_tool_model_response

        try:
            token = await self._response_iterator.__anext__()
            projection = self._stream_item_projection(
                token,
                self._canonical_sequence,
            )
            if projection.event is not None:
                return await self._emit(projection.event)
            canonical_item = projection.canonical_item
            assert canonical_item is not None
            if projection.canonical_source:
                if (
                    canonical_item.kind is StreamItemKind.ANSWER_DELTA
                    and canonical_item.text_delta is not None
                ):
                    canonical_sequence = self._canonical_sequence
                    self._append_canonical_projection_item(canonical_item)
                    token = Token(token=canonical_item.text_delta)
                    return await self._emit(
                        token, canonical_sequence=canonical_sequence
                    )
                else:
                    self._append_canonical_projection_item(canonical_item)
                    return None
            if projection.legacy_tool_call_token is not None:
                self._record_streamed_tool_call_token(
                    projection.legacy_tool_call_token
                )
        except StopAsyncIteration:
            self._response_drained = True
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            self._queue_pending_tool_call_event()
            if self._tool_parser:
                parser_items: list[Token | TokenDetail | Event] = []
                parser_events: list[Event] = []
                for item in await self._tool_parser.flush():
                    projection = self._stream_item_projection(
                        item,
                        self._canonical_sequence,
                    )
                    if projection.event is not None:
                        event = projection.event
                        if event.type == EventType.TOOL_PROCESS:
                            self._put_staging_item(
                                self._tool_process_events,
                                event,
                                "tool process event",
                            )
                        elif event.type == EventType.TOOL_DIAGNOSTIC:
                            parser_events.append(event)
                        else:
                            self._put_staging_item(
                                self._tool_process_events,
                                event,
                                "tool process event",
                            )
                    else:
                        legacy_token = projection.legacy_token
                        assert legacy_token is not None
                        parser_items.append(
                            cast(Token | TokenDetail | Event, legacy_token)
                        )
                assert self._parser_queue
                for item in parser_items:
                    self._put_staging_item(
                        self._parser_queue,
                        item,
                        "parser item",
                    )
                for event in parser_events:
                    self._put_staging_item(
                        self._parser_queue,
                        event,
                        "parser item",
                    )
                if self._parser_queue and not self._parser_queue.empty():
                    return self._parser_queue.get()
                if not self._tool_process_events.empty():
                    event = self._tool_process_events.get()
                    assert event.type == EventType.TOOL_PROCESS
                    self._put_staging_item(
                        self._tool_call_events,
                        event,
                        "tool call event",
                    )
                    return event
            if not self._tool_process_events.empty():
                event = self._tool_process_events.get()
                assert event.type == EventType.TOOL_PROCESS
                self._put_staging_item(
                    self._tool_call_events,
                    event,
                    "tool call event",
                )
                return event
            if (
                not self._tool_result_emit_events.empty()
                or not self._tool_call_events.empty()
                or not self._calls.empty()
                or not self._tool_result_events.empty()
            ):
                return await self._next_item()
            if self._event_manager and not self._finished:
                self._finished = True
                await self._event_manager.trigger(Event(type=EventType.END))

            self._finish_canonical_stream(StreamItemKind.STREAM_COMPLETED)
            raise
        except CancelledError:
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

        return await self._emit(token)

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
            if self._event_manager:
                await self._event_manager.trigger(
                    _legacy_tool_event(EventType.TOOL_DETECT)
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
                outcomes = await self._execute_tool_call_batch(
                    batch,
                    confirm=True,
                    abort_on_reject=False,
                    emit_ready=True,
                )
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
        self._append_canonical_tool_execution_cancelled(call)

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
            if action == "a":
                self._tool_confirm_all = True
            elif action != "y":
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
                event = _legacy_tool_event(
                    EventType.TOOL_RESULT,
                    payload={
                        "call": call,
                        "planned_index": planned_index,
                        "result": diagnostic,
                    },
                )
                return _ToolExecutionOutcome(
                    call=call,
                    context=self._tool_context or ToolCallContext(),
                    event=event,
                    planned_index=planned_index,
                    result=diagnostic,
                )

        start = perf_counter()
        execute_event = _legacy_tool_event(
            EventType.TOOL_EXECUTE,
            payload={"call": call},
            started=start,
        )
        self._append_canonical_tool_execution_started(call)
        if self._event_manager:
            await self._event_manager.trigger(execute_event)

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
            self._append_canonical_tool_execution_cancelled(call)
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

        self._append_canonical_tool_execution_terminal(call, result)

        end = perf_counter()
        result_event = _legacy_tool_event(
            EventType.TOOL_RESULT,
            payload={
                "call": call,
                "planned_index": planned_index,
                "result": result,
            },
            started=start,
            finished=end,
            elapsed=end - start,
        )
        if self._event_manager:
            await self._trigger_tool_diagnostic_event(
                call=call,
                result=result,
                started=start,
                finished=end,
                elapsed=end - start,
            )
            await self._event_manager.trigger(result_event)

        return _ToolExecutionOutcome(
            call=call,
            context=context,
            event=result_event,
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
            if action == "a":
                self._tool_confirm_all = True
                return []
            if action != "y":
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
                self._append_canonical_tool_execution_cancelled(call)
            event = _legacy_tool_event(
                EventType.TOOL_RESULT,
                payload={
                    "call": call,
                    "planned_index": index,
                    "result": diagnostic,
                },
            )
            outcomes.append(
                _ToolExecutionOutcome(
                    call=call,
                    context=self._tool_context or ToolCallContext(),
                    event=event,
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
        if iscoroutine(action):
            action = await action
        return cast(str | bool | None, action)

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
            self._append_canonical_tool_execution_cancelled(call)

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
            calls.append(self._calls.get())
        batch, remaining = self._split_tool_call_batch(calls)
        for call in remaining:
            self._put_staging_item(self._calls, call, "tool call")
        return batch

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
        try:
            if not response.is_async_generator:
                text = await response.to_str()
                self._finish_active_model_continuation(
                    StreamItemKind.MODEL_CONTINUATION_COMPLETED
                )
                return text, []

            text_parts: list[str] = []
            calls: list[ToolCall] = []
            async for item in response:
                await self._raise_if_cancelled(finish_stream=False)
                projection = self._stream_item_projection(
                    item,
                    self._canonical_sequence,
                )
                if projection.event is not None:
                    continue
                canonical_item = projection.canonical_item
                assert canonical_item is not None
                if projection.canonical_source:
                    if (
                        canonical_item.kind is StreamItemKind.ANSWER_DELTA
                        and canonical_item.text_delta is not None
                    ):
                        text_parts.append(canonical_item.text_delta)
                    continue
                legacy_token = projection.legacy_token
                assert legacy_token is not None
                await self._emit_token_generated_event(legacy_token)
                self._step += 1
                tool_call_token = projection.legacy_tool_call_token
                if tool_call_token is not None:
                    if tool_call_token.call is not None:
                        self._collect_streamed_tool_call_token(
                            tool_call_token,
                            calls,
                        )
                    else:
                        self._record_unanchored_tool_call_delta(
                            tool_call_token.token
                        )
                    continue
                text_delta = canonical_item.text_delta
                assert text_delta is not None
                text_parts.append(text_delta)
            pending_call = self._take_pending_tool_call()
            if pending_call is not None:
                calls.append(pending_call)
            self._finish_active_model_continuation(
                StreamItemKind.MODEL_CONTINUATION_COMPLETED
            )
            return "".join(text_parts), calls
        except CancelledError:
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

    async def _emit_token_generated_event(
        self,
        item: Token | TokenDetail | str,
        *,
        canonical_sequence: int | None = None,
    ) -> CanonicalStreamItem | None:
        if not self._should_emit_token_generated_event():
            return None
        token_str = item.token if hasattr(item, "token") else str(item)
        token_id = getattr(item, "id", None)
        if token_id is None and self._should_enrich_token_ids():
            tokenizer = (
                self._engine_agent.engine.tokenizer
                if self._engine_agent.engine
                else None
            )
            if tokenizer:
                ids = tokenizer.encode(token_str, add_special_tokens=False)
                token_id = ids[0] if ids else None

        sequence = (
            canonical_sequence
            if canonical_sequence is not None
            else self._canonical_sequence
        )
        if canonical_sequence is None:
            self._canonical_sequence += 1
        canonical_item = canonical_item_from_token(
            item,
            sequence,
            stream_session_id=self._canonical_stream_session_id,
            run_id=self._canonical_run_id,
            turn_id=self._canonical_turn_id,
        )

        assert self._event_manager
        payload = {
            "token_id": token_id,
            "model_id": self._engine_agent.engine.model_id,
            "token": token_str,
            "token_type": type(item).__qualname__,
            "step": self._step,
        }
        await self._event_manager.trigger(
            Event(
                type=EventType.TOKEN_GENERATED,
                payload=payload,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        stream_observability_payload(canonical_item)
                    )
                ),
            )
        )
        return canonical_item

    def _should_emit_token_generated_event(self) -> bool:
        if not self._event_manager:
            return False
        should_emit = getattr(self._event_manager, "should_emit", None)
        if not callable(should_emit):
            return True
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
        self._append_canonical_model_continuation(
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            continuation_id,
        )
        try:
            await self._raise_if_cancelled(finish_stream=False)
            event_tool_model_run = _legacy_tool_event(
                EventType.TOOL_MODEL_RUN,
                payload={
                    "model_id": self._engine_agent.engine.model_id,
                    "messages": messages,
                    "engine_args": self._engine_args,
                },
            )
            if self._event_manager:
                await self._event_manager.trigger(event_tool_model_run)
            response = await self._engine_agent(context)
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
                details={"outcome_count": len(outcomes)},
            )
            return False

        cycle_signature = self._tool_cycle_signature(tool_messages)
        if cycle_signature in self._tool_cycle_signatures:
            self._append_canonical_guard_diagnostic(
                code="orchestrator.tool_cycle.duplicate_observation",
                message="Tool cycle stopped after repeated observations.",
                details={
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

    def _record_tool_outcome(self, result: ToolCallOutcome | None) -> None:
        if isinstance(result, (ToolCallResult, ToolCallError)):
            self._call_history.append(result.call)

    async def _trigger_tool_diagnostic_event(
        self,
        *,
        call: ToolCall,
        result: ToolCallOutcome | None,
        started: float,
        finished: float,
        elapsed: float,
    ) -> None:
        if not isinstance(result, ToolCallDiagnostic):
            return
        assert self._event_manager
        await self._event_manager.trigger(
            _legacy_tool_event(
                EventType.TOOL_DIAGNOSTIC,
                payload={
                    "call": call,
                    "diagnostic": result,
                    "diagnostics": [result],
                    "result": result,
                },
                started=started,
                finished=finished,
                elapsed=elapsed,
            )
        )

    def _repeated_call_diagnostic(
        self, call: ToolCall
    ) -> ToolCallDiagnostic | None:
        if self._call_signature(call) not in self._attempted_call_signatures:
            return None
        return ToolCallDiagnostic(
            id=uuid4(),
            call_id=call.id,
            requested_name=self._diagnostic_tool_name(call),
            code=ToolCallDiagnosticCode.REPEATED_CALL,
            stage=ToolCallDiagnosticStage.GUARD,
            message="Tool call repeats a previous attempt.",
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

    async def _emit(
        self,
        item: Token | TokenDetail | Event | str,
        *,
        canonical_sequence: int | None = None,
    ) -> Token | TokenDetail | Event | None:
        projection = self._stream_item_projection(
            item,
            (
                canonical_sequence
                if canonical_sequence is not None
                else self._canonical_sequence
            ),
        )
        if self._event_manager and projection.event is None:
            legacy_token = projection.legacy_token
            assert legacy_token is not None
            canonical_item = await self._emit_token_generated_event(
                legacy_token,
                canonical_sequence=canonical_sequence,
            )
            self._last_token_event_canonical_item = (
                canonical_item if canonical_sequence is None else None
            )

        self._step += 1

        if projection.event is not None:
            return self._stage_compatibility_event(projection.event)

        if projection.parser_text is not None and self._tool_parser:
            items = await self._tool_parser.push(projection.parser_text)
            if not items:
                return None
        else:
            items = [item]

        for it in items:
            self._queue_parser_output(it)

        assert self._parser_queue
        return (
            self._parser_queue.get()
            if not self._parser_queue.empty()
            else None
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
        )
        self._canonical_items.append(item)
        self._canonical_sequence += 1
        self._track_canonical_channel_boundary(item)
        return item

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
        self._append_open_canonical_channel_done_items()
        terminal_usage: Any | None = None
        if kind is StreamItemKind.STREAM_COMPLETED:
            terminal_usage = (
                None
                if self._has_canonical_final_usage()
                else (
                    cast(Any, usage)
                    if usage is not None
                    else self._canonical_usage()
                )
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

    def _append_canonical_tool_call_ready(self, call: ToolCall) -> None:
        tool_call_id = self._canonical_tool_call_id(call)
        if tool_call_id in self._canonical_tool_call_ready_ids:
            return
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
        self._append_canonical_item(
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

    def _append_canonical_tool_execution_started(self, call: ToolCall) -> None:
        tool_call_id = self._canonical_tool_call_id(call)
        if (
            tool_call_id in self._canonical_tool_execution_started_ids
            or tool_call_id in self._canonical_tool_execution_terminal_ids
        ):
            return
        self._canonical_tool_execution_started_ids.add(tool_call_id)
        self._append_canonical_item(
            StreamItemKind.TOOL_EXECUTION_STARTED,
            data={"name": call.name},
            correlation=StreamItemCorrelation(tool_call_id=tool_call_id),
        )

    def _append_canonical_tool_execution_terminal(
        self,
        call: ToolCall,
        result: ToolCallOutcome | None,
    ) -> None:
        if isinstance(result, ToolCallError):
            self._append_canonical_tool_execution_result(
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
            self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_CANCELLED,
                call,
                tool_call_diagnostic_payload(result),
            )
        elif isinstance(result, ToolCallDiagnostic):
            self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_ERROR,
                call,
                tool_call_diagnostic_payload(result),
            )
        elif isinstance(result, ToolCallResult):
            self._append_canonical_tool_execution_result(
                StreamItemKind.TOOL_EXECUTION_COMPLETED,
                call,
                {
                    "name": result.name,
                    "result": cast(Any, result.result),
                    "arguments": cast(Any, result.arguments),
                },
            )
        else:
            self._append_canonical_tool_execution_result(
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
        self, call: ToolCall
    ) -> None:
        self._append_canonical_tool_execution_result(
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
            call,
            {"name": call.name},
        )

    def _append_canonical_tool_execution_result(
        self,
        kind: StreamItemKind,
        call: ToolCall,
        data: Any | None,
    ) -> None:
        if kind in {
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
        }:
            tool_call_id = self._canonical_tool_call_id(call)
            if tool_call_id in self._canonical_tool_execution_terminal_ids:
                return
            self._canonical_tool_execution_terminal_ids.add(tool_call_id)
        self._append_canonical_item(
            kind,
            data=data,
            correlation=StreamItemCorrelation(
                tool_call_id=self._canonical_tool_call_id(call)
            ),
        )

    @staticmethod
    def _is_cancellation_diagnostic(
        result: ToolCallOutcome | None,
    ) -> bool:
        return (
            isinstance(result, ToolCallDiagnostic)
            and result.code is ToolCallDiagnosticCode.CANCELLED
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
            correlation = StreamItemCorrelation(tool_call_id=tool_call_id)
            metadata = dict(event.metadata)
            if event.kind in {
                ToolExecutionStreamKind.STDOUT,
                ToolExecutionStreamKind.STDERR,
                ToolExecutionStreamKind.LOG,
            }:
                self._append_canonical_item(
                    StreamItemKind.TOOL_EXECUTION_OUTPUT,
                    text_delta=event.content,
                    data={
                        "category": event.kind.value,
                        "content": event.content,
                        "metadata": cast(Any, metadata),
                    },
                    correlation=correlation,
                )
                return
            self._append_canonical_item(
                StreamItemKind.TOOL_EXECUTION_PROGRESS,
                data={
                    "category": event.kind.value,
                    "content": event.content,
                    "progress": event.progress,
                    "metadata": cast(Any, metadata),
                },
                correlation=correlation,
            )

        return emit

    def _append_canonical_model_continuation(
        self,
        kind: StreamItemKind,
        continuation_id: str,
        *,
        data: Any | None = None,
    ) -> None:
        self._append_canonical_item(
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
    ) -> None:
        continuation_id = self._active_model_continuation_id
        if continuation_id is None:
            return
        self._active_model_continuation_id = None
        self._append_canonical_model_continuation(
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
        if call.id is not None:
            return str(call.id)
        key = id(call)
        tool_call_id = self._canonical_tool_call_ids_by_object.get(key)
        if tool_call_id is None:
            self._canonical_tool_call_index += 1
            tool_call_id = (
                f"orchestrator-tool-call-{self._canonical_tool_call_index}"
            )
            self._canonical_tool_call_ids_by_object[key] = tool_call_id
        return tool_call_id

    def _tool_call_with_canonical_id(self, call: ToolCall) -> ToolCall:
        if call.id is not None:
            return call
        return replace(call, id=self._canonical_tool_call_id(call))

    def _record_streamed_tool_call_token(self, token: ToolCallToken) -> None:
        call = token.call
        if call is None:
            self._record_unanchored_tool_call_delta(token.token)
            return
        anonymous = call.id is None
        call = self._merge_or_assign_tool_call_id(
            call,
            anonymous=anonymous,
            token_text=token.token,
        )
        if (
            self._pending_tool_call is not None
            and self._canonical_tool_call_id(self._pending_tool_call)
            != self._canonical_tool_call_id(call)
        ):
            self._queue_pending_tool_call_event()
        token_text = self._flush_unanchored_tool_call_deltas(call)
        token_text += token.token
        self._set_pending_tool_call(call, anonymous, token_text)
        if token.token:
            self._append_canonical_tool_call_argument_delta(call, token.token)

    def _collect_streamed_tool_call_token(
        self, token: ToolCallToken, calls: list[ToolCall]
    ) -> None:
        call = token.call
        assert call is not None
        anonymous = call.id is None
        call = self._merge_or_assign_tool_call_id(
            call,
            anonymous=anonymous,
            token_text=token.token,
        )
        if (
            self._pending_tool_call is not None
            and self._canonical_tool_call_id(self._pending_tool_call)
            != self._canonical_tool_call_id(call)
        ):
            pending_call = self._take_pending_tool_call()
            assert pending_call is not None
            calls.append(pending_call)
        token_text = self._flush_unanchored_tool_call_deltas(call)
        token_text += token.token
        self._set_pending_tool_call(call, anonymous, token_text)
        if token.token:
            self._append_canonical_tool_call_argument_delta(call, token.token)

    def _record_unanchored_tool_call_delta(self, token_text: str) -> None:
        if token_text:
            self._pending_tool_call_unanchored_deltas.append(token_text)

    def _flush_unanchored_tool_call_deltas(self, call: ToolCall) -> str:
        token_text = "".join(self._pending_tool_call_unanchored_deltas)
        for delta in self._pending_tool_call_unanchored_deltas:
            self._append_canonical_tool_call_argument_delta(call, delta)
        self._pending_tool_call_unanchored_deltas = []
        return token_text

    def _queue_pending_tool_call_event(self) -> None:
        call = self._take_pending_tool_call()
        if call is None:
            return
        event = _legacy_tool_event(
            EventType.TOOL_PROCESS,
            payload=cast(dict[str, Any], [call]),
            started=perf_counter(),
        )
        self._put_staging_item(
            self._tool_process_events,
            event,
            "tool process event",
        )

    def _take_pending_tool_call(self) -> ToolCall | None:
        call = self._pending_tool_call
        self._pending_tool_call = None
        self._pending_tool_call_anonymous = False
        self._pending_tool_call_argument_text = ""
        self._pending_tool_call_unanchored_deltas = []
        return call

    def _set_pending_tool_call(
        self,
        call: ToolCall,
        anonymous: bool,
        token_text: str,
    ) -> None:
        if (
            self._pending_tool_call is not None
            and self._canonical_tool_call_id(self._pending_tool_call)
            == self._canonical_tool_call_id(call)
        ):
            self._pending_tool_call_argument_text += token_text
        else:
            self._pending_tool_call_argument_text = token_text
        self._pending_tool_call = call
        self._pending_tool_call_anonymous = anonymous

    def _merge_or_assign_tool_call_id(
        self,
        call: ToolCall,
        *,
        anonymous: bool,
        token_text: str,
    ) -> ToolCall:
        if (
            anonymous
            and self._pending_tool_call_anonymous
            and self._pending_tool_call is not None
            and self._pending_tool_call.name == call.name
        ):
            tool_call_id = self._canonical_tool_call_id(
                self._pending_tool_call
            )
            if (
                tool_call_id in self._canonical_tool_call_argument_delta_ids
                and self._anonymous_tool_call_fragment_extends_pending(
                    call,
                    token_text,
                )
            ):
                return replace(call, id=tool_call_id)
        return self._tool_call_with_canonical_id(call)

    def _anonymous_tool_call_fragment_extends_pending(
        self,
        call: ToolCall,
        token_text: str,
    ) -> bool:
        try:
            decoded = loads(self._pending_tool_call_argument_text + token_text)
            return bool(decoded == call.arguments)
        except (TypeError, ValueError):
            return not self._looks_like_anonymous_tool_call_start(token_text)

    @staticmethod
    def _looks_like_anonymous_tool_call_start(token_text: str) -> bool:
        stripped = token_text.lstrip()
        return stripped.startswith(("{", "[", "<tool_call", "<|", "<｜"))

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
