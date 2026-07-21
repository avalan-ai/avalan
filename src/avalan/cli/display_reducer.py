"""Reduce canonical stream projections into CLI display snapshots."""

from ..event import EventType
from ..model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamConsumerProjection,
    StreamItemKind,
    iter_stream_consumer_projections,
    stream_projection_text_delta,
)
from ..tool.display import (
    ToolDisplayProjection,
    tool_display_projection_from_metadata,
)
from .display import CliStreamDisplayConfig
from .display_safety import (
    MAX_SUMMARY_CHARS,
    REDACTED,
    contains_sensitive_marker,
    event_type_value,
    safe_text,
    truncate_text,
    value_from,
)
from .display_snapshot import (
    CliStreamSnapshot,
    CliStreamSnapshotBuilder,
    ToolResultStatus,
    ToolStatus,
    display_token_snapshot_from_projection,
)

from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol, runtime_checkable

CliStreamClock = Callable[[], float]


@runtime_checkable
class CliSideChannelEvent(Protocol):
    """Describe event objects emitted by CLI side-channel listeners."""

    type: object
    payload: object
    observability: object
    started: object
    finished: object
    elapsed: object


CliStreamReducerInput = StreamConsumerProjection | CliSideChannelEvent


@dataclass(slots=True)
class _ToolArgumentDisplayState:
    """Retain bounded display text for one streamed tool call."""

    text: str = ""
    dropped_characters: int = 0
    sensitive_seen: bool = False
    _scan_tail: str = ""

    def append(self, chunk: str) -> None:
        assert isinstance(chunk, str)
        if not chunk:
            return
        scan_text = self._scan_tail + chunk
        self.sensitive_seen = self.sensitive_seen or contains_sensitive_marker(
            scan_text
        )
        self._scan_tail = scan_text[-MAX_SUMMARY_CHARS:]
        retained = self.text + chunk
        if len(retained) <= MAX_SUMMARY_CHARS:
            self.text = retained
            return
        dropped = len(retained) - MAX_SUMMARY_CHARS
        self.dropped_characters += dropped
        self.text = truncate_text(
            retained[-MAX_SUMMARY_CHARS:],
            MAX_SUMMARY_CHARS,
        )

    def materialize(self) -> str:
        """Return bounded text or redact truncated sensitive data."""
        if self.sensitive_seen and self.dropped_characters:
            return REDACTED
        return self.text


def default_cli_stream_clock() -> float:
    """Return the default monotonic CLI stream timestamp."""
    return perf_counter()


class CliStreamSnapshotReducer:
    """Reduce CLI stream projections into immutable snapshots."""

    def __init__(
        self,
        config: CliStreamDisplayConfig,
        *,
        clock: CliStreamClock = default_cli_stream_clock,
    ) -> None:
        assert isinstance(config, CliStreamDisplayConfig)
        assert callable(clock)
        self._builder = CliStreamSnapshotBuilder(config)
        self._clock = clock
        self._started_at: float | None = None
        self._terminal_seen = False
        self._first_visible_token_seen = False
        self._visible_token_count = 0
        self._time_to_n_token_recorded = False
        self._reasoning_started_at: float | None = None
        self._tool_started_at: dict[str, float] = {}
        self._tool_argument_state: dict[str, _ToolArgumentDisplayState] = {}
        self._tool_names: dict[str, str] = {}
        self._canonical_tool_call_ids: set[str] = set()
        self._side_channel_tool_event_ids: dict[str, dict[str, int]] = {}
        self._side_channel_tool_event_order: deque[str] = deque()
        self._side_channel_tool_event_counts: dict[str, int] = {}

    @property
    def terminal_completed(self) -> bool:
        """Return whether a terminal projection has completed display state."""
        return self._terminal_seen

    def reduce_projection(
        self,
        projection: StreamConsumerProjection,
    ) -> CliStreamSnapshot:
        """Reduce one canonical consumer projection."""
        self.apply_projection(projection)
        return self.snapshot()

    def apply_projection(
        self,
        projection: StreamConsumerProjection,
    ) -> bool:
        """Apply one canonical projection without building a snapshot."""
        assert isinstance(projection, StreamConsumerProjection)
        if self._is_duplicate_terminal_projection(projection):
            return False
        text_delta = stream_projection_text_delta(projection)
        display_token = display_token_snapshot_from_projection(projection)
        now = self._active_stream_time()
        if now is not None:
            self._mark_started(now)
            self._builder.update_timing(updated_at=now)

        tool_events_changed = False
        if projection.tool_call_id is not None:
            tool_events_changed = self._remember_canonical_tool_call(
                projection.tool_call_id
            )
        self._builder.add_projection_summary(projection)
        self._update_usage(projection)
        self._builder.observe_reasoning_projection(projection)
        if text_delta is not None:
            self._reduce_text_projection(projection, text_delta, now)

        self._reduce_projection_kind(projection, now)
        self._builder.add_display_token_from_projection(projection)
        return self._projection_changes_display(
            projection,
            text_delta=text_delta,
            display_token_seen=display_token is not None,
            tool_events_changed=tool_events_changed,
        )

    def reduce_event(self, event: CliSideChannelEvent) -> CliStreamSnapshot:
        """Reduce one CLI side-channel event summary."""
        self.apply_event(event)
        return self.snapshot()

    def apply_event(self, event: CliSideChannelEvent) -> bool:
        """Apply one CLI side-channel event without building a snapshot."""
        assert isinstance(event, CliSideChannelEvent)
        event_type = event_type_value(event.type)
        if event_type == EventType.TOKEN_GENERATED.value:
            return False
        if event_type == EventType.TOOL_PROGRESS.value:
            return False
        if event_type.startswith("tool_"):
            return self._reduce_tool_event(event)
        self._builder.add_event(event)
        return self._builder.retention.event_history_limit > 0

    def snapshot(self) -> CliStreamSnapshot:
        """Return the current immutable CLI stream snapshot."""
        return self._builder.snapshot()

    def _active_stream_time(self) -> float | None:
        if self._terminal_seen:
            return None
        return self._clock()

    def _mark_started(self, now: float) -> None:
        if self._started_at is not None:
            return
        self._started_at = now
        self._builder.update_timing(started_at=now)

    def _reduce_text_projection(
        self,
        projection: StreamConsumerProjection,
        text_delta: str,
        now: float | None,
    ) -> None:
        if (
            projection.channel is StreamChannel.ANSWER
            and projection.kind is StreamItemKind.ANSWER_DELTA
        ):
            self._finish_reasoning(now)
            self._builder.append_answer_text(text_delta)
            self._record_visible_token(now)
        elif (
            projection.channel is StreamChannel.REASONING
            and projection.kind is StreamItemKind.REASONING_DELTA
        ):
            if self._reasoning_started_at is None and now is not None:
                self._reasoning_started_at = now
            self._record_visible_token(now)
        elif (
            projection.channel is StreamChannel.TOOL_CALL
            and projection.kind is StreamItemKind.TOOL_CALL_ARGUMENT_DELTA
        ):
            tool_id = _tool_call_id(projection)
            self._tool_argument_state.setdefault(
                tool_id,
                _ToolArgumentDisplayState(),
            ).append(text_delta)
            self._builder.append_tool_call_request_text(text_delta)

    def _record_visible_token(self, now: float | None) -> None:
        if now is not None and self._started_at is not None:
            elapsed = now - self._started_at
            if not self._first_visible_token_seen:
                self._first_visible_token_seen = True
                self._builder.update_timing(first_token_seconds=elapsed)
            self._visible_token_count += 1
            time_to_n_token = self._builder.display.display_time_to_n_token
            if (
                time_to_n_token is not None
                and not self._time_to_n_token_recorded
                and self._visible_token_count >= time_to_n_token
            ):
                self._time_to_n_token_recorded = True
                self._builder.update_timing(time_to_n_token_seconds=elapsed)

    def _finish_reasoning(self, now: float | None) -> None:
        if now is None or self._reasoning_started_at is None:
            return
        self._builder.update_timing(
            reasoning_seconds=now - self._reasoning_started_at
        )
        self._reasoning_started_at = None

    def _reduce_projection_kind(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        kind = projection.kind
        if kind is StreamItemKind.TOOL_CALL_READY:
            self._remember_tool_ready(projection)
        elif kind is StreamItemKind.TOOL_EXECUTION_STARTED:
            self._start_tool_execution(projection, now)
        elif kind in (
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
        ):
            self._update_tool_execution(projection, now)
        elif kind in (
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
        ):
            self._finish_tool_execution(projection, now)
        elif kind in (
            StreamItemKind.FLOW_EVENT,
            StreamItemKind.STREAM_DIAGNOSTIC,
        ):
            self._add_projection_event(projection)
        elif kind in (StreamItemKind.MODEL_CONTINUATION_STARTED,):
            self._start_model_continuation(projection, now)
        elif kind in (
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            StreamItemKind.MODEL_CONTINUATION_ERROR,
            StreamItemKind.MODEL_CONTINUATION_CANCELLED,
        ):
            self._finish_model_continuation(projection, now)
        elif kind in (
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
            StreamItemKind.STREAM_CANCELLED,
            StreamItemKind.STREAM_INPUT_REQUIRED,
        ):
            self._finish_stream(projection, now)

    def _projection_changes_display(
        self,
        projection: StreamConsumerProjection,
        *,
        text_delta: str | None,
        display_token_seen: bool,
        tool_events_changed: bool,
    ) -> bool:
        if tool_events_changed:
            return True
        if display_token_seen and self._builder.display.show_token_details:
            return True
        if self._projection_adds_stats_summary(projection):
            return True
        if text_delta:
            if projection.channel is StreamChannel.ANSWER:
                return True
            if projection.channel in (
                StreamChannel.REASONING,
                StreamChannel.TOOL_CALL,
            ):
                return (
                    self._builder.display.show_reasoning
                    if projection.channel is StreamChannel.REASONING
                    else self._builder.display.show_stats
                )
        if projection.kind in (
            StreamItemKind.TOOL_EXECUTION_STARTED,
            StreamItemKind.TOOL_EXECUTION_OUTPUT,
            StreamItemKind.TOOL_EXECUTION_PROGRESS,
            StreamItemKind.TOOL_EXECUTION_COMPLETED,
            StreamItemKind.TOOL_EXECUTION_ERROR,
            StreamItemKind.TOOL_EXECUTION_CANCELLED,
            StreamItemKind.MODEL_CONTINUATION_STARTED,
            StreamItemKind.MODEL_CONTINUATION_COMPLETED,
            StreamItemKind.MODEL_CONTINUATION_ERROR,
            StreamItemKind.MODEL_CONTINUATION_CANCELLED,
        ):
            return self._builder.display.show_tools
        if projection.kind in (
            StreamItemKind.FLOW_EVENT,
            StreamItemKind.STREAM_DIAGNOSTIC,
        ):
            return self._builder.display.show_events
        if projection.kind in (
            StreamItemKind.USAGE_UPDATE,
            StreamItemKind.USAGE_COMPLETED,
        ):
            return self._builder.display.show_stats
        return projection.kind in (
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
            StreamItemKind.STREAM_CANCELLED,
            StreamItemKind.STREAM_INPUT_REQUIRED,
        )

    def _projection_adds_stats_summary(
        self,
        projection: StreamConsumerProjection,
    ) -> bool:
        if not self._builder.display.show_stats:
            return False
        if (
            projection.usage is not None
            and self._builder.retention.usage_summary_history_limit > 0
        ):
            return True
        return (
            self._builder.retention.projection_metadata_history_limit > 0
            and (projection.data is not None or bool(projection.metadata))
        )

    def _is_duplicate_terminal_projection(
        self,
        projection: StreamConsumerProjection,
    ) -> bool:
        return self._terminal_seen and projection.kind in (
            StreamItemKind.STREAM_COMPLETED,
            StreamItemKind.STREAM_ERRORED,
            StreamItemKind.STREAM_CANCELLED,
            StreamItemKind.STREAM_INPUT_REQUIRED,
        )

    def _remember_tool_ready(
        self,
        projection: StreamConsumerProjection,
    ) -> None:
        tool_id = _tool_call_id(projection)
        self._remember_canonical_tool_call(tool_id)
        name = _tool_name_from_projection(projection)
        if name is not None:
            self._tool_names[tool_id] = name

    def _start_tool_execution(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        tool_id = _tool_call_id(projection)
        self._remember_canonical_tool_call(tool_id)
        if now is not None:
            self._tool_started_at[tool_id] = now
        self._builder.add_active_tool(
            tool_call_id=tool_id,
            name=self._tool_name(tool_id, projection),
            arguments=self._tool_arguments(tool_id),
            display_projection=_tool_display_projection(projection),
            provider_name=projection.provider_family,
            sequence=projection.sequence,
            started_at=now,
        )

    def _update_tool_execution(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        tool_id = _tool_call_id(projection)
        self._remember_canonical_tool_call(tool_id)
        self._builder.update_active_tool(
            tool_call_id=tool_id,
            name=self._tool_name(tool_id, projection),
            display_projection=_tool_display_projection(projection),
            sequence=projection.sequence,
            updated_at=now,
        )

    def _finish_tool_execution(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        tool_id = _tool_call_id(projection)
        self._remember_canonical_tool_call(tool_id)
        elapsed = self._tool_elapsed(tool_id, now)
        tool_status: ToolStatus = "completed"
        result_status: ToolResultStatus = "result"
        if projection.kind is StreamItemKind.TOOL_EXECUTION_ERROR:
            tool_status = "error"
            result_status = "error"
        elif projection.kind is StreamItemKind.TOOL_EXECUTION_CANCELLED:
            tool_status = "cancelled"
            result_status = "error"

        name = self._tool_name(tool_id, projection)
        display_projection = _tool_display_projection(projection)
        self._builder.complete_tool(
            tool_call_id=tool_id,
            status=tool_status,
            name=name,
            display_projection=display_projection,
            elapsed_seconds=elapsed,
            sequence=projection.sequence,
        )
        self._builder.add_tool_result_summary(
            tool_call_id=tool_id,
            name=name,
            status=result_status,
            result=_tool_terminal_result(projection),
            arguments_count=_tool_arguments_count(
                self._tool_argument_state.get(tool_id)
            ),
            display_projection=display_projection,
            sequence=projection.sequence,
            elapsed_seconds=elapsed,
        )
        self._tool_argument_state.pop(tool_id, None)
        self._tool_names.pop(tool_id, None)

    def _tool_elapsed(
        self,
        tool_id: str,
        now: float | None,
    ) -> float | None:
        started_at = self._tool_started_at.pop(tool_id, None)
        if started_at is None or now is None:
            return None
        return now - started_at

    def _add_projection_event(
        self,
        projection: StreamConsumerProjection,
    ) -> None:
        self._builder.add_event_summary(
            event_type=projection.kind,
            payload=_projection_event_payload(projection),
            observability=_projection_observability(projection),
            sequence=projection.sequence,
        )

    def _start_model_continuation(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        self._builder.add_active_model_continuation(
            model_continuation_id=_model_continuation_id(projection),
            sequence=projection.sequence,
            started_at=now,
        )

    def _finish_model_continuation(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        self._builder.finish_model_continuation(
            model_continuation_id=_model_continuation_id(projection),
            updated_at=now,
        )

    def _finish_stream(
        self,
        projection: StreamConsumerProjection,
        now: float | None,
    ) -> None:
        self._terminal_seen = True
        if now is not None:
            self._finish_reasoning(now)
            elapsed = (
                None if self._started_at is None else now - self._started_at
            )
            self._builder.update_timing(
                updated_at=now,
                finished_at=now,
                elapsed_seconds=elapsed,
            )
        self._builder.set_terminal(
            completed=True,
            outcome=projection.terminal_outcome,
            sequence=projection.sequence,
            error=(
                projection.data
                if projection.kind is StreamItemKind.STREAM_ERRORED
                else None
            ),
        )

    def _update_usage(self, projection: StreamConsumerProjection) -> None:
        usage = projection.usage
        if not isinstance(usage, Mapping):
            return
        self._builder.update_token_counts(
            input_tokens=_usage_int(usage, "input_tokens"),
            cached_input_tokens=(
                _usage_int(usage, "cached_input_tokens")
                or _nested_usage_int(
                    usage, "input_tokens_details", "cached_tokens"
                )
                or _nested_usage_int(
                    usage, "prompt_tokens_details", "cached_tokens"
                )
            ),
            output_tokens=_usage_int(usage, "output_tokens"),
            reasoning_usage_tokens=(
                _usage_int(usage, "reasoning_usage_tokens")
                or _usage_int(usage, "reasoning_tokens")
                or _nested_usage_int(
                    usage, "output_tokens_details", "reasoning_tokens"
                )
                or _nested_usage_int(
                    usage, "completion_tokens_details", "reasoning_tokens"
                )
            ),
            total_tokens=_usage_int(usage, "total_tokens"),
        )

    def _tool_name(
        self,
        tool_id: str,
        projection: StreamConsumerProjection,
    ) -> str:
        name = _tool_name_from_projection(projection)
        if name is not None:
            self._tool_names[tool_id] = name
            return name
        return self._tool_names.get(tool_id, "tool")

    def _tool_arguments(self, tool_id: str) -> str | None:
        state = self._tool_argument_state.get(tool_id)
        if state is None:
            return None
        text = state.materialize()
        return text or None

    def _remember_canonical_tool_call(self, tool_id: str) -> bool:
        normalized_tool_id = safe_text(tool_id)
        self._canonical_tool_call_ids.add(normalized_tool_id)
        removed_count = self._builder.remove_tool_events_for_tool_call(
            normalized_tool_id
        )
        event_ids = self._side_channel_tool_event_ids.pop(
            normalized_tool_id,
            {},
        )
        for event_id in event_ids:
            removed_count += self._builder.remove_tool_events_for_tool_call(
                event_id
            )
            self._forget_side_channel_tool_event(event_id)
        return removed_count > 0

    def _reduce_tool_event(self, event: CliSideChannelEvent) -> bool:
        tool_call_ids, name = _tool_event_identity(event)
        normalized_tool_ids = tuple(
            _normalized_tool_call_id(tool_call_id)
            for tool_call_id in tool_call_ids
        )
        if normalized_tool_ids and any(
            tool_id in self._canonical_tool_call_ids
            for tool_id in normalized_tool_ids
        ):
            return False
        event_id = "|".join(normalized_tool_ids) or None
        self._builder.add_tool_event(
            event,
            tool_call_id=event_id,
            name=name,
        )
        changed = self._builder.retention.internal_tool_history_limit > 0
        if event_id is not None:
            self._side_channel_tool_event_order.append(event_id)
            self._side_channel_tool_event_counts[event_id] = (
                self._side_channel_tool_event_counts.get(event_id, 0) + 1
            )
            for tool_id in normalized_tool_ids:
                event_counts = self._side_channel_tool_event_ids.setdefault(
                    tool_id,
                    {},
                )
                event_counts[event_id] = event_counts.get(event_id, 0) + 1
            self._trim_side_channel_tool_event_index()
        return changed

    def _drop_side_channel_tool_event_reference(self, event_id: str) -> None:
        for tool_id, event_ids in tuple(
            self._side_channel_tool_event_ids.items()
        ):
            count = event_ids.get(event_id)
            if count is None:
                continue
            if count > 1:
                event_ids[event_id] = count - 1
            else:
                del event_ids[event_id]
            if not event_ids:
                del self._side_channel_tool_event_ids[tool_id]

    def _forget_side_channel_tool_event(self, event_id: str) -> None:
        self._side_channel_tool_event_counts.pop(event_id, None)
        for tool_id, event_ids in tuple(
            self._side_channel_tool_event_ids.items()
        ):
            event_ids.pop(event_id, None)
            if not event_ids:
                del self._side_channel_tool_event_ids[tool_id]

    def _trim_side_channel_tool_event_index(self) -> None:
        self._drop_stale_side_channel_tool_event_order()
        limit = self._builder.retention.internal_tool_history_limit
        retained_count = sum(self._side_channel_tool_event_counts.values())
        while retained_count > limit:
            event_id = self._side_channel_tool_event_order.popleft()
            self._drop_side_channel_tool_event_reference(event_id)
            count = self._side_channel_tool_event_counts[event_id]
            if count > 1:
                self._side_channel_tool_event_counts[event_id] = count - 1
            else:
                del self._side_channel_tool_event_counts[event_id]
            retained_count -= 1
            self._drop_stale_side_channel_tool_event_order()

    def _drop_stale_side_channel_tool_event_order(self) -> None:
        while (
            self._side_channel_tool_event_order
            and self._side_channel_tool_event_order[0]
            not in self._side_channel_tool_event_counts
        ):
            self._side_channel_tool_event_order.popleft()


async def iter_cli_stream_snapshots(
    items: AsyncIterable[CliStreamReducerInput],
    config: CliStreamDisplayConfig,
    *,
    clock: CliStreamClock = default_cli_stream_clock,
) -> AsyncIterator[CliStreamSnapshot]:
    """Yield snapshots while reducing one ordered CLI stream."""
    assert isinstance(items, AsyncIterable)
    reducer = CliStreamSnapshotReducer(config, clock=clock)
    async for item in items:
        if isinstance(item, CliSideChannelEvent):
            yield reducer.reduce_event(item)
        else:
            yield reducer.reduce_projection(item)


async def iter_cli_canonical_stream_snapshots(
    items: AsyncIterable[CanonicalStreamItem],
    config: CliStreamDisplayConfig,
    *,
    clock: CliStreamClock = default_cli_stream_clock,
) -> AsyncIterator[CliStreamSnapshot]:
    """Yield snapshots while reducing one canonical item stream."""
    assert isinstance(items, AsyncIterable)
    reducer = CliStreamSnapshotReducer(config, clock=clock)
    async for projection in iter_stream_consumer_projections(items):
        yield reducer.reduce_projection(projection)


def _tool_call_id(projection: StreamConsumerProjection) -> str:
    tool_id = projection.tool_call_id
    assert tool_id is not None
    return tool_id


def _model_continuation_id(projection: StreamConsumerProjection) -> str:
    continuation_id = projection.correlation.model_continuation_id
    if continuation_id is not None:
        return continuation_id
    return f"model-continuation:{projection.sequence}"


def _tool_name_from_projection(
    projection: StreamConsumerProjection,
) -> str | None:
    data = projection.data
    if data is None:
        return None
    name = value_from(data, "name") or value_from(data, "tool_name")
    return None if name is None else safe_text(name)


def _tool_terminal_result(projection: StreamConsumerProjection) -> object:
    if projection.data is not None:
        return projection.data
    return {"kind": projection.kind.value}


def _tool_display_projection(
    projection: StreamConsumerProjection,
) -> ToolDisplayProjection | None:
    return tool_display_projection_from_metadata(projection.metadata)


def _tool_arguments_count(state: _ToolArgumentDisplayState | None) -> int:
    if state is None or not state.text:
        return 0
    return 1


def _projection_event_payload(
    projection: StreamConsumerProjection,
) -> object | None:
    if projection.text_delta is None:
        return projection.data
    if projection.data is None:
        return {"text": projection.text_delta}
    return {"text": projection.text_delta, "data": projection.data}


def _projection_observability(
    projection: StreamConsumerProjection,
) -> object | None:
    if not projection.metadata:
        return None
    return projection.metadata


def _usage_int(usage: Mapping[str, object], key: str) -> int | None:
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _nested_usage_int(
    usage: Mapping[str, object],
    parent_key: str,
    key: str,
) -> int | None:
    parent = usage.get(parent_key)
    if not isinstance(parent, Mapping):
        return None
    value = parent.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _tool_event_identity(
    event: CliSideChannelEvent,
) -> tuple[tuple[object, ...], object | None]:
    payload = event.payload
    calls = _tool_event_calls(payload)
    result = value_from(payload, "result")
    diagnostics = _tool_event_diagnostics(payload)
    if not calls and result is not None:
        result_call = value_from(result, "call")
        if result_call is not None:
            calls = (result_call,)

    tool_call_ids = tuple(
        tool_call_id
        for candidate in (
            *(value_from(diagnostic, "call_id") for diagnostic in diagnostics),
            *(value_from(call, "id") for call in calls),
        )
        if (tool_call_id := candidate) is not None
    )
    name = (
        _first_tool_event_value(calls, "name")
        or value_from(result, "name")
        or _first_tool_event_value(diagnostics, "canonical_name")
        or _first_tool_event_value(diagnostics, "requested_name")
    )
    return tool_call_ids, name


def _tool_event_calls(payload: object) -> tuple[object, ...]:
    call = value_from(payload, "call")
    if call is not None:
        return (call,)
    if isinstance(payload, list | tuple):
        return tuple(payload)
    return ()


def _tool_event_diagnostics(payload: object) -> tuple[object, ...]:
    diagnostics = value_from(payload, "diagnostics")
    if isinstance(diagnostics, list | tuple):
        return tuple(diagnostics)
    result: list[object] = []
    for key in ("diagnostic", "result"):
        diagnostic = value_from(payload, key)
        if (
            diagnostic is not None
            and value_from(diagnostic, "code") is not None
        ):
            result.append(diagnostic)
    return tuple(result)


def _first_tool_event_value(
    items: tuple[object, ...], key: str
) -> object | None:
    for item in items:
        value = value_from(item, key)
        if value is not None:
            return value
    return None


def _normalized_tool_call_id(value: object) -> str:
    return safe_text(value)
