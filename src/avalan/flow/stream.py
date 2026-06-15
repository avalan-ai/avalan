from ..event import Event, EventObservabilityPayload, EventType
from ..model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
    stream_observability_payload,
)
from ..types import LooseJsonValue, assert_non_empty_string
from .node import CancellationChecker

from asyncio import CancelledError
from asyncio import Event as AsyncEvent
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from math import isfinite

FlowEventSink = Callable[[Event], Awaitable[None] | None]


@dataclass(slots=True)
class FlowCanonicalEventListener:
    downstream: FlowEventSink
    stream_session_id: str
    run_id: str
    turn_id: str
    history_item_limit: int = StreamRetentionPolicy().flow_history_item_limit
    _sequence: int = 0
    _items: list[CanonicalStreamItem] = field(default_factory=list)
    _ui_items: dict[
        tuple[str | None, str | None, str], CanonicalStreamItem
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert callable(self.downstream)
        assert_non_empty_string(self.stream_session_id, "stream_session_id")
        assert_non_empty_string(self.run_id, "run_id")
        assert_non_empty_string(self.turn_id, "turn_id")
        assert isinstance(self.history_item_limit, int)
        assert not isinstance(self.history_item_limit, bool)
        assert self.history_item_limit >= 0

    @property
    def items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(self._items)

    @property
    def ui_items(self) -> tuple[CanonicalStreamItem, ...]:
        return tuple(
            sorted(self._ui_items.values(), key=lambda item: item.sequence)
        )

    def __call__(self, event: Event) -> Awaitable[None] | None:
        assert isinstance(event, Event)
        projected, item = self._project_event(event)
        result = self.downstream(projected)
        if isawaitable(result):
            return self._record_after_await(result, item)
        assert result is None
        if item is not None:
            self._record(item)
        return None

    async def _record_after_await(
        self,
        result: Awaitable[None],
        item: CanonicalStreamItem | None,
    ) -> None:
        await result
        if item is not None:
            self._record(item)

    def _project_event(
        self, event: Event
    ) -> tuple[Event, CanonicalStreamItem | None]:
        if not flow_event_is_projectable(event):
            return event, None
        sequence = self._sequence
        self._sequence += 1
        item = canonical_flow_item_from_event(
            event,
            stream_session_id=self.stream_session_id,
            run_id=self.run_id,
            turn_id=self.turn_id,
            sequence=sequence,
        )
        return (
            Event(
                type=event.type,
                payload=event.payload,
                observability_payload=(
                    EventObservabilityPayload.canonical_stream(
                        stream_observability_payload(item)
                    )
                ),
                started=event.started,
                finished=event.finished,
                elapsed=event.elapsed,
            ),
            item,
        )

    def _record(self, item: CanonicalStreamItem) -> None:
        assert item.sequence < self._sequence
        assert all(
            existing.sequence != item.sequence for existing in self._items
        )
        if self.history_item_limit == 0:
            return
        self._items.append(item)
        self._items.sort(key=lambda current: current.sequence)
        if len(self._items) > self.history_item_limit:
            del self._items[0]
        self._ui_items[_flow_ui_coalescing_key(item)] = item
        while len(self._ui_items) > self.history_item_limit:
            oldest_key = min(
                self._ui_items,
                key=lambda key: self._ui_items[key].sequence,
            )
            del self._ui_items[oldest_key]


@dataclass(slots=True)
class FlowStreamSession:
    stream_session_id: str
    run_id: str
    turn_id: str
    cancellation_checker: CancellationChecker | None = None
    _cancel_event: AsyncEvent = field(
        default_factory=AsyncEvent,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        assert_non_empty_string(self.stream_session_id, "stream_session_id")
        assert_non_empty_string(self.run_id, "run_id")
        assert_non_empty_string(self.turn_id, "turn_id")
        if self.cancellation_checker is not None:
            assert callable(self.cancellation_checker)

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def cancel(self) -> None:
        self._cancel_event.set()

    async def check_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError()
        if self.cancellation_checker is None:
            return
        try:
            await self.cancellation_checker()
        except CancelledError:
            self.cancel()
            raise
        if self.cancelled:
            raise CancelledError()


def flow_stream_session(
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    cancellation_checker: CancellationChecker | None = None,
) -> FlowStreamSession:
    return FlowStreamSession(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        cancellation_checker=cancellation_checker,
    )


def canonical_flow_event_listener(
    downstream: FlowEventSink,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    history_item_limit: int = StreamRetentionPolicy().flow_history_item_limit,
) -> FlowCanonicalEventListener:
    assert callable(downstream)
    return FlowCanonicalEventListener(
        downstream=downstream,
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        history_item_limit=history_item_limit,
    )


def canonical_flow_item_from_event(
    event: Event,
    *,
    stream_session_id: str,
    run_id: str,
    turn_id: str,
    sequence: int,
) -> CanonicalStreamItem:
    assert isinstance(event, Event)
    assert flow_event_is_projectable(event)
    assert_non_empty_string(stream_session_id, "stream_session_id")
    assert_non_empty_string(run_id, "run_id")
    assert_non_empty_string(turn_id, "turn_id")
    assert isinstance(sequence, int)
    assert not isinstance(sequence, bool)
    assert sequence >= 0
    payload = _payload_mapping(event)
    flow_run_id = _optional_string(payload.get("flow_id"))
    node_id = _optional_string(payload.get("node"))
    if node_id is None:
        node_id = _optional_string(payload.get("flow_node"))
    return CanonicalStreamItem(
        stream_session_id=stream_session_id,
        run_id=run_id,
        turn_id=turn_id,
        sequence=sequence,
        kind=StreamItemKind.FLOW_EVENT,
        channel=StreamChannel.FLOW,
        correlation=StreamItemCorrelation(
            flow_run_id=flow_run_id,
            node_id=node_id,
        ),
        data=_loose_mapping(payload),
        metadata=_flow_event_metadata(event),
    )


def flow_event_is_projectable(event: Event) -> bool:
    assert isinstance(event, Event)
    event_type = event.type
    if isinstance(event_type, EventType):
        return event_type.value.startswith("flow_")
    return event_type.startswith("flow_")


def _payload_mapping(event: Event) -> Mapping[str, object]:
    if event.payload is None:
        return {}
    assert isinstance(event.payload, Mapping)
    return event.payload


def _loose_mapping(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: _loose_value(value) for key, value in payload.items()}


def _loose_value(value: object) -> LooseJsonValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return _loose_float(value)
    if isinstance(value, Mapping):
        return {
            str(key): _loose_value(item)
            for key, item in value.items()
            if isinstance(key, str) and key.strip()
        }
    if isinstance(value, list | tuple):
        return [_loose_value(item) for item in value]
    return {"type": type(value).__name__}


def _loose_float(value: float) -> LooseJsonValue:
    if isfinite(value):
        return value
    return {"type": "float", "value": str(value)}


def _flow_event_metadata(event: Event) -> dict[str, LooseJsonValue]:
    event_type = (
        event.type.value
        if isinstance(event.type, EventType)
        else (str(event.type))
    )
    metadata: dict[str, LooseJsonValue] = {"event_type": event_type}
    if event.started is not None:
        metadata["started"] = _loose_float(float(event.started))
    if event.finished is not None:
        metadata["finished"] = _loose_float(float(event.finished))
    if event.elapsed is not None:
        metadata["elapsed"] = _loose_float(float(event.elapsed))
    return metadata


def _optional_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _flow_ui_coalescing_key(
    item: CanonicalStreamItem,
) -> tuple[str | None, str | None, str]:
    event_type = str(item.metadata.get("event_type", item.kind.value))
    node_group = _node_progress_group(event_type)
    return (
        item.correlation.flow_run_id,
        item.correlation.node_id,
        node_group,
    )


def _node_progress_group(event_type: str) -> str:
    if event_type.startswith("flow_node_"):
        return "flow_node_progress"
    if event_type.startswith("flow_edge_"):
        return "flow_edge_progress"
    if event_type in {
        EventType.FLOW_CONDITION_EVALUATED.value,
        EventType.FLOW_JOIN_READY.value,
    }:
        return "flow_route_progress"
    return event_type
