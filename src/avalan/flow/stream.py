from ..event import EventType
from ..model.stream import (
    CanonicalStreamItem,
    StreamChannel,
    StreamItemCorrelation,
    StreamItemKind,
    StreamRetentionPolicy,
)
from ..types import LooseJsonValue, assert_non_empty_string
from .node import CancellationChecker

from asyncio import CancelledError
from asyncio import Event as AsyncEvent
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from inspect import isawaitable
from math import isfinite
from typing import cast

FlowStreamSink = Callable[[CanonicalStreamItem], Awaitable[None] | None]
_FLOW_TEXT_METADATA_FIELDS: tuple[str, ...] = (
    "state",
    "status",
    "route_kind",
    "edge_kind",
    "source",
    "target",
    "output_name",
)
_FLOW_INT_METADATA_FIELDS: tuple[str, ...] = (
    "attempt",
    "attempts",
    "edge_index",
    "node_count",
    "edge_count",
)
_FLOW_FLOAT_METADATA_FIELDS: tuple[str, ...] = (
    "duration_ms",
    "elapsed_ms",
    "progress",
    "progress_percent",
)
_FLOW_BOOL_METADATA_FIELDS: tuple[str, ...] = (
    "matched",
    "eligible",
    "ready",
)


@dataclass(slots=True)
class FlowStreamRecorder:
    downstream: FlowStreamSink
    history_item_limit: int = StreamRetentionPolicy().flow_history_item_limit
    _items: list[CanonicalStreamItem] = field(default_factory=list)
    _ui_items: dict[
        tuple[str | None, str | None, str], CanonicalStreamItem
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert callable(self.downstream)
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

    def __call__(self, item: CanonicalStreamItem) -> Awaitable[None] | None:
        _assert_flow_stream_item(item)
        result = self.downstream(item)
        if isawaitable(result):
            return self._record_after_await(result, item)
        assert result is None
        self._record(item)
        return None

    async def _record_after_await(
        self,
        result: Awaitable[None],
        item: CanonicalStreamItem,
    ) -> None:
        await result
        self._record(item)

    def _record(self, item: CanonicalStreamItem) -> None:
        if self.history_item_limit == 0:
            return
        assert all(
            existing.sequence != item.sequence for existing in self._items
        )
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
    _sequence: int = field(default=0, init=False, repr=False)

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

    def next_sequence(self) -> int:
        sequence = self._sequence
        self._sequence += 1
        return sequence

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


def flow_stream_recorder(
    downstream: FlowStreamSink,
    *,
    history_item_limit: int = StreamRetentionPolicy().flow_history_item_limit,
) -> FlowStreamRecorder:
    assert callable(downstream)
    return FlowStreamRecorder(
        downstream=downstream,
        history_item_limit=history_item_limit,
    )


def canonical_flow_item(
    *,
    stream_session: FlowStreamSession,
    event_type: EventType | str,
    payload: Mapping[str, object],
    sequence: int | None = None,
    started: float | None = None,
    finished: float | None = None,
    parent_sequence: int | None = None,
) -> CanonicalStreamItem:
    assert isinstance(stream_session, FlowStreamSession)
    event_type_value = _flow_event_type_value(event_type)
    assert event_type_value.startswith("flow_")
    assert isinstance(payload, Mapping)
    if sequence is None:
        sequence = stream_session.next_sequence()
    assert isinstance(sequence, int)
    assert not isinstance(sequence, bool)
    assert sequence >= 0
    if started is not None:
        assert isinstance(started, int | float)
        assert not isinstance(started, bool)
    if finished is not None:
        assert isinstance(finished, int | float)
        assert not isinstance(finished, bool)
    if parent_sequence is not None:
        assert isinstance(parent_sequence, int)
        assert not isinstance(parent_sequence, bool)
        assert parent_sequence >= 0
    flow_run_id = _optional_string(payload.get("flow_id"))
    node_id = _optional_string(payload.get("node"))
    if node_id is None:
        node_id = _optional_string(payload.get("flow_node"))
    return CanonicalStreamItem(
        stream_session_id=stream_session.stream_session_id,
        run_id=stream_session.run_id,
        turn_id=stream_session.turn_id,
        sequence=sequence,
        kind=StreamItemKind.FLOW_EVENT,
        channel=StreamChannel.FLOW,
        correlation=StreamItemCorrelation(
            flow_run_id=flow_run_id or stream_session.run_id,
            node_id=node_id,
            parent_sequence=parent_sequence,
        ),
        data=_loose_mapping(payload),
        metadata=_flow_item_metadata(
            event_type_value,
            payload=payload,
            started=started,
            finished=finished,
        ),
    )


def _assert_flow_stream_item(item: CanonicalStreamItem) -> None:
    assert isinstance(item, CanonicalStreamItem)
    assert item.kind is StreamItemKind.FLOW_EVENT
    assert item.channel is StreamChannel.FLOW
    assert isinstance(item.data, Mapping)
    event_type = item.metadata.get("event_type")
    assert_non_empty_string(event_type, "event_type")
    event_type_value = cast(str, event_type)
    assert event_type_value.startswith("flow_")


def _flow_event_type_value(event_type: EventType | str) -> str:
    if isinstance(event_type, EventType):
        return event_type.value
    assert_non_empty_string(event_type, "event_type")
    return event_type


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


def _flow_item_metadata(
    event_type: str,
    *,
    payload: Mapping[str, object],
    started: float | None,
    finished: float | None,
) -> dict[str, LooseJsonValue]:
    metadata: dict[str, LooseJsonValue] = {"event_type": event_type}
    if started is not None:
        metadata["started"] = _loose_float(float(started))
    if finished is not None:
        metadata["finished"] = _loose_float(float(finished))
    if started is not None and finished is not None:
        metadata["elapsed"] = _loose_float(float(finished - started))
    metadata.update(_flow_payload_metadata(payload))
    return metadata


def _flow_payload_metadata(
    payload: Mapping[str, object],
) -> dict[str, LooseJsonValue]:
    metadata: dict[str, LooseJsonValue] = {}
    for key in _FLOW_TEXT_METADATA_FIELDS:
        text_value = _metadata_string(payload.get(key))
        if text_value is not None:
            metadata[key] = text_value
    for key in _FLOW_INT_METADATA_FIELDS:
        int_value = _metadata_non_negative_int(payload.get(key))
        if int_value is not None:
            metadata[key] = int_value
    for key in _FLOW_FLOAT_METADATA_FIELDS:
        number_value = _metadata_non_negative_number(payload.get(key))
        if number_value is not None:
            metadata[key] = number_value
    for key in _FLOW_BOOL_METADATA_FIELDS:
        bool_value = _metadata_bool(payload.get(key))
        if bool_value is not None:
            metadata[key] = bool_value
    if "state" not in metadata and "status" in metadata:
        metadata["state"] = metadata["status"]
    return metadata


def _metadata_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _metadata_non_negative_int(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _metadata_non_negative_number(value: object) -> int | float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    number = float(value)
    if not isfinite(number) or number < 0:
        return None
    if isinstance(value, int):
        return value
    return number


def _metadata_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


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
