from ..event import (
    Event,
    EventObservabilityPayload,
    EventStats,
    EventStatsSnapshot,
    EventType,
)
from ..model.stream import CanonicalStreamItem, stream_observability_payload
from ..types import (
    assert_optional_non_negative_int,
    assert_optional_non_negative_number,
    assert_positive_int,
)

from asyncio import (
    CancelledError,
    Queue,
    QueueEmpty,
    Task,
    TimeoutError,
    create_task,
    current_task,
    gather,
    wait_for,
)
from asyncio import Event as EventSignal
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import StrEnum
from inspect import isawaitable
from time import monotonic
from typing import Any, AsyncIterator, Awaitable, Callable, Iterable, Literal

Listener = Callable[[Event], Awaitable[None] | None]


class EventDeliveryPolicy(StrEnum):
    BLOCK = "block"
    DROP = "drop"
    COALESCE = "coalesce"
    FAIL_CLOSED = "fail_closed"


class EventSubscriberClass(StrEnum):
    LOSSLESS = "lossless"
    UI = "ui"
    CRITICAL = "critical"
    OBSERVABILITY = "observability"


@dataclass(frozen=True, kw_only=True, slots=True)
class EventDeliveryConfig:
    policy: EventDeliveryPolicy = EventDeliveryPolicy.BLOCK
    queue_limit: int = 1
    timeout: float | None = None
    critical: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.policy, EventDeliveryPolicy)
        assert_positive_int(self.queue_limit, "queue_limit")
        assert_optional_non_negative_number(self.timeout, "timeout")
        assert isinstance(self.critical, bool)
        assert not self.critical or self.policy is EventDeliveryPolicy.BLOCK
        assert not self.critical or self.timeout is not None


@dataclass(frozen=True, kw_only=True, slots=True)
class EventHistoryConfig:
    enabled: bool = True
    max_events: int | None = None
    max_bytes: int | None = None
    ttl_seconds: float | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool)
        assert_optional_non_negative_int(self.max_events, "max_events")
        assert_optional_non_negative_int(self.max_bytes, "max_bytes")
        assert_optional_non_negative_number(self.ttl_seconds, "ttl_seconds")


@dataclass(frozen=True, kw_only=True, slots=True)
class EventListenConfig:
    enabled: bool = True
    queue_limit: int = 512
    policy: EventDeliveryPolicy = EventDeliveryPolicy.DROP

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool)
        assert_positive_int(self.queue_limit, "queue_limit")
        assert isinstance(self.policy, EventDeliveryPolicy)
        assert self.policy in (
            EventDeliveryPolicy.DROP,
            EventDeliveryPolicy.COALESCE,
        )


class EventManagerMode(StrEnum):
    SDK = "sdk"
    SERVER = "server"
    CLI = "cli"
    TEST = "test"


@dataclass(frozen=True, kw_only=True, slots=True)
class EventManagerDefaults:
    history_config: EventHistoryConfig
    delivery_config: EventDeliveryConfig
    listen_config: EventListenConfig
    enrich_token_ids: bool = False

    def __post_init__(self) -> None:
        assert isinstance(self.history_config, EventHistoryConfig)
        assert isinstance(self.delivery_config, EventDeliveryConfig)
        assert isinstance(self.listen_config, EventListenConfig)
        assert isinstance(self.enrich_token_ids, bool)


@dataclass(frozen=True, slots=True)
class _HistoryEntry:
    event: Event
    created_at: float
    size_bytes: int


@dataclass(slots=True)
class _QueuedEvent:
    event: Event
    enqueued_at: float


@dataclass(slots=True)
class _Subscriber:
    listener: Listener
    config: EventDeliveryConfig
    queue: Queue[_QueuedEvent] = field(init=False)
    task: Task[None] | None = None
    closed: bool = False

    def __post_init__(self) -> None:
        self.queue = Queue(maxsize=self.config.queue_limit)


class EventManager:
    _subscribers: dict[EventType, list[_Subscriber]]
    _subscriber_index: dict[Listener, _Subscriber]
    _delivery_queue: Queue[Event]
    _history: deque[_HistoryEntry]
    _history_bytes: int
    _history_config: EventHistoryConfig
    _listen_config: EventListenConfig
    _enrich_token_ids: bool
    _stats: EventStats
    _closed: bool

    def __init__(
        self,
        history_length: int | None = None,
        *,
        default_delivery_config: EventDeliveryConfig | None = None,
        enrich_token_ids: bool = False,
        history_config: EventHistoryConfig | None = None,
        listen_config: EventListenConfig | None = None,
        mode: EventManagerMode = EventManagerMode.SDK,
    ) -> None:
        assert isinstance(mode, EventManagerMode)
        assert isinstance(enrich_token_ids, bool)
        mode_defaults = self.defaults_for_mode(mode)
        if history_config is None:
            history_config = (
                EventHistoryConfig(max_events=history_length)
                if history_length is not None
                else mode_defaults.history_config
            )
        else:
            assert history_length is None
        if default_delivery_config is None:
            default_delivery_config = mode_defaults.delivery_config
        if listen_config is None:
            listen_config = mode_defaults.listen_config
        if not enrich_token_ids:
            enrich_token_ids = mode_defaults.enrich_token_ids
        assert isinstance(history_config, EventHistoryConfig)
        assert isinstance(default_delivery_config, EventDeliveryConfig)
        assert isinstance(listen_config, EventListenConfig)
        self._subscribers = defaultdict(list)
        self._subscriber_index = {}
        self._delivery_queue = Queue(maxsize=listen_config.queue_limit)
        self._history = deque()
        self._history_bytes = 0
        self._default_delivery_config = default_delivery_config
        self._history_config = history_config
        self._listen_config = listen_config
        self._enrich_token_ids = enrich_token_ids
        self._stats = EventStats()
        self._closed = False

    @staticmethod
    def defaults_for_mode(mode: EventManagerMode) -> EventManagerDefaults:
        assert isinstance(mode, EventManagerMode)
        match mode:
            case EventManagerMode.SERVER:
                return EventManagerDefaults(
                    history_config=EventHistoryConfig(enabled=False),
                    delivery_config=(
                        EventManager.default_delivery_config_for_subscriber_class(
                            EventSubscriberClass.OBSERVABILITY
                        )
                    ),
                    listen_config=EventListenConfig(enabled=False),
                )
            case EventManagerMode.CLI:
                return EventManagerDefaults(
                    history_config=EventHistoryConfig(max_events=256),
                    delivery_config=(
                        EventManager.default_delivery_config_for_subscriber_class(
                            EventSubscriberClass.UI
                        )
                    ),
                    listen_config=EventListenConfig(
                        queue_limit=256,
                        policy=EventDeliveryPolicy.COALESCE,
                    ),
                )
            case EventManagerMode.TEST:
                return EventManagerDefaults(
                    history_config=EventHistoryConfig(max_events=1024),
                    delivery_config=(
                        EventManager.default_delivery_config_for_subscriber_class(
                            EventSubscriberClass.LOSSLESS
                        )
                    ),
                    listen_config=EventListenConfig(queue_limit=1024),
                )
            case EventManagerMode.SDK:
                return EventManagerDefaults(
                    history_config=EventHistoryConfig(max_events=512),
                    delivery_config=(
                        EventManager.default_delivery_config_for_subscriber_class(
                            EventSubscriberClass.LOSSLESS
                        )
                    ),
                    listen_config=EventListenConfig(queue_limit=512),
                )

    @staticmethod
    def default_delivery_config_for_subscriber_class(
        subscriber_class: EventSubscriberClass,
    ) -> EventDeliveryConfig:
        assert isinstance(subscriber_class, EventSubscriberClass)
        match subscriber_class:
            case EventSubscriberClass.LOSSLESS:
                return EventDeliveryConfig()
            case EventSubscriberClass.UI:
                return EventDeliveryConfig(
                    policy=EventDeliveryPolicy.COALESCE,
                    queue_limit=64,
                )
            case EventSubscriberClass.CRITICAL:
                return EventDeliveryConfig(
                    policy=EventDeliveryPolicy.BLOCK,
                    queue_limit=1,
                    timeout=1.0,
                    critical=True,
                )
            case EventSubscriberClass.OBSERVABILITY:
                return EventDeliveryConfig(
                    policy=EventDeliveryPolicy.DROP,
                    queue_limit=32,
                )

    @property
    def history(self) -> list[Event]:
        self._evict_history()
        return [entry.event for entry in self._history]

    @property
    def default_delivery_config(self) -> EventDeliveryConfig:
        return self._default_delivery_config

    @property
    def history_config(self) -> EventHistoryConfig:
        return self._history_config

    @property
    def listen_config(self) -> EventListenConfig:
        return self._listen_config

    @property
    def enrich_token_ids(self) -> bool:
        return self._enrich_token_ids

    @property
    def stats(self) -> EventStatsSnapshot:
        return self._stats.snapshot()

    @property
    def closed(self) -> bool:
        return self._closed

    def add_listener(
        self,
        listener: Listener,
        event_types: Iterable[EventType] | None = None,
        *,
        delivery_config: EventDeliveryConfig | None = None,
        include_token_events: bool = False,
        subscriber_class: EventSubscriberClass | None = None,
    ) -> None:
        assert not self._closed
        assert callable(listener)
        assert isinstance(include_token_events, bool)
        assert delivery_config is None or isinstance(
            delivery_config, EventDeliveryConfig
        )
        assert subscriber_class is None or isinstance(
            subscriber_class, EventSubscriberClass
        )
        assert delivery_config is None or subscriber_class is None
        types = self._listener_event_types(
            event_types, include_token_events=include_token_events
        )
        if not types:
            return
        resolved_delivery_config = delivery_config or (
            self.default_delivery_config_for_subscriber_class(subscriber_class)
            if subscriber_class
            else self._default_delivery_config
        )
        subscriber = self._subscriber_index.get(listener)
        if subscriber is None:
            subscriber = _Subscriber(
                listener=listener,
                config=resolved_delivery_config,
            )
            self._subscriber_index[listener] = subscriber
        else:
            assert subscriber.config == resolved_delivery_config
        for event_type in types:
            subscribers = self._subscribers[event_type]
            if subscriber not in subscribers:
                subscribers.append(subscriber)

    def add_ui_listener(
        self,
        listener: Listener,
        event_types: Iterable[EventType] | None = None,
        *,
        include_token_events: bool = False,
    ) -> None:
        assert isinstance(include_token_events, bool)
        self.add_listener(
            listener,
            event_types,
            include_token_events=include_token_events,
            subscriber_class=EventSubscriberClass.UI,
        )

    def add_observability_listener(
        self,
        listener: Listener,
        event_types: Iterable[EventType] | None = None,
        *,
        include_token_events: bool = False,
    ) -> None:
        assert isinstance(include_token_events, bool)
        self.add_listener(
            listener,
            event_types,
            include_token_events=include_token_events,
            subscriber_class=EventSubscriberClass.OBSERVABILITY,
        )

    def remove_listener(
        self,
        listener: Listener,
        event_types: Iterable[EventType] | None = None,
    ) -> None:
        types = list(EventType) if event_types is None else list(event_types)
        for event_type in types:
            assert isinstance(event_type, EventType)
        if not types:
            return
        subscriber = self._subscriber_index.get(listener)
        for event_type in types:
            subscribers = self._subscribers.get(event_type)
            if subscribers and subscriber in subscribers:
                subscribers.remove(subscriber)
                if not subscribers:
                    self._subscribers.pop(event_type)
        if not any(
            subscriber in subscribers
            for subscribers in self._subscribers.values()
        ):
            self._close_subscriber(listener, cancel_task=True)

    def _listener_event_types(
        self,
        event_types: Iterable[EventType] | None,
        *,
        include_token_events: bool,
    ) -> list[EventType]:
        types = (
            [
                event_type
                for event_type in EventType
                if include_token_events
                or event_type is not EventType.TOKEN_GENERATED
            ]
            if event_types is None
            else list(event_types)
        )
        for event_type in types:
            assert isinstance(event_type, EventType)
        return types

    def should_emit(self, event_type: EventType) -> bool:
        assert isinstance(event_type, EventType)
        if self._closed:
            return False
        if event_type is EventType.TOKEN_GENERATED:
            return bool(self._subscribers.get(event_type))
        return True

    async def trigger_stream_item(
        self,
        item: CanonicalStreamItem,
        *,
        event_type: EventType = EventType.TOKEN_GENERATED,
    ) -> None:
        assert isinstance(item, CanonicalStreamItem)
        assert isinstance(event_type, EventType)
        if not self.should_emit(event_type):
            return
        payload = EventObservabilityPayload.canonical_stream(
            stream_observability_payload(item)
        )
        await self.trigger(
            Event.from_observability_payload(
                type=event_type,
                observability_payload=payload,
            )
        )

    async def trigger(self, event: Event) -> None:
        if not self.should_emit(event.type):
            return

        self._store_history(event)
        self._enqueue_for_listen(event)
        self._stats.record_published(
            event.type, queue_depth=self._queue_depth()
        )

        delivery_errors: list[BaseException] = []
        for subscriber in list(self._subscribers.get(event.type, [])):
            try:
                await self._deliver(subscriber, event)
            except CancelledError as error:
                if self._current_task_is_cancelling():
                    raise
                delivery_errors.append(error)
            except Exception as error:
                delivery_errors.append(error)

        if delivery_errors:
            raise delivery_errors[0]

    async def listen(
        self, stop_signal: EventSignal | None = None, timeout: float = 0.2
    ) -> AsyncIterator[Event]:
        while True:
            if self._closed:
                break
            try:
                event = await wait_for(
                    self._delivery_queue.get(), timeout=timeout
                )
                self._stats.record_delivered(queue_depth=self._queue_depth())
                yield event
            except TimeoutError:
                if (
                    self._closed
                    or self._delivery_queue.empty()
                    and (stop_signal is None or stop_signal.is_set())
                ):
                    break

    async def aclose(self) -> None:
        self._closed = True
        current = current_task()
        subscribers = list(self._subscriber_index.values())
        tasks = [
            subscriber.task
            for subscriber in subscribers
            if subscriber.task is not None
            and not subscriber.task.done()
            and subscriber.task is not current
        ]
        dropped = 0
        for subscriber in subscribers:
            subscriber.closed = True
            dropped += self._drain_queue(subscriber.queue)
            if (
                subscriber.task is not None
                and not subscriber.task.done()
                and subscriber.task is not current
            ):
                subscriber.task.cancel()
        self._subscriber_index.clear()
        self._subscribers.clear()
        self._history.clear()
        self._history_bytes = 0
        dropped += self._drain_queue(self._delivery_queue)
        if dropped:
            self._stats.record_dropped(dropped, queue_depth=0)
        else:
            self._stats.record_queue_depth(0)
        if tasks:
            await gather(*tasks, return_exceptions=True)

    def _store_history(self, event: Event) -> None:
        config = self._history_config
        if not config.enabled:
            return

        history_event = event.for_history()
        entry = _HistoryEntry(
            event=history_event,
            created_at=monotonic(),
            size_bytes=self._event_size(history_event),
        )
        self._history.append(entry)
        self._history_bytes += entry.size_bytes
        self._evict_history()

    def _evict_history(self) -> None:
        config = self._history_config
        if not config.enabled:
            self._history.clear()
            self._history_bytes = 0
            return

        if config.ttl_seconds is not None:
            cutoff = monotonic() - config.ttl_seconds
            while self._history and self._history[0].created_at <= cutoff:
                self._remove_oldest_history()

        if config.max_events is not None:
            while len(self._history) > config.max_events:
                self._remove_oldest_history()

        if config.max_bytes is not None:
            while self._history and self._history_bytes > config.max_bytes:
                self._remove_oldest_history()

    def _remove_oldest_history(self) -> None:
        entry = self._history.popleft()
        self._history_bytes -= entry.size_bytes

    def _event_size(self, event: Event) -> int:
        return len(repr(event.observability.to_dict()).encode("utf-8"))

    def _enqueue_for_listen(self, event: Event) -> None:
        if not self._listen_config.enabled:
            return
        if self._delivery_queue.full():
            if self._listen_config.policy is EventDeliveryPolicy.DROP:
                self._stats.record_dropped(queue_depth=self._queue_depth())
                return
            _ = self._delivery_queue.get_nowait()
            self._stats.record_coalesced()
        self._delivery_queue.put_nowait(event)

    def _queue_depth(self) -> int:
        queue_depths = [self._listen_queue_depth()]
        queue_depths.extend(
            subscriber.queue.qsize()
            for subscriber in self._subscriber_index.values()
        )
        return max(queue_depths)

    def _listen_queue_depth(self) -> int:
        if not self._listen_config.enabled:
            return 0
        return self._delivery_queue.qsize()

    def _drain_queue(self, queue: Queue[Any]) -> int:
        dropped = 0
        while True:
            try:
                queue.get_nowait()
                dropped += 1
            except QueueEmpty:
                return dropped

    async def _deliver(self, subscriber: _Subscriber, event: Event) -> None:
        if subscriber.closed:
            return

        match subscriber.config.policy:
            case EventDeliveryPolicy.BLOCK:
                try:
                    await self._call_listener(subscriber, event)
                except CancelledError:
                    if (
                        self._current_task_is_cancelling()
                        or subscriber.config.critical
                    ):
                        raise
                except Exception:
                    if subscriber.config.critical:
                        raise
            case EventDeliveryPolicy.DROP:
                self._enqueue_or_drop(subscriber, event)
            case EventDeliveryPolicy.COALESCE:
                self._enqueue_or_coalesce(subscriber, event)
            case EventDeliveryPolicy.FAIL_CLOSED:
                self._enqueue_or_fail_closed(subscriber, event)

    async def _call_listener(
        self, subscriber: _Subscriber, event: Event
    ) -> Literal[True]:
        started = monotonic()
        try:
            result = subscriber.listener(event)
            if isawaitable(result):
                if subscriber.config.timeout is None:
                    await result
                else:
                    wait_started = monotonic()
                    try:
                        await wait_for(
                            result, timeout=subscriber.config.timeout
                        )
                    finally:
                        if subscriber.config.critical:
                            self._stats.record_critical_wait_time(
                                monotonic() - wait_started
                            )
        except CancelledError:
            if self._current_task_is_cancelling():
                raise
            self._stats.record_failed()
            raise
        except Exception:
            self._stats.record_failed()
            raise
        self._stats.record_delivered(queue_depth=self._queue_depth())
        self._stats.record_listener_lag(monotonic() - started)
        return True

    def _enqueue_or_drop(self, subscriber: _Subscriber, event: Event) -> None:
        if subscriber.queue.full():
            self._stats.record_dropped(queue_depth=self._queue_depth())
            return
        self._enqueue(subscriber, event)

    def _enqueue_or_coalesce(
        self, subscriber: _Subscriber, event: Event
    ) -> None:
        if subscriber.queue.full():
            _ = subscriber.queue.get_nowait()
            self._stats.record_coalesced()
        self._enqueue(subscriber, event)

    def _enqueue_or_fail_closed(
        self, subscriber: _Subscriber, event: Event
    ) -> None:
        if subscriber.queue.full():
            self._stats.record_failed()
            self._close_subscriber(subscriber.listener, cancel_task=True)
            return
        self._enqueue(subscriber, event)

    def _enqueue(self, subscriber: _Subscriber, event: Event) -> None:
        subscriber.queue.put_nowait(
            _QueuedEvent(event=event, enqueued_at=monotonic())
        )
        self._stats.record_queue_depth(self._queue_depth())
        self._ensure_worker(subscriber)

    def _ensure_worker(self, subscriber: _Subscriber) -> None:
        if subscriber.task is None or subscriber.task.done():
            subscriber.task = create_task(self._drain_subscriber(subscriber))

    async def _drain_subscriber(self, subscriber: _Subscriber) -> None:
        while not subscriber.closed:
            if subscriber.queue.empty():
                break
            queued = await subscriber.queue.get()
            self._stats.record_listener_lag(monotonic() - queued.enqueued_at)
            try:
                await self._call_listener(subscriber, queued.event)
            except CancelledError:
                if subscriber.closed or self._current_task_is_cancelling():
                    raise
                if subscriber.config.policy is EventDeliveryPolicy.FAIL_CLOSED:
                    self._close_subscriber(
                        subscriber.listener, cancel_task=False
                    )
                    return
                self._stats.record_queue_depth(self._queue_depth())
            except Exception:
                if subscriber.config.policy is EventDeliveryPolicy.FAIL_CLOSED:
                    self._close_subscriber(
                        subscriber.listener, cancel_task=False
                    )
                    return
                self._stats.record_queue_depth(self._queue_depth())

    def _close_subscriber(
        self, listener: Listener, *, cancel_task: bool
    ) -> None:
        subscriber = self._subscriber_index.pop(listener, None)
        if subscriber is None:
            return
        subscriber.closed = True
        dropped = self._drain_queue(subscriber.queue)
        if cancel_task and subscriber.task is not None:
            subscriber.task.cancel()
        for event_type, subscribers in list(self._subscribers.items()):
            if subscriber in subscribers:
                subscribers.remove(subscriber)
                if not subscribers:
                    self._subscribers.pop(event_type)
        if dropped:
            self._stats.record_dropped(
                dropped,
                queue_depth=self._queue_depth(),
            )
        else:
            self._stats.record_queue_depth(self._queue_depth())

    @staticmethod
    def _current_task_is_cancelling() -> bool:
        task = current_task()
        return bool(task and task.cancelling())
