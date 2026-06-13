from ..event import Event, EventType

from asyncio import Event as EventSignal
from asyncio import Queue, TimeoutError, wait_for
from collections import defaultdict, deque
from inspect import iscoroutine
from typing import AsyncIterator, Awaitable, Callable, Iterable

Listener = Callable[[Event], Awaitable[None] | None]


class EventManager:
    _listeners: dict[EventType, list[Listener]]
    _queue: Queue[Event]
    _history: deque[Event]
    _enrich_token_ids: bool

    def __init__(
        self,
        history_length: int | None = None,
        *,
        enrich_token_ids: bool = False,
    ) -> None:
        assert isinstance(enrich_token_ids, bool)
        self._listeners = defaultdict(list)
        self._queue = Queue()
        self._history = deque(maxlen=history_length)
        self._enrich_token_ids = enrich_token_ids

    @property
    def history(self) -> list[Event]:
        return list(self._history)

    @property
    def enrich_token_ids(self) -> bool:
        return self._enrich_token_ids

    def add_listener(
        self,
        listener: Listener,
        event_types: Iterable[EventType] | None = None,
    ) -> None:
        types = list(event_types) if event_types else list(EventType)
        for event_type in types:
            listeners = self._listeners[event_type]
            if listener not in listeners:
                listeners.append(listener)

    def remove_listener(
        self,
        listener: Listener,
        event_types: Iterable[EventType] | None = None,
    ) -> None:
        types = list(event_types) if event_types else list(EventType)
        for event_type in types:
            listeners = self._listeners.get(event_type)
            if listeners and listener in listeners:
                listeners.remove(listener)
                if not listeners:
                    self._listeners.pop(event_type)

    def should_emit(self, event_type: EventType) -> bool:
        assert isinstance(event_type, EventType)
        if event_type is EventType.TOKEN_GENERATED:
            return bool(self._listeners.get(event_type))
        return True

    async def trigger(self, event: Event) -> None:
        self._history.append(event)
        self._queue.put_nowait(event)

        for listener in self._listeners.get(event.type, []):
            result = listener(event)
            if iscoroutine(result):
                await result

    async def listen(
        self, stop_signal: EventSignal | None = None, timeout: float = 0.2
    ) -> AsyncIterator[Event]:
        while True:
            try:
                yield await wait_for(self._queue.get(), timeout=timeout)
            except TimeoutError:
                if self._queue.empty() and (
                    stop_signal is None or stop_signal.is_set()
                ):
                    break
