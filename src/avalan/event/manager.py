from ..event import Event, EventType
from asyncio import Event as EventSignal, Queue, TimeoutError, wait_for
from collections import defaultdict, deque
from inspect import iscoroutine
from typing import Awaitable, Callable, Iterable

Listener = Callable[[Event], Awaitable[None] | None]


class EventManager:
    _listeners: dict[EventType, list[Listener]]
    _queue: Queue[Event]
    _history: deque[Event]

    def __init__(self, history_length: int | None = None) -> None:
        self._listeners = defaultdict(list)
        self._queue = Queue()
        self._history = deque(maxlen=history_length)

    @property
    def history(self) -> list[Event]:
        return list(self._history)

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

    async def trigger(self, event: Event) -> None:
        self._history.append(event)
        self._queue.put_nowait(event)

        for listener in self._listeners.get(event.type, []):
            result = listener(event)
            if iscoroutine(result):
                await result

    async def listen(
        self, stop_signal: EventSignal | None = None, timeout: float = 0.2
    ):
        while True:
            try:
                yield await wait_for(self._queue.get(), timeout=timeout)
            except TimeoutError:
                if self._queue.empty() and (
                    stop_signal is None or stop_signal.is_set()
                ):
                    break
