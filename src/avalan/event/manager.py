from ..event import Event, EventType
from collections import defaultdict
from inspect import iscoroutine
from typing import Awaitable, Callable, Iterable

Listener = Callable[[Event], Awaitable[None] | None]


class EventManager:
    _listeners: dict[EventType, list[Listener]]

    def __init__(self) -> None:
        self._listeners = defaultdict(list)

    def add_listener(
        self, listener: Listener, event_types: Iterable[EventType] | None = None
    ) -> None:
        types = list(event_types) if event_types else list(EventType)
        for event_type in types:
            self._listeners[event_type].append(listener)

    async def trigger(self, event: Event) -> None:
        for listener in self._listeners.get(event.type, []):
            result = listener(event)
            if iscoroutine(result):
                await result
