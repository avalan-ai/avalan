from ..flow.node import Node

from collections.abc import Awaitable, Callable
from inspect import isawaitable
from typing import Any


class Connection:
    def __init__(
        self,
        src: Node,
        dest: Node,
        label: str | None = None,
        conditions: (
            list[Callable[[Any], bool | Awaitable[bool]]] | None
        ) = None,
        filters: list[Callable[[Any], Any | Awaitable[Any]]] | None = None,
    ) -> None:
        self.src: Node = src
        self.dest: Node = dest
        self.label: str | None = label
        self.conditions: list[Callable[[Any], bool | Awaitable[bool]]] = (
            conditions or []
        )
        self.filters: list[Callable[[Any], Any | Awaitable[Any]]] = (
            filters or []
        )

    def check_conditions(self, data: Any) -> bool:
        for condition in self.conditions:
            result = condition(data)
            if isawaitable(result):
                _close_awaitable(result)
                raise TypeError(
                    "Connection condition produced awaitable output; "
                    "use check_conditions_async"
                )
            if not result:
                return False
        return True

    async def check_conditions_async(self, data: Any) -> bool:
        for condition in self.conditions:
            result = condition(data)
            if isawaitable(result):
                result = await result
            if not result:
                return False
        return True

    def apply_filters(self, data: Any) -> Any:
        for filter_function in self.filters:
            data = filter_function(data)
            if isawaitable(data):
                _close_awaitable(data)
                raise TypeError(
                    "Connection filter produced awaitable output; "
                    "use apply_filters_async"
                )
        return data

    async def apply_filters_async(self, data: Any) -> Any:
        for filter_function in self.filters:
            data = filter_function(data)
            if isawaitable(data):
                data = await data
        return data

    def __repr__(self) -> str:
        return f"<Conn {self.src.name}->{self.dest.name}>"


def _close_awaitable(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()
