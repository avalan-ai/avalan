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

    async def check_conditions_async(self, data: Any) -> bool:
        for condition in self.conditions:
            result = condition(data)
            if isawaitable(result):
                result = await result
            if not result:
                return False
        return True

    async def apply_filters_async(self, data: Any) -> Any:
        for filter_function in self.filters:
            data = filter_function(data)
            if isawaitable(data):
                data = await data
        return data

    def __repr__(self) -> str:
        return f"<Conn {self.src.name}->{self.dest.name}>"
