from ..flow.node import Node
from typing import Any, Callable, Optional


class Connection:
    def __init__(
        self,
        src: Node,
        dest: Node,
        label: Optional[str] = None,
        conditions: Optional[list[Callable[[Any], bool]]] = None,
        filters: Optional[list[Callable[[Any], Any]]] = None,
    ) -> None:
        self.src: Node = src
        self.dest: Node = dest
        self.label: Optional[str] = label
        self.conditions: list[Callable[[Any], bool]] = conditions or []
        self.filters: list[Callable[[Any], Any]] = filters or []

    def check_conditions(self, data: Any) -> bool:
        return all(cond(data) for cond in self.conditions)

    def apply_filters(self, data: Any) -> Any:
        for f in self.filters:
            data = f(data)
        return data

    def __repr__(self) -> str:
        return f"<Conn {self.src.name}->{self.dest.name}>"
