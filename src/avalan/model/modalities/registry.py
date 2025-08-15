from collections.abc import Awaitable, Callable
from inspect import isclass
from typing import Any

from ...entities import EngineUri, Modality, Operation
from ...tool.manager import ToolManager

ModalityCallable = Callable[
    [EngineUri, Any, Operation, ToolManager | None],
    Awaitable[Any],
]


class ModalityRegistry:
    _handlers: dict[Modality, ModalityCallable] = {}

    @classmethod
    def register(
        cls, modality: Modality
    ) -> Callable[[ModalityCallable], ModalityCallable]:
        def decorator(handler: ModalityCallable | type) -> ModalityCallable:
            cls._handlers[modality] = (
                handler() if isclass(handler) else handler
            )
            return handler

        return decorator

    @classmethod
    def get(cls, modality: Modality) -> ModalityCallable:
        if modality not in cls._handlers:
            raise NotImplementedError(f"Modality {modality} not registered")
        return cls._handlers[modality]
