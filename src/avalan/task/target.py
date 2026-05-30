from .context import TaskTargetContext
from .definition import TaskDefinition
from .validation import TaskValidationIssue

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationContext:
    execution_roots: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.execution_roots, tuple)
        for root in self.execution_roots:
            assert isinstance(root, Path)


class TaskTargetRunner(Protocol):
    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]: ...

    async def run(self, context: TaskTargetContext) -> object: ...


class CallableTaskTargetRunner:
    def __init__(
        self,
        target: Callable[[TaskTargetContext], Awaitable[object]],
    ) -> None:
        self._target = target

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        return await self._target(context)
