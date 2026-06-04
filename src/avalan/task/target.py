from .context import TaskTargetContext
from .definition import TaskDefinition, TaskTargetType
from .validation import TaskValidationIssue

from collections.abc import Awaitable, Callable, Mapping
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


class TaskTargetRunnerRegistry:
    def __init__(
        self,
        default: TaskTargetRunner,
        runners: Mapping[TaskTargetType, TaskTargetRunner] | None = None,
    ) -> None:
        self._default = default
        self._runners = dict(runners or {})

    def runner_for(self, target_type: TaskTargetType) -> TaskTargetRunner:
        assert isinstance(target_type, TaskTargetType)
        return self._runners.get(target_type, self._default)

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(context, TaskValidationContext)
        runner = self.runner_for(definition.execution.type)
        return await runner.validate_definition(definition, context)

    async def run(self, context: TaskTargetContext) -> object:
        assert isinstance(context, TaskTargetContext)
        runner = self.runner_for(context.definition.execution.type)
        return await runner.run(context)
