from .artifact import ArtifactStore
from .context import TaskTargetContext
from .converters import FileConverter
from .definition import TaskDefinition, TaskTargetType
from .store import TaskStore
from .validation import TaskValidationIssue

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskValidationContext:
    execution_roots: tuple[Path, ...] = ()
    artifact_store: ArtifactStore | None = None
    task_store: TaskStore | None = None
    file_converters: Mapping[str, FileConverter] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        assert isinstance(self.execution_roots, tuple)
        for root in self.execution_roots:
            assert isinstance(root, Path)
        assert isinstance(self.file_converters, Mapping)
        for name, converter in self.file_converters.items():
            assert isinstance(name, str) and name.strip()
            assert hasattr(converter, "convert")


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
