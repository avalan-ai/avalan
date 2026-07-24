from ..interaction import DurableInteractionSuspension, InputRequiredResult
from .artifact import ArtifactStore
from .context import TaskDurableResumeHandle, TaskTargetContext
from .converters import FileConverter
from .definition import TaskDefinition, TaskTargetType
from .store import TaskStore
from .validation import TaskValidationIssue

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import (
    Literal,
    NoReturn,
    Protocol,
    TypeAlias,
    cast,
    final,
    runtime_checkable,
)


class TaskTargetOutcomeKind(StrEnum):
    COMPLETED = "completed"
    SUSPENDED = "suspended"


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskTargetCompleted:
    output: object
    kind: Literal[TaskTargetOutcomeKind.COMPLETED] = field(
        init=False,
        default=TaskTargetOutcomeKind.COMPLETED,
    )

    def __post_init__(self) -> None:
        assert type(self) is TaskTargetCompleted


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskTargetSuspended:
    input_required: InputRequiredResult
    checkpoint_id: str | None = None
    durable: DurableInteractionSuspension | None = None
    kind: Literal[TaskTargetOutcomeKind.SUSPENDED] = field(
        init=False,
        default=TaskTargetOutcomeKind.SUSPENDED,
    )

    def __post_init__(self) -> None:
        assert type(self) is TaskTargetSuspended
        assert type(self.input_required) is InputRequiredResult
        if self.checkpoint_id is not None:
            assert isinstance(self.checkpoint_id, str)
            assert self.checkpoint_id.strip()
        if self.durable is not None:
            assert type(self.durable) is DurableInteractionSuspension
            assert (
                self.checkpoint_id is not None
            ), "durable suspension requires a checkpoint"
            assert (
                self.durable.command.request.request_id
                == self.input_required.request_id
            )
            assert (
                self.durable.command.request.continuation_id
                == self.input_required.continuation_id
            )
            assert self.input_required.detached_resumption_available


TaskTargetOutcome: TypeAlias = TaskTargetCompleted | TaskTargetSuspended


def completed_task_target_outcome(output: object) -> TaskTargetCompleted:
    return TaskTargetCompleted(output=output)


def suspended_task_target_outcome(
    input_required: InputRequiredResult,
    *,
    checkpoint_id: str | None = None,
    durable: DurableInteractionSuspension | None = None,
) -> TaskTargetSuspended:
    return TaskTargetSuspended(
        input_required=input_required,
        checkpoint_id=checkpoint_id,
        durable=durable,
    )


def task_target_outcome(value: object) -> TaskTargetOutcome:
    if type(value) is TaskTargetCompleted:
        return value
    if type(value) is TaskTargetSuspended:
        return value
    raise TypeError("task target must return a completed or suspended outcome")


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

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome: ...


@runtime_checkable
class TaskDurableSuspensionTargetRunner(Protocol):
    """Advertise exact detached suspension support for one target type."""

    def supports_durable_suspension(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        """Return whether this runner can stage detached suspension."""
        ...


@runtime_checkable
class TaskDurableResumeTargetRunner(Protocol):
    """Resume a durable target without replaying its original input."""

    def supports_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        """Return whether this runner owns exact durable semantics."""
        ...

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        """Resume one explicitly supported durable target."""
        ...


_PREPARED_DURABLE_RESUME_TARGET_PROOF = object()


@final
class PreparedTaskDurableResumeTarget(tuple[object, ...]):
    """Carry registry-minted durable resume capability evidence."""

    __slots__ = ()

    def __new__(cls) -> "PreparedTaskDurableResumeTarget":
        raise TypeError(
            "prepared durable resume targets come from a target registry"
        )

    @classmethod
    def _mint(
        cls,
        *,
        target_type: TaskTargetType,
        runner: TaskDurableResumeTargetRunner,
        preparer: object,
        proof: object,
    ) -> "PreparedTaskDurableResumeTarget":
        if proof is not _PREPARED_DURABLE_RESUME_TARGET_PROOF:
            raise TypeError(
                "durable resume target capability is not registry-minted"
            )
        return tuple.__new__(
            cls,
            (target_type, runner, preparer, proof),
        )

    @property
    def target_type(self) -> TaskTargetType:
        """Return the immutable target type bound at mint time."""
        return cast(TaskTargetType, self[0])

    @property
    def runner(self) -> TaskDurableResumeTargetRunner:
        """Return the immutable runner bound at mint time."""
        return cast(TaskDurableResumeTargetRunner, self[1])

    def is_bound_to(
        self,
        preparer: object,
        target_type: TaskTargetType,
    ) -> bool:
        """Return whether this evidence binds the exact prepared target."""
        if not isinstance(preparer, TaskTargetRunnerRegistry):
            return False
        retained = TaskTargetRunnerRegistry._retained_durable_resume_runner(
            preparer,
            self,
            target_type,
        )
        return (
            retained is not None
            and retained is self.runner
            and self[2] is preparer
            and self[3] is _PREPARED_DURABLE_RESUME_TARGET_PROOF
        )

    def __copy__(self) -> NoReturn:
        """Reject copying registry-owned capability identity."""
        raise TypeError("prepared durable resume targets cannot be copied")

    def __deepcopy__(self, memo: object) -> NoReturn:
        """Reject deep-copying registry-owned capability identity."""
        del memo
        raise TypeError("prepared durable resume targets cannot be copied")

    def __reduce_ex__(self, protocol: int) -> NoReturn:
        """Reject serializing registry-local capability identity."""
        del protocol
        raise TypeError("prepared durable resume targets cannot be serialized")


@runtime_checkable
class TaskDurableResumeTargetPreparer(Protocol):
    """Prepare one exact durable runner for a later resume dispatch."""

    def prepare_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> PreparedTaskDurableResumeTarget | None:
        """Select and validate one durable runner exactly once."""
        ...


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

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        return completed_task_target_outcome(await self._target(context))


class TaskTargetRunnerRegistry:
    def __init__(
        self,
        default: TaskTargetRunner,
        runners: Mapping[TaskTargetType, TaskTargetRunner] | None = None,
    ) -> None:
        self._default = default
        self._runners = dict(runners or {})
        self._prepared_durable_resume_targets: dict[
            TaskTargetType,
            tuple[
                PreparedTaskDurableResumeTarget,
                TaskDurableResumeTargetRunner,
            ],
        ] = {}

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

    async def run(self, context: TaskTargetContext) -> TaskTargetOutcome:
        assert isinstance(context, TaskTargetContext)
        runner = self.runner_for(context.definition.execution.type)
        return task_target_outcome(await runner.run(context))

    async def resume(
        self,
        context: TaskTargetContext,
        durable_resume: TaskDurableResumeHandle,
    ) -> TaskTargetOutcome:
        """Resume only a runner with an explicit durable implementation."""
        assert isinstance(context, TaskTargetContext)
        target_type = context.definition.execution.type
        prepared = self.prepare_durable_resume(target_type)
        if type(
            prepared
        ) is not PreparedTaskDurableResumeTarget or not prepared.is_bound_to(
            self, target_type
        ):
            raise TypeError(
                "task target runner does not support durable resume"
            )
        return task_target_outcome(
            await prepared.runner.resume(context, durable_resume)
        )

    def supports_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        """Return exact support from the selected target runner."""
        prepared = self.prepare_durable_resume(target_type)
        return type(
            prepared
        ) is PreparedTaskDurableResumeTarget and prepared.is_bound_to(
            self, target_type
        )

    def prepare_durable_resume(
        self,
        target_type: TaskTargetType,
    ) -> PreparedTaskDurableResumeTarget | None:
        """Select and validate one durable runner exactly once."""
        runner = self.runner_for(target_type)
        if not isinstance(runner, TaskDurableResumeTargetRunner):
            return None
        if runner.supports_durable_resume(target_type) is not True:
            return None
        prepared = PreparedTaskDurableResumeTarget._mint(
            target_type=target_type,
            runner=runner,
            preparer=self,
            proof=_PREPARED_DURABLE_RESUME_TARGET_PROOF,
        )
        self._prepared_durable_resume_targets[target_type] = (
            prepared,
            runner,
        )
        return prepared

    def _retained_durable_resume_runner(
        self,
        prepared: PreparedTaskDurableResumeTarget,
        target_type: TaskTargetType,
    ) -> TaskDurableResumeTargetRunner | None:
        retained = self._prepared_durable_resume_targets.get(target_type)
        if retained is None or retained[0] is not prepared:
            return None
        return retained[1]

    def supports_durable_suspension(
        self,
        target_type: TaskTargetType,
    ) -> bool:
        """Return exact suspension support from the selected runner."""
        runner = self.runner_for(target_type)
        return isinstance(
            runner, TaskDurableSuspensionTargetRunner
        ) and runner.supports_durable_suspension(target_type)
