"""Keep timed-out asynchronous work owned until it actually terminates."""

from .admission import PreparedTriggerAdmission
from .plan import TriggerAdmissionPlan
from .records import TriggerSnapshot

from asyncio import CancelledError, Task, create_task, wait
from collections.abc import Awaitable
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from typing import TypeVar

_Result = TypeVar("_Result")


class SchedulerOperationKind(StrEnum):
    DISCOVERY = "discovery"
    PREPARATION = "preparation"
    ADMISSION = "admission"
    RECOVERY = "recovery"
    FAILURE = "failure"
    CLEANUP = "cleanup"
    CLOSE = "close"


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerOperation:
    """Retain the original operation and recovery context across timeouts."""

    identity: int
    kind: SchedulerOperationKind
    task: Task[object]
    snapshot: TriggerSnapshot | None = None
    plan: TriggerAdmissionPlan | None = None
    prepared: PreparedTriggerAdmission | None = None
    resource_index: int | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SchedulerOperationCompletion:
    operation: SchedulerOperation
    value: object = None
    error: BaseException | None = None


class SchedulerOperationTimeout(TimeoutError):
    """Expose the still-owned operation instead of reporting cancellation."""

    def __init__(self, operation: SchedulerOperation) -> None:
        self.operation = operation
        super().__init__(operation.kind.value)


class SchedulerOperations:
    """Bound waits without losing tasks that delay or reject cancellation."""

    def __init__(self) -> None:
        self._next_identity = 0
        self._pending: dict[int, SchedulerOperation] = {}
        self._failures: dict[int, BaseException] = {}

    @property
    def pending(self) -> tuple[SchedulerOperation, ...]:
        return tuple(self._pending.values())

    @property
    def cancelled_error(self) -> CancelledError | None:
        """Preserve typed cancellation after asyncio consumes its exception."""
        return next(
            (
                error
                for error in self._failures.values()
                if isinstance(error, CancelledError)
            ),
            None,
        )

    async def call(
        self,
        work: Awaitable[_Result],
        *,
        kind: SchedulerOperationKind,
        timeout: float,
        snapshot: TriggerSnapshot | None = None,
        plan: TriggerAdmissionPlan | None = None,
        prepared: PreparedTriggerAdmission | None = None,
        resource_index: int | None = None,
    ) -> _Result:
        """Await one operation and retain work after a bounded wait."""
        assert isfinite(timeout) and timeout >= 0

        self._next_identity += 1
        identity = self._next_identity

        async def invoke() -> _Result:
            try:
                return await work
            except BaseException as error:
                self._failures[identity] = error
                raise

        task = create_task(invoke())
        operation = SchedulerOperation(
            identity=identity,
            kind=kind,
            task=task,
            snapshot=snapshot,
            plan=plan,
            prepared=prepared,
            resource_index=resource_index,
        )
        self._pending[operation.identity] = operation
        try:
            done, _ = await wait((task,), timeout=timeout)
        except CancelledError:
            task.cancel()
            raise
        if not done:
            task.cancel()
            raise SchedulerOperationTimeout(operation)
        try:
            result = task.result()
        except CancelledError:
            # A completed cancellation can contain prepared resource handles.
            # Retain its original exception for the reconciliation owner.
            raise
        except BaseException:
            del self._pending[identity]
            self._failures.pop(identity, None)
            raise
        del self._pending[identity]
        return result

    def completed(self) -> tuple[SchedulerOperationCompletion, ...]:
        """Take terminal results once and retain unfinished work."""
        values = []
        for operation in self.pending:
            if not operation.task.done():
                continue
            del self._pending[operation.identity]
            try:
                value = operation.task.result()
            except BaseException as error:
                values.append(
                    SchedulerOperationCompletion(
                        operation=operation,
                        error=self._failures.pop(operation.identity, error),
                    )
                )
            else:
                values.append(
                    SchedulerOperationCompletion(
                        operation=operation,
                        value=value,
                    )
                )
        return tuple(values)

    async def stop(
        self, timeout: float
    ) -> tuple[SchedulerOperationCompletion, ...]:
        """Cancel owned work within the shutdown bound."""
        assert isfinite(timeout) and timeout >= 0
        tasks = tuple(item.task for item in self.pending)
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await wait(tasks, timeout=timeout)
        return self.completed()
