from .artifact import ArtifactStore
from .context import TaskInputFile
from .definition import RunMode, TaskDefinition
from .event import SanitizedTaskEvent
from .privacy import EncryptionProvider, HmacProvider
from .runner import DirectTaskRunner, TaskDirectTarget, TaskRunResult
from .state import TaskRunState
from .store import (
    TaskAttempt,
    TaskRun,
    TaskSnapshotValue,
    TaskStore,
)
from .target import (
    CallableTaskTargetRunner,
    TaskTargetRunner,
    TaskValidationContext,
)
from .usage import UsageRecord, UsageTotals
from .validation import (
    TaskValidationError,
    TaskValidationIssue,
    validate_task_definition,
    validate_task_input,
)

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast


class TaskClientUnsupportedOperationError(RuntimeError):
    code: str
    operation: str

    def __init__(self, *, code: str, operation: str, message: str) -> None:
        self.code = code
        self.operation = operation
        super().__init__(message)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientValidationResult:
    issues: tuple[TaskValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    def raise_for_issues(self) -> None:
        if self.issues:
            raise TaskValidationError(self.issues)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientOutput:
    run_id: str
    state: TaskRunState
    output_summary: TaskSnapshotValue = None
    error: TaskSnapshotValue = None

    @property
    def ready(self) -> bool:
        return self.state == TaskRunState.SUCCEEDED


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientInspection:
    run: TaskRun
    attempts: tuple[TaskAttempt, ...]
    output: TaskClientOutput
    events: tuple[SanitizedTaskEvent, ...]
    usage: tuple[UsageRecord, ...]
    usage_totals: UsageTotals
    artifacts: tuple[TaskSnapshotValue, ...]


class TaskClient:
    def __init__(
        self,
        store: TaskStore,
        *,
        target: TaskDirectTarget | TaskTargetRunner,
        hmac_provider: HmacProvider | None = None,
        encryption_provider: EncryptionProvider | None = None,
        raw_storage_allowed: bool = False,
        artifact_store: ArtifactStore | None = None,
        definition_hash: Callable[[TaskDefinition], str] | None = None,
        execution_roots: Iterable[str | Path] = (),
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._target = _target_runner(target)
        self._hmac_provider = hmac_provider
        self._encryption_provider = encryption_provider
        self._raw_storage_allowed = raw_storage_allowed
        self._artifact_store = artifact_store
        self._definition_hash = definition_hash
        self._execution_roots = tuple(execution_roots)
        self._clock = clock
        self._sleep = sleep

    async def validate(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
    ) -> TaskClientValidationResult:
        assert isinstance(definition, TaskDefinition)
        issues: list[TaskValidationIssue] = []
        issues.extend(
            validate_task_definition(
                definition,
                hmac_provider=self._hmac_provider,
                encryption_provider=self._encryption_provider,
                require_configured_keys=True,
                raw_storage_allowed=self._raw_storage_allowed,
                execution_roots=self._execution_roots,
            )
        )
        issues.extend(validate_task_input(definition, input_value))
        issues.extend(
            await self._target.validate_definition(
                definition,
                TaskValidationContext(
                    execution_roots=tuple(
                        Path(root) for root in self._execution_roots
                    ),
                ),
            )
        )
        return TaskClientValidationResult(issues=tuple(issues))

    async def run(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
        expires_at: datetime | None = None,
    ) -> TaskRunResult:
        assert isinstance(definition, TaskDefinition)
        if definition.run.mode != RunMode.DIRECT:
            raise _unsupported_queue_operation("run")
        return await self._runner().run(
            definition,
            input_value=input_value,
            files=files,
            idempotency_key=idempotency_key,
            metadata=metadata,
            expires_at=expires_at,
        )

    async def enqueue(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
        files: tuple[TaskInputFile, ...] = (),
        idempotency_key: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskRun:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(files, tuple)
        _ = input_value, idempotency_key, metadata
        raise _unsupported_queue_operation("enqueue")

    async def output(self, run_id: str) -> TaskClientOutput:
        run = await self._store.get_run(run_id)
        result = run.result
        return TaskClientOutput(
            run_id=run.run_id,
            state=run.state,
            output_summary=result.output_summary if result else None,
            error=result.error if result else None,
        )

    async def events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        return await self._store.list_events(run_id, attempt_id=attempt_id)

    async def usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
    ) -> tuple[UsageRecord, ...]:
        return await self._store.list_usage(run_id, attempt_id=attempt_id)

    async def usage_totals(self, run_id: str) -> UsageTotals:
        return await self._store.usage_totals(run_id)

    async def artifacts(self, run_id: str) -> tuple[TaskSnapshotValue, ...]:
        artifacts = await self._store.list_artifacts(run_id)
        if artifacts:
            return tuple(artifact.summary() for artifact in artifacts)
        run = await self._store.get_run(run_id)
        result = run.result
        if result is None:
            return ()
        references = result.metadata.get("artifacts")
        if references is None:
            return ()
        if isinstance(references, tuple):
            return references
        return (references,)

    async def inspect(self, run_id: str) -> TaskClientInspection:
        return TaskClientInspection(
            run=await self._store.get_run(run_id),
            attempts=await self._store.list_attempts(run_id),
            output=await self.output(run_id),
            events=await self.events(run_id),
            usage=await self.usage(run_id),
            usage_totals=await self.usage_totals(run_id),
            artifacts=await self.artifacts(run_id),
        )

    def _runner(self) -> DirectTaskRunner:
        return DirectTaskRunner(
            self._store,
            target=self._target,
            hmac_provider=self._hmac_provider,
            encryption_provider=self._encryption_provider,
            raw_storage_allowed=self._raw_storage_allowed,
            artifact_store=self._artifact_store,
            definition_hash=self._definition_hash,
            execution_roots=self._execution_roots,
            clock=self._clock,
            sleep=self._sleep,
        )


def _unsupported_queue_operation(
    operation: str,
) -> TaskClientUnsupportedOperationError:
    return TaskClientUnsupportedOperationError(
        code="task.queue_unsupported",
        operation=operation,
        message=(
            "Queued task operations are not available in this SDK client yet."
        ),
    )


def _target_runner(
    target: TaskDirectTarget | TaskTargetRunner,
) -> TaskTargetRunner:
    run = getattr(target, "run", None)
    validate_definition = getattr(target, "validate_definition", None)
    if callable(run) and callable(validate_definition):
        return cast(TaskTargetRunner, target)
    return CallableTaskTargetRunner(cast(TaskDirectTarget, target))
