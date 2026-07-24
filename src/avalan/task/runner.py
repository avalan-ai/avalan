from ..container import (
    ContainerAsyncBackend,
    ContainerBackendDiagnosticCode,
    ContainerBackendOperation,
    ContainerBackendProbeResult,
    ContainerBackendSelection,
    ContainerOutputDecisionType,
    ContainerResultStatus,
    ContainerRunPlan,
    run_container_managed_lifecycle,
    select_container_backend,
)
from .artifact import (
    ArtifactStore,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskOutputArtifact,
    task_output_artifact_from_value,
)
from .attempt import TaskAttemptDecision, TaskAttemptPolicy
from .canonical import spec_hash
from .container import (
    TaskContainerPlans,
    TaskContainerVerificationError,
    task_container_event_payload,
    task_container_input_mount_manifest,
    task_container_lifecycle_run_plan,
    task_container_output_artifacts,
    task_container_output_contract,
    task_container_run_metadata,
    task_container_unsupported_input_mount_path,
    verify_task_container_request,
)
from .context import (
    TaskInputFile,
    TaskTargetContext,
    TaskUsageObservationTracker,
)
from .converters import (
    FileConverter,
    TaskFileConversionError,
    convert_task_artifact,
    convert_task_artifact_pages,
)
from .converters.registry import default_file_converters
from .definition import (
    ObservabilitySinkType,
    RunMode,
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
)
from .error import TaskError, TaskErrorValue, classify_task_error
from .event import (
    TaskEventCategory,
    TaskInteractionEventType,
    freeze_task_event_value,
    task_interaction_event_payload,
)
from .input import (
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskFileSourceKind,
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskRemoteUrlPolicy,
)
from .materialization import (
    TaskMaterializedFile,
    TaskRemoteUrlHttpClient,
    TaskRemoteUrlResolver,
    materialize_task_input_files,
    task_file_descriptors_from_input,
    task_provider_reference_input_files_from_input,
)
from .observability import (
    ObservabilitySink,
    TaskEventPipeline,
    TaskSanitizedEventObserver,
    observe_response_usage,
    record_response_usage,
)
from .privacy import (
    EncryptionProvider,
    HmacProvider,
    PrivacyField,
    PrivacySafeValue,
    PrivacySanitizationError,
    PrivacySanitizer,
)
from .schema import (
    TaskSchemaResolutionError,
    resolve_task_definition_schemas,
    task_definition_schema_base_path,
)
from .skills import (
    task_definition_with_skills_identity,
    task_skill_audit_event_publisher,
)
from .state import (
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskRunState,
)
from .store import (
    TaskAttempt,
    TaskAttemptSegment,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskSnapshotMetadata,
    TaskSnapshotValue,
    TaskStore,
    TaskStoreConflictError,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)
from .target import (
    CallableTaskTargetRunner,
    TaskTargetCompleted,
    TaskTargetRunner,
    TaskTargetSuspended,
    TaskValidationContext,
    task_target_outcome,
)
from .usage import usage_observations_from_response
from .validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    deduplicate_task_validation_issues,
    validate_task_definition,
    validate_task_input,
    validate_task_output,
)

from asyncio import (
    CancelledError,
    TimeoutError,
    create_task,
    gather,
    shield,
    wait_for,
)
from asyncio import sleep as asyncio_sleep
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from inspect import isawaitable
from pathlib import Path
from typing import Protocol, cast


class TaskRunnerError(RuntimeError):
    pass


class TaskRunExpiredError(RuntimeError):
    pass


class TaskDirectTarget(Protocol):
    def __call__(
        self,
        context: TaskTargetContext,
    ) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskRunResult:
    run: TaskRun
    attempt: TaskAttempt
    segment: TaskAttemptSegment | None = None
    output: object = None
    input_required: TaskTargetSuspended | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.run, TaskRun)
        assert isinstance(self.attempt, TaskAttempt)
        if self.segment is not None:
            assert isinstance(self.segment, TaskAttemptSegment)
            assert self.segment.run_id == self.run.run_id
            assert self.segment.attempt_id == self.attempt.attempt_id
        if self.input_required is not None:
            assert type(self.input_required) is TaskTargetSuspended
            assert self.run.state == TaskRunState.INPUT_REQUIRED
            assert self.attempt.state == TaskAttemptState.SUSPENDED
            assert self.output is None

    @property
    def suspended(self) -> bool:
        return self.input_required is not None


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutableInputFiles:
    files: tuple[TaskInputFile, ...]
    materialized_files: tuple[TaskMaterializedFile, ...] = ()

    def __post_init__(self) -> None:
        assert isinstance(self.files, tuple)
        for file in self.files:
            assert isinstance(file, TaskInputFile)
        assert isinstance(self.materialized_files, tuple)
        for materialized_file in self.materialized_files:
            assert isinstance(materialized_file, TaskMaterializedFile)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskExecutableInputFileEntry:
    file: TaskInputFile
    materialized_file: TaskMaterializedFile | None = None

    def __post_init__(self) -> None:
        assert isinstance(self.file, TaskInputFile)
        if self.materialized_file is not None:
            assert isinstance(self.materialized_file, TaskMaterializedFile)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskContainerAttemptResult:
    output: object
    output_artifacts_recorded: bool = False


class TaskWorkerRuntimeEnvelopeRunner(Protocol):
    trusted_runtime_envelope_runner: bool

    def __call__(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        plans: TaskContainerPlans,
        input_mounts: tuple[dict[str, object], ...],
    ) -> Awaitable[TaskContainerAttemptResult]: ...


def is_trusted_task_worker_runtime_envelope_runner(
    runner: object,
) -> bool:
    return getattr(runner, "trusted_runtime_envelope_runner", False) is True


async def build_task_executable_input_files(
    definition: TaskDefinition,
    input_value: object,
    *,
    files: tuple[TaskInputFile, ...] = (),
    roots: Iterable[str | Path],
    artifact_store: ArtifactStore | None,
    hmac_provider: HmacProvider | None = None,
    task_store: TaskStore | None = None,
    run: TaskRun | None = None,
    attempt: TaskAttempt | None = None,
    file_converters: Mapping[str, FileConverter] | None = None,
    remote_url_policy: TaskRemoteUrlPolicy | None = None,
    remote_url_http_client: TaskRemoteUrlHttpClient | None = None,
    remote_url_resolver: TaskRemoteUrlResolver | None = None,
    now: datetime | None = None,
) -> TaskExecutableInputFiles:
    assert isinstance(definition, TaskDefinition)
    assert isinstance(files, tuple)
    for file in files:
        assert isinstance(file, TaskInputFile)
    if task_store is not None:
        assert isinstance(run, TaskRun)
        assert isinstance(attempt, TaskAttempt)
    issues = validate_explicit_task_input_files(files, now=now)
    if issues:
        raise TaskValidationError(issues)
    provider_reference_files = task_provider_reference_input_files_from_input(
        definition,
        input_value,
        now=now,
    )
    materialized_files = await materialize_task_input_files(
        definition,
        input_value,
        roots=roots,
        artifact_store=artifact_store,
        hmac_provider=hmac_provider,
        remote_url_policy=remote_url_policy,
        remote_url_http_client=remote_url_http_client,
        remote_url_resolver=remote_url_resolver,
        task_store=task_store,
        run_id=run.run_id if run is not None else None,
        attempt_id=attempt.attempt_id if attempt is not None else None,
    )
    input_files = await task_input_files_from_materialized(
        definition,
        materialized_files,
        artifact_store=artifact_store,
        file_converters=file_converters,
        task_store=task_store,
        run=run,
        attempt=attempt,
    )
    return TaskExecutableInputFiles(
        files=(*files, *provider_reference_files, *input_files),
        materialized_files=materialized_files,
    )


async def task_input_files_from_materialized(
    definition: TaskDefinition,
    materialized_files: tuple[TaskMaterializedFile, ...],
    *,
    artifact_store: ArtifactStore | None,
    file_converters: Mapping[str, FileConverter] | None = None,
    task_store: TaskStore | None = None,
    run: TaskRun | None = None,
    attempt: TaskAttempt | None = None,
) -> tuple[TaskInputFile, ...]:
    assert isinstance(definition, TaskDefinition)
    groups = await task_input_file_groups_from_materialized(
        definition,
        materialized_files,
        artifact_store=artifact_store,
        file_converters=file_converters,
        task_store=task_store,
        run=run,
        attempt=attempt,
    )
    return tuple(file for group in groups for file in group)


async def task_input_file_groups_from_materialized(
    definition: TaskDefinition,
    materialized_files: tuple[TaskMaterializedFile, ...],
    *,
    artifact_store: ArtifactStore | None,
    file_converters: Mapping[str, FileConverter] | None = None,
    task_store: TaskStore | None = None,
    run: TaskRun | None = None,
    attempt: TaskAttempt | None = None,
) -> tuple[tuple[TaskInputFile, ...], ...]:
    assert isinstance(definition, TaskDefinition)
    input_file_groups: list[tuple[TaskInputFile, ...]] = []
    converters = _file_converters(file_converters)
    for index, file in enumerate(materialized_files):
        converted_files = await converted_task_input_files(
            definition,
            file,
            index=index,
            artifact_store=artifact_store,
            file_converters=converters,
            task_store=task_store,
            run=run,
            attempt=attempt,
        )
        input_file_groups.append(converted_files or (file.as_input_file(),))
    return tuple(input_file_groups)


async def converted_task_input_file(
    definition: TaskDefinition,
    file: TaskMaterializedFile,
    *,
    index: int,
    artifact_store: ArtifactStore | None,
    file_converters: Mapping[str, FileConverter],
    task_store: TaskStore | None = None,
    run: TaskRun | None = None,
    attempt: TaskAttempt | None = None,
) -> TaskInputFile | None:
    files = await converted_task_input_files(
        definition,
        file,
        index=index,
        artifact_store=artifact_store,
        file_converters=file_converters,
        task_store=task_store,
        run=run,
        attempt=attempt,
    )
    if not files:
        return None
    if len(files) != 1:
        raise TaskValidationError(
            (
                _conversion_issue(
                    index=index,
                    conversion_index=0,
                    is_array=definition.input.type == TaskInputType.FILE_ARRAY,
                ),
            )
        )
    return files[0]


async def converted_task_input_files(
    definition: TaskDefinition,
    file: TaskMaterializedFile,
    *,
    index: int,
    artifact_store: ArtifactStore | None,
    file_converters: Mapping[str, FileConverter],
    task_store: TaskStore | None = None,
    run: TaskRun | None = None,
    attempt: TaskAttempt | None = None,
) -> tuple[TaskInputFile, ...]:
    source_refs: tuple[TaskArtifactRef, ...] = (file.ref,)
    converted = False
    for conversion_index, request in enumerate(file.descriptor.conversions):
        if request.name in {"native", "none"}:
            continue
        converter = file_converters.get(request.name)
        if (
            converter is None
            or artifact_store is None
            or (
                callable(getattr(converter, "convert_pages", None))
                and (task_store is None or run is None)
            )
        ):
            raise TaskValidationError(
                (
                    _conversion_issue(
                        index=index,
                        conversion_index=conversion_index,
                        is_array=definition.input.type
                        == TaskInputType.FILE_ARRAY,
                    ),
                )
            )
        try:
            source_refs = await _converted_source_refs(
                source_refs,
                request,
                converter=converter,
                artifact_store=artifact_store,
                task_store=task_store,
                run=run,
                attempt=attempt,
                retention=TaskArtifactRetention(
                    delete_after_days=definition.artifact.retention_days,
                ),
            )
        except TaskFileConversionError as error:
            raise TaskValidationError(
                (
                    _conversion_issue(
                        index=index,
                        conversion_index=conversion_index,
                        is_array=definition.input.type
                        == TaskInputType.FILE_ARRAY,
                    ),
                )
            ) from error
        converted = True
    if not converted:
        return ()
    return tuple(
        TaskInputFile(
            logical_path=f"artifact:{ref.artifact_id}",
            artifact_ref=ref,
            media_type=ref.media_type,
            size_bytes=ref.size_bytes,
            metadata={
                **file.identity,
                **ref.metadata,
            },
        )
        for ref in source_refs
    )


async def _converted_source_refs(
    source_refs: tuple[TaskArtifactRef, ...],
    request: TaskFileConversionRequest,
    *,
    converter: FileConverter,
    artifact_store: ArtifactStore,
    task_store: TaskStore | None,
    run: TaskRun | None,
    attempt: TaskAttempt | None,
    retention: TaskArtifactRetention,
) -> tuple[TaskArtifactRef, ...]:
    page_converter = callable(getattr(converter, "convert_pages", None))
    converted_refs: list[TaskArtifactRef] = []
    for source_ref in source_refs:
        if page_converter:
            assert task_store is not None
            assert run is not None
            collection = await convert_task_artifact_pages(
                source_ref,
                request,
                converter=converter,
                artifact_store=artifact_store,
                task_store=task_store,
                run_id=run.run_id,
                attempt_id=(
                    attempt.attempt_id if attempt is not None else None
                ),
                retention=retention,
            )
            converted_refs.extend(page.ref for page in collection.pages)
            continue
        converted_artifact = await convert_task_artifact(
            source_ref,
            request,
            converter=converter,
            artifact_store=artifact_store,
            task_store=task_store,
            run_id=run.run_id if run is not None else None,
            attempt_id=attempt.attempt_id if attempt is not None else None,
            retention=retention,
        )
        converted_refs.append(converted_artifact.ref)
    return tuple(converted_refs)


def task_execution_file_entries_value(
    entries: tuple[TaskExecutableInputFileEntry, ...],
) -> tuple[TaskSnapshotValue, ...]:
    return tuple(
        _execution_file_entry_value(entry, index)
        for index, entry in enumerate(entries)
    )


def task_execution_file_entries_from_value(
    value: object,
) -> tuple[TaskExecutableInputFileEntry, ...]:
    if isinstance(value, Mapping):
        try:
            return (_execution_file_entry_from_value(value),)
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise TaskValidationError(
                (_queue_file_payload_issue(),)
            ) from error
    if not isinstance(value, list | tuple):
        raise TaskValidationError((_queue_file_payload_issue(),))
    entries: list[TaskExecutableInputFileEntry] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TaskValidationError((_queue_file_payload_issue(),))
        try:
            entries.append(_execution_file_entry_from_value(item))
        except (AssertionError, KeyError, TypeError, ValueError) as error:
            raise TaskValidationError(
                (_queue_file_payload_issue(),)
            ) from error
    return tuple(entries)


def task_input_file_entries_for_queue(
    *,
    files: tuple[TaskInputFile, ...],
    provider_reference_files: tuple[TaskInputFile, ...],
    materialized_files: tuple[TaskMaterializedFile, ...],
) -> tuple[TaskExecutableInputFileEntry, ...]:
    return (
        *(
            TaskExecutableInputFileEntry(file=file)
            for file in (*files, *provider_reference_files)
        ),
        *(
            TaskExecutableInputFileEntry(
                file=materialized_file.as_input_file(),
                materialized_file=materialized_file,
            )
            for materialized_file in materialized_files
        ),
    )


def validate_explicit_task_input_files(
    files: tuple[TaskInputFile, ...],
    *,
    now: datetime | None = None,
) -> tuple[TaskValidationIssue, ...]:
    assert isinstance(files, tuple)
    issues: list[TaskValidationIssue] = []
    for index, file in enumerate(files):
        assert isinstance(file, TaskInputFile)
        provider_reference = file.provider_reference
        if provider_reference is None:
            continue
        if provider_reference.is_expired(now):
            issues.append(
                TaskValidationIssue(
                    code="input.invalid_file",
                    path=f"files[{index}].provider_reference.expires_at",
                    message="Task file provider reference has expired.",
                    hint="Refresh the provider reference before execution.",
                    category=TaskValidationCategory.VALUE,
                )
            )
        if (
            file.media_type is not None
            and provider_reference.mime_type is not None
            and file.media_type != provider_reference.mime_type
        ):
            issues.append(
                TaskValidationIssue(
                    code="input.invalid_file",
                    path=f"files[{index}].provider_reference.mime_type",
                    message=(
                        "Task file provider reference MIME type does not "
                        "match."
                    ),
                    hint=(
                        "Use one MIME type for the explicit file and "
                        "provider reference."
                    ),
                    category=TaskValidationCategory.VALUE,
                )
            )
    return tuple(issues)


class TaskRunFinalizer:
    async def finalize_suspended(
        self,
        *,
        store: TaskStore,
        run: TaskRun,
        attempt: TaskAttempt,
        segment: TaskAttemptSegment,
        outcome: TaskTargetSuspended,
    ) -> TaskRunResult:
        required = outcome.input_required
        result = TaskExecutionResult(
            metadata={
                "interaction": {
                    "kind": required.kind.value,
                    "request_id": str(required.request_id),
                    "continuation_id": str(required.continuation_id),
                    "detached_resumption_available": (
                        required.detached_resumption_available
                    ),
                    "checkpoint_id": outcome.checkpoint_id,
                }
            }
        )
        segment = await store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=TaskAttemptSegmentState.SUSPENDED,
            reason="input_required",
            request_id=str(required.request_id),
            continuation_id=str(required.continuation_id),
            checkpoint_id=outcome.checkpoint_id,
        )
        attempt = await store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.SUSPENDED,
            reason="input_required",
            result=result,
        )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.INPUT_REQUIRED,
            reason="input_required",
            result=result,
        )
        await store.append_event(
            run.run_id,
            attempt_id=attempt.attempt_id,
            event_type=TaskInteractionEventType.INPUT_REQUIRED.value,
            category=TaskEventCategory.INTERACTION,
            payload=task_interaction_event_payload(
                event_type=TaskInteractionEventType.INPUT_REQUIRED,
                request_id=str(required.request_id),
                continuation_id=str(required.continuation_id),
                segment_id=segment.segment_id,
            ),
        )
        return TaskRunResult(
            run=run,
            attempt=attempt,
            segment=segment,
            input_required=outcome,
        )

    async def finalize_success(
        self,
        *,
        store: TaskStore,
        run: TaskRun,
        attempt: TaskAttempt,
        output_summary: PrivacySafeValue,
    ) -> TaskRunResult:
        result = TaskExecutionResult(
            output_summary=_snapshot_value(output_summary)
        )
        attempt = await store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.SUCCEEDED,
            reason="finalized_success",
            result=result,
        )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.SUCCEEDED,
            reason="finalized_success",
            result=result,
        )
        return TaskRunResult(run=run, attempt=attempt)

    async def finalize_failure(
        self,
        *,
        store: TaskStore,
        run: TaskRun,
        attempt: TaskAttempt,
        error_summary: PrivacySafeValue,
    ) -> TaskRunResult:
        result = TaskExecutionResult(error=_snapshot_value(error_summary))
        attempt = await store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="finalized_failure",
            result=result,
        )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.FAILED,
            reason="finalized_failure",
            result=result,
        )
        return TaskRunResult(run=run, attempt=attempt)

    async def finalize_attempt_failure(
        self,
        *,
        store: TaskStore,
        attempt: TaskAttempt,
        error_summary: PrivacySafeValue,
        retry_decision: TaskAttemptDecision,
    ) -> TaskAttempt:
        result = TaskExecutionResult(
            error=_snapshot_value(
                _error_summary_with_attempt_policy(
                    error_summary,
                    retry_decision=retry_decision,
                )
            )
        )
        return await store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="attempt_retry",
            result=result,
        )

    async def finalize_cancelled(
        self,
        *,
        store: TaskStore,
        run: TaskRun,
        attempt: TaskAttempt,
        error_summary: PrivacySafeValue,
    ) -> TaskRunResult:
        result = TaskExecutionResult(error=_snapshot_value(error_summary))
        attempt = await _fail_attempt_if_running(
            store=store,
            attempt=attempt,
            reason="cancelled",
            result=result,
        )
        run = await store.get_run(run.run_id)
        if run.state == TaskRunState.RUNNING:
            run = await store.transition_run(
                run.run_id,
                from_states={TaskRunState.RUNNING},
                to_state=TaskRunState.CANCEL_REQUESTED,
                reason="cancel_requested",
            )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.CANCEL_REQUESTED},
            to_state=TaskRunState.CANCELLED,
            reason="cancelled",
            result=result,
        )
        return TaskRunResult(run=run, attempt=attempt)

    async def finalize_expired(
        self,
        *,
        store: TaskStore,
        run: TaskRun,
        attempt: TaskAttempt,
        error_summary: PrivacySafeValue,
    ) -> TaskRunResult:
        result = TaskExecutionResult(error=_snapshot_value(error_summary))
        attempt = await _fail_attempt_if_running(
            store=store,
            attempt=attempt,
            reason="run_expired",
            result=result,
        )
        run = await store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.EXPIRED,
            reason="run_expired",
            result=result,
        )
        return TaskRunResult(run=run, attempt=attempt)


class DirectTaskRunner:
    def __init__(
        self,
        store: TaskStore,
        *,
        target: TaskDirectTarget | TaskTargetRunner,
        hmac_provider: HmacProvider | None = None,
        encryption_provider: EncryptionProvider | None = None,
        raw_storage_allowed: bool = False,
        artifact_store: ArtifactStore | None = None,
        file_converters: Mapping[str, FileConverter] | None = None,
        finalizer: TaskRunFinalizer | None = None,
        definition_hash: (
            Callable[[TaskDefinition], str | Awaitable[str]] | None
        ) = None,
        execution_roots: Iterable[str | Path] = (),
        input_roots: Iterable[str | Path] | None = None,
        remote_url_policy: TaskRemoteUrlPolicy | None = None,
        remote_url_http_client: TaskRemoteUrlHttpClient | None = None,
        remote_url_resolver: TaskRemoteUrlResolver | None = None,
        event_observer: TaskSanitizedEventObserver | None = None,
        metrics_event_observer: TaskSanitizedEventObserver | None = None,
        trace_event_observer: TaskSanitizedEventObserver | None = None,
        observability_sink: ObservabilitySink | None = None,
        container_backend: ContainerAsyncBackend | None = None,
        worker_runtime_envelope_runner: (
            TaskWorkerRuntimeEnvelopeRunner | None
        ) = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._target = _target_runner(target)
        self._hmac_provider = hmac_provider
        self._encryption_provider = encryption_provider
        self._raw_storage_allowed = raw_storage_allowed
        self._artifact_store = artifact_store
        self._file_converters = _file_converters(file_converters)
        self._finalizer = finalizer or TaskRunFinalizer()
        self._definition_hash = definition_hash or spec_hash
        self._execution_roots = tuple(execution_roots)
        self._input_roots = (
            tuple(input_roots)
            if input_roots is not None
            else self._execution_roots
        )
        self._remote_url_policy = remote_url_policy
        self._remote_url_http_client = remote_url_http_client
        self._remote_url_resolver = remote_url_resolver
        self._event_observer = event_observer
        self._metrics_event_observer = metrics_event_observer
        self._trace_event_observer = trace_event_observer
        self._observability_sink = observability_sink
        if container_backend is not None:
            assert isinstance(container_backend, ContainerAsyncBackend)
        self._container_backend = container_backend
        if worker_runtime_envelope_runner is not None:
            assert callable(worker_runtime_envelope_runner)
            assert is_trusted_task_worker_runtime_envelope_runner(
                worker_runtime_envelope_runner
            ), "worker runtime envelope runner must be trusted"
        self._worker_runtime_envelope_runner = worker_runtime_envelope_runner
        self._clock = clock or _utc_now
        self._sleep = sleep or asyncio_sleep

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
        assert isinstance(files, tuple)
        if expires_at is not None:
            assert isinstance(expires_at, datetime)
        for file in files:
            assert isinstance(file, TaskInputFile)
        if definition.run.mode != RunMode.DIRECT:
            raise TaskRunnerError("direct runner requires direct run mode")
        schema_base_path = task_definition_schema_base_path(definition)
        definition = await self._resolve_definition_schemas(definition)
        skill_audit_sanitizer = self._sanitizer(definition)
        definition = await task_definition_with_skills_identity(
            definition,
            event_manager=task_skill_audit_event_publisher(
                sanitizer=skill_audit_sanitizer,
                event_observer=self._event_observer,
                metrics_event_observer=(
                    self._metrics_event_observer
                    if definition.observability.metrics
                    else None
                ),
                trace_event_observer=(
                    self._trace_event_observer
                    if definition.observability.trace
                    else None
                ),
                observability_sink=self._observability_sink_for(definition),
            ),
            schema_base_path=schema_base_path,
        )
        self._validate(definition, input_value)
        await self._validate_target(definition)
        sanitizer = self._sanitizer(definition)
        definition_id = await self._definition_hash_value(definition)
        await self._store.register_definition(
            definition,
            definition_hash=definition_id,
        )
        initial_input_mounts = task_container_input_mount_manifest(
            files,
            allowed_roots=tuple(Path(root) for root in self._execution_roots),
        )
        run = await self._store.create_run(
            TaskExecutionRequest(
                definition_id=definition_id,
                input_summary=_snapshot_value(
                    sanitizer.sanitize(
                        PrivacyField.INPUT,
                        _input_summary_value(definition, input_value),
                    )
                ),
                file_summaries=self._file_summaries(sanitizer, files),
                idempotency_key=idempotency_key,
                queue=definition.run.queue,
                metadata=freeze_snapshot_metadata(
                    task_container_run_metadata(
                        definition,
                        metadata,
                        input_mounts=initial_input_mounts,
                    )
                ),
            ),
            metadata={"runner": "direct"},
        )
        run = await self._store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        attempt = await self._store.create_attempt(run.run_id)
        attempt = await self._store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
        )
        run = await self._store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.RUNNING,
            reason="started",
        )
        try:
            await self._check_cancellation_or_expiry(run, expires_at)
        except (KeyboardInterrupt, SystemExit):
            raise
        except TaskStoreConflictError:
            raise
        except TaskRunExpiredError:
            return await self._finalizer.finalize_expired(
                store=self._store,
                run=run,
                attempt=attempt,
                error_summary=self._safe_task_error_summary(
                    sanitizer,
                    _task_error_with_details(
                        TaskError.timeout(),
                        {"scope": "run"},
                    ),
                ),
            )
        except CancelledError as error:
            return await self._finalizer.finalize_cancelled(
                store=self._store,
                run=run,
                attempt=attempt,
                error_summary=self._safe_error_summary(
                    sanitizer,
                    error,
                ),
            )
        except BaseException as error:
            return await self._finalizer.finalize_failure(
                store=self._store,
                run=run,
                attempt=attempt,
                error_summary=self._safe_error_summary(
                    sanitizer,
                    error,
                ),
            )
        attempt_policy = TaskAttemptPolicy.from_retry_policy(
            definition.retry,
        )
        return await self._run_attempts(
            definition=definition,
            run=run,
            first_attempt=attempt,
            input_value=input_value,
            files=files,
            metadata=metadata,
            sanitizer=sanitizer,
            attempt_policy=attempt_policy,
            expires_at=expires_at,
        )

    async def _run_attempts(
        self,
        *,
        definition: TaskDefinition,
        run: TaskRun,
        first_attempt: TaskAttempt,
        input_value: object,
        files: tuple[TaskInputFile, ...],
        metadata: Mapping[str, object] | None,
        sanitizer: PrivacySanitizer,
        attempt_policy: TaskAttemptPolicy,
        expires_at: datetime | None,
    ) -> TaskRunResult:
        attempt = first_attempt
        while True:
            try:
                return await self._run_single_attempt(
                    definition=definition,
                    run=run,
                    attempt=attempt,
                    input_value=input_value,
                    files=files,
                    metadata=metadata,
                    sanitizer=sanitizer,
                    expires_at=expires_at,
                )
            except (KeyboardInterrupt, SystemExit):
                raise
            except TaskStoreConflictError:
                raise
            except BaseException as error:
                await self._close_running_segment(
                    attempt,
                    state=(
                        TaskAttemptSegmentState.ABANDONED
                        if isinstance(error, CancelledError)
                        else TaskAttemptSegmentState.FAILED
                    ),
                    reason=(
                        "cancelled"
                        if isinstance(error, CancelledError)
                        else "execution_failed"
                    ),
                )
                if isinstance(error, TaskRunExpiredError):
                    return await self._finalizer.finalize_expired(
                        store=self._store,
                        run=run,
                        attempt=attempt,
                        error_summary=self._safe_task_error_summary(
                            sanitizer,
                            _task_error_with_details(
                                TaskError.timeout(),
                                {"scope": "run"},
                            ),
                        ),
                    )
                if isinstance(error, CancelledError):
                    return await self._finalizer.finalize_cancelled(
                        store=self._store,
                        run=run,
                        attempt=attempt,
                        error_summary=self._safe_error_summary(
                            sanitizer,
                            error,
                        ),
                    )
                task_error = classify_task_error(error)
                decision = attempt_policy.decide(
                    attempt_number=attempt.attempt_number,
                    error=task_error,
                )
                if not decision.should_retry:
                    return await self._finalizer.finalize_failure(
                        store=self._store,
                        run=run,
                        attempt=attempt,
                        error_summary=self._safe_task_error_summary(
                            sanitizer,
                            _task_error_with_attempt_counts(
                                task_error,
                                decision,
                            ),
                        ),
                    )
                await self._finalizer.finalize_attempt_failure(
                    store=self._store,
                    attempt=attempt,
                    error_summary=self._safe_task_error_summary(
                        sanitizer,
                        task_error,
                    ),
                    retry_decision=decision,
                )
                try:
                    await self._backoff_before_retry(
                        decision,
                        run=run,
                        expires_at=expires_at,
                    )
                except TaskRunExpiredError:
                    return await self._finalizer.finalize_expired(
                        store=self._store,
                        run=run,
                        attempt=attempt,
                        error_summary=self._safe_task_error_summary(
                            sanitizer,
                            _task_error_with_details(
                                TaskError.timeout(),
                                {"scope": "run"},
                            ),
                        ),
                    )
                except CancelledError as error:
                    return await self._finalizer.finalize_cancelled(
                        store=self._store,
                        run=run,
                        attempt=attempt,
                        error_summary=self._safe_error_summary(
                            sanitizer,
                            error,
                        ),
                    )
                attempt = await self._store.create_attempt(run.run_id)
                attempt = await self._store.transition_attempt(
                    attempt.attempt_id,
                    from_states={TaskAttemptState.CREATED},
                    to_state=TaskAttemptState.RUNNING,
                    reason="started",
                )

    async def _run_single_attempt(
        self,
        *,
        definition: TaskDefinition,
        run: TaskRun,
        attempt: TaskAttempt,
        input_value: object,
        files: tuple[TaskInputFile, ...],
        metadata: Mapping[str, object] | None,
        sanitizer: PrivacySanitizer,
        expires_at: datetime | None,
    ) -> TaskRunResult:
        segment = await self._start_attempt_segment(attempt)

        async def observe_usage(response: object) -> None:
            await self._record_usage(
                response,
                definition=definition,
                run=run,
                attempt=attempt,
                segment=segment,
            )

        observed_usage_ids: set[str] = set()

        async def observe_live_usage(response: object) -> None:
            await observe_response_usage(
                response,
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                segment_id=segment.segment_id,
                usage_observer=self._event_observer,
                observed_usage_ids=observed_usage_ids,
            )

        if definition.observability.metrics:
            usage_observer = observe_usage
        elif self._event_observer is not None:
            usage_observer = observe_live_usage
        else:
            usage_observer = None
        usage_tracker = TaskUsageObservationTracker(
            usage_observer,
            has_observations=lambda response: bool(
                usage_observations_from_response(response)
            ),
        )
        await self._check_cancellation_or_expiry(run, expires_at)
        executable_files = await self._build_executable_input_files(
            definition,
            input_value,
            files=files,
            run=run,
            attempt=attempt,
        )
        files = executable_files.files
        input_mounts = task_container_input_mount_manifest(
            files,
            allowed_roots=tuple(Path(root) for root in self._execution_roots),
        )
        container_result = await self._run_task_container(
            definition,
            run=run,
            attempt=attempt,
            input_mounts=input_mounts,
            sanitizer=sanitizer,
            expires_at=expires_at,
        )
        if container_result is not None:
            output = container_result.output
            await self._check_cancellation_or_expiry(run, expires_at)
            await usage_tracker.observe(output)
            output_issues = validate_task_output(definition, output)
            if output_issues:
                await self._transition_segment(
                    segment,
                    to_state=TaskAttemptSegmentState.FAILED,
                    reason="output_invalid",
                )
                return await self._finalizer.finalize_failure(
                    store=self._store,
                    run=run,
                    attempt=attempt,
                    error_summary=self._safe_task_error_summary(
                        sanitizer,
                        TaskError.output_contract(output_issues),
                    ),
                )
            output_summary = sanitizer.sanitize(
                PrivacyField.OUTPUT,
                _output_summary_value(definition, output),
            )
            if not container_result.output_artifacts_recorded:
                await self._record_output_artifacts(
                    definition,
                    output,
                    run=run,
                    attempt=attempt,
                    sanitizer=sanitizer,
                )
            await self._check_cancellation_or_expiry(run, expires_at)
            segment = await self._transition_segment(
                segment,
                to_state=TaskAttemptSegmentState.SUCCEEDED,
                reason="completed",
            )
            finalized = await self._finalizer.finalize_success(
                store=self._store,
                run=run,
                attempt=attempt,
                output_summary=output_summary,
            )
            return TaskRunResult(
                run=finalized.run,
                attempt=finalized.attempt,
                segment=segment,
                output=output,
            )
        context = TaskTargetContext(
            definition=definition,
            execution=attempt.context,
            input_value=input_value,
            files=files,
            metadata=metadata or {},
            cancellation_checker=lambda: self._check_cancellation_or_expiry(
                run,
                expires_at,
            ),
            event_listener=self._event_pipeline(
                definition,
                run=run,
                attempt=attempt,
                sanitizer=sanitizer,
            ),
            usage_observer=(
                usage_tracker.observe if usage_observer is not None else None
            ),
            artifact_store=self._artifact_store,
            task_store=self._store,
            file_converters=self._file_converters,
        )
        await self._check_cancellation_or_expiry(run, expires_at)
        target_outcome = task_target_outcome(
            await wait_for(
                self._target.run(context),
                timeout=definition.run.timeout_seconds,
            )
        )
        await self._check_cancellation_or_expiry(run, expires_at)
        if type(target_outcome) is TaskTargetSuspended:
            if target_outcome.durable is not None:
                raise TaskRunnerError(
                    "durable task suspension requires a queued coordinator"
                )
            return await self._finalizer.finalize_suspended(
                store=self._store,
                run=run,
                attempt=attempt,
                segment=segment,
                outcome=target_outcome,
            )
        assert type(target_outcome) is TaskTargetCompleted
        output = target_outcome.output
        await usage_tracker.observe(output)
        output_issues = validate_task_output(definition, output)
        if output_issues:
            await self._transition_segment(
                segment,
                to_state=TaskAttemptSegmentState.FAILED,
                reason="output_invalid",
            )
            return await self._finalizer.finalize_failure(
                store=self._store,
                run=run,
                attempt=attempt,
                error_summary=self._safe_task_error_summary(
                    sanitizer,
                    TaskError.output_contract(output_issues),
                ),
            )
        output_summary = sanitizer.sanitize(
            PrivacyField.OUTPUT,
            _output_summary_value(definition, output),
        )
        await self._record_output_artifacts(
            definition,
            output,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
        )
        await self._check_cancellation_or_expiry(run, expires_at)
        segment = await self._transition_segment(
            segment,
            to_state=TaskAttemptSegmentState.SUCCEEDED,
            reason="completed",
        )
        try:
            result = await self._finalizer.finalize_success(
                store=self._store,
                run=run,
                attempt=attempt,
                output_summary=output_summary,
            )
        except TaskStoreConflictError:
            await self._check_cancellation_or_expiry(run, expires_at)
            raise
        return TaskRunResult(
            run=result.run,
            attempt=result.attempt,
            segment=segment,
            output=output,
        )

    async def _start_attempt_segment(
        self,
        attempt: TaskAttempt,
    ) -> TaskAttemptSegment:
        segments = await self._store.list_attempt_segments(attempt.attempt_id)
        previous = segments[-1] if segments else None
        segment = await self._store.create_attempt_segment(
            attempt.attempt_id,
            resumed_from_segment_id=(
                previous.segment_id
                if previous is not None
                and previous.state == TaskAttemptSegmentState.SUSPENDED
                else None
            ),
        )
        return await self._store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.CREATED},
            to_state=TaskAttemptSegmentState.RUNNING,
            reason="started",
        )

    async def _transition_segment(
        self,
        segment: TaskAttemptSegment,
        *,
        to_state: TaskAttemptSegmentState,
        reason: str,
    ) -> TaskAttemptSegment:
        return await self._store.transition_attempt_segment(
            segment.segment_id,
            from_states={TaskAttemptSegmentState.RUNNING},
            to_state=to_state,
            reason=reason,
        )

    async def _close_running_segment(
        self,
        attempt: TaskAttempt,
        *,
        state: TaskAttemptSegmentState,
        reason: str,
    ) -> None:
        segments = await self._store.list_attempt_segments(attempt.attempt_id)
        if (
            not segments
            or segments[-1].state != TaskAttemptSegmentState.RUNNING
        ):
            return
        await self._transition_segment(
            segments[-1],
            to_state=state,
            reason=reason,
        )

    def _validate(
        self,
        definition: TaskDefinition,
        input_value: object,
    ) -> None:
        issues: list[TaskValidationIssue] = []
        issues.extend(
            validate_task_definition(
                definition,
                hmac_provider=self._hmac_provider,
                encryption_provider=self._encryption_provider,
                require_configured_keys=True,
                raw_storage_allowed=self._raw_storage_allowed,
                execution_roots=self._execution_roots,
                file_converters=self._file_converters,
            )
        )
        issues.extend(
            validate_task_input(
                definition,
                input_value,
                file_converters=self._file_converters,
                remote_url_policy=self._remote_url_policy,
            )
        )
        issues = list(deduplicate_task_validation_issues(issues))
        if issues:
            raise TaskValidationError(tuple(issues))

    async def _resolve_definition_schemas(
        self,
        definition: TaskDefinition,
    ) -> TaskDefinition:
        try:
            return await resolve_task_definition_schemas(
                definition,
                schema_base_path=None,
            )
        except TaskSchemaResolutionError as error:
            raise TaskValidationError(
                (_schema_resolution_issue(error),)
            ) from error

    async def _definition_hash_value(
        self,
        definition: TaskDefinition,
    ) -> str:
        value = self._definition_hash(definition)
        if isawaitable(value):
            value = await value
        assert isinstance(value, str) and value.strip()
        return value

    async def _validate_target(self, definition: TaskDefinition) -> None:
        issues = await self._target.validate_definition(
            definition,
            TaskValidationContext(
                execution_roots=tuple(
                    Path(root) for root in self._execution_roots
                ),
                artifact_store=self._artifact_store,
                task_store=self._store,
                file_converters=self._file_converters,
            ),
        )
        if issues:
            raise TaskValidationError(issues)

    async def _run_task_container(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        input_mounts: tuple[dict[str, object], ...],
        sanitizer: PrivacySanitizer,
        expires_at: datetime | None,
    ) -> TaskContainerAttemptResult | None:
        try:
            plans = verify_task_container_request(
                definition,
                run=run,
                attempt=attempt,
                input_mounts=input_mounts,
                allow_dynamic_input_mounts=True,
            )
        except TaskContainerVerificationError as error:
            raise TaskValidationError(
                (_container_validation_issue(error),)
            ) from error
        if not plans.enabled:
            return None
        await self._record_container_event(
            definition,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
            event_type="container_plan_verified",
            plans=plans,
            input_mounts=input_mounts,
            status="verified",
        )
        if plans.worker_envelope is not None:
            if self._worker_runtime_envelope_runner is not None:
                worker_envelope_result = (
                    await self._worker_runtime_envelope_runner(
                        definition,
                        run=run,
                        attempt=attempt,
                        plans=plans,
                        input_mounts=input_mounts,
                    )
                )
                assert isinstance(
                    worker_envelope_result,
                    TaskContainerAttemptResult,
                )
                await self._record_container_event(
                    definition,
                    run=run,
                    attempt=attempt,
                    sanitizer=sanitizer,
                    event_type="container_worker_envelope_completed",
                    plans=plans,
                    input_mounts=input_mounts,
                    status="completed",
                )
                return worker_envelope_result
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.worker_envelope_unsupported",
                        path="container.worker_envelope",
                        message=(
                            "Task worker runtime envelopes require an "
                            "envelope-aware task runtime."
                        ),
                        hint=(
                            "Route this task to a trusted envelope-aware "
                            "worker or remove the worker envelope policy."
                        ),
                    ),
                )
            )
        if plans.attempt is None:
            return None
        if self._container_backend is None:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.backend_unavailable",
                        path="container.backend",
                        message=(
                            "Task container execution requires an injected "
                            "container backend."
                        ),
                        hint=(
                            "Run this task with a trusted container-capable "
                            "runner."
                        ),
                    ),
                )
            )
        run_plan = task_container_lifecycle_run_plan(
            plans,
            input_mounts=input_mounts,
        )
        assert run_plan is not None
        unsupported_input_mount_path = (
            task_container_unsupported_input_mount_path(input_mounts)
        )
        if unsupported_input_mount_path is not None:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.input_mount_unsupported",
                        path=unsupported_input_mount_path,
                        message=(
                            "Task-attempt container inputs must be "
                            "artifact-backed scoped mounts."
                        ),
                        hint=(
                            "Materialize task inputs to task artifacts before "
                            "running them in a required container."
                        ),
                    ),
                )
            )
        output_contract = task_container_output_contract(definition, plans)
        if output_contract is None:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.task_execution_unsupported",
                        path="container.output",
                        message=(
                            "This task output contract cannot be completed "
                            "through the task-attempt container runtime."
                        ),
                        hint=(
                            "Use a file or artifact output contract with "
                            "container artifact output enabled, or run "
                            "without a required task-attempt container."
                        ),
                    ),
                )
            )
        if not output_contract.enabled:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.output_unsupported",
                        path="container.output",
                        message=(
                            "Task-attempt container output artifacts are "
                            "disabled by the trusted container profile."
                        ),
                        hint=(
                            "Enable artifact output in the trusted profile "
                            "or use an output contract that does not require "
                            "container artifact collection."
                        ),
                    ),
                )
            )
        probe = await self._container_backend.probe()
        _raise_for_container_backend_selection(run_plan, probe)
        lifecycle_task = create_task(
            run_container_managed_lifecycle(
                self._container_backend,
                run_plan,
                output_contract=output_contract,
            )
        )
        try:
            while not lifecycle_task.done():
                try:
                    await wait_for(shield(lifecycle_task), timeout=0.1)
                except TimeoutError:
                    await self._check_cancellation_or_expiry(run, expires_at)
                    continue
            result = await lifecycle_task
        finally:
            if not lifecycle_task.done():
                lifecycle_task.cancel()
                try:
                    await lifecycle_task
                except CancelledError:
                    pass
        status = cast(ContainerResultStatus, result.execution.status)
        await self._record_container_event(
            definition,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
            event_type="container_lifecycle_completed",
            plans=plans,
            input_mounts=input_mounts,
            status=status.value,
        )
        if result.execution.status is not ContainerResultStatus.COMPLETED:
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.execution_failed",
                        path="container.result",
                        message=(
                            "Container lifecycle did not complete"
                            " successfully."
                        ),
                        hint=(
                            "Inspect sanitized container events and retry"
                            " safely."
                        ),
                    ),
                )
            )
        if (
            result.output is None
            or result.output.decision is not ContainerOutputDecisionType.ACCEPT
            or not result.output.artifacts
        ):
            raise TaskValidationError(
                (
                    _container_issue(
                        code="container.output_unsupported",
                        path="container.output",
                        message=(
                            "Task-attempt container execution did not return "
                            "accepted task output artifacts."
                        ),
                        hint=(
                            "Configure the container backend to copy accepted "
                            "task artifacts for this output contract."
                        ),
                    ),
                )
            )
        try:
            output = await task_container_output_artifacts(
                definition,
                tuple(result.output.artifacts),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=self._artifact_store,
            )
        except TaskContainerVerificationError as error:
            raise TaskValidationError(
                (_container_validation_issue(error),)
            ) from error
        await self._record_output_artifacts(
            definition,
            output,
            run=run,
            attempt=attempt,
            sanitizer=sanitizer,
        )
        return TaskContainerAttemptResult(
            output=output,
            output_artifacts_recorded=True,
        )

    async def _record_container_event(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
        event_type: str,
        plans: TaskContainerPlans,
        input_mounts: tuple[dict[str, object], ...],
        status: str,
    ) -> None:
        if not definition.observability.capture_events:
            return
        payload = freeze_task_event_value(
            sanitizer.sanitize_event(
                event_type,
                task_container_event_payload(
                    status=status,
                    plans=plans,
                    input_mounts=input_mounts,
                ),
            ),
        )
        await self._store.append_event(
            run.run_id,
            event_type=event_type,
            category=TaskEventCategory.UNKNOWN,
            payload=payload,
            attempt_id=attempt.attempt_id,
        )

    def _sanitizer(self, definition: TaskDefinition) -> PrivacySanitizer:
        return PrivacySanitizer(
            definition.privacy,
            hmac_provider=self._hmac_provider,
            encryption_provider=self._encryption_provider,
            raw_storage_allowed=self._raw_storage_allowed,
        )

    def _event_pipeline(
        self,
        definition: TaskDefinition,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> TaskEventPipeline | None:
        metrics_observer = (
            self._metrics_event_observer
            if definition.observability.metrics
            else None
        )
        trace_observer = (
            self._trace_event_observer
            if definition.observability.trace
            else None
        )
        observability_sink = self._observability_sink_for(definition)
        if (
            not definition.observability.capture_events
            and self._event_observer is None
            and metrics_observer is None
            and trace_observer is None
            and observability_sink is None
        ):
            return None
        return TaskEventPipeline(
            store=self._store,
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            sanitizer=sanitizer,
            capture_events=definition.observability.capture_events,
            event_observer=self._event_observer,
            metrics_observer=metrics_observer,
            trace_observer=trace_observer,
            observability_sink=observability_sink,
        )

    def _file_summaries(
        self,
        sanitizer: PrivacySanitizer,
        files: tuple[TaskInputFile, ...],
    ) -> tuple[TaskSnapshotValue, ...]:
        return tuple(
            _snapshot_value(
                sanitizer.sanitize(PrivacyField.FILES, file.summary())
            )
            for file in files
        )

    async def _backoff_before_retry(
        self,
        decision: TaskAttemptDecision,
        *,
        run: TaskRun,
        expires_at: datetime | None,
    ) -> None:
        await self._check_cancellation_or_expiry(run, expires_at)
        if decision.retry_delay_seconds:
            await self._sleep(decision.retry_delay_seconds)
            await self._check_cancellation_or_expiry(run, expires_at)

    async def _check_cancellation_or_expiry(
        self,
        run: TaskRun,
        expires_at: datetime | None,
    ) -> None:
        current = await self._store.get_run(run.run_id)
        if current.state == TaskRunState.CANCEL_REQUESTED:
            raise CancelledError()
        if expires_at is not None and self._clock() >= expires_at:
            raise TaskRunExpiredError()

    def _safe_error_summary(
        self,
        sanitizer: PrivacySanitizer,
        error: BaseException,
    ) -> PrivacySafeValue:
        return self._safe_task_error_summary(
            sanitizer,
            classify_task_error(error),
        )

    def _safe_task_error_summary(
        self,
        sanitizer: PrivacySanitizer,
        error: TaskError,
    ) -> PrivacySafeValue:
        try:
            return sanitizer.sanitize(
                PrivacyField.ERRORS,
                error.as_dict(),
            )
        except PrivacySanitizationError:
            return {
                "category": error.category.value,
                "code": error.code.value,
                "privacy": "<redacted>",
            }

    async def _record_usage(
        self,
        response: object,
        *,
        definition: TaskDefinition,
        run: TaskRun,
        attempt: TaskAttempt,
        segment: TaskAttemptSegment,
    ) -> None:
        await record_response_usage(
            self._observability_sink_for(definition),
            store=self._store,
            response=response,
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            segment_id=segment.segment_id,
            usage_observer=self._event_observer,
        )

    def _observability_sink_for(
        self,
        definition: TaskDefinition,
    ) -> ObservabilitySink | None:
        if definition.observability.sinks == (ObservabilitySinkType.NOOP,):
            return None
        return self._observability_sink

    async def _record_output_artifacts(
        self,
        definition: TaskDefinition,
        output: object,
        *,
        run: TaskRun,
        attempt: TaskAttempt,
        sanitizer: PrivacySanitizer,
    ) -> None:
        safe_artifacts = tuple(
            _sanitize_output_artifact(artifact, sanitizer)
            for artifact in _output_artifacts_from_output(definition, output)
        )
        await gather(
            *(
                self._store.append_artifact(
                    run.run_id,
                    ref=safe_artifact.ref,
                    purpose=TaskArtifactPurpose.OUTPUT,
                    state=safe_artifact.state,
                    attempt_id=attempt.attempt_id,
                    provenance=safe_artifact.provenance,
                    retention=_output_artifact_retention(
                        definition,
                        safe_artifact,
                    ),
                    metadata=safe_artifact.metadata,
                )
                for safe_artifact in safe_artifacts
            )
        )

    async def _input_files_from_materialized(
        self,
        definition: TaskDefinition,
        materialized_files: tuple[TaskMaterializedFile, ...],
        *,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> tuple[TaskInputFile, ...]:
        return await task_input_files_from_materialized(
            definition,
            materialized_files,
            artifact_store=self._artifact_store,
            file_converters=self._file_converters,
            task_store=self._store,
            run=run,
            attempt=attempt,
        )

    async def _build_executable_input_files(
        self,
        definition: TaskDefinition,
        input_value: object,
        *,
        files: tuple[TaskInputFile, ...],
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> TaskExecutableInputFiles:
        issues = validate_explicit_task_input_files(
            files,
            now=self._clock(),
        )
        if issues:
            raise TaskValidationError(issues)
        provider_reference_files = (
            task_provider_reference_input_files_from_input(
                definition,
                input_value,
                now=self._clock(),
            )
        )
        materialized_files = await materialize_task_input_files(
            definition,
            input_value,
            roots=self._input_roots,
            artifact_store=self._artifact_store,
            hmac_provider=self._hmac_provider,
            remote_url_policy=self._remote_url_policy,
            remote_url_http_client=self._remote_url_http_client,
            remote_url_resolver=self._remote_url_resolver,
            task_store=self._store,
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
        )
        input_files = await self._input_files_from_materialized(
            definition,
            materialized_files,
            run=run,
            attempt=attempt,
        )
        return TaskExecutableInputFiles(
            files=(*files, *provider_reference_files, *input_files),
            materialized_files=materialized_files,
        )

    async def _converted_input_file(
        self,
        definition: TaskDefinition,
        file: TaskMaterializedFile,
        *,
        index: int,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> TaskInputFile | None:
        return await converted_task_input_file(
            definition,
            file,
            index=index,
            artifact_store=self._artifact_store,
            file_converters=self._file_converters,
            task_store=self._store,
            run=run,
            attempt=attempt,
        )


def _snapshot_value(value: PrivacySafeValue) -> TaskSnapshotValue:
    return freeze_snapshot_value(value)


def _execution_file_entry_value(
    entry: TaskExecutableInputFileEntry,
    index: int,
) -> TaskSnapshotValue:
    value: dict[str, object] = {
        "file": _input_file_value(entry.file),
        "order": index,
    }
    if entry.materialized_file is not None:
        value["materialized"] = _materialized_file_value(
            entry.materialized_file
        )
    return freeze_snapshot_value(value)


def _execution_file_entry_from_value(
    value: Mapping[str, object],
) -> TaskExecutableInputFileEntry:
    file_value = value["file"]
    assert isinstance(file_value, Mapping)
    materialized_value = value.get("materialized")
    file = _input_file_from_value(file_value)
    return TaskExecutableInputFileEntry(
        file=file,
        materialized_file=(
            _materialized_file_from_value(materialized_value, file=file)
            if materialized_value is not None
            else None
        ),
    )


def _input_file_value(file: TaskInputFile) -> TaskSnapshotValue:
    value: dict[str, object] = {
        "logical_path": file.logical_path,
        "metadata": file.metadata,
    }
    if file.artifact_ref is not None:
        value["artifact_ref"] = _artifact_ref_value(file.artifact_ref)
    if file.provider_reference is not None:
        value["provider_reference"] = _provider_reference_value(
            file.provider_reference
        )
    if file.media_type is not None:
        value["media_type"] = file.media_type
    if file.size_bytes is not None:
        value["size_bytes"] = file.size_bytes
    return freeze_snapshot_value(value)


def _input_file_from_value(value: Mapping[str, object]) -> TaskInputFile:
    logical_path = value["logical_path"]
    assert isinstance(logical_path, str)
    artifact_ref_value = value.get("artifact_ref")
    provider_reference_value = value.get("provider_reference")
    metadata = value.get("metadata", {})
    assert isinstance(metadata, Mapping)
    media_type = value.get("media_type")
    size_bytes = value.get("size_bytes")
    return TaskInputFile(
        logical_path=logical_path,
        artifact_ref=(
            _artifact_ref_from_value(artifact_ref_value)
            if artifact_ref_value is not None
            else None
        ),
        provider_reference=(
            _provider_reference_from_value(provider_reference_value)
            if provider_reference_value is not None
            else None
        ),
        media_type=cast(str | None, media_type),
        size_bytes=cast(int | None, size_bytes),
        metadata=metadata,
    )


def _materialized_file_value(
    file: TaskMaterializedFile,
) -> TaskSnapshotValue:
    return freeze_snapshot_value(
        {
            "descriptor": _file_descriptor_value(file.descriptor),
            "descriptor_path": file.descriptor_path,
            "identity": file.identity,
            "ref": _artifact_ref_value(file.ref),
        }
    )


def _materialized_file_from_value(
    value: object,
    *,
    file: TaskInputFile,
) -> TaskMaterializedFile:
    assert isinstance(value, Mapping)
    descriptor_value = value["descriptor"]
    assert isinstance(descriptor_value, Mapping)
    descriptor_path = value["descriptor_path"]
    assert isinstance(descriptor_path, str)
    identity = value.get("identity", {})
    assert isinstance(identity, Mapping)
    ref_value = value.get("ref")
    ref = (
        _artifact_ref_from_value(ref_value)
        if ref_value is not None
        else file.artifact_ref
    )
    assert ref is not None
    return TaskMaterializedFile(
        descriptor=_file_descriptor_from_value(descriptor_value),
        descriptor_path=descriptor_path,
        ref=ref,
        identity=identity,
    )


def _artifact_ref_value(ref: TaskArtifactRef) -> TaskSnapshotValue:
    return freeze_snapshot_value(
        {
            "artifact_id": ref.artifact_id,
            "media_type": ref.media_type,
            "metadata": ref.metadata,
            "sha256": ref.sha256,
            "size_bytes": ref.size_bytes,
            "storage_key": ref.storage_key,
            "store": ref.store,
        }
    )


def _artifact_ref_from_value(value: object) -> TaskArtifactRef:
    assert isinstance(value, Mapping)
    return TaskArtifactRef(
        artifact_id=cast(str, value["artifact_id"]),
        store=cast(str, value["store"]),
        storage_key=cast(str, value["storage_key"]),
        media_type=cast(str | None, value.get("media_type")),
        size_bytes=cast(int | None, value.get("size_bytes")),
        sha256=cast(str | None, value.get("sha256")),
        metadata=cast(TaskSnapshotMetadata, value.get("metadata", {})),
    )


def _file_descriptor_value(
    descriptor: TaskFileDescriptor,
) -> TaskSnapshotValue:
    value: dict[str, object] = {
        "conversions": tuple(
            _conversion_request_value(request)
            for request in descriptor.conversions
        ),
        "metadata": descriptor.metadata,
        "reference": descriptor.reference,
        "source_kind": descriptor.source_kind.value,
    }
    if descriptor.role is not None:
        value["role"] = descriptor.role
    if descriptor.mime_type is not None:
        value["mime_type"] = descriptor.mime_type
    if descriptor.size_bytes is not None:
        value["size_bytes"] = descriptor.size_bytes
    if descriptor.sha256 is not None:
        value["sha256"] = descriptor.sha256
    if descriptor.provider_reference is not None:
        value["provider_reference"] = _provider_reference_value(
            descriptor.provider_reference
        )
    return freeze_snapshot_value(value)


def _file_descriptor_from_value(
    value: Mapping[str, object],
) -> TaskFileDescriptor:
    conversions = value.get("conversions", ())
    assert isinstance(conversions, list | tuple)
    provider_reference_value = value.get("provider_reference")
    return TaskFileDescriptor(
        source_kind=TaskFileSourceKind(cast(str, value["source_kind"])),
        reference=cast(str, value["reference"]),
        role=cast(str | None, value.get("role")),
        mime_type=cast(str | None, value.get("mime_type")),
        size_bytes=cast(int | None, value.get("size_bytes")),
        sha256=cast(str | None, value.get("sha256")),
        conversions=tuple(
            _conversion_request_from_value(conversion)
            for conversion in conversions
        ),
        provider_reference=(
            _provider_reference_from_value(provider_reference_value)
            if provider_reference_value is not None
            else None
        ),
        metadata=cast(Mapping[str, object], value.get("metadata", {})),
    )


def _conversion_request_value(
    request: TaskFileConversionRequest,
) -> TaskSnapshotValue:
    return freeze_snapshot_value(
        {
            "name": request.name,
            "options": request.options,
        }
    )


def _conversion_request_from_value(
    value: object,
) -> TaskFileConversionRequest:
    assert isinstance(value, Mapping)
    return TaskFileConversionRequest(
        name=cast(str, value["name"]),
        options=cast(Mapping[str, object], value.get("options", {})),
    )


def _provider_reference_value(
    reference: TaskProviderReference,
) -> TaskSnapshotValue:
    return freeze_snapshot_value(reference.execution_metadata())


def _provider_reference_from_value(value: object) -> TaskProviderReference:
    assert isinstance(value, Mapping)
    expires_at = value.get("expires_at")
    assert expires_at is None or isinstance(expires_at, str)
    return TaskProviderReference(
        kind=TaskProviderReferenceKind(cast(str, value["kind"])),
        provider=cast(str, value["provider"]),
        reference=cast(str, value["reference"]),
        owner_scope=cast(str | None, value.get("owner_scope")),
        expires_at=(
            datetime.fromisoformat(expires_at)
            if isinstance(expires_at, str)
            else None
        ),
        mime_type=cast(str | None, value.get("mime_type")),
        size_bucket=cast(str | None, value.get("size_bucket")),
        identity_hmac=cast(str | None, value.get("identity_hmac")),
        durable=cast(bool, value.get("durable", True)),
        metadata=cast(Mapping[str, object], value.get("metadata", {})),
    )


def _queue_file_payload_issue() -> TaskValidationIssue:
    return TaskValidationIssue(
        code="queue.file_payload_unavailable",
        path="request.input_payload.file_values",
        message=(
            "Queued task file inputs are unavailable for worker execution."
        ),
        hint="Queue file tasks with encrypted file payload storage enabled.",
        category=TaskValidationCategory.PRIVACY,
    )


def _container_validation_issue(
    error: TaskContainerVerificationError,
) -> TaskValidationIssue:
    return _container_issue(
        code=error.code,
        path=error.path,
        message=str(error),
        hint=error.hint,
    )


def _raise_for_container_backend_selection(
    run_plan: ContainerRunPlan,
    probe: ContainerBackendProbeResult,
) -> None:
    selection = select_container_backend(
        run_plan,
        (probe,),
    )
    if selection.ok:
        return
    raise TaskValidationError((_container_backend_selection_issue(selection),))


def _container_backend_selection_issue(
    selection: ContainerBackendSelection,
) -> TaskValidationIssue:
    diagnostic = selection.diagnostics[0] if selection.diagnostics else None
    code = (
        "container.backend_unavailable"
        if diagnostic is not None
        and diagnostic.code
        is ContainerBackendDiagnosticCode.BACKEND_UNAVAILABLE
        else "container.worker_capability_mismatch"
    )
    message = (
        diagnostic.message
        if diagnostic is not None
        else "The selected container backend cannot enforce the task plan."
    )
    operation = (
        ContainerBackendOperation.PROBE
        if diagnostic is None
        else diagnostic.operation
    )
    operation = cast(ContainerBackendOperation, operation)
    return _container_issue(
        code=code,
        path=f"container.backend.{operation.value}",
        message=message,
        hint=(
            "Route the task to a worker whose container backend can enforce "
            "the trusted plan."
        ),
    )


def _container_issue(
    *,
    code: str,
    path: str,
    message: str,
    hint: str,
) -> TaskValidationIssue:
    return TaskValidationIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _schema_resolution_issue(
    error: TaskSchemaResolutionError,
) -> TaskValidationIssue:
    path = error.path
    code = (
        "input.invalid_schema"
        if path.startswith("input.")
        else "output.invalid_schema"
    )
    return TaskValidationIssue(
        code=code,
        path=path,
        message="Task contract schema reference cannot be resolved.",
        hint="Use a local JSON object schema file.",
        category=TaskValidationCategory.VALUE,
    )


async def _fail_attempt_if_running(
    *,
    store: TaskStore,
    attempt: TaskAttempt,
    reason: str,
    result: TaskExecutionResult,
) -> TaskAttempt:
    attempt = await store.get_attempt(attempt.attempt_id)
    if attempt.state != TaskAttemptState.RUNNING:
        return attempt
    return await store.transition_attempt(
        attempt.attempt_id,
        from_states={TaskAttemptState.RUNNING},
        to_state=TaskAttemptState.FAILED,
        reason=reason,
        result=result,
    )


def _task_error_with_attempt_counts(
    error: TaskError,
    decision: TaskAttemptDecision,
) -> TaskError:
    return _task_error_with_details(
        error,
        {
            "failed_attempt_count": decision.attempt_number,
            "max_attempts": decision.max_attempts,
            "retry_exhausted": error.retryable,
        },
    )


def _task_error_with_details(
    error: TaskError,
    details: Mapping[str, TaskErrorValue],
) -> TaskError:
    return TaskError(
        category=error.category,
        code=error.code,
        message=error.message,
        retryable=error.retryable,
        details={**error.details, **details},
    )


def _error_summary_with_attempt_policy(
    error_summary: PrivacySafeValue,
    *,
    retry_decision: TaskAttemptDecision,
) -> PrivacySafeValue:
    if not isinstance(error_summary, Mapping):
        return error_summary
    return {
        **error_summary,
        "attempt": {
            "failed_attempt_count": retry_decision.attempt_number,
            "max_attempts": retry_decision.max_attempts,
            "retry_delay_seconds": retry_decision.retry_delay_seconds,
        },
    }


def _target_runner(
    target: TaskDirectTarget | TaskTargetRunner,
) -> TaskTargetRunner:
    run = getattr(target, "run", None)
    validate_definition = getattr(target, "validate_definition", None)
    if callable(run) and callable(validate_definition):
        return cast(TaskTargetRunner, target)
    return CallableTaskTargetRunner(cast(TaskDirectTarget, target))


def _file_converters(
    converters: Mapping[str, FileConverter] | None,
) -> Mapping[str, FileConverter]:
    values: dict[str, FileConverter] = dict(default_file_converters())
    values.update(converters or {})
    return values


def _conversion_issue(
    *,
    index: int,
    conversion_index: int,
    is_array: bool,
) -> TaskValidationIssue:
    path = (
        f"input[{index}].conversions[{conversion_index}]"
        if is_array
        else f"input.conversions[{conversion_index}]"
    )
    return TaskValidationIssue(
        code="input.invalid_file",
        path=path,
        message="Task file conversion could not be completed.",
        hint="Use a supported conversion with valid options.",
        category=TaskValidationCategory.UNSUPPORTED,
    )


def _input_summary_value(
    definition: TaskDefinition,
    input_value: object,
) -> object:
    if definition.input.type not in {
        TaskInputType.FILE,
        TaskInputType.FILE_ARRAY,
    }:
        return input_value
    try:
        descriptors = task_file_descriptors_from_input(definition, input_value)
    except (AssertionError, KeyError, ValueError):
        return input_value
    if not descriptors:
        return input_value
    values = tuple(
        _file_descriptor_summary(descriptor) for descriptor in descriptors
    )
    if len(values) == 1:
        return values[0]
    return values


def _output_summary_value(
    definition: TaskDefinition,
    output: object,
) -> object:
    artifacts = _output_artifacts_from_output(definition, output)
    if not artifacts:
        return output
    summaries = tuple(artifact.summary() for artifact in artifacts)
    if definition.output.type == TaskOutputType.FILE:
        return summaries[0]
    return summaries


def _output_artifacts_from_output(
    definition: TaskDefinition,
    output: object,
) -> tuple[TaskOutputArtifact, ...]:
    match definition.output.type:
        case TaskOutputType.FILE:
            return (_coerce_output_artifact(output),)
        case TaskOutputType.FILE_ARRAY | TaskOutputType.ARTIFACT_ARRAY:
            assert isinstance(output, list | tuple)
            return tuple(_coerce_output_artifact(item) for item in output)
        case _:
            return ()


def _coerce_output_artifact(value: object) -> TaskOutputArtifact:
    artifact = task_output_artifact_from_value(value)
    assert artifact is not None, "output artifact must be validated first"
    return artifact


def _sanitize_output_artifact(
    artifact: TaskOutputArtifact,
    sanitizer: PrivacySanitizer,
) -> TaskOutputArtifact:
    return TaskOutputArtifact(
        ref=_sanitize_artifact_ref(artifact.ref, sanitizer),
        state=artifact.state,
        provenance=_sanitize_artifact_provenance(
            artifact.provenance,
            sanitizer,
        ),
        retention=_sanitize_artifact_retention(
            artifact.retention,
            sanitizer,
        ),
        metadata=_sanitize_metadata(
            artifact.metadata,
            sanitizer,
            PrivacyField.OUTPUT,
        ),
    )


def _sanitize_artifact_ref(
    ref: TaskArtifactRef,
    sanitizer: PrivacySanitizer,
    field: PrivacyField = PrivacyField.OUTPUT,
) -> TaskArtifactRef:
    if not ref.metadata:
        return ref
    return TaskArtifactRef(
        artifact_id=ref.artifact_id,
        store=ref.store,
        storage_key=ref.storage_key,
        media_type=ref.media_type,
        size_bytes=ref.size_bytes,
        sha256=ref.sha256,
        metadata=_sanitize_metadata(ref.metadata, sanitizer, field),
    )


def _sanitize_artifact_provenance(
    provenance: TaskArtifactProvenance,
    sanitizer: PrivacySanitizer,
) -> TaskArtifactProvenance:
    if not provenance.metadata:
        return provenance
    return TaskArtifactProvenance(
        source_artifact_id=provenance.source_artifact_id,
        source_run_id=provenance.source_run_id,
        source_attempt_id=provenance.source_attempt_id,
        operation=provenance.operation,
        converter=provenance.converter,
        metadata=_sanitize_metadata(
            provenance.metadata,
            sanitizer,
            PrivacyField.OUTPUT,
        ),
    )


def _sanitize_artifact_retention(
    retention: TaskArtifactRetention,
    sanitizer: PrivacySanitizer,
) -> TaskArtifactRetention:
    if not retention.metadata:
        return retention
    return TaskArtifactRetention(
        expires_at=retention.expires_at,
        delete_after_days=retention.delete_after_days,
        retain_metadata=retention.retain_metadata,
        metadata=_sanitize_metadata(
            retention.metadata,
            sanitizer,
            PrivacyField.OUTPUT,
        ),
    )


def _sanitize_metadata(
    metadata: TaskSnapshotMetadata,
    sanitizer: PrivacySanitizer,
    field: PrivacyField,
) -> TaskSnapshotMetadata:
    if not metadata:
        return metadata
    safe_metadata = sanitizer.sanitize(field, metadata)
    assert isinstance(safe_metadata, Mapping)
    return freeze_snapshot_metadata(cast(Mapping[str, object], safe_metadata))


def _output_artifact_retention(
    definition: TaskDefinition,
    artifact: TaskOutputArtifact,
) -> TaskArtifactRetention:
    if (
        artifact.retention.expires_at is not None
        or artifact.retention.delete_after_days is not None
    ):
        return artifact.retention
    return TaskArtifactRetention(
        delete_after_days=definition.artifact.retention_days,
        retain_metadata=artifact.retention.retain_metadata,
        metadata=artifact.retention.metadata,
    )


def _file_descriptor_summary(file: TaskFileDescriptor) -> Mapping[str, object]:
    value: dict[str, object] = {"source_kind": file.source_kind.value}
    if file.provider_reference is not None:
        value["provider_reference"] = file.provider_reference.summary()
    else:
        value["reference"] = {"privacy": "<redacted>"}
    if file.role is not None:
        value["role"] = file.role
    if file.mime_type is not None:
        value["mime_type"] = file.mime_type
    if file.size_bytes is not None:
        value["size_bytes"] = file.size_bytes
    if file.sha256 is not None:
        value["sha256"] = file.sha256
    if file.conversions:
        value["conversions"] = tuple(
            {
                "name": conversion.name,
                "options": {"privacy": "<redacted>"},
            }
            for conversion in file.conversions
        )
    if file.metadata:
        value["metadata"] = {"privacy": "<redacted>"}
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
