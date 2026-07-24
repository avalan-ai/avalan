from ..container import ContainerAsyncBackend
from ..types import (
    assert_non_empty_string,
    assert_optional_non_negative_int,
    assert_optional_non_negative_number,
    assert_positive_number,
)
from .artifact import (
    ArtifactStore,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRetention,
)
from .canonical import spec_hash
from .container import (
    task_container_input_mount_manifest,
    task_container_run_metadata,
)
from .context import TaskInputFile
from .converters import FileConverter
from .definition import (
    ObservabilitySinkType,
    PrivacyAction,
    RunMode,
    TaskDefinition,
    TaskInputType,
)
from .event import SanitizedTaskEvent
from .idempotency import task_idempotency_identity
from .input import (
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskProviderReferenceKind,
    TaskRemoteUrlPolicy,
)
from .materialization import (
    TaskMaterializedFile,
    TaskRemoteUrlHttpClient,
    TaskRemoteUrlResolver,
    materialize_task_input_files,
    task_provider_reference_input_files_from_input,
)
from .observability import ObservabilitySink, TaskSanitizedEventObserver
from .privacy import (
    ENCRYPTED_MARKER,
    REDACTED_MARKER,
    EncryptionProvider,
    HmacProvider,
    PrivacyField,
    PrivacySafeValue,
    PrivacySanitizationError,
    PrivacySanitizer,
)
from .queue import (
    TaskQueue,
    TaskQueueArtifact,
    TaskQueueCompletion,
    TaskQueueItemState,
    TaskQueueSubmission,
)
from .runner import (
    DirectTaskRunner,
    TaskDirectTarget,
    TaskExecutableInputFileEntry,
    TaskRunResult,
    _file_converters,
    _input_summary_value,
    _sanitize_artifact_ref,
    _schema_resolution_issue,
    _snapshot_value,
    task_execution_file_entries_value,
    task_input_file_entries_for_queue,
    validate_explicit_task_input_files,
)
from .schema import (
    TaskSchemaResolutionError,
    resolve_task_definition_schemas,
    task_definition_schema_base_path,
)
from .skills import (
    TASK_SKILLS_METADATA_KEY,
    task_definition_with_skills_identity,
    task_skill_audit_event_publisher,
)
from .state import (
    TASK_RUN_TERMINAL_STATES,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskRunState,
)
from .store import (
    TaskAttempt,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskRun,
    TaskSnapshotMetadata,
    TaskSnapshotValue,
    TaskStore,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
)
from .target import (
    CallableTaskTargetRunner,
    TaskTargetRunner,
    TaskValidationContext,
)
from .usage import (
    UsageRecord,
    UsageSource,
    UsageTotals,
    aggregate_usage_totals,
)
from .validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    deduplicate_task_validation_issues,
    validate_task_definition,
    validate_task_input,
)

from asyncio import gather
from asyncio import sleep as asyncio_sleep
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from inspect import isawaitable
from math import isfinite
from pathlib import Path
from typing import Protocol, cast


class TaskClientUnsupportedOperationError(RuntimeError):
    code: str
    operation: str

    def __init__(self, *, code: str, operation: str, message: str) -> None:
        self.code = code
        self.operation = operation
        super().__init__(message)


class TaskClientWaitTimeoutError(TimeoutError):
    code: str
    operation: str
    run_id: str

    def __init__(self, *, run_id: str) -> None:
        self.code = "task.wait_timeout"
        self.operation = "wait"
        self.run_id = run_id
        super().__init__("Task run did not reach a terminal state in time.")


class TaskClientDurableLifecycleCoordinator(Protocol):
    """Converge durable interaction and task lifecycle state atomically."""

    async def cancel_input_required_task(
        self,
        *,
        task_run_id: str,
        now: datetime,
        metadata: Mapping[str, object],
    ) -> TaskQueueCompletion:
        """Cancel one durable input-required task atomically."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientValidationResult:
    issues: tuple[TaskValidationIssue, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.issues

    def as_dict(self) -> TaskSnapshotValue:
        return freeze_snapshot_value(
            {
                "valid": self.valid,
                "issues": tuple(issue.as_dict() for issue in self.issues),
            }
        )

    def raise_for_issues(self) -> None:
        if self.issues:
            raise TaskValidationError(self.issues)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientOutput:
    run_id: str
    state: TaskRunState
    output_summary: TaskSnapshotValue = None
    error: TaskSnapshotValue = None
    input_required: TaskSnapshotValue = None

    @property
    def ready(self) -> bool:
        return self.state == TaskRunState.SUCCEEDED

    @property
    def waiting_for_input(self) -> bool:
        return self.state == TaskRunState.INPUT_REQUIRED

    def as_dict(self) -> TaskSnapshotValue:
        value: dict[str, object] = {
            "run_id": self.run_id,
            "state": self.state.value,
            "ready": self.ready,
            "waiting_for_input": self.waiting_for_input,
        }
        if self.output_summary is not None:
            value["output_summary"] = self.output_summary
        if self.error is not None:
            value["error"] = self.error
        if self.input_required is not None:
            value["input_required"] = self.input_required
        return freeze_snapshot_value(value)


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientInspection:
    run: TaskRun
    attempts: tuple[TaskAttempt, ...]
    output: TaskClientOutput
    events: tuple[SanitizedTaskEvent, ...]
    usage: tuple[UsageRecord, ...]
    usage_totals: UsageTotals
    artifacts: tuple[TaskSnapshotValue, ...]

    def as_dict(self) -> TaskSnapshotValue:
        return freeze_snapshot_value(
            {
                "run": _run_inspection_value(self.run),
                "attempts": tuple(
                    _attempt_inspection_value(attempt)
                    for attempt in self.attempts
                ),
                "output": self.output.as_dict(),
                "events": tuple(
                    _event_inspection_value(event) for event in self.events
                ),
                "usage": tuple(
                    _usage_inspection_value(record) for record in self.usage
                ),
                "usage_totals": _usage_totals_value(self.usage_totals),
                "artifacts": self.artifacts,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskClientUsageInspection:
    usage: tuple[UsageRecord, ...]
    usage_totals: UsageTotals

    def as_dict(self) -> TaskSnapshotValue:
        return freeze_snapshot_value(
            {
                "usage": tuple(
                    _usage_inspection_value(record) for record in self.usage
                ),
                "usage_totals": _usage_totals_value(self.usage_totals),
            }
        )


class TaskClient:
    def __init__(
        self,
        store: TaskStore,
        *,
        target: TaskDirectTarget | TaskTargetRunner,
        queue: TaskQueue | None = None,
        hmac_provider: HmacProvider | None = None,
        encryption_provider: EncryptionProvider | None = None,
        raw_storage_allowed: bool = False,
        artifact_store: ArtifactStore | None = None,
        file_converters: Mapping[str, FileConverter] | None = None,
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
        durable_lifecycle_coordinator: (
            TaskClientDurableLifecycleCoordinator | None
        ) = None,
        clock: Callable[[], datetime] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._store = store
        self._target = _target_runner(target)
        self._queue = queue
        self._hmac_provider = hmac_provider
        self._encryption_provider = encryption_provider
        self._raw_storage_allowed = raw_storage_allowed
        self._artifact_store = artifact_store
        self._file_converters = _file_converters(file_converters)
        self._definition_hash = definition_hash
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
        self._durable_lifecycle_coordinator = durable_lifecycle_coordinator
        self._clock = clock or _utc_now
        self._sleep = sleep or asyncio_sleep

    @staticmethod
    def file_conversion(
        name: str,
        *,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionRequest:
        return TaskFileConversionRequest(name=name, options=options or {})

    @staticmethod
    def local_file(
        reference: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[
            str | TaskFileConversionRequest | Mapping[str, object]
        ] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return TaskFileDescriptor.local_path(
            reference,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=_file_conversion_requests(conversions),
            metadata=metadata,
        )

    @staticmethod
    def remote_url_file(
        url: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[
            str | TaskFileConversionRequest | Mapping[str, object]
        ] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return TaskFileDescriptor.remote_url(
            url,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=_file_conversion_requests(conversions),
            metadata=metadata,
        )

    @staticmethod
    def artifact_file(
        artifact_id: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[
            str | TaskFileConversionRequest | Mapping[str, object]
        ] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return TaskFileDescriptor.artifact(
            artifact_id,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=_file_conversion_requests(conversions),
            metadata=metadata,
        )

    @staticmethod
    def inline_file(
        encoded_bytes: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        conversions: Iterable[
            str | TaskFileConversionRequest | Mapping[str, object]
        ] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return TaskFileDescriptor.inline_bytes(
            encoded_bytes,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            conversions=_file_conversion_requests(conversions),
            metadata=metadata,
        )

    @staticmethod
    def provider_file_id(
        provider: str,
        file_id: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        size_bucket: str | None = None,
        identity_hmac: str | None = None,
        owner_scope: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return _provider_reference_descriptor(
            provider,
            file_id,
            TaskProviderReferenceKind.PROVIDER_FILE_ID,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            size_bucket=size_bucket,
            identity_hmac=identity_hmac,
            owner_scope=owner_scope,
            metadata=metadata,
        )

    @staticmethod
    def hosted_url(
        provider: str,
        url: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        size_bucket: str | None = None,
        identity_hmac: str | None = None,
        owner_scope: str | None = None,
        expires_at: datetime | None = None,
        durable: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return _provider_reference_descriptor(
            provider,
            url,
            TaskProviderReferenceKind.HOSTED_URL,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            size_bucket=size_bucket,
            identity_hmac=identity_hmac,
            owner_scope=owner_scope,
            expires_at=expires_at,
            durable=durable,
            metadata=metadata,
        )

    @staticmethod
    def object_store_uri(
        provider: str,
        uri: str,
        *,
        role: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        sha256: str | None = None,
        size_bucket: str | None = None,
        identity_hmac: str | None = None,
        owner_scope: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskFileDescriptor:
        return _provider_reference_descriptor(
            provider,
            uri,
            TaskProviderReferenceKind.OBJECT_STORE_URI,
            role=role,
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            size_bucket=size_bucket,
            identity_hmac=identity_hmac,
            owner_scope=owner_scope,
            metadata=metadata,
        )

    async def validate(
        self,
        definition: TaskDefinition,
        *,
        input_value: object = None,
    ) -> TaskClientValidationResult:
        assert isinstance(definition, TaskDefinition)
        issues: list[TaskValidationIssue] = []
        schema_base_path = task_definition_schema_base_path(definition)
        try:
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
                    observability_sink=(
                        self._observability_sink
                        if definition.observability.sinks
                        != (ObservabilitySinkType.NOOP,)
                        else None
                    ),
                ),
                schema_base_path=schema_base_path,
            )
        except TaskValidationError as error:
            issues.extend(error.issues)
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
        target_definition = await self._target_validation_definition(
            definition
        )
        issues.extend(
            await self._target.validate_definition(
                target_definition,
                TaskValidationContext(
                    execution_roots=tuple(
                        Path(root) for root in self._execution_roots
                    ),
                    artifact_store=self._artifact_store,
                    task_store=self._store,
                    file_converters=self._file_converters,
                ),
            )
        )
        return TaskClientValidationResult(
            issues=deduplicate_task_validation_issues(issues)
        )

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
        available_at: datetime | None = None,
        idempotency_expires_at: datetime | None = None,
        idempotency_window: object = None,
        owner_scope: object = "default",
        queue_name: str | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        assert isinstance(definition, TaskDefinition)
        assert isinstance(files, tuple)
        if queue_name is not None:
            assert_non_empty_string(queue_name, "queue_name")
        if available_at is not None:
            assert isinstance(available_at, datetime)
        if idempotency_expires_at is not None:
            assert isinstance(idempotency_expires_at, datetime)
        if definition.run.mode != RunMode.QUEUE:
            raise _unsupported_queue_operation("enqueue")
        if self._queue is None:
            raise _unsupported_queue_operation("enqueue")
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
                observability_sink=(
                    self._observability_sink
                    if definition.observability.sinks
                    != (ObservabilitySinkType.NOOP,)
                    else None
                ),
            ),
            schema_base_path=schema_base_path,
        )
        validation = await self.validate(definition, input_value=input_value)
        validation.raise_for_issues()
        sanitizer = self._sanitizer(definition)
        definition_id = await self._definition_hash_value(definition)
        await self._store.register_definition(
            definition,
            definition_hash=definition_id,
        )
        provider_reference_files = (
            task_provider_reference_input_files_from_input(
                definition,
                input_value,
                now=self._clock(),
            )
        )
        file_issues = validate_explicit_task_input_files(
            files,
            now=self._clock(),
        )
        if file_issues:
            raise TaskValidationError(file_issues)
        materialized_files = await materialize_task_input_files(
            definition,
            input_value,
            roots=self._input_roots,
            artifact_store=self._artifact_store,
            hmac_provider=self._hmac_provider,
            remote_url_policy=self._remote_url_policy,
            remote_url_http_client=self._remote_url_http_client,
            remote_url_resolver=self._remote_url_resolver,
        )
        input_files = tuple(
            materialized_file.as_input_file()
            for materialized_file in materialized_files
        )
        file_entries = task_input_file_entries_for_queue(
            files=files,
            provider_reference_files=provider_reference_files,
            materialized_files=materialized_files,
        )
        queued_files = tuple(entry.file for entry in file_entries)
        assert queued_files == (
            *files,
            *provider_reference_files,
            *input_files,
        )
        input_payload = self._queue_input_payload(
            definition,
            input_value,
            file_entries=file_entries,
            sanitizer=sanitizer,
        )
        explicit_artifacts = self._explicit_queue_artifacts(
            definition,
            sanitizer,
            (*files, *provider_reference_files),
        )
        safe_queue_metadata = _queue_metadata_snapshot(
            queue_metadata,
            input_value=input_value,
            idempotency_key=idempotency_key,
            owner_scope=owner_scope,
        )
        selected_queue_name = queue_name or definition.run.queue
        assert selected_queue_name is not None
        input_summary_value = _input_summary_value(definition, input_value)
        try:
            idempotency = task_idempotency_identity(
                definition,
                definition_hash=definition_id,
                input_value=input_summary_value,
                files=queued_files,
                owner_scope=owner_scope,
                hmac_provider=cast(HmacProvider, self._hmac_provider),
                window=idempotency_key or idempotency_window,
            )
            return await self._queue.enqueue_run(
                TaskExecutionRequest(
                    definition_id=definition_id,
                    input_summary=_snapshot_value(
                        sanitizer.sanitize(
                            PrivacyField.INPUT,
                            input_summary_value,
                        )
                    ),
                    input_payload=input_payload,
                    file_summaries=self._file_summaries(
                        sanitizer,
                        queued_files,
                    ),
                    idempotency_key=(
                        idempotency.identity_key if idempotency else None
                    ),
                    queue=selected_queue_name,
                    metadata=freeze_snapshot_metadata(
                        _task_run_metadata_with_skills(
                            definition,
                            metadata,
                            input_mounts=task_container_input_mount_manifest(
                                queued_files,
                                allowed_roots=tuple(
                                    Path(root) for root in self._input_roots
                                ),
                            ),
                        )
                    ),
                ),
                queue_name=selected_queue_name,
                priority=definition.run.priority or 0,
                available_at=available_at,
                idempotency=idempotency,
                idempotency_expires_at=idempotency_expires_at,
                artifacts=(
                    *explicit_artifacts,
                    *self._queue_artifacts(definition, materialized_files),
                ),
                run_metadata={"runner": "queue"},
                queue_metadata=safe_queue_metadata,
            )
        except BaseException:
            await self._delete_materialized_files(materialized_files)
            raise

    async def wait(
        self,
        run_id: str,
        *,
        timeout_seconds: float | None = None,
        poll_interval_seconds: float = 1.0,
    ) -> TaskClientOutput:
        assert isinstance(run_id, str) and run_id.strip()
        assert_optional_non_negative_number(
            timeout_seconds,
            "timeout_seconds",
        )
        assert_positive_number(
            poll_interval_seconds,
            "poll_interval_seconds",
        )
        deadline = (
            self._clock() + timedelta(seconds=timeout_seconds)
            if timeout_seconds is not None
            else None
        )
        while True:
            output = await self.output(run_id)
            if (
                output.state in TASK_RUN_TERMINAL_STATES
                or output.state == TaskRunState.INPUT_REQUIRED
            ):
                return output
            if deadline is not None:
                now = self._clock()
                remaining = (deadline - now).total_seconds()
                if remaining <= 0:
                    raise TaskClientWaitTimeoutError(run_id=run_id)
                await self._sleep(min(poll_interval_seconds, remaining))
            else:
                await self._sleep(poll_interval_seconds)

    async def cancel(self, run_id: str) -> TaskRun:
        assert isinstance(run_id, str) and run_id.strip()
        run = await self._store.get_run(run_id)
        if run.state in TASK_RUN_TERMINAL_STATES:
            return run
        if run.state == TaskRunState.CANCEL_REQUESTED:
            return run
        if run.state == TaskRunState.INPUT_REQUIRED:
            assert run.last_attempt_id is not None
            attempt = await self._store.get_attempt(run.last_attempt_id)
            segments = await self._store.list_attempt_segments(
                attempt.attempt_id
            )
            suspended_segment = (
                segments[-1]
                if segments
                and segments[-1].state is TaskAttemptSegmentState.SUSPENDED
                else None
            )
            if (
                suspended_segment is not None
                and suspended_segment.checkpoint_id is not None
            ):
                coordinator = self._durable_lifecycle_coordinator
                if coordinator is None:
                    raise _unsupported_durable_cancel_operation()
                completion = await coordinator.cancel_input_required_task(
                    task_run_id=run.run_id,
                    now=self._clock(),
                    metadata={"source": "task_client"},
                )
                if (
                    completion.run.run_id != run.run_id
                    or completion.run.state is not TaskRunState.CANCELLED
                    or completion.attempt.attempt_id != attempt.attempt_id
                    or completion.attempt.state
                    is not TaskAttemptState.ABANDONED
                    or completion.queue_item.run_id != run.run_id
                    or completion.queue_item.state
                    is not TaskQueueItemState.DEAD
                ):
                    raise _invalid_durable_cancel_completion()
                return completion.run
            if attempt.state == TaskAttemptState.SUSPENDED:
                await self._store.transition_attempt(
                    attempt.attempt_id,
                    from_states={TaskAttemptState.SUSPENDED},
                    to_state=TaskAttemptState.ABANDONED,
                    reason="cancelled_while_input_required",
                )
            run = await self._store.transition_run(
                run.run_id,
                from_states={TaskRunState.INPUT_REQUIRED},
                to_state=TaskRunState.CANCEL_REQUESTED,
                reason="cancel_requested",
            )
            return await self._store.transition_run(
                run.run_id,
                from_states={TaskRunState.CANCEL_REQUESTED},
                to_state=TaskRunState.CANCELLED,
                reason="cancelled_while_input_required",
            )
        if run.state not in {
            TaskRunState.QUEUED,
            TaskRunState.RUNNING,
        }:
            raise _unsupported_cancel_operation()
        return await self._store.transition_run(
            run.run_id,
            from_states={run.state},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="cancel_requested",
        )

    async def output(self, run_id: str) -> TaskClientOutput:
        run = await self._store.get_run(run_id)
        result = run.result
        return TaskClientOutput(
            run_id=run.run_id,
            state=run.state,
            output_summary=result.output_summary if result else None,
            error=result.error if result else None,
            input_required=(
                result.metadata.get("interaction")
                if result is not None
                else None
            ),
        )

    async def events(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        after_sequence: int | None = None,
    ) -> tuple[SanitizedTaskEvent, ...]:
        assert_optional_non_negative_int(after_sequence, "after_sequence")
        return await self._store.list_events(
            run_id,
            attempt_id=attempt_id,
            after_sequence=after_sequence,
        )

    async def usage(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        source: UsageSource | None = None,
    ) -> tuple[UsageRecord, ...]:
        if source is not None:
            assert isinstance(source, UsageSource)
        return await self._store.list_usage(
            run_id,
            attempt_id=attempt_id,
            source=source,
        )

    async def usage_totals(
        self,
        run_id: str,
        *,
        source: UsageSource | None = None,
    ) -> UsageTotals:
        if source is not None:
            assert isinstance(source, UsageSource)
        return await self._store.usage_totals(run_id, source=source)

    async def usage_inspection(
        self,
        run_id: str,
        *,
        attempt_id: str | None = None,
        source: UsageSource | None = None,
    ) -> TaskClientUsageInspection:
        records = await self.usage(
            run_id,
            attempt_id=attempt_id,
            source=source,
        )
        return TaskClientUsageInspection(
            usage=records,
            usage_totals=aggregate_usage_totals(records),
        )

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

    async def inspect(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> TaskClientInspection:
        return TaskClientInspection(
            run=await self._store.get_run(run_id),
            attempts=await self._store.list_attempts(run_id),
            output=await self.output(run_id),
            events=await self.events(run_id, after_sequence=after_sequence),
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
            file_converters=self._file_converters,
            definition_hash=self._definition_hash,
            execution_roots=self._execution_roots,
            remote_url_policy=self._remote_url_policy,
            remote_url_http_client=self._remote_url_http_client,
            remote_url_resolver=self._remote_url_resolver,
            event_observer=self._event_observer,
            metrics_event_observer=self._metrics_event_observer,
            trace_event_observer=self._trace_event_observer,
            observability_sink=self._observability_sink,
            container_backend=self._container_backend,
            clock=self._clock,
            sleep=self._sleep,
            input_roots=self._input_roots,
        )

    async def _definition_hash_value(self, definition: TaskDefinition) -> str:
        value = (
            self._definition_hash(definition)
            if self._definition_hash is not None
            else _default_definition_hash(definition)
        )
        if isawaitable(value):
            value = await value
        assert isinstance(value, str) and value.strip()
        return value

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

    async def _target_validation_definition(
        self,
        definition: TaskDefinition,
    ) -> TaskDefinition:
        try:
            return await resolve_task_definition_schemas(
                definition,
                schema_base_path=None,
            )
        except TaskSchemaResolutionError:
            return definition

    def _sanitizer(self, definition: TaskDefinition) -> PrivacySanitizer:
        return PrivacySanitizer(
            definition.privacy,
            hmac_provider=self._hmac_provider,
            encryption_provider=self._encryption_provider,
            raw_storage_allowed=self._raw_storage_allowed,
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

    def _queue_input_payload(
        self,
        definition: TaskDefinition,
        input_value: object,
        *,
        file_entries: tuple[TaskExecutableInputFileEntry, ...],
        sanitizer: PrivacySanitizer,
    ) -> TaskExecutionPayload | None:
        input_required = _queue_input_payload_required(
            definition,
            input_value,
        )
        if not input_required and not file_entries:
            return None
        issues = _queue_input_payload_issues(
            definition,
            raw_storage_allowed=self._raw_storage_allowed,
            encryption_provider=self._encryption_provider,
        )
        if issues:
            raise TaskValidationError(issues)
        encrypted_input: PrivacySafeValue = None
        if input_required:
            try:
                encrypted_input = sanitizer.sanitize_with_action(
                    PrivacyAction.ENCRYPT,
                    input_value,
                )
            except PrivacySanitizationError as error:
                raise TaskValidationError(
                    (
                        _queue_input_payload_issue(
                            path="input",
                            message=(
                                "Queued task input cannot be safely stored "
                                "for worker execution."
                            ),
                            hint=(
                                "Use JSON-compatible input or a durable file "
                                "or provider reference."
                            ),
                        ),
                    )
                ) from error
        try:
            encrypted_files = tuple(
                _snapshot_value(
                    sanitizer.sanitize_with_action(
                        PrivacyAction.ENCRYPT,
                        entry,
                    )
                )
                for entry in task_execution_file_entries_value(file_entries)
            )
        except PrivacySanitizationError as error:
            raise TaskValidationError(
                (
                    _queue_input_payload_issue(
                        path="files",
                        message=(
                            "Queued task files cannot be safely stored for "
                            "worker execution."
                        ),
                        hint=(
                            "Use durable artifacts or provider references "
                            "with encrypted payload storage enabled."
                        ),
                    ),
                )
            ) from error
        return TaskExecutionPayload(
            file_values=encrypted_files,
            input_value=_snapshot_value(encrypted_input),
        )

    def _queue_artifacts(
        self,
        definition: TaskDefinition,
        files: tuple[TaskMaterializedFile, ...],
    ) -> tuple[TaskQueueArtifact, ...]:
        return tuple(
            TaskQueueArtifact(
                ref=file.ref,
                purpose=TaskArtifactPurpose.INPUT,
                provenance=TaskArtifactProvenance(
                    operation="materialization",
                    metadata={
                        "descriptor": _materialized_file_descriptor_metadata(
                            file,
                            index=index,
                        ),
                        "identity": file.identity,
                        "source_kind": file.descriptor.source_kind.value,
                    },
                ),
                retention=TaskArtifactRetention(
                    delete_after_days=definition.artifact.retention_days,
                ),
                metadata={
                    "descriptor": _materialized_file_descriptor_metadata(
                        file,
                        index=index,
                    ),
                    "identity": file.identity,
                },
            )
            for index, file in enumerate(files)
        )

    async def _delete_materialized_files(
        self,
        files: tuple[TaskMaterializedFile, ...],
    ) -> None:
        assert self._artifact_store is not None
        await gather(
            *(self._artifact_store.delete(file.ref) for file in files)
        )

    def _explicit_queue_artifacts(
        self,
        definition: TaskDefinition,
        sanitizer: PrivacySanitizer,
        files: tuple[TaskInputFile, ...],
    ) -> tuple[TaskQueueArtifact, ...]:
        issues: list[TaskValidationIssue] = []
        artifacts: list[TaskQueueArtifact] = []
        for index, file in enumerate(files):
            if file.artifact_ref is None:
                if (
                    file.provider_reference is not None
                    and file.provider_reference.durable_for_queue
                ):
                    continue
                issue_path = (
                    f"files[{index}].provider_reference"
                    if file.provider_reference is not None
                    else f"files[{index}].artifact_ref"
                )
                issues.append(
                    TaskValidationIssue(
                        code="input.invalid_file",
                        path=issue_path,
                        message=(
                            "Queued task file attachments require a durable "
                            "artifact or provider reference."
                        ),
                        hint=(
                            "Store file bytes in an artifact backend or use "
                            "a non-expiring provider reference."
                        ),
                        category=TaskValidationCategory.UNSUPPORTED,
                    )
                )
                continue
            metadata = _snapshot_value(
                sanitizer.sanitize(PrivacyField.FILES, file.summary())
            )
            assert isinstance(metadata, Mapping)
            artifacts.append(
                TaskQueueArtifact(
                    ref=_sanitize_artifact_ref(
                        file.artifact_ref,
                        sanitizer,
                        PrivacyField.FILES,
                    ),
                    purpose=TaskArtifactPurpose.INPUT,
                    provenance=TaskArtifactProvenance(
                        operation="client_attachment",
                        metadata={"file": metadata},
                    ),
                    retention=TaskArtifactRetention(
                        delete_after_days=definition.artifact.retention_days,
                    ),
                    metadata=metadata,
                )
            )
        if issues:
            raise TaskValidationError(tuple(issues))
        return tuple(artifacts)


def _queue_input_payload_required(
    definition: TaskDefinition,
    input_value: object,
) -> bool:
    return input_value is not None and definition.input.type not in {
        TaskInputType.FILE,
        TaskInputType.FILE_ARRAY,
    }


def _task_run_metadata_with_skills(
    definition: TaskDefinition,
    metadata: Mapping[str, object] | None,
    *,
    input_mounts: tuple[dict[str, object], ...],
) -> Mapping[str, object]:
    value = dict(
        task_container_run_metadata(
            definition,
            metadata,
            input_mounts=input_mounts,
        )
    )
    if definition.skills_identity is not None:
        value[TASK_SKILLS_METADATA_KEY] = definition.skills_identity
    return value


def _queue_input_payload_issues(
    definition: TaskDefinition,
    *,
    raw_storage_allowed: bool,
    encryption_provider: EncryptionProvider | None,
) -> tuple[TaskValidationIssue, ...]:
    issues: list[TaskValidationIssue] = []
    if definition.privacy.raw_retention_days <= 0:
        issues.append(
            _queue_input_payload_issue(
                path="privacy.raw_retention_days",
                message=(
                    "Queued task input requires encrypted raw retention for "
                    "worker execution."
                ),
                hint=(
                    "Set a positive raw retention period for queued scalar "
                    "or structured inputs."
                ),
            )
        )
    if not raw_storage_allowed:
        issues.append(
            _queue_input_payload_issue(
                path="privacy",
                message="Queued task input raw storage is not enabled.",
                hint=(
                    "Enable raw storage in runtime configuration for queued "
                    "scalar or structured inputs."
                ),
            )
        )
    if encryption_provider is None:
        issues.append(
            _queue_input_payload_issue(
                path="privacy.input",
                code="privacy.encryption_key_missing",
                message=(
                    "Queued task input requires an encryption key provider."
                ),
                hint=(
                    "Configure a task encryption provider before queueing "
                    "scalar or structured inputs."
                ),
            )
        )
    return tuple(issues)


def _queue_input_payload_issue(
    *,
    path: str,
    message: str,
    hint: str,
    code: str = "queue.input_payload_unavailable",
) -> TaskValidationIssue:
    return TaskValidationIssue(
        code=code,
        path=path,
        message=message,
        hint=hint,
        category=TaskValidationCategory.PRIVACY,
    )


_SENSITIVE_QUEUE_METADATA_TOKENS = frozenset(
    {
        "api",
        "authorization",
        "bytes",
        "content",
        "file",
        "files",
        "filename",
        "handle",
        "idempotency",
        "input",
        "key",
        "owner",
        "path",
        "prompt",
        "provider",
        "reference",
        "scope",
        "secret",
        "token",
        "uri",
        "url",
    }
)


def _queue_metadata_snapshot(
    metadata: Mapping[str, object] | None,
    *,
    input_value: object,
    idempotency_key: str | None,
    owner_scope: object,
) -> TaskSnapshotMetadata:
    if metadata is None:
        return freeze_snapshot_metadata(None)
    sensitive_values = _sensitive_queue_metadata_values(
        input_value,
        idempotency_key,
        owner_scope,
    )
    safe_metadata = _safe_queue_metadata_value(metadata, sensitive_values)
    assert isinstance(safe_metadata, Mapping)
    return freeze_snapshot_metadata(safe_metadata)


def _safe_queue_metadata_value(
    value: object,
    sensitive_values: frozenset[str],
) -> object:
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if isfinite(value) else _redacted_metadata_value()
    if isinstance(value, str):
        return (
            _redacted_metadata_value()
            if _contains_sensitive_metadata_value(value, sensitive_values)
            else value
        )
    if isinstance(value, Mapping):
        safe: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                safe["metadata"] = _redacted_metadata_value()
                continue
            safe[key] = (
                _redacted_metadata_value()
                if _queue_metadata_key_is_sensitive(key)
                else _safe_queue_metadata_value(item, sensitive_values)
            )
        return safe
    if isinstance(value, list | tuple):
        return tuple(
            _safe_queue_metadata_value(item, sensitive_values)
            for item in value
        )
    return _redacted_metadata_value()


def _queue_metadata_key_is_sensitive(key: str) -> bool:
    return bool(_metadata_key_tokens(key) & _SENSITIVE_QUEUE_METADATA_TOKENS)


def _metadata_key_tokens(key: str) -> frozenset[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in key.lower():
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return frozenset(tokens)


def _sensitive_queue_metadata_values(
    *values: object,
) -> frozenset[str]:
    sensitive: set[str] = set()
    for value in values:
        _collect_sensitive_queue_metadata_values(value, sensitive)
    return frozenset(sensitive)


def _collect_sensitive_queue_metadata_values(
    value: object,
    sensitive: set[str],
) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) >= 4:
            sensitive.add(stripped)
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _collect_sensitive_queue_metadata_values(item, sensitive)
        return
    if isinstance(value, list | tuple):
        for item in value:
            _collect_sensitive_queue_metadata_values(item, sensitive)


def _contains_sensitive_metadata_value(
    value: str,
    sensitive_values: frozenset[str],
) -> bool:
    return any(secret in value for secret in sensitive_values)


def _redacted_metadata_value() -> dict[str, str]:
    return {"privacy": REDACTED_MARKER}


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


def _unsupported_cancel_operation() -> TaskClientUnsupportedOperationError:
    return TaskClientUnsupportedOperationError(
        code="task.cancel_unavailable",
        operation="cancel",
        message="Task run cannot be cancelled from its current state.",
    )


def _unsupported_durable_cancel_operation() -> (
    TaskClientUnsupportedOperationError
):
    return TaskClientUnsupportedOperationError(
        code="task.durable_lifecycle_unavailable",
        operation="cancel",
        message=(
            "Durable input-required cancellation requires an atomic "
            "lifecycle coordinator."
        ),
    )


def _invalid_durable_cancel_completion() -> (
    TaskClientUnsupportedOperationError
):
    return TaskClientUnsupportedOperationError(
        code="task.durable_lifecycle_invalid",
        operation="cancel",
        message=(
            "Durable input-required cancellation returned an invalid "
            "task completion."
        ),
    )


async def _default_definition_hash(definition: TaskDefinition) -> str:
    return await spec_hash(definition)


def _file_conversion_requests(
    values: Iterable[str | TaskFileConversionRequest | Mapping[str, object]],
) -> tuple[TaskFileConversionRequest, ...]:
    return tuple(_file_conversion_request(value) for value in values)


def _file_conversion_request(
    value: str | TaskFileConversionRequest | Mapping[str, object],
) -> TaskFileConversionRequest:
    if isinstance(value, TaskFileConversionRequest):
        return value
    if isinstance(value, str):
        return TaskFileConversionRequest(name=value)
    name = value.get("name")
    assert_non_empty_string(cast(str, name), "conversion.name")
    options = value.get("options", {})
    assert isinstance(options, Mapping), "conversion.options must be a mapping"
    return TaskFileConversionRequest(
        name=cast(str, name),
        options=options,
    )


def _materialized_file_descriptor_metadata(
    file: TaskMaterializedFile,
    *,
    index: int,
) -> TaskSnapshotMetadata:
    descriptor = file.descriptor
    value: dict[str, object] = {
        "conversions": tuple(
            {"name": conversion.name} for conversion in descriptor.conversions
        ),
        "descriptor_path": file.descriptor_path,
        "file_order": index,
        "source_kind": descriptor.source_kind.value,
    }
    if descriptor.role is not None:
        value["role"] = descriptor.role
    if descriptor.mime_type is not None:
        value["mime_type"] = descriptor.mime_type
    if descriptor.size_bytes is not None:
        value["size_bytes"] = descriptor.size_bytes
    return freeze_snapshot_metadata(value)


def _provider_reference_descriptor(
    provider: str,
    reference: str,
    kind: TaskProviderReferenceKind,
    *,
    role: str | None = None,
    mime_type: str | None = None,
    size_bytes: int | None = None,
    sha256: str | None = None,
    size_bucket: str | None = None,
    identity_hmac: str | None = None,
    owner_scope: str | None = None,
    expires_at: datetime | None = None,
    durable: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> TaskFileDescriptor:
    return TaskFileDescriptor.provider_reference_descriptor(
        reference,
        kind=kind,
        provider=provider,
        role=role,
        mime_type=mime_type,
        size_bytes=size_bytes,
        sha256=sha256,
        size_bucket=size_bucket,
        identity_hmac=identity_hmac,
        owner_scope=owner_scope,
        expires_at=expires_at,
        durable=durable,
        metadata=metadata,
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _target_runner(
    target: TaskDirectTarget | TaskTargetRunner,
) -> TaskTargetRunner:
    run = getattr(target, "run", None)
    validate_definition = getattr(target, "validate_definition", None)
    if callable(run) and callable(validate_definition):
        return cast(TaskTargetRunner, target)
    return CallableTaskTargetRunner(cast(TaskDirectTarget, target))


def _run_inspection_value(run: TaskRun) -> dict[str, object]:
    value: dict[str, object] = {
        "run_id": run.run_id,
        "definition_id": run.definition_id,
        "state": run.state.value,
        "created_at": _datetime_value(run.created_at),
        "updated_at": _datetime_value(run.updated_at),
    }
    if run.request.input_summary is not None:
        value["input_summary"] = run.request.input_summary
    if run.request.input_payload is not None:
        value["input_payload"] = {"privacy": ENCRYPTED_MARKER}
    if run.request.file_summaries:
        value["file_summaries"] = run.request.file_summaries
    if run.request.queue is not None:
        value["queue"] = run.request.queue
    if run.last_attempt_id is not None:
        value["last_attempt_id"] = run.last_attempt_id
    if run.claim is not None:
        value["claim"] = {
            "worker_id": run.claim.worker_id,
            "claimed_at": _datetime_value(run.claim.claimed_at),
            "lease_expires_at": _datetime_value(run.claim.lease_expires_at),
            "heartbeat_at": (
                _datetime_value(run.claim.heartbeat_at)
                if run.claim.heartbeat_at is not None
                else None
            ),
        }
    if run.result is not None:
        value["result"] = _result_inspection_value(run.result)
    return value


def _attempt_inspection_value(attempt: TaskAttempt) -> dict[str, object]:
    value: dict[str, object] = {
        "attempt_id": attempt.attempt_id,
        "run_id": attempt.run_id,
        "attempt_number": attempt.attempt_number,
        "state": attempt.state.value,
        "created_at": _datetime_value(attempt.created_at),
        "updated_at": _datetime_value(attempt.updated_at),
    }
    if attempt.result is not None:
        value["result"] = _result_inspection_value(attempt.result)
    return value


def _result_inspection_value(result: TaskExecutionResult) -> dict[str, object]:
    value: dict[str, object] = {}
    if result.output_summary is not None:
        value["output_summary"] = result.output_summary
    if result.error is not None:
        value["error"] = result.error
    if result.metadata:
        value["metadata"] = result.metadata
    return value


def _event_inspection_value(event: SanitizedTaskEvent) -> dict[str, object]:
    value: dict[str, object] = {
        "event_id": event.event_id,
        "run_id": event.run_id,
        "sequence": event.sequence,
        "event_type": event.event_type,
        "category": event.category.value,
        "created_at": _datetime_value(event.created_at),
    }
    if event.attempt_id is not None:
        value["attempt_id"] = event.attempt_id
    if event.payload is not None:
        value["payload"] = event.payload
    return value


def _usage_inspection_value(record: UsageRecord) -> dict[str, object]:
    value: dict[str, object] = {
        "usage_id": record.usage_id,
        "run_id": record.run_id,
        "sequence": record.sequence,
        "source": record.source.value,
        "totals": _usage_totals_value(record.totals),
        "created_at": _datetime_value(record.created_at),
    }
    if record.attempt_id is not None:
        value["attempt_id"] = record.attempt_id
    if record.segment_id is not None:
        value["segment_id"] = record.segment_id
    if record.metadata:
        value["metadata"] = record.metadata
    return value


def _usage_totals_value(totals: UsageTotals) -> dict[str, object]:
    return {
        "input_tokens": totals.input_tokens,
        "cached_input_tokens": totals.cached_input_tokens,
        "cache_creation_input_tokens": totals.cache_creation_input_tokens,
        "output_tokens": totals.output_tokens,
        "reasoning_tokens": totals.reasoning_tokens,
        "total_tokens": totals.total_tokens,
    }


def _datetime_value(value: datetime) -> str:
    return value.isoformat()
