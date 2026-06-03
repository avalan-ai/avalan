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
from .context import TaskInputFile, TaskTargetContext
from .converters import (
    FileConverter,
    TaskFileConversionError,
    convert_task_artifact,
)
from .converters.text import TextFileConverter
from .definition import (
    ObservabilitySinkType,
    RunMode,
    TaskDefinition,
    TaskInputType,
    TaskOutputType,
)
from .error import TaskError, TaskErrorValue, classify_task_error
from .input import TaskFileDescriptor
from .materialization import (
    TaskMaterializedFile,
    materialize_task_input_files,
    task_file_descriptors_from_input,
    task_provider_reference_input_files_from_input,
)
from .observability import (
    ObservabilitySink,
    TaskEventPipeline,
    TaskSanitizedEventObserver,
    record_observability_usage,
)
from .privacy import (
    EncryptionProvider,
    HmacProvider,
    PrivacyField,
    PrivacySafeValue,
    PrivacySanitizationError,
    PrivacySanitizer,
)
from .state import TaskAttemptState, TaskRunState
from .store import (
    TaskAttempt,
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
    TaskTargetRunner,
    TaskValidationContext,
)
from .usage import UsageRecord, usage_observation_from_response
from .validation import (
    TaskValidationCategory,
    TaskValidationError,
    TaskValidationIssue,
    validate_task_definition,
    validate_task_input,
    validate_task_output,
)

from asyncio import CancelledError, wait_for
from asyncio import sleep as asyncio_sleep
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
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
    output: object = None


class TaskRunFinalizer:
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
        definition_hash: Callable[[TaskDefinition], str] | None = None,
        execution_roots: Iterable[str | Path] = (),
        metrics_event_observer: TaskSanitizedEventObserver | None = None,
        trace_event_observer: TaskSanitizedEventObserver | None = None,
        observability_sink: ObservabilitySink | None = None,
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
        self._metrics_event_observer = metrics_event_observer
        self._trace_event_observer = trace_event_observer
        self._observability_sink = observability_sink
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
        self._validate(definition, input_value)
        await self._validate_target(definition)
        sanitizer = self._sanitizer(definition)
        definition_id = self._definition_hash(definition)
        await self._store.register_definition(
            definition,
            definition_hash=definition_id,
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
                metadata=freeze_snapshot_metadata(metadata),
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
                roots=self._execution_roots,
                artifact_store=self._artifact_store,
                hmac_provider=self._hmac_provider,
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
        files = (
            *files,
            *provider_reference_files,
            *input_files,
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
                (
                    lambda response: self._record_usage(
                        response,
                        definition=definition,
                        run=run,
                        attempt=attempt,
                    )
                )
                if definition.observability.metrics
                else None
            ),
            artifact_store=self._artifact_store,
        )
        await self._check_cancellation_or_expiry(run, expires_at)
        output = await wait_for(
            self._target.run(context),
            timeout=definition.run.timeout_seconds,
        )
        await self._check_cancellation_or_expiry(run, expires_at)
        output_issues = validate_task_output(definition, output)
        if output_issues:
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
            output=output,
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
            )
        )
        issues.extend(validate_task_input(definition, input_value))
        if issues:
            raise TaskValidationError(tuple(issues))

    async def _validate_target(self, definition: TaskDefinition) -> None:
        issues = await self._target.validate_definition(
            definition,
            TaskValidationContext(
                execution_roots=tuple(
                    Path(root) for root in self._execution_roots
                ),
            ),
        )
        if issues:
            raise TaskValidationError(issues)

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
    ) -> None:
        observation = usage_observation_from_response(response)
        if observation is None:
            return
        usage_record: UsageRecord | None = None
        try:
            usage_record = await self._store.append_usage(
                run.run_id,
                attempt_id=attempt.attempt_id,
                source=observation.source,
                totals=observation.totals,
            )
        except Exception:
            pass
        await record_observability_usage(
            self._observability_sink_for(definition),
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            source=observation.source,
            totals=observation.totals,
            record=usage_record,
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
        for artifact in _output_artifacts_from_output(definition, output):
            safe_artifact = _sanitize_output_artifact(
                artifact,
                sanitizer,
            )
            await self._store.append_artifact(
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

    async def _input_files_from_materialized(
        self,
        definition: TaskDefinition,
        materialized_files: tuple[TaskMaterializedFile, ...],
        *,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> tuple[TaskInputFile, ...]:
        input_files: list[TaskInputFile] = []
        for index, file in enumerate(materialized_files):
            input_files.append(
                (
                    await self._converted_input_file(
                        definition,
                        file,
                        index=index,
                        run=run,
                        attempt=attempt,
                    )
                )
                or file.as_input_file()
            )
        return tuple(input_files)

    async def _converted_input_file(
        self,
        definition: TaskDefinition,
        file: TaskMaterializedFile,
        *,
        index: int,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> TaskInputFile | None:
        source_ref = file.ref
        converted = False
        for conversion_index, request in enumerate(
            file.descriptor.conversions
        ):
            if request.name in {"native", "none"}:
                continue
            converter = self._file_converters.get(request.name)
            if converter is None:
                raise TaskValidationError(
                    (
                        _conversion_issue(
                            index=index,
                            conversion_index=conversion_index,
                            is_array=(
                                definition.input.type
                                == TaskInputType.FILE_ARRAY
                            ),
                        ),
                    )
                )
            try:
                converted_artifact = await convert_task_artifact(
                    source_ref,
                    request,
                    converter=converter,
                    artifact_store=cast(ArtifactStore, self._artifact_store),
                    task_store=self._store,
                    run_id=run.run_id,
                    attempt_id=attempt.attempt_id,
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
                            is_array=(
                                definition.input.type
                                == TaskInputType.FILE_ARRAY
                            ),
                        ),
                    )
                ) from error
            source_ref = converted_artifact.ref
            converted = True
        if not converted:
            return None
        return TaskInputFile(
            logical_path=f"artifact:{source_ref.artifact_id}",
            artifact_ref=source_ref,
            media_type=source_ref.media_type,
            size_bytes=source_ref.size_bytes,
            metadata=file.identity,
        )


def _snapshot_value(value: PrivacySafeValue) -> TaskSnapshotValue:
    return freeze_snapshot_value(value)


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
    values: dict[str, FileConverter] = {"text": TextFileConverter()}
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
        value["reference"] = file.reference
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
                "options": conversion.options,
            }
            for conversion in file.conversions
        )
    if file.metadata:
        value["metadata"] = file.metadata
    return value


def _utc_now() -> datetime:
    return datetime.now(UTC)
