import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    DirectTaskRunner,
    HmacProvider,
    PrivacyAction,
    PrivacySanitizer,
    RetryBackoff,
    TaskArtifactPolicy,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
    TaskAttemptDecision,
    TaskAttemptDecisionType,
    TaskAttemptState,
    TaskDefinition,
    TaskDirectTarget,
    TaskErrorCode,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskRetryPolicy,
    TaskRunFinalizer,
    TaskRunnerError,
    TaskRunPolicy,
    TaskRunResult,
    TaskRunState,
    TaskStoreConflictError,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    safe_target_metadata,
    safe_target_value,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.runner import (
    _error_summary_with_attempt_policy,
    _input_summary_value,
)
from avalan.task.stores import InMemoryTaskStore


class StaticHmacProvider:
    def hmac_key(
        self,
        *,
        purpose: TaskKeyPurpose,
        key_id: str | None = None,
    ) -> TaskKeyMaterial:
        return TaskKeyMaterial(
            key_id=key_id or purpose.value,
            algorithm="hmac-sha256",
            secret=b"test-secret",
        )


class RecordingTarget:
    def __init__(self, output: object) -> None:
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def __call__(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        return self.output


class ArtifactOutputTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        assert context.artifact_store is not None
        return await context.artifact_store.put(
            b"private report bytes",
            media_type="text/plain",
            metadata={"name": "report"},
        )


class ArtifactRecordArrayTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        assert context.artifact_store is not None
        ref = await context.artifact_store.put(
            b"private stale bytes",
            media_type="text/plain",
            metadata={"name": "stale"},
        )
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        return [
            TaskArtifactRecord(
                artifact_id=ref.artifact_id,
                run_id=context.execution.run_id,
                attempt_id=context.execution.attempt_id,
                purpose=TaskArtifactPurpose.OUTPUT,
                state=TaskArtifactState.LOST,
                ref=ref,
                created_at=created_at,
                updated_at=created_at,
                provenance=TaskArtifactProvenance(operation="export"),
                retention=TaskArtifactRetention(delete_after_days=2),
                metadata={"state": "simulated"},
            )
        ]


class FailingTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        raise RuntimeError("raw secret failure")


class FlakyTarget:
    def __init__(self, failures: int, output: object) -> None:
        self.failures = failures
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def __call__(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        if len(self.contexts) <= self.failures:
            raise RuntimeError("raw transient failure")
        return self.output


class SlowTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        await asyncio.sleep(2)
        return "too late"


class CancelledTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        raise asyncio.CancelledError("raw cancellation reason")


class CancellingTarget:
    def __init__(self, store: InMemoryTaskStore) -> None:
        self.store = store

    async def __call__(self, context: TaskTargetContext) -> object:
        await self.store.transition_run(
            context.execution.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="test_cancel",
        )
        return "late success"


class CheckpointCancellingTarget:
    def __init__(self, store: InMemoryTaskStore) -> None:
        self.store = store
        self.after_checkpoint = False

    async def __call__(self, context: TaskTargetContext) -> object:
        await self.store.transition_run(
            context.execution.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.CANCEL_REQUESTED,
            reason="test_cancel",
        )
        await context.check_cancelled()
        self.after_checkpoint = True
        return "late success"


class ConflictFinalizer(TaskRunFinalizer):
    async def finalize_success(
        self,
        *args: object,
        **kwargs: object,
    ) -> TaskRunResult:
        raise TaskStoreConflictError("commit conflict")


class SystemExitRunner(DirectTaskRunner):
    async def _run_single_attempt(
        self,
        *args: object,
        **kwargs: object,
    ) -> TaskRunResult:
        raise SystemExit("stop")


class RejectingTarget(TaskTargetRunner):
    def __init__(self) -> None:
        self.ran = False

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return (
            TaskValidationIssue(
                code="execution.unknown_target",
                path="execution.ref",
                message="Task target could not be loaded.",
                hint="Use a supported execution target.",
                category=TaskValidationCategory.UNSUPPORTED,
            ),
        )

    async def run(self, context: TaskTargetContext) -> object:
        self.ran = True
        return "unused"


class DisappearingDescriptor(Mapping[str, object]):
    def __init__(self) -> None:
        self._data = {
            "source_kind": "local_path",
            "reference": "input.txt",
            "mime_type": "text/plain",
        }

    def __getitem__(self, key: str) -> object:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


def definition(
    *,
    artifact: TaskArtifactPolicy | None = None,
    privacy: TaskPrivacyPolicy | None = None,
    run: TaskRunPolicy | None = None,
    retry: TaskRetryPolicy | None = None,
    input_contract: TaskInputContract | None = None,
    output_contract: TaskOutputContract | None = None,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=input_contract or TaskInputContract.string(),
        output=output_contract or TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
        artifact=artifact or TaskArtifactPolicy.references_only(),
        privacy=privacy or TaskPrivacyPolicy(),
        run=run or TaskRunPolicy.direct(),
        retry=retry or TaskRetryPolicy(),
    )


class DirectTaskRunnerTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.store = InMemoryTaskStore()
        self.hmac_provider: HmacProvider = StaticHmacProvider()

    async def test_fake_target_success_creates_inspectable_lifecycle(
        self,
    ) -> None:
        target = RecordingTarget("short summary")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-success",
        )

        result = await runner.run(
            definition(),
            input_value="private prompt",
            files=(
                TaskInputFile(
                    logical_path="uploads/input.txt",
                    media_type="text/plain",
                    size_bytes=19,
                    metadata={"count": 1},
                ),
            ),
            idempotency_key="request-1",
            metadata={"attempt": 1},
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.attempt.state, TaskAttemptState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(target.contexts[0].input_value, "private prompt")
        run = await self.store.get_run(result.run.run_id)
        input_summary = cast(Mapping[str, object], run.request.input_summary)
        file_summary = cast(
            Mapping[str, object],
            run.request.file_summaries[0],
        )
        output_summary = cast(
            Mapping[str, object],
            result.run.result.output_summary if result.run.result else {},
        )

        self.assertEqual(input_summary["privacy"], "<hmac-sha256>")
        self.assertNotIn("private prompt", str(input_summary))
        self.assertEqual(file_summary["privacy"], "<hmac-sha256>")
        self.assertNotIn("uploads/input.txt", str(file_summary))
        self.assertEqual(output_summary["privacy"], "<redacted>")
        self.assertEqual(run.request.idempotency_key, "request-1")
        self.assertEqual(run.request.metadata["attempt"], 1)
        self.assertEqual(
            [
                transition.to_state
                for transition in await self.store.list_run_transitions(
                    result.run.run_id
                )
            ],
            [
                TaskRunState.VALIDATED,
                TaskRunState.RUNNING,
                TaskRunState.SUCCEEDED,
            ],
        )

    async def test_fake_target_failure_records_safe_failure(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=FailingTarget(),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-failure",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertIsNotNone(result.run.result)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "runnable")
        self.assertEqual(error_summary["code"], "runnable.failed")
        self.assertEqual(error_summary["retryable"], True)
        self.assertNotIn("raw secret failure", str(error_summary))
        self.assertEqual(
            [
                transition.to_state
                for transition in await self.store.list_attempt_transitions(
                    result.attempt.attempt_id
                )
            ],
            [TaskAttemptState.RUNNING, TaskAttemptState.FAILED],
        )

    async def test_cancelled_target_records_safe_failure(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=CancelledTarget(),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-cancelled",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.CANCELLED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "cancellation")
        self.assertEqual(error_summary["code"], "cancellation.requested")
        self.assertNotIn("raw cancellation reason", str(error_summary))

    async def test_retryable_failure_retries_until_success(self) -> None:
        target = FlakyTarget(failures=2, output="short summary")
        delays: list[float] = []
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-retry-success",
            sleep=_record_sleep(delays),
        )

        result = await runner.run(
            definition(
                retry=TaskRetryPolicy(
                    max_attempts=3,
                    backoff=RetryBackoff.LINEAR,
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.attempt.attempt_number, 3)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(delays, [1, 2])
        attempts = await self.store.list_attempts(result.run.run_id)
        self.assertEqual(
            [attempt.state for attempt in attempts],
            [
                TaskAttemptState.FAILED,
                TaskAttemptState.FAILED,
                TaskAttemptState.SUCCEEDED,
            ],
        )

    async def test_retry_exhaustion_records_safe_attempt_counts(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=FailingTarget(),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-retry-exhausted",
            sleep=_record_sleep([]),
        )

        result = await runner.run(
            definition(
                retry=TaskRetryPolicy(
                    max_attempts=2,
                    backoff=RetryBackoff.EXPONENTIAL,
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.attempt_number, 2)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        details = cast(Mapping[str, object], error_summary["details"])
        self.assertEqual(details["failed_attempt_count"], 2)
        self.assertEqual(details["max_attempts"], 2)
        self.assertEqual(details["retry_exhausted"], True)
        self.assertNotIn("raw secret failure", str(error_summary))

    async def test_cancellation_during_retry_backoff_cancels_run(
        self,
    ) -> None:
        target = FlakyTarget(failures=1, output="unused")

        async def sleep(delay: float) -> None:
            await self.store.transition_run(
                target.contexts[0].execution.run_id,
                from_states={TaskRunState.RUNNING},
                to_state=TaskRunState.CANCEL_REQUESTED,
                reason="test_cancel",
            )

        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-retry-cancel",
            sleep=sleep,
        )

        result = await runner.run(
            definition(
                retry=TaskRetryPolicy(
                    max_attempts=2,
                    backoff=RetryBackoff.LINEAR,
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.CANCELLED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(len(target.contexts), 1)

    async def test_expiry_during_retry_backoff_expires_run(self) -> None:
        target = FlakyTarget(failures=1, output="unused")
        expires_at = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
        current_time = datetime(2026, 1, 1, tzinfo=UTC)

        async def sleep(delay: float) -> None:
            nonlocal current_time
            current_time = expires_at

        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-retry-expire",
            clock=lambda: current_time,
            sleep=sleep,
        )

        result = await runner.run(
            definition(
                retry=TaskRetryPolicy(
                    max_attempts=2,
                    backoff=RetryBackoff.LINEAR,
                )
            ),
            input_value="private prompt",
            expires_at=expires_at,
        )

        self.assertEqual(result.run.state, TaskRunState.EXPIRED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(len(target.contexts), 1)

    async def test_attempt_timeout_fails_without_expiring_run(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=SlowTarget(),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-timeout",
        )

        result = await runner.run(
            definition(run=TaskRunPolicy.direct(timeout_seconds=1)),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "timeout")
        self.assertEqual(error_summary["code"], "timeout.exceeded")

    async def test_system_exit_from_target_propagates(self) -> None:
        runner = SystemExitRunner(
            self.store,
            target=RecordingTarget("unused"),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-system-exit",
        )

        with self.assertRaises(SystemExit):
            await runner.run(definition(), input_value="private prompt")

    async def test_cancellation_before_success_commits_cancelled_run(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=CancellingTarget(self.store),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-cancel-before-success",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.CANCELLED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertIsNone(result.output)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                result.run.run_id,
                from_states={TaskRunState.CANCELLED},
                to_state=TaskRunState.SUCCEEDED,
                reason="late_success",
            )

    async def test_target_cancellation_checkpoint_stops_work(self) -> None:
        target = CheckpointCancellingTarget(self.store)
        runner = DirectTaskRunner(
            self.store,
            target=target,
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-cancel-checkpoint",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.CANCELLED)
        self.assertFalse(target.after_checkpoint)

    async def test_success_commit_conflict_is_not_hidden(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget("short summary"),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-success-conflict",
            finalizer=ConflictFinalizer(),
        )

        with self.assertRaises(TaskStoreConflictError):
            await runner.run(definition(), input_value="private prompt")

    async def test_expired_run_age_uses_expired_state_not_timeout_failure(
        self,
    ) -> None:
        now = datetime(2026, 1, 1, tzinfo=UTC)
        target = RecordingTarget("unused")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-expired",
            clock=lambda: now,
        )

        result = await runner.run(
            definition(),
            input_value="private prompt",
            expires_at=now,
        )

        self.assertEqual(result.run.state, TaskRunState.EXPIRED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(target.contexts, [])
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        details = cast(Mapping[str, object], error_summary["details"])
        self.assertEqual(error_summary["category"], "timeout")
        self.assertEqual(details["scope"], "run")

    async def test_future_expiration_uses_default_clock_path(self) -> None:
        target = RecordingTarget("short summary")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-default-clock",
        )

        result = await runner.run(
            definition(),
            input_value="private prompt",
            expires_at=datetime(2999, 1, 1, tzinfo=UTC),
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")

    async def test_input_validation_happens_before_target_execution(
        self,
    ) -> None:
        target = RecordingTarget("unused")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-invalid",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(definition(), input_value={"raw": "not text"})

        self.assertEqual(target.contexts, [])
        self.assertIn("input.invalid_type", str(error.exception))
        self.assertNotIn("not text", str(error.exception))

    async def test_file_input_materializes_when_artifact_store_is_configured(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")
            target = RecordingTarget("short summary")
            runner = DirectTaskRunner(
                self.store,
                target=cast(TaskDirectTarget, target),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-1",
                ),
                definition_hash=lambda task: "hash-file-input",
                execution_roots=(root,),
            )

            result = await runner.run(
                definition(input_contract=TaskInputContract.file()),
                input_value=TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                ),
            )

            self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
            self.assertEqual(len(target.contexts), 1)
            self.assertEqual(
                target.contexts[0].files[0].logical_path,
                "artifact:artifact-1",
            )
            artifacts = await self.store.list_artifacts(result.run.run_id)
            self.assertEqual(len(artifacts), 1)
            self.assertEqual(
                artifacts[0].attempt_id, result.attempt.attempt_id
            )
            self.assertNotIn("input.txt", str(artifacts[0].summary()))

    async def test_file_materialization_failure_finalizes_run(
        self,
    ) -> None:
        target = RecordingTarget("unused")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-file-materialization-failure",
            execution_roots=("/tmp/private-root",),
        )

        result = await runner.run(
            definition(input_contract=TaskInputContract.file()),
            input_value=TaskFileDescriptor.local_path(
                "secret-input.txt",
                mime_type="text/plain",
            ),
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(target.contexts, [])
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "input_contract")
        self.assertEqual(error_summary["code"], "input_contract.failed")
        self.assertNotIn("secret-input", str(error_summary))
        self.assertEqual(
            [
                transition.to_state
                for transition in await self.store.list_run_transitions(
                    result.run.run_id
                )
            ],
            [
                TaskRunState.VALIDATED,
                TaskRunState.RUNNING,
                TaskRunState.FAILED,
            ],
        )

    async def test_target_validation_happens_before_run_creation(self) -> None:
        target = RejectingTarget()
        runner = DirectTaskRunner(
            self.store,
            target=target,
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-target-invalid",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(definition(), input_value="private")

        self.assertFalse(target.ran)
        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["execution.unknown_target"],
        )

    async def test_missing_hmac_provider_returns_validation_diagnostic(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget("unused"),
            definition_hash=lambda task: "hash-missing-key",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(definition(), input_value="private")

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["privacy.hmac_key_missing"],
        )

    async def test_direct_runner_rejects_queued_definitions(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget("unused"),
            hmac_provider=self.hmac_provider,
        )

        with self.assertRaises(TaskRunnerError):
            await runner.run(
                definition(
                    privacy=TaskPrivacyPolicy(input=PrivacyAction.REDACT),
                    run=TaskRunPolicy.queued("default"),
                ),
                input_value="private",
            )

    async def test_error_summary_fallback_is_privacy_safe(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget("unused"),
            hmac_provider=self.hmac_provider,
        )
        sanitizer = PrivacySanitizer(
            TaskPrivacyPolicy(errors=PrivacyAction.ENCRYPT)
        )

        summary = runner._safe_error_summary(  # noqa: SLF001
            sanitizer,
            RuntimeError("raw secret failure"),
        )

        self.assertEqual(
            summary,
            {
                "category": "runnable",
                "code": TaskErrorCode.RUNNABLE_FAILED.value,
                "privacy": "<redacted>",
            },
        )

    async def test_output_validation_happens_before_success_commit(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget({"secret": "raw model output"}),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-output-invalid",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertIsNone(result.output)
        self.assertIsNotNone(result.run.result)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "output_contract")
        self.assertEqual(error_summary["code"], "output_contract.failed")
        self.assertNotIn("raw model output", str(error_summary))
        self.assertEqual(
            [
                transition.to_state
                for transition in await self.store.list_run_transitions(
                    result.run.run_id
                )
            ],
            [
                TaskRunState.VALIDATED,
                TaskRunState.RUNNING,
                TaskRunState.FAILED,
            ],
        )

    async def test_structured_output_validation_allows_success(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget({"answer": "ok"}),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-output-valid",
        )

        result = await runner.run(
            definition(
                output_contract=TaskOutputContract.object(
                    schema={
                        "type": "object",
                        "required": ["answer"],
                        "properties": {"answer": {"type": "string"}},
                    }
                )
            ),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, {"answer": "ok"})

    async def test_file_output_records_artifact_before_success(self) -> None:
        with TemporaryDirectory() as artifacts:
            runner = DirectTaskRunner(
                self.store,
                target=ArtifactOutputTarget(),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-output-1",
                ),
                definition_hash=lambda task: "hash-file-output",
            )

            result = await runner.run(
                definition(
                    output_contract=TaskOutputContract.file(),
                    privacy=TaskPrivacyPolicy(output=PrivacyAction.HASH),
                    artifact=TaskArtifactPolicy.references_only(
                        retention_days=4
                    ),
                ),
                input_value="private prompt",
            )

            self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
            self.assertIsInstance(result.output, TaskArtifactRef)
            records = await self.store.list_artifacts(
                result.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].attempt_id, result.attempt.attempt_id)
            self.assertEqual(records[0].state, TaskArtifactState.READY)
            self.assertEqual(records[0].retention.delete_after_days, 4)
            self.assertNotIn(
                "private report bytes",
                str(records[0].summary()),
            )
            output_summary = cast(
                Mapping[str, object],
                result.run.result.output_summary if result.run.result else {},
            )
            self.assertEqual(output_summary["privacy"], "<hmac-sha256>")

    async def test_artifact_array_output_preserves_record_state(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            runner = DirectTaskRunner(
                self.store,
                target=ArtifactRecordArrayTarget(),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-lost-1",
                ),
                definition_hash=lambda task: "hash-artifact-array-output",
            )

            result = await runner.run(
                definition(
                    output_contract=TaskOutputContract.artifact_array(),
                    privacy=TaskPrivacyPolicy(output=PrivacyAction.REDACT),
                ),
                input_value="private prompt",
            )

            self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
            records = await self.store.list_artifacts(
                result.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            )
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].state, TaskArtifactState.LOST)
            self.assertEqual(records[0].provenance.operation, "export")
            self.assertEqual(records[0].retention.delete_after_days, 2)
            self.assertEqual(records[0].metadata["state"], "simulated")

    async def test_artifact_output_validation_fails_before_success(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget(object()),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-output-artifact-invalid",
        )

        result = await runner.run(
            definition(output_contract=TaskOutputContract.file()),
            input_value="private prompt",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        self.assertEqual(
            await self.store.list_artifacts(
                result.run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            ),
            (),
        )
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "output_contract")
        self.assertEqual(error_summary["code"], "output_contract.failed")

    def test_safe_target_helpers_freeze_values(self) -> None:
        metadata = safe_target_metadata({"counts": [1, 2]})
        value = safe_target_value({"items": ["a"]})

        self.assertEqual(metadata["counts"], (1, 2))
        self.assertEqual(cast(Mapping[str, object], value)["items"], ("a",))
        with self.assertRaises(TypeError):
            cast(dict[str, object], metadata)["raw"] = "leak"

    def test_attempt_policy_summary_leaves_non_mapping_values(self) -> None:
        self.assertEqual(
            _error_summary_with_attempt_policy(
                "<redacted>",
                retry_decision=TaskAttemptDecision(
                    type=TaskAttemptDecisionType.RETRY,
                    attempt_number=1,
                    max_attempts=2,
                    retry_delay_seconds=0,
                ),
            ),
            "<redacted>",
        )

    def test_file_input_summary_covers_descriptor_variants(self) -> None:
        summary = _input_summary_value(
            definition(input_contract=TaskInputContract.file_array()),
            [
                TaskFileDescriptor.local_path(
                    "input-a.txt",
                    role="source",
                    mime_type="text/plain",
                    size_bytes=12,
                    sha256="a" * 64,
                    conversions=(
                        TaskFileConversionRequest(
                            name="markdown",
                            options={"strict": True},
                        ),
                    ),
                    metadata={"caller": "private"},
                ),
                TaskFileDescriptor.local_path("input-b.txt"),
            ],
        )

        self.assertIsInstance(summary, tuple)
        first = cast(tuple[Mapping[str, object], ...], summary)[0]
        self.assertEqual(first["role"], "source")
        self.assertEqual(first["mime_type"], "text/plain")
        self.assertEqual(first["size_bytes"], 12)
        self.assertEqual(first["sha256"], "a" * 64)
        self.assertIn("conversions", first)
        self.assertIn("metadata", first)

    def test_file_input_summary_returns_original_for_malformed_value(
        self,
    ) -> None:
        value = {"source_kind": "local_path"}

        self.assertIs(_input_summary_value(definition(), value), value)

    def test_file_input_summary_returns_original_when_coercion_fails(
        self,
    ) -> None:
        value = DisappearingDescriptor()

        self.assertIs(
            _input_summary_value(
                definition(input_contract=TaskInputContract.file()),
                value,
            ),
            value,
        )


def _record_sleep(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return sleep


if __name__ == "__main__":
    main()
