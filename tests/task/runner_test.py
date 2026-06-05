import asyncio
from collections.abc import Awaitable, Callable, Collection, Iterator, Mapping
from datetime import UTC, datetime
from io import BytesIO
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
    TaskAttempt,
    TaskAttemptDecision,
    TaskAttemptDecisionType,
    TaskAttemptState,
    TaskDefinition,
    TaskDirectTarget,
    TaskErrorCode,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileConversionError,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionRequest,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMaterializedFile,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputContract,
    TaskPrivacyPolicy,
    TaskProviderReferenceKind,
    TaskRemoteUrlPolicy,
    TaskRemoteUrlResponse,
    TaskRetryPolicy,
    TaskRun,
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
    UsageRecord,
    UsageSource,
    UsageTotals,
    safe_target_metadata,
    safe_target_value,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.runner import (
    TaskExecutableInputFileEntry,
    TaskRunExpiredError,
    _conversion_request_from_value,
    _conversion_request_value,
    _error_summary_with_attempt_policy,
    _input_summary_value,
    build_task_executable_input_files,
    task_execution_file_entries_from_value,
    task_execution_file_entries_value,
    task_input_file_entries_for_queue,
    task_input_files_from_materialized,
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


class FakeRemoteResolver:
    def __init__(
        self,
        addresses: Mapping[str, tuple[str, ...]] | None = None,
    ) -> None:
        self.addresses = dict(addresses or {})
        self.calls: list[str] = []

    def resolve(self, hostname: str) -> tuple[str, ...]:
        self.calls.append(hostname)
        return self.addresses.get(hostname, ("8.8.8.8",))


class FakeRemoteClient:
    def __init__(
        self,
        responses: Mapping[str, TaskRemoteUrlResponse],
    ) -> None:
        self.responses = dict(responses)
        self.calls: list[tuple[str, float]] = []

    def open(
        self,
        url: str,
        *,
        timeout_seconds: float,
    ) -> TaskRemoteUrlResponse:
        self.calls.append((url, timeout_seconds))
        return self.responses[url]


class RecordingTarget:
    def __init__(self, output: object) -> None:
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def __call__(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        return self.output


class UsageObservingTarget:
    def __init__(self, response: object) -> None:
        self.response = response
        self.contexts: list[TaskTargetContext] = []

    async def __call__(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.observe_usage(self.response)
        return "short summary"


class DuplicateUsageObservingTarget(UsageObservingTarget):
    async def __call__(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.observe_usage(self.response)
        await context.observe_usage(self.response)
        return "short summary"


class EmptyThenReturnedUsageTarget:
    def __init__(self, output: object) -> None:
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def __call__(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.observe_usage(object())
        return self.output


class CountingUsageResponse:
    input_token_count = 3
    output_token_count = 5


class MultiCallUsageResponse:
    def __init__(self, *responses: object) -> None:
        self.usage_responses = responses


class UsageTextOutput(str):
    _usage: object | None
    _usage_responses: tuple[object, ...]

    def __new__(
        cls,
        value: str,
        *,
        usage: object | None = None,
        usage_responses: tuple[object, ...] = (),
    ) -> "UsageTextOutput":
        output = str.__new__(cls, value)
        output._usage = usage
        output._usage_responses = usage_responses
        return output

    @property
    def usage(self) -> object | None:
        return self._usage

    @property
    def usage_responses(self) -> tuple[object, ...]:
        return self._usage_responses


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


class PrivateArtifactRecordTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        assert context.artifact_store is not None
        ref = await context.artifact_store.put(
            b"private bytes",
            media_type="text/plain",
            metadata={"filename": "private-ref.txt"},
        )
        created_at = datetime(2026, 1, 1, tzinfo=UTC)
        return [
            TaskArtifactRecord(
                artifact_id=ref.artifact_id,
                run_id=context.execution.run_id,
                attempt_id=context.execution.attempt_id,
                purpose=TaskArtifactPurpose.OUTPUT,
                state=TaskArtifactState.READY,
                ref=ref,
                created_at=created_at,
                updated_at=created_at,
                provenance=TaskArtifactProvenance(
                    operation="export",
                    metadata={"prompt": "private prompt"},
                ),
                retention=TaskArtifactRetention(
                    delete_after_days=2,
                    metadata={"tenant": "private retention"},
                ),
                metadata={
                    "filename": "private-output.txt",
                    "state": "simulated",
                },
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


class ExpiringTarget:
    async def __call__(self, context: TaskTargetContext) -> object:
        raise TaskRunExpiredError()


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


class ConflictOnAttemptSuccessStore(InMemoryTaskStore):
    async def transition_attempt(
        self,
        attempt_id: str,
        *,
        from_states: Collection[TaskAttemptState],
        to_state: TaskAttemptState,
        reason: str,
        result: TaskExecutionResult | None = None,
        claim_token: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskAttempt:
        if to_state == TaskAttemptState.SUCCEEDED:
            raise TaskStoreConflictError("attempt commit conflict")
        return await super().transition_attempt(
            attempt_id,
            from_states=from_states,
            to_state=to_state,
            reason=reason,
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )


class FailingUsageStore(InMemoryTaskStore):
    async def append_usage(
        self,
        run_id: str,
        *,
        source: UsageSource,
        totals: UsageTotals,
        attempt_id: str | None = None,
        usage_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> UsageRecord:
        raise RuntimeError("private usage store failure")


class SystemExitRunner(DirectTaskRunner):
    async def _run_single_attempt(
        self,
        *args: object,
        **kwargs: object,
    ) -> TaskRunResult:
        raise SystemExit("stop")


class MaterializationFailureRunner(DirectTaskRunner):
    error: BaseException

    async def _input_files_from_materialized(
        self,
        definition: TaskDefinition,
        materialized_files: tuple[TaskMaterializedFile, ...],
        *,
        run: TaskRun,
        attempt: TaskAttempt,
    ) -> tuple[TaskInputFile, ...]:
        raise self.error


class FailingTextConverter:
    name = "text"
    version = "failure"

    @property
    def capability(self) -> TaskFileConverterCapability:
        return TaskFileConverterCapability(
            source_mime_types=("text/*",),
            output_mime_types=("text/plain",),
            supports_streaming=False,
            max_input_bytes=1024,
            max_output_bytes=1024,
        )

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        raise TaskFileConversionError("private conversion failure")


class PrefixTextConverter:
    name = "text"
    version = "test"
    capability = TaskFileConverterCapability(
        source_mime_types=("text/plain",),
        output_mime_types=("text/plain",),
        supports_streaming=False,
        max_input_bytes=1024,
        max_output_bytes=1024,
    )

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        _ = source_media_type, options
        return TaskFileConversionResult(
            content=b"converted:" + content,
            media_type="text/plain",
            metadata={},
        )


class PdfPageConverter:
    name = "pdf_image"
    version = "test"

    def __init__(
        self,
        pages: tuple[TaskFileConversionPageResult, ...],
    ) -> None:
        self.pages = pages

    @property
    def capability(self) -> TaskFileConverterCapability:
        return TaskFileConverterCapability(
            source_mime_types=("application/pdf",),
            output_mime_types=("image/png",),
            supports_streaming=False,
            max_input_bytes=1024,
            max_output_bytes=1024,
            max_pages=8,
            max_pixels=10_000,
        )

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        _ = content, source_media_type, options
        raise AssertionError("page converter must use page output")

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        _ = content, source_media_type, options
        return TaskFileConversionPageCollection(
            pages=self.pages,
            metadata={"backend": "test"},
        )


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
    observability: TaskObservabilityPolicy | None = None,
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
        observability=observability or TaskObservabilityPolicy(),
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

    async def test_observe_usage_ignores_unrecognized_response(self) -> None:
        target = UsageObservingTarget(object())
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-usage-unavailable",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            target.contexts[0].execution.run_id, result.run.run_id
        )
        self.assertEqual(await self.store.list_usage(result.run.run_id), ())

    async def test_observe_usage_records_each_provider_call(self) -> None:
        target = UsageObservingTarget(
            MultiCallUsageResponse(
                type(
                    "FirstUsageResponse",
                    (),
                    {
                        "usage": {
                            "input_tokens": 3,
                            "cached_input_tokens": 1,
                            "output_tokens": 2,
                            "total_tokens": 5,
                            "provider_family": "openai",
                        }
                    },
                )(),
                type(
                    "SecondUsageResponse",
                    (),
                    {
                        "usage": {
                            "input_tokens": 4,
                            "cache_creation_input_tokens": 2,
                            "output_tokens": 6,
                            "reasoning_tokens": 1,
                            "total_tokens": 10,
                            "provider_family": "openai",
                            "raw_response_id": "private-response-id",
                        }
                    },
                )(),
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-usage-multi-call",
        )

        result = await runner.run(definition(), input_value="private prompt")
        records = await self.store.list_usage(result.run.run_id)
        totals = await self.store.usage_totals(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 2)
        self.assertEqual([record.sequence for record in records], [1, 2])
        self.assertEqual(
            [record.source for record in records],
            [UsageSource.EXACT, UsageSource.EXACT],
        )
        self.assertEqual(records[0].totals.cached_input_tokens, 1)
        self.assertEqual(records[1].totals.cache_creation_input_tokens, 2)
        self.assertEqual(records[1].totals.reasoning_tokens, 1)
        self.assertNotIn("raw_response_id", records[1].metadata)
        self.assertEqual(totals.input_tokens, 7)
        self.assertEqual(totals.output_tokens, 8)
        self.assertEqual(totals.total_tokens, 15)

    async def test_observe_usage_deduplicates_reobserved_response(
        self,
    ) -> None:
        target = DuplicateUsageObservingTarget(CountingUsageResponse())
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-usage-deduplicate",
        )

        result = await runner.run(definition(), input_value="private prompt")
        records = await self.store.list_usage(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].sequence, 1)
        self.assertEqual(records[0].totals.input_tokens, 3)
        self.assertEqual(records[0].totals.output_tokens, 5)

    async def test_observe_usage_drops_malformed_nested_provider_call(
        self,
    ) -> None:
        target = UsageObservingTarget(
            MultiCallUsageResponse(
                type(
                    "InvalidUsageResponse",
                    (),
                    {
                        "usage": {
                            "input_tokens": "private prompt",
                            "output_tokens": -1,
                            "total_tokens": True,
                        }
                    },
                )(),
                type(
                    "ZeroUsageResponse",
                    (),
                    {
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        }
                    },
                )(),
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-usage-malformed-call",
        )

        result = await runner.run(definition(), input_value="private prompt")
        records = await self.store.list_usage(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].totals.input_tokens, 0)
        self.assertEqual(records[0].totals.output_tokens, 0)
        self.assertEqual(records[0].totals.total_tokens, 0)

    async def test_returned_output_usage_is_recorded_without_observer_call(
        self,
    ) -> None:
        target = RecordingTarget(
            UsageTextOutput(
                "short summary",
                usage={
                    "input_tokens": 9,
                    "cached_input_tokens": 4,
                    "output_tokens": 6,
                    "reasoning_tokens": 2,
                    "total_tokens": 15,
                    "provider_family": "openai",
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-returned-output-usage",
        )

        result = await runner.run(definition(), input_value="private prompt")
        records = await self.store.list_usage(result.run.run_id)
        totals = await self.store.usage_totals(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "short summary")
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 9)
        self.assertEqual(records[0].totals.cached_input_tokens, 4)
        self.assertEqual(records[0].totals.output_tokens, 6)
        self.assertEqual(records[0].totals.reasoning_tokens, 2)
        self.assertEqual(records[0].totals.total_tokens, 15)
        self.assertEqual(records[0].metadata, {"provider_family": "openai"})
        self.assertEqual(totals.total_tokens, 15)

    async def test_empty_observation_does_not_hide_returned_output_usage(
        self,
    ) -> None:
        target = EmptyThenReturnedUsageTarget(
            UsageTextOutput(
                "short summary",
                usage={
                    "input_tokens": 6,
                    "output_tokens": 4,
                    "total_tokens": 10,
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-empty-then-returned-usage",
        )

        result = await runner.run(definition(), input_value="private prompt")
        records = await self.store.list_usage(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].totals.input_tokens, 6)
        self.assertEqual(records[0].totals.output_tokens, 4)
        self.assertEqual(records[0].totals.total_tokens, 10)

    async def test_malformed_returned_output_usage_is_ignored(self) -> None:
        target = RecordingTarget(
            UsageTextOutput(
                "short summary",
                usage={
                    "input_tokens": "private prompt",
                    "output_tokens": -1,
                    "total_tokens": True,
                    "raw_response_body": "private provider body",
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-returned-output-bad-usage",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(await self.store.list_usage(result.run.run_id), ())

    async def test_returned_output_usage_survives_output_contract_failure(
        self,
    ) -> None:
        target = RecordingTarget(
            UsageTextOutput(
                "private invalid output",
                usage={
                    "input_tokens": 2,
                    "cache_creation_input_tokens": 1,
                    "output_tokens": 3,
                    "total_tokens": 5,
                    "provider_family": "openai",
                    "raw_response_id": "private-response-id",
                },
            )
        )
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-returned-output-failed",
        )

        result = await runner.run(
            definition(
                output_contract=TaskOutputContract.object(
                    schema={"type": "object"}
                )
            ),
            input_value="private prompt",
        )
        records = await self.store.list_usage(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(result.output, None)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].source, UsageSource.EXACT)
        self.assertEqual(records[0].totals.input_tokens, 2)
        self.assertEqual(records[0].totals.cache_creation_input_tokens, 1)
        self.assertEqual(records[0].totals.total_tokens, 5)
        self.assertNotIn("raw_response_id", records[0].metadata)
        self.assertNotIn(
            "private invalid output",
            str(result.run.result.error if result.run.result else {}),
        )

    async def test_usage_store_failure_does_not_fail_run(self) -> None:
        store = FailingUsageStore()
        target = UsageObservingTarget(CountingUsageResponse())
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-usage-store-failure",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            target.contexts[0].execution.run_id, result.run.run_id
        )
        self.assertEqual(await store.list_usage(result.run.run_id), ())

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

    async def test_target_run_expiry_marks_run_expired(self) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=ExpiringTarget(),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-target-expired",
        )

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.EXPIRED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertEqual(error_summary["category"], "timeout")
        self.assertEqual(error_summary["code"], "timeout.exceeded")

    async def test_materialization_cancellation_finalizes_cancelled_run(
        self,
    ) -> None:
        runner = MaterializationFailureRunner(
            self.store,
            target=RecordingTarget("unused"),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-materialization-cancel",
        )
        runner.error = asyncio.CancelledError("private cancel")

        result = await runner.run(definition(), input_value="private prompt")

        self.assertEqual(result.run.state, TaskRunState.CANCELLED)
        self.assertEqual(result.attempt.state, TaskAttemptState.FAILED)
        error_summary = cast(
            Mapping[str, object],
            result.run.result.error if result.run.result else {},
        )
        self.assertNotIn("private cancel", str(error_summary))

    async def test_materialization_system_exit_propagates(self) -> None:
        runner = MaterializationFailureRunner(
            self.store,
            target=RecordingTarget("unused"),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-materialization-system-exit",
        )
        runner.error = SystemExit("private stop")

        with self.assertRaises(SystemExit):
            await runner.run(definition(), input_value="private prompt")

    async def test_materialization_store_conflict_propagates(self) -> None:
        runner = MaterializationFailureRunner(
            self.store,
            target=RecordingTarget("unused"),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-materialization-conflict",
        )
        runner.error = TaskStoreConflictError("private conflict")

        with self.assertRaises(TaskStoreConflictError):
            await runner.run(definition(), input_value="private prompt")

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

    async def test_success_attempt_conflict_leaves_run_mutable(self) -> None:
        store = ConflictOnAttemptSuccessStore()
        target = RecordingTarget("short summary")
        runner = DirectTaskRunner(
            store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-success-attempt-conflict",
        )

        with self.assertRaises(TaskStoreConflictError):
            await runner.run(definition(), input_value="private prompt")

        context = target.contexts[0]
        run = await store.get_run(context.execution.run_id)
        attempt = await store.get_attempt(context.execution.attempt_id)
        self.assertEqual(run.state, TaskRunState.RUNNING)
        self.assertEqual(attempt.state, TaskAttemptState.RUNNING)

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

    async def test_provider_reference_input_reaches_target_without_bytes(
        self,
    ) -> None:
        target = RecordingTarget("accepted")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-provider-reference",
        )

        result = await runner.run(
            definition(input_contract=TaskInputContract.file()),
            input_value=TaskFileDescriptor.provider_reference_descriptor(
                "file-openai",
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="openai",
                mime_type="application/pdf",
                owner_scope="tenant-a",
                identity_hmac="hmac-value",
            ),
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(target.contexts), 1)
        self.assertEqual(len(target.contexts[0].files), 1)
        file = target.contexts[0].files[0]
        self.assertIsNotNone(file.provider_reference)
        assert file.provider_reference is not None
        self.assertEqual(file.provider_reference.reference, "file-openai")
        self.assertIsNone(file.artifact_ref)
        self.assertEqual(
            await self.store.list_artifacts(result.run.run_id),
            (),
        )
        self.assertNotIn("file-openai", str(result.run.request.input_summary))

    async def test_provider_reference_bad_expiry_rejects_before_target(
        self,
    ) -> None:
        target = RecordingTarget("unused")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-provider-reference-expiry",
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner.run(
                definition(input_contract=TaskInputContract.file()),
                input_value={
                    "source_kind": "provider_reference",
                    "reference": "file-private",
                    "provider_reference": {
                        "kind": "provider_file_id",
                        "provider": "openai",
                        "reference": "file-private",
                        "expires_at": "private-invalid-expiry",
                        "durable": False,
                    },
                },
            )

        self.assertEqual(target.contexts, [])
        self.assertEqual(
            error.exception.issues[0].path,
            "input.provider_reference.expires_at",
        )
        self.assertNotIn("file-private", str(error.exception))
        self.assertNotIn("private-invalid-expiry", str(error.exception))

    async def test_explicit_provider_reference_expiry_fails_before_target(
        self,
    ) -> None:
        target = RecordingTarget("unused")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-explicit-provider-expiry",
            clock=lambda: datetime(2026, 1, 1, tzinfo=UTC),
        )
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            expires_at=datetime(2025, 12, 31, tzinfo=UTC),
            durable=False,
        )
        assert descriptor.provider_reference is not None

        result = await runner.run(
            definition(input_contract=TaskInputContract.string()),
            input_value="prompt",
            files=(
                TaskInputFile(
                    logical_path="provider:openai:provider_file_id",
                    provider_reference=descriptor.provider_reference,
                ),
            ),
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        assert result.run.result is not None
        self.assertNotIn("file-private", str(result.run.result.error))

    async def test_explicit_provider_reference_mime_mismatch_fails(
        self,
    ) -> None:
        target = RecordingTarget("unused")
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, target),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda task: "hash-explicit-provider-mime",
        )
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            mime_type="application/pdf",
        )
        assert descriptor.provider_reference is not None

        result = await runner.run(
            definition(input_contract=TaskInputContract.string()),
            input_value="prompt",
            files=(
                TaskInputFile(
                    logical_path="provider:openai:provider_file_id",
                    provider_reference=descriptor.provider_reference,
                    media_type="text/plain",
                ),
            ),
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        assert result.run.result is not None
        self.assertNotIn("file-private", str(result.run.result.error))

    async def test_native_file_conversion_uses_materialized_input(
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
                definition_hash=lambda task: "hash-native-file-input",
                execution_roots=(root,),
            )

            result = await runner.run(
                definition(
                    input_contract=TaskInputContract.file(
                        conversions=("native", "none"),
                    )
                ),
                input_value=TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                    conversions=(
                        TaskFileConversionRequest(name="native"),
                        TaskFileConversionRequest(name="none"),
                    ),
                ),
            )

            self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
            self.assertEqual(
                target.contexts[0].files[0].logical_path,
                "artifact:artifact-1",
            )
            artifacts = await self.store.list_artifacts(result.run.run_id)
            self.assertEqual(
                [artifact.purpose for artifact in artifacts],
                [TaskArtifactPurpose.INPUT],
            )

    async def test_page_file_conversion_expands_materialized_input(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.pdf").write_bytes(b"%PDF private")
            target = RecordingTarget("short summary")
            artifact_ids = iter(("source-pdf", "page-1", "page-2"))
            runner = DirectTaskRunner(
                self.store,
                target=cast(TaskDirectTarget, target),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: next(artifact_ids),
                ),
                file_converters={
                    "pdf_image": PdfPageConverter(
                        (
                            TaskFileConversionPageResult(
                                page_index=1,
                                page_count=2,
                                content=b"page one",
                                media_type="image/png",
                                width_pixels=10,
                                height_pixels=20,
                                metadata={"filename": "private-page.png"},
                            ),
                            TaskFileConversionPageResult(
                                page_index=2,
                                page_count=2,
                                content=b"page two",
                                media_type="image/png",
                                width_pixels=30,
                                height_pixels=40,
                            ),
                        )
                    )
                },
                definition_hash=lambda task: "hash-page-file-input",
                execution_roots=(root,),
            )

            result = await runner.run(
                definition(
                    input_contract=TaskInputContract.file(
                        conversions=("pdf_image",),
                        mime_types=("application/pdf",),
                    )
                ),
                input_value=TaskFileDescriptor.local_path(
                    "input.pdf",
                    mime_type="application/pdf",
                    conversions=(TaskFileConversionRequest(name="pdf_image"),),
                ),
            )
            records = await self.store.list_artifacts(result.run.run_id)

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(target.contexts), 1)
        files = target.contexts[0].files
        self.assertEqual(
            [file.logical_path for file in files],
            ["artifact:page-1", "artifact:page-2"],
        )
        self.assertEqual(
            [file.media_type for file in files],
            ["image/png", "image/png"],
        )
        self.assertEqual([file.size_bytes for file in files], [8, 8])
        self.assertEqual(
            [record.purpose for record in records],
            [
                TaskArtifactPurpose.INPUT,
                TaskArtifactPurpose.CONVERTED,
                TaskArtifactPurpose.CONVERTED,
            ],
        )
        self.assertEqual(
            [record.artifact_id for record in records],
            ["source-pdf", "page-1", "page-2"],
        )
        converted = records[1:]
        self.assertEqual(converted[0].metadata["page_index"], 1)
        self.assertEqual(converted[1].metadata["page_index"], 2)
        self.assertEqual(converted[0].metadata["format"], "png")
        identity = cast(
            Mapping[str, object], converted[0].metadata["identity"]
        )
        self.assertEqual(identity["privacy"], "<hmac-sha256>")
        rendered = str(tuple(record.summary() for record in converted))
        self.assertNotIn("private-page.png", rendered)
        self.assertNotIn("%PDF private", rendered)
        assert converted[0].ref.sha256 is not None
        assert converted[1].ref.sha256 is not None
        self.assertNotIn(converted[0].ref.sha256, rendered)
        self.assertNotIn(converted[1].ref.sha256, rendered)

    async def test_page_file_conversion_requires_recording_context(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            materialized = TaskMaterializedFile(
                descriptor=TaskFileDescriptor.artifact(
                    "source-pdf",
                    mime_type="application/pdf",
                    conversions=(TaskFileConversionRequest(name="pdf_image"),),
                ),
                descriptor_path="input",
                ref=TaskArtifactRef(
                    artifact_id="source-pdf",
                    store="local",
                    storage_key="source-pdf",
                    media_type="application/pdf",
                    size_bytes=12,
                ),
                identity={},
            )

            with self.assertRaises(TaskValidationError) as error:
                await task_input_files_from_materialized(
                    definition(
                        input_contract=TaskInputContract.file(
                            conversions=("pdf_image",),
                            mime_types=("application/pdf",),
                        )
                    ),
                    (materialized,),
                    artifact_store=LocalArtifactStore(
                        artifacts,
                        raw_storage_allowed=True,
                    ),
                    file_converters={
                        "pdf_image": PdfPageConverter(
                            (
                                TaskFileConversionPageResult(
                                    page_index=1,
                                    page_count=1,
                                    content=b"page",
                                    media_type="image/png",
                                    width_pixels=10,
                                    height_pixels=10,
                                ),
                            )
                        )
                    },
                    task_store=None,
                )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertNotIn("source-pdf", str(error.exception))

    async def test_shared_input_file_builder_materializes_inputs(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")
            store = InMemoryTaskStore()
            definition_value = definition(
                input_contract=TaskInputContract.file()
            )
            await store.register_definition(
                definition_value,
                definition_hash="shared-builder",
            )
            run = await store.create_run(
                TaskExecutionRequest(definition_id="shared-builder")
            )
            attempt = await store.create_attempt(run.run_id)

            result = await build_task_executable_input_files(
                definition_value,
                TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                ),
                files=(TaskInputFile(logical_path="explicit:file"),),
                roots=(root,),
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-1",
                ),
                hmac_provider=self.hmac_provider,
                task_store=store,
                run=run,
                attempt=attempt,
            )

        self.assertEqual(len(result.files), 2)
        self.assertEqual(result.files[0].logical_path, "explicit:file")
        self.assertEqual(result.files[1].logical_path, "artifact:artifact-1")
        self.assertEqual(len(result.materialized_files), 1)

    async def test_shared_input_file_builder_materializes_remote_url(
        self,
    ) -> None:
        url = "https://example.test/input.txt"
        remote_client = FakeRemoteClient(
            {
                url: TaskRemoteUrlResponse(
                    status_code=200,
                    headers={
                        "Content-Length": "11",
                        "Content-Type": "text/plain",
                    },
                    stream=BytesIO(b"remote text"),
                )
            }
        )
        remote_resolver = FakeRemoteResolver()
        with TemporaryDirectory() as artifacts:
            store = InMemoryTaskStore()
            definition_value = definition(
                input_contract=TaskInputContract.file()
            )
            await store.register_definition(
                definition_value,
                definition_hash="shared-builder-remote",
            )
            run = await store.create_run(
                TaskExecutionRequest(definition_id="shared-builder-remote")
            )
            attempt = await store.create_attempt(run.run_id)

            result = await build_task_executable_input_files(
                definition_value,
                TaskFileDescriptor.remote_url(
                    url,
                    mime_type="text/plain",
                    size_bytes=11,
                ),
                roots=(),
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-remote",
                ),
                hmac_provider=self.hmac_provider,
                task_store=store,
                run=run,
                attempt=attempt,
                remote_url_policy=TaskRemoteUrlPolicy(
                    enabled=True,
                    max_bytes=64,
                ),
                remote_url_http_client=remote_client,
                remote_url_resolver=remote_resolver,
            )

        self.assertEqual(remote_resolver.calls, ["example.test"])
        self.assertEqual(remote_client.calls, [(url, 10.0)])
        self.assertEqual(len(result.files), 1)
        self.assertEqual(
            result.files[0].logical_path,
            "artifact:artifact-remote",
        )
        self.assertEqual(len(result.materialized_files), 1)
        self.assertNotIn("remote text", str(result.materialized_files[0]))

    async def test_shared_input_file_builder_rejects_remote_url_disabled(
        self,
    ) -> None:
        url = "https://example.test/input.txt"
        remote_client = FakeRemoteClient(
            {
                url: TaskRemoteUrlResponse(
                    status_code=200,
                    headers={
                        "Content-Length": "11",
                        "Content-Type": "text/plain",
                    },
                    stream=BytesIO(b"remote text"),
                )
            }
        )
        with TemporaryDirectory() as artifacts:
            with self.assertRaises(TaskValidationError) as error:
                await build_task_executable_input_files(
                    definition(input_contract=TaskInputContract.file()),
                    TaskFileDescriptor.remote_url(
                        url,
                        mime_type="text/plain",
                        size_bytes=11,
                    ),
                    roots=(),
                    artifact_store=LocalArtifactStore(
                        artifacts,
                        raw_storage_allowed=True,
                    ),
                    hmac_provider=self.hmac_provider,
                    remote_url_http_client=remote_client,
                    remote_url_resolver=FakeRemoteResolver(),
                )

        self.assertEqual(
            [issue.code for issue in error.exception.issues],
            ["feature.remote_url_file_inputs_disabled"],
        )
        self.assertEqual(remote_client.calls, [])
        self.assertNotIn(url, str(error.exception))
        self.assertNotIn("remote text", str(error.exception))

    async def test_shared_input_file_builder_rejects_bad_explicit_file(
        self,
    ) -> None:
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            mime_type="application/pdf",
        )
        assert descriptor.provider_reference is not None

        with self.assertRaises(TaskValidationError) as error:
            await build_task_executable_input_files(
                definition(input_contract=TaskInputContract.string()),
                "prompt",
                files=(
                    TaskInputFile(
                        logical_path="provider:openai:provider_file_id",
                        provider_reference=descriptor.provider_reference,
                        media_type="text/plain",
                    ),
                ),
                roots=(),
                artifact_store=None,
                hmac_provider=self.hmac_provider,
            )

        self.assertEqual(
            error.exception.issues[0].path,
            "files[0].provider_reference.mime_type",
        )
        self.assertNotIn("file-private", str(error.exception))

    async def test_execution_file_payload_round_trips_descriptor_fields(
        self,
    ) -> None:
        descriptor = TaskFileDescriptor.provider_reference_descriptor(
            "file-private",
            kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
            provider="openai",
            role="evidence",
            mime_type="application/pdf",
            size_bytes=42,
            sha256="a" * 64,
            owner_scope="tenant-a",
            identity_hmac="hmac-value",
        )
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="artifact-1",
            media_type="application/pdf",
            size_bytes=42,
            sha256="b" * 64,
        )
        entry = TaskExecutableInputFileEntry(
            file=TaskInputFile(
                logical_path="provider:openai:provider_file_id",
                artifact_ref=ref,
                provider_reference=descriptor.provider_reference,
                media_type="application/pdf",
                size_bytes=42,
                metadata={"identity": "safe"},
            ),
            materialized_file=TaskMaterializedFile(
                descriptor=descriptor,
                descriptor_path="input.files[0]",
                ref=ref,
                identity={"identity": "safe"},
            ),
        )

        values = task_execution_file_entries_value((entry,))
        decoded = task_execution_file_entries_from_value(values)

        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].file.logical_path, entry.file.logical_path)
        self.assertIsNotNone(decoded[0].file.provider_reference)
        self.assertIsNotNone(decoded[0].materialized_file)
        assert decoded[0].materialized_file is not None
        decoded_descriptor = decoded[0].materialized_file.descriptor
        self.assertEqual(decoded_descriptor.role, "evidence")
        self.assertEqual(decoded_descriptor.size_bytes, 42)
        self.assertEqual(decoded_descriptor.sha256, "a" * 64)
        self.assertIsNotNone(decoded_descriptor.provider_reference)

    async def test_execution_file_payload_rejects_malformed_values(
        self,
    ) -> None:
        for value in (
            object(),
            (object(),),
            ({"file": {"logical_path": ""}},),
            {"file": {"logical_path": ""}},
        ):
            with self.subTest(value=repr(value)):
                with self.assertRaises(TaskValidationError) as error:
                    task_execution_file_entries_from_value(value)

                self.assertEqual(
                    error.exception.issues[0].code,
                    "queue.file_payload_unavailable",
                )

    async def test_file_conversion_failure_finalizes_run_safely(self) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")
            target = RecordingTarget("unused")
            runner = DirectTaskRunner(
                self.store,
                target=cast(TaskDirectTarget, target),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-1",
                ),
                file_converters={"text": FailingTextConverter()},
                definition_hash=lambda task: "hash-file-conversion-failure",
                execution_roots=(root,),
            )

            result = await runner.run(
                definition(
                    input_contract=TaskInputContract.file(
                        conversions=("text",),
                    )
                ),
                input_value=TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="text"),),
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
            self.assertNotIn("private conversion failure", str(error_summary))
            self.assertNotIn("input.txt", str(error_summary))

    async def test_missing_file_converter_rejects_before_run(
        self,
    ) -> None:
        with TemporaryDirectory() as root, TemporaryDirectory() as artifacts:
            Path(root, "input.txt").write_bytes(b"private text")
            target = RecordingTarget("unused")
            runner = DirectTaskRunner(
                self.store,
                target=cast(TaskDirectTarget, target),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-1",
                ),
                definition_hash=lambda task: "hash-missing-file-converter",
                execution_roots=(root,),
            )

            with self.assertRaises(TaskValidationError) as error:
                await runner.run(
                    definition(
                        input_contract=TaskInputContract.file(
                            conversions=("unknown",),
                        )
                    ),
                    input_value=TaskFileDescriptor.local_path(
                        "input.txt",
                        mime_type="text/plain",
                        conversions=(
                            TaskFileConversionRequest(name="unknown"),
                        ),
                    ),
                )

            self.assertEqual(target.contexts, [])
            self.assertEqual(
                error.exception.issues[0].code,
                "input.invalid_file",
            )
            self.assertEqual(
                error.exception.issues[0].path,
                "input.file_conversions[0]",
            )
            self.assertNotIn("unknown", str(error.exception))
            self.assertNotIn("input.txt", str(error.exception))

    async def test_converted_input_file_rejects_missing_converter_guard(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            self.store,
            target=cast(TaskDirectTarget, RecordingTarget("unused")),
            hmac_provider=self.hmac_provider,
            file_converters={},
        )
        definition_value = definition(
            input_contract=TaskInputContract.file(
                conversions=("unknown",),
            )
        )
        await self.store.register_definition(
            definition_value,
            definition_hash="definition-1",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="definition-1")
        )
        attempt = await self.store.create_attempt(run.run_id)
        materialized = TaskMaterializedFile(
            descriptor=TaskFileDescriptor.local_path(
                "input.txt",
                conversions=(TaskFileConversionRequest(name="unknown"),),
            ),
            descriptor_path="input",
            ref=TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="artifact-1",
                media_type="text/plain",
                size_bytes=4,
            ),
            identity={},
        )

        with self.assertRaises(TaskValidationError) as error:
            await runner._converted_input_file(
                definition_value,
                materialized,
                index=0,
                run=run,
                attempt=attempt,
            )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertEqual(
            error.exception.issues[0].path,
            "input.conversions[0]",
        )
        self.assertNotIn("unknown", str(error.exception))

    async def test_converted_input_file_wrapper_handles_none_and_scalar(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_ids = iter(("source-text", "converted-text"))
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            source_ref = await artifact_store.put(
                b"private text",
                media_type="text/plain",
            )
            runner = DirectTaskRunner(
                self.store,
                target=cast(TaskDirectTarget, RecordingTarget("unused")),
                hmac_provider=self.hmac_provider,
                artifact_store=artifact_store,
                file_converters={"text": PrefixTextConverter()},
            )
            definition_value = definition(
                input_contract=TaskInputContract.file(conversions=("text",))
            )
            await self.store.register_definition(
                definition_value,
                definition_hash="definition-scalar-wrapper",
            )
            run = await self.store.create_run(
                TaskExecutionRequest(definition_id="definition-scalar-wrapper")
            )
            attempt = await self.store.create_attempt(run.run_id)
            native = TaskMaterializedFile(
                descriptor=TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                ),
                descriptor_path="input",
                ref=source_ref,
                identity={},
            )
            converted = TaskMaterializedFile(
                descriptor=TaskFileDescriptor.local_path(
                    "input.txt",
                    mime_type="text/plain",
                    conversions=(TaskFileConversionRequest(name="text"),),
                ),
                descriptor_path="input",
                ref=source_ref,
                identity={},
            )

            self.assertIsNone(
                await runner._converted_input_file(
                    definition_value,
                    native,
                    index=0,
                    run=run,
                    attempt=attempt,
                )
            )
            file = await runner._converted_input_file(
                definition_value,
                converted,
                index=0,
                run=run,
                attempt=attempt,
            )

        assert file is not None
        self.assertEqual(file.logical_path, "artifact:converted-text")
        self.assertEqual(file.media_type, "text/plain")
        self.assertEqual(file.size_bytes, len(b"converted:private text"))

    async def test_converted_input_file_wrapper_rejects_page_collection(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_ids = iter(("source-pdf", "page-1", "page-2"))
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=lambda: next(artifact_ids),
            )
            source_ref = await artifact_store.put(
                b"%PDF private",
                media_type="application/pdf",
            )
            runner = DirectTaskRunner(
                self.store,
                target=cast(TaskDirectTarget, RecordingTarget("unused")),
                hmac_provider=self.hmac_provider,
                artifact_store=artifact_store,
                file_converters={
                    "pdf_image": PdfPageConverter(
                        (
                            TaskFileConversionPageResult(
                                page_index=1,
                                page_count=2,
                                content=b"page one",
                                media_type="image/png",
                                width_pixels=10,
                                height_pixels=20,
                            ),
                            TaskFileConversionPageResult(
                                page_index=2,
                                page_count=2,
                                content=b"page two",
                                media_type="image/png",
                                width_pixels=10,
                                height_pixels=20,
                            ),
                        )
                    )
                },
            )
            definition_value = definition(
                input_contract=TaskInputContract.file(
                    conversions=("pdf_image",),
                    mime_types=("application/pdf",),
                )
            )
            await self.store.register_definition(
                definition_value,
                definition_hash="definition-page-wrapper",
            )
            run = await self.store.create_run(
                TaskExecutionRequest(definition_id="definition-page-wrapper")
            )
            attempt = await self.store.create_attempt(run.run_id)
            materialized = TaskMaterializedFile(
                descriptor=TaskFileDescriptor.local_path(
                    "input.pdf",
                    mime_type="application/pdf",
                    conversions=(TaskFileConversionRequest(name="pdf_image"),),
                ),
                descriptor_path="input",
                ref=source_ref,
                identity={},
            )

            with self.assertRaises(TaskValidationError) as error:
                await runner._converted_input_file(
                    definition_value,
                    materialized,
                    index=0,
                    run=run,
                    attempt=attempt,
                )

        self.assertEqual(error.exception.issues[0].code, "input.invalid_file")
        self.assertNotIn("%PDF private", str(error.exception))

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
            self.assertIn("privacy", records[0].ref.metadata)
            self.assertNotIn("report", str(records[0].ref.metadata))
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

    async def test_artifact_output_sanitizes_record_metadata(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            runner = DirectTaskRunner(
                self.store,
                target=PrivateArtifactRecordTarget(),
                hmac_provider=self.hmac_provider,
                artifact_store=LocalArtifactStore(
                    artifacts,
                    raw_storage_allowed=True,
                    id_factory=lambda: "artifact-private-1",
                ),
                definition_hash=lambda task: "hash-artifact-metadata",
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
            self.assertEqual(records[0].metadata, {"state": "simulated"})
            self.assertEqual(
                records[0].ref.metadata,
                {"privacy": "<redacted>"},
            )
            self.assertEqual(
                records[0].provenance.metadata,
                {"privacy": "<redacted>"},
            )
            self.assertEqual(
                records[0].retention.metadata,
                {"privacy": "<redacted>"},
            )
            persisted = str(records)
            self.assertNotIn("private-ref.txt", persisted)
            self.assertNotIn("private-output.txt", persisted)
            self.assertNotIn("private prompt", persisted)
            self.assertNotIn("private retention", persisted)

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

    async def test_runner_noop_observability_branches(self) -> None:
        task = definition(observability=TaskObservabilityPolicy.noop())
        runner = DirectTaskRunner(
            self.store,
            target=RecordingTarget("ok"),
            hmac_provider=self.hmac_provider,
            definition_hash=lambda definition: "hash-observability-noop",
        )
        await self.store.register_definition(
            task,
            definition_hash="definition-observability-noop",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(definition_id="definition-observability-noop")
        )
        attempt = await self.store.create_attempt(run.run_id)

        self.assertIsNone(runner._observability_sink_for(task))
        self.assertIsNone(
            runner._event_pipeline(
                task,
                run=run,
                attempt=attempt,
                sanitizer=runner._sanitizer(task),
            )
        )

    def test_queue_file_entries_and_conversion_values_round_trip(
        self,
    ) -> None:
        request = TaskFileConversionRequest(
            name="text",
            options={"mode": "safe"},
        )
        source = TaskMaterializedFile(
            descriptor=TaskFileDescriptor.artifact(
                "artifact-source",
                conversions=(request,),
                mime_type="application/pdf",
            ),
            descriptor_path="input",
            ref=TaskArtifactRef(
                artifact_id="artifact-source",
                store="local",
                storage_key="ar/artifact-source",
                media_type="application/pdf",
                size_bytes=4,
            ),
            identity={"privacy": "<hmac-sha256>"},
        )
        entries = task_input_file_entries_for_queue(
            files=(TaskInputFile(logical_path="inline:file"),),
            provider_reference_files=(),
            materialized_files=(source,),
        )

        value = _conversion_request_value(request)
        restored = _conversion_request_from_value(value)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].file.logical_path, "inline:file")
        self.assertEqual(entries[1].materialized_file, source)
        self.assertEqual(restored.name, "text")
        self.assertEqual(restored.options, {"mode": "safe"})

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
        conversions = cast(
            tuple[Mapping[str, object], ...], first["conversions"]
        )
        self.assertEqual(first["reference"], {"privacy": "<redacted>"})
        self.assertEqual(conversions[0]["options"], {"privacy": "<redacted>"})
        self.assertEqual(first["metadata"], {"privacy": "<redacted>"})
        self.assertNotIn("input-a.txt", str(first))
        self.assertNotIn("private", str(first))

    def test_file_input_summary_returns_original_for_malformed_value(
        self,
    ) -> None:
        value = {"source_kind": "local_path"}

        self.assertIs(_input_summary_value(definition(), value), value)

    def test_file_input_summary_returns_original_without_descriptor(
        self,
    ) -> None:
        self.assertIs(
            _input_summary_value(
                definition(input_contract=TaskInputContract.file()),
                None,
            ),
            None,
        )

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

    def test_late_schema_resolution_failure_is_sanitized(self) -> None:
        runner = DirectTaskRunner(
            InMemoryTaskStore(),
            target=RecordingTarget("unused"),
            hmac_provider=StaticHmacProvider(),
        )

        with self.assertRaises(TaskValidationError) as error:
            runner._resolve_definition_schemas(
                definition(
                    input_contract=TaskInputContract.object(
                        schema_ref="schemas/private-input.json"
                    )
                )
            )

        self.assertEqual(
            [(issue.code, issue.path) for issue in error.exception.issues],
            [("input.invalid_schema", "input.schema_ref")],
        )
        self.assertNotIn("private-input", str(error.exception))


def _record_sleep(delays: list[float]) -> Callable[[float], Awaitable[None]]:
    async def sleep(delay: float) -> None:
        delays.append(delay)

    return sleep


if __name__ == "__main__":
    main()
