from asyncio import CancelledError, Event, create_task, sleep
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.container import (
    ContainerBackend,
    ContainerBackendCapabilities,
    ContainerBackendOperation,
    ContainerDurablePlanMetadata,
    ContainerEffectiveSettings,
    ContainerExecutionScope,
    ContainerFakeBackend,
    ContainerFakeBackendScript,
    ContainerMountType,
    ContainerOutputArtifact,
    ContainerOutputContract,
    ContainerOutputContractType,
    ContainerOutputDecisionType,
    ContainerOutputPolicy,
    ContainerOutputValidationResult,
    ContainerProfile,
    ContainerProfileSelection,
    ContainerResultStatus,
    ContainerRunPlan,
    ContainerSettings,
    ContainerSettingsSource,
    ContainerSurface,
    ContainerTrustLevel,
    run_container_managed_lifecycle,
)
from avalan.task import (
    DirectTaskRunner,
    IdempotencyMode,
    RetryBackoff,
    SanitizedTaskEvent,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskClient,
    TaskContainerExecutionSettings,
    TaskDefinition,
    TaskEventCategory,
    TaskEventValue,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskFileDescriptor,
    TaskInputContract,
    TaskInputFile,
    TaskInputType,
    TaskKeyMaterial,
    TaskKeyPurpose,
    TaskMetadata,
    TaskObservabilityPolicy,
    TaskOutputArtifact,
    TaskOutputContract,
    TaskProviderReference,
    TaskProviderReferenceKind,
    TaskQueueAbandonment,
    TaskQueueArtifact,
    TaskQueueClaim,
    TaskQueueCompletion,
    TaskQueueDepth,
    TaskQueueHealth,
    TaskQueueItem,
    TaskQueueItemState,
    TaskQueueRetry,
    TaskQueueSubmission,
    TaskRetryPolicy,
    TaskRunPolicy,
    TaskRunState,
    TaskTargetContext,
    TaskTargetRunner,
    TaskValidationCategory,
    TaskValidationContext,
    TaskValidationError,
    TaskValidationIssue,
    TaskWorker,
)
from avalan.task.artifacts.local import LocalArtifactStore
from avalan.task.canonical import canonical_definition, spec_hash
from avalan.task.container import (
    TASK_CONTAINER_ATTEMPT_KEY,
    TASK_CONTAINER_METADATA_KEY,
    TASK_CONTAINER_WORKER_ENVELOPE_KEY,
    TaskContainerPlans,
    TaskContainerVerificationError,
    task_container_event_payload,
    task_container_input_mount_manifest,
    task_container_lifecycle_run_plan,
    task_container_output_artifacts,
    task_container_output_contract,
    task_container_plans,
    task_container_request_metadata,
    task_container_run_metadata,
    verify_task_container_request,
)
from avalan.task.event import task_event_category
from avalan.task.idempotency import TaskIdempotencyIdentity
from avalan.task.runner import TaskContainerAttemptResult
from avalan.task.state import TaskAttemptState
from avalan.task.store import TaskStoreConflictError
from avalan.task.stores import InMemoryTaskStore
from avalan.task.validation import validate_task_definition

_IMAGE = "registry.example/task@sha256:" + ("1" * 64)
_OUTPUT_DIGEST = (
    "57aa86c72be6c88e37be171f86f25951c4204ec73391361a00ee97f2d85a488f"
)
_DEFAULT_ATTEMPT = object()


class RecordingTarget(TaskTargetRunner):
    def __init__(self, output: object = "done") -> None:
        self.output = output
        self.contexts: list[TaskTargetContext] = []

    async def validate_definition(
        self,
        definition: TaskDefinition,
        context: TaskValidationContext,
    ) -> tuple[TaskValidationIssue, ...]:
        return ()

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        await context.check_cancelled()
        return self.output


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
            secret=b"phase13-secret",
        )


class FailingThenSucceedingTarget(RecordingTarget):
    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        if len(self.contexts) == 1:
            raise RuntimeError("first attempt")
        return self.output


class WaitingTarget(RecordingTarget):
    def __init__(self) -> None:
        super().__init__("done")
        self.started = Event()

    async def run(self, context: TaskTargetContext) -> object:
        self.contexts.append(context)
        self.started.set()
        while True:
            await sleep(0)
            await context.check_cancelled()


class InitialCheckFailureRunner(DirectTaskRunner):
    def __init__(
        self,
        store: InMemoryTaskStore,
        *,
        error: BaseException,
    ) -> None:
        super().__init__(
            store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
        )
        self.error = error
        self.checks = 0

    async def _check_cancellation_or_expiry(
        self,
        run: object,
        expires_at: datetime | None,
    ) -> None:
        _ = run
        _ = expires_at
        self.checks += 1
        if self.checks == 1:
            raise self.error


class CancelAfterPlanStore(InMemoryTaskStore):
    async def append_event(
        self,
        run_id: str,
        *,
        event_type: str,
        category: TaskEventCategory,
        payload: TaskEventValue,
        attempt_id: str | None = None,
    ) -> SanitizedTaskEvent:
        event = await super().append_event(
            run_id,
            event_type=event_type,
            category=category,
            payload=payload,
            attempt_id=attempt_id,
        )
        if event_type == "container_plan_verified":
            await self.transition_run(
                run_id,
                from_states={TaskRunState.RUNNING},
                to_state=TaskRunState.CANCEL_REQUESTED,
                reason="cancel_requested",
            )
        return event


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2027, 1, 1, tzinfo=UTC)


class SingleItemQueue:
    def __init__(self, store: InMemoryTaskStore, clock: Clock) -> None:
        self.store = store
        self.clock = clock
        self.item: TaskQueueItem | None = None
        self.completed: TaskQueueCompletion | None = None
        self.retried: TaskQueueRetry | None = None
        self.abandoned: TaskQueueAbandonment | None = None

    async def enqueue_run(
        self,
        request: TaskExecutionRequest,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        idempotency: TaskIdempotencyIdentity | None = None,
        idempotency_expires_at: datetime | None = None,
        artifacts: tuple[TaskQueueArtifact, ...] = (),
        run_metadata: Mapping[str, object] | None = None,
        queue_metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueSubmission:
        _ = idempotency
        _ = idempotency_expires_at
        run = await self.store.create_run(request, metadata=run_metadata)
        records = []
        for artifact in artifacts:
            records.append(
                await self.store.append_artifact(
                    run.run_id,
                    ref=artifact.ref,
                    purpose=artifact.purpose,
                    state=artifact.state,
                    provenance=artifact.provenance,
                    retention=artifact.retention,
                    metadata=artifact.metadata,
                )
            )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        now = self.clock.now
        self.item = TaskQueueItem(
            queue_item_id="queue-item-1",
            run_id=run.run_id,
            queue_name=queue_name,
            state=TaskQueueItemState.AVAILABLE,
            priority=priority,
            available_at=available_at or now,
            attempts=0,
            created_at=now,
            updated_at=now,
            run_state=run.state,
            metadata=queue_metadata or {},
        )
        return TaskQueueSubmission(
            run=run,
            created=True,
            queue_item=self.item,
            artifacts=tuple(records),
        )

    async def enqueue(
        self,
        run_id: str,
        *,
        queue_name: str,
        priority: int = 0,
        available_at: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueItem:
        raise AssertionError("enqueue should not be used")

    async def claim(
        self,
        queue_name: str,
        *,
        worker_id: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueClaim | None:
        if (
            self.item is None
            or self.item.queue_name != queue_name
            or self.item.state != TaskQueueItemState.AVAILABLE
        ):
            return None
        run = await self.store.assign_claim(
            self.item.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id=worker_id,
            lease_expires_at=lease_expires_at,
            reason="claimed",
            metadata=metadata,
        )
        claim_token = run.claim.claim_token if run.claim else ""
        attempt = await self.store.create_attempt(
            run.run_id,
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = replace(
            self.item,
            state=TaskQueueItemState.CLAIMED,
            updated_at=now or self.clock.now,
            run_state=run.state,
            claimed_at=run.claim.claimed_at if run.claim else None,
            lease_expires_at=run.claim.lease_expires_at if run.claim else None,
            worker_id=worker_id,
            claim_token=claim_token,
            heartbeat_at=run.claim.heartbeat_at if run.claim else None,
            metadata=metadata or {},
        )
        return TaskQueueClaim(queue_item=self.item, run=run, attempt=attempt)

    async def heartbeat(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        lease_expires_at: datetime,
        now: datetime | None = None,
    ) -> TaskQueueItem:
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        assert self.item.claim_token == claim_token
        self.item = replace(
            self.item,
            lease_expires_at=lease_expires_at,
            heartbeat_at=now or self.clock.now,
            updated_at=now or self.clock.now,
        )
        return self.item

    async def complete(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        run_state: TaskRunState,
        attempt_state: TaskAttemptState,
        result: TaskExecutionResult | None = None,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueCompletion:
        assert self.item is not None
        assert self.item.queue_item_id == queue_item_id
        run = await self.store.get_run(self.item.run_id)
        attempt = await self.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=attempt_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        completed_run = await self.store.transition_run(
            run.run_id,
            from_states={run.state},
            to_state=run_state,
            reason="completed",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        self.item = TaskQueueItem(
            queue_item_id=self.item.queue_item_id,
            run_id=self.item.run_id,
            queue_name=self.item.queue_name,
            state=(
                TaskQueueItemState.DONE
                if run_state == TaskRunState.SUCCEEDED
                else TaskQueueItemState.DEAD
            ),
            priority=self.item.priority,
            available_at=self.item.available_at,
            attempts=self.item.attempts,
            created_at=self.item.created_at,
            updated_at=now or self.clock.now,
            run_state=completed_run.state,
        )
        self.completed = TaskQueueCompletion(
            queue_item=self.item,
            run=completed_run,
            attempt=attempt,
        )
        return self.completed

    async def retry(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        result: TaskExecutionResult,
        available_at: datetime,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueRetry:
        assert self.item is not None
        run = await self.store.get_run(self.item.run_id)
        attempt = await self.store.transition_attempt(
            run.last_attempt_id or "",
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="retry",
            result=result,
            claim_token=claim_token,
            metadata=metadata,
        )
        queued_run = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.QUEUED,
            reason="retry",
            claim_token=claim_token,
            metadata=metadata,
        )
        queued_run = replace(queued_run, claim=None)
        self.store._runs[run.run_id] = queued_run
        exhausted = self.item.attempts + 1 >= max_attempts
        self.item = replace(
            self.item,
            state=(
                TaskQueueItemState.DEAD
                if exhausted
                else TaskQueueItemState.AVAILABLE
            ),
            attempts=self.item.attempts + 1,
            available_at=available_at,
            updated_at=now or self.clock.now,
            run_state=(
                TaskRunState.FAILED if exhausted else TaskRunState.QUEUED
            ),
            claimed_at=None,
            lease_expires_at=None,
            worker_id=None,
            claim_token=None,
            heartbeat_at=None,
        )
        self.retried = TaskQueueRetry(
            queue_item=self.item,
            run=queued_run,
            attempt=attempt,
        )
        return self.retried

    async def abandon(
        self,
        queue_item_id: str,
        *,
        claim_token: str,
        max_attempts: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskQueueAbandonment:
        raise AssertionError("abandon should not be used")

    async def abandon_expired(
        self,
        queue_name: str,
        *,
        max_attempts: int,
        limit: int,
        now: datetime | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> tuple[TaskQueueAbandonment, ...]:
        return ()

    async def drain(
        self,
        queue_name: str,
        *,
        limit: int,
        now: datetime | None = None,
    ) -> tuple[TaskQueueItem, ...]:
        return ()

    async def depth(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueDepth:
        return TaskQueueDepth(
            queue_name=queue_name,
            available=1 if self.item is not None else 0,
            scheduled=0,
            claimed=0,
            dead=0,
            cancel_requested=0,
        )

    async def health(
        self,
        queue_name: str,
        *,
        now: datetime | None = None,
    ) -> TaskQueueHealth:
        return TaskQueueHealth(
            queue_name=queue_name,
            depth=await self.depth(queue_name, now=now),
            checked_at=now or self.clock.now,
        )


class TaskContainerExecutionTest(IsolatedAsyncioTestCase):
    async def test_container_settings_canonicalize_policy_identity(
        self,
    ) -> None:
        definition = _definition(container=_container_settings())
        self.assertTrue(definition.container.enabled)
        self.assertFalse(TaskContainerExecutionSettings().enabled)
        self.assertEqual(
            task_event_category("container_plan_verified"),
            TaskEventCategory.UNKNOWN,
        )
        canonical = await canonical_definition(definition)
        container = cast(dict[str, object], canonical["container"])
        attempt = cast(dict[str, object], container["attempt"])
        attempt_spec = cast(dict[str, object], container["attempt_spec"])

        self.assertEqual(attempt["profile_registry_id"], "task-registry")
        self.assertEqual(attempt["policy_version"], "policy-v1")
        self.assertEqual(attempt_spec["profile_name"], "strict")
        self.assertEqual(
            TaskContainerExecutionSettings.from_dict(
                definition.container.to_dict()
            ).to_dict(),
            definition.container.to_dict(),
        )

        changed = _definition(
            container=_container_settings(
                attempt=_effective_settings(policy_version="policy-v2")
            )
        )
        self.assertNotEqual(
            await spec_hash(definition),
            await spec_hash(changed),
        )

    async def test_container_metadata_helpers_cover_envelope_and_mounts(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition(
            container=_container_settings(
                attempt=None,
                worker_envelope=_effective_settings(),
            )
        )
        await store.register_definition(
            definition,
            definition_hash=await spec_hash(definition),
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(definition),
                metadata=task_container_request_metadata(definition),
            )
        )
        attempt = await store.create_attempt(run.run_id)
        plans = task_container_plans(definition, run=run, attempt=attempt)
        metadata = plans.to_metadata()

        self.assertIn(TASK_CONTAINER_WORKER_ENVELOPE_KEY, metadata)
        self.assertNotIn(TASK_CONTAINER_ATTEMPT_KEY, metadata)

        attempt_definition = _definition(container=_container_settings())
        await store.register_definition(
            attempt_definition,
            definition_hash=await spec_hash(attempt_definition),
        )
        attempt_run = await store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(attempt_definition),
                metadata=task_container_request_metadata(attempt_definition),
            )
        )
        attempt_attempt = await store.create_attempt(attempt_run.run_id)
        self.assertIn(
            TASK_CONTAINER_ATTEMPT_KEY,
            task_container_plans(
                attempt_definition,
                run=attempt_run,
                attempt=attempt_attempt,
            ).to_metadata(),
        )

        payload = task_container_event_payload(
            status="verified",
            plans=plans,
            input_mounts=(
                {"source_kind": "artifact"},
                {"source_kind": "provider"},
            ),
        )
        self.assertEqual(payload["mount_count"], 2)
        self.assertEqual(payload["artifact_count"], 1)
        self.assertIn("worker_envelope_plan_fingerprint", payload)

        manifest = task_container_input_mount_manifest(
            (
                TaskInputFile(
                    logical_path="plain.txt",
                    media_type="text/plain",
                ),
                TaskInputFile(
                    logical_path="inline:bytes",
                    media_type="text/plain",
                    size_bytes=4,
                ),
                TaskInputFile(
                    logical_path="provider:file",
                    provider_reference=TaskProviderReference(
                        kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                        provider="provider",
                        reference="file-1",
                    ),
                ),
            )
        )
        self.assertEqual(
            [item["source_kind"] for item in manifest],
            [
                "inline",
                "provider",
            ],
        )

    async def test_container_helper_fail_closed_edges(self) -> None:
        self.assertIsNone(
            task_container_lifecycle_run_plan(TaskContainerPlans())
        )
        self.assertIsNone(
            task_container_output_contract(
                _container_file_definition(),
                TaskContainerPlans(),
            )
        )

        store = InMemoryTaskStore()
        definition = _container_file_definition()
        await store.register_definition(
            definition,
            definition_hash=await spec_hash(definition),
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(definition),
                metadata=task_container_request_metadata(definition),
            )
        )
        attempt = await store.create_attempt(run.run_id)
        plans = task_container_plans(definition, run=run, attempt=attempt)
        disabled = replace(
            cast(
                ContainerOutputContract,
                task_container_output_contract(
                    definition,
                    plans,
                ),
            ),
            enabled=False,
        )
        self.assertIsNone(
            task_container_output_contract(
                _definition(
                    container=_container_settings(allow_artifacts=True),
                    output=TaskOutputContract.text(),
                ),
                plans,
            )
        )
        self.assertEqual(disabled.enabled, False)

        no_attempt_settings = _definition(
            container=TaskContainerExecutionSettings(attempt=None),
            output=TaskOutputContract.file(),
        )
        self.assertIsNone(
            task_container_output_contract(
                no_attempt_settings,
                plans,
            )
        )
        with patch(
            "avalan.task.container.output_contracts_from_policy",
            return_value=(),
        ):
            self.assertIsNone(
                task_container_output_contract(definition, plans)
            )

        worker_only = _definition(
            container=_container_settings(
                attempt=None,
                worker_envelope=_effective_settings(),
            )
        )
        await store.register_definition(
            worker_only,
            definition_hash=await spec_hash(worker_only),
        )
        worker_run = await store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(worker_only),
                metadata={
                    TASK_CONTAINER_METADATA_KEY: (
                        task_container_request_metadata(
                            worker_only,
                            input_mounts=(
                                {
                                    "source_kind": "provider",
                                    "target": "/inputs/0",
                                },
                            ),
                        )
                    )
                },
            )
        )
        worker_attempt = await store.create_attempt(worker_run.run_id)
        worker_plans = verify_task_container_request(
            worker_only,
            run=worker_run,
            attempt=worker_attempt,
            input_mounts=(
                {
                    "source_kind": "provider",
                    "target": "/inputs/0",
                },
            ),
        )
        self.assertIsNotNone(task_container_lifecycle_run_plan(worker_plans))

        invalid_manifest = task_container_input_mount_manifest(
            (
                TaskInputFile(
                    logical_path="artifact:input-artifact",
                    artifact_ref=TaskArtifactRef(
                        artifact_id="input-artifact",
                        store="local",
                        storage_key="input-artifact",
                        metadata={"container_mount_source": "\0"},
                    ),
                ),
            ),
            allowed_roots=("/tmp/task-inputs",),
        )
        self.assertNotIn("source", invalid_manifest[0])

        with self.assertRaises(TaskContainerVerificationError) as missing:
            await task_container_output_artifacts(
                definition,
                (_container_output_artifact(),),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=None,
            )
        self.assertEqual(
            missing.exception.code,
            "container.output_unsupported",
        )

        with self.assertRaises(TaskContainerVerificationError) as count:
            await task_container_output_artifacts(
                definition,
                (_container_output_artifact(), _container_output_artifact()),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=_artifact_store(self),
            )
        self.assertEqual(count.exception.path, "container.output")

        with self.assertRaises(TaskContainerVerificationError) as no_bytes:
            await task_container_output_artifacts(
                definition,
                (_container_output_artifact(content=None),),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=_artifact_store(self),
            )
        self.assertEqual(no_bytes.exception.path, "container.output")

        with self.assertRaises(TaskContainerVerificationError) as digest:
            await task_container_output_artifacts(
                definition,
                (_container_output_artifact(digest="0" * 64),),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=_artifact_store(self),
            )
        self.assertEqual(digest.exception.path, "container.output.digest")

        artifact_array_definition = replace(
            definition,
            output=TaskOutputContract.artifact_array(),
        )
        limited = replace(
            artifact_array_definition,
            artifact=replace(definition.artifact, max_count=1),
        )
        with self.assertRaises(TaskContainerVerificationError) as max_count:
            await task_container_output_artifacts(
                limited,
                (_container_output_artifact(), _container_output_artifact()),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=_artifact_store(self),
            )
        self.assertEqual(max_count.exception.path, "container.output")

        byte_limited = replace(
            artifact_array_definition,
            artifact=replace(definition.artifact, max_bytes=1),
        )
        with self.assertRaises(TaskContainerVerificationError) as max_bytes:
            await task_container_output_artifacts(
                byte_limited,
                (_container_output_artifact(),),
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                artifact_store=_artifact_store(self),
            )
        self.assertEqual(max_bytes.exception.path, "container.output")

        array_output = await task_container_output_artifacts(
            artifact_array_definition,
            (_container_output_artifact(),),
            run_id=run.run_id,
            attempt_id=attempt.attempt_id,
            artifact_store=_artifact_store(self),
        )
        self.assertIsInstance(array_output, tuple)

    async def test_container_request_verification_rejects_bad_metadata(
        self,
    ) -> None:
        definition = _definition(container=_container_settings())

        with self.subTest("missing top-level"):
            error = await _verification_error(definition, {})
            self.assertEqual(error.code, "container.plan_missing")
            self.assertEqual(error.path, TASK_CONTAINER_METADATA_KEY)

        with self.subTest("missing attempt"):
            error = await _verification_error(
                definition,
                {TASK_CONTAINER_METADATA_KEY: {}},
            )
            self.assertEqual(error.code, "container.plan_missing")
            self.assertEqual(
                error.path,
                f"{TASK_CONTAINER_METADATA_KEY}.{TASK_CONTAINER_ATTEMPT_KEY}",
            )

        with self.subTest("stale policy"):
            metadata = task_container_request_metadata(definition)
            container = dict(cast(Mapping[str, object], metadata))
            attempt = dict(
                cast(
                    Mapping[str, object],
                    container[TASK_CONTAINER_ATTEMPT_KEY],
                )
            )
            attempt["policy_version"] = "stale"
            container[TASK_CONTAINER_ATTEMPT_KEY] = attempt
            error = await _verification_error(
                definition,
                {TASK_CONTAINER_METADATA_KEY: container},
            )
            self.assertEqual(error.code, "container.plan_mismatch")
            self.assertEqual(
                error.path,
                f"{TASK_CONTAINER_METADATA_KEY}."
                f"{TASK_CONTAINER_ATTEMPT_KEY}.policy_version",
            )

        with self.subTest("unexpected envelope"):
            metadata = task_container_request_metadata(definition)
            container = dict(cast(Mapping[str, object], metadata))
            container[TASK_CONTAINER_WORKER_ENVELOPE_KEY] = {
                "profile_name": "strict"
            }
            error = await _verification_error(
                definition,
                {TASK_CONTAINER_METADATA_KEY: container},
            )
            self.assertEqual(error.code, "container.plan_unexpected")

    async def test_container_policy_validation_reports_required_gaps(
        self,
    ) -> None:
        definition = _definition(
            container=TaskContainerExecutionSettings(
                attempt=_disabled_required_settings(),
                worker_envelope=_enabled_without_profile_settings(),
            )
        )

        issues = validate_task_definition(
            definition,
            hmac_provider=StaticHmacProvider(),
        )

        self.assertEqual(
            [(issue.code, issue.path) for issue in issues],
            [
                ("container.backend_required", "container.attempt.backend"),
                (
                    "container.profile_unknown",
                    "container.worker_envelope.profile_name",
                ),
            ],
        )

    async def test_direct_container_execution_verifies_and_records_events(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget()
        backend = _backend(output_result=_container_output_result())
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=backend,
        )

        result = await runner.run(
            _container_file_definition(),
            input_value="hello",
            metadata={"user": "visible"},
        )

        self.assertIsInstance(result.output, TaskOutputArtifact)
        self.assertEqual(target.contexts, [])
        self.assertIn(ContainerBackendOperation.PROBE, backend.operations)
        self.assertIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertIn(
            ContainerBackendOperation.COPY_OUTPUTS,
            backend.operations,
        )
        artifacts = await store.list_artifacts(
            result.run.run_id,
            purpose=TaskArtifactPurpose.OUTPUT,
        )
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].ref.store, "local")
        self.assertEqual(
            result.output.metadata["container_path"],
            "result.txt",
        )
        events = await store.list_events(result.run.run_id)
        self.assertEqual(
            [event.event_type for event in events],
            [
                "container_plan_verified",
                "container_lifecycle_completed",
            ],
        )
        self.assertEqual(events[0].payload["status"], "verified")
        self.assertEqual(events[0].payload["profile_name"], "strict")

    async def test_container_input_mounts_are_in_lifecycle_plan(
        self,
    ) -> None:
        captured_plans: list[object] = []

        async def capture_lifecycle(
            backend: object,
            plan: object,
            **kwargs: object,
        ) -> object:
            captured_plans.append(plan)
            return await run_container_managed_lifecycle(
                backend,
                plan,
                **kwargs,
            )

        store = InMemoryTaskStore()
        runner = DirectTaskRunner(
            store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            execution_roots=("/tmp/task-inputs",),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        input_file = TaskInputFile(
            logical_path="artifact:input-artifact",
            artifact_ref=_input_artifact_ref(),
            media_type="text/plain",
            size_bytes=5,
        )

        with patch(
            "avalan.task.runner.run_container_managed_lifecycle",
            capture_lifecycle,
        ):
            result = await runner.run(
                _container_file_definition(file_input=True),
                files=(input_file,),
            )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(len(captured_plans), 1)
        plan = captured_plans[0]
        input_mounts = [
            mount
            for mount in plan.mounts
            if mount.mount_type is ContainerMountType.INPUT
        ]
        assert input_file.artifact_ref is not None
        self.assertEqual(len(input_mounts), 1)
        self.assertEqual(
            input_mounts[0].source,
            input_file.artifact_ref.storage_key,
        )
        self.assertEqual(input_mounts[0].target, "/inputs/0")

    async def test_direct_materialized_input_mounts_use_dynamic_plan(
        self,
    ) -> None:
        captured_plans: list[object] = []

        async def capture_lifecycle(
            backend: object,
            plan: object,
            **kwargs: object,
        ) -> object:
            captured_plans.append(plan)
            return await run_container_managed_lifecycle(
                backend,
                plan,
                **kwargs,
            )

        with TemporaryDirectory() as input_root:
            with TemporaryDirectory() as artifact_root:
                Path(input_root, "input.txt").write_text("hello")
                artifact_store = LocalArtifactStore(
                    artifact_root,
                    raw_storage_allowed=True,
                )
                runner = DirectTaskRunner(
                    InMemoryTaskStore(),
                    target=RecordingTarget(),
                    hmac_provider=StaticHmacProvider(),
                    artifact_store=artifact_store,
                    input_roots=(input_root,),
                    execution_roots=(artifact_root,),
                    container_backend=_backend(
                        output_result=_container_output_result()
                    ),
                )
                with patch(
                    "avalan.task.runner.run_container_managed_lifecycle",
                    capture_lifecycle,
                ):
                    result = await runner.run(
                        _container_file_definition(file_input=True),
                        input_value=TaskFileDescriptor.local_path(
                            "input.txt",
                            mime_type="text/plain",
                        ),
                    )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        plan = cast(ContainerRunPlan, captured_plans[0])
        input_mounts = [
            mount
            for mount in plan.mounts
            if mount.mount_type == ContainerMountType.INPUT
        ]
        self.assertEqual(len(input_mounts), 1)
        Path(input_mounts[0].source).resolve(strict=False).relative_to(
            Path(artifact_root).resolve(strict=False)
        )

    async def test_container_backend_capability_mismatch_fails_closed(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget()
        backend = _backend(mount_types=(ContainerMountType.WORKSPACE,))
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
            container_backend=backend,
        )
        input_file = TaskInputFile(
            logical_path="artifact:input-artifact",
            artifact_ref=_input_artifact_ref(),
        )

        result = await runner.run(
            _container_file_definition(file_input=True),
            files=(input_file,),
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    async def test_non_artifact_container_input_mount_fails_closed(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget()
        backend = _backend(output_result=_container_output_result())
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
            container_backend=backend,
        )
        input_file = TaskInputFile(
            logical_path="provider:file",
            provider_reference=TaskProviderReference(
                kind=TaskProviderReferenceKind.PROVIDER_FILE_ID,
                provider="provider",
                reference="file-1",
            ),
        )

        result = await runner.run(
            _container_file_definition(file_input=True),
            files=(input_file,),
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    async def test_required_container_does_not_fall_back_to_host_target(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget("host-output")
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
        )

        result = await runner.run(
            _definition(container=_container_settings()),
            input_value="hello",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        self.assertEqual(result.output, None)

    async def test_direct_container_output_contract_failures(
        self,
    ) -> None:
        cases = (
            (
                "text output unsupported",
                _definition(container=_container_settings()),
                _backend(),
            ),
            (
                "artifact disabled",
                _definition(
                    container=_container_settings(),
                    output=TaskOutputContract.file(),
                ),
                _backend(),
            ),
            (
                "missing accepted artifacts",
                _container_file_definition(),
                _backend(output_result=_rejected_output_result()),
            ),
            (
                "missing copied bytes",
                _container_file_definition(),
                _backend(output_result=_container_output_result(content=None)),
            ),
        )
        for _name, definition, backend in cases:
            with self.subTest(_name):
                store = InMemoryTaskStore()
                runner = DirectTaskRunner(
                    store,
                    target=RecordingTarget(),
                    hmac_provider=StaticHmacProvider(),
                    artifact_store=_artifact_store(self),
                    container_backend=backend,
                )

                result = await runner.run(definition, input_value="hello")

                self.assertEqual(result.run.state, TaskRunState.FAILED)
                assert result.run.result is not None
                self.assertEqual(
                    result.run.result.error["code"],
                    "runnable.failed",
                )

    async def test_direct_container_result_reuses_output_handling(
        self,
    ) -> None:
        async def fake_container(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            return SimpleNamespace(
                output="done",
                output_artifacts_recorded=False,
            )

        runner = DirectTaskRunner(
            InMemoryTaskStore(),
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
        )
        with patch.object(runner, "_run_task_container", fake_container):
            result = await runner.run(_definition(), input_value="hello")

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "done")

    async def test_direct_container_output_validation_failure(
        self,
    ) -> None:
        runner = DirectTaskRunner(
            InMemoryTaskStore(),
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        with patch(
            "avalan.task.runner.validate_task_output",
            return_value=(_validation_issue("output.invalid"),),
        ):
            result = await runner.run(
                _container_file_definition(),
                input_value="hello",
            )

        self.assertEqual(result.run.state, TaskRunState.FAILED)

    async def test_container_lifecycle_requires_accepted_artifacts(
        self,
    ) -> None:
        async def rejected_lifecycle(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            return SimpleNamespace(
                execution=SimpleNamespace(
                    status=ContainerResultStatus.COMPLETED
                ),
                output=_rejected_output_result(),
            )

        direct_store = InMemoryTaskStore()
        direct_definition = _container_file_definition()
        await direct_store.register_definition(
            direct_definition,
            definition_hash=await spec_hash(direct_definition),
        )
        direct_run = await direct_store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(direct_definition),
                metadata={
                    TASK_CONTAINER_METADATA_KEY: (
                        task_container_request_metadata(direct_definition)
                    )
                },
            )
        )
        direct_attempt = await direct_store.create_attempt(direct_run.run_id)
        direct_runner = DirectTaskRunner(
            direct_store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        with patch(
            "avalan.task.runner.run_container_managed_lifecycle",
            rejected_lifecycle,
        ):
            with self.assertRaises(TaskValidationError) as direct_error:
                await direct_runner._run_task_container(
                    direct_definition,
                    run=direct_run,
                    attempt=direct_attempt,
                    input_mounts=(),
                    sanitizer=direct_runner._sanitizer(direct_definition),
                    expires_at=None,
                )
        self.assertEqual(
            direct_error.exception.issues[0].code,
            "container.output_unsupported",
        )

        worker_store = InMemoryTaskStore()
        worker_definition = _container_file_definition()
        await worker_store.register_definition(
            worker_definition,
            definition_hash=await spec_hash(worker_definition),
        )
        worker_run = await worker_store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(worker_definition),
                metadata={
                    TASK_CONTAINER_METADATA_KEY: (
                        task_container_request_metadata(worker_definition)
                    )
                },
            )
        )
        worker_attempt = await worker_store.create_attempt(worker_run.run_id)
        worker = TaskWorker(
            worker_store,
            SingleItemQueue(worker_store, Clock()),
            target=RecordingTarget(),
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        with patch(
            "avalan.task.worker.run_container_managed_lifecycle",
            rejected_lifecycle,
        ):
            with self.assertRaises(TaskValidationError) as worker_error:
                await worker._run_task_container(
                    worker_definition,
                    run=worker_run,
                    attempt=worker_attempt,
                    input_mounts=(),
                    sanitizer=worker._sanitizer(worker_definition),
                )
        self.assertEqual(
            worker_error.exception.issues[0].code,
            "container.output_unsupported",
        )

    async def test_direct_container_no_attempt_plan_returns_none(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition(container=_container_settings())
        await store.register_definition(
            definition,
            definition_hash=await spec_hash(definition),
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id=await spec_hash(definition))
        )
        attempt = await store.create_attempt(run.run_id)
        runner = DirectTaskRunner(
            store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
        )
        fake_plans = SimpleNamespace(
            enabled=True,
            attempt=None,
            worker_envelope=None,
        )

        with patch(
            "avalan.task.runner.verify_task_container_request",
            return_value=fake_plans,
        ):
            result = await runner._run_task_container(
                definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=runner._sanitizer(definition),
                expires_at=None,
            )

        self.assertIsNone(result)

    async def test_direct_initial_check_error_paths_are_structured(
        self,
    ) -> None:
        with self.subTest("system exit"):
            with self.assertRaises(SystemExit):
                await InitialCheckFailureRunner(
                    InMemoryTaskStore(),
                    error=SystemExit("stop"),
                ).run(_definition(), input_value="hello")

        with self.subTest("store conflict"):
            with self.assertRaises(TaskStoreConflictError):
                await InitialCheckFailureRunner(
                    InMemoryTaskStore(),
                    error=TaskStoreConflictError("conflict"),
                ).run(_definition(), input_value="hello")

        with self.subTest("cancelled"):
            cancelled = await InitialCheckFailureRunner(
                InMemoryTaskStore(),
                error=CancelledError(),
            ).run(_definition(), input_value="hello")
            self.assertEqual(cancelled.run.state, TaskRunState.CANCELLED)

        with self.subTest("failure"):
            failed = await InitialCheckFailureRunner(
                InMemoryTaskStore(),
                error=RuntimeError("boom"),
            ).run(_definition(), input_value="hello")
            self.assertEqual(failed.run.state, TaskRunState.FAILED)

    async def test_direct_container_verification_error_is_structured(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition(container=_container_settings())
        definition_hash = await spec_hash(definition)
        await store.register_definition(
            definition,
            definition_hash=definition_hash,
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id=definition_hash)
        )
        attempt = await store.create_attempt(run.run_id)
        runner = DirectTaskRunner(
            store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
        )

        with self.assertRaises(TaskValidationError) as caught:
            await runner._run_task_container(
                definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=runner._sanitizer(definition),
                expires_at=None,
            )

        self.assertEqual(
            caught.exception.issues[0].code,
            "container.plan_missing",
        )

    async def test_direct_container_rejects_backend_drift_and_failure(
        self,
    ) -> None:
        cases = (
            ("unavailable", _backend(available=False), ()),
            (
                "mismatch",
                _backend(backend=ContainerBackend.APPLE_CONTAINER),
                (),
            ),
            (
                "failed",
                _backend(
                    wait_exit_code=1,
                    output_result=_container_output_result(),
                ),
                ("container_lifecycle_completed",),
            ),
        )
        for _name, backend, extra_events in cases:
            with self.subTest(_name):
                store = InMemoryTaskStore()
                target = RecordingTarget()
                runner = DirectTaskRunner(
                    store,
                    target=target,
                    hmac_provider=StaticHmacProvider(),
                    container_backend=backend,
                )

                result = await runner.run(
                    _container_file_definition(),
                    input_value="hello",
                )

                self.assertEqual(result.run.state, TaskRunState.FAILED)
                self.assertEqual(target.contexts, [])
                events = await store.list_events(result.run.run_id)
                self.assertEqual(
                    tuple(event.event_type for event in events),
                    ("container_plan_verified", *extra_events),
                )

    async def test_direct_container_polling_and_no_capture_paths(self) -> None:
        no_capture_store = InMemoryTaskStore()
        no_capture_runner = DirectTaskRunner(
            no_capture_store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                delay_seconds=0.15,
                output_result=_container_output_result(),
            ),
        )
        no_capture = await no_capture_runner.run(
            _container_file_definition(
                observability=TaskObservabilityPolicy.noop(),
            ),
            input_value="hello",
        )
        self.assertEqual(no_capture.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(
            await no_capture_store.list_events(no_capture.run.run_id),
            (),
        )

        cancel_store = CancelAfterPlanStore()
        cancel_runner = DirectTaskRunner(
            cancel_store,
            target=RecordingTarget(),
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
        )
        with patch(
            "avalan.task.runner.run_container_managed_lifecycle",
            _hanging_lifecycle,
        ):
            cancelled = await cancel_runner.run(
                _container_file_definition(),
                input_value="hello",
            )
        self.assertEqual(cancelled.run.state, TaskRunState.CANCELLED)

    async def test_worker_envelope_only_profile_fails_closed(self) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget()
        backend = _backend()
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
            container_backend=backend,
        )

        result = await runner.run(
            _definition(
                container=_container_settings(
                    attempt=None,
                    worker_envelope=_effective_settings(),
                )
            ),
            input_value="hello",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)
        self.assertEqual(target.contexts, [])
        events = await store.list_events(result.run.run_id)
        self.assertIn(
            "worker_envelope_plan_fingerprint",
            events[0].payload,
        )

    async def test_worker_envelope_delegates_to_trusted_runner(self) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget()
        backend = _backend()

        class FakeWorkerEnvelopeRunner:
            trusted_runtime_envelope_runner = True

            def __init__(self) -> None:
                self.plans: TaskContainerPlans | None = None
                self.input_mounts: tuple[dict[str, object], ...] | None = None

            async def __call__(
                self,
                definition: TaskDefinition,
                *,
                run,
                attempt,
                plans: TaskContainerPlans,
                input_mounts: tuple[dict[str, object], ...],
            ) -> TaskContainerAttemptResult:
                self.plans = plans
                self.input_mounts = input_mounts
                return TaskContainerAttemptResult(output="from-envelope")

        envelope_runner = FakeWorkerEnvelopeRunner()
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
            container_backend=backend,
            worker_runtime_envelope_runner=envelope_runner,
        )

        result = await runner.run(
            _definition(
                container=_container_settings(
                    attempt=None,
                    worker_envelope=_effective_settings(),
                )
            ),
            input_value="hello",
        )

        self.assertEqual(result.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "from-envelope")
        self.assertEqual(target.contexts, [])
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)
        assert envelope_runner.plans is not None
        assert envelope_runner.plans.worker_envelope is not None
        self.assertEqual(
            envelope_runner.plans.worker_envelope.envelope_plan.profile_name,
            "strict",
        )
        events = await store.list_events(result.run.run_id)
        self.assertEqual(
            tuple(event.event_type for event in events),
            (
                "container_plan_verified",
                "container_worker_envelope_completed",
            ),
        )

    async def test_queued_worker_envelope_delegates_to_trusted_runner(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        target = RecordingTarget()

        class FakeWorkerEnvelopeRunner:
            trusted_runtime_envelope_runner = True

            def __init__(self) -> None:
                self.plans: TaskContainerPlans | None = None
                self.input_mounts: tuple[dict[str, object], ...] | None = None

            async def __call__(
                self,
                definition: TaskDefinition,
                *,
                run,
                attempt,
                plans: TaskContainerPlans,
                input_mounts: tuple[dict[str, object], ...],
            ) -> TaskContainerAttemptResult:
                _ = definition
                _ = run
                _ = attempt
                self.plans = plans
                self.input_mounts = input_mounts
                return TaskContainerAttemptResult(output="from-worker")

        client = TaskClient(
            store,
            target=target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
        )
        definition = _definition(
            run=TaskRunPolicy.queued("tasks"),
            container=_container_settings(
                attempt=None,
                worker_envelope=_effective_settings(),
            ),
        )
        submission = await client.enqueue(definition)
        envelope_runner = FakeWorkerEnvelopeRunner()
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
            worker_runtime_envelope_runner=envelope_runner,
            clock=lambda: clock.now,
        )

        result = await worker.process_once()

        self.assertEqual(result.output, "from-worker")
        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(target.contexts, [])
        assert envelope_runner.plans is not None
        assert envelope_runner.plans.worker_envelope is not None
        self.assertEqual(
            envelope_runner.plans.worker_envelope.envelope_plan.profile_name,
            "strict",
        )
        events = await store.list_events(submission.run.run_id)
        self.assertEqual(
            tuple(event.event_type for event in events),
            (
                "container_plan_verified",
                "container_worker_envelope_completed",
            ),
        )

    async def test_worker_envelope_runner_must_be_trusted(self) -> None:
        class UntrustedWorkerEnvelopeRunner:
            async def __call__(self, *args, **kwargs):
                return TaskContainerAttemptResult(output=None)

        with self.assertRaisesRegex(
            AssertionError,
            "worker runtime envelope runner must be trusted",
        ):
            DirectTaskRunner(
                InMemoryTaskStore(),
                target=RecordingTarget(),
                hmac_provider=StaticHmacProvider(),
                worker_runtime_envelope_runner=UntrustedWorkerEnvelopeRunner(),
            )

        worker_store = InMemoryTaskStore()
        with self.assertRaisesRegex(
            AssertionError,
            "worker runtime envelope runner must be trusted",
        ):
            TaskWorker(
                worker_store,
                SingleItemQueue(worker_store, Clock()),
                target=RecordingTarget(),
                queue_name="tasks",
                hmac_provider=StaticHmacProvider(),
                worker_runtime_envelope_runner=UntrustedWorkerEnvelopeRunner(),
            )

    async def test_worker_envelope_metadata_survives_restart(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition(
            container=_container_settings(
                attempt=None,
                worker_envelope=_effective_settings(),
            )
        )
        definition_id = await spec_hash(definition)
        await store.register_definition(
            definition,
            definition_hash=definition_id,
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id=definition_id,
                metadata=task_container_run_metadata(definition, None),
            )
        )
        attempt = await store.create_attempt(run.run_id)

        plans = verify_task_container_request(
            definition,
            run=run,
            attempt=attempt,
        )

        assert plans.worker_envelope is not None
        raw = run.request.metadata[TASK_CONTAINER_METADATA_KEY]
        assert isinstance(raw, Mapping)
        request_spec = raw[TASK_CONTAINER_WORKER_ENVELOPE_KEY]
        assert isinstance(request_spec, Mapping)
        self.assertEqual(
            request_spec["plan_fingerprint"],
            plans.worker_envelope.plan_fingerprint,
        )
        metadata = plans.to_metadata()[TASK_CONTAINER_WORKER_ENVELOPE_KEY]
        assert isinstance(metadata, Mapping)
        durable = ContainerDurablePlanMetadata.from_dict(metadata)
        durable.assert_matches(plans.worker_envelope)

    async def test_queued_worker_container_backend_guards(self) -> None:
        cases = (
            ("missing", None, TaskRunState.FAILED),
            ("unavailable", _backend(available=False), TaskRunState.FAILED),
            (
                "mismatch",
                _backend(backend=ContainerBackend.APPLE_CONTAINER),
                TaskRunState.FAILED,
            ),
            (
                "failed",
                _backend(
                    wait_exit_code=1,
                    output_result=_container_output_result(),
                ),
                TaskRunState.FAILED,
            ),
            (
                "delayed",
                _backend(
                    delay_seconds=0.15,
                    output_result=_container_output_result(),
                ),
                TaskRunState.SUCCEEDED,
            ),
        )
        for _name, backend, expected_state in cases:
            with self.subTest(_name):
                store = InMemoryTaskStore()
                clock = Clock()
                queue = SingleItemQueue(store, clock)
                target = RecordingTarget("done")
                definition = _container_file_definition(
                    run=TaskRunPolicy.queued("tasks"),
                    file_input=True,
                )
                client = TaskClient(
                    store,
                    target=target,
                    queue=queue,
                    hmac_provider=StaticHmacProvider(),
                    container_backend=_backend(),
                )
                await client.enqueue(definition)
                worker = TaskWorker(
                    store,
                    queue,
                    target=target,
                    queue_name="tasks",
                    hmac_provider=StaticHmacProvider(),
                    artifact_store=_artifact_store(self),
                    container_backend=backend,
                    clock=lambda: clock.now,
                )

                result = await worker.process_once()

                assert queue.completed is not None
                self.assertEqual(queue.completed.run.state, expected_state)
                self.assertEqual(
                    target.contexts,
                    [],
                )
                if expected_state == TaskRunState.SUCCEEDED:
                    self.assertIsInstance(result.output, TaskOutputArtifact)
                else:
                    self.assertIsNone(result.output)

    async def test_queued_worker_container_no_capture_skips_events(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        target = RecordingTarget("done")
        definition = _container_file_definition(
            run=TaskRunPolicy.queued("tasks"),
            file_input=True,
            observability=TaskObservabilityPolicy.noop(),
        )
        client = TaskClient(
            store,
            target=target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
        )
        submission = await client.enqueue(definition)
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
            clock=lambda: clock.now,
        )

        result = await worker.process_once()

        self.assertIsInstance(result.output, TaskOutputArtifact)
        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(await store.list_events(submission.run.run_id), ())
        self.assertEqual(target.contexts, [])

    async def test_direct_container_required_without_backend_fails_closed(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        target = RecordingTarget()
        runner = DirectTaskRunner(
            store,
            target=target,
            hmac_provider=StaticHmacProvider(),
        )

        result = await runner.run(
            _definition(container=_container_settings()),
            input_value="hello",
        )

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        events = await store.list_events(result.run.run_id)
        self.assertEqual(
            [event.event_type for event in events],
            ["container_plan_verified"],
        )

    async def test_queued_worker_verifies_container_metadata_before_target(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        definition = _container_file_definition(
            run=TaskRunPolicy.queued("tasks"),
            file_input=True,
        )
        await store.register_definition(
            definition,
            definition_hash=await spec_hash(definition),
        )
        request = TaskExecutionRequest(
            definition_id=await spec_hash(definition),
            metadata={"container": {"attempt": {"policy_version": "stale"}}},
        )
        submission = await queue.enqueue_run(
            request,
            queue_name="tasks",
            run_metadata={"runner": "queue"},
        )
        target = RecordingTarget()
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
            clock=lambda: clock.now,
        )

        result = await worker.process_once()

        self.assertIsNotNone(result.claimed)
        self.assertIsNotNone(queue.completed)
        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        self.assertEqual(await store.list_events(submission.run.run_id), ())
        self.assertEqual(submission.queue_item.run_id, queue.item.run_id)

    async def test_client_and_worker_preserve_direct_queued_equivalence(
        self,
    ) -> None:
        direct_store = InMemoryTaskStore()
        direct_target = RecordingTarget("same")
        direct = TaskClient(
            direct_store,
            target=direct_target,
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        direct_result = await direct.run(
            _container_file_definition(file_input=True),
            metadata={"user": "visible"},
        )

        queued_store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(queued_store, clock)
        queued_target = RecordingTarget("same")
        queued_definition = _container_file_definition(
            run=TaskRunPolicy.queued(
                "tasks",
                idempotency=IdempotencyMode.NONE,
            ),
            file_input=True,
        )
        client = TaskClient(
            queued_store,
            target=queued_target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        submission = await client.enqueue(
            queued_definition,
            metadata={"user": "visible"},
        )
        worker = TaskWorker(
            queued_store,
            queue,
            target=queued_target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
            clock=lambda: clock.now,
        )
        worker_result = await worker.process_once()

        self.assertIsInstance(direct_result.output, TaskOutputArtifact)
        self.assertIsInstance(worker_result.output, TaskOutputArtifact)
        self.assertEqual(direct_target.contexts, [])
        self.assertEqual(queued_target.contexts, [])
        self.assertIn("container", submission.run.request.metadata)
        self.assertEqual(queue.completed.run.state, TaskRunState.SUCCEEDED)

    async def test_retry_does_not_fall_back_to_host_state(
        self,
    ) -> None:
        backend = _backend()
        target = FailingThenSucceedingTarget("ok")
        runner = DirectTaskRunner(
            InMemoryTaskStore(),
            target=target,
            hmac_provider=StaticHmacProvider(),
            container_backend=backend,
        )
        definition = _definition(
            container=_container_settings(),
            retry=TaskRetryPolicy(
                max_attempts=2,
                backoff=RetryBackoff.NONE,
            ),
        )

        result = await runner.run(definition, input_value="hello")

        self.assertEqual(result.run.state, TaskRunState.FAILED)
        self.assertEqual(target.contexts, [])
        self.assertNotIn(ContainerBackendOperation.CREATE, backend.operations)

    async def test_worker_container_execution_fail_closed_edges(
        self,
    ) -> None:
        cases = (
            (
                "worker envelope",
                _definition(
                    run=TaskRunPolicy.queued("tasks"),
                    container=_container_settings(
                        attempt=None,
                        worker_envelope=_effective_settings(),
                    ),
                    output=TaskOutputContract.file(),
                ),
                _backend(),
            ),
            (
                "text output",
                _definition(
                    run=TaskRunPolicy.queued("tasks"),
                    container=_container_settings(),
                ),
                _backend(),
            ),
            (
                "artifact disabled",
                _definition(
                    run=TaskRunPolicy.queued("tasks"),
                    container=_container_settings(),
                    output=TaskOutputContract.file(),
                ),
                _backend(),
            ),
            (
                "missing accepted artifacts",
                _container_file_definition(
                    run=TaskRunPolicy.queued("tasks"),
                ),
                _backend(output_result=_rejected_output_result()),
            ),
            (
                "missing copied bytes",
                _container_file_definition(
                    run=TaskRunPolicy.queued("tasks"),
                ),
                _backend(output_result=_container_output_result(content=None)),
            ),
        )
        for _name, definition, backend in cases:
            with self.subTest(_name):
                store = InMemoryTaskStore()
                clock = Clock()
                queue = SingleItemQueue(store, clock)
                target = RecordingTarget()
                client = TaskClient(
                    store,
                    target=target,
                    queue=queue,
                    hmac_provider=StaticHmacProvider(),
                    artifact_store=_artifact_store(self),
                    container_backend=_backend(),
                )
                await client.enqueue(definition)
                worker = TaskWorker(
                    store,
                    queue,
                    target=target,
                    queue_name="tasks",
                    hmac_provider=StaticHmacProvider(),
                    artifact_store=_artifact_store(self),
                    container_backend=backend,
                    clock=lambda: clock.now,
                )

                await worker.process_once()

                assert queue.completed is not None
                self.assertEqual(
                    queue.completed.run.state, TaskRunState.FAILED
                )
                assert queue.completed.run.result is not None
                self.assertEqual(
                    queue.completed.run.result.error["code"],
                    "runnable.failed",
                )
                self.assertEqual(target.contexts, [])

    async def test_worker_container_result_reuses_output_handling(
        self,
    ) -> None:
        async def fake_container(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            return SimpleNamespace(
                output="done",
                output_artifacts_recorded=False,
            )

        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        target = RecordingTarget()
        definition = _definition(run=TaskRunPolicy.queued("tasks"))
        client = TaskClient(
            store,
            target=target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
        )
        await client.enqueue(definition)
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            clock=lambda: clock.now,
        )
        with patch.object(worker, "_run_task_container", fake_container):
            result = await worker.process_once()

        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(result.output, "done")

    async def test_worker_container_output_validation_failure(
        self,
    ) -> None:
        async def fake_container(
            *_args: object,
            **_kwargs: object,
        ) -> object:
            return SimpleNamespace(
                output=object(),
                output_artifacts_recorded=True,
            )

        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        target = RecordingTarget()
        definition = _definition(run=TaskRunPolicy.queued("tasks"))
        client = TaskClient(
            store,
            target=target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
        )
        await client.enqueue(definition)
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            clock=lambda: clock.now,
        )
        with patch.object(worker, "_run_task_container", fake_container):
            result = await worker.process_once()

        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.FAILED)
        self.assertIsNone(result.output)

    async def test_worker_container_no_attempt_plan_returns_none(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _definition(container=_container_settings())
        await store.register_definition(
            definition,
            definition_hash=await spec_hash(definition),
        )
        run = await store.create_run(
            TaskExecutionRequest(definition_id=await spec_hash(definition))
        )
        attempt = await store.create_attempt(run.run_id)
        worker = TaskWorker(
            store,
            SingleItemQueue(store, Clock()),
            target=RecordingTarget(),
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
        )
        fake_plans = SimpleNamespace(
            enabled=True,
            attempt=None,
            worker_envelope=None,
        )
        with patch(
            "avalan.task.worker.verify_task_container_request",
            return_value=fake_plans,
        ):
            result = await worker._run_task_container(
                definition,
                run=run,
                attempt=attempt,
                input_mounts=(),
                sanitizer=worker._sanitizer(definition),
            )

        self.assertIsNone(result)

    async def test_worker_container_rejects_unscoped_input_mount(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        definition = _container_file_definition(
            run=TaskRunPolicy.queued("tasks"),
            file_input=True,
        )
        input_mounts = (
            {
                "source_kind": "provider",
                "target": "/inputs/0",
            },
        )
        await store.register_definition(
            definition,
            definition_hash=await spec_hash(definition),
        )
        run = await store.create_run(
            TaskExecutionRequest(
                definition_id=await spec_hash(definition),
                metadata={
                    TASK_CONTAINER_METADATA_KEY: (
                        task_container_request_metadata(
                            definition,
                            input_mounts=input_mounts,
                        )
                    )
                },
            )
        )
        attempt = await store.create_attempt(run.run_id)
        worker = TaskWorker(
            store,
            SingleItemQueue(store, Clock()),
            target=RecordingTarget(),
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )

        with self.assertRaises(TaskValidationError) as caught:
            await worker._run_task_container(
                definition,
                run=run,
                attempt=attempt,
                input_mounts=input_mounts,
                sanitizer=worker._sanitizer(definition),
            )

        self.assertEqual(
            caught.exception.issues[0].code,
            "container.input_mount_unsupported",
        )

    async def test_worker_container_output_does_not_run_host_target(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        target = RecordingTarget("host-output")
        definition = _container_file_definition(
            run=TaskRunPolicy.queued("tasks"),
            file_input=True,
        )
        client = TaskClient(
            store,
            target=target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
        )
        await client.enqueue(definition)
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            artifact_store=_artifact_store(self),
            container_backend=_backend(
                output_result=_container_output_result()
            ),
            clock=lambda: clock.now,
        )
        result = await worker.process_once()

        self.assertIsNotNone(result.claimed)
        self.assertIsInstance(result.output, TaskOutputArtifact)
        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.SUCCEEDED)
        self.assertEqual(target.contexts, [])

    async def test_worker_cancellation_cancels_running_container_lifecycle(
        self,
    ) -> None:
        store = InMemoryTaskStore()
        clock = Clock()
        queue = SingleItemQueue(store, clock)
        target = RecordingTarget()
        definition = _container_file_definition(
            run=TaskRunPolicy.queued("tasks"),
            file_input=True,
        )
        client = TaskClient(
            store,
            target=target,
            queue=queue,
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
        )
        submission = await client.enqueue(definition)
        worker = TaskWorker(
            store,
            queue,
            target=target,
            queue_name="tasks",
            hmac_provider=StaticHmacProvider(),
            container_backend=_backend(),
            clock=lambda: clock.now,
        )

        with patch(
            "avalan.task.worker.run_container_managed_lifecycle",
            _hanging_lifecycle,
        ):
            worker_task = create_task(worker.process_once())
            for _ in range(100):
                run = await store.get_run(submission.run.run_id)
                if run.state == TaskRunState.RUNNING:
                    break
                await sleep(0)
            else:
                self.fail("worker did not start running the task")

            await store.transition_run(
                submission.run.run_id,
                from_states={TaskRunState.RUNNING},
                to_state=TaskRunState.CANCEL_REQUESTED,
                reason="cancel_requested",
            )
            result = await worker_task

        self.assertIsNotNone(result.claimed)
        assert queue.completed is not None
        self.assertEqual(queue.completed.run.state, TaskRunState.CANCELLED)
        self.assertEqual(target.contexts, [])


def _definition(
    *,
    run: TaskRunPolicy | None = None,
    retry: TaskRetryPolicy | None = None,
    observability: TaskObservabilityPolicy | None = None,
    container: TaskContainerExecutionSettings | None = None,
    output: TaskOutputContract | None = None,
    file_input: bool = False,
) -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="phase13", version="1"),
        input=(
            TaskInputContract(type=TaskInputType.FILE, required=False)
            if file_input
            else TaskInputContract(type=TaskInputType.STRING, required=False)
        ),
        output=output or TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/phase13.toml"),
        run=run or TaskRunPolicy.direct(timeout_seconds=30),
        retry=retry or TaskRetryPolicy(),
        observability=observability or TaskObservabilityPolicy(),
        container=container or TaskContainerExecutionSettings(),
    )


def _container_file_definition(
    *,
    run: TaskRunPolicy | None = None,
    retry: TaskRetryPolicy | None = None,
    observability: TaskObservabilityPolicy | None = None,
    file_input: bool = False,
) -> TaskDefinition:
    return _definition(
        run=run,
        retry=retry,
        observability=observability,
        container=_container_settings(allow_artifacts=True),
        output=TaskOutputContract.file(),
        file_input=file_input,
    )


def _container_settings(
    *,
    attempt: ContainerEffectiveSettings | None | object = _DEFAULT_ATTEMPT,
    worker_envelope: ContainerEffectiveSettings | None = None,
    allow_artifacts: bool = False,
) -> TaskContainerExecutionSettings:
    return TaskContainerExecutionSettings(
        attempt=(
            _effective_settings(allow_artifacts=allow_artifacts)
            if attempt is _DEFAULT_ATTEMPT
            else cast(ContainerEffectiveSettings | None, attempt)
        ),
        worker_envelope=worker_envelope,
    )


def _effective_settings(
    *,
    policy_version: str = "policy-v1",
    profile_name: str = "strict",
    allow_artifacts: bool = False,
) -> ContainerEffectiveSettings:
    profile = ContainerProfile.minimal_readonly(
        name=profile_name,
        image_reference=_IMAGE,
    )
    if allow_artifacts:
        profile = replace(
            profile,
            output=ContainerOutputPolicy(
                allow_artifacts=True,
                max_artifact_bytes=1024,
            ),
        )
    settings = ContainerSettings(
        source=_trusted_source(),
        backend=ContainerBackend.DOCKER,
        default_profile=profile_name,
        allowed_profiles=(profile_name,),
        profiles={profile_name: profile},
        profile_registry_id="task-registry",
        policy_version=policy_version,
    )
    return settings.select_profile(
        ContainerProfileSelection(
            required=True,
            profile=profile_name,
        )
    )


def _disabled_required_settings() -> ContainerEffectiveSettings:
    return ContainerSettings(
        source=_trusted_source(),
        backend=ContainerBackend.NONE,
    ).select_profile(ContainerProfileSelection(required=True))


def _enabled_without_profile_settings() -> ContainerEffectiveSettings:
    return ContainerEffectiveSettings(
        backend=ContainerBackend.DOCKER,
        required=True,
        scope=ContainerExecutionScope.SHELL_CONTAINER_EXECUTION,
        source=_trusted_source(),
        policy_version="policy-v1",
        profile_registry_id="task-registry",
    )


def _trusted_source() -> ContainerSettingsSource:
    return ContainerSettingsSource(
        surface=ContainerSurface.SDK,
        trust_level=ContainerTrustLevel.TRUSTED_DEPLOYMENT,
    )


def _validation_issue(code: str) -> TaskValidationIssue:
    return TaskValidationIssue(
        code=code,
        path="output",
        message="Output is invalid.",
        hint="Return a valid output.",
        category=TaskValidationCategory.VALUE,
    )


def _artifact_store(
    test_case: IsolatedAsyncioTestCase,
) -> LocalArtifactStore:
    return LocalArtifactStore(
        test_case.enterContext(TemporaryDirectory()),
        raw_storage_allowed=True,
    )


async def _verification_error(
    definition: TaskDefinition,
    metadata: Mapping[str, object],
) -> TaskContainerVerificationError:
    store = InMemoryTaskStore()
    definition_hash = await spec_hash(definition)
    await store.register_definition(
        definition,
        definition_hash=definition_hash,
    )
    run = await store.create_run(
        TaskExecutionRequest(
            definition_id=definition_hash,
            metadata=metadata,
        )
    )
    attempt = await store.create_attempt(run.run_id)
    try:
        verify_task_container_request(definition, run=run, attempt=attempt)
    except TaskContainerVerificationError as error:
        return error
    raise AssertionError("container verification unexpectedly succeeded")


async def _hanging_lifecycle(
    *_args: object,
    **_kwargs: object,
) -> object:
    while True:
        await sleep(1)


def _container_output_result(
    *,
    path: str = "result.txt",
    content: bytes | None = b"container-output",
) -> ContainerOutputValidationResult:
    contract = ContainerOutputContract(
        contract_type=ContainerOutputContractType.TASK_ARTIFACT,
        max_bytes=1024,
        max_files=100,
        enabled=True,
    )
    return ContainerOutputValidationResult(
        decision=ContainerOutputDecisionType.ACCEPT,
        contract=contract,
        artifacts=(_container_output_artifact(path=path, content=content),),
        total_bytes=len(content or b"container-output"),
        file_count=1,
    )


def _rejected_output_result() -> ContainerOutputValidationResult:
    return replace(
        _container_output_result(),
        decision=ContainerOutputDecisionType.REJECT,
        artifacts=(),
        total_bytes=0,
        file_count=0,
    )


def _container_output_artifact(
    *,
    path: str = "result.txt",
    content: bytes | None = b"container-output",
    digest: str = _OUTPUT_DIGEST,
) -> ContainerOutputArtifact:
    return ContainerOutputArtifact(
        artifact_type=ContainerOutputContractType.TASK_ARTIFACT,
        path=path,
        size_bytes=len(content or b"container-output"),
        media_type="text/plain",
        digest=f"sha256:{digest}",
        content=content,
    )


def _input_artifact_ref() -> TaskArtifactRef:
    return TaskArtifactRef(
        artifact_id="input-artifact",
        store="task-inputs",
        storage_key="/tmp/task-inputs/input-artifact",
        media_type="text/plain",
        size_bytes=5,
        sha256="3" * 64,
        metadata={"container_mount_source": "/tmp/task-inputs/input-artifact"},
    )


def _backend(
    *,
    backend: ContainerBackend = ContainerBackend.DOCKER,
    available: bool = True,
    wait_exit_code: int = 0,
    delay_seconds: float = 0,
    output_result: ContainerOutputValidationResult | None = None,
    mount_types: tuple[ContainerMountType, ...] = (
        ContainerMountType.WORKSPACE,
        ContainerMountType.INPUT,
    ),
) -> ContainerFakeBackend:
    return ContainerFakeBackend(
        ContainerFakeBackendScript(
            available=available,
            wait_exit_code=wait_exit_code,
            output_result=output_result,
            operation_delay_seconds=(
                {ContainerBackendOperation.STREAM: delay_seconds}
                if delay_seconds
                else {}
            ),
            capabilities=ContainerBackendCapabilities(
                backend=backend,
                host_os="linux",
                guest_os="linux",
                architecture="amd64",
                rootless=True,
                mount_types=mount_types,
                streaming_attach=True,
                stats=True,
            ),
        )
    )


if __name__ == "__main__":
    main()
