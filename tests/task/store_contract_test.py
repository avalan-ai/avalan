from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Any, cast
from unittest import IsolatedAsyncioTestCase, TestCase, main

from avalan.task import (
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
    TaskAttemptSegmentState,
    TaskAttemptState,
    TaskDefinition,
    TaskExecutionPayload,
    TaskExecutionRequest,
    TaskExecutionResult,
    TaskExecutionTarget,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    TaskRun,
    TaskRunState,
    TaskStoreConflictError,
    TaskStoreNotFoundError,
    freeze_snapshot_metadata,
    freeze_snapshot_value,
    validate_attempt_segment_transition_request,
    validate_attempt_transition_request,
    validate_run_transition_request,
)
from avalan.task.store import (
    ensure_attempt_segment_is_mutable,
    ensure_run_is_mutable,
)
from avalan.task.stores import InMemoryTaskStore


class SequenceClock:
    def __init__(self) -> None:
        self._next = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        value = self._next
        self._next = self._next + timedelta(seconds=1)
        return value


class SequenceIds:
    def __init__(self) -> None:
        self._next = 1

    def __call__(self) -> str:
        value = f"id-{self._next}"
        self._next = self._next + 1
        return value


def definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="summarize", version="1"),
        input=TaskInputContract.string(),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/summarize.toml"),
    )


def _artifact_ref(artifact_id: str) -> TaskArtifactRef:
    return TaskArtifactRef(
        artifact_id=artifact_id,
        store="local",
        storage_key=f"{artifact_id[:2]}/{artifact_id}",
    )


class StoreContractAssertions:
    async def asyncSetUp(self) -> None:
        self.clock = SequenceClock()
        self.store = self.create_store()
        self.definition = definition()

    def create_store(self) -> Any:
        return InMemoryTaskStore(
            clock=self.clock,
            id_factory=SequenceIds(),
        )

    async def test_registers_definitions_immutably(self) -> None:
        record = await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
            metadata={"labels": ["stable"], "nested": {"owner": "runtime"}},
        )
        same_record = await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )

        self.assertEqual(record, same_record)
        self.assertEqual(record.definition_id, "hash-a")
        self.assertEqual(record.spec_hash, "hash-a")
        self.assertEqual(record.metadata["labels"], ("stable",))
        self.assertIsInstance(record.metadata, MappingProxyType)
        with self.assertRaises(FrozenInstanceError):
            cast(object, setattr(record, "spec_hash", "changed"))
        with self.assertRaises(TypeError):
            cast(dict[str, object], record.metadata)["raw"] = "value"
        with self.assertRaises(TaskStoreConflictError):
            await self.store.register_definition(
                TaskDefinition(
                    task=TaskMetadata(name="different", version="1"),
                    input=TaskInputContract.string(),
                    output=TaskOutputContract.text(),
                    execution=TaskExecutionTarget.agent("agents/other.toml"),
                ),
                definition_hash="hash-a",
            )

    async def test_create_run_requires_registered_definition(self) -> None:
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.create_run(
                TaskExecutionRequest(definition_id="missing")
            )

    async def test_run_lifecycle_uses_compare_and_swap_transitions(
        self,
    ) -> None:
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<hmac-sha256>"},
                file_summaries=({"name": "input.pdf", "bytes": 100},),
            )
        )

        self.assertEqual(run.state, TaskRunState.CREATED)
        self.assertEqual(
            cast(dict[str, object], run.request.input_summary)["privacy"],
            "<hmac-sha256>",
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], run.request.input_summary)["raw"] = "leak"

        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                run.run_id,
                from_states={TaskRunState.QUEUED},
                to_state=TaskRunState.RUNNING,
                reason="stale_state",
            )
        self.assertEqual(
            await self.store.list_run_transitions(run.run_id),
            (),
        )

        validated = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        running = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.RUNNING,
            reason="started",
        )
        result = TaskExecutionResult(output_summary={"privacy": "<redacted>"})
        succeeded = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.SUCCEEDED,
            reason="completed",
            result=result,
        )

        self.assertEqual(validated.state, TaskRunState.VALIDATED)
        self.assertEqual(running.state, TaskRunState.RUNNING)
        self.assertEqual(succeeded.state, TaskRunState.SUCCEEDED)
        self.assertEqual(succeeded.result, result)
        transitions = await self.store.list_run_transitions(run.run_id)
        self.assertEqual(
            [transition.to_state for transition in transitions],
            [
                TaskRunState.VALIDATED,
                TaskRunState.RUNNING,
                TaskRunState.SUCCEEDED,
            ],
        )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                run.run_id,
                from_states={TaskRunState.SUCCEEDED},
                to_state=TaskRunState.FAILED,
                reason="late_failure",
            )

    async def test_execution_payload_is_durable_and_immutable(self) -> None:
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        run = await self.store.create_run(
            TaskExecutionRequest(
                definition_id="hash-a",
                input_summary={"privacy": "<hmac-sha256>"},
                input_payload=TaskExecutionPayload(
                    file_values=(
                        {
                            "privacy": "<encrypted>",
                            "ciphertext": "ZmlsZQ==",
                        },
                    ),
                    input_value={
                        "privacy": "<encrypted>",
                        "ciphertext": "Ynl0ZXM=",
                    },
                ),
            )
        )
        loaded = await self.store.get_run(run.run_id)

        self.assertIsNotNone(loaded.request.input_payload)
        assert loaded.request.input_payload is not None
        payload = cast(
            Mapping[str, object],
            loaded.request.input_payload.input_value,
        )
        self.assertEqual(payload["privacy"], "<encrypted>")
        self.assertEqual(len(loaded.request.input_payload.file_values), 1)
        file_payload = cast(
            Mapping[str, object],
            loaded.request.input_payload.file_values[0],
        )
        self.assertEqual(file_payload["privacy"], "<encrypted>")
        with self.assertRaises(TypeError):
            cast(dict[str, object], payload)["raw"] = "leak"
        with self.assertRaises(TypeError):
            cast(dict[str, object], file_payload)["raw"] = "leak"
        with self.assertRaises(AssertionError):
            TaskExecutionPayload(input_value={"raw": object()})
        with self.assertRaises(AssertionError):
            TaskExecutionPayload(
                input_value=None,
                file_values=({"raw": object()},),
            )

    async def test_attempts_are_ordered_and_one_active_attempt_is_allowed(
        self,
    ) -> None:
        run = await self._created_run()
        first = await self.store.create_attempt(run.run_id)

        self.assertEqual(first.attempt_number, 1)
        self.assertEqual(first.state, TaskAttemptState.CREATED)
        self.assertEqual(first.context.run_id, run.run_id)
        self.assertEqual(first.context.attempt_id, first.attempt_id)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(run.run_id)

        running = await self.store.transition_attempt(
            first.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="started",
        )
        failed = await self.store.transition_attempt(
            first.attempt_id,
            from_states={TaskAttemptState.RUNNING},
            to_state=TaskAttemptState.FAILED,
            reason="failed",
            result=TaskExecutionResult(error={"code": "runnable"}),
        )
        second = await self.store.create_attempt(run.run_id)

        self.assertEqual(running.state, TaskAttemptState.RUNNING)
        self.assertIsNotNone(failed.result)
        result = failed.result
        error = cast(Mapping[str, object], result.error) if result else {}
        self.assertEqual(error["code"], "runnable")
        self.assertEqual(second.attempt_number, 2)
        self.assertEqual(
            [
                attempt.attempt_id
                for attempt in await self.store.list_attempts(run.run_id)
            ],
            [first.attempt_id, second.attempt_id],
        )
        transitions = await self.store.list_attempt_transitions(
            first.attempt_id
        )
        self.assertEqual(
            [transition.to_state for transition in transitions],
            [TaskAttemptState.RUNNING, TaskAttemptState.FAILED],
        )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_attempt(
                first.attempt_id,
                from_states={TaskAttemptState.FAILED},
                to_state=TaskAttemptState.RUNNING,
                reason="reopen",
            )

    async def test_claim_token_fences_queued_run_updates(self) -> None:
        run = await self._created_run()
        queued = await self.store.transition_run(
            run.run_id,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
            reason="validated",
        )
        queued = await self.store.transition_run(
            queued.run_id,
            from_states={TaskRunState.VALIDATED},
            to_state=TaskRunState.QUEUED,
            reason="queued",
        )
        claimed = await self.store.assign_claim(
            queued.run_id,
            from_states={TaskRunState.QUEUED},
            worker_id="worker-1",
            lease_expires_at=datetime(2026, 1, 1, 1, tzinfo=UTC),
            reason="claimed",
        )

        self.assertIsNotNone(claimed.claim)
        self.assertEqual(claimed.state, TaskRunState.CLAIMED)
        claim_token = claimed.claim.claim_token if claimed.claim else ""
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                claimed.run_id,
                from_states={TaskRunState.CLAIMED},
                to_state=TaskRunState.RUNNING,
                reason="missing_token",
            )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_run(
                claimed.run_id,
                from_states={TaskRunState.CLAIMED},
                to_state=TaskRunState.RUNNING,
                reason="wrong_token",
                claim_token="stale",
            )

        running = await self.store.transition_run(
            claimed.run_id,
            from_states={TaskRunState.CLAIMED},
            to_state=TaskRunState.RUNNING,
            reason="started",
            claim_token=claim_token,
        )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(running.run_id)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.create_attempt(
                running.run_id,
                claim_token="stale",
            )

        attempt = await self.store.create_attempt(
            running.run_id,
            claim_token=claim_token,
        )
        self.assertEqual(attempt.context.claim, claimed.claim)
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_attempt(
                attempt.attempt_id,
                from_states={TaskAttemptState.CREATED},
                to_state=TaskAttemptState.RUNNING,
                reason="missing_token",
            )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_attempt(
                attempt.attempt_id,
                from_states={TaskAttemptState.CREATED},
                to_state=TaskAttemptState.RUNNING,
                reason="wrong_token",
                claim_token="stale",
            )
        running_attempt = await self.store.transition_attempt(
            attempt.attempt_id,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
            reason="attempt_started",
            claim_token=claim_token,
        )
        self.assertEqual(running_attempt.state, TaskAttemptState.RUNNING)
        succeeded = await self.store.transition_run(
            running.run_id,
            from_states={TaskRunState.RUNNING},
            to_state=TaskRunState.SUCCEEDED,
            reason="finished",
            claim_token=claim_token,
        )
        self.assertIsNone(succeeded.claim)

    async def test_artifact_metadata_is_appended_and_filtered_by_store(
        self,
    ) -> None:
        run = await self._created_run()
        attempt = await self.store.create_attempt(run.run_id)
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        other_attempt = await self.store.create_attempt(other_run.run_id)
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
            media_type="text/plain",
            size_bytes=12,
            sha256="a" * 64,
        )
        record = await self.store.append_artifact(
            run.run_id,
            attempt_id=attempt.attempt_id,
            ref=ref,
            purpose=TaskArtifactPurpose.INPUT,
            provenance=TaskArtifactProvenance(operation="upload"),
            retention=TaskArtifactRetention(delete_after_days=1),
            metadata={"labels": ["safe"]},
        )

        self.assertEqual(record.artifact_id, "artifact-1")
        self.assertEqual(record.state, TaskArtifactState.READY)
        self.assertEqual(record.metadata["labels"], ("safe",))
        self.assertEqual(await self.store.get_artifact("artifact-1"), record)
        self.assertEqual(
            await self.store.list_artifacts(run.run_id),
            (record,),
        )
        self.assertEqual(
            await self.store.list_artifacts(
                run.run_id,
                attempt_id=attempt.attempt_id,
                purpose=TaskArtifactPurpose.INPUT,
                state=TaskArtifactState.READY,
            ),
            (record,),
        )
        self.assertEqual(
            await self.store.list_artifacts(
                run.run_id,
                purpose=TaskArtifactPurpose.OUTPUT,
            ),
            (),
        )
        with self.assertRaises(TypeError):
            cast(dict[str, object], record.metadata)["raw"] = "value"
        with self.assertRaises(TaskStoreConflictError):
            await self.store.append_artifact(
                run.run_id,
                ref=ref,
                purpose=TaskArtifactPurpose.INPUT,
            )
        with self.assertRaises(AssertionError):
            await self.store.append_artifact(
                run.run_id,
                ref=TaskArtifactRef(
                    artifact_id="artifact-2",
                    store="local",
                    storage_key="ar/artifact-2",
                ),
                purpose=TaskArtifactPurpose.INPUT,
                metadata={"raw": cast(object, b"bytes")},
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.append_artifact(
                run.run_id,
                attempt_id=other_attempt.attempt_id,
                ref=TaskArtifactRef(
                    artifact_id="artifact-3",
                    store="local",
                    storage_key="ar/artifact-3",
                ),
                purpose=TaskArtifactPurpose.INPUT,
            )
        with self.assertRaises(TaskStoreNotFoundError):
            await self.store.list_artifacts(
                run.run_id,
                attempt_id=other_attempt.attempt_id,
            )

    async def test_retention_artifact_discovery_filters_ready_expired_records(
        self,
    ) -> None:
        run = await self._created_run()
        other_run = await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )
        input_record = await self.store.append_artifact(
            run.run_id,
            ref=_artifact_ref("input-1"),
            purpose=TaskArtifactPurpose.INPUT,
            retention=TaskArtifactRetention(delete_after_days=1),
        )
        await self.store.append_artifact(
            run.run_id,
            ref=_artifact_ref("output-1"),
            purpose=TaskArtifactPurpose.OUTPUT,
            retention=TaskArtifactRetention(delete_after_days=30),
        )
        converted_record = await self.store.append_artifact(
            other_run.run_id,
            ref=_artifact_ref("converted-1"),
            purpose=TaskArtifactPurpose.CONVERTED,
            retention=TaskArtifactRetention(
                expires_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            ),
        )
        await self.store.append_artifact(
            other_run.run_id,
            ref=_artifact_ref("lost-1"),
            purpose=TaskArtifactPurpose.INPUT,
            state=TaskArtifactState.LOST,
            retention=TaskArtifactRetention(delete_after_days=1),
        )

        expired = await self.store.list_retention_artifacts(
            expired_at=datetime(2026, 1, 3, tzinfo=UTC),
        )
        input_expired = await self.store.list_retention_artifacts(
            expired_at=datetime(2026, 1, 3, tzinfo=UTC),
            purpose=TaskArtifactPurpose.INPUT,
        )
        limited = await self.store.list_retention_artifacts(
            expired_at=datetime(2026, 1, 3, tzinfo=UTC),
            limit=1,
        )

        self.assertEqual(
            {record.artifact_id for record in expired},
            {input_record.artifact_id, converted_record.artifact_id},
        )
        self.assertEqual(input_expired, (input_record,))
        self.assertEqual(len(limited), 1)
        self.assertIn(limited[0], expired)
        with self.assertRaises(AssertionError):
            await self.store.list_retention_artifacts(
                expired_at=cast(Any, "2026-01-03"),
            )
        with self.assertRaises(AssertionError):
            await self.store.list_retention_artifacts(
                expired_at=datetime(2026, 1, 3, tzinfo=UTC),
                purpose=cast(Any, "input"),
            )
        with self.assertRaises(AssertionError):
            await self.store.list_retention_artifacts(
                expired_at=datetime(2026, 1, 3, tzinfo=UTC),
                limit=0,
            )

    async def test_artifact_state_transitions_are_compare_and_swap(
        self,
    ) -> None:
        run = await self._created_run()
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
        )
        await self.store.append_artifact(
            run.run_id,
            ref=ref,
            purpose=TaskArtifactPurpose.OUTPUT,
        )

        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_artifact(
                "artifact-1",
                from_states={TaskArtifactState.LOST},
                to_state=TaskArtifactState.DELETED,
                reason="stale",
            )
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_artifact(
                "artifact-1",
                from_states={TaskArtifactState.READY},
                to_state=TaskArtifactState.READY,
                reason="invalid_transition",
            )
        deleted = await self.store.transition_artifact(
            "artifact-1",
            from_states={TaskArtifactState.READY},
            to_state=TaskArtifactState.DELETED,
            reason="retention_expired",
            metadata={"retention": "expired"},
        )

        self.assertEqual(deleted.state, TaskArtifactState.DELETED)
        self.assertEqual(deleted.metadata["retention"], "expired")
        with self.assertRaises(TaskStoreConflictError):
            await self.store.transition_artifact(
                "artifact-1",
                from_states={TaskArtifactState.DELETED},
                to_state=TaskArtifactState.LOST,
                reason="late_loss",
            )

    async def _created_run(self) -> TaskRun:
        await self.store.register_definition(
            self.definition,
            definition_hash="hash-a",
        )
        return await self.store.create_run(
            TaskExecutionRequest(definition_id="hash-a")
        )


class StoreContractTest(StoreContractAssertions, IsolatedAsyncioTestCase):
    pass


class StoreHelperTest(TestCase):
    def test_segment_transition_helpers_reject_stale_invalid_and_terminal(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "task attempt segment state did not match",
        ):
            validate_attempt_segment_transition_request(
                current_state=TaskAttemptSegmentState.CREATED,
                from_states={TaskAttemptSegmentState.RUNNING},
                to_state=TaskAttemptSegmentState.ABANDONED,
            )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "task attempt segment transition is not valid",
        ):
            validate_attempt_segment_transition_request(
                current_state=TaskAttemptSegmentState.CREATED,
                from_states={TaskAttemptSegmentState.CREATED},
                to_state=TaskAttemptSegmentState.SUCCEEDED,
            )
        with self.assertRaisesRegex(
            TaskStoreConflictError,
            "terminal task attempt segment cannot be changed",
        ):
            ensure_attempt_segment_is_mutable(
                TaskAttemptSegmentState.SUCCEEDED
            )

    def test_transition_request_helpers_reject_stale_and_invalid_states(
        self,
    ) -> None:
        validate_run_transition_request(
            current_state=TaskRunState.CREATED,
            from_states={TaskRunState.CREATED},
            to_state=TaskRunState.VALIDATED,
        )
        validate_attempt_transition_request(
            current_state=TaskAttemptState.CREATED,
            from_states={TaskAttemptState.CREATED},
            to_state=TaskAttemptState.RUNNING,
        )

        with self.assertRaises(TaskStoreConflictError):
            validate_run_transition_request(
                current_state=TaskRunState.CREATED,
                from_states={TaskRunState.QUEUED},
                to_state=TaskRunState.RUNNING,
            )
        with self.assertRaises(TaskStoreConflictError):
            validate_attempt_transition_request(
                current_state=TaskAttemptState.SUCCEEDED,
                from_states={TaskAttemptState.SUCCEEDED},
                to_state=TaskAttemptState.RUNNING,
            )
        with self.assertRaises(TaskStoreConflictError):
            validate_attempt_transition_request(
                current_state=TaskAttemptState.CREATED,
                from_states={TaskAttemptState.RUNNING},
                to_state=TaskAttemptState.SUCCEEDED,
            )
        with self.assertRaises(TaskStoreConflictError):
            validate_run_transition_request(
                current_state=TaskRunState.SUCCEEDED,
                from_states={TaskRunState.SUCCEEDED},
                to_state=TaskRunState.SUCCEEDED,
            )
        with self.assertRaises(TaskStoreConflictError):
            ensure_run_is_mutable(TaskRunState.SUCCEEDED)

    def test_snapshot_values_are_recursive_immutable_safe_values(self) -> None:
        value = freeze_snapshot_value(
            {
                "confidence": 0.5,
                "items": [{"count": 1}],
                "flags": (True, None),
            }
        )

        frozen = cast(Mapping[str, object], value)
        self.assertEqual(frozen["confidence"], 0.5)
        items = cast(tuple[Mapping[str, object], ...], frozen["items"])
        self.assertEqual(items[0]["count"], 1)
        with self.assertRaises(TypeError):
            cast(dict[str, object], value)["raw"] = "value"
        with self.assertRaises(AssertionError):
            freeze_snapshot_value(object())
        with self.assertRaises(AssertionError):
            freeze_snapshot_metadata({"bad": cast(object, b"bytes")})


if __name__ == "__main__":
    main()
