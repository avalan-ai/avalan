from asyncio import CancelledError, Event, create_task
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from client_test import (
    RecordingArtifactStore,
    RecordingQueue,
    StaticEncryptionProvider,
    StaticHmacProvider,
    _definition,
    _noop_target,
)

from avalan.pgsql import PgsqlUnitOfWork
from avalan.task.client import (
    TaskClient,
    TaskClientUnsupportedOperationError,
    _copy_submission_input,
    _queue_metadata_snapshot,
    _run_inspection_value,
)
from avalan.task.definition import (
    IdempotencyMode,
    TaskDefinition,
    TaskInputContract,
    TaskRunPolicy,
)
from avalan.task.input import TaskFileDescriptor
from avalan.task.queue import TaskQueue
from avalan.task.stores import InMemoryTaskStore
from avalan.task.submission import (
    PreparedTaskSubmission,
    TaskSubmissionCancelledError,
    TaskSubmissionKeyboardInterrupt,
    TaskSubmissionOutcome,
    TaskSubmissionRequest,
    TaskSubmissionResult,
    TaskSubmissionSystemExit,
    TaskSubmissionUnsettledError,
    TaskSubmissionWrite,
)


class AcknowledgmentQueue(RecordingQueue):
    def __init__(self, store: InMemoryTaskStore) -> None:
        super().__init__(store)
        self.commit_allowed = Event()
        self.commit_allowed.set()
        self.written = Event()
        self.ack_lost = False
        self.recovery_failed = False
        self.cancel_submission = False
        self.duplicate_without_new_ownership = False
        self.recovery_started = Event()
        self.recovery_allowed = Event()
        self.recovery_allowed.set()
        self.recoveries = 0
        self.transactions = 0

    @asynccontextmanager
    async def submission_transaction(self) -> AsyncIterator[PgsqlUnitOfWork]:
        self.transactions += 1
        yield cast(PgsqlUnitOfWork, object())
        await self.commit_allowed.wait()
        if self.ack_lost:
            raise ConnectionError("private database details")

    async def submit_prepared(
        self,
        prepared: PreparedTaskSubmission,
        *,
        unit_of_work: PgsqlUnitOfWork,
    ) -> TaskSubmissionWrite:
        if self.cancel_submission:
            raise CancelledError()
        write = await super().submit_prepared(
            prepared, unit_of_work=unit_of_work
        )
        if self.duplicate_without_new_ownership:
            write = replace(write, created=False, artifacts=())
            self.record_submission_write(write)
        self.written.set()
        return write

    async def reconcile_submission(
        self, prepared: PreparedTaskSubmission
    ) -> TaskSubmissionResult:
        self.recoveries += 1
        self.recovery_started.set()
        await self.recovery_allowed.wait()
        if self.recovery_failed:
            raise ConnectionError("private recovery details")
        result = await super().reconcile_submission(prepared)
        assert isinstance(result, TaskSubmissionResult)
        return result


def _client(
    root: Path, queue: AcknowledgmentQueue, artifacts: RecordingArtifactStore
) -> TaskClient:
    return TaskClient(
        queue.store,
        target=_noop_target,
        queue=cast(TaskQueue, queue),
        hmac_provider=StaticHmacProvider(),
        encryption_provider=StaticEncryptionProvider(),
        raw_storage_allowed=True,
        artifact_store=artifacts,
        definition_hash=lambda definition: (
            f"definition-{definition.run.idempotency.value}"
        ),
        execution_roots=(root,),
        owner_scope="trusted-owner",
        execution_deployment_id="host-resolved-deployment",
    )


def _file_request(root: Path) -> TaskSubmissionRequest:
    (root / "private.txt").write_text("private body")
    return TaskSubmissionRequest(
        input_value=TaskFileDescriptor.local_path(
            "private.txt", mime_type="text/plain"
        )
    )


def _file_definition() -> TaskDefinition:
    return _definition(
        input_contract=TaskInputContract.file(), run=TaskRunPolicy.queued("q")
    )


class TaskSubmissionLifecycleTest(IsolatedAsyncioTestCase):
    def test_preparation_snapshots_nested_input_and_absent_metadata(
        self,
    ) -> None:
        nested = ["original"]
        value = {"items": (nested,)}
        snapshot = _copy_submission_input(value)
        nested.append("later mutation")
        self.assertEqual(snapshot, {"items": (["original"],)})
        self.assertEqual(
            dict(
                _queue_metadata_snapshot(
                    None,
                    input_value="private",
                    idempotency_key=None,
                    owner_scope="trusted-owner",
                )
            ),
            {},
        )

    async def test_committed_inspection_redacts_encrypted_input(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            client = _client(root, queue, RecordingArtifactStore())
            result = await client.submit(
                _file_definition(), request=_file_request(root)
            )
            inspected = _run_inspection_value(result.run)
            self.assertIsNotNone(result.run.request.input_payload)
            self.assertIn("input_payload", inspected)
            self.assertNotIn("private body", str(inspected))

    async def test_public_result_waits_for_outer_commit(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.commit_allowed.clear()
            artifacts = RecordingArtifactStore()
            client = _client(root, queue, artifacts)
            pending = create_task(
                client.submit(_file_definition(), request=_file_request(root))
            )
            await queue.written.wait()
            self.assertFalse(pending.done())
            self.assertEqual(artifacts.deleted, [])
            queue.commit_allowed.set()
            result = await pending
            self.assertEqual(result.outcome, TaskSubmissionOutcome.COMMITTED)
            self.assertEqual(
                result.run.run_id, result.committed_write().run.run_id
            )
            self.assertTrue(result.created)
            self.assertIsNotNone(result.queue_item)
            self.assertIsNone(result.idempotency)
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(queue.recoveries, 0)

    async def test_lost_ack_reconciles_before_releasing_ownership(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.ack_lost = True
            artifacts = RecordingArtifactStore()
            result = await _client(root, queue, artifacts).submit(
                _file_definition(), request=_file_request(root)
            )
            self.assertEqual(result.outcome, TaskSubmissionOutcome.COMMITTED)
            self.assertEqual(queue.recoveries, 1)
            self.assertEqual(artifacts.deleted, [])
            self.assertNotIn("private", repr(result))

    async def test_unknown_retains_identity_and_all_temporary_artifacts(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.ack_lost = queue.recovery_failed = True
            artifacts = RecordingArtifactStore()
            client = _client(root, queue, artifacts)
            result = await client.submit(
                _file_definition(), request=_file_request(root)
            )
            self.assertEqual(result.outcome, TaskSubmissionOutcome.UNKNOWN)
            self.assertEqual(artifacts.deleted, [])
            with self.assertRaises(TaskSubmissionUnsettledError) as error:
                result.committed_write()
            self.assertIs(error.exception.result, result)
            self.assertNotIn("private", str(error.exception))
            prepared = result.prepared
            assert prepared is not None
            queue.recovery_failed = False
            recovered = await client.reconcile_submission(prepared)
            self.assertEqual(
                recovered.outcome, TaskSubmissionOutcome.COMMITTED
            )
            self.assertEqual(recovered.submission_id, result.submission_id)
            self.assertEqual(queue.transactions, 1)
            self.assertEqual(artifacts.deleted, [])

    async def test_cancellation_reconciles_then_releases_confirmed_rollback(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.cancel_submission = True
            artifacts = RecordingArtifactStore()
            with self.assertRaises(CancelledError):
                await _client(root, queue, artifacts).submit(
                    _file_definition(), request=_file_request(root)
                )
            self.assertEqual(queue.recoveries, 1)
            self.assertEqual(len(artifacts.deleted), 1)

    async def test_cleanup_failure_does_not_hide_commit_acknowledgment(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.duplicate_without_new_ownership = True
            artifacts = RecordingArtifactStore()
            with patch.object(
                artifacts,
                "delete",
                side_effect=RuntimeError("private cleanup details"),
            ):
                result = await _client(root, queue, artifacts).submit(
                    _file_definition(), request=_file_request(root)
                )
            self.assertEqual(result.outcome, TaskSubmissionOutcome.COMMITTED)
            self.assertFalse(result.created)
            self.assertEqual(len(result.cleanup_pending), 1)
            self.assertNotIn("private", repr(result))

    async def test_occurrence_window_is_scoped_without_enabling_idempotency(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            client = _client(root, queue, RecordingArtifactStore())
            enabled = _definition(
                run=TaskRunPolicy.queued(
                    "q", idempotency=IdempotencyMode.INPUT_HASH
                )
            )
            one = await client.prepare_submission(
                enabled,
                request=TaskSubmissionRequest(input_value="same"),
                occurrence_id="one",
            )
            two = await client.prepare_submission(
                enabled,
                request=TaskSubmissionRequest(input_value="same"),
                occurrence_id="two",
            )
            assert one.idempotency is not None and two.idempotency is not None
            self.assertNotEqual(
                one.idempotency.identity_key, two.idempotency.identity_key
            )
            disabled = await client.prepare_submission(
                _definition(
                    run=TaskRunPolicy.queued(
                        "q", idempotency=IdempotencyMode.NONE
                    )
                ),
                request=TaskSubmissionRequest(input_value="same"),
                occurrence_id="one",
            )
            self.assertIsNone(disabled.idempotency)
            self.assertEqual(disabled.occurrence_id, "one")
            self.assertEqual(
                disabled.execution_deployment_id, "host-resolved-deployment"
            )
            for request in (
                TaskSubmissionRequest(idempotency_key="manual"),
                TaskSubmissionRequest(idempotency_window="manual"),
            ):
                with self.assertRaises(AssertionError):
                    await client.prepare_submission(
                        enabled, request=request, occurrence_id="one"
                    )
            foreign = replace(one, owner_scope="another-owner")
            with self.assertRaises(AssertionError):
                await client.reconcile_submission(foreign)

    async def test_cancelled_write_exposes_committed_or_unknown_handle(
        self,
    ) -> None:
        for unknown in (False, True):
            with (
                self.subTest(unknown=unknown),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                queue = AcknowledgmentQueue(InMemoryTaskStore())
                queue.commit_allowed.clear()
                queue.recovery_failed = unknown
                artifacts = RecordingArtifactStore()
                client = _client(root, queue, artifacts)
                pending = create_task(
                    client.submit(
                        _file_definition(), request=_file_request(root)
                    )
                )
                await queue.written.wait()
                pending.cancel()
                with self.assertRaises(TaskSubmissionCancelledError) as caught:
                    await pending
                result = caught.exception.result
                self.assertEqual(
                    result.outcome,
                    (
                        TaskSubmissionOutcome.UNKNOWN
                        if unknown
                        else TaskSubmissionOutcome.COMMITTED
                    ),
                )
                self.assertIsNotNone(result.prepared)
                self.assertEqual(artifacts.deleted, [])
                assert result.prepared is not None
                queue.recovery_failed = False
                recovered = await client.reconcile_submission(result.prepared)
                self.assertEqual(
                    recovered.outcome, TaskSubmissionOutcome.COMMITTED
                )
                self.assertEqual(recovered.submission_id, result.submission_id)

    async def test_repeated_cancellation_during_recovery_keeps_unknown_handle(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.commit_allowed.clear()
            queue.recovery_allowed.clear()
            artifacts = RecordingArtifactStore()
            client = _client(root, queue, artifacts)
            pending = create_task(
                client.submit(_file_definition(), request=_file_request(root))
            )
            await queue.written.wait()
            pending.cancel()
            await queue.recovery_started.wait()
            pending.cancel()
            with self.assertRaises(TaskSubmissionCancelledError) as caught:
                await pending
            self.assertEqual(
                caught.exception.result.outcome, TaskSubmissionOutcome.UNKNOWN
            )
            self.assertIsNotNone(caught.exception.result.prepared)
            self.assertEqual(artifacts.deleted, [])

    async def test_repeated_cancellation_during_cleanup_preserves_known_commit(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            queue.commit_allowed.clear()
            queue.duplicate_without_new_ownership = True
            artifacts = RecordingArtifactStore()
            cleanup_started = Event()
            cleanup_allowed = Event()

            async def delete(ref: object) -> None:
                cleanup_started.set()
                await cleanup_allowed.wait()

            with patch.object(artifacts, "delete", side_effect=delete):
                pending = create_task(
                    _client(root, queue, artifacts).submit(
                        _file_definition(), request=_file_request(root)
                    )
                )
                await queue.written.wait()
                pending.cancel()
                await cleanup_started.wait()
                pending.cancel()
                with self.assertRaises(TaskSubmissionCancelledError) as caught:
                    await pending
            result = caught.exception.result
            self.assertEqual(result.outcome, TaskSubmissionOutcome.COMMITTED)
            self.assertIsNotNone(result.prepared)
            self.assertFalse(result.created)
            self.assertEqual(artifacts.deleted, [])

    async def test_termination_preserves_type_exit_code_and_recovery(
        self,
    ) -> None:
        for original, expected in (
            (KeyboardInterrupt(), TaskSubmissionKeyboardInterrupt),
            (SystemExit(7), TaskSubmissionSystemExit),
        ):
            with (
                self.subTest(expected=expected),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                queue = AcknowledgmentQueue(InMemoryTaskStore())
                artifacts = RecordingArtifactStore()
                with patch.object(
                    queue, "submit_prepared", side_effect=original
                ):
                    with self.assertRaises(expected) as caught:
                        await _client(root, queue, artifacts).submit(
                            _file_definition(), request=_file_request(root)
                        )
                self.assertEqual(
                    caught.exception.result.outcome,
                    TaskSubmissionOutcome.NOT_COMMITTED,
                )
                self.assertIsNotNone(caught.exception.result.prepared)
                if isinstance(caught.exception, SystemExit):
                    self.assertEqual(caught.exception.code, 7)
                self.assertEqual(len(artifacts.deleted), 1)

    async def test_schema_preflight_rejects_before_preparation_writes(
        self,
    ) -> None:
        for failure in (
            AssertionError("incompatible head"),
            RuntimeError("private missing schema"),
        ):
            with (
                self.subTest(failure=type(failure)),
                TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                queue = AcknowledgmentQueue(InMemoryTaskStore())
                artifacts = RecordingArtifactStore()
                with (
                    patch.object(
                        queue, "preflight_submission", side_effect=failure
                    ),
                    patch.object(
                        queue.store,
                        "register_definition",
                        new_callable=AsyncMock,
                    ) as register,
                ):
                    with self.assertRaises(
                        TaskClientUnsupportedOperationError
                    ) as caught:
                        await _client(root, queue, artifacts).submit(
                            _file_definition(), request=_file_request(root)
                        )
                register.assert_not_awaited()
                self.assertEqual(artifacts.puts, [])
                self.assertEqual(queue.transactions, 0)
                self.assertFalse(hasattr(queue, "_submission_writes"))
                self.assertEqual(artifacts.deleted, [])
                self.assertNotIn("private", str(caught.exception))

    async def test_recovery_cancellation_does_not_swallow_termination(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            artifacts = RecordingArtifactStore()
            with (
                patch.object(
                    queue, "submit_prepared", side_effect=SystemExit(9)
                ),
                patch.object(
                    queue, "reconcile_submission", side_effect=CancelledError()
                ),
            ):
                with self.assertRaises(TaskSubmissionSystemExit) as caught:
                    await _client(root, queue, artifacts).submit(
                        _file_definition(), request=_file_request(root)
                    )
            self.assertEqual(caught.exception.code, 9)
            self.assertEqual(
                caught.exception.result.outcome, TaskSubmissionOutcome.UNKNOWN
            )
            self.assertIsNotNone(caught.exception.result.prepared)
            self.assertEqual(artifacts.deleted, [])
