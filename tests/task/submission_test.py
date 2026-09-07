from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol, cast
from unittest import TestCase

from avalan.task.artifact import (
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactState,
)
from avalan.task.context import TaskInputFile
from avalan.task.definition import IdempotencyMode
from avalan.task.idempotency import (
    TaskIdempotencyDigest,
    TaskIdempotencyIdentity,
    TaskIdempotencyReservation,
    TaskIdempotencyReservationResult,
)
from avalan.task.queue import (
    TaskQueueItem,
    TaskQueueItemState,
)
from avalan.task.state import TaskRunState
from avalan.task.store import (
    TaskExecutionRequest,
    TaskRun,
    freeze_snapshot_metadata,
)
from avalan.task.submission import (
    _PREPARATION_AUTHORITY,
    PreparedTaskSubmission,
    TaskSubmissionArtifact,
    TaskSubmissionOutcome,
    TaskSubmissionRequest,
    TaskSubmissionResult,
    TaskSubmissionWrite,
)


class _UncheckedCall(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> object: ...


_NOW = datetime(2026, 9, 7, tzinfo=UTC)


def _ref(key: str = "temporary") -> TaskArtifactRef:
    return TaskArtifactRef(artifact_id=key, store="memory", storage_key=key)


def _prepared() -> PreparedTaskSubmission:
    return PreparedTaskSubmission(
        submission_id="submission",
        run_id="proposed-run",
        owner_scope="private-owner",
        execution=TaskExecutionRequest(definition_id="definition", queue="q"),
        priority=0,
        available_at=None,
        idempotency=None,
        idempotency_expires_at=None,
        artifacts=(TaskSubmissionArtifact(ref=_ref()),),
        temporary_artifacts=(_ref(),),
        run_metadata={},
        queue_metadata={},
        _authority=_PREPARATION_AUTHORITY,
    )


def _run() -> TaskRun:
    return TaskRun(
        run_id="existing-run",
        definition_id="definition",
        state=TaskRunState.QUEUED,
        request=TaskExecutionRequest(definition_id="definition", queue="q"),
        created_at=_NOW,
        updated_at=_NOW,
    )


def _record(ref: TaskArtifactRef) -> TaskArtifactRecord:
    return TaskArtifactRecord(
        artifact_id=ref.artifact_id,
        run_id="existing-run",
        purpose=TaskArtifactPurpose.INPUT,
        state=TaskArtifactState.READY,
        ref=ref,
        created_at=_NOW,
        updated_at=_NOW,
    )


class TaskSubmissionTest(TestCase):
    def test_request_freezes_metadata_and_hides_private_values(self) -> None:
        metadata = {"secret": ["private"]}
        request = TaskSubmissionRequest(
            input_value="private",
            files=(TaskInputFile(logical_path="private"),),
            metadata=metadata,
            queue_metadata=metadata,
            idempotency_key="private",
            idempotency_window="private",
            queue_name="q",
            available_at=_NOW,
            idempotency_expires_at=_NOW,
        )
        metadata["secret"].append("changed")
        self.assertEqual(request.metadata["secret"], ("private",))
        self.assertEqual(request.queue_metadata, request.metadata)
        self.assertNotIn("private", repr(request))
        self.assertIsNone(TaskSubmissionRequest().queue_name)
        with self.assertRaises(TypeError):
            cast(_UncheckedCall, TaskSubmissionRequest)(
                owner_scope="untrusted"
            )

    def test_request_rejects_invalid_queue_files_and_timestamps(self) -> None:
        for field, value in (
            ("queue_name", " "),
            ("idempotency_key", ""),
            ("files", []),
            ("files", (object(),)),
            ("available_at", datetime(2026, 9, 7)),
            ("idempotency_expires_at", "2026-09-07"),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(AssertionError):
                    cast(_UncheckedCall, replace)(
                        TaskSubmissionRequest(), **{field: value}
                    )

    def test_prepared_requires_internal_validation_authority(self) -> None:
        prepared = _prepared()
        self.assertNotIn("private-owner", repr(prepared))
        self.assertEqual(prepared.temporary_artifacts, (_ref(),))
        for changes in (
            {"_authority": object()},
            {"submission_id": ""},
            {"priority": True},
            {"execution": TaskExecutionRequest(definition_id="definition")},
            {"temporary_artifacts": (_ref(), _ref())},
            {"temporary_artifacts": (_ref("not-in-plan"),)},
            {"available_at": datetime(2026, 9, 7)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(AssertionError):
                    replace(prepared, **changes)

    def test_prepared_pins_idempotency_identity_and_frozen_metadata(
        self,
    ) -> None:
        digest = TaskIdempotencyDigest(
            algorithm="hmac-sha256", digest="digest", key_id="key"
        )
        identity = TaskIdempotencyIdentity(
            identity_key="identity",
            task_name="task",
            task_version="1",
            spec_hash="definition",
            owner_scope=digest,
            strategy=IdempotencyMode.INPUT_HASH,
            input=digest,
        )
        prepared = replace(
            _prepared(),
            execution=TaskExecutionRequest(
                definition_id="definition",
                queue="q",
                idempotency_key="identity",
            ),
            idempotency=identity,
            available_at=_NOW,
            idempotency_expires_at=_NOW,
            run_metadata=freeze_snapshot_metadata({"labels": ["safe"]}),
        )
        self.assertEqual(prepared.run_metadata["labels"], ("safe",))
        with self.assertRaises(AssertionError):
            replace(prepared, idempotency=None)
        with self.assertRaises(AssertionError):
            replace(_prepared(), idempotency=identity)
        item = TaskQueueItem(
            queue_item_id="item",
            run_id="existing-run",
            queue_name="q",
            state=TaskQueueItemState.AVAILABLE,
            priority=0,
            available_at=_NOW,
            attempts=0,
            created_at=_NOW,
            updated_at=_NOW,
            run_state=TaskRunState.QUEUED,
        )
        reservation = TaskIdempotencyReservationResult(
            reservation=TaskIdempotencyReservation(
                identity=identity, run_id="existing-run", created_at=_NOW
            ),
            created=False,
        )
        write = TaskSubmissionWrite(
            submission_id="submission",
            run=_run(),
            created=False,
            queue_item=item,
            idempotency=reservation,
            artifacts=(_record(_ref()),),
        )
        self.assertFalse(write.created)
        for changes in (
            {"queue_item": replace(item, run_id="other-run")},
            {
                "idempotency": replace(
                    reservation,
                    reservation=replace(
                        reservation.reservation, run_id="other-run"
                    ),
                )
            },
            {"artifacts": (replace(_record(_ref()), run_id="other-run"),)},
        ):
            with self.subTest(changes=changes):
                with self.assertRaises(AssertionError):
                    replace(write, **changes)

    def test_provisional_write_cannot_be_exposed_as_unknown_or_rollback(
        self,
    ) -> None:
        write = TaskSubmissionWrite(
            submission_id="submission", run=_run(), created=True
        )
        for outcome in (
            TaskSubmissionOutcome.UNKNOWN,
            TaskSubmissionOutcome.NOT_COMMITTED,
        ):
            with self.assertRaises(AssertionError):
                TaskSubmissionResult(
                    submission_id="submission", outcome=outcome, write=write
                )
        with self.assertRaises(AssertionError):
            TaskSubmissionResult(
                submission_id="submission",
                outcome=TaskSubmissionOutcome.COMMITTED,
            )
        with self.assertRaises(AssertionError):
            TaskSubmissionResult(
                submission_id="other",
                outcome=TaskSubmissionOutcome.COMMITTED,
                write=write,
            )
        with self.assertRaises(AssertionError):
            TaskSubmissionResult(
                submission_id="submission",
                outcome=cast(TaskSubmissionOutcome, "committed"),
            )

    def test_unknown_retains_temporary_resources_and_rollback_releases(
        self,
    ) -> None:
        prepared = _prepared()
        unknown = TaskSubmissionResult(
            submission_id="submission", outcome=TaskSubmissionOutcome.UNKNOWN
        )
        self.assertEqual(unknown.unused_artifacts(prepared), ())
        rollback = replace(
            unknown, outcome=TaskSubmissionOutcome.NOT_COMMITTED
        )
        self.assertEqual(rollback.unused_artifacts(prepared), (_ref(),))
        with self.assertRaises(AssertionError):
            unknown.unused_artifacts(replace(prepared, submission_id="other"))

    def test_duplicate_cleanup_protects_owned_storage_despite_metadata_changes(
        self,
    ) -> None:
        prepared = _prepared()
        write = TaskSubmissionWrite(
            submission_id="submission",
            run=_run(),
            created=False,
            artifacts=(
                _record(replace(_ref(), metadata={"sanitized": True})),
            ),
        )
        committed = TaskSubmissionResult(
            submission_id="submission",
            outcome=TaskSubmissionOutcome.COMMITTED,
            write=write,
        )
        self.assertEqual(committed.unused_artifacts(prepared), ())
        duplicate = replace(committed, write=replace(write, artifacts=()))
        self.assertEqual(duplicate.unused_artifacts(prepared), (_ref(),))
