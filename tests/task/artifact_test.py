from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from typing import cast
from unittest import TestCase, main

from avalan.task import (
    ArtifactStoreConflictError,
    TaskArtifactProvenance,
    TaskArtifactPurpose,
    TaskArtifactRecord,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactState,
    TaskOutputArtifact,
    assert_artifact_state_collection,
    is_terminal_artifact_state,
    is_valid_artifact_transition,
    task_output_artifact_from_value,
    validate_artifact_transition,
)


class TaskArtifactTest(TestCase):
    def test_artifact_models_freeze_sanitized_metadata(self) -> None:
        expires_at = datetime(2026, 1, 2, tzinfo=UTC)
        retention_metadata = {"labels": ["raw"], "nested": {"keep": True}}
        provenance_metadata = {"converter": {"version": "1"}}
        ref_metadata = {"checks": ["sha256"]}
        record_metadata = {"audit": {"retained": True}}

        retention = TaskArtifactRetention(
            expires_at=expires_at,
            delete_after_days=3,
            metadata=retention_metadata,
        )
        provenance = TaskArtifactProvenance(
            source_artifact_id="source-1",
            source_run_id="run-1",
            source_attempt_id="attempt-1",
            operation="conversion",
            converter="markdown",
            metadata=provenance_metadata,
        )
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
            media_type="text/plain",
            size_bytes=12,
            sha256="a" * 64,
            metadata=ref_metadata,
        )
        record = TaskArtifactRecord(
            artifact_id="artifact-1",
            run_id="run-1",
            attempt_id="attempt-1",
            purpose=TaskArtifactPurpose.CONVERTED,
            state=TaskArtifactState.READY,
            ref=ref,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC),
            provenance=provenance,
            retention=retention,
            metadata=record_metadata,
        )

        retention_metadata["labels"].append("mutated")
        cast(dict[str, object], retention_metadata["nested"])["keep"] = False
        cast(dict[str, object], provenance_metadata["converter"])[
            "version"
        ] = "2"
        ref_metadata["checks"].append("mutated")
        cast(dict[str, object], record_metadata["audit"])["retained"] = False

        self.assertEqual(retention.metadata["labels"], ("raw",))
        nested = cast(Mapping[str, object], retention.metadata["nested"])
        self.assertTrue(nested["keep"])
        converter = cast(
            Mapping[str, object], provenance.metadata["converter"]
        )
        self.assertEqual(converter["version"], "1")
        self.assertEqual(ref.metadata["checks"], ("sha256",))
        audit = cast(Mapping[str, object], record.metadata["audit"])
        self.assertTrue(audit["retained"])
        self.assertNotIn(
            "storage_key", cast(Mapping[str, object], ref.summary())
        )
        summary = cast(Mapping[str, object], record.summary())
        self.assertEqual(summary["artifact_id"], "artifact-1")
        self.assertEqual(summary["purpose"], "converted")
        self.assertEqual(summary["state"], "ready")
        self.assertIn("retention", summary)
        self.assertIn("provenance", summary)
        with self.assertRaises(FrozenInstanceError):
            record.state = TaskArtifactState.DELETED
        with self.assertRaises(TypeError):
            cast(dict[str, object], record.metadata)["raw"] = "value"

    def test_output_artifact_helper_normalizes_refs_and_records(self) -> None:
        ref = TaskArtifactRef(
            artifact_id="artifact-1",
            store="local",
            storage_key="ar/artifact-1",
            media_type="text/plain",
        )
        record = TaskArtifactRecord(
            artifact_id="artifact-2",
            run_id="run-1",
            purpose=TaskArtifactPurpose.OUTPUT,
            state=TaskArtifactState.DELETED,
            ref=TaskArtifactRef(
                artifact_id="artifact-2",
                store="local",
                storage_key="ar/artifact-2",
            ),
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
            updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            metadata={"state": "deleted"},
        )

        from_ref = task_output_artifact_from_value(ref)
        from_record = task_output_artifact_from_value(record)

        self.assertIsInstance(from_ref, TaskOutputArtifact)
        self.assertEqual(from_ref.ref, ref)
        self.assertEqual(from_ref.state, TaskArtifactState.READY)
        self.assertEqual(from_ref.provenance.operation, "output")
        self.assertIsInstance(from_record, TaskOutputArtifact)
        self.assertEqual(from_record.state, TaskArtifactState.DELETED)
        self.assertEqual(from_record.metadata["state"], "deleted")
        summary = cast(Mapping[str, object], from_record.summary())
        self.assertEqual(summary["state"], "deleted")
        self.assertNotIn("storage_key", str(summary))
        self.assertIsNone(task_output_artifact_from_value(object()))

    def test_transition_helpers_reject_invalid_state_changes(self) -> None:
        self.assertFalse(is_terminal_artifact_state(TaskArtifactState.READY))
        self.assertTrue(is_terminal_artifact_state(TaskArtifactState.DELETED))
        self.assertTrue(
            is_valid_artifact_transition(
                TaskArtifactState.READY,
                TaskArtifactState.DELETED,
            )
        )
        self.assertFalse(
            is_valid_artifact_transition(
                TaskArtifactState.DELETED,
                TaskArtifactState.LOST,
            )
        )
        validate_artifact_transition(
            TaskArtifactState.READY,
            TaskArtifactState.LOST,
        )

        with self.assertRaises(ArtifactStoreConflictError):
            validate_artifact_transition(
                TaskArtifactState.LOST,
                TaskArtifactState.DELETED,
            )
        with self.assertRaises(AssertionError):
            assert_artifact_state_collection((), "from_states")
        with self.assertRaises(AssertionError):
            assert_artifact_state_collection(
                ("ready",),
                "from_states",
            )

    def test_artifact_models_reject_malformed_values(self) -> None:
        with self.assertRaises(AssertionError):
            TaskArtifactRetention(delete_after_days=0)
        with self.assertRaises(AssertionError):
            TaskArtifactProvenance(operation="")
        with self.assertRaises(AssertionError):
            TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="ar/artifact-1",
                size_bytes=-1,
            )
        with self.assertRaises(AssertionError):
            TaskArtifactRef(
                artifact_id="artifact-1",
                store="local",
                storage_key="ar/artifact-1",
                sha256="A" * 64,
            )
        with self.assertRaises(AssertionError):
            TaskArtifactRecord(
                artifact_id="artifact-1",
                run_id="run-1",
                purpose=TaskArtifactPurpose.INPUT,
                state=TaskArtifactState.READY,
                ref=TaskArtifactRef(
                    artifact_id="artifact-2",
                    store="local",
                    storage_key="ar/artifact-2",
                ),
                created_at=datetime(2026, 1, 1, tzinfo=UTC),
                updated_at=datetime(2026, 1, 1, tzinfo=UTC),
            )


if __name__ == "__main__":
    main()
