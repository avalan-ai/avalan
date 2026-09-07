from asyncio import Event, create_task
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, patch

from client_test import RecordingArtifactStore, _definition
from pytest import raises
from submission_lifecycle_test import AcknowledgmentQueue, _client

from avalan.task.definition import TaskInputContract, TaskRunPolicy
from avalan.task.preparation import prepare_task_input
from avalan.task.stores import InMemoryTaskStore
from avalan.task.validation import TaskValidationError


class TaskInputPreparationTest(IsolatedAsyncioTestCase):
    async def test_actual_validation_materializes_without_a_run(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input.txt").write_text("registered bytes")
            task_store = InMemoryTaskStore()
            queue = AcknowledgmentQueue(task_store)
            artifacts = RecordingArtifactStore()
            client = _client(root, queue, artifacts)
            definition = _definition(
                input_contract=TaskInputContract.file(),
                run=TaskRunPolicy.queued("q"),
            )
            value = {
                "source_kind": "local_path",
                "reference": "input.txt",
                "mime_type": "text/plain",
            }
            prepared = await prepare_task_input(
                client, definition, input_value=value
            )
            assert not task_store._runs
            assert len(prepared.materialized_files) == 1
            assert await task_store.get_definition(prepared.definition_id)
            first = prepared.files_for_submission(
                client, prepared.definition_id, value
            )
            second = prepared.files_for_submission(
                client, prepared.definition_id, value
            )
            assert first[0].ref.artifact_id != second[0].ref.artifact_id
            assert first[0].ref.storage_key == second[0].ref.storage_key
            with raises(AssertionError):
                prepared.files_for_submission(client, "other", value)
            with raises(AssertionError):
                prepared.files_for_submission(
                    client,
                    prepared.definition_id,
                    {**value, "reference": "changed.txt"},
                )
            with raises(AssertionError):
                replace(prepared, _authority=object())

    async def test_capability_and_schema_reject_before_writes(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            task_store = InMemoryTaskStore()
            queue = AcknowledgmentQueue(task_store)
            artifacts = RecordingArtifactStore()
            client = _client(root, queue, artifacts)
            definition = _definition(run=TaskRunPolicy.queued("q"))
            with (
                patch.object(
                    queue,
                    "preflight_submission",
                    AsyncMock(side_effect=RuntimeError),
                ),
                raises(RuntimeError),
            ):
                await prepare_task_input(
                    client, definition, input_value="input"
                )
            assert not task_store._definitions and not task_store._runs
            with raises(TaskValidationError):
                await prepare_task_input(
                    client,
                    replace(definition, input=TaskInputContract.file()),
                    input_value={"invalid": "descriptor"},
                )
            assert not task_store._definitions and not task_store._runs

    async def test_scalar_snapshot_without_materialization(self) -> None:
        with TemporaryDirectory() as directory:
            task_store = InMemoryTaskStore()
            queue = AcknowledgmentQueue(task_store)
            client = _client(Path(directory), queue, RecordingArtifactStore())
            prepared = await prepare_task_input(
                client,
                _definition(run=TaskRunPolicy.queued("q")),
                input_value="hello",
            )
            assert prepared.input_value == "hello"
            assert prepared.materialized_files == ()

    async def test_input_is_frozen_before_preflight_waits(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "original.txt").write_text("original")
            (root / "mutated.txt").write_text("mutated")
            queue = AcknowledgmentQueue(InMemoryTaskStore())
            client = _client(root, queue, RecordingArtifactStore())
            entered, release = Event(), Event()

            async def preflight() -> None:
                entered.set()
                await release.wait()

            value = {
                "source_kind": "local_path",
                "reference": "original.txt",
                "mime_type": "text/plain",
            }
            with patch.object(queue, "preflight_submission", preflight):
                running = create_task(
                    prepare_task_input(
                        client,
                        _definition(
                            input_contract=TaskInputContract.file(),
                            run=TaskRunPolicy.queued("q"),
                        ),
                        input_value=value,
                    )
                )
                await entered.wait()
                value["reference"] = "mutated.txt"
                release.set()
                prepared = await running
            assert (
                prepared.materialized_files[0].descriptor.reference
                == "original.txt"
            )
            assert prepared.input_value != value
