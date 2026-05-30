from collections.abc import Callable, Mapping
from tempfile import TemporaryDirectory
from typing import cast
from unittest import IsolatedAsyncioTestCase, main

from avalan.task import (
    TaskArtifactPurpose,
    TaskArtifactRetention,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskFileConversionRequest,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.converters import (
    TaskFileConversionError,
    convert_task_artifact,
)
from avalan.task.converters.text import TextFileConverter
from avalan.task.stores import InMemoryTaskStore


class TextFileConverterTest(IsolatedAsyncioTestCase):
    async def test_text_converter_decodes_and_normalizes_text(self) -> None:
        converter = TextFileConverter()

        result = await converter.convert(
            "café\r\nline".encode("latin-1"),
            source_media_type="text/plain",
            options={"encoding": "latin-1", "newline": "lf"},
        )

        self.assertEqual(result.content, "café\nline".encode())
        self.assertEqual(result.media_type, "text/plain")
        self.assertEqual(result.metadata["source_encoding"], "latin-1")
        self.assertEqual(result.metadata["characters"], 9)

    async def test_text_converter_rejects_binary_and_bad_options(
        self,
    ) -> None:
        converter = TextFileConverter()

        with self.assertRaises(TaskFileConversionError) as binary_error:
            await converter.convert(
                b"\x00private",
                source_media_type="application/octet-stream",
            )
        with self.assertRaises(TaskFileConversionError):
            await converter.convert(
                b"private",
                source_media_type="text/plain",
                options={"raw": "value"},
            )
        with self.assertRaises(TaskFileConversionError):
            await converter.convert(
                b"\xff",
                source_media_type="text/plain",
            )

        self.assertNotIn("private", str(binary_error.exception))

    async def test_conversion_service_writes_and_records_artifact(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=_id_factory(),
            )
            source_ref = await artifact_store.put(
                b"private text",
                artifact_id="source-1",
                media_type="text/plain",
            )
            task_store = InMemoryTaskStore(id_factory=_id_factory())
            definition = _definition()
            await task_store.register_definition(
                definition,
                definition_hash="definition-1",
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="definition-1")
            )
            attempt = await task_store.create_attempt(run.run_id)
            request = TaskFileConversionRequest(
                name="text",
                options={"newline": "lf"},
            )

            converted = await convert_task_artifact(
                source_ref,
                request,
                converter=TextFileConverter(),
                artifact_store=artifact_store,
                task_store=task_store,
                run_id=run.run_id,
                attempt_id=attempt.attempt_id,
                retention=TaskArtifactRetention(delete_after_days=7),
            )
            records = await task_store.list_artifacts(
                run.run_id,
                purpose=TaskArtifactPurpose.CONVERTED,
            )

            self.assertEqual(converted.ref.artifact_id, "converted-1")
            self.assertEqual(converted.ref.media_type, "text/plain")
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record.artifact_id, "converted-1")
            self.assertEqual(record.attempt_id, attempt.attempt_id)
            self.assertEqual(record.provenance.source_artifact_id, "source-1")
            self.assertEqual(record.provenance.converter, "text")
            self.assertEqual(record.retention.delete_after_days, 7)
            provenance = cast(
                Mapping[str, object],
                record.provenance.metadata,
            )
            converter = cast(Mapping[str, object], provenance["converter"])
            identity = cast(Mapping[str, object], provenance["identity"])
            options = cast(Mapping[str, object], provenance["options"])
            self.assertEqual(converter["name"], "text")
            self.assertEqual(converter["version"], "1")
            self.assertEqual(provenance["target_media_type"], "text/plain")
            self.assertEqual(options["newline"], "lf")
            self.assertEqual(identity["size_bytes"], 12)
            self.assertNotIn("private text", str(record.summary()))
            reader = await artifact_store.open(converted.ref)
            try:
                self.assertEqual(reader.read(), b"private text")
            finally:
                reader.close()

    async def test_conversion_service_rejects_invalid_runtime_values(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
            )
            source_ref = await artifact_store.put(
                b"private text",
                artifact_id="source-1",
                media_type="text/plain",
            )

            with self.assertRaises(AssertionError):
                await convert_task_artifact(
                    source_ref,
                    TaskFileConversionRequest(name="text"),
                    converter=TextFileConverter(),
                    artifact_store=artifact_store,
                    task_store=InMemoryTaskStore(),
                )

    async def test_text_converter_covers_boundary_options(self) -> None:
        converter = TextFileConverter()

        no_media_type = await converter.convert(b"private")

        self.assertEqual(no_media_type.content, b"private")
        for options in (
            {"errors": "bad"},
            {"newline": "crlf"},
            {"encoding": 1},
        ):
            with self.assertRaises(TaskFileConversionError):
                await converter.convert(
                    b"private",
                    source_media_type="text/plain",
                    options=options,
                )


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="convert", version="1"),
        input=TaskInputContract.file(mime_types=("text/plain",)),
        output=TaskOutputContract.text(),
        execution=TaskExecutionTarget.agent("agents/convert.toml"),
    )


def _id_factory() -> Callable[[], str]:
    ids = iter(
        (
            "converted-1",
            "definition-1",
            "run-1",
            "transition-1",
            "attempt-1",
            "event-1",
        )
    )
    return lambda: next(ids)


if __name__ == "__main__":
    main()
