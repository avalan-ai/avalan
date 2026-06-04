from asyncio import CancelledError
from collections.abc import Callable, Mapping
from hashlib import sha256
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.task import (
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactStat,
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
    TaskFileConversionDependencyError,
    TaskFileConversionError,
    TaskFileConversionResult,
    TaskFileConverterCapability,
    UnavailableFileConverter,
    convert_task_artifact,
    markdown_converter_capability,
    validate_conversion_request,
)
from avalan.task.converters.registry import default_file_converters
from avalan.task.converters.text import TextFileConverter
from avalan.task.stores import InMemoryTaskStore


class TextFileConverterTest(IsolatedAsyncioTestCase):
    def test_default_converter_registry_includes_text_and_markdown(
        self,
    ) -> None:
        converters = default_file_converters()

        self.assertEqual(
            converters["text"].capability.max_input_bytes,
            1024 * 1024,
        )
        self.assertIn(
            "text/plain",
            converters["text"].capability.output_mime_types,
        )
        self.assertIn("markdown", converters)

        with patch(
            "avalan.task.converters.registry.feature_available",
            return_value=False,
        ):
            fallback = default_file_converters()["markdown"]

        self.assertEqual(fallback.name, "markdown")
        self.assertEqual(fallback.version, "unavailable")
        self.assertEqual(
            fallback.capability.dependency_gates[0].value,
            "document_conversion",
        )

    async def test_unavailable_converter_reports_dependency(self) -> None:
        converter = UnavailableFileConverter(
            name="markdown",
            version="unavailable",
            capability=markdown_converter_capability(),
        )

        with self.assertRaises(TaskFileConversionDependencyError):
            await converter.convert(b"private")

    def test_text_converter_capability_preflight(self) -> None:
        converter = TextFileConverter()
        request = TaskFileConversionRequest(
            name="text",
            options={"newline": "lf"},
        )

        validate_conversion_request(
            converter,
            request,
            source_media_type="application/activity+json",
            source_size_bytes=10,
        )

        with self.assertRaises(TaskFileConversionError):
            validate_conversion_request(
                converter,
                request,
                source_media_type="application/pdf",
                source_size_bytes=10,
            )
        with self.assertRaises(TaskFileConversionError):
            validate_conversion_request(
                converter,
                request,
                source_media_type="text/plain",
                source_size_bytes=1024 * 1024 + 1,
            )
        with self.assertRaises(TaskFileConversionError):
            validate_conversion_request(
                converter,
                TaskFileConversionRequest(
                    name="text",
                    options={"newline": "crlf"},
                ),
                source_media_type="text/plain",
                source_size_bytes=10,
            )

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

    async def test_conversion_service_enforces_source_and_output_caps(
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
            request = TaskFileConversionRequest(name="small")

            with self.assertRaises(TaskFileConversionError) as source_error:
                await convert_task_artifact(
                    source_ref,
                    request,
                    converter=SmallCapConverter(max_input_bytes=4),
                    artifact_store=artifact_store,
                )
            with self.assertRaises(TaskFileConversionError) as output_error:
                await convert_task_artifact(
                    source_ref,
                    request,
                    converter=SmallCapConverter(max_output_bytes=4),
                    artifact_store=artifact_store,
                )
            with self.assertRaises(TaskFileConversionError):
                await convert_task_artifact(
                    source_ref,
                    request,
                    converter=NoInputCapConverter(),
                    artifact_store=artifact_store,
                )
            with self.assertRaises(TaskFileConversionError):
                await convert_task_artifact(
                    source_ref,
                    request,
                    converter=WrongMediaConverter(),
                    artifact_store=artifact_store,
                )

            rendered = f"{source_error.exception} {output_error.exception}"
            self.assertNotIn("private text", rendered)

    async def test_conversion_service_rejects_stream_growth(self) -> None:
        store = GrowingArtifactStore()
        ref = TaskArtifactRef(
            artifact_id="source-1",
            store="fake",
            storage_key="source-1",
            media_type="text/plain",
            size_bytes=4,
            sha256=sha256(b"safe").hexdigest(),
        )

        with self.assertRaises(TaskFileConversionError) as error:
            await convert_task_artifact(
                ref,
                TaskFileConversionRequest(name="small"),
                converter=SmallCapConverter(max_input_bytes=4),
                artifact_store=store,
            )

        self.assertNotIn("private", str(error.exception))

    async def test_conversion_service_propagates_cancellation(self) -> None:
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

            with self.assertRaises(CancelledError):
                await convert_task_artifact(
                    source_ref,
                    TaskFileConversionRequest(name="cancel"),
                    converter=CancellingConverter(),
                    artifact_store=artifact_store,
                )

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


class SmallCapConverter:
    name = "small"
    version = "1"

    def __init__(
        self,
        *,
        max_input_bytes: int | None = 1024,
        max_output_bytes: int | None = 1024,
    ) -> None:
        self._capability = TaskFileConverterCapability(
            source_mime_types=("text/plain",),
            output_mime_types=("text/plain",),
            supports_streaming=False,
            max_input_bytes=max_input_bytes,
            max_output_bytes=max_output_bytes,
        )

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        return TaskFileConversionResult(
            content=content,
            media_type="text/plain",
            metadata={},
        )


class NoInputCapConverter(SmallCapConverter):
    def __init__(self) -> None:
        super().__init__(max_input_bytes=None)


class CancellingConverter(SmallCapConverter):
    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        raise CancelledError()


class WrongMediaConverter(SmallCapConverter):
    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionResult:
        return TaskFileConversionResult(
            content=content,
            media_type="application/pdf",
            metadata={},
        )


class GrowingArtifactStore:
    async def put(
        self,
        content: bytes,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TaskArtifactRef:
        raise AssertionError("put must not be called")

    async def put_stream(
        self,
        stream: BinaryIO,
        *,
        artifact_id: str | None = None,
        media_type: str | None = None,
        metadata: Mapping[str, object] | None = None,
        max_bytes: int | None = None,
        expected_size_bytes: int | None = None,
        expected_sha256: str | None = None,
    ) -> TaskArtifactRef:
        raise AssertionError("put_stream must not be called")

    async def open(self, ref: TaskArtifactRef) -> BinaryIO:
        raise AssertionError("open must not be called")

    async def open_stream(
        self,
        ref: TaskArtifactRef,
        *,
        max_bytes: int | None = None,
    ) -> BinaryIO:
        return BytesIO(b"private text")

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return TaskArtifactStat(
            ref=ref,
            size_bytes=4,
            sha256=sha256(b"safe").hexdigest(),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        raise AssertionError("delete must not be called")


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
