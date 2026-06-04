from collections.abc import Callable, Mapping
from hashlib import sha256
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import BinaryIO, cast
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.task import (
    ArtifactStoreNotFoundError,
    PdfImageFileConverter,
    TaskArtifactPurpose,
    TaskArtifactRef,
    TaskArtifactRetention,
    TaskArtifactStat,
    TaskArtifactState,
    TaskDefinition,
    TaskExecutionRequest,
    TaskExecutionTarget,
    TaskFeature,
    TaskFileConversionError,
    TaskFileConversionPageCollection,
    TaskFileConversionPageResult,
    TaskFileConversionRequest,
    TaskFileConverterCapability,
    TaskInputContract,
    TaskMetadata,
    TaskOutputContract,
    convert_task_artifact_pages,
    pdf_image_converter_capability,
    safe_pdf_image_capability_metadata,
    validate_conversion_request,
)
from avalan.task.artifacts import LocalArtifactStore
from avalan.task.converters import (
    TaskFileConversionDependencyError,
    UnavailableFileConverter,
)
from avalan.task.converters.registry import default_file_converters
from avalan.task.stores import InMemoryTaskStore


class PdfImageFileConverterTest(IsolatedAsyncioTestCase):
    def test_capability_records_pdf_image_limits(self) -> None:
        capability = pdf_image_converter_capability()
        metadata = safe_pdf_image_capability_metadata(capability)

        self.assertEqual(capability.source_mime_types, ("application/pdf",))
        self.assertEqual(
            capability.output_mime_types,
            ("image/png", "image/jpeg"),
        )
        self.assertFalse(capability.supports_streaming)
        self.assertEqual(capability.max_input_bytes, 100 * 1024 * 1024)
        self.assertEqual(capability.max_output_bytes, 50 * 1024 * 1024)
        self.assertEqual(capability.max_pages, 256)
        self.assertEqual(capability.min_dpi, 36)
        self.assertEqual(capability.max_dpi, 300)
        self.assertEqual(capability.min_quality, 1)
        self.assertEqual(capability.max_quality, 100)
        self.assertEqual(capability.max_pixels, 40_000_000)
        self.assertEqual(capability.estimated_memory_bytes, 512 * 1024 * 1024)
        self.assertEqual(capability.timeout_seconds, 120)
        self.assertEqual(
            capability.dependency_gates,
            (TaskFeature.PDF_IMAGE_CONVERSION,),
        )
        self.assertEqual(metadata["max_pages"], 256)
        self.assertEqual(metadata["options_schema"], capability.options_schema)

    def test_default_registry_exposes_unavailable_pdf_image_converter(
        self,
    ) -> None:
        with patch(
            "avalan.task.converters.registry.feature_available",
            side_effect=lambda feature: (
                feature != (TaskFeature.PDF_IMAGE_CONVERSION)
            ),
        ):
            converter = default_file_converters()["pdf_image"]

        self.assertIsInstance(converter, UnavailableFileConverter)
        self.assertEqual(converter.name, "pdf_image")
        self.assertEqual(converter.version, "unavailable")
        self.assertEqual(
            converter.capability.dependency_gates,
            (TaskFeature.PDF_IMAGE_CONVERSION,),
        )

    def test_default_registry_uses_available_pdf_image_converter(self) -> None:
        with patch(
            "avalan.task.converters.registry.feature_available",
            return_value=True,
        ):
            converter = default_file_converters()["pdf_image"]

        self.assertIsInstance(converter, PdfImageFileConverter)
        self.assertEqual(converter.name, "pdf_image")
        self.assertEqual(converter.version, "1")

    def test_preflight_accepts_supported_options(self) -> None:
        converter = PdfImageFileConverter()

        with patch(
            "avalan.task.converters.feature_available", return_value=True
        ):
            validate_conversion_request(
                converter,
                TaskFileConversionRequest(
                    name="pdf_image",
                    options={
                        "dpi": 144,
                        "format": "jpeg",
                        "pages": {"start": 2, "end": 4},
                        "quality": 85,
                    },
                ),
                source_media_type="application/pdf",
                source_size_bytes=10 * 1024,
            )

    def test_preflight_rejects_dependency_and_unsafe_inputs(self) -> None:
        converter = PdfImageFileConverter()
        request = TaskFileConversionRequest(name="pdf_image")

        with self.assertRaises(TaskFileConversionDependencyError):
            validate_conversion_request(
                converter,
                request,
                source_media_type="application/pdf",
                source_size_bytes=10,
            )

        with patch(
            "avalan.task.converters.feature_available", return_value=True
        ):
            for media_type, size_bytes in (
                ("text/plain", 10),
                ("application/pdf", 100 * 1024 * 1024 + 1),
            ):
                with self.assertRaises(TaskFileConversionError):
                    validate_conversion_request(
                        converter,
                        request,
                        source_media_type=media_type,
                        source_size_bytes=size_bytes,
                    )

    def test_preflight_rejects_malformed_options(self) -> None:
        converter = PdfImageFileConverter()
        bad_options: tuple[Mapping[str, object], ...] = (
            {"format": "gif"},
            {"format": "png", "quality": 80},
            {"format": "jpeg", "quality": 0},
            {"format": "jpeg", "quality": 101},
            {"format": "jpeg", "quality": True},
            {"dpi": 35},
            {"dpi": 301},
            {"dpi": "144"},
            {"dpi": False},
            {"pages": "1-3"},
            {"pages": {"start": 0}},
            {"pages": {"start": 4, "end": 3}},
            {"pages": {"start": 1, "end": 257}},
            {"pages": {"start": True}},
            {"pages": {"end": False}},
            {"pages": {"start": 1, "raw": "private.pdf"}},
            {"raw": "private.pdf"},
        )

        with patch(
            "avalan.task.converters.feature_available", return_value=True
        ):
            for options in bad_options:
                with self.subTest(options=options):
                    with self.assertRaises(TaskFileConversionError) as error:
                        validate_conversion_request(
                            converter,
                            TaskFileConversionRequest(
                                name="pdf_image",
                                options=options,
                            ),
                            source_media_type="application/pdf",
                            source_size_bytes=10,
                        )
                    self.assertNotIn("private.pdf", str(error.exception))

    async def test_unavailable_converter_reports_pdf_image_dependency(
        self,
    ) -> None:
        converter = UnavailableFileConverter(
            name="pdf_image",
            version="unavailable",
            capability=pdf_image_converter_capability(),
        )

        with self.assertRaises(TaskFileConversionDependencyError) as error:
            await converter.convert(b"private pdf")

        self.assertEqual(
            error.exception.feature, TaskFeature.PDF_IMAGE_CONVERSION
        )
        self.assertNotIn("private pdf", str(error.exception))

    async def test_converter_runtime_failure_is_sanitized(self) -> None:
        converter = PdfImageFileConverter()

        with self.assertRaises(TaskFileConversionError) as error:
            await converter.convert(
                b"%PDF-private",
                source_media_type="application/pdf",
                options={"format": "png"},
            )

        self.assertNotIn("%PDF-private", str(error.exception))

    async def test_converter_rejects_non_pdf_media_type_before_backend(
        self,
    ) -> None:
        converter = PdfImageFileConverter()

        with self.assertRaises(TaskFileConversionError) as error:
            await converter.convert(
                b"private content",
                source_media_type="text/plain",
                options={"raw": "private.pdf"},
            )

        self.assertEqual(
            str(error.exception),
            "PDF image source media type is not supported",
        )
        self.assertNotIn("private content", str(error.exception))
        self.assertNotIn("private.pdf", str(error.exception))

    async def test_converter_accepts_pdf_media_type_case_insensitively(
        self,
    ) -> None:
        converter = PdfImageFileConverter()

        with self.assertRaises(TaskFileConversionError) as error:
            await converter.convert(
                b"%PDF-private",
                source_media_type="Application/PDF",
                options={"format": "png"},
            )

        self.assertEqual(
            str(error.exception), "PDF image conversion is unavailable"
        )
        self.assertNotIn("%PDF-private", str(error.exception))

    def test_page_collection_contract_preserves_order_and_metadata(
        self,
    ) -> None:
        page = TaskFileConversionPageResult(
            page_index=2,
            page_count=3,
            content=b"\x89PNG page",
            media_type="image/png",
            width_pixels=100,
            height_pixels=200,
            metadata={"identity": {"privacy": "<hmac-sha256>"}},
        )
        metadata = {"backend": "fake"}
        collection = TaskFileConversionPageCollection(
            pages=(
                page,
                TaskFileConversionPageResult(
                    page_index=3,
                    page_count=3,
                    content=b"\x89PNG page",
                    media_type="image/png",
                    width_pixels=100,
                    height_pixels=200,
                ),
            ),
            metadata=metadata,
        )

        metadata["backend"] = "mutated"

        self.assertEqual(collection.pages[0].page_index, 2)
        self.assertEqual(collection.metadata["backend"], "fake")
        self.assertEqual(
            cast(Mapping[str, object], page.metadata["identity"])["privacy"],
            "<hmac-sha256>",
        )
        with self.assertRaises(AssertionError):
            TaskFileConversionPageResult(
                page_index=0,
                page_count=1,
                content=b"page",
                media_type="image/png",
                width_pixels=1,
                height_pixels=1,
            )
        with self.assertRaises(AssertionError):
            TaskFileConversionPageResult(
                page_index=True,
                page_count=1,
                content=b"page",
                media_type="image/png",
                width_pixels=1,
                height_pixels=1,
            )
        with self.assertRaises(AssertionError):
            TaskFileConversionPageCollection(
                pages=(
                    TaskFileConversionPageResult(
                        page_index=2,
                        page_count=3,
                        content=b"page",
                        media_type="image/png",
                        width_pixels=1,
                        height_pixels=1,
                    ),
                    TaskFileConversionPageResult(
                        page_index=1,
                        page_count=3,
                        content=b"page",
                        media_type="image/png",
                        width_pixels=1,
                        height_pixels=1,
                    ),
                )
            )
        with self.assertRaises(AssertionError):
            TaskFileConversionPageCollection(
                pages=(
                    TaskFileConversionPageResult(
                        page_index=1,
                        page_count=1,
                        content=b"page",
                        media_type="image/png",
                        width_pixels=1,
                        height_pixels=1,
                    ),
                    TaskFileConversionPageResult(
                        page_index=2,
                        page_count=2,
                        content=b"page",
                        media_type="image/png",
                        width_pixels=1,
                        height_pixels=1,
                    ),
                )
            )

    async def test_page_conversion_writes_ordered_converted_artifacts(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=_id_factory(("page-1", "page-2")),
            )
            source_ref = await artifact_store.put(
                b"%PDF-private source",
                artifact_id="source-1",
                media_type="application/pdf",
                metadata={"filename": "private.pdf"},
            )
            task_store = InMemoryTaskStore(
                id_factory=_id_factory(_store_ids())
            )
            definition = _definition()
            await task_store.register_definition(
                definition,
                definition_hash="definition-1",
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="definition-1")
            )
            attempt = await task_store.create_attempt(run.run_id)

            with patch(
                "avalan.task.converters.feature_available",
                return_value=True,
            ):
                converted = await convert_task_artifact_pages(
                    source_ref,
                    TaskFileConversionRequest(
                        name="pdf_image",
                        options={"format": "png", "pages": {"start": 1}},
                    ),
                    converter=FakePdfPageConverter(
                        (
                            _page_result(1, b"page one"),
                            _page_result(2, b"page two"),
                        )
                    ),
                    artifact_store=artifact_store,
                    task_store=task_store,
                    run_id=run.run_id,
                    attempt_id=attempt.attempt_id,
                    retention=TaskArtifactRetention(delete_after_days=3),
                )

            records = await task_store.list_artifacts(
                run.run_id,
                purpose=TaskArtifactPurpose.CONVERTED,
            )

            self.assertEqual(
                [page.ref.artifact_id for page in converted.pages],
                ["page-1", "page-2"],
            )
            self.assertEqual(
                [record.artifact_id for record in records],
                [
                    "page-1",
                    "page-2",
                ],
            )
            self.assertEqual(records[0].attempt_id, attempt.attempt_id)
            self.assertEqual(records[0].retention.delete_after_days, 3)
            self.assertEqual(records[0].provenance.converter, "pdf_image")
            provenance = cast(
                Mapping[str, object],
                records[0].provenance.metadata,
            )
            dimensions = cast(Mapping[str, object], provenance["dimensions"])
            self.assertEqual(provenance["page_index"], 1)
            self.assertEqual(provenance["page_count"], 2)
            self.assertEqual(dimensions["width_pixels"], 100)
            self.assertEqual(dimensions["height_pixels"], 200)
            self.assertNotIn("private.pdf", str(records[0].summary()))
            self.assertNotIn("%PDF-private", str(records[0].summary()))
            reader = await artifact_store.open(converted.pages[1].ref)
            try:
                self.assertEqual(reader.read(), b"page two")
            finally:
                reader.close()

    async def test_page_conversion_cleans_up_partial_page_outputs(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
                id_factory=_id_factory(("page-1", "page-2")),
            )
            source_ref = await artifact_store.put(
                b"%PDF-private source",
                artifact_id="source-1",
                media_type="application/pdf",
            )
            task_store = InMemoryTaskStore(
                id_factory=_id_factory(_store_ids())
            )
            definition = _definition()
            await task_store.register_definition(
                definition,
                definition_hash="definition-1",
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="definition-1")
            )

            with patch(
                "avalan.task.converters.feature_available",
                return_value=True,
            ):
                with self.assertRaises(TaskFileConversionError) as error:
                    await convert_task_artifact_pages(
                        source_ref,
                        TaskFileConversionRequest(name="pdf_image"),
                        converter=FakePdfPageConverter(
                            (
                                _page_result(1, b"page one"),
                                _page_result(2, b""),
                            )
                        ),
                        artifact_store=artifact_store,
                        task_store=task_store,
                        run_id=run.run_id,
                    )

            records = await task_store.list_artifacts(run.run_id)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].state, TaskArtifactState.LOST)
            self.assertNotIn("%PDF-private", str(error.exception))
            with self.assertRaises(ArtifactStoreNotFoundError):
                await artifact_store.open(records[0].ref)

    async def test_page_conversion_rejects_missing_page_contract(
        self,
    ) -> None:
        with TemporaryDirectory() as artifacts:
            artifact_store = LocalArtifactStore(
                artifacts,
                raw_storage_allowed=True,
            )
            source_ref = await artifact_store.put(
                b"%PDF-private source",
                artifact_id="source-1",
                media_type="application/pdf",
            )
            task_store = InMemoryTaskStore(
                id_factory=_id_factory(_store_ids())
            )
            definition = _definition()
            await task_store.register_definition(
                definition,
                definition_hash="definition-1",
            )
            run = await task_store.create_run(
                TaskExecutionRequest(definition_id="definition-1")
            )

            with patch(
                "avalan.task.converters.feature_available",
                return_value=True,
            ):
                with self.assertRaises(TaskFileConversionError) as error:
                    await convert_task_artifact_pages(
                        source_ref,
                        TaskFileConversionRequest(name="pdf_image"),
                        converter=PdfImageFileConverter(),
                        artifact_store=artifact_store,
                        task_store=task_store,
                        run_id=run.run_id,
                    )

            self.assertNotIn("%PDF-private", str(error.exception))

    async def test_page_conversion_rejects_runtime_page_limit_violations(
        self,
    ) -> None:
        cases = (
            (
                FakePdfPageConverter((_page_result(1, b"oversized"),), 4),
                "byte",
            ),
            (
                FakePdfPageConverter(
                    (
                        TaskFileConversionPageResult(
                            page_index=1,
                            page_count=1,
                            content=b"page",
                            media_type="application/pdf",
                            width_pixels=1,
                            height_pixels=1,
                        ),
                    )
                ),
                "media type",
            ),
            (
                FakePdfPageConverter(
                    (
                        TaskFileConversionPageResult(
                            page_index=1,
                            page_count=1,
                            content=b"page",
                            media_type="image/png",
                            width_pixels=40_000_001,
                            height_pixels=1,
                        ),
                    )
                ),
                "pixel",
            ),
        )
        for converter, expected in cases:
            with self.subTest(expected=expected):
                error = await _convert_pages_expect_error(converter)
                self.assertIn(expected, str(error))
                self.assertNotIn("%PDF-private", str(error))

    async def test_page_conversion_rejects_stream_growth(self) -> None:
        source_ref = TaskArtifactRef(
            artifact_id="source-1",
            store="fake",
            storage_key="source-1",
            media_type="application/pdf",
            size_bytes=4,
            sha256=sha256(b"safe").hexdigest(),
        )
        task_store = InMemoryTaskStore(id_factory=_id_factory(_store_ids()))
        definition = _definition()
        await task_store.register_definition(
            definition,
            definition_hash="definition-1",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="definition-1")
        )

        with patch(
            "avalan.task.converters.feature_available",
            return_value=True,
        ):
            with self.assertRaises(TaskFileConversionError) as error:
                await convert_task_artifact_pages(
                    source_ref,
                    TaskFileConversionRequest(name="pdf_image"),
                    converter=FakePdfPageConverter(
                        (_page_result(1, b"page"),)
                    ),
                    artifact_store=GrowingPdfArtifactStore(),
                    task_store=task_store,
                    run_id=run.run_id,
                )

        self.assertNotIn("private", str(error.exception))


class FakePdfPageConverter:
    name = "pdf_image"
    version = "fake"

    def __init__(
        self,
        pages: tuple[TaskFileConversionPageResult, ...],
        max_output_bytes: int | None = 1024,
    ) -> None:
        base = pdf_image_converter_capability()
        self._pages = pages
        self._capability = TaskFileConverterCapability(
            source_mime_types=base.source_mime_types,
            output_mime_types=base.output_mime_types,
            supports_streaming=base.supports_streaming,
            max_input_bytes=base.max_input_bytes,
            max_output_bytes=max_output_bytes,
            max_pages=base.max_pages,
            min_dpi=base.min_dpi,
            max_dpi=base.max_dpi,
            min_quality=base.min_quality,
            max_quality=base.max_quality,
            max_pixels=base.max_pixels,
            estimated_memory_bytes=base.estimated_memory_bytes,
            timeout_seconds=base.timeout_seconds,
            options_schema=base.options_schema,
            dependency_gates=base.dependency_gates,
        )

    @property
    def capability(self) -> TaskFileConverterCapability:
        return self._capability

    def validate_options(self, options: Mapping[str, object]) -> None:
        PdfImageFileConverter().validate_options(options)

    async def convert(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> object:
        raise AssertionError("page converter must use convert_pages")

    async def convert_pages(
        self,
        content: bytes,
        *,
        source_media_type: str | None = None,
        options: Mapping[str, object] | None = None,
    ) -> TaskFileConversionPageCollection:
        return TaskFileConversionPageCollection(
            pages=self._pages,
            metadata={"backend": "fake"},
        )


class GrowingPdfArtifactStore:
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
        return BytesIO(b"private pdf growth")

    async def stat(self, ref: TaskArtifactRef) -> TaskArtifactStat:
        return TaskArtifactStat(
            ref=ref,
            size_bytes=4,
            sha256=sha256(b"safe").hexdigest(),
        )

    async def delete(self, ref: TaskArtifactRef) -> None:
        raise AssertionError("delete must not be called")


async def _convert_pages_expect_error(
    converter: FakePdfPageConverter,
) -> TaskFileConversionError:
    with TemporaryDirectory() as artifacts:
        artifact_store = LocalArtifactStore(
            artifacts,
            raw_storage_allowed=True,
            id_factory=_id_factory(("page-1",)),
        )
        source_ref = await artifact_store.put(
            b"%PDF-private source",
            artifact_id="source-1",
            media_type="application/pdf",
        )
        task_store = InMemoryTaskStore(id_factory=_id_factory(_store_ids()))
        definition = _definition()
        await task_store.register_definition(
            definition,
            definition_hash="definition-1",
        )
        run = await task_store.create_run(
            TaskExecutionRequest(definition_id="definition-1")
        )
        with patch(
            "avalan.task.converters.feature_available",
            return_value=True,
        ):
            try:
                await convert_task_artifact_pages(
                    source_ref,
                    TaskFileConversionRequest(name="pdf_image"),
                    converter=converter,
                    artifact_store=artifact_store,
                    task_store=task_store,
                    run_id=run.run_id,
                )
            except TaskFileConversionError as error:
                return error
    raise AssertionError("conversion should have failed")


def _page_result(
    page_index: int,
    content: bytes,
) -> TaskFileConversionPageResult:
    return TaskFileConversionPageResult(
        page_index=page_index,
        page_count=2,
        content=content,
        media_type="image/png",
        width_pixels=100,
        height_pixels=200,
        metadata={"identity": {"privacy": "<hmac-sha256>"}},
    )


def _definition() -> TaskDefinition:
    return TaskDefinition(
        task=TaskMetadata(name="render", version="1"),
        input=TaskInputContract.file(mime_types=("application/pdf",)),
        output=TaskOutputContract.file(),
        execution=TaskExecutionTarget.agent("agents/render.toml"),
    )


def _id_factory(values: tuple[str, ...]) -> Callable[[], str]:
    ids = iter(values)
    return lambda: next(ids)


def _store_ids() -> tuple[str, ...]:
    return (
        "run-1",
        "transition-1",
        "attempt-1",
        "event-1",
        "cleanup-transition-1",
    )


if __name__ == "__main__":
    main()
