from collections.abc import Mapping
from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.task import (
    PdfImageFileConverter,
    TaskFeature,
    TaskFileConversionError,
    TaskFileConversionRequest,
    pdf_image_converter_capability,
    safe_pdf_image_capability_metadata,
    validate_conversion_request,
)
from avalan.task.converters import (
    TaskFileConversionDependencyError,
    UnavailableFileConverter,
)
from avalan.task.converters.registry import default_file_converters


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


if __name__ == "__main__":
    main()
