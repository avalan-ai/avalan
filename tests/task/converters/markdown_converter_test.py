from unittest import IsolatedAsyncioTestCase, main
from unittest.mock import patch

from avalan.task.converters import (
    TaskFileConversionDependencyError,
    TaskFileConversionError,
)
from avalan.task.converters.markdown import MarkdownFileConverter


class MarkdownFileConverterTest(IsolatedAsyncioTestCase):
    def test_markdown_converter_capability_and_options(self) -> None:
        converter = MarkdownFileConverter()

        self.assertIn("text/html", converter.capability.source_mime_types)
        converter.validate_options({"html_heading_style": "SETEXT"})

        with self.assertRaises(TaskFileConversionError):
            converter.validate_options({"html_heading_style": "PRIVATE"})

    async def test_markdown_converter_preserves_markdown_text(self) -> None:
        converter = MarkdownFileConverter()

        result = await converter.convert(
            b"# Title\n\nPrivate body",
            source_media_type="text/markdown",
        )

        self.assertEqual(result.content, b"# Title\n\nPrivate body")
        self.assertEqual(result.media_type, "text/markdown")
        self.assertEqual(result.metadata["characters"], 21)

    async def test_markdown_converter_converts_html(self) -> None:
        converter = MarkdownFileConverter()

        result = await converter.convert(
            b"<h1>Title</h1><p>Body</p>",
            source_media_type="text/html",
            options={"html_heading_style": "ATX"},
        )

        self.assertIn(b"# Title", result.content)
        self.assertIn(b"Body", result.content)
        self.assertEqual(result.metadata["source_media_type"], "text/html")

    async def test_markdown_converter_reports_missing_dependency(
        self,
    ) -> None:
        converter = MarkdownFileConverter()

        with patch(
            "avalan.task.converters.markdown.import_module",
            side_effect=ModuleNotFoundError("private missing module"),
        ):
            with self.assertRaises(TaskFileConversionDependencyError) as error:
                await converter.convert(
                    b"<h1>Private</h1>",
                    source_media_type="text/html",
                )

        self.assertNotIn("Private", str(error.exception))

    async def test_markdown_converter_rejects_binary_and_bad_options(
        self,
    ) -> None:
        converter = MarkdownFileConverter()

        with self.assertRaises(TaskFileConversionError) as media_error:
            await converter.convert(
                b"private bytes",
                source_media_type="application/octet-stream",
            )
        with self.assertRaises(TaskFileConversionError):
            await converter.convert(
                b"private bytes",
                source_media_type="text/plain",
                options={"html_heading_style": "UNKNOWN"},
            )
        with self.assertRaises(TaskFileConversionError):
            await converter.convert(
                b"private bytes",
                source_media_type="text/plain",
                options={"raw": "value"},
            )
        with self.assertRaises(TaskFileConversionError):
            await converter.convert(
                b"\xff",
                source_media_type="text/plain",
            )

        self.assertNotIn("private bytes", str(media_error.exception))

    async def test_markdown_converter_covers_boundary_options(self) -> None:
        converter = MarkdownFileConverter()

        no_media_type = await converter.convert(b"plain")

        self.assertEqual(no_media_type.content, b"plain")
        for options in (
            {"errors": "bad"},
            {"encoding": 1},
        ):
            with self.assertRaises(TaskFileConversionError):
                await converter.convert(
                    b"private",
                    source_media_type="text/plain",
                    options=options,
                )


if __name__ == "__main__":
    main()
