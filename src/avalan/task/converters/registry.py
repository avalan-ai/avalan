from ..feature_gate import TaskFeature, feature_available
from . import (
    FileConverter,
    UnavailableFileConverter,
    markdown_converter_capability,
)
from .markdown import MarkdownFileConverter
from .pdf_image import PdfImageFileConverter, pdf_image_converter_capability
from .text import TextFileConverter

from collections.abc import Mapping
from types import MappingProxyType


def default_file_converters() -> Mapping[str, FileConverter]:
    values: dict[str, FileConverter] = {"text": TextFileConverter()}
    markdown_capability = markdown_converter_capability()
    if feature_available(TaskFeature.DOCUMENT_CONVERSION):
        values["markdown"] = MarkdownFileConverter()
    else:
        values["markdown"] = UnavailableFileConverter(
            name="markdown",
            version="unavailable",
            capability=markdown_capability,
        )
    pdf_image_capability = pdf_image_converter_capability()
    if feature_available(TaskFeature.PDF_IMAGE_CONVERSION):
        values["pdf_image"] = PdfImageFileConverter()
    else:
        values["pdf_image"] = UnavailableFileConverter(
            name="pdf_image",
            version="unavailable",
            capability=pdf_image_capability,
        )
    return MappingProxyType(values)
